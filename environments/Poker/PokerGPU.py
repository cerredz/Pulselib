import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
import struct

class PokerGPU(gym.Env):
    metadata = {'render.modes': ['human']}
    NUM_ACTIONS=13
    ACTIVE, FOLDED, ALLIN, SITOUT = 0, 1, 2, 3
    STATE_SPACE=28 # details in depth_notes in rl folder
    MIN_EQUITY_RANK=4145.0
    MAX_EQUITY_RANK=36874.0
    MAX_FLOP_EQUITY=823779.0
    MIN_FLOP_EQUITY=74359.0
    MIN_TURN_RIVER_EQUITY=4109.0
    MAX_TURN_RIVER_EQUITY=36874.0

    def __init__(self, device, agents, n_players=6, max_players=10, n_games=100, starting_bbs=100, max_bbs=1000, w1=.5, w2=.5, K=20, alpha=300):
        super().__init__()
        self.device=device

        self.w1=torch.tensor(w1, device=self.device, dtype=torch.float32)
        self.w2=torch.tensor(w2, device=self.device, dtype=torch.float32)
        self.K=torch.tensor(K, device=self.device, dtype=torch.int32)
        self.alpha=torch.tensor(alpha, device=self.device, dtype=torch.int32)
        # env 
        
        self.agents=agents
        self.n_players=n_players
        self.n_games=n_games
        self.starting_bbs=starting_bbs
        self.max_bbs=max_bbs
        self.max_players=max_players

        # action space / observation space
        self.action_space=spaces.Discrete(self.NUM_ACTIONS)
        self.raise_fractions=torch.tensor([0.25, 0.33, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00], device=self.device)
        self.obs_size=13+((self.max_players-1)*3)
        self.observation_space = spaces.Box(low=0, high=10000, shape=(self.obs_size,), dtype=np.float32)
        
        # Initialize state tensors as None (will be set in reset)
        self.stacks = None

        # initialize hand ranks table for hand strength calculator
        hand_rank_file=Path(__file__).parent/"HandRanks.dat"
        if not hand_rank_file.exists():
            raise FileNotFoundError(f"HandRanks.dat not found at {hand_rank_file}. Download link: https://github.com/chenosaurus/poker-evaluator/raw/master/data/HandRanks.dat. Then put file in the environments/Poker folder.")

        file_size_bytes = hand_rank_file.stat().st_size
        num_elements = file_size_bytes // 4
        self.hand_ranks=torch.from_file(
            filename=str(hand_rank_file),
            shared=False,
            dtype=torch.int32,
            size=num_elements
        ).to(device=self.device, non_blocking=True)

        self.last_raise_size = None
        self.g=torch.arange(self.n_games, device=self.device)
        self.is_truncated=torch.zeros(self.n_games, dtype=torch.bool, device=self.device) # games never truncated early
        self.bb_amounts=torch.ones(self.n_games, device=self.device, dtype=torch.int32)

        self.equity_turn_denom=torch.tensor(max(self.MAX_TURN_RIVER_EQUITY - self.MIN_TURN_RIVER_EQUITY, 1e-8), device=self.device, dtype=torch.int32)
        self.equity_flop_denom=torch.tensor(max(self.MAX_FLOP_EQUITY - self.MIN_FLOP_EQUITY, 1e-8), device=self.device, dtype=torch.int32)
        self.zeros=torch.zeros(self.n_games, 1, device=self.device)
        
    def set_agents(self, agents):
        self.agents=agents

    def reset(self, seed=None, options=None, rotation=0):
        super().reset(seed=seed)
        if options and options['active_players']:
            candidate_players=torch.randint(2, self.n_players+1, (1,), device=self.device).item()
        else: candidate_players = self.n_players
        q_seat = options.get('q_agent_seat', 0)
        self.active_players = max(candidate_players, q_seat + 1)
        self.last_raise_size = torch.ones(self.n_games, dtype=torch.int32, device=self.device) 

        # game state tensors
        self.decks=torch.rand(self.n_games, 52, device=self.device).argsort(dim=1)+1
        self.deck_positions=torch.zeros(self.n_games, device=self.device, dtype=torch.int32)
        
        self.board = torch.full((self.n_games, 5), -1, dtype=torch.int32, device=self.device)
        self.pots = torch.zeros(self.n_games, dtype=torch.int32, device=self.device)
        self.stages = torch.zeros(self.n_games, dtype=torch.int32, device=self.device)

        # player state tensors
        # refill/reset stacks
        if self.stacks is None:
            self.stacks = torch.full((self.n_games, self.n_players), self.starting_bbs, dtype=torch.int32, device=self.device)
        else:
            busted = (self.stacks == 0)
            above_max = (self.stacks > self.max_bbs)
            self.stacks[busted] = self.starting_bbs
            self.stacks[above_max] = self.starting_bbs
            if options and options['rotation'] != 0:
                self.stacks = torch.roll(self.stacks, rotation, dims=1)
        
        all_cards = self.deal_players_cards(self.active_players * 2)
        self.hands = torch.full((self.n_games, self.n_players, 2), -1, dtype=torch.int32, device=self.device)
        self.hands[:, :self.active_players] = all_cards.view(self.n_games, self.active_players, 2)

        self.current_round_bet = torch.zeros((self.n_games, self.n_players), dtype=torch.int32, device=self.device)
        self.total_invested = torch.zeros((self.n_games, self.n_players), dtype=torch.int32, device=self.device)
        self.status = torch.full((self.n_games, self.n_players), self.ACTIVE, dtype=torch.int32, device=self.device)
        self.status[:, self.active_players:self.n_players]=self.SITOUT

        self.button = (self.button + 1) % self.active_players if hasattr(self, 'button_pos') else torch.zeros(self.n_games, dtype=torch.int32, device=self.device)
        self.sb = (self.button + 1) % self.active_players
        self.bb = (self.button + 2) % self.active_players
        self.post_blinds()
        self.idx = (self.bb + 1) % self.active_players
        self.highest = torch.ones(self.n_games, dtype=torch.int32, device=self.device)
        self.agg = self.bb.clone()
        self.acted = torch.zeros(self.n_games, dtype=torch.int32, device=self.device)
        self.is_done = torch.zeros(self.n_games, dtype=torch.bool, device=self.device)
        self.offsets=torch.zeros(self.n_games, dtype=torch.int32, device=self.device) # offset tensor for hand strength
        self.raise_amounts=torch.zeros(self.n_games, dtype=torch.int32, device=self.device)

        # buffers that we call in our step function (pre-allocate the tensors)
        self.is_round_over=torch.zeros(self.n_games, dtype=torch.bool, device=self.device)
        self.searching=torch.ones(self.n_games, device=self.device, dtype=torch.bool)
        self.equities=torch.full((self.n_games, self.active_players), self.MIN_EQUITY_RANK+1, device=self.device, dtype=torch.float32)
        self.prev_stacks=torch.zeros(self.n_games, device=self.device, dtype=torch.int32)
        self.prev_invested=torch.zeros(self.n_games, device=self.device, dtype=torch.int32)
        self.active_player_idx=torch.arange(self.active_players, device=self.device)
        self.s=torch.zeros(self.n_games, device=self.device, dtype=torch.float32)
        self.obs=torch.zeros((self.n_games, self.obs_size), device=self.device)
        self.o=torch.arange(1, self.active_players, device=self.device)

        return self.get_obs(), self.get_info()

    def get_obs(self):
        self.obs[:, 0:5]=self.board
        self.obs[:, 5:7]=self.hands[self.g, self.idx]
        self.obs[:, 7:8]=self.stages.unsqueeze(1)
        self.obs[:, 8:9]=((self.idx - self.button) % self.active_players).unsqueeze(1)
        self.obs[:, 9:10]=self.pots.unsqueeze(1)
        self.obs[:, 10:11]=(self.highest - self.current_round_bet[self.g, self.idx]).unsqueeze(1)
        self.obs[:, 11:12]=self.stacks[self.g, self.idx].unsqueeze(1)
        self.obs[:, 12:13]=self.status[self.g, self.idx].unsqueeze(1)

        # Calculate number of opponents
        n_opp = self.active_players - 1
        opp_idx = (self.idx.unsqueeze(1) + self.o) % self.active_players        
        self.obs[:, 13 : 13+n_opp] = self.stacks[self.g].gather(1, opp_idx)
        active_mask = (self.status[self.g].gather(1, opp_idx) == self.ACTIVE)
        self.obs[:, 13+n_opp : 13+2*n_opp] = active_mask.float()
        self.obs[:, 13+2*n_opp : 13+3*n_opp] = self.current_round_bet[self.g].gather(1, opp_idx)
        return self.obs

    def get_info(self):
        return {
            'active_players': self.active_players,
            'stacks': self.stacks
        }

    def post_blinds(self):
        # handle the logic of when we "bet" the blinds for the small and big blinds
        # handle only big blind, small blind gets rounded down to 0
        self.stacks[self.g, self.bb] -= self.bb_amounts
        self.current_round_bet[self.g, self.bb] += self.bb_amounts
        self.total_invested[self.g, self.bb] += self.bb_amounts
        self.pots += self.bb_amounts
        self.status[self.g, self.bb] = torch.where(
            self.stacks[self.g, self.bb] == 0,
            self.ALLIN,
            self.ACTIVE
        ).to(torch.int32)

    def deal_players_cards(self, n_cards):
        """Deal n_cards from each game's deck, deal to players in the preflop_stage"""
        card_idx = self.deck_positions.unsqueeze(1) + torch.arange(n_cards, device=self.device).unsqueeze(0)
        cards = self.decks[self.g.unsqueeze(1), card_idx]
        self.deck_positions += n_cards
        return cards

    def deal_cards(self, g, n_cards):
        # deals cards at a stage past the preflop, no longer dealing to players, but rather to 
        # the board
        card_idx = self.deck_positions[g].unsqueeze(1) + torch.arange(n_cards, device=self.device).unsqueeze(0)        
        cards = self.decks[g.unsqueeze(1), card_idx]
        self.deck_positions[g] += n_cards
        return cards.to(torch.int32)

    def execute_actions(self, actions):
        # executes the actions from the actions tensor for each of the current players
        call_costs=self.highest-(self.current_round_bet[self.g, self.idx])
        active_mask=(self.status[self.g, self.idx] != self.FOLDED) & (self.status[self.g, self.idx] != self.ALLIN) & (self.status[self.g, self.idx] != self.SITOUT)

        # fold
        fold_mask=(actions==0) & active_mask
        self.status[self.g[fold_mask], self.idx[fold_mask]] = self.FOLDED
        self.acted[fold_mask] += 1

        # call / check
        call_mask=(actions==1) & active_mask
        actual_amounts = torch.min(call_costs[call_mask], self.stacks[self.g[call_mask], self.idx[call_mask]])
        self.stacks[self.g[call_mask], self.idx[call_mask]] -= actual_amounts
        self.current_round_bet[self.g[call_mask], self.idx[call_mask]] += actual_amounts
        self.total_invested[self.g[call_mask], self.idx[call_mask]] += actual_amounts
        self.pots[call_mask] += actual_amounts
        self.status[self.g[call_mask], self.idx[call_mask]] = torch.where(
            self.stacks[self.g[call_mask], self.idx[call_mask]] == 0,
            self.ALLIN,
            self.status[self.g[call_mask], self.idx[call_mask]]
        ).to(torch.int32)
        self.acted[call_mask] += 1

        # raising
        raise_mask=(actions >= 2) & active_mask
        self.raise_amounts.fill_(0)
        # min-raise
        min_raise_mask=(actions==2)&raise_mask
        self.raise_amounts[min_raise_mask] = call_costs[min_raise_mask] + self.last_raise_size[min_raise_mask]

        # all in 
        all_in_mask=(actions == 12) & raise_mask
        # handle all ins
        self.raise_amounts[all_in_mask]=self.stacks[self.g[all_in_mask], self.idx[all_in_mask]]

        # pot_sized fraction bets
        potsize_mask=((actions>=3) & (actions<=11)) & raise_mask
        frac_indices=actions[potsize_mask]-3
        fractions=self.raise_fractions[frac_indices]
        self.raise_amounts[potsize_mask] = (self.pots[potsize_mask] * fractions).to(torch.int32)
            
        total_needed=call_costs+self.raise_amounts
        actual_bets=torch.min(total_needed[raise_mask], self.stacks[self.g[raise_mask], self.idx[raise_mask]])
        is_call=actual_bets <= call_costs[raise_mask]
        is_raise=~is_call

        self.stacks[self.g[raise_mask], self.idx[raise_mask]] -= actual_bets
        self.current_round_bet[self.g[raise_mask], self.idx[raise_mask]] += actual_bets
        self.total_invested[self.g[raise_mask], self.idx[raise_mask]] += actual_bets
        self.pots[raise_mask] += actual_bets

        self.status[self.g[raise_mask], self.idx[raise_mask]] = torch.where(
            self.stacks[self.g[raise_mask], self.idx[raise_mask]] == 0,
            self.ALLIN,
            self.status[self.g[raise_mask], self.idx[raise_mask]]
        ).to(torch.int32)

        raise_indices=torch.where(raise_mask)[0]
        pure_raise_indices=raise_indices[is_raise]
        new_bets = self.current_round_bet[self.g[pure_raise_indices], self.idx[pure_raise_indices]]
        self.highest[pure_raise_indices] = torch.max(self.highest[pure_raise_indices], new_bets)
        self.agg[pure_raise_indices] = self.idx[pure_raise_indices]
        self.acted[pure_raise_indices] = 0 # 'new round' of betting on a raise, set acted to 0
        self.acted[raise_mask] += 1 # increase acted for raisers and callers

        if len(pure_raise_indices) > 0:
            actual_raise_sizes = new_bets - self.highest[pure_raise_indices]
            self.last_raise_size[pure_raise_indices] = actual_raise_sizes
            self.highest[pure_raise_indices] = new_bets

    def poker_reward_gpu(self, actions):
        self.s.fill_(0)
        investments = self.prev_stacks - self.stacks[self.g, self.idx]
        stack_changes = self.stacks[self.g, self.idx] - self.prev_stacks
        active_counts = ((self.status == self.ACTIVE) | (self.status == self.ALLIN)).sum(dim=1).float()
        fair_shares = 1.0 / torch.clamp(active_counts, min=1.0)
        call_costs = torch.maximum(torch.zeros_like(self.highest), self.highest - self.prev_invested)
        
        e = self.equities[self.g, self.idx]
        m = ((e * self.pots) - investments) + stack_changes
        o = call_costs / (self.pots + call_costs + 1e-6)
        
        # Use pre-allocated masks and buffer
        call_mask = (actions == 1)
        fold_mask = (actions == 0)
        raise_mask = (actions >= 2)
        
        self.s[call_mask] = (e[call_mask] - o[call_mask]) * self.pots[call_mask]
        self.s[fold_mask] = (o[fold_mask] - e[fold_mask]) * self.pots[fold_mask]
        self.s[raise_mask] = (e[raise_mask] - fair_shares[raise_mask]) * self.pots[raise_mask]
        
        return self.alpha * torch.tanh(((self.w1 * m) + (self.w2 * self.s)) / self.K)

    def resolve_fold_winners(self):
        # calculate the winners of the pot and add to their stack
        ended=self.is_done[self.g]
        won=((self.status[self.g]==self.ACTIVE) | (self.status[self.g] == self.ALLIN)).sum(dim=1) == 1
        gg=self.g[ended & won]
        survivor=((self.status[gg] == self.ACTIVE) | (self.status[gg] == self.ALLIN)).long().argmax(dim=1)
        self.stacks[gg, survivor] += self.pots[gg]
        self.pots[gg]=0

    def resolve_terminated_games(self):
        # resolves the terminated games
        # games can either be terminated post-river or terminated 'early' where there are no players left to act
        # filter by games with more than 1 player left (already implemented win by fold in resolve_fold_winners)
        
        # get all of the terminated games to the river stage
        needs_resolution = (self.stages[self.g] == 4)
        g_not_done = self.g[needs_resolution] 
        if len(g_not_done) == 0: return

        multiple_players = ((self.status[self.g] == self.ACTIVE) | (self.status[self.g] == self.ALLIN)).sum(dim=1) > 1
        if not multiple_players.any(): return
        flop_mask = (self.stages[self.g] == 1) & multiple_players
        turn_mask = (self.stages[self.g] == 2) & multiple_players

        flop_games = self.g[flop_mask]   
        self.deck_positions[flop_games] += 1  # burn
        turn_cards = self.deal_cards(flop_games, 1)
        self.board[flop_games, 3] = turn_cards.squeeze(1) 
        self.deck_positions[flop_games] += 1  # burn
        river_cards = self.deal_cards(flop_games, 1)
        self.board[flop_games, 4] = river_cards.squeeze(1) 

        turn_games = self.g[turn_mask]
        self.deck_positions[turn_games] += 1  # burn
        river_cards = self.deal_cards(turn_games, 1)
        self.board[turn_games, 4] = river_cards.squeeze(1)

        showdown_mask = multiple_players
        if not showdown_mask.any():return

        showdown_games = self.g[showdown_mask]
        n_showdown = showdown_games.shape[0]
        
        boards = self.board[showdown_games]  # [n_showdown, 5]
        hands = self.hands[showdown_games, :self.active_players]  # [n_showdown, active_players, 2]

        hands_7 = torch.zeros((n_showdown, self.active_players, 7), dtype=torch.int32, device=self.device)
        hands_7[:, :, 0:2] = hands
        hands_7[:, :, 2:7] = boards.unsqueeze(1)
    
        # Evaluate all hands using hand_ranks lookup table
        batch_size = n_showdown * self.active_players
        hands_flat = hands_7.view(batch_size, 7)
        offsets = torch.full((batch_size,), 53, dtype=torch.int32, device=self.device)
    
        for card_idx in range(7):
            indices = offsets + hands_flat[:, card_idx]
            offsets = self.hand_ranks[indices]
    
        hand_ranks = offsets.view(n_showdown, self.active_players)
        player_status = self.status[showdown_games, :self.active_players]
        eligible_mask = (player_status == self.ACTIVE) | (player_status == self.ALLIN)
        hand_ranks = torch.where(eligible_mask, hand_ranks, torch.tensor(-1, device=self.device))
        best_ranks = hand_ranks.min(dim=1, keepdim=True).values  # [n_showdown, 1]
        winner_mask = (hand_ranks == best_ranks)  # [n_showdown, active_players] - True for winners
    
        # Count winners per game (for split pots)
        num_winners = winner_mask.sum(dim=1)  # [n_showdown]
    
        # Distribute pot to winners
        pots = self.pots[showdown_games].unsqueeze(1)  # [n_showdown, 1]
        winnings = (pots * winner_mask.float()) / num_winners.unsqueeze(1)  # [n_showdown, active_players]
        self.stacks[showdown_games.unsqueeze(1), torch.arange(self.active_players, device=self.device)] += winnings.to(torch.int32)
        self.pots[showdown_games] = 0
        self.stages[showdown_games] = 5

    def calculate_equities(self):
        # creates the equities tensor (might have to move before execute actions function in step function)
        river_mask, turn_mask, flop_mask = (self.stages == 3), (self.stages==2), (self.stages==1)
        counts = torch.bincount(self.stages, minlength=4)

        if not river_mask.any() and not turn_mask.any() and not flop_mask.any():
            pass
        else:
            n_river_games, n_turn_games, n_flop_games = counts[3], counts[2], counts[1],

            if n_river_games > 0:
                river_boards = self.board[river_mask]
                river_hands = self.hands[river_mask, :self.active_players]
                
                hands_7_cards_river = torch.zeros(
                    (n_river_games, self.active_players, 7),
                    dtype=torch.int32,
                    device=self.device
                )
                
                hands_7_cards_river[:, :, 0] = river_hands[:, :, 0]
                hands_7_cards_river[:, :, 1] = river_hands[:, :, 1]
                hands_7_cards_river[:, :, 2:7] = river_boards.unsqueeze(1)
                
                # Evaluate river hands
                batch_size = n_river_games * self.active_players
                hands_flat = hands_7_cards_river.view(batch_size, 7)
                river_offsets = torch.full((batch_size,), 53, dtype=torch.int32, device=self.device)
                
                for card_idx in range(7):
                    indices = river_offsets + hands_flat[:, card_idx]
                    river_offsets = self.hand_ranks[indices]
                
                ranks = river_offsets.view(n_river_games, self.active_players)
                self.equities[river_mask] = ranks.to(torch.float32)
                self.equities[river_mask] = ((self.equities[river_mask] - self.MIN_TURN_RIVER_EQUITY) / self.equity_turn_denom).clamp(0.0, 1.0)

            if n_turn_games > 0:
                turn_boards, turn_hands = self.board[turn_mask], self.hands[turn_mask, :self.active_players]
                hands_6_cards=torch.zeros(
                    (n_turn_games, self.active_players, 6), dtype=torch.int32,
                    device=self.device
                )

                hands_6_cards[:, :, 0] = turn_hands[:, :, 0]
                hands_6_cards[:, :, 1] = turn_hands[:, :, 1]
                hands_6_cards[:, :, 2:] = turn_boards[:, 0:4].unsqueeze(1)
                batch_size=n_turn_games*self.active_players
                hands_flat=hands_6_cards.view(batch_size, 6)
                turn_offsets = torch.full((batch_size,), 53, dtype=torch.int32, device=self.device)
                
                for card_idx in range(6):
                    indices = turn_offsets + hands_flat[:, card_idx]
                    turn_offsets = self.hand_ranks[indices]
                
                final_values=self.hand_ranks[turn_offsets]
                ranks = final_values.view(n_turn_games, self.active_players)
                self.equities[turn_mask] = ranks.to(torch.float32)
                self.equities[turn_mask] = ((self.equities[turn_mask] - self.MIN_TURN_RIVER_EQUITY) / self.equity_turn_denom).clamp(0.0, 1.0)

            if n_flop_games > 0:
                flop_boards, flop_hands = self.board[flop_mask], self.hands[flop_mask, :self.active_players]
                hands_5_cards = torch.zeros((n_flop_games, self.active_players, 5), dtype=torch.int32, device=self.device)
                hands_5_cards[:, :, 0:2] = flop_hands
                hands_5_cards[:, :, 2:] = flop_boards[:, 0:3].unsqueeze(1)  # flop is first 3 board cards
                batch_size = n_flop_games * self.active_players
                hands_flat = hands_5_cards.view(batch_size, 5)
                flop_offsets = torch.full((batch_size,), 53, dtype=torch.int32, device=self.device)
                for card_idx in range(5):
                    indices = flop_offsets + hands_flat[:, card_idx]
                    flop_offsets = self.hand_ranks[indices]
                offsets = self.hand_ranks[flop_offsets]      # first +0
                final_ranks = self.hand_ranks[offsets]  # second +0
                ranks = final_ranks.view(n_flop_games, self.active_players)
                self.equities[flop_mask] = ranks.to(torch.float32)
                self.equities[flop_mask] = ((self.equities[flop_mask] - self.MIN_FLOP_EQUITY) / self.equity_flop_denom).clamp(0.0, 1.0)

        preflop_mask = self.stages == 0
        self.equities[preflop_mask] = 0.5
    
    def step(self, actions):
        # step function to handle logic of n_games actions at once
        # get game indices ready
        self.prev_stacks.copy_(self.stacks[self.g, self.idx])
        self.prev_invested.copy_(self.current_round_bet[self.g, self.idx])

        # 1) calculate the equties of players hands
        self.equities.fill_(self.MAX_EQUITY_RANK+1)
        self.calculate_equities()

        # 2) execute actions 
        self.execute_actions(actions)
        truly_active = ((self.status == self.ACTIVE)).sum(dim=1)
        all_allin_or_folded = (truly_active == 0)

        # 3) find next player to act in current round
        next_player_idx=self.idx.clone()
        self.is_round_over.fill_(False)
        self.is_round_over[all_allin_or_folded]=True
        self.searching[:]=~self.is_round_over

        # NOTE: need way to parallelize the below code, eliminate this ugly for loop
        for _ in range(self.active_players):
            next_player_idx[self.searching]=(next_player_idx[self.searching]+1)%self.active_players
            # round over check
            back_to_agg=(next_player_idx==self.agg)
            truly_active_counts = (self.status == self.ACTIVE).sum(dim=1)  # FIXED
            all_acted=(self.acted >= truly_active_counts)
            round_over=back_to_agg & all_acted & self.searching
            self.is_round_over |= round_over
            self.searching[round_over]=False

            # still eligible player left check
            player_status=self.status[self.g, next_player_idx]
            is_eligible=((player_status==self.ACTIVE) | (player_status==self.ALLIN)) & self.searching
            self.searching[is_eligible]=False

        no_over_mask=~self.is_round_over
        self.idx[no_over_mask]=next_player_idx[no_over_mask]
    
        # 3) handle round transitions & game ends
        active_counts=((self.status == self.ACTIVE) | (self.status == self.ALLIN)).sum(dim=1)
        early_term=(active_counts <= 1)&self.is_round_over
        self.is_done[early_term]=True

        transition_mask=self.is_round_over&~early_term
        self.last_raise_size[transition_mask] = 1
        if transition_mask.any():
            g_over=self.g[transition_mask]
            self.stages[transition_mask]+=1
            self.highest[transition_mask] = 0
            self.agg[transition_mask] = (self.button[transition_mask] + 1) % self.active_players
            self.acted[transition_mask] = 0
            self.current_round_bet[g_over, :]=0

            flop_mask = (self.stages[transition_mask] == 1)
            turn_mask = (self.stages[transition_mask] == 2)
            river_mask = (self.stages[transition_mask] == 3)

            post_river=(self.stages[transition_mask]>3)
            post_river_games = g_over[post_river]
            self.is_done[post_river_games] = True
            self.stages[post_river_games] = 4

            # flop mask
            street_games = g_over[flop_mask]
            self.deck_positions[street_games] += 1
            self.board[street_games, 0:3] = self.deal_cards(street_games, 3)
            
            # turn mask
            street_games = g_over[turn_mask]
            self.deck_positions[street_games] += 1
            self.board[street_games, 3] = self.deal_cards(street_games, 1).squeeze(1)
            
            # river mask
            street_games = g_over[river_mask]
            self.deck_positions[street_games] += 1
            self.board[street_games, 4] = self.deal_cards(street_games, 1).squeeze(1)

        # 4) resolve fold winners, resolve games that are over with more than 1 active player currently
        self.resolve_fold_winners()
        self.resolve_terminated_games()

        all_done = self.is_done[self.g]
        self.current_round_bet[self.g[all_done], :]=0
        self.total_invested[self.g[all_done], :]=0
        self.highest[self.g[all_done]]=0        
        
        # 5) Calculate the actual reward
        rewards = self.poker_reward_gpu(actions=actions)
        return self.get_obs(), rewards, self.is_done, self.is_truncated, self.get_info()

