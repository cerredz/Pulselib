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
    MAX_EQUITY_NUM=7462

    def __init__(self, device, agents, n_players=6, max_players=10, n_games=100, starting_bbs=100, max_bbs=1000, w1=.5, w2=.5, K=20):
        super().__init__()

        self.w1=w1
        self.w2=w2
        self.K=K
        # env 
        self.device=device
        self.agents=agents
        self.n_players=n_players
        self.n_games=n_games
        self.starting_bbs=starting_bbs
        self.max_bbs=max_bbs
        self.max_players=max_players

        # action space / observation space
        self.action_space=spaces.Discrete(self.NUM_ACTIONS)
        self.raise_fractions=torch.tensor([0.25, 0.33, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00], device=self.device)
        self.obs_size=12+((self.max_players-1)*3)
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

        print("hand ranks: ", self.hand_ranks)
        print(self.hand_ranks.shape)

    def set_agents(self, agents):
        self.agents=agents

    def reset(self, seed=None, options=None, rotation=0):
        super().reset(seed=seed)
        if options and options['active_players']:
            self.active_players=torch.randint(2, self.n_players+1, (1,), device=self.device).item()
        else: self.active_players=self.n_players

        # game state tensors
        self.decks=torch.rand(self.n_games, 52, device=self.device).argsort(dim=1)
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
        return self.get_obs(), self.get_info()

    def get_obs(self):
        g = torch.arange(self.n_games, device=self.device)
        obs_parts = [
            self.board,  # [n_games, 5]
            self.hands[g, self.idx],  # [n_games, 2]
            self.stages.unsqueeze(1),  # [n_games, 1]
            ((self.idx - self.button) % self.active_players).unsqueeze(1),  # [n_games, 1]
            self.pots.unsqueeze(1),  # [n_games, 1]
            (self.highest - self.current_round_bet[g, self.idx]).unsqueeze(1),  # [n_games, 1]
            self.stacks[g, self.idx].unsqueeze(1)  # [n_games, 1]
        ]
        
        for i in range(1, self.max_players):
            if i < self.active_players:
                opp = (self.idx + i) % self.active_players
                obs_parts.extend([
                    self.stacks[g, opp].unsqueeze(1),
                    (self.status[g, opp] == self.ACTIVE).float().unsqueeze(1),
                    self.current_round_bet[g, opp].unsqueeze(1)
                ])
            else:
                obs_parts.extend([
                    torch.zeros(self.n_games, 1, device=self.device),
                    torch.zeros(self.n_games, 1, device=self.device),
                    torch.zeros(self.n_games, 1, device=self.device)
                ])

        return torch.cat(obs_parts, dim=1)

    def get_info(self):
        return {
            'active_players': self.active_players
        }

    def post_blinds(self):
        # handle the logic of when we "bet" the blinds for the small and big blinds
        # handle only big blind, small blind gets rounded down to 0
        game_idx=torch.arange(self.n_games, device=self.device)
        bb_amount=torch.ones(self.n_games, dtype=torch.int32, device=self.device)
        self.stacks[game_idx, self.bb] -= bb_amount
        self.current_round_bet[game_idx, self.bb] += bb_amount
        self.total_invested[game_idx, self.bb] += bb_amount
        self.pots += bb_amount
        self.status[game_idx, self.bb] = torch.where(
            self.stacks[game_idx, self.bb] == 0,
            self.ALLIN,
            self.ACTIVE
        ).to(torch.int32)

    def deal_players_cards(self, n_cards):
        """Deal n_cards from each game's deck, deal to players in the preflop_stage"""
        game_idx = torch.arange(self.n_games, device=self.device).unsqueeze(1)
        card_idx = self.deck_positions.unsqueeze(1) + torch.arange(n_cards, device=self.device).unsqueeze(0)
        cards = self.decks[game_idx, card_idx]
        self.deck_positions += n_cards
        return cards

    def deal_cards(self, g, n_cards):
        # deals cards at a stage past the preflop, no longer dealing to players, but rather to 
        # the board
        n=len(g)
        if n==0: return torch.empty(0, n_cards, dtype=torch.int32, device=self.device)

        card_idx = self.deck_positions[g].unsqueeze(1) + torch.arange(n_cards, device=self.device).unsqueeze(0)        
        cards = self.decks[g.unsqueeze(1), card_idx]
        self.deck_positions[g] += n_cards
        return cards.to(torch.int32)

    def execute_actions(self, g, actions):
        # executes the actions from the actions tensor for each of the current players
        call_costs=self.highest-(self.current_round_bet[g, self.idx])
        active_mask=(self.status[g, self.idx] != self.FOLDED) & (self.status[g, self.idx] != self.ALLIN) & (self.status[g, self.idx] != self.SITOUT)

        # fold
        fold_mask=(actions==0) & active_mask
        self.status[g[fold_mask], self.idx[fold_mask]] = self.FOLDED
        self.acted[fold_mask] += 1

        # call / check
        call_mask=(actions==1) & active_mask
        actual_amounts = torch.min(call_costs[call_mask], self.stacks[g[call_mask], self.idx[call_mask]])
        self.stacks[g[call_mask], self.idx[call_mask]] -= actual_amounts
        self.current_round_bet[g[call_mask], self.idx[call_mask]] += actual_amounts
        self.total_invested[g[call_mask], self.idx[call_mask]] += actual_amounts
        self.pots[call_mask] += actual_amounts
        self.status[g[call_mask], self.idx[call_mask]] = torch.where(
            self.stacks[g[call_mask], self.idx[call_mask]] == 0,
            self.ALLIN,
            self.status[g[call_mask], self.idx[call_mask]]
        ).to(torch.int32)
        self.acted[call_mask] += 1

        # raising
        raise_mask=(actions >= 2) & active_mask
        raise_amounts=torch.zeros(self.n_games, dtype=torch.int32, device=self.device)
        pt=self.pots+call_costs
        # min-raise
        min_raise_mask=(actions==2)&raise_mask
        raise_amounts[min_raise_mask] = torch.max(
            torch.ones(min_raise_mask.sum(), dtype=torch.int32, device=self.device), # case where no one has bet yet
            call_costs[min_raise_mask]
        )
        # all in 
        all_in_mask=(actions == 12) & raise_mask
        # handle all ins
        if all_in_mask.any():
            raise_amounts[all_in_mask]=self.stacks[g[all_in_mask], self.idx[all_in_mask]]

        # pot_sized fraction bets
        potsize_mask=((actions>=3) & (actions<= 11)) & raise_mask
        frac_indices=actions[potsize_mask]-3
        fractions=self.raise_fractions[frac_indices]
        raise_amounts[potsize_mask] = (self.pots[potsize_mask] * fractions).to(torch.int32)
            
        total_needed=call_costs+raise_amounts
        actual_bets=torch.min(total_needed[raise_mask], self.stacks[g[raise_mask], self.idx[raise_mask]])
        is_call=actual_bets <= call_costs[raise_mask]
        is_raise=~is_call

        self.stacks[g[raise_mask], self.idx[raise_mask]] -= actual_bets
        self.current_round_bet[g[raise_mask], self.idx[raise_mask]] += actual_bets
        self.total_invested[g[raise_mask], self.idx[raise_mask]] += actual_bets
        self.pots[raise_mask] += actual_bets

        self.status[g[raise_mask], self.idx[raise_mask]] = torch.where(
            self.stacks[g[raise_mask], self.idx[raise_mask]] == 0,
            self.ALLIN,
            self.status[g[raise_mask], self.idx[raise_mask]]
        ).to(torch.int32)

        raise_indices=torch.where(raise_mask)[0]
        pure_raise_indices=raise_indices[is_raise]
        new_bets = self.current_round_bet[g[pure_raise_indices], self.idx[pure_raise_indices]]
        self.highest[pure_raise_indices] = torch.max(self.highest[pure_raise_indices], new_bets)
        self.agg[pure_raise_indices] = self.idx[pure_raise_indices]
        self.acted[pure_raise_indices] = 0 # 'new round' of betting on a raise, set acted to 0
        self.acted[raise_mask] += 1 # increase acted for raisers and callers

    def calculate_showdown_winners(self, g):
        # need way to calculate winners on the gpu when multiple people are left
        # here is where we will do it

        # algorithmic approach:
            # we need to get the games that are past the post flop and have more than 1 active player
            # board has 5 cards, each player has 2 cards
            # card value split by suit -> 0-12=clubs, 13-25=diamonds, etc, each category is ascending order from 2->ace
            # need masks for each type of hand in descending order of hands (depending on card values above)
                # royal flush mask
                # straight flush mask
                # four of a kind mask, etc
            # using masks, evaluate strength of each hand in the games we need to, handle side pots, and 
                # award the pots to the winners
                # hardest parts are SIDE POTS and TIE BREAKERS
        
        # find showdown games
        pass

    def resolve_winner_by_fold(self):
        pass

    def poker_reward_gpu(self, equities, pots, investments, stack_changes, call_costs, fair_shares, actions):
        m=.5*((equities*pots)-investments)+.5*stack_changes
        o=call_costs/(pots+call_costs+1e-6)
        s=torch.zeros_like(m)
        fold_mask, call_mask, raise_mask=(actions==0), (actions==1), (actions>=2)

        s[call_mask] = (equities[call_mask] - o[call_mask]) * pots[call_mask]
        s[fold_mask] = (o[fold_mask] - equities[fold_mask]) * pots[fold_mask]
        s[raise_mask] = equities[raise_mask] - fair_shares[raise_mask] * pots[raise_mask] * 1.2

        weighted = (self.w1 * m) + (self.w2 * s)
        r=self.n_players * torch.tanh(weighted/self.K)
        return r

    def resolve_fold_winners(self, g):
        # recieves the mask of the terminated games, for these games we are going to 
        # calculate the winners of the pot and add to their stack
        ended=self.is_done[g]
        won=((self.status[g]==self.ACTIVE) | (self.status[g] == self.ALLIN)).sum(dim=1) == 1
        gg=g[ended & won]
        if gg.any():
            survivor=((self.status[gg] == self.ACTIVE) | (self.status[gg] == self.ALLIN)).long().argmax(dim=1)
            self.stacks[gg, survivor] += self.pots[gg]
            self.pots[gg]=0

    def calculate_equities(self):
        # calculate the equities of all players using the precomputed tables:
        # for anyone that has won, award them the chips
        
        equities=torch.zeros((self.n_games, self.active_players), device=self.device, dtype=torch.int32)
        # river equity calculation
        river_mask=(self.stages==3)
        active_counts = ((self.status == self.ACTIVE) | (self.status == self.ALLIN)).sum(dim=1)
        multi_player_mask=active_counts>1
        river_eval_mask=river_mask & multi_player_mask

        if river_eval_mask.any():
            eval_games=torch.where(river_eval_mask)
            

        # h1, h2 = self.hands[river, active_river, 0], self.hands[river, active_river, 1]
        #boards=self.board[river]

        pass

    def step(self, actions):
        # step function to handle logic of n_games actions at once
        # get game indices ready
        g=torch.arange(self.n_games, device=self.device)
        prev_stacks=self.stacks[g, self.idx].clone()
        prev_invested=self.current_round_bet[g, self.idx].clone()

        # 1)  execute actions
        self.execute_actions(g, actions)

        truly_active = ((self.status == self.ACTIVE)).sum(dim=1)
        all_allin_or_folded = (truly_active == 0)

        # 2) find next player to act in current round
        
        next_player_idx=self.idx.clone()
        is_round_over=torch.zeros(self.n_games, dtype=torch.bool, device=self.device)
        is_round_over[all_allin_or_folded] = True  # Round over if no one can act

        searching=~is_round_over
        searching=torch.ones(self.n_games, dtype=torch.bool, device=self.device)
        searching[all_allin_or_folded] = False  # Don't search if already over

        # NOTE: need way to parallelize the below code, eliminate this ugly for loop
        for _ in range(self.active_players):
            next_player_idx[searching]=(next_player_idx[searching]+1)%self.active_players
            # round over check
            back_to_agg=(next_player_idx==self.agg)
            truly_active_counts = (self.status == self.ACTIVE).sum(dim=1)  # FIXED
            all_acted=(self.acted >= truly_active_counts)
            round_over=back_to_agg & all_acted & searching
            is_round_over |= round_over
            searching[round_over]=False

            # still eligible player left check
            player_status=self.status[g, next_player_idx]
            is_eligible=((player_status==self.ACTIVE) | (player_status==self.ALLIN)) & searching
            searching[is_eligible]=False

        no_over_mask=~is_round_over
        self.idx[no_over_mask]=next_player_idx[no_over_mask]
    
        # find the next players in each round
        """
        offsets=torch.arange(1, self.active_players+1, dtype=torch.int32, device=self.device)
        candidates=(next_player_idx.unsqueeze(1)+offsets)%self.active_players  
        candidates_status=self.status[g.unsqueeze(1), candidates]
        is_eligible=(candidates_status==self.ACTIVE) | (candidates_status == self.ALLIN)
        has_eligible=is_eligible.any(dim=1)
        first_eligible_idx = torch.where(
            has_eligible,
            is_eligible.float().argmax(dim=1),
            torch.zeros_like(next_player_idx)
        )
        next_player=candidates[g, first_eligible_idx]

        back_to_agg = (next_player == self.agg)
        truly_active_counts = (self.status == self.ACTIVE).sum(dim=1)
        all_acted = (self.acted >= truly_active_counts)
        round_over_now = ((back_to_agg & all_acted) | ~has_eligible) & searching
        is_round_over |= round_over_now
        should_update = searching & has_eligible & ~round_over_now
        next_player_idx = torch.where(should_update, next_player, next_player_idx)
        no_over_mask = ~is_round_over
        self.idx[no_over_mask] = next_player_idx[no_over_mask]
        """
        # 3) handle round transitions & game ends
        terminated = torch.zeros(self.n_games, dtype=torch.bool, device=self.device)
        stack_changes = torch.zeros(self.n_games, dtype=torch.int32, device=self.device)
        active_counts=((self.status == self.ACTIVE) | (self.status == self.ALLIN)).sum(dim=1)
        early_term=(active_counts <= 1)&is_round_over
        terminated[early_term]=True

        transition_mask=is_round_over&~early_term
        if transition_mask.any():
            g_over=g[transition_mask]
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
            terminated[post_river_games] = True

            for street_mask, n_cards, start_col in [(flop_mask, 3, 0), (turn_mask, 1, 3), (river_mask, 1, 4)]:
                if street_mask.any():
                    street_games=g_over[street_mask]
                    self.deck_positions[street_games]+= 1 # burn 1 card
                    cards=self.deal_cards(street_games, n_cards)
                    self.board[street_games, start_col:start_col+n_cards]=cards

        # showdown resolution, resolve winner by fold, stack change calculation for rewards
        self.is_done=terminated.clone()
        self.resolve_fold_winners(g)
        # self.resolve_post_river_games()

        # calculate reward
        equities=torch.full((self.n_games, ), .5, dtype=torch.float32, device=self.device)
        # for equity calculation, we will be using (for right now) a 7-card lookup table 
            # takes into account a players hand and the board
        e=self.calculate_equities()

        stack_changes=self.stacks[g, self.idx]-prev_stacks
        active_counts = ((self.status == self.ACTIVE) | (self.status == self.ALLIN)).sum(dim=1).float()  # [n_games]
        fair_shares = 1.0 / torch.clamp(active_counts, min=1.0)  # [n_games]
        investment_this_step=prev_stacks-self.stacks[g, self.idx]
        call_costs = torch.maximum(torch.zeros_like(self.highest), self.highest - prev_invested)  # [n_games] non-negative
        
        rewards = self.poker_reward_gpu(
            equities=equities,
            pots=self.pots,
            investments=investment_this_step,
            stack_changes=stack_changes,
            call_costs=call_costs,
            fair_shares=fair_shares,
            actions=actions
        )
        
        #terminated=torch.ones(self.n_games, device=self.device, dtype=torch.bool)
        truncated = torch.zeros(self.n_games, dtype=torch.bool, device=self.device)  # All False
        info = {}

        return self.get_obs(), rewards, terminated, truncated, info

