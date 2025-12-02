from multiprocessing import Value
from operator import truediv
from re import L
from environments.Poker.utils import Agent, validate_agents
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import List

class PokerGPU(gym.Env):
    metadata = {'render.modes': ['human']}
    NUM_ACTIONS=13
    ACTIVE, FOLDED, ALLIN = 0, 1, 2
    STATE_SPACE=28 # details in depth_notes in rl folder

    def __init__(self, device, agents, n_players=6, n_games=100, starting_bbs=100, max_bbs=1000):
        super().__init__()

        # env 
        self.device=device
        self.agents=validate_agents(agents=agents)
        self.n_players=n_players
        self.n_games=n_games
        self.starting_bbs=starting_bbs
        self.max_bbs=max_bbs

        # action space / observation space
        self.action_space=spaces.Discrete(self.NUM_ACTIONS)
        self.raise_fractions=torch.tensor([0.25, 0.33, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00], device=self.device)
        self.obs_size=12+((self.n_players-1)*3)
        self.observation_space = spaces.Box(low=0, high=10000, shape=(self.obs_size,), dtype=torch.float32)

        # Initialize state tensors as None (will be set in reset)
        self.stacks = None
        self.total_busted=0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # game state tensors
        self.decks=torch.stack([torch.randperm(52, device=self.device) for _ in range(self.n_games)])
        self.deck_positions=torch.zeros(self.n_games, device=self.device, dtype=torch.uint8)
        
        self.board = torch.full((self.n_games, 5), -1, dtype=torch.uint8, device=self.device)
        self.pots = torch.zeros(self.n_games, dtype=torch.uint32, device=self.device)
        self.stages = torch.zeros(self.n_games, dtype=torch.uint32, device=self.device)

        # player state tensors
        # refill/reset stacks
        if self.stacks is None:
            self.stacks = torch.full((self.n_games, self.n_players), self.starting_bbs, dtype=torch.float32, device=self.device)
        else:
            busted = (self.stacks == 0)
            above_max = (self.stacks > self.max_bbs)
            self.stacks[busted] = self.starting_bbs
            self.stacks[above_max] = self.starting_bbs
            self.total_busted+=busted.sum().items()
        
        self.hands = self.deal_players_cards(self.n_players * 2).view(self.n_games, self.n_players, 2)

        self.current_round_bet = torch.zeros((self.n_games, self.n_players), dtype=torch.uint32, device=self.device)
        self.total_invested = torch.zeros((self.n_games, self.n_players), dtype=torch.uint32, device=self.device)
        self.status = torch.full((self.n_games, self.n_players), self.ACTIVE, dtype=torch.uint8, device=self.device)

        self.button = (self.button + 1) % self.n_players if hasattr(self, 'button_pos') else torch.zeros(self.n_games, dtype=torch.long, device=self.device)
        self.sb = (self.button_pos + 1) % self.n_players
        self.bb = (self.button_pos + 2) % self.n_players

        self.post_blinds()

        self.idx = (self.bb_pos + 1) % self.n_players
        self.highest = torch.ones(self.n_games, dtype=torch.uint32, device=self.device)
        self.agg = self.bb.clone()
        self.acted = torch.zeros(self.n_games, dtype=torch.uint8, device=self.device)

        self.is_done = torch.zeros(self.n_games, dtype=torch.bool, device=self.device)
        return self.get_obs(), self.get_info()

    def get_obs(self):
        g = torch.arange(self.n_games, device=self.device)
        obs_parts = [
            self.board,  # [n_games, 5]
            self.hands[g, self.idx],  # [n_games, 2]
            self.stages.unsqueeze(1),  # [n_games, 1]
            ((self.idx - self.button) % self.n_players).unsqueeze(1),  # [n_games, 1]
            self.pots.unsqueeze(1),  # [n_games, 1]
            (self.highest - self.current_round_bet[g, self.idx]).unsqueeze(1),  # [n_games, 1]
            self.stacks[g, self.idx].unsqueeze(1)  # [n_games, 1]
        ]
        for i in range(1, self.n_players):
            opp = (self.idx + i) % self.n_players
            obs_parts.extend([
                self.stacks[g, opp].unsqueeze(1),
                (self.status[g, opp] == self.ACTIVE).float().unsqueeze(1),
                self.current_round_bet[g, opp].unsqueeze(1)
            ])
        return torch.cat(obs_parts, dim=1)

    def get_info(self):
        pass

    def post_blinds(self):
        # handle the logic of when we "bet" the blinds for the small and big blinds
        # handle only big blind, small blind gets rounded down to 0
        game_idx=torch.arange(self.n_games, device=self.device)
        bb_amount=torch.ones(self.n_games, dtype=torch.uint8, device=self.device)
        self.stacks[game_idx, self.bb] -= bb_amount
        self.current_round_bet[game_idx, self.bb] += bb_amount
        self.total_invested[game_idx, self.bb] += bb_amount
        self.pots += bb_amount
        self.status[game_idx, self.bb] = torch.where(
            self.stacks[game_idx, self.bb] == 0,
            self.ALLIN,
            self.ACTIVE
        )

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
        if n==0: return torch.empty(0, n_cards, dtype=torch.uint8, device=self.device)

        card_idx = self.deck_positions[g].unsqueeze(1) + torch.arange(n_cards, device=self.device).unsqueeze(0)        
        cards = self.decks[g.unsqueeze(1), card_idx]
        self.deck_positions[g] += n_cards
        return cards

    def execute_actions(self, g, actions):
        # executes the actions from the actions tensor for each of the current players
        call_costs=self.highest-(self.current_round_bet[g, self.idx])
        active_mask=(self.status[g, self.idx] != self.FOLDED) & (self.status[g, self.idx] != self.ALLIN)

        # fold
        fold_mask=(actions==0) & active_mask
        if fold_mask.any():
            self.status[g[fold_mask], self.idx[fold_mask]] = self.FOLDED
            self.acted[fold_mask] += 1

        # call / check
        call_mask=(actions==1) & active_mask
        if call_mask.any():
            actual_amounts = torch.min(call_costs[call_mask], self.stacks[g[call_mask], self.idx[call_mask]], device=self.device)
            self.stacks[g[call_mask], self.idx[call_mask]] -= actual_amounts
            self.current_round_bet[g[call_mask], self.idx[call_mask]] += actual_amounts
            self.total_invested[g[call_mask], self.idx[call_mask]] += actual_amounts
            self.pots[call_mask] += actual_amounts
            self.status[g[call_mask], self.idx[call_mask]] = torch.where(
                self.stacks[g[call_mask], self.idx[call_mask]] == 0,
                self.ALLIN,
                self.status[g[call_mask], self.idx[call_mask]]
            )
            self.acted[call_mask] += 1

        # raising
        raise_mask=(actions >= 2) & active_mask
        if raise_mask.any():
            raise_amounts=torch.zeros(self.n_games, dtype=torch.uint32, device=self.device)
            pt=self.pots+call_costs

            # min-raise
            min_raise_mask=(actions==2)&raise_mask
            if min_raise_mask.any():
                raise_amounts[min_raise_mask] = torch.max(
                    torch.ones(min_raise_mask.sum(), device=self.device), # case where no one has bet yet
                    call_costs[min_raise_mask]
                )

            # all in 
            all_in_mask=(actions == 12) & raise_mask
            # handle all ins
            if all_in_mask.any():
                raise_amounts[all_in_mask]=self.stacks[g[all_in_mask], self.idx[all_in_mask]]

            # pot_sized fraction bets
            potsize_mask=(actions>=3 & actions<= 11) & raise_mask
            if potsize_mask.any():
                frac_indices=actions[potsize_mask]-3
                fractions=self.raise_fractions[frac_indices]
                raise_amounts[potsize_mask] = (self.pots[potsize_mask] * fractions).to(torch.uint32)
            
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
            )

            raise_indices=torch.where(raise_mask)[0]
            pure_raise_indices=raise_indices[is_raise]
            if len(pure_raise_indices) > 0:
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
        active_counts=((self.status==self.ACTIVE) | (self.status==self.ALLIN)).sum(dim=1)
        showdown_mask=(self.stages>3) & (active_counts>1)
        if not showdown_mask.any(): return # no showdowns in any games

        # find active players in these games
        g_show=torch.where(showdown_mask)[0]
        num_show=len(g_show)
        
        player_mask = ((self.status[g_show] == self.ACTIVE) | (self.status[g_show] == self.ALLIN))
        full_hands = torch.cat([self.hands[g_show], self.board[g_show].unsqueeze(1).expand(-1, self.n_players, -1)], dim=2)  # [num_show, n_players, 7]
        valid_cards=(full_hands>0)
        full_hands[~valid_cards]=0

        ranks, suits = full_hands%13, full_hands//13

        sorted_ranks, _= torch.sort(ranks, dim=2, device=self.device, descending=True) # [num_show, n_players, 7]

        pass

    def step(self, actions):
        # step function to handle logic of n_games actions at once
        # get game indices ready
        g=torch.arange(self.n_games, device=self.device)
        #prev_stacks=self.stacks[g, self.idx].clone()
        #prev_invested=self.current_round_bet[g, self.idx].clone()

        # 1)  execute actions
        self.execute_actions(g, actions)

        # 2) find next player to act in current round
        active_counts=((self.status == self.ACTIVE) | (self.status == self.ALLIN)).sum(dim=1)
        next_player_idx=self.idx.clone()
        is_round_over=torch.zeros(self.n_games, dtype=torch.bool, device=self.device)
        searching=torch.ones(self.n_games, dtype=torch.bool, device=self.device)
        for _ in range(self.n_games):
            next_player_idx[searching]=(next_player_idx[searching]+1)%self.n_players
            # round over check
            back_to_agg=(next_player_idx==self.agg)
            all_acted=(self.acted >= active_counts)
            round_over=back_to_agg & all_acted & searching
            is_round_over |= round_over
            searching[round_over]=False

            # still eligible player left check
            player_status=self.status[g, next_player_idx]
            is_eligible=((player_status==self.ACTIVE) | (player_status==self.ALLIN)) & searching
            searching[is_eligible]=False
        
        no_over_mask=~is_round_over
        self.idx[no_over_mask]=next_player_idx[no_over_mask]

        # 3) handle round transitions & game ends
        terminated = torch.zeros(self.n_games, dtype=torch.bool, device=self.device)
        stack_changes = torch.zeros(self.n_games, dtype=torch.uint32, device=self.device)
        active_counts=((self.status == self.ACTIVE) | (self.status == self.ALLIN)).sum(dim=1)
        early_term=(active_counts <= 1)&is_round_over
        terminated[early_term]=True

        transition_mask=is_round_over&~early_term
        if transition_mask.any():
            g_over=g[transition_mask]
            self.stages[transition_mask]+=1
            self.highest[transition_mask] = 0
            self.agg[transition_mask] = (self.button[transition_mask] + 1) % self.n_players
            self.acted[transition_mask] = 0
            self.current_round_bet[g_over, :]=0

            flop_mask = (self.stages[transition_mask] == 1)
            turn_mask = (self.stages[transition_mask] == 2)
            river_mask = (self.stages[transition_mask] == 3)

            post_river=(self.stages[transition_mask]>3)
            terminated[transition_mask & post_river]=True

            for street_mask, n_cards, start_col in [(flop_mask, 3, 0), (turn_mask, 1, 3), (river_mask, 1, 4)]:
                if street_mask.any():
                    street_games=g_over[street_mask]
                    self.deck_positions[street_games]+= 1 # burn 1 card
                    cards=self.deal_cards(street_games, n_cards)
                    self.board[street_games, start_col:start_col+n_cards]=cards

        # showdown resolution, resolve winner by fold, stack change calculation for rewards





            

        
    
