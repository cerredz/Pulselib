import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PokerGPU(gym.Env):
    metadata = {'render.modes': ['human']}
    NUM_ACTIONS=13
    ACTIVE, FOLDED, ALLIN = 0, 1, 2
    STATE_SPACE=28 # details in depth_notes in rl folder

    def __init__(self, device, agents, n_players=6, n_games=100, starting_bbs=100, max_bbs=1000, w1=.5, w2=.5, K=20):
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

        # action space / observation space
        self.action_space=spaces.Discrete(self.NUM_ACTIONS)
        self.raise_fractions=torch.tensor([0.25, 0.33, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00], device=self.device)
        self.obs_size=12+((self.n_players-1)*3)
        self.observation_space = spaces.Box(low=0, high=10000, shape=(self.obs_size,), dtype=np.float32)
        
        # Initialize state tensors as None (will be set in reset)
        self.stacks = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
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
        
        self.hands = self.deal_players_cards(self.n_players * 2).view(self.n_games, self.n_players, 2)

        self.current_round_bet = torch.zeros((self.n_games, self.n_players), dtype=torch.int32, device=self.device)
        self.total_invested = torch.zeros((self.n_games, self.n_players), dtype=torch.int32, device=self.device)
        self.status = torch.full((self.n_games, self.n_players), self.ACTIVE, dtype=torch.int32, device=self.device)

        self.button = (self.button + 1) % self.n_players if hasattr(self, 'button_pos') else torch.zeros(self.n_games, dtype=torch.int32, device=self.device)
        self.sb = (self.button + 1) % self.n_players
        self.bb = (self.button + 2) % self.n_players

        self.post_blinds()

        self.idx = (self.bb + 1) % self.n_players
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
        return {}

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
        active_mask=(self.status[g, self.idx] != self.FOLDED) & (self.status[g, self.idx] != self.ALLIN)

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

    def mask_royal_flush(self, num_show, suits, ranks):
        royal_flush_mask = torch.zeros(num_show, self.n_players, dtype=torch.bool, device=self.device)
        for suit in range(4): 
            suit_mask = (suits == suit)  # [num_show, n_players, 7]
            royal_ranks = torch.tensor([8, 9, 10, 11, 12], device=self.device)  # 10, J, Q, K, A
            has_all_royal_ranks = torch.ones(num_show, self.n_players, dtype=torch.bool, device=self.device)
            
            for rank in royal_ranks:
                # Check if this rank exists in this suit
                has_rank_in_suit = ((ranks == rank) & suit_mask).any(dim=2)  # [num_show, n_players]
                has_all_royal_ranks &= has_rank_in_suit
            
            # Players who have all 5 royal ranks in this suit have a royal flush
            royal_flush_mask |= has_all_royal_ranks
        return royal_flush_mask

    def mask_straight_flush(self, num_show, suits, ranks, royal_flush_mask):
        straight_flush_mask = torch.zeros(num_show, self.n_players, dtype=torch.bool, device=self.device)
        straight_flush_high_card = torch.zeros(num_show, self.n_players, dtype=torch.long, device=self.device)
        
        for suit in range(4):
            suit_mask = (suits == suit)
            
            # Check all possible straights (A-2-3-4-5 through 9-10-J-Q-K)
            # Straights are 5 consecutive ranks
            # Possible high cards: 3 (for A-2-3-4-5), 4, 5, 6, 7, 8, 9, 10, 11, 12 (for 8-9-10-J-Q)
            
            for high_card in range(3, 13):  # High card of straight: 3 through 12
                # Build the 5 consecutive ranks
                straight_ranks = torch.tensor([high_card-3, high_card-2, high_card-1, high_card, high_card+1] if high_card < 12 
                                            else [high_card-4, high_card-3, high_card-2, high_card-1, high_card], 
                                            device=self.device)
                
                # Special case: wheel straight (A-2-3-4-5, high card is 5 which is rank 3)
                if high_card == 3:
                    straight_ranks = torch.tensor([0, 1, 2, 3, 12], device=self.device)  # 2,3,4,5,Ace
                else:
                    straight_ranks = torch.arange(high_card-4, high_card+1, device=self.device)
                
                # Check if all ranks in this straight exist in this suit
                has_all_straight_ranks = torch.ones(num_show, self.n_players, dtype=torch.bool, device=self.device)
                
                for rank in straight_ranks:
                    has_rank_in_suit = ((ranks == rank) & suit_mask).any(dim=2)
                    has_all_straight_ranks &= has_rank_in_suit
                
                # Update straight flush mask and track highest straight flush
                is_new_straight_flush = has_all_straight_ranks & ~straight_flush_mask
                straight_flush_mask |= has_all_straight_ranks
                
                # Track the high card of the best straight flush
                straight_flush_high_card = torch.where(
                    is_new_straight_flush,
                    torch.full_like(straight_flush_high_card, high_card),
                    straight_flush_high_card
                )
        
        # Exclude royal flushes from straight flush mask
        straight_flush_mask = straight_flush_mask & ~royal_flush_mask
        return straight_flush_mask, straight_flush_high_card

    def mask_four_of_a_kind(self, num_show, suits, ranks):
        four_of_a_kind_mask = torch.zeros(num_show, self.n_players, dtype=torch.bool, device=self.device)
        four_kind_rank = torch.zeros(num_show, self.n_players, dtype=torch.long, device=self.device)
        four_kind_kicker = torch.zeros(num_show, self.n_players, dtype=torch.long, device=self.device)
        
        # For each possible rank (0-12), check if player has 4 of that rank
        for rank in range(13):  # 0=2, 1=3, ..., 12=Ace
            # Count how many cards of this rank each player has
            rank_mask = (ranks == rank)  # [num_show, n_players, 7]
            rank_count = rank_mask.sum(dim=2)  # [num_show, n_players] - count of this rank
            
            # Players with exactly 4 of this rank have four of a kind
            has_four = (rank_count == 4)  # [num_show, n_players]
            
            # Update mask (use OR to catch if player has multiple four of a kinds - take highest)
            is_new_four = has_four & ~four_of_a_kind_mask
            four_of_a_kind_mask |= has_four
            
            # Track the rank of the four of a kind (higher rank wins tiebreaker)
            # Only update if this is a new four OR a higher-ranked four
            is_better_four = has_four & (rank > four_kind_rank)
            four_kind_rank = torch.where(is_better_four, rank, four_kind_rank)
            
            # For players with this four of a kind, find their best kicker
            if has_four.any():
                # Get all cards that are NOT part of the four of a kind
                not_four_mask = ranks != rank  # [num_show, n_players, 7]
                
                # Find the highest kicker (max rank among non-four cards)
                # Set four-of-a-kind cards to -1 so they don't get picked as kicker
                kicker_ranks = torch.where(not_four_mask, ranks, torch.tensor(-1, device=self.device))
                best_kicker = kicker_ranks.max(dim=2).values  # [num_show, n_players]
                
                # Update kicker only for players with this specific four of a kind
                # and only if it's better than their current kicker
                should_update_kicker = has_four & (rank == four_kind_rank)
                four_kind_kicker = torch.where(should_update_kicker, best_kicker, four_kind_kicker)
        
        return four_of_a_kind_mask, four_kind_rank, four_kind_kicker

    def mask_full_house(self, num_show, suits, ranks, player_mask):
        """
        Detect full house hands (three of a kind + pair).
        
        Returns:
            full_house_mask: [num_show, n_players] - True where player has full house
            full_house_trips: [num_show, n_players] - Rank of the three of a kind
            full_house_pair: [num_show, n_players] - Rank of the pair
        """
        full_house_mask = torch.zeros(num_show, self.n_players, dtype=torch.bool, device=self.device)
        full_house_trips = torch.full((num_show, self.n_players), -1, dtype=torch.long, device=self.device)
        full_house_pair = torch.full((num_show, self.n_players), -1, dtype=torch.long, device=self.device)
        
        # Count occurrences of each rank for each player
        rank_counts = torch.zeros(num_show, self.n_players, 13, dtype=torch.long, device=self.device)
        
        for rank in range(13):
            rank_mask = (ranks == rank)  # [num_show, n_players, 7]
            rank_counts[:, :, rank] = rank_mask.sum(dim=2)  # Count this rank
        
        # Find all ranks where player has 3+ cards (potential trips)
        trips_mask = (rank_counts >= 3)  # [num_show, n_players, 13]
        
        # Find all ranks where player has 2+ cards (potential pairs)
        pair_mask = (rank_counts >= 2)  # [num_show, n_players, 13]
        
        # For each player, find their best trips and best pair
        for game_idx in range(num_show):
            for player_idx in range(self.n_players):
                if not player_mask[game_idx, player_idx]:
                    continue  # Skip folded players
                
                # Get ranks where this player has trips (3+)
                trips_ranks = torch.where(trips_mask[game_idx, player_idx])[0]
                
                # Get ranks where this player has pairs (2+)
                pair_ranks = torch.where(pair_mask[game_idx, player_idx])[0]
                
                if len(trips_ranks) == 0:
                    continue  # No trips, no full house
                
                # Best trips is the highest rank with 3+
                best_trips = trips_ranks.max().item()
                
                # For the pair, we need a DIFFERENT rank with 2+
                # Case 1: Player has two different trips (e.g., 3 Aces + 3 Kings) -> use second trips as pair
                # Case 2: Player has trips + separate pair (e.g., 3 Aces + 2 Kings)
                # Case 3: Player has quads + pair (e.g., 4 Aces + 2 Kings) -> trips is Aces, pair is Kings
                
                # Find potential pair ranks (exclude the trips rank we're using)
                potential_pairs = pair_ranks[pair_ranks != best_trips]
                
                if len(potential_pairs) == 0:
                    # Check if we have 4+ of the trips rank (quads can be split into trips + pair)
                    if rank_counts[game_idx, player_idx, best_trips] >= 5:
                        # 5+ of same rank (e.g., board + hand both have Aces) -> trips and pair are same rank
                        best_pair = best_trips
                    else:
                        continue  # No valid pair, no full house
                else:
                    # Use the highest pair rank
                    best_pair = potential_pairs.max().item()
                
                # Valid full house found
                full_house_mask[game_idx, player_idx] = True
                full_house_trips[game_idx, player_idx] = best_trips
                full_house_pair[game_idx, player_idx] = best_pair
        
        return full_house_mask, full_house_trips, full_house_pair

    def mask_flush(self, num_show, suits, ranks, player_mask, straight_flush_mask, royal_flush_mask):
        # logs
        flush_mask = torch.zeros(num_show, self.n_players, dtype=torch.bool, device=self.device)
        flush_cards = torch.full((num_show, self.n_players, 5), -1, dtype=torch.long, device=self.device)
        
        # For each suit, check if player has 5+ cards
        for suit in range(4):  # 0=clubs, 1=diamonds, 2=spades, 3=hearts
            suit_mask = (suits == suit)  # [num_show, n_players, 7]
            suit_count = suit_mask.sum(dim=2)  # [num_show, n_players] - count of this suit
            
            # Players with 5+ cards of this suit have a flush
            has_flush = (suit_count >= 5)  # [num_show, n_players]
            
            if not has_flush.any():
                continue  # No flushes in this suit
            
            # For players with flush in this suit, extract the ranks of suited cards
            for game_idx in range(num_show):
                for player_idx in range(self.n_players):
                    if not has_flush[game_idx, player_idx]:
                        continue
                    if not player_mask[game_idx, player_idx]:
                        continue  # Skip folded players
                    
                    # Get all cards of this suit for this player
                    suited_cards_mask = suit_mask[game_idx, player_idx]  # [7]
                    suited_ranks = ranks[game_idx, player_idx][suited_cards_mask]  # [5, 6, or 7] ranks
                    
                    # Sort in descending order and take top 5
                    sorted_suited_ranks, _ = torch.sort(suited_ranks, descending=True)
                    top_5_ranks = sorted_suited_ranks[:5]  # Take best 5 cards
                    
                    # Update flush mask and cards
                    flush_mask[game_idx, player_idx] = True
                    flush_cards[game_idx, player_idx] = top_5_ranks
        
        # Exclude straight flushes and royal flushes
        flush_mask = flush_mask & ~straight_flush_mask & ~royal_flush_mask
        
        # Clear flush_cards for non-flush players (set to -1)
        flush_cards = torch.where(
            flush_mask.unsqueeze(2).expand(-1, -1, 5),
            flush_cards,
            torch.tensor(-1, device=self.device)
        )
        
        return flush_mask, flush_cards

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
        
        # fill out imcomplete boards
        incomplete_mask = (self.board[g_show] == -1).any(dim=1)  # [num_show] - True if board has any -1
        if incomplete_mask.any():
            incomplete_games = g_show[incomplete_mask]
            for game_idx in incomplete_games:
                # Find how many cards are missing
                board_cards = self.board[game_idx]
                missing_positions = (board_cards == -1)
                n_missing = missing_positions.sum().item()
                
                if n_missing > 0:
                    # Deal missing cards from deck
                    new_cards = self.deal_cards(game_idx.unsqueeze(0), n_missing).squeeze(0)
                    # Fill in the missing positions
                    self.board[game_idx, missing_positions] = new_cards

        full_hands = torch.cat([self.hands[g_show], self.board[g_show].unsqueeze(1).expand(-1, self.n_players, -1)], dim=2)  # [num_show, n_players, 7]
        ranks, suits = full_hands%13, full_hands//13

        # find hand strength masks
        #royal_flush_mask=self.mask_royal_flush(num_show, suits, ranks) & player_mask
        #straight_flush_mask, straight_flush_high_card=self.mask_straight_flush(num_show, suits, ranks, royal_flush_mask) & player_mask
        #four_of_a_kind_mask, four_kind_rank, four_kind_kicker = self.mask_four_of_a_kind(num_show, suits, ranks) & player_mask
        #full_house_mask, full_house_trips, full_house_pair = self.mask_full_house(num_show, suits, ranks, player_mask)
        #flush_mask, flush_cards = self.mask_flush(num_show, suits, ranks, player_mask, straight_flush_mask, royal_flush_mask)

        return terminated


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

    def step(self, actions):
        # step function to handle logic of n_games actions at once
        # get game indices ready
        g=torch.arange(self.n_games, device=self.device)
        prev_stacks=self.stacks[g, self.idx].clone()
        prev_invested=self.current_round_bet[g, self.idx].clone()

        # 1)  execute actions
        self.execute_actions(g, actions)

        # 2) find next player to act in current round
        active_counts=((self.status == self.ACTIVE) | (self.status == self.ALLIN)).sum(dim=1)
        next_player_idx=self.idx.clone()
        is_round_over=torch.zeros(self.n_games, dtype=torch.bool, device=self.device)
        searching=torch.ones(self.n_games, dtype=torch.bool, device=self.device)
        for _ in range(self.n_players):
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
        stack_changes = torch.zeros(self.n_games, dtype=torch.int32, device=self.device)
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
            post_river_games = g_over[post_river]
            terminated[post_river_games] = True

            for street_mask, n_cards, start_col in [(flop_mask, 3, 0), (turn_mask, 1, 3), (river_mask, 1, 4)]:
                if street_mask.any():
                    street_games=g_over[street_mask]
                    self.deck_positions[street_games]+= 1 # burn 1 card
                    cards=self.deal_cards(street_games, n_cards)
                    self.board[street_games, start_col:start_col+n_cards]=cards

        # showdown resolution, resolve winner by fold, stack change calculation for rewards

        # calculate reward
        equities=torch.full((self.n_games, ), .5, dtype=torch.float32, device=self.device)
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
        
        terminated=torch.ones(self.n_games, device=self.device, dtype=torch.bool)
        truncated = torch.zeros(self.n_games, dtype=torch.bool, device=self.device)  # All False
        info = {}

        return self.get_obs(), rewards, terminated, truncated, info




            

        
    
