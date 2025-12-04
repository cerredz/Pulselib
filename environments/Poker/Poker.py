import gymnasium as gym
import numpy as np
import eval7
from gymnasium import spaces
from environments.Poker.Player import Player
from environments.Poker.utils import calculate_equity, encode_card, poker_reward
from utils.steps import steps

_FULL_RANGE = eval7.HandRange("AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AKo,AQs,AQo,AJs,AJo,ATs,ATo,A9s,A9o,A8s,A8o,A7s,A7o,A6s,A6o,A5s,A5o,A4s,A4o,A3s,A3o,A2s,A2o,KQs,KQo,KJs,KJo,KTs,KTo,K9s,K9o,K8s,K8o,K7s,K7o,K6s,K6o,K5s,K5o,K4s,K4o,K3s,K3o,K2s,K2o,QJs,QJo,QTs,QTo,Q9s,Q9o,Q8s,Q8o,Q7s,Q7o,Q6s,Q6o,Q5s,Q5o,Q4s,Q4o,Q3s,Q3o,Q2s,Q2o,JTs,JTo,J9s,J9o,J8s,J8o,J7s,J7o,J6s,J6o,J5s,J5o,J4s,J4o,J3s,J3o,J2s,J2o,T9s,T9o,T8s,T8o,T7s,T7o,T6s,T6o,T5s,T5o,T4s,T4o,T3s,T3o,T2s,T2o,98s,98o,97s,97o,96s,96o,95s,95o,94s,94o,93s,93o,92s,92o,87s,87o,86s,86o,85s,85o,84s,84o,83s,83o,82s,82o,76s,76o,75s,75o,74s,74o,73s,73o,72s,72o,65s,65o,64s,64o,63s,63o,62s,62o,54s,54o,53s,53o,52s,52o,43s,43o,42s,42o,32s,32o")


class Poker(gym.Env):
    metadata = {'render.modes': ['human']}
    NUM_ACTIONS = 13
    
    def __init__(self, agents=None, n=6, bb=2, starting_stack=100):
        super().__init__()
        
        self.n = n
        self.starting_stack = starting_stack
        self.bb = bb
        self.sb = bb // 2
        
        # Action Space
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.raise_fractions = [0.25, 0.33, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00]
        
        # Observation Space
        obs_size = 12 + ((self.n-1) * 3)
        self.observation_space = spaces.Box(low=0, high=10000, shape=(obs_size,), dtype=np.float32)
        self.deck = eval7.Deck()
        self.players = agents if agents else []
        self.button_pos=0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.deck=eval7.Deck()
        self.deck.shuffle()
        self.board = []
        self.pot = 0
        self.stage = 0
                
        for i, p in enumerate(self.players):
            hand = self.deck.deal(2)
            stack_val = self.starting_stack if p.stack==0 else None
            p.reset_state(new_hand=hand, starting_stack=stack_val)
            
        # Blinds & Position Logic
        self.button_pos = (self.button_pos + 1) % self.n
        self.sb_pos = (self.button_pos + 1) % self.n
        self.bb_pos = (self.button_pos + 2) % self.n
        
        self._post_blind(self.sb_pos, 0) # small blind is .5 blinds, round to 0
        self._post_blind(self.bb_pos, 1) # big blind is 1 blind
        
        self.curr_idx = (self.bb_pos + 1) % self.n
        self.highest_bet = 1 # big blind bet 
        self.aggressor_idx = self.bb_pos
        self.players_acted_this_street = 0
        
        # Initial equity calculation for reward delta
        self.prev_equity = self._calculate_equity(self.players[self.curr_idx])
        
        return self._get_obs(), {}

    @steps(reported_every_sec=5.0)
    def step(self, action):
        # executes action, calcs reward, game moves forward
        current_player = self.players[self.curr_idx]
        prev_stack = current_player.stack
        prev_invested = current_player.current_round_bet
        
        # 1. Logic for Current Player
        self.action_player_logic(action)
        
        # 2. Find Next Player
        next_player_idx = self.curr_idx
        is_round_over = False
        
        while True:
            next_player_idx = (next_player_idx + 1) % self.n
            
            # Round Over Condition: Back to Aggressor AND everyone acted
            if next_player_idx == self.aggressor_idx and self.players_acted_this_street >= self.count_active_players():
                is_round_over = True
                break
                
            if self.players[next_player_idx].status == 'active' or self.players[next_player_idx].status == 'allin':
                break
        
        self.curr_idx = next_player_idx
        # 3. Handle Street Transitions & Game End
        terminated = False
        stack_change = 0
        
        if is_round_over:
            self.next_round()
            if self.stage > 3: # After River
                self.resolve_showdown()
                terminated = True
                # Calculate stack change for reward
                stack_change = current_player.stack - prev_stack 
                
        # Check for Fold Win
        if self.count_active_players() == 1:
            self.resolve_winner_by_fold()
            terminated = True
            stack_change = current_player.stack - prev_stack

        # 4. Calculate Reward
        new_equity = self._calculate_equity(current_player)
        call_cost = max(0, self.highest_bet - prev_invested)
        investment_this_step = prev_stack - current_player.stack
        
        reward = poker_reward(
            w1=1.0, w2=0.5, n=self.n, K=20.0,
            equity=new_equity,
            pot=self.pot,
            investment=investment_this_step,
            stack=stack_change,
            cost_to_call=call_cost,
            fair_share=1.0/max(1, self.count_active_players()),
            action_type=action
        )
        
        self.prev_equity = new_equity
        return self._get_obs(), reward, terminated, False, {}

    def action_player_logic(self, action):
        player = self.players[self.curr_idx]
        call_cost = self.highest_bet - player.current_round_bet
        
        if action == 0: # Fold
            player.status = 'folded'
            self.players_acted_this_street += 1
            return

        if action == 1: # Check/Call
            amount = min(call_cost, player.stack)
            self._bet_chips(player, amount)
            self.players_acted_this_street += 1
            return

        # Raise Logic
        current_pot_total = self.pot + call_cost
        raise_amount = 0
        
        if action == 2: raise_amount = max(self.bb, call_cost)
        elif action == 12: raise_amount = player.stack - call_cost
        else:
            idx = action - 3
            if 0 <= idx < len(self.raise_fractions):
                raise_amount = int(current_pot_total * self.raise_fractions[idx])
            else:
                raise_amount = 1

        total_needed = call_cost + int(raise_amount)
        actual_bet = min(total_needed, player.stack)
        
        self._bet_chips(player, actual_bet)
    
        if actual_bet > call_cost:
            self.highest_bet = player.current_round_bet
            self.aggressor_idx = player.id
            self.players_acted_this_street = 0
        
        self.players_acted_this_street += 1

    def _bet_chips(self, player, amount):
        player.stack -= amount
        player.current_round_bet += amount
        player.total_invested += amount
        self.pot += amount
        if player.stack == 0:
            player.status = 'allin'

    def _post_blind(self, player_idx, amount):
        player = self.players[player_idx]
        actual = min(amount, player.stack)
        self._bet_chips(player, actual)

    def next_round(self):
        self.stage += 1
        self.highest_bet = 0
        self.aggressor_idx = (self.button_pos + 1) % self.n
        self.players_acted_this_street = 0
        
        for p in self.players:
            p.current_round_bet = 0
            
        if self.stage == 1: 
            self.board = self.deck.deal(3)
        elif self.stage == 2 or self.stage == 3: 
            self.board.append(self.deck.deal(1)[0])

    def _get_obs(self):
        """Flattened state vector for Neural Net."""
        obs = []
        
        # Board (5 ints) - handle up to 5 cards
        for i in range(5):
            if i < len(self.board): obs.append(encode_card(self.board[i]))
            else: obs.append(0)
        
        # Hero Cards (2 ints)
        hero = self.players[self.curr_idx]
        obs.append(encode_card(hero.hand[0]))
        obs.append(encode_card(hero.hand[1]))
        
        # Globals (5 ints)
        obs.append(self.stage)
        obs.append((self.curr_idx - self.button_pos) % self.n)
        obs.append(int(self.pot))
        call_cost = self.highest_bet - hero.current_round_bet
        obs.append(int(call_cost))
        obs.append(int(hero.stack))
        
        # Opponents (N * 3 ints)
        for i in range(1, self.n):
            opp_idx = (self.curr_idx + i) % self.n
            opp = self.players[opp_idx]
            obs.append(int(opp.stack / self.bb))
            obs.append(1 if opp.status == 'active' else 0)
            obs.append(int(opp.current_round_bet / self.bb))
            
        return tuple(obs)

    def _calculate_equity_preflop(self, player):
        # calculates the equity of a players hand preflop
        # extrapolated into another function because the preflop equity can be caluclated with a lookup table
        pass

    def _calculate_equity(self, player):
        """Optimized equity calculation using imported function."""
        return calculate_equity(
            player.hand,
            self.board,
            self.stage,
            self.count_active_players(),
            player.status
        )

    def resolve_showdown(self):
        active_players = [p for p in self.players if p.status != 'folded']
        if not active_players: return

        player_scores = {p.id: eval7.evaluate(p.hand + self.board) for p in active_players}
        active_players.sort(key=lambda x: x.total_invested)
        undistributed = {p.id: p.total_invested for p in self.players}

        for pot_owner in active_players:
            if undistributed[pot_owner.id] <= 0: continue
            
            chunk_size = undistributed[pot_owner.id]
            side_pot = 0
            contributors = []
            
            for p in self.players:
                amount_taken = min(undistributed[p.id], chunk_size)
                side_pot += amount_taken
                undistributed[p.id] -= amount_taken
                if p.status != 'folded' and amount_taken > 0:
                    contributors.append(p)
            
            if not contributors: continue
            
            best_score = -1
            winners = []
            for p in contributors:
                score = player_scores[p.id]
                if score > best_score:
                    best_score = score
                    winners = [p]
                elif score == best_score:
                    winners.append(p)
            
            if side_pot > 0:
                share = side_pot // len(winners)
                remainder = side_pot % len(winners)
                for w in winners:
                    w.stack += share
                if winners:
                    winners[0].stack += remainder

    def resolve_winner_by_fold(self):
        for p in self.players:
            if p.status in ['active', 'allin']:
                p.stack += self.pot
                return

    def count_active_players(self):
        return sum(1 for p in self.players if p.status in ['active', 'allin'])