from abc import ABC, abstractmethod
import random
import eval7
import math
from environments.Poker.utils import decode_card

class Player(ABC):
    """
    Stateful object representing a player at the table.
    We reuse these objects to avoid Garbage Collection overhead.
    """
    def __init__(self, stack_size: int, player_id: int):
        self.id = player_id
        self.stack = stack_size      # Current chips
        self.current_round_bet = 0   # Amount bet in current street
        self.total_invested = 0      # Amount bet in entire hand
        self.status = 'active'       # 'active', 'folded', 'allin'
        self.hand = []               # List of eval7 cards

    @abstractmethod
    def action(self, state):
        pass

    @abstractmethod
    def learn(self, state):
        pass

    def reset_state(self, new_hand, starting_stack=None):
        self.hand = new_hand
        self.current_round_bet = 0
        self.total_invested = 0
        self.status = 'active'
        if starting_stack is not None:
            self.stack = starting_stack

class RandomPlayer(Player):
    def action(self, state, valid_actions=None): return random.randint(0, 12)
    def learn(self, episode): pass

class HeuristicPlayer(Player):
    def action(self, state, valid_actions=None):
        # 1. Parse Pot Odds (Price)
        pot, call_cost = state[9], state[10]
        pot_odds = call_cost / (pot + call_cost) if (pot + call_cost) > 0 else 0

        # 2. Parse Cards
        hand = [decode_card(state[5]), decode_card(state[6])]
        board = [decode_card(state[i]) for i in range(5) if state[i] != -1]
                
        hand = [c for c in hand if c]
        board = [c for c in board if c]

        if not hand: return 0

        # 3. Calculate Strength (Proxy for Equity)
        strength = 0.5
        if not board: # Pre-flop Heuristics
            r1, r2 = hand[0].rank, hand[1].rank
            if r1 == r2: strength = 0.8       # Pair
            elif r1 > 9 and r2 > 9: strength = 0.6 # High Cards
        else: # Post-flop: Normalized Hand Rank
            score = eval7.evaluate(hand + board)
            strength = min(1.0, math.log(score + 1) / 18.5)

        # 4. Decision: Value > Price
        if strength > pot_odds + 0.1: 
            if strength > 0.8: return 8 
            return 1 
        return 1 if call_cost <= 0 else 0 

    def learn(self, episode): pass

