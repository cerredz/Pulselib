from environments.Poker.utils import PlayerStatus
from environments.Poker.Player import Player
import gymnasium as gym
import numpy as np
from typing import List, Any
from environments.Poker.utils import Round
import random
import math
import eval7

"""
- Poker environment implementation for rl:
"""

# Action Mapping
# 0: Fold
# 1: Check/Call (Context dependent)
# 2: Raise Min
# 3: Raise 25% Pot
# 4: Raise 33% Pot
# 5: Raise 50% Pot
# 6: Raise 75% Pot
# 7: Raise 100% Pot
# 8: Raise 150% Pot
# 9: Raise 200% Pot
# 10: Raise 300% Pot
# 11: All-In

class Poker(gym.Env):
    NUM_ACTIONS=12
    
    def __init__(self, n: int, sm: int, bb: int, starting_bbs:int=100):
        assert n > 1 and n < 10, "Number of players must be between 2 and 9"
        assert sm > 0 and bb > 0, "Small blind and big blind must be greater than 0"
        self.n=n
        self.action_space=gym.spaces.Discrete(self.NUM_ACTIONS)
        observation_space_size=12 + self.n*3
        self.observation_space=gym.spaces.Box(
            low=0, high=np.int32, shape=(observation_space_size,), dtype=np.int32
        )

        self.sm=sm
        self.bb=bb
        self.deck=eval7.Deck()
        self.round=Round.PREFLOP

        self.board=[]
        self.pot_size=0
        self.button_position=random.randint(0, self.n-1)
        self.curr_idx=self.button_position
        self.players=[Player(blinds=starting_bbs, current_round_bet=0, status=PlayerStatus.ACTIVE) for _ in range(self.n)]

    # board, our agent's cards, stage in game, position from button, pot size, our money, opponent info
    def _get_obs(self):
        our_cards=self.players[self.curr_idx].cards
        our_money=self.players[self.curr_idx].blinds
        position=abs(self.curr_idx - self.button_position)
        opponent_info=np.concatenate([np.array(self.players[i].status, self.players[i].blinds, self.players[i].current_round_bet) for i in range(self.n) if i != self.curr_idx])
        state=np.concatenate(self.board, our_cards, self.round, position, self.pot_size//self.bb, our_money, opponent_info)
        return state
    
    def _get_info(self):
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board=[]
        self.round=Round.PREFLOP
        self.pot_size=0
        self.button_position = self.button_position + 1 % self.n
        self.curr_idx=self.button_position

    def step(self, action):
        assert 0 <= action < self.NUM_ACTIONS, 'invalid action idx (0-11)'
        
        



if __name__ == "__main__":
    deck = eval7.Deck()
    print(deck)
    deck.shuffle()
    hand = deck.deal(7)
    print(hand)
    hr = eval7.HandRange("AJ+, ATs, KQ+, 33-JJ, 0.8(QQ+, KJs)")
    print(len(hr))
    print(deck.deal(5))
    deck.shuffle()
    print(deck)
