from abc import ABC, abstractmethod
import random
import eval7
import math
from environments.Poker.utils import decode_card
import torch
import torch.nn as nn

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
        self.hand = []             # List of eval7 cards

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
        strength = 0.5
        if not board: 
            r1, r2 = hand[0].rank, hand[1].rank
            if r1 == r2: strength = 0.8
            elif r1 > 9 and r2 > 9: strength = 0.6 
        else: 
            score = eval7.evaluate(hand + board)
            strength = min(1.0, math.log(score + 1) / 18.5)

        if strength > pot_odds + 0.1: 
            if strength > 0.8: return 8 
            return 1 
        return 1 if call_cost <= 0 else 0 

    def learn(self, episode): pass

# simple heuristic player, bet solely based on its own hand strength (nothing to do with the board)
# very simple, one of the first agents that our nn will play against
class HeuristicHandsPlayerGPU(Player):
    def __init__(self, starting_stack: int, player_id: int, device):        
        super().__init__(starting_stack, player_id)
        self.device = device
        self.raise_distribution = torch.arange(2, 11, device=device) 

    def action(self, states):
        # get players hands
        n_games=states.shape[0]
        hands = states[:, 5:7]
        ranks = hands%13

        # extract ranks
        rank1, rank2 = ranks[:, 0], ranks[:, 1]
        actions = torch.zeros(n_games, dtype=torch.long, device=self.device)

        # actions based on ranks
        fold_mask = (rank1 < 8) & (rank2 < 8)
        actions[fold_mask] = 0
        pair_mask = (rank1 == rank2)
        high_card_mask = (rank1 >= 10) | (rank2 >= 10)  # King or Ace
        raise_mask = (pair_mask | high_card_mask) & ~fold_mask

        n_raises = raise_mask.sum().item()
        indices = torch.randint(0, 9, (n_raises,), device=self.device)
        actions[raise_mask] = self.raise_distribution[indices]
        return actions

    def learn(self): pass

class PokerQNetwork(nn.Module):
    def __init__(self, device, state_dim=27, action_dim=13, hidden_dim=256, lr=1e-4):
        super().__init__()

        self.device=device        
        # Simple feedforward network
        self.network = nn.Sequential(
            nn.Linear(state_dim, 23),
            nn.GELU(),
            nn.Linear(23, 19),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(19, 16),
            nn.GELU(),
            nn.Linear(16, action_dim)
        )
        
        self.lr = lr
    
    def forward(self, states):
        """
        states: [batch_size, 27] tensor
        returns: [batch_size, 13] Q-values
        """
        states=states.to(self.device)
        return self.network(states)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


