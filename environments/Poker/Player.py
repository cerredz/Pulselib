from abc import ABC, abstractmethod
import random
from environments.Poker.PokerGPU import PokerGPU
import eval7
import math
from environments.Poker.utils import decode_card
import torch
import torch.nn as nn
import copy
from pathlib import Path

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

class TightAggressivePlayerGPU(Player):
    def __init__(self, starting_stack: int, player_id: int, device):        
        super().__init__(starting_stack, player_id)
        self.device = device
        self.raise_distribution = torch.arange(2, 11, device=device) 

    def action(self, states):
        n_games=states.shape[0]
        hands = states[:, 5:7].long()
        ranks = hands%13
        rank1, rank2 = ranks[:, 0], ranks[:, 1]
        actions = torch.ones(n_games, dtype=torch.long, device=self.device)
        fold_mask=(rank1 < 7) & (rank2 < 7) & (torch.abs(rank1-rank2) > 5)
        actions[fold_mask]=0
        pair_mask=(rank1==rank2)
        high_card_mask = ((rank1 >= 10) & (rank2 > 5)) | ((rank2 >= 10) & (rank1 > 5))        
        raise_mask=(pair_mask | high_card_mask) & ~fold_mask
        n_raises=raise_mask.sum().item()
        indices=torch.randint(5, 9, (n_raises,), device=self.device)
        actions[raise_mask]=self.raise_distribution[indices]
        return actions

    def learn(self): pass

class LoosePassivePlayerGPU(Player):
    def __init__(self, starting_stack: int, player_id: int, device):        
        super().__init__(starting_stack, player_id)
        self.device = device
        self.raise_distribution = torch.arange(2, 11, device=device) 

    def action(self, states):
        n_games=states.shape[0]
        hands = states[:, 5:7].long()
        ranks = hands%13
        rank1, rank2 = ranks[:, 0], ranks[:, 1]
        actions = torch.zeros(n_games, dtype=torch.long, device=self.device)
        probs=torch.rand(n_games, device=self.device)
        fold_mask=(rank1 <= 4) & (rank2 <= 4) & (torch.abs(rank1-rank2) > 9)
        actions[fold_mask]=0
        pair_mask = (rank1 == rank2) & (rank1>8)
        high_card_mask = ((rank1 >= 11) & (rank2 > 9)) | ((rank2 >= 11) & (rank1 > 9)) 
        call_mask=(pair_mask | high_card_mask) & ~fold_mask
        raise_mask=(probs>.9) & call_mask
        n_raises = raise_mask.sum().item()
        actions[call_mask]=1
        indices = torch.randint(0, 4, (n_raises,), device=self.device)        
        actions[raise_mask]=self.raise_distribution[indices]
        return actions

    def learn(self): pass

class SmallBallPlayerGPU(Player):
    def __init__(self, starting_stack: int, player_id: int, device):        
        super().__init__(starting_stack, player_id)
        self.device = device
        self.raise_distribution = torch.arange(2, 11, device=device) 

    def action(self, states):
        n_games=states.shape[0]
        hands = states[:, 5:7].long()
        pot_size=states[:, 9]
        ranks = hands%13
        rank1, rank2 = ranks[:, 0], ranks[:, 1]
        actions = torch.zeros(n_games, dtype=torch.long, device=self.device)
        fold_mask = ((rank1 < 6) & (rank2 < 6) & (pot_size > 30)) | \
                    ((rank1 < 9) & (rank2 < 9) & (pot_size > 80)) 
        actions[fold_mask]=0

        pair_mask=(rank1==rank2)
        high_card_mask = ((rank1 >= 10) & (rank2 > 5)) | ((rank2 >= 10) & (rank1 > 5)) 
        raise_mask=(pair_mask | high_card_mask) & ~fold_mask
        n_raises = raise_mask.sum().item()
        indices=torch.randint(0, 3, (n_raises, ), device=self.device)
        actions[raise_mask]=self.raise_distribution[indices]
        return actions

    def learn(self): pass

class PokerQNetwork(nn.Module):
    def __init__(self, weights_path, device, gamma, update_freq:int, state_dim=27, action_dim=13, hidden_dim=256, lr=1e-3):
        super().__init__()
        self.update_freq=update_freq
        self.device=device  
        self.gamma=gamma      
        # Simple feedforward network
        self.network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.GELU(),
            nn.Linear(32, 24),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(24, 18),
            nn.GELU(),
            nn.Linear(18, action_dim)
        )

        self.network2 = nn.Sequential(
            nn.Linear(state_dim, 96),
            nn.GELU(),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Dropout(.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(.2),
            nn.Linear(32, 24),
            nn.GELU(),
            nn.Linear(24, action_dim)
        )

        if Path(weights_path).exists():
            model_weights=torch.load(weights_path, map_location=device)
            self.network.load_state_dict(model_weights)

        self.target_network=copy.deepcopy(self.network)
        self.target_network.eval()

        self.lr = lr
        self.step_count=0
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def forward(self, states):
        """
        states: [batch_size, 39] tensor
        returns: [batch_size, 13] Q-values
        """
        return self.network(states)
    
    def get_actions(self, states, epsilon=.1):
        with torch.no_grad():
            q_values = self.forward(states)
            explore_mask = torch.rand(len(states), device=self.device) < epsilon
            greedy_actions = q_values.argmax(dim=1)
            random_actions = torch.randint(0, 13, (len(states),), device=self.device)
            return torch.where(explore_mask, random_actions, greedy_actions)

    def train_step(self, states, actions, rewards, next_states, dones):
        # get state and action tensor of all games, need to filter on the states/actions we want to train on
        # dont want to train states based on 3 conditions:
            # already stages >= 4 (game in post-river stage, no more actions our agent can make)
            # already foled in prev round
            # states where reward is 0 (terminated game)
        
        valid_games = (states[:, 7] < 4) & (rewards.abs() > 1e-6) & ((states[:, 12] == 0) | (states[:, 12] == 2))
        if not valid_games.any(): return 0.0

        states=states[valid_games]
        actions=actions[valid_games]
        rewards=rewards[valid_games]
        next_states = next_states[valid_games]
        dones = dones[valid_games]

        q_values=self.forward(states)
        q_values_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch]

        with torch.no_grad():
            next_q_values=self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q_values * (~dones).float() 

        loss=self.criterion(q_values_for_actions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return loss.item()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


