import struct
import random
from environments.Poker.Player import Player

# --- Format String Explanation ---
# Total Items: 27
# 9B: Board(5) + Hand(2) + Stage(1) + Pos(1)  [Indices 0-8]
# 3H: Pot(1) + CallCost(1) + Stack(1)         [Indices 9-11]
# (H B H) * 5: For each opponent -> Stack(H), Status(B), Bet(H) [Indices 12-26]
POKER_STATE_FORMAT = '9B3H' + 'HBH'*5
PACKER = struct.Struct(POKER_STATE_FORMAT)

class PokerQLearning(Player):
    def __init__(self, id, starting_stack, action_space_n, ep=.1, gamma=.9, alpha=.9):
        super().__init__(starting_stack, id)
        self.ep = ep
        self.gamma = gamma
        self.alpha = alpha 
        self.action_space_n = action_space_n
        self.q = {}
        self.default_q = [0.0] * action_space_n

    def state_key(self, state):
        # packed byte helper function
        return PACKER.pack(*state)

    def get_q(self, key):
        # helper function to safely get the key in our q-dict
        return self.q.get(key, self.default_q)

    def action(self, state):
        # e-greedy policy
        if random.random() < self.ep:
            return random.randrange(self.action_space_n)
        
        # select the best action
        key = self.state_key(state)
        q_vals = self.get_q(key)
        a, a_val = 0, q_vals[0]
        for i in range(1, len(q_vals)):
            if q_vals[i] > a_val:
                a = i
                a_val = q_vals[i]
        return a

    def learn(self, state, action, reward, next_state, done):
        # "learning" function, called after we call the action function and get the next state and reward
        # get q_vals
        key = self.state_key(state)
        q_vals = self.q.get(key)
        
        if q_vals is None:
            q_vals = self.default_q.copy()
            self.q[key] = q_vals
        
        # calc new action val using sutton's psuedocode and update q-table
        if done:
            target = reward
        else:
            next_key = self.state_key(next_state)
            next_q_vals = self.get_q(next_key) 
            target = reward + self.gamma * max(next_q_vals)
        
        q_vals[action] += self.alpha * (target - q_vals[action])
    
    def save(self):
        pass