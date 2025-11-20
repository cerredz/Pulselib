import random
from typing import Dict, Tuple, List
from collections import defaultdict
import gymnasium as gym

class OnPolicyFirstVisitMC():
    def __init__(self, gamma: float, epsilon: float, action_space):
        assert isinstance(action_space, gym.spaces.Discrete), "On Policy First Visit Monte Carlo only implemented for Discrete action spaces for now. Use discrete action space"

        self.q: Dict[Tuple, float] = defaultdict(float)        
        self.returns: Dict[Tuple, List[float]] = defaultdict(lambda: [0.0, 0.0]) # sum of state-action, size of returns at state-action
        self.gamma=gamma
        self.epsilon=epsilon
        self.n=action_space.n
        self.actions=[i for i in range(self.n)]
        self.policy: Dict[Tuple, List[float]] = defaultdict(lambda: [1.0/self.n for _ in range(self.n)])
        # self.policy: maps probabilities to our actions in our action space given a state

    def action(self, state):
        probs=self.policy[state]
        a=random.choices(population=self.actions, weights=probs, k=1)[0]
        return a

    def learn(self, episode: List[tuple]):
        # each index in episode is denoted as a tuple of: state, action, reward at time step t
        # episode lists contains all time steps of all stace, action, reward tuples in the episode
        g=0
        n=len(episode)

        first_visit_indices = {}
        visited_pairs=set()
        for t, (state, action, reward) in enumerate(episode):
            sa=state + (action,)
            if sa not in visited_pairs:
                first_visit_indices[sa] = t
                visited_pairs.add(sa)

        # for each step of episode
        for t in range(n-1, -1, -1):
            # update the q and returns table
            state, action, reward = episode[t]
            g = self.gamma * g + reward
            sa=state+ (action,)

            if t == first_visit_indices[sa]:

                self.returns[sa][0] += g
                self.returns[sa][1] += 1.0
                self.q[sa] = self.returns[sa][0] / self.returns[sa][1]
            
            # find the best action
            best_action, max_q_val=0, float("-inf")
            for a in self.actions:
                qsa=state+(a,)
                q_val=self.q[qsa]

                if q_val > max_q_val:
                    max_q_val = q_val
                    best_action = a
                elif q_val == max_q_val:
                    if random.random() > 0.5:
                        best_action = a

            # Update Probabilities
            for i in range(self.n):
                if i == best_action:
                    # Exploit: 1 - e + e/n
                    self.policy[state][i] = 1 - self.epsilon + (self.epsilon / self.n)
                else:
                    # Explore: e/n
                    self.policy[state][i] = self.epsilon / self.n     

            
                






    


        

        
        



