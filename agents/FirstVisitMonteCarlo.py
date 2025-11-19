from typing import Dict, Tuple, List
from collections import defaultdict, Counter

class FirstVisitMonteCarlo():
    def __init__(self, gamma:float, ):
        self.values: Dict[Tuple,float] = defaultdict(float)
        self.returns: Dict[Tuple, List[float]] = defaultdict(lambda: [0.0, 0.0]) # sum of state, size of returns at state
        self.gamma=gamma

    def action(self, action_space):
        return action_space.sample()

    def learn(self, episode: List[tuple]):
        # each index in list should be a time step of an episode defined by:
        # state, action, reward
        g=0
        n = len(episode)
        
        first_visit_indices = {}
        for t, time_step in enumerate(episode):
            state = time_step[0] 
            if state not in first_visit_indices:
                first_visit_indices[state] = t

        for i in range(n-1, -1, -1):
            state, action, reward= episode[i]
            g = self.gamma * g + reward
            if first_visit_indices[state] == i:
                self.returns[state][0] += g
                self.returns[state][1] += 1
                self.values[state] = self.returns[state][0] / self.returns[state][1]


            