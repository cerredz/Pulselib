# utility functions for our poker environment
import math
from typing import List
import enum
import eval7

def decode_card(rank_int, suit_int):
    """
    Reconstructs an eval7 Card object from the state vector integers.
    """
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['c', 'd', 'h', 's']
    
    if rank_int < 0 or rank_int > 12 or suit_int < 0 or suit_int > 3:
        return None
        
    card_str = f"{ranks[int(rank_int)]}{suits[int(suit_int)]}"
    return eval7.Card(card_str)

def encode_card(card):
    # converts card to int between 0 and 51
    print(card.rank, card.suit)
    return card.rank + 13 * card.suit

def poker_reward(
        w1: float,
        w2: float,
        n:int, 
        K: float,
        equity: float, 
        pot: float, 
        investment: float,
        stack: float, # amount of money won or lost
        cost_to_call: float,
        fair_share: float, 
        action_type: int
    ):

    m=.5*((equity*pot)-investment)+.5*(stack)
    o=cost_to_call / (pot + cost_to_call)
    if action_type==0: # calling
        s=(equity-o)*pot
    elif action_type==1: # folding
        s=(o-equity)*pot
    else: # raising/betting
        s=equity-fair_share*pot*1.2
    r=n*math.tanh((w1*m+w2*s)/K)
    return r

class PokerAgentType(enum.Enum):
    QLEARNING='qlearning'
    HEURISTIC="heuistic"
    RANDOM='random'

def load_agents(num_players: int, agent_types: list, starting_stack: int) -> list:
    # Local imports to avoid circular import with Player -> utils
    from agents.TemperalDifference.PokerQLearning import PokerQLearning
    from environments.Poker.Player import HeuristicPlayer, RandomPlayer
    players = []
    assert len(agent_types) == num_players
    for i, a_type in enumerate(agent_types):
        if a_type == 'qlearning': p = PokerQLearning(starting_stack, i)
        elif a_type == 'random': p = RandomPlayer(starting_stack, i)
        else: p = HeuristicPlayer(starting_stack, i)
        players.append(p)
    return players

