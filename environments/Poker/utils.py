# utility functions for our poker environment
import math
from typing import List
import enum
import eval7

def decode_card(card_int):
    """
    Reconstructs an eval7 Card object from the single state integer (0-51).
    Returns None if card_int is -1 or out of bounds.
    """
    if card_int == -1 or card_int == 52: # Handle padding
        return None
        
    # Validation
    if card_int < 0 or card_int > 51:
        return None

    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['c', 'd', 'h', 's']
    
    # Reverse the formula: rank + (13 * suit)
    rank_idx = card_int % 13
    suit_idx = card_int // 13
    
    card_str = f"{ranks[rank_idx]}{suits[suit_idx]}"
    return eval7.Card(card_str)

def encode_card(card):
    # converts card to int between 0 and 51
    return card.rank + (13 * card.suit)

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

def load_agents(num_players: int, agent_types: list, starting_stack: int, action_space_n: int) -> list:
    # Local imports to avoid circular import with Player -> utils
    from agents.TemperalDifference.PokerQLearning import PokerQLearning
    from environments.Poker.Player import HeuristicPlayer, RandomPlayer
    players = []
    assert len(agent_types) == num_players
    for i, a_type in enumerate(agent_types):
        if a_type == 'qlearning': p = PokerQLearning(i, starting_stack, action_space_n)
        elif a_type == 'random': p = RandomPlayer(starting_stack, i)
        else: p = HeuristicPlayer(starting_stack, i)
        players.append(p)
    return players

