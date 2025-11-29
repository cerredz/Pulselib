# utility functions for our poker environment
import math

def encode_card(card):
    """
    Converts eval7 Card object to a fixed vector.
    We use Rank (0-12) and Suit (0-3). 
    Returns: [Rank, Suit] (2 ints) for Q-Learning or [OneHot] for NN.
    For this example, we return simple Ints to keep state small.
    """
    if card is None: return [0, 0]
    return [card.rank, card.suit]

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

