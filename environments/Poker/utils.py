# utility functions for our poker environment
import math
import enum
import eval7
import torch
import torch.optim as optim

_FULL_RANGE = eval7.HandRange("AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AKo,AQs,AQo,AJs,AJo,ATs,ATo,A9s,A9o,A8s,A8o,A7s,A7o,A6s,A6o,A5s,A5o,A4s,A4o,A3s,A3o,A2s,A2o,KQs,KQo,KJs,KJo,KTs,KTo,K9s,K9o,K8s,K8o,K7s,K7o,K6s,K6o,K5s,K5o,K4s,K4o,K3s,K3o,K2s,K2o,QJs,QJo,QTs,QTo,Q9s,Q9o,Q8s,Q8o,Q7s,Q7o,Q6s,Q6o,Q5s,Q5o,Q4s,Q4o,Q3s,Q3o,Q2s,Q2o,JTs,JTo,J9s,J9o,J8s,J8o,J7s,J7o,J6s,J6o,J5s,J5o,J4s,J4o,J3s,J3o,J2s,J2o,T9s,T9o,T8s,T8o,T7s,T7o,T6s,T6o,T5s,T5o,T4s,T4o,T3s,T3o,T2s,T2o,98s,98o,97s,97o,96s,96o,95s,95o,94s,94o,93s,93o,92s,92o,87s,87o,86s,86o,85s,85o,84s,84o,83s,83o,82s,82o,76s,76o,75s,75o,74s,74o,73s,73o,72s,72o,65s,65o,64s,64o,63s,63o,62s,62o,54s,54o,53s,53o,52s,52o,43s,43o,42s,42o,32s,32o")

def calculate_equity(player_hand, board, stage, num_active_players, player_status):
    # Fast exits for edge cases
    if player_status == 'folded':
        return 0.0
    if num_active_players == 1:
        return 1.0
    
    # Aggressive Monte Carlo iterations based on stage
    # Lower on early streets for speed, higher on later streets for accuracy
    sims = 500 if stage == 0 else 1000 if stage == 1 else 2000 if stage == 2 else 3000
    
    # Use pre-computed range for speed
    return eval7.py_hand_vs_range_monte_carlo(
        player_hand,
        _FULL_RANGE,
        board,
        sims
    )

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
    HEURISTIC="heuristic"
    RANDOM='random'
    HEURISTIC_HANDS='heuristic_hands'
    TIGHT_AGGRESSIVE="tight_aggressive"
    LOOSE_PASSIVE="loose_passive"
    SMALL_BALL="small_ball"

def load_agents(num_players: int, agent_types: list, starting_stack: int, action_space_n: int) -> list:
    # Local imports to avoid circular import with Player -> utils
    from agents.TemperalDifference.PokerQLearning import PokerQLearning
    from environments.Poker.Player import HeuristicPlayer, RandomPlayer
    players = []
    types=[]
    assert len(agent_types) == num_players
    for i, a_type in enumerate(agent_types):
        agent_type=None
        if a_type == 'random': 
            p = RandomPlayer(starting_stack, i)
            agent_type=PokerAgentType.RANDOM
        else:
            p = HeuristicPlayer(starting_stack, i)
            agent_type=PokerAgentType.HEURISTIC
        players.append(p)
        types.append(agent_type)
    return players, types

def build_actions(state, curr_players, agents, agent_types, device, epsilon=0.1):
    n_games = state.shape[0]
    actions = torch.zeros(n_games, dtype=torch.long, device=device)

    for agent_idx, agent_type in enumerate(agent_types):
        mask = (curr_players == agent_idx)
        agent_states = state[mask]
        if agent_type == PokerAgentType.QLEARNING:
            agent_actions=agents[agent_idx].get_actions(agent_states)
        elif agent_type == PokerAgentType.RANDOM:
            agent_actions = torch.randint(0, 13, (mask.sum(),), device=device)
        # heuristic players that we created
        else:
            agent_actions = agents[agent_idx].action(agent_states)
        actions[mask] = agent_actions
    return actions

def load_gpu_agents(device, num_players: int, agent_types: list, starting_stack: int, action_space_n: int) -> list:
    from environments.Poker.Player import HeuristicPlayer, RandomPlayer
    from environments.Poker.Player import HeuristicHandsPlayerGPU
    from environments.Poker.Player import PokerQNetwork
    from environments.Poker.Player import LoosePassivePlayerGPU, SmallBallPlayerGPU, TightAggressivePlayerGPU

    players = []
    types=[]
    assert len(agent_types) == num_players
    for i, a_type in enumerate(agent_types):
        agent_type=None
        p=None
        if a_type == 'random': 
            p = RandomPlayer(starting_stack, i)
            agent_type = PokerAgentType.RANDOM
        elif a_type == 'heuristic_hands': 
            p = HeuristicHandsPlayerGPU(starting_stack, i, device)
            agent_type = PokerAgentType.HEURISTIC_HANDS
        elif a_type == 'tight_aggressive':
            p = TightAggressivePlayerGPU(starting_stack, i, device)
            agent_type = PokerAgentType.TIGHT_AGGRESSIVE
        elif a_type == 'loose_passive':
            p = LoosePassivePlayerGPU(starting_stack, i, device)
            agent_type = PokerAgentType.LOOSE_PASSIVE
        elif a_type == 'small_ball':
            p = SmallBallPlayerGPU(starting_stack, i, device)
            agent_type = PokerAgentType.SMALL_BALL
        else:
            raise ValueError(f"Unknown agent type: {a_type}")

        players.append(p)
        types.append(agent_type)
    return players, types

def debug_state(st, pid, aid):
    """Debug poker state - prints state, player ID, and action."""
    actions = ["Fold", "Check/Call", "MinRaise", "R25%", "R33%", "R50%", "R75%", "R100%", "R150%", "R200%", "R300%", "R400%", "All-in"]
    stages = ["Preflop", "Flop", "Turn", "River", "Game Over"]
    
    board = ' '.join(str(decode_card(c)) for c in st[0:5] if c != 0) or "(empty)"
    hand = f"{decode_card(st[5])} {decode_card(st[6])}"
    n_opp = (len(st) - 12) // 3
    opps = ', '.join(f"O{i+1}:{st[12+i*3]}BB/{'A' if st[13+i*3] else 'F'}/{st[14+i*3]}BB" 
                     for i in range(n_opp))
    
    print(f"Player: {pid} Action: {actions[aid]} | {stages[st[7]]} | Hand:{hand} Stack:{st[11]}BB Pos:{st[8]}\n"
          f"Board:{board} Pot:{st[9]}BB Call:{st[10]}BB | {opps}")

def get_rotated_agents(agents, agent_types, episode_idx=None):
    n = len(agents)
    q_idx = agent_types.index(PokerAgentType.QLEARNING)           # original index of Q-agent
    target_seat = (episode_idx % n) if episode_idx is not None else 0
    rotation = (target_seat - q_idx) % n
    
    rotated_agents = agents[-rotation:] + agents[:-rotation]
    rotated_types  = agent_types[-rotation:] + agent_types[:-rotation]
    new_q_seat = target_seat                                      # Q-agent now sits here
    
    return rotated_agents, rotated_types, new_q_seat, rotation