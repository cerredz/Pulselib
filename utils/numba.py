from numba import njit
import random
# numba function to choose action A from a list of action values using e-greedy
# returns the index of the action to take
@njit('int32(float64[:], float64)')
def select_action_epsilon_greedy_numba(actions, epsilon):
    n = len(actions)
    p = random.random()
    
    if p < epsilon:
        return random.randint(0, n - 1)
    
    idx = 0
    max_val = actions[0]
    
    for i in range(1, n):
        if actions[i] > max_val:
            max_val = actions[i]
            idx = i

    return idx

# updates the q-table for q learning
# necessary parameters are parameters for the q-learning update formula
@njit('void(float64[:], int32, float64[:], float64, float64, float64, boolean)')
def update_q_entry(current_q_vals, action, next_q_vals, alpha, reward, gamma, is_terminal):
    
    # 1. Calculate Max Q(S', a)
    # We initialize with the first element, not -inf, to be safe and fast
    max_next_q = next_q_vals[0]
    for i in range(1, len(next_q_vals)):
        if next_q_vals[i] > max_next_q:
            max_next_q = next_q_vals[i]
            
    # 2. Calculate Target
    # If terminal, future value is 0
    if is_terminal:
        target = reward
    else:
        target = reward + gamma * max_next_q
        
    # 3. Update Q(S, A)
    # Q(S, A) <- Q(S, A) + alpha * [Target - Q(S, A)]
    old_val = current_q_vals[action]
    current_q_vals[action] = old_val + alpha * (target - old_val)





