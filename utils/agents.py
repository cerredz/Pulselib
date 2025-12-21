import gymnasium as gym
import torch
import torch.nn as nn
"""
def get_agent(agent_id: str, env: gym.Env, config: dict):
    if agent_id == "q_learning":
        return QLearning(env, config)
    # elif agent_id == "monte_carlo":
    #     return OnPolicyFirstVisitMC(...)
    else:
        raise ValueError(f"Unknown Agent ID: {agent_id}")
"""

def default_actor_critic_params(env, state_dim, device=torch.device("cpu")):
    n_actions = env.action_space.n
    
    actor_network=nn.Sequential(
        nn.Linear(state_dim, 32), nn.ReLU(), 
        nn.Linear(32, n_actions), nn.Softmax(dim=-1)
    )

    critic_network=nn.Sequential(
        nn.Linear(state_dim, 32), nn.ReLU(), 
        nn.Linear(32, 1)
    )

    return {
        'env_action_space': env.action_space,
        'state_dim': state_dim,
        'device': device,
        'actor_network': actor_network,
        'critic_network': critic_network,
        'gamma': 0.99,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4
    }