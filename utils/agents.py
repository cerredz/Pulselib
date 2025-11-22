from agents.TemperalDifference.QLearning import QLearning
import gymnasium as gym

def get_agent(agent_id: str, env: gym.Env, config: dict):
    if agent_id == "q_learning":
        return QLearning(env, config)
    # elif agent_id == "monte_carlo":
    #     return OnPolicyFirstVisitMC(...)
    else:
        raise ValueError(f"Unknown Agent ID: {agent_id}")