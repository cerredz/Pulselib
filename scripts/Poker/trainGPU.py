
from environments.Poker.Player import PokerQNetwork
from environments.Poker.utils import PokerAgentType, build_actions, load_agents, load_gpu_agents
from utils.config import get_config_file
import gymnasium as gym
from utils.torch import load_device
import torch

CONFIG_FILENAME="pokerGPU.yaml"
POKER_ACTION_SPACE_N=13

def train_agent(env: gym.Env, agents, agent_types, episodes, n_games, device):
    g=torch.arange(n_games, device=device, dtype=torch.int32)
    for i in range(episodes):
        state, info = env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            curr_player_idxs=state[:, 8].long()
            actions=build_actions(state, curr_player_idxs, agents, agent_types, device)
            next_state, rewards, dones, truncated, info = env.step(actions)
            state=next_state
            terminated |= dones

if __name__ == "__main__":
    config=get_config_file(file_name=CONFIG_FILENAME)
    agents, agent_types=load_gpu_agents(config["NUM_PLAYERS"], config["AGENTS"], config["STARTING_BBS"], POKER_ACTION_SPACE_N)
    
    q_net=PokerQNetwork(state_dim=config["STATE_SPACE"], action_dim=config["ACTION_SPACE"])
    target_net=PokerQNetwork(state_dim=config["STATE_SPACE"], action_dim=config["ACTION_SPACE"])
    target_net.load_state_dict(q_net.state_dict())

    agents.append(q_net)
    agents.append(target_net)
    agent_types.append(PokerAgentType.QLEARNING)
    agent_types.append(PokerAgentType.QLEARNING)
    
    device=load_device()
    print(device)

    env=gym.make(
        config["ENV_ID"],
        device=device,
        agents=agents, 
        n_players=config["NUM_PLAYERS"],
        n_games=config["N_GAMES"], 
        starting_bbs=config["STARTING_BBS"], 
    )

    train_agent(env, agents, agent_types, config["EPISODES"], config["N_GAMES"], device=device)
