
from environments.Poker.Player import PokerQNetwork
from environments.Poker.utils import PokerAgentType, build_actions, load_agents, load_gpu_agents
#from scripts.TFE.train import next_states
from utils.config import get_config_file, get_result_folder
import gymnasium as gym
from utils.plotting import plot_learning_curve
from utils.torch import load_device
from cProfile import Profile
import torch
import pstats
import time                         

CONFIG_FILENAME="pokerGPU.yaml"
PLOT_FILENAME="rewards_learning_curve"
POKER_ACTION_SPACE_N=13

def train_agent(env: gym.Env, agents, agent_types, episodes, n_games, device, results_dir):
    g=torch.arange(n_games, device=device, dtype=torch.int32)
    total_steps = 0
    start_time = time.time()

    q_agent_idx=agent_types.index(PokerAgentType.QLEARNING)
    scores=[]
    
    for episode in range(episodes):
        q_seat = episode % 7
        rotation = (q_seat - q_agent_idx) % 7

        rotated_agents = agents[-rotation:] + agents[:-rotation] if rotation else agents.copy()
        rotated_types = agent_types[-rotation:] + agent_types[:-rotation] if rotation else agent_types.copy()

        state, info = env.reset(options={'rotation': rotation, 'active_players': True})
        terminated = torch.zeros(n_games, dtype=torch.bool, device=device)
        episode_reward=0
        
        while terminated.float().mean() < .9:
            curr_player_idxs = state[:, 8].long()
            actions = build_actions(state, curr_player_idxs, rotated_agents, rotated_types, device)
            next_state, rewards, dones, truncated, info = env.step(actions)
            #q_mask = (curr_player_idxs == q_seat[0:info["active_players"]])
            q_mask = (curr_player_idxs == q_seat)


            if q_mask.any():
                loss = q_net.train_step(
                    states=state[q_mask],
                    actions=actions[q_mask],
                    rewards=rewards[q_mask],
                    next_states=next_state[q_mask],
                    dones=dones[q_mask]
                )
            episode_reward += rewards[q_mask].sum().item() if q_mask.any() else 0
            #replay_buffer.add(state, actions, rewards, next_state, dones)
            state = next_state
            terminated |= dones
            total_steps += n_games

        scores.append(episode_reward)
        elapsed = time.time() - start_time
        steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
            print(f"Episode {episode+1:5d}/{episodes} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Speed: {steps_per_sec:6.1f} steps/sec")

    torch.save(q_net.network.state_dict(), f"{results_dir}/poker_qnet_final.pth")
    
    plot_path=results_dir/PLOT_FILENAME
    plot_learning_curve(
        scores=scores, 
        file_path=str(plot_path), 
        window_size=10, 
        title="Poker Q-Learning – Total Reward per Episode Batch"
    )

if __name__ == "__main__":
    config=get_config_file(file_name=CONFIG_FILENAME)
    result_dir=get_result_folder(config["RESULTS_DIR"])
    q_learning_model_weights=result_dir/"poker_qnet_final.pth"

    device=load_device()
    agents, agent_types=load_gpu_agents(device, config["NUM_PLAYERS"], config["AGENTS"], config["STARTING_BBS"], POKER_ACTION_SPACE_N)
    q_net=PokerQNetwork(weights_path=q_learning_model_weights, device=device, gamma=config["GAMMA"], update_freq=config["UPDATE_FREQ"], state_dim=config["STATE_SPACE"], action_dim=config["ACTION_SPACE"]).to(device)

    agents.append(q_net)
    agent_types.append(PokerAgentType.QLEARNING)
    
    env=gym.make(
        config["ENV_ID"],
        device=device,
        agents=agents, 
        n_players=config["NUM_PLAYERS"]+1, # account for the manually added q-net
        n_games=config["N_GAMES"], 
        starting_bbs=config["STARTING_BBS"], 
        w1=config["W1"],
        w2=config["W2"],
        K=config["K"]
    )
    profiler = Profile()
    profiler.enable()
    train_agent(env, agents, agent_types, config["EPISODES"], config["N_GAMES"], device=device, results_dir=result_dir)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20) 
