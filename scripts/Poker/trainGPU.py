
from environments.Poker.Player import PokerQNetwork
from environments.Poker.utils import PokerAgentType, build_actions, get_rotated_agents, load_agents, load_gpu_agents
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
REWARDS_FILENAME="rewards_learning_curve"
CHIPS_FILENAME="total_chips_curve"
POKER_ACTION_SPACE_N=13

def train_agent(env: gym.Env, agents, agent_types, episodes, n_games, device, results_dir):
    total_steps = 0
    start_time = time.time()
    scores, reward_scores=[], []
    
    for episode in range(episodes):
        rotated_agents, rotated_types, q_seat, rotations = get_rotated_agents(
            agents, agent_types, episode_idx=episode
        )

        state, info = env.reset(options={
            'rotation': rotations,                    
            'active_players': True,
            'q_agent_seat': q_seat   
        })

        initial_stacks = info['stacks'][:, q_seat].clone()
        terminated = torch.zeros(n_games, dtype=torch.bool, device=device)
        episode_reward=0
        
        while terminated.float().mean() < .97:
            curr_player_idxs = state[:, 8].long()
            actions = build_actions(state, curr_player_idxs, rotated_agents, rotated_types, device)
            next_state, rewards, dones, truncated, info = env.step(actions)
            q_mask = (curr_player_idxs == q_seat)
            if q_mask.any():
                print("training")
                agents[agent_types.index(PokerAgentType.QLEARNING)].train_step(
                    states=state[q_mask],
                    actions=actions[q_mask],
                    rewards=rewards[q_mask],
                    next_states=next_state[q_mask],
                    dones=dones[q_mask]
                )

            episode_reward += rewards[q_mask].sum().item() if q_mask.any() else 0
            state = next_state
            terminated |= dones
            total_steps += n_games

        final_stacks=info['stacks'][:, q_seat]
        episode_profit = (final_stacks - initial_stacks).sum().item()
        reward_scores.append(episode_reward)
        scores.append(episode_profit)
        elapsed = time.time() - start_time
        steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
            print(f"Episode {episode+1:5d}/{episodes} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Q-Agent Profit: {episode_profit:10.2f} chips | "
                f"Speed: {steps_per_sec:6.1f} steps/sec")

    q_net = next(a for a in agents if isinstance(a, PokerQNetwork))
    torch.save(q_net.network.state_dict(), f"{results_dir}/poker_qnet_final.pth")
    reward_path=results_dir/REWARDS_FILENAME
    chips_path=result_dir/CHIPS_FILENAME

    plot_learning_curve(
        scores=reward_scores, 
        file_path=str(reward_path), 
        window_size=10, 
        title="Poker Q-Learning – Total Reward per Episode Batch"
    )

    plot_learning_curve(
        scores=scores, 
        file_path=str(chips_path), 
        window_size=10, 
        title="Poker Q-Learning – Total Chip profit per episode batch"
    )

if __name__ == "__main__":
    config=get_config_file(file_name=CONFIG_FILENAME)
    result_dir=get_result_folder(config["RESULTS_DIR"])
    q_learning_model_weights=result_dir/"poker_qnet_final.pth"

    device=load_device()
    agents, agent_types=load_gpu_agents(device, config["NUM_PLAYERS"], config["AGENTS"], config["STARTING_BBS"], POKER_ACTION_SPACE_N)
    q_net=PokerQNetwork(weights_path=q_learning_model_weights, device=device, gamma=config["GAMMA"], update_freq=config["UPDATE_FREQ"], state_dim=config["STATE_SPACE"], action_dim=config["ACTION_SPACE"]).to(device)

    agents.insert(0, q_net)
    agent_types.insert(0, PokerAgentType.QLEARNING)
    
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
