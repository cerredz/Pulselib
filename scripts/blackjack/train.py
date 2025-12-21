from agents.TemperalDifference.ActorCritic import ActorCritic
from agents.TemperalDifference.DQN import DQN
from agents.TemperalDifference.DoubleDQN import DoubleDQN
from environments.blackjack.utils import blackjack_training_utils
import torch
from utils.agents import default_actor_critic_params
from utils.config import get_config_file, get_result_folder
from utils.plotting import plot_learning_curve
from utils.torch import load_device
import gymnasium as gym
import environments
import time


def train_agent(env, device, config, agent, results_dir):
    n = config["NUM_EPISODES"]
    episode_rewards = torch.zeros(n, device=device)
    start_time = time.time()
    total_steps = 0
    batch_size = config["BATCH_SIZE"]

    for i in range(config["NUM_EPISODES"]):
        states, info = env.reset()
        terminated = torch.zeros(batch_size, device=device, dtype=torch.bool)
        truncated = torch.zeros(batch_size, device=device, dtype=torch.bool)
        episode_reward = 0

        while terminated.float().mean() < .95:
            active_mask = ~terminated
            states=states.float().to(device)
            actions = agent.action(states)
            next_states, rewards, dones, truncated, info = env.step(actions)
            episode_reward += rewards[active_mask].sum().item()
            total_steps += active_mask.sum().item()
            next_states=next_states.float().to(device)
            agent.train_step(states[active_mask], actions[active_mask], rewards[active_mask], next_states[active_mask], dones[active_mask])
            terminated |= dones
            states = next_states

        #agent.decay_epsilon()
        episode_rewards[i] = episode_reward
       
        if (i + 1) % 500 == 0:
            # Calculate mean of last 10 episodes
            start_idx = max(0, i - 500)
            avg_reward = episode_rewards[start_idx:i+1].mean().item()
            
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
            print(f"Episode {i+1:5d}/{n} | "
                  f"Avg Reward (last 50): {avg_reward:8.2f} | "
                  f"Speed: {steps_per_sec:6.1f} steps/sec")

    scores = episode_rewards.detach().cpu().tolist()

    plot_learning_curve(
        scores=scores, 
        file_path=str(results_dir/"reward_learning_curve"), 
        window_size=10,
        title="Blackjack Q Learning – Total Reward per Episode Batch"
    )

if __name__ == "__main__":
    CONFIG_FILENAME = "blackjack.yaml"
    config = get_config_file(file_name=CONFIG_FILENAME)
    results_dir=get_result_folder(config["RESULTS_DIR"])
    device = load_device()
    network, target_network, criterion = blackjack_training_utils(config) 
    
    env = gym.make(
        config["ENV_ID"],
        device=device,
        batch_size=config["BATCH_SIZE"]
    )
    """
    agent = DoubleDQN(
        env_action_space=env.action_space,
        state_dim=config["STATE_DIM"],
        device=device,
        gamma=config["Q_LEARNING_RATE"],
        criterion=criterion,
        learning_rate=config["LEARNING_RATE"],
        weight_decay=config["WEIGHT_DECAY"],
        network=network,
        epsilon=config["EPSILON"],
        update=config["UPDATE"],
        epsilon_decay=config["EPSILON_DECAY"],
        epsilon_min=config["EPSILON_MIN"],
        optimizer=None,
        target_network=None,
        weights_path=None,
    )
    """

    params=default_actor_critic_params(env, config["STATE_DIM"], device)
    agent=ActorCritic(**params)

    train_agent(env, device, config, agent, results_dir)