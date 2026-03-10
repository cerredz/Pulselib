from agents.TemperalDifference.DDPG import DDPG
from agents.TemperalDifference.DQN import DQN
from environments.blackjack.utils import blackjack_training_utils
import torch
from utils import ReplayBuffer
from utils.agents import default_actor_critic_params, default_ddpg_actor_critic
from utils.config import get_config_file, get_result_folder
from utils.plotting import plot_learning_curve
from utils.torch import load_device
import gymnasium as gym
import environments
import time
from torchrl.data import ReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict

def train_agent(env, device, config, agent, results_dir, replay_buffer=None):
    n = config["NUM_EPISODES"]
    episode_rewards = torch.zeros(n, device=device)
    start_time = time.time()
    total_steps = 0
    batch_size = config["BATCH_SIZE"]
    
    terminated = torch.zeros(batch_size, device=device, dtype=torch.bool)

    for i in range(config["NUM_EPISODES"]):
        states, info = env.reset()
        terminated.fill_(0)
        episode_reward = 0

        while terminated.float().mean() < .95:
            active_mask = ~terminated
            states = states.float().to(device)
            actions = agent.action(states)
            next_states, rewards, dones, truncated, info = env.step(actions)
            episode_reward += rewards[active_mask].sum().item()
            total_steps += active_mask.sum().item()
            next_states = next_states.float().to(device)
            
            # Store transitions in replay buffer
            transition = TensorDict({
                "state": states,
                "action": actions,
                "reward": rewards,
                "next_state": next_states,
                "done": dones
            }, batch_size=[batch_size])
            replay_buffer.extend(transition)
            
            # Train from replay buffer
            if len(replay_buffer) >= config["MIN_SAMPLES"]:
                batch = replay_buffer.sample()
                agent.train_step(
                    batch["state"], 
                    batch["action"], 
                    batch["reward"], 
                    batch["next_state"], 
                    batch["done"]
                )
            
            terminated |= dones
            states = next_states

        agent.reset_noise()
        episode_rewards[i] = episode_reward
       
        if (i + 1) % 500 == 0:
            start_idx = max(0, i - 500)
            avg_reward = episode_rewards[start_idx:i+1].mean().item()
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
            print(f"Episode {i+1:5d}/{n} | "
                  f"Avg Reward (last 500): {avg_reward:8.2f} | "
                  f"Speed: {steps_per_sec:6.1f} steps/sec")

    scores = episode_rewards.detach().cpu().tolist()
    plot_learning_curve(
        scores=scores, 
        file_path=str(results_dir/"reward_learning_curve"), 
        window_size=10,
        title="DDPG Blackjack – Total Reward per Episode Batch"
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

    ddpg_actor, ddpg_critic = default_ddpg_actor_critic(env, config["STATE_DIM"])
    
    storage = LazyMemmapStorage(config['CAPACITY'])
    replay_buffer = ReplayBuffer(storage=storage, batch_size=config["BATCH_SIZE"])  

    agent = DDPG(
        env_action_space=env.action_space,
        state_dim=config["STATE_DIM"],
        device=device,
        gamma=config["Q_LEARNING_RATE"],
        criterion=criterion,
        learning_rate=config["LEARNING_RATE"],
        weight_decay=config["WEIGHT_DECAY"],
        actor_network=ddpg_actor,
        critic_network=ddpg_critic,
        update=config["UPDATE"],
        replay_buffer=replay_buffer,
        batch_size=config["BATCH_SIZE"],
        actor_optimizer=None,
        critic_optimizer=None,
        target_actor_network=None,
        target_critic_network=None,
        actor_weights_path=None,
        critic_weights_path=None,
        mu=config["MU"],
        theta=config["THETA"],
        sigma=config["SIGMA"]
    )

    #params=default_actor_critic_params(env, config["STATE_DIM"], device)
    #agent=ActorCritic(**params)

    train_agent(env, device, config, agent, results_dir, replay_buffer=replay_buffer)