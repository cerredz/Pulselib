import torch
import torch.nn as nn
import time
from torchrl.data import ReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
from agents.TemperalDifference.DDPG import DDPG
import environments
from environments.Particle2D.Particle2D import Particle2D

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x): return self.net(x)

def train_agent(env, agent, config, device):
    episode_rewards = torch.zeros(config["NUM_EPISODES"], device=device)
    start_time = time.time()
    total_steps = 0
    batch_size = config["BATCH_SIZE"]
    
    storage = LazyMemmapStorage(config['CAPACITY'])
    replay_buffer = ReplayBuffer(storage=storage, batch_size=batch_size)
    
    for i in range(config["NUM_EPISODES"]):
        states, _ = env.reset()
        terminated = torch.zeros(batch_size, device=device, dtype=torch.bool)
        episode_reward = 0

        while terminated.float().mean() < 0.95:
            active_mask = ~terminated
            actions = agent.action(states)
            next_states, rewards, dones, _, _ = env.step(actions)
            episode_reward += rewards[active_mask].sum().item()
            total_steps += active_mask.sum().item()
            
            replay_buffer.extend(TensorDict({
                "state": states, "action": actions, "reward": rewards,
                "next_state": next_states, "done": dones
            }, batch_size=[batch_size]))
            
            if len(replay_buffer) >= config["MIN_SAMPLES"]:
                batch = replay_buffer.sample()
                agent.train_step(batch["state"], batch["action"], batch["reward"], 
                               batch["next_state"], batch["done"])
            
            terminated |= dones
            states = next_states

        agent.reset_noise()
        episode_rewards[i] = episode_reward
       
        if (i + 1) % 5 == 0:
            avg_reward = episode_rewards[max(0, i-99):i+1].mean().item()
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
            print(f"Episode {i+1:5d}/{config['NUM_EPISODES']} | "
                  f"Avg Reward (last 100): {avg_reward:8.2f} | "
                  f"Speed: {steps_per_sec:6.1f} steps/sec")

    return episode_rewards.cpu().numpy()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        "BATCH_SIZE": 256,
        "NUM_EPISODES": 1000,
        "STATE_DIM": 4,
        "ACTION_DIM": 2,
        "LEARNING_RATE": 1e-3,
        "WEIGHT_DECAY": 1e-4,
        "Q_LEARNING_RATE": 0.99,  # gamma
        "CAPACITY": 100000,
        "MIN_SAMPLES": 1000,
        "UPDATE": 1,
        "MU": 0.0,
        "THETA": 0.15,
        "SIGMA": 0.2
    }
    
    env = Particle2D(device=device, batch_size=config["BATCH_SIZE"])
    
    agent = DDPG(
        env_action_space=env.action_space,
        state_dim=config["STATE_DIM"],
        device=device,
        gamma=config["Q_LEARNING_RATE"],
        criterion=nn.MSELoss(),
        learning_rate=config["LEARNING_RATE"],
        weight_decay=config["WEIGHT_DECAY"],
        actor_network=Actor(config["STATE_DIM"], config["ACTION_DIM"]),
        critic_network=Critic(config["STATE_DIM"], config["ACTION_DIM"]),
        target_actor_network=None,
        target_critic_network=None,
        batch_size=config["BATCH_SIZE"],
        update=config["UPDATE"],
        replay_buffer=None,
        mu=config["MU"],
        theta=config["THETA"],
        sigma=config["SIGMA"]
    )
    
    rewards = train_agent(env, agent, config, device)
    print(f"\nTraining complete! Final avg reward: {rewards[-100:].mean():.2f}")