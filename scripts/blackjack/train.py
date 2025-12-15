import torch
from utils.config import get_config_file
from utils.torch import load_device
import gymnasium as gym
import environments

def train_agent(env, device, config):
    n=config["NUM_EPISODES"]
    episode_rewards=torch.zeros(n, device=device)
    b=0

    for i in range(config["NUM_EPISODES"]):
        state, info = env.reset()
        terminated = torch.zeros(n, device=device, dtype=torch.int32)
        truncated=torch.zeros(n, device=device, dtype=torch.int32)

        while terminated.float().mean() < .95:
            actions=torch.randint(0, 2, (n,), device=device, dtype=torch.int32)
            next_states, rewards, dones, truncated, info = env.step(actions)
            terminated |= dones
            print(rewards)

        episode_rewards[i] = rewards.sum()   

        if i % 10 == 0:
            print(f"Rewards: {episode_rewards[i*b:i*b+10].mean()}")
            b += 1

if __name__ == "__main__":
    CONFIG_FILENAME="blackjack.yaml"
    config=get_config_file(file_name=CONFIG_FILENAME)
    device=load_device()

    env=gym.make(
        config["ENV_ID"],
        device=device,
        batch_size=config["BATCH_SIZE"]
    )

    train_agent(env, device, config)
