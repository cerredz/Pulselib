from math import trunc
from agents.FirstVisitMonteCarlo import FirstVisitMonteCarlo
from scripts.TFE.train import NUM_EPISODES
from utils.config import get_config_file
from typing import List
import gymnasium as gym
import numpy as np
from utils.plotting import plot_learning_curve
from pathlib import Path

CONFIG_FILENAME="fvmc.yaml"
ENV_ID="Pulse-2048-v1"
RESULTS_DIR=Path(__file__).parent.parent.parent/"results"/"2048" 
PLOT_FILENAME="fv_monte_carlo_learning_curve" 

if __name__ == "__main__":
    config=get_config_file(file_name=CONFIG_FILENAME)
    agent=FirstVisitMonteCarlo(gamma=config["GAMMA"])
    plot_filepath=RESULTS_DIR/PLOT_FILENAME
    env = gym.make(ENV_ID)
    print(config)

    scores: List[float] = []
    steps=0

    for i in range(config["NUM_EPISODES"]):
        raw_state, info = env.reset()
        state = tuple(raw_state.flatten())
        episode: List[tuple] = []
        terminated, truncated, episode_score = False, False, 0
        
        while not terminated and not truncated:
            action=agent.action(env.action_space)
            next_state, reward, terminated, truncated, total_score = env.step(action)
            episode.append((state, action, reward))
            state=tuple(next_state.flatten())
            steps += 1
            episode_score += reward
            break

        agent.learn(episode=episode)
        scores.append(episode_score)
        
        if i % 50 == 0:
            print(f"Episode {i}: Score: {episode_score}, Avg Score (last 50): {np.mean(scores[-50:]):.2f}")
            
    plot_learning_curve(scores, plot_filepath, window_size=100, title="DQN Agent on 2048")