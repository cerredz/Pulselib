

from agents.random_agent import RandomAgent
import environments
from environments.TwentyFourtyEight import Game2048Env
from pathlib import Path
from typing import List
import gymnasium as gym
from collections import deque

from utils.ReplayBuffer import ReplayBuffer
from utils.plotting import plot_learning_curve

NUM_EPISODES=10000
ENV_ID = 'Pulse-2048-v1'
RESULTS_DIR=Path(__file__).parent.parent.parent/"results"/"2048"
PLOT_FILENAME="random_agent_learning_curve"
SCORES_FILENAME="scores.csv"

if __name__ == "__main__":
    plot_filepath=RESULTS_DIR/PLOT_FILENAME
    env=gym.make(ENV_ID)
    agent=RandomAgent(action_space=env.action_space)
    buffer=ReplayBuffer(file_path=RESULTS_DIR/SCORES_FILENAME)
    scores: List[float]=[]
    
    print(f"Running agent on {ENV_ID} for {NUM_EPISODES} episodes...")

    for i in range(NUM_EPISODES):
        state = env._get_obs()
        terminated, truncated, score = False, False, 0
        env.reset()

        while not terminated and not truncated:
            action=agent.action()
            board, reward, terminated, truncated, env_info= env.step(action)
            score += reward

        scores.append(score)
        if (i + 1) % 100 == 0:
            print(f"Episode {i + 1}/{NUM_EPISODES} finished. Score: {score}")
    
    env.close()
    plot_learning_curve(scores, plot_filepath, title="Random Agent on 2048")