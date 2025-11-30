from math import trunc
from agents.FirstVisitMonteCarlo import FirstVisitMonteCarlo
from agents.MonteCarlo.OnPolicyFirstVisit import OnPolicyFirstVisitMC
from scripts.TFE.train import NUM_EPISODES
from utils.config import get_config_file
from typing import List
import gymnasium as gym
import numpy as np
from utils.plotting import plot_learning_curve
from pathlib import Path
import environments.TFE
import time  # <--- 1. Added import

CONFIG_FILENAME="on_policy_first_visit_monte_carlo.yaml"
ENV_ID="Pulse-2048-v2"
RESULTS_DIR=Path(__file__).parent.parent.parent/"results"/"2048" 
PLOT_FILENAME="fv_monte_carlo_learning_curve"
SCORES_FILENAME="fv_monte_carlo_scores"

if __name__ == "__main__":
    config=get_config_file(file_name=CONFIG_FILENAME)
    print(config)
    plot_filepath=RESULTS_DIR/PLOT_FILENAME
    score_filepath=RESULTS_DIR/SCORES_FILENAME

    env = gym.make(ENV_ID, board_height=3, board_width=3)
    agent=OnPolicyFirstVisitMC(gamma=config["GAMMA"], epsilon=config["EPSILON"], action_space=env.action_space)
    scores: List[float] = []
    final_scores: List[float]=[]
    
    steps=0
    start_time = time.time() # <--- 2. Initialize start time

    for i in range(config["NUM_EPISODES"]):
        raw_state, info = env.reset()
        state = tuple(raw_state.flatten())
        episode: List[tuple] = []
        terminated, truncated, episode_score = False, False, 0
        
        while not terminated and not truncated:
            action=agent.action(state)
            next_state, reward, terminated, truncated, total_score = env.step(action)
            episode.append((state, action, reward))
            state=tuple(next_state.flatten())
            steps += 1
            episode_score += reward

        agent.learn(episode=episode)
        scores.append(episode_score)
        final_scores.append(total_score["score"])
        
        if i % 5000 == 0:
            # 3. Updated print statement to calculate total steps divided by elapsed time
            print(f"Episode {i}: Score: {episode_score}, Avg Score (last 50): {np.mean(scores[-50:]):.2f}, Final score: {final_scores[-1]}, Steps/sec: {steps / (time.time() - start_time):.2f}")
            
    plot_learning_curve(scores, plot_filepath, window_size=5000, title="DQN Agent on 2048")
    plot_learning_curve(final_scores, score_filepath, window_size=5000, title="DQN Agent on 2048")

    print(f"Highest Score: {max(final_scores)}")