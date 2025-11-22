import time
import gymnasium as gym
import numpy as np
from pathlib import Path
from typing import List, Tuple
from utils.agents import get_agent
from utils.config import get_config_file, get_paths
from utils.plotting import plot_learning_curve
import environments.TFE

CONFIG_FILENAME = "qlearning.yaml"

if __name__ == "__main__":
    # 1. Load Config
    config = get_config_file(CONFIG_FILENAME)
    print(f"Loaded Config: {config}")
    reward_path, score_path, steps_path = get_paths(config)
    
    env = gym.make(
        config["ENV_ID"], 
        board_height=config.get("BOARD_HEIGHT", 4), 
        board_width=config.get("BOARD_WIDTH", 4)
    )
    
    agent = get_agent(config["AGENT_ID"], env, config)

    rewards_history: List[float] = []
    game_scores_history: List[float] = []
    
    global_step_count = 0
    start_time = time.time()

    print(f"Starting training for {config['NUM_EPISODES']} episodes...")

    # 5. Training Loop
    for episode in range(config["NUM_EPISODES"]):
        raw_state, info = env.reset()
        state = tuple(raw_state.flatten())
        
        terminated, truncated = False, False
        episode_reward = 0
        final_game_score = 0

        while not terminated and not truncated:
            # A. Select Action
            action = agent.get_action(state)

            # B. Step Environment
            next_raw_state, reward, terminated, truncated, info = env.step(action)
            next_state = tuple(next_raw_state.flatten())

            # C. Learn (Q-Learning updates every step)
            agent.update(state, action, next_state, reward, terminated)

            # D. Updates
            state = next_state
            episode_reward += reward
            global_step_count += 1
            final_game_score = info['score']

        # End of Episode Logging
        rewards_history.append(episode_reward)
        game_scores_history.append(final_game_score)

        # Logging Interval
        if (episode + 1) % config["SAVE_INTERVAL"] == 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = global_step_count / elapsed_time
            avg_reward = np.mean(rewards_history[-50:])
            
            print(f"Episode {episode + 1}: "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Game Score: {final_game_score} | "
                  f"Speed: {steps_per_sec:.0f} steps/sec")

    # 6. Final Plotting
    print("Training Complete. Saving plots...")
    
    plot_learning_curve(rewards_history, reward_path, window_size=100, title=f"{config['AGENT_ID']} Rewards")
    plot_learning_curve(game_scores_history, score_path, window_size=100, title=f"{config['AGENT_ID']} Game Scores")
    
    print(f"Best Game Score Achieved: {max(game_scores_history)}")