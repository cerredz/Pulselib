from agents import DQN_agent 
#from agents.DQN_agent import DQNAgent 
import environments 
#from environments.TwentyFourtyEight import Game2048Env 
from pathlib import Path 
from typing import List 
import gymnasium as gym
from collections import deque 
import yaml

from models.tfe import TFE, TFELightning 
from utils.ReplayBuffer import ReplayBuffer 
from utils.plotting import plot_learning_curve 
import numpy as np 
import torch
import time

NUM_EPISODES=5000
ENV_ID = 'Pulse-2048-v1' 
RESULTS_DIR=Path(__file__).parent.parent.parent/"results"/"2048" 
PLOT_FILENAME="random_agent_learning_curve" 
SCORES_FILENAME="scores.csv" 
MODEL_WEIGHTS_FILENAME="tfe_light_model_weights.pt" 
CAPACITY=1000000
NUM_ENVS=16

if __name__ == "__main__":
    with open("config/tfe.yaml", 'r') as file: 
        config=yaml.safe_load(file)

    print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    plot_filepath=RESULTS_DIR/PLOT_FILENAME
    env = gym.make_vec(ENV_ID, num_envs=NUM_ENVS, vectorization_mode="sync")    
    model=TFELightning(lr=config['learning_rate']).to(device)

    agent=DQNAgent(
        action_space=env.single_action_space, 
        model=model,
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_decay=config['epsilon_decay'],
        epsilon_end=config['epsilon_end'],
        batch_size=config['batch_size'],
        weight_decay=config['weight_decay'],
        target_update=config['target_update']
    )

    agent.model.to(device)
    agent.target_model.to(device)

    buffer=ReplayBuffer(file_path=RESULTS_DIR/SCORES_FILENAME, capacity=CAPACITY)
    scores: List[float]=[]
    losses: List[float]=[]

    print(f"Running agent on {ENV_ID} with {NUM_ENVS} environments...")

    states, _ = env.reset()
    episode_scores = np.zeros(NUM_ENVS)
    completed_episodes = 0

    start_time = time.time()
    total_steps = 0

    while completed_episodes < NUM_EPISODES:
        actions = [agent.action(s) for s in states]
        
        next_states, rewards, terminateds, truncateds, infos = env.step(actions)
        total_steps += NUM_ENVS
        
        for i in range(NUM_ENVS):
            done = terminateds[i] or truncateds[i]
            buffer.add(states[i], actions[i], rewards[i], next_states[i], done)
            episode_scores[i] += rewards[i]
            
            if done:
                real_score = infos["total_score"][-1]
                scores.append(real_score)
                episode_scores[i] = 0
                completed_episodes += 1
                if completed_episodes % 10 == 0:
                    elapsed_time = time.time() - start_time
                    sps = total_steps / elapsed_time
                    print(f"Episodes {completed_episodes}/{NUM_EPISODES}. Last Score: {scores[-1]}. Avg Loss: {np.mean(losses[-100:]) if losses else 0:.4f}. SPS: {int(sps)}")        
        states = next_states

        if len(buffer) > agent.batch_size:
            batch=buffer.sample(agent.batch_size)
            loss=agent.learn(batch)
            losses.append(loss)

    env.close()
    torch.save(model.state_dict(), RESULTS_DIR/MODEL_WEIGHTS_FILENAME)
    plot_learning_curve(scores, plot_filepath, window_size=100, title="DQN Agent on 2048")