

from agents import DQN_agent
from agents.DQN_agent import DQNAgent
import environments
from environments.TwentyFourtyEight import Game2048Env
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

NUM_EPISODES=1000
ENV_ID = 'Pulse-2048-v1'
RESULTS_DIR=Path(__file__).parent.parent.parent/"results"/"2048"
PLOT_FILENAME="random_agent_learning_curve"
SCORES_FILENAME="scores.csv"
MODEL_WEIGHTS_FILENAME="tfe_light_model_weights.pt"
CAPACITY=100000
LEARN_EVERY=4

if __name__ == "__main__":
    # load the config
    with open("config/tfe.yaml", 'r') as file:
        config=yaml.safe_load(file)
    
    print(config)

    # initialize environment, agent, and replay buffer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_filepath=RESULTS_DIR/PLOT_FILENAME
    env=gym.make(ENV_ID)
    model=TFELightning(lr=config['learning_rate']).to(device)
    if Path.exists(RESULTS_DIR/MODEL_WEIGHTS_FILENAME):
        print('loading model weights...')
        model.load_state_dict(torch.load(RESULTS_DIR/MODEL_WEIGHTS_FILENAME))
    
    agent=DQNAgent(
        action_space=env.action_space,
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

    #agent=RandomAgent(action_space=env.action_space)
    buffer=ReplayBuffer(file_path=RESULTS_DIR/SCORES_FILENAME, capacity=CAPACITY)
    scores: List[float]=[]
    losses: List[float]=[]
    
    print(f"Running agent on {ENV_ID} for {NUM_EPISODES} episodes...")
    
    for i in range(NUM_EPISODES):
        state, info=env.reset()
        #print(state)
        terminated, truncated, score = False, False, 0
        step = 0

        while not terminated and not truncated:
            action=agent.action(state)
            next_state, reward, terminated, truncated, env_info= env.step(action)
            buffer.add(state, action, reward, next_state, terminated)
            state=next_state
            score += reward
            step += 1 

            if len(buffer) > agent.batch_size and step % LEARN_EVERY==0:
                batch=buffer.sample(agent.batch_size)
                loss=agent.learn(batch)
                losses.append(loss)

        scores.append(score)
        if (i + 1) % 1 == 0:
            print(f"Episode {i + 1}/{NUM_EPISODES} finished. Score: {score}. Avg loss: {np.mean(losses[-100:]) if losses else 0}")    
    env.close()
    torch.save(model.state_dict(), RESULTS_DIR/MODEL_WEIGHTS_FILENAME)
    plot_learning_curve(scores, plot_filepath, title="Random Agent on 2048")