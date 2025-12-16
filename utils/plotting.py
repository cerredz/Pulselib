import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_learning_curve(scores: List[float], file_path: str, window_size: int = 100, 
                        title: str = "Agent Learning Curve", extend_plot: bool = False):
    data_file = file_path.rsplit('.', 1)[0] + '_scores.pkl'
    if extend_plot and os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            previous_scores = pickle.load(f)
        scores = previous_scores + scores
        print(f"Loaded {len(previous_scores)} previous scores. Now plotting {len(scores)} total.")
    scores_series = pd.Series(scores)
    moving_avg = scores_series.rolling(window=window_size).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(scores, 'b-', alpha=0.1, label='Raw Episode Score')
    ax.plot(moving_avg.index, moving_avg, 'r-', linewidth=2, label=f'Moving Average (Window={window_size})')
    ax.set_title(title)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(file_path)
        print(f"Plot successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    with open(data_file, 'wb') as f:
        pickle.dump(scores, f)
    plt.close(fig)

def plot_learning_curves(agent_scores: Dict[str, List[float]], file_path: str, window_size: int = 100,
                         title: str = "Multi-Agent Learning Curves", extend_plot: bool = False):
    data_file = file_path.rsplit('.', 1)[0] + '_scores.pkl'
    if extend_plot and os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            previous_scores = pickle.load(f)
        agent_scores = {agent: previous_scores.get(agent, []) + scores for agent, scores in agent_scores.items()}
        print(f"Loaded previous scores. Total episodes per agent: {[(k, len(v)) for k, v in agent_scores.items()]}")
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(range(len(agent_scores)))
    for (agent_name, scores), color in zip(agent_scores.items(), colors):
        scores_series = pd.Series(scores)
        moving_avg = scores_series.rolling(window=window_size).mean()
        ax.plot(scores, alpha=0.1, color=color)
        ax.plot(moving_avg.index, moving_avg, linewidth=2, label=f'{agent_name} (MA={window_size})', color=color)
    ax.set_title(title)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(file_path)
        print(f"Plot successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    with open(data_file, 'wb') as f:
        pickle.dump(agent_scores, f)
    plt.close(fig)