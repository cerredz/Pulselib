import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List

def plot_learning_curve(scores: List[float], file_path: str, window_size: int = 100, title: str = "Agent Learning Curve"):
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
    plt.close(fig)