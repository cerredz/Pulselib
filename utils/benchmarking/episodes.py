# benchmarking utility function for episode values
import torch
import numpy as np
import torch

def benchmark_episode(episodes_return):
    assert isinstance(episodes_return, list) or isinstance(episodes_return, torch.Tensor), 'episode return must be either list or torch tensor'
    if isinstance(episodes_return, list): 
        episode_return = np.array(episodes_return)
    elif isinstance(episodes_return, torch.Tensor):
        episode_return = episodes_return.detach().cpu().numpy()

    mean = episode_return.mean()
    std = episode_return.std()
    mi = episode_return.min()
    ma = episode_return.max()
    median = np.median(episode_return)
    l = len(episode_return)

    return mean, std, mi, ma, median, l


