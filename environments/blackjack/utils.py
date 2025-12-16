
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# some default agent architectures, loss function, and optimizers that can be used for 
# training on the blackjack environment

def blackjack_training_utils(config):
    assert config["STATE_DIM"] is not None, "blackjack-training_utils requires STATE_DIM in config file"
    assert config['ACTION_DIM'] is not None, "blackjack-training_utils requires ACTION_DIM in config file"
    state_dim, action_dim =config["STATE_DIM"], config["ACTION_DIM"]
    network=basic_blackjack_network(state_dim=state_dim, action_dim=action_dim)
    target_network=copy.deepcopy(network)
    criterion=nn.MSELoss()
    return network, target_network, criterion

def basic_blackjack_network(state_dim, action_dim):
    return nn.Sequential(
        nn.Linear(state_dim, 8),
        nn.GELU(),
        nn.Linear(8, 32),
        nn.GELU(),
        nn.Dropout(.1),
        nn.Linear(32, 16),
        nn.GELU(),
        nn.Dropout(.1),
        nn.Linear(16, action_dim)
    )