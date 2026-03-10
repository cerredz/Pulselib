import time

import gymnasium as gym
import torch

from environments.Poker.Player import PokerQNetwork
from environments.Poker.utils import PokerAgentType, build_actions
from scripts.Poker import trainGPU as trainer


def _load_training_stack(n_games: int):
    config = trainer.get_config_file(file_name=trainer.CONFIG_FILENAME)
    device = trainer.load_device()
    agents, agent_types = trainer.load_gpu_agents(
        device,
        config["NUM_PLAYERS"],
        config["AGENTS"],
        config["STARTING_BBS"],
        trainer.POKER_ACTION_SPACE_N,
    )
    q_net = PokerQNetwork(
        weights_path="missing.pth",
        device=device,
        gamma=config["GAMMA"],
        update_freq=config["UPDATE_FREQ"],
        state_dim=config["STATE_SPACE"],
        action_dim=config["ACTION_SPACE"],
        learning_rate=config["LEARNING_RATE"],
        weight_decay=config["WEIGHT_DECAY"],
    ).to(device)
    agents.insert(0, q_net)
    agent_types.insert(0, PokerAgentType.QLEARNING)
    env = gym.make(
        config["ENV_ID"],
        device=device,
        agents=agents,
        n_players=config["NUM_PLAYERS"] + 1,
        n_games=n_games,
        starting_bbs=config["STARTING_BBS"],
        w1=config["W1"],
        w2=config["W2"],
        K=config["K"],
        alpha=config["ALPHA"],
    )
    return device, agents, agent_types, env


def run_build_actions_benchmark(n_games: int = 100_000, iters: int = 20) -> float:
    device, agents, agent_types, env = _load_training_stack(n_games=n_games)
    state, info = env.reset(options={"rotation": 0, "active_players": True, "q_agent_seat": 0})
    actions = torch.zeros(n_games, dtype=torch.long, device=device)

    for _ in range(5):
        actions.zero_()
        build_actions(state, actions, info["seat_idx"], agents, agent_types, device)
        state, _, _, _, info = env.step(actions)
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = 0.0
    for _ in range(iters):
        start = time.perf_counter()
        actions.zero_()
        build_actions(state, actions, info["seat_idx"], agents, agent_types, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        state, _, _, _, info = env.step(actions)

    ms = elapsed * 1000 / iters
    print(f"heuristic_build_actions_ms={ms:.3f}")
    return ms


def test_training_stack_builds_valid_heuristic_actions():
    device, agents, agent_types, env = _load_training_stack(n_games=2048)
    state, info = env.reset(options={"rotation": 0, "active_players": True, "q_agent_seat": 0})
    actions = torch.zeros(2048, dtype=torch.long, device=device)
    build_actions(state, actions, info["seat_idx"], agents, agent_types, device)
    assert actions.dtype == torch.long
    assert actions.min().item() >= 0
    assert actions.max().item() < trainer.POKER_ACTION_SPACE_N


def test_heuristic_action_benchmark_smoke():
    assert run_build_actions_benchmark() > 0
