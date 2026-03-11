from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import torch

import environments  # noqa: F401
from environments.Poker.Player import PokerQNetwork
from environments.Poker.utils import build_actions, load_gpu_agents
from scripts.Poker import trainGPU as train_gpu
from utils.config import get_config_file
from utils.torch import load_device


CONFIG_FILENAME = "pokerGPU.yaml"
POKER_ACTION_SPACE_N = 13


class NullPlotter:
    def plot_learning_curve(self, *, scores, file_path, window_size, title):
        del scores, file_path, window_size, title
        return None


class NullBenchmarker:
    def create_benchmark_file(self, **kwargs):
        del kwargs
        return None


@dataclass
class BenchmarkContext:
    config: dict
    benchmark_config: dict
    device: torch.device
    root_dir: Path
    results_dir: Path
    weights_path: Path


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return load_device()
    return torch.device(device_name)


def load_benchmark_context(preset_config: dict, root_dir: Path) -> BenchmarkContext:
    config = get_config_file(CONFIG_FILENAME)
    if config is None:
        raise FileNotFoundError(f"Could not load config/{CONFIG_FILENAME}")

    benchmark_config = dict(config)
    benchmark_config["N_GAMES"] = int(preset_config["env"]["n_games"])
    benchmark_config["EPISODES"] = int(preset_config["env"]["episodes"])
    benchmark_config["RESULTS_DIR"] = "PokerGPU"

    device = resolve_device(preset_config.get("device", "auto"))
    if device.type != "cuda":
        raise RuntimeError(
            "Poker GPU benchmarks require a CUDA device. "
            "This suite targets the live GPU poker environment and trainer only."
        )
    results_dir = root_dir / "results" / "benchmarks" / "Poker"
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_path = results_dir / "benchmark_qnet_weights.pth"
    return BenchmarkContext(
        config=config,
        benchmark_config=benchmark_config,
        device=device,
        root_dir=root_dir,
        results_dir=results_dir,
        weights_path=weights_path,
    )


def create_agents_and_types(context: BenchmarkContext):
    agent_names = context.benchmark_config["AGENTS"]
    agents, agent_types = load_gpu_agents(
        context.device,
        context.benchmark_config["NUM_PLAYERS"],
        agent_names,
        context.benchmark_config["STARTING_BBS"],
        POKER_ACTION_SPACE_N,
    )
    q_net = PokerQNetwork(
        weights_path=context.weights_path,
        device=context.device,
        gamma=context.benchmark_config["GAMMA"],
        update_freq=context.benchmark_config["UPDATE_FREQ"],
        state_dim=context.benchmark_config["STATE_SPACE"],
        action_dim=context.benchmark_config["ACTION_SPACE"],
        learning_rate=context.benchmark_config["LEARNING_RATE"],
        weight_decay=context.benchmark_config["WEIGHT_DECAY"],
    ).to(context.device)
    agents.insert(0, q_net)
    agent_types.insert(0, train_gpu.PokerAgentType.QLEARNING)
    return agents, agent_types, q_net


def create_env(context: BenchmarkContext, agents) -> gym.Env:
    cfg = context.benchmark_config
    return gym.make(
        cfg["ENV_ID"],
        device=context.device,
        agents=agents,
        n_players=cfg["NUM_PLAYERS"] + 1,
        n_games=cfg["N_GAMES"],
        starting_bbs=cfg["STARTING_BBS"],
        w1=cfg["W1"],
        w2=cfg["W2"],
        K=cfg["K"],
        alpha=cfg["ALPHA"],
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(fn, *, device: torch.device):
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    elapsed = time.perf_counter() - start
    return elapsed, result


def build_default_actions(state, info, agents, agent_types, device):
    actions = torch.zeros(state.shape[0], dtype=torch.long, device=device)
    build_actions(state, actions, info["seat_idx"], agents, agent_types, device)
    return actions


def create_train_run_dependencies(context: BenchmarkContext):
    agents, agent_types, _ = create_agents_and_types(context)
    env = create_env(context, agents)
    return {
        "env": env,
        "agents": agents,
        "agent_types": agent_types,
        "plotter": NullPlotter(),
        "benchmarker": NullBenchmarker(),
    }
