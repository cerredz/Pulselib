from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from scripts.Poker import trainGPU_stability as stability_script
from utils.logging.logger import TrainingLogger
from utils.stability import (
    calculate_final_stability_metrics,
    run_stability_measured_q_learning_step,
    summarize_episode_stability_metrics,
)


class DummyStabilityQNetwork(nn.Module):
    def __init__(
        self,
        *,
        weights_path: str = "",
        device: torch.device | None = None,
        gamma: float = 0.5,
        update_freq: int = 1,
        state_dim: int = 13,
        action_dim: int = 13,
        learning_rate: float = 0.1,
        weight_decay: float = 0.0,
        **_: object,
    ) -> None:
        del weights_path, weight_decay
        super().__init__()
        self.device = device or torch.device("cpu")
        self.gamma = gamma
        self.update_freq = update_freq
        self.step_count = 0
        self.network = nn.Linear(state_dim, action_dim, bias=False)
        self.target_network = nn.Linear(state_dim, action_dim, bias=False)
        nn.init.zeros_(self.network.weight)
        nn.init.zeros_(self.target_network.weight)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


class SingleStepBenchmarkEnv:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, options: dict[str, object]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del options
        self.reset_calls += 1
        state = torch.zeros((1, 13), dtype=torch.float32)
        state[:, 0] = 1.0
        state[:, 12] = 0.0
        info = {"seat_idx": torch.tensor([0], dtype=torch.long)}
        return state, info

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self.step_calls += 1
        assert actions.tolist() == [2]
        next_state = torch.zeros((1, 13), dtype=torch.float32)
        next_state[:, 1] = 1.0
        next_state[:, 12] = 0.0
        rewards = torch.tensor([1.5], dtype=torch.float32)
        dones = torch.tensor([True], dtype=torch.bool)
        truncated = torch.tensor([False], dtype=torch.bool)
        info = {"seat_idx": torch.tensor([0], dtype=torch.long)}
        return next_state, rewards, dones, truncated, info


class FakeLogger:
    instances: list["FakeLogger"] = []

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.run_number = 1
        self.entries: list[tuple[str, dict[str, object] | None]] = []
        self.__class__.instances.append(self)

    def log(self, message: str, metrics: dict[str, object] | None = None) -> None:
        self.entries.append((message, metrics))


def test_run_stability_measured_q_learning_step_returns_metrics_and_updates_target() -> None:
    q_network = DummyStabilityQNetwork()
    states = torch.zeros((2, 13), dtype=torch.float32)
    states[0, 0] = 1.0
    states[0, 12] = 0.0
    states[1, 12] = 1.0
    actions = torch.tensor([2, 3], dtype=torch.long)
    rewards = torch.tensor([1.0, 5.0], dtype=torch.float32)
    next_states = torch.zeros((2, 13), dtype=torch.float32)
    dones = torch.tensor([True, False], dtype=torch.bool)

    metrics = run_stability_measured_q_learning_step(
        q_network=q_network,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        clip_threshold=1e-6,
    )

    assert metrics is not None
    assert metrics["loss"].device == states.device
    assert metrics["td_error"].device == states.device
    assert metrics["loss"].item() == pytest.approx(1.0)
    assert metrics["td_error"].item() == pytest.approx(1.0)
    assert metrics["q_mean"].item() == pytest.approx(0.0)
    assert metrics["q_min"].item() == pytest.approx(0.0)
    assert metrics["q_max"].item() == pytest.approx(0.0)
    assert metrics["clip_rate"].item() == pytest.approx(1.0)
    assert q_network.step_count == 1
    assert torch.equal(q_network.network.weight, q_network.target_network.weight)


def test_run_stability_measured_q_learning_step_returns_none_for_empty_valid_batch() -> None:
    q_network = DummyStabilityQNetwork()
    states = torch.ones((1, 13), dtype=torch.float32)
    states[:, 12] = 1.0

    metrics = run_stability_measured_q_learning_step(
        q_network=q_network,
        states=states,
        actions=torch.tensor([1], dtype=torch.long),
        rewards=torch.tensor([0.5], dtype=torch.float32),
        next_states=states.clone(),
        dones=torch.tensor([False], dtype=torch.bool),
    )

    assert metrics is None
    assert q_network.step_count == 0


def test_episode_and_final_stability_summaries_aggregate_expected_values() -> None:
    episode_summary = summarize_episode_stability_metrics(
        episode_reward=torch.tensor(12.5),
        step_metrics=[
            {"q_mean": torch.tensor(1.0), "q_min": torch.tensor(-2.0), "q_max": torch.tensor(3.0), "td_error": torch.tensor(0.5), "clip_rate": torch.tensor(0.0)},
            {"q_mean": torch.tensor(3.0), "q_min": torch.tensor(-1.0), "q_max": torch.tensor(5.0), "td_error": torch.tensor(1.5), "clip_rate": torch.tensor(1.0)},
        ],
    )

    assert episode_summary["reward"].item() == pytest.approx(12.5)
    assert episode_summary["q_mean"].item() == pytest.approx(2.0)
    assert episode_summary["q_min"].item() == pytest.approx(-2.0)
    assert episode_summary["q_max"].item() == pytest.approx(5.0)
    assert episode_summary["td_error"].item() == pytest.approx(1.0)
    assert episode_summary["clip_rate"].item() == pytest.approx(0.5)

    final_metrics = calculate_final_stability_metrics(
        epoch_rewards=[torch.tensor(10.0), torch.tensor(14.0)],
        epoch_q_means=[torch.tensor(1.0), torch.tensor(3.0)],
        epoch_q_mins=[torch.tensor(-2.0), torch.tensor(-1.0)],
        epoch_q_maxs=[torch.tensor(3.0), torch.tensor(5.0)],
        epoch_td_errors=[torch.tensor(2.0), torch.tensor(1.0)],
        epoch_clip_rates=[torch.tensor(0.0), torch.tensor(1.0)],
        elapsed_seconds=7.5,
    )

    assert final_metrics["reward_std"].item() == pytest.approx(2.0)
    assert final_metrics["mean_reward"].item() == pytest.approx(12.0)
    assert final_metrics["q_bounds"]["global_min"].item() == pytest.approx(-2.0)
    assert final_metrics["q_bounds"]["global_max"].item() == pytest.approx(5.0)
    assert final_metrics["q_bounds"]["mean_q"].item() == pytest.approx(2.0)
    assert final_metrics["td_error_trend"].item() == pytest.approx(-1.0)
    assert final_metrics["average_clip_rate"].item() == pytest.approx(0.5)
    assert final_metrics["total_time_seconds"].item() == pytest.approx(7.5)


def test_training_logger_serializes_nested_tensor_metrics(tmp_path: Path) -> None:
    logger = TrainingLogger(str(tmp_path), run_number=1)
    logger.log(
        "nested metrics",
        {
            "reward_std": torch.tensor(1.25),
            "q_bounds": {
                "global_min": torch.tensor(-2.0),
                "global_max": torch.tensor(3.5),
            },
        },
    )

    contents = (tmp_path / "logs_1.txt").read_text()

    assert "\"reward_std\": 1.25" in contents
    assert "\"q_bounds\": {\"global_min\": -2.0, \"global_max\": 3.5}" in contents


def test_run_stability_benchmark_accepts_small_override_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = SingleStepBenchmarkEnv()
    FakeLogger.instances.clear()

    monkeypatch.setattr(stability_script, "load_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        stability_script,
        "load_gpu_agents",
        lambda device, num_players, agent_strings, starting_bbs, action_space_n: ([], []),
    )
    monkeypatch.setattr(stability_script, "PokerQNetwork", DummyStabilityQNetwork)
    monkeypatch.setattr(stability_script, "TrainingLogger", FakeLogger)
    monkeypatch.setattr(stability_script.gym, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(
        stability_script,
        "get_rotated_agents",
        lambda agents, agent_types, episode_idx: (agents, agent_types, 0, 0),
    )
    monkeypatch.setattr(
        stability_script,
        "build_actions",
        lambda state, actions, seat_idx, rotated_agents, rotated_types, device: actions.fill_(2),
    )

    final_metrics = stability_script.run_stability_benchmark(
        {
            "NUM_PLAYERS": 0,
            "N_GAMES": 1,
            "EPISODES": 2,
            "STATE_SPACE": 13,
            "ACTION_SPACE": 13,
            "AGENT_STRINGS": [],
            "LOG_DIR": str(tmp_path / "logs"),
        }
    )

    logger = FakeLogger.instances[0]

    assert env.reset_calls == 2
    assert env.step_calls == 2
    assert final_metrics["reward_std"].item() == pytest.approx(0.0)
    assert final_metrics["mean_reward"].item() == pytest.approx(1.5)
    assert final_metrics["q_bounds"]["global_min"].item() == pytest.approx(0.0)
    assert final_metrics["q_bounds"]["global_max"].item() >= 0.0
    assert final_metrics["q_bounds"]["mean_q"].item() >= 0.0
    assert 0.0 <= final_metrics["average_clip_rate"].item() <= 1.0
    assert logger.entries[0][0] == "Starting stability benchmark run #1"
    assert logger.entries[-1][0] == "Training Stability Benchmark Completed"
