from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from scripts.Poker import trainGPU_performance as performance_script
from utils.performance import (
    calculate_final_performance_metrics,
    calculate_rolling_window_averages,
    summarize_episode_performance_metrics,
)


class DummyPerformanceQNetwork(nn.Module):
    def __init__(
        self,
        *,
        weights_path: str = "",
        device: torch.device | None = None,
        **_: object,
    ) -> None:
        del weights_path
        super().__init__()
        self.device = device or torch.device("cpu")
        self.train_calls: list[dict[str, torch.Tensor]] = []

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return states

    def train_step(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        self.train_calls.append(
            {
                "states": states.clone(),
                "actions": actions.clone(),
                "rewards": rewards.clone(),
                "next_states": next_states.clone(),
                "dones": dones.clone(),
            }
        )
        return torch.zeros((), device=states.device if states.numel() else self.device)


class TwoEpisodePerformanceEnv:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0
        self.active_players = 2
        self.button = torch.tensor([0], dtype=torch.int32)
        self.stages = torch.tensor([0], dtype=torch.int32)

    def reset(self, options: dict[str, object]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del options
        self.reset_calls += 1
        self.button = torch.tensor([0], dtype=torch.int32)
        self.stages = torch.tensor([0], dtype=torch.int32)
        state = torch.zeros((1, 13), dtype=torch.float32)
        state[:, 12] = 0.0
        info = {
            "seat_idx": torch.tensor([0], dtype=torch.long),
            "stacks": torch.tensor([[5]], dtype=torch.int32),
        }
        return state, info

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self.step_calls += 1
        assert actions.tolist() == [2]
        next_state = torch.zeros((1, 13), dtype=torch.float32)
        next_state[:, 12] = 0.0
        dones = torch.tensor([True], dtype=torch.bool)
        truncated = torch.tensor([False], dtype=torch.bool)
        if self.step_calls == 1:
            self.stages = torch.tensor([0], dtype=torch.int32)
            rewards = torch.tensor([1.5], dtype=torch.float32)
            stacks = torch.tensor([[7]], dtype=torch.int32)
        else:
            self.stages = torch.tensor([5], dtype=torch.int32)
            rewards = torch.tensor([0.5], dtype=torch.float32)
            stacks = torch.tensor([[4]], dtype=torch.int32)
        info = {
            "seat_idx": torch.tensor([0], dtype=torch.long),
            "stacks": stacks,
        }
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


def test_calculate_q_seat_positions_uses_button_relative_positions() -> None:
    buttons = torch.tensor([0, 1, 2], dtype=torch.int32)
    positions = performance_script.calculate_q_seat_positions(buttons, q_seat=2, active_players=4)

    assert positions.tolist() == [2, 1, 0]


def test_episode_and_final_performance_summaries_aggregate_expected_values() -> None:
    episode_summary = summarize_episode_performance_metrics(
        episode_reward=torch.tensor(12.5),
        cumulative_reward=torch.tensor(20.0),
        hand_bb_deltas=[torch.tensor([2.0, -1.0], dtype=torch.float32)],
    )

    assert episode_summary["reward"].item() == pytest.approx(12.5)
    assert episode_summary["cumulative_reward"].item() == pytest.approx(20.0)
    assert episode_summary["mean_bb_delta"].item() == pytest.approx(0.5)
    assert episode_summary["hand_win_rate"].item() == pytest.approx(0.5)
    assert episode_summary["hands_completed"].item() == 2

    rolling_averages = calculate_rolling_window_averages(
        [torch.tensor([1.0, -1.0, 2.0, 3.0], dtype=torch.float32)],
        window_size=2,
    )
    assert [value.item() for value in rolling_averages] == pytest.approx([0.0, 0.5, 2.5])

    final_metrics = calculate_final_performance_metrics(
        epoch_rewards=[torch.tensor(10.0), torch.tensor(20.0)],
        hand_bb_deltas=[
            torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32),
            torch.tensor([3.0], dtype=torch.float32),
        ],
        hand_terminal_stages=[
            torch.tensor([0, 1, 5], dtype=torch.int64),
            torch.tensor([3], dtype=torch.int64),
        ],
        hand_positions=[
            torch.tensor([0, 1, 0], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int64),
        ],
        elapsed_seconds=7.5,
        rolling_window_size=2,
    )

    assert final_metrics["cumulative_reward"].item() == pytest.approx(30.0)
    assert final_metrics["mean_reward"].item() == pytest.approx(15.0)
    assert final_metrics["reward_improvement"]["slope"].item() == pytest.approx(10.0)
    assert final_metrics["reward_improvement"]["first_to_last_percent_change"].item() == pytest.approx(100.0)
    assert final_metrics["total_bb_won"].item() == pytest.approx(5.0)
    assert final_metrics["overall_hand_win_rate"].item() == pytest.approx(0.75)
    assert final_metrics["total_hands"].item() == 4
    assert [value.item() for value in final_metrics["rolling_bb_window"]["values"]] == pytest.approx([0.0, 0.5, 2.5])
    assert final_metrics["rolling_bb_window"]["last_average"].item() == pytest.approx(2.5)
    assert final_metrics["street_win_percentages"]["preflop"].item() == pytest.approx(0.25)
    assert final_metrics["street_win_percentages"]["flop"].item() == pytest.approx(0.0)
    assert final_metrics["street_win_percentages"]["river"].item() == pytest.approx(0.25)
    assert final_metrics["street_win_percentages"]["showdown"].item() == pytest.approx(0.25)
    assert final_metrics["position_win_rates"]["position_0"].item() == pytest.approx(1.0)
    assert final_metrics["position_win_rates"]["position_1"].item() == pytest.approx(0.5)
    assert final_metrics["position_hand_counts"]["position_0"].item() == 2
    assert final_metrics["position_hand_counts"]["position_1"].item() == 2
    assert final_metrics["total_time_seconds"].item() == pytest.approx(7.5)


def test_run_performance_benchmark_accepts_small_override_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = TwoEpisodePerformanceEnv()
    FakeLogger.instances.clear()

    monkeypatch.setattr(performance_script, "load_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        performance_script,
        "load_gpu_agents",
        lambda device, num_players, agent_strings, starting_bbs, action_space_n: ([], []),
    )
    monkeypatch.setattr(performance_script, "PokerQNetwork", DummyPerformanceQNetwork)
    monkeypatch.setattr(performance_script, "TrainingLogger", FakeLogger)
    monkeypatch.setattr(performance_script.gym, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(
        performance_script,
        "get_rotated_agents",
        lambda agents, agent_types, episode_idx: (agents, agent_types, 0, 0),
    )
    monkeypatch.setattr(
        performance_script,
        "build_actions",
        lambda state, actions, seat_idx, rotated_agents, rotated_types, device: actions.fill_(2),
    )

    final_metrics = performance_script.run_performance_benchmark(
        {
            "NUM_PLAYERS": 0,
            "N_GAMES": 1,
            "EPISODES": 2,
            "STATE_SPACE": 13,
            "ACTION_SPACE": 13,
            "AGENT_STRINGS": [],
            "ROLLING_WINDOW_SIZE": 2,
            "LOG_DIR": str(tmp_path / "logs"),
        }
    )

    logger = FakeLogger.instances[0]

    assert env.reset_calls == 2
    assert env.step_calls == 2
    assert final_metrics["cumulative_reward"].item() == pytest.approx(2.0)
    assert final_metrics["reward_improvement"]["slope"].item() == pytest.approx(-1.0)
    assert final_metrics["reward_improvement"]["first_to_last_percent_change"].item() == pytest.approx(-66.666666, rel=1e-5)
    assert final_metrics["total_bb_won"].item() == pytest.approx(1.0)
    assert final_metrics["rolling_bb_window"]["last_average"].item() == pytest.approx(0.5)
    assert final_metrics["street_win_percentages"]["preflop"].item() == pytest.approx(0.5)
    assert final_metrics["street_win_percentages"]["showdown"].item() == pytest.approx(0.0)
    assert final_metrics["position_win_rates"]["position_0"].item() == pytest.approx(0.5)
    assert final_metrics["position_hand_counts"]["position_0"].item() == 2
    assert logger.entries[0][0] == "Starting performance benchmark run #1"
    assert logger.entries[-1][0] == "Training Performance Benchmark Completed"
