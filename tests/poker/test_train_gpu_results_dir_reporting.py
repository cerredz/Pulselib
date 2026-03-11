from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from environments.Poker.Player import PokerQNetwork
from environments.Poker.utils import PokerAgentType
from scripts.Poker import trainGPU as train_gpu


class DummyPokerQNetwork(PokerQNetwork):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.network = nn.Linear(1, 1)
        self.train_calls: list[dict[str, torch.Tensor]] = []

    def train_step(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.train_calls.append(
            {
                "states": states.clone(),
                "actions": actions.clone(),
                "rewards": rewards.clone(),
                "next_states": next_states.clone(),
                "dones": dones.clone(),
            }
        )


class FailOnUseEnv:
    def reset(self, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise AssertionError("episodes=0 should not call env.reset()")

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        raise AssertionError("episodes=0 should not call env.step()")


class SingleStepEnv:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, options: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.reset_calls += 1
        state = torch.zeros((1, 1), dtype=torch.float32)
        info = {
            "stacks": torch.tensor([[100.0]], dtype=torch.float32),
            "seat_idx": torch.tensor([0], dtype=torch.long),
        }
        return state, info

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self.step_calls += 1
        assert actions.tolist() == [4]
        next_state = torch.ones((1, 1), dtype=torch.float32)
        rewards = torch.tensor([2.5], dtype=torch.float32)
        dones = torch.tensor([True], dtype=torch.bool)
        truncated = torch.tensor([False], dtype=torch.bool)
        info = {
            "stacks": torch.tensor([[112.0]], dtype=torch.float32),
            "seat_idx": torch.tensor([0], dtype=torch.long),
        }
        return next_state, rewards, dones, truncated, info


def _install_reporting_spies(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plot_calls: list[dict[str, Any]] = []
    benchmark_calls: list[dict[str, Any]] = []

    def fake_plot_learning_curve(
        *,
        scores: list[float],
        file_path: str,
        window_size: int = 100,
        title: str = "Agent Learning Curve",
        extend_plot: bool = False,
    ) -> None:
        plot_calls.append(
            {
                "scores": list(scores),
                "file_path": file_path,
                "window_size": window_size,
                "title": title,
                "extend_plot": extend_plot,
            }
        )

    def fake_create_benchmark_file(
        *,
        env_name: str,
        episodes_return: list[float],
        start_time: float,
        end_time: float,
        total_steps: int,
        config: dict[str, Any],
    ) -> None:
        benchmark_calls.append(
            {
                "env_name": env_name,
                "episodes_return": list(episodes_return),
                "start_time": start_time,
                "end_time": end_time,
                "total_steps": total_steps,
                "config": config,
            }
        )

    monkeypatch.setattr(train_gpu, "plot_learning_curve", fake_plot_learning_curve)
    monkeypatch.setattr(train_gpu, "create_benchmark_file", fake_create_benchmark_file)
    return plot_calls, benchmark_calls


def _run_zero_episode_train_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    plot_calls, benchmark_calls = _install_reporting_spies(monkeypatch)
    q_net = DummyPokerQNetwork()

    train_gpu.train_agent(
        env=FailOnUseEnv(),
        agents=[q_net],
        agent_types=[PokerAgentType.QLEARNING],
        episodes=0,
        n_games=1,
        device=torch.device("cpu"),
        results_dir=results_dir,
        config={"ticket": 20},
    )
    return results_dir, plot_calls, benchmark_calls


def test_train_agent_zero_episodes_skips_env_interaction_and_completes_reporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, plot_calls, benchmark_calls = _run_zero_episode_train_agent(tmp_path, monkeypatch)

    assert len(plot_calls) == 2
    assert len(benchmark_calls) == 1


def test_train_agent_uses_passed_results_dir_for_reward_and_chip_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong_dir = tmp_path / "wrong-global-dir"
    monkeypatch.setattr(train_gpu, "result_dir", wrong_dir, raising=False)

    results_dir, plot_calls, _ = _run_zero_episode_train_agent(tmp_path, monkeypatch)

    assert Path(plot_calls[0]["file_path"]) == results_dir / train_gpu.REWARDS_FILENAME
    assert Path(plot_calls[1]["file_path"]) == results_dir / train_gpu.CHIPS_FILENAME


def test_train_agent_saves_q_network_weights_in_results_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir, _, _ = _run_zero_episode_train_agent(tmp_path, monkeypatch)

    weights_path = results_dir / "poker_qnet_final.pth"
    saved_state = torch.load(weights_path, map_location="cpu")

    assert weights_path.is_file()
    assert set(saved_state) == {"weight", "bias"}


def test_train_agent_zero_episode_benchmark_reports_empty_scores_and_zero_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, benchmark_calls = _run_zero_episode_train_agent(tmp_path, monkeypatch)

    benchmark_call = benchmark_calls[0]

    assert benchmark_call["env_name"] == train_gpu.ENV_NAME
    assert benchmark_call["episodes_return"] == []
    assert benchmark_call["total_steps"] == 0
    assert benchmark_call["config"] == {"ticket": 20}
    assert benchmark_call["end_time"] >= benchmark_call["start_time"]


def test_train_agent_single_episode_reports_reward_and_chip_profit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    plot_calls, benchmark_calls = _install_reporting_spies(monkeypatch)
    q_net = DummyPokerQNetwork()

    monkeypatch.setattr(
        train_gpu,
        "get_rotated_agents",
        lambda agents, agent_types, episode_idx: (agents, agent_types, 0, 0),
    )
    monkeypatch.setattr(
        train_gpu,
        "build_actions",
        lambda state, actions, seat_idx, rotated_agents, rotated_types, device: actions.fill_(4),
    )

    env = SingleStepEnv()
    train_gpu.train_agent(
        env=env,
        agents=[q_net],
        agent_types=[PokerAgentType.QLEARNING],
        episodes=1,
        n_games=1,
        device=torch.device("cpu"),
        results_dir=results_dir,
        config={"ticket": 20},
    )

    assert env.reset_calls == 1
    assert env.step_calls == 1
    assert len(q_net.train_calls) == 1
    assert plot_calls[0]["scores"] == [pytest.approx(2.5)]
    assert plot_calls[1]["scores"] == [pytest.approx(12.0)]
    assert benchmark_calls[0]["episodes_return"] == [pytest.approx(2.5)]
