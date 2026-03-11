import time
from pathlib import Path

import pytest
import torch

from environments.Poker.utils import PokerAgentType
from scripts.Poker import trainGPU as trainer


class CountingAgentTypes(list):
    def __init__(self, values):
        super().__init__(values)
        self.index_calls = 0

    def index(self, value, *args):
        self.index_calls += 1
        return super().index(value, *args)


class DummyNetwork:
    def state_dict(self):
        return {"weight": torch.tensor([1.0])}


class FakeQNetwork:
    def __init__(self):
        self.network = DummyNetwork()
        self.train_batches = []

    def get_actions(self, states):
        return torch.zeros(states.shape[0], dtype=torch.long, device=states.device)

    def train_step(self, *, states, actions, rewards, next_states, dones):
        self.train_batches.append(
            {
                "states": states.clone(),
                "actions": actions.clone(),
                "rewards": rewards.clone(),
                "next_states": next_states.clone(),
                "dones": dones.clone(),
            }
        )
        return torch.tensor(0.0)


class FakeOpponent:
    pass


class BenchmarkQAgent:
    def train_step(self, **kwargs):
        del kwargs
        return None


class FakePlotter:
    def __init__(self):
        self.calls = []

    def plot_learning_curve(self, *, scores, file_path, window_size, title):
        self.calls.append(
            {
                "scores": list(scores),
                "file_path": file_path,
                "window_size": window_size,
                "title": title,
            }
        )


class FakeBenchmarker:
    def __init__(self):
        self.calls = []

    def create_benchmark_file(self, **kwargs):
        self.calls.append(kwargs)
        return Path(kwargs["config"]["RESULTS_DIR"]) / "benchmark.yaml"


class ScriptedEnv:
    def __init__(self, episodes):
        self.episodes = episodes
        self.episode_idx = -1
        self.step_idx = 0
        self.step_calls = 0

    def reset(self, options):
        self.episode_idx += 1
        self.step_idx = 0
        episode = self.episodes[self.episode_idx]
        return episode["state"].clone(), _clone_info(episode["info"])

    def step(self, actions):
        self.step_calls += 1
        step = self.episodes[self.episode_idx]["steps"][self.step_idx]
        self.step_idx += 1
        return (
            step["next_state"].clone(),
            step["rewards"].clone(),
            step["dones"].clone(),
            step["truncated"].clone(),
            _clone_info(step["info"]),
        )


def _clone_info(info):
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in info.items()
    }


def _fake_build_actions(state, actions, curr_players, agents, agent_types, device):
    del state, agents, agent_types, device
    actions.copy_(curr_players.long())


def _make_state(rows):
    return torch.tensor([[row] + [0] * 12 for row in rows], dtype=torch.float32)


def _make_episode(n_games, initial_rows, initial_seats, steps, initial_stacks, q_seat=0):
    return {
        "state": _make_state(initial_rows),
        "info": {
            "seat_idx": torch.tensor(initial_seats, dtype=torch.long),
            "stacks": torch.tensor(initial_stacks, dtype=torch.float32),
        },
        "steps": steps,
    }


def _make_step(next_rows, rewards, dones, seat_idx, stacks):
    n_games = len(rewards)
    return {
        "next_state": _make_state(next_rows),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "dones": torch.tensor(dones, dtype=torch.bool),
        "truncated": torch.zeros(n_games, dtype=torch.bool),
        "info": {
            "seat_idx": torch.tensor(seat_idx, dtype=torch.long),
            "stacks": torch.tensor(stacks, dtype=torch.float32),
        },
    }


def _padding_steps(count, n_games, seat_idx, stacks, row_start=100):
    steps = []
    for offset in range(count):
        steps.append(
            _make_step(
                next_rows=list(range(row_start + offset * n_games, row_start + (offset + 1) * n_games)),
                rewards=[0.0] * n_games,
                dones=[True] * n_games,
                seat_idx=seat_idx,
                stacks=stacks,
            )
        )
    return steps


def _run_train_agent(monkeypatch, tmp_path, episodes, agent_types=None):
    monkeypatch.setattr(trainer, "PokerQNetwork", FakeQNetwork)
    monkeypatch.setattr(trainer, "build_actions", _fake_build_actions)
    q_agent = FakeQNetwork()
    agents = [q_agent, FakeOpponent(), FakeOpponent()]
    types = agent_types or CountingAgentTypes(
        [PokerAgentType.QLEARNING, PokerAgentType.HEURISTIC_HANDS, PokerAgentType.RANDOM]
    )
    plotter = FakePlotter()
    benchmarker = FakeBenchmarker()
    env = ScriptedEnv(episodes)
    config = {"RESULTS_DIR": str(tmp_path)}
    trainer.train_agent(
        env=env,
        agents=agents,
        agent_types=types,
        episodes=len(episodes),
        n_games=len(episodes[0]["info"]["seat_idx"]),
        device=torch.device("cpu"),
        results_dir=tmp_path,
        config=config,
        plotter=plotter,
        benchmarker=benchmarker,
    )
    return q_agent, types, plotter, benchmarker, env


def _baseline_inner_loop_step(terminated, q_mask, dones):
    terminated_before = terminated.clone()
    terminated |= dones
    active_games = q_mask & ~terminated_before
    should_stop = terminated.float().mean() > torch.tensor(0.8, device=terminated.device)
    return active_games, should_stop


def run_train_loop_inner_overhead_benchmark(
    n_games: int = 2_048,
    iters: int = 2_000,
    device: torch.device | None = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and n_games == 2_048 and iters == 2_000:
        n_games, iters = 200_000, 500
    q_mask = torch.arange(n_games, device=device) % 2 == 0
    dones = torch.arange(n_games, device=device) % 5 == 0
    states = torch.zeros((n_games, 13), dtype=torch.float32, device=device)
    next_states = torch.ones((n_games, 13), dtype=torch.float32, device=device)
    actions = torch.zeros(n_games, dtype=torch.long, device=device)
    rewards = torch.ones(n_games, dtype=torch.float32, device=device)
    q_agent = BenchmarkQAgent()
    agents = [q_agent, FakeOpponent(), FakeOpponent(), FakeOpponent(), FakeOpponent()]
    agent_types = [
        PokerAgentType.QLEARNING,
        PokerAgentType.HEURISTIC_HANDS,
        PokerAgentType.RANDOM,
        PokerAgentType.LOOSE_PASSIVE,
        PokerAgentType.SMALL_BALL,
    ]
    termination_threshold = torch.tensor(0.8, device=device)

    for _ in range(10):
        baseline_terminated = torch.zeros(n_games, dtype=torch.bool, device=device)
        baseline_active_games, _ = _baseline_inner_loop_step(baseline_terminated, q_mask, dones)
        agents[agent_types.index(PokerAgentType.QLEARNING)].train_step(
            states=states[baseline_active_games],
            actions=actions[baseline_active_games],
            rewards=rewards[baseline_active_games],
            next_states=next_states[baseline_active_games],
            dones=dones[baseline_active_games],
        )
        optimized_terminated = torch.zeros(n_games, dtype=torch.bool, device=device)
        optimized_mask = q_mask & ~optimized_terminated
        optimized_terminated |= dones
        q_agent.train_step(
            states=states[optimized_mask],
            actions=actions[optimized_mask],
            rewards=rewards[optimized_mask],
            next_states=next_states[optimized_mask],
            dones=dones[optimized_mask],
        )
        _ = bool(optimized_terminated.float().mean() > termination_threshold)
    if device.type == "cuda":
        torch.cuda.synchronize()

    baseline_elapsed = 0.0
    for _ in range(iters):
        baseline_terminated = torch.zeros(n_games, dtype=torch.bool, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        baseline_active_games, _ = _baseline_inner_loop_step(baseline_terminated, q_mask, dones)
        agents[agent_types.index(PokerAgentType.QLEARNING)].train_step(
            states=states[baseline_active_games],
            actions=actions[baseline_active_games],
            rewards=rewards[baseline_active_games],
            next_states=next_states[baseline_active_games],
            dones=dones[baseline_active_games],
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        baseline_elapsed += time.perf_counter() - start

    optimized_elapsed = 0.0
    for _ in range(iters):
        optimized_terminated = torch.zeros(n_games, dtype=torch.bool, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        optimized_mask = q_mask & ~optimized_terminated
        optimized_terminated |= dones
        q_agent.train_step(
            states=states[optimized_mask],
            actions=actions[optimized_mask],
            rewards=rewards[optimized_mask],
            next_states=next_states[optimized_mask],
            dones=dones[optimized_mask],
        )
        _ = bool(optimized_terminated.float().mean() > termination_threshold)
        if device.type == "cuda":
            torch.cuda.synchronize()
        optimized_elapsed += time.perf_counter() - start

    baseline_ms = baseline_elapsed * 1000 / iters
    optimized_ms = optimized_elapsed * 1000 / iters
    print(f"train_loop_baseline_ms={baseline_ms:.6f}")
    print(f"train_loop_optimized_ms={optimized_ms:.6f}")
    return baseline_ms, optimized_ms


def test_active_q_mask_excludes_games_terminated_before_the_step():
    terminated = torch.tensor([False, True, False, True])
    q_mask = torch.tensor([True, True, False, True])

    active_games = trainer._get_active_q_mask(terminated, q_mask)

    assert active_games.tolist() == [True, False, False, False]


def test_should_stop_loop_matches_original_five_step_cadence():
    threshold = torch.tensor(0.8)

    assert not trainer._should_stop_loop(4, torch.tensor([True, True, True, True, True]), threshold)
    assert not trainer._should_stop_loop(5, torch.tensor([True, True, True, True, False]), threshold)
    assert trainer._should_stop_loop(5, torch.tensor([True, True, True, True, True]), threshold)


def test_train_agent_caches_q_agent_lookup_and_skips_empty_q_updates(monkeypatch, tmp_path):
    episode = _make_episode(
        n_games=4,
        initial_rows=[0, 1, 2, 3],
        initial_seats=[1, 1, 2, 2],
        initial_stacks=[[10.0, 20.0, 30.0]] * 4,
        steps=[
            _make_step(
                next_rows=[10, 11, 12, 13],
                rewards=[0.0, 0.0, 0.0, 0.0],
                dones=[True, False, False, False],
                seat_idx=[1, 1, 2, 2],
                stacks=[[10.0, 20.0, 30.0]] * 4,
            ),
            _make_step(
                next_rows=[20, 21, 22, 23],
                rewards=[0.0, 0.0, 0.0, 0.0],
                dones=[False, True, False, False],
                seat_idx=[1, 1, 2, 2],
                stacks=[[10.0, 20.0, 30.0]] * 4,
            ),
            _make_step(
                next_rows=[30, 31, 32, 33],
                rewards=[0.0, 0.0, 0.0, 0.0],
                dones=[False, False, True, False],
                seat_idx=[1, 1, 2, 2],
                stacks=[[10.0, 20.0, 30.0]] * 4,
            ),
            _make_step(
                next_rows=[40, 41, 42, 43],
                rewards=[0.0, 0.0, 0.0, 0.0],
                dones=[False, False, False, True],
                seat_idx=[1, 1, 2, 2],
                stacks=[[10.0, 20.0, 30.0]] * 4,
            ),
            _make_step(
                next_rows=[50, 51, 52, 53],
                rewards=[0.0, 0.0, 0.0, 0.0],
                dones=[True, True, True, True],
                seat_idx=[1, 1, 2, 2],
                stacks=[[10.0, 20.0, 30.0]] * 4,
            ),
            _make_step(
                next_rows=[60, 61, 62, 63],
                rewards=[0.0, 0.0, 0.0, 0.0],
                dones=[True, True, True, True],
                seat_idx=[1, 1, 2, 2],
                stacks=[[10.0, 20.0, 30.0]] * 4,
            ),
        ],
    )

    q_agent, agent_types, _, _, env = _run_train_agent(monkeypatch, tmp_path, [episode])

    assert env.step_calls == 6
    assert agent_types.index_calls == 1
    assert q_agent.train_batches == []


def test_train_agent_uses_active_mask_before_updating_terminated(monkeypatch, tmp_path):
    steps = [
        _make_step(
            next_rows=[10, 11, 12, 13],
            rewards=[1.0, 2.0, 3.0, 4.0],
            dones=[False, False, True, False],
            seat_idx=[0, 0, 0, 0],
            stacks=[[20.0, 30.0, 40.0]] * 4,
        ),
        _make_step(
            next_rows=[20, 21, 22, 23],
            rewards=[5.0, 6.0, 7.0, 8.0],
            dones=[False, True, False, True],
            seat_idx=[1, 1, 1, 1],
            stacks=[[20.0, 30.0, 40.0]] * 4,
        ),
        *_padding_steps(
            count=4,
            n_games=4,
            seat_idx=[1, 1, 1, 1],
            stacks=[[20.0, 30.0, 40.0]] * 4,
            row_start=30,
        ),
    ]
    episode = _make_episode(
        n_games=4,
        initial_rows=[0, 1, 2, 3],
        initial_seats=[0, 1, 0, 1],
        initial_stacks=[[10.0, 20.0, 30.0]] * 4,
        steps=steps,
    )

    q_agent, _, _, _, _ = _run_train_agent(monkeypatch, tmp_path, [episode])

    assert len(q_agent.train_batches) == 2
    assert q_agent.train_batches[0]["states"][:, 0].tolist() == [0.0, 2.0]
    assert q_agent.train_batches[0]["dones"].tolist() == [False, True]
    assert q_agent.train_batches[1]["states"][:, 0].tolist() == [10.0, 11.0, 13.0]
    assert q_agent.train_batches[1]["dones"].tolist() == [False, True, True]


def test_train_agent_reports_reward_and_profit_without_changing_metrics(monkeypatch, tmp_path):
    steps = [
        _make_step(
            next_rows=[10, 11, 12, 13, 14],
            rewards=[1.0, 100.0, 2.0, 100.0, 3.0],
            dones=[False, False, False, False, True],
            seat_idx=[0, 1, 0, 1, 0],
            stacks=[[10.0, 20.0, 30.0]] * 5,
        ),
        _make_step(
            next_rows=[20, 21, 22, 23, 24],
            rewards=[4.0, 100.0, 5.0, 100.0, 6.0],
            dones=[True, True, True, True, True],
            seat_idx=[0, 0, 0, 0, 0],
            stacks=[
                [17.0, 20.0, 30.0],
                [9.0, 20.0, 30.0],
                [14.0, 20.0, 30.0],
                [11.0, 20.0, 30.0],
                [10.0, 20.0, 30.0],
            ],
        ),
        *_padding_steps(
            count=4,
            n_games=5,
            seat_idx=[0, 0, 0, 0, 0],
            stacks=[
                [17.0, 20.0, 30.0],
                [9.0, 20.0, 30.0],
                [14.0, 20.0, 30.0],
                [11.0, 20.0, 30.0],
                [10.0, 20.0, 30.0],
            ],
            row_start=30,
        ),
    ]
    episode = _make_episode(
        n_games=5,
        initial_rows=[0, 1, 2, 3, 4],
        initial_seats=[0, 1, 0, 1, 0],
        initial_stacks=[[10.0, 20.0, 30.0]] * 5,
        steps=steps,
    )

    _, _, plotter, benchmarker, _ = _run_train_agent(monkeypatch, tmp_path, [episode])

    assert plotter.calls[0]["scores"] == [15.0]
    assert plotter.calls[1]["scores"] == [11.0]
    assert benchmarker.calls[0]["episodes_return"] == [15.0]


def test_train_loop_inner_overhead_benchmark_smoke():
    baseline_ms, optimized_ms = run_train_loop_inner_overhead_benchmark()

    assert baseline_ms > 0
    assert optimized_ms > 0
