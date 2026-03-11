from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch

from benchmarking.Poker import runtime
from scripts.Poker import trainGPU as train_gpu


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    category: str
    description: str
    primary_metric_name: str
    primary_metric_unit: str
    lower_is_better: bool
    runner: callable


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _result(case: BenchmarkCase, timings: list[float], *, metadata: dict, derived_metrics: list[dict] | None = None):
    summary = _stats(timings)
    return {
        "name": case.name,
        "category": case.category,
        "description": case.description,
        "primary_metric": {
            "name": case.primary_metric_name,
            "unit": case.primary_metric_unit,
            "value": summary["mean"],
            "lower_is_better": case.lower_is_better,
        },
        "timings": {
            "unit": case.primary_metric_unit,
            "trials": timings,
            **summary,
        },
        "derived_metrics": derived_metrics or [],
        "metadata": metadata,
    }


def _per_second(name: str, count: float, elapsed_seconds: float, *, unit: str) -> dict:
    value = count / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "higher_is_better": True,
    }


def run_env_reset(case: BenchmarkCase, context, warmup_iterations: int, measure_iterations: int):
    agents, _, _ = runtime.create_agents_and_types(context)
    env = runtime.create_env(context, agents)

    def call():
        env.reset(options={"rotation": 0, "active_players": True, "q_agent_seat": 0})

    for _ in range(warmup_iterations):
        runtime.timed_call(call, device=context.device)

    timings = []
    for _ in range(measure_iterations):
        elapsed, _ = runtime.timed_call(call, device=context.device)
        timings.append(elapsed)

    env.close()
    n_games = context.benchmark_config["N_GAMES"]
    derived = [_per_second("games_reset_per_second", n_games, _stats(timings)["mean"], unit="games_per_second")]
    return _result(case, timings, metadata={"n_games": n_games}, derived_metrics=derived)


def run_env_calculate_equities(case: BenchmarkCase, context, warmup_iterations: int, measure_iterations: int):
    agents, _, _ = runtime.create_agents_and_types(context)
    env = runtime.create_env(context, agents)
    env.reset(options={"rotation": 0, "active_players": True, "q_agent_seat": 0})
    poker_env = env.unwrapped

    def prepare_stage(stage: int):
        env.reset(options={"rotation": 0, "active_players": True, "q_agent_seat": 0})
        poker_env.stages.fill_(stage)
        if stage >= 1:
            poker_env.board[:, 0:3] = poker_env.deal_cards(poker_env.g, 3)
        if stage >= 2:
            poker_env.board[:, 3] = poker_env.deal_cards(poker_env.g, 1).squeeze(1)
        if stage >= 3:
            poker_env.board[:, 4] = poker_env.deal_cards(poker_env.g, 1).squeeze(1)
        poker_env.equities.fill_(0.5)

    def call():
        prepare_stage(3)
        poker_env.calculate_equities()

    for _ in range(warmup_iterations):
        runtime.timed_call(call, device=context.device)

    timings = []
    for _ in range(measure_iterations):
        elapsed, _ = runtime.timed_call(call, device=context.device)
        timings.append(elapsed)

    env.close()
    n_games = context.benchmark_config["N_GAMES"]
    derived = [_per_second("equity_batches_per_second", n_games, _stats(timings)["mean"], unit="games_per_second")]
    return _result(case, timings, metadata={"n_games": n_games, "street": "river"}, derived_metrics=derived)


def run_env_execute_actions(case: BenchmarkCase, context, warmup_iterations: int, measure_iterations: int):
    agents, _, _ = runtime.create_agents_and_types(context)
    env = runtime.create_env(context, agents)
    env.reset(options={"rotation": 0, "active_players": True, "q_agent_seat": 0})
    poker_env = env.unwrapped

    # Use a deterministic non-destructive action profile to limit state degeneration.
    fixed_actions = torch.ones(context.benchmark_config["N_GAMES"], dtype=torch.long, device=context.device)

    def fixed_call():
        poker_env.execute_actions(fixed_actions)

    for _ in range(warmup_iterations):
        runtime.timed_call(fixed_call, device=context.device)

    timings = []
    for _ in range(measure_iterations):
        env.reset(options={"rotation": 0, "active_players": True, "q_agent_seat": 0})
        elapsed, _ = runtime.timed_call(fixed_call, device=context.device)
        timings.append(elapsed)

    env.close()
    n_games = context.benchmark_config["N_GAMES"]
    derived = [_per_second("action_batches_per_second", n_games, _stats(timings)["mean"], unit="games_per_second")]
    return _result(case, timings, metadata={"n_games": n_games, "action_profile": "check_call"}, derived_metrics=derived)


def run_env_step(case: BenchmarkCase, context, warmup_iterations: int, measure_iterations: int):
    agents, agent_types, _ = runtime.create_agents_and_types(context)
    env = runtime.create_env(context, agents)

    def prepare():
        state, info = env.reset(options={"rotation": 0, "active_players": True, "q_agent_seat": 0})
        actions = runtime.build_default_actions(state, info, agents, agent_types, context.device)
        return actions

    for _ in range(warmup_iterations):
        actions = prepare()
        runtime.timed_call(lambda: env.step(actions), device=context.device)

    timings = []
    for _ in range(measure_iterations):
        actions = prepare()
        elapsed, _ = runtime.timed_call(lambda: env.step(actions), device=context.device)
        timings.append(elapsed)

    env.close()
    n_games = context.benchmark_config["N_GAMES"]
    derived = [_per_second("env_steps_per_second", n_games, _stats(timings)["mean"], unit="games_per_second")]
    return _result(case, timings, metadata={"n_games": n_games}, derived_metrics=derived)


def run_trainer_build_actions(case: BenchmarkCase, context, warmup_iterations: int, measure_iterations: int):
    agents, agent_types, _ = runtime.create_agents_and_types(context)
    env = runtime.create_env(context, agents)
    state, info = env.reset(options={"rotation": 0, "active_players": True, "q_agent_seat": 0})

    def call():
        actions = torch.zeros(context.benchmark_config["N_GAMES"], dtype=torch.long, device=context.device)
        train_gpu.build_actions(state, actions, info["seat_idx"], agents, agent_types, context.device)
        return actions

    for _ in range(warmup_iterations):
        runtime.timed_call(call, device=context.device)

    timings = []
    for _ in range(measure_iterations):
        elapsed, _ = runtime.timed_call(call, device=context.device)
        timings.append(elapsed)

    env.close()
    n_games = context.benchmark_config["N_GAMES"]
    derived = [_per_second("actions_built_per_second", n_games, _stats(timings)["mean"], unit="games_per_second")]
    return _result(case, timings, metadata={"n_games": n_games}, derived_metrics=derived)


def run_trainer_q_network_train_step(case: BenchmarkCase, context, warmup_iterations: int, measure_iterations: int):
    _, _, q_net = runtime.create_agents_and_types(context)
    n_games = context.benchmark_config["N_GAMES"]
    state_dim = context.benchmark_config["STATE_SPACE"]
    action_dim = context.benchmark_config["ACTION_SPACE"]
    states = torch.randn((n_games, state_dim), dtype=torch.float32, device=context.device)
    next_states = torch.randn((n_games, state_dim), dtype=torch.float32, device=context.device)
    actions = torch.randint(0, action_dim, (n_games,), dtype=torch.long, device=context.device)
    rewards = torch.randn((n_games,), dtype=torch.float32, device=context.device)
    dones = torch.zeros((n_games,), dtype=torch.bool, device=context.device)
    states[:, 12] = 0

    def call():
        loss = q_net.train_step(states, actions, rewards, next_states, dones)
        if isinstance(loss, torch.Tensor):
            return float(loss.detach().item())
        return float(loss)

    for _ in range(warmup_iterations):
        runtime.timed_call(call, device=context.device)

    timings = []
    for _ in range(measure_iterations):
        elapsed, _ = runtime.timed_call(call, device=context.device)
        timings.append(elapsed)

    derived = [_per_second("q_updates_per_second", n_games, _stats(timings)["mean"], unit="samples_per_second")]
    return _result(case, timings, metadata={"batch_size": n_games}, derived_metrics=derived)


def run_trainer_short_run(case: BenchmarkCase, context, warmup_iterations: int, measure_iterations: int):
    del warmup_iterations
    timings = []
    steps_per_second = []
    episodes = context.benchmark_config["EPISODES"]
    n_games = context.benchmark_config["N_GAMES"]

    for _ in range(measure_iterations):
        deps = runtime.create_train_run_dependencies(context)

        def call():
            train_gpu.train_agent(
                env=deps["env"],
                agents=deps["agents"],
                agent_types=deps["agent_types"],
                episodes=episodes,
                n_games=n_games,
                device=context.device,
                results_dir=context.results_dir,
                config=context.benchmark_config,
                plotter=deps["plotter"],
                benchmarker=deps["benchmarker"],
            )

        elapsed, _ = runtime.timed_call(call, device=context.device)
        timings.append(elapsed)
        steps_per_second.append((episodes * n_games) / elapsed if elapsed > 0 else 0.0)
        deps["env"].close()

    derived = [
        {
            "name": "trainer_steps_per_second",
            "value": statistics.fmean(steps_per_second),
            "unit": "episode_games_per_second",
            "higher_is_better": True,
        }
    ]
    return _result(
        case,
        timings,
        metadata={"episodes": episodes, "n_games": n_games},
        derived_metrics=derived,
    )


CASE_REGISTRY = {
    "env_reset": BenchmarkCase(
        name="env_reset",
        category="environment",
        description="Times live PokerGPU.reset() throughput for vectorized game batches.",
        primary_metric_name="elapsed_seconds",
        primary_metric_unit="seconds",
        lower_is_better=True,
        runner=run_env_reset,
    ),
    "env_calculate_equities": BenchmarkCase(
        name="env_calculate_equities",
        category="environment",
        description="Times live PokerGPU.calculate_equities() on prepared river-state batches.",
        primary_metric_name="elapsed_seconds",
        primary_metric_unit="seconds",
        lower_is_better=True,
        runner=run_env_calculate_equities,
    ),
    "env_execute_actions": BenchmarkCase(
        name="env_execute_actions",
        category="environment",
        description="Times live PokerGPU.execute_actions() across the current vectorized game batch.",
        primary_metric_name="elapsed_seconds",
        primary_metric_unit="seconds",
        lower_is_better=True,
        runner=run_env_execute_actions,
    ),
    "env_step": BenchmarkCase(
        name="env_step",
        category="environment",
        description="Times one live PokerGPU.step(...) call including reward and round progression work.",
        primary_metric_name="elapsed_seconds",
        primary_metric_unit="seconds",
        lower_is_better=True,
        runner=run_env_step,
    ),
    "trainer_build_actions": BenchmarkCase(
        name="trainer_build_actions",
        category="trainer",
        description="Times live trainer action routing through build_actions(...) for one batch.",
        primary_metric_name="elapsed_seconds",
        primary_metric_unit="seconds",
        lower_is_better=True,
        runner=run_trainer_build_actions,
    ),
    "trainer_q_network_train_step": BenchmarkCase(
        name="trainer_q_network_train_step",
        category="trainer",
        description="Times live PokerQNetwork.train_step(...) update cost for one batch.",
        primary_metric_name="elapsed_seconds",
        primary_metric_unit="seconds",
        lower_is_better=True,
        runner=run_trainer_q_network_train_step,
    ),
    "trainer_short_run": BenchmarkCase(
        name="trainer_short_run",
        category="end_to_end",
        description="Times a short live trainGPU.train_agent(...) run with no-op plotting and benchmark file output.",
        primary_metric_name="elapsed_seconds",
        primary_metric_unit="seconds",
        lower_is_better=True,
        runner=run_trainer_short_run,
    ),
}
