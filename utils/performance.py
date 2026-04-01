from __future__ import annotations

from typing import Any

import torch


STREET_NAMES = {
    0: "preflop",
    1: "flop",
    2: "turn",
    3: "river",
}


def resolve_metric_device(*metric_lists: list[torch.Tensor]) -> torch.device:
    """Pick the first available tensor device so aggregates stay on that backend."""
    for metric_list in metric_lists:
        if metric_list:
            return metric_list[0].device
    return torch.device("cpu")


def flatten_metric_batches(
    metric_batches: list[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Flatten a list of scalar or vector batches into one one-dimensional tensor."""
    if not metric_batches:
        return torch.empty(0, device=device, dtype=dtype)

    return torch.cat([batch.reshape(-1).to(device=device, dtype=dtype) for batch in metric_batches], dim=0)


def calculate_linear_trend(values: list[torch.Tensor]) -> torch.Tensor:
    """Compute a simple linear slope with tensor math only."""
    if len(values) < 2:
        if values:
            return torch.zeros((), device=values[0].device, dtype=values[0].dtype)
        return torch.zeros(())

    y = torch.stack(values)
    x = torch.arange(y.shape[0], device=y.device, dtype=y.dtype)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = (x_centered * x_centered).sum().clamp_min(torch.finfo(y.dtype).eps)
    return (x_centered * y_centered).sum() / denominator


def calculate_reward_percent_change(epoch_rewards: list[torch.Tensor]) -> torch.Tensor:
    """Measure first-to-last reward change as a percentage."""
    if len(epoch_rewards) < 2:
        if epoch_rewards:
            return torch.zeros((), device=epoch_rewards[0].device, dtype=epoch_rewards[0].dtype)
        return torch.zeros(())

    first_reward = epoch_rewards[0]
    last_reward = epoch_rewards[-1]
    denominator = torch.abs(first_reward).clamp_min(torch.finfo(first_reward.dtype).eps)
    hundred = torch.tensor(100.0, device=first_reward.device, dtype=first_reward.dtype)
    return ((last_reward - first_reward) / denominator) * hundred


def calculate_rolling_window_averages(hand_bb_deltas: list[torch.Tensor], *, window_size: int) -> list[torch.Tensor]:
    """Compute rolling mean big-blind deltas across completed hands."""
    device = resolve_metric_device(hand_bb_deltas)
    deltas = flatten_metric_batches(hand_bb_deltas, dtype=torch.float32, device=device)
    if deltas.numel() < window_size or window_size <= 0:
        return []

    return list(deltas.unfold(0, window_size, 1).mean(dim=1).unbind())


def summarize_episode_performance_metrics(
    *,
    episode_reward: torch.Tensor,
    cumulative_reward: torch.Tensor,
    hand_bb_deltas: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Summarize one episode's reward and hand-level performance."""
    reward = episode_reward.detach()
    cumulative = cumulative_reward.detach()
    deltas = flatten_metric_batches(hand_bb_deltas, dtype=reward.dtype, device=reward.device)
    if deltas.numel() == 0:
        zero = torch.zeros((), device=reward.device, dtype=reward.dtype)
        zero_count = torch.zeros((), device=reward.device, dtype=torch.int64)
        return {
            "reward": reward,
            "cumulative_reward": cumulative,
            "mean_bb_delta": zero,
            "hand_win_rate": zero,
            "hands_completed": zero_count,
        }

    return {
        "reward": reward,
        "cumulative_reward": cumulative,
        "mean_bb_delta": deltas.mean(),
        "hand_win_rate": (deltas > 0).float().mean(),
        "hands_completed": torch.tensor(deltas.numel(), device=reward.device, dtype=torch.int64),
    }


def calculate_street_win_percentages(
    *,
    hand_bb_deltas: list[torch.Tensor],
    hand_terminal_stages: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return the percentage of completed hands won on each street bucket."""
    device = resolve_metric_device(hand_bb_deltas, hand_terminal_stages)
    deltas = flatten_metric_batches(hand_bb_deltas, dtype=torch.float32, device=device)
    stages = flatten_metric_batches(hand_terminal_stages, dtype=torch.int64, device=device)
    zero = torch.zeros((), device=device)
    if deltas.numel() == 0:
        return {name: zero for name in (*STREET_NAMES.values(), "showdown")}

    win_mask = deltas > 0
    total_hands = torch.tensor(float(deltas.numel()), device=device)
    percentages: dict[str, torch.Tensor] = {}
    for stage_value, stage_name in STREET_NAMES.items():
        percentages[stage_name] = ((win_mask & (stages == stage_value)).float().sum()) / total_hands
    percentages["showdown"] = ((win_mask & (stages >= 4)).float().sum()) / total_hands
    return percentages


def calculate_position_win_rates(
    *,
    hand_bb_deltas: list[torch.Tensor],
    hand_positions: list[torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    """Return button-relative position win rates over hands played from each position."""
    device = resolve_metric_device(hand_bb_deltas, hand_positions)
    deltas = flatten_metric_batches(hand_bb_deltas, dtype=torch.float32, device=device)
    positions = flatten_metric_batches(hand_positions, dtype=torch.int64, device=device)
    if deltas.numel() == 0:
        return {}

    position_metrics: dict[str, dict[str, torch.Tensor]] = {}
    for position in positions.unique(sorted=True).unbind():
        position_mask = positions == position
        position_deltas = deltas[position_mask]
        position_key = f"position_{int(position.item())}"
        position_metrics[position_key] = {
            "hands": torch.tensor(position_deltas.numel(), device=device, dtype=torch.int64),
            "wins": (position_deltas > 0).sum().to(dtype=torch.int64),
            "win_rate": (position_deltas > 0).float().mean() if position_deltas.numel() else torch.zeros((), device=device),
        }
    return position_metrics


def calculate_final_performance_metrics(
    *,
    epoch_rewards: list[torch.Tensor],
    hand_bb_deltas: list[torch.Tensor],
    hand_terminal_stages: list[torch.Tensor],
    hand_positions: list[torch.Tensor],
    elapsed_seconds: float,
    rolling_window_size: int,
) -> dict[str, Any]:
    """Aggregate the full benchmark into tensor-native performance metrics."""
    device = resolve_metric_device(epoch_rewards, hand_bb_deltas, hand_terminal_stages, hand_positions)
    elapsed = torch.tensor(elapsed_seconds, device=device)
    zero = torch.zeros((), device=device)
    zero_count = torch.zeros((), device=device, dtype=torch.int64)

    rewards = flatten_metric_batches(epoch_rewards, dtype=torch.float32, device=device)
    deltas = flatten_metric_batches(hand_bb_deltas, dtype=torch.float32, device=device)
    rolling_averages = calculate_rolling_window_averages(hand_bb_deltas, window_size=rolling_window_size)
    street_win_percentages = calculate_street_win_percentages(
        hand_bb_deltas=hand_bb_deltas,
        hand_terminal_stages=hand_terminal_stages,
    )
    position_metrics = calculate_position_win_rates(
        hand_bb_deltas=hand_bb_deltas,
        hand_positions=hand_positions,
    )

    if rewards.numel() == 0:
        return {
            "cumulative_reward": zero,
            "mean_reward": zero,
            "reward_improvement": {
                "slope": zero,
                "first_to_last_percent_change": zero,
            },
            "total_bb_won": zero,
            "overall_hand_win_rate": zero,
            "total_hands": zero_count,
            "rolling_bb_window": {
                "window_size": rolling_window_size,
                "num_windows": zero_count,
                "last_average": zero,
                "best_average": zero,
                "values": [],
            },
            "street_win_percentages": street_win_percentages,
            "position_win_rates": {},
            "position_hand_counts": {},
            "total_time_seconds": elapsed,
        }

    return {
        "cumulative_reward": rewards.sum(),
        "mean_reward": rewards.mean(),
        "reward_improvement": {
            "slope": calculate_linear_trend(list(rewards.unbind())),
            "first_to_last_percent_change": calculate_reward_percent_change(list(rewards.unbind())),
        },
        "total_bb_won": deltas.sum() if deltas.numel() else zero,
        "overall_hand_win_rate": (deltas > 0).float().mean() if deltas.numel() else zero,
        "total_hands": torch.tensor(deltas.numel(), device=device, dtype=torch.int64),
        "rolling_bb_window": {
            "window_size": rolling_window_size,
            "num_windows": torch.tensor(len(rolling_averages), device=device, dtype=torch.int64),
            "last_average": rolling_averages[-1] if rolling_averages else zero,
            "best_average": torch.stack(rolling_averages).max() if rolling_averages else zero,
            "values": [value.detach() for value in rolling_averages],
        },
        "street_win_percentages": street_win_percentages,
        "position_win_rates": {
            position: metrics["win_rate"] for position, metrics in position_metrics.items()
        },
        "position_hand_counts": {
            position: metrics["hands"] for position, metrics in position_metrics.items()
        },
        "total_time_seconds": elapsed,
    }
