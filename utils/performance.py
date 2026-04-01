from __future__ import annotations

from typing import Any

import torch


STREET_DEPTH_NAMES = {
    0: "preflop",
    1: "flop",
    2: "turn",
    3: "river",
    4: "showdown",
}
CONFIDENCE_Z_95 = 1.959963984540054


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


def format_scalar_metric_map(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    """Convert scalar tensor maps into plain floats for console output."""
    return {name: float(value.item()) for name, value in metrics.items()}


def format_nested_metric_values(value: Any) -> Any:
    """Recursively convert nested tensor metrics into plain Python values for output."""
    if isinstance(value, dict):
        return {key: format_nested_metric_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [format_nested_metric_values(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def calculate_q_seat_positions(buttons: torch.Tensor, *, q_seat: int, active_players: int) -> torch.Tensor:
    """Convert the Q-agent seat into button-relative positions for each batched hand."""
    q_seat_tensor = torch.full_like(buttons, q_seat)
    active_players_tensor = torch.full_like(buttons, active_players)
    return torch.remainder(q_seat_tensor - buttons, active_players_tensor).to(dtype=torch.int64)


def build_prefixed_deck_batch(*, n_games: int, seed: int, device: torch.device) -> torch.Tensor:
    """Build one deterministic shuffled deck per game using a fixed CPU generator seed."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shuffled = torch.rand((n_games, 52), generator=generator).argsort(dim=1) + 1
    return shuffled.to(device=device, dtype=torch.int32)


def build_opponent_mix_description(agent_strings: list[str]) -> str:
    """Create a stable human-readable label for the configured opponent pool."""
    return "+".join(agent_strings) if agent_strings else "no_opponents"


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


def calculate_bb_per_100_from_tensor(bb_deltas: torch.Tensor) -> torch.Tensor:
    """Return mean big blinds won per 100 hands for one flat delta tensor."""
    if bb_deltas.numel() == 0:
        return torch.zeros((), device=bb_deltas.device, dtype=bb_deltas.dtype)
    hundred = torch.tensor(100.0, device=bb_deltas.device, dtype=bb_deltas.dtype)
    return bb_deltas.mean() * hundred


def calculate_lcb95_bb_per_100_from_tensor(bb_deltas: torch.Tensor) -> torch.Tensor:
    """Return the lower 95% confidence bound of BB/100."""
    if bb_deltas.numel() == 0:
        return torch.zeros((), device=bb_deltas.device, dtype=bb_deltas.dtype)

    if bb_deltas.numel() == 1:
        return calculate_bb_per_100_from_tensor(bb_deltas)

    sample_std = bb_deltas.std(unbiased=False)
    sample_count = torch.tensor(float(bb_deltas.numel()), device=bb_deltas.device, dtype=bb_deltas.dtype)
    standard_error = sample_std / sample_count.sqrt()
    z_value = torch.tensor(CONFIDENCE_Z_95, device=bb_deltas.device, dtype=bb_deltas.dtype)
    hundred = torch.tensor(100.0, device=bb_deltas.device, dtype=bb_deltas.dtype)
    return (bb_deltas.mean() - (z_value * standard_error)) * hundred


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
            "field_bb_per_100": zero,
        }

    return {
        "reward": reward,
        "cumulative_reward": cumulative,
        "mean_bb_delta": deltas.mean(),
        "hand_win_rate": (deltas > 0).float().mean(),
        "hands_completed": torch.tensor(deltas.numel(), device=reward.device, dtype=torch.int64),
        "field_bb_per_100": calculate_bb_per_100_from_tensor(deltas),
    }


def bucketize_terminal_stages(stages: torch.Tensor) -> torch.Tensor:
    """Collapse terminal stage values into the preflop/flop/turn/river/showdown buckets."""
    showdown_bucket = torch.full_like(stages, 4)
    return torch.where(stages >= 4, showdown_bucket, stages.clamp(min=0, max=3))


def calculate_street_win_percentages(
    *,
    hand_bb_deltas: list[torch.Tensor],
    hand_terminal_stages: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return the percentage of completed hands won on each street bucket."""
    device = resolve_metric_device(hand_bb_deltas, hand_terminal_stages)
    deltas = flatten_metric_batches(hand_bb_deltas, dtype=torch.float32, device=device)
    stages = bucketize_terminal_stages(
        flatten_metric_batches(hand_terminal_stages, dtype=torch.int64, device=device)
    )
    zero = torch.zeros((), device=device)
    if deltas.numel() == 0:
        return {name: zero for name in STREET_DEPTH_NAMES.values()}

    win_mask = deltas > 0
    total_hands = torch.tensor(float(deltas.numel()), device=device)
    percentages: dict[str, torch.Tensor] = {}
    for stage_value, stage_name in STREET_DEPTH_NAMES.items():
        percentages[stage_name] = ((win_mask & (stages == stage_value)).float().sum()) / total_hands
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


def calculate_grouped_bb_per_100(
    *,
    hand_bb_deltas: torch.Tensor,
    group_ids: torch.Tensor,
    label_map: dict[int, str],
) -> dict[str, torch.Tensor]:
    """Compute BB/100 for each integer-labelled group."""
    if hand_bb_deltas.numel() == 0:
        return {}

    grouped_metrics: dict[str, torch.Tensor] = {}
    for group_id in group_ids.unique(sorted=True).unbind():
        group_value = int(group_id.item())
        group_mask = group_ids == group_id
        grouped_metrics[label_map[group_value]] = calculate_bb_per_100_from_tensor(hand_bb_deltas[group_mask])
    return grouped_metrics


def calculate_seat_balanced_bb_per_100(*, hand_bb_deltas: list[torch.Tensor], hand_positions: list[torch.Tensor]) -> torch.Tensor:
    """Average position BB/100 equally across observed seats to remove seat volume bias."""
    device = resolve_metric_device(hand_bb_deltas, hand_positions)
    deltas = flatten_metric_batches(hand_bb_deltas, dtype=torch.float32, device=device)
    positions = flatten_metric_batches(hand_positions, dtype=torch.int64, device=device)
    if deltas.numel() == 0:
        return torch.zeros((), device=device)

    seat_values = []
    for position in positions.unique(sorted=True).unbind():
        seat_values.append(calculate_bb_per_100_from_tensor(deltas[positions == position]))
    return torch.stack(seat_values).mean() if seat_values else torch.zeros((), device=device)


def calculate_slice_metrics(
    *,
    hand_bb_deltas: list[torch.Tensor],
    hand_positions: list[torch.Tensor],
    hand_player_counts: list[torch.Tensor],
    hand_terminal_stages: list[torch.Tensor],
    hand_opponent_mix_ids: list[torch.Tensor],
    opponent_mix_descriptions: dict[str, str],
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute BB/100 across the requested slice families."""
    device = resolve_metric_device(
        hand_bb_deltas,
        hand_positions,
        hand_player_counts,
        hand_terminal_stages,
        hand_opponent_mix_ids,
    )
    deltas = flatten_metric_batches(hand_bb_deltas, dtype=torch.float32, device=device)
    positions = flatten_metric_batches(hand_positions, dtype=torch.int64, device=device)
    player_counts = flatten_metric_batches(hand_player_counts, dtype=torch.int64, device=device)
    street_depths = bucketize_terminal_stages(
        flatten_metric_batches(hand_terminal_stages, dtype=torch.int64, device=device)
    )
    opponent_mix_ids = flatten_metric_batches(hand_opponent_mix_ids, dtype=torch.int64, device=device)
    if deltas.numel() == 0:
        return {
            "opponent_mix": {},
            "seat": {},
            "player_count": {},
            "street_depth": {},
        }

    opponent_mix_label_map = {
        int(slice_name.split("_")[1]): slice_name for slice_name in opponent_mix_descriptions
    }
    seat_label_map = {int(position.item()): f"position_{int(position.item())}" for position in positions.unique(sorted=True).unbind()}
    player_count_label_map = {
        int(player_count.item()): f"players_{int(player_count.item())}"
        for player_count in player_counts.unique(sorted=True).unbind()
    }

    return {
        "opponent_mix": calculate_grouped_bb_per_100(
            hand_bb_deltas=deltas,
            group_ids=opponent_mix_ids,
            label_map=opponent_mix_label_map,
        ),
        "seat": calculate_grouped_bb_per_100(
            hand_bb_deltas=deltas,
            group_ids=positions,
            label_map=seat_label_map,
        ),
        "player_count": calculate_grouped_bb_per_100(
            hand_bb_deltas=deltas,
            group_ids=player_counts,
            label_map=player_count_label_map,
        ),
        "street_depth": calculate_grouped_bb_per_100(
            hand_bb_deltas=deltas,
            group_ids=street_depths,
            label_map=STREET_DEPTH_NAMES,
        ),
    }


def calculate_worst_slice_metrics(
    slice_metrics: dict[str, dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Find the minimum BB/100 across all configured slice families."""
    slice_values: list[torch.Tensor] = []
    metadata: list[tuple[str, str]] = []
    for family_name, family_metrics in slice_metrics.items():
        for slice_name, value in family_metrics.items():
            metadata.append((family_name, slice_name))
            slice_values.append(value)

    zero = torch.zeros((), device=device)
    if not slice_values:
        return {
            "bb_per_100": zero,
            "family": "",
            "slice": "",
        }

    stacked = torch.stack(slice_values)
    worst_index = int(stacked.argmin().item())
    family_name, slice_name = metadata[worst_index]
    return {
        "bb_per_100": stacked[worst_index],
        "family": family_name,
        "slice": slice_name,
    }


def calculate_final_performance_metrics(
    *,
    epoch_rewards: list[torch.Tensor],
    hand_bb_deltas: list[torch.Tensor],
    hand_terminal_stages: list[torch.Tensor],
    hand_positions: list[torch.Tensor],
    hand_player_counts: list[torch.Tensor],
    hand_opponent_mix_ids: list[torch.Tensor],
    elapsed_seconds: float,
    rolling_window_size: int,
    use_prefixed_decks: bool,
    opponent_mix_descriptions: dict[str, str],
) -> dict[str, Any]:
    """Aggregate the full benchmark into tensor-native performance metrics."""
    device = resolve_metric_device(
        epoch_rewards,
        hand_bb_deltas,
        hand_terminal_stages,
        hand_positions,
        hand_player_counts,
        hand_opponent_mix_ids,
    )
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
    slice_metrics = calculate_slice_metrics(
        hand_bb_deltas=hand_bb_deltas,
        hand_positions=hand_positions,
        hand_player_counts=hand_player_counts,
        hand_terminal_stages=hand_terminal_stages,
        hand_opponent_mix_ids=hand_opponent_mix_ids,
        opponent_mix_descriptions=opponent_mix_descriptions,
    )
    worst_slice = calculate_worst_slice_metrics(slice_metrics, device=device)

    if rewards.numel() == 0:
        return {
            "cumulative_reward": zero,
            "mean_reward": zero,
            "reward_improvement": {
                "slope": zero,
                "first_to_last_percent_change": zero,
            },
            "total_bb_won": zero,
            "field_bb_per_100": zero,
            "paired_field_bb_per_100": zero,
            "lcb95_bb_per_100": zero,
            "seat_balanced_bb_per_100": zero,
            "paired_prefixed_decks_enabled": use_prefixed_decks,
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
            "slice_bb_per_100": slice_metrics,
            "worst_slice_bb_per_100": zero,
            "worst_slice_details": worst_slice,
            "opponent_mix_descriptions": opponent_mix_descriptions,
            "total_time_seconds": elapsed,
        }

    field_bb_per_100 = calculate_bb_per_100_from_tensor(deltas)

    return {
        "cumulative_reward": rewards.sum(),
        "mean_reward": rewards.mean(),
        "reward_improvement": {
            "slope": calculate_linear_trend(list(rewards.unbind())),
            "first_to_last_percent_change": calculate_reward_percent_change(list(rewards.unbind())),
        },
        "total_bb_won": deltas.sum() if deltas.numel() else zero,
        "field_bb_per_100": field_bb_per_100,
        "paired_field_bb_per_100": field_bb_per_100 if use_prefixed_decks else zero,
        "lcb95_bb_per_100": calculate_lcb95_bb_per_100_from_tensor(deltas),
        "seat_balanced_bb_per_100": calculate_seat_balanced_bb_per_100(
            hand_bb_deltas=hand_bb_deltas,
            hand_positions=hand_positions,
        ),
        "paired_prefixed_decks_enabled": use_prefixed_decks,
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
        "slice_bb_per_100": slice_metrics,
        "worst_slice_bb_per_100": worst_slice["bb_per_100"],
        "worst_slice_details": worst_slice,
        "opponent_mix_descriptions": opponent_mix_descriptions,
        "total_time_seconds": elapsed,
    }
