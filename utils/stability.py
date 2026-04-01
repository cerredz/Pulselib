from __future__ import annotations

from typing import Any

import torch


def build_valid_q_learning_batch(states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Filter the Q-learning batch to rows that should contribute to the measured update."""
    valid_mask = (states[:, 12] == 0) | (states[:, 12] == 2)
    if not valid_mask.any():
        return None

    return states[valid_mask], actions[valid_mask], rewards[valid_mask], next_states[valid_mask], dones[valid_mask]


def calculate_q_learning_targets(q_network: Any, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    """Compute bootstrap targets on the same device as the incoming training batch."""
    with torch.no_grad():
        next_q_values = q_network.target_network(next_states).max(dim=1).values
        return rewards + q_network.gamma * next_q_values * (~dones).float()


def calculate_q_value_summary(q_values_for_actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Summarize chosen-action Q-values without leaving the tensor device path."""
    return {
        "q_mean": q_values_for_actions.mean(),
        "q_min": q_values_for_actions.min(),
        "q_max": q_values_for_actions.max(),
    }


def calculate_td_error(q_values_for_actions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the mean absolute TD error as a scalar tensor on the active device."""
    return torch.abs(q_values_for_actions - targets).mean()


def calculate_gradient_clip_rate(total_grad_norm: torch.Tensor, clip_threshold: float = 1.0) -> torch.Tensor:
    """Report whether the current step exceeded the configured clipping threshold."""
    threshold = torch.as_tensor(clip_threshold, device=total_grad_norm.device, dtype=total_grad_norm.dtype)
    one = torch.ones((), device=total_grad_norm.device, dtype=total_grad_norm.dtype)
    zero = torch.zeros((), device=total_grad_norm.device, dtype=total_grad_norm.dtype)
    return torch.where(total_grad_norm > threshold, one, zero)


def run_stability_measured_q_learning_step(q_network: Any, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor, *, clip_threshold: float = 1.0) -> dict[str, torch.Tensor] | None:
    """Run one measured optimizer step while keeping the derived metrics on-device."""
    batch = build_valid_q_learning_batch(states, actions, rewards, next_states, dones)
    if batch is None:
        return None

    valid_states, valid_actions, valid_rewards, valid_next_states, valid_dones = batch
    q_values = q_network(valid_states)
    q_values_for_actions = q_values.gather(1, valid_actions.unsqueeze(1)).squeeze(1)
    targets = calculate_q_learning_targets(q_network=q_network, rewards=valid_rewards, next_states=valid_next_states, dones=valid_dones)
    loss = q_network.criterion(q_values_for_actions, targets)

    q_network.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    total_grad_norm = torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=clip_threshold)
    q_network.optimizer.step()

    q_network.step_count += 1
    if q_network.step_count % q_network.update_freq == 0:
        q_network.target_network.load_state_dict(q_network.network.state_dict())

    metrics = {
        "loss": loss.detach(),
        "td_error": calculate_td_error(q_values_for_actions, targets),
        "grad_norm": total_grad_norm,
        "clip_rate": calculate_gradient_clip_rate(total_grad_norm, clip_threshold=clip_threshold),
    }
    metrics.update(calculate_q_value_summary(q_values_for_actions))
    return metrics


def stack_metric_values(step_metrics: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    """Stack one named metric across measured steps using torch tensors only."""
    return torch.stack([metric[key] for metric in step_metrics])


def summarize_episode_stability_metrics(episode_reward: torch.Tensor, step_metrics: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Aggregate one episode of step metrics into scalar tensors on the active device."""
    reward = episode_reward.detach()
    if not step_metrics:
        zero = torch.zeros((), device=reward.device, dtype=reward.dtype)
        return {
            "reward": reward,
            "q_mean": zero,
            "q_min": zero,
            "q_max": zero,
            "td_error": zero,
            "clip_rate": zero,
        }

    return {
        "reward": reward,
        "q_mean": stack_metric_values(step_metrics, "q_mean").mean(),
        "q_min": stack_metric_values(step_metrics, "q_min").min(),
        "q_max": stack_metric_values(step_metrics, "q_max").max(),
        "td_error": stack_metric_values(step_metrics, "td_error").mean(),
        "clip_rate": stack_metric_values(step_metrics, "clip_rate").mean(),
    }


def calculate_td_error_trend(td_errors: list[torch.Tensor]) -> torch.Tensor:
    """Compute the TD-error slope with torch math instead of switching to NumPy."""
    if len(td_errors) < 2:
        if td_errors:
            return torch.zeros((), device=td_errors[0].device, dtype=td_errors[0].dtype)
        return torch.zeros(())

    y = torch.stack(td_errors)
    x = torch.arange(y.shape[0], device=y.device, dtype=y.dtype)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = (x_centered * x_centered).sum().clamp_min(torch.finfo(y.dtype).eps)
    return (x_centered * y_centered).sum() / denominator


def resolve_metric_device(*metric_lists: list[torch.Tensor]) -> torch.device:
    """Pick the first available tensor device so final aggregates stay on that backend."""
    for metric_list in metric_lists:
        if metric_list:
            return metric_list[0].device
    return torch.device("cpu")


def calculate_final_stability_metrics(*, epoch_rewards: list[torch.Tensor], epoch_q_means: list[torch.Tensor], epoch_q_mins: list[torch.Tensor], epoch_q_maxs: list[torch.Tensor], epoch_td_errors: list[torch.Tensor], epoch_clip_rates: list[torch.Tensor], elapsed_seconds: float) -> dict[str, Any]:
    """Aggregate the full benchmark with torch reductions and device-preserving scalars."""
    device = resolve_metric_device(epoch_rewards, epoch_q_means, epoch_q_mins, epoch_q_maxs, epoch_td_errors, epoch_clip_rates)
    elapsed = torch.tensor(elapsed_seconds, device=device)
    if not epoch_rewards:
        zero = torch.zeros((), device=device)
        return {
            "reward_std": zero,
            "mean_reward": zero,
            "q_bounds": {
                "global_min": zero,
                "global_max": zero,
                "mean_q": zero,
            },
            "td_error_trend": zero,
            "average_clip_rate": zero,
            "total_time_seconds": elapsed,
        }

    rewards = torch.stack(epoch_rewards)
    q_means = torch.stack(epoch_q_means) if epoch_q_means else None
    q_mins = torch.stack(epoch_q_mins) if epoch_q_mins else None
    q_maxs = torch.stack(epoch_q_maxs) if epoch_q_maxs else None
    clip_rates = torch.stack(epoch_clip_rates) if epoch_clip_rates else None

    return {
        "reward_std": rewards.std(unbiased=False),
        "mean_reward": rewards.mean(),
        "q_bounds": {
            "global_min": q_mins.min() if q_mins is not None else torch.zeros((), device=device),
            "global_max": q_maxs.max() if q_maxs is not None else torch.zeros((), device=device),
            "mean_q": q_means.mean() if q_means is not None else torch.zeros((), device=device),
        },
        "td_error_trend": calculate_td_error_trend(epoch_td_errors),
        "average_clip_rate": clip_rates.mean() if clip_rates is not None else torch.zeros((), device=device),
        "total_time_seconds": elapsed,
    }
