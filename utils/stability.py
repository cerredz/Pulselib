from __future__ import annotations

from typing import Any

import numpy as np
import torch


def build_valid_q_learning_batch(
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    valid_mask = (states[:, 12] == 0) | (states[:, 12] == 2)
    if not valid_mask.any():
        return None

    return (
        states[valid_mask],
        actions[valid_mask],
        rewards[valid_mask],
        next_states[valid_mask],
        dones[valid_mask],
    )


def calculate_q_learning_targets(
    q_network: Any,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        next_q_values = q_network.target_network(next_states).max(dim=1).values
        return rewards + q_network.gamma * next_q_values * (~dones).float()


def calculate_q_value_summary(q_values_for_actions: torch.Tensor) -> dict[str, float]:
    return {
        "q_mean": float(q_values_for_actions.mean().item()),
        "q_min": float(q_values_for_actions.min().item()),
        "q_max": float(q_values_for_actions.max().item()),
    }


def calculate_td_error(q_values_for_actions: torch.Tensor, targets: torch.Tensor) -> float:
    return float(torch.abs(q_values_for_actions - targets).mean().item())


def calculate_gradient_clip_rate(total_grad_norm: float, clip_threshold: float = 1.0) -> float:
    return 1.0 if total_grad_norm > clip_threshold else 0.0


def run_stability_measured_q_learning_step(
    q_network: Any,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    *,
    clip_threshold: float = 1.0,
) -> dict[str, float] | None:
    batch = build_valid_q_learning_batch(states, actions, rewards, next_states, dones)
    if batch is None:
        return None

    valid_states, valid_actions, valid_rewards, valid_next_states, valid_dones = batch

    q_values = q_network(valid_states)
    q_values_for_actions = q_values.gather(1, valid_actions.unsqueeze(1)).squeeze(1)
    targets = calculate_q_learning_targets(
        q_network=q_network,
        rewards=valid_rewards,
        next_states=valid_next_states,
        dones=valid_dones,
    )

    loss = q_network.criterion(q_values_for_actions, targets)

    q_network.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    total_grad_norm = float(
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=clip_threshold).item()
    )
    q_network.optimizer.step()

    q_network.step_count += 1
    if q_network.step_count % q_network.update_freq == 0:
        q_network.target_network.load_state_dict(q_network.network.state_dict())

    metrics = {
        "loss": float(loss.item()),
        "td_error": calculate_td_error(q_values_for_actions, targets),
        "grad_norm": total_grad_norm,
        "clip_rate": calculate_gradient_clip_rate(total_grad_norm, clip_threshold=clip_threshold),
    }
    metrics.update(calculate_q_value_summary(q_values_for_actions))
    return metrics


def summarize_episode_stability_metrics(
    episode_reward: float,
    step_metrics: list[dict[str, float]],
) -> dict[str, float]:
    if not step_metrics:
        return {
            "reward": float(episode_reward),
            "q_mean": 0.0,
            "q_min": 0.0,
            "q_max": 0.0,
            "td_error": 0.0,
            "clip_rate": 0.0,
        }

    return {
        "reward": float(episode_reward),
        "q_mean": float(np.mean([metric["q_mean"] for metric in step_metrics])),
        "q_min": float(np.min([metric["q_min"] for metric in step_metrics])),
        "q_max": float(np.max([metric["q_max"] for metric in step_metrics])),
        "td_error": float(np.mean([metric["td_error"] for metric in step_metrics])),
        "clip_rate": float(np.mean([metric["clip_rate"] for metric in step_metrics])),
    }


def calculate_td_error_trend(td_errors: list[float]) -> float:
    if len(td_errors) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(td_errors)), td_errors, 1)[0])


def calculate_final_stability_metrics(
    *,
    epoch_rewards: list[float],
    epoch_q_means: list[float],
    epoch_q_mins: list[float],
    epoch_q_maxs: list[float],
    epoch_td_errors: list[float],
    epoch_clip_rates: list[float],
    elapsed_seconds: float,
) -> dict[str, Any]:
    if not epoch_rewards:
        return {
            "reward_std": 0.0,
            "mean_reward": 0.0,
            "q_bounds": {
                "global_min": 0.0,
                "global_max": 0.0,
                "mean_q": 0.0,
            },
            "td_error_trend": 0.0,
            "average_clip_rate": 0.0,
            "total_time_seconds": float(elapsed_seconds),
        }

    return {
        "reward_std": float(np.std(epoch_rewards)),
        "mean_reward": float(np.mean(epoch_rewards)),
        "q_bounds": {
            "global_min": float(np.min(epoch_q_mins)) if epoch_q_mins else 0.0,
            "global_max": float(np.max(epoch_q_maxs)) if epoch_q_maxs else 0.0,
            "mean_q": float(np.mean(epoch_q_means)) if epoch_q_means else 0.0,
        },
        "td_error_trend": calculate_td_error_trend(epoch_td_errors),
        "average_clip_rate": float(np.mean(epoch_clip_rates)) if epoch_clip_rates else 0.0,
        "total_time_seconds": float(elapsed_seconds),
    }
