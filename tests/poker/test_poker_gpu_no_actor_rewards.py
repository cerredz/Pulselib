import random

import eval7
import numpy as np
import pytest
import torch

from environments.Poker.PokerGPU import PokerGPU


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_env(n_players: int, *, n_games: int = 1, seed: int = 0) -> PokerGPU:
    _seed_everything(seed)
    env = PokerGPU(
        device=torch.device("cpu"),
        agents=[],
        n_players=n_players,
        max_players=n_players,
        n_games=n_games,
    )
    env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})
    env.active_players = n_players
    return env


def _encode(card_str: str) -> int:
    card = eval7.Card(card_str)
    return card.rank + (13 * card.suit) + 1


def _configure_no_actor_runout(env: PokerGPU, *, game: int = 0, stage: int = 0) -> None:
    env.status[game] = torch.tensor([env.ALLIN, env.ALLIN, env.FOLDED], dtype=torch.int32)
    env.stacks[game] = torch.tensor([90, 90, 100], dtype=torch.int32)
    env.idx[game] = torch.tensor(0, dtype=torch.int32)
    env.agg[game] = torch.tensor(0, dtype=torch.int32)
    env.acted[game] = torch.tensor(0, dtype=torch.int32)
    env.highest[game] = torch.tensor(10, dtype=torch.int32)
    env.current_round_bet[game] = torch.tensor([10, 10, 0], dtype=torch.int32)
    env.total_invested[game] = torch.tensor([10, 10, 0], dtype=torch.int32)
    env.pots[game] = torch.tensor(20, dtype=torch.int32)
    env.stages[game] = torch.tensor(stage, dtype=torch.int32)
    env.is_done[game] = False


def _configure_live_actor_state(env: PokerGPU, *, game: int = 0) -> None:
    env.status[game] = torch.tensor([env.ACTIVE, env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.stacks[game] = torch.tensor([100, 100, 100], dtype=torch.int32)
    env.idx[game] = torch.tensor(0, dtype=torch.int32)
    env.agg[game] = torch.tensor(1, dtype=torch.int32)
    env.acted[game] = torch.tensor(0, dtype=torch.int32)
    env.highest[game] = torch.tensor(10, dtype=torch.int32)
    env.current_round_bet[game] = torch.tensor([0, 10, 10], dtype=torch.int32)
    env.total_invested[game] = torch.tensor([0, 10, 10], dtype=torch.int32)
    env.pots[game] = torch.tensor(20, dtype=torch.int32)
    env.stages[game] = torch.tensor(0, dtype=torch.int32)
    env.is_done[game] = False


def test_step_auto_runout_zeroes_placeholder_rewards_across_action_ids() -> None:
    reward_by_action: dict[int, float] = {}

    for action in (0, 1, 12):
        env = _build_env(n_players=3, seed=11)
        _configure_no_actor_runout(env)
        _, rewards, dones, _, _ = env.step(torch.tensor([action], dtype=torch.long))
        reward_by_action[action] = float(rewards[0].item())

        assert not dones[0].item()
        assert env.stages[0].item() == 1

    assert reward_by_action[0] == pytest.approx(0.0)
    assert reward_by_action[1] == pytest.approx(0.0)
    assert reward_by_action[12] == pytest.approx(0.0)


def test_step_auto_runout_keeps_transition_state_identical_for_placeholder_actions() -> None:
    snapshots: list[tuple[int, list[int], int, bool]] = []

    for action in (0, 1, 12):
        env = _build_env(n_players=3, seed=19)
        _configure_no_actor_runout(env)
        _, rewards, dones, _, _ = env.step(torch.tensor([action], dtype=torch.long))
        snapshots.append(
            (
                int(env.stages[0].item()),
                env.board[0].tolist(),
                int(env.pots[0].item()),
                bool(dones[0].item()),
            )
        )
        assert rewards[0].item() == pytest.approx(0.0)

    assert snapshots[0] == snapshots[1] == snapshots[2]
    assert snapshots[0][0] == 1
    assert snapshots[0][1][0:3] != [-1, -1, -1]


def test_step_auto_runout_on_river_resolves_showdown_without_action_reward() -> None:
    env = _build_env(n_players=3, seed=23)
    _configure_no_actor_runout(env, stage=3)
    env.board[0] = torch.tensor([_encode(card) for card in ["2c", "7d", "9h", "Js", "Kd"]], dtype=torch.int32)
    env.hands[0, 0] = torch.tensor([_encode("Ah"), _encode("Qh")], dtype=torch.int32)
    env.hands[0, 1] = torch.tensor([_encode("3c"), _encode("4d")], dtype=torch.int32)
    env.hands[0, 2] = torch.tensor([_encode("5c"), _encode("5d")], dtype=torch.int32)

    _, rewards, dones, _, _ = env.step(torch.tensor([12], dtype=torch.long))

    assert rewards[0].item() == pytest.approx(0.0)
    assert dones[0].item()
    assert env.pots[0].item() == 0
    assert env.stages[0].item() == 5
    assert env.stacks[0].tolist() == [110, 90, 100]


def test_step_batched_rewards_zero_only_games_without_legal_actor() -> None:
    env = _build_env(n_players=3, n_games=2, seed=31)
    _configure_no_actor_runout(env, game=0)
    env.status[1] = torch.tensor([env.ACTIVE, env.ACTIVE, env.FOLDED], dtype=torch.int32)
    env.stacks[1] = torch.tensor([100, 100, 100], dtype=torch.int32)
    env.idx[1] = torch.tensor(0, dtype=torch.int32)
    env.agg[1] = torch.tensor(1, dtype=torch.int32)
    env.acted[1] = torch.tensor(0, dtype=torch.int32)
    env.highest[1] = torch.tensor(10, dtype=torch.int32)
    env.current_round_bet[1] = torch.tensor([0, 10, 0], dtype=torch.int32)
    env.total_invested[1] = torch.tensor([0, 10, 0], dtype=torch.int32)
    env.pots[1] = torch.tensor(20, dtype=torch.int32)
    env.stages[1] = torch.tensor(0, dtype=torch.int32)
    env.is_done[1] = False

    _, rewards, dones, _, info = env.step(torch.tensor([12, 1], dtype=torch.long))

    assert rewards[0].item() == pytest.approx(0.0)
    assert rewards[1].item() != pytest.approx(0.0)
    assert not dones[0].item()
    assert not dones[1].item()
    assert env.stages.tolist() == [1, 0]
    assert env.current_round_bet[1].tolist() == [10, 10, 0]
    assert info["seat_idx"][1].item() == 1


def test_step_preserves_action_sensitive_rewards_when_a_legal_actor_exists() -> None:
    rewards_by_action: dict[int, float] = {}

    for action in (0, 1):
        env = _build_env(n_players=3, seed=37)
        _configure_live_actor_state(env)
        _, rewards, dones, _, _ = env.step(torch.tensor([action], dtype=torch.long))
        rewards_by_action[action] = float(rewards[0].item())

        assert not dones[0].item()

    assert rewards_by_action[1] > rewards_by_action[0]
