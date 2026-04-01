import eval7
import pytest
import torch

from environments.Poker.PokerGPU import PokerGPU


def _encode(card_str: str) -> int:
    card = eval7.Card(card_str)
    return card.rank + (13 * card.suit) + 1


def _build_env(*, n_players: int = 2, n_games: int = 1) -> PokerGPU:
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


def test_poker_reward_gpu_zero_pot_and_call_cost_returns_finite_zero_rewards() -> None:
    env = _build_env(n_players=3, n_games=3)
    env.status[:] = torch.tensor(
        [
            [env.ACTIVE, env.ACTIVE, env.ACTIVE],
            [env.ACTIVE, env.ACTIVE, env.ACTIVE],
            [env.ACTIVE, env.ACTIVE, env.ACTIVE],
        ],
        dtype=torch.int32,
    )
    env.pots.zero_()
    env.highest.zero_()
    env.prev_invested.zero_()
    env.equities[:] = 0.5

    rewards = env.poker_reward_gpu(
        actions=torch.tensor([0, 1, 12], dtype=torch.long),
        actor_idx=torch.tensor([0, 0, 0], dtype=torch.int32),
    )

    assert torch.isfinite(rewards).all()
    assert rewards.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_poker_reward_gpu_call_reward_increases_with_equity() -> None:
    env = _build_env(n_players=2, n_games=2)
    env.status[:] = torch.tensor([[env.ACTIVE, env.ACTIVE], [env.ACTIVE, env.ACTIVE]], dtype=torch.int32)
    env.pots[:] = torch.tensor([20, 20], dtype=torch.int32)
    env.highest[:] = torch.tensor([10, 10], dtype=torch.int32)
    env.prev_invested[:] = torch.tensor([0, 0], dtype=torch.int32)
    env.equities[0] = torch.tensor([0.2, 0.8], dtype=torch.float32)
    env.equities[1] = torch.tensor([0.8, 0.2], dtype=torch.float32)

    rewards = env.poker_reward_gpu(
        actions=torch.tensor([1, 1], dtype=torch.long),
        actor_idx=torch.tensor([0, 0], dtype=torch.int32),
    )

    assert rewards[1].item() > rewards[0].item()


def test_poker_reward_gpu_fold_reward_decreases_with_equity() -> None:
    env = _build_env(n_players=2, n_games=2)
    env.w1 = torch.tensor(0.0, device=env.device, dtype=torch.float32)
    env.w2 = torch.tensor(1.0, device=env.device, dtype=torch.float32)
    env.status[:] = torch.tensor([[env.ACTIVE, env.ACTIVE], [env.ACTIVE, env.ACTIVE]], dtype=torch.int32)
    env.pots[:] = torch.tensor([20, 20], dtype=torch.int32)
    env.highest[:] = torch.tensor([10, 10], dtype=torch.int32)
    env.prev_invested[:] = torch.tensor([0, 0], dtype=torch.int32)
    env.equities[0] = torch.tensor([0.1, 0.8], dtype=torch.float32)
    env.equities[1] = torch.tensor([0.8, 0.2], dtype=torch.float32)

    rewards = env.poker_reward_gpu(
        actions=torch.tensor([0, 0], dtype=torch.long),
        actor_idx=torch.tensor([0, 0], dtype=torch.int32),
    )

    assert rewards[1].item() < rewards[0].item()


def test_poker_reward_gpu_raise_reward_tracks_equity_vs_fair_share() -> None:
    env = _build_env(n_players=3, n_games=2)
    env.w1 = torch.tensor(0.0, device=env.device, dtype=torch.float32)
    env.w2 = torch.tensor(1.0, device=env.device, dtype=torch.float32)
    env.status[:] = torch.tensor(
        [[env.ACTIVE, env.ACTIVE, env.ACTIVE], [env.ACTIVE, env.ACTIVE, env.ACTIVE]],
        dtype=torch.int32,
    )
    env.pots[:] = torch.tensor([30, 30], dtype=torch.int32)
    env.highest[:] = torch.tensor([0, 0], dtype=torch.int32)
    env.prev_invested[:] = torch.tensor([0, 0], dtype=torch.int32)
    env.equities[0] = torch.tensor([0.2, 0.5, 0.5], dtype=torch.float32)
    env.equities[1] = torch.tensor([0.8, 0.5, 0.5], dtype=torch.float32)

    rewards = env.poker_reward_gpu(
        actions=torch.tensor([2, 2], dtype=torch.long),
        actor_idx=torch.tensor([0, 0], dtype=torch.int32),
    )

    assert rewards[0].item() < 0.0
    assert rewards[1].item() > 0.0


def test_calculate_equities_postflop_rows_stay_bounded_and_rank_stronger_river_hand_higher() -> None:
    env = _build_env(n_players=2, n_games=3)
    env.equity_dirty[:] = torch.tensor([True, True, True], dtype=torch.bool)
    env.stages[:] = torch.tensor([1, 2, 3], dtype=torch.int32)

    env.board[0] = torch.tensor([_encode("2c"), _encode("7d"), _encode("9h"), -1, -1], dtype=torch.int32)
    env.hands[0, 0] = torch.tensor([_encode("Ah"), _encode("Kd")], dtype=torch.int32)
    env.hands[0, 1] = torch.tensor([_encode("3c"), _encode("4d")], dtype=torch.int32)

    env.board[1] = torch.tensor([_encode("2d"), _encode("5s"), _encode("8c"), _encode("Jh"), -1], dtype=torch.int32)
    env.hands[1, 0] = torch.tensor([_encode("Ad"), _encode("Qs")], dtype=torch.int32)
    env.hands[1, 1] = torch.tensor([_encode("4h"), _encode("4s")], dtype=torch.int32)

    env.board[2] = torch.tensor([_encode("2h"), _encode("7s"), _encode("9d"), _encode("Jc"), _encode("Kd")], dtype=torch.int32)
    env.hands[2, 0] = torch.tensor([_encode("Ah"), _encode("Ad")], dtype=torch.int32)
    env.hands[2, 1] = torch.tensor([_encode("3c"), _encode("4d")], dtype=torch.int32)

    env.calculate_equities()

    assert torch.all((env.equities >= 0.0) & (env.equities <= 1.0))
    assert not env.equity_dirty.any()
    assert env.equities[2, 0].item() > env.equities[2, 1].item()
