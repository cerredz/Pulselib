import math
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


def _encode(card_str: str) -> int:
    card = eval7.Card(card_str)
    return card.rank + (13 * card.suit) + 1


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


def _configure_game(
    env: PokerGPU,
    *,
    game: int,
    board: list[str],
    hands: list[tuple[str, str]],
    status: list[int],
    stacks: list[int],
    current_round_bet: list[int],
    total_invested: list[int],
    idx: int,
    agg: int,
    acted: int,
    stage: int,
    highest: int,
    pot: int,
    last_raise_size: int = 1,
) -> None:
    env.board[game] = torch.tensor([_encode(card) for card in board], dtype=torch.int32)
    env.hands[game] = torch.tensor(
        [[_encode(card_a), _encode(card_b)] for card_a, card_b in hands],
        dtype=torch.int32,
    )
    env.status[game] = torch.tensor(status, dtype=torch.int32)
    env.stacks[game] = torch.tensor(stacks, dtype=torch.int32)
    env.current_round_bet[game] = torch.tensor(current_round_bet, dtype=torch.int32)
    env.total_invested[game] = torch.tensor(total_invested, dtype=torch.int32)
    env.idx[game] = torch.tensor(idx, dtype=torch.int32)
    env.agg[game] = torch.tensor(agg, dtype=torch.int32)
    env.acted[game] = torch.tensor(acted, dtype=torch.int32)
    env.stages[game] = torch.tensor(stage, dtype=torch.int32)
    env.highest[game] = torch.tensor(highest, dtype=torch.int32)
    env.pots[game] = torch.tensor(pot, dtype=torch.int32)
    env.last_raise_size[game] = torch.tensor(last_raise_size, dtype=torch.int32)
    env.is_done[game] = False
    env.is_truncated[game] = False


def _capture_actor_equity(env: PokerGPU, actor_idx: int, *, game: int = 0) -> float:
    env.equities.fill_(0.5)
    env.calculate_equities()
    return float(env.equities[game, actor_idx].item())


def _expected_actor_reward(
    env: PokerGPU,
    *,
    action: int,
    actor_equity: float,
    prev_invested: int,
    game: int = 0,
) -> float:
    active_count = int(
        ((env.status[game] == env.ACTIVE) | (env.status[game] == env.ALLIN)).sum().item()
    )
    fair_share = 1.0 / max(active_count, 1)
    pot = float(env.pots[game].item())
    highest = float(env.highest[game].item())
    call_cost = max(0.0, highest - float(prev_invested))
    market_odds = call_cost / (pot + call_cost + 1e-6)

    if action == 0:
        shaped_component = (market_odds - actor_equity) * pot
    elif action == 1:
        shaped_component = (actor_equity - market_odds) * pot
    else:
        shaped_component = (actor_equity - fair_share) * pot

    magnitude = actor_equity * pot
    logits = (
        (float(env.w1.item()) * magnitude) + (float(env.w2.item()) * shaped_component)
    ) / float(env.K.item())
    return float(env.alpha.item()) * math.tanh(logits)


def test_call_reward_stays_with_actor_when_next_players_flop_equity_changes() -> None:
    weak_env = _build_env(n_players=3, seed=1)
    strong_env = _build_env(n_players=3, seed=1)
    common_kwargs = {
        "game": 0,
        "board": ["Ah", "Kh", "2c", "3d", "4s"],
        "status": [weak_env.ACTIVE, weak_env.ACTIVE, weak_env.FOLDED],
        "stacks": [100, 95, 100],
        "current_round_bet": [0, 5, 0],
        "total_invested": [0, 5, 0],
        "idx": 0,
        "agg": 1,
        "acted": 0,
        "stage": 1,
        "highest": 5,
        "pot": 10,
    }
    _configure_game(
        weak_env,
        hands=[("Qh", "Jh"), ("7d", "8s"), ("3c", "3s")],
        **common_kwargs,
    )
    _configure_game(
        strong_env,
        hands=[("Qh", "Jh"), ("Th", "9h"), ("3c", "3s")],
        **common_kwargs,
    )

    actor_equity = _capture_actor_equity(weak_env, actor_idx=0)
    weak_reward = weak_env.step(torch.tensor([1], dtype=torch.long))[1]
    strong_reward = strong_env.step(torch.tensor([1], dtype=torch.long))[1]
    expected = _expected_actor_reward(weak_env, action=1, actor_equity=actor_equity, prev_invested=0)

    assert weak_env.idx[0].item() == 1
    assert strong_env.idx[0].item() == 1
    assert weak_reward[0].item() == pytest.approx(expected, abs=1e-4)
    assert strong_reward[0].item() == pytest.approx(expected, abs=1e-4)


def test_fold_reward_ignores_next_players_turn_equity_with_all_in_side_player() -> None:
    weak_env = _build_env(n_players=3, seed=2)
    strong_env = _build_env(n_players=3, seed=2)
    common_kwargs = {
        "game": 0,
        "board": ["Ah", "Kh", "2c", "5d", "4s"],
        "status": [weak_env.ACTIVE, weak_env.ACTIVE, weak_env.ALLIN],
        "stacks": [100, 94, 0],
        "current_round_bet": [0, 6, 6],
        "total_invested": [0, 6, 18],
        "idx": 0,
        "agg": 1,
        "acted": 0,
        "stage": 2,
        "highest": 6,
        "pot": 30,
    }
    _configure_game(
        weak_env,
        hands=[("7c", "8d"), ("Qh", "Jh"), ("9s", "9c")],
        **common_kwargs,
    )
    _configure_game(
        strong_env,
        hands=[("7c", "8d"), ("Ac", "Ad"), ("9s", "9c")],
        **common_kwargs,
    )

    actor_equity = _capture_actor_equity(weak_env, actor_idx=0)
    weak_reward = weak_env.step(torch.tensor([0], dtype=torch.long))[1]
    strong_reward = strong_env.step(torch.tensor([0], dtype=torch.long))[1]
    expected = _expected_actor_reward(weak_env, action=0, actor_equity=actor_equity, prev_invested=0)

    assert weak_env.idx[0].item() == 1
    assert strong_env.idx[0].item() == 1
    assert weak_reward[0].item() == pytest.approx(expected, abs=1e-4)
    assert strong_reward[0].item() == pytest.approx(expected, abs=1e-4)


def test_raise_reward_uses_actor_equity_when_next_active_seat_skips_folded_player() -> None:
    weak_env = _build_env(n_players=4, seed=3)
    strong_env = _build_env(n_players=4, seed=3)
    common_kwargs = {
        "game": 0,
        "board": ["As", "Kd", "7h", "2c", "3d"],
        "status": [weak_env.ACTIVE, weak_env.FOLDED, weak_env.ACTIVE, weak_env.ACTIVE],
        "stacks": [100, 100, 96, 96],
        "current_round_bet": [2, 0, 4, 4],
        "total_invested": [2, 0, 4, 4],
        "idx": 0,
        "agg": 3,
        "acted": 0,
        "stage": 1,
        "highest": 4,
        "pot": 10,
        "last_raise_size": 2,
    }
    _configure_game(
        weak_env,
        hands=[("Qd", "Jh"), ("4d", "5d"), ("2s", "3s"), ("9d", "9c")],
        **common_kwargs,
    )
    _configure_game(
        strong_env,
        hands=[("Qd", "Jh"), ("4d", "5d"), ("Ac", "Kc"), ("9d", "9c")],
        **common_kwargs,
    )

    actor_equity = _capture_actor_equity(weak_env, actor_idx=0)
    weak_reward = weak_env.step(torch.tensor([2], dtype=torch.long))[1]
    strong_reward = strong_env.step(torch.tensor([2], dtype=torch.long))[1]
    expected = _expected_actor_reward(weak_env, action=2, actor_equity=actor_equity, prev_invested=2)

    assert weak_env.idx[0].item() == 2
    assert strong_env.idx[0].item() == 2
    assert weak_reward[0].item() == pytest.approx(expected, abs=1e-4)
    assert strong_reward[0].item() == pytest.approx(expected, abs=1e-4)


def test_reward_uses_previous_actor_equity_after_street_transition_reassigns_idx() -> None:
    weak_env = _build_env(n_players=2, seed=4)
    strong_env = _build_env(n_players=2, seed=4)
    common_kwargs = {
        "game": 0,
        "board": ["Ah", "Kh", "2c", "3d", "4s"],
        "status": [weak_env.ACTIVE, weak_env.ACTIVE],
        "stacks": [100, 100],
        "current_round_bet": [0, 0],
        "total_invested": [5, 5],
        "idx": 0,
        "agg": 1,
        "acted": 1,
        "stage": 1,
        "highest": 0,
        "pot": 10,
    }
    _configure_game(
        weak_env,
        hands=[("Qh", "Jh"), ("7d", "8s")],
        **common_kwargs,
    )
    _configure_game(
        strong_env,
        hands=[("Qh", "Jh"), ("Th", "9h")],
        **common_kwargs,
    )

    actor_equity = _capture_actor_equity(weak_env, actor_idx=0)
    weak_reward = weak_env.step(torch.tensor([1], dtype=torch.long))[1]
    strong_reward = strong_env.step(torch.tensor([1], dtype=torch.long))[1]
    expected = _expected_actor_reward(weak_env, action=1, actor_equity=actor_equity, prev_invested=0)

    assert weak_env.stages[0].item() == 2
    assert strong_env.stages[0].item() == 2
    assert weak_env.idx[0].item() == 1
    assert strong_env.idx[0].item() == 1
    assert weak_reward[0].item() == pytest.approx(expected, abs=1e-4)
    assert strong_reward[0].item() == pytest.approx(expected, abs=1e-4)


def test_batched_rewards_follow_each_games_actor_instead_of_post_step_cursor() -> None:
    env = _build_env(n_players=4, n_games=2, seed=5)
    _configure_game(
        env,
        game=0,
        board=["Ah", "Kh", "2c", "3d", "4s"],
        hands=[("Qh", "Jh"), ("Th", "9h"), ("3c", "3s"), ("5c", "6d")],
        status=[env.ACTIVE, env.ACTIVE, env.FOLDED, env.FOLDED],
        stacks=[100, 95, 100, 100],
        current_round_bet=[0, 5, 0, 0],
        total_invested=[0, 5, 0, 0],
        idx=0,
        agg=1,
        acted=0,
        stage=1,
        highest=5,
        pot=10,
    )
    _configure_game(
        env,
        game=1,
        board=["Qs", "Jh", "8c", "2d", "4h"],
        hands=[("Ac", "Ad"), ("7d", "7s"), ("Th", "9h"), ("2s", "2c")],
        status=[env.ACTIVE, env.FOLDED, env.ACTIVE, env.ACTIVE],
        stacks=[94, 100, 98, 94],
        current_round_bet=[6, 0, 2, 6],
        total_invested=[6, 0, 2, 6],
        idx=2,
        agg=0,
        acted=0,
        stage=2,
        highest=6,
        pot=20,
    )

    actor_equities = [
        _capture_actor_equity(env, actor_idx=0, game=0),
        _capture_actor_equity(env, actor_idx=2, game=1),
    ]
    _, rewards, _, _, info = env.step(torch.tensor([1, 1], dtype=torch.long))

    expected_game0 = _expected_actor_reward(env, action=1, actor_equity=actor_equities[0], prev_invested=0, game=0)
    expected_game1 = _expected_actor_reward(env, action=1, actor_equity=actor_equities[1], prev_invested=2, game=1)

    assert info["seat_idx"].tolist() == [1, 3]
    assert rewards[0].item() == pytest.approx(expected_game0, abs=1e-4)
    assert rewards[1].item() == pytest.approx(expected_game1, abs=1e-4)
