import random

import numpy as np
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


def _set_round_end_state(
    env: PokerGPU,
    *,
    game: int,
    status: list[int],
    idx: int,
    agg: int,
    acted: int,
    stage: int,
) -> None:
    env.status[game] = torch.tensor(status, dtype=torch.int32)
    env.idx[game] = torch.tensor(idx, dtype=torch.int32)
    env.agg[game] = torch.tensor(agg, dtype=torch.int32)
    env.acted[game] = torch.tensor(acted, dtype=torch.int32)
    env.highest[game] = torch.tensor(0, dtype=torch.int32)
    env.stages[game] = torch.tensor(stage, dtype=torch.int32)
    env.current_round_bet[game] = torch.zeros(env.n_players, dtype=torch.int32)
    env.total_invested[game] = torch.zeros(env.n_players, dtype=torch.int32)
    env.is_done[game] = False
    env.pots[game] = torch.tensor(0, dtype=torch.int32)


def test_flop_transition_restarts_action_from_first_seat_left_of_button() -> None:
    env = _build_env(n_players=4)
    _set_round_end_state(
        env,
        game=0,
        status=[env.ACTIVE, env.ACTIVE, env.ACTIVE, env.ACTIVE],
        idx=0,
        agg=1,
        acted=3,
        stage=0,
    )

    _, _, dones, _, info = env.step(torch.tensor([1], dtype=torch.long))

    assert not dones[0].item()
    assert env.stages[0].item() == 1
    assert env.idx[0].item() == 1
    assert info["seat_idx"][0].item() == 1


def test_turn_transition_skips_folded_seats_left_of_button() -> None:
    env = _build_env(n_players=4)
    _set_round_end_state(
        env,
        game=0,
        status=[env.ACTIVE, env.FOLDED, env.ACTIVE, env.ACTIVE],
        idx=0,
        agg=2,
        acted=2,
        stage=1,
    )

    env.step(torch.tensor([1], dtype=torch.long))

    assert env.stages[0].item() == 2
    assert env.idx[0].item() == 2
    assert env.agg[0].item() == 1


def test_river_transition_skips_all_in_seats_left_of_button() -> None:
    env = _build_env(n_players=4)
    _set_round_end_state(
        env,
        game=0,
        status=[env.ACTIVE, env.ALLIN, env.ACTIVE, env.ACTIVE],
        idx=0,
        agg=2,
        acted=2,
        stage=2,
    )

    env.step(torch.tensor([1], dtype=torch.long))

    assert env.stages[0].item() == 3
    assert env.idx[0].item() == 2
    assert env.board[0, 4].item() > 0


def test_heads_up_transition_uses_only_other_active_seat() -> None:
    env = _build_env(n_players=2)
    _set_round_end_state(
        env,
        game=0,
        status=[env.ACTIVE, env.ACTIVE],
        idx=0,
        agg=1,
        acted=1,
        stage=1,
    )

    _, _, dones, _, info = env.step(torch.tensor([1], dtype=torch.long))

    assert not dones[0].item()
    assert env.stages[0].item() == 2
    assert env.idx[0].item() == 1
    assert info["seat_idx"][0].item() == 1


def test_batched_transition_resets_only_finished_rounds_and_updates_observation() -> None:
    env = _build_env(n_players=4, n_games=2)
    _set_round_end_state(
        env,
        game=0,
        status=[env.ACTIVE, env.FOLDED, env.ACTIVE, env.ACTIVE],
        idx=0,
        agg=2,
        acted=2,
        stage=0,
    )
    _set_round_end_state(
        env,
        game=1,
        status=[env.ACTIVE, env.FOLDED, env.ALLIN, env.ACTIVE],
        idx=0,
        agg=0,
        acted=0,
        stage=0,
    )

    obs, _, dones, _, info = env.step(torch.tensor([1, 1], dtype=torch.long))

    assert not dones[0].item()
    assert not dones[1].item()
    assert env.stages.tolist() == [1, 0]
    assert env.idx.tolist() == [2, 3]
    assert info["seat_idx"].tolist() == [2, 3]
    assert obs[0, 5:7].to(torch.int32).tolist() == env.hands[0, 2].tolist()
    assert obs[1, 5:7].to(torch.int32).tolist() == env.hands[1, 3].tolist()
