import eval7
import pytest
import torch
from unittest.mock import patch

from environments.Poker.PokerGPU import PokerGPU


def _encode(card_str: str) -> int:
    card = eval7.Card(card_str)
    return card.rank + (13 * card.suit) + 1


def _build_env(*, n_players: int = 6, max_players: int | None = None, n_games: int = 1) -> PokerGPU:
    max_players = max_players or n_players
    env = PokerGPU(
        device=torch.device("cpu"),
        agents=[],
        n_players=n_players,
        max_players=max_players,
        n_games=n_games,
    )
    return env


def test_reset_accepts_none_options_and_initializes_full_ring() -> None:
    env = _build_env(n_players=4)

    obs, info = env.reset()

    assert obs.shape == (1, env.obs_size)
    assert info["active_players"] == 4
    assert env.active_players == 4
    assert env.deck_positions.tolist() == [8]
    assert env.button.tolist() == [0]
    assert env.sb.tolist() == [1]
    assert env.bb.tolist() == [2]
    assert env.idx.tolist() == [3]
    assert env.status[0].tolist() == [env.ACTIVE, env.ACTIVE, env.ACTIVE, env.ACTIVE]


def test_reset_accepts_partial_options_without_active_players_key() -> None:
    env = _build_env(n_players=4)

    obs, info = env.reset(options={"rotation": 0})

    assert obs.shape == (1, env.obs_size)
    assert info["active_players"] == 4
    assert env.active_players == 4
    assert env.status[0].tolist() == [env.ACTIVE, env.ACTIVE, env.ACTIVE, env.ACTIVE]


def test_reset_random_active_players_respects_q_seat_floor() -> None:
    env = _build_env(n_players=6, max_players=6)

    def fake_randint(low, high, size, device=None):
        return torch.tensor([2], device=device)

    with patch("torch.randint", fake_randint):
        env.reset(options={"active_players": True, "q_agent_seat": 4, "rotation": 0})

    assert env.active_players == 5
    assert env.status[0].tolist() == [env.ACTIVE, env.ACTIVE, env.ACTIVE, env.ACTIVE, env.ACTIVE, env.SITOUT]
    assert env.hands[0, 5].tolist() == [-1, -1]


def test_reset_marks_inactive_seats_sitout_and_unused_hands_negative_one() -> None:
    env = _build_env(n_players=6, max_players=6)

    def fake_randint(low, high, size, device=None):
        return torch.tensor([4], device=device)

    with patch("torch.randint", fake_randint):
        env.reset(options={"active_players": True, "q_agent_seat": 0, "rotation": 0})

    assert env.active_players == 4
    assert env.status[0].tolist() == [env.ACTIVE, env.ACTIVE, env.ACTIVE, env.ACTIVE, env.SITOUT, env.SITOUT]
    assert env.hands[0, 4:].tolist() == [[-1, -1], [-1, -1]]


def test_reset_decks_are_permutations_and_active_hole_cards_are_unique_per_game() -> None:
    env = _build_env(n_players=6, n_games=3)
    env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})

    expected_deck = list(range(1, 53))
    for game in range(env.n_games):
        assert sorted(env.decks[game].tolist()) == expected_deck
        dealt_cards = env.hands[game, :env.active_players].reshape(-1).tolist()
        assert len(dealt_cards) == len(set(dealt_cards))


def test_second_reset_advances_button_and_recomputes_blinds_and_first_actor() -> None:
    env = _build_env(n_players=4)
    env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})

    env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})

    assert env.button.tolist() == [1]
    assert env.sb.tolist() == [2]
    assert env.bb.tolist() == [3]
    assert env.idx.tolist() == [0]


def test_resolve_terminated_games_returns_extra_chip_to_only_eligible_contributor_in_tied_main_pot() -> None:
    env = _build_env(n_players=2)
    env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})
    env.board[0] = torch.tensor([_encode(card) for card in ["Ah", "Kd", "Qc", "Js", "9d"]], dtype=torch.int32)
    env.hands[0, 0] = torch.tensor([_encode("2c"), _encode("3d")], dtype=torch.int32)
    env.hands[0, 1] = torch.tensor([_encode("2d"), _encode("3c")], dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.stacks[0] = torch.tensor([10, 20], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([12, 13], dtype=torch.int32)
    env.pots[0] = torch.tensor(25, dtype=torch.int32)
    env.stages[0] = torch.tensor(4, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.stacks[0].tolist() == [22, 33]
    assert env.pots[0].item() == 0
    assert env.stages[0].item() == 5


def test_resolve_terminated_games_done_single_survivor_row_is_left_for_fold_resolution() -> None:
    env = _build_env(n_players=2)
    env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})
    env.status[0] = torch.tensor([env.ACTIVE, env.FOLDED], dtype=torch.int32)
    env.stacks[0] = torch.tensor([50, 60], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([10, 10], dtype=torch.int32)
    env.pots[0] = torch.tensor(20, dtype=torch.int32)
    env.stages[0] = torch.tensor(2, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.stacks[0].tolist() == [50, 60]
    assert env.pots[0].item() == 20
    assert env.stages[0].item() == 2


def test_step_mixed_batch_keeps_each_row_isolated() -> None:
    env = _build_env(n_players=4, n_games=4)
    env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})

    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE, env.FOLDED, env.FOLDED], dtype=torch.int32)
    env.stacks[0] = torch.tensor([45, 40, 100, 100], dtype=torch.int32)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.agg[0] = torch.tensor(1, dtype=torch.int32)
    env.acted[0] = torch.tensor(0, dtype=torch.int32)
    env.highest[0] = torch.tensor(10, dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([5, 10, 0, 0], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([5, 10, 0, 0], dtype=torch.int32)
    env.pots[0] = torch.tensor(15, dtype=torch.int32)
    env.is_done[0] = False

    env.status[1] = torch.tensor([env.ACTIVE, env.FOLDED, env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.idx[1] = torch.tensor(0, dtype=torch.int32)
    env.agg[1] = torch.tensor(2, dtype=torch.int32)
    env.acted[1] = torch.tensor(2, dtype=torch.int32)
    env.highest[1] = torch.tensor(0, dtype=torch.int32)
    env.current_round_bet[1] = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    env.total_invested[1] = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    env.pots[1] = torch.tensor(0, dtype=torch.int32)
    env.stages[1] = torch.tensor(0, dtype=torch.int32)
    env.is_done[1] = False

    env.status[2] = torch.tensor([env.ALLIN, env.ALLIN, env.FOLDED, env.FOLDED], dtype=torch.int32)
    env.stacks[2] = torch.tensor([90, 90, 100, 100], dtype=torch.int32)
    env.idx[2] = torch.tensor(0, dtype=torch.int32)
    env.agg[2] = torch.tensor(0, dtype=torch.int32)
    env.acted[2] = torch.tensor(0, dtype=torch.int32)
    env.highest[2] = torch.tensor(10, dtype=torch.int32)
    env.current_round_bet[2] = torch.tensor([10, 10, 0, 0], dtype=torch.int32)
    env.total_invested[2] = torch.tensor([10, 10, 0, 0], dtype=torch.int32)
    env.pots[2] = torch.tensor(20, dtype=torch.int32)
    env.stages[2] = torch.tensor(0, dtype=torch.int32)
    env.is_done[2] = False

    env.status[3] = torch.tensor([env.ACTIVE, env.ACTIVE, env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.stacks[3] = torch.tensor([70, 80, 90, 100], dtype=torch.int32)
    env.idx[3] = torch.tensor(2, dtype=torch.int32)
    env.agg[3] = torch.tensor(1, dtype=torch.int32)
    env.acted[3] = torch.tensor(1, dtype=torch.int32)
    env.highest[3] = torch.tensor(5, dtype=torch.int32)
    env.current_round_bet[3] = torch.tensor([5, 5, 5, 5], dtype=torch.int32)
    env.total_invested[3] = torch.tensor([5, 5, 5, 5], dtype=torch.int32)
    env.pots[3] = torch.tensor(20, dtype=torch.int32)
    env.stages[3] = torch.tensor(2, dtype=torch.int32)
    env.is_done[3] = True

    before_done_row = {
        "status": env.status[3].clone(),
        "stacks": env.stacks[3].clone(),
        "idx": env.idx[3].clone(),
        "pots": env.pots[3].clone(),
        "stages": env.stages[3].clone(),
    }

    _, rewards, dones, _, info = env.step(torch.tensor([0, 1, 12, 12], dtype=torch.long))

    assert dones[0].item()
    assert env.stacks[0].tolist() == [45, 55, 100, 100]
    assert env.pots[0].item() == 0

    assert not dones[1].item()
    assert env.stages[1].item() == 1
    assert env.idx[1].item() == 2
    assert torch.all(env.board[1, 0:3] > 0)

    assert not dones[2].item()
    assert env.stages[2].item() == 1
    assert rewards[2].item() == pytest.approx(0.0)
    assert torch.all(env.board[2, 0:3] > 0)

    assert dones[3].item()
    assert torch.equal(env.status[3], before_done_row["status"])
    assert torch.equal(env.stacks[3], before_done_row["stacks"])
    assert torch.equal(env.idx[3], before_done_row["idx"])
    assert torch.equal(env.pots[3], before_done_row["pots"])
    assert torch.equal(env.stages[3], before_done_row["stages"])
    assert info["seat_idx"][1].item() == 2
    assert info["seat_idx"][3].item() == before_done_row["idx"].item()
