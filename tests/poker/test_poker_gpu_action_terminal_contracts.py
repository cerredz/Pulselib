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


def test_execute_actions_call_uses_remaining_stack_and_marks_allin() -> None:
    env = _build_env(n_players=2)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.highest[0] = torch.tensor(10, dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([4, 10], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([4, 10], dtype=torch.int32)
    env.stacks[0] = torch.tensor([3, 50], dtype=torch.int32)
    env.pots[0] = torch.tensor(14, dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.acted[0] = torch.tensor(0, dtype=torch.int32)

    env.execute_actions(torch.tensor([1], dtype=torch.long))

    assert env.stacks[0].tolist() == [0, 50]
    assert env.current_round_bet[0].tolist() == [7, 10]
    assert env.total_invested[0].tolist() == [7, 10]
    assert env.pots[0].item() == 17
    assert env.status[0, 0].item() == env.ALLIN
    assert env.acted[0].item() == 1


def test_execute_actions_min_raise_reopens_action_and_updates_raise_size() -> None:
    env = _build_env(n_players=3)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.agg[0] = torch.tensor(2, dtype=torch.int32)
    env.acted[0] = torch.tensor(2, dtype=torch.int32)
    env.highest[0] = torch.tensor(10, dtype=torch.int32)
    env.last_raise_size[0] = torch.tensor(4, dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([6, 0, 10], dtype=torch.int32)
    env.total_invested[0] = env.current_round_bet[0].clone()
    env.stacks[0] = torch.tensor([50, 50, 50], dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.pots[0] = torch.tensor(16, dtype=torch.int32)

    env.execute_actions(torch.tensor([2], dtype=torch.long))

    assert env.current_round_bet[0].tolist() == [14, 0, 10]
    assert env.total_invested[0].tolist() == [14, 0, 10]
    assert env.pots[0].item() == 24
    assert env.stacks[0].tolist() == [42, 50, 50]
    assert env.highest[0].item() == 14
    assert env.agg[0].item() == 0
    assert env.last_raise_size[0].item() == 4
    assert env.acted[0].item() == 1


def test_resolve_fold_winners_is_idempotent_after_pot_is_cleared() -> None:
    env = _build_env(n_players=3)
    env.status[0] = torch.tensor([env.FOLDED, env.ACTIVE, env.FOLDED], dtype=torch.int32)
    env.stacks[0] = torch.tensor([10, 20, 30], dtype=torch.int32)
    env.pots[0] = torch.tensor(15, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_fold_winners()
    first_payout = env.stacks[0].tolist()
    env.resolve_fold_winners()

    assert first_payout == [10, 35, 30]
    assert env.stacks[0].tolist() == first_payout
    assert env.pots[0].item() == 0


def test_resolve_terminated_games_noops_when_no_done_rows_need_resolution() -> None:
    env = _build_env(n_players=2)
    before_board = env.board.clone()
    before_stacks = env.stacks.clone()
    before_pots = env.pots.clone()
    before_stages = env.stages.clone()
    before_positions = env.deck_positions.clone()

    env.resolve_terminated_games()

    assert torch.equal(env.board, before_board)
    assert torch.equal(env.stacks, before_stacks)
    assert torch.equal(env.pots, before_pots)
    assert torch.equal(env.stages, before_stages)
    assert torch.equal(env.deck_positions, before_positions)


def test_resolve_terminated_games_turn_runout_preserves_existing_board_and_burns_once() -> None:
    env = _build_env(n_players=2)
    env.decks[0] = torch.arange(1, 53, dtype=torch.int32)
    env.deck_positions[0] = torch.tensor(10, dtype=torch.int32)
    env.board[0] = torch.tensor([11, 22, 33, 44, -1], dtype=torch.int32)
    env.hands[0, 0] = torch.tensor([1, 2], dtype=torch.int32)
    env.hands[0, 1] = torch.tensor([3, 4], dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.stacks[0] = torch.tensor([100, 100], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([10, 10], dtype=torch.int32)
    env.pots[0] = torch.tensor(20, dtype=torch.int32)
    env.stages[0] = torch.tensor(2, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.board[0].tolist()[0:4] == [11, 22, 33, 44]
    assert env.board[0, 4].item() == 12
    assert env.deck_positions[0].item() == 12
    assert env.pots[0].item() == 0
    assert env.stages[0].item() == 5
    assert int(env.stacks[0].sum().item()) == 220


def test_resolve_terminated_games_flop_runout_preserves_flop_and_deals_turn_then_river() -> None:
    env = _build_env(n_players=2)
    env.decks[0] = torch.arange(1, 53, dtype=torch.int32)
    env.deck_positions[0] = torch.tensor(10, dtype=torch.int32)
    env.board[0] = torch.tensor([11, 22, 33, -1, -1], dtype=torch.int32)
    env.hands[0, 0] = torch.tensor([1, 2], dtype=torch.int32)
    env.hands[0, 1] = torch.tensor([3, 4], dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.stacks[0] = torch.tensor([100, 100], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([10, 10], dtype=torch.int32)
    env.pots[0] = torch.tensor(20, dtype=torch.int32)
    env.stages[0] = torch.tensor(1, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.board[0].tolist() == [11, 22, 33, 12, 14]
    assert env.deck_positions[0].item() == 14
    assert env.pots[0].item() == 0
    assert env.stages[0].item() == 5
    assert int(env.stacks[0].sum().item()) == 220


def test_calculate_equities_leaves_clean_rows_untouched() -> None:
    env = _build_env(n_players=2, n_games=2)
    env.equities[0] = torch.tensor([0.2, 0.8], dtype=torch.float32)
    env.equities[1] = torch.tensor([0.7, 0.3], dtype=torch.float32)
    env.equity_dirty[:] = torch.tensor([False, True], dtype=torch.bool)
    env.stages[:] = torch.tensor([0, 0], dtype=torch.int32)

    env.calculate_equities()

    assert env.equities[0].tolist() == pytest.approx([0.2, 0.8])
    assert env.equities[1].tolist() == pytest.approx([0.5, 0.5])
    assert env.equity_dirty.tolist() == [False, False]


def test_step_clears_round_state_after_fold_ends_hand() -> None:
    env = _build_env(n_players=2)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.agg[0] = torch.tensor(1, dtype=torch.int32)
    env.acted[0] = torch.tensor(0, dtype=torch.int32)
    env.highest[0] = torch.tensor(10, dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([5, 10], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([5, 10], dtype=torch.int32)
    env.pots[0] = torch.tensor(15, dtype=torch.int32)
    env.stacks[0] = torch.tensor([45, 40], dtype=torch.int32)
    env.is_done[0] = False

    _, _, dones, _, _ = env.step(torch.tensor([0], dtype=torch.long))

    assert dones[0].item()
    assert env.pots[0].item() == 0
    assert env.current_round_bet[0].tolist() == [0, 0]
    assert env.total_invested[0].tolist() == [0, 0]
    assert env.highest[0].item() == 0
    assert env.stacks[0].tolist() == [45, 55]


def test_step_clears_round_state_after_river_showdown() -> None:
    env = _build_env(n_players=2)
    env.board[0] = torch.tensor([_encode(card) for card in ["2c", "7d", "9h", "Js", "Kd"]], dtype=torch.int32)
    env.hands[0, 0] = torch.tensor([_encode("Ah"), _encode("Qh")], dtype=torch.int32)
    env.hands[0, 1] = torch.tensor([_encode("3c"), _encode("4d")], dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.agg[0] = torch.tensor(1, dtype=torch.int32)
    env.acted[0] = torch.tensor(1, dtype=torch.int32)
    env.highest[0] = torch.tensor(0, dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([0, 0], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([10, 10], dtype=torch.int32)
    env.pots[0] = torch.tensor(20, dtype=torch.int32)
    env.stacks[0] = torch.tensor([50, 50], dtype=torch.int32)
    env.stages[0] = torch.tensor(3, dtype=torch.int32)
    env.is_done[0] = False
    env.equity_dirty[0] = False
    env.equities[0] = torch.tensor([0.8, 0.2], dtype=torch.float32)

    _, _, dones, _, _ = env.step(torch.tensor([1], dtype=torch.long))

    assert dones[0].item()
    assert env.pots[0].item() == 0
    assert env.stages[0].item() == 5
    assert env.current_round_bet[0].tolist() == [0, 0]
    assert env.total_invested[0].tolist() == [0, 0]
    assert env.highest[0].item() == 0
    assert env.stacks[0].tolist() == [70, 50]
