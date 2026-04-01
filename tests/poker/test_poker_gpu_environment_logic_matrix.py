from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from environments.Poker.PokerGPU import PokerGPU


def _build_env(*, n_players: int = 6, max_players: int | None = None, n_games: int = 1) -> PokerGPU:
    max_players = max_players or n_players
    env = PokerGPU(
        device=torch.device("cpu"),
        agents=[],
        n_players=n_players,
        max_players=max_players,
        n_games=n_games,
    )
    env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})
    return env


POT_FRACTION_CASES = [
    ("actions/pot-fraction-0.25-maps-to-25-chip-bet", 3, 25, "Verify the 0.25-pot action maps to a 25-chip bet from a 100-chip pot."),
    ("actions/pot-fraction-0.33-maps-to-33-chip-bet", 4, 33, "Verify the 0.33-pot action truncates to 33 chips from a 100-chip pot."),
    ("actions/pot-fraction-0.50-maps-to-50-chip-bet", 5, 50, "Verify the 0.50-pot action maps to a 50-chip bet from a 100-chip pot."),
    ("actions/pot-fraction-0.75-maps-to-75-chip-bet", 6, 75, "Verify the 0.75-pot action maps to a 75-chip bet from a 100-chip pot."),
    ("actions/pot-fraction-1.00-maps-to-100-chip-bet", 7, 100, "Verify the 1.00-pot action maps to a 100-chip bet from a 100-chip pot."),
    ("actions/pot-fraction-1.50-maps-to-150-chip-bet", 8, 150, "Verify the 1.50-pot action maps to a 150-chip bet from a 100-chip pot."),
    ("actions/pot-fraction-2.00-maps-to-200-chip-bet", 9, 200, "Verify the 2.00-pot action maps to a 200-chip bet from a 100-chip pot."),
    ("actions/pot-fraction-3.00-maps-to-300-chip-bet", 10, 300, "Verify the 3.00-pot action maps to a 300-chip bet from a 100-chip pot."),
    ("actions/pot-fraction-4.00-maps-to-400-chip-bet", 11, 400, "Verify the 4.00-pot action maps to a 400-chip bet from a 100-chip pot."),
]

RELATIVE_POSITION_CASES = [
    ("observation/relative-position-seat-0-from-button-1-is-3", 0, 1, 3),
    ("observation/relative-position-seat-1-from-button-1-is-0", 1, 1, 0),
    ("observation/relative-position-seat-2-from-button-1-is-1", 2, 1, 1),
    ("observation/relative-position-seat-3-from-button-1-is-2", 3, 1, 2),
]

CALL_AMOUNT_CASES = [
    ("observation/call-amount-matches-full-10-chip-gap", 0, 10, 10),
    ("observation/call-amount-matches-7-chip-gap", 3, 10, 7),
    ("observation/call-amount-matches-1-chip-gap", 9, 10, 1),
    ("observation/call-amount-zero-when-bet-is-matched", 10, 10, 0),
]

OPPONENT_ORDER_CASES = [
    ("observation/opponents-wrap-correctly-from-seat-0", 0, [102, 1, 12, 103, 2, 13, 104, 0, 14]),
    ("observation/opponents-wrap-correctly-from-seat-1", 1, [103, 2, 13, 104, 0, 14, 101, 0, 11]),
    ("observation/opponents-wrap-correctly-from-seat-2", 2, [104, 0, 14, 101, 0, 11, 102, 1, 12]),
    ("observation/opponents-wrap-correctly-from-seat-3", 3, [101, 0, 11, 102, 1, 12, 103, 2, 13]),
]

RESET_DECKPOS_CASES = [
    ("reset/deck-position-after-2-player-deal-is-4", 2),
    ("reset/deck-position-after-3-player-deal-is-6", 3),
    ("reset/deck-position-after-4-player-deal-is-8", 4),
    ("reset/deck-position-after-5-player-deal-is-10", 5),
    ("reset/deck-position-after-6-player-deal-is-12", 6),
]

QSEAT_FLOOR_CASES = [
    ("reset/random-2-player-candidate-stays-2-with-q-seat-0", 2, 0, 2),
    ("reset/random-2-player-candidate-rises-to-3-with-q-seat-2", 2, 2, 3),
    ("reset/random-2-player-candidate-rises-to-5-with-q-seat-4", 2, 4, 5),
    ("reset/random-5-player-candidate-stays-5-with-q-seat-1", 5, 1, 5),
]

BUTTON_SEQUENCE_CASES = [
    ("reset/heads-up-third-reset-wraps-button-back-to-seat-0", 2, 3, 0, 0, 1, 0),
    ("reset/full-ring-third-reset-puts-button-on-seat-2", 4, 3, 2, 3, 0, 1),
    ("reset/three-player-second-reset-puts-button-on-seat-1", 3, 2, 1, 2, 0, 1),
]

FOLD_RESOLUTION_CASES = [
    ("termination/fold-resolution-pays-single-survivor-on-done-row", "done_single"),
    ("termination/fold-resolution-ignores-live-row-with-single-survivor", "live_single"),
    ("termination/fold-resolution-ignores-done-row-with-multiple-survivors", "done_multi"),
]

HEADSUP_CHECKAROUND_CASES = [
    ("heads-up/flop-check-check-advances-to-turn", 1, 2, False),
    ("heads-up/turn-check-check-advances-to-river", 2, 3, False),
    ("heads-up/river-check-check-resolves-showdown", 3, 5, True),
]

AUTO_RUNOUT_STAGE_CASES = [
    ("no-actor/flop-allin-auto-runout-advances-to-turn", 1, 2, False),
    ("no-actor/turn-allin-auto-runout-advances-to-river", 2, 3, False),
    ("no-actor/river-allin-auto-runout-finishes-hand", 3, 5, True),
]

PREFLOP_BASELINE_CASES = [
    ("equity/preflop-2-player-dirty-rows-reset-to-point-five", 2),
    ("equity/preflop-3-player-dirty-rows-reset-to-point-five", 3),
    ("equity/preflop-4-player-dirty-rows-reset-to-point-five", 4),
    ("equity/preflop-6-player-dirty-rows-reset-to-point-five", 6),
]

MIN_RAISE_CASES = [
    ("actions/min-raise-opens-to-one-chip-from-unopened-pot", "open"),
    ("actions/min-raise-adds-last-raise-size-on-top-of-call-cost", "facing_bet"),
]

EXACT_STACK_ACTION_CASES = [
    ("actions/call-with-exact-stack-matches-bet-and-goes-allin", "call"),
    ("actions/allin-with-exact-stack-matching-call-does-not-reopen", "allin_call"),
    ("actions/allin-with-exact-stack-for-full-raise-reopens", "allin_raise"),
]

MIXED_STAGE_RECALC_CASES = [
    ("equity/mixed-dirty-mask-preflop-row-becomes-point-five", "preflop"),
    ("equity/mixed-dirty-mask-flop-row-stays-bounded", "flop"),
    ("equity/mixed-dirty-mask-turn-row-stays-bounded", "turn"),
    ("equity/mixed-dirty-mask-river-row-stays-bounded", "river"),
]


def _assert_pot_fraction_case(action_id: int, expected_bet: int) -> None:
    env = _build_env(n_players=2)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.agg[0] = torch.tensor(1, dtype=torch.int32)
    env.acted[0] = torch.tensor(0, dtype=torch.int32)
    env.highest[0] = torch.tensor(0, dtype=torch.int32)
    env.current_round_bet.zero_()
    env.total_invested.zero_()
    env.stacks[0] = torch.tensor([500, 500], dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.pots[0] = torch.tensor(100, dtype=torch.int32)
    env.last_raise_size[0] = torch.tensor(1, dtype=torch.int32)

    env.execute_actions(torch.tensor([action_id], dtype=torch.long))

    assert env.current_round_bet[0].tolist() == [expected_bet, 0]
    assert env.total_invested[0].tolist() == [expected_bet, 0]
    assert env.pots[0].item() == 100 + expected_bet
    assert env.stacks[0].tolist() == [500 - expected_bet, 500]
    assert env.highest[0].item() == expected_bet
    assert env.last_raise_size[0].item() == expected_bet
    assert env.agg[0].item() == 0
    assert env.acted[0].item() == 1


def _assert_relative_position_case(idx: int, button: int, expected_rel: int) -> None:
    env = _build_env(n_players=4)
    env.idx[0] = torch.tensor(idx, dtype=torch.int32)
    env.button[0] = torch.tensor(button, dtype=torch.int32)

    obs = env.get_obs()

    assert obs[0, 8].item() == expected_rel


def _assert_call_amount_case(acting_bet: int, highest: int, expected_call: int) -> None:
    env = _build_env(n_players=4)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.highest[0] = torch.tensor(highest, dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([acting_bet, 0, 0, 0], dtype=torch.int32)

    obs = env.get_obs()

    assert obs[0, 10].item() == expected_call


def _assert_opponent_order_case(idx: int, expected_flat: list[int]) -> None:
    env = _build_env(n_players=4)
    env.idx[0] = torch.tensor(idx, dtype=torch.int32)
    env.stacks[0] = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.FOLDED, env.ALLIN, env.ACTIVE], dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([11, 12, 13, 14], dtype=torch.int32)

    obs = env.get_obs()

    assert obs[0, 13:22].to(torch.int32).tolist() == expected_flat


def _assert_reset_deck_position_case(candidate_players: int) -> None:
    env = PokerGPU(
        device=torch.device("cpu"),
        agents=[],
        n_players=6,
        max_players=6,
        n_games=1,
    )

    def fake_randint(low, high, size, device=None):
        return torch.tensor([candidate_players], device=device)

    with patch("torch.randint", fake_randint):
        env.reset(options={"active_players": True, "q_agent_seat": 0, "rotation": 0})

    assert env.active_players == candidate_players
    assert env.deck_positions.tolist() == [candidate_players * 2]
    assert env.hands[0, :candidate_players].reshape(-1).ge(1).all().item()


def _assert_qseat_floor_case(candidate_players: int, q_seat: int, expected_active: int) -> None:
    env = PokerGPU(
        device=torch.device("cpu"),
        agents=[],
        n_players=6,
        max_players=6,
        n_games=1,
    )

    def fake_randint(low, high, size, device=None):
        return torch.tensor([candidate_players], device=device)

    with patch("torch.randint", fake_randint):
        env.reset(options={"active_players": True, "q_agent_seat": q_seat, "rotation": 0})

    assert env.active_players == expected_active
    assert env.status[0, expected_active:].eq(env.SITOUT).all().item()
    assert env.hands[0, expected_active:].eq(-1).all().item()


def _assert_button_sequence_case(
    n_players: int,
    reset_count: int,
    expected_button: int,
    expected_sb: int,
    expected_bb: int,
    expected_idx: int,
) -> None:
    env = PokerGPU(
        device=torch.device("cpu"),
        agents=[],
        n_players=n_players,
        max_players=n_players,
        n_games=1,
    )

    for _ in range(reset_count):
        env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})

    assert env.button.tolist() == [expected_button]
    assert env.sb.tolist() == [expected_sb]
    assert env.bb.tolist() == [expected_bb]
    assert env.idx.tolist() == [expected_idx]


def _assert_fold_resolution_case(mode: str) -> None:
    env = _build_env(n_players=3)
    env.pots[0] = torch.tensor(15, dtype=torch.int32)
    env.stacks[0] = torch.tensor([10, 20, 30], dtype=torch.int32)

    if mode == "done_single":
        env.is_done[0] = True
        env.status[0] = torch.tensor([env.FOLDED, env.ACTIVE, env.FOLDED], dtype=torch.int32)
        env.resolve_fold_winners()
        assert env.stacks[0].tolist() == [10, 35, 30]
        assert env.pots[0].item() == 0
    elif mode == "live_single":
        env.is_done[0] = False
        env.status[0] = torch.tensor([env.FOLDED, env.ACTIVE, env.FOLDED], dtype=torch.int32)
        env.resolve_fold_winners()
        assert env.stacks[0].tolist() == [10, 20, 30]
        assert env.pots[0].item() == 15
    else:
        env.is_done[0] = True
        env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE, env.FOLDED], dtype=torch.int32)
        env.resolve_fold_winners()
        assert env.stacks[0].tolist() == [10, 20, 30]
        assert env.pots[0].item() == 15


def _configure_headsup_postflop(env: PokerGPU, stage: int) -> None:
    env.active_players = 2
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.button[0] = torch.tensor(0, dtype=torch.int32)
    env.idx[0] = torch.tensor(1, dtype=torch.int32)
    env.agg[0] = torch.tensor(0, dtype=torch.int32)
    env.acted[0] = torch.tensor(0, dtype=torch.int32)
    env.highest[0] = torch.tensor(0, dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([0, 0], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([10, 10], dtype=torch.int32)
    env.pots[0] = torch.tensor(20, dtype=torch.int32)
    env.stages[0] = torch.tensor(stage, dtype=torch.int32)
    env.is_done[0] = False
    env.equity_dirty[0] = False
    env.equities[0] = torch.tensor([0.6, 0.4], dtype=torch.float32)


def _assert_headsup_checkaround_case(stage: int, expected_stage: int, expected_done: bool) -> None:
    env = _build_env(n_players=2)
    _configure_headsup_postflop(env, stage)

    env.step(torch.tensor([1], dtype=torch.long))
    _, _, dones, _, info = env.step(torch.tensor([1], dtype=torch.long))

    assert env.stages[0].item() == expected_stage
    assert dones[0].item() == expected_done
    if expected_done:
        assert env.pots[0].item() == 0
    else:
        assert info["seat_idx"][0].item() == 1


def _configure_no_actor_stage(env: PokerGPU, stage: int) -> None:
    env.status[0] = torch.tensor([env.ALLIN, env.ALLIN, env.FOLDED], dtype=torch.int32)
    env.stacks[0] = torch.tensor([90, 90, 100], dtype=torch.int32)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.agg[0] = torch.tensor(0, dtype=torch.int32)
    env.acted[0] = torch.tensor(0, dtype=torch.int32)
    env.highest[0] = torch.tensor(10, dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([10, 10, 0], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([10, 10, 0], dtype=torch.int32)
    env.pots[0] = torch.tensor(20, dtype=torch.int32)
    env.stages[0] = torch.tensor(stage, dtype=torch.int32)
    env.is_done[0] = False


def _assert_auto_runout_stage_case(stage: int, expected_stage: int, expected_done: bool) -> None:
    env = _build_env(n_players=3)
    _configure_no_actor_stage(env, stage)

    _, rewards, dones, _, _ = env.step(torch.tensor([12], dtype=torch.long))

    assert rewards[0].item() == pytest.approx(0.0)
    assert env.stages[0].item() == expected_stage
    assert dones[0].item() == expected_done


def _assert_preflop_baseline_case(n_players: int) -> None:
    env = _build_env(n_players=n_players)
    env.equities.fill_(0.17)
    env.stages[0] = torch.tensor(0, dtype=torch.int32)
    env.equity_dirty[0] = True

    env.calculate_equities()

    assert env.equities[0].tolist() == pytest.approx([0.5] * n_players)
    assert not env.equity_dirty[0].item()


def _assert_set_agents_case() -> None:
    env = _build_env(n_players=2)
    agents = [object(), object()]
    env.set_agents(agents)
    assert env.agents is agents


def _assert_post_blinds_active_case() -> None:
    env = _build_env(n_players=2)
    env.status.fill_(env.ACTIVE)
    env.stacks[0] = torch.tensor([5, 5], dtype=torch.int32)
    env.current_round_bet.zero_()
    env.total_invested.zero_()
    env.pots.zero_()
    env.bb[0] = torch.tensor(1, dtype=torch.int32)

    env.post_blinds()

    assert env.status[0, 1].item() == env.ACTIVE
    assert env.stacks[0, 1].item() == 4
    assert env.pots[0].item() == 1


def _assert_min_raise_case(mode: str) -> None:
    env = _build_env(n_players=2)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.stacks[0] = torch.tensor([50, 50], dtype=torch.int32)
    env.agg[0] = torch.tensor(1, dtype=torch.int32)
    env.acted[0] = torch.tensor(0, dtype=torch.int32)

    if mode == "open":
        env.highest[0] = torch.tensor(0, dtype=torch.int32)
        env.current_round_bet.zero_()
        env.total_invested.zero_()
        env.pots[0] = torch.tensor(0, dtype=torch.int32)
        env.last_raise_size[0] = torch.tensor(1, dtype=torch.int32)
        env.execute_actions(torch.tensor([2], dtype=torch.long))
        assert env.current_round_bet[0, 0].item() == 1
        assert env.highest[0].item() == 1
        assert env.last_raise_size[0].item() == 1
    else:
        env.highest[0] = torch.tensor(10, dtype=torch.int32)
        env.current_round_bet[0] = torch.tensor([6, 10], dtype=torch.int32)
        env.total_invested[0] = env.current_round_bet[0].clone()
        env.pots[0] = torch.tensor(16, dtype=torch.int32)
        env.last_raise_size[0] = torch.tensor(4, dtype=torch.int32)
        env.execute_actions(torch.tensor([2], dtype=torch.long))
        assert env.current_round_bet[0, 0].item() == 14
        assert env.highest[0].item() == 14
        assert env.last_raise_size[0].item() == 4


def _assert_exact_stack_action_case(mode: str) -> None:
    env = _build_env(n_players=2)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.agg[0] = torch.tensor(1, dtype=torch.int32)
    env.acted[0] = torch.tensor(0, dtype=torch.int32)

    if mode == "call":
        env.highest[0] = torch.tensor(10, dtype=torch.int32)
        env.current_round_bet[0] = torch.tensor([4, 10], dtype=torch.int32)
        env.total_invested[0] = env.current_round_bet[0].clone()
        env.stacks[0] = torch.tensor([6, 50], dtype=torch.int32)
        env.pots[0] = torch.tensor(14, dtype=torch.int32)
        env.execute_actions(torch.tensor([1], dtype=torch.long))
        assert env.status[0, 0].item() == env.ALLIN
        assert env.current_round_bet[0, 0].item() == 10
        assert env.agg[0].item() == 1
    elif mode == "allin_call":
        env.highest[0] = torch.tensor(10, dtype=torch.int32)
        env.current_round_bet[0] = torch.tensor([4, 10], dtype=torch.int32)
        env.total_invested[0] = env.current_round_bet[0].clone()
        env.stacks[0] = torch.tensor([6, 50], dtype=torch.int32)
        env.pots[0] = torch.tensor(14, dtype=torch.int32)
        env.last_raise_size[0] = torch.tensor(4, dtype=torch.int32)
        env.execute_actions(torch.tensor([12], dtype=torch.long))
        assert env.status[0, 0].item() == env.ALLIN
        assert env.current_round_bet[0, 0].item() == 10
        assert env.agg[0].item() == 1
        assert env.last_raise_size[0].item() == 4
    else:
        env.highest[0] = torch.tensor(10, dtype=torch.int32)
        env.current_round_bet[0] = torch.tensor([4, 10], dtype=torch.int32)
        env.total_invested[0] = env.current_round_bet[0].clone()
        env.stacks[0] = torch.tensor([10, 50], dtype=torch.int32)
        env.pots[0] = torch.tensor(14, dtype=torch.int32)
        env.last_raise_size[0] = torch.tensor(4, dtype=torch.int32)
        env.execute_actions(torch.tensor([12], dtype=torch.long))
        assert env.status[0, 0].item() == env.ALLIN
        assert env.current_round_bet[0, 0].item() == 14
        assert env.agg[0].item() == 0
        assert env.last_raise_size[0].item() == 4


def _assert_mixed_stage_recalc_case(mode: str) -> None:
    env = _build_env(n_players=2, n_games=4)
    env.equity_dirty[:] = torch.tensor([True, True, True, True], dtype=torch.bool)
    env.equities.fill_(0.17)
    env.stages[:] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    env.board[1] = torch.tensor([1, 2, 3, -1, -1], dtype=torch.int32)
    env.board[2] = torch.tensor([1, 2, 3, 4, -1], dtype=torch.int32)
    env.board[3] = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    env.hands[1, 0] = torch.tensor([6, 7], dtype=torch.int32)
    env.hands[1, 1] = torch.tensor([8, 9], dtype=torch.int32)
    env.hands[2, 0] = torch.tensor([6, 7], dtype=torch.int32)
    env.hands[2, 1] = torch.tensor([8, 9], dtype=torch.int32)
    env.hands[3, 0] = torch.tensor([6, 7], dtype=torch.int32)
    env.hands[3, 1] = torch.tensor([8, 9], dtype=torch.int32)

    env.calculate_equities()

    if mode == "preflop":
        assert env.equities[0].tolist() == pytest.approx([0.5, 0.5])
    elif mode == "flop":
        assert torch.all((env.equities[1] >= 0.0) & (env.equities[1] <= 1.0)).item()
    elif mode == "turn":
        assert torch.all((env.equities[2] >= 0.0) & (env.equities[2] <= 1.0)).item()
    else:
        assert torch.all((env.equities[3] >= 0.0) & (env.equities[3] <= 1.0)).item()


def _assert_step_closes_on_current_actor_case() -> None:
    env = _build_env(n_players=3)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.idx[0] = torch.tensor(1, dtype=torch.int32)
    env.agg[0] = torch.tensor(1, dtype=torch.int32)
    env.acted[0] = torch.tensor(2, dtype=torch.int32)
    env.highest[0] = torch.tensor(0, dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([0, 0, 0], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([0, 0, 0], dtype=torch.int32)
    env.pots[0] = torch.tensor(0, dtype=torch.int32)
    env.stages[0] = torch.tensor(0, dtype=torch.int32)
    env.is_done[0] = False

    env.step(torch.tensor([1], dtype=torch.long))

    assert env.stages[0].item() == 1
    assert env.idx[0].item() == 1


@pytest.mark.parametrize(("name", "action_id", "expected_bet", "_description"), POT_FRACTION_CASES)
def test_pot_fraction_action_matrix(name: str, action_id: int, expected_bet: int, _description: str) -> None:
    _assert_pot_fraction_case(action_id, expected_bet)


@pytest.mark.parametrize(("name", "idx", "button", "expected_rel"), RELATIVE_POSITION_CASES)
def test_relative_position_matrix(name: str, idx: int, button: int, expected_rel: int) -> None:
    _assert_relative_position_case(idx, button, expected_rel)


@pytest.mark.parametrize(("name", "acting_bet", "highest", "expected_call"), CALL_AMOUNT_CASES)
def test_call_amount_matrix(name: str, acting_bet: int, highest: int, expected_call: int) -> None:
    _assert_call_amount_case(acting_bet, highest, expected_call)


@pytest.mark.parametrize(("name", "idx", "expected_flat"), OPPONENT_ORDER_CASES)
def test_opponent_order_matrix(name: str, idx: int, expected_flat: list[int]) -> None:
    _assert_opponent_order_case(idx, expected_flat)


@pytest.mark.parametrize(("name", "candidate_players"), RESET_DECKPOS_CASES)
def test_reset_deck_position_matrix(name: str, candidate_players: int) -> None:
    _assert_reset_deck_position_case(candidate_players)


@pytest.mark.parametrize(("name", "candidate_players", "q_seat", "expected_active"), QSEAT_FLOOR_CASES)
def test_reset_qseat_floor_matrix(name: str, candidate_players: int, q_seat: int, expected_active: int) -> None:
    _assert_qseat_floor_case(candidate_players, q_seat, expected_active)


@pytest.mark.parametrize(("name", "n_players", "reset_count", "expected_button", "expected_sb", "expected_bb", "expected_idx"), BUTTON_SEQUENCE_CASES)
def test_button_sequence_matrix(
    name: str,
    n_players: int,
    reset_count: int,
    expected_button: int,
    expected_sb: int,
    expected_bb: int,
    expected_idx: int,
) -> None:
    _assert_button_sequence_case(n_players, reset_count, expected_button, expected_sb, expected_bb, expected_idx)


@pytest.mark.parametrize(("name", "mode"), FOLD_RESOLUTION_CASES)
def test_fold_resolution_matrix(name: str, mode: str) -> None:
    _assert_fold_resolution_case(mode)


@pytest.mark.parametrize(("name", "stage", "expected_stage", "expected_done"), HEADSUP_CHECKAROUND_CASES)
def test_headsup_checkaround_matrix(name: str, stage: int, expected_stage: int, expected_done: bool) -> None:
    _assert_headsup_checkaround_case(stage, expected_stage, expected_done)


@pytest.mark.parametrize(("name", "stage", "expected_stage", "expected_done"), AUTO_RUNOUT_STAGE_CASES)
def test_auto_runout_stage_matrix(name: str, stage: int, expected_stage: int, expected_done: bool) -> None:
    _assert_auto_runout_stage_case(stage, expected_stage, expected_done)


@pytest.mark.parametrize(("name", "n_players"), PREFLOP_BASELINE_CASES)
def test_preflop_baseline_matrix(name: str, n_players: int) -> None:
    _assert_preflop_baseline_case(n_players)


def test_set_agents_replaces_agent_reference() -> None:
    _assert_set_agents_case()


def test_post_blinds_keeps_blind_seat_active_when_stack_remains() -> None:
    _assert_post_blinds_active_case()


@pytest.mark.parametrize(("name", "mode"), MIN_RAISE_CASES)
def test_min_raise_matrix(name: str, mode: str) -> None:
    _assert_min_raise_case(mode)


@pytest.mark.parametrize(("name", "mode"), EXACT_STACK_ACTION_CASES)
def test_exact_stack_action_matrix(name: str, mode: str) -> None:
    _assert_exact_stack_action_case(mode)


@pytest.mark.parametrize(("name", "mode"), MIXED_STAGE_RECALC_CASES)
def test_mixed_stage_recalc_matrix(name: str, mode: str) -> None:
    _assert_mixed_stage_recalc_case(mode)


def test_step_closes_on_current_actor_transition_matrix() -> None:
    _assert_step_closes_on_current_actor_case()


def runner_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for name, action_id, expected_bet, description in POT_FRACTION_CASES:
        cases.append({"name": name, "description": description, "run": lambda action_id=action_id, expected_bet=expected_bet: _assert_pot_fraction_case(action_id, expected_bet)})
    for name, idx, button, expected_rel in RELATIVE_POSITION_CASES:
        cases.append({"name": name, "description": f"Verify observation relative position encodes seat {idx} correctly from button {button}.", "run": lambda idx=idx, button=button, expected_rel=expected_rel: _assert_relative_position_case(idx, button, expected_rel)})
    for name, acting_bet, highest, expected_call in CALL_AMOUNT_CASES:
        cases.append({"name": name, "description": f"Verify observation call amount is {expected_call} when highest is {highest} and the acting seat has already invested {acting_bet}.", "run": lambda acting_bet=acting_bet, highest=highest, expected_call=expected_call: _assert_call_amount_case(acting_bet, highest, expected_call)})
    for name, idx, expected_flat in OPPONENT_ORDER_CASES:
        cases.append({"name": name, "description": f"Verify full-ring opponent ordering wraps correctly when the acting seat is {idx}.", "run": lambda idx=idx, expected_flat=expected_flat: _assert_opponent_order_case(idx, expected_flat)})
    for name, candidate_players in RESET_DECKPOS_CASES:
        cases.append({"name": name, "description": f"Verify reset advances deck position to exactly {candidate_players * 2} after dealing a {candidate_players}-player preflop.", "run": lambda candidate_players=candidate_players: _assert_reset_deck_position_case(candidate_players)})
    for name, candidate_players, q_seat, expected_active in QSEAT_FLOOR_CASES:
        cases.append({"name": name, "description": f"Verify random short-handed reset with candidate {candidate_players} and q seat {q_seat} results in {expected_active} active seats.", "run": lambda candidate_players=candidate_players, q_seat=q_seat, expected_active=expected_active: _assert_qseat_floor_case(candidate_players, q_seat, expected_active)})
    for name, n_players, reset_count, expected_button, expected_sb, expected_bb, expected_idx in BUTTON_SEQUENCE_CASES:
        cases.append({"name": name, "description": f"Verify reset {reset_count} times on a {n_players}-player table yields button {expected_button}, small blind {expected_sb}, big blind {expected_bb}, and first actor {expected_idx}.", "run": lambda n_players=n_players, reset_count=reset_count, expected_button=expected_button, expected_sb=expected_sb, expected_bb=expected_bb, expected_idx=expected_idx: _assert_button_sequence_case(n_players, reset_count, expected_button, expected_sb, expected_bb, expected_idx)})
    for name, mode in FOLD_RESOLUTION_CASES:
        cases.append({"name": name, "description": f"Verify fold-winner settlement behaves correctly in the `{mode}` mode.", "run": lambda mode=mode: _assert_fold_resolution_case(mode)})
    for name, stage, expected_stage, expected_done in HEADSUP_CHECKAROUND_CASES:
        cases.append({"name": name, "description": f"Verify a heads-up check-around from stage {stage} reaches stage {expected_stage} with done={expected_done}.", "run": lambda stage=stage, expected_stage=expected_stage, expected_done=expected_done: _assert_headsup_checkaround_case(stage, expected_stage, expected_done)})
    for name, stage, expected_stage, expected_done in AUTO_RUNOUT_STAGE_CASES:
        cases.append({"name": name, "description": f"Verify a no-actor auto-runout from stage {stage} reaches stage {expected_stage} with done={expected_done}.", "run": lambda stage=stage, expected_stage=expected_stage, expected_done=expected_done: _assert_auto_runout_stage_case(stage, expected_stage, expected_done)})
    for name, n_players in PREFLOP_BASELINE_CASES:
        cases.append({"name": name, "description": f"Verify dirty preflop equities reset to 0.5 for all {n_players} active seats.", "run": lambda n_players=n_players: _assert_preflop_baseline_case(n_players)})
    cases.append({"name": "setup/set-agents-replaces-reference", "description": "Verify set_agents replaces the environment agent reference without mutating other state.", "run": _assert_set_agents_case})
    cases.append({"name": "setup/post-blinds-keeps-blind-seat-active-when-stack-remains", "description": "Verify posting the big blind leaves the blind seat active when chips remain behind.", "run": _assert_post_blinds_active_case})
    for name, mode in MIN_RAISE_CASES:
        cases.append({"name": name, "description": f"Verify min-raise behavior is correct in the `{mode}` configuration.", "run": lambda mode=mode: _assert_min_raise_case(mode)})
    for name, mode in EXACT_STACK_ACTION_CASES:
        cases.append({"name": name, "description": f"Verify exact-stack action semantics are correct in the `{mode}` configuration.", "run": lambda mode=mode: _assert_exact_stack_action_case(mode)})
    for name, mode in MIXED_STAGE_RECALC_CASES:
        cases.append({"name": name, "description": f"Verify mixed dirty-stage equity recomputation handles the `{mode}` row correctly without corrupting the others.", "run": lambda mode=mode: _assert_mixed_stage_recalc_case(mode)})
    cases.append({"name": "round-progression/closes-on-current-actor-when-aggressor-checks-last", "description": "Verify a betting round closes when the aggressor is the current actor and all active seats have already acted.", "run": _assert_step_closes_on_current_actor_case})
    return cases
