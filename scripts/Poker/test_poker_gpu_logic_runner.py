from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.Poker import trainGPU  # noqa: E402


@dataclass(frozen=True)
class TestCase:
    name: str
    description: str
    run: Callable[[], None]


def _load(module_name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


showdown = _load("test_poker_gpu_showdown", "tests/poker/test_poker_gpu_showdown.py")
side_pots = _load("test_poker_gpu_side_pot_showdown", "tests/poker/test_poker_gpu_side_pot_showdown.py")
preflop = _load("test_poker_gpu_preflop_allin_resolver", "tests/poker/test_poker_gpu_preflop_allin_resolver.py")
round_progression = _load("test_poker_gpu_round_progression", "tests/poker/test_poker_gpu_round_progression.py")
no_actor_rewards = _load("test_poker_gpu_no_actor_rewards", "tests/poker/test_poker_gpu_no_actor_rewards.py")
reset_rotation = _load("test_poker_gpu_reset_rotation", "tests/poker/test_poker_gpu_reset_rotation.py")
street_actor_reset = _load("test_poker_gpu_street_actor_reset", "tests/poker/test_poker_gpu_street_actor_reset.py")
state_contracts = _load("test_poker_gpu_state_contracts", "tests/poker/test_poker_gpu_state_contracts.py")
action_terminal_contracts = _load(
    "test_poker_gpu_action_terminal_contracts",
    "tests/poker/test_poker_gpu_action_terminal_contracts.py",
)


def _assert_hand_ranks_present() -> None:
    hand_ranks = PROJECT_ROOT / "environments" / "Poker" / "HandRanks.dat"
    assert hand_ranks.exists(), (
        "PokerGPU requires environments/Poker/HandRanks.dat. "
        "The runner cannot execute live environment cases without it."
    )


def _assert_active_q_mask_excludes_pre_terminated_games() -> None:
    terminated = torch.tensor([False, True, False, True])
    q_mask = torch.tensor([True, True, False, True])
    active_games = trainGPU._get_active_q_mask(terminated, q_mask)
    assert active_games.tolist() == [True, False, False, False]


def _assert_should_stop_loop_matches_five_step_cadence() -> None:
    threshold = torch.tensor(0.8)
    assert not trainGPU._should_stop_loop(4, torch.tensor([True, True, True, True, True]), threshold)
    assert not trainGPU._should_stop_loop(5, torch.tensor([True, True, True, True, False]), threshold)
    assert trainGPU._should_stop_loop(5, torch.tensor([True, True, True, True, True]), threshold)


def _build_cases() -> list[TestCase]:
    return [
        TestCase(
            name="asset/hand-ranks-present",
            description=(
                "Verify the required HandRanks lookup asset is present before running poker logic.\n"
                "This is a preflight check for the evaluator-backed GPU environment.\n"
                "Without it, every live PokerGPU case would fail during environment construction."
            ),
            run=_assert_hand_ranks_present,
        ),
        TestCase(
            name="reset/options-rotation-rolls-persistent-stacks",
            description=(
                "Verify reset honors the requested stack rotation from the options payload.\n"
                "This protects persistent bankroll carryover between hands.\n"
                "It also confirms blind posting happens after the rotated seat order is applied."
            ),
            run=reset_rotation.test_reset_rolls_persistent_stacks_from_options_rotation,
        ),
        TestCase(
            name="reset/options-and-explicit-rotation-match",
            description=(
                "Verify reset uses the same stack rotation whether it comes from options or the explicit argument.\n"
                "This prevents two public reset inputs from drifting apart semantically.\n"
                "It also guards the training path that passes rotation through reset options."
            ),
            run=reset_rotation.test_reset_options_rotation_matches_explicit_rotation_argument,
        ),
        TestCase(
            name="reset/zero-rotation-keeps-order",
            description=(
                "Verify a zero rotation leaves the persistent stack order unchanged before blind posting.\n"
                "This protects the default episode path where no seat rotation is requested.\n"
                "It also catches accidental stack shuffles during reset."
            ),
            run=reset_rotation.test_reset_zero_rotation_keeps_stack_order_before_blind_post,
        ),
        TestCase(
            name="reset/wraps-large-rotation-values",
            description=(
                "Verify rotation values larger than the table size wrap cleanly around the table.\n"
                "This protects long-running training loops that may accumulate arbitrary rotation counts.\n"
                "It also confirms stack order follows modular seat arithmetic."
            ),
            run=reset_rotation.test_reset_wraps_options_rotation_values_larger_than_table_size,
        ),
        TestCase(
            name="reset/restores-invalid-persistent-stacks-before-rotation",
            description=(
                "Verify busted and over-cap persistent stacks are normalized before seat rotation.\n"
                "This protects bankroll carryover from impossible or stale values.\n"
                "It also ensures later blind posting starts from a legal stack state."
            ),
            run=reset_rotation.test_reset_restores_invalid_persistent_stacks_before_rotating,
        ),
        TestCase(
            name="observation/core-fields-pack-correctly",
            description=(
                "Verify the observation vector packs board cards, acting hand, stage, relative position, pot,\n"
                "call amount, stack, and status into their documented slots for the current actor.\n"
                "This protects the fundamental state contract consumed by the poker agents."
            ),
            run=state_contracts.test_get_obs_packs_core_fields_for_current_actor,
        ),
        TestCase(
            name="observation/short-handed-live-opponents-zero-pad-unused-slots",
            description=(
                "Verify a short-handed table embedded in a larger max-player layout lists only live opponents\n"
                "in order and leaves the remaining observation padding as zeros.\n"
                "This protects variable-player training from leaking sit-out seats into real opponent features."
            ),
            run=state_contracts.test_get_obs_short_handed_orders_only_live_opponents_and_zero_pads_unused_slots,
        ),
        TestCase(
            name="setup/post-blinds-updates-pot-bets-and-allin-state",
            description=(
                "Verify posting blinds updates the pot, the round bet, total invested, and the blind seat status.\n"
                "This includes the edge case where posting the blind consumes the entire stack.\n"
                "It protects the opening chip-flow contract of every hand."
            ),
            run=state_contracts.test_post_blinds_updates_pot_bets_and_allin_status,
        ),
        TestCase(
            name="setup/deal-players-cards-advances-positions-per-game",
            description=(
                "Verify preflop dealing advances deck positions by the requested number of cards and preserves\n"
                "per-game card order inside a batched environment.\n"
                "This protects card uniqueness and deterministic deck slicing."
            ),
            run=state_contracts.test_deal_players_cards_advances_positions_and_preserves_per_game_order,
        ),
        TestCase(
            name="setup/deal-cards-updates-only-selected-games",
            description=(
                "Verify board-card dealing mutates only the selected game rows and leaves other rows untouched.\n"
                "This protects batched street transitions from cross-row contamination.\n"
                "It also confirms deck positions advance only for the targeted games."
            ),
            run=state_contracts.test_deal_cards_updates_only_selected_games,
        ),
        TestCase(
            name="setup/get-info-reports-live-contract",
            description=(
                "Verify get_info returns the active-player count, the live stack tensor, and the current seat index.\n"
                "This protects the trainer and runner contract that consumes info-side metadata.\n"
                "It also confirms the returned values match the environment's current state."
            ),
            run=state_contracts.test_get_info_reports_active_players_stacks_and_current_seat,
        ),
        TestCase(
            name="actions/inactive-seats-ignore-incoming-actions",
            description=(
                "Verify folded, all-in, sit-out, and already-done rows ignore incoming actions entirely.\n"
                "This protects batched action execution from mutating illegal actors.\n"
                "It also confirms inactive rows do not consume chips or increment acted counters."
            ),
            run=state_contracts.test_execute_actions_ignores_folded_allin_sitout_and_done_rows,
        ),
        TestCase(
            name="actions/check-with-zero-call-cost-only-marks-acted",
            description=(
                "Verify a check in a zero-call-cost spot only marks the actor as having acted.\n"
                "This protects no-cost action semantics from accidentally moving chips.\n"
                "It also confirms the pot and investment state remain unchanged."
            ),
            run=state_contracts.test_execute_actions_check_only_marks_actor_as_acted_when_call_cost_is_zero,
        ),
        TestCase(
            name="actions/fractional-raise-rounds-down-to-int-chips",
            description=(
                "Verify a fractional pot raise is truncated to an integer number of chips before application.\n"
                "This protects integer chip accounting and ensures pot-sized action ids have deterministic semantics.\n"
                "It also confirms the resulting highest bet and pot size reflect the rounded amount."
            ),
            run=state_contracts.test_execute_actions_fractional_raise_rounds_down_to_int_chips,
        ),
        TestCase(
            name="actions/short-call-allin-caps-at-stack",
            description=(
                "Verify a call that costs more chips than the actor has only spends the remaining stack\n"
                "and marks the actor all-in instead of driving the stack negative.\n"
                "This protects partial-call all-in accounting."
            ),
            run=action_terminal_contracts.test_execute_actions_call_uses_remaining_stack_and_marks_allin,
        ),
        TestCase(
            name="actions/min-raise-reopens-and-updates-raise-size",
            description=(
                "Verify a direct min-raise applies the stored last-raise size, reopens action,\n"
                "and transfers aggressor ownership to the raising seat.\n"
                "This protects the no-limit betting contract for explicit min-raises."
            ),
            run=action_terminal_contracts.test_execute_actions_min_raise_reopens_action_and_updates_raise_size,
        ),
        TestCase(
            name="termination/fold-winner-resolution-is-idempotent",
            description=(
                "Verify resolving a fold winner twice does not pay the same pot twice after the first award.\n"
                "This protects terminal settlement from duplicate invocation bugs.\n"
                "It also confirms the pot is the sole one-shot payout source."
            ),
            run=action_terminal_contracts.test_resolve_fold_winners_is_idempotent_after_pot_is_cleared,
        ),
        TestCase(
            name="termination/no-done-rows-leaves-state-untouched",
            description=(
                "Verify showdown resolution is a no-op when no rows are marked done and no runout is required.\n"
                "This protects mixed batches from accidental board dealing or payouts on live hands.\n"
                "It also confirms deck positions stay stable."
            ),
            run=action_terminal_contracts.test_resolve_terminated_games_noops_when_no_done_rows_need_resolution,
        ),
        TestCase(
            name="termination/turn-runout-preserves-board-and-burns-once",
            description=(
                "Verify a turn-stage all-in runout keeps the existing flop and turn cards intact,\n"
                "burns exactly one card, and deals only the river before showdown.\n"
                "This protects partial-board runout correctness."
            ),
            run=action_terminal_contracts.test_resolve_terminated_games_turn_runout_preserves_existing_board_and_burns_once,
        ),
        TestCase(
            name="termination/flop-runout-preserves-flop-and-deals-turn-river",
            description=(
                "Verify a flop-stage all-in runout preserves the existing flop,\n"
                "then deals the turn and river with the correct burn-card pattern.\n"
                "This protects staged board completion from overwriting public cards."
            ),
            run=action_terminal_contracts.test_resolve_terminated_games_flop_runout_preserves_flop_and_deals_turn_then_river,
        ),
        TestCase(
            name="equity/clean-rows-remain-untouched",
            description=(
                "Verify equity recomputation updates only dirty rows and leaves already-clean rows unchanged.\n"
                "This protects the hot-path dirty-bit cache from silently rewriting stable values.\n"
                "It also confirms dirty rows fall back to the documented preflop baseline when appropriate."
            ),
            run=action_terminal_contracts.test_calculate_equities_leaves_clean_rows_untouched,
        ),
        TestCase(
            name="termination/fold-end-clears-round-state",
            description=(
                "Verify a fold that ends the hand clears current-round bets, total invested amounts, and highest bet.\n"
                "This protects reset-free terminal cleanup for rows that finish before showdown.\n"
                "It also confirms the winner receives the pot exactly once."
            ),
            run=action_terminal_contracts.test_step_clears_round_state_after_fold_ends_hand,
        ),
        TestCase(
            name="termination/river-showdown-clears-round-state",
            description=(
                "Verify a river showdown step clears current-round bets, total invested amounts, and highest bet\n"
                "after the hand is settled and the pot is awarded.\n"
                "This protects post-showdown cleanup for immediately resettable rows."
            ),
            run=action_terminal_contracts.test_step_clears_round_state_after_river_showdown,
        ),
        TestCase(
            name="street-actor/flop-starts-left-of-button",
            description=(
                "Verify a flop transition restarts action from the first active seat left of the button.\n"
                "This is the core post-transition actor-order rule for multiway poker.\n"
                "It also ensures the next observation belongs to the legal actor."
            ),
            run=street_actor_reset.test_flop_transition_restarts_action_from_first_seat_left_of_button,
        ),
        TestCase(
            name="street-actor/turn-skips-folded-seats",
            description=(
                "Verify a turn transition skips folded seats when choosing the next legal actor.\n"
                "This protects actor order after players drop out mid-hand.\n"
                "It also confirms folded seats do not re-enter the action scan."
            ),
            run=street_actor_reset.test_turn_transition_skips_folded_seats_left_of_button,
        ),
        TestCase(
            name="street-actor/river-skips-all-in-seats",
            description=(
                "Verify a river transition skips all-in seats when choosing the next legal actor.\n"
                "This protects tables where some players remain eligible for showdown but not for action.\n"
                "It also confirms the board still advances correctly on that transition."
            ),
            run=street_actor_reset.test_river_transition_skips_all_in_seats_left_of_button,
        ),
        TestCase(
            name="street-actor/heads-up-uses-only-other-seat",
            description=(
                "Verify a heads-up street transition selects the only other active seat.\n"
                "This protects the special two-player actor-order path.\n"
                "It also confirms no phantom seat selection is possible on short tables."
            ),
            run=street_actor_reset.test_heads_up_transition_uses_only_other_active_seat,
        ),
        TestCase(
            name="street-actor/batched-transition-updates-only-finished-rounds",
            description=(
                "Verify a batched step only resets actor order for rows that actually finished the street.\n"
                "This protects mixed batches where some games transition and others keep acting.\n"
                "It also confirms observations line up with the new actor seat per row."
            ),
            run=street_actor_reset.test_batched_transition_resets_only_finished_rounds_and_updates_observation,
        ),
        TestCase(
            name="showdown/strongest-active-hand-wins",
            description=(
                "Verify showdown awards the pot to the strongest eligible hand.\n"
                "This protects the central winner-selection rule.\n"
                "It also catches evaluator-rank direction mistakes."
            ),
            run=showdown.test_resolve_terminated_games_pays_strongest_active_hand,
        ),
        TestCase(
            name="showdown/folded-seats-cannot-win",
            description=(
                "Verify folded seats are excluded from showdown eligibility even if their cards would otherwise win.\n"
                "This protects winner masking at settlement time.\n"
                "It also keeps dead money in the pot while preserving legal eligibility."
            ),
            run=showdown.test_resolve_terminated_games_excludes_folded_players_from_winning,
        ),
        TestCase(
            name="showdown/tied-even-pot-splits-cleanly",
            description=(
                "Verify a tied even pot splits evenly between all winning hands.\n"
                "This protects chip conservation for the simplest split-pot case.\n"
                "It also confirms showdown settlement clears the pot and terminal stage."
            ),
            run=showdown.test_resolve_terminated_games_splits_tied_even_pot,
        ),
        TestCase(
            name="showdown/terminal-games-do-not-mutate-on-step",
            description=(
                "Verify already-terminal rows are not mutated by later step calls.\n"
                "This protects vectorized batches from corrupting completed hands.\n"
                "It also ensures rewards and state transitions stay anchored to live rows only."
            ),
            run=showdown.test_step_does_not_mutate_terminal_games,
        ),
        TestCase(
            name="side-pot/main-and-side-pot-follow-commitment",
            description=(
                "Verify showdown separates the main pot from the side pot when investments differ.\n"
                "This protects short-stack all-in semantics.\n"
                "It also confirms each layer is awarded only among eligible contributors."
            ),
            run=side_pots.test_resolve_terminated_games_awards_main_and_side_pot_by_commitment,
        ),
        TestCase(
            name="side-pot/split-main-pot-before-side-pot",
            description=(
                "Verify a split main pot is settled before a later side pot is awarded.\n"
                "This protects layered settlement order when winner sets differ by pot layer.\n"
                "It also catches chip-loss bugs across mixed split and non-split layers."
            ),
            run=side_pots.test_resolve_terminated_games_splits_main_pot_before_awarding_side_pot,
        ),
        TestCase(
            name="side-pot/folded-chips-stay-live-without-restoring-folded-seat",
            description=(
                "Verify folded contributions remain in side pots without making the folded seat eligible to win.\n"
                "This protects dead-money semantics after a player folds.\n"
                "It also confirms fold state and pot eligibility remain separate concepts."
            ),
            run=side_pots.test_resolve_terminated_games_keeps_folded_chips_in_side_pot_without_making_folded_player_eligible,
        ),
        TestCase(
            name="side-pot/multiple-layers-settle-correctly",
            description=(
                "Verify multiple side-pot layers settle correctly in a deeper multi-all-in hand.\n"
                "This protects the generalized side-pot decomposition path.\n"
                "It also confirms payout accumulation is correct across several contribution levels."
            ),
            run=side_pots.test_resolve_terminated_games_handles_multiple_side_pot_layers,
        ),
        TestCase(
            name="side-pot/batched-mixed-showdown-shapes",
            description=(
                "Verify batched showdown resolution handles different showdown shapes in different rows.\n"
                "This protects vectorized settlement across heterogeneous game states.\n"
                "It also confirms per-row pots and stacks remain isolated."
            ),
            run=side_pots.test_resolve_terminated_games_supports_batched_games_with_mixed_showdown_shapes,
        ),
        TestCase(
            name="preflop-runout/deals-full-board",
            description=(
                "Verify a preflop all-in showdown runs out the entire board with the correct burn-card pattern.\n"
                "This protects the automatic runout path when no further betting is possible.\n"
                "It also confirms deck position advances by the exact consumed-card count."
            ),
            run=preflop.test_resolve_terminated_games_deals_full_board_for_preflop_all_in_showdown,
        ),
        TestCase(
            name="preflop-runout/runout-picks-winner",
            description=(
                "Verify the generated runout is actually used to decide the winner in a preflop all-in.\n"
                "This protects the link between board completion and showdown resolution.\n"
                "It also ensures settlement is not based on stale or placeholder board state."
            ),
            run=preflop.test_resolve_terminated_games_preflop_all_in_uses_runout_to_pick_winner,
        ),
        TestCase(
            name="preflop-runout/tied-runout-splits-correctly",
            description=(
                "Verify a tied preflop all-in runout splits the resulting pot correctly.\n"
                "This protects chip conservation on the automatic runout path.\n"
                "It also confirms tie handling survives the preflop-to-showdown shortcut."
            ),
            run=preflop.test_resolve_terminated_games_preflop_all_in_splits_tied_runout,
        ),
        TestCase(
            name="preflop-runout/batched-preflop-and-river-resolution",
            description=(
                "Verify a batch can resolve a preflop all-in row and a river showdown row at the same time.\n"
                "This protects heterogeneous resolution inside a vectorized batch.\n"
                "It also confirms one row's runout does not mutate another row's settled board."
            ),
            run=preflop.test_resolve_terminated_games_handles_batched_preflop_and_river_showdowns,
        ),
        TestCase(
            name="preflop-runout/single-survivor-does-not-run-board",
            description=(
                "Verify the board is not run out when a hand is already won by fold before showdown.\n"
                "This protects the distinction between fold resolution and showdown resolution.\n"
                "It also avoids consuming deck cards in a hand that is already over."
            ),
            run=preflop.test_resolve_terminated_games_does_not_run_preflop_board_when_single_survivor_remains,
        ),
        TestCase(
            name="round-progression/skips-folded-and-all-in-next-actor",
            description=(
                "Verify the same-street actor scan skips folded and all-in seats.\n"
                "This protects turn order during normal betting rounds.\n"
                "It also ensures only legally actionable seats receive the next observation."
            ),
            run=round_progression.test_step_skips_folded_and_all_in_seats_when_selecting_next_actor,
        ),
        TestCase(
            name="round-progression/round-over-transitions-street",
            description=(
                "Verify a completed betting round advances to the next street and resets round-local state.\n"
                "This protects stage progression, betting reset, and aggressor reset.\n"
                "It also confirms the flop is actually dealt on a preflop transition."
            ),
            run=round_progression.test_step_marks_round_over_and_transitions_to_next_street,
        ),
        TestCase(
            name="round-progression/fold-leaves-single-survivor",
            description=(
                "Verify a fold that leaves one survivor ends the hand immediately.\n"
                "This protects the early-termination path for non-showdown wins.\n"
                "It also confirms the pot is awarded and cleared in the same step."
            ),
            run=round_progression.test_step_ends_hand_early_when_fold_leaves_single_survivor,
        ),
        TestCase(
            name="round-progression/preflop-all-call-closes-on-big-blind",
            description=(
                "Verify a no-raise preflop round closes correctly after the big blind checks the option.\n"
                "This protects a very common multiway hand shape.\n"
                "It also guards against runaway action loops on the same street."
            ),
            run=round_progression.test_step_closes_multiway_preflop_after_big_blind_checks_option,
        ),
        TestCase(
            name="round-progression/postflop-checkaround-advances",
            description=(
                "Verify a postflop check-around closes the street and advances to the next board card.\n"
                "This protects no-raise street progression after the flop.\n"
                "It also confirms actor order does not loop back to already-acted seats."
            ),
            run=round_progression.test_step_closes_multiway_postflop_checkaround_and_advances_to_turn,
        ),
        TestCase(
            name="round-progression/heads-up-postflop-opener",
            description=(
                "Verify the correct heads-up seat opens action after a street transition.\n"
                "This protects the two-player postflop actor-order rule.\n"
                "It also catches incorrect carryover of the prior street's closer."
            ),
            run=round_progression.test_step_sets_heads_up_postflop_opener_to_first_active_seat_left_of_button,
        ),
        TestCase(
            name="round-progression/short-all-in-does-not-reopen",
            description=(
                "Verify a sub-minimum all-in increases the amount to call without reopening action.\n"
                "This protects no-limit raise semantics for short stacks.\n"
                "It also confirms the minimum legal raise size is not incorrectly reduced."
            ),
            run=round_progression.test_execute_actions_short_all_in_does_not_reopen_action_or_shrink_min_raise,
        ),
        TestCase(
            name="round-progression/full-all-in-reopens-action",
            description=(
                "Verify a full legal all-in raise reopens action and updates the minimum raise size.\n"
                "This protects the aggressor-reset path for real raises.\n"
                "It also distinguishes full raises from short all-ins."
            ),
            run=round_progression.test_execute_actions_full_all_in_reopens_action_and_updates_min_raise,
        ),
        TestCase(
            name="round-progression/reward-uses-acting-seat-equity",
            description=(
                "Verify step-based rewards are attributed to the acting seat rather than the next actor.\n"
                "This protects Bellman targets in the Q-learning path.\n"
                "It also catches reward corruption caused by mutating the turn index too early."
            ),
            run=round_progression.test_step_reward_uses_acting_seat_equity_before_turn_advances,
        ),
        TestCase(
            name="round-progression/reuses-equities-within-street",
            description=(
                "Verify equity calculation is reused across consecutive actions on the same street.\n"
                "This protects the intended dirty-flag caching behavior.\n"
                "It also helps prevent avoidable recomputation in the hot step path."
            ),
            run=round_progression.test_step_reuses_equities_across_same_street_actions,
        ),
        TestCase(
            name="round-progression/recomputes-equities-after-street-change",
            description=(
                "Verify equities are recomputed after a street transition changes the public board.\n"
                "This protects reward accuracy after new community cards appear.\n"
                "It also confirms the dirty flag is re-enabled exactly when the board changes."
            ),
            run=round_progression.test_step_recomputes_equities_after_street_transition,
        ),
        TestCase(
            name="no-actor/placeholder-actions-zero-reward",
            description=(
                "Verify auto-runout rows with no legal actor return zero placeholder rewards across action ids.\n"
                "This protects mixed batches where some rows are resolving automatically.\n"
                "It also ensures meaningless action ids do not leak reward signal."
            ),
            run=no_actor_rewards.test_step_auto_runout_zeroes_placeholder_rewards_across_action_ids,
        ),
        TestCase(
            name="no-actor/placeholder-actions-preserve-transition-state",
            description=(
                "Verify placeholder actions in a no-actor row all produce the same transition state.\n"
                "This protects determinism when a row is auto-running to the next street.\n"
                "It also confirms the chosen placeholder action id is semantically ignored."
            ),
            run=no_actor_rewards.test_step_auto_runout_keeps_transition_state_identical_for_placeholder_actions,
        ),
        TestCase(
            name="no-actor/river-auto-runout-has-zero-action-reward",
            description=(
                "Verify a river auto-runout resolves showdown without paying action-based reward.\n"
                "This protects the distinction between automatic settlement and live decision reward.\n"
                "It also confirms the hand is fully settled and the pot is cleared."
            ),
            run=no_actor_rewards.test_step_auto_runout_on_river_resolves_showdown_without_action_reward,
        ),
        TestCase(
            name="no-actor/batched-zero-reward-only-for-no-actor-rows",
            description=(
                "Verify a mixed batch zeroes rewards only for rows with no legal actor.\n"
                "This protects reward integrity when live and auto-runout rows share a batch.\n"
                "It also confirms legal-actor rows still receive action-sensitive rewards."
            ),
            run=no_actor_rewards.test_step_batched_rewards_zero_only_games_without_legal_actor,
        ),
        TestCase(
            name="no-actor/legal-actor-keeps-action-sensitive-reward",
            description=(
                "Verify live rows still differentiate fold and call rewards when a legal actor exists.\n"
                "This protects learning signal quality on ordinary decision steps.\n"
                "It also guards against zeroing or flattening the whole reward batch."
            ),
            run=no_actor_rewards.test_step_preserves_action_sensitive_rewards_when_a_legal_actor_exists,
        ),
        TestCase(
            name="train-loop/active-q-mask-excludes-pre-terminated-games",
            description=(
                "Verify the training helper excludes rows that terminated before the current step.\n"
                "This protects the Q-update mask from revisiting dead rows.\n"
                "It also prevents stale rewards and next states from being trained twice."
            ),
            run=_assert_active_q_mask_excludes_pre_terminated_games,
        ),
        TestCase(
            name="train-loop/stop-cadence-matches-five-step-check",
            description=(
                "Verify the train loop only checks the stop threshold on the intended five-step cadence.\n"
                "This protects episode-boundary behavior from drifting silently.\n"
                "It also confirms the helper preserves the existing scheduling contract."
            ),
            run=_assert_should_stop_loop_matches_five_step_cadence,
        ),
    ]


def main() -> int:
    cases = _build_cases()
    failures: list[tuple[TestCase, BaseException]] = []

    print(f"Running {len(cases)} Poker GPU logic cases from {Path(__file__).name}")
    print("=" * 79)

    for index, case in enumerate(cases, start=1):
        print(f"[{index:02d}/{len(cases):02d}] {case.name}")
        print(case.description)
        try:
            case.run()
        except Exception as exc:  # pragma: no cover - this is the reporting path
            failures.append((case, exc))
            print("RESULT: FAIL")
            traceback.print_exc()
        else:
            print("RESULT: PASS")
        print("-" * 79)

    passed = len(cases) - len(failures)
    print(f"Summary: {passed} passed, {len(failures)} failed, {len(cases)} total.")
    if failures:
        print("Failed cases:")
        for case, exc in failures:
            print(f"- {case.name}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
