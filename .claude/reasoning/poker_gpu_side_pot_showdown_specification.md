# Task Definition and Objectives

Fix GPU showdown payout resolution in `environments/Poker/PokerGPU.py` so terminated hands are awarded in side-pot layers derived from `self.total_invested` instead of paying the entire pot to the best overall hand. The objective is to make GPU bankroll updates match poker side-pot rules and the existing CPU implementation in `environments/Poker/Poker.py`, eliminating silent chip corruption in uneven all-in hands.

# In-Scope / Out-of-Scope

In scope:
- Modify `environments/Poker/PokerGPU.py` to distribute showdown pots by commitment tiers.
- Add regression coverage in `tests/poker/test_poker_gpu_side_pot_showdown.py` for uneven all-ins, folded contributors, multi-layer side pots, split pots, and batched resolution.
- Preserve the existing `step()` contract in `environments/Poker/PokerGPU.py`, including post-resolution cleanup of `total_invested`.

Out of scope:
- Changing betting semantics, raise validation, or action encoding.
- Refactoring the CPU environment in `environments/Poker/Poker.py`.
- Broader showdown evaluator changes, reward shaping changes, or training-loop performance work unrelated to terminal payout correctness.

# Current State and Architecture Context

Current behavior:
- `environments/Poker/PokerGPU.py:299` resolves terminated games by evaluating all non-folded hands, building one `winner_mask`, and splitting `self.pots[showdown_games]` across those winners.
- `environments/Poker/PokerGPU.py:107`, `:170`, `:208`, and `:242` already maintain exact per-player hand investment in `self.total_invested`.
- `environments/Poker/PokerGPU.py:449` calls `resolve_fold_winners()` first, then `resolve_terminated_games()`, then zeroes `self.total_invested` for finished hands. That makes showdown resolution the last point where contribution eligibility can be honored.
- `environments/Poker/Poker.py:243` already implements CPU side-pot chunking by sorting active players on `total_invested`, peeling contribution layers, and selecting winners only from eligible contributors.

What changes:
- GPU showdown payout logic will stop treating the whole pot as one winner-take-all bucket.
- GPU showdown payout will instead derive side-pot tiers from `self.total_invested` and award each tier only among eligible non-folded contributors using tensorized operations.

What does not change:
- Hand ranking remains the current `HandRanks.dat` lookup path.
- Fold-win resolution remains in `resolve_fold_winners()`.
- `step()` ordering and the post-hand reset of `current_round_bet`, `total_invested`, and `highest` remain unchanged.

# Proposed Design and Integration Plan

Implementation approach:
- Keep showdown hand evaluation batched in `environments/Poker/PokerGPU.py` and remove Python-level loops from the side-pot payout path.
- Replace the one-shot whole-pot payout block in `resolve_terminated_games()` with tensorized side-pot distribution driven by `self.total_invested[showdown_games, :self.active_players]`.
- Sort each game's committed chips once, derive per-layer chip deltas by subtracting the shifted sorted commitments, and materialize a `[n_showdown, active_players, active_players]` contributor tensor via broadcasted `invested >= level` comparison.
- Restrict winner selection per layer to seats that both contributed to that layer and are still eligible at showdown (`ACTIVE` or `ALLIN`) using masked `max` across the player axis.
- Award each layer with integer floor shares plus deterministic remainder assignment to the first winning seat in table order using tensor `argmax` and `scatter_add_`, not per-game loops. This preserves chip conservation while staying GPU-friendly.

Implementation tasks:
- Modify `environments/Poker/PokerGPU.py` inside `resolve_terminated_games()` to compute `invested`, `eligible_mask`, and `hand_ranks` once for all showdown games, then distribute side pots through broadcasted layer tensors with no Python loops in the payout algorithm. Why: this is the root-cause location where total investment is currently ignored, and the revised request explicitly requires GPU-oriented vectorized execution. Downstream impact: `step()` in the same file will now receive corrected stack updates before it clears per-hand investment state. Potential issues: masking mistakes can accidentally include folded players or exclude short-stack winners; integer division can leak chips if remainder handling is omitted; broadcast shapes must remain `[n_showdown, active_players, active_players]`. Validation: run `pytest tests/poker/test_poker_gpu_side_pot_showdown.py -q` and `pytest tests/poker/test_poker_gpu_showdown.py -q`.
- Add `tests/poker/test_poker_gpu_side_pot_showdown.py` with five self-contained tests covering main-pot-only eligibility, split-main-plus-side-pot resolution, folded contributors, multiple side-pot tiers, and mixed batched showdown resolution. Why: the existing showdown test file only covers strongest-hand and even split cases. Downstream impact: future changes to showdown payout logic will now fail fast if they regress side-pot semantics. Potential issues: incorrect card setups can accidentally create ties or stronger hidden hands. Validation: each test asserts exact final stack vectors, zeroed pots, and completed stage markers.
- Update `tests/poker/test_poker_gpu_showdown.py` to populate `total_invested` consistently with each synthetic pot fixture. Why: once side-pot logic uses contribution data as the payout source of truth, showdown fixtures that omit it no longer describe a valid hand state. Downstream impact: the legacy showdown tests continue to validate strongest-hand, folded-player, and split-pot behavior under the corrected contract. Potential issues: fixture investment vectors must sum to the configured pot. Validation: run `pytest tests/poker/test_poker_gpu_showdown.py -q`.
- Leave `environments/Poker/Poker.py` unchanged but use its `resolve_showdown()` behavior as the semantic reference during implementation review. Why: the CPU path already encodes the expected payout contract. Downstream impact: CPU and GPU environments remain aligned on showdown math. Potential issues: the CPU code iterates over player objects while the GPU path uses tensors, so the implementation must preserve semantics without introducing shape bugs. Validation: compare layer construction and remainder behavior conceptually against the CPU algorithm.

# Data and API Contract Changes

No external API, schema, or public interface changes are expected.

Internal behavior change:
- Before: `resolve_terminated_games()` paid `self.pots[game]` to the best overall eligible hand(s), ignoring contribution caps.
- After: `resolve_terminated_games()` pays each commitment layer only to eligible contributors for that layer, preserving total-chip conservation and stack correctness.

Tensor contract details:
- `self.total_invested`: remains `int32` on `self.device`, shape `[n_games, n_players]`, and becomes the authoritative source for showdown eligibility tiers.
- `self.stacks`: still receives integer chip updates in place on `self.device`.
- `self.pots`: still resolves to zero for completed showdown games.

# Edge Cases and Failure Modes

- Uneven all-in with one short-stack winner: the short stack must win only the main pot, not the side pot. Covered by `tests/poker/test_poker_gpu_side_pot_showdown.py::test_resolve_terminated_games_awards_main_and_side_pot_by_commitment`.
- Main-pot tie with a different side-pot winner: tied short stacks split only the main pot while the deeper stack wins the side pot uncontested by the short stack. Covered by `tests/poker/test_poker_gpu_side_pot_showdown.py::test_resolve_terminated_games_splits_main_pot_before_awarding_side_pot`.
- Folded player contributions remain in the pot but the folded seat must never become eligible to win any layer. Covered by `tests/poker/test_poker_gpu_side_pot_showdown.py::test_resolve_terminated_games_keeps_folded_chips_in_side_pot_without_making_folded_player_eligible`.
- Multiple side-pot layers in one hand: each tier must independently select winners from the correct eligible subset. Covered by `tests/poker/test_poker_gpu_side_pot_showdown.py::test_resolve_terminated_games_handles_multiple_side_pot_layers`.
- Batched GPU resolution across multiple concurrent games: a side-pot game and a normal showdown must both resolve correctly in the same call. Covered by `tests/poker/test_poker_gpu_side_pot_showdown.py::test_resolve_terminated_games_supports_batched_games_with_mixed_showdown_shapes`.

Failure modes to guard against:
- Using the full-game `winner_mask` for every layer will reintroduce the bug.
- Using only showdown-eligible players as contributors will incorrectly discard folded-chip contributions from the pot.
- Omitting remainder distribution on integer splits will silently destroy chips.
- Falling back to Python loops over games, layers, or players would violate the explicit GPU-vectorization requirement for this ticket.

# Security, Reliability, and Performance Considerations

Security:
- No new I/O, subprocess, serialization, or user-input surfaces are introduced.

Reliability:
- Keep all tensors on `self.device`; do not introduce implicit CPU transfers during payout resolution.
- Use the already computed `hand_ranks` tensor as the single source of showdown strength to avoid divergent winner calculation across layers.
- Preserve integer chip accounting end-to-end so the sum of awarded chips equals the committed pot.

Performance:
- Hand evaluation remains batched across all showdown games, which is the expensive part of the path.
- Side-pot distribution stays fully tensorized across showdown games, side-pot layers, and player seats so the payout path can exploit GPU parallelism instead of falling back to Python control flow.
- No new tensor copies should move data off-device; use broadcasted masking, reduction, and scatter operations in place where safe.

PyTorch best-practice cross-reference:
- Device placement: all new tensors must be created on `self.device`.
- Tensor operations: keep hand evaluation and side-pot payout vectorized end-to-end in `resolve_terminated_games()`; do not introduce Python loops over games, layers, or players.
- Numerical stability: integer chip math remains exact; avoid float payout math entirely.
- Memory management: reuse computed tensors inside `resolve_terminated_games()` rather than building redundant rank buffers per side-pot layer.

# Acceptance Criteria

- `resolve_terminated_games()` awards the main pot and each side pot only to seats that invested into that layer and are still `ACTIVE` or `ALLIN`.
- A short-stack winner cannot receive chips from contributions above that seat’s total investment.
- Folded seats never win showdown chips but their committed chips remain distributable to eligible players.
- Multiple side-pot tiers in the same hand resolve in order without chip loss.
- Batched showdown resolution still works when different games require different payout structures.
- All tests in `tests/poker/test_poker_gpu_side_pot_showdown.py` pass.
- Existing non-side-pot showdown tests in `tests/poker/test_poker_gpu_showdown.py` remain green.

# Test Plan

Unit and regression targets:
- `tests/poker/test_poker_gpu_side_pot_showdown.py`: new direct regression coverage for side-pot behavior.
- `tests/poker/test_poker_gpu_showdown.py`: existing baseline coverage for strongest-hand and split-pot showdowns.

Validation steps:
- Run `pytest tests/poker/test_poker_gpu_side_pot_showdown.py -q`.
- Run `pytest tests/poker/test_poker_gpu_showdown.py -q`.

Coverage gap intentionally not addressed in this ticket:
- No new test currently exercises forced turn/flop reveal plus side-pot resolution from `resolve_terminated_games()` in the same hand. The current change still preserves that code path because only payout distribution changes after board completion.

# Rollout, Recovery, and Monitoring Plan

Rollout:
- Land as a bug-fix change scoped to GPU showdown resolution only.
- Rely on targeted regression coverage before merge because the behavior change is fully local to terminal payout accounting.

Recovery:
- If regression appears, revert the `resolve_terminated_games()` payout block in `environments/Poker/PokerGPU.py` and the associated test file additions.
- Because the change is isolated to showdown resolution, rollback risk is low and does not require data migration.

Monitoring:
- Watch for test failures in `tests/poker/test_poker_gpu_side_pot_showdown.py` and `tests/poker/test_poker_gpu_showdown.py`.
- In downstream training runs, bankroll drift or impossible chip totals after all-in hands would indicate payout regression.

# Open Questions and Explicit Assumptions

Assumptions:
- `self.pots[game]` equals `self.total_invested[game].sum()` for resolved showdown hands; the implementation will derive payout layers from `self.total_invested` and zero the pot after distribution.
- Remainder chips on split pots should be assigned deterministically in seat order, matching the CPU environment’s current behavior closely enough for parity.
- Unmatched overbets are not expected to survive into `resolve_terminated_games()` because betting logic caps actual committed chips during action execution.

Open question:
- The ticket text does not provide a GitHub issue number, so the PR body will need a placeholder or an omitted closing reference unless that number is discoverable from repo context.
