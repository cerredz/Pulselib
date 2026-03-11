# PokerGPU Preflop All-In Resolver Specification

## Task Definition and Objectives
Fix `environments/Poker/PokerGPU.py::resolve_terminated_games()` so a terminated showdown that starts from preflop (`stage == 0`) burns and deals flop, turn, and river before showdown evaluation. The objective is to prevent silent settlement against the undealt placeholder board `[-1, -1, -1, -1, -1]` and make direct resolver calls produce valid showdown outcomes for preflop all-in terminal states.

## In-Scope / Out-of-Scope
In scope:
- Modify `environments/Poker/PokerGPU.py` to add the missing stage-0 board runout branch inside `resolve_terminated_games()`.
- Add a dedicated regression file at `tests/poker/test_poker_gpu_preflop_allin_resolver.py` with five distinct direct-resolver cases.
- Validate compatibility with the existing showdown and round-progression tests already present on `main`.

Out of scope:
- Refactoring showdown payout behavior beyond the current whole-pot winner logic on `main`.
- Changing betting semantics, action encoding, or `step()` turn-order behavior.
- Modifying CPU poker environment code.
- Broader performance or training-loop changes.

## Current State and Architecture Context
- `environments/Poker/PokerGPU.py::step()` already burns and deals board cards street by street during normal progression.
- `environments/Poker/PokerGPU.py::resolve_terminated_games()` currently fills missing streets only for `stage == 1` and `stage == 2`, then evaluates the showdown immediately.
- `resolve_fold_winners()` handles single-survivor terminal hands before showdown resolution, so `resolve_terminated_games()` should only deal a board when multiple seats remain eligible.
- Existing `tests/poker/test_poker_gpu_showdown.py` covers river-stage showdown payouts, and `tests/poker/test_poker_gpu_round_progression.py` covers street dealing, but neither file covers a direct preflop all-in termination.

## Proposed Design and Integration Plan
1. Modify `environments/Poker/PokerGPU.py` inside `resolve_terminated_games()` to add `preflop_mask = (self.stages[g_res] == 0) & multiple_players`.
Why: this is the missing root-cause branch from the ticket.
Downstream impact: preflop terminated showdown games receive a valid five-card board before the existing ranking logic runs.
Potential issues: incorrect mask construction could deal boards for single-survivor hands or wrong stages.
Validation: `pytest tests/poker/test_poker_gpu_preflop_allin_resolver.py -q`.

2. Reuse the exact `step()` burn/deal order for preflop resolver completion:
- burn one, deal flop to `board[:, 0:3]`
- burn one, deal turn to `board[:, 3]`
- burn one, deal river to `board[:, 4]`
Why: direct resolution should produce the same board state normal street progression would have produced from the same deck position.
Downstream impact: `self.deck_positions` stays consistent with existing environment semantics.
Potential issues: forgetting that `deal_cards()` already increments `self.deck_positions` will corrupt deck advancement.
Validation: regression assertions on both final board contents and final `deck_positions`.

3. Add `tests/poker/test_poker_gpu_preflop_allin_resolver.py` with five cases:
- happy path: full preflop runout deals all five cards
- correctness: actual stronger hand wins after the runout
- boundary: tied runout still splits correctly once cards are dealt
- integration: batched preflop and river showdowns resolve in one call without mutating the river board
- guard rail: single-survivor preflop terminal hands do not trigger showdown dealing
Why: these cases cover the bug, a payout correctness check, a split-pot boundary, a mixed-stage batch, and the single-survivor non-showdown path.
Downstream impact: future resolver regressions fail fast.
Potential issues: deck fixtures must avoid duplicate cards and accidental hidden stronger hands.
Validation: each test asserts exact board and stack outcomes.

4. Leave `tests/poker/test_poker_gpu_showdown.py` unchanged and use it as backward-compatibility validation.
Why: the fix should not alter river-stage showdown behavior on `main`.
Downstream impact: limits ticket scope and keeps the branch reviewable.
Potential issues: if these tests fail after the stage-0 change, the resolver logic has regressed beyond the ticket scope.
Validation: `pytest tests/poker/test_poker_gpu_showdown.py -q`.

## Data and API Contract Changes
No public API change.

Internal behavior change:
- Before: stage-0 terminated showdowns could be evaluated with an undealt board.
- After: stage-0 terminated multi-player showdowns are completed to five board cards before hand ranking.

Tensor contract details:
- `self.board`: still `[n_games, 5]`, `int32`, device-local; stage-0 resolved showdown games must exit with all five entries populated.
- `self.deck_positions`: still `int32`; stage-0 resolved showdown games advance by 8 cards from the post-hole-card position.
- `self.stacks`, `self.pots`, `self.stages`: unchanged shape/type, but now reflect correct direct preflop showdown results.

## Edge Cases and Failure Modes
- Direct preflop two-player all-in should fill the entire board. Covered by `test_resolve_terminated_games_deals_full_board_for_preflop_all_in_showdown`.
- Direct preflop two-player all-in should award the actual winner instead of a placeholder-board artifact. Covered by `test_resolve_terminated_games_preflop_all_in_uses_runout_to_pick_winner`.
- Direct preflop tied showdown should still split the pot correctly after runout. Covered by `test_resolve_terminated_games_preflop_all_in_splits_tied_runout`.
- Mixed-stage batched resolution should only deal the preflop board and leave a river board unchanged. Covered by `test_resolve_terminated_games_handles_batched_preflop_and_river_showdowns`.
- Single-survivor terminal states should not enter the showdown deal path. Covered by `test_resolve_terminated_games_does_not_run_preflop_board_when_single_survivor_remains`.

Failure modes to guard against:
- Dealing only turn and river for preflop states.
- Misordered burn cards relative to `step()`.
- Mutating river-stage boards in mixed batches.
- Dealing a board for a fold-win state that should remain untouched here.

## Security, Reliability, and Performance Considerations
Security:
- No new I/O, subprocess, or external-input surface is introduced.

Reliability:
- Keep all operations on `self.device`; the fix is device-local tensor logic only.
- Preserve existing showdown ranking and payout flow after the board is completed.
- Match the `step()` path’s burn/deal order exactly so direct resolution and normal progression remain semantically consistent.

Performance:
- The expensive hand-ranking path remains unchanged and batched.
- The new preflop branch is just one additional masked tensor path on terminated hands, not a hot-path loop.
- No Python loops over games or players are added by the fix itself.

PyTorch best-practice cross-reference:
- Device placement remains explicit and local to `self.device`.
- Tensor operations stay vectorized across the masked game set.
- Memory management reuses existing board and deck buffers in place.
- No float-heavy or numerically sensitive math is introduced.

## Acceptance Criteria
- `resolve_terminated_games()` burns and deals flop, turn, and river for terminated preflop multi-player showdowns.
- The resulting board matches the deterministic deck order and advances `self.deck_positions` consistently with existing street progression.
- Preflop showdown payouts depend on the dealt board rather than placeholder values.
- Existing river-stage showdown behavior remains unchanged.
- Single-survivor terminal preflop states do not trigger showdown dealing.
- `tests/poker/test_poker_gpu_preflop_allin_resolver.py` passes.
- `tests/poker/test_poker_gpu_showdown.py` and `tests/poker/test_poker_gpu_round_progression.py` remain green.

## Test Plan
- `pytest tests/poker/test_poker_gpu_preflop_allin_resolver.py -q`
- `pytest tests/poker/test_poker_gpu_showdown.py -q`
- `pytest tests/poker/test_poker_gpu_round_progression.py -q`

Coverage gap not addressed here:
- This ticket does not compare `step()` and `resolve_terminated_games()` against the exact same seeded all-in trajectory end to end; it confines the regression surface to the direct resolver path.

## Rollout, Recovery, and Monitoring Plan
Rollout:
- Land the resolver change and new regression file together as a single bug-fix PR.

Recovery:
- Revert the stage-0 branch and the new regression file if any downstream regression appears.
- Because the change is local to terminal board completion, rollback does not require data migration.

Monitoring:
- Watch the targeted pytest suites above.
- In downstream usage, a reappearance of `[-1, -1, -1, -1, -1]` after direct preflop terminal resolution indicates regression.

## Open Questions and Explicit Assumptions
Assumptions:
- `self.deck_positions` already points at the next undealt card after hole cards when direct preflop resolution is invoked.
- `resolve_fold_winners()` remains the correct owner of single-survivor hands.
- The `main` branch’s existing whole-pot showdown payout behavior is the correct baseline for this ticket; the bug is board completion, not payout structure.

Open questions:
- None blocking for this option-A implementation.
