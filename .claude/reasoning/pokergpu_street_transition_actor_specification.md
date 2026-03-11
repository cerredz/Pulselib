# PokerGPU Street Transition Actor Specification

## Task Definition and Objectives
Fix `environments/Poker/PokerGPU.py` so that when `PokerGPU.step()` advances from one betting street to the next, `self.idx` is reassigned to the first legal acting seat to the left of the button instead of leaking the previous street's last actor. The objective is to restore legal betting order, emit the correct next observation, and keep `info["seat_idx"]` aligned with the real acting player for downstream policy selection in `environments/Poker/utils.py`.

## In-Scope / Out-of-Scope
In scope:
- `environments/Poker/PokerGPU.py`: update street-transition actor selection.
- `tests/poker/test_poker_gpu_street_actor_reset.py`: add regression coverage for flop/turn/river, skipped inactive seats, heads-up, and batched transitions.

Out of scope:
- Blind posting or preflop actor-selection semantics in `reset()`.
- Showdown, side-pot, reward, or equity logic beyond the actor-order side effects already covered by the regression tests.
- Training-loop changes in `scripts/Poker/trainGPU.py`; the contract there remains unchanged and should benefit automatically once `seat_idx` is corrected.

## Current State and Architecture Context
- `environments/Poker/PokerGPU.py` computes the current-street next actor by scanning `candidate_seats` from `self.idx + 1` and only updates `self.idx` when the street continues.
- When `self.is_round_over` becomes true, the transition block resets `self.stages`, `self.highest`, `self.agg`, `self.acted`, and the board cards, but it never reassigns `self.idx`.
- `PokerGPU.get_obs()` reads `self.idx` to expose the current seat's hole cards, position, stack, and call cost, so stale `self.idx` immediately corrupts the observation returned from `step()`.
- `PokerGPU.get_info()` returns `seat_idx`, and `environments/Poker/utils.py::build_actions` uses that seat index to route observations to the correct agent. A stale `idx` therefore misroutes both environment state and chosen actions.
- Existing `tests/poker/test_poker_gpu_round_progression.py` covers street advancement and early termination, but it does not assert the next actor after a street transition.

## Proposed Design and Integration Plan
Task 1:
- File: `C:\Users\422mi\Pulselib\tests\poker\test_poker_gpu_street_actor_reset.py`
- Change: add five regression tests that force a round-ending action and assert the post-transition actor through `env.idx`, `info["seat_idx"]`, and the returned observation when relevant.
- Why: the ticket is specifically about silent actor-order corruption; the tests define observable pass/fail behavior before implementation.
- Downstream impact: validates the `PokerGPU.get_obs()` and `PokerGPU.get_info()` contracts consumed by `environments/Poker/utils.py`.
- Potential issues: tests must force round-end states precisely; otherwise they can accidentally exercise the existing in-street actor scan instead of the transition path.
- Validation: run `pytest tests/poker/test_poker_gpu_street_actor_reset.py -q` and confirm the tests fail before implementation, then pass after the fix.

Task 2:
- File: `C:\Users\422mi\Pulselib\environments\Poker\PokerGPU.py`
- Change: add a small helper with a typed signature that computes the first active seat to the left of the button for a boolean game mask, then call it from the street-transition block before returning observations.
- Why: this removes the stale-actor bug while keeping actor-order logic vectorized across games. A helper is preferable to open-coded duplication because the same seat-selection rule may need reuse and is easier to reason about in isolation.
- Downstream impact: `get_obs()` and `get_info()` will now emit the correct next seat after flop/turn/river transitions, which keeps `build_actions` aligned with the legal actor.
- Potential issues: the helper must remain device-local, avoid implicit CPU transfers, and only target games in `transition_mask`; it must not disturb hands that end early or post-river games marked done.
- Validation: run the new regression file plus `tests/poker/test_poker_gpu_round_progression.py` to verify no behavioral regression in existing round-progression coverage.

Task 3:
- File: `C:\Users\422mi\Pulselib\environments\Poker\PokerGPU.py`
- Change: leave the current in-street next-actor scan intact and update only the transition path so that the bug fix is narrowly scoped.
- Why: the ticket identifies the missing reassignment only after `transition_mask`; changing the current-street scan would widen risk without evidence.
- Downstream impact: preserves existing round-continuation behavior already covered by `tests/poker/test_poker_gpu_round_progression.py`.
- Potential issues: if the helper is wired into the wrong branch, it could accidentally change pre-existing in-street action order.
- Validation: ensure the mixed batched regression still shows one game transitioning and another continuing with the legacy current-street scan.

## Data and API Contract Changes
- No schema or public API shape changes.
- Behavioral contract change:
  - Before: after a street transition, `get_obs()` and `get_info()["seat_idx"]` could still point at the previous street's last actor.
  - After: after a street transition, both contracts point at the first active seat to the left of the button for that game.
- Tensor shapes remain unchanged. `self.idx` stays `torch.int32` on `self.device`.

## Edge Cases and Failure Modes
- All seats active on transition: action must restart at the immediate seat left of the button.
  - Covered by `test_flop_transition_restarts_action_from_first_seat_left_of_button`.
- Folded seat immediately left of the button: actor selection must skip folded seats.
  - Covered by `test_turn_transition_skips_folded_seats_left_of_button`.
- All-in seat immediately left of the button: actor selection must skip non-active seats even when they remain in the hand.
  - Covered by `test_river_transition_skips_all_in_seats_left_of_button`.
- Heads-up boundary: with only two active seats, the next street must move to the only other active seat rather than leaving the closer.
  - Covered by `test_heads_up_transition_uses_only_other_active_seat`.
- Mixed batched execution: only transitioned games should reset to the button-left actor; games that continue the same street must keep the existing candidate scan behavior.
  - Covered by `test_batched_transition_resets_only_finished_rounds_and_updates_observation`.

## Security, Reliability, and Performance Considerations
- Security: no new external inputs, I/O, or privilege boundaries are introduced.
- Reliability: this is a silent correctness bug in a core rules engine. Fixing it reduces RL data corruption and illegal action routing without changing external interfaces.
- Performance:
  - Keep the solution vectorized with tensor operations over masked games; do not introduce Python loops over `n_games` or seat dimensions in the hot path.
  - Keep all computations on `self.device` and avoid `.item()`, `.cpu()`, or `.numpy()` inside `step()`.
  - Reuse existing tensors such as `self.active_player_idx` where practical to avoid unnecessary allocations.
- PyTorch best-practice cross-check:
  - Device placement: all new tensors must be created on `self.device`.
  - Tensor operations: the actor scan should use broadcasting/masking rather than Python loops.
  - Numerical stability: not applicable to this integer seat-selection path.
  - Distributed/multi-GPU: no DDP changes are needed; the fix must remain batch-safe across many simultaneous games.
  - Memory management: no additional persistent buffers are required unless a reusable helper can operate on existing tensors.

## Acceptance Criteria
- `pytest tests/poker/test_poker_gpu_street_actor_reset.py -q` passes with all five tests green.
- `pytest tests/poker/test_poker_gpu_round_progression.py -q` still passes.
- After any flop/turn/river transition, `env.idx` equals the first active seat left of the button for transitioned games.
- `step()` returns observations and `info["seat_idx"]` that match the corrected actor immediately after the transition.
- Games that do not transition streets in the same batch retain the existing current-street actor-selection behavior.

## Test Plan
- New regression coverage:
  - `C:\Users\422mi\Pulselib\tests\poker\test_poker_gpu_street_actor_reset.py`
  - Targets flop, turn, river, heads-up, skipped inactive seats, and batched mixed-state transitions.
- Existing regression coverage to re-run:
  - `C:\Users\422mi\Pulselib\tests\poker\test_poker_gpu_round_progression.py`
  - Confirms the narrow fix does not break current round-progression expectations.
- Coverage gap intentionally left open:
  - No additional GPU benchmark or full training-stack test is required for this ticket because the interface does not change and the bug is reproducible in CPU unit tests.

## Rollout, Recovery, and Monitoring Plan
- Rollout: ship as a narrow bug fix in `PokerGPU.step()` with regression tests included in the same change set.
- Recovery: if the fix regresses actor ordering, revert the helper call in the transition path and rerun the new regression file to confirm the failure returns to the prior known state.
- Monitoring signals:
  - Unit test failures in the new regression file.
  - Any downstream failures in heuristic-agent tests caused by mismatched `seat_idx`.
  - Manual symptom: `step()` returns board cards for a new street while `seat_idx` still names the prior closer.

## Open Questions and Explicit Assumptions
- Assumption: the intended rule is exactly "first active seat to the left of the button" for every post-preflop street, including heads-up in this environment's current blind model.
- Assumption: only seats with `status == ACTIVE` are eligible to act on a new street; `FOLDED`, `ALLIN`, and `SITOUT` seats must be skipped.
- Assumption: no clarifying question is blocking implementation because the ticket already defines the target actor-selection rule precisely.
- Open question not addressed here: `reset()` uses a simplified heads-up blind assignment; this ticket does not change that behavior.
