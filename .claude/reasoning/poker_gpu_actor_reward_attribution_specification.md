# PokerGPU Actor Reward Attribution Specification

## Task Definition and Objectives
Fix `PokerGPU.step()` so action rewards are always attributed to the acting seat, even after `self.idx` is advanced to the next player or reset during a street transition. The objective is to preserve the RL invariant that a reward depends on the actor's own pre/post-action state, not on a later cursor value consumed by `poker_reward_gpu()`.

## In-Scope / Out-of-Scope
In scope:
- Modify the GPU poker environment reward path in [environments/Poker/PokerGPU.py](/C:/Users/422mi/Pulselib/environments/Poker/PokerGPU.py) so reward computation reads the acting seat explicitly.
- Add regression coverage in [tests/poker/test_poker_gpu_actor_reward_attribution.py](/C:/Users/422mi/Pulselib/tests/poker/test_poker_gpu_actor_reward_attribution.py) for continuing actions, skipped-seat turn advancement, street transitions, and batched execution.
- Validate that the corrected reward still flows unchanged into the existing Q-learning update path in [scripts/Poker/trainGPU.py](/C:/Users/422mi/Pulselib/scripts/Poker/trainGPU.py).

Out of scope:
- Changing the numeric reward formula itself.
- Refactoring the broader environment API beyond the actor-index input needed for reward attribution.
- Modifying Q-learning target logic in [environments/Poker/Player.py](/C:/Users/422mi/Pulselib/environments/Poker/Player.py).
- Addressing unrelated environment issues, optimizer behavior, or multi-GPU training architecture.

## Current State and Architecture Context
- [environments/Poker/PokerGPU.py](/C:/Users/422mi/Pulselib/environments/Poker/PokerGPU.py) captures actor-local buffers with `self.prev_stacks` and `self.prev_invested` before action execution, but `step()` later mutates `self.idx` during next-player selection and street transitions before calling `poker_reward_gpu()`.
- `poker_reward_gpu()` currently reads `self.equities[self.g, self.idx]`, which makes the reward dependent on the post-step cursor instead of the acting seat.
- [scripts/Poker/trainGPU.py](/C:/Users/422mi/Pulselib/scripts/Poker/trainGPU.py) feeds `rewards` directly into `PokerQNetwork.train_step`, so any misattribution corrupts TD targets without further validation.
- [environments/Poker/Player.py](/C:/Users/422mi/Pulselib/environments/Poker/Player.py) assumes the reward tensor it receives is already semantically correct; no downstream correction exists.
- Existing tests in [tests/poker/test_poker_gpu_no_actor_rewards.py](/C:/Users/422mi/Pulselib/tests/poker/test_poker_gpu_no_actor_rewards.py) cover zero-reward no-actor transitions, and [tests/poker/test_poker_gpu_street_actor_reset.py](/C:/Users/422mi/Pulselib/tests/poker/test_poker_gpu_street_actor_reset.py) covers cursor reset semantics, but neither asserts that rewards are actor-indexed after cursor mutation.

## Proposed Design and Integration Plan
1. Modify [environments/Poker/PokerGPU.py](/C:/Users/422mi/Pulselib/environments/Poker/PokerGPU.py) `step()` to snapshot the acting seat with `actor_idx = self.idx.clone()` before any mutation.
Why: `self.prev_invested` and `self.prev_stacks` already snapshot actor-local state; reward attribution must use the same actor identity.
Downstream impact: `poker_reward_gpu()` becomes actor-explicit, while the return shape to callers remains unchanged.
Potential issues: forgetting to clone would alias the mutable cursor and preserve the bug. Validation: regression tests in [tests/poker/test_poker_gpu_actor_reward_attribution.py](/C:/Users/422mi/Pulselib/tests/poker/test_poker_gpu_actor_reward_attribution.py) fail before this change and pass after it.

2. Modify [environments/Poker/PokerGPU.py](/C:/Users/422mi/Pulselib/environments/Poker/PokerGPU.py) `poker_reward_gpu()` to accept `actor_idx: torch.Tensor` and index `self.equities[self.g, actor_idx]` instead of `self.idx`.
Why: making actor identity an explicit input removes hidden dependence on mutable environment state.
Downstream impact: only the internal caller in `step()` changes; external API consumers still receive `(obs, rewards, dones, truncated, info)`.
Potential issues: dtype or device mismatch if `actor_idx` leaves the current device. Validation: run the new reward-attribution tests plus the existing poker GPU regressions to ensure shape and device behavior remain stable.

3. Keep the reward formula tensorized in [environments/Poker/PokerGPU.py](/C:/Users/422mi/Pulselib/environments/Poker/PokerGPU.py) with no Python loops over games.
Why: the environment is batched; vectorized indexing is the correct PyTorch pattern for both correctness and throughput.
Downstream impact: no training-loop synchronization changes in [scripts/Poker/trainGPU.py](/C:/Users/422mi/Pulselib/scripts/Poker/trainGPU.py).
Potential issues: introducing per-game Python control flow would degrade throughput and violate the batch design. Validation: batched regression `test_batched_rewards_follow_each_games_actor_instead_of_post_step_cursor`.

4. Add [tests/poker/test_poker_gpu_actor_reward_attribution.py](/C:/Users/422mi/Pulselib/tests/poker/test_poker_gpu_actor_reward_attribution.py) with five concrete regressions:
- continuing flop call with next-player hand changes
- continuing turn fold with an all-in side player
- raise path that skips a folded seat while selecting the next actor
- street transition that resets `idx`
- batched multi-game reward attribution
Why: the ticket reports both ordinary turn advancement and street-transition cursor changes, and training uses batched rewards.
Downstream impact: protects future refactors of reward computation and turn-order logic.
Potential issues: tests that only compare two buggy code paths would be weak, so each case also compares against a manual actor-index reward calculation. Validation: targeted pytest run for the new file.

## Data and API Contract Changes
- Internal function contract change in [environments/Poker/PokerGPU.py](/C:/Users/422mi/Pulselib/environments/Poker/PokerGPU.py):
Before: `def poker_reward_gpu(self, actions):`
After: `def poker_reward_gpu(self, actions: torch.Tensor, actor_idx: torch.Tensor) -> torch.Tensor:`
- External environment API contract:
No change. `step()` still returns the same five-tuple and `scripts/Poker/trainGPU.py` continues consuming `rewards` without modification.
- Tensor semantics:
Before: reward equity source was implicitly `self.idx` after cursor mutation.
After: reward equity source is explicitly the pre-mutation acting seat for each game.

## Edge Cases and Failure Modes
- Continuing non-terminal action where `self.idx` advances to the next seat. Covered by `test_call_reward_stays_with_actor_when_next_players_flop_equity_changes`.
- Fold action with remaining active/all-in players, where reward must not inherit the next player's turn equity. Covered by `test_fold_reward_ignores_next_players_turn_equity_with_all_in_side_player`.
- Turn-order scan skips folded seats before picking the next actor. Covered by `test_raise_reward_uses_actor_equity_when_next_active_seat_skips_folded_player`.
- Street transition resets `idx` to the first active seat on the next street before reward calculation. Covered by `test_reward_uses_previous_actor_equity_after_street_transition_reassigns_idx`.
- Batched execution where different games have different actors and different post-step cursors. Covered by `test_batched_rewards_follow_each_games_actor_instead_of_post_step_cursor`.

## Security, Reliability, and Performance Considerations
- Security: no new I/O, serialization, or external inputs are introduced.
- Reliability: actor identity is captured once per step and passed explicitly, removing a silent state-coupling failure mode in core RL reward generation.
- Performance: the fix must remain device-local and batched. `actor_idx` should stay as a tensor on `self.device`; do not introduce `.item()`, `.cpu()`, or NumPy conversions inside environment hot paths.
- Memory management: `actor_idx = self.idx.clone()` adds one small per-step tensor copy of shape `[n_games]`, which is acceptable and avoids aliasing. Do not create extra per-player copies of equities or status.
- Numerical stability: preserve the existing `torch.tanh` reward shaping and `1e-6` denominator guard. No manual softmax/log operations are added.
- PyTorch best-practice alignment:
  - Vectorized tensor indexing is preserved instead of Python loops.
  - Device placement remains explicit through tensors already resident on `self.device`.
  - No CPU/GPU synchronization is added in the training loop or reward path.
  - No DDP/AMP changes are required because this ticket only fixes environment-side reward attribution.

## Acceptance Criteria
- `PokerGPU.step()` returns identical rewards when only the next acting player's hole cards change and the acting player's state is fixed.
- Reward values match a manual actor-index calculation for call, fold, and raise actions in postflop states.
- Rewards remain actor-correct when next-player selection skips folded seats.
- Rewards remain actor-correct when the action ends the street and `idx` is reassigned for the next street.
- Batched `step()` calls attribute each game's reward to that game's acting seat.
- [tests/poker/test_poker_gpu_actor_reward_attribution.py](/C:/Users/422mi/Pulselib/tests/poker/test_poker_gpu_actor_reward_attribution.py) passes in full.

## Test Plan
- Unit/regression:
  - Run `pytest tests/poker/test_poker_gpu_actor_reward_attribution.py`.
  - Run `pytest tests/poker/test_poker_gpu_no_actor_rewards.py`.
  - Run `pytest tests/poker/test_poker_gpu_street_actor_reset.py`.
- Integration:
  - Confirm [scripts/Poker/trainGPU.py](/C:/Users/422mi/Pulselib/scripts/Poker/trainGPU.py) requires no code changes because the reward tensor shape and environment API are unchanged.
- Coverage gaps:
  - No full training benchmark is required for this bugfix, so longer-horizon learning improvements remain observational rather than directly tested here.

## Rollout, Recovery, and Monitoring Plan
- Rollout: land the environment change and regression tests together in one commit so the invariant is enforced atomically.
- Recovery: if unexpected regressions appear, revert the `actor_idx` plumbing in [environments/Poker/PokerGPU.py](/C:/Users/422mi/Pulselib/environments/Poker/PokerGPU.py) and rerun the prior poker GPU regression suite.
- Monitoring:
  - In local validation, compare reward outputs across duplicated states that only vary the next player's hole cards.
  - In subsequent training runs, watch for shifts in reward distributions and Q-loss stability in the logs emitted by [environments/Poker/Player.py](/C:/Users/422mi/Pulselib/environments/Poker/Player.py).

## Open Questions and Explicit Assumptions
- Assumption: the intended reward input equity is the actor's pre-step equity under the board state that existed before the action, not a recomputed post-transition equity after turn/river dealing.
- Assumption: no caller outside [environments/Poker/PokerGPU.py](/C:/Users/422mi/Pulselib/environments/Poker/PokerGPU.py) invokes `poker_reward_gpu()` directly in production code.
- Assumption: preserving the existing reward formula is correct; only actor attribution is wrong.
- Open question kept non-blocking: whether a future refactor should also snapshot actor status explicitly alongside `prev_invested` for additional defensive clarity. This is not required for the current ticket because the reported corruption comes from seat indexing, not status mutation.
