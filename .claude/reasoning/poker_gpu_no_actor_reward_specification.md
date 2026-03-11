Task Definition and Objectives

This task fixes a reward-contract bug in [`environments/Poker/PokerGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/environments/Poker/PokerGPU.py) where `PokerGPU.step()` auto-advances all-in runoff states with no legal actor, but still forwards the raw placeholder action tensor into `poker_reward_gpu()`. The objective is to implement ticket option B: detect games that have no legal actor during `step()` and bypass action scoring for those games so that no-decision states produce no action-dependent reward. The change must preserve normal reward semantics for games with a legal actor and must remain batched and GPU-friendly.

In-Scope / Out-of-Scope

In scope:
- Modify [`environments/Poker/PokerGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/environments/Poker/PokerGPU.py) so `step()` masks out action scoring for games with no legal actor.
- Add regression coverage in [`tests/poker/test_poker_gpu_no_actor_rewards.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/tests/poker/test_poker_gpu_no_actor_rewards.py) for preflop auto-runout, state-transition invariance, river showdown resolution, mixed batched execution, and normal legal-actor reward behavior.
- Document downstream implications for [`scripts/Poker/trainGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/scripts/Poker/trainGPU.py) and [`environments/Poker/Player.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/environments/Poker/Player.py) without changing their interfaces.

Out of scope:
- Changing the trainer to skip `env.step(actions)` when a game is auto-advancing.
- Adding a new `info` legality mask or changing the `gym.Env.step()` return signature.
- Refactoring `poker_reward_gpu()` into a different reward model.
- Fixing unrelated in-progress work in the original dirty worktree.

Current State and Architecture Context

[`environments/Poker/PokerGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/environments/Poker/PokerGPU.py) currently has split responsibility:
- `execute_actions()` already computes an `active_mask` and mutates state only for seats that can legally act.
- `step()` computes `truly_active = (self.status == self.ACTIVE).sum(dim=1)` and sets `all_allin_or_folded = (truly_active == 0)`, which is the exact auto-runout condition described in the ticket.
- `step()` then advances streets or showdown for those games even though no seat acted.
- `poker_reward_gpu(actions)` derives reward exclusively from raw action ids and current tensors; it does not know whether a game had a legal actor.

Downstream consumers:
- [`scripts/Poker/trainGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/scripts/Poker/trainGPU.py) always fills an action tensor and always consumes the returned rewards inside Q-learning updates.
- [`environments/Poker/Player.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/environments/Poker/Player.py) uses those rewards directly in `train_step()`, so the environment reward contract is the correct fix point.

Proposed Design and Integration Plan

1. Modify [`tests/poker/test_poker_gpu_no_actor_rewards.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/tests/poker/test_poker_gpu_no_actor_rewards.py) to define the contract before implementation.
Why: The ticket explicitly describes a reward mismatch between execution and scoring paths. The tests encode the expected behavior for both no-actor and legal-actor states.
Adjacent impact: No runtime modules are affected; this file constrains behavior in [`environments/Poker/PokerGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/environments/Poker/PokerGPU.py).
Potential issues: Tests could accidentally depend on unrelated actor-progression behavior if they assert too much about `idx`; keep assertions focused on reward semantics and auto-runout progression.
Validation: `pytest tests\\poker\\test_poker_gpu_no_actor_rewards.py -q` must fail before the fix and pass after it.

2. Modify [`environments/Poker/PokerGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/environments/Poker/PokerGPU.py) inside `step()` to derive a batched `has_legal_actor` mask before `execute_actions()` and use it after reward computation to zero-out no-actor games.
Why: Option B belongs in control flow, not in trainer code. `step()` already computes the round-level auto-runout condition, so it is the narrowest place to enforce that auto-runout steps are not scored as decisions.
Specific change:
- Derive `has_legal_actor` from the same seat-specific legality criteria used by `execute_actions()`: current seat status is not `FOLDED`, `ALLIN`, or `SITOUT`, and game is not already done.
- Keep `execute_actions(actions)` unchanged so the execution path remains single-source for action mutation.
- After `rewards = self.poker_reward_gpu(actions=actions)`, set `rewards[~has_legal_actor] = 0`.
- Keep `rewards[prev_done] = 0` semantics by incorporating `prev_done` into the same mask.
Adjacent impact: [`scripts/Poker/trainGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/scripts/Poker/trainGPU.py) and [`environments/Poker/Player.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/environments/Poker/Player.py) will receive corrected zero rewards without interface changes.
Potential issues:
- If `has_legal_actor` is computed after `execute_actions()`, legality could change because the action mutated status or `is_done`; compute it before action execution.
- If the mask is derived from `truly_active == 0` only, it will miss other no-actor states where the indexed seat is not legally allowed to act; use seat-level legality.
- Avoid allocating Python lists or syncing tensors to CPU in the hot path; keep the mask as a device tensor.
Validation: All five new tests must pass, especially the batched mixed-state regression and the legal-actor control test.

3. Leave [`scripts/Poker/trainGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/scripts/Poker/trainGPU.py) unchanged, but explicitly document why.
Why: The trainer submits placeholder actions every tick by design. Once the environment zeroes no-actor rewards, the trainer no longer ingests label noise for those steps.
Adjacent impact: `build_actions()` in [`environments/Poker/utils.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/environments/Poker/utils.py) continues to fill all games; no API break.
Potential issues: Future callers might still want a legality mask for analytics, but that is a separate contract change and out of scope here.
Validation: The mixed-batch test proves a placeholder action can coexist with a live action in one batched `env.step()` call without corrupting the reward tensor.

Data and API Contract Changes

There are no schema or public API changes.

Before:
- `PokerGPU.step(actions)` returned action-dependent rewards even when the current game had no legal actor.

After:
- `PokerGPU.step(actions)` returns zero reward for any game where no legal actor existed for that tick.
- The `info` payload remains unchanged.
- `poker_reward_gpu(actions)` signature remains unchanged; the gating happens at the `step()` control-flow layer.

Edge Cases and Failure Modes

- Preflop all-in runoff with `status = [ALLIN, ALLIN, FOLDED]` and placeholder actions `0`, `1`, or `12` must yield identical zero rewards while still progressing to the flop. Covered by `test_step_auto_runout_zeroes_placeholder_rewards_across_action_ids`.
- No-actor auto-runout must not leak placeholder-action differences into street dealing or pot state. Covered by `test_step_auto_runout_keeps_transition_state_identical_for_placeholder_actions`.
- River no-actor states must resolve showdown normally while producing zero action reward. Covered by `test_step_auto_runout_on_river_resolves_showdown_without_action_reward`.
- Mixed batches must zero only the no-actor rows and preserve real reward updates for legal actors in the same tensor operation. Covered by `test_step_batched_rewards_zero_only_games_without_legal_actor`.
- The fix must not flatten all reward semantics; legal-actor games must still differentiate between fold and call. Covered by `test_step_preserves_action_sensitive_rewards_when_a_legal_actor_exists`.

Security, Reliability, and Performance Considerations

- Memory management: the change uses one additional boolean tensor mask on the existing device. No host-device copies, no `.item()` calls, and no new long-lived buffers are required in the runtime path.
- GPU utilization: the mask is vectorized and batched, matching the current tensorized control flow. This follows the PyTorch constraint to avoid Python-level loops across batch dimension.
- Numerical stability: the fix only zeroes selected reward entries after the existing reward computation. It does not alter the `tanh`-based reward transform or introduce new floating-point instability.
- Precision tradeoffs: none. Reward values are either unchanged for legal-actor games or forced to exact zero for no-actor games.
- Reliability: placing the gate in `step()` keeps execution legality and reward legality aligned without adding a new downstream contract.
- Reproducibility: tests seed `random`, `numpy`, and `torch`, including `torch.cuda.manual_seed_all` when CUDA is available, so street-dealing comparisons remain deterministic.

Acceptance Criteria

- `pytest tests\\poker\\test_poker_gpu_no_actor_rewards.py -q` passes completely.
- `test_step_auto_runout_zeroes_placeholder_rewards_across_action_ids` proves placeholder actions `0`, `1`, and `12` all produce zero reward in a no-actor auto-runout state.
- `test_step_auto_runout_keeps_transition_state_identical_for_placeholder_actions` proves action ids do not alter the resulting auto-runout transition state.
- `test_step_auto_runout_on_river_resolves_showdown_without_action_reward` proves showdown resolution still occurs and rewards remain zero.
- `test_step_batched_rewards_zero_only_games_without_legal_actor` proves the fix is row-selective within a batch.
- `test_step_preserves_action_sensitive_rewards_when_a_legal_actor_exists` proves legal-actor reward semantics still differ by action.

Test Plan

Primary regression suite:
- [`tests/poker/test_poker_gpu_no_actor_rewards.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/tests/poker/test_poker_gpu_no_actor_rewards.py)

Targeted adjacent suites:
- [`tests/poker/test_poker_gpu_showdown.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/tests/poker/test_poker_gpu_showdown.py) to confirm showdown resolution remains correct.
- [`tests/poker/test_poker_gpu_round_progression.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/tests/poker/test_poker_gpu_round_progression.py) to confirm ordinary actor progression remains intact.

Coverage gap notes:
- No trainer-level integration test currently asserts that zeroed no-actor rewards reduce Q-learning label noise in [`scripts/Poker/trainGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/scripts/Poker/trainGPU.py). This task relies on the environment contract being correct.
- Multi-street repeated auto-runout beyond a single step is partially covered by transition-state assertions but not by a full training-loop scenario.

Rollout, Recovery, and Monitoring Plan

- Rollout: ship as a small environment-only bug fix. No migration or data backfill is needed because this changes online reward generation only.
- Recovery: if the fix introduces an unintended regression, revert the `step()` reward mask in [`environments/Poker/PokerGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/poker-gpu-no-actor-reward/environments/Poker/PokerGPU.py) and rerun the targeted poker tests.
- Monitoring: watch training metrics for reduced arbitrary reward spikes during all-in runoff hands and verify no unexpected collapse in non-zero reward frequency for normal decision states.

Open Questions and Explicit Assumptions

Open questions:
- No ticket number was provided, so the PR will need a placeholder or follow-up update for the `Closes #<ticket number>` line if a number is later supplied.

Explicit assumptions:
- The correct semantic for "no legal actor" is seat-level legality of the currently indexed seat before action execution, not merely `truly_active == 0`.
- The trainer should continue calling `env.step(actions)` for every game every tick; the environment is responsible for making no-actor ticks reward-neutral.
