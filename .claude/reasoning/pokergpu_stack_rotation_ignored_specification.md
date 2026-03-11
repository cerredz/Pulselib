# Task Definition and Objectives

Fix the persistent-stack seat-rotation regression in the GPU poker environment so bankroll state follows the same seat rotation that already applies to agent order during training. The target behavior is narrow and explicit: when `scripts/Poker/trainGPU.py` passes a non-zero seat rotation through `env.reset(options={"rotation": ...})`, `environments/Poker/PokerGPU.py` must roll `self.stacks` by that same value before blinds are posted for the next hand.

This task uses Option A from the ticket: read the rotation from `options` inside `PokerGPU.reset()` and apply it directly. The objective is to restore contract correctness without broadening the public API or redesigning the training loop.

# In-Scope / Out-of-Scope

In scope:
- Add regression coverage in `tests/poker/test_poker_gpu_reset_rotation.py` for the options-only reset path, zero rotation, large rotation values, invalid persistent stacks, and parity with the legacy positional argument.
- Modify `environments/Poker/PokerGPU.py` so persistent stacks are rolled using the rotation value carried in `options`.
- Preserve the existing trainer contract in `scripts/Poker/trainGPU.py` and the rotation computation in `environments/Poker/utils.py`.

Out of scope:
- Removing the positional `rotation` argument from `PokerGPU.reset()` (Option B).
- Updating the trainer to pass both `options["rotation"]` and `rotation=...` (Option C).
- Any changes to reward shaping, player rotation logic, episode termination, side-pot logic, or showdown behavior.
- Broader PyTorch architecture changes such as AMP, DDP, `torch.compile`, dataloader tuning, or seeding. Those are valid production concerns, but they are unrelated to this regression and would widen the ticket.

# Current State and Architecture Context

Current call path:
- `environments/Poker/utils.py:get_rotated_agents(...)` computes `rotation` and the new Q-agent seat from episode index and original agent order.
- `scripts/Poker/trainGPU.py:train_agent(...)` passes that rotation only through `env.reset(options={"rotation": rotations, ...})`.
- `environments/Poker/PokerGPU.py:reset(...)` checks `options["rotation"] != 0`, but applies `torch.roll(self.stacks, rotation, dims=1)` using the separate positional parameter `rotation`, which defaults to `0`.

Current behavior:
- First reset creates uniform stacks, so rotation is not observable.
- On later resets, if `self.stacks` already exists and `options["rotation"]` is non-zero, busted stacks (`== 0`) and over-cap stacks (`> max_bbs`) are restored to `starting_bbs`.
- The environment intends to rotate persistent stacks before dealing a new hand, but the actual roll amount stays `0` unless a caller redundantly passes the positional `rotation` argument.
- Blinds are then posted based on the next button position, which makes the missing rotation visible in the resulting bankroll order.

What will change:
- Only the source of the roll amount in `PokerGPU.reset()` will change. The environment will continue to sanitize persistent stacks before rolling them and posting blinds afterward.

What will not change:
- `get_rotated_agents(...)` remains the source of seat rotation.
- `trainGPU.py` continues to pass rotation via `options`.
- The legacy explicit `rotation` argument remains present and usable, but it is no longer required for the trainer path to work.

# Proposed Design and Integration Plan

1. Modify `tests/poker/test_poker_gpu_reset_rotation.py` to define the regression contract.
Specific change:
- Add five self-contained tests around `PokerGPU.reset()` with persistent stacks already initialized.
Why necessary:
- The ticket describes a silent correctness bug; the tests become the executable definition of done.
Adjacent modules affected:
- `environments/Poker/PokerGPU.py` through direct invocation.
- By implication, `scripts/Poker/trainGPU.py` because it depends on the options-only contract.
Potential issues:
- Expected stacks must account for both stack sanitation and blind posting order or the tests will assert the wrong contract.
Validation:
- `pytest tests/poker/test_poker_gpu_reset_rotation.py -q` must fail before the fix and pass after it.

2. Modify `environments/Poker/PokerGPU.py` inside `reset()` around the persistent-stack reuse branch.
Specific change:
- Read a local rotation value from `options` and use that value for `torch.roll(self.stacks, ..., dims=1)` instead of the unrelated positional argument.
Why necessary:
- This is the direct root-cause fix for the mismatch between the trainer’s options-based API and the environment’s reset implementation.
Adjacent modules affected:
- `scripts/Poker/trainGPU.py` because its existing `options={"rotation": rotations}` path becomes correct.
- `environments/Poker/utils.py` because its computed rotation again propagates through the existing trainer contract.
- Existing poker GPU tests, because they instantiate `PokerGPU` and rely on reset behavior.
Potential issues:
- Accidentally changing the order of stack sanitation, rotation, and blind posting would alter bankroll semantics.
- Using a CPU scalar extracted via `.item()` inside a GPU path would introduce an unnecessary sync. Avoid that.
Validation:
- `test_reset_rolls_persistent_stacks_from_options_rotation`
- `test_reset_options_rotation_matches_explicit_rotation_argument`
- `test_reset_zero_rotation_keeps_stack_order_before_blind_post`
- `test_reset_wraps_options_rotation_values_larger_than_table_size`
- `test_reset_restores_invalid_persistent_stacks_before_rotating`

3. Re-run adjacent poker GPU regressions to confirm the narrow fix does not regress reset/step behavior elsewhere.
Specific change:
- Re-run the new reset-rotation suite plus the existing poker GPU suites that already exercise `PokerGPU`.
Why necessary:
- This environment is heavily stateful; even a one-line bug fix should be checked against nearby behavior, especially around reset and showdown flow.
Adjacent modules affected:
- `tests/poker/test_poker_gpu_round_progression.py`
- `tests/poker/test_poker_gpu_showdown.py`
Potential issues:
- Hidden coupling to reset semantics could surface if any existing tests depended on the broken behavior.
Validation:
- All targeted poker tests pass together.

# Data and API Contract Changes

External API changes:
- None.

Internal behavioral contract change:
- `PokerGPU.reset(options={"rotation": N, ...})` now honors `N` for persistent-stack rotation even when the separate positional `rotation` argument is left at its default.

Before:
- Guard path: `if options and options['rotation'] != 0:`
- Applied roll amount: positional `rotation`, usually `0` in trainer calls.

After:
- Guard path and applied roll amount both use the same logical rotation carried through `options`.

Compatibility:
- Existing callers that pass `rotation` explicitly still work.
- Existing callers that pass rotation only in `options` now behave as intended.

# Edge Cases and Failure Modes

Edge case: non-zero options rotation on a reused stack tensor.
- Risk: bankroll order stays attached to old seat indices.
- Covered by `test_reset_rolls_persistent_stacks_from_options_rotation`.

Boundary case: zero rotation.
- Risk: the fix could accidentally rotate or otherwise disturb stack order when no seat change is requested.
- Covered by `test_reset_zero_rotation_keeps_stack_order_before_blind_post`.

Edge case: rotation magnitude larger than table size.
- Risk: options-based rotation might not wrap the same way as `torch.roll`.
- Covered by `test_reset_wraps_options_rotation_values_larger_than_table_size`.

Failure mode: broken parity between the options-only trainer path and the legacy explicit-argument path.
- Risk: two reset entry paths produce different bankroll state for the same logical rotation.
- Covered by `test_reset_options_rotation_matches_explicit_rotation_argument`.

Edge case: invalid persistent stacks (`0` or `> max_bbs`) on rotated reset.
- Risk: fixing rotation could accidentally reorder the sanitize-then-roll sequence and produce incorrect bankroll restoration.
- Covered by `test_reset_restores_invalid_persistent_stacks_before_rotating`.

# Security, Reliability, and Performance Considerations

Security:
- No new I/O, deserialization, network, or privilege boundary changes are introduced.

Reliability:
- This is a silent state-integrity bug in a training environment. The fix must preserve stack sanitation and blind posting order while aligning seat state with agent rotation.
- The regression suite explicitly checks the observable post-reset stacks instead of internal implementation details.

Performance:
- The change should remain on-device and use the existing `torch.roll` tensor operation; no Python loops or host-device copies are needed.
- Avoid `.item()` extraction or any CPU-side branching derived from tensor values in the reset hot path.
- The fix is constant-time relative to current behavior because `torch.roll` was already being called on the same tensor shape.

Numerical stability:
- No floating-point computation changes are introduced.

PyTorch best-practice notes:
- This implementation follows the constraint to prefer vectorized tensor ops over Python-level seat shuffling.
- Device placement remains explicit because `self.stacks` stays on `self.device`.
- No extra tensor copies beyond the existing `torch.roll` behavior are introduced.
- AMP/GradScaler, DDP, profiling, and seeding do not apply to this narrow environment-state fix and remain out of scope by design.

# Acceptance Criteria

- `PokerGPU.reset()` rolls persistent stacks by the rotation supplied in `options["rotation"]` when stacks are being reused.
- The options-only reset path produces the same stack order as the legacy explicit `rotation` argument for the same logical rotation.
- Zero rotation preserves stack order aside from the expected blind posting.
- Large rotation values wrap with `torch.roll` semantics.
- Invalid persistent stacks are restored before the rotation is applied.
- `tests/poker/test_poker_gpu_reset_rotation.py` passes in full.
- Existing poker GPU regression suites continue to pass.

# Test Plan

Primary regression file:
- `tests/poker/test_poker_gpu_reset_rotation.py`

Cases:
- Unit/integration: `test_reset_rolls_persistent_stacks_from_options_rotation`
- Integration/parity: `test_reset_options_rotation_matches_explicit_rotation_argument`
- Boundary: `test_reset_zero_rotation_keeps_stack_order_before_blind_post`
- Edge case: `test_reset_wraps_options_rotation_values_larger_than_table_size`
- Edge case/order-of-operations: `test_reset_restores_invalid_persistent_stacks_before_rotating`

Adjacent suites to rerun:
- `tests/poker/test_poker_gpu_round_progression.py`
- `tests/poker/test_poker_gpu_showdown.py`

Coverage gap:
- This change does not add a full end-to-end trainer integration test that runs multiple rotated episodes through `train_agent(...)`; the regression suite instead validates the exact environment contract that the trainer depends on.

# Rollout, Recovery, and Monitoring Plan

Rollout:
- Ship as a narrow environment bug fix plus regression coverage.

Recovery:
- If the fix causes unexpected bankroll ordering or breaks existing callers, revert the `PokerGPU.reset()` change and the new regression file together.

Monitoring:
- For manual validation, compare per-seat bankroll continuity across consecutive rotated episodes in a GPU training run.
- Watch for future regressions where agent order and stack order drift again after reset; the new regression suite is the primary guardrail.

# Open Questions and Explicit Assumptions

Explicit assumptions:
- The ticket’s requested Option A is the desired scope, so the redundant positional `rotation` argument remains in place for compatibility.
- `trainGPU.py` is the primary production caller and will continue passing rotation only through `options`.
- The initial reset case does not need explicit coverage because uniform starting stacks make rotation unobservable there.

Open questions:
- None that block implementation. The intended contract and the minimal fix are both explicit in the ticket.
