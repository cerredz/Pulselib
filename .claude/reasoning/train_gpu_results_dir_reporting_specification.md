Task Definition and Objectives

This task fixes ticket #20 in [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/scripts/Poker/trainGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/scripts/Poker/trainGPU.py), where `train_agent()` accepts a `results_dir` parameter but builds `chips_path` from an undefined `result_dir` name in the post-training reporting block. The objective is to make the helper use its `results_dir` argument consistently so training runs, including `episodes=0`, complete the weights save, plot generation, and benchmark reporting path without raising `NameError`.

In-Scope / Out-of-Scope

In scope:
- Add a dedicated regression suite in [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/tests/poker/test_train_gpu_results_dir_reporting.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/tests/poker/test_train_gpu_results_dir_reporting.py) with five distinct cases covering zero-episode reporting, path selection, weight saving, benchmark payloads, and a one-episode integration path.
- Modify the post-training reporting path in [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/scripts/Poker/trainGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/scripts/Poker/trainGPU.py) so both plots derive from the `results_dir` function argument.
- Preserve the existing training loop, environment contract, and benchmark writer interface.

Out of scope:
- Refactoring the broader naming style in the `__main__` block from `result_dir` to `results_dir`.
- Changing reward logic, GPU kernels, optimizer configuration, or PyTorch model architecture in [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/environments/Poker/Player.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/environments/Poker/Player.py).
- Altering plotting or benchmark file formats in [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/utils/plotting.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/utils/plotting.py) or [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/utils/benchmarking/benchmarking.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/utils/benchmarking/benchmarking.py).
- Touching unrelated files outside the new regression test and the target trainer bug.

Current State and Architecture Context

[`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/scripts/Poker/trainGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/scripts/Poker/trainGPU.py) has two scopes with similar naming:
- `train_agent(..., results_dir, ...)` receives the destination directory for model weights and training plots.
- The `__main__` block creates `result_dir = get_result_folder(...)` and passes it to `train_agent(results_dir=result_dir, ...)`.

The bug is confined to `train_agent()`:
- Weights are saved with `torch.save(..., f"{results_dir}/poker_qnet_final.pth")`, which uses the correct argument.
- `reward_path` is built from `results_dir`.
- `chips_path` is built from `result_dir`, which is not defined in the helper scope and raises `NameError` unless a module-level `result_dir` happens to exist.

Connected dependencies already reviewed:
- [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/utils/plotting.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/utils/plotting.py): `plot_learning_curve()` accepts a string path and writes image/pickle artifacts.
- [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/utils/benchmarking/benchmarking.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/utils/benchmarking/benchmarking.py): `create_benchmark_file()` is called after plotting and receives the accumulated reward history plus timing metadata.
- [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/environments/Poker/Player.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/environments/Poker/Player.py): `PokerQNetwork` is discovered via `isinstance()` before state-dict serialization.

Proposed Design and Integration Plan

1. Modify [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/tests/poker/test_train_gpu_results_dir_reporting.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/tests/poker/test_train_gpu_results_dir_reporting.py).
Specific change:
- Keep the new five-test regression suite as the executable contract for this ticket.
- Use a lightweight `DummyPokerQNetwork` subclass so `isinstance(..., PokerQNetwork)` remains true while avoiding full optimizer/device setup.
- Use zero-episode invocations to isolate the reporting tail and a one-episode fake environment to confirm reporting still works after a real loop iteration.
Why this is necessary:
- The ticket is a deterministic scope bug. The fastest reliable proof is a direct call to `train_agent()` with controlled collaborators.
Adjacent modules affected downstream:
- [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/scripts/Poker/trainGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/scripts/Poker/trainGPU.py) is the only production module these tests constrain.
Potential issues:
- Tests must not depend on the live poker environment or real benchmark directories, or they will become brittle and over-broad.
- Injecting a fake module-global `result_dir` is necessary to catch the subtler wrong-scope path leak, not just the `NameError`.
Validation:
- `pytest tests/poker/test_train_gpu_results_dir_reporting.py -q` fails before the fix and passes after it.

2. Modify [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/scripts/Poker/trainGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/scripts/Poker/trainGPU.py) at the post-training reporting block.
Specific change:
- Replace `chips_path = result_dir / CHIPS_FILENAME` with `chips_path = results_dir / CHIPS_FILENAME`.
Why this is necessary:
- The helper contract is already `results_dir`; the code must use the parameter it was passed rather than a name from a different scope.
Adjacent modules affected downstream:
- [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/utils/plotting.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/utils/plotting.py) will receive the correct chip-plot path.
- [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/utils/benchmarking/benchmarking.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/utils/benchmarking/benchmarking.py) becomes reachable in zero-episode and normal runs because the helper no longer crashes first.
Potential issues:
- None expected beyond preserving the existing `Path` semantics used by the reward plot and weight save path.
- Avoid unnecessary refactors in the same file; the ticket is about a single scope bug and the skill requires minimal, high-signal changes.
Validation:
- Re-run the new regression suite and confirm all five tests pass.

Data and API Contract Changes

There are no public API or schema changes.

Before:
- `train_agent()` could raise `NameError` in its reporting section because `chips_path` referenced `result_dir`.

After:
- `train_agent()` uses `results_dir` consistently for model save, reward plot, and chip plot outputs.
- Callers continue to pass `results_dir` exactly as before.
- Output artifact names remain `poker_qnet_final.pth`, `rewards_learning_curve`, and `total_chips_curve`.

Edge Cases and Failure Modes

- Zero-episode runs must not touch the environment but still complete reporting. Covered by `test_train_agent_zero_episodes_skips_env_interaction_and_completes_reporting`.
- A stray module-global `result_dir` must not hijack the chip plot destination. Covered by `test_train_agent_uses_passed_results_dir_for_reward_and_chip_plots`.
- Weight serialization must still land in the caller-provided directory after the reporting fix. Covered by `test_train_agent_saves_q_network_weights_in_results_dir`.
- Zero-episode benchmark payloads must remain well-formed with empty rewards and zero steps. Covered by `test_train_agent_zero_episode_benchmark_reports_empty_scores_and_zero_steps`.
- A real loop iteration must still propagate reward and chip-profit histories through the reporting tail. Covered by `test_train_agent_single_episode_reports_reward_and_chip_profit`.

Security, Reliability, and Performance Considerations

- Security: no new file-system surface is introduced; the fix only restores use of the already-supplied output directory.
- Reliability: this converts a deterministic post-training crash into the intended reporting flow and prevents accidental dependence on a module-global variable.
- Performance: the change is not in the GPU hot path. No tensor allocations, synchronization points, or data movement changes are introduced.
- PyTorch best-practice alignment:
  - State serialization already uses `state_dict()`, which matches the skill’s checkpointing requirement for model persistence.
  - The fix avoids any changes to device placement, mixed precision, gradient flow, or training-loop tensor operations, so no new PyTorch performance or numerical-stability risk is introduced.
  - The regression tests deliberately stub the reporting functions rather than adding `.item()` or CPU syncs inside the actual training loop code.

Acceptance Criteria

- `pytest tests/poker/test_train_gpu_results_dir_reporting.py -q` passes completely.
- `test_train_agent_zero_episodes_skips_env_interaction_and_completes_reporting` proves the helper reaches both plotting calls and benchmark reporting without using the environment.
- `test_train_agent_uses_passed_results_dir_for_reward_and_chip_plots` proves both plot paths are derived from the `results_dir` argument even if a wrong global `result_dir` exists.
- `test_train_agent_saves_q_network_weights_in_results_dir` proves the Q-network weights are persisted into the caller-provided directory.
- `test_train_agent_zero_episode_benchmark_reports_empty_scores_and_zero_steps` proves the benchmark payload remains valid for `episodes=0`.
- `test_train_agent_single_episode_reports_reward_and_chip_profit` proves the fix does not break the normal post-episode reporting flow.

Test Plan

Primary regression target:
- [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/tests/poker/test_train_gpu_results_dir_reporting.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/tests/poker/test_train_gpu_results_dir_reporting.py)

Execution plan:
- Run `pytest tests/poker/test_train_gpu_results_dir_reporting.py -q` before implementation to confirm the reported crash.
- Apply the one-line production fix in [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/scripts/Poker/trainGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/scripts/Poker/trainGPU.py).
- Re-run `pytest tests/poker/test_train_gpu_results_dir_reporting.py -q` until green.

Coverage gaps:
- The current ticket does not add a full end-to-end trainer invocation through `__main__`; the direct `train_agent()` contract is sufficient because the scope bug exists entirely inside that helper.
- The benchmark writer itself remains mocked in the regression suite to avoid coupling this fix to repository-level `results/` side effects.

Rollout, Recovery, and Monitoring Plan

- Rollout: ship as a narrow bug fix with no migration. Any caller already passing a valid `results_dir` will simply stop crashing.
- Recovery: if an unexpected regression appears, revert the one-line path change in [`/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/scripts/Poker/trainGPU.py`](/C:/Users/422mi/Pulselib/.worktrees/train-gpu-results-dir-reporting/scripts/Poker/trainGPU.py) and re-run the targeted regression suite.
- Monitoring: verify training runs now produce both learning-curve artifacts and benchmark output after completion, including `episodes=0` smoke runs.

Open Questions and Explicit Assumptions

Open questions:
- None that block implementation. The ticket defines the bug and the expected behavior precisely.

Explicit assumptions:
- The intended contract is to use the `results_dir` function argument everywhere inside `train_agent()`, even though the `__main__` block uses the singular local name `result_dir`.
- The minimal fix is preferred over a wider naming refactor because the ticket is deterministic, isolated, and already fully specified by the regression suite.
