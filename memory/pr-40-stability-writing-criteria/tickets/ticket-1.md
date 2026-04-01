Title: Refactor poker GPU stability metrics into reusable utilities

Intent:
Implement the PR `#40` review feedback by removing the unintended `Player.py` changes and moving stability metric calculations into explicit reusable helpers that future scripts can import.

Scope:
- Restore `environments/Poker/Player.py` so this PR no longer changes the Q-network contract.
- Add `utils/stability.py` with reusable helper functions for batch filtering and stability metric aggregation.
- Update `scripts/Poker/trainGPU_stability.py` to use the new helpers.
- Add focused tests for the helper module and script integration path.
- Make only the minimal supporting logging adjustment needed for metric serialization robustness.
- Keep the stability helper path torch-native so metric aggregation stays on the active tensor device rather than using NumPy/Python reductions.
- Add the requested PyTorch best-practices artifact under `artifacts/`.

Relevant Files:
- `environments/Poker/Player.py` - revert unintended PR `#40` changes.
- `scripts/Poker/trainGPU_stability.py` - consume reusable stability helpers instead of inline calculations.
- `utils/stability.py` - new helper module for stability metrics.
- `utils/logging/logger.py` - ensure nested numeric metric payloads serialize safely.
- `tests/poker/test_train_gpu_stability_metrics.py` - regression coverage for helper behavior and script integration.
- `artifacts/pytorch_codebase_best_practices.md` - PyTorch GPU best-practices artifact requested in review.

Approach:
Use the existing `PokerQNetwork` object as an input to helper functions that inspect the same batch passed into training. The script will compute valid masks, evaluate pre-update Q-values and targets, call `train_step(...)` unchanged, and then aggregate per-step, per-episode, and final benchmark metrics through dedicated helper functions. Those helpers will use torch-only reductions and tensor math so metric aggregation stays on-device until the final logging and print boundary. This preserves the current benchmark outputs without expanding the Q-network API.

Assumptions:
- The stability benchmark may compute summary metrics externally as long as it does not change `PokerQNetwork.train_step(...)`.
- The logger should be allowed to serialize nested benchmark payloads because final stability summaries naturally include nested structures.

Acceptance Criteria:
- [ ] PR `#40` no longer modifies `environments/Poker/Player.py`.
- [ ] Stability calculations are implemented via reusable helper functions under `utils/stability.py`.
- [ ] `scripts/Poker/trainGPU_stability.py` uses those helpers rather than inline metric math.
- [ ] Final benchmark metrics still include reward stability, TD-error trend, Q-value bounds, gradient clip rate, and runtime.
- [ ] Added tests cover helper behavior and the script integration path.
- [ ] The stability helper path uses torch operations rather than NumPy for metric aggregation and preserves the input device for tensor work.
- [ ] `artifacts/pytorch_codebase_best_practices.md` exists and documents the requested GPU-focused PyTorch best practices.

Verification Steps:
1. Apply a manual style pass because no repo linter is configured.
2. Confirm no project type checker is configured; add type hints in new helper code.
3. Run `python -m pytest tests/poker/test_train_gpu_stability_metrics.py -q`.
4. Run at least one adjacent poker pytest target that confirms nothing regressed in the existing training path.
5. Inspect the final git diff to confirm `environments/Poker/Player.py` is restored and only intended files changed.

Dependencies:
- None.

Drift Guard:
This ticket must not redesign the poker training loop, retune RL hyperparameters, or introduce broader Q-network instrumentation. The goal is only to satisfy PR review feedback by relocating stability metric calculations into reusable utilities while preserving existing runtime behavior.
