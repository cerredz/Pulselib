Title: Refactor poker GPU stability metrics into reusable utilities

Intent:
Implement the PR `#40` review feedback by removing the unintended `Player.py` changes and moving stability metric calculations into explicit reusable helpers that future scripts can import.

Scope:
- Restore `environments/Poker/Player.py` so this PR no longer changes the Q-network contract.
- Add `utils/stability.py` with reusable helper functions for batch filtering and stability metric aggregation.
- Update `scripts/Poker/trainGPU_stability.py` to use the new helpers.
- Add focused tests for the helper module and script integration path.
- Make only the minimal supporting logging adjustment needed for metric serialization robustness.

Relevant Files:
- `environments/Poker/Player.py` - revert unintended PR `#40` changes.
- `scripts/Poker/trainGPU_stability.py` - consume reusable stability helpers instead of inline calculations.
- `utils/stability.py` - new helper module for stability metrics.
- `utils/logging/logger.py` - ensure nested numeric metric payloads serialize safely.
- `tests/poker/test_train_gpu_stability_metrics.py` - regression coverage for helper behavior and script integration.

Approach:
Use the existing `PokerQNetwork` object as an input to helper functions that inspect the same batch passed into training. The script will compute valid masks, evaluate pre-update Q-values and targets, call `train_step(...)` unchanged, and then aggregate per-step, per-episode, and final benchmark metrics through dedicated helper functions. This preserves the current benchmark outputs without expanding the Q-network API.

Assumptions:
- The stability benchmark may compute summary metrics externally as long as it does not change `PokerQNetwork.train_step(...)`.
- The logger should be allowed to serialize nested benchmark payloads because final stability summaries naturally include nested structures.

Acceptance Criteria:
- [ ] PR `#40` no longer modifies `environments/Poker/Player.py`.
- [ ] Stability calculations are implemented via reusable helper functions under `utils/stability.py`.
- [ ] `scripts/Poker/trainGPU_stability.py` uses those helpers rather than inline metric math.
- [ ] Final benchmark metrics still include reward stability, TD-error trend, Q-value bounds, gradient clip rate, and runtime.
- [ ] Added tests cover helper behavior and the script’s use of the helpers.

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
