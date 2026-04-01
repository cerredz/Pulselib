### 1a: Structural Survey

Repository shape:
- `agents/`: RL agent implementations, including temporal-difference and Monte Carlo agents.
- `config/`: YAML training configs for each environment.
- `environments/`: environment implementations, with `environments/Poker/` holding both CPU and GPU poker logic plus shared poker utilities.
- `scripts/`: training and benchmark entrypoints. `scripts/Poker/trainGPU.py` is the main GPU poker training loop pattern adjacent to this task.
- `utils/`: shared helpers for config loading, benchmarking, plotting, torch device selection, replay buffer utilities, and logging.
- `tests/`: pytest-based regression coverage, with `tests/poker/` holding targeted poker bug and training-loop tests.
- `rl/`: handwritten reinforcement-learning notes, including `rl/important_notes.txt`, which describes high-level RL design questions rather than executable criteria.
- `memory/`: prior engineering notes and tickets created during earlier poker work.

Technology and conventions observed:
- Python project with direct module imports; no packaged build layer is present.
- Training code uses PyTorch and Gymnasium directly.
- Tests use `pytest` and monkeypatch-heavy targeted regressions rather than large integration harnesses.
- Configuration is file-based through YAML helpers in `utils.config`.
- Utility modules are plain Python files under `utils/`; reusable poker-specific helpers already live in `environments/Poker/utils.py`.
- The codebase is partially inconsistent on typing and formatting: newer tests use type hints heavily, while environment and script files are looser and more compact.
- There is no visible project-wide linter or type-checker configuration in the repo root.

Relevant poker data flow:
1. A poker training script loads config and device helpers.
2. The script builds heuristic agents through `environments/Poker.utils.load_gpu_agents(...)`.
3. The script constructs a `PokerQNetwork` from `environments/Poker.Player`.
4. A Gymnasium poker GPU environment is created and stepped in batches.
5. Action routing goes through `environments/Poker.utils.build_actions(...)`.
6. Training scripts compute and report aggregate metrics, benchmark artifacts, and saved weights.

Test strategy:
- Targeted poker regressions live in `tests/poker/`.
- Tests commonly isolate scripts with dummy envs and monkeypatched helpers, which is the right pattern for the stability benchmark changes as well.

Codebase inconsistencies relevant to this task:
- `scripts/Poker/trainGPU_stability.py` is currently mixing benchmark orchestration with metric calculations.
- PR `#40` currently changes `environments/Poker/Player.py` to surface metric dictionaries from `PokerQNetwork.train_step(...)`, but the review explicitly rejects modifying that file for this task.
- `utils/logging/logger.py` serializes only top-level scalar metrics, which is fragile once nested metric dictionaries are logged.

### 1b: Task Cross-Reference

User request mapping:
- “Read the important notes file”: `rl/important_notes.txt` contains high-level RL prompts about returns, action values, exploration, simulated vs real updates, update targets, and memory horizons. It provides conceptual motivation for a stability-writing criteria artifact but does not dictate a specific implementation.
- “Set up the stability writing criteria file”: the concrete implementation lives in `scripts/Poker/trainGPU_stability.py`, which is the grading criteria artifact introduced by PR `#40`.
- “Implement the solution for these comments on pull request number 40”: the actionable requirements come from PR review comments on `https://github.com/cerredz/Pulselib/pull/40`.

PR `#40` review mapping:
- Comment 1 on `environments/Poker/Player.py`: this file must not be changed in this PR. The existing branch diff shows `PokerQNetwork.train_step(...)` was altered to return detailed metrics instead of its prior behavior. That change must be removed.
- Comment 2 on `scripts/Poker/trainGPU_stability.py`: metric calculations should be defined as reusable utility functions in `utils/`.
- Comment 3 on `scripts/Poker/trainGPU_stability.py`: create a dedicated `utils/stability.py` module with explicit helper names so the metric calculations can be reused by future scripts.

Files touched by the requested fix:
- `scripts/Poker/trainGPU_stability.py`: must delegate metric calculation to reusable helpers and stop depending on a modified `PokerQNetwork.train_step(...)` return contract.
- `utils/stability.py`: net-new module for step-level, episode-level, and final benchmark metric calculations.
- `utils/logging/logger.py`: likely needs a small robustness fix so nested metric payloads from final stability summaries serialize safely.
- `utils/logging/__init__.py`: may remain unchanged unless export surface needs extension.
- `tests/poker/...`: new regression coverage should validate the helper module and the script integration path.
- `environments/Poker/Player.py`: must be restored so PR `#40` no longer contains unrelated changes here.

Behavior that must be preserved:
- The stability benchmark must still report reward stability, TD-error trend, Q-value bounds, and gradient clip rates.
- The Q-network API must remain compatible with existing training code and adjacent tests.
- The current branch/PR target remains `main`, but implementation should stay on the existing PR branch rather than creating a separate parallel PR flow.

Blast radius:
- Low-to-moderate. The main risk is breaking `PokerQNetwork.train_step(...)` or making the stability script inconsistent with how the environment batches terminal rows and active Q-seat rows.
- Logging changes affect any future callers of `TrainingLogger`, so that code should be made more robust rather than more specialized.

### 1c: Assumption & Risk Inventory

Assumptions:
- The review comments on PR `#40` are the source of truth for implementation scope.
- `rl/important_notes.txt` is contextual guidance, not a strict specification for exact formulas.
- The user wants changes applied to the existing PR branch `feature/poker-gpu-stability-benchmark`, not a new branch or new PR.
- The intended reusable helpers belong in `utils/` rather than `environments/Poker/utils.py`, because the review explicitly asks for `utils/stability.py`.

Risks:
- Reverting `PokerQNetwork.train_step(...)` naively would remove the data the stability script currently consumes. The script therefore needs an alternate way to compute metrics around the existing training call.
- If metric collection is done after the optimizer step without care, the values may no longer match the training batch used for the update. Helper APIs should therefore receive the pre-update batch tensors and model references explicitly.
- `TrainingLogger.log(...)` currently converts only top-level scalar values; nested dicts containing numpy scalars are likely to fail JSON serialization when final benchmark metrics are logged.
- The working tree already contains unrelated user artifacts and untracked files, so changes must stay tightly scoped and avoid broad git operations.

Follow-up review note:
- A later PR `#40` comment tightened the scope further: the reusable stability helpers and benchmark path should avoid NumPy entirely, preserve tensor work on the incoming device, and only convert to Python scalars at the final output boundary. That pushes the implementation toward scalar tensor aggregation rather than Python-float summaries inside the training loop.

Phase 1 complete
