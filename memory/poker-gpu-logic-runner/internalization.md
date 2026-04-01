### 1a: Structural Survey

Top-level architecture:
- `agents/`: RL algorithm implementations outside the environment layer.
- `config/`: environment and training configuration files, including poker GPU config.
- `environments/`: gym-style environments. `environments/Poker/` contains both CPU and GPU poker paths, heuristic agents, and evaluator assets.
- `models/`: model wrappers not directly central to this task.
- `rl/`: design notes and environment rationale. Useful for intended semantics, especially variable-player support and performance priorities.
- `scripts/`: runnable training entrypoints. `scripts/Poker/` is the user-facing execution surface for poker training and benchmarking.
- `tests/`: pytest-based regression coverage. `tests/poker/` already captures several GPU poker logic bugs and trainer invariants.
- `utils/`: shared configuration, plotting, device loading, benchmarking, logging, and stability helpers.
- `memory/`: local engineering artifacts, ticket notes, milestones, and prior investigations.

Relevant technology and patterns:
- Python with Gymnasium environments.
- PyTorch tensors used directly inside the GPU poker environment, including batched multi-game stepping.
- Tests are plain pytest modules with direct tensor assertions; no custom test framework is configured.
- No repository-level linter or type-checker configuration file is present at the repo root.

Poker-specific structure:
- `environments/Poker/PokerGPU.py`: batched GPU environment core. Handles reset, observation building, dealing, action execution, reward shaping, round progression, and termination resolution.
- `environments/Poker/Player.py`: Q-network plus heuristic poker agents.
- `environments/Poker/utils.py`: agent loading and action dispatch.
- `scripts/Poker/trainGPU.py`: primary poker GPU training loop.
- `scripts/Poker/trainGPU_stability.py`: stability/benchmark-oriented training path.
- `tests/poker/`: focused regressions for showdown resolution, side pots, round progression, no-actor rewards, heuristic action decoding, and trainer-state aliasing.

Observed conventions:
- Environment methods are vectorized over `n_games`.
- Existing tests favor narrowly scoped helper builders with explicit tensor setup.
- Known poker issues are documented as markdown tickets in `memory/tickets/`, often with a precise proof sketch and suggested regression direction.
- Poker runner scripts live in `scripts/Poker/` and are meant to be directly executable with `python`.

Codebase inconsistencies relevant to this task:
- Several poker regressions exist as test artifacts or ticket notes without a single consolidated executable runner in `scripts/Poker/`.
- The working tree currently contains unrelated generated files and untracked artifacts; implementation must avoid depending on that incidental state.
- There is no single configured lint/type pipeline to reuse, so verification will need to rely on local Python compilation and targeted pytest/script execution.

### 1b: Task Cross-Reference

User request mapping:
- "produce a test file (in the same directory as the pokerGPU training script)":
  - Target directory is `scripts/Poker/`, alongside `trainGPU.py`.
  - Net-new file is appropriate because current poker regressions live under `tests/poker/`, not as a single runnable script.
- "single entry point into the file where when we run the file all of the test cases are ran":
  - The new file should expose a `main()` and `if __name__ == '__main__': main()` flow.
  - The runner should enumerate all cases itself and exit non-zero on failure.
- "Each test case should have a multi line description as to what it is testing":
  - Each case should be represented by structured metadata, not bare functions.
  - The descriptions should print before execution so the script doubles as readable test documentation.
- "tests all of the following edge cases":
  - The relevant source of truth is the previously requested comprehensive checklist, constrained by the actual `PokerGPU` implementation and the regressions already captured in `tests/poker/` and `memory/tickets/`.
  - The highest-value coverage areas are reset/setup invariants, observations, action execution edge cases, reward attribution, round progression, showdown and side-pot resolution, preflop runouts, and trainer-boundary invariants that are directly adjacent to `trainGPU.py`.
- "create a pull request into main":
  - This requires GitHub CLI workflow after local implementation and verification.
  - The repo currently sits on `feature/poker-gpu-stability-benchmark`, so the requested PR flow will need a fresh branch/worktree based on `main`.

Concrete files touched by this task:
- Create `scripts/Poker/test_poker_gpu_logic_runner.py`:
  - standalone executable test harness for the poker GPU environment and trainer-adjacent invariants.
- Create workflow artifacts under `memory/poker-gpu-logic-runner/`:
  - `internalization.md`
  - `clarifications.md`
  - `tickets/index.md`
  - `tickets/ticket-1.md`
  - later quality and critique records required by the workflow.

Existing code and behavior to preserve:
- Do not change `PokerGPU.py`, `trainGPU.py`, or existing pytest files unless a hard blocker appears.
- The new runner should reuse existing semantics instead of inventing alternate poker rules.
- Existing test modules remain the authoritative regression set for pytest; the new script is an additional executable aggregation surface, not a replacement.

Blast radius:
- Low runtime blast radius if implemented as a net-new script.
- Moderate maintenance blast radius because the runner will encode many environment invariants; it should therefore centralize helpers, clear naming, and explicit case descriptions.

### 1c: Assumption & Risk Inventory

Assumptions:
- "same directory as the pokerGPU training script" means `scripts/Poker/`, next to `trainGPU.py`.
- The requested file is a standalone runnable script, not a new pytest module.
- It is acceptable for the script to reuse helper logic and/or import existing pytest-style regression functions where that improves coverage without weakening clarity.
- The user wants a high-coverage practical runner covering the major and known edge cases, not a mathematically exhaustive proof of every tensor state permutation.

Risks:
- The prior checklist is broader than can fit credibly into one monolithic file if every permutation is expanded separately; the runner needs disciplined grouping and reusable helpers to stay maintainable.
- Some historical regressions exist only as ticket notes or pycache traces, not as tracked source files. Re-implementing those cases directly in the runner is safer than depending on missing modules.
- The dirty working tree means git operations and PR prep must avoid unrelated artifacts.
- GitHub issue/PR creation may fail if `gh` is unauthenticated or if repo policy blocks issue deletion.

Resolution strategy:
- Build a case registry with descriptions plus compact helper setup functions so coverage is broad without becoming unreadable.
- Prefer direct in-script assertions for critical known regressions.
- Keep all code changes net-new and localized.

Phase 1 complete.
