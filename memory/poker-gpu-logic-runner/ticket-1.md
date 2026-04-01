Title: Add a standalone Poker GPU logic regression runner

Issue URL: https://github.com/cerredz/Pulselib/issues/41

Intent:
Provide a single executable file in `scripts/Poker/` that runs a comprehensive set of Poker GPU environment logic checks from one entrypoint, with readable multi-line descriptions for every case, so the user can review and run the full regression surface without depending on pytest discovery.

Scope:
- Create one new runnable Python script in `scripts/Poker/`.
- Cover high-value Poker GPU logic cases spanning reset/setup, observation encoding, action execution, reward attribution, round progression, showdown resolution, side-pot handling, preflop runouts, and trainer-adjacent state safety.
- Print or otherwise surface a multi-line description for each test case before execution.
- Provide a single entrypoint that executes the full suite and returns a failing exit code on the first or accumulated failure.

Out of scope:
- Modifying `environments/Poker/PokerGPU.py`.
- Refactoring the training loop or existing pytest suites.
- Replacing the existing `tests/poker/` regression files.

Relevant Files:
- `C:\Users\422mi\Pulselib\scripts\Poker\test_poker_gpu_logic_runner.py`
  - New standalone regression runner with a case registry and `main()` entrypoint.
- `C:\Users\422mi\Pulselib\memory\poker-gpu-logic-runner\tickets\ticket-1-quality.md`
  - Quality pipeline results for this ticket.
- `C:\Users\422mi\Pulselib\memory\poker-gpu-logic-runner\tickets\ticket-1-critique.md`
  - Post-implementation self-critique and refinements.

Approach:
- Implement a lightweight runner using a dataclass-like case registry: each case has a name, a multi-line description, and a callable.
- Reuse the same environment-building and card-encoding patterns already established in `tests/poker/` so the new script stays aligned with current conventions.
- Keep the script self-contained enough to run directly with `python scripts/Poker/test_poker_gpu_logic_runner.py`.
- Prefer direct behavior assertions over introspecting implementation details.
- Group helpers around natural seams:
  - environment builder
  - card encoding / ordered decks
  - per-case assertion helpers
  - execution/reporting harness

Assumptions:
- The user wants a standalone executable script rather than a pytest-only file.
- Covering the main logic seams and known regressions satisfies the "comprehensive" request more effectively than attempting every combinatorial table state in one file.
- Existing untracked/dirty repository artifacts are unrelated and must remain untouched.

Acceptance Criteria:
- [ ] A new file exists in `scripts/Poker/` alongside `trainGPU.py`.
- [ ] Running the new file executes all registered cases from a single entrypoint.
- [ ] Every test case has a multi-line human-readable description.
- [ ] The runner covers the major Poker GPU logic seams: reset/setup, observation ordering, action execution, reward attribution, round progression, showdown/side-pot resolution, and runout behavior.
- [ ] The runner exits non-zero when any case fails.
- [ ] The runner passes locally on the current codebase.

Verification Steps:
1. Compile the new script with `python -m py_compile scripts/Poker/test_poker_gpu_logic_runner.py`.
2. Run the standalone runner with `python scripts/Poker/test_poker_gpu_logic_runner.py`.
3. Run the adjacent poker pytest regressions that overlap the runner's coverage:
   - `python -m pytest tests/poker/test_poker_gpu_showdown.py -q`
   - `python -m pytest tests/poker/test_poker_gpu_side_pot_showdown.py -q`
   - `python -m pytest tests/poker/test_poker_gpu_preflop_allin_resolver.py -q`
   - `python -m pytest tests/poker/test_poker_gpu_round_progression.py -q`
   - `python -m pytest tests/poker/test_poker_gpu_no_actor_rewards.py -q`

Dependencies:
- None.

Drift Guard:
This ticket must not turn into a refactor of the poker environment or the trainer. The work is complete once there is a readable, runnable, high-coverage script in `scripts/Poker/` plus the normal verification and review artifacts. If a missing regression suggests an environment bug, that bug should be documented by the test harness and surfaced in review rather than fixed as part of this ticket.
