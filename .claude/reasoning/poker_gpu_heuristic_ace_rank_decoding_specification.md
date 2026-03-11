# Poker GPU Heuristic Ace Rank Decoding Specification

## Task Definition and Objectives
Fix the GPU poker heuristic agents so they decode the `PokerGPU` environment's 1-based card IDs correctly. Today the GPU heuristics in `environments/Poker/Player.py` derive ranks with `hands % 13`, which misclassifies aces as rank `0`. The objective is to make ace-high hole cards evaluate according to poker rank ordering across all GPU heuristic opponents without changing the environment's existing 1-based card encoding.

## In-Scope / Out-of-Scope
In scope:
- Add regression tests in `tests/poker/test_poker_gpu_heuristic_ace_rank_decoding.py` that fail under the current bug and pass after the fix.
- Modify `environments/Poker/Player.py` so every GPU heuristic class uses a shared 1-based rank-decoding path.
- Validate the fix through direct heuristic tests and the adjacent heuristic smoke suite.

Out of scope:
- Changing `environments/Poker/PokerGPU.py` card encoding from `1..52` to `0..51`.
- Changing CPU heuristic logic in `HeuristicPlayer`.
- Retuning heuristic thresholds, action distributions, or training config composition in `config/pokerGPU.yaml`.
- Broader training-loop, reward, or showdown changes.

## Current State and Architecture Context
Observed current state:
- `environments/Poker/PokerGPU.py` builds shuffled decks with `argsort(dim=1) + 1`, so dealt cards are `1..52`.
- `environments/Poker/Player.py` has four GPU heuristic classes:
  - `HeuristicHandsPlayerGPU`
  - `TightAggressivePlayerGPU`
  - `LoosePassivePlayerGPU`
  - `SmallBallPlayerGPU`
- Each class extracts hole-card ranks independently with `hands % 13`.
- `environments/Poker/utils.py` routes live actions through `build_actions(...)` and constructs GPU heuristic seats through `load_gpu_agents(...)`.
- `scripts/Poker/trainGPU.py` loads `config/pokerGPU.yaml`, builds the GPU heuristic table, inserts the Q-network, and uses those heuristics as the default training pool.

What changes:
- Shared GPU rank decoding in `environments/Poker/Player.py`.
- New regression coverage in `tests/poker/test_poker_gpu_heuristic_ace_rank_decoding.py`.

What does not change:
- The `PokerGPU` environment's storage format for cards.
- The shape or dtype contract of action tensors returned by GPU heuristics.
- Existing agent loading and action routing APIs in `environments/Poker/utils.py`.

## Proposed Design and Integration Plan
Design choice:
- Introduce a small shared helper in `environments/Poker/Player.py` that converts 1-based card IDs to zero-based rank buckets with `(hands.long() - 1) % 13`.
- Reuse that helper in all four GPU heuristic policies instead of duplicating raw `% 13`.

Rationale:
- This is the minimum fix that matches the environment contract.
- Centralizing the decode removes copy-pasted rank logic and prevents future GPU heuristics from repeating the same bug.
- The helper stays fully vectorized on the input tensor's device, which aligns with PyTorch best practice for tensor operations and avoids Python loops or implicit CPU/GPU transfers.

Implementation tasks:

1. Modify `C:\Users\422mi\Pulselib\tests\poker\test_poker_gpu_heuristic_ace_rank_decoding.py`.
- Specific change: keep the six tests already written as the contract for the bug fix.
- Why necessary: they reproduce the issue directly at the heuristic boundary and through `build_actions(...)`.
- Downstream impact: these tests validate both individual agents and the routing path used by training.
- Potential issues: action outputs include random raise sizes for some heuristics, so assertions must target observable action classes (`fold`, `call`, `raise`) rather than exact raise indices unless deterministic by rule.
- Validation: `python -m pytest tests/poker/test_poker_gpu_heuristic_ace_rank_decoding.py -q` must pass.

2. Modify `C:\Users\422mi\Pulselib\environments\Poker\Player.py`.
- Specific change: add a typed helper near the GPU heuristic classes, for example `def _decode_gpu_hole_card_ranks(hands: torch.Tensor) -> torch.Tensor:`, returning `(hands.long() - 1) % 13`.
- Why necessary: the ticketed bug exists in four copy-pasted `% 13` sites; one helper fixes the contract once.
- Downstream impact: `HeuristicHandsPlayerGPU.action`, `TightAggressivePlayerGPU.action`, `LoosePassivePlayerGPU.action`, and `SmallBallPlayerGPU.action` will all consume the corrected ranks.
- Potential issues: the helper must preserve device placement and integer dtype semantics; using `long()` explicitly prevents dtype surprises from float observations while avoiding implicit host transfers.
- Validation: the new regression tests must switch from all-failing to passing, and existing heuristic tests must remain green.

3. Modify `C:\Users\422mi\Pulselib\environments\Poker\Player.py` inside each GPU heuristic action method.
- Specific change: replace local `hands % 13` expressions with the shared helper result.
- Why necessary: all four agents currently misdecode ace ranks independently.
- Downstream impact: `build_actions(...)` in `environments/Poker/utils.py` will begin producing corrected ace-high actions for live training and evaluation without any API change.
- Potential issues: `SmallBallPlayerGPU` also branches on `pot_size`; the fix must only change rank evaluation, not pot threshold behavior.
- Validation: targeted tests for `A2`, `A7`, and `AK` across the four GPU heuristics must pass.

4. Validate adjacent behavior in `C:\Users\422mi\Pulselib\tests\poker\test_heuristic_agents.py`.
- Specific change: no code change expected; this file is an adjacent verification target because it exercises `load_gpu_agents(...)` and `build_actions(...)`.
- Why necessary: confirms that the fix does not break the existing heuristic action smoke path used by the training stack.
- Downstream impact: provides confidence that training bootstrapping remains intact.
- Potential issues: the benchmark-style tests may be slower than unit tests, but they already exist as local guardrails.
- Validation: run `python -m pytest tests/poker/test_heuristic_agents.py -q`.

PyTorch best-practice alignment:
- Memory management: the helper will operate directly on the incoming tensor and preserve current device placement, so there are no implicit CPU-GPU transfers.
- Tensor operations: the decode is fully batched and vectorized; no Python loops or per-row `.item()` calls are introduced.
- Precision and stability: rank decoding is integer arithmetic only; mixed precision guidance is not applicable here.
- Distributed safety: no global mutable state is added, so the helper is safe under batched inference in multi-process launches.

## Data and API Contract Changes
No external API or schema changes.

Before:
- GPU heuristics interpret `1..52` card IDs as though they were `0..51`, producing rank buckets shifted by one and mapping aces to `0`.

After:
- GPU heuristics interpret `1..52` card IDs correctly by converting to zero-based rank buckets before thresholding.

Unchanged contracts:
- Input state shape remains unchanged.
- Output action dtype remains `torch.long`.
- `load_gpu_agents(...)` and `build_actions(...)` signatures remain unchanged.

## Edge Cases and Failure Modes
- Ace at the maximum card ID boundary (`As` -> `52`) must still decode as the highest rank.
  - Covered by `test_heuristic_hands_gpu_raises_ace_two_with_one_based_ids`.
- Batched inference must not regress when one row is ace-high and another is a weak low-card hand.
  - Covered by `test_heuristic_hands_gpu_batched_ace_high_and_low_cards_diverge`.
- Tight-aggressive logic must stop folding ace-high hands that merely miss the raise threshold.
  - Covered by `test_tight_aggressive_gpu_calls_ace_seven_instead_of_folding`.
- Loose-passive logic must stop silently folding premium ace-high hands due to the wrapped ace rank.
  - Covered by `test_loose_passive_gpu_does_not_fold_ace_king`.
- Small-ball logic must preserve pot-size branching while restoring ace-high raises.
  - Covered by `test_small_ball_gpu_raises_ace_king_when_pot_is_manageable`.
- Integration routing through grouped agent dispatch must preserve the corrected behavior for configured heuristic seat types.
  - Covered by `test_build_actions_routes_configured_gpu_heuristics_with_ace_high_hands`.

## Security, Reliability, and Performance Considerations
- Security: no new external inputs, file access, serialization, or privilege boundaries are introduced.
- Reliability: centralizing the decode reduces the chance of future copy-paste divergence across GPU heuristics.
- Performance: the helper is a single vectorized integer transform on an already resident tensor. This is effectively free relative to the surrounding action logic and avoids any Python-side loops.
- Memory: no additional persistent buffers are introduced; the temporary rank tensor is the same scale as the prior per-method `ranks` tensor.
- Numerical behavior: integer modulo on `long` tensors is exact; no precision tradeoff exists.

## Acceptance Criteria
- `HeuristicHandsPlayerGPU` raises for ace-two (`As`, `2c`) instead of folding.
- `HeuristicHandsPlayerGPU` still folds weak low-card hands in the same batch while raising ace-high hands.
- `TightAggressivePlayerGPU` returns `call` for ace-seven (`As`, `7c`) instead of folding it.
- `LoosePassivePlayerGPU` returns a non-fold action for ace-king (`As`, `Kd`).
- `SmallBallPlayerGPU` returns a raise action for ace-king at a manageable pot size.
- `build_actions(...)` preserves the corrected behavior across configured GPU heuristic seat types.
- `tests/poker/test_heuristic_agents.py` continues to pass without modification.

## Test Plan
Unit and regression:
- `C:\Users\422mi\Pulselib\tests\poker\test_poker_gpu_heuristic_ace_rank_decoding.py`
  - Direct per-agent regressions for ace-high decoding.
  - Batched regression for vectorized behavior.
  - Integration regression through `load_gpu_agents(...)` and `build_actions(...)`.

Adjacent integration:
- `C:\Users\422mi\Pulselib\tests\poker\test_heuristic_agents.py`
  - Existing training-stack smoke verification for heuristic action construction.

Coverage gap noted:
- No end-to-end training-quality assertion is added here; that is intentionally out of scope for a targeted logic fix.

## Rollout, Recovery, and Monitoring Plan
- Rollout: land as a small logic-only patch with regression tests.
- Recovery: revert the helper change in `environments/Poker/Player.py` if unexpected heuristic behavior appears.
- Monitoring signals:
  - Regression test failures in `tests/poker/test_poker_gpu_heuristic_ace_rank_decoding.py`.
  - Heuristic action smoke failures in `tests/poker/test_heuristic_agents.py`.
  - Training runs showing unexpected increases in heuristic fold frequency for premium ace-high starting hands would indicate a regression.

## Open Questions and Explicit Assumptions
Assumptions:
- The environment contract of `PokerGPU` using `1..52` card IDs is intentional and should remain unchanged.
- Existing heuristic thresholds are semantically correct once ranks are decoded properly.
- CPU heuristic behavior is already correct because it uses `decode_card(...)` and is not part of this ticket.

Open questions:
- None critical for implementation. No architecture, security, or contract ambiguity remains after reading `Player.py`, `PokerGPU.py`, `environments/Poker/utils.py`, `scripts/Poker/trainGPU.py`, `config/pokerGPU.yaml`, and adjacent tests.
