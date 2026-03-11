# Poker GPU Benchmarking

This folder benchmarks the live GPU poker stack rather than a copied benchmark-only implementation. The benchmark cases import the current runtime modules in `scripts/Poker/trainGPU.py`, `environments/Poker/PokerGPU.py`, `environments/Poker/utils.py`, and `environments/Poker/Player.py`, so future runs automatically exercise the updated environment and trainer.

## What It Measures

The first benchmark suite covers multiple parts of the poker stack instead of one coarse timing number:

- `env_reset` measures vectorized environment reset throughput.
- `env_calculate_equities` measures the environment's live equity calculation path on prepared river-state batches.
- `env_execute_actions` measures batched action execution inside the environment.
- `env_step` measures a full `PokerGPU.step(...)` call, including reward and round progression work.
- `trainer_build_actions` measures trainer-side action routing through `build_actions(...)`.
- `trainer_q_network_train_step` measures one live `PokerQNetwork.train_step(...)` update.
- `trainer_short_run` measures a short live `train_agent(...)` run using no-op plotting and benchmark-summary writers so the timing stays focused on the trainer loop itself.

## How To Run

List cases:

```bash
python -m benchmarking.Poker.run --list-cases
```

Run the quick preset:

```bash
python -m benchmarking.Poker.run --preset quick
```

Run on a specific CUDA device:

```bash
python -m benchmarking.Poker.run --preset quick --device cuda:0
```

Run a specific case:

```bash
python -m benchmarking.Poker.run --preset quick --case env_step
```

## Output Shape

Each run writes a JSON report to `results/benchmarks/Poker/` by default. The report includes benchmark metadata, per-case timing statistics, and derived throughput values. The runner also prints a rigid stdout block between `LLM_BENCHMARK_SUMMARY_BEGIN` and `LLM_BENCHMARK_SUMMARY_END` so an LLM or regression parser can extract the key metrics without guessing.

The suite is intentionally GPU-only. If CUDA is not available, the runner fails early with a direct error instead of silently switching to a non-representative CPU codepath.

## Extension Guidance

Add new benchmark cases in `benchmarking/Poker/cases.py`. Keep benchmark setup inside this package. Do not duplicate production poker logic in the benchmark suite. If a future benchmark cannot call the live codepath cleanly, prefer a small benchmark harness workaround over changing the live trainer or environment unless the production code truly needs a cleaner seam for reasons beyond benchmarking.
