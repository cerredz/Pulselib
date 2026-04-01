# Repository File Index

This document provides context and descriptions for the primary top-level folders within the repository, acting as a quick reference for their use cases.

- **`.claude/`**: Stores conversational history, project reasoning, issue tracking, planning documents, and reports used by AI assistants for project management.
- **`.worktrees/`**: Contains active Git worktrees used for developing and testing parallel features or fixes (e.g., various Poker GPU optimizations).
- **`agents/`**: Houses implementations for various Reinforcement Learning agents, including Monte Carlo and Temporal Difference learning approaches.
- **`benchmarking/`**: Scripts and utilities dedicated to measuring and benchmarking agent and environment performance (e.g., Poker benchmarks).
- **`config/`**: YAML-based configuration files that define hyperparameters and settings for different environments and training runs.
- **`environments/`**: The core simulation and game environments designed for the RL agents, including 2048, Blackjack, Particle2D, Poker, Tetris, and Wordle.
- **`memory/`**: A structured knowledge base tracking project milestones, reference papers, pull requests, agent skills, and tickets.
- **`models/`**: Contains neural network architectures and model definitions.
- **`results/`**: Output directory for storing intermediate training checkpoints, model weights (`.pth` files), logs, and evaluation results.
- **`rl/`**: Documentation, reference materials, and notes pertaining to Reinforcement Learning theory and project-specific strategies.
- **`scripts/`**: Executable Python scripts for initiating training loops (`qlearningtrain.py`), running environments, and orchestrating experiments.
- **`tests/`**: Unit and integration test suites for environments, utilities, and models to ensure code correctness and prevent regressions.
- **`tmp/`**: A temporary storage area for intermediate artifacts like git patches and pull request drafts.
- **`trained_models/`**: Dedicated storage for final, fully trained, and production-ready models.
- **`utils/`**: Reusable helper modules and utilities for plotting, PyTorch tensor management, Replay Buffers, Numba JIT optimizations, and environment step logic.
