from __future__ import annotations

from copy import deepcopy


DEFAULT_CASES = [
    "env_reset",
    "env_calculate_equities",
    "env_execute_actions",
    "env_step",
    "trainer_build_actions",
    "trainer_q_network_train_step",
    "trainer_short_run",
]


PRESETS = {
    "quick": {
        "device": "auto",
        "cases": list(DEFAULT_CASES),
        "warmup_iterations": 1,
        "measure_iterations": 3,
        "env": {
            "n_games": 256,
            "episodes": 2,
        },
    },
    "standard": {
        "device": "auto",
        "cases": list(DEFAULT_CASES),
        "warmup_iterations": 2,
        "measure_iterations": 5,
        "env": {
            "n_games": 1024,
            "episodes": 3,
        },
    },
    "stress": {
        "device": "auto",
        "cases": list(DEFAULT_CASES),
        "warmup_iterations": 2,
        "measure_iterations": 7,
        "env": {
            "n_games": 4096,
            "episodes": 5,
        },
    },
}


def resolve_preset(name: str) -> dict:
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")
    return deepcopy(PRESETS[name])
