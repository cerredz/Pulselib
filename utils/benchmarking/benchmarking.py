from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Mapping

import yaml

from utils.benchmarking.episodes import benchmark_episode
from utils.benchmarking.files import create_files
from utils.config import get_result_folder_env


DEFAULT_BENCHMARK_MASK = {
    "training_summary": True,
}


class Benchmarker(ABC):
    def __init__(
        self,
        enabled: bool = True,
        feature_mask: Mapping[str, bool] | None = None,
        results_dir_resolver: Callable[[str], Path] | None = None,
    ):
        self.enabled = enabled
        self.feature_mask = {**DEFAULT_BENCHMARK_MASK, **(feature_mask or {})}
        self.results_dir_resolver = results_dir_resolver or get_result_folder_env

    def is_enabled(self, feature_name: str) -> bool:
        return self.enabled and self.feature_mask.get(feature_name, True)

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None = None) -> "Benchmarker":
        config = config or {}
        return cls(
            enabled=bool(config.get("enabled", True)),
            feature_mask=config.get("mask"),
        )

    @abstractmethod
    def create_benchmark_file(
        self,
        env_name,
        episodes_return,
        start_time,
        end_time,
        total_steps,
        config,
    ):
        raise NotImplementedError


class NullBenchmarker(Benchmarker):
    def create_benchmark_file(
        self,
        env_name,
        episodes_return,
        start_time,
        end_time,
        total_steps,
        config,
    ):
        return None


class YamlBenchmarker(Benchmarker):
    def create_benchmark_file(
        self,
        env_name,
        episodes_return,
        start_time,
        end_time,
        total_steps,
        config,
    ):
        if not self.is_enabled("training_summary"):
            return None

        results_dir = self.results_dir_resolver(env_name)
        _, current_path = create_files(results_dir=results_dir)
        mean, std, minimum, maximum, median, count = benchmark_episode(episodes_return=episodes_return)
        training_seconds = end_time - start_time
        benchmark_file = {
            "env": env_name,
            "config": config,
            "start_time": start_time,
            "end_time": end_time,
            "total_training_seconds": training_seconds,
            "total_steps": total_steps,
            "sps": round(float(total_steps / training_seconds), 4) if training_seconds > 0 else 0.0,
            "episode_stats": {
                "count": count,
                "mean": float(mean),
                "std": float(std),
                "min": float(minimum),
                "max": float(maximum),
                "median": float(median),
            },
        }

        print(current_path)
        with open(current_path, "w") as file:
            yaml.dump(benchmark_file, file, default_flow_style=False)
        return current_path


DEFAULT_BENCHMARKER = YamlBenchmarker()


def create_benchmark_file(
    env_name,
    episodes_return,
    start_time,
    end_time,
    total_steps,
    config,
    benchmarker: Benchmarker | None = None,
):
    active_benchmarker = benchmarker or DEFAULT_BENCHMARKER
    return active_benchmarker.create_benchmark_file(
        env_name=env_name,
        episodes_return=episodes_return,
        start_time=start_time,
        end_time=end_time,
        total_steps=total_steps,
        config=config,
    )
