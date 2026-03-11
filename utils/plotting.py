from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_PLOT_MASK = {
    "learning_curve": True,
    "multi_learning_curve": True,
}


class Plotter(ABC):
    def __init__(self, enabled: bool = True, feature_mask: Mapping[str, bool] | None = None):
        self.enabled = enabled
        self.feature_mask = {**DEFAULT_PLOT_MASK, **(feature_mask or {})}

    def is_enabled(self, feature_name: str) -> bool:
        return self.enabled and self.feature_mask.get(feature_name, True)

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None = None) -> "Plotter":
        config = config or {}
        return cls(
            enabled=bool(config.get("enabled", True)),
            feature_mask=config.get("mask"),
        )

    @abstractmethod
    def plot_learning_curve(
        self,
        scores: Sequence[float],
        file_path: str | Path,
        window_size: int = 100,
        title: str = "Agent Learning Curve",
        extend_plot: bool = False,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def plot_learning_curves(
        self,
        agent_scores: Dict[str, Sequence[float]],
        file_path: str | Path,
        window_size: int = 100,
        title: str = "Multi-Agent Learning Curves",
        extend_plot: bool = False,
    ) -> None:
        raise NotImplementedError


class NullPlotter(Plotter):
    def plot_learning_curve(
        self,
        scores: Sequence[float],
        file_path: str | Path,
        window_size: int = 100,
        title: str = "Agent Learning Curve",
        extend_plot: bool = False,
    ) -> None:
        return None

    def plot_learning_curves(
        self,
        agent_scores: Dict[str, Sequence[float]],
        file_path: str | Path,
        window_size: int = 100,
        title: str = "Multi-Agent Learning Curves",
        extend_plot: bool = False,
    ) -> None:
        return None


class MatplotlibPlotter(Plotter):
    def plot_learning_curve(
        self,
        scores: Sequence[float],
        file_path: str | Path,
        window_size: int = 100,
        title: str = "Agent Learning Curve",
        extend_plot: bool = False,
    ) -> None:
        if not self.is_enabled("learning_curve"):
            return

        path = Path(file_path)
        merged_scores = self._merge_series_history(path, list(scores), extend_plot)
        moving_avg = pd.Series(merged_scores).rolling(window=window_size).mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(merged_scores, "b-", alpha=0.1, label="Raw Episode Score")
        ax.plot(
            moving_avg.index,
            moving_avg,
            "r-",
            linewidth=2,
            label=f"Moving Average (Window={window_size})",
        )
        self._finalize_plot(fig, ax, path, title, "Total Reward")
        self._save_history(path, merged_scores)

    def plot_learning_curves(
        self,
        agent_scores: Dict[str, Sequence[float]],
        file_path: str | Path,
        window_size: int = 100,
        title: str = "Multi-Agent Learning Curves",
        extend_plot: bool = False,
    ) -> None:
        if not self.is_enabled("multi_learning_curve"):
            return

        path = Path(file_path)
        merged_scores = {agent_name: list(scores) for agent_name, scores in agent_scores.items()}
        if extend_plot and self._history_path(path).exists():
            previous_scores = self._load_history(path)
            merged_scores = {
                agent_name: list(previous_scores.get(agent_name, [])) + scores
                for agent_name, scores in merged_scores.items()
            }
            totals = [(name, len(scores)) for name, scores in merged_scores.items()]
            print(f"Loaded previous scores. Total episodes per agent: {totals}")

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.tab10(range(len(merged_scores)))
        for (agent_name, scores), color in zip(merged_scores.items(), colors):
            moving_avg = pd.Series(scores).rolling(window=window_size).mean()
            ax.plot(scores, alpha=0.1, color=color)
            ax.plot(
                moving_avg.index,
                moving_avg,
                linewidth=2,
                label=f"{agent_name} (MA={window_size})",
                color=color,
            )

        self._finalize_plot(fig, ax, path, title, "Total Reward")
        self._save_history(path, merged_scores)

    def _merge_series_history(self, file_path: Path, scores: list[float], extend_plot: bool) -> list[float]:
        if not extend_plot or not self._history_path(file_path).exists():
            return scores

        previous_scores = list(self._load_history(file_path))
        merged_scores = previous_scores + scores
        print(f"Loaded {len(previous_scores)} previous scores. Now plotting {len(merged_scores)} total.")
        return merged_scores

    def _finalize_plot(self, fig, ax, file_path: Path, title: str, y_label: str) -> None:
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        try:
            os.makedirs(file_path.parent, exist_ok=True)
            fig.savefig(file_path)
            print(f"Plot successfully saved to {file_path}")
        except Exception as error:
            print(f"Error saving plot: {error}")
        finally:
            plt.close(fig)

    def _history_path(self, file_path: Path) -> Path:
        return file_path.with_name(f"{file_path.stem}_scores.pkl")

    def _load_history(self, file_path: Path):
        with open(self._history_path(file_path), "rb") as file:
            return pickle.load(file)

    def _save_history(self, file_path: Path, payload) -> None:
        with open(self._history_path(file_path), "wb") as file:
            pickle.dump(payload, file)


DEFAULT_PLOTTER = MatplotlibPlotter()


def plot_learning_curve(
    scores: Sequence[float],
    file_path: str | Path,
    window_size: int = 100,
    title: str = "Agent Learning Curve",
    extend_plot: bool = False,
    plotter: Plotter | None = None,
) -> None:
    active_plotter = plotter or DEFAULT_PLOTTER
    active_plotter.plot_learning_curve(
        scores=scores,
        file_path=file_path,
        window_size=window_size,
        title=title,
        extend_plot=extend_plot,
    )


def plot_learning_curves(
    agent_scores: Dict[str, Sequence[float]],
    file_path: str | Path,
    window_size: int = 100,
    title: str = "Multi-Agent Learning Curves",
    extend_plot: bool = False,
    plotter: Plotter | None = None,
) -> None:
    active_plotter = plotter or DEFAULT_PLOTTER
    active_plotter.plot_learning_curves(
        agent_scores=agent_scores,
        file_path=file_path,
        window_size=window_size,
        title=title,
        extend_plot=extend_plot,
    )
