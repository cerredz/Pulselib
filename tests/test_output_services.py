from pathlib import Path

import yaml

from utils.benchmarking.benchmarking import YamlBenchmarker, create_benchmark_file
from utils.plotting import MatplotlibPlotter, plot_learning_curve


def test_plotter_respects_feature_mask(tmp_path):
    plot_path = tmp_path / "reward_curve.png"
    plotter = MatplotlibPlotter(feature_mask={"learning_curve": False})

    plotter.plot_learning_curve([1.0, 2.0, 3.0], plot_path, window_size=2)

    assert not plot_path.exists()
    assert not (tmp_path / "reward_curve_scores.pkl").exists()


def test_plot_wrapper_creates_plot_and_history(tmp_path):
    plot_path = tmp_path / "reward_curve.png"

    plot_learning_curve([1.0, 2.0, 3.0], plot_path, window_size=2)

    assert plot_path.exists()
    assert (tmp_path / "reward_curve_scores.pkl").exists()


def test_benchmarker_respects_feature_mask(tmp_path):
    benchmarker = YamlBenchmarker(
        feature_mask={"training_summary": False},
        results_dir_resolver=lambda _env_name: tmp_path,
    )

    result = benchmarker.create_benchmark_file("Pulse-Poker-GPU-v1", [1.0, 2.0], 0.0, 2.0, 10, {})

    assert result is None
    assert not (tmp_path / "runs").exists()


def test_benchmark_wrapper_writes_yaml_summary(tmp_path):
    benchmarker = YamlBenchmarker(results_dir_resolver=lambda _env_name: tmp_path)

    result_path = create_benchmark_file(
        env_name="Pulse-Poker-GPU-v1",
        episodes_return=[1.0, 2.0, 3.0],
        start_time=0.0,
        end_time=2.0,
        total_steps=20,
        config={"alpha": 1},
        benchmarker=benchmarker,
    )

    assert result_path == Path(tmp_path / "runs" / "run_1.yaml")
    with open(result_path, "r") as file:
        payload = yaml.safe_load(file)

    assert payload["env"] == "Pulse-Poker-GPU-v1"
    assert payload["sps"] == 10.0
    assert payload["episode_stats"]["count"] == 3
