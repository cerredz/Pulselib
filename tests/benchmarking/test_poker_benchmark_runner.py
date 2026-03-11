from pathlib import Path

from benchmarking.Poker.cases import BenchmarkCase
from benchmarking.Poker.runner import run_benchmarks


def test_poker_benchmark_runner_writes_structured_report(tmp_path, monkeypatch):
    import benchmarking.Poker.runner as runner_module

    def fake_resolve_preset(name):
        assert name == "quick"
        return {
            "device": "cuda",
            "cases": ["fake_case"],
            "warmup_iterations": 1,
            "measure_iterations": 2,
            "env": {"n_games": 8, "episodes": 2},
        }

    class FakeContext:
        device = "cuda"
        benchmark_config = {"N_GAMES": 8, "EPISODES": 2}

    def fake_load_context(preset, root_dir):
        del preset, root_dir
        return FakeContext()

    def fake_runner(case, context, warmups, trials):
        del warmups
        return {
            "name": case.name,
            "category": case.category,
            "description": case.description,
            "primary_metric": {
                "name": case.primary_metric_name,
                "unit": case.primary_metric_unit,
                "value": 0.123,
                "lower_is_better": case.lower_is_better,
            },
            "timings": {
                "unit": "seconds",
                "trials": [0.12, 0.13],
                "mean": 0.123,
                "median": 0.123,
                "min": 0.12,
                "max": 0.13,
                "stdev": 0.007,
            },
            "derived_metrics": [
                {
                    "name": "throughput",
                    "value": 99.0,
                    "unit": "items_per_second",
                    "higher_is_better": True,
                }
            ],
            "metadata": {"trials": trials, "context_device": context.device},
        }

    monkeypatch.setattr(runner_module, "resolve_preset", fake_resolve_preset)
    monkeypatch.setattr(runner_module, "load_benchmark_context", fake_load_context)
    monkeypatch.setitem(
        runner_module.CASE_REGISTRY,
        "fake_case",
        BenchmarkCase(
            name="fake_case",
            category="trainer",
            description="Fake benchmark case for runner smoke testing.",
            primary_metric_name="elapsed_seconds",
            primary_metric_unit="seconds",
            lower_is_better=True,
            runner=fake_runner,
        ),
    )

    report = run_benchmarks(
        preset_name="quick",
        selected_cases=["fake_case"],
        output_dir=tmp_path,
    )

    assert report["metadata"]["suite_name"] == "poker_gpu_benchmarking"
    assert len(report["cases"]) == 1
    case = report["cases"][0]
    assert case["name"] == "fake_case"
    assert case["primary_metric"]["value"] >= 0
    assert case["derived_metrics"]

    output_path = Path(report["output_path"])
    assert output_path.exists()
    assert output_path.parent == tmp_path
