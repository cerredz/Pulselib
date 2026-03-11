from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from benchmarking.Poker.cases import CASE_REGISTRY
from benchmarking.Poker.presets import resolve_preset
from benchmarking.Poker.reporting import build_output_path, emit_llm_summary, ensure_output_dir, write_json_report
from benchmarking.Poker.runtime import load_benchmark_context


def run_benchmarks(
    *,
    preset_name: str = "standard",
    selected_cases: list[str] | None = None,
    output_dir: str | Path | None = None,
    device_override: str | None = None,
) -> dict:
    preset = resolve_preset(preset_name)
    if device_override is not None:
        preset["device"] = device_override
    root_dir = Path(__file__).resolve().parents[2]
    context = load_benchmark_context(preset, root_dir)
    case_names = selected_cases or preset["cases"]
    output_root = ensure_output_dir(Path(output_dir) if output_dir else root_dir / "results" / "benchmarks" / "Poker")

    case_results = []
    for case_name in case_names:
        if case_name not in CASE_REGISTRY:
            available = ", ".join(sorted(CASE_REGISTRY))
            raise ValueError(f"Unknown case '{case_name}'. Available cases: {available}")
        case = CASE_REGISTRY[case_name]
        case_result = case.runner(
            case,
            context,
            preset["warmup_iterations"],
            preset["measure_iterations"],
        )
        case_results.append(case_result)

    report = {
        "metadata": {
            "suite_name": "poker_gpu_benchmarking",
            "preset": preset_name,
            "device": str(context.device),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "warmup_iterations": preset["warmup_iterations"],
            "measure_iterations": preset["measure_iterations"],
            "config_source": "config/pokerGPU.yaml",
            "benchmark_overrides": {
                "N_GAMES": context.benchmark_config["N_GAMES"],
                "EPISODES": context.benchmark_config["EPISODES"],
            },
        },
        "cases": case_results,
    }
    output_path = build_output_path(output_root, preset_name)
    report["output_path"] = str(write_json_report(report, output_path))
    emit_llm_summary(report)
    return report
