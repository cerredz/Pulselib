from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_output_path(output_dir: Path, preset: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"poker_gpu_benchmark_{preset}_{stamp}.json"


def write_json_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return output_path


def emit_llm_summary(report: dict[str, Any]) -> None:
    metadata = report["metadata"]
    print("LLM_BENCHMARK_SUMMARY_BEGIN")
    print(f"benchmark_suite={metadata['suite_name']}")
    print(f"preset={metadata['preset']}")
    print(f"device={metadata['device']}")
    print(f"cases_run={len(report['cases'])}")
    print(f"output_path={report['output_path']}")
    for case in report["cases"]:
        print(
            "case="
            f"{case['name']} "
            f"category={case['category']} "
            f"unit={case['primary_metric']['unit']} "
            f"value={case['primary_metric']['value']:.6f} "
            f"lower_is_better={str(case['primary_metric']['lower_is_better']).lower()}"
        )
        for derived in case.get("derived_metrics", []):
            print(
                "derived="
                f"{case['name']} "
                f"{derived['name']}={derived['value']:.6f} "
                f"unit={derived['unit']} "
                f"higher_is_better={str(derived['higher_is_better']).lower()}"
            )
    print("LLM_BENCHMARK_SUMMARY_END")
