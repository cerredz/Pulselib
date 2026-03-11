from __future__ import annotations

import argparse
from pathlib import Path

from benchmarking.Poker.cases import CASE_REGISTRY
from benchmarking.Poker.runner import run_benchmarks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Poker GPU benchmarks against the live PulseLib codepaths.")
    parser.add_argument(
        "--preset",
        default="standard",
        help="Benchmark preset to run. Default: standard",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        help="Run only the named benchmark case. Repeat to select multiple cases.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for the JSON report. Default: results/benchmarks/Poker",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override such as cuda or cuda:0.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available benchmark cases and exit.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_cases:
        for case_name, case in CASE_REGISTRY.items():
            print(f"{case_name}: {case.description}")
        return 0

    run_benchmarks(
        preset_name=args.preset,
        selected_cases=args.cases,
        output_dir=args.output_dir,
        device_override=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
