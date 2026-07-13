"""Chapter 10 project: end-to-end migration of a toy decoder subgraph.

This remains the main compatibility entrypoint. The real implementation now
lives in `examples/ch10_end_to_end_core.py`, while the chapter also exposes
focused scripts for the generic, operator, and sweep variants.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.ch10_end_to_end_core import ProjectConfig, run_project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 10 end-to-end migration project")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--block-size", type=int, default=8, help="Block size for blockwise attention")
    parser.add_argument("--runs", type=int, default=10, help="Benchmark runs")
    parser.add_argument("--case-style", choices=["generic", "operator", "public"], default="operator", help="Case wording style for chapter 10")
    parser.add_argument("--subgraph-label", type=str, default="decoder block", help="Readable label for the migrated subgraph")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["cuda", "rocm", "ascend"],
        help="Target platforms to record migration notes for",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a small sequence-length sweep to mimic a more realistic project study",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write a JSON summary to artifacts/ch10_end_to_end_migration_report.json",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Custom path for the JSON summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig(
        batch=args.batch,
        seq_len=args.seq_len,
        dim=args.dim,
        block_size=args.block_size,
        runs=args.runs,
    )
    run_project(
        config,
        args.targets,
        case_style=args.case_style,
        subgraph_label=args.subgraph_label,
        sweep=args.sweep,
        write_report=args.write_report,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()
