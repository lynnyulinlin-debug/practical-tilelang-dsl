"""Chapter 10 focused demo: operator-level project layer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.ch10_end_to_end_core import ProjectConfig, run_project


def main(
    batch: int,
    seq_len: int,
    dim: int,
    block_size: int,
    runs: int,
    case_style: str,
    subgraph_label: str,
    targets: list[str],
    sweep: bool,
    write_report: bool,
    report_path: str | None,
) -> None:
    run_project(
        ProjectConfig(batch=batch, seq_len=seq_len, dim=dim, block_size=block_size, runs=runs),
        targets,
        case_style=case_style,
        subgraph_label=subgraph_label,
        sweep=sweep,
        write_report=write_report,
        report_path=report_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 10 operator-level demo")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--block-size", type=int, default=8, help="Block size for blockwise attention")
    parser.add_argument("--runs", type=int, default=10, help="Benchmark runs")
    parser.add_argument("--case-style", choices=["generic", "operator", "public"], default="operator", help="Case wording style for chapter 10")
    parser.add_argument("--subgraph-label", type=str, default="decoder block", help="Readable label for the migrated subgraph")
    parser.add_argument("--targets", nargs="*", default=["cuda", "rocm", "ascend"], help="Target platforms to record migration notes for")
    parser.add_argument("--sweep", action="store_true", help="Run a small sequence-length sweep")
    parser.add_argument("--write-report", action="store_true", help="Write a JSON summary")
    parser.add_argument("--report-path", type=str, default=None, help="Custom path for the JSON summary")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.batch,
        args.seq_len,
        args.dim,
        args.block_size,
        args.runs,
        args.case_style,
        args.subgraph_label,
        args.targets,
        args.sweep,
        args.write_report,
        args.report_path,
    )
