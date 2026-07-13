"""Chapter 7 demo: standard attention and blockwise online softmax.

This script is CPU-only and focuses on the data-flow relationship between
standard attention and a FlashAttention-style blockwise implementation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmark_record_template import set_benchmark_record
from examples.ch07_attention_core import attention_blockwise, attention_reference, benchmark, make_attention_inputs


def demo(
    mode: str = "all",
    batch: int = 1,
    seq_len: int = 32,
    dim: int = 64,
    block_size: int = 8,
    runs: int = 20,
) -> None:
    np.random.seed(0)
    q, k, v = make_attention_inputs(batch, seq_len, dim)

    print("Chapter 7: Attention Demo")
    print("=" * 72)
    ref = None
    blk = None
    if mode in {"reference", "all"}:
        ref, ref_ms = benchmark(attention_reference, q, k, v, runs=runs)
        print(f"Reference attention: {ref_ms:.4f} ms per run")
    if mode in {"blockwise", "all"}:
        blk, blk_ms = benchmark(attention_blockwise, q, k, v, runs=runs)
        print(f"Blockwise attention:  {blk_ms:.4f} ms per run (block_size={block_size})")
    if ref is not None and blk is not None:
        max_err = np.max(np.abs(ref - blk))
        print(f"Max error: {max_err:.2e}")
    print()
    print("Interpretation:")
    print("- The reference version materializes the full scores/probs tensors.")
    print("- The blockwise version keeps only running statistics on the fly.")
    print("- This is the core idea behind FlashAttention-style IO reduction.")

    record = set_benchmark_record(
        scenario="chapter 7 attention demo",
        operator="Attention",
        platform="CPU",
        target="numpy",
        dtype="float32",
        shape=f"B={batch},S={seq_len},D={dim}",
        baseline="standard attention",
        optimization=f"blockwise online softmax (block_size={block_size})",
    )
    print()
    print("Benchmark record skeleton:")
    for key in ["scenario", "operator", "platform", "target", "dtype", "shape", "baseline", "optimization"]:
        print(f"  {key}: {record[key]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Chapter 7 attention demo")
    parser.add_argument(
        "--mode",
        choices=["reference", "blockwise", "all"],
        default="all",
        help="Which attention variant to run",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--block-size", type=int, default=8, help="Block size for blockwise attention")
    parser.add_argument("--runs", type=int, default=20, help="Benchmark runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo(args.mode, args.batch, args.seq_len, args.dim, args.block_size, args.runs)
