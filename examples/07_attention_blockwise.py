"""Chapter 7 focused demo: blockwise online softmax only."""

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


def main(batch: int = 1, seq_len: int = 32, dim: int = 64, block_size: int = 8, runs: int = 20) -> None:
    np.random.seed(0)
    q, k, v = make_attention_inputs(batch, seq_len, dim)
    ref, ref_ms = benchmark(attention_reference, q, k, v, runs=runs)
    out, elapsed_ms = benchmark(attention_blockwise, q, k, v, runs=runs, block_size=block_size)

    print("Chapter 7: Blockwise Attention")
    print("=" * 72)
    print(f"Shape: B={batch}, S={seq_len}, D={dim}")
    print(f"Block size: {block_size}")
    print(f"Reference latency: {ref_ms:.4f} ms per run")
    print(f"Blockwise latency:  {elapsed_ms:.4f} ms per run")
    print(f"Max error: {np.max(np.abs(ref - out)):.2e}")
    print()
    print("Interpretation:")
    print("- This version keeps only running statistics on the fly.")
    print("- It demonstrates the IO-reduction idea behind FlashAttention.")
    print()

    record = set_benchmark_record(
        scenario="chapter 7 attention blockwise",
        operator="Attention",
        platform="CPU",
        target="numpy",
        dtype="float32",
        shape=f"B={batch},S={seq_len},D={dim}",
        baseline="standard attention",
        optimization=f"blockwise online softmax (block_size={block_size})",
    )
    print("Benchmark record skeleton:")
    for key in ["scenario", "operator", "platform", "target", "dtype", "shape", "baseline", "optimization"]:
        print(f"  {key}: {record[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 7 blockwise attention demo")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--block-size", type=int, default=8, help="Block size for blockwise attention")
    parser.add_argument("--runs", type=int, default=20, help="Benchmark runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.batch, args.seq_len, args.dim, args.block_size, args.runs)
