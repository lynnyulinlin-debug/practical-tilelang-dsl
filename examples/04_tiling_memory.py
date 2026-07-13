"""Chapter 4 focused demo: tile, memory tiers, and data reuse."""

from __future__ import annotations

import argparse

import numpy as np


def estimate_bytes(shape: tuple[int, ...], dtype=np.float32) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n * np.dtype(dtype).itemsize


def main(m: int = 64, n: int = 64, k: int = 64, bm: int = 32, bn: int = 32, bk: int = 16) -> None:
    naive = estimate_bytes((m, k)) + estimate_bytes((k, n)) + estimate_bytes((m, n))
    blocked = estimate_bytes((m, k)) + estimate_bytes((k, n)) + estimate_bytes((bm, bk)) + estimate_bytes((bk, bn))

    print("Chapter 4: Tiling and memory")
    print("=" * 72)
    print(f"Shape: M={m}, N={n}, K={k}")
    print(f"Tile:  bm={bm}, bn={bn}, bk={bk}")
    print(f"Naive working-set estimate:   {naive / (1024**2):.3f} MiB")
    print(f"Blocked working-set estimate: {blocked / (1024**2):.3f} MiB")
    print()
    print("Interpretation:")
    print("- Tile controls where data lives and who reuses it.")
    print("- Shared memory helps only when the tile matches the reuse pattern.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 tiling and memory demo")
    parser.add_argument("--m", type=int, default=64, help="GEMM M dimension")
    parser.add_argument("--n", type=int, default=64, help="GEMM N dimension")
    parser.add_argument("--k", type=int, default=64, help="GEMM K dimension")
    parser.add_argument("--bm", type=int, default=32, help="Tile M size")
    parser.add_argument("--bn", type=int, default=32, help="Tile N size")
    parser.add_argument("--bk", type=int, default=16, help="Tile K size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.m, args.n, args.k, args.bm, args.bn, args.bk)
