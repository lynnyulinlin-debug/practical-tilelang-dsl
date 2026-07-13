"""Chapter 1 focused demo: kernel vs DSL vs compiler decisions."""

from __future__ import annotations

import argparse


def classify_scenario(maintainability: str, shape_volatility: str, backend_count: int) -> str:
    score = 0
    score += 2 if maintainability == "high" else 0
    score += 2 if shape_volatility == "high" else 0
    score += 2 if backend_count >= 2 else 0
    if score >= 4:
        return "Prefer DSL + compiler"
    if score >= 2:
        return "Hybrid: keep a kernel baseline, move control to DSL"
    return "Hand-written kernel is still reasonable"


def main() -> None:
    cases = [
        ("low", "low", 1),
        ("high", "low", 1),
        ("high", "high", 2),
    ]

    print("Chapter 1: Kernel vs DSL decisions")
    print("=" * 72)
    for maintainability, shape_volatility, backend_count in cases:
        decision = classify_scenario(maintainability, shape_volatility, backend_count)
        print(
            f"maintainability={maintainability:<4} "
            f"shape_volatility={shape_volatility:<4} "
            f"backends={backend_count} -> {decision}"
        )

    print()
    print("Interpretation:")
    print("- Kernel is the execution unit.")
    print("- DSL is the way to describe a family of kernels.")
    print("- Compiler gives optimization room, but only within the described constraints.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 1 DSL decision demo")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    main()
