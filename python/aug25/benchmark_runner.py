import argparse
import dataclasses
import gzip
import json
import os
import timeit
from pathlib import Path

from .grammar_constraint import GrammarConstraint
from .models import get_model_from_path


@dataclasses.dataclass
class BenchmarkResult:
    reference_model_name: str
    competitor_model_name: str
    code_file: str
    constraint_file: str
    reference_time_ms: float
    competitor_time_ms: float
    speedup_factor: float


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for grammar constraint models.")
    parser.add_argument(
        "--code",
        type=str,
        required=True,
        help="Path to the code file to use as input.",
    )
    parser.add_argument(
        "--constraint-file",
        type=str,
        required=True,
        help="Path to the pre-compiled .json.gz constraint file.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to the reference model Python file.",
    )
    parser.add_argument(
        "--competitor",
        type=str,
        required=True,
        help="Path to the competitor model Python file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save benchmark results.",
    )
    args = parser.parse_args()

    print(f"Loading code from {args.code}...")
    code = Path(args.code).read_text()
    print(f"Loading constraint from {args.constraint_file}...")
    with gzip.open(args.constraint_file, "rt", encoding="utf-8") as f:
        constraint = GrammarConstraint.from_json_string(f.read())
    print("Constraint loaded.")

    print(f"Loading reference model from {args.reference}...")
    reference_model = get_model_from_path(args.reference)
    print(f"Loading competitor model from {args.competitor}...")
    competitor_model = get_model_from_path(args.competitor)

    # --- Benchmark Reference Model ---
    print(f"\nBenchmarking reference model: {reference_model.name}...")
    reference_time = timeit.timeit(
        lambda: reference_model.check_constraint(code, constraint), number=100
    )
    print(f"Reference model time: {reference_time:.4f} seconds")

    # --- Benchmark Competitor Model ---
    print(f"\nBenchmarking competitor model: {competitor_model.name}...")
    competitor_time = timeit.timeit(
        lambda: competitor_model.check_constraint(code, constraint), number=100
    )
    print(f"Competitor model time: {competitor_time:.4f} seconds")

    # --- Calculate Speedup ---
    speedup_factor = reference_time / competitor_time
    print(f"\nSpeedup factor (Reference / Competitor): {speedup_factor:.2f}x")

    # --- Save Results ---
    results = BenchmarkResult(
        reference_model_name=reference_model.name,
        competitor_model_name=competitor_model.name,
        code_file=os.path.basename(args.code),
        constraint_file=os.path.basename(args.constraint_file),
        reference_time_ms=reference_time * 1000,
        competitor_time_ms=competitor_time * 1000,
        speedup_factor=speedup_factor,
    )

    output_filename = (
        f"{reference_model.name}_vs_{competitor_model.name}_"
        f"{os.path.basename(args.code).replace('.', '_')}.json"
    )
    output_path = Path(args.output) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(dataclasses.asdict(results), f, indent=2)

    print(f"\nBenchmark results saved to: {output_path}")


if __name__ == "__main__":
    main()
