#!/usr/bin/env python3
"""
Benchmark nullable-inlining strategies against a synthetic grammar
parameterised by m: the number of nullable nonterminals in a sequence.

Grammar shape:
    start: NT_1 NT_2 ... NT_m
    NT_k: "x" | ""            <- nullable singleton for each k

Usage:
    python bench_null_inline.py [--m-values 1 2 4 8 16 32] [--timeout 30]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

COMPILER = Path(__file__).parent / "target" / "debug" / "grammar-compiler"

STRATEGIES = [
    "exhaustive",
    "right_chain",
    "left_chain",
    "balanced_tree_2",
    "balanced_tree_4",
    "balanced_tree_8",
]


def make_lark_grammar(m: int) -> str:
    """Generate a Lark grammar: start → NT_0 NT_1 … NT_{m-1}.
    Each NT_k → "t" k_str to use two separate terminal tokens per NT.
    Using two-terminal bodies prevents unit-elimination from folding NT_k
    into a single terminal, preserving distinct grammar structure per NT."""
    lines = []
    nt_names = [f"nt_{k}" for k in range(m)]
    lines.append("start: " + " ".join(nt_names))
    for k, nt in enumerate(nt_names):
        lines.append(f'{nt}: "t" "{k}" | ""')
    return "\n".join(lines) + "\n"


def make_dummy_vocab(m: int) -> str:
    """Write a JSON vocab file: 't' for the common prefix token, and '0'..'m-1' for the index tokens."""
    path = "/tmp/bench_null_inline_vocab.json"
    vocab = {"t": 0}
    for k in range(m):
        vocab[str(k)] = k + 1
    with open(path, "w") as f:
        json.dump(vocab, f)
    return path


def run_strategy(grammar_path: str, vocab_path: str, strategy: str, timeout: float) -> dict:
    """Run the grammar-compiler with the given strategy.  Returns a dict with timing/stats."""
    env = dict(os.environ)
    env["MACRO_DEBUG_LEVEL"] = "5"
    env["NULL_INLINE_STRATEGY"] = strategy

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [str(COMPILER), "--grammar", grammar_path, "--format", "lark",
             "--vocab", vocab_path, "--output", "/tmp/bench_null_inline_out.bin"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"elapsed": timeout, "timed_out": True, "prods": None, "states": None}

    elapsed = time.monotonic() - t0
    prods = states = None
    for line in result.stderr.splitlines():
        m = re.search(r"glr_normalization_loop.*?(\d+) prods", line)
        if m:
            prods = int(m.group(1))
        m2 = re.search(r"glr_stage1_lr0.*?(\d+) states", line)
        if m2:
            states = int(m2.group(1))

    return {
        "elapsed": elapsed,
        "timed_out": result.returncode != 0 and prods is None,
        "prods": prods,
        "states": states,
        "rc": result.returncode,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-values", type=int, nargs="+",
                        default=[1, 2, 4, 8, 12, 16, 20, 24, 28, 32],
                        help="Values of m to test")
    parser.add_argument("--timeout", type=float, default=1.0,
                        help="Timeout per test run in seconds (default: 1s; skip strategy after first timeout)")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES,
                        help="Strategies to test")
    args = parser.parse_args()

    if not COMPILER.exists():
        print(f"ERROR: compiler not found at {COMPILER}", file=sys.stderr)
        print("Run: cargo build --release", file=sys.stderr)
        sys.exit(1)

    # Header
    col_w = 18
    strategy_cols = args.strategies
    header = f"{'m':>4} | " + " | ".join(f"{s[:col_w]:>{col_w}}" for s in strategy_cols)
    sep = "-" * len(header)

    def run_table(label: str, cell_fn):
        print(f"\n{label}")
        print(sep)
        print(header + "  (T=timeout, -=skipped)")
        print(sep)

        timed_out: set[str] = set()
        for m in args.m_values:
            grammar = make_lark_grammar(m)
            vocab_path = make_dummy_vocab(m)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".lark", delete=False) as f:
                f.write(grammar)
                grammar_path = f.name

            row_parts = [f"{m:>4}"]
            for strategy in strategy_cols:
                if strategy in timed_out:
                    row_parts.append(f"{'-':>{col_w}}")
                    continue
                r = run_strategy(grammar_path, vocab_path, strategy, args.timeout)
                if r["timed_out"]:
                    timed_out.add(strategy)
                    row_parts.append(f"{'T':>{col_w}}")
                else:
                    row_parts.append(f"{cell_fn(r):>{col_w}}")

            os.unlink(grammar_path)
            print(f"{' | '.join(row_parts)}")
            sys.stdout.flush()

        print(sep)

    run_table("GLR states vs m", lambda r: str(r["states"]) if r["states"] is not None else "?")
    run_table("Build time (s) vs m", lambda r: f"{r['elapsed']:.3f}")


if __name__ == "__main__":
    main()
