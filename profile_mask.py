#!/usr/bin/env python3
"""Profile sep1 mask generation step breakdown (seed vs worklist)."""

import json
import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path

import _sep1

CFA_ROOT = Path(os.path.expanduser("~/Projects2/constraint-framework-analysis"))
GRAMMARS_ROOT = Path(os.path.expanduser("~/Projects2/grammars2024"))
COMPILER = GRAMMARS_ROOT / "target" / "release" / "grammar-compiler"

sys.path.insert(0, str(GRAMMARS_ROOT))
from python.aug25.models.rust_model import Model as RustModel


def compile_schema(schema_path: Path) -> str:
    """Compile a JSON schema and return the constraint JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as of:
        output_path = Path(of.name)

    try:
        result = subprocess.run(
            [str(COMPILER), "--vocab", "/tmp/vocab.json",
             "--json-schema", str(schema_path), "--output", str(output_path)],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "MACRO_DEBUG_LEVEL": "0"},
        )
        if result.returncode != 0:
            print(f"Compile failed: {result.stderr[:500]}")
            return None
        with open(output_path) as f:
            return f.read()
    finally:
        if output_path.exists():
            os.unlink(output_path)


def profile_schema(schema_id: str, n_steps: int = 20):
    """Profile mask generation for a schema."""
    # schema_id like "Github_easy/o10008"
    schema_path = CFA_ROOT / "data" / "sources" / "jsonschemabench" / "data" / (schema_id + ".json")
    if not schema_path.exists():
        print(f"Schema not found: {schema_path}")
        return

    print(f"\n{'='*70}")
    print(f"Schema: {schema_id}")

    # Compile
    t0 = time.time()
    constraint_json = compile_schema(schema_path)
    compile_time = time.time() - t0
    if constraint_json is None:
        return
    print(f"Compile time: {compile_time:.3f}s")

    # Load model
    model = RustModel.from_json_string(constraint_json)

    # Enable benchmark mode
    _sep1.set_benchmark_mode(True)

    # Run mask steps
    step_data = []
    for step in range(n_steps):
        t0 = time.time()
        mask = model.get_mask()
        wall_time = time.time() - t0

        seed_ns = _sep1.get_last_mask_seed_time_ns()
        worklist_ns = _sep1.get_last_mask_worklist_time_ns()
        worklist_iters = _sep1.get_last_mask_worklist_iter_count()
        wl_expand_ns = _sep1.get_last_mask_wl_expand_ns()
        wl_intersect_ns = _sep1.get_last_mask_wl_intersect_ns()
        wl_gss_ns = _sep1.get_last_mask_wl_gss_ns()
        wl_merge_ns = _sep1.get_last_mask_wl_merge_ns()
        wl_final_ns = _sep1.get_last_mask_wl_final_ns()
        wl_expand_count = _sep1.get_last_mask_wl_expand_count()

        # Get allowed tokens
        ranges = list(mask.to_ranges())
        num_allowed = sum(end - start + 1 for start, end in ranges)

        seed_us = seed_ns / 1000.0
        wl_us = worklist_ns / 1000.0
        wall_us = wall_time * 1e6
        other_us = wall_us - seed_us - wl_us

        step_data.append({
            "step": step,
            "wall_us": wall_us,
            "seed_us": seed_us,
            "wl_us": wl_us,
            "other_us": other_us,
            "wl_iters": worklist_iters,
            "num_allowed": num_allowed,
            "wl_expand_us": wl_expand_ns / 1000.0,
            "wl_intersect_us": wl_intersect_ns / 1000.0,
            "wl_gss_us": wl_gss_ns / 1000.0,
            "wl_merge_us": wl_merge_ns / 1000.0,
            "wl_final_us": wl_final_ns / 1000.0,
            "wl_expand_count": wl_expand_count,
        })

        if num_allowed == 0:
            print(f"  Step {step}: no allowed tokens, stopping")
            break

        # Commit the first allowed token
        first_token = ranges[0][0]
        model.commit(first_token)

    # Print table
    fmt = "{:>4s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>8s} {:>10s} {:>8s}"
    print(fmt.format("Step", "Wall(us)", "Seed(us)", "WL(us)", "  Expand", "  Intersct", "  GSS", "  Merge", "  Final", "Allowed"))
    for d in step_data:
        print(f"{d['step']:4d} {d['wall_us']:10.1f} {d['seed_us']:10.1f} {d['wl_us']:10.1f} {d['wl_expand_us']:10.1f} {d['wl_intersect_us']:10.1f} {d['wl_gss_us']:8.1f} {d['wl_merge_us']:10.1f} {d['wl_final_us']:8.1f} {d['num_allowed']:8d}")

    if len(step_data) > 1:
        data = step_data[1:]  # exclude first step
        n = len(data)
        avg_wall = sum(d['wall_us'] for d in data) / n
        avg_seed = sum(d['seed_us'] for d in data) / n
        avg_wl = sum(d['wl_us'] for d in data) / n
        avg_other = sum(d['other_us'] for d in data) / n
        avg_iters = sum(d['wl_iters'] for d in data) / n
        avg_expand = sum(d['wl_expand_us'] for d in data) / n
        avg_intersect = sum(d['wl_intersect_us'] for d in data) / n
        avg_gss = sum(d['wl_gss_us'] for d in data) / n
        avg_merge = sum(d['wl_merge_us'] for d in data) / n
        avg_final = sum(d['wl_final_us'] for d in data) / n
        avg_exp_count = sum(d['wl_expand_count'] for d in data) / n
        print(f"\nAvg (excl step 0): Wall={avg_wall:.1f}us  Seed={avg_seed:.1f}us ({100*avg_seed/avg_wall:.0f}%)  WL={avg_wl:.1f}us ({100*avg_wl/avg_wall:.0f}%)")
        print(f"  WL breakdown: Expand={avg_expand:.1f}us  Intersect={avg_intersect:.1f}us  GSS={avg_gss:.1f}us  Merge={avg_merge:.1f}us  Final={avg_final:.1f}us")
        print(f"  Expand count={avg_exp_count:.1f}  WL iters={avg_iters:.1f}")

    return step_data


def main():
    schemas = [
        "Github_easy/o36272",   # worst sep1 vs llg (1.451ms, 19x)
        "Github_hard/o69862",   # complex schema
        "Github_easy/o10008",   # simple
    ]

    for schema_id in schemas:
        try:
            profile_schema(schema_id, n_steps=20)
        except Exception as e:
            print(f"Error profiling {schema_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
