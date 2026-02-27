#!/usr/bin/env python3
"""Profile mask gen breakdown for representative schemas."""
import json, os, sys, time, subprocess, tempfile
from pathlib import Path

import _sep1

CFA_ROOT = Path(os.path.expanduser("~/Projects2/constraint-framework-analysis"))
GRAMMARS_ROOT = Path(os.path.expanduser("~/Projects2/grammars2024"))
COMPILER = GRAMMARS_ROOT / "target" / "release" / "grammar-compiler"

sys.path.insert(0, str(GRAMMARS_ROOT))
from python.aug25.models.rust_model import Model as RustModel

def compile_schema(schema_id):
    schema_path = CFA_ROOT / "data" / "sources" / "jsonschemabench" / "data" / (schema_id + ".json")
    if not schema_path.exists():
        return None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as of:
        output_path = Path(of.name)
    try:
        result = subprocess.run(
            [str(COMPILER), "--vocab", "/tmp/vocab.json",
             "--json-schema", str(schema_path), "--output", str(output_path)],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "MACRO_DEBUG_LEVEL": "0"},
        )
        if result.returncode != 0:
            return None
        with open(output_path) as f:
            return f.read()
    except Exception:
        return None
    finally:
        if output_path.exists():
            os.unlink(output_path)

SCHEMAS = [
    "Github_hard/o9767",      # worst new ratio (9.8x)
    "Github_hard/o82811",     # regressed (4.2x)
    "Github_easy/o30452",     # typical slow (2.7x)
    "Github_medium/o29961",   # improved (1.7x)
    "Github_easy/o68689",     # improved (1.8x)
    "Github_easy/o10008",     # fast schema
]

N_STEPS = 30

for sid in SCHEMAS:
    cj = compile_schema(sid)
    if cj is None:
        print(f"FAIL: {sid}")
        continue
    
    model = RustModel.from_json_string(cj)
    _sep1.set_benchmark_mode(True)
    
    print(f"\n{'='*80}")
    print(f"Schema: {sid}")
    print(f"{'Step':>4s} {'Wall':>8s} {'Compute':>8s} {'Convert':>8s} {'Seed':>8s} {'WL':>8s} {'Expand':>8s} {'Intrs':>8s} {'GSS':>8s} {'Merge':>8s} {'Final':>8s} {'Iters':>5s} {'ExpN':>4s}")
    print("-" * 108)
    
    for step in range(N_STEPS):
        t0 = time.perf_counter()
        mask = model.get_mask()
        t1 = time.perf_counter()
        
        wall_us = (t1 - t0) * 1e6
        compute_us = _sep1.get_last_mask_compute_time_ns() / 1000
        convert_us = _sep1.get_last_mask_convert_time_ns() / 1000
        seed_us = _sep1.get_last_mask_seed_time_ns() / 1000
        wl_us = _sep1.get_last_mask_worklist_time_ns() / 1000
        expand_us = _sep1.get_last_mask_wl_expand_ns() / 1000
        intersect_us = _sep1.get_last_mask_wl_intersect_ns() / 1000
        gss_us = _sep1.get_last_mask_wl_gss_ns() / 1000
        merge_us = _sep1.get_last_mask_wl_merge_ns() / 1000
        final_us = _sep1.get_last_mask_wl_final_ns() / 1000
        iters = _sep1.get_last_mask_worklist_iter_count()
        exp_count = _sep1.get_last_mask_wl_expand_count()
        
        print(f"{step:>4d} {wall_us:>7.0f}μs {compute_us:>7.0f}μs {convert_us:>7.0f}μs {seed_us:>7.0f}μs {wl_us:>7.0f}μs {expand_us:>7.0f}μs {intersect_us:>7.0f}μs {gss_us:>7.0f}μs {merge_us:>7.0f}μs {final_us:>7.0f}μs {iters:>5d} {exp_count:>4d}")
        
        ranges = list(mask.to_ranges())
        num_allowed = sum(end - start + 1 for start, end in ranges)
        if num_allowed == 0:
            break
        model.commit(ranges[0][0])
