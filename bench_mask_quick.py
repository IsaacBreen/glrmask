#!/usr/bin/env python3
"""Quick benchmark: sep1 mask gen with caching vs llguidance baseline."""
import json, os, sys, time, subprocess, tempfile
from pathlib import Path

import _sep1

CFA_ROOT = Path(os.path.expanduser("~/Projects2/constraint-framework-analysis"))
GRAMMARS_ROOT = Path(os.path.expanduser("~/Projects2/grammars2024"))
COMPILER = GRAMMARS_ROOT / "target" / "release" / "grammar-compiler"

sys.path.insert(0, str(GRAMMARS_ROOT))
from python.aug25.models.rust_model import Model as RustModel

# Load LLG baseline from sweep
with open(CFA_ROOT / "results" / "full_jsb_sweep.json") as f:
    sweep = json.load(f)
llg_baseline = {}
old_sep1_baseline = {}
for r in sweep["results"]:
    mt = r.get("mask_times_avg", {})
    sid = r.get("schema_id", "")
    if "llguidance" in mt and mt["llguidance"] > 0:
        llg_baseline[sid] = mt["llguidance"]
    if "sep1" in mt and mt["sep1"] > 0:
        old_sep1_baseline[sid] = mt["sep1"]

# Representative sample: worst sep1 cases + typical cases
SCHEMAS = [
    "Github_easy/o36272",     # worst ratio
    "Github_easy/o40223",     # 2nd worst
    "Github_medium/o30410",   # 3rd
    "Github_medium/o42988",   # 4th
    "Github_easy/o10030",     # 5th
    "Github_easy/o12418",     # 6th
    "Github_hard/o69862",     # complex
    "Github_easy/o10008",     # simple
    "Github_easy/o25962",     # top 10
    "Github_medium/o15131",   # top 10
    "Github_easy/o30452",     # top 10
    "Github_medium/o29961",   # top 10
    "Github_easy/o22983",     # random easy
    "Github_medium/o53019",   # random medium
    "Github_hard/o28543",     # random hard
]

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
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "MACRO_DEBUG_LEVEL": "0"},
        )
        if result.returncode != 0:
            return None
        with open(output_path) as f:
            return f.read()
    finally:
        if output_path.exists():
            os.unlink(output_path)

def benchmark_schema(schema_id, n_steps=100):
    cj = compile_schema(schema_id)
    if cj is None:
        return None
    model = RustModel.from_json_string(cj)
    _sep1.set_benchmark_mode(True)
    
    mask_times = []
    for step in range(n_steps):
        t0 = time.perf_counter()
        mask = model.get_mask()
        t1 = time.perf_counter()
        mask_times.append(t1 - t0)
        
        ranges = list(mask.to_ranges())
        num_allowed = sum(end - start + 1 for start, end in ranges)
        if num_allowed == 0:
            break
        model.commit(ranges[0][0])
    
    return mask_times

print(f"{'Schema':<40s} {'Old Sep1':>10s} {'New Sep1':>10s} {'LLG':>10s} {'Old/LLG':>8s} {'New/LLG':>8s} {'Speedup':>8s}")
print("-" * 106)

for sid in SCHEMAS:
    mask_times = benchmark_schema(sid, n_steps=100)
    if mask_times is None:
        print(f"{sid:<40s} {'FAIL':>10s}")
        continue
    
    # Exclude step 0 (cold start)
    warm_times = mask_times[1:] if len(mask_times) > 1 else mask_times
    new_avg = sum(warm_times) / len(warm_times)
    
    old_avg = old_sep1_baseline.get(sid, 0)
    llg_avg = llg_baseline.get(sid, 0)
    
    old_ms = old_avg * 1000
    new_ms = new_avg * 1000
    llg_ms = llg_avg * 1000
    
    old_ratio = f"{old_ms/llg_ms:.1f}x" if llg_ms > 0 else "N/A"
    new_ratio = f"{new_ms/llg_ms:.1f}x" if llg_ms > 0 else "N/A"
    speedup = f"{old_avg/new_avg:.1f}x" if new_avg > 0 else "N/A"
    
    print(f"{sid:<40s} {old_ms:>9.3f}ms {new_ms:>9.3f}ms {llg_ms:>9.3f}ms {old_ratio:>8s} {new_ratio:>8s} {speedup:>8s}")
