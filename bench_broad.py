#!/usr/bin/env python3
"""Broader benchmark: sample 100 schemas, compare sep1 vs llguidance."""
import json, os, sys, time, subprocess, tempfile, random
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
schema_ids_in_sweep = []
for r in sweep["results"]:
    mt = r.get("mask_times_avg", {})
    sid = r.get("schema_id", "")
    if "llguidance" in mt and mt["llguidance"] > 0 and "sep1" in mt and mt["sep1"] > 0:
        llg_baseline[sid] = mt["llguidance"]
        old_sep1_baseline[sid] = mt["sep1"]
        schema_ids_in_sweep.append(sid)

# Sample: top 20 worst ratio + 80 random
sorted_by_ratio = sorted(schema_ids_in_sweep, key=lambda s: old_sep1_baseline[s]/llg_baseline[s], reverse=True)
top_worst = sorted_by_ratio[:20]
remaining = [s for s in schema_ids_in_sweep if s not in set(top_worst)]
random.seed(42)
random_sample = random.sample(remaining, min(80, len(remaining)))
sample = top_worst + random_sample

print(f"Benchmarking {len(sample)} schemas ({len(top_worst)} worst + {len(random_sample)} random)")
print(f"LLG baseline from sweep with {len(schema_ids_in_sweep)} schemas")
print()

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
    except subprocess.TimeoutExpired:
        return None
    finally:
        if output_path.exists():
            os.unlink(output_path)

N_STEPS = 50

def benchmark_schema(schema_id):
    cj = compile_schema(schema_id)
    if cj is None:
        return None
    try:
        model = RustModel.from_json_string(cj)
    except Exception:
        return None
    _sep1.set_benchmark_mode(True)
    
    mask_times = []
    for step in range(N_STEPS):
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

results = []
fail_count = 0
for i, sid in enumerate(sample):
    try:
        mask_times = benchmark_schema(sid)
    except Exception as e:
        print(f"  Error for {sid}: {e}", file=sys.stderr)
        fail_count += 1
        continue
    if mask_times is None:
        fail_count += 1
        continue
    
    warm_times = mask_times[2:] if len(mask_times) > 2 else mask_times
    if not warm_times:
        continue
    new_avg = sum(warm_times) / len(warm_times)
    llg_avg = llg_baseline.get(sid, 0)
    old_avg = old_sep1_baseline.get(sid, 0)
    
    if llg_avg > 0:
        new_ratio = new_avg / llg_avg
    else:
        new_ratio = float('inf')
    
    results.append({
        'schema_id': sid,
        'new_sep1_ms': new_avg * 1000,
        'old_sep1_ms': old_avg * 1000,
        'llg_ms': llg_avg * 1000,
        'new_ratio': new_ratio,
        'old_ratio': old_avg / llg_avg if llg_avg > 0 else float('inf'),
        'speedup': old_avg / new_avg if new_avg > 0 else float('inf'),
        'n_steps': len(mask_times),
    })
    
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{len(sample)}", file=sys.stderr)

# Sort by new ratio (worst first)
results.sort(key=lambda r: r['new_ratio'], reverse=True)

print(f"\nResults ({len(results)} successful, {fail_count} failed):")
print(f"\n{'Schema':<40s} {'New Sep1':>10s} {'LLG':>10s} {'New/LLG':>8s} {'Old/LLG':>8s} {'Speedup':>8s}")
print("-" * 86)

for r in results[:30]:  # Top 30 worst
    print(f"{r['schema_id']:<40s} {r['new_sep1_ms']:>9.3f}ms {r['llg_ms']:>9.3f}ms {r['new_ratio']:>7.1f}x {r['old_ratio']:>7.1f}x {r['speedup']:>7.1f}x")

print(f"\n--- Distribution stats ---")
ratios = [r['new_ratio'] for r in results if r['new_ratio'] < float('inf')]
old_ratios = [r['old_ratio'] for r in results if r['old_ratio'] < float('inf')]
speedups = [r['speedup'] for r in results if r['speedup'] < float('inf')]

ratios.sort()
old_ratios.sort()

n = len(ratios)
print(f"  N schemas: {n}")
print(f"  New Sep1/LLG: median={ratios[n//2]:.2f}x  P90={ratios[int(n*0.9)]:.2f}x  P95={ratios[int(n*0.95)]:.2f}x  P99={ratios[int(n*0.99)]:.2f}x  max={ratios[-1]:.2f}x")
print(f"  Old Sep1/LLG: median={old_ratios[n//2]:.2f}x  P90={old_ratios[int(n*0.9)]:.2f}x  P95={old_ratios[int(n*0.95)]:.2f}x  max={old_ratios[-1]:.2f}x")
print(f"  Sep1 faster than LLG (new): {sum(1 for r in ratios if r < 1.0)}/{n} ({sum(1 for r in ratios if r < 1.0)*100/n:.1f}%)")
print(f"  Sep1 within 1.5x of LLG:   {sum(1 for r in ratios if r < 1.5)}/{n} ({sum(1 for r in ratios if r < 1.5)*100/n:.1f}%)")
print(f"  Sep1 within 2.0x of LLG:   {sum(1 for r in ratios if r < 2.0)}/{n} ({sum(1 for r in ratios if r < 2.0)*100/n:.1f}%)")
print(f"  Speedup from caching: median={sorted(speedups)[len(speedups)//2]:.2f}x  max={max(speedups):.1f}x")

# Save full results
with open('/tmp/bench_broad_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nFull results saved to /tmp/bench_broad_results.json")
