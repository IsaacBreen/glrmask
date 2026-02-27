#!/usr/bin/env python3
"""Profile sep1 mask generation step breakdown (seed vs worklist)."""

import json
import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path

# We need _sep1 from the .venv
import _sep1

CFA_ROOT = Path(os.path.expanduser("~/Projects2/constraint-framework-analysis"))
GRAMMARS_ROOT = Path(os.path.expanduser("~/Projects2/grammars2024"))
COMPILER = GRAMMARS_ROOT / "target" / "release" / "grammar-compiler"

def load_vocab():
    vocab_path = CFA_ROOT / ".cache" / "vocab_cache" / "gpt2_vocab.json"
    if vocab_path.exists():
        with open(vocab_path) as f:
            return json.load(f)
    # Fallback
    with open("/tmp/vocab.json") as f:
        return json.load(f)

def compile_schema(schema_path: Path, vocab: dict) -> str:
    """Compile a JSON schema and return the constraint JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as vf:
        json.dump(vocab, vf)
        vocab_path = Path(vf.name)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as of:
        output_path = Path(of.name)
    
    try:
        result = subprocess.run(
            [str(COMPILER), "--vocab", str(vocab_path), "--json-schema", str(schema_path), "--output", str(output_path)],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "SKIP_SERIALIZATION": "0"},
        )
        if result.returncode != 0:
            print(f"Compile failed: {result.stderr[:500]}")
            return None
        with open(output_path) as f:
            return f.read()
    finally:
        os.unlink(vocab_path)
        if output_path.exists():
            os.unlink(output_path)

def profile_schema(schema_id: str, vocab: dict):
    """Profile mask generation for a schema."""
    parts = schema_id.split("/")
    schema_path = CFA_ROOT / "data" / "sources" / "jsonschemabench" / "data" / parts[0] / parts[1] + ".json"
    if not schema_path.exists():
        # Try finding it
        import glob
        matches = glob.glob(str(CFA_ROOT / "data" / "sources" / "jsonschemabench" / "data" / "**" / (parts[-1] + ".json")), recursive=True)
        if matches:
            schema_path = Path(matches[0])
        else:
            print(f"Schema not found: {schema_id}")
            return
    
    print(f"\n{'='*60}")
    print(f"Schema: {schema_id} ({schema_path})")
    
    # Compile
    t0 = time.time()
    constraint_json = compile_schema(schema_path, vocab)
    compile_time = time.time() - t0
    if constraint_json is None:
        return
    print(f"Compile time: {compile_time:.3f}s")
    
    # Load model
    from python.aug25.models.rust_model import Model
    model = Model.from_json_string(constraint_json)
    
    # Enable benchmark mode
    _sep1.set_benchmark_mode(True)
    
    # Get initial mask and commit a valid token
    step_data = []
    for step in range(20):
        t0 = time.time()
        mask = model.get_mask()
        wall_time = time.time() - t0
        
        seed_ns = _sep1.get_last_mask_seed_time_ns()
        worklist_ns = _sep1.get_last_mask_worklist_time_ns()
        worklist_iters = _sep1.get_last_mask_worklist_iter_count()
        
        # Get allowed tokens
        ranges = list(mask.to_ranges())
        num_allowed = sum(end - start + 1 for start, end in ranges)
        
        step_data.append({
            "step": step,
            "wall_us": wall_time * 1e6,
            "seed_us": seed_ns / 1000,
            "worklist_us": worklist_ns / 1000,
            "other_us": (wall_time * 1e6) - (seed_ns + worklist_ns) / 1000,
            "worklist_iters": worklist_iters,
            "num_allowed": num_allowed,
        })
        
        if num_allowed == 0:
            print(f"  Step {step}: no allowed tokens, stopping")
            break
        
        # Commit the first allowed token
        first_token = ranges[0][0]
        model.commit(first_token)
    
    # Print results
    print(f"\n{'Step':>4s} {'Wall(μs)':>10s} {'Seed(μs)':>10s} {'WL(μs)':>10s} {'Other(μs)':>10s} {'WL_iters':>10s} {'Allowed':>10s}")
    for d in step_data:
        print(f"{d['step']:4d} {d['wall_us']:10.1f} {d['seed_us']:10.1f} {d['worklist_us']:10.1f} {d['other_us']:10.1f} {d['worklist_iters']:10d} {d['num_allowed']:10d}")
    
    if len(step_data) > 1:
        # Averages excluding first step
        avg_wall = sum(d['wall_us'] for d in step_data[1:]) / (len(step_data) - 1)
        avg_seed = sum(d['seed_us'] for d in step_data[1:]) / (len(step_data) - 1)
        avg_wl = sum(d['worklist_us'] for d in step_data[1:]) / (len(step_data) - 1)
        avg_other = sum(d['other_us'] for d in step_data[1:]) / (len(step_data) - 1)
        avg_iters = sum(d['worklist_iters'] for d in step_data[1:]) / (len(step_data) - 1)
        print(f"\nAverages (excl first step):")
        print(f"  Wall: {avg_wall:.1f}μs, Seed: {avg_seed:.1f}μs ({100*avg_seed/avg_wall:.1f}%), WL: {avg_wl:.1f}μs ({100*avg_wl/avg_wall:.1f}%), Other: {avg_other:.1f}μs ({100*avg_other/avg_wall:.1f}%)")
        print(f"  Avg WL iters: {avg_iters:.1f}")

def main():
    vocab = load_vocab()
    print(f"Vocab size: {len(vocab)}")
    
    schemas = [
        "Github_easy/o10008",
        "Github_hard/o69862",
        "Github_easy/o36272",  # worst case sep1 vs llg
        "Github_medium/o42988",  # 2nd worst
    ]
    
    for schema_id in schemas:
        try:
            profile_schema(schema_id, vocab)
        except Exception as e:
            print(f"Error profiling {schema_id}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
