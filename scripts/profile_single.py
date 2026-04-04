#!/usr/bin/env python3
"""Profile a single schema with GLRMASK_PROFILE_COMPILE=1 for full phase breakdown."""
import json, os, sys, time
from pathlib import Path

schema_name = sys.argv[1] if len(sys.argv) > 1 else "Github_medium---o82370"

os.environ["GLRMASK_PROFILE_COMPILE"] = "1"
os.environ["GLRMASK_PROFILE_COMPILE_SUMMARY"] = "1"
os.environ["GLRMASK_PROFILE_PARSER_DWA"] = "1"
os.environ["RAYON_NUM_THREADS"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "constraint-framework-analysis"))

import _glrmask as gm
from cfa.tokenization import load_vocab_info

CFA_DIR = Path(__file__).resolve().parents[1].parent / "constraint-framework-analysis"
MASKBENCH_DIR = CFA_DIR / "data" / "sources" / "jsonschemabench" / "maskbench" / "data"
vi = load_vocab_info(cache_dir=CFA_DIR / ".cache" / "vocab_cache")
tok_bytes = {v: k for k, v in vi.id_to_token_bytes.items()}
vocab = gm.Vocab.from_dict(tok_bytes)

def load_schema(name):
    path = MASKBENCH_DIR / f"{name}.json"
    payload = json.load(open(path))
    if isinstance(payload, dict) and "schema" in payload:
        return json.dumps(payload["schema"])
    return json.dumps(payload)

# Warmup with boolean
print("=== Warmup ===", flush=True)
_ = gm.Constraint.from_json_schema('{"type":"boolean"}', vocab)
print("Warmup done\n", flush=True)

schema_json = load_schema(schema_name)
print(f"=== Profiling: {schema_name} ===", flush=True)
t0 = time.perf_counter()
c = gm.Constraint.from_json_schema(schema_json, vocab)
elapsed = time.perf_counter() - t0
print(f"\n=== Total Python wall time: {elapsed*1000:.0f}ms ===", flush=True)
