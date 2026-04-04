#!/usr/bin/env python3
"""Profile top-10 slowest schemas to identify the 3 slowest."""
import json, os, sys, time
from pathlib import Path

os.environ["GLRMASK_PROFILE_COMPILE_SUMMARY"] = "1"
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

# Problem ID -> maskbench filename mapping
# jsb/data/Github_medium---o82370 -> Github_medium---o82370.json
TOP10 = [
    "Github_medium---o82370",
    "Github_hard---o56012",
    "Kubernetes---kb_684_Normalized",
    "Kubernetes---kb_815_Normalized",
    "Github_medium---o53116",
    "Kubernetes---kb_273_Normalized",
    "Github_hard---o82974",
    "Snowplow---sp_367_Normalized",
    "Kubernetes---kb_143_K5r",
    "Kubernetes---kb_143_A5",
]

def load_schema(name):
    path = MASKBENCH_DIR / f"{name}.json"
    payload = json.load(open(path))
    if isinstance(payload, dict) and "schema" in payload:
        return json.dumps(payload["schema"])
    return json.dumps(payload)

# Warmup
print("=== Warmup ===", flush=True)
schema_json = load_schema(TOP10[0])
_ = gm.Constraint.from_json_schema(schema_json, vocab)
print("Warmup done\n", flush=True)

results = []
for name in TOP10:
    schema_json = load_schema(name)
    
    print(f"=== {name} ===", flush=True)
    t0 = time.perf_counter()
    c = gm.Constraint.from_json_schema(schema_json, vocab)
    elapsed = time.perf_counter() - t0
    results.append((name, elapsed))
    print(f"  Python wall time: {elapsed*1000:.0f}ms\n", flush=True)

print("\n=== RANKING ===")
results.sort(key=lambda x: x[1], reverse=True)
for i, (name, t) in enumerate(results):
    print(f"  {i+1}. {name}: {t*1000:.0f}ms")
