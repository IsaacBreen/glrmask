#!/usr/bin/env python3
"""Analyze weight structure and range counts for slow schemas."""
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
            return json.load(f)
    except Exception:
        return None
    finally:
        if output_path.exists():
            os.unlink(output_path)

SCHEMAS = [
    "Github_hard/o9767",      # 9.0x
    "Github_hard/o82811",     # 3.5x
    "Github_easy/o30452",     # 1.9x
    "Github_easy/o10008",     # fast
]

for sid in SCHEMAS:
    data = compile_schema(sid)
    if data is None:
        print(f"FAIL: {sid}")
        continue
    
    dwa = data.get("dwa", {})
    states = dwa.get("states", [])
    weight_pool = dwa.get("weight_pool", [])
    num_tsids = data.get("num_tsids", 0)
    
    print(f"\n{'='*60}")
    print(f"Schema: {sid}")
    print(f"  DWA states: {len(states)}")
    print(f"  Weight pool entries: {len(weight_pool)}")
    print(f"  num_tsids: {num_tsids}")
    
    # Analyze weight pool
    for i, w in enumerate(weight_pool[:10]):
        w_type = w.get("type", "unknown")
        if w_type == "RangeMap":
            entries = w.get("entries", [])
            total_tokens = 0
            total_tsid_ranges = 0
            for entry in entries:
                token_range = entry.get("key_range", [0, 0])
                tsid_ranges = entry.get("value", {}).get("ranges", [])
                span = token_range[1] - token_range[0] + 1
                total_tokens += span
                total_tsid_ranges += len(tsid_ranges)
            print(f"  Weight[{i}]: RangeMap, {len(entries)} entries, ~{total_tokens} tokens, ~{total_tsid_ranges} tsid_ranges")
        elif w_type == "RangeSet":
            ranges = w.get("ranges", [])
            print(f"  Weight[{i}]: RangeSet, {len(ranges)} ranges")
        elif w_type == "Factorized":
            pairs = w.get("pairs", [])
            print(f"  Weight[{i}]: Factorized, {len(pairs)} pairs")
        else:
            print(f"  Weight[{i}]: {w_type}")
