#!/usr/bin/env python3
"""Two-pass oracle compaction experiment.

Pass 1: Compile normally with GLRMASK_ORACLE_DUMP to save post-compact mappings.
Pass 2: Compile with GLRMASK_ORACLE_LOAD to inject oracle mappings, skipping equiv analysis.

Reports timing deltas and correctness (whether DWA weights match).
"""

import json
import os
import re
import subprocess
import sys
import time

VOCAB_PATH = "/Users/isaacbreen/Projects2/grammars2024/benchmarking/gpt2_vocab.json"

SCHEMAS = {
    "o82370": "/Users/isaacbreen/Projects2/constraint-framework-analysis/data/sources/jsonschemabench/data/Github_medium/o82370.json",
    "kb_684": "/Users/isaacbreen/Projects2/constraint-framework-analysis/data/sources/jsonschemabench/data/Kubernetes/kb_684_Normalized.json",
}

CONFIGS = {
    "no_split": {
        "GLRMASK_NO_OPEN_QUOTE_SPLIT": "1",
        "GLRMASK_SPLIT_CLOSE_QUOTE": "",
    },
    "open_only": {
        "GLRMASK_NO_OPEN_QUOTE_SPLIT": "",
        "GLRMASK_SPLIT_CLOSE_QUOTE": "",
    },
    "close_only": {
        "GLRMASK_NO_OPEN_QUOTE_SPLIT": "1",
        "GLRMASK_SPLIT_CLOSE_QUOTE": "1",
    },
    "open_close": {
        "GLRMASK_NO_OPEN_QUOTE_SPLIT": "",
        "GLRMASK_SPLIT_CLOSE_QUOTE": "1",
    },
}


def make_env(config_env, extra_env=None):
    env = os.environ.copy()
    # Clear split flags
    for k in ["GLRMASK_NO_OPEN_QUOTE_SPLIT", "GLRMASK_SPLIT_CLOSE_QUOTE",
              "GLRMASK_SPLIT_KEY_COLON_SUFFIX", "GLRMASK_ORACLE_DUMP", "GLRMASK_ORACLE_LOAD"]:
        env.pop(k, None)
    for k, v in config_env.items():
        if v:
            env[k] = v
    env["GLRMASK_PROFILE_COMPILE_SUMMARY"] = "1"
    if extra_env:
        env.update(extra_env)
    return env


def run_compile(schema_path, vocab_path, env, log_path):
    """Run a compile and capture stderr to log_path. Returns (wall_time_s, stderr_text)."""
    code = f'''
import json, time, _glrmask as glrmask
with open("{vocab_path}") as f:
    vd = json.load(f)
vocab = glrmask.Vocab.from_dict({{k.encode(): v for k,v in vd.items()}})
with open("{schema_path}") as f:
    schema = f.read()
t0 = time.perf_counter()
c = glrmask.Constraint.from_json_schema(schema, vocab)
t1 = time.perf_counter()
print(f"WALL_TIME={{t1-t0:.6f}}")
# Dump mask fingerprint for correctness check
import hashlib
m = c.get_mask()
h = hashlib.sha256(bytes(m)).hexdigest()[:16]
print(f"MASK_HASH={{h}}")
n_allowed = sum(1 for x in m if x)
print(f"MASK_ALLOWED={{n_allowed}}")
'''
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env, capture_output=True, text=True, timeout=300
        )
        log_f.write(result.stderr)

    return result.stdout.strip(), result.stderr


def parse_compile_line(stderr):
    """Extract [glrmask/profile][compile] fields."""
    for line in stderr.splitlines():
        if "[glrmask/profile][compile]" in line:
            fields = {}
            for m in re.finditer(r'(\w+)=([\d.]+)', line):
                fields[m.group(1)] = float(m.group(2))
            return fields
    return {}


def parse_compact_line(stderr):
    """Extract [glrmask/profile][compact] fields."""
    for line in stderr.splitlines():
        if "[glrmask/profile][compact]" in line:
            m = re.search(r'tsids=(\d+)=>(\d+)\s+tokens=(\d+)=>(\d+)', line)
            if m:
                return {
                    "tsids_before": int(m.group(1)),
                    "tsids_after": int(m.group(2)),
                    "tokens_before": int(m.group(3)),
                    "tokens_after": int(m.group(4)),
                }
    return {}


def parse_oracle_line(stderr):
    for line in stderr.splitlines():
        if "[glrmask/oracle]" in line:
            return line.strip()
    return None


def main():
    results = {}
    
    # Pass 1: Baseline with oracle dump
    print("=" * 70)
    print("PASS 1: Baseline compile (with oracle dump)")
    print("=" * 70)
    
    for config_name, config_env in CONFIGS.items():
        for schema_name, schema_path in SCHEMAS.items():
            key = f"{config_name}_{schema_name}"
            oracle_path = f"/tmp/oracle_{key}.json"
            log_path = f"/tmp/oracle_pass1_{key}.log"
            
            env = make_env(config_env, {"GLRMASK_ORACLE_DUMP": oracle_path})
            
            print(f"\n  {key}...", end=" ", flush=True)
            stdout, stderr = run_compile(schema_path, VOCAB_PATH, env, log_path)
            
            compile_fields = parse_compile_line(stderr)
            compact_fields = parse_compact_line(stderr)
            oracle_line = parse_oracle_line(stderr)
            
            # Parse stdout
            wall_time = None
            mask_hash = None
            mask_allowed = None
            for line in stdout.splitlines():
                if line.startswith("WALL_TIME="):
                    wall_time = float(line.split("=")[1])
                elif line.startswith("MASK_HASH="):
                    mask_hash = line.split("=")[1]
                elif line.startswith("MASK_ALLOWED="):
                    mask_allowed = int(line.split("=")[1])
            
            results[key] = {
                "pass1": {
                    "wall_time": wall_time,
                    "mask_hash": mask_hash,
                    "mask_allowed": mask_allowed,
                    "compile": compile_fields,
                    "compact": compact_fields,
                    "oracle_line": oracle_line,
                },
            }
            
            oracle_size = os.path.getsize(oracle_path) if os.path.exists(oracle_path) else 0
            print(f"done (wall={wall_time:.3f}s, id_map={compile_fields.get('id_map_ms',0):.1f}ms, oracle={oracle_size}B)")
    
    # Pass 2: Oracle load
    print("\n" + "=" * 70)
    print("PASS 2: Oracle compile (skip equiv analysis)")
    print("=" * 70)
    
    for config_name, config_env in CONFIGS.items():
        for schema_name, schema_path in SCHEMAS.items():
            key = f"{config_name}_{schema_name}"
            oracle_path = f"/tmp/oracle_{key}.json"
            log_path = f"/tmp/oracle_pass2_{key}.log"
            
            if not os.path.exists(oracle_path):
                print(f"\n  {key}: SKIP (no oracle file)")
                continue
            
            env = make_env(config_env, {"GLRMASK_ORACLE_LOAD": oracle_path})
            
            print(f"\n  {key}...", end=" ", flush=True)
            stdout, stderr = run_compile(schema_path, VOCAB_PATH, env, log_path)
            
            compile_fields = parse_compile_line(stderr)
            compact_fields = parse_compact_line(stderr)
            oracle_line = parse_oracle_line(stderr)
            
            wall_time = None
            mask_hash = None
            mask_allowed = None
            for line in stdout.splitlines():
                if line.startswith("WALL_TIME="):
                    wall_time = float(line.split("=")[1])
                elif line.startswith("MASK_HASH="):
                    mask_hash = line.split("=")[1]
                elif line.startswith("MASK_ALLOWED="):
                    mask_allowed = int(line.split("=")[1])
            
            results[key]["pass2"] = {
                "wall_time": wall_time,
                "mask_hash": mask_hash,
                "mask_allowed": mask_allowed,
                "compile": compile_fields,
                "compact": compact_fields,
                "oracle_line": oracle_line,
            }
            
            match = results[key]["pass1"]["mask_hash"] == mask_hash
            print(f"done (wall={wall_time:.3f}s, id_map={compile_fields.get('id_map_ms',0):.1f}ms, mask_match={match})")
    
    # Summary report
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Config':<25} {'Pass1 id_map':>12} {'Pass2 id_map':>12} {'Delta':>8} {'Pass1 DWA':>10} {'Pass2 DWA':>10} {'Pass1 compact':>13} {'Pass2 compact':>13} {'Mask OK':>8}")
    print("-" * 120)
    
    for config_name in CONFIGS:
        for schema_name in SCHEMAS:
            key = f"{config_name}_{schema_name}"
            r = results.get(key, {})
            p1 = r.get("pass1", {})
            p2 = r.get("pass2", {})
            
            p1_idmap = p1.get("compile", {}).get("id_map_ms", 0)
            p2_idmap = p2.get("compile", {}).get("id_map_ms", 0)
            delta = p2_idmap - p1_idmap
            
            p1_dwa = p1.get("compile", {}).get("terminal_dwa_ms", 0)
            p2_dwa = p2.get("compile", {}).get("terminal_dwa_ms", 0)
            
            p1_compact = p1.get("compile", {}).get("compact_ms", 0)
            p2_compact = p2.get("compile", {}).get("compact_ms", 0)
            
            mask_ok = "YES" if p1.get("mask_hash") == p2.get("mask_hash") else "NO"
            
            print(f"{key:<25} {p1_idmap:>10.1f}ms {p2_idmap:>10.1f}ms {delta:>+7.1f}ms {p1_dwa:>8.1f}ms {p2_dwa:>8.1f}ms {p1_compact:>11.1f}ms {p2_compact:>11.1f}ms {mask_ok:>8}")
    
    # Detailed per-stage comparison
    print(f"\n{'Config':<25} {'Stage':<25} {'Pass1 (ms)':>12} {'Pass2 (ms)':>12} {'Delta':>10}")
    print("-" * 90)
    
    stages = ["id_map_ms", "terminal_dwa_ms", "compact_ms", "compile_ms", "total_ms"]
    for config_name in CONFIGS:
        for schema_name in SCHEMAS:
            key = f"{config_name}_{schema_name}"
            r = results.get(key, {})
            p1c = r.get("pass1", {}).get("compile", {})
            p2c = r.get("pass2", {}).get("compile", {})
            for stage in stages:
                v1 = p1c.get(stage, 0)
                v2 = p2c.get(stage, 0)
                print(f"{key:<25} {stage:<25} {v1:>10.1f}ms {v2:>10.1f}ms {v2-v1:>+9.1f}ms")
            print()
    
    # Compact dimension changes
    print(f"\n{'Config':<25} {'Pass1 tsids':>15} {'Pass2 tsids':>15} {'Pass1 tokens':>15} {'Pass2 tokens':>15}")
    print("-" * 90)
    for config_name in CONFIGS:
        for schema_name in SCHEMAS:
            key = f"{config_name}_{schema_name}"
            r = results.get(key, {})
            p1c = r.get("pass1", {}).get("compact", {})
            p2c = r.get("pass2", {}).get("compact", {})
            p1_tsids = f"{p1c.get('tsids_before', '?')}=>{p1c.get('tsids_after', '?')}"
            p2_tsids = f"{p2c.get('tsids_before', '?')}=>{p2c.get('tsids_after', '?')}"
            p1_tokens = f"{p1c.get('tokens_before', '?')}=>{p1c.get('tokens_after', '?')}"
            p2_tokens = f"{p2c.get('tokens_before', '?')}=>{p2c.get('tokens_after', '?')}"
            print(f"{key:<25} {p1_tsids:>15} {p2_tsids:>15} {p1_tokens:>15} {p2_tokens:>15}")
    
    # Dump full results as JSON
    with open("/tmp/oracle_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results: /tmp/oracle_experiment_results.json")


if __name__ == "__main__":
    main()
