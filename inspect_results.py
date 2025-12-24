import gzip
import json
import os
from pathlib import Path

RESULTS_DIR = Path("gcg-paper/benchmarks/results")

def inspect_file(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"File not found: {filename}")
        return

    print(f"\n--- {filename} ---")
    with gzip.open(path, "rt") as f:
        for line in f:
            data = json.loads(line)
            print("Schema:", data.get("schema_id"))
            print("System:", path.stem.split('_')[0])
            print("Success:", data.get("success"))
            print("Error:", data.get("error"))
            
            # Correctness
            token_validity = data.get("token_was_valid", [])
            valid_count = sum(token_validity)
            total_count = len(token_validity)
            
            print(f"Token Validity: {valid_count}/{total_count}")
            # print("Sample validity sequence:", token_validity[:20])

inspect_file("sep1_PackageJson.jsonl.gz")
inspect_file("xgrammar_PackageJson.jsonl.gz")
inspect_file("llguidance_PackageJson.jsonl.gz")
