#!/usr/bin/env python3
"""
Test Sep1 JSON Schema compilation and constraint testing.

Usage:
    # With default hard schema
    python scripts/test_json_schema.py
    
    # With custom schema file
    SCHEMA_FILE="gcg-paper/downloads/repos/jsonschemabench/data/Github_easy/o10008.json" python scripts/test_json_schema.py
    
    # With schema ID (searches benchmark data directories)
    SCHEMA_ID="Snowplow---sp_136_Normalized" python scripts/test_json_schema.py
"""

import json
import time
import os
import glob
import _sep1

def find_schema_by_id(schema_id: str) -> str:
    """Find a schema file by its ID in benchmark data directories."""
    search_dirs = [
        "gcg-paper/downloads/repos/jsonschemabench/data",
        "gcg-paper/downloads/repos/jsonschemabench/maskbench/data",
        "gcg-paper/hard_schemas/data",
        "gcg-paper/json_schema_test_suite/data",
    ]
    
    for base_dir in search_dirs:
        # Try exact match first
        pattern = f"{base_dir}/**/{schema_id}.json"
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
        
        # Also try with schema_id as the full filename (category---name format)
        pattern = f"{base_dir}/{schema_id}.json"
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    raise FileNotFoundError(f"Schema ID '{schema_id}' not found in benchmark data directories")

# Load schema
schema_id = os.environ.get("SCHEMA_ID")
schema_file = os.environ.get("SCHEMA_FILE")

if schema_id:
    schema_file = find_schema_by_id(schema_id)
elif not schema_file:
    schema_file = "gcg-paper/downloads/repos/jsonschemabench/data/Github_hard/o69862.json"

print(f"Loading schema from: {schema_file}")
with open(schema_file) as f:
    schema = json.load(f)


# Load vocabulary (GPT-2)
print("Loading vocabulary...")
import urllib.request
vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"
with urllib.request.urlopen(vocab_url) as resp:
    vocab = json.loads(resp.read().decode())

# Convert to token_bytes -> id format
token_to_id = {k.encode('utf-8'): v for k, v in vocab.items()}

# Step 1: Convert JSON schema to EBNF
print("\n1. Converting JSON schema to EBNF...")
start = time.time()
ebnf = _sep1.json_schema_to_ebnf_py(json.dumps(schema))
ebnf_time = time.time() - start
print(f"   EBNF conversion: {ebnf_time*1000:.1f}ms ({len(ebnf)} chars)")

# Step 2: Parse EBNF to GrammarDefinition
print("\n2. Parsing EBNF to GrammarDefinition...")
start = time.time()
grammar_def = _sep1.grammar_definition_from_json_schema(json.dumps(schema))
parse_time = time.time() - start
print(f"   Grammar parsing: {parse_time*1000:.1f}ms")

# Step 3: Optimize grammar
print("\n3. Optimizing grammar...")
start = time.time()
grammar_def.optimize()
optimize_time = time.time() - start
print(f"   Optimization: {optimize_time*1000:.1f}ms")

# Step 4: Compile to GLR parser
print("\n4. Compiling grammar...")
start = time.time()
compiled = grammar_def.compile()
compile_time = time.time() - start

# Step 5: Create constraint with vocabulary
print("\n5. Creating constraint with vocabulary...")
start = time.time()
constraint = _sep1.GrammarConstraint(compiled, token_to_id)
constraint_time = time.time() - start
print(f"   Constraint creation: {constraint_time*1000:.1f}ms")

# Step 6: Test stepping through a valid input
print("\n6. Testing constraint on valid input...")
state = _sep1.GrammarConstraintState(constraint)

# A minimal valid JSON that should match most schemas
test_input = "{}"
print(f"   Input: {test_input}")

for char in test_input:
    char_bytes = char.encode('utf-8')
    # Find token ID for this character
    if char_bytes in token_to_id:
        token_id = token_to_id[char_bytes]
        mask = state.get_mask()
        if mask[token_id]:
            state.commit(token_id)
            print(f"   Accepted: '{char}' (token {token_id})")
        else:
            print(f"   Rejected: '{char}' (token {token_id}) - not in mask")
            break

print(f"\n   Final state active: {state.is_active()}")
print(f"   Final state valid: {state.is_valid()}")

# Summary
total_time = ebnf_time + parse_time + optimize_time + compile_time + constraint_time
print(f"\n=== Total compile time: {total_time*1000:.1f}ms ===")
