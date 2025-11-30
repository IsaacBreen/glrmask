#!/usr/bin/env python3
"""
Test Sep1 JSON Schema compilation and constraint testing.

Usage:
    # With default hard schema
    python scripts/test_json_schema.py
    
    # With custom schema
    SCHEMA_FILE="gcg-paper/downloads/repos/jsonschemabench/data/Github_easy/o10008.json" python scripts/test_json_schema.py
"""

import json
import time
import os
import _sep1

# Load schema
schema_file = os.environ.get(
    "SCHEMA_FILE", 
    "gcg-paper/downloads/repos/jsonschemabench/data/Github_hard/o69862.json"
)
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

# Step 3: Compile to GLR parser
print("\n3. Compiling grammar...")
start = time.time()
compiled = grammar_def.compile()
compile_time = time.time() - start
print(f"   Compilation: {compile_time*1000:.1f}ms")

# Step 4: Create constraint with vocabulary
print("\n4. Creating constraint with vocabulary...")
start = time.time()
constraint = _sep1.GrammarConstraint(compiled, token_to_id)
constraint_time = time.time() - start
print(f"   Constraint creation: {constraint_time*1000:.1f}ms")

# Step 5: Test stepping through a valid input
print("\n5. Testing constraint on valid input...")
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
total_time = ebnf_time + parse_time + compile_time + constraint_time
print(f"\n=== Total compile time: {total_time*1000:.1f}ms ===")
