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
import sys
import glob

# Add project root and python/ dir to sys.path to find _sep1 module
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "python"))

import _sep1

def find_schema_by_id(schema_id: str) -> str:
    """Find a schema file by its ID in benchmark data directories."""
    search_dirs = [
        "gcg-paper/downloads/repos/jsonschemabench/data",
        "gcg-paper/downloads/repos/jsonschemabench/maskbench/data",
        "gcg-paper/hard_schemas/data",
        "gcg-paper/hard_schemas/schemas",
        "gcg-paper/hard_schemas/tests",
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
    data = json.load(f)

if 'schema' in data:
    schema = data['schema']
else:
    schema = data

if 'title' in schema:
    print(f"Schema title: {schema['title']}")

# If PRINT_JSON_SCHEMA is set, output the schema JSON and exit
if os.environ.get("PRINT_JSON_SCHEMA"):
    print(json.dumps(schema, indent=2))
    sys.exit(0)


# Load vocabulary
try:
    import tiktoken
    print("Loading vocabulary using tiktoken (matching benchmark runner)...")
    enc = tiktoken.get_encoding("gpt2")
    
    print("\nGeneration token_to_id map (matching Sep1Adapter._get_token_to_id)...")
    start = time.time()
    token_to_id = {}
    for token_id in range(enc.n_vocab):
        token_bytes = enc.decode_single_token_bytes(token_id)
        token_to_id[token_bytes] = token_id
    vocab_time = time.time() - start
    print(f"   Token map generation: {vocab_time*1000:.1f}ms")

except ImportError:
    print("tiktoken not found, falling back to downloading vocab.json...")
    import urllib.request
    vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"
    start = time.time()
    with urllib.request.urlopen(vocab_url) as resp:
        vocab = json.loads(resp.read().decode())
    token_to_id = {k.encode('utf-8'): v for k, v in vocab.items()}
    vocab_time = time.time() - start
    print(f"   Vocab download/parse: {vocab_time*1000:.1f}ms")

# Step 1: Convert JSON schema to EBNF
print("\n1. Converting JSON schema to EBNF...", file=sys.stderr)
start = time.time()
ebnf = _sep1.json_schema_to_ebnf_py(json.dumps(schema))
ebnf_time = time.time() - start
print(f"   EBNF conversion: {ebnf_time*1000:.1f}ms ({len(ebnf)} chars)", file=sys.stderr)

# If PRINT_RAW_EBNF is set, output the raw EBNF and exit
if os.environ.get("PRINT_RAW_EBNF"):
    print(ebnf)
    sys.exit(0)

# If PRINT_EBNF is set, output the parsed EBNF and exit
if os.environ.get("PRINT_EBNF"):
    print("\n2. Parsing EBNF to GrammarDefinition...", file=sys.stderr)
    start = time.time()
    # Parse into GrammarDefinition (optimization is disabled by default)
    grammar_def = _sep1.grammar_definition_from_ebnf(ebnf)
    
    # Output the parsed EBNF
    parsed_ebnf = grammar_def.to_ebnf()
    
    ebnf_time = time.time() - start
    print(f"   Parsing: {ebnf_time*1000:.1f}ms", file=sys.stderr)

    out_file = os.environ.get("OUT_FILE")
    if out_file:
        with open(out_file, 'w') as f:
            f.write(parsed_ebnf)
        print(f"Parsed EBNF written to: {out_file}", file=sys.stderr)
    else:
        print(parsed_ebnf)
    sys.exit(0)

# Step 2: Parse EBNF to GrammarDefinition
# Note: from_ebnf() does NOT optimize by default
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
print(f"   Grammar compilation: {compile_time*1000:.1f}ms")

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
total_time = vocab_time + ebnf_time + parse_time + compile_time + constraint_time
print(f"\n=== Total compile time: {total_time*1000:.1f}ms ===")
