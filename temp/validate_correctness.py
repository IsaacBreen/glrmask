#!/usr/bin/env python3
"""
Quick correctness validation script.
Compares the constraint masks from rust_model.py against a brute-force reference.
"""

import sys
import json
import gzip
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Disable debug output before importing ffi
import os
os.environ["MACRO_DEBUG_LEVEL"] = "0"

import _sep1 as ffi


def validate_mask(constraint_path: str, code_path: str, max_steps: int = 50):
    """
    Validate that the optimized get_mask matches brute-force verification.
    """
    from transformers import GPT2Tokenizer
    
    print(f"Loading constraint from: {constraint_path}")
    print(f"Loading code from: {code_path}")
    
    # Load constraint
    if constraint_path.endswith('.gz'):
        with gzip.open(constraint_path, 'rt') as f:
            constraint_json = f.read()
    else:
        with open(constraint_path) as f:
            constraint_json = f.read()
    
    constraint = ffi.GrammarConstraint.from_json_string(constraint_json)
    state = ffi.GrammarConstraintState(constraint)
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Create id -> bytes mapping
    id_to_bytes = {}
    for token_id in range(tokenizer.vocab_size):
        try:
            token_str = tokenizer.decode([token_id])
            id_to_bytes[token_id] = token_str.encode('utf-8')
        except:
            pass
    
    # Tokenize code
    with open(code_path, 'r') as f:
        code_text = f.read()
    
    tokens = tokenizer.encode(code_text)
    print(f"Tokenized into {len(tokens)} tokens")
    
    # Validate each step
    steps_to_check = min(max_steps, len(tokens))
    print(f"\nValidating first {steps_to_check} steps...")
    
    mismatches = 0
    for i, token_id in enumerate(tokens[:steps_to_check]):
        # Get optimized mask
        mask_bv = state.get_mask_bv()
        optimized_valid = mask_bv.contains(token_id)
        
        # Check with brute force: try committing and see if valid
        temp_state = state.clone()
        token_bytes = id_to_bytes.get(token_id, b'')
        if token_bytes:
            temp_state.commit_bytes(token_bytes)
            bruteforce_valid = temp_state.is_valid()
        else:
            bruteforce_valid = False
        
        if optimized_valid != bruteforce_valid:
            print(f"Step {i}: token={token_id} ({repr(token_bytes[:20])}) "
                  f"optimized={optimized_valid} bruteforce={bruteforce_valid} ✗ MISMATCH!")
            mismatches += 1
        else:
            token_preview = repr(id_to_bytes.get(token_id, b'')[:10])
            print(f"Step {i}: token={token_id} {token_preview} optimized={optimized_valid} ✓")
        
        # Commit the token for next step
        if token_bytes:
            state.commit_bytes(token_bytes)
    
    print(f"\n{'='*50}")
    if mismatches == 0:
        print(f"✓ All {steps_to_check} steps validated successfully!")
    else:
        print(f"✗ Found {mismatches} mismatches out of {steps_to_check} steps")
    
    return mismatches == 0


if __name__ == "__main__":
    constraint_path = ".cache/test_vocabs/constraint_js.json.gz"
    code_path = "src/example_code10.js"
    
    if len(sys.argv) > 1:
        constraint_path = sys.argv[1]
    if len(sys.argv) > 2:
        code_path = sys.argv[2]
    
    success = validate_mask(constraint_path, code_path)
    sys.exit(0 if success else 1)
