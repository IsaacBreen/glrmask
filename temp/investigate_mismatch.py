#!/usr/bin/env python3
import sys, os
os.environ['MACRO_DEBUG_LEVEL'] = '0'
import gzip
import _sep1 as ffi
from transformers import GPT2Tokenizer

with gzip.open('.cache/test_vocabs/constraint_js.json.gz', 'rt') as f:
    constraint_json = f.read()

constraint = ffi.GrammarConstraint.from_json_string(constraint_json)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

state = ffi.GrammarConstraintState(constraint)
mask = state.get_mask_bv()

# Investigate specific mismatch: token 35713 ' ..."'
token_id = 35713
token_str = tokenizer.decode([token_id])
print(f'Token {token_id}: {repr(token_str)}')
token_bytes = token_str.encode('utf-8')
print(f'Bytes: {token_bytes!r}')

# Check mask
print(f'In mask: {mask.contains(token_id)}')

# Check brute force
temp = state.clone()
temp.commit_bytes(token_bytes)
print(f'is_valid after commit: {temp.is_valid()}')

# Try committing bytes individually
print("\nCommitting bytes individually:")
state2 = ffi.GrammarConstraintState(constraint)
for b in token_bytes:
    byte_char = chr(b) if 32 <= b < 127 else '?'
    print(f'Committing byte {b!r} ({byte_char})...')
    state2.commit_bytes(bytes([b]))
    print(f'  is_valid: {state2.is_valid()}')
