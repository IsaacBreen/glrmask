#!/usr/bin/env python3
"""Quick llguidance benchmark with JSON schema."""

import llguidance as llg
import llguidance.hf
from llguidance.numpy import fill_next_token_bitmask, allocate_token_bitmask
from transformers import AutoTokenizer
import json
import time
import statistics

# Use built-in JSON schema instead of Lark grammar
schema = {"type": "object"}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
llg_tokenizer = llguidance.hf.from_tokenizer(tokenizer)
vocab_size = llg_tokenizer.vocab_size

# Compile with JSON schema
grammars = json.dumps({"grammars": [{"json_schema": schema}]})
interp = llg.LLMatcher(llg_tokenizer, grammars)

if interp.is_error():
    print(f"Error: {interp.get_error()}")
else:
    mask_data = allocate_token_bitmask(1, vocab_size)
    
    # Warm up
    for _ in range(10):
        fill_next_token_bitmask(interp, mask_data, 0)
    
    # Count valid tokens
    fill_next_token_bitmask(interp, mask_data, 0)
    valid_count = 0
    for i in range(vocab_size):
        word_idx = i // 32
        bit_idx = i % 32
        if (mask_data[0, word_idx] & (1 << bit_idx)) != 0:
            valid_count += 1
    print(f"Valid tokens at start: {valid_count}")
    
    # Benchmark
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        fill_next_token_bitmask(interp, mask_data, 0)
        end = time.perf_counter()
        times.append((end - start) * 1e6)
    
    print(f"Mean: {statistics.mean(times):.1f} us")
    print(f"Median: {statistics.median(times):.1f} us")
    print(f"Min: {min(times):.1f} us")
