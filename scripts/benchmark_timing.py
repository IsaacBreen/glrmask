#!/usr/bin/env python3
"""Benchmark the Rust implementation for paper figures."""

import _sep1 as ffi
import gzip
import time
import sys

def main():
    # Load constraint JSON
    print("Loading constraint file...")
    with gzip.open('.cache/test_vocabs/constraint_js.json.gz', 'rt') as f:
        json_str = f.read()

    # Time constraint loading
    start = time.perf_counter()
    constraint = ffi.GrammarConstraint.from_json_string(json_str)
    load_time = time.perf_counter() - start
    print(f'Constraint load time: {load_time*1000:.2f} ms')

    # Create initial state
    start = time.perf_counter()
    state = ffi.GrammarConstraintState(constraint)
    state_time = time.perf_counter() - start
    print(f'Initial state creation: {state_time*1000:.4f} ms')

    # Warm up
    for _ in range(100):
        mask_bv = state.get_mask_bv()

    # Benchmark get_mask
    N = 10000
    start = time.perf_counter()
    for _ in range(N):
        mask_bv = state.get_mask_bv()
    end = time.perf_counter()

    per_call_us = (end - start) / N * 1e6
    print(f'')
    print(f'=== get_mask performance ({N} iterations) ===')
    print(f'  Per-call: {per_call_us:.2f} us')
    print(f'  Per-call: {per_call_us/1000:.4f} ms')
    print(f'  Throughput: {1e6/per_call_us:.0f} calls/sec')

    # Get the mask and find valid tokens for commit
    mask_bv = state.get_mask_bv()
    valid_tokens = list(mask_bv)[:20]  # First 20 valid tokens
    print(f'')
    print(f'Sample valid tokens: {valid_tokens[:10]}')
    
    if len(valid_tokens) > 0:
        # Test commit with a valid token
        token_id = valid_tokens[0]
        
        N = 5000
        times = []
        for _ in range(N):
            state = ffi.GrammarConstraintState(constraint)
            try:
                start = time.perf_counter()
                state.commit(token_id)
                times.append(time.perf_counter() - start)
            except Exception as e:
                print(f'  Commit failed for token {token_id}: {e}')
                break

        if times:
            avg_commit = sum(times) / len(times) * 1e6
            print(f'')
            print(f'=== commit performance ({len(times)} iterations) ===')
            print(f'  Per-call: {avg_commit:.2f} us')

    # Test multiple commits in sequence (simulating generation)
    print(f'')
    print(f'=== Multi-step generation simulation ===')
    
    for num_steps in [5, 10, 20]:
        N = 500
        times = []
        for _ in range(N):
            state = ffi.GrammarConstraintState(constraint)
            step_start = time.perf_counter()
            success = True
            for step in range(num_steps):
                mask_bv = state.get_mask_bv()
                valid = list(mask_bv)[:5]
                if not valid:
                    success = False
                    break
                try:
                    state.commit(valid[0])
                except:
                    success = False
                    break
            if success:
                times.append(time.perf_counter() - step_start)
        
        if times:
            avg_time = sum(times) / len(times) * 1e6
            per_token = avg_time / num_steps
            print(f'  {num_steps} steps: {avg_time:.2f} us total, {per_token:.2f} us/token')

if __name__ == '__main__':
    main()
