#!/usr/bin/env python3
"""
Estimate what range counts would look like if we used factored representation.

In factored representation, each weight is stored as (base_weight, tsid_mask).
The base_weight is in N-space (LLM tokens), not N×M-space.

This script estimates the savings by "un-expanding" the current weights.
"""

import json
import sys


def load_weights(filename):
    print(f"--- Loading {filename} ---", flush=True)
    with open(filename, 'r') as f:
        weights = json.load(f)
    
    max_val = 0
    clipped = []
    for w in weights:
        c = [(s, e) for s, e in w if e < (1 << 32)]
        clipped.append(c)
        for s, e in c:
            max_val = max(max_val, e)
    
    return clipped, max_val


def estimate_factored_ranges(filename, num_tsids):
    """
    Estimate what range count would be if we used factored representation.
    
    Current representation: each weight is in N×M space
    - Position = llm_token * num_tsids + tsid
    
    Factored representation: (base_ranges, tsid_mask)
    - base_ranges in N-space: llm_token IDs
    - tsid_mask: which tsids this weight covers
    
    To "un-expand", we:
    1. For each position, compute (llm_token, tsid) = divmod(position, num_tsids)
    2. Collect all llm_tokens into base_ranges
    3. Collect all tsids into tsid_mask
    """
    weights, max_val = load_weights(filename)
    
    print(f"\n=== Factored Representation Estimate ===", flush=True)
    print(f"Using num_tsids = {num_tsids}", flush=True)
    
    total_current_ranges = 0
    total_base_ranges = 0
    total_mask_ranges = 0
    
    for w_idx, w in enumerate(weights):
        if (w_idx + 1) % 200 == 0:
            print(f"  Processing {w_idx+1}/{len(weights)}...", flush=True)
        
        total_current_ranges += len(w)
        
        # Extract (llm_token, tsid) pairs
        tokens = set()
        tsids = set()
        
        for start, end in w:
            for pos in range(start, min(end + 1, start + 1000000)):  # Cap to avoid OOM
                llm_token = pos // num_tsids
                tsid = pos % num_tsids
                tokens.add(llm_token)
                tsids.add(tsid)
            
            # For large ranges, estimate
            if end - start >= 1000000:
                # Just add the boundary tokens
                tokens.add(start // num_tsids)
                tokens.add(end // num_tsids)
                for t in range(num_tsids):
                    tsids.add(t)
        
        # Count ranges in tokens and tsids
        # Sort and count contiguous runs
        def count_ranges(s):
            if not s:
                return 0
            sorted_vals = sorted(s)
            ranges = 1
            for i in range(1, len(sorted_vals)):
                if sorted_vals[i] > sorted_vals[i-1] + 1:
                    ranges += 1
            return ranges
        
        base_ranges = count_ranges(tokens)
        mask_ranges = count_ranges(tsids)
        
        total_base_ranges += base_ranges
        total_mask_ranges += mask_ranges
    
    print(f"\n  Results:", flush=True)
    print(f"    Weights: {len(weights)}", flush=True)
    print(f"    Current total ranges: {total_current_ranges}", flush=True)
    print(f"    Factored base ranges: {total_base_ranges}", flush=True)
    print(f"    Factored mask ranges: {total_mask_ranges}", flush=True)
    print(f"    Factored total: {total_base_ranges + total_mask_ranges}", flush=True)
    print(f"    Reduction: {total_current_ranges / (total_base_ranges + total_mask_ranges):.2f}x", flush=True)


if __name__ == "__main__":
    # num_tsids is typically the number of tokenizer DFA states
    # For ApolloRouter, let's estimate based on the data
    
    # First, let's find what num_tsids might be by looking at the pattern
    # In expanded representation: positions are llm_token * num_tsids + tsid
    # If we find the GCD of position differences, we might infer num_tsids
    
    # For now, try common values
    for num_tsids in [4476]:  # Based on earlier pattern analysis showing 4476 stride
        for fname in ["range_weights_terminal_dwa.json"]:
            estimate_factored_ranges(fname, num_tsids)
