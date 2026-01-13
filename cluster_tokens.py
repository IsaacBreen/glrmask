#!/usr/bin/env python3
"""
Cluster Token IDs to Minimize Range Fragmentation.

The goal is to find a permutation of Token IDs such that tokens that frequently
co-occur in weights are placed adjacent to each other.
This transforms fragmented point ranges [x,x] into dense spans [start, end].

Approach:
1. Load base token sets (sets of valid tokens for each weight)
2. Build affinity matrix (how often pair (i, j) appears in same weight)
3. Use spectral clustering (Fiedler vector) or greedy ordering to linearize
"""

import json
import math
from collections import defaultdict

def load_base_sets(filename):
    print(f"--- Loading {filename} ---", flush=True)
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Convert ranges to sets of integers
    weights = []
    max_token = 0
    for ranges in data:
        tokens = set()
        for start, end in ranges:
            max_token = max(max_token, end)
            for t in range(start, end + 1):
                tokens.add(t)
        weights.append(tokens)
    
    print(f"  Loaded {len(weights)} weights", flush=True)
    print(f"  Max token ID: {max_token}", flush=True)
    return weights, max_token

def count_ranges(weights):
    total = 0
    for w in weights:
        if not w:
            continue
        sorted_w = sorted(w)
        ranges = 1
        for i in range(1, len(sorted_w)):
            if sorted_w[i] > sorted_w[i-1] + 1:
                ranges += 1
        total += ranges
    return total

def greedy_ordering(weights, max_token):
    print("Building co-occurrence graph...", flush=True)
    
    # Count frequency of each token
    freq = defaultdict(int)
    # Count co-occurrence
    cooccur = defaultdict(lambda: defaultdict(int))
    
    for w in weights:
        w_list = list(w)
        for t in w_list:
            freq[t] += 1
            for t2 in w_list:
                if t < t2:
                    cooccur[t][t2] += 1
                    cooccur[t2][t] += 1
    
    print(f"  Graph built. Ordering {max_token + 1} tokens...", flush=True)
    
    # Simple Greedy:
    # 1. Start with most frequent token
    # 2. Pick next token that has highest affinity with last added
    
    ordered = []
    visited = set()
    
    # Handle unconnected components by restarting with max freq unvisited
    while len(ordered) <= max_token:
        # Pick start node (max freq unvisited)
        start_node = -1
        max_f = -1
        
        candidates = [t for t in range(max_token + 1) if t not in visited]
        if not candidates:
            break
            
        # Optimization: finding max freq every time is slow if naive
        # Just sort candidates by freq once?
        if start_node == -1:
             # Sort once for restart strategy
             sorted_candidates = sorted(candidates, key=lambda t: freq[t], reverse=True)
             start_node = sorted_candidates[0]
        
        ordered.append(start_node)
        visited.add(start_node)
        
        current = start_node
        
        # Grow chain
        while True:
            best_next = -1
            best_score = -1
            
            # Identify neighbors
            neighbors = cooccur[current]
            
            # Find best unvisited neighbor
            # Optimization: only check neighbors, not all candidates
            for neighbor, score in neighbors.items():
                if neighbor not in visited:
                    if score > best_score:
                        best_score = score
                        best_next = neighbor
            
            if best_next != -1:
                ordered.append(best_next)
                visited.add(best_next)
                current = best_next
            else:
                break # Chain ends, restart
                
    # Add any missing tokens (if gaps in range)
    for t in range(max_token + 1):
        if t not in visited:
            ordered.append(t)
            
    return ordered

def apply_ordering(weights, ordered_tokens):
    # Mapping: old_id -> new_id
    perm = {old: new for new, old in enumerate(ordered_tokens)}
    
    new_weights = []
    for w in weights:
        new_w = set()
        for t in w:
            if t in perm:
                new_w.add(perm[t])
        new_weights.append(new_w)
    return new_weights

if __name__ == "__main__":
    import sys
    filename = "base_token_sets_terminal.json"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        
    weights, max_token = load_base_sets(filename)
    
    original_ranges = count_ranges(weights)
    print(f"\nOriginal Total Ranges: {original_ranges}")
    
    ordering = greedy_ordering(weights, max_token)
    
    new_weights = apply_ordering(weights, ordering)
    new_ranges = count_ranges(new_weights)
    
    print(f"\nReordered Total Ranges: {new_ranges}")
    print(f"Reduction Factor: {original_ranges / new_ranges:.2f}x")
    
    # Estimate total Terminal DWA ranges with this base reduction
    # Current factored base was ~? (we'll see output)
    # Total ranges = base_ranges + mask_ranges
    # mask_ranges won't change.
    # We can estimate new total.

