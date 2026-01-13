#!/usr/bin/env python3
"""
Cluster Token IDs to Minimize Range Fragmentation (V2).

Optimized for speed and memory using Scipy sparse matrices and RCM.

Approach:
1. Load base token sets
2. Build sparse adjacency matrix of Tokens co-occurring in sparse sets (density < 0.5)
3. Use Reverse Cuthill-McKee (RCM) to minimize bandwidth (cluster tokens)
4. Measure reduction on ALL sets (including dense ones)
"""

import json
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import sys

def load_base_sets(filename):
    print(f"--- Loading {filename} ---", flush=True)
    with open(filename, 'r') as f:
        data = json.load(f)
    
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

def count_ranges(weights, perm=None):
    total = 0
    for w in weights:
        if not w:
            continue
            
        if perm is not None:
             mapped = sorted([perm[t] for t in w])
        else:
             mapped = sorted(list(w))
             
        ranges = 1
        for i in range(1, len(mapped)):
            if mapped[i] > mapped[i-1] + 1:
                ranges += 1
        total += ranges
    return total

def build_sparse_adjacency(weights, max_token):
    print("Building sparse adjacency matrix...", flush=True)
    num_tokens = max_token + 1
    
    # We want A[i,j] = 1 if i and j appear in same sparse weight
    # Construct incidence matrix B where B[w, t] = 1
    # Then A = B.T * B
    
    rows = []
    cols = []
    data = []
    
    skipped_dense = 0
    used_weights = 0
    
    for w_idx, w in enumerate(weights):
        # Skip dense weights - aggressively!
        # Only use small sets to determine local topology.
        # Large sets (even 1000 tokens) create 1M edges in adjacency mask.
        if len(w) > 200:
            skipped_dense += 1
            continue
            
        used_weights += 1
        for t in w:
            rows.append(w_idx)
            cols.append(t)
            data.append(1)
            
    print(f"  Used {used_weights} sparse weights (skipped {skipped_dense} dense)", flush=True)
    
    B = csr_matrix((data, (rows, cols)), shape=(len(weights), num_tokens))
    
    # Co-occurrence
    A = B.T @ B
    A.setdiag(0)
    return A

def main():
    filename = "base_token_sets_terminal.json"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        
    weights, max_token = load_base_sets(filename)
    num_tokens = max_token + 1
    
    original_ranges = count_ranges(weights)
    print(f"\nOriginal N-space Ranges: {original_ranges}")
    
    # Build graph
    try:
        A = build_sparse_adjacency(weights, max_token)
        
        print("Running Reverse Cuthill-McKee...", flush=True)
        perm_idx = reverse_cuthill_mckee(A)
        
        # perm_idx is permutation of rows. New order of node i is found by inverse?
        # No, perm_idx[k] is the node index that comes k-th in the new ordering.
        # So Node P is at new position: find where P is in perm_idx.
        
        perm_map = np.zeros(num_tokens, dtype=int)
        perm_map[perm_idx] = np.arange(len(perm_idx))
        
        new_ranges = count_ranges(weights, perm_map)
        
        print(f"\nReordered N-space Ranges: {new_ranges}")
        print(f"Reduction Factor: {original_ranges / new_ranges:.2f}x")
        
        # Estimate Total Terminal DWA Ranges
        # mask ranges ~ 2000 (conservatively)
        mask_ranges = 2500 
        total_orig = original_ranges + mask_ranges # Approx
        total_new = new_ranges + mask_ranges
        
        print(f"\nEstimated Terminal DWA Total Ranges:")
        print(f"  Current: ~{total_orig}")
        print(f"  Projected: ~{total_new}")
        
        if total_new < 10000:
             print("\nSUCCESS! < 10K Target Achievable.")
        else:
             print("\nStill above 10K target.")
             
    except Exception as e:
        print(f"Clustering failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
