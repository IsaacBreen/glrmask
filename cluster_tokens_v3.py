#!/usr/bin/env python3
"""
Cluster Token IDs to Minimize Range Fragmentation (V4 - Dense Mapping).

Optimized for speed and memory. Handles sparse, large Token IDs by mapping to dense space.
Does NOT store expanded sets in memory.

Approach:
1. Stream load base token sets ranges.
2. Build map from Large Token ID -> Dense ID (0..N).
3. Build sparse adjacency matrix of Dense Tokens.
4. Use Reverse Cuthill-McKee (RCM).
5. Measure reduction.
"""

import json
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csgraph
import sys
import gc

def count_ranges_dense(dense_weights, perm_map=None):
    """
    Count ranges in dense weights (lists of dense IDs).
    """
    total = 0
    for w in dense_weights:
        if not w:
            continue
            
        # Map
        if perm_map is not None:
             mapped = sorted([perm_map[t] for t in w])
        else:
             mapped = sorted(w)
             
        ranges_count = 1
        for i in range(1, len(mapped)):
            if mapped[i] > mapped[i-1] + 1:
                ranges_count += 1
        total += ranges_count
    return total

def main():
    filename = "base_token_sets_terminal.json"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    print(f"--- Loading {filename} ---", flush=True)
    with open(filename, 'r') as f:
        data = json.load(f)
        
    # 1. Collect Universe and Map to Dense
    print("Collecting unique tokens...", flush=True)
    unique_tokens = set()
    for ranges in data:
         for start, end in ranges:
             # If range is huge, iterating is slow. But max range size?
             # If sparse weights have small ranges, it's fine.
             # If dense weights have huge ranges, we skip them for collecting?
             # No, we need uniform mapping.
             # Assume huge ranges are RARE (usually complements).
             # We can't iterate 4 quadrillion items.
             # But ranges are in N-space. N-space is only 4 quadrillion if sparse.
             # Wait. If tokens are 0, 10, 20. And weight is [0, 20]. It covers 0,1...20.
             # Does it cover 21 integers? Yes.
             # My assumption that N-space is small (12k) implies tokens are dense?
             # BUT `Max Token: 4 quadrillion` implies huge gaps?
             # Or N-space uses Hashed IDs?
             # If N-space uses Hashed IDs, ranges [start, end] might cover huge empty space?
             # No, ranges imply contiguous validity.
             # If Token 0 and Token 100 are valid, and [0,100] is the range, then 1..99 must be valid too.
             # Valid implies "maps to valid LLM token".
             # If Tokens are large ints, maybe they are just large ints.
             # Are there 4 quadrillion valid tokens? No.
             # So huge ranges [0, 4e15] are impossible?
             # OR they are `complements` (everything valid).
             # If I iterate [0, 4e15], I die.
             
             size = end - start + 1
             if size > 100000:
                 # Skip massive ranges for universe collection? 
                 # Risky. If dense weight has unique tokens...
                 # But usually universal set uses existing tokens.
                 continue
                 
             for t in range(start, end + 1):
                 unique_tokens.add(t)
                 
    sorted_universe = sorted(unique_tokens)
    token_map = {t: i for i, t in enumerate(sorted_universe)}
    num_dense = len(sorted_universe)
    print(f"  Unique Tokens (filtered large ranges): {num_dense}", flush=True)
    
    # 2. Convert Sparse Weights to Dense Sets
    # Only convert small weights to avoid expanding huge ranges
    print("Converting weights to dense space...", flush=True)
    dense_weights = []
    skipped_large = 0
    
    for ranges in data:
        w_dense = []
        is_large = False
        
        # Check size first
        total_size = sum(end - start + 1 for start, end in ranges)
        if total_size > 10000: # Arbitrary "large" cuttoff
            skipped_large += 1
            is_large = True
        else:
            for start, end in ranges:
                for t in range(start, end + 1):
                    if t in token_map:
                         w_dense.append(token_map[t])
        
        # If it was large, we might have partial data if we filtered mapping?
        # Let's just track sparse weights for clustering.
        # But for counting total reduction, we need ALL weights.
        # Dense weights (spans) in sparse space -> spans in dense space?
        # If we map 0->0, 10->1. Range [0, 10] (11 items). 2 are valid.
        # Dense weight: [0, 1].
        # So "Range" in sparse space maps to "Set of valid dense tokens".
        # Yes.
        
        if not is_large:
            dense_weights.append(w_dense)
        else:
            # We can't represent large weight in dense space easily if we excluded tokens?
            # It's fine. We optimize for the sparse weights.
            pass
            
    print(f"  Converted {len(dense_weights)} sparse weights (skipped {skipped_large})", flush=True)
    
    # 3. Count Original (Dense) Ranges
    # This checks "How fragmented is it if we just repack 0..K?"
    # Repacking by sorting universe IS a form of clustering (identity).
    original_ranges = count_ranges_dense(dense_weights, None)
    print(f"Original Dense Ranges: {original_ranges}", flush=True)
    
    # 4. Build Graph
    # Filter for very sparse
    rows = []
    cols = []
    vals = []
    
    for i, w in enumerate(dense_weights):
        if len(w) > 200: continue
        for t in w:
            rows.append(i)
            cols.append(t)
            vals.append(1)
            
    if not rows:
         print("No sparse weights for graph.", flush=True)
         return
         
    B = csr_matrix((vals, (rows, cols)), shape=(len(dense_weights), num_dense))
    A = B.T @ B
    A.setdiag(0)
    
    # 5. RCM
    print("Running RCM...", flush=True)
    perm_idx = csgraph.reverse_cuthill_mckee(A, symmetric_mode=True)
    
    perm_map = np.empty(num_dense, dtype=np.int64)
    perm_map[perm_idx] = np.arange(num_dense)
    
    # 6. Count New
    new_ranges = count_ranges_dense(dense_weights, perm_map)
    print(f"Reordered Dense Ranges: {new_ranges}", flush=True)
    
    if new_ranges > 0:
        print(f"Reduction Factor: {original_ranges / new_ranges:.2f}x", flush=True)
    
    # Projection
    mask_total = 38322 - original_ranges # Sort of
    # Wait, original_ranges here is "Dense Ranges".
    # Original Sparse Ranges was ~36k.
    # If "Original Dense Ranges" is ~36k, then mapping didn't compress ranges, just space.
    # Good.
    
    print(f"\nProjection:")
    print(f"  Base Ranges: {original_ranges} -> {new_ranges}")
    print(f"  Estimated Mask: 2000")
    print(f"  New Total: {new_ranges + 2000}")

if __name__ == "__main__":
    main()
