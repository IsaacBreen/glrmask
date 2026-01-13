#!/usr/bin/env python3
"""
Advanced Bit/Range Reordering to Minimize Total Ranges.

Uses Spectral Seriation and Reverse Cuthill-McKee (RCM) to find optimal 
permutations of the position space that maximize range adjacency.

Algorithms:
1. Reverse Cuthill-McKee (RCM): Minimizes matrix bandwidth.
2. Spectral Seriation: Uses Fiedler vector of Laplacian to linearize.
"""

import json
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.sparse.csgraph import reverse_cuthill_mckee
import sys

def load_weights(filename):
    print(f"--- Loading {filename} ---", flush=True)
    with open(filename, 'r') as f:
        weights = json.load(f)
    
    # Flatten to list of sets
    weight_sets = []
    all_positions = set()
    
    print("  Expanding ranges to positions...", flush=True)
    for i, w in enumerate(weights):
        positions = set()
        for start, end in w:
            # Safety cap for huge ranges (though Terminal DWA shouldn't have them)
            if end - start > 10000:
                continue 
            for p in range(start, end + 1):
                positions.add(p)
        if positions:
            weight_sets.append(sorted(list(positions)))
            all_positions.update(positions)
            
    # Map sparse universe to dense indices 0..K
    sorted_universe = sorted(list(all_positions))
    orig_to_dense = {p: i for i, p in enumerate(sorted_universe)}
    dense_to_orig = {i: p for i, p in enumerate(sorted_universe)}
    
    dense_weights = []
    for w in weight_sets:
        dense_weights.append([orig_to_dense[p] for p in w])
        
    print(f"  Loaded {len(dense_weights)} weights covering {len(all_positions)} unique positions.", flush=True)
    return dense_weights, dense_to_orig

def count_ranges_dense(weights, perm_map=None):
    """
    Count ranges in the weights.
    If perm_map is provided (dense_idx -> new_order), use it.
    """
    total_ranges = 0
    for w in weights:
        if not w:
            continue
            
        # Map to reordered positions
        if perm_map:
            mapped = sorted([perm_map[p] for p in w])
        else:
            mapped = sorted(w) # Already sorted dense indices? No, usually not contiguous
            
        # Count ranges
        if not mapped:
             continue
             
        ranges = 1
        for i in range(1, len(mapped)):
            if mapped[i] > mapped[i-1] + 1:
                ranges += 1
        total_ranges += ranges
    return total_ranges

def build_adjacency(weights, num_nodes):
    """
    Build adjacency matrix of positions co-occurring in weights.
    A[i,j] = 1 if i and j appear in the same weight.
    """
    print("  Building adjacency matrix...", flush=True)
    # This can be huge, use sparse matrix
    # A = W.T * W where W is (num_weights x num_nodes)
    
    # Construct W
    row_ind = []
    col_ind = []
    data = []
    
    for i, w in enumerate(weights):
        for p in w:
            row_ind.append(i)
            col_ind.append(p)
            data.append(1)
            
    W = sp.csr_matrix((data, (row_ind, col_ind)), shape=(len(weights), num_nodes))
    
    # Co-occurrence: A = W.T * W
    # A[i,j] = count of weights containing both i and j
    A = W.T @ W
    
    # Remove self-loops and make binary? Or keep weighted?
    # Weighted is better for spectral
    A.setdiag(0)
    return A

def spectral_ordering(A):
    print("  Computing Spectral Ordering (Fiedler vector)...", flush=True)
    # A is affinity matrix. Laplacian L = D - A
    # Fiedler vector is eigenvector matching 2nd smallest eigenvalue
    
    # Use scipy's laplacian
    L = csgraph.laplacian(A, normed=False)
    
    # Eigen decomposition
    # Only need first 2 eigenvalues/vectors
    # 'SM' = Smallest Magnitude
    try:
        eigvals, eigvecs = sp.linalg.eigsh(L, k=2, which='SM', sigma=1e-6) # Sigma shift to find near 0
        fiedler = eigvecs[:, 1]
        perm = np.argsort(fiedler)
        return perm
    except Exception as e:
        print(f"Spectral failed: {e}")
        return None

def main():
    filename = "range_weights_terminal_dwa.json"
    dense_weights, dense_to_orig = load_weights(filename)
    num_nodes = len(dense_to_orig)
    
    # Base count (using dense indices implies identity permutation of observed universe)
    # Note: This is counts in the "Dense Compressed" space (0..K) not original sparse space.
    # To compare fairly with original ranges, we should count ranges in original values first?
    # Actually, the problem is about *adjacency*.
    # Original sparse values: might be [100, 200, 300] -> 3 ranges
    # Dense values: [0, 1, 2] -> 1 range (if we map 100->0, 200->1, 300->2)
    # Reordering effectively *defines* the new integer space.
    # So counting ranges in the dense 0..K space is correct for the "best possible case for this permutation".
    
    base_dense_ranges = count_ranges_dense(dense_weights)
    print(f"\nBaseline (Dense Packing): {base_dense_ranges} ranges")
    print("(This assumes we can map the sparse universe to 0..K arbitrarily, effectively compressing gaps)")
    
    # But wait, the user wants "reordering". 
    # If we output a permutation, we are essentially saying:
    # "Token ID X is now Integer Y".
    # And "Token ID Z is now Integer Y+1".
    
    A = build_adjacency(dense_weights, num_nodes)
    
    # 1. Reverse Cuthill-McKee
    print("\nRunning Reverse Cuthill-McKee...")
    perm_rcm = reverse_cuthill_mckee(A)
    # perm_rcm is the new order of nodes. 
    # i.e. perm_rcm[0] is the node that should be at index 0.
    # We need map: node_idx -> distinct_order_idx
    rcm_map = np.zeros(num_nodes, dtype=int)
    rcm_map[perm_rcm] = np.arange(num_nodes)
    
    ranges_rcm = count_ranges_dense(dense_weights, rcm_map)
    print(f"RCM Total Ranges: {ranges_rcm}")
    print(f"Reduction vs Dense Baseline: {base_dense_ranges/ranges_rcm:.2f}x")
    
    # 2. Spectral
    print("\nRunning Spectral Seriation...")
    perm_spectral = spectral_ordering(A)
    if perm_spectral is not None:
        spectral_map = np.zeros(num_nodes, dtype=int)
        spectral_map[perm_spectral] = np.arange(num_nodes)
        
        ranges_spec = count_ranges_dense(dense_weights, spectral_map)
        print(f"Spectral Total Ranges: {ranges_spec}")
        print(f"Reduction vs Dense Baseline: {base_dense_ranges/ranges_spec:.2f}x")

if __name__ == "__main__":
    main()
