#!/usr/bin/env python3
"""
Cluster Token IDs to Minimize Range Fragmentation (V5 - Greedy Chain).

Optimized for speed. Handles sparse, large Token IDs by mapping to dense space.
Uses Greedy Chain Aggregation to explicitly maximize adjacency.

Approach:
1. Stream load base token sets.
2. Map to Dense (4k nodes).
3. Build Adjacency Matrix (Graph).
4. Run Greedy Chain algorithm:
   - Pick start node.
   - Pick neighbor with edge weight.
   - Extend chain.
"""

import json
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import sys
import collections

def count_ranges_dense(dense_weights, perm_map=None):
    total = 0
    for w in dense_weights:
        if not w: continue
        if perm_map is not None:
             mapped = sorted([perm_map[t] for t in w])
        else:
             mapped = sorted(w)
        ranges = 1
        for i in range(1, len(mapped)):
            if mapped[i] > mapped[i-1] + 1:
                ranges += 1
        total += ranges
    return total

def greedy_chain_ordering(A, num_nodes):
    print("Running Greedy Chain Ordering...", flush=True)
    # A is sparse adjacency.
    # Convert to dense adj lists (sorted by weight desc) for speed?
    # 4k nodes. 4k*4k = 16M. Dense matrix is fine.
    
    # Actually A is sparse. Let's use it.
    # We want: for current u, find unvisited v with max weight A[u,v].
    
    visited = set()
    ordering = []
    
    # Precompute neighbors sorted by weight for each node
    adj = collections.defaultdict(list)
    cx = A.tocoo()
    for u, v, w in zip(cx.row, cx.col, cx.data):
        if u != v:
            adj[u].append((w, v))
            
    # Sort neighbors
    for u in adj:
        adj[u].sort(key=lambda x: x[0], reverse=True)
        
    # Start with global max degree/weight node?
    degrees = np.array(A.sum(axis=1)).flatten()
    start_candidates = np.argsort(degrees)[::-1] # descending degree
    
    for start_node in start_candidates:
        if start_node in visited:
            continue
            
        # Start new chain
        chain = [start_node]
        visited.add(start_node)
        current = start_node
        
        while True:
            # Find best neighbor of current
            best_neigh = -1
            best_w = -1
            
            # Check pre-sorted neighbors
            found = False
            for w, v in adj[current]:
                if v not in visited:
                    best_neigh = v
                    best_w = w
                    found = True
                    break
            
            if found:
                visited.add(best_neigh)
                chain.append(best_neigh)
                current = best_neigh
            else:
                # Chain ends
                break
        
        ordering.extend(chain)
        
    # Add isolated nodes
    for i in range(num_nodes):
        if i not in visited:
            ordering.append(i)
            
    return ordering

def main():
    filename = "base_token_sets_terminal.json"
    if len(sys.argv) > 1: filename = sys.argv[1]
    
    print(f"--- Loading {filename} ---", flush=True)
    with open(filename, 'r') as f: data = json.load(f)
        
    # 1. Map to Dense
    print("Collecting unique tokens...", flush=True)
    unique_tokens = set()
    for ranges in data:
         for start, end in ranges:
             # Just map everything. Assuming 4k tokens based on previous run.
             if end - start > 100000: continue 
             for t in range(start, end + 1): unique_tokens.add(t)
    sorted_universe = sorted(unique_tokens)
    token_map = {t: i for i, t in enumerate(sorted_universe)}
    num_dense = len(sorted_universe)
    print(f"  Unique Tokens: {num_dense}", flush=True)
    
    # 2. Convert
    print("Converting...", flush=True)
    dense_weights = []
    for ranges in data:
        w_dense = []
        size = sum(end - start + 1 for start, end in ranges)
        if size > 10000: pass # Skip huge for clustering, but...
        else:
            for start, end in ranges:
                for t in range(start, end + 1):
                    if t in token_map: w_dense.append(token_map[t])
        dense_weights.append(w_dense)
        
    # 3. Original Count
    base_ranges = count_ranges_dense(dense_weights, None)
    print(f"Original Dense Ranges (Sorted IDs): {base_ranges}", flush=True)
    
    # 4. Build Graph
    rows, cols, vals = [], [], []
    for i, w in enumerate(dense_weights):
        if len(w) > 200: continue
        for t in w:
            rows.append(i); cols.append(t); vals.append(1)
            
    B = csr_matrix((vals, (rows, cols)), shape=(len(dense_weights), num_dense))
    A = B.T @ B
    A.setdiag(0)
    
    # 5. Greedy
    perm_idx = greedy_chain_ordering(A, num_dense)
    
    perm_map = np.empty(num_dense, dtype=np.int64)
    # perm_idx is ordered list of nodes.
    # We want node perm_idx[0] to map to index 0.
    # map[node] = rank
    for rank, node in enumerate(perm_idx):
        perm_map[node] = rank
        
    new_ranges = count_ranges_dense(dense_weights, perm_map)
    print(f"Greedy Reordered Ranges: {new_ranges}", flush=True)
    if new_ranges > 0:
        print(f"Reduction Factor: {base_ranges / new_ranges:.2f}x", flush=True)
        
    print(f"New Total Est: {new_ranges} + 2000 = {new_ranges + 2000}")
    if new_ranges + 2000 < 10000:
        print("SUCCESS!")

if __name__ == "__main__":
    main()
