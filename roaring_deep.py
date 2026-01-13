#!/usr/bin/env python3
"""
Deep analysis of Roaring bitmap internals to understand compression.
"""

import json
import sys
import time
from pyroaring import BitMap


def load_weights(filename):
    print(f"--- Loading {filename} ---", flush=True)
    try:
        with open(filename, 'r') as f:
            weights = json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found.", flush=True)
        return None, 0
    
    max_val = 0
    clipped = []
    for w in weights:
        c = []
        for start, end in w:
            if end < (1 << 32):
                max_val = max(max_val, end)
                c.append((start, min(end, max_val)))
        clipped.append(c)
    return clipped, max_val


def analyze_roaring_detail(filename):
    weights, max_val = load_weights(filename)
    if not weights:
        return
    
    print(f"\n=== Roaring Detailed Analysis: {filename} ===", flush=True)
    
    total_ranges = sum(len(w) for w in weights)
    raw_size = total_ranges * 2 * 4
    print(f"  Total ranges: {total_ranges}", flush=True)
    print(f"  Raw size: {raw_size} bytes", flush=True)
    
    # Build bitmaps
    bitmaps = []
    total_cardinality = 0
    
    stats_before = {"array": 0, "bitset": 0, "run": 0, "bytes": 0}
    stats_after = {"array": 0, "bitset": 0, "run": 0, "bytes": 0}
    
    for i, w in enumerate(weights):
        bm = BitMap()
        for start, end in w:
            bm.add_range(start, end + 1)
        
        total_cardinality += len(bm)
        
        # Get stats before run_optimize
        s = bm.get_statistics()
        stats_before["array"] += s['n_array_containers']
        stats_before["bitset"] += s['n_bitset_containers']
        stats_before["run"] += s['n_run_containers']
        stats_before["bytes"] += len(bm.serialize())
        
        # Run optimize
        bm.run_optimize()
        
        s = bm.get_statistics()
        stats_after["array"] += s['n_array_containers']
        stats_after["bitset"] += s['n_bitset_containers']
        stats_after["run"] += s['n_run_containers']
        stats_after["bytes"] += len(bm.serialize())
        
        bitmaps.append(bm)
    
    print(f"\n  Total cardinality (elements in sets): {total_cardinality}", flush=True)
    print(f"  Avg elements per weight: {total_cardinality / len(weights):.0f}", flush=True)
    
    print(f"\n  Before run_optimize():", flush=True)
    print(f"    Array containers: {stats_before['array']}", flush=True)
    print(f"    Bitset containers: {stats_before['bitset']}", flush=True)
    print(f"    Run containers: {stats_before['run']}", flush=True)
    print(f"    Total bytes: {stats_before['bytes']}", flush=True)
    print(f"    Compression ratio: {raw_size / stats_before['bytes']:.2f}x", flush=True)
    
    print(f"\n  After run_optimize():", flush=True)
    print(f"    Array containers: {stats_after['array']}", flush=True)
    print(f"    Bitset containers: {stats_after['bitset']}", flush=True)
    print(f"    Run containers: {stats_after['run']}", flush=True)
    print(f"    Total bytes: {stats_after['bytes']}", flush=True)
    print(f"    Compression ratio: {raw_size / stats_after['bytes']:.2f}x", flush=True)
    
    # Check if sharing bitmaps helps
    print(f"\n  Checking bitmap sharing...", flush=True)
    unique_bitmaps = {}
    for bm in bitmaps:
        key = bytes(bm.serialize())
        if key not in unique_bitmaps:
            unique_bitmaps[key] = bm
    
    shared_bytes = sum(len(k) for k in unique_bitmaps.keys())
    index_bytes = len(weights) * 4
    shared_total = shared_bytes + index_bytes
    
    print(f"    Unique bitmaps: {len(unique_bitmaps)}", flush=True)
    print(f"    Shared bytes: {shared_bytes}", flush=True)
    print(f"    Index bytes: {index_bytes}", flush=True)
    print(f"    Total shared: {shared_total}", flush=True)
    print(f"    Compression ratio (shared): {raw_size / shared_total:.2f}x", flush=True)


if __name__ == "__main__":
    for fname in ["range_weights_terminal_dwa.json", "range_weights_parser_dwa.json"]:
        analyze_roaring_detail(fname)
