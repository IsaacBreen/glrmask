#!/usr/bin/env python3
"""
Compression Benchmark: Compare multiple strategies for weight range-sets.

Tests:
1. Roaring Bitmaps (pyroaring)
2. Pattern Factorization (custom)
3. Raw representation baseline

All strategies clip usize::MAX to actual max value.
"""

import json
import sys
import time
from collections import defaultdict

# Try importing libraries
try:
    from pyroaring import BitMap
    HAS_ROARING = True
except ImportError:
    HAS_ROARING = False
    print("pyroaring not available", flush=True)


def load_weights(filename):
    """Load weights from JSON, clipping sentinel values."""
    print(f"--- Loading {filename} ---", flush=True)
    
    try:
        with open(filename, 'r') as f:
            weights = json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found.", flush=True)
        return None, 0
    
    # Find actual max value (ignore sentinels like u64::MAX)
    max_val = 0
    for w in weights:
        for start, end in w:
            if end < (1 << 32):  # Skip sentinel values
                max_val = max(max_val, end)
    
    print(f"Loaded {len(weights)} weights, max_val={max_val}", flush=True)
    
    # Clip all ranges to max_val
    clipped_weights = []
    for w in weights:
        clipped = []
        for start, end in w:
            if start > max_val:
                continue
            clipped.append((start, min(end, max_val)))
        clipped_weights.append(clipped)
    
    return clipped_weights, max_val


def test_roaring(weights, max_val):
    """Test Roaring Bitmap compression."""
    if not HAS_ROARING:
        return None
    
    print("\n=== Roaring Bitmaps ===", flush=True)
    start_time = time.time()
    
    total_ranges = 0
    total_roaring_bytes = 0
    bitmaps = []
    
    for i, w in enumerate(weights):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(weights)}...", flush=True)
        
        bm = BitMap()
        for start, end in w:
            total_ranges += 1
            # Add range [start, end] inclusive
            bm.add_range(start, end + 1)
        
        bitmaps.append(bm)
        total_roaring_bytes += bm.get_statistics()['n_bytes_array_containers']
        total_roaring_bytes += bm.get_statistics()['n_bytes_bitset_containers']
        total_roaring_bytes += bm.get_statistics()['n_bytes_run_containers']
    
    elapsed = time.time() - start_time
    
    # Get total serialized size
    total_serialized = sum(len(bm.serialize()) for bm in bitmaps)
    raw_size = total_ranges * 2 * 4  # 2 u32s per range
    
    print(f"  Total ranges: {total_ranges}", flush=True)
    print(f"  Raw size (bytes): {raw_size}", flush=True)
    print(f"  Roaring serialized (bytes): {total_serialized}", flush=True)
    print(f"  Compression ratio: {raw_size / total_serialized:.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    return {
        "method": "Roaring",
        "raw_bytes": raw_size,
        "compressed_bytes": total_serialized,
        "ratio": raw_size / total_serialized if total_serialized > 0 else 0,
        "time": elapsed,
        "bitmaps": bitmaps,
    }


def test_pattern_factorization(weights, max_val):
    """
    Test Pattern Factorization: identify shared patterns across weights.
    
    Idea: Many weights may share the same underlying structure. Deduplicate
    the range-lists and count unique patterns.
    """
    print("\n=== Pattern Factorization ===", flush=True)
    start_time = time.time()
    
    # Convert each weight to a hashable tuple
    pattern_to_id = {}
    weight_pattern_ids = []
    
    for w in weights:
        pattern = tuple(tuple(r) for r in w)
        if pattern not in pattern_to_id:
            pattern_to_id[pattern] = len(pattern_to_id)
        weight_pattern_ids.append(pattern_to_id[pattern])
    
    num_unique_patterns = len(pattern_to_id)
    
    # Calculate storage: unique patterns + index per weight
    total_ranges_in_patterns = sum(len(p) for p in pattern_to_id.keys())
    pattern_storage = total_ranges_in_patterns * 2 * 4  # 2 u32s per range
    index_storage = len(weights) * 4  # 1 u32 per weight (pattern ID)
    factorized_size = pattern_storage + index_storage
    
    total_ranges = sum(len(w) for w in weights)
    raw_size = total_ranges * 2 * 4
    
    elapsed = time.time() - start_time
    
    print(f"  Total weights: {len(weights)}", flush=True)
    print(f"  Unique patterns: {num_unique_patterns}", flush=True)
    print(f"  Total ranges (raw): {total_ranges}", flush=True)
    print(f"  Total ranges (factorized): {total_ranges_in_patterns}", flush=True)
    print(f"  Raw size (bytes): {raw_size}", flush=True)
    print(f"  Factorized size (bytes): {factorized_size}", flush=True)
    print(f"  Compression ratio: {raw_size / factorized_size:.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    return {
        "method": "Pattern Factorization",
        "raw_bytes": raw_size,
        "compressed_bytes": factorized_size,
        "ratio": raw_size / factorized_size if factorized_size > 0 else 0,
        "time": elapsed,
        "unique_patterns": num_unique_patterns,
    }


def test_roaring_with_sharing(weights, max_val):
    """
    Roaring with pattern sharing: deduplicate identical bitmaps.
    """
    if not HAS_ROARING:
        return None
    
    print("\n=== Roaring + Sharing ===", flush=True)
    start_time = time.time()
    
    # Build bitmaps and hash them
    bitmap_to_id = {}
    weight_bitmap_ids = []
    unique_bitmaps = []
    
    for i, w in enumerate(weights):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(weights)}...", flush=True)
        
        bm = BitMap()
        for start, end in w:
            bm.add_range(start, end + 1)
        
        # Use serialized form as hash key
        serialized = bytes(bm.serialize())
        if serialized not in bitmap_to_id:
            bitmap_to_id[serialized] = len(unique_bitmaps)
            unique_bitmaps.append(bm)
        weight_bitmap_ids.append(bitmap_to_id[serialized])
    
    elapsed = time.time() - start_time
    
    # Size calculation
    unique_serialized = sum(len(bm.serialize()) for bm in unique_bitmaps)
    index_storage = len(weights) * 4  # u32 per weight
    total_shared_size = unique_serialized + index_storage
    
    total_ranges = sum(len(w) for w in weights)
    raw_size = total_ranges * 2 * 4
    
    print(f"  Total weights: {len(weights)}", flush=True)
    print(f"  Unique bitmaps: {len(unique_bitmaps)}", flush=True)
    print(f"  Raw size (bytes): {raw_size}", flush=True)
    print(f"  Shared size (bytes): {total_shared_size}", flush=True)
    print(f"  Compression ratio: {raw_size / total_shared_size:.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    return {
        "method": "Roaring + Sharing",
        "raw_bytes": raw_size,
        "compressed_bytes": total_shared_size,
        "ratio": raw_size / total_shared_size if total_shared_size > 0 else 0,
        "time": elapsed,
        "unique_bitmaps": len(unique_bitmaps),
    }


def test_range_set_hashing(weights, max_val):
    """
    Range-set hashing: Hash individual ranges and count unique ones.
    Then represent each weight as a set of range IDs.
    """
    print("\n=== Range-Set Hashing ===", flush=True)
    start_time = time.time()
    
    range_to_id = {}
    weight_range_ids = []
    
    for w in weights:
        ids = []
        for r in w:
            key = tuple(r)
            if key not in range_to_id:
                range_to_id[key] = len(range_to_id)
            ids.append(range_to_id[key])
        weight_range_ids.append(ids)
    
    elapsed = time.time() - start_time
    
    num_unique_ranges = len(range_to_id)
    total_range_refs = sum(len(ids) for ids in weight_range_ids)
    
    # Storage: unique ranges + references
    range_storage = num_unique_ranges * 2 * 4  # 2 u32s per unique range
    ref_storage = total_range_refs * 4  # u32 per reference
    total_size = range_storage + ref_storage
    
    total_ranges = sum(len(w) for w in weights)
    raw_size = total_ranges * 2 * 4
    
    print(f"  Total ranges (raw): {total_ranges}", flush=True)
    print(f"  Unique ranges: {num_unique_ranges}", flush=True)
    print(f"  Raw size (bytes): {raw_size}", flush=True)
    print(f"  Hashed size (bytes): {total_size}", flush=True)
    print(f"  Compression ratio: {raw_size / total_size:.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    return {
        "method": "Range-Set Hashing",
        "raw_bytes": raw_size,
        "compressed_bytes": total_size,
        "ratio": raw_size / total_size if total_size > 0 else 0,
        "time": elapsed,
        "unique_ranges": num_unique_ranges,
    }


def main():
    results = {}
    
    for filename in ["range_weights_terminal_dwa.json", "range_weights_parser_dwa.json"]:
        print(f"\n{'='*60}", flush=True)
        print(f"BENCHMARKING: {filename}", flush=True)
        print(f"{'='*60}", flush=True)
        
        weights, max_val = load_weights(filename)
        if weights is None:
            continue
        
        file_results = []
        
        # Test all methods
        r = test_pattern_factorization(weights, max_val)
        if r: file_results.append(r)
        
        r = test_range_set_hashing(weights, max_val)
        if r: file_results.append(r)
        
        r = test_roaring(weights, max_val)
        if r: file_results.append(r)
        
        r = test_roaring_with_sharing(weights, max_val)
        if r: file_results.append(r)
        
        results[filename] = file_results
        
        # Summary for this file
        print(f"\n--- Summary for {filename} ---", flush=True)
        print(f"{'Method':<25} {'Raw':>12} {'Compressed':>12} {'Ratio':>8}", flush=True)
        print("-" * 60, flush=True)
        for r in sorted(file_results, key=lambda x: -x['ratio']):
            print(f"{r['method']:<25} {r['raw_bytes']:>12} {r['compressed_bytes']:>12} {r['ratio']:>7.2f}x", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("OVERALL WINNER", flush=True)
    print("="*60, flush=True)
    
    all_results = []
    for fname, fr in results.items():
        for r in fr:
            all_results.append({**r, "file": fname})
    
    best = max(all_results, key=lambda x: x['ratio'])
    print(f"Best method: {best['method']}", flush=True)
    print(f"  File: {best['file']}", flush=True)
    print(f"  Compression ratio: {best['ratio']:.2f}x", flush=True)


if __name__ == "__main__":
    main()
