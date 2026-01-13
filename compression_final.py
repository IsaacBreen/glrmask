#!/usr/bin/env python3
"""
Final Compression Benchmark: Delta + LZMA combo and other aggressive strategies.
"""

import json
import sys
import time
import zlib
import lzma


def load_weights(filename):
    """Load weights from JSON, clipping sentinel values."""
    print(f"--- Loading {filename} ---", flush=True)
    
    try:
        with open(filename, 'r') as f:
            weights = json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found.", flush=True)
        return None, 0
    
    max_val = 0
    for w in weights:
        for start, end in w:
            if end < (1 << 32):
                max_val = max(max_val, end)
    
    print(f"Loaded {len(weights)} weights, max_val={max_val}", flush=True)
    
    clipped_weights = []
    for w in weights:
        clipped = []
        for start, end in w:
            if start > max_val:
                continue
            clipped.append((start, min(end, max_val)))
        clipped_weights.append(clipped)
    
    return clipped_weights, max_val


def test_delta_lzma(weights, max_val):
    """Delta encoding + LZMA compression."""
    print("\n=== Delta + LZMA ===", flush=True)
    start_time = time.time()
    
    # Collect all endpoints and sort
    all_ranges = []
    for w in weights:
        all_ranges.extend(w)
    all_ranges.sort()
    
    # Delta encode
    def zigzag(n):
        return (n << 1) ^ (n >> 31)
    
    delta_bytes = b''
    prev = 0
    for start, end in all_ranges:
        d1 = zigzag(start - prev)
        d2 = zigzag(end - start)
        while d1 >= 128:
            delta_bytes += bytes([d1 & 0x7f | 0x80])
            d1 >>= 7
        delta_bytes += bytes([d1])
        while d2 >= 128:
            delta_bytes += bytes([d2 & 0x7f | 0x80])
            d2 >>= 7
        delta_bytes += bytes([d2])
        prev = end
    
    # LZMA compress the delta stream
    compressed = lzma.compress(delta_bytes)
    
    elapsed = time.time() - start_time
    
    total_ranges = len(all_ranges)
    raw_size = total_ranges * 2 * 4
    
    print(f"  Total ranges: {total_ranges}", flush=True)
    print(f"  Raw size (bytes): {raw_size}", flush=True)
    print(f"  Delta size (bytes): {len(delta_bytes)}", flush=True)
    print(f"  Delta + LZMA (bytes): {len(compressed)}", flush=True)
    print(f"  Compression ratio: {raw_size / len(compressed):.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    return {
        "method": "Delta + LZMA",
        "raw_bytes": raw_size,
        "compressed_bytes": len(compressed),
        "ratio": raw_size / len(compressed),
    }


def test_per_weight_lzma(weights, max_val):
    """LZMA compress each weight individually and sum."""
    print("\n=== Per-Weight LZMA ===", flush=True)
    start_time = time.time()
    
    total_compressed = 0
    total_ranges = 0
    
    for i, w in enumerate(weights):
        if (i + 1) % 200 == 0:
            print(f"  Processing {i+1}/{len(weights)}...", flush=True)
        
        data = b''
        for start, end in w:
            data += start.to_bytes(4, 'little')
            data += end.to_bytes(4, 'little')
        
        if data:
            compressed = lzma.compress(data)
            total_compressed += len(compressed)
        total_ranges += len(w)
    
    elapsed = time.time() - start_time
    
    raw_size = total_ranges * 2 * 4
    
    print(f"  Total weights: {len(weights)}", flush=True)
    print(f"  Total ranges: {total_ranges}", flush=True)
    print(f"  Raw size (bytes): {raw_size}", flush=True)
    print(f"  Per-weight LZMA (bytes): {total_compressed}", flush=True)
    print(f"  Compression ratio: {raw_size / total_compressed:.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    return {
        "method": "Per-Weight LZMA",
        "raw_bytes": raw_size,
        "compressed_bytes": total_compressed,
        "ratio": raw_size / total_compressed if total_compressed > 0 else 0,
    }


def test_grouped_lzma(weights, max_val):
    """Group weights by size, compress each group with LZMA."""
    print("\n=== Grouped LZMA (by range count) ===", flush=True)
    start_time = time.time()
    
    # Group by range count
    groups = {}
    for w in weights:
        key = len(w)
        if key not in groups:
            groups[key] = []
        groups[key].append(w)
    
    total_compressed = 0
    total_ranges = 0
    
    for key, group in groups.items():
        data = b''
        for w in group:
            for start, end in w:
                data += start.to_bytes(4, 'little')
                data += end.to_bytes(4, 'little')
            total_ranges += len(w)
        
        if data:
            compressed = lzma.compress(data)
            total_compressed += len(compressed)
    
    elapsed = time.time() - start_time
    
    raw_size = total_ranges * 2 * 4
    
    print(f"  Groups: {len(groups)}", flush=True)
    print(f"  Raw size (bytes): {raw_size}", flush=True)
    print(f"  Grouped LZMA (bytes): {total_compressed}", flush=True)
    print(f"  Compression ratio: {raw_size / total_compressed:.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    return {
        "method": "Grouped LZMA",
        "raw_bytes": raw_size,
        "compressed_bytes": total_compressed,
        "ratio": raw_size / total_compressed if total_compressed > 0 else 0,
    }


def test_sorted_all_lzma(weights, max_val):
    """Sort all ranges globally and compress with LZMA."""
    print("\n=== Sorted Global LZMA ===", flush=True)
    start_time = time.time()
    
    all_ranges = []
    for w in weights:
        all_ranges.extend(w)
    all_ranges.sort()
    
    data = b''
    for start, end in all_ranges:
        data += start.to_bytes(4, 'little')
        data += end.to_bytes(4, 'little')
    
    compressed = lzma.compress(data)
    
    elapsed = time.time() - start_time
    
    raw_size = len(data)
    
    print(f"  Total ranges: {len(all_ranges)}", flush=True)
    print(f"  Raw size (bytes): {raw_size}", flush=True)
    print(f"  Sorted LZMA (bytes): {len(compressed)}", flush=True)
    print(f"  Compression ratio: {raw_size / len(compressed):.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    return {
        "method": "Sorted Global LZMA",
        "raw_bytes": raw_size,
        "compressed_bytes": len(compressed),
        "ratio": raw_size / len(compressed),
    }


def main():
    results = {}
    
    for filename in ["range_weights_terminal_dwa.json", "range_weights_parser_dwa.json"]:
        print(f"\n{'='*60}", flush=True)
        print(f"FINAL BENCHMARK: {filename}", flush=True)
        print(f"{'='*60}", flush=True)
        
        weights, max_val = load_weights(filename)
        if weights is None:
            continue
        
        file_results = []
        
        r = test_delta_lzma(weights, max_val)
        if r: file_results.append(r)
        
        r = test_sorted_all_lzma(weights, max_val)
        if r: file_results.append(r)
        
        r = test_grouped_lzma(weights, max_val)
        if r: file_results.append(r)
        
        r = test_per_weight_lzma(weights, max_val)
        if r: file_results.append(r)
        
        results[filename] = file_results
        
        print(f"\n--- Summary for {filename} ---", flush=True)
        print(f"{'Method':<25} {'Raw':>12} {'Compressed':>12} {'Ratio':>8}", flush=True)
        print("-" * 60, flush=True)
        for r in sorted(file_results, key=lambda x: -x['ratio']):
            print(f"{r['method']:<25} {r['raw_bytes']:>12} {r['compressed_bytes']:>12} {r['ratio']:>7.2f}x", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("FINAL RESULTS", flush=True)
    print("="*60, flush=True)
    
    for fname, fr in results.items():
        best = max(fr, key=lambda x: x['ratio'])
        print(f"{fname}:", flush=True)
        print(f"  Best: {best['method']} at {best['ratio']:.2f}x", flush=True)


if __name__ == "__main__":
    main()
