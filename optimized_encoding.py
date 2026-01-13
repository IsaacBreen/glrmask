#!/usr/bin/env python3
"""
Test optimized range encoding strategies.

Key insight: 90.7% of Terminal DWA ranges are point ranges [x,x].
Current encoding: 8 bytes per range (start, end as u32)
Optimized encoding: 
  - Point ranges: 4 bytes (just the value + 1-bit flag)
  - Span ranges: ~8 bytes (start + length or start + end)
  - Dictionary: Frequently reused ranges get short IDs
"""

import json
from collections import defaultdict


def load_weights(filename):
    with open(filename, 'r') as f:
        weights = json.load(f)
    # Clip sentinels
    clipped = []
    for w in weights:
        c = [(s, e) for s, e in w if e < (1 << 32)]
        clipped.append(c)
    return clipped


def test_point_optimized(filename):
    """
    Encoding scheme:
    - If MSB of first u32 is 0: point range, value in lower 31 bits
    - If MSB is 1: span range, next u32 is end, start in lower 31 bits
    """
    print(f"\n=== Point-Optimized Encoding: {filename} ===", flush=True)
    
    weights = load_weights(filename)
    
    total_raw = 0
    total_optimized = 0
    
    for w in weights:
        for start, end in w:
            total_raw += 8  # 2 x u32
            
            if start == end:
                total_optimized += 4  # Just the value
            else:
                total_optimized += 8  # start + end
    
    print(f"  Raw size: {total_raw} bytes", flush=True)
    print(f"  Optimized size: {total_optimized} bytes", flush=True)
    print(f"  Compression ratio: {total_raw / total_optimized:.2f}x", flush=True)
    
    return total_raw / total_optimized


def test_dictionary_encoding(filename):
    """
    Build a dictionary of frequently reused ranges.
    Ranges appearing >= threshold times get a short ID.
    """
    print(f"\n=== Dictionary Encoding: {filename} ===", flush=True)
    
    weights = load_weights(filename)
    
    # Count range frequencies
    range_counts = defaultdict(int)
    total_ranges = 0
    for w in weights:
        for r in w:
            range_counts[tuple(r)] += 1
            total_ranges += 1
    
    # Sort by frequency
    sorted_ranges = sorted(range_counts.items(), key=lambda x: -x[1])
    
    # Try different thresholds
    for threshold in [2, 3, 5, 10]:
        dict_ranges = [(r, c) for r, c in sorted_ranges if c >= threshold]
        dict_size = len(dict_ranges) * 8  # 8 bytes per dictionary entry
        
        # Calculate references
        ref_bytes = 0
        inline_bytes = 0
        dict_lookup = {r: i for i, (r, c) in enumerate(dict_ranges)}
        
        for w in weights:
            for r in w:
                key = tuple(r)
                if key in dict_lookup:
                    # Reference ID (assume 2 bytes if dict small, else 4)
                    ref_bytes += 2 if len(dict_ranges) < 65536 else 4
                else:
                    inline_bytes += 8
        
        total_size = dict_size + ref_bytes + inline_bytes
        raw_size = total_ranges * 8
        
        print(f"  Threshold >= {threshold}: dict_size={len(dict_ranges)}, "
              f"total={total_size}, ratio={raw_size/total_size:.2f}x", flush=True)


def test_combined(filename):
    """
    Combine point optimization + dictionary encoding.
    """
    print(f"\n=== Combined (Point + Dict): {filename} ===", flush=True)
    
    weights = load_weights(filename)
    
    # Count range frequencies
    range_counts = defaultdict(int)
    total_ranges = 0
    point_count = 0
    for w in weights:
        for start, end in w:
            range_counts[(start, end)] += 1
            total_ranges += 1
            if start == end:
                point_count += 1
    
    raw_size = total_ranges * 8
    
    # Build dictionary of reused ranges (threshold >= 2)
    dict_ranges = [(r, c) for r, c in range_counts.items() if c >= 2]
    dict_lookup = {r: i for i, (r, c) in enumerate(dict_ranges)}
    
    # Dictionary storage: 
    # - Point entries: 4 bytes
    # - Span entries: 8 bytes
    dict_size = 0
    for r, c in dict_ranges:
        if r[0] == r[1]:
            dict_size += 4
        else:
            dict_size += 8
    
    # References
    ref_bytes = 0
    inline_bytes = 0
    
    for w in weights:
        for start, end in w:
            key = (start, end)
            if key in dict_lookup:
                ref_bytes += 2  # Short ID
            else:
                if start == end:
                    inline_bytes += 4
                else:
                    inline_bytes += 8
    
    total_size = dict_size + ref_bytes + inline_bytes
    
    print(f"  Dictionary entries: {len(dict_ranges)}", flush=True)
    print(f"  Dictionary size: {dict_size} bytes", flush=True)
    print(f"  Reference bytes: {ref_bytes}", flush=True)
    print(f"  Inline bytes: {inline_bytes}", flush=True)
    print(f"  Total: {total_size} bytes", flush=True)
    print(f"  Raw size: {raw_size} bytes", flush=True)
    print(f"  Compression ratio: {raw_size / total_size:.2f}x", flush=True)


def test_varint_points(filename):
    """
    Use variable-length encoding for points.
    Many point values might be clusterable.
    """
    print(f"\n=== Varint Point Encoding: {filename} ===", flush=True)
    
    weights = load_weights(filename)
    
    # Collect all point values
    points = []
    spans = []
    for w in weights:
        for start, end in w:
            if start == end:
                points.append(start)
            else:
                spans.append((start, end))
    
    # Sort points and delta encode
    all_points_sorted = sorted(points)
    
    if len(all_points_sorted) > 1:
        deltas = [all_points_sorted[0]]
        for i in range(1, len(all_points_sorted)):
            deltas.append(all_points_sorted[i] - all_points_sorted[i-1])
        
        # Varint encoding: count bytes needed
        varint_bytes = 0
        for d in deltas:
            if d < 128:
                varint_bytes += 1
            elif d < 16384:
                varint_bytes += 2
            elif d < 2097152:
                varint_bytes += 3
            else:
                varint_bytes += 4
        
        raw_point_bytes = len(points) * 4
        span_bytes = len(spans) * 8
        
        print(f"  Point ranges: {len(points)}", flush=True)
        print(f"  Span ranges: {len(spans)}", flush=True)
        print(f"  Raw point bytes: {raw_point_bytes}", flush=True)
        print(f"  Varint delta bytes: {varint_bytes}", flush=True)
        print(f"  Point compression: {raw_point_bytes / varint_bytes:.2f}x", flush=True)
        print(f"  Overall (with spans): raw={raw_point_bytes*2 + span_bytes}, "
              f"opt={varint_bytes + span_bytes}", flush=True)


if __name__ == "__main__":
    for fname in ["range_weights_terminal_dwa.json", "range_weights_parser_dwa.json"]:
        test_point_optimized(fname)
        test_dictionary_encoding(fname)
        test_combined(fname)
        test_varint_points(fname)
