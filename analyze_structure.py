#!/usr/bin/env python3
"""
Analyze the structure of weight ranges to find compression opportunities.
"""

import json
from collections import defaultdict


def analyze_structure(filename):
    print(f"\n=== Structure Analysis: {filename} ===", flush=True)
    
    with open(filename, 'r') as f:
        weights = json.load(f)
    
    # Categorize ranges
    point_ranges = 0  # [x, x]
    small_ranges = 0  # span <= 10
    medium_ranges = 0  # span <= 1000
    large_ranges = 0  # span <= 100000
    huge_ranges = 0  # span > 100000
    
    total_ranges = 0
    
    point_weights = []  # weights with only point ranges
    mixed_weights = []
    dense_weights = []  # weights with mostly large ranges
    
    range_spans = []
    
    for w_idx, w in enumerate(weights):
        w_point = 0
        w_large = 0
        
        for start, end in w:
            if end > (1 << 32):
                continue  # Skip sentinels
            
            span = end - start
            range_spans.append(span)
            total_ranges += 1
            
            if span == 0:
                point_ranges += 1
                w_point += 1
            elif span <= 10:
                small_ranges += 1
            elif span <= 1000:
                medium_ranges += 1
            elif span <= 100000:
                large_ranges += 1
            else:
                huge_ranges += 1
                w_large += 1
        
        if len(w) > 0:
            if w_point == len(w):
                point_weights.append(w_idx)
            elif w_large > len(w) / 2:
                dense_weights.append(w_idx)
            else:
                mixed_weights.append(w_idx)
    
    print(f"  Total weights: {len(weights)}", flush=True)
    print(f"  Total ranges: {total_ranges}", flush=True)
    
    print(f"\n  Range type distribution:", flush=True)
    print(f"    Point ranges [x,x]: {point_ranges} ({100*point_ranges/total_ranges:.1f}%)", flush=True)
    print(f"    Small (span<=10): {small_ranges} ({100*small_ranges/total_ranges:.1f}%)", flush=True)
    print(f"    Medium (span<=1K): {medium_ranges} ({100*medium_ranges/total_ranges:.1f}%)", flush=True)
    print(f"    Large (span<=100K): {large_ranges} ({100*large_ranges/total_ranges:.1f}%)", flush=True)
    print(f"    Huge (span>100K): {huge_ranges} ({100*huge_ranges/total_ranges:.1f}%)", flush=True)
    
    print(f"\n  Weight type distribution:", flush=True)
    print(f"    Pure point-set weights: {len(point_weights)}", flush=True)
    print(f"    Dense (mostly huge) weights: {len(dense_weights)}", flush=True)
    print(f"    Mixed weights: {len(mixed_weights)}", flush=True)
    
    # Check for common ranges across weights
    range_counts = defaultdict(int)
    for w in weights:
        for r in w:
            if r[1] < (1 << 32):
                range_counts[tuple(r)] += 1
    
    # Distribution of range reuse
    reuse_dist = defaultdict(int)
    for count in range_counts.values():
        reuse_dist[count] += 1
    
    print(f"\n  Range reuse distribution:", flush=True)
    for count in sorted(reuse_dist.keys())[:10]:
        print(f"    Appears in {count} weight(s): {reuse_dist[count]} unique ranges", flush=True)
    
    # Highly reused ranges
    highly_reused = [(r, c) for r, c in range_counts.items() if c >= 10]
    highly_reused.sort(key=lambda x: -x[1])
    
    print(f"\n  Top 10 most reused ranges:", flush=True)
    for r, c in highly_reused[:10]:
        print(f"    {r}: appears in {c} weights", flush=True)


if __name__ == "__main__":
    for fname in ["range_weights_terminal_dwa.json", "range_weights_parser_dwa.json"]:
        analyze_structure(fname)
