#!/usr/bin/env python3
"""
Focused compression experiments - faster algorithms.
"""

import json
from collections import Counter
from typing import List, Tuple

Range = Tuple[int, int]
Weight = List[Range]

def load_weights(filename: str) -> List[Weight]:
    with open(filename, 'r') as f:
        data = json.load(f)
    return [[(s, e) for s, e in w] for w in data]

def total_cost(weights: List[Weight]) -> int:
    return sum(len(w) * 2 for w in weights)

# =============================================================================
# STRATEGY: Varint Potential (measure delta magnitudes)
# =============================================================================
def strategy_varint_potential(weights: List[Weight]) -> dict:
    """
    Measure how much smaller the actual values are vs worst case.
    Varints save space when values are small.
    """
    values = []
    deltas = []
    
    for w in weights:
        if not w:
            continue
        sorted_w = sorted(w)
        
        # First range: absolute
        s, e = sorted_w[0]
        values.extend([s, e])
        
        # Rest: compute deltas
        for i in range(1, len(sorted_w)):
            prev_end = sorted_w[i-1][1]
            curr_start, curr_end = sorted_w[i]
            gap = curr_start - prev_end - 1
            width = curr_end - curr_start
            deltas.extend([gap, width])
            values.extend([curr_start, curr_end])
    
    # Varint savings: log2(value) bits vs 32 bits
    import math
    
    def varint_bytes(v):
        if v < 0: v = -v  # Handle negatives
        if v == 0: return 1
        return (v.bit_length() + 6) // 7  # 7 bits per byte
    
    absolute_bytes = len(values) * 4  # 32-bit ints
    varint_abs_bytes = sum(varint_bytes(v) for v in values)
    varint_delta_bytes = sum(varint_bytes(d) for d in deltas) + sum(varint_bytes(v) for v in values[:2*len(weights)])
    
    return {
        "name": "Varint Potential",
        "absolute_bytes": absolute_bytes,
        "varint_absolute_bytes": varint_abs_bytes,
        "varint_delta_bytes": varint_delta_bytes,
        "ratio_abs": varint_abs_bytes / absolute_bytes,
        "ratio_delta": varint_delta_bytes / absolute_bytes if absolute_bytes else 0,
        "avg_absolute_value": sum(values) / len(values) if values else 0,
        "avg_delta": sum(deltas) / len(deltas) if deltas else 0,
    }

# =============================================================================
# STRATEGY: Range Clustering (group by start // bucket_size)
# =============================================================================
def strategy_range_clustering(weights: List[Weight], bucket_size: int = 256) -> dict:
    """
    Group ranges by start // bucket_size.
    Within a bucket, store offset from bucket base.
    """
    all_ranges = []
    for w in weights:
        all_ranges.extend(w)
    
    buckets = {}
    for s, e in all_ranges:
        bucket_id = s // bucket_size
        if bucket_id not in buckets:
            buckets[bucket_id] = []
        offset = s % bucket_size
        buckets[bucket_id].append((offset, e))
    
    # Cost: 1 int per bucket header + ranges within
    # If offset fits in 8 bits and end in 24 bits, we could pack
    # For now, count as 2 ints per range + 1 per bucket
    bucket_cost = len(buckets)  # Headers
    range_cost = len(all_ranges) * 2  # Still need start offset + end
    total = bucket_cost + range_cost
    
    original = len(all_ranges) * 2
    
    return {
        "name": f"Range Clustering (bucket={bucket_size})",
        "original_ints": original,
        "compressed_ints": total,
        "ratio": total / original if original else 0,
        "num_buckets": len(buckets),
    }

# =============================================================================
# STRATEGY: Weight Hash Dedup with Content ID
# =============================================================================
def strategy_content_addressing(weights: List[Weight]) -> dict:
    """
    Hash weights by content, store unique ones once.
    Each weight reference is just a hash/ID.
    """
    weight_tuples = [tuple(sorted(w)) for w in weights]
    unique = set(weight_tuples)
    
    # Dictionary cost
    dict_cost = sum(len(w) * 2 for w in unique)
    # Reference cost: one ID per weight usage
    ref_cost = len(weights)
    
    total = dict_cost + ref_cost
    original = total_cost(weights)
    
    # Sharing analysis
    counts = Counter(weight_tuples)
    shared = sum(1 for c in counts.values() if c > 1)
    max_share = max(counts.values())
    
    return {
        "name": "Content Addressing (Weight Dedup)",
        "original_ints": original,
        "compressed_ints": total,
        "ratio": total / original if original else 0,
        "unique_weights": len(unique),
        "weights_with_sharing": shared,
        "max_sharing": max_share,
    }

# =============================================================================
# STRATEGY: Run-Length on Range Widths
# =============================================================================
def strategy_width_rle(weights: List[Weight]) -> dict:
    """
    Many ranges might have same width (e.g. all single-chars).
    RLE encode widths separately from starts.
    """
    starts = []
    widths = []
    
    for w in weights:
        for s, e in w:
            starts.append(s)
            widths.append(e - s)
    
    # RLE on widths
    width_counts = Counter(widths)
    unique_widths = len(width_counts)
    
    # If most widths are same, RLE helps
    most_common_width, most_common_count = width_counts.most_common(1)[0]
    
    # Estimate RLE cost: for each run, store (width, count)
    # But we'd need to track positions... skip full RLE, just report potential
    
    return {
        "name": "Width Analysis (RLE Potential)",
        "total_ranges": len(starts),
        "unique_widths": unique_widths,
        "most_common_width": most_common_width,
        "most_common_count": most_common_count,
        "width_0_count": width_counts.get(0, 0),  # Single-point ranges
        "note": f"{most_common_count}/{len(starts)} = {100*most_common_count/len(starts):.1f}% same width",
    }

# =============================================================================
# MAIN
# =============================================================================
def run_strategies(filename: str):
    print(f"\n{'='*60}")
    print(f"Fast Compression Analysis: {filename}")
    print(f"{'='*60}")
    
    weights = load_weights(filename)
    print(f"Loaded {len(weights)} weights")
    
    original = total_cost(weights)
    print(f"Original: {original} ints ({original * 4 / 1024:.1f} KB)\n")
    
    strategies = [
        strategy_content_addressing,
        strategy_varint_potential,
        lambda w: strategy_range_clustering(w, 256),
        lambda w: strategy_range_clustering(w, 1024),
        strategy_width_rle,
    ]
    
    for strategy in strategies:
        try:
            result = strategy(weights)
            print(f"[{result['name']}]")
            for k, v in result.items():
                if k != 'name':
                    if isinstance(v, float):
                        print(f"  {k}: {v:.3f}")
                    else:
                        print(f"  {k}: {v}")
            print()
        except Exception as e:
            print(f"ERROR: {e}\n")

if __name__ == "__main__":
    for fname in ["range_weights_terminal_dwa_HEAVY.json", "range_weights_parser_dwa_HEAVY.json"]:
        try:
            run_strategies(fname)
        except FileNotFoundError:
            print(f"File not found: {fname}")
