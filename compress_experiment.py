#!/usr/bin/env python3
"""
Weight Compression Experiments for HEAVY mode DWA weights.
Anything goes - exploring various compression strategies.
"""

import json
import sys
from collections import defaultdict, Counter
from typing import List, Tuple, Set, FrozenSet
import heapq

# Type aliases
Range = Tuple[int, int]  # (start, end) inclusive
Weight = List[Range]

def load_weights(filename: str) -> List[Weight]:
    with open(filename, 'r') as f:
        data = json.load(f)
    return [[(s, e) for s, e in w] for w in data]

def weight_cost(w: Weight) -> int:
    """Cost of storing a weight = number of integers needed (2 per range)."""
    return len(w) * 2

def total_cost(weights: List[Weight]) -> int:
    return sum(weight_cost(w) for w in weights)

# =============================================================================
# STRATEGY 1: Range Deduplication with Reference Count
# =============================================================================
def strategy_range_dedup(weights: List[Weight]) -> dict:
    """
    Basic deduplication: store each unique range once, reference by ID.
    Cost = unique_ranges * 2 + total_references
    """
    all_ranges = []
    for w in weights:
        all_ranges.extend(w)
    
    unique_ranges = set(all_ranges)
    range_counts = Counter(all_ranges)
    
    # Cost model: dictionary has unique_ranges * 2 ints, each weight stores IDs
    dict_cost = len(unique_ranges) * 2
    ref_cost = len(all_ranges)  # Each range ref is 1 ID
    total = dict_cost + ref_cost
    
    original = total_cost(weights)
    
    # Stats on sharing
    multi_use = sum(1 for r, c in range_counts.items() if c > 1)
    max_reuse = max(range_counts.values())
    
    return {
        "name": "Range Deduplication",
        "original_ints": original,
        "compressed_ints": total,
        "ratio": total / original,
        "unique_ranges": len(unique_ranges),
        "multi_use_ranges": multi_use,
        "max_reuse": max_reuse,
    }

# =============================================================================
# STRATEGY 2: Weight Deduplication (full weight reuse)
# =============================================================================
def strategy_weight_dedup(weights: List[Weight]) -> dict:
    """
    Store each unique weight once, reference by ID.
    """
    weight_tuples = [tuple(w) for w in weights]
    unique_weights = set(weight_tuples)
    weight_counts = Counter(weight_tuples)
    
    # Cost: dictionary has sum of weight costs, each usage is 1 ID
    dict_cost = sum(weight_cost(list(w)) for w in unique_weights)
    ref_cost = len(weights)  # One ID per weight reference
    total = dict_cost + ref_cost
    
    original = total_cost(weights)
    
    multi_use = sum(1 for w, c in weight_counts.items() if c > 1)
    
    return {
        "name": "Weight Deduplication",
        "original_ints": original,
        "compressed_ints": total,
        "ratio": total / original,
        "unique_weights": len(unique_weights),
        "multi_use_weights": multi_use,
    }

# =============================================================================
# STRATEGY 3: Complement Encoding for Dense Weights
# =============================================================================
def strategy_complement_encoding(weights: List[Weight], universe_max: int = 0xFFFF) -> dict:
    """
    For weights that are 'almost full', store the complement instead.
    A weight covering most of [0, universe_max] can be stored as NOT(small_set).
    """
    original = total_cost(weights)
    compressed = 0
    complement_used = 0
    
    for w in weights:
        if not w:
            compressed += 0
            continue
            
        # Compute coverage
        covered = sum(e - s + 1 for s, e in w)
        total_gaps = universe_max + 1 - covered
        
        # Cost of storing complement (gaps)
        # Gaps are: before first range, between ranges, after last range
        gaps = []
        prev_end = -1
        for s, e in sorted(w):
            if s > prev_end + 1:
                gaps.append((prev_end + 1, s - 1))
            prev_end = e
        if prev_end < universe_max:
            gaps.append((prev_end + 1, universe_max))
        
        complement_cost = len(gaps) * 2 + 1  # +1 for "negate" flag
        direct_cost = len(w) * 2
        
        if complement_cost < direct_cost:
            compressed += complement_cost
            complement_used += 1
        else:
            compressed += direct_cost
    
    return {
        "name": "Complement Encoding",
        "original_ints": original,
        "compressed_ints": compressed,
        "ratio": compressed / original if original else 0,
        "complement_used": complement_used,
        "total_weights": len(weights),
    }

# =============================================================================
# STRATEGY 4: Delta Encoding for Sequential Ranges
# =============================================================================
def strategy_delta_encoding(weights: List[Weight]) -> dict:
    """
    For sorted ranges, store deltas instead of absolute values.
    First range: (start, end)
    Subsequent: (delta_start, width) where delta_start = start - prev_end - 1
    
    This saves bits if ranges are clustered.
    """
    original = total_cost(weights)
    delta_cost = 0
    
    for w in weights:
        if not w:
            continue
        sorted_w = sorted(w)
        
        # First range: absolute
        delta_cost += 2  # start, end
        
        # Rest: delta from previous end
        for i in range(1, len(sorted_w)):
            prev_end = sorted_w[i-1][1]
            curr_start, curr_end = sorted_w[i]
            delta = curr_start - prev_end - 1  # Gap size (can be 0 if adjacent)
            width = curr_end - curr_start      # Range width
            # If deltas are small, they could be varint-encoded
            # For simplicity, count as 2 integers still
            delta_cost += 2
    
    return {
        "name": "Delta Encoding",
        "original_ints": original,
        "compressed_ints": delta_cost,
        "ratio": delta_cost / original if original else 0,
        "note": "Same int count but deltas may have smaller magnitude for varint",
    }

# =============================================================================
# STRATEGY 5: Bitmap Encoding for Dense Small Ranges
# =============================================================================
def strategy_bitmap_hybrid(weights: List[Weight], bitmap_threshold: int = 64) -> dict:
    """
    For weights where ranges span <= bitmap_threshold, use bitmap.
    Otherwise use range-list.
    
    A 64-bit bitmap can represent 64 consecutive values with 1 u64.
    """
    original = total_cost(weights)
    compressed = 0
    bitmap_used = 0
    
    for w in weights:
        if not w:
            continue
        
        # Span of weight
        min_val = min(s for s, e in w)
        max_val = max(e for s, e in w)
        span = max_val - min_val + 1
        
        if span <= bitmap_threshold:
            # Bitmap: 1 offset + ceil(span/64) u64s ~ ceil(span/64) * 2 + 1 ints
            bitmap_ints = 1 + ((span + 63) // 64) * 2  # Offset + bitmap words
            range_ints = len(w) * 2
            
            if bitmap_ints < range_ints:
                compressed += bitmap_ints
                bitmap_used += 1
            else:
                compressed += range_ints
        else:
            compressed += len(w) * 2
    
    return {
        "name": "Bitmap Hybrid",
        "original_ints": original,
        "compressed_ints": compressed,
        "ratio": compressed / original if original else 0,
        "bitmap_used": bitmap_used,
        "total_weights": len(weights),
    }

# =============================================================================
# STRATEGY 6: Prefix-Free Range Tree (Trie-like)
# =============================================================================
def strategy_range_trie(weights: List[Weight]) -> dict:
    """
    Build a trie of range start values, collapsing common prefixes.
    Experimental - may not compress well.
    """
    # Collect all (start, end) pairs and count
    all_pairs = []
    for w in weights:
        all_pairs.extend(w)
    
    # Group by start
    by_start = defaultdict(list)
    for s, e in all_pairs:
        by_start[s].append(e)
    
    # For each unique start, store: start + [ends]
    # With dedup: start + unique_ends
    trie_cost = 0
    for start, ends in by_start.items():
        unique_ends = set(ends)
        # 1 int for start + 1 int per unique end
        trie_cost += 1 + len(unique_ends)
    
    # Reference cost: each original range becomes 1 reference (index into trie)
    ref_cost = len(all_pairs)
    total = trie_cost + ref_cost
    
    original = total_cost(weights)
    
    return {
        "name": "Range Trie (by start)",
        "original_ints": original,
        "compressed_ints": total,
        "ratio": total / original if original else 0,
        "unique_starts": len(by_start),
    }

# =============================================================================
# STRATEGY 7: Frequent Subweight Mining (Re-Pair style)
# =============================================================================
def strategy_frequent_subweights(weights: List[Weight], max_iters: int = 500) -> dict:
    """
    Find frequently occurring sub-sets of ranges across weights.
    Replace with dictionary entries.
    """
    # Convert weights to frozensets for subset operations
    weight_sets = [frozenset(w) for w in weights]
    
    # Count all pairs of ranges that co-occur
    pair_counts = Counter()
    for ws in weight_sets:
        if len(ws) < 2:
            continue
        ranges = list(ws)
        for i in range(len(ranges)):
            for j in range(i+1, len(ranges)):
                pair = frozenset([ranges[i], ranges[j]])
                pair_counts[pair] += 1
    
    if not pair_counts:
        return {
            "name": "Frequent Subweights (Pair Mining)",
            "original_ints": total_cost(weights),
            "compressed_ints": total_cost(weights),
            "ratio": 1.0,
            "note": "No pairs found",
        }
    
    # Iteratively merge most frequent pairs
    dictionary = {}  # id -> frozenset of ranges
    next_id = 0
    current_weights = [set(w) for w in weights]
    
    for iteration in range(max_iters):
        # Find most frequent pair
        pair_counts = Counter()
        for ws in current_weights:
            if len(ws) < 2:
                continue
            items = list(ws)
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    a, b = items[i], items[j]
                    pair = (min(a, b, key=str), max(a, b, key=str))
                    pair_counts[pair] += 1
        
        if not pair_counts:
            break
        
        best_pair, count = pair_counts.most_common(1)[0]
        if count < 2:
            break
        
        # Create dictionary entry
        dict_entry = frozenset(best_pair)
        dictionary[next_id] = dict_entry
        
        # Replace in all weights
        for ws in current_weights:
            if best_pair[0] in ws and best_pair[1] in ws:
                ws.discard(best_pair[0])
                ws.discard(best_pair[1])
                ws.add(("$", next_id))  # Marker for dict ref
        
        next_id += 1
    
    # Compute final cost
    # Dictionary: each entry is 2 ranges * 2 ints = 4 ints (for original pairs)
    # But dict entries could themselves contain refs... simplified: count original ranges
    dict_cost = 0
    for did, entry in dictionary.items():
        for item in entry:
            if isinstance(item, tuple) and item[0] == "$":
                dict_cost += 1  # Reference
            else:
                dict_cost += 2  # Range
    
    # Weight cost: each remaining item
    weight_cost_total = 0
    for ws in current_weights:
        for item in ws:
            if isinstance(item, tuple) and item[0] == "$":
                weight_cost_total += 1  # Reference
            else:
                weight_cost_total += 2  # Range
    
    compressed = dict_cost + weight_cost_total
    original = total_cost(weights)
    
    return {
        "name": "Frequent Subweights (Pair Mining)",
        "original_ints": original,
        "compressed_ints": compressed,
        "ratio": compressed / original if original else 0,
        "dictionary_entries": len(dictionary),
        "iterations": min(iteration + 1, max_iters),
    }

# =============================================================================
# MAIN
# =============================================================================
def run_all_strategies(filename: str):
    print(f"\n{'='*60}")
    print(f"Compression Experiments: {filename}")
    print(f"{'='*60}")
    
    weights = load_weights(filename)
    print(f"Loaded {len(weights)} weights")
    
    original = total_cost(weights)
    print(f"Original cost: {original} integers ({original * 4 / 1024:.1f} KB @ 4 bytes/int)\n")
    
    strategies = [
        strategy_range_dedup,
        strategy_weight_dedup,
        strategy_complement_encoding,
        strategy_delta_encoding,
        strategy_bitmap_hybrid,
        strategy_range_trie,
        strategy_frequent_subweights,
    ]
    
    results = []
    for strategy in strategies:
        try:
            result = strategy(weights)
            results.append(result)
            print(f"[{result['name']}]")
            print(f"  Compressed: {result['compressed_ints']} ints")
            print(f"  Ratio: {result['ratio']:.3f}")
            for k, v in result.items():
                if k not in ['name', 'original_ints', 'compressed_ints', 'ratio']:
                    print(f"  {k}: {v}")
            print()
        except Exception as e:
            print(f"[{strategy.__name__}] ERROR: {e}\n")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY (sorted by compression ratio)")
    print("="*60)
    for r in sorted(results, key=lambda x: x['ratio']):
        print(f"  {r['ratio']:.3f}  {r['name']}")

if __name__ == "__main__":
    # Run on HEAVY mode weights
    for fname in ["range_weights_terminal_dwa_HEAVY.json", "range_weights_parser_dwa_HEAVY.json"]:
        try:
            run_all_strategies(fname)
        except FileNotFoundError:
            print(f"File not found: {fname}")
