#!/usr/bin/env python3
"""
Bit/Range Reordering to Minimize Total Ranges.

Idea: If we reorder the positions (change the mapping from LLM token IDs to internal IDs),
we might be able to make more ranges adjacent, causing them to merge and reducing total count.

Approach:
1. Load weights
2. Build a graph where nodes are positions and edges connect positions that appear together
3. Find an ordering that groups co-occurring positions together
4. Count ranges after reordering
"""

import json
from collections import defaultdict


def load_weights(filename):
    print(f"--- Loading {filename} ---", flush=True)
    with open(filename, 'r') as f:
        weights = json.load(f)
    
    clipped = []
    for w in weights:
        c = [(s, e) for s, e in w if e < (1 << 32)]
        clipped.append(c)
    
    return clipped


def count_ranges(weights):
    """Count total ranges across all weights."""
    return sum(len(w) for w in weights)


def apply_permutation(weights, perm):
    """
    Apply a permutation to all weights.
    perm[old_id] = new_id
    
    For each range [start, end], map both endpoints.
    Then re-merge into sorted ranges.
    """
    new_weights = []
    for w in weights:
        # Expand to individual positions, apply permutation, then re-range
        positions = set()
        for start, end in w:
            # For small ranges, expand
            if end - start < 10000:
                for p in range(start, end + 1):
                    if p in perm:
                        positions.add(perm[p])
            else:
                # For large ranges, just map endpoints (approximation)
                if start in perm:
                    positions.add(perm[start])
                if end in perm:
                    positions.add(perm[end])
        
        # Convert back to ranges
        if not positions:
            new_weights.append([])
            continue
            
        sorted_pos = sorted(positions)
        ranges = []
        range_start = sorted_pos[0]
        range_end = sorted_pos[0]
        
        for p in sorted_pos[1:]:
            if p == range_end + 1:
                range_end = p
            else:
                ranges.append((range_start, range_end))
                range_start = p
                range_end = p
        ranges.append((range_start, range_end))
        
        new_weights.append(ranges)
    
    return new_weights


def greedy_ordering(weights, max_positions=100000):
    """
    Build a greedy ordering that groups co-occurring positions.
    
    Heuristic: 
    - Start with position that appears most frequently
    - Add positions that co-occur most frequently with already-added positions
    """
    print("Building position co-occurrence graph...", flush=True)
    
    # Collect all positions
    all_positions = set()
    for w in weights:
        for start, end in w:
            # Limit enumeration
            if end - start < 1000:
                for p in range(start, end + 1):
                    all_positions.add(p)
                    if len(all_positions) > max_positions:
                        break
            else:
                all_positions.add(start)
                all_positions.add(end)
            if len(all_positions) > max_positions:
                break
        if len(all_positions) > max_positions:
            break
    
    print(f"  Positions: {len(all_positions)}", flush=True)
    
    if len(all_positions) > 50000:
        print("  Too many positions for full analysis, sampling...", flush=True)
        # Just use identity permutation for large sets
        all_positions = sorted(all_positions)[:50000]
    
    # Build co-occurrence: positions that appear in same weight
    cooccur = defaultdict(lambda: defaultdict(int))
    freq = defaultdict(int)
    
    for w in weights:
        w_positions = set()
        for start, end in w:
            if end - start < 100:  # Only small ranges
                for p in range(start, end + 1):
                    if p in all_positions:
                        w_positions.add(p)
        
        for p in w_positions:
            freq[p] += 1
            for q in w_positions:
                if p != q:
                    cooccur[p][q] += 1
    
    print(f"  Building greedy ordering...", flush=True)
    
    # Start with most frequent position
    if not freq:
        return {p: p for p in sorted(all_positions)}
    
    ordered = []
    remaining = set(all_positions)
    
    # Start with most frequent
    start_pos = max(freq.keys(), key=lambda p: freq[p])
    ordered.append(start_pos)
    remaining.discard(start_pos)
    
    # Greedily add positions that co-occur most with already-added
    while remaining and len(ordered) < len(all_positions):
        best_next = None
        best_score = -1
        
        for p in list(remaining)[:1000]:  # Limit search
            score = sum(cooccur[p].get(q, 0) for q in ordered[-100:])  # Look at recent
            if score > best_score:
                best_score = score
                best_next = p
        
        if best_next is None:
            # No co-occurrence, just add any
            best_next = next(iter(remaining))
        
        ordered.append(best_next)
        remaining.discard(best_next)
        
        if len(ordered) % 5000 == 0:
            print(f"    Ordered {len(ordered)} / {len(all_positions)}...", flush=True)
    
    # Add any remaining
    ordered.extend(sorted(remaining))
    
    # Build permutation: old -> new
    perm = {old: new for new, old in enumerate(ordered)}
    
    return perm


def analyze_reordering(filename):
    weights = load_weights(filename)
    
    original_ranges = count_ranges(weights)
    print(f"\n=== Reordering Analysis: {filename} ===", flush=True)
    print(f"  Original total ranges: {original_ranges}", flush=True)
    
    # Try greedy ordering
    perm = greedy_ordering(weights)
    
    if not perm:
        print("  Could not build permutation.", flush=True)
        return
    
    print("  Applying permutation...", flush=True)
    reordered = apply_permutation(weights, perm)
    new_ranges = count_ranges(reordered)
    
    print(f"  Reordered total ranges: {new_ranges}", flush=True)
    print(f"  Reduction: {original_ranges / new_ranges:.2f}x", flush=True)


if __name__ == "__main__":
    analyze_reordering("range_weights_terminal_dwa.json")
