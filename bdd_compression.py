#!/usr/bin/env python3
"""
BDD Compression Analysis for DWA Weights.

Uses the `dd` package to encode weight range-sets as Binary Decision Diagrams (BDDs)
and measure the compression achieved vs. raw representation.

OPTIMIZED: Uses dd's expression parsing for faster interval encoding.
"""

import json
import sys
from dd.autoref import BDD

def make_interval_expr(lo, hi, num_bits):
    """
    Create a boolean expression string for the interval [lo, hi].
    Uses comparison encoding: lo <= x <= hi
    
    For efficiency, we encode intervals directly as disjunctions of 
    "prefix patterns" where possible.
    """
    # Clip values to num_bits
    max_val = (1 << num_bits) - 1
    lo = max(0, min(lo, max_val))
    hi = max(0, min(hi, max_val))
    
    if lo > hi:
        return "FALSE"
    if lo == 0 and hi == max_val:
        return "TRUE"
    
    # Simple case: single value
    if lo == hi:
        bits = []
        for i in range(num_bits):
            if (lo >> i) & 1:
                bits.append(f"x{i}")
            else:
                bits.append(f"!x{i}")
        return " & ".join(bits)
    
    # For larger intervals, we use a recursive split approach
    # but generate the expression string instead of BDD operations
    return interval_expr_recursive(lo, hi, num_bits)


def interval_expr_recursive(lo, hi, num_bits):
    """
    Recursively build expression for interval [lo, hi].
    """
    max_val = (1 << num_bits) - 1
    
    if lo > hi:
        return "FALSE"
    if lo == 0 and hi >= max_val:
        return "TRUE"
    if lo == hi:
        bits = []
        for i in range(num_bits):
            if (lo >> i) & 1:
                bits.append(f"x{i}")
            else:
                bits.append(f"!x{i}")
        return "(" + " & ".join(bits) + ")"
    
    # Find highest bit
    top_bit = num_bits - 1
    mid = 1 << top_bit
    
    if hi < mid:
        # Both in lower half: top bit = 0
        return f"(!x{top_bit} & {interval_expr_recursive(lo, hi, num_bits - 1)})"
    elif lo >= mid:
        # Both in upper half: top bit = 1
        return f"(x{top_bit} & {interval_expr_recursive(lo - mid, hi - mid, num_bits - 1)})"
    else:
        # Spans both halves
        lower_part = interval_expr_recursive(lo, mid - 1, num_bits - 1) if lo < mid else "FALSE"
        upper_part = interval_expr_recursive(0, hi - mid, num_bits - 1) if hi >= mid else "FALSE"
        
        if lower_part == "FALSE":
            return f"(x{top_bit} & {upper_part})"
        if upper_part == "FALSE":
            return f"(!x{top_bit} & {lower_part})"
        
        return f"((!x{top_bit} & {lower_part}) | (x{top_bit} & {upper_part}))"


def analyze_weights_bdd(filename, num_bits=24):
    """
    Load weights from JSON, encode as BDDs, and measure compression.
    """
    print(f"--- BDD Compression Analysis: {filename} ---", flush=True)
    
    try:
        with open(filename, 'r') as f:
            weights = json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found.", flush=True)
        return
    
    print(f"Loaded {len(weights)} weights.", flush=True)
    
    # Find the maximum value to determine required bit width
    max_val = 0
    for w in weights:
        for start, end in w:
            # Clip to reasonable range (ignore u64 max sentinels)
            if end < (1 << 32):  # Ignore obvious sentinel values
                max_val = max(max_val, end)
    
    required_bits = max(max_val.bit_length(), 1) if max_val > 0 else 1
    print(f"Max value (ignoring sentinels): {max_val}, Required bits: {required_bits}", flush=True)
    
    # Use the required bits, but cap at num_bits
    actual_bits = min(required_bits, num_bits)
    print(f"Using {actual_bits} bits for encoding.", flush=True)
    
    bdd = BDD()
    
    # Declare variables (bitvector bits)
    print(f"Declaring {actual_bits} BDD variables...", flush=True)
    for i in range(actual_bits):
        bdd.declare(f"x{i}")
    
    # Build BDDs for each weight
    weight_bdds = []
    total_ranges = 0
    skipped_ranges = 0
    
    print(f"Building BDDs for {len(weights)} weights...", flush=True)
    
    max_clip = (1 << actual_bits) - 1
    
    for i, w in enumerate(weights):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Weight {i + 1}/{len(weights)}: {len(w)} ranges, BDD nodes: {len(bdd)}...", flush=True)
        
        # Build expression for this weight (union of ranges)
        range_exprs = []
        for start, end in w:
            # Skip sentinel values
            if start > max_clip or end > (1 << 32):
                skipped_ranges += 1
                continue
            
            total_ranges += 1
            expr = make_interval_expr(start, min(end, max_clip), actual_bits)
            if expr not in ("FALSE", "TRUE"):
                range_exprs.append(expr)
            elif expr == "TRUE":
                range_exprs = ["TRUE"]
                break
        
        if not range_exprs:
            w_bdd = bdd.false
        elif "TRUE" in range_exprs:
            w_bdd = bdd.true
        else:
            # Parse and OR all expressions
            combined_expr = " | ".join(range_exprs)
            try:
                w_bdd = bdd.add_expr(combined_expr)
            except Exception as e:
                print(f"  Error parsing weight {i}: {e}", flush=True)
                w_bdd = bdd.false
        
        weight_bdds.append(w_bdd)
    
    # Measure BDD size
    total_bdd_nodes = len(bdd)
    
    # Raw size: each range = 2 integers (start, end)
    raw_size = total_ranges * 2
    
    print(f"\nResults:", flush=True)
    print(f"  Unique Weights: {len(weights)}", flush=True)
    print(f"  Total Ranges (used): {total_ranges}", flush=True)
    print(f"  Skipped Ranges (sentinels): {skipped_ranges}", flush=True)
    print(f"  Raw Size (range endpoints): {raw_size}", flush=True)
    print(f"  BDD Nodes (shared): {total_bdd_nodes}", flush=True)
    if total_bdd_nodes > 0:
        print(f"  Compression Ratio: {raw_size / total_bdd_nodes:.2f}x", flush=True)
    
    return {
        "unique_weights": len(weights),
        "total_ranges": total_ranges,
        "raw_size": raw_size,
        "bdd_nodes": total_bdd_nodes,
    }


if __name__ == "__main__":
    results = {}
    
    for filename in ["range_weights_terminal_dwa.json", "range_weights_parser_dwa.json"]:
        result = analyze_weights_bdd(filename)
        if result:
            results[filename] = result
        print("", flush=True)
    
    print("--- Summary ---", flush=True)
    for filename, r in results.items():
        if r['bdd_nodes'] > 0:
            print(f"{filename}: {r['bdd_nodes']} BDD nodes vs {r['raw_size']} raw endpoints ({r['raw_size'] / r['bdd_nodes']:.2f}x compression)", flush=True)
        else:
            print(f"{filename}: No data", flush=True)
