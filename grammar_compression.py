#!/usr/bin/env python3
"""
Grammar-Based Compression Analysis using scikit-sequitur.

Approach:
1. Atomize ranges: Decompose all weight ranges into disjoint atomic intervals.
2. Symbolize: Map each weight to a sorted sequence of Atom IDs.
3. Factorize: Use Sequitur to infer a grammar from the sequence of weight symbols.
4. Measure: Size of grammar rules + size of weight definitions (start rule).

This supports fast AND/OR if the grammar structure is preserved and operations
are performed on the non-terminals (a la SLP compression).
"""

import json
import sys
import time
from collections import defaultdict

try:
    from sksequitur import Grammar, Parser
    HAS_SEQUITUR = True
except ImportError:
    HAS_SEQUITUR = False
    print("scikit-sequitur not found. Please pip install scikit-sequitur", flush=True)


def load_weights(filename):
    print(f"--- Loading {filename} ---", flush=True)
    try:
        with open(filename, 'r') as f:
            weights = json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found.", flush=True)
        return None, 0
    
    max_val = 0
    clipped_weights = []
    for w in weights:
        clipped = []
        for start, end in w:
            if end < (1 << 32):
                max_val = max(max_val, end)
                clipped.append((start, end))
        clipped_weights.append(clipped)
    
    return clipped_weights, max_val


def compute_atoms(weights):
    """
    Decompose all ranges into disjoint atomic intervals.
    Returns:
      atoms: List of (start, end) tuples
      weight_atoms: List of lists of atom indices
    """
    print("Computing atoms...", flush=True)
    points = set()
    for w in weights:
        for start, end in w:
            points.add(start)
            points.add(start)       # ensure start is a split point
            points.add(end + 1)     # ensure end+1 is a split point
    
    sorted_points = sorted(points)
    
    atoms = []
    atom_map = {}  # (start, end) -> idx
    
    for i in range(len(sorted_points) - 1):
        start, next_start = sorted_points[i], sorted_points[i+1]
        if start < next_start:
            atom = (start, next_start - 1)
            atom_map[atom] = len(atoms)
            atoms.append(atom)
            
    num_atoms = len(atoms)
    print(f"  Generated {num_atoms} atoms.", flush=True)
    
    # Map each weight to a sequence of atom IDs
    weight_atom_seqs = []
    for w in weights:
        seq = []
        # Sort ranges to ensure atom sequence is sorted
        w.sort()
        for start, end in w:
            # Find atoms covered by this range
            # Since atoms are disjoint and cover the space, we can iterate/search
            # Optimization: could be O(log N) or O(1) with better structures
            # For analysis, simplistic iteration over relevant atoms is fine
            # Actually, `start` MUST be in `sorted_points` and `end+1` MUST be in `sorted_points`
            # So the range [start, end] corresponds to a contiguous block of atoms.
            
            # Find index of start in sorted_points
            # using binary search would be faster, but let's assume we can map directly
            # A range [s, e] covers atoms starting at s, s_next, ..., up to e
             pass # Placeholder for actual logic below
             
        # Optimized mapping:
        # Use atom_map is tricky because of the splitting logic.
        # Instead, iterate sorted_points.
        current_seq = []
        for start, end in w:
             # Find start index in sorted_points
             # This is slow O(N) per range. Let's do better.
             pass
        weight_atom_seqs.append(seq)
        
    # Re-implementation of fast mapping
    # 1. Create a map: point -> index in sorted_points
    point_to_idx = {p: i for i, p in enumerate(sorted_points)}
    
    final_seqs = []
    for w in weights:
        seq = []
        for start, end in w:
            if start not in point_to_idx or (end + 1) not in point_to_idx:
                 # Should not happen if points set is correct
                 continue
            
            s_idx = point_to_idx[start]
            e_idx = point_to_idx[end + 1]
            
            # Add all atom indices in range [s_idx, e_idx)
            # Atom i corresponds to interval [sorted_points[i], sorted_points[i+1]-1]
            for i in range(s_idx, e_idx):
                seq.append(i)
        final_seqs.append(seq)
        
    return atoms, final_seqs


def analyze_grammar(filename):
    if not HAS_SEQUITUR:
        return

    weights, max_val = load_weights(filename)
    if not weights:
        return

    start_time = time.time()
    
    atoms, weight_seqs = compute_atoms(weights)
    
    # Symbols need to be hashable/unique. We can use atom indices (integers).
    # However, Sequitur might assume characters. scikit-sequitur handles iterables.
    
    print("Inferring grammar...", flush=True)
    
    # Flatten all sequences into one huge sequence, separated by a separator symbol
    # OR feed them one by one if the library supports incremental
    # scikit-sequitur example: parser = Parser(); parser.feed(iterable)
    
    parser = Parser()
    
    # Use a separator that is not an atom ID. Atom IDs are 0..num_atoms-1
    # Use num_atoms as separator
    separator = len(atoms)
    
    full_sequence = []
    for seq in weight_seqs:
        full_sequence.extend(seq)
        full_sequence.append(separator)
        
    parser.feed(full_sequence)
    grammar = Grammar(parser.tree)
    
    elapsed = time.time() - start_time
    
    # Measure Size
    num_rules = len(grammar)
    num_symbols_in_rules = 0
    
    # Count symbols on RHS of all rules
    for rule in grammar:
        num_symbols_in_rules += len(grammar[rule])
        
    # Each symbol is an ID (rule ID or terminal/atom ID).
    # Assume 4 bytes per symbol for simplicity (u32).
    # Plus overhead for rule structure (say 4 bytes per rule head)
    
    grammar_size = (num_symbols_in_rules * 4) + (num_rules * 4)
    atom_table_size = len(atoms) * 2 * 4  # (start, end) per atom
    
    total_compressed_size = grammar_size + atom_table_size
    
    total_ranges = sum(len(w) for w in weights)
    raw_size = total_ranges * 2 * 4
    
    print(f"\n--- Grammar Results for {filename} ---", flush=True)
    print(f"  Atoms: {len(atoms)}", flush=True)
    print(f"  Grammar Rules: {num_rules}", flush=True)
    print(f"  Total Symbols in Rules: {num_symbols_in_rules}", flush=True)
    print(f"  Atom Table Size: {atom_table_size}", flush=True)
    print(f"  Grammar Structure Size: {grammar_size}", flush=True)
    print(f"  Total Compressed Size: {total_compressed_size}", flush=True)
    print(f"  Raw Size: {raw_size}", flush=True)
    print(f"  Compression Ratio: {raw_size / total_compressed_size:.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    # Analyze rule reuse
    # Ideally, we want high reuse of heavy rules
    print("\n  Sample Rules (Head -> Production):", flush=True)
    # Print top 5 rules by expansion size?
    # Or just first few
    count = 0
    for rule in grammar:
        if count < 5:
            # simple print
            print(f"    {rule} -> {grammar[rule]}", flush=True)
        count += 1
        
    return {
        "ratio": raw_size / total_compressed_size
    }

if __name__ == "__main__":
    for fname in ["range_weights_terminal_dwa.json", "range_weights_parser_dwa.json"]:
        analyze_grammar(fname)
