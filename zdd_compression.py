#!/usr/bin/env python3
"""
ZDD Compression Analysis for DWA Weights.

Uses Zero-suppressed Decision Diagrams (ZDDs) which are optimized for sparse sets.
Tests if ZDDs can achieve better compression than BDDs for weight range-sets.

Key insight: ZDDs suppress zeros, making them more compact for sets where
most elements are NOT in the set.
"""

import json
import sys
import time

try:
    from dd.autoref import BDD
    HAS_DD = True
except ImportError:
    HAS_DD = False
    print("dd not found. pip install dd", flush=True)


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


def test_zdd_on_atoms(filename):
    """
    Test ZDD compression using atom-based representation.
    Instead of encoding integers directly, we:
    1. Compute atoms (disjoint intervals)
    2. Represent each weight as a set of atom IDs
    3. Use ZDD to represent the family of sets
    """
    if not HAS_DD:
        return None
    
    weights, max_val = load_weights(filename)
    if not weights:
        return None
    
    print(f"\n=== ZDD on Atoms: {filename} ===", flush=True)
    start_time = time.time()
    
    # Step 1: Compute atoms
    print("  Computing atoms...", flush=True)
    points = set()
    for w in weights:
        for start, end in w:
            points.add(start)
            points.add(end + 1)
    sorted_points = sorted(points)
    
    atoms = []
    for i in range(len(sorted_points) - 1):
        atoms.append((sorted_points[i], sorted_points[i+1] - 1))
    
    num_atoms = len(atoms)
    print(f"  Atoms: {num_atoms}", flush=True)
    
    # Step 2: Map weights to atom sets
    point_to_idx = {p: i for i, p in enumerate(sorted_points)}
    
    weight_atom_sets = []
    for w in weights:
        atom_set = set()
        for start, end in w:
            if start in point_to_idx and (end + 1) in point_to_idx:
                s_idx = point_to_idx[start]
                e_idx = point_to_idx[end + 1]
                for i in range(s_idx, e_idx):
                    atom_set.add(i)
        weight_atom_sets.append(frozenset(atom_set))
    
    # Step 3: Build BDD/ZDD for the family of sets
    # Using dd's autoref BDD (no native ZDD in pure Python dd)
    # But we can simulate ZDD-like behavior by encoding sets
    
    # Actually, dd.autoref doesn't have native ZDD.
    # Let's just encode each weight as a conjunction of atom variables.
    # This is essentially a BDD, but on the atom space (much smaller than integer space).
    
    print("  Building BDDs on atom space...", flush=True)
    
    bdd = BDD()
    
    # Declare variables for each atom
    # Limit to reasonable number to avoid explosion
    MAX_ATOMS_FOR_BDD = 5000
    if num_atoms > MAX_ATOMS_FOR_BDD:
        print(f"  WARNING: Too many atoms ({num_atoms}). Sampling first {MAX_ATOMS_FOR_BDD}.", flush=True)
        num_atoms_actual = MAX_ATOMS_FOR_BDD
    else:
        num_atoms_actual = num_atoms
    
    for i in range(num_atoms_actual):
        bdd.declare(f"a{i}")
    
    # Encode each weight as a BDD (characteristic function over atoms)
    weight_bdds = []
    for idx, atom_set in enumerate(weight_atom_sets):
        if (idx + 1) % 100 == 0:
            print(f"    Weight {idx+1}/{len(weights)}, BDD nodes: {len(bdd)}...", flush=True)
        
        # Build conjunction: atoms in set are True, others are False
        # This represents a single "point" in the BDD (the specific set)
        expr_parts = []
        for i in range(num_atoms_actual):
            if i in atom_set:
                expr_parts.append(f"a{i}")
            else:
                expr_parts.append(f"!a{i}")
        
        if expr_parts:
            # Limit expression size
            if len(expr_parts) > 200:
                # Too large, skip or sample
                w_bdd = bdd.false  # Placeholder
            else:
                expr = " & ".join(expr_parts)
                try:
                    w_bdd = bdd.add_expr(expr)
                except Exception as e:
                    print(f"    Error on weight {idx}: {e}", flush=True)
                    w_bdd = bdd.false
        else:
            w_bdd = bdd.true
        
        weight_bdds.append(w_bdd)
    
    elapsed = time.time() - start_time
    
    # Measure
    bdd_nodes = len(bdd)
    atom_storage = num_atoms * 2 * 4  # (start, end) per atom
    
    total_ranges = sum(len(w) for w in weights)
    raw_size = total_ranges * 2 * 4
    
    # BDD storage: assume 16 bytes per node (variable, low, high, etc.)
    bdd_storage = bdd_nodes * 16 + atom_storage
    
    print(f"\n  Results:", flush=True)
    print(f"    Atoms: {num_atoms}", flush=True)
    print(f"    BDD Nodes: {bdd_nodes}", flush=True)
    print(f"    Atom Storage: {atom_storage}", flush=True)
    print(f"    BDD Storage (est.): {bdd_storage}", flush=True)
    print(f"    Raw Size: {raw_size}", flush=True)
    if bdd_storage > 0:
        print(f"    Compression Ratio: {raw_size / bdd_storage:.2f}x", flush=True)
    print(f"    Time: {elapsed:.2f}s", flush=True)
    
    return {
        "ratio": raw_size / bdd_storage if bdd_storage > 0 else 0
    }


def test_simple_shared_atoms(filename):
    """
    Simpler approach: Just count unique atom-sets and measure sharing.
    No BDD, just hash-based deduplication.
    """
    weights, max_val = load_weights(filename)
    if not weights:
        return None
    
    print(f"\n=== Shared Atom Sets: {filename} ===", flush=True)
    start_time = time.time()
    
    # Step 1: Compute atoms
    print("  Computing atoms...", flush=True)
    points = set()
    for w in weights:
        for start, end in w:
            points.add(start)
            points.add(end + 1)
    sorted_points = sorted(points)
    
    atoms = []
    for i in range(len(sorted_points) - 1):
        atoms.append((sorted_points[i], sorted_points[i+1] - 1))
    
    num_atoms = len(atoms)
    point_to_idx = {p: i for i, p in enumerate(sorted_points)}
    
    # Step 2: Map weights to atom sets
    unique_atom_sets = {}
    weight_refs = []
    
    for w in weights:
        atom_set = set()
        for start, end in w:
            if start in point_to_idx and (end + 1) in point_to_idx:
                s_idx = point_to_idx[start]
                e_idx = point_to_idx[end + 1]
                for i in range(s_idx, e_idx):
                    atom_set.add(i)
        
        key = frozenset(atom_set)
        if key not in unique_atom_sets:
            unique_atom_sets[key] = len(unique_atom_sets)
        weight_refs.append(unique_atom_sets[key])
    
    elapsed = time.time() - start_time
    
    num_unique = len(unique_atom_sets)
    
    # Storage: atoms + unique atom-sets (as lists of IDs) + weight refs
    atom_storage = num_atoms * 2 * 4
    
    # Atom sets: sum of lengths * 4 bytes per ID
    atomset_storage = sum(len(s) for s in unique_atom_sets.keys()) * 4
    
    # Refs: 4 bytes per weight
    ref_storage = len(weights) * 4
    
    total_compressed = atom_storage + atomset_storage + ref_storage
    
    total_ranges = sum(len(w) for w in weights)
    raw_size = total_ranges * 2 * 4
    
    print(f"  Atoms: {num_atoms}", flush=True)
    print(f"  Unique Atom Sets: {num_unique}", flush=True)
    print(f"  Atom Storage: {atom_storage}", flush=True)
    print(f"  Atom-Set Storage: {atomset_storage}", flush=True)
    print(f"  Ref Storage: {ref_storage}", flush=True)
    print(f"  Total Compressed: {total_compressed}", flush=True)
    print(f"  Raw Size: {raw_size}", flush=True)
    print(f"  Compression Ratio: {raw_size / total_compressed:.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    return {
        "method": "Shared Atom Sets",
        "ratio": raw_size / total_compressed
    }


if __name__ == "__main__":
    for fname in ["range_weights_terminal_dwa.json", "range_weights_parser_dwa.json"]:
        test_simple_shared_atoms(fname)
        # Skip BDD test if atoms are too many
        # test_zdd_on_atoms(fname)
