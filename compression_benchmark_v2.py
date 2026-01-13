#!/usr/bin/env python3
"""
Advanced Compression Benchmark: More aggressive strategies.

Tests:
1. Delta Encoding (sort ranges, encode deltas)
2. Hierarchical Range Sharing (decompose overlapping ranges)
3. LZ-style suffix sharing
4. Alphabet reduction (map to smaller domain)
"""

import json
import sys
import time
import zlib
import lzma
from collections import defaultdict


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


def test_delta_encoding(weights, max_val):
    """
    Delta encoding: Sort all range endpoints and encode as deltas.
    """
    print("\n=== Delta Encoding ===", flush=True)
    start_time = time.time()
    
    # Collect all endpoints
    all_ranges = []
    for w in weights:
        all_ranges.extend(w)
    
    # Sort by start
    all_ranges.sort()
    
    # Encode as deltas
    deltas = []
    prev = 0
    for start, end in all_ranges:
        deltas.append(start - prev)
        deltas.append(end - start)  # Length (always positive)
        prev = end
    
    # Convert to bytes (varint encoding with zigzag for signed)
    def zigzag(n):
        return (n << 1) ^ (n >> 31)
    
    raw_bytes = b''
    for d in deltas:
        d = zigzag(d)  # Make non-negative
        while d >= 128:
            raw_bytes += bytes([d & 0x7f | 0x80])
            d >>= 7
        raw_bytes += bytes([d])
    
    elapsed = time.time() - start_time
    
    total_ranges = len(all_ranges)
    raw_size = total_ranges * 2 * 4
    delta_size = len(raw_bytes)
    
    print(f"  Total ranges: {total_ranges}", flush=True)
    print(f"  Raw size (bytes): {raw_size}", flush=True)
    print(f"  Delta encoded (bytes): {delta_size}", flush=True)
    print(f"  Compression ratio: {raw_size / delta_size:.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    return {
        "method": "Delta Encoding",
        "raw_bytes": raw_size,
        "compressed_bytes": delta_size,
        "ratio": raw_size / delta_size if delta_size > 0 else 0,
    }


def test_zlib_raw(weights, max_val):
    """
    Just compress the raw range data with zlib.
    """
    print("\n=== zlib on Raw Ranges ===", flush=True)
    start_time = time.time()
    
    # Flatten to bytes
    data = b''
    for w in weights:
        for start, end in w:
            data += start.to_bytes(4, 'little')
            data += end.to_bytes(4, 'little')
    
    compressed = zlib.compress(data, level=9)
    
    elapsed = time.time() - start_time
    
    print(f"  Raw size (bytes): {len(data)}", flush=True)
    print(f"  Compressed (bytes): {len(compressed)}", flush=True)
    print(f"  Compression ratio: {len(data) / len(compressed):.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    return {
        "method": "zlib",
        "raw_bytes": len(data),
        "compressed_bytes": len(compressed),
        "ratio": len(data) / len(compressed) if len(compressed) > 0 else 0,
    }


def test_lzma_raw(weights, max_val):
    """
    Just compress the raw range data with LZMA.
    """
    print("\n=== LZMA on Raw Ranges ===", flush=True)
    start_time = time.time()
    
    data = b''
    for w in weights:
        for start, end in w:
            data += start.to_bytes(4, 'little')
            data += end.to_bytes(4, 'little')
    
    compressed = lzma.compress(data)
    
    elapsed = time.time() - start_time
    
    print(f"  Raw size (bytes): {len(data)}", flush=True)
    print(f"  Compressed (bytes): {len(compressed)}", flush=True)
    print(f"  Compression ratio: {len(data) / len(compressed):.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    return {
        "method": "LZMA",
        "raw_bytes": len(data),
        "compressed_bytes": len(compressed),
        "ratio": len(data) / len(compressed) if len(compressed) > 0 else 0,
    }


def test_range_tree_sharing(weights, max_val):
    """
    Build a shared interval tree across all weights.
    Represent each weight as a bitmap over shared "atoms".
    """
    print("\n=== Range Tree Sharing (Atoms) ===", flush=True)
    start_time = time.time()
    
    # Collect all endpoints
    points = set()
    for w in weights:
        for start, end in w:
            points.add(start)
            points.add(end + 1)
    
    sorted_points = sorted(points)
    
    # Create atoms: intervals between consecutive points
    atoms = []
    for i in range(len(sorted_points) - 1):
        atoms.append((sorted_points[i], sorted_points[i+1] - 1))
    
    num_atoms = len(atoms)
    print(f"  Number of atoms: {num_atoms}", flush=True)
    
    # Map each atom to an index
    atom_to_idx = {a: i for i, a in enumerate(atoms)}
    
    # For each weight, determine which atoms it covers
    weight_atom_sets = []
    for w in weights:
        atom_set = set()
        for start, end in w:
            # Find overlapping atoms
            for a in atoms:
                if a[0] >= start and a[1] <= end:
                    atom_set.add(atom_to_idx[a])
        weight_atom_sets.append(atom_set)
    
    elapsed = time.time() - start_time
    
    # Storage: atoms + per-weight atom sets
    # Atoms: num_atoms * 2 * 4 bytes
    # Atom sets: could be bitmaps or lists
    
    # Use bitmap approach: each weight = num_atoms bits
    bitmap_bytes = (num_atoms + 7) // 8 * len(weights)
    atom_storage = num_atoms * 2 * 4
    total_size = atom_storage + bitmap_bytes
    
    total_ranges = sum(len(w) for w in weights)
    raw_size = total_ranges * 2 * 4
    
    print(f"  Atom storage (bytes): {atom_storage}", flush=True)
    print(f"  Bitmap storage (bytes): {bitmap_bytes}", flush=True)
    print(f"  Total size (bytes): {total_size}", flush=True)
    print(f"  Raw size (bytes): {raw_size}", flush=True)
    print(f"  Compression ratio: {raw_size / total_size:.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    return {
        "method": "Atom Bitmaps",
        "raw_bytes": raw_size,
        "compressed_bytes": total_size,
        "ratio": raw_size / total_size if total_size > 0 else 0,
        "num_atoms": num_atoms,
    }


def test_zlib_on_atoms(weights, max_val):
    """
    Atom bitmaps + zlib compression.
    """
    print("\n=== Atom Bitmaps + zlib ===", flush=True)
    start_time = time.time()
    
    # Collect all endpoints
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
    atom_to_idx = {a: i for i, a in enumerate(atoms)}
    
    # Build bitmaps
    bitmaps = []
    for w in weights:
        bitmap = [0] * ((num_atoms + 7) // 8)
        for start, end in w:
            for a in atoms:
                if a[0] >= start and a[1] <= end:
                    idx = atom_to_idx[a]
                    bitmap[idx // 8] |= (1 << (idx % 8))
        bitmaps.append(bytes(bitmap))
    
    # Concatenate and compress
    all_bitmaps = b''.join(bitmaps)
    compressed = zlib.compress(all_bitmaps, level=9)
    
    elapsed = time.time() - start_time
    
    atom_storage = num_atoms * 2 * 4
    total_size = atom_storage + len(compressed)
    
    total_ranges = sum(len(w) for w in weights)
    raw_size = total_ranges * 2 * 4
    
    print(f"  Atoms: {num_atoms}", flush=True)
    print(f"  Bitmap raw (bytes): {len(all_bitmaps)}", flush=True)
    print(f"  Bitmap compressed (bytes): {len(compressed)}", flush=True)
    print(f"  Total size (bytes): {total_size}", flush=True)
    print(f"  Raw size (bytes): {raw_size}", flush=True)
    print(f"  Compression ratio: {raw_size / total_size:.2f}x", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)
    
    return {
        "method": "Atom Bitmaps + zlib",
        "raw_bytes": raw_size,
        "compressed_bytes": total_size,
        "ratio": raw_size / total_size if total_size > 0 else 0,
    }


def main():
    results = {}
    
    for filename in ["range_weights_terminal_dwa.json", "range_weights_parser_dwa.json"]:
        print(f"\n{'='*60}", flush=True)
        print(f"ADVANCED BENCHMARK: {filename}", flush=True)
        print(f"{'='*60}", flush=True)
        
        weights, max_val = load_weights(filename)
        if weights is None:
            continue
        
        file_results = []
        
        r = test_delta_encoding(weights, max_val)
        if r: file_results.append(r)
        
        r = test_zlib_raw(weights, max_val)
        if r: file_results.append(r)
        
        r = test_lzma_raw(weights, max_val)
        if r: file_results.append(r)
        
        r = test_range_tree_sharing(weights, max_val)
        if r: file_results.append(r)
        
        r = test_zlib_on_atoms(weights, max_val)
        if r: file_results.append(r)
        
        results[filename] = file_results
        
        print(f"\n--- Summary for {filename} ---", flush=True)
        print(f"{'Method':<25} {'Raw':>12} {'Compressed':>12} {'Ratio':>8}", flush=True)
        print("-" * 60, flush=True)
        for r in sorted(file_results, key=lambda x: -x['ratio']):
            print(f"{r['method']:<25} {r['raw_bytes']:>12} {r['compressed_bytes']:>12} {r['ratio']:>7.2f}x", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("BEST RESULTS", flush=True)
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
