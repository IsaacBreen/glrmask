#!/usr/bin/env python3
"""
Analyze GrammarConstraint JSON file to identify space usage by component.
Shows which parts of the JSON use the most space and their compression potential.
"""

import json
import gzip
import sys
from pathlib import Path
from typing import Any, Dict
from collections import defaultdict


def get_size_bytes(obj: Any) -> int:
    """Get the JSON string size of an object"""
    return len(json.dumps(obj, separators=(',', ':')))


def analyze_json_structure(data: Dict[str, Any], path: str = "") -> Dict[str, Dict[str, int]]:
    """
    Recursively analyze JSON structure and report sizes.
    Returns dict mapping paths to {'raw_size', 'compressed_size', 'item_count'}
    """
    results = {}
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path }.{key}" if path else key
            raw_size = get_size_bytes(value)
            compressed_size = len(gzip.compress(json.dumps(value, separators=(',', ':')).encode()))
            
            # Count items if it's an array or dict
            if isinstance(value, list):
                item_count = len(value)
            elif isinstance(value, dict):
                item_count = len(value)
            else:
                item_count = 1
                
            results[current_path] = {
                'raw_size': raw_size,
                'compressed_size': compressed_size,
                'item_count': item_count,
                'compression_ratio': raw_size / compressed_size if compressed_size> 0 else 0
            }
            
            # Recurse into nested structures (but only one level deep for top-level analysis)
            if isinstance(value, dict) and not path:
                nested = analyze_json_structure(value, current_path)
                results.update(nested)
                
    return results


def format_size(bytes_val: int) -> str:
    """Format byte size as human-readable string"""
    if bytes_val < 1024:
        return f"{bytes_val}B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val/1024:.2f}KB"
    else:
        return f"{bytes_val/(1024*1024):.2f}MB"


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_json_size.py <json_file>")
        sys.exit(1)
        
    json_path = Path(sys.argv[1])
    
    if not json_path.exists():
        print(f"Error: File {json_path} not found")
        sys.exit(1)
        
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    total_size = json_path.stat().st_size
    print(f"Total file size: {format_size(total_size)}\n")
    
    # Analyze structure
    print("Analyzing JSON structure...\n")
    results = analyze_json_structure(data)
    
    # Sort by raw size
    sorted_results = sorted(results.items(), key=lambda x: x[1]['raw_size'], reverse=True)
    
    # Print table header
    print(f"{'Field Path':<50} {'Raw Size':<12} {'Compressed':<12} {'Items':<10} {'Ratio':<8} {'Waste%':<8}")
    print("=" * 110)
    
    # Print results
    for path, stats in sorted_results:
        waste_pct = (1 - stats['compressed_size'] / stats['raw_size']) * 100 if stats['raw_size'] > 0 else 0
        print(f"{path:<50} {format_size(stats['raw_size']):<12} {format_size(stats['compressed_size']):<12} "
              f"{stats['item_count']:<10} {stats['compression_ratio']:<8.2f} {waste_pct:<8.1f}")
    
    # Summary of top-level fields
    print("\n" + "=" * 110)
    print("\nTop-level field summary:")
    top_level = {k: v for k, v in results.items() if '.' not in k}
    total_top_level = sum(v['raw_size'] for v in top_level.values())
    
    for path, stats in sorted(top_level.items(), key=lambda x: x[1]['raw_size'], reverse=True):
        pct = (stats['raw_size'] / total_top_level * 100) if total_top_level > 0 else 0
        print(f"{path:<30} {format_size(stats['raw_size']):<12} ({pct:5.1f}%)")
    
    # Overall compression potential
    print("\n" + "=" * 110)
    total_compressed = sum(v['compressed_size'] for v in top_level.values())
    print(f"\nOverall compression potential: {format_size(total_top_level)} -> {format_size(total_compressed)}")
    print(f"Compression ratio: {total_top_level/total_compressed:.2f}x")
    print(f"Potential space savings: {(1 - total_compressed/total_top_level)*100:.1f}%")


if __name__ == "__main__":
    main()
