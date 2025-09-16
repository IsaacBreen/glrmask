import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import itertools

def analyze_results(result_files: List[Path], reference_file: Path = None):
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    implementations: List[str] = []

    for file_path in result_files:
        try:
            data = json.loads(file_path.read_text())
            impl_name = data["implementation"]
            all_results[impl_name] = data["results"]
            implementations.append(impl_name)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping invalid result file {file_path}: {e}")


    if not implementations:
        print("No valid result files found to analyze.")
        return

    # --- Equivalence Partitioning ---
    # A signature is a tuple of all yielded states and their line numbers.
    Signature = Tuple[Tuple[int, str], ...]

    def get_signature(results: List[Dict[str, Any]]) -> Signature:
        # Serialize state to a string to make it hashable and canonical.
        return tuple(
            (r['line'], json.dumps(r['state'], sort_keys=True)) for r in results
        )

    partitions: Dict[Signature, List[str]] = {}
    for impl_name, results in all_results.items():
        sig = get_signature(results)
        if sig not in partitions:
            partitions[sig] = []
        partitions[sig].append(impl_name)

    print("--- Consistency Analysis ---")
    print(f"Found {len(partitions)} equivalence class(es) among {len(implementations)} implementations.\n")

    # --- Reference Comparison ---
    ref_sig: Signature = None
    ref_impl_name: str = None
    if reference_file:
        try:
            ref_data = json.loads(reference_file.read_text())
            ref_impl_name = ref_data["implementation"]
            ref_sig = get_signature(ref_data["results"])
            print(f"Reference implementation: {ref_impl_name}\n")
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not load reference file {reference_file}: {e}")


    # --- Display Partitions ---
    sorted_partitions = sorted(partitions.items(), key=lambda item: len(item[1]), reverse=True)
    for i, (sig, impls) in enumerate(sorted_partitions):
        header = f"Class {i+1} ({len(impls)} implementation(s))"
        if ref_sig:
            match_icon = "✅" if sig == ref_sig else "❌"
            header += f" [Ref Match: {match_icon}]"
        
        print(header)
        print("-" * len(header))
        for impl in sorted(impls):
            print(f"  - {impl}")
        print()

    # --- Detailed Divergence Report ---
    if len(partitions) > 1:
        print("\n--- Divergence Report ---")
        # Compare every pair of partitions
        for (sig1, impls1), (sig2, impls2) in itertools.combinations(partitions.items(), 2):
            impl_name1 = sorted(impls1)[0]
            impl_name2 = sorted(impls2)[0]
            print(f"\nComparing '{impl_name1}' with '{impl_name2}':")
            
            results1 = all_results[impl_name1]
            results2 = all_results[impl_name2]
            
            max_len = max(len(results1), len(results2))
            for i in range(max_len):
                res1 = results1[i] if i < len(results1) else None
                res2 = results2[i] if i < len(results2) else None

                sig_item1 = (res1['line'], json.dumps(res1['state'], sort_keys=True)) if res1 else None
                sig_item2 = (res2['line'], json.dumps(res2['state'], sort_keys=True)) if res2 else None

                if sig_item1 != sig_item2:
                    print(f"  - First divergence at yield index {i}:")
                    if res1:
                        print(f"    - {impl_name1} (L{res1['line']}): {json.dumps(res1['state'])}")
                    else:
                        print(f"    - {impl_name1}: No yield at this index.")
                    
                    if res2:
                        print(f"    - {impl_name2} (L{res2['line']}): {json.dumps(res2['state'])}")
                    else:
                        print(f"    - {impl_name2}: No yield at this index.")
                    break # Show only the first divergence for this pair

def main():
    parser = argparse.ArgumentParser(description="Analyze GSS implementation consistency from result files.")
    parser.add_argument(
        "result_files",
        nargs='+',
        type=Path,
        help="Paths to result JSON files."
    )
    parser.add_argument(
        "-r", "--reference",
        type=Path,
        default=None,
        help="Path to a reference result file for comparison."
    )
    args = parser.parse_args()

    valid_files = [f for f in args.result_files if f.exists() and f.is_file()]
    if not valid_files:
        print("Error: No valid result files found.")
        return

    analyze_results(valid_files, args.reference)

if __name__ == "__main__":
    main()
