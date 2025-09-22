import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import itertools

MAX_STACKS_PREVIEW = 5

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

    def pretty_stacks(stacks: Any, max_items: int = MAX_STACKS_PREVIEW) -> str:
        try:
            if not isinstance(stacks, list):
                return json.dumps(stacks, ensure_ascii=False)
            shown = stacks[:max_items]
            s = json.dumps(shown, ensure_ascii=False)
            if len(stacks) > max_items:
                s += " ..."
            return s
        except Exception:
            return repr(stacks)

    def extract_full_trace(results: List[Dict[str, Any]], up_to_index: int) -> List[Tuple[int, Dict[str, Any]]]:
        items: List[Tuple[int, Dict[str, Any]]] = []
        for i in range(up_to_index + 1):
            if i < len(results):
                r = results[i]
                t = r.get("trace")
                if isinstance(t, dict):
                    items.append((i, t))
        return items

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

                    # Rich trace context (if available)
                    # Immediate operation at divergence index
                    t1 = res1.get("trace") if res1 else None
                    t2 = res2.get("trace") if res2 else None
                    if t1 or t2:
                        print("  - Divergence operation details:")
                        if t1:
                            print(f"    - {impl_name1}: op={t1.get('op')} step={t1.get('step')} seed={t1.get('seed')}")
                            args1 = t1.get("args", {})
                            print(f"      args={json.dumps(args1, ensure_ascii=False)}")
                            ss1 = t1.get("source_stacks")
                            rs1 = t1.get("result_stacks")
                            if ss1 is not None:
                                print(f"      source_stacks={pretty_stacks(ss1)}")
                            if rs1 is not None:
                                print(f"      result_stacks={pretty_stacks(rs1)}")
                        else:
                            print(f"    - {impl_name1}: No trace info at divergence index.")
                        if t2:
                            print(f"    - {impl_name2}: op={t2.get('op')} step={t2.get('step')} seed={t2.get('seed')}")
                            args2 = t2.get("args", {})
                            print(f"      args={json.dumps(args2, ensure_ascii=False)}")
                            ss2 = t2.get("source_stacks")
                            rs2 = t2.get("result_stacks")
                            if ss2 is not None:
                                print(f"      source_stacks={pretty_stacks(ss2)}")
                            if rs2 is not None:
                                print(f"      result_stacks={pretty_stacks(rs2)}")
                        else:
                            print(f"    - {impl_name2}: No trace info at divergence index.")

                    # --- Trace Diff ---
                    print(f"\n  --- Trace Diff (-{impl_name1}, +{impl_name2}) ---")

                    trace1 = extract_full_trace(all_results[impl_name1], i)
                    trace2 = extract_full_trace(all_results[impl_name2], i)

                    def format_trace_for_display(tr: Dict[str, Any]) -> List[str]:
                        op = tr.get("op")
                        step = tr.get("step")
                        args = json.dumps(tr.get("args", {}), ensure_ascii=False)
                        avail = tr.get("available_ops")
                        src_idx = tr.get("source_index")
                        src = pretty_stacks(tr.get("source_stacks"))
                        res = pretty_stacks(tr.get("result_stacks"))
                        
                        return [
                            f"step={step} op={op} args={args} src_idx={src_idx} avail_ops={avail}",
                            f"  src: {src}",
                            f"  res: {res}"
                        ]

                    max_len = max(len(trace1), len(trace2))
                    for idx in range(max_len):
                        tr1_data = trace1[idx][1] if idx < len(trace1) else None
                        tr2_data = trace2[idx][1] if idx < len(trace2) else None

                        lines1 = format_trace_for_display(tr1_data) if tr1_data else None
                        lines2 = format_trace_for_display(tr2_data) if tr2_data else None

                        if lines1 and lines2 and lines1 == lines2:
                            print(f"  [{idx: >3}] {lines1[0]}\n       {lines1[1]}\n       {lines1[2]}")
                        else:
                            if lines1:
                                print(f"- [{idx: >3}] {lines1[0]}\n-      {lines1[1]}\n-      {lines1[2]}")
                            if lines2:
                                print(f"+ [{idx: >3}] {lines2[0]}\n+      {lines2[1]}\n+      {lines2[2]}")

                    # Add explanation
                    t1 = res1.get("trace") if res1 else None
                    t2 = res2.get("trace") if res2 else None
                    if t1 and t2 and t1.get('op') != t2.get('op'):
                        print("\n  [Analysis]: The operations differ because the fuzzer's internal state pool diverged at a prior step.")
                        print("              Review the trace diff above to find the first step where 'res' (result_stacks) differs.")
                        print("              This earlier difference caused the fuzzer to select different source GSSs for this step, leading to different available operations ('avail_ops').")

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
