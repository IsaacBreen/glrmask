import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple, Optional, Type, Callable
import itertools

# --- Path setup for importing GSS interface and implementations ---
# This allows the script to be run from the project root.
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gss_tester.interface import GSS, MergeableInt

MAX_STACKS_PREVIEW = 5
TRACE_CONTEXT_WINDOW = 10


def analyze_results(result_files: List[Path], reference_file: Path = None, minimize: bool = False):
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

    def extract_trace_window(results: List[Dict[str, Any]], up_to_index: int, window: int = TRACE_CONTEXT_WINDOW) -> List[Tuple[int, Dict[str, Any]]]:
        start = max(0, up_to_index - window + 1)
        items: List[Tuple[int, Dict[str, Any]]] = []
        for i in range(start, up_to_index + 1):
            r = results[i]
            t = r.get("trace")
            if isinstance(t, dict):
                items.append((i, t))
        return items[-window:]

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
        divergence_found = False
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
                    divergence_found = True
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

                    # Rolling window of previous fuzz ops for each impl (up to TRACE_CONTEXT_WINDOW)
                    ctx1 = extract_trace_window(all_results[impl_name1], i, TRACE_CONTEXT_WINDOW)
                    ctx2 = extract_trace_window(all_results[impl_name2], i, TRACE_CONTEXT_WINDOW)
                    if ctx1:
                        seed1 = ctx1[-1][1].get("seed")
                        print(f"  - Trace context for {impl_name1} (seed={seed1}, last {len(ctx1)} op(s)):")
                        for idx, tr in ctx1:
                            op = tr.get("op")
                            step = tr.get("step")
                            args = tr.get("args", {})
                            ss = tr.get("source_stacks")
                            rs = tr.get("result_stacks")
                            print(f"    [{idx}] step={step} op={op} args={json.dumps(args, ensure_ascii=False)}")
                            if ss is not None:
                                print(f"         src={pretty_stacks(ss)}")
                            if rs is not None:
                                print(f"         res={pretty_stacks(rs)}")
                    else:
                        print(f"  - No fuzz trace context available for {impl_name1}.")

                    if ctx2:
                        seed2 = ctx2[-1][1].get("seed")
                        print(f"  - Trace context for {impl_name2} (seed={seed2}, last {len(ctx2)} op(s)):")
                        for idx, tr in ctx2:
                            op = tr.get("op")
                            step = tr.get("step")
                            args = tr.get("args", {})
                            ss = tr.get("source_stacks")
                            rs = tr.get("result_stacks")
                            print(f"    [{idx}] step={step} op={op} args={json.dumps(args, ensure_ascii=False)}")
                            if ss is not None:
                                print(f"         src={pretty_stacks(ss)}")
                            if rs is not None:
                                print(f"         res={pretty_stacks(rs)}")

                    # --- Minimization ---
                    if minimize:
                        # Check if this divergence came from a fuzz test
                        is_fuzz_failure = t1 and t1.get("phase") == "fuzz"
                        if is_fuzz_failure:
                            # The reference implementation determines the "correct" trace
                            ref_impl = ref_impl_name if ref_impl_name in [impl_name1, impl_name2] else impl_name2
                            other_impl = impl_name1 if ref_impl == impl_name2 else impl_name1
                            
                            ref_results = all_results[ref_impl]
                            full_trace = [r['trace'] for r in ref_results if 'trace' in r and r['trace'].get("phase") == "fuzz"]
                            
                            minimize_divergence(ref_impl, other_impl, full_trace)

                    break # Show only the first divergence for this pair
            if divergence_found and minimize:
                break # Only minimize the first pair found

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
    parser.add_argument(
        "-m", "--minimize",
        action="store_true",
        help="Attempt to minimize any found fuzzing divergence to a minimal reproducible trace."
    )
    args = parser.parse_args()

    valid_files = [f for f in args.result_files if f.exists() and f.is_file()]
    if not valid_files:
        print("Error: No valid result files found.")
        return
    
    analyze_results(valid_files, args.reference, args.minimize)


def _run_replay(gss_class: Type[GSS], trace_log: List[Dict[str, Any]], max_gss_states: int = 10) -> List[Dict[str, Any]]:
    """
    Executes a trace log against a GSS implementation and returns the results.
    This is a non-yielding, replay-only version of the fuzzer.
    """
    gss_states: List[GSS] = []
    results: List[Dict[str, Any]] = []

    for i, trace_op in enumerate(trace_log):
        op_choice = trace_op["op"]

        if op_choice == 'init' or op_choice == 'restart_empty_pool':
            new_gss = gss_class.from_stacks([([], MergeableInt(0))])
            if op_choice == 'init':
                gss_states = [new_gss]
            else:
                gss_states.append(new_gss)
            results.append({"state": new_gss.to_stacks(), "trace": trace_op})
            continue

        if not gss_states:
            break

        source_index = trace_op["source_index"]
        if source_index is None or source_index >= len(gss_states):
            break
        source_gss = gss_states[source_index]

        args = trace_op["args"]
        new_gss: Optional[GSS] = None

        try:
            if op_choice == 'push':
                new_gss = source_gss.push(args["value"])
            elif op_choice == 'pop':
                new_gss = source_gss.pop()
            elif op_choice == 'popn':
                new_gss = source_gss.popn(args["n"])
            elif op_choice == 'isolate':
                new_gss = source_gss.isolate(args["value"])
            elif op_choice == 'apply':
                amount = args["amount"]
                func: Callable[[MergeableInt], MergeableInt] = lambda acc, amt=amount: acc + amt
                new_gss = source_gss.apply(func)
            elif op_choice == 'prune':
                threshold = args["threshold"]
                predicate: Callable[[MergeableInt], bool] = lambda acc, thr=threshold: acc.real > thr
                new_gss = source_gss.prune(predicate)
            elif op_choice == 'merge':
                can_merge = len(gss_states) >= 2
                if not can_merge:
                    continue
                other_index = trace_op["other_index"]
                if other_index is None or other_index >= len(gss_states):
                    break
                other_gss = gss_states[other_index]
                new_gss = source_gss.merge(other_gss)
            else:
                continue

            if new_gss is not None:
                if new_gss is not source_gss and not new_gss.is_empty():
                    gss_states.append(new_gss)
                results.append({"state": new_gss.to_stacks(), "trace": trace_op})

            if len(gss_states) > max_gss_states:
                gss_states = gss_states[-max_gss_states:]
        except Exception:
            break

    return results


def _check_divergence(impl_name1: str, impl_name2: str, trace: List[Dict[str, Any]]) -> bool:
    """Runs a trace on two implementations and returns True if they diverge."""
    try:
        mod_name1, cls_name1 = impl_name1.rsplit('.', 1)
        gss_class1 = getattr(importlib.import_module(mod_name1), cls_name1)

        mod_name2, cls_name2 = impl_name2.rsplit('.', 1)
        gss_class2 = getattr(importlib.import_module(mod_name2), cls_name2)
    except (ImportError, AttributeError) as e:
        print(f"Error loading implementations for minimization: {e}", file=sys.stderr)
        return False

    results1 = _run_replay(gss_class1, trace)
    results2 = _run_replay(gss_class2, trace)

    sig1 = tuple(json.dumps(r['state'], sort_keys=True) for r in results1)
    sig2 = tuple(json.dumps(r['state'], sort_keys=True) for r in results2)

    return sig1 != sig2


def minimize_divergence(ref_impl: str, other_impl: str, full_trace: List[Dict[str, Any]]):
    """
    Attempts to find the smallest subset of a trace that still causes a divergence.
    """
    print("\n--- Minimizing Divergence ---")
    print(f"Minimizing trace for '{ref_impl}' vs '{other_impl}' (starting with {len(full_trace)} ops)...")

    minimized_trace = list(full_trace)
    
    # Simple forward deletion shrinker
    i = 0
    while i < len(minimized_trace):
        shrunk_trace = minimized_trace[:i] + minimized_trace[i+1:]
        if not shrunk_trace:
            i += 1
            continue
        
        if _check_divergence(ref_impl, other_impl, shrunk_trace):
            minimized_trace = shrunk_trace
            # Restart scan from the beginning
            i = 0
        else:
            i += 1
    
    print(f"\n--- Minimized Failing Trace ({len(minimized_trace)} operations) ---")
    for i, op in enumerate(minimized_trace):
        op_name = op['op']
        args = json.dumps(op['args'])
        print(f"[{i+1}] op={op_name}, args={args}, source_idx={op['source_index']}, other_idx={op['other_index']}")

if __name__ == "__main__":
    main()
