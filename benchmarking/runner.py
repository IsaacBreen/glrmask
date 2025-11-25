import argparse
import json
import time
from pathlib import Path
import sys
import importlib

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.systems.sep1 import Sep1System
from benchmarking.systems.outlines_wrapper import OutlinesSystem

def run_benchmark(system, grammar_path, input_tokens, repeat=1):
    print(f"Benchmarking {system.name} with {grammar_path.name}...")
    
    # Compile
    try:
        compilation = system.compile_grammar(grammar_path, {})
        print(f"Compilation time: {compilation.compilation_time_sec:.4f}s")
    except Exception as e:
        print(f"Compilation failed: {e}")
        return None

    results = {
        "get_mask_times": [],
        "commit_times": [],
        "total_time": 0
    }
    
    for r in range(repeat):
        state = system.create_state(compilation.compiled)
        
        run_start = time.perf_counter()
        for i, token_id in enumerate(input_tokens):
            # Get mask
            mask_res = system.get_mask(state)
            results["get_mask_times"].append(mask_res.time_sec)
            
            # Commit
            commit_res = system.commit(state, token_id)
            results["commit_times"].append(commit_res.time_sec)
            state = commit_res.new_state
            
        results["total_time"] += time.perf_counter() - run_start
        
    results["total_time"] /= repeat
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar", type=Path, required=True)
    parser.add_argument("--input-tokens", type=str, help="Comma separated token IDs")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    
    # Setup systems
    systems = [
        Sep1System(),
        # OutlinesSystem() # Uncomment when ready
    ]
    
    # Dummy tokens if not provided
    if args.input_tokens:
        tokens = [int(t) for t in args.input_tokens.split(",")]
    else:
        tokens = [1, 2, 3] # Dummy
        
    for system in systems:
        res = run_benchmark(system, args.grammar, tokens, args.repeat)
        if res:
            import numpy as np
            print(f"Results for {system.name}:")
            print(f"  Mean get_mask: {np.mean(res['get_mask_times'])*1e6:.2f} us")
            print(f"  Mean commit:   {np.mean(res['commit_times'])*1e6:.2f} us")

if __name__ == "__main__":
    main()
