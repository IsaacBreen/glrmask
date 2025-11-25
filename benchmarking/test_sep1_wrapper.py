import sys
from pathlib import Path
import time

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.systems.sep1 import Sep1System

def test_sep1():
    print("Testing sep1 wrapper...")
    system = Sep1System()
    
    # Locate a constraint file
    # We'll use one from the root directory if available
    constraint_file = PROJECT_ROOT / "python/serialized_compiled_grammar.json"
    if not constraint_file.exists():
        print(f"Constraint file not found at {constraint_file}")
        # Try to find any json.gz file
        candidates = list(PROJECT_ROOT.glob("*.json.gz"))
        if candidates:
            constraint_file = candidates[0]
            print(f"Using alternative: {constraint_file}")
        else:
            print("No constraint file found. Skipping test.")
            return

    print(f"Compiling/Loading {constraint_file}...")
    # vocab is not strictly needed for loading the precompiled JSON in sep1
    # but the interface requires it. We'll pass an empty dict for now.
    compilation_result = system.compile_grammar(constraint_file, {})
    print(f"Compilation took {compilation_result.compilation_time_sec:.4f}s")
    
    print("Creating state...")
    state = system.create_state(compilation_result.compiled)
    
    print("Getting mask...")
    mask_result = system.get_mask(state)
    print(f"Mask has {len(mask_result.valid_token_ids)} valid tokens")
    print(f"Mask time: {mask_result.time_sec:.6f}s")
    
    # Commit a token (just pick the first valid one)
    if mask_result.valid_token_ids:
        token_id = mask_result.valid_token_ids[0]
        print(f"Committing token {token_id}...")
        commit_result = system.commit(state, token_id)
        print(f"Commit time: {commit_result.time_sec:.6f}s")
        
        # Get mask again
        mask_result2 = system.get_mask(commit_result.new_state)
        print(f"New mask has {len(mask_result2.valid_token_ids)} valid tokens")
    
    print("Test passed!")

if __name__ == "__main__":
    test_sep1()
