import sys
from pathlib import Path
import time

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.systems.outlines_wrapper import OutlinesSystem

def test_outlines():
    print("Testing Outlines wrapper...")
    system = OutlinesSystem()
    
    grammar_file = PROJECT_ROOT / "benchmarking/grammars/simple.json"
    
    print(f"Compiling {grammar_file}...")
    # Mock vocab
    compilation_result = system.compile_grammar(grammar_file, {})
    print(f"Compilation took {compilation_result.compilation_time_sec:.4f}s")
    
    print("Creating state...")
    state = system.create_state(compilation_result.compiled)
    
    print("Getting mask...")
    mask_result = system.get_mask(state)
    print(f"Mask has {len(mask_result.valid_token_ids)} valid tokens")
    print(f"Mask time: {mask_result.time_sec:.6f}s")
    
    # Commit a token (dummy)
    print("Committing token 1...")
    commit_result = system.commit(state, 1)
    print(f"Commit time: {commit_result.time_sec:.6f}s")
    
    print("Test passed!")

if __name__ == "__main__":
    test_outlines()
