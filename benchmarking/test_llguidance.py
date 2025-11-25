"""Test llguidance wrapper."""

import sys
from pathlib import Path
import json
import tempfile

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from benchmarking.systems.llguidance_wrapper import LLGuidanceSystem, LLGUIDANCE_AVAILABLE
from benchmarking.grammars.test_schemas import SIMPLE_USER

def main():
    print("Testing llguidance wrapper...")
    print()
    
    if not LLGUIDANCE_AVAILABLE:
        print("ERROR: llguidance not available!")
        return 1
    
    # Create schema file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(SIMPLE_USER, f)
        schema_file = Path(f.name)
    
    try:
        # Initialize system
        print("1. Initializing llguidance system...")
        system = LLGuidanceSystem(tokenizer_name="gpt2")
        print(f"   System: {system.name}")
        print()
        
        # Compile schema
        print("2. Compiling JSON schema...")
        compilation_result = system.compile_grammar(schema_file, {})
        print(f"   Compilation time: {compilation_result.compilation_time_sec:.4f}s")
        print()
        
        # Create interpreter
        print("3. Creating interpreter...")
        interpreter = system.create_state(compilation_result.compiled)
        print("   Interpreter created")
        print()
        
        # Get initial mask
        print("4. Getting initial token mask...")
        mask_result = system.get_mask(interpreter)
        print(f"   Time: {mask_result.time_sec*1000:.3f}ms")
        print(f"   Valid tokens: {len(mask_result.valid_token_ids)}")
        if mask_result.valid_token_ids:
            print(f"   First 10: {mask_result.valid_token_ids[:10]}")
        print()
        
        # Commit a token
        if mask_result.valid_token_ids:
            token = mask_result.valid_token_ids[0]
            print(f"5. Committing token {token}...")
            commit_result = system.commit(interpreter, token)
            print(f"   Time: {commit_result.time_sec*1000:.3f}ms")
            print()
            
            # Get mask after commit
            print("6. Getting mask after commit...")
            mask_result2 = system.get_mask(interpreter)
            print(f"   Time: {mask_result2.time_sec*1000:.3f}ms")
            print(f"   Valid tokens: {len(mask_result2.valid_token_ids)}")
            print()
        
        print("✓ llguidance wrapper test passed!")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        schema_file.unlink()

if __name__ == "__main__":
    sys.exit(main())
