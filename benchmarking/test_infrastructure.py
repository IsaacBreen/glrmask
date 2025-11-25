"""Simple test runner to validate benchmarking infrastructure.

This tests that the base interface and sep1 wrapper work correctly before
adding complexity.
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarking.systems import Sep1System
import json


def load_gpt2_vocab():
    """Load a minimal GPT-2 vocabulary for testing."""
    # For testing, create a minimal vocab
    vocab = {}
    # Add some common tokens
    for i in range(256):  # Byte tokens
        vocab[i] = bytes([i])
    
    # Add some word tokens
    vocab[256] = b'if'
    vocab[257] = b' '
    vocab[258] = b'('
    vocab[259] = b'true'
    vocab[260] = b')'
    vocab[261] = b'{'
    vocab[262] = b'}'
    
    return vocab


def main():
    print("Testing benchmarking infrastructure...")
    print()
    
    # Initialize system
    print("1. Initializing sep1 system...")
    system = Sep1System()
    print(f"   System name: {system.name}")
    print()
    
    # Load vocab
    print("2. Loading vocabulary...")
    vocab = load_gpt2_vocab()
    print(f"   Loaded {len(vocab)} tokens")
    print()
    
    # Find a precompiled grammar
    print("3. Looking for precompiled grammars...")
    js_grammar_candidates = [
        Path(_project_root) / "reduced_js_grammar_constraint.json.gz",
        Path(_project_root) / "reduced_js_grammar_constraint11.json.gz",
    ]
    
    grammar_path = None
    for candidate in js_grammar_candidates:
        if candidate.exists():
            grammar_path = candidate
            print(f"   Found: {grammar_path.name}")
            break
    
    if not grammar_path:
        print("   ERROR: No precompiled grammar found!")
        print("   Searched for:", [c.name for c in js_grammar_candidates])
        return 1
    print()
    
    # Compile grammar
    print("4. Compiling grammar...")
    try:
        compilation_result = system.compile_grammar(grammar_path, vocab)
        print(f"   Compilation time: {compilation_result.compilation_time_sec:.4f}s")
        print(f"   Metadata: {compilation_result.metadata}")
    except Exception as e:
        print(f"   ERROR during compilation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print()
    
    #Create initial state
    print("5. Creating initial state...")
    try:
        state = system.create_state(compilation_result.compiled)
        print("   State created successfully")
    except Exception as e:
        print(f"   ERROR creating state: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print()
    
    # Test get_mask
    print("6. Testing get_mask...")
    try:
        mask_result = system.get_mask(state)
        print(f"   Time: {mask_result.time_sec:.6f}s")
        print(f"   Valid tokens: {len(mask_result.valid_token_ids)}")
        if len(mask_result.valid_token_ids) > 0:
            print(f"   First few: {mask_result.valid_token_ids[:10]}")
    except Exception as e:
        print(f"   ERROR in get_mask: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print()
    
    # Test commit
    print("7. Testing commit...")
    if len(mask_result.valid_token_ids) > 0:
        try:
            token_to_commit = mask_result.valid_token_ids[0]
            commit_result = system.commit(state, token_to_commit)
            print(f"   Committed token {token_to_commit}")
            print(f"   Time: {commit_result.time_sec:.6f}s")
        except Exception as e:
            print(f"   ERROR in commit: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("   Skipping (no valid tokens)")
    print()
    
    # Test another get_mask after commit
    print("8. Testing get_mask after commit...")
    try:
        mask_result2 = system.get_mask(state)
        print(f"   Time: {mask_result2.time_sec:.6f}s")
        print(f"   Valid tokens: {len(mask_result2.valid_token_ids)}")
    except Exception as e:
        print(f"   ERROR in get_mask: {e}")
        import traceback
        traceback.print_exc()
        return 1
    print()
    
    print("✓ All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
