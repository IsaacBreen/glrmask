import sys
import json
import subprocess
import tempfile
from pathlib import Path
import os

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.json_schema_to_ebnf import convert_schema_to_ebnf

def test_sep1_compilation():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"}
        },
        "required": ["name", "age"]
    }
    
    print("Using simple EBNF...")
    ebnf_str = """
    root ::= '{' '}' ;
    """
    print("EBNF:")
    print(ebnf_str)
    
    # Save EBNF
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ebnf', delete=False) as f:
        f.write(ebnf_str)
        ebnf_path = Path(f.name)
        
    # Load actual GPT2 vocab
    try:
        with open(PROJECT_ROOT / "benchmarking/gpt2_vocab.json") as f:
            gpt2_vocab_json = json.load(f)
            # Convert to {string: id} format expected by compiler
            vocab = gpt2_vocab_json
    except:
        # Fallback to dummy vocab
        vocab = {str(i): i for i in range(50257)}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(vocab, f)
        vocab_path = Path(f.name)
        
    output_path = Path("debug_output.json")
    
    print("Running compiler...")
    compiler_bin = PROJECT_ROOT / "target" / "release" / "grammar-compiler"
    
    cmd = [str(compiler_bin), 
           "--grammar", str(ebnf_path), 
           "--vocab", str(vocab_path), 
           "--output", str(output_path)]
           
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError("Compiler failed")
        
    print("Compiler finished successfully.")
    
    # Load compiled constraint
    from benchmarking.systems.sep1 import Sep1System
    
    # We need to mock the system or just use the model directly
    # Let's use the Sep1System wrapper logic partially
    
    # We need to load the model.
    # The wrapper does: RustModel.from_json_string(json_str)
    
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from aug25.models.rust_model import Model as RustModel
    
    with open(output_path, 'r') as f:
        constraint_json = f.read()
        
    print("Loading model...")
    model = RustModel.from_json_string(constraint_json)
    
    print("Creating state...")
    # The wrapper uses system.create_state(model) -> model.start_state() ??
    # No, sep1.py: create_state(compiled) -> compiled (which is the model)
    # Wait, sep1.py create_state returns the model itself?
    # Let's check sep1.py
    
    # sep1.py:
    # def create_state(self, compiled: Any) -> Any:
    #    return compiled # The RustModel instance
    
    # def get_mask(self, state: Any) -> MaskResult:
    #    mask = state.get_mask() 
    
    # So the "state" IS the model?
    # But RustModel has internal state?
    # No, RustModel wraps GrammarConstraint.
    # Where is the state?
    # RustModel.get_mask() creates a NEW state?
    # No, RustModel likely maintains state or we need to create a state object.
    
    # Let's check rust_model.py
    # class RustModel:
    #    def __init__(self, constraint): ...
    #    def get_mask(self): ...
    #    def commit(self, token_id): ...
    
    # So RustModel IS the stateful object in the python wrapper?
    # Yes.
    
    state = model
    
    print("Starting generation loop...")
    for i in range(10):
        print(f"Step {i}")
        mask = state.get_mask()
        # mask is FFIBitset
        # We need to find a valid token
        # FFIBitset has to_ranges() ?
        # Or we can just iterate?
        
        # In sep1.py:
        # ranges = mask.to_ranges()
        # valid_tokens = []
        # for start, end in ranges:
        #     valid_tokens.extend(range(start, end)) # Exclusive?
        
        # Let's check sep1.py logic for ranges
        # It says: # sep1 returns [start, end) ranges (exclusive end)
        
        ranges = mask.to_ranges()
        print(f"  Ranges: {ranges}")
        
        valid_tokens = []
        for start, end in ranges:
            valid_tokens.extend(range(start, end + 1))
            
        if not valid_tokens:
            print("  No valid tokens!")
            break
            
        token_id = valid_tokens[0]
        print(f"  Committing token {token_id}")
        state.commit(token_id)
    
    print("Done.")
    
    # Cleanup
    ebnf_path.unlink()
    vocab_path.unlink()
    if output_path.exists():
        output_path.unlink()

if __name__ == "__main__":
    test_sep1_compilation()
