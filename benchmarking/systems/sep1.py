import sys
from pathlib import Path
from typing import Any, List, Tuple
import time
import json
import gzip
import subprocess
import tempfile
import os

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.systems.base import BaseSystem, CompilationResult, MaskResult, CommitResult, time_function
from benchmarking.json_schema_to_ebnf import convert_schema_to_ebnf

# Import the Rust model
try:
    from python.aug25.models.rust_model import Model as RustModel
    import _sep1 as ffi
except ImportError as e:
    print(f"Warning: Could not import sep1 Rust model: {e}")
    RustModel = None

class Sep1System(BaseSystem):
    @property
    def name(self) -> str:
        return "sep1"

    def compile_grammar(
        self,
        grammar_path: Path,
        vocab: dict[int, bytes],
        **kwargs
    ) -> CompilationResult:
        start_time = time.perf_counter()
        
        # 1. Determine input format and convert to EBNF if needed
        if grammar_path.suffix == '.json' and not grammar_path.name.endswith('.json.gz'):
            # It's a JSON Schema (unless it's a precompiled constraint, but we assume schema for .json)
            # Check if it looks like a schema
            try:
                with open(grammar_path) as f:
                    data = json.load(f)
                if "llm_token_map" in data:
                    # It's a precompiled constraint
                    ebnf_path = None
                    compiled_json_path = grammar_path
                else:
                    # It's a JSON Schema
                    # Convert to EBNF
                    ebnf_str = convert_schema_to_ebnf(data)
                    
                    # Save EBNF to temp file
                    fd, ebnf_path_str = tempfile.mkstemp(suffix=".ebnf", text=True)
                    os.close(fd)
                    ebnf_path = Path(ebnf_path_str)
                    ebnf_path.write_text(ebnf_str)
                    
                    # We need to compile this EBNF
                    compiled_json_path = None
            except Exception as e:
                raise ValueError(f"Failed to parse JSON file {grammar_path}: {e}")
        elif grammar_path.suffix == '.ebnf':
            ebnf_path = grammar_path
            compiled_json_path = None
        elif grammar_path.name.endswith('.json.gz'):
             # Precompiled
             ebnf_path = None
             compiled_json_path = grammar_path
        else:
            raise ValueError(f"Unsupported file format: {grammar_path}")

        # 2. Compile EBNF to GrammarConstraint if needed
        if ebnf_path:
            # We need a vocab file for the compiler
            # Create a temp vocab file
            # The compiler expects a JSON map: {"token": id}
            # We have vocab: {id: bytes}
            # We need to decode bytes to string. This might be lossy for raw bytes tokens!
            # But the compiler handles "Ġ" etc replacement.
            # Let's try to reconstruct the vocab map.
            
            vocab_map = {}
            for tid, tbytes in vocab.items():
                # Try to decode utf-8, fallback to latin1 or repr?
                # The compiler expects UTF-8 strings usually.
                # GPT-2 vocab is byte-level BPE.
                try:
                    tstr = tbytes.decode('utf-8')
                    # The compiler replaces Ġ with space, etc.
                    # We should provide the raw string if possible?
                    # Actually, the compiler expects the string representation that matches the tokenizer.
                    # For GPT2, we might need to handle the byte encoder.
                    # But for now, let's assume standard utf-8 decoding works for most tokens.
                    vocab_map[tstr] = tid
                except UnicodeDecodeError:
                    # Skip tokens that aren't valid utf-8?
                    # Or use a placeholder?
                    pass
            
            fd, vocab_path_str = tempfile.mkstemp(suffix=".json", text=True)
            os.close(fd)
            vocab_path = Path(vocab_path_str)
            with open(vocab_path, 'w') as f:
                json.dump(vocab_map, f)
                
            # Output path
            fd, output_path_str = tempfile.mkstemp(suffix=".json", text=True)
            os.close(fd)
            output_path = Path(output_path_str)
            
            # Run compiler
            # Assuming 'grammar-compiler' binary is in target/release/grammar-compiler
            compiler_bin = PROJECT_ROOT / "target" / "release" / "grammar-compiler"
            if not compiler_bin.exists():
                # Try debug
                compiler_bin = PROJECT_ROOT / "target" / "debug" / "grammar-compiler"
                
            if not compiler_bin.exists():
                # Try running via cargo
                cmd = ["cargo", "run", "--release", "--bin", "grammar-compiler", "--", 
                       "--grammar", str(ebnf_path), 
                       "--vocab", str(vocab_path), 
                       "--output", str(output_path)]
            else:
                cmd = [str(compiler_bin), 
                       "--grammar", str(ebnf_path), 
                       "--vocab", str(vocab_path), 
                       "--output", str(output_path)]
                       
            # print(f"Running compiler: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
            
            # Cleanup temp files
            vocab_path.unlink()
            if grammar_path.suffix == '.json' and not grammar_path.name.endswith('.json.gz'):
                 # We created the ebnf file
                 # ebnf_path.unlink() # Keep for debugging
                 pass
            
            if result.returncode != 0:
                print(f"Compiler STDOUT: {result.stdout}")
                print(f"Compiler STDERR: {result.stderr}")
                print(f"EBNF Content:\n{ebnf_path.read_text()}")
                raise RuntimeError(f"Compiler failed:\n{result.stderr}")
                
            compiled_json_path = output_path
            
            # Cleanup EBNF if success
            if grammar_path.suffix == '.json' and not grammar_path.name.endswith('.json.gz'):
                ebnf_path.unlink()
            
        # 3. Load the compiled constraint
        if compiled_json_path.name.endswith('.gz'):
            with gzip.open(compiled_json_path, 'rt', encoding='utf-8') as f:
                json_str = f.read()
        else:
            json_str = compiled_json_path.read_text(encoding='utf-8')
            
        # Cleanup output file if we created it
        if ebnf_path:
            compiled_json_path.unlink()

        elapsed = time.perf_counter() - start_time
        
        return CompilationResult(
            compiled=json_str,
            compilation_time_sec=elapsed,
            metadata={"source": str(grammar_path)}
        )

    def create_state(self, compiled: Any) -> Any:
        # compiled is the JSON string
        return RustModel.from_json_string(compiled)

    def get_mask(self, state: Any) -> MaskResult:
        model: RustModel = state
        
        start = time.perf_counter()
        mask_bitset = model.get_mask()
        elapsed = time.perf_counter() - start
        
        # sep1 returns [start, end] inclusive ranges
        ranges = mask_bitset.to_ranges()
        
        valid_tokens = []
        # If to_ranges returns inclusive [start, end], then range(start, end + 1) is correct.
        # Assuming inclusive for now based on previous code.
        for start_idx, end_idx in ranges:
            valid_tokens.extend(range(start_idx, end_idx + 1))
            
        return MaskResult(
            valid_token_ids=valid_tokens,
            time_sec=elapsed
        )

    def commit(self, state: Any, token_id: int) -> CommitResult:
        model: RustModel = state
        
        start = time.perf_counter()
        model.commit(token_id)
        elapsed = time.perf_counter() - start
        
        return CommitResult(
            new_state=model, # State is mutated in place
            time_sec=elapsed
        )

    def supports_grammar_format(self, format: str) -> bool:
        return format in ["sep1_json", "sep1_gz", "json_schema", "ebnf"]

