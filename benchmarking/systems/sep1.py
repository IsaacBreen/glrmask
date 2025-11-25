import sys
from pathlib import Path
from typing import Any, List, Tuple
import time
import json
import gzip

# Add project root to sys.path to allow importing python.aug25
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.systems.base import BaseSystem, CompilationResult, MaskResult, CommitResult, time_function

# Import the Rust model
# We try to import the rust_model from python.aug25.models
# This assumes the C++ extension _sep1 is available in the python path
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
        """
        For sep1, 'compilation' usually means loading the precomputed JSON.
        If the input is a raw grammar file (EBNF), we would need to run the full compilation pipeline.
        For now, we assume the input IS the precomputed JSON/GZ file, as per our current workflow.
        
        TODO: Implement EBNF -> JSON compilation if needed.
        """
        start_time = time.perf_counter()
        
        # Check if it's already a compiled JSON
        if grammar_path.suffix == '.json' or grammar_path.name.endswith('.json.gz'):
            if grammar_path.name.endswith('.gz'):
                with gzip.open(grammar_path, 'rt', encoding='utf-8') as f:
                    json_str = f.read()
            else:
                json_str = grammar_path.read_text(encoding='utf-8')
                
            # We don't actually "compile" here, we just verify we can load it?
            # Or we return the JSON string as the "compiled" artifact?
            # The RustModel.from_json_string takes the string.
            compiled = json_str
            
        else:
            raise ValueError(f"sep1 currently expects a precompiled .json or .json.gz file, got {grammar_path}")

        elapsed = time.perf_counter() - start_time
        
        return CompilationResult(
            compiled=compiled,
            compilation_time_sec=elapsed,
            metadata={"source": str(grammar_path)}
        )

    def create_state(self, compiled: Any) -> Any:
        # compiled is the JSON string
        return RustModel.from_json_string(compiled)

    def get_mask(self, state: Any) -> MaskResult:
        model: RustModel = state
        
        start = time.perf_counter()
        # get_mask returns a Bitset (from ffi_bitset.py)
        mask_bitset = model.get_mask()
        elapsed = time.perf_counter() - start
        
        # Convert to list of integers for the result
        # The Bitset has a .to_ranges() method, we can convert that to a flat list
        ranges = mask_bitset.to_ranges()
        valid_tokens = []
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
        # We support our own precompiled format
        return format in ["sep1_json", "sep1_gz"]
