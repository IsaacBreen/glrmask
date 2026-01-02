import sys
from pathlib import Path
from typing import Any, List, Tuple
import time
import json
import gzip
import os

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.systems.base import BaseSystem, CompilationResult, MaskResult, CommitResult, time_function

# Import the Rust model
try:
    from python.aug25.models.rust_model import Model as RustModel
    import _sep1 as ffi
except ImportError as e:
    print(f"Warning: Could not import sep1 Rust model: {e}")
    RustModel = None
    ffi = None

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
        if ffi is None:
             raise ImportError("sep1 native module not loaded")

        start_time = time.perf_counter()
        
        # 1. Determine input format and get/compile constraint
        constraint = None
        
        if grammar_path.name.endswith('.json.gz'):
             # Precompiled constraint JSON
             with gzip.open(grammar_path, 'rt', encoding='utf-8') as f:
                json_str = f.read()
             # Return as string, create_state handles it
             return CompilationResult(compiled=json_str, compilation_time_sec=0, metadata={})
             
        elif grammar_path.suffix == '.ebnf':
             # EBNF File
             gd = ffi.GrammarDefinition.from_ebnf_file(str(grammar_path))
             # gd.optimize()  # Optimization disabled by default
             cg = gd.compile()
             
             token_to_id = {v: k for k, v in vocab.items()}
             max_id = max(vocab.keys()) if vocab else 0
             constraint = ffi.GrammarConstraint(cg, token_to_id, max_id)
             
        elif grammar_path.suffix == '.json':
             # Schema
             with open(grammar_path, 'r') as f:
                 schema_str = f.read()
                 
             # Check if it is precompiled constraint (has llm_token_map)
             try:
                 data = json.loads(schema_str)
                 if "llm_token_map" in data:
                     # Precompiled constraint JSON
                     return CompilationResult(compiled=schema_str, compilation_time_sec=0, metadata={})
             except:
                 pass
             
             # Native compile from JSON Schema
             gd = ffi.grammar_definition_from_json_schema(schema_str)
             # gd.optimize()  # Optimization disabled by default
             cg = gd.compile()
             
             token_to_id = {v: k for k, v in vocab.items()}
             max_id = max(vocab.keys()) if vocab else 0
             constraint = ffi.GrammarConstraint(cg, token_to_id, max_id)
             
        else:
            raise ValueError(f"Unsupported file format: {grammar_path}")

        compilation_time = time.perf_counter() - start_time
        
        return CompilationResult(
            compiled=constraint,
            compilation_time_sec=compilation_time,
            metadata={"source": str(grammar_path)}
        )

    def create_state(self, compiled: Any) -> Any:
        # compiled can be JSON string or ffi.GrammarConstraint
        if isinstance(compiled, str):
            return RustModel.from_json_string(compiled)
        elif ffi and isinstance(compiled, ffi.GrammarConstraint):
            state = ffi.GrammarConstraintState(compiled)
            return RustModel(compiled, state)
        else:
            raise TypeError(f"Unknown compiled type: {type(compiled)}")

    def get_mask(self, state: Any) -> MaskResult:
        model: RustModel = state
        
        start = time.perf_counter()
        mask_bitset = model.get_mask()
        elapsed = time.perf_counter() - start
        
        # sep1 returns [start, end] inclusive ranges
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
        return format in ["sep1_json", "sep1_gz", "json_schema", "ebnf"]