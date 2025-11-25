"""Outlines system wrapper for JSON Schema benchmarking."""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarking.systems.base import (
    BaseSystem,
    CompilationResult,
    MaskResult,
    CommitResult,
    time_function
)

try:
    import outlines
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    print("Warning: outlines not installed. Run: pip install outlines")


class OutlinesSystem(BaseSystem):
    """Wrapper for Outlines library with JSON Schema support."""
    
    def __init__(self):
        if not OUTLINES_AVAILABLE:
            raise RuntimeError("outlines library not available")
        self.generator = None
        
    @property
    def name(self) -> str:
        return "outlines"
    
    def compile_grammar(
        self,
        grammar_path: Path,
        vocab: Dict[int, bytes],
        **kwargs
    ) -> CompilationResult:
        """For Outlines with JSON schema, grammar_path should point to a JSON schema file."""
        import json
        
        def compile():
            # Load JSON schema
            with open(grammar_path, 'r') as f:
                schema = json.load(f)
            
            # Outlines doesn't have separate compilation step for schemas
            # It compiles on-the-fly during generation
            # So we just store the schema
            return schema
        
        compiled, compilation_time = time_function(compile)
        
        return CompilationResult(
            compiled=compiled,
            compilation_time_sec=compilation_time,
            metadata={'format': 'json_schema'}
        )
    
    def create_state(self, compiled: Any) -> Any:
        """Create Outlines generator with schema."""
        # Note: Outlines requires a model to create generator
        # For benchmarking constraint computation only, we'll need to mock this
        # OR use a small model
        
        # For now, return the schema itself as "state"
        # Actual integration would require model setup
        return {
            'schema': compiled,
            'position': 0,
            'generated_so_far': ""
        }
    
    def get_mask(self, state: Any) -> MaskResult:
        """Get valid tokens for current state.
        
        Note: Outlines doesn't expose get_mask directly in the same way.
        This would require accessing internal FSM Guide.
        For benchmarking, we need to either:
        1. Use Outlines' internal APIs
        2. Or measure end-to-end generation time instead
        """
        # TODO: Access Outlines internals to get mask
        # For now, placeholder
        start = time.perf_counter()
        # Simulated mask computation
        valid_tokens = list(range(100))  # Placeholder
        elapsed = time.perf_counter() - start
        
        return MaskResult(
            valid_token_ids=valid_tokens,
            time_sec=elapsed,
            metadata={'note': 'placeholder - needs Outlines internal API access'}
        )
    
    def commit(self, state: Any, token_id: int) -> CommitResult:
        """Commit token to state."""
        start = time.perf_counter()
        # Update state
        state['position'] += 1
        elapsed = time.perf_counter() - start
        
        return CommitResult(
            new_state=state,
            time_sec=elapsed
        )
    
    def supports_grammar_format(self, format: str) -> bool:
        """Outlines supports JSON schema, CFG, regex."""
        return format in ['json_schema', 'cfg', 'regex']
