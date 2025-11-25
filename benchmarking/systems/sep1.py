"""sep1 wrapper updated for JSON Schema support.

Since our system currently uses EBNF grammars, we need to either:
1. Add JSON Schema support to sep1
2. Or use EBNF grammars for benchmarking instead

For now, documenting the interface.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
import json

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


class Sep1System(BaseSystem):
    """Wrapper for sep1 - our native Rust implementation.
    
    Note: Currently sep1 uses precompiled EBNF grammars in JSON format.
    For JSON Schema benchmarking, we would need to either:
    1. Add JSON Schema -> EBNF conversion
    2. Or use EBNF grammars directly
    
    For now, this is a placeholder showing what the interface would look like.
    """
    
    @property
    def name(self) -> str:
        return "sep1"
    
    def compile_grammar(
        self,
        grammar_path: Path,
        vocab: Dict[int, bytes],
        **kwargs
    ) -> CompilationResult:
        """Placeholder - would need to convert JSON Schema to our format."""
        raise NotImplementedError(
            "sep1 JSON Schema support not yet implemented. "
            "Use EBNF grammars instead."
        )
    
    def create_state(self, compiled: Any) -> Any:
        """Create initial state."""
        raise NotImplementedError("See compile_grammar")
    
    def get_mask(self, state: Any) -> MaskResult:
        """Get mask."""
        raise NotImplementedError("See compile_grammar")
    
    def commit(self, state: Any, token_id: int) -> CommitResult:
        """Commit token."""
        raise NotImplementedError("See compile_grammar")
    
    def supports_grammar_format(self, format: str) -> bool:
        """sep1 supports EBNF (precompiled to JSON)."""
        return format in ['ebnf', 'precompiled_json']
