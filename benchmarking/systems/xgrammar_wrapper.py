"""XGrammar system wrapper for JSON Schema benchmarking."""

import sys
import time
from pathlib import Path
from typing import Any, Dict

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
    from xgrammar import GrammarCompiler, TokenizerInfo, GrammarMatcher
    from transformers import AutoTokenizer
    XGRAMMAR_AVAILABLE = True
    XGRAMMAR_ERROR = None
except Exception as e:
    XGRAMMAR_AVAILABLE = False
    XGRAMMAR_ERROR = str(e)
    print(f"Warning: xgrammar not available: {e}")


class XGrammarSystem(BaseSystem):
    """Wrapper for XGrammar with JSON Schema support."""
    
    def __init__(self, tokenizer_name="gpt2"):
        if not XGRAMMAR_AVAILABLE:
            raise RuntimeError("XGrammar not available")
        
        # Initialize tokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_info = TokenizerInfo.from_huggingface(hf_tokenizer)
        self.compiler = GrammarCompiler(self.tokenizer_info, max_threads=8)
        
    @property
    def name(self) -> str:
        return "xgrammar"
    
    def compile_grammar(
        self,
        grammar_path: Path,
        vocab: Dict[int, bytes],
        **kwargs
    ) -> CompilationResult:
        """Compile JSON schema using XGrammar."""
        import json
        
        def compile():
            with open(grammar_path, 'r') as f:
                schema = json.load(f)
            
            # Compile with XGrammar
            compiled = self.compiler.compile_json_schema(
                schema,
                indent=None,  # Compact JSON
                strict_mode=True
            )
            return compiled
        
        compiled, compilation_time = time_function(compile)
        
        return CompilationResult(
            compiled=compiled,
            compilation_time_sec=compilation_time,
            metadata={
                'format': 'json_schema',
                'memory_bytes': compiled.memory_size_bytes
            }
        )
    
    def create_state(self, compiled: Any) -> Any:
        """Create XGrammar matcher."""
        matcher = GrammarMatcher(compiled)
        return matcher
    
    def get_mask(self, state: Any) -> MaskResult:
        """Get valid token bitmask."""
        def get_bitmask():
            # XGrammar returns a bitmask
            bitmask = state.get_next_token_bitmask()
            return bitmask
        
        bitmask, elapsed = time_function(get_bitmask)
        
        # Convert bitmask to list of token IDs
        valid_tokens = []
        for i in range(len(bitmask)):
            if bitmask[i]:
                valid_tokens.append(i)
        
        return MaskResult(
            valid_token_ids=valid_tokens,
            time_sec=elapsed,
            metadata={'num_valid': len(valid_tokens)}
        )
    
    def commit(self, state: Any, token_id: int) -> CommitResult:
        """Accept token in matcher."""
        def accept_token():
            state.accept_token(token_id)
        
        _, elapsed = time_function(accept_token)
        
        return CommitResult(
            new_state=state,
            time_sec=elapsed
        )
    
    def supports_grammar_format(self, format: str) -> bool:
        """XGrammar supports JSON schema and EBNF."""
        return format in ['json_schema', 'ebnf']
