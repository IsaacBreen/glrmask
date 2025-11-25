"""llguidance system wrapper for JSON Schema benchmarking."""

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
    from llguidance import JsonCompiler, LLTokenizer, LLInterpreter
    from transformers import AutoTokenizer
    LLGUIDANCE_AVAILABLE = True
    LLGUIDANCE_ERROR = None
except Exception as e:
    LLGUIDANCE_AVAILABLE = False
    LLGUIDANCE_ERROR = str(e)
    print(f"Warning: llguidance not available: {e}")


class LLGuidanceSystem(BaseSystem):
    """Wrapper for llguidance with JSON Schema support."""
    
    def __init__(self, tokenizer_name="gpt2"):
        if not LLGUIDANCE_AVAILABLE:
            raise RuntimeError(f"llguidance not available: {LLGUIDANCE_ERROR}")
        
        # Initialize tokenizer chain: HF -> TokenizerWrapper -> LLToken izer
        from llguidance import TokenizerWrapper
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Create wrapper that adds .tokens attribute
        class HFTokenizerWithTokens:
            def __init__(self, hf_tok):
                self._hf_tok = hf_tok
                # Tokens must be bytes for llguidance
                self.tokens = [hf_tok.convert_ids_to_tokens(i).encode('utf-8') if isinstance(hf_tok.convert_ids_to_tokens(i), str) else hf_tok.convert_ids_to_tokens(i) for i in range(hf_tok.vocab_size)]
                self.eos_token_id = hf_tok.eos_token_id
                self.bos_token_id = getattr(hf_tok, 'bos_token_id', None)
            
            def __getattr__(self, name):
                return getattr(self._hf_tok, name)
            
            def __call__(self, text):
                if isinstance(text, bytes):
                    text = text.decode('utf-8')
                return self._hf_tok.encode(text)
        
        wrapped_tokenizer = HFTokenizerWithTokens(hf_tokenizer)
        tokenizer_wrapper = TokenizerWrapper(wrapped_tokenizer)
        self.ll_tokenizer = LLTokenizer(tokenizer_wrapper)
        self.hf_tokenizer = hf_tokenizer
        
    @property
    def name(self) -> str:
        return "llguidance"
    
    def compile_grammar(
        self,
        grammar_path: Path,
        vocab: Dict[int, bytes],
        **kwargs
    ) -> CompilationResult:
        """Compile JSON schema using llguidance."""
        import json
        
        def compile():
            with open(grammar_path, 'r') as f:
                schema = json.load(f)
            
            # Compile with llguidance JsonCompiler
            compiler = JsonCompiler()
            # Convert schema to JSON string for llguidance
            schema_str = json.dumps(schema)
            compiled = compiler.compile(schema_str)
            
            return compiled
        
        compiled, compilation_time = time_function(compile)
        
        return CompilationResult(
            compiled=compiled,
            compilation_time_sec=compilation_time,
            metadata={'format': 'json_schema'}
        )
    
    def create_state(self, compiled: Any) -> Any:
        """Create llguidance interpreter."""
        from llguidance import LLInterpreter
        
        # Create interpreter with compiled grammar and tokenizer
        interpreter = LLInterpreter(self.ll_tokenizer, compiled)
        return interpreter
    
    def get_mask(self, state: Any) -> MaskResult:
        """Get valid token mask."""
        def get_bitmask():
            # llguidance returns mask via get_mask()
            mask = state.get_mask()
            return mask
        
        mask, elapsed = time_function(get_bitmask)
        
        # Convert mask to list of valid token IDs
        # Assuming mask is a list/array of booleans or integers
        if hasattr(mask, '__iter__'):
            valid_tokens = [i for i, v in enumerate(mask) if v]
        else:
            # Fallback - might need to adjust based on actual return type
            valid_tokens = []
        
        return MaskResult(
            valid_token_ids=valid_tokens,
            time_sec=elapsed,
            metadata={'num_valid': len(valid_tokens)}
        )
    
    def commit(self, state: Any, token_id: int) -> CommitResult:
        """Commit token to interpreter."""
        def commit_token():
            state.commit_token(token_id)
        
        _, elapsed = time_function(commit_token)
        
        return CommitResult(
            new_state=state,
            time_sec=elapsed
        )
    
    def supports_grammar_format(self, format: str) -> bool:
        """llguidance supports JSON schema, Lark grammar, regex."""
        return format in ['json_schema', 'lark', 'regex']
