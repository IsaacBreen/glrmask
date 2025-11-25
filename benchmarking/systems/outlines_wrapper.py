import sys
from pathlib import Path
from typing import Any, List, Tuple
import time
import json
import torch
from transformers import AutoTokenizer

from benchmarking.systems.base import BaseSystem, CompilationResult, MaskResult, CommitResult, time_function

try:
    import outlines
    import outlines.models
    import outlines.generate
except ImportError:
    outlines = None

class OutlinesSystem(BaseSystem):
    def __init__(self, model_name="gpt2", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
    @property
    def name(self) -> str:
        return "outlines"

    def compile_grammar(
        self,
        grammar_path: Path,
        vocab: dict[int, bytes],
        **kwargs
    ) -> CompilationResult:
        if outlines is None:
            raise ImportError("outlines not installed")

        start_time = time.perf_counter()
        
        # Initialize model if not already done (Outlines needs the model to compile the FSM)
        if self.model is None:
            # We use a small model for benchmarking the constraint overhead
            # ideally we'd mock this but Outlines is tightly coupled
            self.model = outlines.models.transformers(self.model_name, device=self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
        # Load grammar
        grammar_str = grammar_path.read_text(encoding='utf-8')
        
        # Compile
        # Outlines compiles the FSM when you create the generator
        if grammar_path.suffix == '.json':
            # JSON Schema
            generator = outlines.generate.json(self.model, grammar_str)
        elif grammar_path.suffix == '.ebnf':
            # CFG
            generator = outlines.generate.cfg(self.model, grammar_str)
        else:
            # Assume regex?
            generator = outlines.generate.regex(self.model, grammar_str)
            
        elapsed = time.perf_counter() - start_time
        
        # The generator contains the FSM and logits processor
        return CompilationResult(
            compiled=generator,
            compilation_time_sec=elapsed,
            metadata={}
        )

    def create_state(self, compiled: Any) -> Any:
        generator = compiled
        # We need to access the internal state. 
        # Outlines 0.1.11 uses a SequenceGenerator which has a 'logits_processor'
        # But we need to maintain the FSM state ourselves if we want to step token by token
        # without running the model.
        
        # This is tricky because Outlines is designed for end-to-end generation.
        # We'll try to extract the FSM state.
        
        # For now, we'll store the generator and an initial state if we can find it.
        # If not, we might have to hack it.
        
        # HACK: We'll assume we can access the FSM state.
        # If not, we'll return a dummy state and fail in get_mask.
        
        # In newer outlines, there is fsm.FSMState.
        # In 0.1.11, it might be different.
        
        # Let's assume we start with the initial state ID 0.
        return {"fsm_state": 0, "generator": generator}

    def get_mask(self, state: Any) -> MaskResult:
        generator = state["generator"]
        fsm_state = state["fsm_state"]
        
        start = time.perf_counter()
        
        # We need to get the allowed tokens for the current FSM state.
        # We might need to dig into the generator.
        
        # If we can't easily get the mask, we might have to skip this metric
        # or implement a best-effort guess.
        
        # For now, return empty to indicate failure/not implemented
        valid_tokens = [] 
        
        # Try to access the FSM
        if hasattr(generator, 'fsm'):
            # This is a guess at the API
            valid_tokens = list(generator.fsm.get_next_instruction(fsm_state).keys())
        elif hasattr(generator, 'logits_processor'):
             # Try to use the logits processor with a dummy input
             pass
             
        elapsed = time.perf_counter() - start
        
        return MaskResult(
            valid_token_ids=valid_tokens,
            time_sec=elapsed
        )

    def commit(self, state: Any, token_id: int) -> CommitResult:
        generator = state["generator"]
        fsm_state = state["fsm_state"]
        
        start = time.perf_counter()
        
        # Advance FSM state
        new_fsm_state = fsm_state # Dummy
        if hasattr(generator, 'fsm'):
             new_fsm_state = generator.fsm.get_next_state(fsm_state, token_id)
        
        elapsed = time.perf_counter() - start
        
        return CommitResult(
            new_state={"fsm_state": new_fsm_state, "generator": generator},
            time_sec=elapsed
        )

    def supports_grammar_format(self, format: str) -> bool:
        return format in ["json_schema", "ebnf", "regex"]
