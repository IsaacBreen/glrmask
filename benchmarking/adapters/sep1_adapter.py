import time
import json
import gzip
from typing import List, Tuple, Dict
import _sep1 as ffi
from benchmarking.core.framework import AbstractBenchmarkAdapter
from benchmarking.core.tokenizer import greedy_tokenizer

class Sep1Adapter(AbstractBenchmarkAdapter):
    def __init__(self):
        self.constraint = None
        self.constraint_state = None
        self.id_to_token = {}

    def load_grammar(self, grammar_path: str) -> float:
        start_time = time.perf_counter()
        
        # Load JSON content
        if grammar_path.endswith('.gz'):
            with gzip.open(grammar_path, 'rt', encoding='utf-8') as f:
                json_str = f.read()
        else:
            with open(grammar_path, 'r', encoding='utf-8') as f:
                json_str = f.read()
                
        # Parse JSON to get vocab
        data = json.loads(json_str)
        self.id_to_token = {v: bytes(k) for k, v in data['original_llm_vocab']['llm_token_map']}
        
        # Load Rust constraint
        self.constraint = ffi.GrammarConstraint.from_json_string(json_str)
        self.constraint_state = ffi.GrammarConstraintState(self.constraint)
        
        return time.perf_counter() - start_time

    def tokenize(self, input_bytes: bytes) -> List[int]:
        return greedy_tokenizer(input_bytes, self.id_to_token)

    def get_mask(self) -> Tuple[List[Tuple[int, int]], float]:
        start_time = time.perf_counter()
        mask_bv = self.constraint_state.get_mask_bv()
        # Convert bitvector to ranges
        # Note: to_ranges() returns a list of (start, end) tuples
        ranges = mask_bv.to_ranges()
        duration = time.perf_counter() - start_time
        return ranges, duration

    def commit(self, token_id: int) -> float:
        start_time = time.perf_counter()
        self.constraint_state.commit(token_id)
        return time.perf_counter() - start_time

    def reset(self):
        if self.constraint:
            self.constraint_state = ffi.GrammarConstraintState(self.constraint)
