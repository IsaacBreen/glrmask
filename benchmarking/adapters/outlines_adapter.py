import time
import json
from typing import List, Tuple
try:
    import outlines
except ImportError:
    outlines = None

from benchmarking.core.framework import AbstractBenchmarkAdapter
from benchmarking.core.tokenizer import greedy_tokenizer

class OutlinesAdapter(AbstractBenchmarkAdapter):
    def __init__(self):
        self.model = None
        self.fsm = None
        self.state = None
        self.id_to_token = {}

    def load_grammar(self, grammar_path: str) -> float:
        if outlines is None:
            raise ImportError("outlines library not installed")

        start_time = time.perf_counter()
        
        with open(grammar_path, 'r') as f:
            data = json.load(f)
        self.id_to_token = {v: bytes(k, 'utf-8') if isinstance(k, str) else bytes(k) 
                           for k, v in data.get('original_llm_vocab', {}).get('llm_token_map', [])}

        # outlines.models.transformers...
        # self.fsm = outlines.fsm.guide.RegexGuide(...)
        
        time.sleep(0.1)
        return time.perf_counter() - start_time

    def tokenize(self, input_bytes: bytes) -> List[int]:
        return greedy_tokenizer(input_bytes, self.id_to_token)

    def get_mask(self) -> Tuple[List[Tuple[int, int]], float]:
        start_time = time.perf_counter()
        # mask = self.fsm.get_next_instruction(self.state)
        ranges = []
        return ranges, time.perf_counter() - start_time

    def commit(self, token_id: int) -> float:
        start_time = time.perf_counter()
        # self.state = self.fsm.get_next_state(self.state, token_id)
        return time.perf_counter() - start_time

    def reset(self):
        # self.state = 0
        pass
