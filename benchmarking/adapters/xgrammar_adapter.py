import time
import json
from typing import List, Tuple
try:
    import xgrammar
except ImportError:
    xgrammar = None

from benchmarking.core.framework import AbstractBenchmarkAdapter
from benchmarking.core.tokenizer import greedy_tokenizer

class XGrammarAdapter(AbstractBenchmarkAdapter):
    def __init__(self):
        self.compiler = None
        self.matcher = None
        self.id_to_token = {}

    def load_grammar(self, grammar_path: str) -> float:
        if xgrammar is None:
            raise ImportError("xgrammar library not installed")

        start_time = time.perf_counter()
        
        with open(grammar_path, 'r') as f:
            data = json.load(f)
        self.id_to_token = {v: bytes(k, 'utf-8') if isinstance(k, str) else bytes(k) 
                           for k, v in data.get('original_llm_vocab', {}).get('llm_token_map', [])}

        # self.compiler = xgrammar.GrammarCompiler(...)
        # self.matcher = self.compiler.compile(...)
        
        time.sleep(0.1)
        return time.perf_counter() - start_time

    def tokenize(self, input_bytes: bytes) -> List[int]:
        return greedy_tokenizer(input_bytes, self.id_to_token)

    def get_mask(self) -> Tuple[List[Tuple[int, int]], float]:
        start_time = time.perf_counter()
        # mask = self.matcher.get_mask()
        ranges = []
        return ranges, time.perf_counter() - start_time

    def commit(self, token_id: int) -> float:
        start_time = time.perf_counter()
        # self.matcher.accept_token(token_id)
        return time.perf_counter() - start_time

    def reset(self):
        # self.matcher.reset()
        pass
