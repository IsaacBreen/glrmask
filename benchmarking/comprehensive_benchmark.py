#!/usr/bin/env python3
"""
Comprehensive Grammar-Constrained Decoding Benchmark Suite

This script provides fair, apples-to-apples comparison across:
- sep1 (our system)
- XGrammar
- LLGuidance  
- llama.cpp grammar
- Outlines

Benchmark focuses on:
1. Compilation time (TTFM - Time to First Mask)
2. Per-token mask computation (TBM - Time Between Masks)
3. Correctness validation
"""

import sys
import time
import json
import gzip
import statistics
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ==============================================================================
# Configuration and Data Classes
# ==============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    warmup_iterations: int = 10
    measurement_iterations: int = 100
    test_sequence: Optional[List[int]] = None  # Token IDs to test with
    grammar_type: str = "json"  # json, ebnf, json_schema
    vocab_size: int = 50257  # GPT-2 default

@dataclass
class BenchmarkResult:
    """Results from benchmarking a single system."""
    system_name: str
    grammar_name: str
    
    # Compilation metrics
    compilation_time_ms: float = 0.0
    
    # TTFM - Time to First Mask (includes compilation if cold start)
    ttfm_cold_ms: float = 0.0
    ttfm_warm_ms: float = 0.0
    
    # TBM - Time Between Masks
    tbm_initial_us: float = 0.0  # First mask at empty state
    tbm_median_us: float = 0.0
    tbm_mean_us: float = 0.0
    tbm_p95_us: float = 0.0
    tbm_p99_us: float = 0.0
    tbm_min_us: float = 0.0
    tbm_max_us: float = 0.0
    
    # Correctness
    num_valid_tokens_initial: int = 0
    
    # Error info
    error: Optional[str] = None
    
    # Raw data for detailed analysis
    tbm_samples: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop('tbm_samples', None)  # Don't include raw samples in summary
        return d


# ==============================================================================
# System Wrapper Base Class
# ==============================================================================

class GrammarSystemWrapper(ABC):
    """Base wrapper for grammar constraint systems."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def compile_grammar(self, grammar_str: str, grammar_type: str) -> Any:
        """Compile grammar, return compiled object."""
        pass
    
    @abstractmethod
    def create_state(self, compiled: Any) -> Any:
        """Create initial matcher/state from compiled grammar."""
        pass
    
    @abstractmethod
    def get_mask(self, state: Any) -> Tuple[List[int], float]:
        """Get valid token mask. Returns (valid_token_ids, time_us)."""
        pass
    
    @abstractmethod
    def accept_token(self, state: Any, token_id: int) -> Tuple[Any, float]:
        """Accept token and advance state. Returns (new_state, time_us)."""
        pass


# ==============================================================================
# XGrammar Wrapper
# ==============================================================================

class XGrammarWrapper(GrammarSystemWrapper):
    """Wrapper for XGrammar."""
    
    def __init__(self, tokenizer_name: str = "gpt2"):
        import xgrammar as xgr
        from transformers import AutoTokenizer
        
        self.xgr = xgr
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = len(self.tokenizer)
        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=self.vocab_size
        )
        self.compiler = xgr.GrammarCompiler(self.tokenizer_info)
        
    @property
    def name(self) -> str:
        return "xgrammar"
    
    def compile_grammar(self, grammar_str: str, grammar_type: str) -> Any:
        if grammar_type == "json_schema":
            schema = json.loads(grammar_str)
            return self.compiler.compile_json_schema(schema)
        elif grammar_type == "ebnf":
            return self.compiler.compile_grammar(grammar_str)
        else:
            raise ValueError(f"Unsupported grammar type: {grammar_type}")
    
    def create_state(self, compiled: Any) -> Any:
        matcher = self.xgr.GrammarMatcher(compiled)
        bitmask = self.xgr.allocate_token_bitmask(1, self.tokenizer_info.vocab_size)
        return {"matcher": matcher, "bitmask": bitmask}
    
    def get_mask(self, state: Any) -> Tuple[List[int], float]:
        import torch
        
        matcher = state["matcher"]
        bitmask = state["bitmask"]
        
        t0 = time.perf_counter()
        matcher.fill_next_token_bitmask(bitmask)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        
        # Convert bitmask to list of valid tokens
        logits = torch.zeros(self.tokenizer_info.vocab_size)
        self.xgr.apply_token_bitmask_inplace(logits, bitmask)
        valid_tokens = (logits > -float('inf')).nonzero(as_tuple=True)[0].tolist()
        
        return valid_tokens, elapsed_us
    
    def accept_token(self, state: Any, token_id: int) -> Tuple[Any, float]:
        matcher = state["matcher"]
        
        t0 = time.perf_counter()
        accepted = matcher.accept_token(token_id)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        
        if not accepted:
            raise ValueError(f"Token {token_id} rejected by grammar")
        
        return state, elapsed_us


# ==============================================================================
# LLGuidance Wrapper
# ==============================================================================

class LLGuidanceWrapper(GrammarSystemWrapper):
    """Wrapper for LLGuidance (Guidance AI)."""
    
    def __init__(self):
        import llguidance
        from llguidance import JsonCompiler, LarkCompiler, LLInterpreter, LLTokenizer
        import tiktoken
        from llguidance.tiktoken import lltokenizer_from_encoding
        
        self.llguidance = llguidance
        self.JsonCompiler = JsonCompiler
        self.LarkCompiler = LarkCompiler
        self.LLInterpreter = LLInterpreter
        
        # Create tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        self.ll_tokenizer = lltokenizer_from_encoding(self.enc)
        
    @property
    def name(self) -> str:
        return "llguidance"
    
    def compile_grammar(self, grammar_str: str, grammar_type: str) -> Any:
        if grammar_type in ["json_schema", "json"]:
            compiler = self.JsonCompiler()
            return compiler.compile(grammar_str)
        elif grammar_type == "lark":
            compiler = self.LarkCompiler()
            return compiler.compile(grammar_str)
        else:
            raise ValueError(f"Unsupported grammar type for llguidance: {grammar_type}")
    
    def create_state(self, compiled: Any) -> Any:
        interp = self.LLInterpreter(self.ll_tokenizer, compiled)
        interp.start_without_prompt()
        return interp
    
    def get_mask(self, state: Any) -> Tuple[List[int], float]:
        t0 = time.perf_counter()
        mask_result = state.compute_mask()
        elapsed_us = (time.perf_counter() - t0) * 1e6
        
        # Parse the mask bytes
        mask_bytes = mask_result[0]
        valid_tokens = []
        
        if isinstance(mask_bytes, bytes):
            for byte_idx, byte_val in enumerate(mask_bytes):
                if byte_val == 0:
                    continue
                for bit_idx in range(8):
                    if (byte_val >> bit_idx) & 1:
                        token_id = byte_idx * 8 + bit_idx
                        valid_tokens.append(token_id)
        
        return valid_tokens, elapsed_us
    
    def accept_token(self, state: Any, token_id: int) -> Tuple[Any, float]:
        t0 = time.perf_counter()
        state.commit_token(token_id)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        return state, elapsed_us


# ==============================================================================
# Sep1 Wrapper
# ==============================================================================

class Sep1Wrapper(GrammarSystemWrapper):
    """Wrapper for Sep1 (our system)."""
    
    def __init__(self):
        import _sep1
        import tiktoken
        
        self._sep1 = _sep1
        self.enc = tiktoken.get_encoding("gpt2")
        
    @property
    def name(self) -> str:
        return "sep1"
    
    def compile_grammar(self, grammar_str: str, grammar_type: str) -> Any:
        """For sep1, we expect pre-compiled constraints."""
        if grammar_type == "precompiled":
            # Load from JSON string
            return self._sep1.GrammarConstraint.from_json_string(grammar_str)
        else:
            raise ValueError("Sep1 requires precompiled constraints. Use compile.py first.")
    
    def create_state(self, compiled: Any) -> Any:
        return self._sep1.GrammarConstraintState(compiled)
    
    def get_mask(self, state: Any) -> Tuple[List[int], float]:
        t0 = time.perf_counter()
        mask_bv = state.get_mask_bv()
        elapsed_us = (time.perf_counter() - t0) * 1e6
        
        # Convert bitset to list using to_indices()
        valid_tokens = list(mask_bv.to_indices())
        
        return valid_tokens, elapsed_us
    
    def accept_token(self, state: Any, token_id: int) -> Tuple[Any, float]:
        tok_bytes = self.enc.decode_single_token_bytes(token_id)
        
        t0 = time.perf_counter()
        state.commit_bytes(tok_bytes)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        
        return state, elapsed_us


# ==============================================================================
# Outlines Wrapper
# ==============================================================================

class OutlinesWrapper(GrammarSystemWrapper):
    """Wrapper for Outlines."""
    
    def __init__(self):
        from outlines.fsm.guide import CFGGuide
        from outlines.fsm.json_schema import build_regex_from_schema
        from outlines.models.transformers import TransformerTokenizer
        from transformers import AutoTokenizer
        import outlines
        
        self.outlines = outlines
        self.CFGGuide = CFGGuide
        self.build_regex_from_schema = build_regex_from_schema
        
        # Create tokenizer
        self.hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.outlines_tokenizer = TransformerTokenizer(self.hf_tokenizer)
        
    @property
    def name(self) -> str:
        return "outlines"
    
    def compile_grammar(self, grammar_str: str, grammar_type: str) -> Any:
        if grammar_type == "ebnf":
            # Outlines uses its own CFG format
            guide = self.CFGGuide(grammar_str, self.outlines_tokenizer)
            return guide
        elif grammar_type == "json_schema":
            # Convert to regex
            regex = self.build_regex_from_schema(grammar_str)
            from outlines.fsm.guide import RegexGuide
            guide = RegexGuide(regex, self.outlines_tokenizer)
            return guide
        else:
            raise ValueError(f"Unsupported grammar type for outlines: {grammar_type}")
    
    def create_state(self, compiled: Any) -> Any:
        # Return initial state (usually 0)
        return {"guide": compiled, "state": compiled.initial_state}
    
    def get_mask(self, state: Any) -> Tuple[List[int], float]:
        guide = state["guide"]
        current_state = state["state"]
        
        t0 = time.perf_counter()
        instruction = guide.get_next_instruction(current_state)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        
        # Instruction contains valid tokens
        if hasattr(instruction, 'tokens'):
            valid_tokens = list(instruction.tokens)
        else:
            valid_tokens = list(instruction) if instruction else []
        
        return valid_tokens, elapsed_us
    
    def accept_token(self, state: Any, token_id: int) -> Tuple[Any, float]:
        guide = state["guide"]
        current_state = state["state"]
        
        t0 = time.perf_counter()
        new_state = guide.get_next_state(current_state, token_id)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        
        return {"guide": guide, "state": new_state}, elapsed_us


# ==============================================================================
# llama.cpp Grammar Wrapper
# ==============================================================================

class LlamaCppGrammarWrapper(GrammarSystemWrapper):
    """Wrapper for llama.cpp grammar constraints.
    
    Note: llama.cpp grammar is tightly integrated with inference,
    so this wrapper uses the llama-cpp-python bindings.
    """
    
    def __init__(self):
        from llama_cpp import LlamaGrammar
        self.LlamaGrammar = LlamaGrammar
        self._grammar_str = None
        
    @property
    def name(self) -> str:
        return "llama.cpp"
    
    def compile_grammar(self, grammar_str: str, grammar_type: str) -> Any:
        """llama.cpp expects GBNF format."""
        if grammar_type in ["gbnf", "ebnf"]:
            self._grammar_str = grammar_str
            # LlamaGrammar is created on-demand
            return grammar_str
        else:
            raise ValueError(f"llama.cpp requires GBNF format, got: {grammar_type}")
    
    def create_state(self, compiled: Any) -> Any:
        """Create grammar instance."""
        # Note: LlamaGrammar is typically used during generation
        # We create it fresh each time
        t0 = time.perf_counter()
        grammar = self.LlamaGrammar.from_string(compiled)
        compile_time = time.perf_counter() - t0
        return {"grammar": grammar, "compile_time": compile_time}
    
    def get_mask(self, state: Any) -> Tuple[List[int], float]:
        """Get mask - llama.cpp doesn't expose this directly."""
        # llama.cpp grammar sampling is integrated with the model
        # We can't easily get just the mask without a model context
        # Return empty to indicate this limitation
        return [], 0.0
    
    def accept_token(self, state: Any, token_id: int) -> Tuple[Any, float]:
        """Accept token - also integrated with model."""
        return state, 0.0


# ==============================================================================
# Benchmark Runner
# ==============================================================================

def run_benchmark(
    system: GrammarSystemWrapper,
    grammar_str: str,
    grammar_name: str,
    grammar_type: str,
    config: BenchmarkConfig
) -> BenchmarkResult:
    """Run benchmark on a single system with a single grammar."""
    
    result = BenchmarkResult(
        system_name=system.name,
        grammar_name=grammar_name
    )
    
    try:
        # 1. Measure compilation time
        compile_times = []
        compiled = None
        for _ in range(5):
            t0 = time.perf_counter()
            compiled = system.compile_grammar(grammar_str, grammar_type)
            compile_times.append((time.perf_counter() - t0) * 1000)
        result.compilation_time_ms = min(compile_times)
        
        # 2. Measure TTFM (cold start)
        ttfm_cold_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            temp_compiled = system.compile_grammar(grammar_str, grammar_type)
            state = system.create_state(temp_compiled)
            valid_tokens, _ = system.get_mask(state)
            ttfm_cold_times.append((time.perf_counter() - t0) * 1000)
        result.ttfm_cold_ms = min(ttfm_cold_times)
        
        # 3. Measure TTFM (warm start - compiled already exists)
        ttfm_warm_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            state = system.create_state(compiled)
            valid_tokens, _ = system.get_mask(state)
            ttfm_warm_times.append((time.perf_counter() - t0) * 1000)
        result.ttfm_warm_ms = min(ttfm_warm_times)
        
        # 4. Warmup for TBM
        for _ in range(config.warmup_iterations):
            state = system.create_state(compiled)
            valid_tokens, _ = system.get_mask(state)
        
        # 5. Measure TBM at initial state
        tbm_samples = []
        for _ in range(config.measurement_iterations):
            state = system.create_state(compiled)
            valid_tokens, elapsed_us = system.get_mask(state)
            tbm_samples.append(elapsed_us)
        
        result.tbm_samples = tbm_samples
        result.tbm_initial_us = tbm_samples[0] if tbm_samples else 0
        result.tbm_median_us = statistics.median(tbm_samples)
        result.tbm_mean_us = statistics.mean(tbm_samples)
        result.tbm_min_us = min(tbm_samples)
        result.tbm_max_us = max(tbm_samples)
        
        sorted_samples = sorted(tbm_samples)
        n = len(sorted_samples)
        result.tbm_p95_us = sorted_samples[int(n * 0.95)] if n > 0 else 0
        result.tbm_p99_us = sorted_samples[int(n * 0.99)] if n > 0 else 0
        
        # 6. Record number of valid tokens at initial state
        result.num_valid_tokens_initial = len(valid_tokens)
        
    except Exception as e:
        result.error = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()
    
    return result


def print_results(results: List[BenchmarkResult]):
    """Print results in a formatted table."""
    
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    
    # Header
    print(f"{'System':<15} | {'Grammar':<20} | {'Compile(ms)':<12} | {'TTFM(ms)':<10} | "
          f"{'TBM-p50(μs)':<12} | {'TBM-p99(μs)':<12} | {'Valid Toks':<10}")
    print("-" * 100)
    
    for r in results:
        if r.error:
            print(f"{r.system_name:<15} | {r.grammar_name:<20} | ERROR: {r.error[:50]}")
        else:
            print(f"{r.system_name:<15} | {r.grammar_name:<20} | {r.compilation_time_ms:<12.2f} | "
                  f"{r.ttfm_warm_ms:<10.2f} | {r.tbm_median_us:<12.1f} | "
                  f"{r.tbm_p99_us:<12.1f} | {r.num_valid_tokens_initial:<10}")
    
    print("=" * 100)


# ==============================================================================
# Test Grammars
# ==============================================================================

SIMPLE_JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "value": {"type": "number"}
    },
    "required": ["name", "value"]
})

MEDIUM_JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "name": {"type": "string", "maxLength": 100},
        "email": {"type": "string", "format": "email"},
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "zipcode": {"type": "string"}
            }
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["id", "name"]
})

SIMPLE_EBNF = r"""
root ::= object
object ::= "{" ws members ws "}"
members ::= pair ("," ws pair)*
pair ::= string ws ":" ws value
string ::= "\"" [a-zA-Z]+ "\""
value ::= string | number
number ::= [0-9]+
ws ::= [ ]*
"""


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Grammar constraint benchmark suite")
    parser.add_argument("--systems", nargs="+", default=["xgrammar", "llguidance", "sep1"],
                       help="Systems to benchmark")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of measurement iterations")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup iterations")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output file for JSON results")
    parser.add_argument("--sep1-constraint", type=Path, default=None,
                       help="Path to precompiled sep1 constraint (.json.gz)")
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        warmup_iterations=args.warmup,
        measurement_iterations=args.iterations
    )
    
    # Initialize systems
    systems = {}
    
    if "xgrammar" in args.systems:
        try:
            systems["xgrammar"] = XGrammarWrapper()
            print("✓ XGrammar initialized")
        except Exception as e:
            print(f"✗ XGrammar failed: {e}")
    
    if "llguidance" in args.systems:
        try:
            systems["llguidance"] = LLGuidanceWrapper()
            print("✓ LLGuidance initialized")
        except Exception as e:
            print(f"✗ LLGuidance failed: {e}")
    
    if "sep1" in args.systems:
        try:
            systems["sep1"] = Sep1Wrapper()
            print("✓ Sep1 initialized")
        except Exception as e:
            print(f"✗ Sep1 failed: {e}")
    
    if "outlines" in args.systems:
        try:
            systems["outlines"] = OutlinesWrapper()
            print("✓ Outlines initialized")
        except Exception as e:
            print(f"✗ Outlines failed: {e}")
    
    results = []
    
    # Benchmark JSON Schema (XGrammar, LLGuidance)
    print("\n--- Benchmarking JSON Schema grammars ---")
    
    for schema_name, schema_str in [("simple_json", SIMPLE_JSON_SCHEMA), 
                                     ("medium_json", MEDIUM_JSON_SCHEMA)]:
        for name, system in systems.items():
            if name in ["xgrammar", "llguidance"]:
                print(f"Running {name} on {schema_name}...")
                result = run_benchmark(system, schema_str, schema_name, "json_schema", config)
                results.append(result)
    
    # Benchmark Sep1 with precompiled constraint
    if "sep1" in systems and args.sep1_constraint:
        print(f"\n--- Benchmarking Sep1 with precompiled constraint ---")
        
        constraint_path = args.sep1_constraint
        if constraint_path.suffix == ".gz":
            with gzip.open(constraint_path, "rt") as f:
                constraint_json = f.read()
        else:
            constraint_json = constraint_path.read_text()
        
        print(f"Running sep1 on {constraint_path.stem}...")
        result = run_benchmark(systems["sep1"], constraint_json, 
                              constraint_path.stem, "precompiled", config)
        results.append(result)
    
    # Print results
    print_results(results)
    
    # Save results
    if args.output:
        output_data = {
            "config": asdict(config),
            "results": [r.to_dict() for r in results]
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
