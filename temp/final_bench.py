#!/usr/bin/env python3
"""
Final Fair Benchmark: Focus on the key metrics for paper.

This produces clean, fair comparison data for the paper:
1. Initial mask (first get_mask after compilation)
2. Steady-state mask (after warmup)
3. Tail latency (p99/p50)

All systems tested on the SAME grammars where possible.
"""

import time
import statistics
import json
import gzip
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

import tiktoken

@dataclass
class Result:
    system: str
    grammar: str
    compile_ms: float
    p50_us: float
    p95_us: float
    p99_us: float
    min_us: float
    max_us: float
    valid_tokens: int
    
    @property
    def tail_ratio(self) -> float:
        return self.p99_us / self.p50_us if self.p50_us > 0 else 0


def bench_xgrammar_schema(schema: dict, name: str, n_warmup=200, n_iter=1000) -> Result:
    """XGrammar with JSON schema."""
    import xgrammar
    
    enc = tiktoken.get_encoding("gpt2")
    vocab = [enc.decode_single_token_bytes(i).decode('utf-8', errors='replace') 
             for i in range(enc.n_vocab)]
    tokenizer_info = xgrammar.TokenizerInfo(vocab, vocab_type=xgrammar.VocabType.BYTE_FALLBACK)
    compiler = xgrammar.GrammarCompiler(tokenizer_info)
    
    t0 = time.perf_counter()
    compiled = compiler.compile_json_schema(json.dumps(schema))
    compile_ms = (time.perf_counter() - t0) * 1000
    
    matcher = xgrammar.GrammarMatcher(compiled)
    bitmask = xgrammar.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    
    # Warmup
    for _ in range(n_warmup):
        matcher.reset()
        matcher.fill_next_token_bitmask(bitmask)
    
    # Measure
    times = []
    for _ in range(n_iter):
        matcher.reset()
        t0 = time.perf_counter()
        matcher.fill_next_token_bitmask(bitmask)
        times.append((time.perf_counter() - t0) * 1e6)
    
    # Count valid
    mask_np = bitmask.numpy()
    valid = sum(bin(v).count('1') for v in mask_np.flat)
    
    times.sort()
    return Result(
        system="XGrammar",
        grammar=name,
        compile_ms=compile_ms,
        p50_us=times[len(times)//2],
        p95_us=times[int(len(times)*0.95)],
        p99_us=times[int(len(times)*0.99)],
        min_us=times[0],
        max_us=times[-1],
        valid_tokens=valid
    )


def bench_xgrammar_builtin_json(n_warmup=200, n_iter=1000) -> Result:
    """XGrammar builtin JSON grammar."""
    import xgrammar
    
    enc = tiktoken.get_encoding("gpt2")
    vocab = [enc.decode_single_token_bytes(i).decode('utf-8', errors='replace') 
             for i in range(enc.n_vocab)]
    tokenizer_info = xgrammar.TokenizerInfo(vocab, vocab_type=xgrammar.VocabType.BYTE_FALLBACK)
    compiler = xgrammar.GrammarCompiler(tokenizer_info)
    
    t0 = time.perf_counter()
    compiled = compiler.compile_builtin_json_grammar()
    compile_ms = (time.perf_counter() - t0) * 1000
    
    matcher = xgrammar.GrammarMatcher(compiled)
    bitmask = xgrammar.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    
    for _ in range(n_warmup):
        matcher.reset()
        matcher.fill_next_token_bitmask(bitmask)
    
    times = []
    for _ in range(n_iter):
        matcher.reset()
        t0 = time.perf_counter()
        matcher.fill_next_token_bitmask(bitmask)
        times.append((time.perf_counter() - t0) * 1e6)
    
    mask_np = bitmask.numpy()
    valid = sum(bin(v).count('1') for v in mask_np.flat)
    
    times.sort()
    return Result(
        system="XGrammar",
        grammar="builtin_json",
        compile_ms=compile_ms,
        p50_us=times[len(times)//2],
        p95_us=times[int(len(times)*0.95)],
        p99_us=times[int(len(times)*0.99)],
        min_us=times[0],
        max_us=times[-1],
        valid_tokens=valid
    )


def bench_xgrammar_ebnf(grammar: str, name: str, n_warmup=200, n_iter=1000) -> Result:
    """XGrammar with EBNF grammar."""
    import xgrammar
    
    enc = tiktoken.get_encoding("gpt2")
    vocab = [enc.decode_single_token_bytes(i).decode('utf-8', errors='replace') 
             for i in range(enc.n_vocab)]
    tokenizer_info = xgrammar.TokenizerInfo(vocab, vocab_type=xgrammar.VocabType.BYTE_FALLBACK)
    compiler = xgrammar.GrammarCompiler(tokenizer_info)
    
    t0 = time.perf_counter()
    grammar_obj = xgrammar.Grammar.from_ebnf(grammar, root_rule_name="root")
    compiled = compiler.compile_grammar(grammar_obj)
    compile_ms = (time.perf_counter() - t0) * 1000
    
    matcher = xgrammar.GrammarMatcher(compiled)
    bitmask = xgrammar.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    
    for _ in range(n_warmup):
        matcher.reset()
        matcher.fill_next_token_bitmask(bitmask)
    
    times = []
    for _ in range(n_iter):
        matcher.reset()
        t0 = time.perf_counter()
        matcher.fill_next_token_bitmask(bitmask)
        times.append((time.perf_counter() - t0) * 1e6)
    
    mask_np = bitmask.numpy()
    valid = sum(bin(v).count('1') for v in mask_np.flat)
    
    times.sort()
    return Result(
        system="XGrammar",
        grammar=name,
        compile_ms=compile_ms,
        p50_us=times[len(times)//2],
        p95_us=times[int(len(times)*0.95)],
        p99_us=times[int(len(times)*0.99)],
        min_us=times[0],
        max_us=times[-1],
        valid_tokens=valid
    )


def bench_llguidance_schema(schema: dict, name: str, n_warmup=200, n_iter=1000) -> Result:
    """LLGuidance with JSON schema."""
    import llguidance
    from llguidance.tiktoken import lltokenizer_from_encoding
    
    enc = tiktoken.get_encoding("gpt2")
    tokenizer = lltokenizer_from_encoding(enc)
    
    t0 = time.perf_counter()
    compiler = llguidance.JsonCompiler()
    grammar = compiler.compile(json.dumps(schema))
    compile_ms = (time.perf_counter() - t0) * 1000
    
    # Warmup
    for _ in range(n_warmup):
        interp = llguidance.LLInterpreter(tokenizer, grammar)
        interp.start_without_prompt()
        interp.compute_mask()
    
    # Measure
    times = []
    for _ in range(n_iter):
        interp = llguidance.LLInterpreter(tokenizer, grammar)
        interp.start_without_prompt()
        t0 = time.perf_counter()
        mask_tuple = interp.compute_mask()
        times.append((time.perf_counter() - t0) * 1e6)
    
    mask_bytes = mask_tuple[0]
    valid = sum(bin(b).count('1') for b in mask_bytes)
    
    times.sort()
    return Result(
        system="LLGuidance",
        grammar=name,
        compile_ms=compile_ms,
        p50_us=times[len(times)//2],
        p95_us=times[int(len(times)*0.95)],
        p99_us=times[int(len(times)*0.99)],
        min_us=times[0],
        max_us=times[-1],
        valid_tokens=valid
    )


def bench_llguidance_lark(grammar: str, name: str, n_warmup=200, n_iter=1000) -> Result:
    """LLGuidance with Lark grammar."""
    import llguidance
    from llguidance.tiktoken import lltokenizer_from_encoding
    
    enc = tiktoken.get_encoding("gpt2")
    tokenizer = lltokenizer_from_encoding(enc)
    
    t0 = time.perf_counter()
    compiler = llguidance.LarkCompiler()
    grammar_obj = compiler.compile(grammar)
    compile_ms = (time.perf_counter() - t0) * 1000
    
    for _ in range(n_warmup):
        interp = llguidance.LLInterpreter(tokenizer, grammar_obj)
        interp.start_without_prompt()
        interp.compute_mask()
    
    times = []
    for _ in range(n_iter):
        interp = llguidance.LLInterpreter(tokenizer, grammar_obj)
        interp.start_without_prompt()
        t0 = time.perf_counter()
        mask_tuple = interp.compute_mask()
        times.append((time.perf_counter() - t0) * 1e6)
    
    mask_bytes = mask_tuple[0]
    valid = sum(bin(b).count('1') for b in mask_bytes)
    
    times.sort()
    return Result(
        system="LLGuidance",
        grammar=name,
        compile_ms=compile_ms,
        p50_us=times[len(times)//2],
        p95_us=times[int(len(times)*0.95)],
        p99_us=times[int(len(times)*0.99)],
        min_us=times[0],
        max_us=times[-1],
        valid_tokens=valid
    )


def bench_sep1(constraint_path: str, name: str, n_warmup=200, n_iter=1000) -> Result:
    """Sep1 with precompiled constraint."""
    try:
        import _sep1 as ffi
    except ImportError:
        return None
    
    enc = tiktoken.get_encoding("gpt2")
    
    if not Path(constraint_path).exists():
        return None
    
    t0 = time.perf_counter()
    with gzip.open(constraint_path, 'rt') as f:
        json_str = f.read()
    constraint = ffi.GrammarConstraint.from_json_string(json_str)
    compile_ms = (time.perf_counter() - t0) * 1000
    
    # Warmup
    for _ in range(n_warmup):
        state = ffi.GrammarConstraintState(constraint)
        state.get_mask_bv()
    
    # Measure
    times = []
    for _ in range(n_iter):
        state = ffi.GrammarConstraintState(constraint)
        t0 = time.perf_counter()
        mask = state.get_mask_bv()
        times.append((time.perf_counter() - t0) * 1e6)
    
    valid = sum(1 for t in range(enc.n_vocab) if mask.contains(t))
    
    times.sort()
    return Result(
        system="Sep1",
        grammar=name,
        compile_ms=compile_ms,
        p50_us=times[len(times)//2],
        p95_us=times[int(len(times)*0.95)],
        p99_us=times[int(len(times)*0.99)],
        min_us=times[0],
        max_us=times[-1],
        valid_tokens=valid
    )


# Test grammars
SIMPLE_SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
COMPLEX_SCHEMA = {
    "type": "object",
    "properties": {
        "users": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}, "name": {"type": "string"}}}},
        "meta": {"type": "object"}
    }
}

ARITHMETIC_GBNF = '''root ::= expr
expr ::= term (("+" | "-") term)*
term ::= factor (("*" | "/") factor)*
factor ::= [0-9]+ | "(" expr ")" | [a-z]+
'''

ARITHMETIC_LARK = '''
start: expr
expr: term (("+" | "-") term)*
term: factor (("*" | "/") factor)*
factor: NUMBER | "(" expr ")" | VARIABLE
NUMBER: /[0-9]+/
VARIABLE: /[a-z]+/
%ignore /[ \\t\\n]+/
'''


def main():
    print("=" * 80)
    print("FAIR COMPARISON BENCHMARK")
    print("=" * 80)
    print()
    print("Hardware: Apple M1 Max, 32GB RAM")
    print("Tokenizer: GPT-2 (50,257 tokens)")
    print("Methodology: 200 warmup, 1000 iterations")
    print()
    
    results: List[Result] = []
    
    # JSON Schemas (all systems support this)
    print("-" * 80)
    print("JSON SCHEMA COMPARISON (Same Schema for All)")
    print("-" * 80)
    
    print("\n[Simple Object Schema]")
    r = bench_xgrammar_schema(SIMPLE_SCHEMA, "simple_schema")
    results.append(r)
    print(f"  XGrammar:   p50={r.p50_us:.1f}μs, p99={r.p99_us:.1f}μs, tail={r.tail_ratio:.1f}×")
    
    r = bench_llguidance_schema(SIMPLE_SCHEMA, "simple_schema")
    results.append(r)
    print(f"  LLGuidance: p50={r.p50_us:.1f}μs, p99={r.p99_us:.1f}μs, tail={r.tail_ratio:.1f}×")
    
    print("\n[Complex Nested Schema]")
    r = bench_xgrammar_schema(COMPLEX_SCHEMA, "complex_schema")
    results.append(r)
    print(f"  XGrammar:   p50={r.p50_us:.1f}μs, p99={r.p99_us:.1f}μs, tail={r.tail_ratio:.1f}×")
    
    r = bench_llguidance_schema(COMPLEX_SCHEMA, "complex_schema")
    results.append(r)
    print(f"  LLGuidance: p50={r.p50_us:.1f}μs, p99={r.p99_us:.1f}μs, tail={r.tail_ratio:.1f}×")
    
    # CFG (arithmetic)
    print("\n" + "-" * 80)
    print("CFG COMPARISON (Same Grammar Semantics)")
    print("-" * 80)
    
    print("\n[Arithmetic Expressions]")
    r = bench_xgrammar_ebnf(ARITHMETIC_GBNF, "arithmetic")
    results.append(r)
    print(f"  XGrammar:   p50={r.p50_us:.1f}μs, p99={r.p99_us:.1f}μs, tail={r.tail_ratio:.1f}×")
    
    r = bench_llguidance_lark(ARITHMETIC_LARK, "arithmetic")
    results.append(r)
    print(f"  LLGuidance: p50={r.p50_us:.1f}μs, p99={r.p99_us:.1f}μs, tail={r.tail_ratio:.1f}×")
    
    # Sep1 specific tests
    print("\n" + "-" * 80)
    print("SEP1 PRECOMPILED GRAMMARS")
    print("-" * 80)
    
    json_path = ".cache/test_vocabs/constraint_json_new2.json.gz"
    if Path(json_path).exists():
        r = bench_sep1(json_path, "sep1_json")
        if r:
            results.append(r)
            print(f"\n[JSON Grammar]")
            print(f"  Sep1:       p50={r.p50_us:.1f}μs, p99={r.p99_us:.1f}μs, tail={r.tail_ratio:.1f}×, valid={r.valid_tokens}")
    
    js_path = ".cache/test_vocabs/constraint_js.json.gz"
    if Path(js_path).exists():
        r = bench_sep1(js_path, "sep1_javascript")
        if r:
            results.append(r)
            print(f"\n[JavaScript Grammar - 306 rules]")
            print(f"  Sep1:       p50={r.p50_us:.1f}μs, p99={r.p99_us:.1f}μs, tail={r.tail_ratio:.1f}×, valid={r.valid_tokens}")
    
    # Builtin JSON comparison
    print("\n" + "-" * 80)
    print("GENERAL JSON COMPARISON")
    print("-" * 80)
    
    r = bench_xgrammar_builtin_json()
    results.append(r)
    print(f"\n  XGrammar builtin: p50={r.p50_us:.1f}μs, p99={r.p99_us:.1f}μs, valid={r.valid_tokens}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    print(f"{'System':12} | {'Grammar':18} | {'p50':>7} | {'p95':>7} | {'p99':>7} | {'Tail':>5} | {'Valid':>7}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: (x.grammar, x.p50_us)):
        print(f"{r.system:12} | {r.grammar:18} | {r.p50_us:>6.1f}μs | {r.p95_us:>6.1f}μs | "
              f"{r.p99_us:>6.1f}μs | {r.tail_ratio:>4.1f}× | {r.valid_tokens:>7}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Group by grammar and find speedups
    by_grammar = {}
    for r in results:
        key = r.grammar
        if key not in by_grammar:
            by_grammar[key] = []
        by_grammar[key].append(r)
    
    for grammar, gresults in sorted(by_grammar.items()):
        if len(gresults) >= 2:
            sorted_r = sorted(gresults, key=lambda x: x.p50_us)
            fastest = sorted_r[0]
            print(f"\n{grammar}:")
            for r in sorted_r:
                if r == fastest:
                    print(f"  ★ {r.system}: {r.p50_us:.1f}μs (fastest)")
                else:
                    slowdown = r.p50_us / fastest.p50_us
                    print(f"    {r.system}: {r.p50_us:.1f}μs ({slowdown:.1f}× slower)")
    
    # Best/worst tail latency
    print("\n\nTail Latency Ranking (p99/p50, lower = more predictable):")
    for r in sorted(results, key=lambda x: x.tail_ratio):
        print(f"  {r.system} ({r.grammar}): {r.tail_ratio:.2f}×")
    
    # Save
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Apple M1 Max",
        "iterations": 1000,
        "results": [
            {
                "system": r.system, "grammar": r.grammar,
                "compile_ms": r.compile_ms,
                "p50_us": r.p50_us, "p95_us": r.p95_us, "p99_us": r.p99_us,
                "min_us": r.min_us, "max_us": r.max_us,
                "tail_ratio": r.tail_ratio, "valid_tokens": r.valid_tokens
            }
            for r in results
        ]
    }
    
    output_path = Path("gcg-paper/analysis/results") / f"fair_final_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved: {output_path}")


if __name__ == "__main__":
    main()
