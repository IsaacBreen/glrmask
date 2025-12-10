#!/usr/bin/env python3
"""
Correctness Testing for Grammar-Constrained Decoding Systems

Compares the token masks produced by different systems to ensure consistency.

For each system and grammar:
1. Process a sequence of tokens
2. At each step, record the valid token mask
3. Compare masks across systems

A system's mask should be a SUPERSET of the true valid tokens (no false negatives).
False positives are acceptable as they get corrected on the next step.

Usage:
    python -m python.benchmarks.correctness_test \\
        --grammar src/js.ebnf \\
        --input src/example_code.js \\
        --output results/correctness.json

Output format:
{
    "grammar": "js.ebnf",
    "input_file": "example_code.js",
    "token_count": 100,
    "systems_tested": ["sep1", "xgrammar", "llguidance"],
    "results": {
        "sep1_vs_xgrammar": {
            "matching_steps": 95,
            "sep1_only": [...],  // tokens in sep1 but not xgrammar
            "xgrammar_only": [...],  // tokens in xgrammar but not sep1
        },
        ...
    }
}
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Set, List, Optional, Any
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.benchmarks.common import (
    load_gpt2_vocab,
    build_id_to_token_bytes,
    greedy_tokenize,
)


@dataclass
class MaskComparison:
    """Result of comparing masks between two systems at a single step."""
    step: int
    token_id: int  # Token that was committed before this step
    system1_mask: Set[int]
    system2_mask: Set[int]
    
    @property
    def matches(self) -> bool:
        return self.system1_mask == self.system2_mask
    
    @property
    def system1_only(self) -> Set[int]:
        """Tokens in system1 but not system2."""
        return self.system1_mask - self.system2_mask
    
    @property
    def system2_only(self) -> Set[int]:
        """Tokens in system2 but not system1."""
        return self.system2_mask - self.system1_mask
    
    @property
    def intersection(self) -> Set[int]:
        return self.system1_mask & self.system2_mask


@dataclass
class CorrectnessResult:
    """Full correctness test result."""
    grammar_name: str
    input_file: str
    token_count: int
    systems_tested: List[str]
    timestamp: str
    
    # Per-pair comparison results
    comparisons: Dict[str, Dict[str, Any]]
    
    # Overall statistics
    all_masks_match: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save_json(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_sep1_masks(
    grammar_path: Path,
    token_ids: List[int],
    vocab: Dict[str, int],
) -> List[Set[int]]:
    """
    Get masks from sep1 for a sequence of tokens.
    
    Returns: List of valid token sets, one per step
    """
    import gzip
    import tempfile
    import subprocess
    import os
    
    # Compile grammar
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(vocab, f)
        vocab_path = Path(f.name)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json.gz', delete=False) as f:
        output_path = Path(f.name)
    
    try:
        compiler_path = PROJECT_ROOT / "target" / "release" / "grammar-compiler"
        if not compiler_path.exists():
            subprocess.run(["cargo", "build", "--release", "-q"], check=True, cwd=PROJECT_ROOT)
        
        subprocess.run(
            [str(compiler_path), "--grammar", str(grammar_path), "--vocab", str(vocab_path), "--output", str(output_path)],
            check=True, capture_output=True,
            env={**os.environ, "ENABLE_PROGRESS_BAR": "0", "MACRO_DEBUG_LEVEL": "0"},
        )
        
        with gzip.open(output_path, 'rt') as f:
            constraint_json = f.read()
        
    finally:
        vocab_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
    
    # Load model and get masks
    from python.aug25.models.rust_model import Model as RustModel
    model = RustModel.from_json_string(constraint_json)
    
    masks = []
    
    # Initial mask
    initial_mask = model.get_mask()
    valid_tokens = set()
    for start, end in initial_mask.to_ranges():
        valid_tokens.update(range(start, end + 1))
    masks.append(valid_tokens)
    
    # Process tokens
    for token_id in token_ids:
        model.commit(token_id)
        mask = model.get_mask()
        valid_tokens = set()
        for start, end in mask.to_ranges():
            valid_tokens.update(range(start, end + 1))
        masks.append(valid_tokens)
        
        if not valid_tokens:
            break
    
    return masks


def get_xgrammar_masks(
    grammar_path: Path,
    token_ids: List[int],
    grammar_type: str = "ebnf",
    schema: Optional[dict] = None,
) -> List[Set[int]]:
    """Get masks from XGrammar for a sequence of tokens."""
    try:
        import xgrammar as xgr
        from transformers import AutoTokenizer
        import torch
    except ImportError:
        return []
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer))
    compiler = xgr.GrammarCompiler(tokenizer_info)
    
    if grammar_type == "json_schema" and schema is not None:
        compiled = compiler.compile_json_schema(schema)
    else:
        grammar_str = grammar_path.read_text()
        compiled = compiler.compile_grammar(grammar_str)
    
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    
    masks = []
    
    # Initial mask
    matcher.fill_next_token_bitmask(bitmask)
    logits = torch.zeros(tokenizer_info.vocab_size)
    xgr.apply_token_bitmask_inplace(logits, bitmask)
    valid_tokens = set((logits > -float('inf')).nonzero(as_tuple=True)[0].tolist())
    masks.append(valid_tokens)
    
    # Process tokens
    for token_id in token_ids:
        if not matcher.accept_token(token_id):
            break
        
        matcher.fill_next_token_bitmask(bitmask)
        logits = torch.zeros(tokenizer_info.vocab_size)
        xgr.apply_token_bitmask_inplace(logits, bitmask)
        valid_tokens = set((logits > -float('inf')).nonzero(as_tuple=True)[0].tolist())
        masks.append(valid_tokens)
    
    return masks


def get_llguidance_masks(
    schema: dict,
    token_ids: List[int],
) -> List[Set[int]]:
    """Get masks from llguidance for a sequence of tokens (JSON schema only)."""
    try:
        from llguidance import JsonCompiler, LLInterpreter, LLTokenizer
        import tiktoken
        from llguidance.tiktoken import lltokenizer_from_encoding
    except ImportError:
        return []
    
    enc = tiktoken.get_encoding("gpt2")
    ll_tokenizer = lltokenizer_from_encoding(enc)
    
    compiler = JsonCompiler()
    schema_str = json.dumps(schema)
    compiled = compiler.compile(schema_str)
    
    interpreter = LLInterpreter(ll_tokenizer, compiled)
    interpreter.start_without_prompt()
    
    def parse_mask(mask_bytes):
        valid_tokens = set()
        if isinstance(mask_bytes, bytes):
            for byte_idx, byte_val in enumerate(mask_bytes):
                if byte_val == 0:
                    continue
                for bit_idx in range(8):
                    if (byte_val >> bit_idx) & 1:
                        valid_tokens.add(byte_idx * 8 + bit_idx)
        return valid_tokens
    
    masks = []
    
    # Initial mask
    mask_result = interpreter.compute_mask()
    masks.append(parse_mask(mask_result[0]))
    
    # Process tokens
    for token_id in token_ids:
        try:
            interpreter.commit_token(token_id)
        except:
            break
        
        mask_result = interpreter.compute_mask()
        masks.append(parse_mask(mask_result[0]))
    
    return masks


def compare_masks(
    system1_name: str,
    system1_masks: List[Set[int]],
    system2_name: str,
    system2_masks: List[Set[int]],
    token_ids: List[int],
) -> Dict[str, Any]:
    """Compare masks from two systems."""
    min_len = min(len(system1_masks), len(system2_masks))
    
    matching_steps = 0
    differences = []
    
    for i in range(min_len):
        mask1 = system1_masks[i]
        mask2 = system2_masks[i]
        
        if mask1 == mask2:
            matching_steps += 1
        else:
            diff = {
                "step": i,
                "token_before": token_ids[i-1] if i > 0 else None,
                f"{system1_name}_only_count": len(mask1 - mask2),
                f"{system2_name}_only_count": len(mask2 - mask1),
                f"{system1_name}_only_sample": list(mask1 - mask2)[:10],
                f"{system2_name}_only_sample": list(mask2 - mask1)[:10],
                "intersection_size": len(mask1 & mask2),
            }
            differences.append(diff)
    
    return {
        "total_steps": min_len,
        "matching_steps": matching_steps,
        "match_rate": matching_steps / min_len if min_len > 0 else 0,
        f"{system1_name}_steps": len(system1_masks),
        f"{system2_name}_steps": len(system2_masks),
        "differences": differences[:20],  # Keep only first 20
    }


def main():
    parser = argparse.ArgumentParser(description="Correctness test for grammar-constrained decoding systems")
    parser.add_argument("--grammar", type=Path, help="Path to grammar file (.ebnf)")
    parser.add_argument("--schema", type=Path, help="Path to JSON schema file")
    parser.add_argument("--input", type=Path, required=True, help="Path to input code file")
    parser.add_argument("--output", type=Path, help="Output JSON file for results")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to process")
    parser.add_argument("--vocab-cache", type=Path, default=Path(".cache/vocab_cache"))
    
    args = parser.parse_args()
    
    if not args.grammar and not args.schema:
        parser.error("Either --grammar or --schema must be provided")
    
    # Load vocab and tokenize
    print("Loading vocabulary...")
    vocab = load_gpt2_vocab(args.vocab_cache)
    id_to_token = build_id_to_token_bytes(vocab)
    
    print(f"Tokenizing input: {args.input}")
    input_bytes = args.input.read_bytes()
    tokens = greedy_tokenize(input_bytes, id_to_token)
    token_ids = [t[0] for t in tokens][:args.max_tokens]
    print(f"  Processing {len(token_ids)} tokens")
    
    # Determine grammar type and name
    if args.schema:
        grammar_type = "json_schema"
        grammar_name = args.schema.name
        with open(args.schema) as f:
            schema = json.load(f)
    else:
        grammar_type = "ebnf"
        grammar_name = args.grammar.name
        schema = None
    
    result = CorrectnessResult(
        grammar_name=grammar_name,
        input_file=str(args.input),
        token_count=len(token_ids),
        systems_tested=[],
        timestamp=datetime.now(timezone.utc).isoformat(),
        comparisons={},
    )
    
    # Get masks from each system
    all_masks = {}
    
    if args.grammar:
        print("\nGetting sep1 masks...")
        try:
            all_masks["sep1"] = get_sep1_masks(args.grammar, token_ids, vocab)
            result.systems_tested.append("sep1")
            print(f"  Got {len(all_masks['sep1'])} masks")
        except Exception as e:
            print(f"  Failed: {e}")
    
    print("\nGetting xgrammar masks...")
    try:
        all_masks["xgrammar"] = get_xgrammar_masks(args.grammar or args.schema, token_ids, grammar_type, schema)
        if all_masks["xgrammar"]:
            result.systems_tested.append("xgrammar")
            print(f"  Got {len(all_masks['xgrammar'])} masks")
        else:
            print("  Not available")
    except Exception as e:
        print(f"  Failed: {e}")
    
    if args.schema:
        print("\nGetting llguidance masks...")
        try:
            all_masks["llguidance"] = get_llguidance_masks(schema, token_ids)
            if all_masks["llguidance"]:
                result.systems_tested.append("llguidance")
                print(f"  Got {len(all_masks['llguidance'])} masks")
            else:
                print("  Not available")
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Compare masks between systems
    print("\nComparing masks...")
    systems = list(all_masks.keys())
    for i, sys1 in enumerate(systems):
        for sys2 in systems[i+1:]:
            key = f"{sys1}_vs_{sys2}"
            comparison = compare_masks(
                sys1, all_masks[sys1],
                sys2, all_masks[sys2],
                token_ids,
            )
            result.comparisons[key] = comparison
            
            print(f"\n  {sys1} vs {sys2}:")
            print(f"    Matching steps: {comparison['matching_steps']}/{comparison['total_steps']}")
            print(f"    Match rate: {comparison['match_rate']:.1%}")
            
            if comparison['differences']:
                print(f"    First difference at step {comparison['differences'][0]['step']}")
                result.all_masks_match = False
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"All masks match: {result.all_masks_match}")
    
    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.save_json(args.output)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
