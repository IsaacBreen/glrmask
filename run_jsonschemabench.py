#!/usr/bin/env python3
"""Run jsonschemabench validation test on a subset of schemas.

This script tests both sep1 and llguidance (if available) on JSON schema validation.
"""

import json
import sys
import os
import time
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'python'))

DATA_DIR = PROJECT_ROOT / 'external' / 'jsonschemabench' / 'maskbench' / 'data'


def run_sep1_test(schema: dict, tests: list) -> dict:
    """Test schema with sep1 engine."""
    from external.jsonschemabench.maskbench.maskbench.sep1_engine import Sep1Engine
    
    result = {
        'engine': 'sep1',
        'compile_ok': False,
        'compile_time_ms': 0,
        'valid_pass': 0,
        'valid_fail': 0,
        'invalid_pass': 0,
        'invalid_fail': 0,
        'error': None
    }
    
    try:
        engine = Sep1Engine()
        engine.init()
        
        t0 = time.time()
        engine.compile_grammar(schema)
        result['compile_time_ms'] = (time.time() - t0) * 1000
        result['compile_ok'] = True
        
        for test in tests:
            expected_valid = test.get('valid', True)
            instance = json.dumps(test['data'], ensure_ascii=False)
            tokens = engine.tokenizer.encode(instance, add_special_tokens=False)
            
            engine.reset()
            accepted = True
            for t in tokens:
                if not engine.commit_token(t):
                    accepted = False
                    break
            
            if expected_valid:
                if accepted:
                    result['valid_pass'] += 1
                else:
                    result['valid_fail'] += 1
            else:
                if not accepted:
                    result['invalid_pass'] += 1
                else:
                    result['invalid_fail'] += 1
                    
    except Exception as e:
        result['error'] = str(e)[:200]
    
    return result


def run_llg_test(schema: dict, tests: list) -> Optional[dict]:
    """Test schema with llguidance engine."""
    try:
        import llguidance as llg
        import llguidance.hf
        from llguidance.numpy import fill_next_token_bitmask, allocate_token_bitmask
        from transformers import AutoTokenizer
    except ImportError:
        return None
    
    result = {
        'engine': 'llguidance',
        'compile_ok': False,
        'compile_time_ms': 0,
        'valid_pass': 0,
        'valid_fail': 0,
        'invalid_pass': 0,
        'invalid_fail': 0,
        'error': None
    }
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        llg_tokenizer = llguidance.hf.from_tokenizer(tokenizer)
        mask_data = allocate_token_bitmask(1, llg_tokenizer.vocab_size)
        
        t0 = time.time()
        grammars = json.dumps({"grammars": [{"json_schema": schema}]})
        interp = llg.LLMatcher(llg_tokenizer, grammars)
        result['compile_time_ms'] = (time.time() - t0) * 1000
        
        if interp.is_error():
            raise ValueError(interp.get_error())
        result['compile_ok'] = True
        
        for test in tests:
            expected_valid = test.get('valid', True)
            instance = json.dumps(test['data'], ensure_ascii=False)
            tokens = tokenizer.encode(instance, add_special_tokens=False)
            
            # Reset by deep copying
            parser = interp.deep_copy()
            accepted = True
            for t in tokens:
                fill_next_token_bitmask(parser, mask_data, 0)
                ok = (mask_data[0, t // 32] & (1 << (t % 32))) != 0
                if not ok:
                    accepted = False
                    break
                parser.consume_token(t)
            
            if expected_valid:
                if accepted:
                    result['valid_pass'] += 1
                else:
                    result['valid_fail'] += 1
            else:
                if not accepted:
                    result['invalid_pass'] += 1
                else:
                    result['invalid_fail'] += 1
                    
    except Exception as e:
        result['error'] = str(e)[:200]
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=100, help='Number of schemas to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--engines', default='sep1,llg', help='Comma-separated engines to test')
    args = parser.parse_args()
    
    # Get list of files and sample
    all_files = sorted(DATA_DIR.glob('*.json'))
    random.seed(args.seed)
    files = random.sample(all_files, min(args.limit, len(all_files)))
    
    engines_to_test = [e.strip() for e in args.engines.split(',')]
    
    print(f"Testing {len(files)} schemas...")
    print(f"Engines: {', '.join(engines_to_test)}")
    print()
    
    # Initialize results
    results = {engine: defaultdict(int) for engine in engines_to_test}
    errors = {engine: [] for engine in engines_to_test}
    
    for i, f in enumerate(files):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(files)}")
        
        with open(f) as fp:
            data = json.load(fp)
        
        schema = data['schema']
        tests = data.get('tests', [])
        
        if not tests:
            continue
        
        # Test each engine
        if 'sep1' in engines_to_test:
            r = run_sep1_test(schema, tests)
            results['sep1']['schemas'] += 1
            if r['compile_ok']:
                results['sep1']['compile_ok'] += 1
                results['sep1']['valid_pass'] += r['valid_pass']
                results['sep1']['valid_fail'] += r['valid_fail']
                results['sep1']['invalid_pass'] += r['invalid_pass']
                results['sep1']['invalid_fail'] += r['invalid_fail']
                results['sep1']['compile_time_ms'] += r['compile_time_ms']
            else:
                results['sep1']['compile_fail'] += 1
                errors['sep1'].append((f.name, r['error']))
        
        if 'llg' in engines_to_test:
            r = run_llg_test(schema, tests)
            if r is None:
                if results['llg']['schemas'] == 0:
                    print("  (llguidance not available, skipping)")
                engines_to_test = [e for e in engines_to_test if e != 'llg']
            else:
                results['llg']['schemas'] += 1
                if r['compile_ok']:
                    results['llg']['compile_ok'] += 1
                    results['llg']['valid_pass'] += r['valid_pass']
                    results['llg']['valid_fail'] += r['valid_fail']
                    results['llg']['invalid_pass'] += r['invalid_pass']
                    results['llg']['invalid_fail'] += r['invalid_fail']
                    results['llg']['compile_time_ms'] += r['compile_time_ms']
                else:
                    results['llg']['compile_fail'] += 1
                    errors['llg'].append((f.name, r['error']))
    
    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for engine in engines_to_test:
        r = results[engine]
        total_schemas = r['schemas']
        if total_schemas == 0:
            continue
        
        print(f"\n{engine.upper()}")
        print(f"  Schemas: {total_schemas}")
        print(f"  Compile: {r['compile_ok']} ok, {r['compile_fail']} fail ({100*r['compile_ok']/total_schemas:.1f}%)")
        
        if r['compile_ok'] > 0:
            avg_compile = r['compile_time_ms'] / r['compile_ok']
            print(f"  Avg compile time: {avg_compile:.1f}ms")
        
        valid_total = r['valid_pass'] + r['valid_fail']
        invalid_total = r['invalid_pass'] + r['invalid_fail']
        
        if valid_total > 0:
            print(f"  Valid tests: {r['valid_pass']}/{valid_total} pass ({100*r['valid_pass']/valid_total:.1f}%)")
        if invalid_total > 0:
            print(f"  Invalid tests: {r['invalid_pass']}/{invalid_total} pass ({100*r['invalid_pass']/invalid_total:.1f}%)")
        
        if r['invalid_fail'] > 0:
            print(f"  ⚠️  {r['invalid_fail']} invalid instances wrongly ACCEPTED")
        if r['valid_fail'] > 0:
            print(f"  ⚠️  {r['valid_fail']} valid instances wrongly REJECTED")
    
    # Show sample errors
    for engine in engines_to_test:
        if errors[engine]:
            print(f"\n{engine} errors ({len(errors[engine])}):")
            for fname, err in errors[engine][:5]:
                print(f"  {fname}: {err[:60]}...")


if __name__ == '__main__':
    main()
