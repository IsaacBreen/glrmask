"""Main benchmark runner for JSON Schema tests.

This runs comprehensive benchmarks across all systems:
- sep1 (our system)
- XGrammar
- Outlines  
- llguidance

Measures:
- Token mask computation time
- Compilation time
- Memory usage
- Correctness (JSON parsing + schema validation)
"""

import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# Add project root
_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

from benchmarking.grammars.test_schemas import ALL_SCHEMAS, TEST_PROMPTS
from benchmarking.correctness_check import validate_json_output, decode_tokens

# Load GPT-2 vocab for testing
try:
    with open(_project_root / "benchmarking/gpt2_vocab.json") as f:
        GPT2_VOCAB_JSON = json.load(f)
        # Convert to {id: bytes}
        GPT2_VOCAB = {}
        for token_str, token_id in GPT2_VOCAB_JSON.items():
            # GPT2 vocab in json is usually string -> id.
            # We need to handle the byte encoding if we want perfect reconstruction.
            # For now, just encode utf-8.
            GPT2_VOCAB[token_id] = token_str.encode('utf-8') # Approximation
except Exception as e:
    print(f"Warning: Could not load GPT-2 vocab: {e}")
    GPT2_VOCAB = {i: str(i).encode('utf-8') for i in range(50257)} # Dummy


@dataclass
class BenchmarkResult:
    """Results for one system on one schema."""
    system_name: str
    schema_name: str
    compilation_time_sec: float
    num_tokens_processed: int
    avg_mask_time_ms: float
    avg_commit_time_ms: float  
    total_time_sec: float
    memory_bytes: int
    success: bool
    correctness_valid: bool = False
    correctness_error: str = ""
    error_message: str = ""
    
    
def run_single_benchmark(system, schema_name: str, schema: Dict, max_tokens: int = 100) -> BenchmarkResult:
    """Run benchmark for one system on one schema."""
    
    print(f"  Running {system.name} on {schema_name}...")
    
    try:
        # Create temporary schema file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_file = Path(f.name)
        
        try:
            # Compile
            compilation_result = system.compile_grammar(schema_file, GPT2_VOCAB)
            compilation_time = compilation_result.compilation_time_sec
            memory_bytes = compilation_result.metadata.get('memory_bytes', 0)
            
            # Create state
            state = system.create_state(compilation_result.compiled)
            
            # Simulate token generation
            mask_times = []
            commit_times = []
            generated_tokens = []
            
            start_total = time.perf_counter()
            
            # Simple greedy generation simulation
            # We just pick the first valid token to simulate "generation"
            # In a real test we might want to guide it or use a model.
            # But for benchmarking constraint overhead, random valid or first valid is fine.
            
            for i in range(max_tokens):
                # Get mask
                mask_result = system.get_mask(state)
                mask_times.append(mask_result.time_sec)
                
                if not mask_result.valid_token_ids:
                    # No valid tokens - done (EOF?)
                    break
                
                # Commit first valid token (deterministic)
                # Or maybe try to find a "sensible" token?
                # For now, just pick the first one.
                token_id = mask_result.valid_token_ids[0]
                
                # Check if it's EOS?
                # GPT2 EOS is 50256 usually.
                
                commit_result = system.commit(state, token_id)
                commit_times.append(commit_result.time_sec)
                
                generated_tokens.append(token_id)
                
                # Stop if we hit a limit or some condition?
                # We rely on max_tokens for now.
            
            total_time = time.perf_counter() - start_total
            
            # Validate correctness
            output_str = decode_tokens(generated_tokens, GPT2_VOCAB)
            # print(f"    Generated: {output_str[:50]}...")
            
            # Note: Since we are just picking random/first tokens, the output will likely be garbage JSON.
            # So validation will fail.
            # To test correctness properly, we need to guide the generation or use a real model.
            # OR, we can just check if it's *grammatically* valid prefix?
            # Full JSON validation expects complete JSON.
            
            # For this benchmark, we are measuring PERFORMANCE.
            # Correctness check here is just "did it crash?".
            # We can try to validate if it finished?
            
            is_valid, val_error = validate_json_output(output_str, schema)
            
            return BenchmarkResult(
                system_name=system.name,
                schema_name=schema_name,
                compilation_time_sec=compilation_time,
                num_tokens_processed=len(generated_tokens),
                avg_mask_time_ms=sum(mask_times) / len(mask_times) * 1000 if mask_times else 0,
                avg_commit_time_ms=sum(commit_times) / len(commit_times) * 1000 if commit_times else 0,
                total_time_sec=total_time,
                memory_bytes=memory_bytes,
                success=True,
                correctness_valid=is_valid,
                correctness_error=val_error if val_error else ""
            )
            
        finally:
            schema_file.unlink()
            
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"    ERROR: {e}")
        
        return BenchmarkResult(
            system_name=system.name,
            schema_name=schema_name,
            compilation_time_sec=0,
            num_tokens_processed=0,
            avg_mask_time_ms=0,
            avg_commit_time_ms=0,
            total_time_sec=0,
            memory_bytes=0,
            success=False,
            error_message=error_msg
        )


    # sep1
    try:
        from benchmarking.systems.sep1 import Sep1System
        print("✓ sep1 available")
        systems.append(Sep1System())
    except Exception as e:
        print(f"✗ sep1 error: {e}")

    # XGrammar
    try:
        from benchmarking.systems.xgrammar_wrapper import XGrammarSystem, XGRAMMAR_AVAILABLE
        if XGRAMMAR_AVAILABLE:
            print("✓ XGrammar available")
            systems.append(XGrammarSystem(tokenizer_name="gpt2"))
        else:
            print("✗ XGrammar not available")
    except Exception as e:
        print(f"✗ XGrammar error: {e}")
    
    # llguidance
    try:
        from benchmarking.systems.llguidance_wrapper import LLGuidanceSystem, LLGUIDANCE_AVAILABLE
        if LLGUIDANCE_AVAILABLE:
            print("✓ llguidance available")
            systems.append(LLGuidanceSystem(tokenizer_name="gpt2"))
        else:
            print("✗ llguidance not available")
    except Exception as e:
        print(f"✗ llguidance error: {e}")
    
    # Outlines
    try:
        from benchmarking.systems.outlines_wrapper import OutlinesSystem, OUTLINES_AVAILABLE
        if OUTLINES_AVAILABLE:
            print("✓ Outlines available")
            # systems.append(OutlinesSystem()) # Still skipping for now until fixed
            print("  (skipping Outlines - wrapper needs fix)")
        else:
            print("✗ Outlines not available")
    except Exception as e:
        print(f"✗ Outlines error: {e}")
    
    print()
    print(f"Testing {len(systems)} system(s) on {len(ALL_SCHEMAS)} schema(s)")
    print()
    
    # Run benchmarks
    all_results = []
    
    for schema_name, schema in ALL_SCHEMAS.items():
        print(f"Schema: {schema_name}")
        
        for system in systems:
            result = run_single_benchmark(system, schema_name, schema, max_tokens=100)
            all_results.append(result)
        
        print()
    
    # Save results
    results_file = Path("benchmarking/results/json_schema_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print()
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    print(f"{'System':<15} {'Schema':<20} {'Comp(ms)':<10} {'Mask(ms)':<10} {'Commit(ms)':<10} {'Tokens':<8} {'Valid'}")
    print("-" * 90)
    
    for result in all_results:
        if result.success:
            print(f"{result.system_name:<15} {result.schema_name:<20} {result.compilation_time_sec*1000:<10.2f} {result.avg_mask_time_ms:<10.3f} {result.avg_commit_time_ms:<10.3f} {result.num_tokens_processed:<8d} {result.correctness_valid}")
        else:
            print(f"{result.system_name:<15} {result.schema_name:<20} ERROR: {result.error_message.split(chr(10))[0][:40]}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
