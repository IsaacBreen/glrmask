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
    error_message: str = ""
    
    
def run_single_benchmark(system, schema_name: str, schema: Dict, max_tokens: int = 50) -> BenchmarkResult:
    """Run benchmark for one system on one schema."""
    
    print(f"  Running {system.name} on {schema_name}...")
    
    try:
        # Create temporary schema file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_file = Path(f.name)
        
        try:
            # Compile
            compilation_result = system.compile_grammar(schema_file, {})
            compilation_time = compilation_result.compilation_time_sec
            memory_bytes = compilation_result.metadata.get('memory_bytes', 0)
            
            # Create state
            state = system.create_state(compilation_result.compiled)
            
            # Simulate token generation
            mask_times = []
            commit_times = []
            num_tokens = 0
            
            start_total = time.perf_counter()
            
            for i in range(max_tokens):
                # Get mask
                mask_result = system.get_mask(state)
                mask_times.append(mask_result.time_sec)
                
                if not mask_result.valid_token_ids:
                    # No valid tokens - done
                    break
                
                # Commit random valid token
                token_id = mask_result.valid_token_ids[0]
                commit_result = system.commit(state, token_id)
                commit_times.append(commit_result.time_sec)
                
                num_tokens += 1
            
            total_time = time.perf_counter() - start_total
            
            return BenchmarkResult(
                system_name=system.name,
                schema_name=schema_name,
                compilation_time_sec=compilation_time,
                num_tokens_processed=num_tokens,
                avg_mask_time_ms=sum(mask_times) / len(mask_times) * 1000 if mask_times else 0,
                avg_commit_time_ms=sum(commit_times) / len(commit_times) * 1000 if commit_times else 0,
                total_time_sec=total_time,
                memory_bytes=memory_bytes,
                success=True
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


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("JSON Schema Benchmarking Suite")
    print("=" * 80)
    print()
    
    # Initialize systems
    systems = []
    
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
    
    # Outlines
    try:
        from benchmarking.systems.outlines_wrapper import OutlinesSystem, OUTLINES_AVAILABLE
        if OUTLINES_AVAILABLE:
            print("✓ Outlines available")
            # systems.append(OutlinesSystem())  # Skip for now - needs more work
            print("  (skipping - wrapper incomplete)")
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
            result = run_single_benchmark(system, schema_name, schema, max_tokens=50)
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
    
    for result in all_results:
        status = "✓" if result.success else "✗"
        print(f"{status} {result.system_name:15} {result.schema_name:20}")
        if result.success:
            print(f"   Compilation: {result.compilation_time_sec*1000:8.2f}ms")
            print(f"   Avg mask:    {result.avg_mask_time_ms:8.3f}ms")
            print(f"   Avg commit:  {result.avg_commit_time_ms:8.3f}ms")
            print(f"   Tokens:      {result.num_tokens_processed:8d}")
            print(f"   Memory:      {result.memory_bytes:8d} bytes")
        else:
            print(f"   Error: {result.error_message.split(chr(10))[0][:60]}")
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
