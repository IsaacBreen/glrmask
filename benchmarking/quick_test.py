"""Quick validation test - minimal token count to complete fast."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarking.run_json_schema_benchmarks import run_single_benchmark, ALL_SCHEMAS
from benchmarking.systems.xgrammar_wrapper import XGrammarSystem, XGRAMMAR_AVAILABLE
from benchmarking.systems.llguidance_wrapper import LLGuidanceSystem, LLGUIDANCE_AVAILABLE

print("=" * 60)
print("QUICK VALIDATION TEST (5 tokens max)")
print("=" * 60)
print()

results = []

# Test XGrammar if available
if XGRAMMAR_AVAILABLE:
    try:
        print("Testing XGrammar...")
        system = XGrammarSystem(tokenizer_name="gpt2")
        result = run_single_benchmark(system, "simple_user", ALL_SCHEMAS["simple_user"], max_tokens=5)
        results.append(result)
        print(f"  ✓ {result.system_name}: {result.num_tokens_processed} tokens, {result.avg_mask_time_ms:.3f}ms avg")
    except Exception as e:
        print(f"  ✗ Error: {e}")

# Test llguidance if available  
if LLGUIDANCE_AVAILABLE:
    try:
        print("Testing llguidance...")
        system = LLGuidanceSystem(tokenizer_name="gpt2")
        result = run_single_benchmark(system, "simple_user", ALL_SCHEMAS["simple_user"], max_tokens=5)
        results.append(result)
        print(f"  ✓ {result.system_name}: {result.num_tokens_processed} tokens, {result.avg_mask_time_ms:.3f}ms avg")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print()
print("=" * 60)
print(f"COMPLETED: {len(results)} system(s) tested")
print("=" * 60)

for r in results:
    print(f"\n{r.system_name}:")
    print(f"  Compilation: {r.compilation_time_sec*1000:.2f}ms")
    print(f"  Avg mask: {r.avg_mask_time_ms:.3f}ms")  
    print(f"  Tokens: {r.num_tokens_processed}")
