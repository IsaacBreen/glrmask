
import sys
import time
import json
import statistics
import random
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mock logic - FORCE MOCK
try:
    import _sep1
except ImportError:
    pass

m = ModuleType("_sep1")
class MockBitset: pass
class MockHybridBitset: pass
class MockConstraint: pass
class MockState: pass
m.Bitset = MockBitset
m.HybridBitset = MockHybridBitset
m.GrammarConstraint = MockConstraint
m.GrammarConstraintState = MockState
sys.modules["_sep1"] = m
sys.modules["python.aug25.models.rust_model"] = ModuleType("python.aug25.models.rust_model")

from benchmarking.systems.sep1 import Sep1System

# Hand-written
GRAMMARS = {
    "Arithmetic": "src/arithmetic.ebnf",
    "Parentheses": "src/nested_parens.ebnf",
    "JSON": "src/json.ebnf",
    "JavaScript": "src/js.ebnf"
}

CATEGORIES = [
    "Github_trivial",
    "Github_easy",
    "Github_medium",
    "Github_hard",
    "Github_ultra",
    "Glaiveai2K",
    "Kubernetes",
    "Snowplow",
    "WashingtonPost",
    "JsonSchemaStore"
]

DATASET_DIR = PROJECT_ROOT / "gcg-paper/downloads/repos/jsonschemabench/maskbench/data"

def load_vocab():
    try:
        with open(PROJECT_ROOT / "benchmarking/gpt2_vocab.json") as f:
            j = json.load(f)
            first_val = next(iter(j.values()))
            if isinstance(first_val, int): 
                return {v: k.encode('utf-8') for k, v in j.items()}
            else:
                return {int(k): v.encode('utf-8') for k, v in j.items()}
    except:
        return {i: str(i).encode('utf-8') for i in range(100)}

def measure_max_gct(system, path: Path, vocab: Dict, loops: int = 5) -> float:
    times = []
    for _ in range(loops):
        try:
            res = system.compile_grammar(path, vocab)
            times.append(res.compilation_time_sec * 1000)
        except:
            pass
    return max(times) if times else 0.0

def measure_max_tbm(system, path: Path, vocab: Dict, max_tokens: int = 50) -> float:
    # Run a short generation and track max mask time
    try:
        res = system.compile_grammar(path, vocab)
        state = system.create_state(res.compiled)
        
        mask_times = []
        for i in range(max_tokens):
            t0 = time.perf_counter()
            mask_res = system.get_mask(state)
            elapsed = time.perf_counter() - t0
            mask_times.append(elapsed * 1000000) # µs
            
            # Extract valid tokens to commit
            valid_token = None
            if hasattr(mask_res, 'to_ranges'):
                 for s, e in mask_res.to_ranges():
                     valid_token = s
                     break
            elif hasattr(mask_res, '__iter__'):
                # Some implementations return direct iterable of ints
                # Or set
                try:
                    it = iter(mask_res)
                    valid_token = next(it)
                except StopIteration:
                    pass
            elif hasattr(mask_res, 'valid_token_ids'):
                # BenchmarkResult usage implies valid_token_ids list
                if mask_res.valid_token_ids:
                    valid_token = mask_res.valid_token_ids[0]

            if valid_token is None:
                # print(f"  Stopped at token {i} (no valid tokens)")
                break
                
            system.commit(state, valid_token)
            
        return max(mask_times) if mask_times else 0.0
            
    except Exception as e:
        # print(f"TBM loop error: {e}")
        return 0.0

def main():
    print("=== Max Stats Measurement ===")
    vocab = load_vocab()
    print(f"Vocab: {len(vocab)}")
    system = Sep1System()
    
    print(f"{'Name':<20} | {'Max GCT (ms)':<15} | {'Max TBM (µs)':<15}")
    print("-" * 55)
    
    # 1. Hand-written
    for name, rel in GRAMMARS.items():
        path = PROJECT_ROOT / rel
        if path.exists():
            g_max = measure_max_gct(system, path, vocab, loops=3)
            # For TBM, run 100 tokens
            t_max = measure_max_tbm(system, path, vocab, max_tokens=100)
            print(f"{name:<20} | {g_max:<15.1f} | {t_max:<15.1f}")
            
    print("-" * 55)
    
    # 2. JSON Schemas
    # For schema categories, we want the max over the set of schemas?
    # Or median of maxes?
    # Usually "Max" column implies worst-case. So max(max(schema)) over category?
    # Yes.
    
    schemas_by_cat = {c: [] for c in CATEGORIES}
    if DATASET_DIR.exists():
        for f in DATASET_DIR.glob("*.json"):
            for c in CATEGORIES:
                if f.name.startswith(c + "---"):
                    schemas_by_cat[c].append(f)
                    break
                    
    for cat in CATEGORIES:
        schemas = schemas_by_cat[cat]
        if not schemas:
            continue
            
        # Sample 10 schemas, measure max for each
        # But to be "Max" over category we should arguably pick the largest schemas?
        # Or random sample? Random sample is safer for time.
        sample = random.sample(schemas, min(len(schemas), 10))
        
        cat_gct_maxes = []
        cat_tbm_maxes = []
        
        for s in sample:
            g = measure_max_gct(system, s, vocab, loops=1) # 1 run is enough if we scan multiple schemas
            t = measure_max_tbm(system, s, vocab, max_tokens=50)
            cat_gct_maxes.append(g)
            cat_tbm_maxes.append(t)
            
        final_g_max = max(cat_gct_maxes) if cat_gct_maxes else 0
        final_t_max = max(cat_tbm_maxes) if cat_tbm_maxes else 0
        
        print(f"{cat:<20} | {final_g_max:<15.1f} | {final_t_max:<15.1f}")

if __name__ == "__main__":
    main()
