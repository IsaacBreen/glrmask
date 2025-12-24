# Handoff Prompt: Investigating HardSchemas Benchmark Failures

### Context
I've been working on benchmarking `sep1`, `xgrammar`, and `llguidance` on "HardSchemas" (complex JSON schemas like `PackageJson`, `VegaLite`, `Kubernetes_Hard`). The benchmark infrastructure is in `gcg-paper/benchmarks/` and results are in `results/hard_schemas/`.

### Current State
- **Rust tests pass**: `test_json_schema_gpt2_real_vocab` in `src/test_constraint_basic.rs` confirms `sep1` correctly handles `PackageJson`-like sequences with real GPT-2 vocab.
- **Benchmarks ran**: All 3 systems on 10 HardSchema categories. Results stored in `results/hard_schemas/*.jsonl.gz`.

### Issues to Investigate (Priority Order)

**1. PackageJson Correctness Failure (ALL systems)**
- All systems report `token_was_valid: [true, true, true, true, false, false, ...]` - they fail after ~4 tokens.
- `sep1` shows `valid_token_counts: [1, 21, 42, 42, 1233, 0, 0, 0, 0, 0]` - count drops to 0!
- Likely cause: test harness issue in `gcg-paper/benchmarks/core.py` (`benchmark_schema` function) or test data format.
- Inspect: `gzcat results/hard_schemas/sep1_PackageJson.jsonl.gz`

**2. llguidance VegaLite Correctness Failure**
- `sep1` and `xgrammar` pass VegaLite correctness. `llguidance` fails all tokens.
- `llguidance` shows `valid_token_counts: [1, 1, 1, 1, 1]` - only 1 valid token at each step!
- Check: `gzcat results/hard_schemas/llguidance_VegaLite.jsonl.gz`
- Likely cause: `LLGuidanceAdapter` in `core.py` might have incorrect mask extraction or state management.

**3. Kubernetes_Hard / Kestra / ApolloRouter (ALL systems fail)**
- These fail compilation or timeout for all systems.
- Check error messages: `gzcat results/hard_schemas/sep1_Kubernetes_Hard.jsonl.gz | jq .error`
- May require schema simplification or timeout increase.

### Key Files
- `gcg-paper/benchmarks/core.py`: Contains `benchmark_schema()`, `Sep1Adapter`, `XGrammarAdapter`, `LLGuidanceAdapter`
- `gcg-paper/benchmarks/debug_trace.py`: Token-by-token tracing tool
- `gcg-paper/hard_schemas/data/`: Schema files with embedded test samples
- `src/test_constraint_basic.rs`: Rust tests proving `sep1` engine correctness

### Commands to Start
```bash
# View PackageJson results for all systems
gzcat results/hard_schemas/sep1_PackageJson.jsonl.gz results/hard_schemas/xgrammar_PackageJson.jsonl.gz results/hard_schemas/llguidance_PackageJson.jsonl.gz | jq -c '{sys: .schema_id, valid: .token_was_valid, counts: .valid_token_counts}'

# Run debug trace on PackageJson
python gcg-paper/benchmarks/debug_trace.py --schema gcg-paper/hard_schemas/data/PackageJson---package.json --system sep1

# View Kubernetes_Hard errors
gzcat results/hard_schemas/*_Kubernetes_Hard.jsonl.gz | jq -c '{sys: .schema_id, error: .error}'
```

### Goal
Fix the benchmark harness so that:
1. `PackageJson` passes correctness for at least `sep1` (we know the engine is correct from Rust tests)
2. `llguidance` passes correctness on `VegaLite` (or document why it legitimately fails)
3. Understand why `Kubernetes_Hard` etc. fail (timeout? schema complexity? unsupported features?)
