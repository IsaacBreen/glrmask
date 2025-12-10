# Grammar-Constrained Decoding Benchmarks

This directory contains dedicated benchmark scripts for measuring the performance of grammar-constrained decoding systems.

## Benchmark Philosophy

Each system has its **own dedicated benchmark script** to ensure fair measurement:

- `benchmark_sep1.py` - Our system (sep1)
- `benchmark_xgrammar.py` - XGrammar
- `benchmark_llguidance.py` - llguidance

This approach ensures:
1. **Fair GCT measurement**: Each script measures the FULL end-to-end compilation time appropriate for that system
2. **System-specific setup**: Each system can have its own initialization requirements
3. **Clear attribution**: Easy to understand what's being measured for each system

## Metrics

### GCT (Grammar Compilation Time)
The time to compile a grammar + vocabulary into a ready-to-use constraint.

For sep1, this includes:
- Grammar parsing
- GLR parser construction
- Terminal characterizations
- NWA construction and determinization
- JSON serialization

For XGrammar/llguidance, this includes:
- Tokenizer initialization
- Grammar/schema compilation

### TBM (Time Between Masks)
The time to compute `get_mask()` after committing a token.

Statistics reported:
- **p50**: Median latency
- **p99**: 99th percentile latency (tail)
- **mean, min, max**: Additional context

## Usage

### Sep1 (Our System)
```bash
# Full GCT + TBM measurement
python -m python.benchmarks.benchmark_sep1 \
    --grammar src/js.ebnf \
    --input src/example_code11.js \
    --output results/sep1_js.json \
    --gct-runs 5 \
    --tbm-runs 3

# Using pre-compiled constraint (GCT from metadata)
python -m python.benchmarks.benchmark_sep1 \
    --grammar src/js.ebnf \
    --constraint .cache/test_vocabs/constraint_js.json.gz \
    --input src/example_code11.js \
    --output results/sep1_js_precompiled.json
```

### XGrammar
```bash
# EBNF grammar
python -m python.benchmarks.benchmark_xgrammar \
    --grammar src/js.ebnf \
    --input src/example_code11.js \
    --output results/xgrammar_js.json \
    --gct-runs 5

# JSON Schema
python -m python.benchmarks.benchmark_xgrammar \
    --schema schemas/example.json \
    --output results/xgrammar_schema.json
```

### llguidance
```bash
# Lark grammar
python -m python.benchmarks.benchmark_llguidance \
    --grammar grammar.lark \
    --input code.txt \
    --output results/llguidance.json

# JSON Schema
python -m python.benchmarks.benchmark_llguidance \
    --schema schemas/example.json \
    --output results/llguidance_schema.json
```

## Correctness Testing

Compare masks across systems to verify consistency:

```bash
python -m python.benchmarks.correctness_test \
    --grammar src/js.ebnf \
    --input src/example_code.js \
    --output results/correctness.json \
    --max-tokens 100
```

## Output Format

All benchmark scripts produce unified JSON output:

```json
{
  "system_name": "sep1",
  "grammar_name": "js.ebnf",
  "vocabulary_name": "gpt2",
  
  "gct_samples_sec": [4.1, 4.2, 4.0],
  "gct_p50_sec": 4.1,
  "gct_p99_sec": 4.2,
  "gct_mean_sec": 4.1,
  
  "tbm_samples_us": [65, 70, 68, ...],
  "tbm_p50_us": 68,
  "tbm_p99_us": 150,
  "tbm_mean_us": 72,
  
  "initial_mask_us": 120,
  "num_tokens_processed": 929,
  "input_file": "src/example_code11.js",
  "timestamp": "2025-12-09T..."
}
```

## Dependencies

### Sep1
- Rust toolchain (for grammar-compiler)
- Python bindings: `cd python && maturin develop -r`

### XGrammar
```bash
pip install xgrammar transformers torch
```

### llguidance
```bash
pip install llguidance tiktoken
```

## Environment Variables

- `MACRO_DEBUG_LEVEL=0`: Disable debug output (default for benchmarks)
- `ENABLE_PROGRESS_BAR=0`: Disable progress bars
