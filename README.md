# Sep1: Efficient Grammar-Constrained Decoding

**Sep1** is a Rust library for grammar-constrained decoding in large language models. It compiles context-free grammars and tokenizers into deterministic weighted automata, enabling microsecond-scale mask computation during inference.

## Key Features

- **Fast mask queries**: 70μs median on JavaScript grammar (40× faster than alternatives)
- **Precomputed constraints**: One-time compilation, O(1) queries thereafter
- **Sparse bitvector representation**: Memory-efficient mask storage
- **Python bindings**: Easy integration via PyO3/Maturin
- **Correctness guaranteed**: Sound constraint enforcement (no false negatives)

## Performance

| Grammar | Compile Time | Mask Query (p50) | Mask Query (p99) |
|---------|-------------|------------------|------------------|
| JSON | 70ms | 25μs | 261μs |
| JavaScript | 4.0s | 70μs | 183μs |
| Arithmetic | 90ms | 24μs | 222μs |

Benchmarks on Apple M1 Max with GPT-2 tokenizer (50,257 tokens).

## Quick Start

### Building

```bash
# Build the Rust library
cargo build --release

# Build Python bindings
cd python && RUSTFLAGS=-Awarnings maturin develop -r
```

### Compiling a Grammar

```bash
# Compile an EBNF grammar to a constraint file
cargo run --release --bin grammar-compiler -- \
    --grammar src/js.ebnf \
    --format ebnf \
    --vocab .cache/test_vocabs/gpt2_vocab.json \
        --output .cache/test_vocabs/constraint_js.json.gz
```

The `--format` argument is optional and defaults to auto-detection based on file extension (`.ebnf` or `.lark`). Explicitly supported values are `ebnf` and `lark`.

### Using in Rust

```rust
use sep1::pipeline::{Pipeline, PipelineConfig};
use sep1::constraint::GrammarConstraint;
use std::collections::BTreeMap;

// Simple: parse grammar and build constraint
let constraint = Pipeline::from_ebnf_file("grammar.ebnf")?
    .with_vocab_file("vocab.json")?
    .build()?;

// With custom config (e.g., skip optimization)
let constraint = Pipeline::from_ebnf_file("grammar.ebnf")?
    .with_config(PipelineConfig::no_optimization())
    .with_vocab_file("vocab.json")?
    .build()?;

// Manual stage-by-stage building for advanced use cases
let pipeline = Pipeline::from_ebnf_file("grammar.ebnf")?;
let compiled = pipeline.build_compiled();  // Get CompiledGrammar
// ... inspect or modify, then build constraint with vocabulary
```

### Using in Python
```python
import _sep1 as sep1
import tiktoken

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")

# Create constraint from compiled file
constraint = sep1.GrammarConstraint.from_file(
    ".cache/test_vocabs/constraint_js.json.gz",
    enc.n_vocab
)

# Create state for decoding
state = sep1.GrammarConstraintState(constraint)

# Get valid token mask
mask = state.get_mask_bv()
valid_tokens = [i for i in range(enc.n_vocab) if mask.contains(i)]
```

## Pipeline Architecture

The compilation pipeline has three stages:

1. **Parsing**: Convert grammar source (EBNF, Lark, expressions) → `GrammarDefinition`
2. **Compilation**: Build tokenizer + GLR parser → `CompiledGrammar`
3. **Precomputation**: Build Parser DWA → `GrammarConstraint`

Each stage can be customized or accessed independently:

```rust
// Access intermediate representations
let pipeline = Pipeline::from_ebnf_file("grammar.ebnf")?;
let definition = pipeline.definition();     // Stage 1 output
let compiled = pipeline.build_compiled();   // Stage 2 output
let constraint = pipeline.build()?;         // Stage 3 output
```

## How It Works

Sep1 uses a novel **terminal characterization** of LR parsing:

1. **Grammar Analysis**: Compile grammar to GLR parser with per-terminal characterization automata
2. **Tokenizer Integration**: Compose characterizations with tokenizer DFA  
3. **Determinization**: Convert non-deterministic weighted automaton to deterministic form
4. **Mask Precomputation**: Each automaton state carries sparse bitvector of valid tokens

At runtime, mask queries reduce to single automaton transitions and weight reads.

## Project Structure

```
src/                    # Rust source code
├── lib.rs             # Library entry point
├── precompute4/       # Core precomputation algorithms
│   ├── full_dwa.rs    # Parser DWA construction
│   └── characterize.rs # Terminal characterization
└── constraint.rs      # Grammar constraint implementation

python/                 # Python bindings
scripts/               # Build and test scripts
gcg-paper/             # Research paper and analysis
```

## Testing

```bash
# Run all tests
RUST_TEST_THREADS=1 RUSTFLAGS=-Awarnings ENABLE_PROGRESS_BAR=0 \
    CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --package sep1 --lib -- --nocapture

# Run benchmarks
cd python && bash run_benchmarks.sh
```

## JSON Schema Support

Sep1 supports JSON Schema constraints via EBNF conversion:

```bash
# Test with a JSON schema from MaskBench
SCHEMA_FILE="gcg-paper/downloads/repos/jsonschemabench/data/Github_ultra/o21378.json" \
    python3 scripts/test_json_schema.py
```

## Related Work

- [XGrammar](https://github.com/mlc-ai/xgrammar) - Byte-level pushdown automata
- [llguidance](https://github.com/guidance-ai/llguidance) - Microsoft's constraint engine
- [Outlines](https://github.com/dottxt-ai/outlines) - Token-level automata for regex/grammars

## Citation

If you use Sep1 in your research, please cite:

```bibtex
@article{sep1,
  title={Efficient Grammar-Constrained Decoding via Precomputed Deterministic Weighted Automata},
  author={Breen, Isaac},
  year={2025}
}
```

## License

MIT License
