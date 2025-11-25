# Benchmarking Framework

> Comprehensive benchmarking infrastructure for fair comparison of grammar-constrained generation systems

## Overview

This framework provides:
- **Fair comparison** between sep1, Outlines, XGrammar, and llguidance
- **Correctness validation** by parsing generated outputs
- **Complex grammars** (JavaScript, Python, JSON Schema, SQL)
- **Minimal overhead** using native implementations where possible

## Structure

```
benchmarking/
├── systems/           # System wrappers with common interface
│   ├── base.py       # Base interface
│   ├── sep1.py       # Our system
│   ├── outlines.py   # Outlines wrapper
│   ├── xgrammar.py   # XGrammar wrapper
│   └── llguidance.py # llguidance wrapper
├── grammars/         # Test grammars in various formats
├── validation/       # Correctness validation
│   └── parsers/      # Reference parsers
├── configs/          # Benchmark configurations
├── results/          # Benchmark results (gitignored)
├── runner.py         # Main benchmark orchestration
└── analyzer.py       # Results analysis and plotting
```

## Usage

### Run All Benchmarks
```bash
cd benchmarking/
python runner.py --all
```

### Run Specific System
```bash
python runner.py --system sep1 --grammar javascript
```

### Analyze Results
```bash
python analyzer.py results/latest/ --output plots/
```

### Validate Correctness
```bash
python validation/validator.py results/latest/
```

## Adding New Systems

1. Create wrapper in `systems/yoursystem.py`
2. Inherit from `BaseSystem` interface
3. Implement required methods:
   - `compile_grammar(grammar_path)` - Returns compiled constraint
   - `create_state(compiled)` - Returns initial state
   - `get_mask(state)` - Returns valid token bitvector
   - `commit(state, token_id)` - Advances state

See `systems/base.py` for full interface documentation.

## Adding New Grammars

1. Add grammar file to `grammars/`
2. Create reference parser in `validation/parsers/`
3. Add configuration in `configs/`

## Fair Comparison Principles

- Native implementations (no Python overhead in measurement)
- Compilation time measured separately
- Multiple runs with statistical significance testing
- Identical test inputs across all systems
- Correctness validation for all outputs
