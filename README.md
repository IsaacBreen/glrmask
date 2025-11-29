```bash
RUSTFLAGS=-Awarnings CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --color=always --package sep1 --lib test_js_constraint_integration --profile test -- --nocapture 2>&1 | tee .temp2; python analyze_timings.py .temp2
```

```bash
RUSTFLAGS=-Awarnings CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --color=always --package sep1 --lib test_js_constraint_isolated_and_minimized --profile test -- --nocapture 2>&1 | tee .temp2; python analyze_timings.py .temp2
```

## Testing Sep1 on JSON Schemas

```bash
cd python && RUSTFLAGS=-Awarnings maturin develop -r && cd ..


# With default hard schema (o69862.json - 12k lines, takes ~1-2s to compile)
python3 scripts/test_json_schema.py

# Or with a simpler/faster schema
SCHEMA_FILE="gcg-paper/downloads/repos/jsonschemabench/data/Github_easy/o10008.json" python3 scripts/test_json_schema.py
```

The test script (`scripts/test_json_schema.py`):
1. Loads a JSON schema from the maskbench dataset
2. Converts it to EBNF using the native Rust converter
3. Compiles it to a Sep1 grammar constraint  
4. Tests it by stepping through a valid JSON input token by token

### Schema locations

The maskbench dataset contains schemas of varying complexity:

- `Github_trivial/` - Very simple schemas
- `Github_easy/` - Simple schemas
- `Github_medium/` - Medium complexity
- `Github_hard/` - Complex schemas (good for benchmarking)
- `Github_ultra/` - Very complex schemas
- `Kubernetes/` - Kubernetes API schemas
- `JsonSchemaStore/` - Various JSON Schema Store schemas