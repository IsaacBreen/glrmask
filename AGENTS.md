Common commands to compile/run the code:

```bash
MACRO_DEBUG_LEVEL=4 python scripts/compile.py \
    --grammar src/js.ebnf \
    --output .cache/test_vocabs/constraint_js.json.gz \
    --vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"

# bruteforce_rust_model.py is a slow reference model, but example_code8.js is small. 
SKIP_PLOTS=1 REPEAT=1 AGG_METHOD="min" SKIP_CPP_BUILD=1 SKIP_RUST_BUILD=1 MACRO_DEBUG_LEVEL=2 CONSTRAINT_FILE=".cache/test_vocabs/constraint_js.json.gz" CODE_FILE=./src/example_code8.js bash python/run_benchmarks.sh python/aug25/models/bruteforce_rust_model.py python/aug25/models/precompute4_model_pure_python.py python/aug25/models/rust_model.py

# bruteforce_rust_model.py is a faster reference model, so example_code10.js is slightly bigger. 
SKIP_PLOTS=1 REPEAT=1 AGG_METHOD="min" SKIP_CPP_BUILD=1 SKIP_RUST_BUILD=1 MACRO_DEBUG_LEVEL=2 CONSTRAINT_FILE=".cache/test_vocabs/constraint_js.json.gz" CODE_FILE=./src/example_code10.js bash python/run_benchmarks.sh python/aug25/models/bruteforce_fast_rust_model.py python/aug25/models/precompute4_model_pure_python.py python/aug25/models/rust_model.py

# Run without reference model and benchmark
REPEAT=3 AGG_METHOD="min" SKIP_CPP_BUILD=1 SKIP_RUST_BUILD=1 MACRO_DEBUG_LEVEL=2 CONSTRAINT_FILE=".cache/test_vocabs/constraint_js.json.gz" CODE_FILE=./src/example_code11.js bash python/run_benchmarks.sh python/aug25/models/precompute4_model_pure_python.py
```

```bash
# Build a grammar constraint representing a valid git diff of some file (choosing src/constraint.rs here but could be anything).anything).
python scripts/generate_diff_grammar.py src/constraint.rs -o .cache/test_vocabs/example_diff_constraint.ebnf
MACRO_DEBUG_LEVEL=4 python scripts/compile.py \
    --grammar .cache/test_vocabs/example_diff_constraint.ebnf \
    --output .cache/test_vocabs/example_diff_constraint.json.gz \
    --vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"
```