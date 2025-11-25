Common commands to compile/run the code:

```bash
MACRO_DEBUG_LEVEL=5 python scripts/compile.py \
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
python scripts/generate_diff_grammar.py .cache/test_vocabs/constraint.rs.old -o .cache/test_vocabs/example_diff_constraint.ebnf
MACRO_DEBUG_LEVEL=5 python scripts/compile.py \
    --grammar .cache/test_vocabs/example_diff_constraint.ebnf \
    --output .cache/test_vocabs/example_diff_constraint.json.gz \
    --vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"
```

---

As a result of debug! macro usages, the stdout of this will contain timings like this:

```
src/precompute4/full_dwa.rs
   178  Optimizing precomputed1 via NWA/DWA conversion... +20ms
   187  Unrolling cycles in precomputed1 DWA... +1.35s
```

Note that the timing values (e.g., `+20ms`, `+1.35s`) displayed in purple at the end of the lines represent the **time elapsed since the previous debug message was printed**.

This is a "delta" timestamp, calculated globally across the application.

*   **`+20ms` on line 178**: Indicates that 20 milliseconds passed between the log message *prior* to line 178 and the moment line 178 was reached.
*   **`+1.35s` on line 187**: Indicates that 1.35 seconds passed between printing line 178 and printing line 187.

Therefore, to read performance from these logs, you usually look at the time appended to the **next** log line to understand how long the **current** block took to execute. In the example above, the step "Optimizing precomputed1 via NWA/DWA conversion..." took approximately 1.35 seconds.

If you see a log line like:
```text
   187  ... (...) +1.35s
```
The `+1.35s` is the "gap" time. If you use `debug_timer_end!`, you will see an explicit duration in parentheses (e.g., `(500ms)`) followed by the gap time.

---

To run the tests, run

```bash
RUST_TEST_THREADS=1 RUSTFLAGS=-Awarnings ENABLE_PROGRESS_BAR=0 CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --color=always --package sep1 --lib --profile test -- --nocapture
```

---

IMPORTANT GLOBAL INSTRUCTIONS

DO NOT STOP GOING until the job is done. Do not the user for additional input. Keep going without supervision until it's done. Make whatever adjustments or assumptions are needed. The user will be watching over you so it's safe. But do NOT stop.
Once it's done, ensure any final tasks/clean-up are finished. Do it, again, without further instruction from me. Be autonomous.
KEEP GOING. If you find you aren't getting anywhere, experiment. Don't stop trying. Don't ask me for additional input.
Any productive contributions from you are appreciated. You have full autonomy to infer and pursue my long-term goals. You are me.