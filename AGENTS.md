Common commands to compile/run the code:

To build the Rust binding for Python, run:
```bash
cd python && RUSTFLAGS=-Awarnings maturin develop -r
```

```bash
MACRO_DEBUG_LEVEL=5 timeout 120 python scripts/compile.py \
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
MACRO_DEBUG_LEVEL=5 timeout 120 python scripts/compile.py \
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

## Research & Paper Writing

Paper-related material is stored in `gcg-paper/`.

### Notes Structure

Notes are organized in `notes/`:

```
notes/
├── index.md              # Master index, always update this
├── daily/
│   └── YYYY-MM-DD.md     # Daily notes, one per day
└── attachments/
    ├── references.md     # Literature review & citations
    ├── ideas.md          # Running ideas & TODOs
    └── ...               # Other long-running notes
```

We treat this like a research journal or like a personal wiki. Try to keep track of your thoughts and ideas here. It's important that your progress is traceable so in future people can understand what you did and why, and benefit from any insights you may have discovered.

**Daily Notes Protocol:**
1. Each day gets one note file: `notes/daily/YYYY-MM-DD.md`
2. Append new sections chronologically with timestamp headers
3. When creating a new day's note, update `notes/index.md` with summary
4. Link to attachments for detailed/lengthy content

### Paper

LaTeX paper is in `paper/`:

```
paper/
├── main.tex              # Main document
├── references.bib        # Bibliography
├── figures/              # Figures (PDF, PNG, etc.)
└── sections/             # Optional: modular sections
```

**Build paper:**
```bash
cd paper && latexmk -pdf main.tex
```

**Continuous build:**
```bash
cd paper && latexmk -pdf -pvc main.tex
```

### Downloads

External resources go in `downloads/`:

```
downloads/
├── papers/               # PDF papers
├── repos/                # Cloned repositories  
└── data/                 # Datasets
```

**Naming conventions:**
- Papers: `YYYY_AuthorLastName_ShortTitle.pdf`
- Repos: `owner_reponame/`

### Research Tools

**Web search:**
Use `vscode-websearchforcopilot_webSearch` for literature discovery.

**Download papers:**
- arXiv: Fetch PDF URLs directly (e.g., `https://arxiv.org/pdf/XXXX.XXXXX.pdf`)
- Use `fetch_webpage` to get abstract/metadata first
- Download via curl: `curl -L "https://arxiv.org/pdf/XXXX.XXXXX.pdf" -o downloads/papers/YYYY_Author_Title.pdf`

**Clone GitHub repos:**
Use terminal with shallow clone to save space:
```bash
git clone --depth 1 https://github.com/owner/repo.git downloads/repos/owner_repo
```

**Search GitHub code (without cloning):**
Use `github_repo` tool to search code snippets in public repos.

**When doing literature review:**
1. Search for papers on topic
2. Download PDFs to `downloads/papers/`
3. Add entry to `notes/attachments/references.md`
4. Note key findings in daily note

---

Avoid running code like this:

```bash
# Example of what NOT to do
cd /Users/isaacbreen/Projects2/grammars2024 && python3 << 'PYTHON_SCRIPT'
cmdand heredoc> import llguidance as llg
...
PYTHON_SCRIPT
```

It tends to cause escape errors. Instead, write a script to `temp/` and run it.

Also avoid HEREDOC for writing to files. Avoid HEREDOC in general.

---

IMPORTANT GLOBAL INSTRUCTIONS

DO NOT STOP GOING until the job is done. Do not the user for additional input. Keep going without supervision until it's done. Make whatever adjustments or assumptions are needed. The user will be watching over you so it's safe. But do NOT stop.
Once it's done, ensure any final tasks/clean-up are finished. Do it, again, without further instruction from me. Be autonomous.
KEEP GOING. If you find you aren't getting anywhere, experiment. Don't stop trying. Don't ask me for additional input.
Any productive contributions from you are appreciated. You have full autonomy to infer and pursue my long-term goals. You are me.