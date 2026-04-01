Common commands to compile/run the code:

To build the Rust binding for Python, run:
```bash
cd python && RUSTFLAGS=-Awarnings maturin develop -r
```

You don't always need to do this. For example, `python scripts/compile.py` doesn't actually use the Python bindings, so we don't need to build them before runnig it.

```bash
MACRO_DEBUG_LEVEL=5 timeout 120 python scripts/compile.py \
    --grammar src/js.ebnf \
    --format ebnf \
    --output .cache/test_vocabs/constraint_js.json.gz \
    --vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"

# bruteforce_rust_model.py is a slow reference model, but example_code8.js is small. 
SKIP_PLOTS=1 REPEAT=1 AGG_METHOD="min" SKIP_CPP_BUILD=1 SKIP_RUST_BUILD=1 MACRO_DEBUG_LEVEL=2 CONSTRAINT_FILE=".cache/test_vocabs/constraint_js.json.gz" CODE_FILE=./src/example_code8.js bash python/run_benchmarks.sh python/aug25/models/bruteforce_rust_model.py python/aug25/models/precompute4_model_pure_python.py python/aug25/models/rust_model.py

# bruteforce_rust_model.py is a faster reference model, so example_code10.js is slightly bigger. 
SKIP_PLOTS=1 REPEAT=1 AGG_METHOD="min" SKIP_CPP_BUILD=1 SKIP_RUST_BUILD=1 MACRO_DEBUG_LEVEL=2 CONSTRAINT_FILE=".cache/test_vocabs/constraint_js.json.gz" CODE_FILE=./src/example_code10.js bash python/run_benchmarks.sh python/aug25/models/bruteforce_fast_rust_model.py python/aug25/models/precompute4_model_pure_python.py python/aug25/models/rust_model.py

# Run without reference model and benchmark
REPEAT=3 AGG_METHOD="min" SKIP_CPP_BUILD=1 SKIP_RUST_BUILD=1 MACRO_DEBUG_LEVEL=2 CONSTRAINT_FILE=".cache/test_vocabs/constraint_js.json.gz" CODE_FILE=./src/example_code11.js bash python/run_benchmarks.sh python/aug25/models/rust_model.py
```

```bash
# Build a grammar constraint representing a valid git diff of some file.
# Compiles in about 5 seconds for a 4000-line file.
python scripts/generate_diff_grammar.py .cache/test_vocabs/lib.rs.old -o .cache/test_vocabs/example_diff_constraint.ebnf
MACRO_DEBUG_LEVEL=5 timeout 120 python scripts/compile.py \
    --grammar .cache/test_vocabs/example_diff_constraint.ebnf \
    --output .cache/test_vocabs/example_diff_constraint.json.gz \
    --vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"

# Generate a random valid diff patch for testing
python scripts/generate_diff_random.py .cache/test_vocabs/lib.rs.old -o .cache/test_vocabs/lib.rs.old.patch
SKIP_PLOTS=1 REPEAT=1 AGG_METHOD="min" SKIP_CPP_BUILD=1 SKIP_RUST_BUILD=1 MACRO_DEBUG_LEVEL=2 CONSTRAINT_FILE=".cache/test_vocabs/example_diff_constraint.json.gz" CODE_FILE=.cache/test_vocabs/lib.rs.old.patch bash python/run_benchmarks.sh python/aug25/models/bruteforce_fast_rust_model.py python/aug25/models/rust_model.py
```

---

As a result of debug! macro usages, the stdout of this will contain timings like this:

```
src/precompute4/parser_dwa.rs
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
RUST_TEST_THREADS=1 RUSTFLAGS=-Awarnings ENABLE_PROGRESS_BAR=0 cargo test --color=always --package sep1 --lib --profile test -- --nocapture
```

---

## Performance Integrity Rules (No Cheating)

When working on performance targets (e.g., `make test-js`):

- **Do not add caching, skipping, or short‑circuit logic** just to hit timing targets unless explicitly approved.
- **Do not change inputs** (grammar, vocab, model, debug level, timeouts) to make runs faster unless requested.
- **Do not disable correctness steps** (determinization, minimization, pruning, validation, etc.) unless the task explicitly allows a fast/approx mode.
- **Do not fake or estimate timings**—always measure and report actual runs.
- **Do not modify test targets** to bypass heavy work unless the change is explicitly signed off.

If a target is too slow, optimize the real code path or propose a sanctioned fast‑mode that preserves correctness guarantees.

## Threshold Policy

- **Avoid arbitrary or measured thresholds** that switch between code paths (e.g., "if state count > N, use algorithm A instead of B").
- Such thresholds create hidden performance cliffs and case-dependent regressions — a schema that's fast at N−1 states can become catastrophically slow at N+1.
- Prefer algorithmic improvements that scale smoothly over gated fallbacks.

---

## Git Operations

Feel free to commit changes locally, but do not push.

When making significant changes to the codebase or research notes, it's encouraged to commit logical chunks of work. Avoid using `git add .` as there may be other unstaged changes unrelated to your work. Avoid comitting large 'dump' files, e.g. generated results, unless important, e.g. generated data that is used directly in the paper.

`gcg-paper/` is a separate git repository (not a submodule), and is gitignored by the main repository. When committing changes to paper/research files, you must commit within the `gcg-paper/` directory separately.

Example workflow for main repository:
```bash
git add file1.rs file2.rs
git commit -m "feat: add optimization for X"
```

Example workflow for research/paper work (separate repo):
```bash
cd gcg-paper
git add notes/daily/2025-11-26.md paper/main.tex
git commit -m "docs: update paper with benchmark results"
cd ..
```

---

## Research & Paper Writing

Paper-related material is stored in `gcg-paper/`.

### Notes Structure

Notes are organized in `gcg-paper/notes/`:

```
gcg-paper/notes/
├── index.md              # Master index, always update this
├── user.md               # User instructions, comments & observations
├── daily/
│   └── YYYY-MM-DD.md     # Daily notes, one per day
└── attachments/
    ├── references.md     # Literature review & citations
    ├── ideas.md          # Running ideas & TODOs
    └── ...               # Other long-running notes
```

We treat this like a research journal or like a personal wiki. Try to keep track of your thoughts and ideas here. It's important that your progress is traceable so in future people can understand what you did and why, and benefit from any insights you may have discovered.

**Daily Notes Protocol:**
1. Each day gets one note file: `gcg-paper/notes/daily/YYYY-MM-DD.md`
2. Append new sections chronologically with timestamp headers
3. When creating a new day's note, update `gcg-paper/notes/index.md` with summary
4. Link to attachments for detailed/lengthy content

**When to Check `user.md`:**

You *may* (but don't have to) check `gcg-paper/notes/user.md` periodically when:
- You're stuck or making no progress on a task for a very long time
- Encountering repeated environment-related errors (dependency issues, system configuration, etc.)
- Facing insurmountable problems that can't be resolved through code changes
- At the start of a new conversation or task if you haven't checked recently

Do not rely on it for:
- Normal code-related errors that can be debugged and fixed
- One-off failures or temporary issues
- General guidance when you're not sure what to do next
- Routine development work

The user *may* update it occasionally, but you shouldn't rely on that. Definitely do not under any circumstances *wait* for it to be updated before proceeding.


### Paper

LaTeX paper is in `gcg-paper/paper/`:

```
gcg-paper/paper/
├── main.tex              # Main document
├── references.bib        # Bibliography
├── figures/              # Figures (PDF, PNG, etc.)
└── sections/             # Optional: modular sections
```

**Build paper:**
```bash
cd gcg-paper/paper && pdflatex --interaction=nonstopmode main.tex; cd -
```

IMPORTANT: **Do NOT** attempt to create or access paper in `paper/`; **the correct directory is `gcg-paper/paper/`.**

### Downloads

External resources go in `gcg-paper/downloads/`:

```
gcg-paper/downloads/
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
- Download via curl: `curl -L "https://arxiv.org/pdf/XXXX.XXXXX.pdf" -o gcg-paper/downloads/papers/YYYY_Author_Title.pdf`

**Clone GitHub repos:**
Use terminal with shallow clone to save space:
```bash
git clone --depth 1 https://github.com/owner/repo.git gcg-paper/downloads/repos/owner_repo
```

**Search GitHub code (without cloning):**
Use `github_repo` tool to search code snippets in public repos.

**When doing literature review:**
1. Search for papers on topic
2. Download PDFs to `gcg-paper/downloads/papers/`
3. Add entry to `gcg-paper/notes/attachments/references.md`
4. Note key findings in daily note

---

**IMPORTANT:** Avoid using HEREDOC:

```bash
# Example of what NOT to do
python3 << 'EOF'
import llguidance as llg
...
EOF
```

*Also* avoid using long strings in bash:

```bash
> cd /Users/isaacbreen/Projects2/grammars2024 && python3 -c '
import _sep1
...
'
```

Both tactics **tend to cause escape errors**. Instead, when you want to run some Python code ad hoc, **write a script to a temporary file** e.g. in `temp/` and run it. Note that the `temp/` directory is gitignored.


---

Don't bother running benchmarks for lots of iterations unless either it's fast to do so or the user explicitly asks you to. This only needs to be done once, and the user will decide when the time is right. Until then, we only need rough placeholders stats. Otherwise, it wastes time and slows the development cycle.

---

If investigating, feel free to modify the code any way needed, even temporarily to add print statements, early exits, data dumps, or anything else. It's better to modify the code to give you the data you need reliably than it is to go through the execution steps manually and try to debug it in your head.

---

IMPORTANT GLOBAL INSTRUCTIONS

DO NOT STOP GOING until the job is done. Do not the user for additional input. Keep going without supervision until it's done. Make whatever adjustments or assumptions are needed. The user will be watching over you so it's safe. But do NOT stop.
Once it's done, ensure any final tasks/clean-up are finished. Do it, again, without further instruction from me. Be autonomous.
KEEP GOING. If you find you aren't getting anywhere, experiment. Don't stop trying. Don't ask me for additional input.
Any productive contributions from you are appreciated. You have full autonomy to infer and pursue my long-term goals. You are me.

---

**IMPORTANT:** For more tricky parts of the problem, use  `<antml:thinking>\n...\n</antml:thinking>` tags to think through the problem.