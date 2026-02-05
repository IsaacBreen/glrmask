# Research Project Makefile
# Common commands for paper writing and research
#
# NOTE: There are TWO separate Rust compilation targets:
#   1. grammar-compiler binary (built by `cargo build --release`)
#   2. Python FFI (_sep1, built by `maturin develop -r`)
# 
# When source code changes, BOTH may need to be recompiled. This is expected
# because they have different features (pyo3/extension-module for FFI) and
# different crate types (bin vs cdylib). Use SKIP_MATURIN=1 or --no-recompile
# to skip unnecessary rebuilds when testing.

.PHONY: paper paper-watch paper-clean notes-today help build test test-release test-js test-json-schema test-schema-packagejson test-schema-github test-schema-sarif test-schema-meta test-schema-extra test-schema-kestra test-schema-vegalite test-schema-apollo test-schema-liquibase test-diff test-diff-dfa show-schema-id schema-id ffi viz viz-clean all jsonschemabench jsonschemabench-quick jsonschemabench-subset jsonschemabench-analyze jsonschemabench-llg jsonschemabench-llg-analyze jsonschemabench-compare

# Default timeout for test-js (override with TEST_TIMEOUT=...)
TEST_TIMEOUT ?= 300

# Skip serialization by default to speed up test targets (override with SKIP_SERIALIZATION=0)
SKIP_SERIALIZATION ?= 1

# === Build All ===

all: ffi viz paper ## Build all components: FFI, visualizations, and paper

# === Paper Commands ===

paper: ## Build the paper PDF
	cd gcg-paper/paper && latexmk -pdf main.tex

paper-watch: ## Build paper continuously (watch mode)
	cd gcg-paper/paper && latexmk -pdf -pvc main.tex

paper-clean: ## Clean paper build artifacts
	cd gcg-paper/paper && latexmk -C

paper-open: paper ## Build and open paper
	open gcg-paper/paper/main.pdf

# === Visualization Commands ===

ffi: ## Build the Python FFI binding (required for visualizations)
	@if [ -z "$(SKIP_MATURIN)" ]; then \
		cd python && RUSTFLAGS=-Awarnings maturin develop -r; \
	else \
		echo "Skipping maturin compilation (SKIP_MATURIN is set)"; \
	fi

viz: ffi ## Generate all visualization components
	cd gcg-paper/paper/figures/components && make all

viz-clean: ## Clean visualization artifacts
	cd gcg-paper/paper/figures/components && make clean

# === Notes Commands ===

notes-today: ## Create/open today's daily note
	@mkdir -p notes/daily
	@TODAY=$$(date +%Y-%m-%d); \
	if [ ! -f "notes/daily/$$TODAY.md" ]; then \
		echo "# $$TODAY — Research Notes\n\n## $$(date +%H:%M) — \n\n---\n\n## Notes\n\n*Add notes throughout the day below this line.*\n" > "notes/daily/$$TODAY.md"; \
		echo "Created notes/daily/$$TODAY.md"; \
	fi; \
	echo "notes/daily/$$TODAY.md"

# === Download Commands ===

clone-repo: ## Clone a repo: make clone-repo URL=https://github.com/owner/repo
	@if [ -z "$(URL)" ]; then echo "Usage: make clone-repo URL=https://github.com/owner/repo"; exit 1; fi
	@REPO_NAME=$$(echo "$(URL)" | sed 's|.*/\([^/]*/[^/]*\)\.git$$|\1|' | sed 's|.*/\([^/]*/[^/]*\)$$|\1|' | tr '/' '_'); \
	git clone --depth 1 "$(URL)" "downloads/repos/$$REPO_NAME"

download-arxiv: ## Download arXiv paper: make download-arxiv ID=2305.13971 NAME=2023_Geng_GCD
	@if [ -z "$(ID)" ] || [ -z "$(NAME)" ]; then echo "Usage: make download-arxiv ID=2305.13971 NAME=2023_Geng_GCD"; exit 1; fi
	curl -L "https://arxiv.org/pdf/$(ID).pdf" -o "downloads/papers/$(NAME).pdf"
	@echo "Downloaded to downloads/papers/$(NAME).pdf"

# === Test/Build Commands ===

test: ## Run Rust tests
	RUST_TEST_THREADS=1 RUSTFLAGS=-Awarnings ENABLE_PROGRESS_BAR=0 CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --color=always --package sep1 --lib --profile test -- --nocapture

test-release: ## Run all crate tests in release profile + test_nwa_minimize_determinize_minimize (ignored by default)
	@echo "Running all non-ignored tests in release profile..."
	RUST_TEST_THREADS=1 RUSTFLAGS=-Awarnings ENABLE_PROGRESS_BAR=0 cargo test --color=always --package sep1 --lib --release -- --nocapture
	@echo ""
	@echo "Running test_nwa_minimize_determinize_minimize (ignored by default)..."
	RUST_TEST_THREADS=1 RUSTFLAGS=-Awarnings ENABLE_PROGRESS_BAR=0 cargo test --color=always --package sep1 --lib --release -- test_nwa_minimize_determinize_minimize --nocapture --include-ignored

test-js: ## Compile the JavaScript grammar (verifies it compiles)
	NWA_SUFFIX_PRUNE=1 SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) timeout $(TEST_TIMEOUT) python scripts/compile.py \
		--grammar src/js.ebnf \
		--format ebnf \
		--output .cache/test_vocabs/constraint_js.json.gz \
		--vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json" \
		$(if $(SKIP_MATURIN),--no-recompile,)

test-json-schema: ## Compile a JSON schema grammar (verifies schema-to-EBNF works)
	SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) \
	SCHEMA_FILE="gcg-paper/downloads/repos/jsonschemabench/data/Github_ultra/o21378.json" \
		python3 scripts/test_json_schema.py

test-json-schema-o1051: build ## Compile o1051 (Github Hard) schema
	SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/downloads/repos/jsonschemabench/data/Github_hard/o1051.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_o1051.json.gz

test-tsconfig: ## Compile TSConfig schema
	SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) \
	SCHEMA_FILE="gcg-paper/hard_schemas/data/TSConfig---tsconfig.json" \
		python3 scripts/test_json_schema.py

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

test-schema-id: ffi ## Compile any benchmark schema by ID (usage: make test-schema-id ID=ApolloRouter---apollo-router-2.9.0)
	@if [ -z "$(ID)" ]; then echo "Usage: make test-schema-id ID=<schema_id>"; exit 1; fi
	SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) SCHEMA_ID="$(ID)" $(PYTHON) scripts/test_json_schema.py

show-schema-id: ffi ## Print EBNF for a schema by ID to stdout (usage: make show-schema-id ID=...)
	@if [ -z "$(ID)" ]; then echo "Usage: make show-schema-id ID=<schema_id>"; exit 1; fi
	SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) SCHEMA_ID="$(ID)" PRINT_EBNF=1 $(PYTHON) scripts/test_json_schema.py

schema-id: ffi ## Write EBNF for a schema by ID to a file (usage: make schema-id ID=... OUT=file.ebnf)
	@if [ -z "$(ID)" ] || [ -z "$(OUT)" ]; then echo "Usage: make schema-id ID=<schema_id> OUT=<file.ebnf>"; exit 1; fi
	SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) SCHEMA_ID="$(ID)" PRINT_EBNF=1 OUT_FILE="$(OUT)" $(PYTHON) scripts/test_json_schema.py

show-json-schema: ffi ## Print the original JSON schema to stdout (usage: make show-json-schema ID=...)
	@if [ -z "$(ID)" ]; then echo "Usage: make show-json-schema ID=<schema_id>"; exit 1; fi
	SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) SCHEMA_ID="$(ID)" PRINT_JSON_SCHEMA=1 $(PYTHON) scripts/test_json_schema.py

test-diff: ffi ## Test diff grammar for any text file (usage: make test-diff FILE=path/to/file.txt)
	@if [ -z "$(FILE)" ]; then echo "Usage: make test-diff FILE=<path_to_text_file>"; exit 1; fi
	SOURCE_FILE="$(FILE)" $(PYTHON) scripts/test_diff.py

test-diff-dfa: ffi ## Test diff grammar on static finite_automata.rs (no FILE arg needed)
	SOURCE_FILE="testdata/finite_automata.rs" $(PYTHON) scripts/test_diff.py

diff-grammar: ffi ## Generate EBNF grammar for a file (usage: make diff-grammar FILE=test12.txt OUT=temp.ebnf)
	@if [ -z "$(FILE)" ] || [ -z "$(OUT)" ]; then echo "Usage: make diff-grammar FILE=<file> OUT=<out.ebnf>"; exit 1; fi
	SOURCE_FILE="$(FILE)" OUT_FILE="$(OUT)" ONLY_GRAMMAR=1 $(PYTHON) scripts/test_diff.py

show-diff-grammar: ffi ## Print EBNF grammar for a file to stdout (usage: make show-diff-grammar FILE=test12.txt)
	@if [ -z "$(FILE)" ]; then echo "Usage: make show-diff-grammar FILE=<file>"; exit 1; fi
	SOURCE_FILE="$(FILE)" PRINT_GRAMMAR=1 ONLY_GRAMMAR=1 $(PYTHON) scripts/test_diff.py

compile-ebnf: ## Compile an EBNF grammar (usage: make compile-ebnf FILE=src/js.ebnf [OUT=out.json.gz])
	@if [ -z "$(FILE)" ]; then echo "Usage: make compile-ebnf FILE=<ebnf_file> [OUT=<output_file>]"; exit 1; fi
	SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) python scripts/compile.py \
		--grammar "$(FILE)" \
		--format ebnf \
		--output "$${OUT:-.cache/test_vocabs/constraint_$$(basename "$(FILE)" .ebnf).json.gz}" \
		--vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json" \
		$(if $(SKIP_MATURIN),--no-recompile,)

# === Token Error Reproduction (sep1 failures) ===
# These reproduce cases where sep1 rejects valid tokens from test data

repro-jstsdraft4-enum: ffi ## Reproduce JSTSDraft4 enum failure (whitespace issue at step 1)
	SCHEMA_ID="JSTSDraft4---enum_8_enumwithtruedoesnotmatch1" \
		$(PYTHON) scripts/test_json_schema.py

repro-openapi: ffi ## Reproduce OpenAPI failure (property order at step 500)
	SCHEMA_ID="OpenAPI---openapi-3.0" \
		$(PYTHON) scripts/test_json_schema.py

repro-washingtonpost: ffi ## Reproduce WashingtonPost failure (property key at step 45)
	SCHEMA_ID="WashingtonPost---wp_68_Normalized" \
		$(PYTHON) scripts/test_json_schema.py
# === Hard Schema Compilation Tests ===
# These use the Rust grammar_compiler binary directly with --json-schema

test-schema-packagejson: build ## Compile PackageJson schema
	SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/hard_schemas/data/PackageJson---package.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_packagejson.json.gz

test-schema-github: build ## Compile GithubWorkflow schema
	SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/hard_schemas/data/GithubWorkflow---github-workflow.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_github.json.gz

test-schema-sarif: build ## Compile SARIF schema
	SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/hard_schemas/data/SARIF---sarif-2.1.0-rtm.1.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_sarif.json.gz

test-schema-meta: build ## Compile JSON Schema meta-schema (draft v4)
	SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/hard_schemas/data/JsonSchemaMeta---schema-draft-v4.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_meta.json.gz

test-schema-extra: build ## Compile bamboo-spec from SchemaStore_Extra
	SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/hard_schemas/data/SchemaStore_Extra---bamboo-spec.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_bamboo.json.gz

test-schema-kestra: build ## Compile Kestra schema (WARNING: ~8MB, very slow)
	SKIP_SERIALIZATION=$(SKIP_SERIALIZATION) MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/hard_schemas/data/Kestra---kestra-0.19.0.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_kestra.json.gz

test-schema-vegalite: build ## Compile VegaLite schema (very_high complexity)
	MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/hard_schemas/data/VegaLite---vega-lite.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_vegalite.json.gz

test-schema-apollo: build ## Compile ApolloRouter schema (very_high complexity)
	MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/hard_schemas/data/ApolloRouter---apollo-router-2.9.0.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_apollo.json.gz

test-schema-liquibase: build ## Compile Liquibase schema (high complexity)
	MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/hard_schemas/data/Liquibase---liquibase.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_liquibase.json.gz

build: ## Build the Rust project
	@if [ -z "$(SKIP_MATURIN)" ]; then \
		cargo build --release; \
	else \
		echo "Skipping cargo build (SKIP_MATURIN is set)"; \
	fi

# === JSON Schema Benchmark ===

BENCH_OPTS :=
ifneq ($(LIMIT),)
    BENCH_OPTS += --limit $(LIMIT)
endif
ifneq ($(FRACTION),)
    BENCH_OPTS += --fraction $(FRACTION)
endif
ifneq ($(SHUFFLE),)
    BENCH_OPTS += --shuffle
endif

jsonschemabench: ffi ## Run sep1 benchmarks on JSonSchemaBench suite (full, ~11K schemas). Vars: LIMIT, FRACTION, SHUFFLE
	cd gcg-paper/external-benchmarks/jsonschemabench && make run BENCH_ARGS="$(BENCH_OPTS)"

jsonschemabench-quick: ffi ## Run sep1 benchmark on a single schema (quick test)
	cd gcg-paper/external-benchmarks/jsonschemabench/jsonschemabench/maskbench && \
		python3 -m maskbench.runner --sep1 data/JME_0.json

jsonschemabench-subset: ffi ## Run sep1 benchmarks on a subset of schemas (usage: make jsonschemabench-subset PATTERN="JME_*.json")
	@if [ -z "$(PATTERN)" ]; then echo "Usage: make jsonschemabench-subset PATTERN=\"JME_*.json\""; exit 1; fi
	cd gcg-paper/external-benchmarks/jsonschemabench/jsonschemabench/maskbench && \
		python3 -m maskbench.runner --sep1 "data/$(PATTERN)" $(BENCH_OPTS)

jsonschemabench-analyze: ## Analyze sep1 benchmark results (run after jsonschemabench)
	cd gcg-paper/external-benchmarks/jsonschemabench/jsonschemabench/maskbench && \
		python3 scripts/maskbench_results.py tmp/out--sep1

jsonschemabench-llg: ## Run llguidance benchmarks on JSonSchemaBench suite (full, ~11K schemas). Vars: LIMIT, FRACTION, SHUFFLE
	cd gcg-paper/external-benchmarks/jsonschemabench/jsonschemabench/maskbench && \
		python3 scripts/run_maskbench.py --llg data/ $(BENCH_OPTS)

jsonschemabench-llg-analyze: ## Analyze llguidance benchmark results
	cd gcg-paper/external-benchmarks/jsonschemabench/jsonschemabench/maskbench && \
		python3 scripts/maskbench_results.py tmp/out--llg

jsonschemabench-compare: ## Compare sep1 and llguidance benchmark results side by side
	cd gcg-paper/external-benchmarks/jsonschemabench/jsonschemabench/maskbench && \
		python3 scripts/maskbench_results.py tmp/out--sep1 tmp/out--llg

# === Help ===

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
