# Research Project Makefile
# Common commands for paper writing and research

.PHONY: paper paper-watch paper-clean notes-today help build test test-js test-json-schema test-schema-packagejson test-schema-github test-schema-sarif test-schema-meta test-schema-extra test-schema-kestra test-schema-vegalite test-schema-apollo test-schema-liquibase test-diff test-diff-dfa ffi viz viz-clean all

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

test-js: ## Compile the JavaScript grammar (verifies it compiles)
	MACRO_DEBUG_LEVEL=4 timeout 120 python scripts/compile.py \
		--grammar src/js.ebnf \
		--format ebnf \
		--output .cache/test_vocabs/constraint_js.json.gz \
		--vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json" \
		$(if $(SKIP_MATURIN),--no-recompile,)

test-json-schema: ## Compile a JSON schema grammar (verifies schema-to-EBNF works)
	SCHEMA_FILE="gcg-paper/downloads/repos/jsonschemabench/data/Github_ultra/o21378.json" \
		python3 scripts/test_json_schema.py

test-json-schema-o1051: ## Compile o1051 (Github Hard) schema
	SCHEMA_FILE="gcg-paper/downloads/repos/jsonschemabench/data/Github_hard/o1051.json" \
		python3 scripts/test_json_schema.py

test-tsconfig: ## Compile TSConfig schema
	SCHEMA_FILE="gcg-paper/hard_schemas/data/TSConfig---tsconfig.json" \
		python3 scripts/test_json_schema.py

PYTHON ?= python

test-schema-id: ## Compile any benchmark schema by ID (usage: make test-schema-id ID=ApolloRouter---apollo-router-2.9.0)
	@if [ -z "$(ID)" ]; then echo "Usage: make test-schema-id ID=<schema_id>"; exit 1; fi
	SCHEMA_ID="$(ID)" $(PYTHON) scripts/test_json_schema.py

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



# === Hard Schema Compilation Tests ===
# These use the Rust grammar_compiler binary directly with --json-schema

test-schema-packagejson: build ## Compile PackageJson schema
	MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/hard_schemas/data/PackageJson---package.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_packagejson.json.gz

test-schema-github: build ## Compile GithubWorkflow schema
	MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/hard_schemas/data/GithubWorkflow---github-workflow.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_github.json.gz

test-schema-sarif: build ## Compile SARIF schema
	MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/hard_schemas/data/SARIF---sarif-2.1.0-rtm.1.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_sarif.json.gz

test-schema-meta: build ## Compile JSON Schema meta-schema (draft v4)
	MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/hard_schemas/data/JsonSchemaMeta---schema-draft-v4.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_meta.json.gz

test-schema-extra: build ## Compile bamboo-spec from SchemaStore_Extra
	MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
		--json-schema gcg-paper/hard_schemas/data/SchemaStore_Extra---bamboo-spec.json \
		--vocab .cache/test_vocabs/gpt2_vocab.json \
		--output .cache/test_vocabs/constraint_bamboo.json.gz

test-schema-kestra: build ## Compile Kestra schema (WARNING: ~8MB, very slow)
	MACRO_DEBUG_LEVEL=4 ./target/release/grammar-compiler \
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

# === Help ===

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
