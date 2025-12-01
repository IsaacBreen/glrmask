# Research Project Makefile
# Common commands for paper writing and research

.PHONY: paper paper-watch paper-clean notes-today help build test ffi viz viz-clean all

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
	cd python && RUSTFLAGS=-Awarnings maturin develop -r

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

build: ## Build the Rust project
	cargo build --release

# === Help ===

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
