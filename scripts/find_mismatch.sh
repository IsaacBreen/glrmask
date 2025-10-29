#!/bin/bash
set -o pipefail

# --- Configuration ---
# Number of parallel jobs to run
MAX_JOBS=3

# List of optimization flags to test
OPTS=(
    "prune_dead_paths"
    "prune_unproductive_paths"
    "canonicalize_end_nodes"
    "compress_edges"
    "compress_unary_chains"
    "factor_common_destinations"
    "merge_structural"
    "merge_bisimulation"
    "merge_global_atoms"
    "eliminate_pop0_except_roots"
    "merge_equivalent_llm_tokens"
    "reorder_llm_tokens"
    "generalize_sids"
)

# Base JSON config with all optimizations disabled
BASE_JSON_CONFIG='{"prune_dead_paths": false, "prune_unproductive_paths": false, "canonicalize_end_nodes": false, "compress_edges": false, "compress_unary_chains": false, "factor_common_destinations": false, "merge_structural": false, "merge_bisimulation": false, "merge_global_atoms": false, "eliminate_pop0_except_roots": false, "merge_equivalent_llm_tokens": false, "reorder_llm_tokens": false, "generalize_sids": false}'

# --- Setup ---
# Create directories for logs and compiled constraints to avoid race conditions
LOG_DIR="bisection_logs"
CACHE_DIR=".cache/test_vocabs_bisection"
mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_DIR"

# Check for jq dependency
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is not installed. Please install it to run this script."
    echo "On Debian/Ubuntu: sudo apt-get install jq"
    echo "On macOS: brew install jq"
    exit 1
fi

# --- Colors for output ---
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# --- Main test function ---
# This function runs the test for a single optimization flag.
run_test() {
    local opt_name="$1"
    local log_file="$LOG_DIR/${opt_name}.log"
    local constraint_file="$CACHE_DIR/js_constraint_${opt_name}.json.gz"

    printf "[${YELLOW}RUNNING${NC}] Testing optimization: %s\n" "$opt_name"

    # 1. Generate the specific AICI_TRIE3_CONFIG for this run
    local current_config
    current_config=$(jq --arg key "$opt_name" '.[$key] = true' <<< "$BASE_JSON_CONFIG")

    # 2. Construct and run the full command, redirecting output to a log file
    (
        AICI_TRIE3_CONFIG="$current_config" python scripts/compile.py \
            --grammar src/js_simplified0_5.ebnf \
            --output "$constraint_file" \
            --vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json" --token-len 1 \
        && \
        SKIP_PLOTS=1 REPEAT=1 AGG_METHOD="min" SKIP_CPP_BUILD=1 CONSTRAINT_FILE="$constraint_file" CODE_FILE=./src/example_code9.js \
        bash python/run_benchmarks.sh python/aug25/models/bruteforce_rust_model.py python/aug25/models/rust_model.py
    ) > "$log_file" 2>&1

    # 3. Check the result and report status
    if [ $? -ne 0 ]; then
        printf "[${RED} FAILED ${NC}] %-35s (Command failed. Check log: %s)\n" "$opt_name" "$log_file"
        return
    fi

    # Parse the log file to find the mismatch count for the rust_model
    # This grep is now more specific to avoid matching 'bruteforce_rust_model'
    local mismatch_count
    mismatch_count=$(grep '^[[:space:]]*rust_model' "$log_file" | awk '{print $3}')

    # Robustly check the result
    if ! [[ "$mismatch_count" =~ ^[0-9]+$ ]]; then
        # This catches empty strings, multi-line strings, or non-numeric values
        printf "[${RED} FAILED ${NC}] %-35s (Could not parse mismatch count. Check log: %s)\n" "$opt_name" "$log_file"
    elif [[ "$mismatch_count" -gt 0 ]]; then
        printf "[${RED}MISMATCH${NC}] %-35s (Mismatch count: %s)\n" "$opt_name" "$mismatch_count"
    else
        printf "[${GREEN}   OK   ${NC}] %-35s\n" "$opt_name"
    fi
}

# --- Parallel Execution Logic ---
echo "Starting optimization bisection..."
echo "Running up to $MAX_JOBS jobs in parallel."
echo "Logs will be stored in the '$LOG_DIR' directory."
echo "-----------------------------------------------------"

# Export the function so it's available to subshells
export -f run_test
export BASE_JSON_CONFIG LOG_DIR CACHE_DIR GREEN RED YELLOW NC

# Use a simple background job management loop
for opt in "${OPTS[@]}"; do
    # Wait until there is a free spot in the job pool
    while (( $(jobs -p | wc -l) >= MAX_JOBS )); do
        sleep 1
    done

    # Run the test function in the background
    run_test "$opt" &
done

# Wait for all remaining background jobs to complete
wait

echo "-----------------------------------------------------"
echo "All tests complete."