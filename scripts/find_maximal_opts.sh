#!/bin/bash
set -o pipefail

# --- Configuration ---
# List of optimization flags to test, potentially ordered by expected impact or stability
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
LOG_DIR="maximal_set_logs"
CACHE_DIR=".cache/test_vocabs_maximal_set"

# --- Colors for output ---
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Test function ---
# This function runs the test for a given configuration.
# Returns:
#   0: Success (OK)
#   1: Command failed (non-zero exit code)
#   2: Mismatch detected
#   3: Could not parse mismatch count from log
run_test_with_config() {
    local test_config="$1"
    local log_file="$2"
    local constraint_file="$3"

    # Construct and run the full command, redirecting output to the log file
    (
        AICI_TRIE3_CONFIG="$test_config" python scripts/compile.py \
            --grammar src/js_simplified0_5.ebnf \
            --output "$constraint_file" \
            --vocab-url "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json" --token-len 1 \
        && \
        SKIP_PLOTS=1 REPEAT=1 AGG_METHOD="min" SKIP_CPP_BUILD=1 SKIP_RUST_BUILD=1 CONSTRAINT_FILE="$constraint_file" CODE_FILE=./src/example_code9.js \
        bash python/run_benchmarks.sh python/aug25/models/bruteforce_rust_model.py python/aug25/models/rust_model.py
    ) > "$log_file" 2>&1

    # Check the result
    if [ $? -ne 0 ]; then
        return 1 # Command failed
    fi

    # Use -e to specify the pattern, preventing grep from interpreting
    # the leading '---' as an option.
    local mismatch_count
    mismatch_count=$(grep -A 5 -e '--- get_mask() Timings ---' "$log_file" | grep '^[[:space:]]*rust_model' | awk '{print $3}')

    # Robustly check the result
    if ! [[ "$mismatch_count" =~ ^[0-9]+$ ]]; then
        return 3 # Could not parse
    elif [[ "$mismatch_count" -gt 0 ]]; then
        return 2 # Mismatch found
    else
        return 0 # OK
    fi
}

# --- Main Execution Logic ---
main() {
    # --- Setup within main function ---
    echo "Cleaning up previous run directories..."
    rm -rf "$LOG_DIR" "$CACHE_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$CACHE_DIR"

    # Check for jq dependency
    if ! command -v jq &> /dev/null; then
        echo "Error: 'jq' is not installed. Please install it to run this script."
        echo "On Debian/Ubuntu: sudo apt-get install jq"
        echo "On macOS: brew install jq"
        exit 1
    fi

    echo "Starting search for the maximal set of compatible optimizations..."
    echo "Logs will be stored in the '$LOG_DIR' directory."
    echo "--------------------------------------------------------------------"

    # State variables
    local good_opts=()
    local current_good_config="$BASE_JSON_CONFIG"
    local step_counter=0

    for opt_to_try in "${OPTS[@]}"; do
        step_counter=$((step_counter + 1))
        
        printf "\n[${BLUE}STEP %02d${NC}] Testing addition of: ${YELLOW}%s${NC}\n" "$step_counter" "$opt_to_try"
        
        # 1. Create the configuration for the next test run
        local next_config
        next_config=$(jq --arg key "$opt_to_try" '.[$key] = true' <<< "$current_good_config")
        
        # 2. Define unique files for this run
        local log_file="$LOG_DIR/step_${step_counter}_add_${opt_to_try}.log"
        local constraint_file="$CACHE_DIR/constraint_step_${step_counter}.json.gz"

        # 3. Run the test
        printf "Running test... (log: %s)\n" "$log_file"
        run_test_with_config "$next_config" "$log_file" "$constraint_file"
        local result=$?

        # 4. Evaluate the result and update state
        case $result in
            0)
                printf "[${GREEN}SUCCESS${NC}] Optimization '${YELLOW}%s${NC}' is compatible. Adding to the set.\n" "$opt_to_try"
                current_good_config="$next_config"
                good_opts+=("$opt_to_try")
                ;;
            1)
                printf "[${RED} FAILED${NC}]  Optimization '${YELLOW}%s${NC}' caused a command failure. Skipping. (log: %s)\n" "$opt_to_try" "$log_file"
                ;;
            2)
                local mismatch_count
                mismatch_count=$(grep -A 5 -e '--- get_mask() Timings ---' "$log_file" | grep '^[[:space:]]*rust_model' | awk '{print $3}')
                printf "[${RED}MISMATCH${NC}] Optimization '${YELLOW}%s${NC}' caused %s mismatches. Skipping. (log: %s)\n" "$opt_to_try" "$mismatch_count" "$log_file"
                ;;
            3)
                printf "[${RED} FAILED${NC}]  Optimization '${YELLOW}%s${NC}' produced unparsable output. Skipping. (log: %s)\n" "$opt_to_try" "$log_file"
                ;;
        esac
        
        echo "Current compatible set: ${good_opts[*]}"
    done

    # --- Final Summary ---
    echo "--------------------------------------------------------------------"
    echo "Search complete."
    if [ ${#good_opts[@]} -eq 0 ]; then
        echo -e "${YELLOW}No compatible optimizations were found.${NC}"
    else
        echo -e "${GREEN}Maximal compatible set of optimizations found:${NC}"
        for opt in "${good_opts[@]}"; do
            echo "  - $opt"
        done

        echo
        echo -e "${GREEN}Final working AICI_TRIE3_CONFIG JSON:${NC}"
        echo "$current_good_config" | jq .
    fi
    echo "--------------------------------------------------------------------"
}

# --- Script Entry Point ---
# This ensures the script's logic runs only when executed.
main "$@"