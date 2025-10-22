#!/usr/bin/env python3
import json
import subprocess
import os
import tempfile
import sys
import uuid
import argparse
import re
import datetime
from typing import List

# --- Configuration ---

# The original, full vocabulary file that causes the mismatch.
ORIGINAL_VOCAB_PATH = ".temp.vocab.json"

# The command to run. Use a placeholder `{vocab_path}` which the script
# will replace with the path to the temporary vocab file for each test.
# NOTE: Using shell=True, so ensure this command is safe.
COMMAND_TEMPLATE = """
python scripts/compile.py \
    --grammar src/js_simplified2.ebnf \
    --output {constraint_path} \
    --vocab-path {vocab_path} \
    && \
REPEAT=1 AGG_METHOD="min" SKIP_CPP_BUILD=1 SKIP_PLOTS=1 \
CONSTRAINT_FILE="{constraint_path}" \
CODE_FILE=./src/example_code8.js \
bash python/run_benchmarks.sh \
    python/aug25/models/bruteforce_model.py \
    python/aug25/models/rust_model.py
"""

# The string to look for in the command's output that indicates a mismatch.
MISMATCH_INDICATOR = r"rust_model__js_constraint.*❌"

# The base directory where results will be saved.
RESULTS_BASE_DIR = "delta_debug_results"

# --- Script Logic ---

def run_test_with_vocab(vocab_list: List[str]) -> bool:
    """
    Runs the benchmark command with a given vocabulary list.

    Args:
        vocab_list: A list of strings representing the vocabulary.

    Returns:
        True if a mismatch occurs, False otherwise.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump(vocab_list, tmp_file)
        temp_vocab_path = tmp_file.name

    # Generate a unique path for the constraint file
    constraint_filename = f"js_constraint_{uuid.uuid4()}.json.gz"
    temp_constraint_path = os.path.join(tempfile.gettempdir(), constraint_filename)

    try:
        command = COMMAND_TEMPLATE.format(
            vocab_path=temp_vocab_path,
            constraint_path=temp_constraint_path
        ).strip()
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        output = result.stdout + result.stderr
        mismatch_found = bool(re.search(MISMATCH_INDICATOR, output))
        # if not mismatch_found:
        #     # Added for debugging: print command and output if the check fails.
        #     print("\n--- DEBUG: Mismatch indicator not found in output ---")
        #     print(f"Command executed:\n{command}")
        #     print(f"\nReturn code: {result.returncode}")
        #     print("\n--- Combined STDOUT and STDERR ---")
        #     print(output)
        #     print("--- END DEBUG ---")

        return mismatch_found
    finally:
        os.remove(temp_vocab_path)
        if os.path.exists(temp_constraint_path):
            os.remove(temp_constraint_path)


def main():
    """
    Main function to perform the vocabulary minimization.
    """
    parser = argparse.ArgumentParser(description="Delta-debug a vocabulary file to find the minimal set of tokens causing a mismatch.")
    parser.add_argument(
        '--keep-tokens',
        type=str,
        nargs='+',
        default=[],
        help="List of tokens to always keep in the vocabulary (e.g., --keep-tokens token1 token2). These tokens will not be removed during minimization."
    )
    args = parser.parse_args()

    print("--- Vocabulary Minimizer (Fast Delta-Debugging Method) ---")

    # Create a unique directory for this run's results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(RESULTS_BASE_DIR, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: '{results_dir}'")

    # 1. Load the original vocabulary
    if not os.path.exists(ORIGINAL_VOCAB_PATH):
        print(f"Error: Original vocabulary file not found at '{ORIGINAL_VOCAB_PATH}'")
        sys.exit(1)

    with open(ORIGINAL_VOCAB_PATH, 'r') as f:
        original_vocab = json.load(f)

    # --- New logic to separate fixed and removable vocab ---
    fixed_vocab = set(args.keep_tokens)

    # The list of tokens we will actually be minimizing
    removable_vocab = [token for token in original_vocab if token not in fixed_vocab]

    # Re-verify that all keep_tokens are actually in the original vocab
    if not fixed_vocab.issubset(set(original_vocab)):
        missing = fixed_vocab - set(original_vocab)
        print(f"Warning: The following --keep-tokens were not found in '{ORIGINAL_VOCAB_PATH}': {missing}")

    print(f"Loaded original vocabulary with {len(original_vocab)} tokens.")
    if fixed_vocab:
        print(f"Keeping {len(fixed_vocab)} tokens fixed. Minimizing on {len(removable_vocab)} removable tokens.")

    # 2. Initial check: Verify that the full vocabulary causes a mismatch
    print("\nStep 1: Verifying that the original vocabulary causes a mismatch...")
    full_test_vocab = list(fixed_vocab) + removable_vocab
    if not run_test_with_vocab(full_test_vocab):
        print("Error: The original vocabulary does not seem to cause a mismatch.")
        print("Please ensure the COMMAND_TEMPLATE and MISMATCH_INDICATOR are correct.")
        sys.exit(1)
    print("✅ Success: Original vocabulary confirmed to cause a mismatch.")

    # 3. Start the minimization process
    minimal_removable_vocab = list(removable_vocab)
    print(f"\nStep 2: Starting minimization from {len(minimal_removable_vocab)} removable tokens using chunk removal...")

    reduction_step = 0
    chunk_size = len(minimal_removable_vocab) // 2

    while chunk_size >= 1:
        print(f"\n--- Testing with chunk size: {chunk_size} ---")

        # Keep trying with this chunk size as long as we are making progress
        while True:
            num_chunks = (len(minimal_removable_vocab) + chunk_size - 1) // chunk_size
            if num_chunks < 2:
                break  # Not enough chunks to test for removal

            found_reduction_in_pass = False
            start_index = 0
            while start_index < len(minimal_removable_vocab):
                end_index = min(start_index + chunk_size, len(minimal_removable_vocab))

                # Create a test vocabulary by removing the current chunk from the removable list
                test_removable_vocab = minimal_removable_vocab[:start_index] + minimal_removable_vocab[end_index:]
                excluded_vocab = minimal_removable_vocab[start_index:end_index]

                # The actual vocab to test is the fixed part plus the test removable part
                test_vocab = list(fixed_vocab) + test_removable_vocab

                print(f"Testing removal of chunk {start_index}-{end_index - 1} ({len(excluded_vocab)} tokens)... ", end='', flush=True)
                mismatch_persists = run_test_with_vocab(test_vocab)

                if mismatch_persists:
                    # Mismatch still occurs, so the removal was successful
                    minimal_removable_vocab = test_removable_vocab

                    # Save intermediate progress
                    reduction_step += 1
                    new_size = len(minimal_removable_vocab) + len(fixed_vocab)
                    filename = f"step_{reduction_step:04d}_size_{new_size}.json"
                    output_path = os.path.join(results_dir, filename)
                    with open(output_path, 'w') as f:
                        json.dump(list(fixed_vocab) + minimal_removable_vocab, f, indent=2)

                    found_reduction_in_pass = True
                    print(f"REMOVED. New removable vocab size: {len(minimal_removable_vocab)}")
                    if len(excluded_vocab) < 10:
                        print(f"  (Removed: {excluded_vocab})")

                    # Restart the scan for this chunk size with the smaller vocab
                    break # from `while start_index < ...`
                else:
                    # This chunk is necessary, so move to the next one
                    print("KEPT.")
                    start_index = end_index

            if not found_reduction_in_pass:
                # No more reductions possible with this chunk size.
                print("No more reductions possible with this chunk size.")
                break  # from `while True`

        chunk_size //= 2

    # 4. Final result
    print("\n--- Minimization Complete ---")
    minimal_vocab = list(fixed_vocab) + minimal_removable_vocab
    print(f"\nOriginal vocabulary size: {len(original_vocab)}")
    print(f"Minimal vocabulary size: {len(minimal_vocab)}")
    print(f"Fixed tokens: {len(fixed_vocab)}")
    print(f"Removable tokens reduced from {len(removable_vocab)} to {len(minimal_removable_vocab)}")
    print("\nEssential tokens causing the mismatch:")
    print(json.dumps(minimal_vocab, indent=2))

    # Save the result to a file
    final_output_path = os.path.join(results_dir, "minimal_vocab.json")
    with open(final_output_path, 'w') as f:
        json.dump(minimal_vocab, f, indent=2)
    print(f"\n✅ Minimal vocabulary saved to '{final_output_path}'")
    print(f"Intermediate steps are also saved in '{results_dir}'")


if __name__ == "__main__":
    # Ensure required files/scripts exist before running
    required_scripts = [
        "scripts/compile.py",
        "python/run_benchmarks.sh",
    ]
    for script in required_scripts:
        if not os.path.exists(script):
            print(f"Error: Required script '{script}' not found in the current directory.")
            sys.exit(1)

    main()