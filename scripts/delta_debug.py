#!/usr/bin/env python3
import json
import subprocess
import os
import tempfile
import sys
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
    --output .cache/test_vocabs/js_constraint.json.gz \
    --vocab-path {vocab_path} \
    && \
REPEAT=1 AGG_METHOD="min" SKIP_CPP_BUILD=1 SKIP_PLOTS=1 \
CONSTRAINT_FILE=".cache/test_vocabs/js_constraint.json.gz" \
CODE_FILE=./src/example_code8.js \
bash python/run_benchmarks.sh \
    python/aug25/models/bruteforce_model.py \
    python/aug25/models/rust_model.py
"""

# The string to look for in the command's output that indicates a mismatch.
MISMATCH_INDICATOR = "❌"

# The file where the final, minimal vocabulary will be saved.
MINIMAL_VOCAB_OUTPUT_PATH = "minimal_vocab.json"

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

    try:
        command = COMMAND_TEMPLATE.format(vocab_path=temp_vocab_path).strip()
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        output = result.stdout + result.stderr
        return MISMATCH_INDICATOR in output
    finally:
        os.remove(temp_vocab_path)


def main():
    """
    Main function to perform the vocabulary minimization.
    """
    print("--- Vocabulary Minimizer (Fast Delta-Debugging Method) ---")

    # 1. Load the original vocabulary
    if not os.path.exists(ORIGINAL_VOCAB_PATH):
        print(f"Error: Original vocabulary file not found at '{ORIGINAL_VOCAB_PATH}'")
        sys.exit(1)

    with open(ORIGINAL_VOCAB_PATH, 'r') as f:
        original_vocab = json.load(f)

    print(f"Loaded original vocabulary with {len(original_vocab)} tokens.")

    # 2. Initial check: Verify that the full vocabulary causes a mismatch
    print("\nStep 1: Verifying that the original vocabulary causes a mismatch...")
    if not run_test_with_vocab(original_vocab):
        print("Error: The original vocabulary does not seem to cause a mismatch.")
        print("Please ensure the COMMAND_TEMPLATE and MISMATCH_INDICATOR are correct.")
        sys.exit(1)
    print("✅ Success: Original vocabulary confirmed to cause a mismatch.")

    # 3. Start the minimization process
    minimal_vocab = list(original_vocab)
    print(f"\nStep 2: Starting minimization from {len(minimal_vocab)} tokens using chunk removal...")

    chunk_size = len(minimal_vocab) // 2
    while chunk_size >= 1:
        print(f"\n--- Testing with chunk size: {chunk_size} ---")
        removed_in_pass = False
        start_index = 0
        while start_index < len(minimal_vocab):
            end_index = min(start_index + chunk_size, len(minimal_vocab))

            # Create a test vocabulary by removing the current chunk
            test_vocab = minimal_vocab[:start_index] + minimal_vocab[end_index:]

            # Don't test an empty vocabulary, just skip
            if not test_vocab:
                start_index = end_index
                continue

            num_chunks = (len(minimal_vocab) + chunk_size - 1) // chunk_size
            current_chunk_num = (start_index // chunk_size) + 1

            progress = f"[{current_chunk_num}/{num_chunks}]"
            print(f"{progress} Trying to remove {end_index - start_index} tokens (indices {start_index}-{end_index-1})...", end='', flush=True)

            if run_test_with_vocab(test_vocab):
                # Mismatch still occurs, so the removal was successful
                minimal_vocab = test_vocab
                removed_in_pass = True
                print(f" REMOVED. New size: {len(minimal_vocab)}")
                # Restart the scan for this chunk size, as the list has changed
                start_index = 0
            else:
                # Mismatch disappeared, this chunk is essential. Move to the next one.
                print(" KEPT (essential).")
                if len(test_vocab) < 10:
                    print(test_vocab)
                start_index = end_index

        # If we completed a full pass for this chunk size without removing anything,
        # it's time to try smaller chunks.
        if not removed_in_pass:
            chunk_size //= 2

    # 4. Final result
    print("\n--- Minimization Complete ---")
    print(f"\nOriginal vocabulary size: {len(original_vocab)}")
    print(f"Minimal vocabulary size: {len(minimal_vocab)}")
    print("\nEssential tokens causing the mismatch:")
    print(json.dumps(minimal_vocab, indent=2))

    # Save the result to a file
    with open(MINIMAL_VOCAB_OUTPUT_PATH, 'w') as f:
        json.dump(minimal_vocab, f, indent=2)
    print(f"\n✅ Minimal vocabulary saved to '{MINIMAL_VOCAB_OUTPUT_PATH}'")


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