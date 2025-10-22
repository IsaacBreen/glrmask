#!/usr/bin/env python3
import json
import subprocess
import os
import tempfile
import random
import shutil
import sys
from typing import List

# --- Configuration ---

# The original, full vocabulary file that causes the mismatch.
# The script will create a copy of this to work with.
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
SKIP_CPP_BUILD=1 REPEAT=1 AGG_METHOD="min" SKIP_RUST_BUILD=1 SKIP_CPP_BUILD=1 \
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
    # Use a temporary file for the vocabulary
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump(vocab_list, tmp_file)
        temp_vocab_path = tmp_file.name

    try:
        # Format the command with the temporary vocab path
        command = COMMAND_TEMPLATE.format(vocab_path=temp_vocab_path).strip()

        # Execute the command
        # We capture output and don't check for exit codes, as the script might "succeed"
        # but still show a mismatch.
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        # Check stdout for the mismatch indicator
        output = result.stdout + result.stderr
        if MISMATCH_INDICATOR in output:
            # print("DEBUG: Mismatch detected.")
            return True
        else:
            # print("DEBUG: No mismatch.")
            return False

    finally:
        # Clean up the temporary file
        os.remove(temp_vocab_path)


def main():
    """
    Main function to perform the vocabulary minimization.
    """
    print("--- Vocabulary Minimizer ---")

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
    print(f"\nStep 2: Starting minimization from {len(minimal_vocab)} tokens...")

    pass_num = 0
    while True:
        pass_num += 1
        tokens_removed_this_pass = 0

        # Shuffle the list to avoid any order-dependent bias
        tokens_to_try = list(minimal_vocab)
        random.shuffle(tokens_to_try)

        print(f"\n--- Pass {pass_num} (current size: {len(minimal_vocab)}) ---")

        for i, token_to_remove in enumerate(tokens_to_try):
            # Don't try to remove the last token
            if len(minimal_vocab) <= 1:
                break

            # Create a test vocabulary without the current token
            test_vocab = [t for t in minimal_vocab if t != token_to_remove]

            progress = f"[{i + 1}/{len(tokens_to_try)}]"
            print(f"{progress} Trying to remove token: '{token_to_remove}'...", end='', flush=True)

            # Run the test
            if run_test_with_vocab(test_vocab):
                # Mismatch still occurs, so the removal was successful
                minimal_vocab = test_vocab
                tokens_removed_this_pass += 1
                print(f" REMOVED. New size: {len(minimal_vocab)}")
                # Restart the pass with the smaller set for efficiency
                break
            else:
                # Mismatch disappeared, so this token is essential
                print(" KEPT (essential).")

        # If a full pass completes with no removals, we are done
        if tokens_removed_this_pass == 0:
            print("\n--- Minimization Complete ---")
            print("A full pass was completed with no tokens removed.")
            break

    # 4. Final result
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
        "python/aug25/models/bruteforce_model.py",
        "python/aug25/models/rust_model.py"
    ]
    for script in required_scripts:
        if not os.path.exists(script):
            print(f"Error: Required script '{script}' not found in the current directory.")
            sys.exit(1)

    main()