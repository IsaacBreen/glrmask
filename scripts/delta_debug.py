#!/usr/bin/env python3
import json
import subprocess
import os
import tempfile
import sys
import uuid
import datetime
from multiprocessing import Pool
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
MISMATCH_INDICATOR = "rust_model__js_constraint                ❌"

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
        return MISMATCH_INDICATOR in output
    finally:
        os.remove(temp_vocab_path)
        if os.path.exists(temp_constraint_path):
            os.remove(temp_constraint_path)


def main():
    """
    Main function to perform the vocabulary minimization.
    """
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

    num_cpus = max(1, os.cpu_count() // 2)
    print(f"Using {num_cpus} parallel processes.")

    reduction_step = 0
    chunk_size = len(minimal_vocab) // 2

    with Pool(processes=num_cpus) as pool:
        while chunk_size >= 1:
            print(f"\n--- Testing with chunk size: {chunk_size} ---")
            removed_in_pass = False

            while True:  # Keep trying with this chunk size as long as we are making progress
                num_chunks = (len(minimal_vocab) + chunk_size - 1) // chunk_size
                if num_chunks < 2:
                    break  # Not enough chunks to test for removal

                tasks = []
                chunk_metadata = []
                start_index = 0
                while start_index < len(minimal_vocab):
                    end_index = min(start_index + chunk_size, len(minimal_vocab))
                    test_vocab = minimal_vocab[:start_index] + minimal_vocab[end_index:]

                    if not test_vocab:
                        start_index = end_index
                        continue

                    tasks.append(test_vocab)
                    chunk_metadata.append({
                        "start": start_index,
                        "end": end_index,
                        "excluded": minimal_vocab[start_index:end_index]
                    })
                    start_index = end_index

                print(f"Testing removal of {len(tasks)} chunks in parallel...")
                results = pool.map(run_test_with_vocab, tasks)

                found_reduction = False
                for i, mismatch_persists in enumerate(results):
                    if mismatch_persists:
                        # Mismatch still occurs, so the removal was successful
                        meta = chunk_metadata[i]
                        minimal_vocab = tasks[i]

                        # Save intermediate progress
                        reduction_step += 1
                        new_size = len(minimal_vocab)
                        filename = f"step_{reduction_step:04d}_size_{new_size}.json"
                        output_path = os.path.join(results_dir, filename)
                        with open(output_path, 'w') as f:
                            json.dump(minimal_vocab, f, indent=2)

                        removed_in_pass = True
                        found_reduction = True
                        print(f"REMOVED chunk (indices {meta['start']}-{meta['end']-1}). New size: {len(minimal_vocab)}")
                        if len(meta['excluded']) < 10:
                            print(meta['excluded'])

                        # Restart the scan for this chunk size with the smaller vocab
                        break

                if not found_reduction:
                    print("No more reductions possible with this chunk size.")
                    break  # Exit the inner while True loop

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