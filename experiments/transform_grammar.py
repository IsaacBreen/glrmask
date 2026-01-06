#!/usr/bin/env python3
"""
Grammar Transformation Experiment Script.

1. Loads a JSON schema.
2. Generates a GrammarDefinition.
3. Dumps it to JSON.
4. Loads it back from JSON.
5. Compiles it, capturing debug output to measure timing.
"""

import argparse
import json
import os
import sys
import glob
import time
import re
import tempfile

# Add project root and python/ dir to sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "python"))

def find_schema_by_id(schema_id: str) -> str:
    """Find a schema file by its ID."""
    search_dirs = [
        "gcg-paper/downloads/repos/jsonschemabench/data",
        "gcg-paper/downloads/repos/jsonschemabench/maskbench/data",
        "gcg-paper/hard_schemas/data",
        "gcg-paper/hard_schemas/schemas",
        "gcg-paper/hard_schemas/tests",
        "gcg-paper/json_schema_test_suite/data",
    ]
    
    # Try absolute path first
    if os.path.isfile(schema_id):
        return schema_id

    for base_dir in search_dirs:
        full_base = os.path.join(repo_root, base_dir)
        # Try exact match first
        pattern = f"{full_base}/**/{schema_id}.json"
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
        
        # Also try with schema_id as the full filename
        pattern = f"{full_base}/{schema_id}.json"
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
            
    return None

class StdoutCapturer:
    """Captures C-level stdout (like Rust println!) to a file."""
    def __init__(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.original_stdout_fd = sys.stdout.fileno()
        self.saved_stdout_fd = os.dup(self.original_stdout_fd)
        
    def __enter__(self):
        # Flush python stdout
        sys.stdout.flush()
        # Redirect stdout to file
        os.dup2(self.temp_file.fileno(), self.original_stdout_fd)
        return self.temp_file.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush and restore
        sys.stdout.flush()
        os.dup2(self.saved_stdout_fd, self.original_stdout_fd)
        os.close(self.saved_stdout_fd)
        self.temp_file.close()

def parse_timings(log_content: str):
    """Extracts timings from the log content."""
    timings = {}
    
    # build_template_dwas
    # Pattern: "Built {} terminal DWAs in {:?}"
    match = re.search(r"Built \d+ terminal DWAs in (\d+\.\d+)(s|ms|µs|m)", log_content)
    if match:
        timings['build_template_dwas'] = match.group(0)
    
    return timings

def main():
    parser = argparse.ArgumentParser(description="Grammar Transformation Experiment")
    parser.add_argument("--schema-id", required=True, help="ID of the schema to test")
    parser.add_argument("--debug-level", type=int, default=4, help="Debug level for Rust macros")
    args = parser.parse_args()

    # Set logging environment early
    os.environ["MACRO_DEBUG_LEVEL"] = str(args.debug_level)
    
    # Import _sep1 NOW, after env var is set
    global _sep1
    import _sep1
    import urllib.request

    schema_path = find_schema_by_id(args.schema_id)
    if not schema_path:
        print(f"Error: Schema {args.schema_id} not found.")
        sys.exit(1)

    print(f"Loading schema: {schema_path}")
    with open(schema_path, 'r') as f:
        schema_str = f.read()

    # 1. Generate Grammar
    print("Generating grammar definition...")
    # Does grammar_definition_from_json_schema perform optimizations? Yes.
    # We want to emulate the full process.
    gd = _sep1.grammar_definition_from_json_schema(schema_str)
    
    # 2. Dump to JSON
    print("Dumping to JSON...")
    json_str = gd.to_json_string()
    
    with open("temp_grammar.json", "w") as f:
        f.write(json_str)
    print(f"Dumped {len(json_str)} bytes to temp_grammar.json")
    
    # 3. Load from JSON
    print("Loading from JSON...")
    start_load = time.time()
    gd2 = _sep1.GrammarDefinition.from_json_string(json_str)
    print(f"Loaded in {(time.time() - start_load)*1000:.2f}ms")

    # 3b. Load Vocabulary (needed for Constraint)
    print("Loading vocabulary...")
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        token_to_id = {}
        for token_id in range(enc.n_vocab):
            token_bytes = enc.decode_single_token_bytes(token_id)
            token_to_id[token_bytes] = token_id
    except ImportError:
        print("tiktoken not found, downloading vocab.json...")
        vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"
        with urllib.request.urlopen(vocab_url) as resp:
            vocab = json.loads(resp.read().decode())
        token_to_id = {k.encode('utf-8'): v for k, v in vocab.items()}

    # 4. Compile and Time
    print("Compiling grammar & Building Constraint (capturing logs)...")
    
    logs = ""
    start_compile = time.time()
    
    with StdoutCapturer() as log_file:
        print("DEBUG: Python stdout check inside capture")
        sys.stdout.flush()
        
        # Compile Grammar
        compiled = gd2.compile()
        
        # Build Constraint (triggers precompute1, build_parser_dwa, etc.)
        constraint = _sep1.GrammarConstraint(compiled, token_to_id)
        
        sys.stdout.flush()
        
    total_compile_time = (time.time() - start_compile) * 1000
    
    with open(log_file, 'r') as f:
        logs = f.read()
    
    os.unlink(log_file)
    
    # 5. Extract Timings
    print("\n--- Captured Logs (First 50 lines) ---")
    print("\n".join(logs.splitlines()[:50]))
    print("...\n")

    metrics = {
        "Total Pipeline (Python measured)": f"{total_compile_time:.2f}ms"
    }
    
    # Helper to clean regex matches
    def get_time(pattern, text):
        m = re.search(pattern, text)
        return m.group(1) if m else "N/A"

    # Tokenizer
    # NFA -> DFA: 67 -> 35 states (369.50µs)
    # Minimized DFA 35 -> 34 states in 228.88µs
    metrics["Tokenizer (NFA->DFA)"] = get_time(r"NFA → DFA: .*? \((.*?)\)", logs)
    metrics["Tokenizer (Minimize)"] = get_time(r"Minimized DFA .*? in ([^\n]+)", logs)
    
    # Combined Equivalence (Pre-calculation)
    # Combined equivalence analysis complete: 206 vocab classes, 34 representative states (total 8.475875ms)
    metrics["Combined Equivalence (Total)"] = get_time(r"Combined equivalence analysis complete: .*? \(total (.*?)\)", logs)
        
    # build_template_dwas
    # Built 11 terminal DWAs in 1.497ms
    metrics["build_template_dwas"] = get_time(r"Built \d+ terminal DWAs in ([^\n]+)", logs)
    
    # build_parser_dwa components
    # DWA Vocab Optimization: Tokens 206 -> 126, Ranges 271 -> 444. Time: 885.67µs
    metrics["Parser DWA Vocab Opt"] = get_time(r"DWA Vocab Optimization: .*? Time: ([^\n]+)", logs)
    
    print("\n=== Timing Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    
    print("\n(Note: Timings extracted from debug logs. 'Total Pipeline' includes Python overhead.)")
    with open("compilation.log", "w") as f:
        f.write(logs)
    print("Full logs saved to compilation.log")

if __name__ == "__main__":
    main()
