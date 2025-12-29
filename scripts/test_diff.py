"""
Test Sep1 diff grammar compilation and constraint creation.

This script takes a text file and:
1. Generates an EBNF grammar that describes valid git diffs for that file
2. Compiles the grammar and creates a constraint

Usage:
    # With environment variable
    SOURCE_FILE="path/to/file.txt" python scripts/test_diff.py
    
    # Or directly via make
    make test-diff FILE=path/to/file.txt
"""

import json
import time
import os
import sys

# Add project root and python/ dir to sys.path to find _sep1 module
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "python"))

import _sep1


def generate_diff_grammar(source_path: str) -> str:
    """
    Generates an EBNF grammar that validates a 'git diff'-like format
    for a specific source file.

    Args:
        source_path: The path to the input file to base the grammar on.
        
    Returns:
        The EBNF grammar as a string.
    """
    print(f"Reading source file: {source_path}")
    with open(source_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    num_lines = len(lines)
    grammar_parts = []

    # --- 1. Preamble and Top-Level Rules ---
    grammar_parts.append("root ::= DIFF;")
    grammar_parts.append("DIFF ::= FILE_HEADER? ( HUNK_HEADER S0 )? EOF;")
    grammar_parts.append("FILE_HEADER ::= GIT_LINE INDEX_LINE? MINUS_LINE PLUS_FILE_LINE;")
    grammar_parts.append("EOF  ::= '<|EOF|>';")  # Ensure this matches your tokenizer's EOF
    grammar_parts.append("")

    # --- 2. 'S' Rules (Search for Hunk Start) ---
    grammar_parts.append("// 'S' rules: Find the start of a hunk")
    for i in range(num_lines):
        # Try to match line i, or skip and try line i+1
        grammar_parts.append(f"S{i} ::= LINE{i} | S{i+1};")

    # If we reach the end of the file, we only allow trailing additions
    grammar_parts.append(f"S{num_lines} ::= PLUS_LINE*;")
    grammar_parts.append("")

    # --- 3. 'LINE' Rules (Match Context/Deletion) ---
    grammar_parts.append("// 'LINE' rules: Match content exactly, then continue or new hunk")
    for i in range(num_lines):
        # After matching line i, we can:
        # 1. Continue immediately to line i+1
        # 2. Have some additions, then a Hunk Header, skipping to i+1
        if i < num_lines - 1:
            continuation = f"( LINE{i+1} | PLUS_LINE* HUNK_HEADER S{i+1} )?"
        else:
            continuation = f"( PLUS_LINE* HUNK_HEADER S{num_lines} )?"

        # NOTE: PLUS_LINE* allows insertions *before* the context/deletion line
        grammar_parts.append(f"LINE{i} ::= PLUS_LINE* CONTENT{i} {continuation};")
    grammar_parts.append("")

    # --- 4. Terminal Definitions ---
    grammar_parts.append("// --- TERMINALS ---")
    grammar_parts.append(r"GIT_LINE         ::= 'diff --git' [^\n\r]* NEWLINE;")
    grammar_parts.append(r"INDEX_LINE       ::= 'index' [^\n\r]* NEWLINE;")
    grammar_parts.append(r"MINUS_LINE       ::= '---' [^\n\r]* NEWLINE;")
    grammar_parts.append(r"PLUS_FILE_LINE   ::= '+++' [^\n\r]* NEWLINE;")
    grammar_parts.append(r"HUNK_HEADER      ::= '@@' [^\n\r]* NEWLINE;")
    grammar_parts.append(r"PLUS_LINE        ::= '+' [^\n\r]* NEWLINE;")

    # Safer NEWLINE definition (using escaped chars rather than literal line breaks)
    grammar_parts.append(r"NEWLINE          ::= '\n' | '\r\n';")
    grammar_parts.append("")

    # --- 5. Content Lines ---
    grammar_parts.append("// Context-line terminals")
    for i, line in enumerate(lines):
        content = line.rstrip('\r\n')

        if not content:
            # Strict diffs require a space or minus even for empty lines
            grammar_parts.append(f"CONTENT{i} ::= ( ' ' | '-' ) NEWLINE;")
        else:
            # Escape backslashes and quotes for the EBNF string literal
            escaped_content = content.replace('\\', '\\\\').replace('"', '\\"')
            grammar_parts.append(f'CONTENT{i} ::= ( " " | "-" ) "{escaped_content}" NEWLINE;')

    return '\n'.join(grammar_parts)


def main():
    # Get source file from environment or argument
    source_file = os.environ.get("SOURCE_FILE")
    
    if not source_file:
        if len(sys.argv) > 1:
            source_file = sys.argv[1]
        else:
            # Default test file
            source_file = "test.txt"
    
    if not os.path.exists(source_file):
        print(f"Error: Source file not found: {source_file}")
        print("Usage: SOURCE_FILE=path/to/file.txt python scripts/test_diff_grammar.py")
        sys.exit(1)
    
    print(f"Testing diff grammar for: {source_file}")
    print("=" * 60)
    
    # Step 1: Generate diff grammar
    print("\n1. Generating diff grammar...")
    start = time.time()
    ebnf = generate_diff_grammar(source_file)
    gen_time = time.time() - start
    print(f"   Grammar generation: {gen_time*1000:.1f}ms ({len(ebnf)} chars)")
    
    # Show first few lines of grammar
    lines = ebnf.split('\n')
    print(f"   Grammar has {len(lines)} lines")
    print("   First 10 lines:")
    for line in lines[:10]:
        print(f"      {line}")
    if len(lines) > 10:
        print("      ...")
    
    # Step 2: Load vocabulary
    print("\n2. Loading vocabulary...")
    try:
        import tiktoken
        print("   Using tiktoken (GPT-2 encoding)...")
        start = time.time()
        enc = tiktoken.get_encoding("gpt2")
        
        token_to_id = {}
        for token_id in range(enc.n_vocab):
            token_bytes = enc.decode_single_token_bytes(token_id)
            token_to_id[token_bytes] = token_id
        vocab_time = time.time() - start
        print(f"   Token map generation: {vocab_time*1000:.1f}ms ({len(token_to_id)} tokens)")
    
    except ImportError:
        print("   tiktoken not found, falling back to vocab.json download...")
        import urllib.request
        vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"
        start = time.time()
        with urllib.request.urlopen(vocab_url) as resp:
            vocab = json.loads(resp.read().decode())
        token_to_id = {k.encode('utf-8'): v for k, v in vocab.items()}
        vocab_time = time.time() - start
        print(f"   Vocab download/parse: {vocab_time*1000:.1f}ms")
    
    # Step 3: Parse EBNF to GrammarDefinition
    print("\n3. Parsing EBNF to GrammarDefinition...")
    start = time.time()
    grammar_def = _sep1.grammar_definition_from_ebnf(ebnf)
    parse_time = time.time() - start
    print(f"   Grammar parsing: {parse_time*1000:.1f}ms")
    
    # Step 4: Optimize grammar
    print("\n4. Optimizing grammar...")
    start = time.time()
    grammar_def.optimize()
    optimize_time = time.time() - start
    print(f"   Optimization: {optimize_time*1000:.1f}ms")
    
    # Step 5: Compile to GLR parser
    print("\n5. Compiling grammar...")
    start = time.time()
    compiled = grammar_def.compile()
    compile_time = time.time() - start
    print(f"   Grammar compilation: {compile_time*1000:.1f}ms")
    
    # Step 6: Create constraint with vocabulary
    print("\n6. Creating constraint with vocabulary...")
    start = time.time()
    constraint = _sep1.GrammarConstraint(compiled, token_to_id)
    constraint_time = time.time() - start
    print(f"   Constraint creation: {constraint_time*1000:.1f}ms")
    
    # Step 7: Test the constraint
    print("\n7. Testing constraint...")
    state = _sep1.GrammarConstraintState(constraint)
    print(f"   Initial state active: {state.is_active()}")
    
    # Try a minimal valid diff (empty diff with just EOF)
    test_input = "<|EOF|>"
    print(f"   Testing minimal input: {repr(test_input)}")
    
    # Get initial mask
    mask = state.get_mask()
    allowed_count = sum(mask)
    print(f"   Initial mask has {allowed_count} allowed tokens")
    
    # Summary
    total_compile_time = gen_time + parse_time + optimize_time + compile_time + constraint_time
    total_time = total_compile_time + vocab_time
    print(f"\n{'=' * 60}")
    print(f"=== Grammar Compile Time (GCT): {total_compile_time*1000:.1f}ms ===")
    print(f"=== Total time (incl. vocab): {total_time*1000:.1f}ms ===")
    print("SUCCESS: Diff grammar constraint created!")


if __name__ == "__main__":
    main()
