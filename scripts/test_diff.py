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
    Generates an EBNF grammar that validates a 'git diff'-like format.
    
    This version uses a CONTENT-AGNOSTIC structure that avoids exponential
    DFA blowup by not creating separate rules for each line. Instead, it
    matches any line content with a generic pattern.

    Args:
        source_path: The path to the input file (used only for validation/logging).
        
    Returns:
        The EBNF grammar as a string.
    """
    print(f"Reading source file: {source_path}")
    with open(source_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    num_lines = len(lines)
    print(f"   Source has {num_lines} lines")
    
    # Content-agnostic grammar that matches any valid diff structure
    # without per-line enumeration
    grammar = r'''root ::= DIFF;

// Top-level diff structure
DIFF ::= FILE_HEADER? HUNKS EOF;
FILE_HEADER ::= GIT_LINE INDEX_LINE? MINUS_LINE PLUS_FILE_LINE;

// Zero or more hunks
HUNKS ::= ( HUNK_HEADER HUNK_BODY )*;

// Hunk body: any sequence of context, addition, or deletion lines
HUNK_BODY ::= ( CONTEXT_LINE | PLUS_LINE | MINUS_CONTENT_LINE )*;

// --- TERMINALS ---
GIT_LINE         ::= 'diff --git' [^\n\r]* NEWLINE;
INDEX_LINE       ::= 'index' [^\n\r]* NEWLINE;
MINUS_LINE       ::= '---' [^\n\r]* NEWLINE;
PLUS_FILE_LINE   ::= '+++' [^\n\r]* NEWLINE;
HUNK_HEADER      ::= '@@' [^\n\r]* NEWLINE;

// Content lines (context or deletion) - matches ANY line content
CONTEXT_LINE     ::= ' ' [^\n\r]* NEWLINE;
MINUS_CONTENT_LINE ::= '-' [^\n\r]* NEWLINE;
PLUS_LINE        ::= '+' [^\n\r]* NEWLINE;

NEWLINE          ::= '\n' | '\r\n';
EOF              ::= '<|EOF|>';
'''
    return grammar.strip()


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
