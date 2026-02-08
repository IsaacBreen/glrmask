"""
Test Sep1 diff grammar compilation and constraint creation.

This script takes a text file and:
1. Generates an EBNF grammar that describes valid git diffs for that file
2. Compiles the grammar and creates a constraint
3. Caches the compiled constraint for faster subsequent runs

Usage:
    # With environment variable
    SOURCE_FILE="path/to/file.txt" python scripts/test_diff.py
    
    # Or directly via make
    make test-diff FILE=path/to/file.txt
    
    # Disable caching (always recompile)
    NO_CACHE=1 SOURCE_FILE="path/to/file.txt" python scripts/test_diff.py
"""

import json
import time
import os
import sys
import hashlib
import gzip

# Add project root and python/ dir to sys.path to find _sep1 module
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "python"))

import _sep1


# Cache directory for compiled constraints
CACHE_DIR = os.path.join(repo_root, ".cache", "diff_constraints")


def get_cache_key(ebnf: str) -> str:
    """Compute cache key from EBNF grammar."""
    return hashlib.sha256(ebnf.encode('utf-8')).hexdigest()[:16]


def get_cache_path(cache_key: str) -> str:
    """Get cache file path for a given key."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{cache_key}.json.gz")


def load_cached_constraint(cache_key: str, token_to_id):
    """Load constraint from cache if available."""
    cache_path = get_cache_path(cache_key)
    if not os.path.exists(cache_path):
        return None
    
    try:
        with gzip.open(cache_path, 'rt', encoding='utf-8') as f:
            constraint_json = json.load(f)
        
        # Create GrammarConstraint from JSON
        constraint = _sep1.GrammarConstraint.from_json(constraint_json, token_to_id)
        return constraint
    except Exception as e:
        print(f"   Warning: Failed to load cached constraint: {e}")
        return None


def save_constraint_to_cache(constraint, cache_key: str):
    """Save constraint to cache."""
    cache_path = get_cache_path(cache_key)
    try:
        constraint_json = constraint.to_json()
        with gzip.open(cache_path, 'wt', encoding='utf-8') as f:
            json.dump(constraint_json, f)
    except Exception as e:
        print(f"   Warning: Failed to save constraint to cache: {e}")


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
    use_balanced_tree = os.environ.get("DIFF_BALANCED_TREE") == "1"
    balanced_no_hunk = os.environ.get("DIFF_BALANCED_NO_HUNK") == "1"
    use_one_big_terminal = os.environ.get("DIFF_ONE_BIG_TERMINAL") == "1"
    seg_root = f"seg0_{num_lines - 1}" if num_lines > 0 else "seg_empty"

    # --- 1. Preamble and Top-Level Rules ---
    grammar_parts.append("root ::= diff;")
    if use_one_big_terminal:
        if balanced_no_hunk:
            grammar_parts.append("diff ::= file_header? diff_body? EOF;")
        else:
            grammar_parts.append("diff ::= file_header? ( HUNK_HEADER diff_body )? EOF;")
        grammar_parts.append("diff_body ::= ( PLUS_LINE | VALID_LINE )*;")
    elif use_balanced_tree:
        if balanced_no_hunk:
            grammar_parts.append(f"diff ::= file_header? {seg_root}? EOF;")
        else:
            grammar_parts.append(f"diff ::= file_header? ( HUNK_HEADER {seg_root} )? EOF;")
    else:
        grammar_parts.append("diff ::= file_header? ( HUNK_HEADER s0 )? EOF;")
    grammar_parts.append("file_header ::= GIT_LINE INDEX_LINE? MINUS_LINE PLUS_FILE_LINE;")
    grammar_parts.append("EOF  ::= '<|EOF|>';")  # Ensure this matches your tokenizer's EOF
    grammar_parts.append("")

    # --- 2. 's' Rules (Search for Hunk Start) ---
    if use_one_big_terminal:
        grammar_parts.append("// 's' rules skipped: one-big-terminal mode")
        grammar_parts.append("")
    elif use_balanced_tree:
        grammar_parts.append("// 'seg' rules: Balanced tree for line subsequences")
        seen_segments = set()

        def emit_segment(start: int, end: int) -> str:
            name = f"seg{start}_{end}"
            if (start, end) in seen_segments:
                return name
            seen_segments.add((start, end))

            if start > end:
                grammar_parts.append(f"{name} ::= PLUS_LINE*;")
                return name
            if start == end:
                grammar_parts.append(f"{name} ::= line{start}?;")
                return name

            mid = (start + end) // 2
            left = emit_segment(start, mid)
            right = emit_segment(mid + 1, end)
            grammar_parts.append(f"{name} ::= {left} {right}?;")
            return name

        emit_segment(0, num_lines - 1)
        grammar_parts.append("")
    else:
        grammar_parts.append("// 's' rules: Find the start of a hunk")
        for i in range(num_lines):
            # Try to match line i, or skip and try line i+1
            grammar_parts.append(f"s{i} ::= line{i} | s{i+1};")

        # If we reach the end of the file, we only allow trailing additions
        grammar_parts.append(f"s{num_lines} ::= PLUS_LINE*;")
        grammar_parts.append("")

    # --- 3. 'line' Rules (Match Context/Deletion) ---
    grammar_parts.append("// 'line' rules: Match content exactly, then continue or new hunk")
    for i in range(num_lines):
        if use_one_big_terminal:
            continue
        if use_balanced_tree:
            # Balanced tree mode: keep line rule minimal for sequencing via segments.
            grammar_parts.append(f"line{i} ::= PLUS_LINE* content{i} PLUS_LINE*;")
            continue

        # After matching line i, we can:
        # 1. Continue immediately to line i+1
        # 2. Have some additions, then a Hunk Header, skipping to i+1
        if i < num_lines - 1:
            continuation = f"( line{i+1} | PLUS_LINE* HUNK_HEADER s{i+1} )?"
        else:
            continuation = f"( PLUS_LINE* HUNK_HEADER s{num_lines} )?"

        # NOTE: PLUS_LINE* allows insertions *before* the context/deletion line
        grammar_parts.append(f"line{i} ::= PLUS_LINE* content{i} {continuation};")
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
    if use_one_big_terminal:
        non_empty_lines = []
        has_empty = False
        for line in lines:
            content = line.rstrip('\r\n')
            if not content:
                has_empty = True
                continue
            escaped = content.replace("\\", "\\\\").replace('"', "\\\"")
            non_empty_lines.append(f'"{escaped}"')

        if non_empty_lines:
            body = " | ".join(non_empty_lines)
            if has_empty:
                grammar_parts.append(
                    f"VALID_LINE ::= ( \" \" | \"-\" ) ( NEWLINE | ( {body} ) NEWLINE );"
                )
            else:
                grammar_parts.append(
                    f"VALID_LINE ::= ( \" \" | \"-\" ) ( {body} ) NEWLINE;"
                )
        else:
            grammar_parts.append("VALID_LINE ::= ( \" \" | \"-\" ) NEWLINE;")
    else:
        # Use lowercase rule names so line content is parsed, not treated as a terminal regex.
        for i, line in enumerate(lines):
            content = line.rstrip('\r\n')

            if not content:
                # Strict diffs require a space or minus even for empty lines
                grammar_parts.append(f"content{i} ::= ( ' ' | '-' ) NEWLINE;")
            else:
                # Emit per-character literals to keep the terminal set small.
                escaped_chars = []
                for ch in content:
                    if ch == "\\":
                        escaped_chars.append("\\\\")
                    elif ch == '"':
                        escaped_chars.append("\\\"")
                    else:
                        escaped_chars.append(ch)
                char_terms = " ".join(f'"{ch}"' for ch in escaped_chars)
                grammar_parts.append(
                    f"content{i} ::= ( \" \" | \"-\" ) {char_terms} NEWLINE;"
                )

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
    
    # Optional: Write EBNF to file
    out_file = os.environ.get("OUT_FILE")
    if out_file:
        try:
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(ebnf)
            print(f"   Wrote EBNF to: {out_file}")
        except Exception as e:
            print(f"   Error writing to file: {e}")

    # Optional: Print EBNF to stdout
    if os.environ.get("PRINT_GRAMMAR"):
        print("\n=== Generated Grammar ===")
        print(ebnf)
        print("=========================\n")

    if os.environ.get("ONLY_GRAMMAR"):
        print("   Exiting after grammar generation (ONLY_GRAMMAR=1)")
        sys.exit(0)
    
    # Show first few lines of grammar (only if not printing full grammar)
    if not os.environ.get("PRINT_GRAMMAR"):
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
    
    # Step 3: Check cache (unless NO_CACHE=1)
    use_cache = not os.environ.get("NO_CACHE")
    constraint = None
    cache_key = None
    parse_time = optimize_time = compile_time = constraint_time = 0.0
    
    if use_cache:
        print("\n3. Checking cache...")
        start = time.time()
        cache_key = get_cache_key(ebnf)
        cache_path = get_cache_path(cache_key)
        print(f"   Cache key: {cache_key}")
        print(f"   Cache path: {cache_path}")
        
        constraint = load_cached_constraint(cache_key, token_to_id)
        cache_time = time.time() - start
        
        if constraint is not None:
            print(f"   ✓ Cache hit! Loaded in {cache_time*1000:.1f}ms")
        else:
            print(f"   ✗ Cache miss (checked in {cache_time*1000:.1f}ms)")
    else:
        print("\n3. Cache disabled (NO_CACHE=1)")
        cache_key = get_cache_key(ebnf)
    
    # Step 4-7: Compile if not cached
    if constraint is None:
        # Step 4: Parse EBNF to GrammarDefinition
        print("\n4. Parsing EBNF to GrammarDefinition...")
        start = time.time()
        grammar_def = _sep1.grammar_definition_from_ebnf(ebnf)
        parse_time = time.time() - start
        print(f"   Grammar parsing: {parse_time*1000:.1f}ms")
        
        # Step 5: Optimize grammar (skip for diff grammars — collapses to a
        # single massive regex terminal that explodes NFA/DFA state count)
        print("\n5. Optimizing grammar... (skipped)")
        optimize_time = 0.0
        
        # Step 6: Compile to GLR parser
        print("\n6. Compiling grammar...")
        start = time.time()
        compiled = grammar_def.compile()
        compile_time = time.time() - start
        print(f"   Grammar compilation: {compile_time*1000:.1f}ms")
        
        # Step 7: Create constraint with vocabulary
        print("\n7. Creating constraint with vocabulary...")
        start = time.time()
        constraint = _sep1.GrammarConstraint(compiled, token_to_id)
        constraint_time = time.time() - start
        print(f"   Constraint creation: {constraint_time*1000:.1f}ms")
        
        # Step 8: Save to cache
        if use_cache and cache_key:
            print("\n8. Saving to cache...")
            start = time.time()
            save_constraint_to_cache(constraint, cache_key)
            save_time = time.time() - start
            print(f"   Cache saved in {save_time*1000:.1f}ms")
    
    # Final step: Test the constraint
    step_num = 9 if constraint_time > 0 else 4
    print(f"\n{step_num}. Testing constraint...")
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
    if constraint_time == 0:
        print("=== (Loaded from cache) ===")
    print("SUCCESS: Diff grammar constraint created!")


if __name__ == "__main__":
    main()
