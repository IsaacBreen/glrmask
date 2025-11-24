import os
import re
import argparse
import time

def generate_diff_grammar(source_path: str, grammar_path: str):
    """
    Generates an EBNF grammar that validates a 'git diff'-like format
    for a specific source file.

    Args:
        source_path: The path to the input file to base the grammar on.
        grammar_path: The path where the generated .ebnf grammar will be saved.
    """
    print(f"Reading source file: {source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"Error reading source file: {e}")
        return

    num_lines = len(lines)
    grammar_parts = []

    # --- 1. Preamble and Top-Level Rules ---
    grammar_parts.append("#![ignore(IGNORE)]")
    grammar_parts.append("")
    grammar_parts.append("diff ::= FILE_HEADER? ( HUNK_HEADER s0 )? EOF;")
    grammar_parts.append("FILE_HEADER ::= GIT_LINE INDEX_LINE? MINUS_LINE PLUS_FILE_LINE;")
    grammar_parts.append("EOF  ::= '<|EOF|>';")
    grammar_parts.append("")

    # --- 2. 's' Rules (Start of a context block) ---
    grammar_parts.append("// 's' rules: Allow starting a context block at a given line or skipping.")
    for i in range(num_lines):
        grammar_parts.append(f"s{i} ::= l{i} | s{i+1};")

    grammar_parts.append(f"s{num_lines} ::= PLUS_LINE*;")
    grammar_parts.append("")

    # --- 3. 'l' Rules (Continuation of a context block) ---
    grammar_parts.append("// 'l' rules: Match a specific context line and continue or start a new hunk.")
    for i in range(num_lines):
        if i < num_lines - 1:
            continuation = f"( l{i+1} | PLUS_LINE* HUNK_HEADER s{i+1} )?"
        else:
            continuation = f"( PLUS_LINE* HUNK_HEADER s{num_lines} )?"

        grammar_parts.append(f"l{i} ::= PLUS_LINE* L{i} {continuation};")
    grammar_parts.append("")

    # --- 4. Terminal Definitions ---
    grammar_parts.append("// --- TERMINALS ---")

    grammar_parts.append(r"GIT_LINE         ::= 'diff --git' [^\n\r]* NEWLINE;")
    grammar_parts.append(r"INDEX_LINE       ::= 'index' [^\n\r]* NEWLINE;")
    grammar_parts.append(r"MINUS_LINE       ::= '---' [^\n\r]* NEWLINE;")
    grammar_parts.append(r"PLUS_FILE_LINE   ::= '+++' [^\n\r]* NEWLINE;")
    grammar_parts.append("")

    # Use raw strings (r"...") to prevent Python from interpreting \n and \r.
    grammar_parts.append(r"HUNK_HEADER ::= '@@' [^\n\r]* NEWLINE;")
    grammar_parts.append(r"PLUS_LINE   ::= '+' [^\n\r]* NEWLINE;")

    # This line is intentionally NOT a raw string. We WANT Python to interpret '\\'
    # as a single '\', so the EBNF parser sees '\r' and '\n' and matches the
    # actual control characters.
    grammar_parts.append("NEWLINE     ::= ( '\\r'? '\\n' );")
    grammar_parts.append("")

    # Use a raw string here as well for the same reason.
    grammar_parts.append(r"IGNORE ::= ( [ \t]+ | '//'[^\n\r]* | '/*'( [^*] | '*'[^/] )*'*/' )+ ;")
    grammar_parts.append("")

    # Context-line terminals (one for each line in the source file)
    grammar_parts.append("// Context-line terminals (one for each line in the source file)")
    for i, line in enumerate(lines):
        content = line.rstrip('\r\n')

        if not content:
            grammar_parts.append(f"L{i} ::= ( ' ' | '-' )? NEWLINE;")
        else:
            # This logic for escaping the file's content remains correct.
            # It handles backslashes and quotes that might be in the source file.
            escaped_content = content.replace('\\', '\\\\')
            escaped_content = escaped_content.replace("'", "\\'")

            grammar_parts.append(f"L{i} ::= ( ' ' | '-' ) '{escaped_content}' NEWLINE;")

    # --- 5. Write the grammar to the output file ---
    try:
        with open(grammar_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(grammar_parts))
        print(f"Successfully generated grammar at: {grammar_path}")
    except IOError as e:
        print(f"Error writing grammar file: {e}")


def benchmark_regex(source_path: str):
    """
    Constructs an equivalent regex for the grammar and benchmarks the build time.
    This demonstrates the exponential complexity of expanding the state machine into a single regex.
    """
    print(f"\n--- Regex Benchmark ---")
    print(f"Reading source file for regex: {source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"Error reading source file: {e}")
        return

    num_lines = len(lines)
    if num_lines > 15:
        print(f"Warning: File has {num_lines} lines. The equivalent regex size is O(2^N).")
        print("Construction might hang, consume massive memory, or fail.")

    print("Constructing regex string (iterative bottom-up)...")
    start_build = time.time()

    # Regex patterns for terminals
    P_NEWLINE = r'(?:\r?\n)'
    P_PLUS_LINE = r'(?:\+[^\n\r]*' + P_NEWLINE + r')'
    P_PLUS_STAR = P_PLUS_LINE + r'*'
    P_HUNK_HEADER = r'(?:@@[^\n\r]*' + P_NEWLINE + r')'

    # Base case: s{num_lines} ::= PLUS_LINE*;
    s_next = P_PLUS_STAR
    l_next = None

    for i in range(num_lines - 1, -1, -1):
        line_content = lines[i].rstrip('\r\n')
        
        if not line_content:
            T_i = r'(?:[ -]?' + P_NEWLINE + r')'
        else:
            T_i = r'(?:[ -]' + re.escape(line_content) + P_NEWLINE + r')'

        if i == num_lines - 1:
            continuation = r'(?:' + P_PLUS_STAR + P_HUNK_HEADER + s_next + r')?'
        else:
            continuation = r'(?:' + l_next + r'|' + P_PLUS_STAR + P_HUNK_HEADER + s_next + r')?'

        l_curr = P_PLUS_STAR + T_i + continuation
        s_curr = r'(?:' + l_curr + r'|' + s_next + r')'

        l_next = l_curr
        s_next = s_curr

    P_GIT_LINE = r'(?:diff --git [^\n\r]*' + P_NEWLINE + r')'
    P_INDEX_LINE = r'(?:index [^\n\r]*' + P_NEWLINE + r')'
    P_MINUS_LINE = r'(?:--- [^\n\r]*' + P_NEWLINE + r')'
    P_PLUS_FILE_LINE = r'(?:\+\+\+ [^\n\r]*' + P_NEWLINE + r')'
    
    P_FILE_HEADER = r'(?:' + P_GIT_LINE + r'(?:' + P_INDEX_LINE + r')?' + P_MINUS_LINE + P_PLUS_FILE_LINE + r')'
    P_EOF = r'(?:<\|EOF\|>)'

    full_regex = r'\A' + P_FILE_HEADER + r'?' + r'(?:' + P_HUNK_HEADER + s_next + r')?' + P_EOF + r'\Z'

    build_time = time.time() - start_build
    print(f"Regex string constructed. Length: {len(full_regex):,} chars.")
    print(f"String build time: {build_time:.4f} seconds.")

    print("Compiling regex (re.compile)...")
    start_compile = time.time()
    try:
        _ = re.compile(full_regex)
        compile_time = time.time() - start_compile
        print(f"Regex compiled successfully in {compile_time:.4f} seconds.")
    except Exception as e:
        print(f"Regex compilation failed: {e}")

def main():
    """Command-line interface to generate the diff grammar."""
    parser = argparse.ArgumentParser(
        description="Generates an EBNF grammar that validates a 'git diff'-like format for a specific source file."
    )
    parser.add_argument(
        "source_path",
        help="The path to the input file to base the grammar on."
    )
    parser.add_argument(
        "-o", "--output",
        dest="grammar_path",
        help="The path where the generated .ebnf grammar will be saved. "
             "If not provided, it defaults to the source file's path with an '.ebnf' extension."
    )
    args = parser.parse_args()

    source_path = args.source_path
    grammar_path = args.grammar_path

    if not grammar_path:
        grammar_path = os.path.splitext(source_path)[0] + ".ebnf"

    generate_diff_grammar(source_path, grammar_path)

    benchmark_regex(source_path)

if __name__ == "__main__":
    main()
