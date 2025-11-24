import os
import re
import argparse

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
    # REMOVED: #![ignore(IGNORE)] - Diffs are whitespace sensitive!

    grammar_parts.append("root ::= diff;")
    grammar_parts.append("diff ::= FILE_HEADER? ( HUNK_HEADER s0 )? EOF;")
    grammar_parts.append("FILE_HEADER ::= GIT_LINE INDEX_LINE? MINUS_LINE PLUS_FILE_LINE;")
    grammar_parts.append("EOF  ::= '<|EOF|>';") # Ensure this matches your tokenizer's EOF
    grammar_parts.append("")

    # --- 2. 's' Rules (Search for Hunk Start) ---
    grammar_parts.append("// 's' rules: Find the start of a hunk")
    for i in range(num_lines):
        # Try to match line i, or skip and try line i+1
        grammar_parts.append(f"s{i} ::= l{i} | s{i+1};")

    # If we reach the end of the file, we only allow trailing additions
    grammar_parts.append(f"s{num_lines} ::= PLUS_LINE*;")
    grammar_parts.append("")

    # --- 3. 'l' Rules (Match Context/Deletion) ---
    grammar_parts.append("// 'l' rules: Match content exactly, then continue or new hunk")
    for i in range(num_lines):
        # After matching line i, we can:
        # 1. Continue immediately to line i+1
        # 2. Have some additions, then a Hunk Header, skipping to i+1
        if i < num_lines - 1:
            continuation = f"( l{i+1} | PLUS_LINE* HUNK_HEADER s{i+1} )?"
        else:
            continuation = f"( PLUS_LINE* HUNK_HEADER s{num_lines} )?"

        # NOTE: PLUS_LINE* allows insertions *before* the context/deletion line
        grammar_parts.append(f"l{i} ::= PLUS_LINE* L{i} {continuation};")
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
            grammar_parts.append(f"L{i} ::= ( ' ' | '-' ) NEWLINE;")
        else:
            # Escape backslashes and quotes for the EBNF string literal
            escaped_content = content.replace('\\', '\\\\').replace('"', '\\"')
            grammar_parts.append(f'L{i} ::= ( " " | "-" ) "{escaped_content}" NEWLINE;')

    try:
        with open(grammar_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(grammar_parts))
        print(f"Successfully generated grammar at: {grammar_path}")
    except IOError as e:
        print(f"Error writing grammar file: {e}")


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


if __name__ == "__main__":
    main()
