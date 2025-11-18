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
    grammar_parts.append("diff ::= ( HUNK_HEADER s0 )? '<|EOF|>';")
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
