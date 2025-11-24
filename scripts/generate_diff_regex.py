import os
import re
import time
import argparse
import sys

def get_escaped_line_pattern(content):
    """Returns the regex pattern for a specific content line (Context/Delete)."""
    content = content.rstrip('\r\n')
    if not content:
        # Matches an empty context line (space or minus followed by newline)
        return r'(?:[ -](?:\r?\n))'
    else:
        # Matches space or minus, then exact content, then newline
        return r'(?:[ -]' + re.escape(content) + r'(?:\r?\n))'

def build_regex_for_lines(lines):
    """
    Constructs the full regex string for a subset of lines using
    the iterative bottom-up approach (O(2^N) string length).
    """
    num_lines = len(lines)

    # --- Common Terminals ---
    P_NEWLINE = r'(?:\r?\n)'
    # + line: starts with +, not followed by newline immediately (content), then newline
    P_PLUS_LINE = r'(?:\+[^\n\r]*' + P_NEWLINE + r')'
    P_PLUS_STAR = P_PLUS_LINE + r'*'
    P_HUNK_HEADER = r'(?:@@[^\n\r]*' + P_NEWLINE + r')'

    # --- Bottom-Up Construction ---

    # Base case: s{num_lines} (The end state after the last line)
    # Allows trailing additions.
    s_next = P_PLUS_STAR
    l_next = None # Represents l{i+1}

    # Iterate backwards from the last line down to 0
    for i in range(num_lines - 1, -1, -1):
        T_i = get_escaped_line_pattern(lines[i])

        # Logic:
        # l{i} matches: PLUS* -> T_i -> Continuation
        # Continuation is: ( l{i+1} | PLUS* HUNK s{i+1} )?

        if i == num_lines - 1:
            # End of file logic: continuation only allows new hunk pointing to End State
            continuation = r'(?:' + P_PLUS_STAR + P_HUNK_HEADER + s_next + r')?'
        else:
            # Normal logic: continuation allows going to next line OR jumping to new hunk
            continuation = r'(?:' + l_next + r'|' + P_PLUS_STAR + P_HUNK_HEADER + s_next + r')?'

        l_curr = P_PLUS_STAR + T_i + continuation

        # s{i} matches: l{i} | s{i+1} (Skip logic)
        s_curr = r'(?:' + l_curr + r'|' + s_next + r')'

        # Move pointers up
        l_next = l_curr
        s_next = s_curr

    # --- File Headers ---
    P_GIT_LINE = r'(?:diff --git [^\n\r]*' + P_NEWLINE + r')'
    P_INDEX_LINE = r'(?:index [^\n\r]*' + P_NEWLINE + r')'
    P_MINUS_LINE = r'(?:--- [^\n\r]*' + P_NEWLINE + r')'
    P_PLUS_FILE_LINE = r'(?:\+\+\+ [^\n\r]*' + P_NEWLINE + r')'

    P_FILE_HEADER = r'(?:' + P_GIT_LINE + r'(?:' + P_INDEX_LINE + r')?' + P_MINUS_LINE + P_PLUS_FILE_LINE + r')'
    P_EOF = r'(?:<\|EOF\|>)'

    # Final Root Pattern
    full_regex = r'\A' + P_FILE_HEADER + r'?' + r'(?:' + P_HUNK_HEADER + s_next + r')?' + P_EOF + r'\Z'

    return full_regex

def main():
    parser = argparse.ArgumentParser(
        description="Incrementally builds and compiles a Git Diff regex to demonstrate exponential complexity."
    )
    parser.add_argument("source_path", help="The path to the input file.")
    args = parser.parse_args()

    if not os.path.exists(args.source_path):
        print(f"Error: File not found: {args.source_path}")
        sys.exit(1)

    try:
        with open(args.source_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
    except IOError as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    print(f"Input file loaded: {len(all_lines)} lines.")
    print("Starting incremental regex construction...")
    print(f"{'Line Count':<12} | {'Regex Length (chars)':<20} | {'Time (sec)':<12} | {'Status'}")
    print("-" * 65)

    # Loop from 1 line up to total lines
    for i in range(1, len(all_lines) + 1):
        subset_lines = all_lines[:i]

        step_start = time.time()

        try:
            # 1. Build String
            regex_str = build_regex_for_lines(subset_lines)
            str_len = len(regex_str)

            # 2. Compile Regex
            _ = re.compile(regex_str)

            step_end = time.time()
            duration = step_end - step_start

            print(f"{i:<12} | {str_len:<20,} | {duration:.4f}       | OK")

            # PANIC CONDITION
            if duration > 1.0:
                print("\n!!! PANIC !!!")
                print(f"Process took {duration:.4f}s which exceeds the 1.0s limit.")
                print(f"Exponential explosion detected at line {i}.")
                print(f"Regex length was {str_len:,} characters.")
                sys.exit(1)

        except re.error as e:
            print(f"\n!!! REGEX ERROR !!!")
            print(f"Compilation failed at line {i} with regex length {str_len:,}.")
            print(f"Reason: {e}")
            print("The pattern likely exceeded Python's maximum recursion depth or pattern size.")
            sys.exit(1)
        except MemoryError:
            print(f"\n!!! MEMORY ERROR !!!")
            print(f"OOM (Out of Memory) at line {i}.")
            sys.exit(1)
        except OverflowError:
             print(f"\n!!! OVERFLOW ERROR !!!")
             print(f"String/Pattern too large at line {i}.")
             sys.exit(1)

    print("\nSuccess! Reached end of file without exceeding time limits (unlikely for large files).")

if __name__ == "__main__":
    main()