#!/usr/bin/env python3
"""Check JSON-schema grammar dumps for lexer-only string bodies in nonterminals.

Usage:
  <grammar dump command> | scripts/check_json_schema_terminal_leaks.py -
  scripts/check_json_schema_terminal_leaks.py /tmp/show_grammar.log

This is a text-level guard for generated grammar dumps. Rust tests should enforce
the same invariant on the AST where possible; this script is for quick inspection
of benchmark grammars and CI logs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


LEAK_MARKERS = (
    "JSON_STRING_CHAR",
    " & /",
)


def read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text()


def check_dump(name: str, text: str) -> list[str]:
    errors: list[str] = []
    in_fa = False
    current_fa = ""
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        is_nonterminal = stripped.startswith("nt ")
        if stripped.startswith("fa "):
            in_fa = True
            current_fa = stripped.split("::=", 1)[0].strip()
            is_nonterminal = True
        elif in_fa:
            is_nonterminal = True
            if stripped == "};":
                in_fa = False

        if not is_nonterminal:
            continue

        for marker in LEAK_MARKERS:
            if marker in line:
                prefix = current_fa if in_fa and not stripped.startswith("fa ") else ""
                where = f"{name}:{lineno}"
                if prefix:
                    where = f"{where} ({prefix})"
                errors.append(f"{where}: terminal-only marker {marker!r} in nonterminal: {line}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="grammar dump path(s), or '-' for stdin")
    args = parser.parse_args()

    all_errors: list[str] = []
    for path in args.paths:
        all_errors.extend(check_dump(path, read_input(path)))

    if all_errors:
        for error in all_errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
