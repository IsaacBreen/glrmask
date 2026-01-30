# Reference Prefix Checker

Slow, simple, correctness‑oriented prefix validator for CFGs.

## Purpose
Provide an independent, rock‑solid implementation for checking whether a byte string is a valid **prefix** of a grammar, without using the DWA/constraint pipeline.

## Input format
Uses `GrammarDefinition` JSON from sep1:
- `productions`: list of `{lhs, rhs}`
- `regex_terminals`: list of `{name, group_id, expr}` (Expr JSON from `dfa_u8::Expr`)
- `literal_terminals`: list of `{value, group_id}`

## Usage (CLI)
```
python prefix_checker.py --grammar-json-file path/to/grammar.json --text "{"
```

## Usage (Python)
```python
from reference_prefix_checker.prefix_checker import ReferencePrefixChecker

checker = ReferencePrefixChecker.from_grammar_definition_json(grammar_json)
checker.is_valid_prefix("{")
```

## Building from EBNF or JSON Schema
The script provides helpers:
- `build_from_ebnf_string(ebnf)`
- `build_from_json_schema(schema_json)`

These require the `_sep1` Python bindings.

## Notes
- Tokenization uses **per‑terminal longest match** (matches the tokenizer’s per‑group max length).
- Uses the `regex` Python module for partial matching.
- Honors `ignore_terminal_ids` by consuming those tokens without advancing the parse state.
- Designed to be correct over performance.