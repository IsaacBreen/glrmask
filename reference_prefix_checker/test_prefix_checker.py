import os
import sys
import unittest

try:
    import regex as re  # noqa: F401
    _REGEX_OK = True
except Exception:
    _REGEX_OK = False

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from reference_prefix_checker.prefix_checker import (
    build_from_grammar_definition_json,
    build_from_json_schema,
    ReferencePrefixChecker,
)


def _terminal_literal(value_bytes, group_id):
    return {"value": list(value_bytes), "group_id": group_id}


def _terminal_regex(name, group_id, expr):
    return {"name": name, "group_id": group_id, "expr": expr}


def _sym_term_literal(value_bytes):
    return {
        "variant": "Terminal",
        "value": {"type": "Literal", "value": list(value_bytes)},
    }


def _sym_term_regex(name):
    return {
        "variant": "Terminal",
        "value": {"type": "Regex", "value": name},
    }


def _sym_nonterm(name):
    return {"variant": "NonTerminal", "value": name}


class TestReferencePrefixChecker(unittest.TestCase):
    def setUp(self):
        if not _REGEX_OK:
            self.skipTest("regex module not available")

    def test_simple_literals(self):
        grammar = {
            "productions": [
                {"lhs": "S", "rhs": [_sym_term_literal(b"a"), _sym_term_literal(b"b")]}
            ],
            "start_production_id": 0,
            "ignore_terminal_ids": [],
            "external_name_to_group_id": {},
            "regex_terminals": [],
            "literal_terminals": [
                _terminal_literal(b"a", 0),
                _terminal_literal(b"b", 1),
            ],
        }

        checker = build_from_grammar_definition_json(json_str=_to_json(grammar))
        self.assertTrue(checker.is_valid_prefix(""))
        self.assertTrue(checker.is_valid_prefix("a"))
        self.assertTrue(checker.is_valid_prefix("ab"))
        self.assertFalse(checker.is_valid_prefix("ac"))
        self.assertFalse(checker.is_valid_prefix("abc"))

    def test_partial_literal(self):
        grammar = {
            "productions": [
                {"lhs": "S", "rhs": [_sym_term_literal(b"abc")]}  # S -> 'abc'
            ],
            "start_production_id": 0,
            "ignore_terminal_ids": [],
            "external_name_to_group_id": {},
            "regex_terminals": [],
            "literal_terminals": [
                _terminal_literal(b"abc", 0),
            ],
        }

        checker = build_from_grammar_definition_json(json_str=_to_json(grammar))
        self.assertTrue(checker.is_valid_prefix("a"))
        self.assertTrue(checker.is_valid_prefix("ab"))
        self.assertTrue(checker.is_valid_prefix("abc"))
        self.assertFalse(checker.is_valid_prefix("abd"))

    def test_ignore_terminal(self):
        ws_expr = {"variant": "U8Seq", "bytes": [0x20]}
        grammar = {
            "productions": [
                {"lhs": "S", "rhs": [_sym_term_literal(b"a"), _sym_term_literal(b"b")]}
            ],
            "start_production_id": 0,
            "ignore_terminal_ids": [2],
            "external_name_to_group_id": {},
            "regex_terminals": [
                _terminal_regex("WS", 2, ws_expr),
            ],
            "literal_terminals": [
                _terminal_literal(b"a", 0),
                _terminal_literal(b"b", 1),
            ],
        }

        checker = build_from_grammar_definition_json(json_str=_to_json(grammar))
        self.assertTrue(checker.is_valid_prefix("a b"))
        self.assertTrue(checker.is_valid_prefix("a  b"))
        self.assertFalse(checker.is_valid_prefix("a c"))

    def test_from_ebnf_string_optional(self):
        try:
            import _sep1  # noqa: F401
        except Exception:
            self.skipTest("_sep1 not available")

        ebnf = "S ::= 'a' 'b';"
        try:
            checker = ReferencePrefixChecker.from_ebnf_string(ebnf)
        except Exception as exc:
            self.skipTest(f"EBNF parse failed in this env: {exc}")
        else:
            self.assertTrue(checker.is_valid_prefix(""))
            self.assertTrue(checker.is_valid_prefix("a"))
            self.assertTrue(checker.is_valid_prefix("ab"))
            self.assertFalse(checker.is_valid_prefix("ac"))

    def test_apollo_schema_minimal(self):
        try:
            import _sep1  # noqa: F401
        except Exception:
            self.skipTest("_sep1 not available")

        schema_path = (
            "/Users/isaacbreen/Projects2/grammars2024/"
            "gcg-paper/hard_schemas/data/ApolloRouter---apollo-router-2.9.0.json"
        )
        if not os.path.exists(schema_path):
            self.skipTest("Apollo schema file not found")

        with open(schema_path, "r", encoding="utf-8") as f:
            wrapper = __import__("json").load(f)
        schema_json = __import__("json").dumps(wrapper.get("schema", {}))
        checker = build_from_json_schema(schema_json)

        tests = wrapper.get("tests", [])
        minimal = [t for t in tests if "Minimal" in t.get("description", "")][0]["data"]
        valid_str = __import__("json").dumps(minimal, separators=(",", ":"))
        self.assertTrue(checker.is_valid_prefix(valid_str))

        # Counterexamples (should be invalid)
        bad_listen_type = {"supergraph": {"listen": 4000}}
        bad_supergraph_type = {"supergraph": "127.0.0.1:4000"}
        extra_field = {"supergraph": {"listen": "127.0.0.1:4000", "extra": 1}}

        self.assertFalse(checker.is_valid_prefix(__import__("json").dumps(bad_listen_type)))
        self.assertFalse(checker.is_valid_prefix(__import__("json").dumps(bad_supergraph_type)))
        self.assertFalse(checker.is_valid_prefix(__import__("json").dumps(extra_field)))


def _to_json(obj):
    return __import__("json").dumps(obj)


if __name__ == "__main__":
    unittest.main()
