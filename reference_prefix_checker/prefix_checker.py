"""
Reference prefix checker (Earley + regex tokenizer).

Goal: a rock-solid, simple, slow-but-correct implementation that determines
whether a given byte string is a valid *prefix* of a grammar.

Input grammar format:
- Productions: list of {lhs: str, rhs: [Symbol]}
- Symbol: {variant: "Terminal"|"NonTerminal", value: str|bytes}
- Regex terminals: list of {name: str, group_id: int, expr: ExprJSON}
- Literal terminals: list of {value: [u8], group_id: int}

This aligns with GrammarDefinition JSON produced by sep1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import regex as re  # supports partial matching
except Exception as exc:  # pragma: no cover - runtime check
    re = None
    _REGEX_IMPORT_ERROR = exc
else:
    _REGEX_IMPORT_ERROR = None


# -----------------------------
# Regex expression AST
# -----------------------------

@dataclass(frozen=True)
class Expr:
    kind: str
    value: Optional[object] = None


def _parse_expr_json(node: object) -> Expr:
    if isinstance(node, dict):
        variant = node.get("variant")
        if variant:
            if variant == "U8Seq":
                return Expr("U8Seq", node.get("bytes", []))
            if variant == "U8Class":
                return Expr("U8Class", node.get("u8set", []))
            if variant == "Shared":
                return Expr("Shared", _parse_expr_json(node.get("inner")))
            if variant == "Quantifier":
                return Expr("Quantifier", (node.get("q_type"), _parse_expr_json(node.get("expr"))))
            if variant == "Choice":
                return Expr("Choice", [_parse_expr_json(v) for v in node.get("exprs", [])])
            if variant == "Seq":
                return Expr("Seq", [_parse_expr_json(v) for v in node.get("exprs", [])])
            if variant == "Epsilon":
                return Expr("Epsilon")
        # Fallback for alternative enum encoding
        if "U8Seq" in node:
            return Expr("U8Seq", node["U8Seq"].get("bytes", []))
        if "U8Class" in node:
            return Expr("U8Class", node["U8Class"].get("u8set", []))
        if "Shared" in node:
            return Expr("Shared", _parse_expr_json(node["Shared"].get("inner")))
        if "Quantifier" in node:
            payload = node["Quantifier"]
            return Expr("Quantifier", (payload.get("q_type"), _parse_expr_json(payload.get("expr"))))
        if "Choice" in node:
            return Expr("Choice", [_parse_expr_json(v) for v in node["Choice"].get("exprs", [])])
        if "Seq" in node:
            return Expr("Seq", [_parse_expr_json(v) for v in node["Seq"].get("exprs", [])])
        if "Epsilon" in node:
            return Expr("Epsilon")
    raise ValueError(f"Unsupported Expr JSON: {node}")


def _u8set_to_ranges(u8set_json: object) -> List[Tuple[int, int]]:
    if not isinstance(u8set_json, list):
        raise ValueError(f"U8Set JSON must be a list, got: {u8set_json}")
    ranges: List[Tuple[int, int]] = []
    for item in u8set_json:
        if isinstance(item, int):
            ranges.append((item, item))
        elif isinstance(item, list) and len(item) == 2:
            start, end = item
            ranges.append((int(start), int(end)))
        else:
            raise ValueError(f"Invalid U8Set entry: {item}")
    return ranges


def _byte_escape(b: int) -> str:
    return f"\\x{b:02x}"


def _ranges_to_charclass(ranges: List[Tuple[int, int]]) -> str:
    parts = []
    for start, end in ranges:
        if start == end:
            parts.append(_byte_escape(start))
        else:
            parts.append(f"{_byte_escape(start)}-{_byte_escape(end)}")
    return "[" + "".join(parts) + "]"


def _expr_to_regex(expr: Expr) -> str:
    kind = expr.kind
    if kind == "U8Seq":
        return "".join(_byte_escape(b) for b in expr.value)
    if kind == "U8Class":
        return _ranges_to_charclass(_u8set_to_ranges(expr.value))
    if kind == "Shared":
        return _expr_to_regex(expr.value)
    if kind == "Quantifier":
        q_type, inner = expr.value
        inner_pat = _expr_to_regex(inner)
        if q_type == "ZeroOrMore":
            return f"(?:{inner_pat})*"
        if q_type == "OneOrMore":
            return f"(?:{inner_pat})+"
        if q_type == "ZeroOrOne":
            return f"(?:{inner_pat})?"
        raise ValueError(f"Unknown quantifier: {q_type}")
    if kind == "Choice":
        parts = [_expr_to_regex(p) for p in expr.value]
        return "(?:" + "|".join(parts) + ")"
    if kind == "Seq":
        return "".join(_expr_to_regex(p) for p in expr.value)
    if kind == "Epsilon":
        return ""
    raise ValueError(f"Unknown Expr kind: {kind}")


class RegexMatcher:
    def __init__(self, expr: Expr):
        if _REGEX_IMPORT_ERROR is not None:
            raise RuntimeError(
                "The 'regex' module is required for partial matching. "
                f"Import error: {_REGEX_IMPORT_ERROR}"
            )
        pattern = _expr_to_regex(expr)
        self._pattern = re.compile(pattern.encode("ascii"))

    def match_max(self, data: bytes) -> int:
        m = self._pattern.match(data)
        if m is None:
            return 0
        return m.end()

    def partial_at_end(self, data: bytes) -> bool:
        m = self._pattern.match(data, partial=True)
        return bool(m is not None and m.partial and m.end() == len(data))


# -----------------------------
# Grammar structures
# -----------------------------

@dataclass(frozen=True)
class TerminalKey:
    kind: str  # "regex" or "literal"
    value: Union[str, bytes]


@dataclass(frozen=True)
class Symbol:
    kind: str  # "T" or "N"
    value: Union[TerminalKey, str]


@dataclass(frozen=True)
class Production:
    lhs: str
    rhs: Tuple[Symbol, ...]


# -----------------------------
# Earley parser (prefix)
# -----------------------------

@dataclass(frozen=True)
class Item:
    prod_id: int
    dot: int
    origin: int


class EarleyPrefixChecker:
    def __init__(
        self,
        productions: Sequence[Production],
        start_symbol: str,
        token_matchers: Dict[TerminalKey, RegexMatcher],
        ignore_terminals: Optional[set[TerminalKey]] = None,
    ) -> None:
        self.productions = list(productions)
        self.start_symbol = start_symbol
        self.token_matchers = token_matchers
        self.ignore_terminals = ignore_terminals or set()

        self._prods_by_lhs: Dict[str, List[int]] = {}
        for i, prod in enumerate(self.productions):
            self._prods_by_lhs.setdefault(prod.lhs, []).append(i)

        self._productive_nts = self._compute_productive_nonterminals()

    def _compute_productive_nonterminals(self) -> set[str]:
        productive: set[str] = set()
        changed = True
        while changed:
            changed = False
            for prod in self.productions:
                if prod.lhs in productive:
                    continue
                if all(self._symbol_productive(sym, productive) for sym in prod.rhs):
                    productive.add(prod.lhs)
                    changed = True
        return productive

    @staticmethod
    def _symbol_productive(sym: Symbol, productive_nts: set[str]) -> bool:
        if sym.kind == "T":
            return True
        return sym.value in productive_nts

    def _suffix_productive(self, prod: Production, dot: int) -> bool:
        return all(self._symbol_productive(sym, self._productive_nts) for sym in prod.rhs[dot:])

    def is_valid_prefix(self, text: Union[str, bytes]) -> bool:
        if isinstance(text, str):
            text_bytes = text.encode("utf-8")
        else:
            text_bytes = text

        n = len(text_bytes)
        chart: List[set[Item]] = [set() for _ in range(n + 1)]
        for pid in self._prods_by_lhs.get(self.start_symbol, []):
            chart[0].add(Item(pid, 0, 0))

        prefix_ok = False
        match_cache: Dict[int, Tuple[Dict[TerminalKey, int], Dict[TerminalKey, bool]]] = {}

        def compute_matches(pos: int) -> Tuple[Dict[TerminalKey, int], Dict[TerminalKey, bool]]:
            if pos in match_cache:
                return match_cache[pos]
            remaining = text_bytes[pos:]
            full_len_by_term: Dict[TerminalKey, int] = {}
            partial_by_term: Dict[TerminalKey, bool] = {}
            for term, matcher in self.token_matchers.items():
                full_len = matcher.match_max(remaining)
                partial = matcher.partial_at_end(remaining)
                full_len_by_term[term] = full_len
                partial_by_term[term] = partial
            match_cache[pos] = (full_len_by_term, partial_by_term)
            return match_cache[pos]

        for i in range(n + 1):
            # Closure: predict + complete
            agenda = list(chart[i])
            while agenda:
                item = agenda.pop()
                prod = self.productions[item.prod_id]
                if item.dot < len(prod.rhs):
                    sym = prod.rhs[item.dot]
                    if sym.kind == "N":
                        for pid in self._prods_by_lhs.get(sym.value, []):
                            new_item = Item(pid, 0, i)
                            if new_item not in chart[i]:
                                chart[i].add(new_item)
                                agenda.append(new_item)
                else:
                    # Completion
                    for prev in list(chart[item.origin]):
                        prev_prod = self.productions[prev.prod_id]
                        if prev.dot < len(prev_prod.rhs):
                            sym = prev_prod.rhs[prev.dot]
                            if sym.kind == "N" and sym.value == prod.lhs:
                                new_item = Item(prev.prod_id, prev.dot + 1, prev.origin)
                                if new_item not in chart[i]:
                                    chart[i].add(new_item)
                                    agenda.append(new_item)

            # Scan
            if i <= n:
                full_len_by_term, partial_by_term = compute_matches(i)
                for item in list(chart[i]):
                    prod = self.productions[item.prod_id]
                    if item.dot >= len(prod.rhs):
                        continue
                    sym = prod.rhs[item.dot]
                    if sym.kind != "T":
                        continue
                    term = sym.value
                    full_len = full_len_by_term.get(term, 0)
                    if full_len > 0:
                        next_pos = i + full_len
                        if next_pos <= n:
                            chart[next_pos].add(Item(item.prod_id, item.dot + 1, item.origin))
                    # Prefix via partial token match that consumes all remaining input
                    if partial_by_term.get(term, False):
                        prefix_ok = True

                # Ignore terminals: consume input without advancing parse state
                if self.ignore_terminals:
                    for term in self.ignore_terminals:
                        full_len = full_len_by_term.get(term, 0)
                        if full_len > 0:
                            next_pos = i + full_len
                            if next_pos <= n:
                                for item in chart[i]:
                                    chart[next_pos].add(item)

        # Accept if a completed start production exists
        for item in chart[n]:
            prod = self.productions[item.prod_id]
            if prod.lhs == self.start_symbol and item.dot == len(prod.rhs):
                return True

        # Otherwise accept if any item at end can still be extended productively
        if any(self._suffix_productive(self.productions[it.prod_id], it.dot) for it in chart[n]):
            return True

        return prefix_ok


# -----------------------------
# GrammarDefinition JSON helpers
# -----------------------------


def _parse_quoted_literal(value: str) -> Optional[bytes]:
    if not value:
        return None
    if (value.startswith("\"") and value.endswith("\"")) or (value.startswith("'") and value.endswith("'")):
        try:
            import ast
            parsed = ast.literal_eval(value)
            if isinstance(parsed, str):
                return parsed.encode("utf-8")
        except Exception:
            return None
    return None


def _parse_symbol_json(
    node: dict,
    literal_map: Dict[str, bytes],
    regex_names: set[str],
) -> Symbol:
    variant = node.get("variant")
    value = node.get("value")
    if variant == "NonTerminal":
        return Symbol("N", value)
    if variant == "Terminal":
        # Terminal value may be a Regex/Literal object or a string.
        if isinstance(value, dict):
            t_type = value.get("type") or value.get("variant")
            if t_type == "Literal":
                literal_bytes = bytes(value.get("value", []))
                return Symbol("T", TerminalKey("literal", literal_bytes))
            if t_type == "Regex":
                return Symbol("T", TerminalKey("regex", value.get("value")))
        if isinstance(value, str):
            if value in regex_names:
                return Symbol("T", TerminalKey("regex", value))
            if value in literal_map:
                return Symbol("T", TerminalKey("literal", literal_map[value]))
            parsed = _parse_quoted_literal(value)
            if parsed is not None:
                return Symbol("T", TerminalKey("literal", parsed))
            # Fallback: treat as literal bytes if it isn't a known regex name
            return Symbol("T", TerminalKey("literal", value.encode("utf-8")))
    raise ValueError(f"Unknown Symbol variant: {variant}")


def _extract_literal_map(grammar_json: dict) -> Dict[str, bytes]:
    literal_map: Dict[str, bytes] = {}
    for entry in grammar_json.get("literal_terminals", []):
        value = entry.get("value", [])
        if isinstance(value, list):
            b = bytes(value)
            literal_map[b.decode("utf-8", errors="backslashreplace")] = b
    return literal_map


def build_from_grammar_definition_json(json_str: str) -> EarleyPrefixChecker:
    data = json.loads(json_str)

    # terminals
    regex_terminals = data.get("regex_terminals", [])
    literal_terminals = data.get("literal_terminals", [])

    token_matchers: Dict[TerminalKey, RegexMatcher] = {}
    group_id_to_terminal: Dict[int, TerminalKey] = {}
    for t in regex_terminals:
        name = t.get("name")
        group_id = t.get("group_id")
        expr = _parse_expr_json(t.get("expr"))
        key = TerminalKey("regex", name)
        token_matchers[key] = RegexMatcher(expr)
        if isinstance(group_id, int):
            group_id_to_terminal[group_id] = key

    for t in literal_terminals:
        value = t.get("value", [])
        group_id = t.get("group_id")
        literal = bytes(value)
        expr = Expr("U8Seq", list(literal))
        key = TerminalKey("literal", literal)
        token_matchers[key] = RegexMatcher(expr)
        if isinstance(group_id, int):
            group_id_to_terminal[group_id] = key

    # productions
    literal_map = _extract_literal_map(data)
    regex_names = {t.get("name") for t in regex_terminals if isinstance(t, dict)}
    productions: List[Production] = []
    for p in data.get("productions", []):
        lhs = p.get("lhs")
        rhs_nodes = p.get("rhs", [])
        rhs = tuple(_parse_symbol_json(n, literal_map, regex_names) for n in rhs_nodes)
        productions.append(Production(lhs, rhs))

    # Ensure matchers exist for any literal terminals referenced in productions
    for prod in productions:
        for sym in prod.rhs:
            if sym.kind == "T":
                term = sym.value
                if isinstance(term, TerminalKey) and term.kind == "literal":
                    if term not in token_matchers:
                        expr = Expr("U8Seq", list(term.value))
                        token_matchers[term] = RegexMatcher(expr)

    start_prod_id = data.get("start_production_id", 0)
    start_symbol = productions[start_prod_id].lhs if productions else ""

    ignore_ids = set(data.get("ignore_terminal_ids", []))
    ignore_terminals = {group_id_to_terminal[i] for i in ignore_ids if i in group_id_to_terminal}

    return EarleyPrefixChecker(
        productions,
        start_symbol,
        token_matchers,
        ignore_terminals=ignore_terminals,
    )


class ReferencePrefixChecker:
    """Simple wrapper: holds grammar + tokenizer and exposes is_valid_prefix(text)."""

    def __init__(self, checker: EarleyPrefixChecker) -> None:
        self._checker = checker

    @staticmethod
    def from_grammar_definition_json(json_str: str) -> "ReferencePrefixChecker":
        return ReferencePrefixChecker(build_from_grammar_definition_json(json_str))

    @staticmethod
    def from_ebnf_string(ebnf: str) -> "ReferencePrefixChecker":
        return ReferencePrefixChecker(build_from_ebnf_string(ebnf))

    @staticmethod
    def from_json_schema(schema_json: str) -> "ReferencePrefixChecker":
        return ReferencePrefixChecker(build_from_json_schema(schema_json))

    def is_valid_prefix(self, text: Union[str, bytes]) -> bool:
        return self._checker.is_valid_prefix(text)


# -----------------------------
# Optional sep1 bindings helpers
# -----------------------------


def build_from_ebnf_string(ebnf: str) -> EarleyPrefixChecker:
    try:
        import _sep1 as sep1
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("_sep1 is required for EBNF import") from exc

    # GrammarDefinition only has from_ebnf_file in the bindings; use a temp file.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".ebnf", delete=False) as f:
        f.write(ebnf)
        path = f.name
    gd = sep1.GrammarDefinition.from_ebnf_file(path)
    return build_from_grammar_definition_json(gd.to_json_string())


def build_from_json_schema(schema_json: str) -> EarleyPrefixChecker:
    try:
        import _sep1 as sep1
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("_sep1 is required for JSON Schema import") from exc
    if hasattr(sep1, "json_schema_to_ebnf"):
        ebnf = sep1.json_schema_to_ebnf(schema_json)
    elif hasattr(sep1, "json_schema_to_ebnf_py"):
        ebnf = sep1.json_schema_to_ebnf_py(schema_json)
    else:
        raise RuntimeError("_sep1 does not expose json_schema_to_ebnf")
    return build_from_ebnf_string(ebnf)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reference prefix checker (Earley + regex)")
    parser.add_argument("--grammar-json", type=str, help="GrammarDefinition JSON string or path")
    parser.add_argument("--grammar-json-file", type=str, help="Path to GrammarDefinition JSON")
    parser.add_argument("--text", type=str, required=True, help="Input text to check")
    args = parser.parse_args()

    if args.grammar_json_file:
        with open(args.grammar_json_file, "r", encoding="utf-8") as f:
            grammar_json = f.read()
    elif args.grammar_json:
        if args.grammar_json.strip().startswith("{"):
            grammar_json = args.grammar_json
        else:
            with open(args.grammar_json, "r", encoding="utf-8") as f:
                grammar_json = f.read()
    else:
        raise SystemExit("Provide --grammar-json or --grammar-json-file")

    checker = build_from_grammar_definition_json(grammar_json)
    ok = checker.is_valid_prefix(args.text)
    print("valid_prefix" if ok else "invalid_prefix")
