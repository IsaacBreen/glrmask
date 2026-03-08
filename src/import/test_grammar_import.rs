//! Ported grammar-import tests from sep1.
//!
//! Sources:
//!   - `grammars2024/src/interface/ebnf.rs` (9 tests total; 2 ported, 7 skipped)
//!   - `grammars2024/src/interface/lark.rs` (4 tests total; 2 ported, 2 skipped)
//!
//! Skipped tests:
//!   EBNF (7):
//!     - test_ebnf_parser_greedy_group_directives: needs greedy group directives (absent)
//!     - test_ebnf_parser_allows_wildcard_greedy_group: needs greedy groups
//!     - test_ebnf_parser_rejects_greedy_all_directive: needs greedy_all directive
//!     - test_ebnf_parser_rejects_mixed_greedy_all_and_greedy_group: needs mixed greedy
//!     - test_grammar_definition_from_ebnf_wildcard_greedy_group_directive: needs GrammarDefinition
//!     - test_ebnf_parser_allows_wildcard_and_explicit_greedy_groups: needs wildcard groups
//!     - test_grammar_definition_from_ebnf_wildcard_and_explicit_greedy_groups: needs GrammarDefinition
//!
//!   Lark (2):
//!     - test_lark_ignore_directive: NamedGrammar has no ignore_symbol_name field
//!     - test_lark_repeat_bounded: GrammarExpr has no RepeatBounded variant (desugared)

use crate::import::ast::{GrammarExpr, NamedGrammar};
use crate::import::ebnf::parse_ebnf_to_named;
use crate::import::lark::parse_lark_to_named;

// ── EBNF tests ───────────────────────────────────────────────────────────────

/// Ported from `test_ebnf_parser_simple`.
///
/// Parses a basic EBNF grammar and checks the resulting AST structure.
#[test]
fn test_ebnf_parser_simple() {
    // glrmask EBNF uses newlines (not semicolons) as rule separators
    let ebnf = "\
s ::= a b
a ::= 'a' |
b ::= c*
c ::= 'c'?";
    let named = parse_ebnf_to_named(ebnf).expect("EBNF should parse");

    let expected_rules: Vec<(String, GrammarExpr)> = vec![
        (
            "s".to_string(),
            GrammarExpr::Sequence(vec![
                GrammarExpr::Ref("a".into()),
                GrammarExpr::Ref("b".into()),
            ]),
        ),
        (
            "a".to_string(),
            GrammarExpr::Choice(vec![
                GrammarExpr::Literal(b"a".to_vec()),
                GrammarExpr::Sequence(vec![]),
            ]),
        ),
        (
            "b".to_string(),
            GrammarExpr::Repeat(Box::new(GrammarExpr::Ref("c".into()))),
        ),
        (
            "c".to_string(),
            GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"c".to_vec()))),
        ),
    ];

    assert_eq!(
        named.rules, expected_rules,
        "EBNF parse should produce expected AST rules"
    );
}

/// Ported from `test_ebnf_parser_error_with_span` (adapted).
///
/// Verifies the parser returns an error for invalid EBNF (double '?').
#[test]
fn test_ebnf_parser_error_for_invalid_syntax() {
    // Double '?' is invalid — parser should fail
    let ebnf = "\
s ::= a b
a ::= 'a' |
b ::= c*
c ::= 'c'??";
    let result = parse_ebnf_to_named(ebnf);
    assert!(
        result.is_err(),
        "EBNF with 'c'?? should produce a parse error"
    );
}

// ── Lark tests ───────────────────────────────────────────────────────────────

/// Ported from `test_lark_parser_simple`.
///
/// Parses a simple Lark grammar and checks rule count and names at the AST level.
#[test]
fn test_lark_parser_simple() {
    let lark = r#"
start: expr

expr: term ("+" term)*

term: NUMBER
    | "(" expr ")"

NUMBER: /[0-9]+/
"#;
    let named = parse_lark_to_named(lark).expect("Lark should parse");

    assert_eq!(
        named.rules.len(),
        4,
        "Expected 4 rules, got {}",
        named.rules.len()
    );
    assert_eq!(named.rules[0].0, "start");
    assert_eq!(named.rules[1].0, "expr");
    assert_eq!(named.rules[2].0, "term");
    assert_eq!(named.rules[3].0, "NUMBER");
}

/// Ported from `test_lark_regex_charclass_not_nested`.
///
/// Verifies that regex char classes like `/[^"\\\x00-\x1F]/` are parsed correctly
/// as a flat CharClass (not nested `[[...]]`).
#[test]
fn test_lark_regex_charclass_not_nested() {
    let lark = r#"
start: STR_CHAR
STR_CHAR: /[^"\\\x00-\x1F]/
"#;
    let named = parse_lark_to_named(lark).expect("Lark should parse");

    let str_char_expr = named
        .rules
        .iter()
        .find(|(name, _)| name == "STR_CHAR")
        .map(|(_, expr)| expr)
        .expect("STR_CHAR rule should exist");

    // glrmask keeps the full regex pattern as RawRegex rather than
    // extracting into CharClass during Lark import.
    match str_char_expr {
        GrammarExpr::RawRegex(pattern) => {
            assert!(
                pattern.contains("^\""),
                "Pattern should contain the negated quote class, got: {}",
                pattern
            );
            assert!(
                !pattern.starts_with("[["),
                "Regex should not be double-nested"
            );
        }
        other => panic!("Expected RawRegex, got {:?}", other),
    }
}
