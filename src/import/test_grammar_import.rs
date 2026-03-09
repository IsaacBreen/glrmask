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
use crate::grammar::ast::lower;

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

/// Ported from `test_lark_repeat_bounded` (adapted to glrmask's desugared AST).
#[test]
fn test_lark_repeat_bounded_desugars_to_optional_tail() {
    let lark = r#"
start: STR_CHAR~3..5
STR_CHAR: "a"
"#;
    let named = parse_lark_to_named(lark).expect("Lark should parse bounded repeats");

    assert_eq!(named.rules[0].0, "start");
    assert_eq!(
        named.rules[0].1,
        GrammarExpr::Sequence(vec![
            GrammarExpr::Literal(b"a".to_vec()),
            GrammarExpr::Literal(b"a".to_vec()),
            GrammarExpr::Literal(b"a".to_vec()),
            GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::Literal(b"a".to_vec()),
                GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"a".to_vec()))),
            ]))),
        ])
    );
}

/// Regression: common Lark syntax features should parse together.
#[test]
fn test_lark_parser_supports_single_quotes_ranges_aliases_and_priority() {
    let lark = "?start: DIGIT -> picked\nDIGIT.2: '0'..'9'";
    let named = parse_lark_to_named(lark).expect("Lark syntax subset should parse");

    assert_eq!(named.rules.len(), 1);
    assert_eq!(named.rules[0].0, "start");
    assert_eq!(
        named.rules[0].1,
        GrammarExpr::CharClass {
            def: "0-9".into(),
            negate: false,
            utf8: true,
        }
    );
}

#[test]
fn test_lark_terminal_rules_follow_capitalization_convention() {
    let lark = "start: WORD\nWORD: LETTER+\nLETTER: 'a' | 'b'";
    let named = parse_lark_to_named(lark).expect("Lark terminal rules should inline into parser rules");

    assert_eq!(named.rules.len(), 1);
    assert_eq!(named.rules[0].0, "start");
    assert_eq!(
        named.rules[0].1,
        GrammarExpr::RepeatOne(Box::new(GrammarExpr::Choice(vec![
            GrammarExpr::Literal(b"a".to_vec()),
            GrammarExpr::Literal(b"b".to_vec()),
        ])))
    );
}

#[test]
fn test_lark_terminal_rule_rejects_parser_rule_reference() {
    let err = parse_lark_to_named("start: WORD\nitem: 'a'\nWORD: item")
        .expect_err("uppercase terminal rules should not reference lowercase parser rules");
    assert!(
        err.to_string().contains("terminal rule cannot reference parser rule item"),
        "unexpected error: {err}"
    );
}

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
        3,
        "Expected 3 parser rules after terminal inlining, got {}",
        named.rules.len()
    );
    assert_eq!(named.rules[0].0, "start");
    assert_eq!(named.rules[1].0, "expr");
    assert_eq!(named.rules[2].0, "term");
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

    let str_char_expr = &named.rules[0].1;

    // After terminal-rule normalization, the start rule inlines the regex.
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

/// Ported from `test_duplicate_terminals_are_merged`.
#[test]
fn test_lark_duplicate_terminals_are_merged() {
    let lark = r#"
start: A B C D E F G H
A: "x"
B: "x"
C: "x"
D: "x"
E: "x"
F: "x"
G: "x"
H: "x"
"#;
    let named = parse_lark_to_named(lark).expect("Lark should parse");
    let lowered = lower(&named).expect("lowering should succeed");

    assert_eq!(
        lowered.terminals.len(),
        1,
        "identical terminals should be deduplicated into one lowered terminal"
    );
}

/// Ported from `test_lark_terminal_chain_differs_from_ebnf_terminal_chain_when_utf8_enabled`.
#[test]
fn test_lark_json_string_chain_differs_from_ebnf_char_class_chain() {
    let lark = r#"
start: obj
obj: "{" pair ("," pair)* "}"
pair: JSON_STRING ":" JSON_STRING
JSON_STRING: "\"" STR_CHAR* "\""
STR_CHAR: /[^"\\\x00-\x1F]/
"#;

    let ebnf = r#"
start ::= obj
obj ::= "{" pair ("," pair)* "}"
pair ::= JSON_STRING ":" JSON_STRING
JSON_STRING ::= '"' STR_CHAR* '"'
STR_CHAR ::= [^"\\\x00-\x1F]
"#;

    let lark_named = parse_lark_to_named(lark).expect("Lark grammar should parse");
    let ebnf_named = parse_ebnf_to_named(ebnf).expect("EBNF grammar should parse");

    assert_eq!(
        lark_named
            .rules
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["start", "obj", "pair"],
        "Lark normalization should inline terminal rules into the parser-rule skeleton"
    );
    assert!(
        ebnf_named.rules.iter().any(|(name, _)| name == "JSON_STRING")
            && ebnf_named.rules.iter().any(|(name, _)| name == "STR_CHAR"),
        "EBNF import should keep the explicit terminal-chain helper rules visible in the named grammar"
    );
    assert!(
        lark_named.rules.len() < ebnf_named.rules.len(),
        "Lark terminal inlining should produce fewer named rules than the EBNF path"
    );
    assert_ne!(
        lark_named.rules,
        ebnf_named.rules,
        "Lark regex terminals and EBNF char classes should remain distinct in the normalized AST"
    );

    let lark_lowered = lower(&lark_named).expect("Lark lowering should succeed");
    let ebnf_lowered = lower(&ebnf_named).expect("EBNF lowering should succeed");
    assert_ne!(
        lark_lowered.rules,
        ebnf_lowered.rules,
        "the importer-path difference should remain observable after lowering"
    );
    assert!(
        lark_lowered.rules.len() > ebnf_lowered.rules.len(),
        "Lark terminal inlining should introduce a larger lowered rule graph than the EBNF helper-rule form"
    );
}
