//! Regression tests for grammar import.
//!
//! Cases that depend on greedy directives or missing ignore metadata stay
//! omitted because they do not map onto glrmask's current import surface.

use std::fmt::Display;

use crate::import::ast::{GrammarExpr, NamedGrammar, NamedRule};
use crate::import::ebnf::parse_ebnf_to_named;
use crate::import::lark::parse_lark_to_named;
use crate::grammar::ast::lower;

fn parse_ebnf_named(input: &str) -> NamedGrammar {
    parse_ebnf_to_named(input).expect("EBNF should parse")
}

fn parse_lark_named(input: &str) -> NamedGrammar {
    parse_lark_to_named(input).expect("Lark should parse")
}

fn nonterminal_rule(name: &str, expr: GrammarExpr) -> NamedRule {
    NamedRule {
        name: name.to_string(),
        expr,
        is_terminal: false,
        is_internal: false,
    }
}

fn assert_error_contains<T, E>(result: Result<T, E>, expected: &str, context: &str)
where
    T: std::fmt::Debug,
    E: Display,
{
    let error = result.expect_err(context);
    assert!(error.to_string().contains(expected), "unexpected error: {error}");
}

/// Adapted from `test_ebnf_parser_simple`.
///
/// Parses a basic EBNF grammar and checks the resulting AST structure.
#[test]
fn test_ebnf_parser_simple() {
    let ebnf = "\
s ::= a b
a ::= 'a' |
b ::= c*
c ::= 'c'?";
    let named = parse_ebnf_named(ebnf);

    let expected_rules: Vec<NamedRule> = vec![
        nonterminal_rule("s", GrammarExpr::Sequence(vec![
            GrammarExpr::Ref("a".into()),
            GrammarExpr::Ref("b".into()),
        ])),
        nonterminal_rule("a", GrammarExpr::Choice(vec![
            GrammarExpr::Literal(b"a".to_vec()),
            GrammarExpr::Sequence(vec![]),
        ])),
        nonterminal_rule("b", GrammarExpr::Repeat(Box::new(GrammarExpr::Ref("c".into())))),
        nonterminal_rule("c", GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"c".to_vec())))),
    ];

    assert_eq!(
        named.rules, expected_rules,
        "EBNF parse should produce expected AST rules"
    );
}

/// Adapted from `test_ebnf_parser_error_with_span`.
///
/// Verifies the parser returns an error for invalid EBNF (double '?').
#[test]
fn test_ebnf_parser_error_for_invalid_syntax() {
    let ebnf = "\
s ::= a b
a ::= 'a' |
b ::= c*
c ::= 'c'??";
    assert!(parse_ebnf_to_named(ebnf).is_err(), "EBNF with 'c'?? should produce a parse error");
}

/// Adapted from `test_lark_repeat_bounded`.
#[test]
fn test_lark_repeat_bounded_preserves_range_node() {
    let lark = r#"
start: STR_CHAR~3..5
STR_CHAR: "a"
"#;
    let named = parse_lark_named(lark);

    assert_eq!(named.rules[0].name, "STR_CHAR");
    assert_eq!(named.rules[0].expr, GrammarExpr::Literal(b"a".to_vec()));
    assert_eq!(named.rules[1].name, "start");
    assert_eq!(
        named.rules[1].expr,
        GrammarExpr::RepeatRange {
            expr: Box::new(GrammarExpr::Ref("STR_CHAR".into())),
            min: 3,
            max: 5,
        }
    );
}

/// Regression: common Lark syntax features should parse together.
#[test]
fn test_lark_parser_supports_single_quotes_ranges_aliases_and_priority() {
    let lark = "?start: DIGIT -> picked\nDIGIT.2: '0'..'9'";
    let named = parse_lark_named(lark);

    assert_eq!(named.rules.len(), 2);
    assert_eq!(named.rules[0].name, "DIGIT");
    assert_eq!(
        named.rules[0].expr,
        GrammarExpr::CharClass { def: "0-9".into(), negate: false, utf8: true }
    );
    assert_eq!(named.rules[1].name, "start");
    assert_eq!(named.rules[1].expr, GrammarExpr::Ref("DIGIT".into()));
}

#[test]
fn test_lark_terminal_rules_follow_capitalization_convention() {
    let lark = "start: WORD\nWORD: LETTER+\nLETTER: 'a' | 'b'";
    let named = parse_lark_named(lark);

    assert_eq!(named.rules.len(), 3);
    assert_eq!(named.rules[0].name, "WORD");
    assert_eq!(named.rules[1].name, "LETTER");
    assert_eq!(
        named.rules[1].expr,
        GrammarExpr::Choice(vec![
            GrammarExpr::Literal(b"a".to_vec()),
            GrammarExpr::Literal(b"b".to_vec()),
        ])
    );
    assert_eq!(named.rules[2].name, "start");
    assert_eq!(named.rules[2].expr, GrammarExpr::Ref("WORD".into()));

    let term_set = named.terminal_names_set();
    assert!(term_set.contains("WORD"));
    assert!(term_set.contains("LETTER"));
    assert!(!term_set.contains("start"));
    assert_eq!(term_set.len(), 2);
}

#[test]
fn test_lark_terminal_rule_rejects_parser_rule_reference() {
    assert_error_contains(
        parse_lark_to_named("start: WORD\nitem: 'a'\nWORD: item"),
        "references nonterminal item",
        "uppercase terminal rules should not reference lowercase parser rules",
    );
}

#[test]
fn test_lark_terminal_rule_rejects_undefined_reference() {
    assert_error_contains(
        parse_lark_to_named("start: WORD\nWORD: MISSING"),
        "references undefined rule MISSING",
        "terminal referencing undefined rule should fail",
    );
}

/// Adapted from the original simple Lark parser smoke test.
///
/// Parses a simple Lark grammar and checks rule count and names at the AST level.
#[test]
fn test_lark_parser_reports_expected_rule_names() {
    let lark = r#"
start: expr

expr: term ("+" term)*

term: NUMBER
    | "(" expr ")"

NUMBER: /[0-9]+/
"#;
    let named = parse_lark_named(lark);

    assert_eq!(
        named.rules.len(),
        4,
        "Expected 4 rules (1 terminal + 3 parser), got {}",
        named.rules.len()
    );
    assert_eq!(named.rules[0].name, "NUMBER");
    assert_eq!(named.rules[1].name, "start");
    assert_eq!(named.rules[2].name, "expr");
    assert_eq!(named.rules[3].name, "term");
}

/// Adapted from `test_lark_regex_charclass_not_nested`.
///
/// Verifies that regex char classes like `/[^"\\\x00-\x1F]/` are parsed correctly
/// as a flat CharClass (not nested `[[...]]`).
#[test]
fn test_lark_regex_charclass_not_nested() {
    let lark = r#"
start: STR_CHAR
STR_CHAR: /[^"\\\x00-\x1F]/
"#;
    let named = parse_lark_named(lark);

    let str_char_expr = &named.rules[0].expr;

    match str_char_expr {
        GrammarExpr::RawRegex(pattern) => {
            assert!(
                pattern.contains("^\"") || pattern.contains(r#"^""#),
                "Pattern should contain the negated quote class, got: {}",
                pattern
            );
        }
        other => panic!("Expected RawRegex (expanded terminal), got {:?}", other),
    }
}

/// Adapted from `test_duplicate_terminals_are_merged`.
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
    let named = parse_lark_named(lark);
    let lowered = lower(&named).expect("lowering should succeed");

    assert_eq!(
        lowered.terminals.len(),
        1,
        "identical terminals should be deduplicated into one lowered terminal"
    );
}

/// Adapted from `test_lark_terminal_chain_differs_from_ebnf_terminal_chain_when_utf8_enabled`.
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

    let lark_named = parse_lark_named(lark);
    let ebnf_named = parse_ebnf_named(ebnf);

    assert!(
        lark_named.rules.iter().any(|r| r.name == "JSON_STRING"),
        "Lark should keep JSON_STRING as a named rule (compiled to regex)"
    );
    assert!(
        ebnf_named.rules.iter().any(|r| r.name == "JSON_STRING")
            && ebnf_named.rules.iter().any(|r| r.name == "STR_CHAR"),
        "EBNF import should keep the explicit terminal-chain helper rules visible in the named grammar"
    );

    let lark_json_string = &lark_named.rules.iter().find(|r| r.name == "JSON_STRING").unwrap().expr;
    assert!(
        !matches!(lark_json_string, GrammarExpr::Ref(_)),
        "Lark JSON_STRING should be an expanded terminal body, got {:?}",
        lark_json_string
    );

    let lark_lowered = lower(&lark_named).expect("Lark lowering should succeed");
    let ebnf_lowered = lower(&ebnf_named).expect("EBNF lowering should succeed");

    assert!(
        lark_lowered.terminals.len() <= ebnf_lowered.terminals.len(),
        "Lark composite terminals should produce no more lowered terminals ({}) than EBNF decomposed form ({})",
        lark_lowered.terminals.len(),
        ebnf_lowered.terminals.len()
    );
}
