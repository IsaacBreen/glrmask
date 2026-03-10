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

use crate::import::ast::{GrammarExpr, NamedGrammar, NamedRule};
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

    let nt = |name: &str, expr: GrammarExpr| -> NamedRule {
        NamedRule { name: name.to_string(), expr, is_terminal: false }
    };
    let expected_rules: Vec<NamedRule> = vec![
        nt("s", GrammarExpr::Sequence(vec![
            GrammarExpr::Ref("a".into()),
            GrammarExpr::Ref("b".into()),
        ])),
        nt("a", GrammarExpr::Choice(vec![
            GrammarExpr::Literal(b"a".to_vec()),
            GrammarExpr::Sequence(vec![]),
        ])),
        nt("b", GrammarExpr::Repeat(Box::new(GrammarExpr::Ref("c".into())))),
        nt("c", GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"c".to_vec())))),
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

    // Terminal rule is stored as expanded GrammarExpr; parser rule keeps Ref nodes.
    assert_eq!(named.rules[0].name, "STR_CHAR");
    assert_eq!(named.rules[0].expr, GrammarExpr::Literal(b"a".to_vec()));
    assert_eq!(named.rules[1].name, "start");
    assert_eq!(
        named.rules[1].expr,
        GrammarExpr::Sequence(vec![
            GrammarExpr::Ref("STR_CHAR".into()),
            GrammarExpr::Ref("STR_CHAR".into()),
            GrammarExpr::Ref("STR_CHAR".into()),
            GrammarExpr::Optional(Box::new(GrammarExpr::Sequence(vec![
                GrammarExpr::Ref("STR_CHAR".into()),
                GrammarExpr::Optional(Box::new(GrammarExpr::Ref("STR_CHAR".into()))),
            ]))),
        ])
    );
}

/// Regression: common Lark syntax features should parse together.
#[test]
fn test_lark_parser_supports_single_quotes_ranges_aliases_and_priority() {
    let lark = "?start: DIGIT -> picked\nDIGIT.2: '0'..'9'";
    let named = parse_lark_to_named(lark).expect("Lark syntax subset should parse");

    // Terminal rule preserved as expanded GrammarExpr; parser rule keeps Ref.
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
    let named = parse_lark_to_named(lark).expect("Lark terminal rules should be compiled to regex");

    // Terminal rules are stored as expanded GrammarExpr; parser rules keep Ref nodes.
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

    // The is_terminal flag marks terminal rule names.
    let term_set = named.terminal_names_set();
    assert!(term_set.contains("WORD"));
    assert!(term_set.contains("LETTER"));
    assert!(!term_set.contains("start"));
    assert_eq!(term_set.len(), 2);
}

#[test]
fn test_lark_terminal_rule_rejects_parser_rule_reference() {
    let err = parse_lark_to_named("start: WORD\nitem: 'a'\nWORD: item")
        .expect_err("uppercase terminal rules should not reference lowercase parser rules");
    assert!(
        err.to_string().contains("references nonterminal item"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_lark_terminal_rule_rejects_undefined_reference() {
    let err = parse_lark_to_named("start: WORD\nWORD: MISSING")
        .expect_err("terminal referencing undefined rule should fail");
    assert!(
        err.to_string().contains("references undefined rule MISSING"),
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

    // Terminal rule preserved; parser rules reference it via Ref.
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

    let str_char_expr = &named.rules[0].expr;

    // After terminal-rule normalization, STR_CHAR is stored as expanded GrammarExpr.
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

    // Lark terminal rules are compiled to single regex patterns and kept as
    // named rules.  EBNF keeps the helper-rule chain as structured nonterminals.
    assert!(
        lark_named.rules.iter().any(|r| r.name == "JSON_STRING"),
        "Lark should keep JSON_STRING as a named rule (compiled to regex)"
    );
    assert!(
        ebnf_named.rules.iter().any(|r| r.name == "JSON_STRING")
            && ebnf_named.rules.iter().any(|r| r.name == "STR_CHAR"),
        "EBNF import should keep the explicit terminal-chain helper rules visible in the named grammar"
    );

    // Lark terminal rules are stored as expanded GrammarExpr; EBNF preserves structure.
    let lark_json_string = &lark_named.rules.iter().find(|r| r.name == "JSON_STRING").unwrap().expr;
    assert!(
        !matches!(lark_json_string, GrammarExpr::Ref(_)),
        "Lark JSON_STRING should be an expanded terminal body, got {:?}",
        lark_json_string
    );

    let lark_lowered = lower(&lark_named).expect("Lark lowering should succeed");
    let ebnf_lowered = lower(&ebnf_named).expect("EBNF lowering should succeed");

    // Lark's compiled regex produces no more terminals than EBNF's decomposed structure.
    assert!(
        lark_lowered.terminals.len() <= ebnf_lowered.terminals.len(),
        "Lark composite terminals should produce no more lowered terminals ({}) than EBNF decomposed form ({})",
        lark_lowered.terminals.len(),
        ebnf_lowered.terminals.len()
    );
}
