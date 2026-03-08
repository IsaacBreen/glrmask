//! Ported from sep1: `grammars2024/src/glr/tests.rs`
//!
//! Source had 35 tests. 8 were already ported (as `test_ported_glr_*` in parser.rs).
//! 8 more involve `analyze::validate` (absent in glrmask). 4 need
//! `remove_productions_with_undefined_nonterminals` (absent). 1 needs
//! `filter_productions_by_reachability` (absent). 4 are `#[ignore]` (need stats/explain APIs).
//! 3 are duplicates of already-ported grammar patterns (expression_grammar, unit_production).
//!
//! Newly ported: 7 tests below.
//!
//! Skipped tests and reasons:
//!   Already ported (8):
//!     - test_ambiguous_dangling_else → test_ported_glr_ambiguous_dangling_else
//!     - test_reduce_reduce_conflict → test_ported_glr_reduce_reduce_conflict
//!     - test_epsilon_rules_ambiguity → test_ported_glr_epsilon_ambiguity
//!     - test_highly_ambiguous_potentially_slow → test_ported_glr_highly_ambiguous
//!     - test_right_recursive_grammar_parse → test_ported_glr_right_recursive
//!     - test_nullable_nonterminal_before_terminal → test_ported_glr_nullable_before_terminal
//!     - test_expression_parse_table_generation_and_parse → test_ported_glr_expression_grammar
//!     - test_hidden_left_recursion → test_ported_glr_left_recursive (conceptual match;
//!       sep1 only tests validate() which is absent; the parse test was commented out)
//!
//!   Needs analyze::validate (8):
//!     - validation_fails_direct_length_1_recursion
//!     - validation_fails_indirect_length_1_recursion
//!     - validation_fails_direct_length_1_recursion_nullable_prefix
//!     - validation_fails_indirect_length_1_recursion_nullable_prefix
//!     - validation_passes_non_unit_recursion
//!     - validation_fails_left_nullable_left_recursion
//!     - validation_fails_missing_nonterminal
//!     - validation_passes_complex_unit_rules_no_cycle
//!
//!   Needs remove_productions_with_undefined_nonterminals (4):
//!     - test_remove_undefined_simple
//!     - test_remove_undefined_iterative
//!     - test_remove_undefined_no_change
//!     - test_remove_undefined_empty_input
//!
//!   Needs filter_productions_by_reachability (1):
//!     - test_filter_productions_selectivity
//!
//!   #[ignore] in source / needs absent APIs (4):
//!     - test_resolve_right_recursion (#[ignore], needs analyze::resolve_direct_right_recursion)
//!     - test_explain_stack (#[ignore], needs parser.explain_stack)
//!     - test_parser_stats_conflicts (#[ignore], needs stats::get_stats)
//!     - test_lr1_not_lalr1_grammar (#[ignore], needs stats::get_stats)
//!
//!   Duplicate grammar patterns already covered (3):
//!     - test_standard_expression_grammar_parse (same grammar as test_ported_glr_expression_grammar)
//!     - test_unit_production_elimination (same grammar; only unique part is stats, absent)
//!     - validation_passes_standard_grammars (not a test, no #[test] attribute)

use super::analysis::AnalyzedGrammar;
use super::parser::{stacks_finished, GLRParser};
use super::table::GLRTable;
use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal, TerminalID};

// ── helpers ──────────────────────────────────────────────────────────────────

fn tdef(id: u32, name: &str) -> Terminal {
    Terminal {
        id,
        name: name.into(),
    }
}

fn make_grammar(rules: Vec<Rule>, start: u32, terminals: Vec<Terminal>) -> GrammarDef {
    let terminal_patterns = terminals.iter().map(|t| t.name.clone()).collect();
    GrammarDef {
        rules,
        start,
        terminals,
        terminal_patterns,
    }
}

fn build_parser(gdef: &GrammarDef) -> GLRParser {
    let grammar = AnalyzedGrammar::from_grammar_def(gdef);
    let table = GLRTable::build(&grammar);
    GLRParser::new(table)
}

fn accepts(parser: &GLRParser, input: &[TerminalID]) -> bool {
    let mut current = GLRParser {
        table: parser.table.clone(),
        stack: parser.stack.clone(),
    };
    for &token in input {
        let (next, progressed) = current.step(token);
        if !progressed {
            return false;
        }
        current = next;
    }
    stacks_finished(&current.table, &current.stack)
}

/// Check whether the parser can continue after consuming `input` (i.e. there are
/// valid terminals it could shift next, or it is already in an accepting state).
fn can_continue(parser: &GLRParser, input: &[TerminalID]) -> bool {
    let mut current = GLRParser {
        table: parser.table.clone(),
        stack: parser.stack.clone(),
    };
    for &token in input {
        let (next, progressed) = current.step(token);
        if !progressed {
            return false;
        }
        current = next;
    }
    // Can continue if either already accepted or there are valid next terminals
    stacks_finished(&current.table, &current.stack) || !current.valid_terminals().is_empty()
}

// ── ported tests ─────────────────────────────────────────────────────────────

/// Ported from `test_repetition_no_eof_1`.
///
/// Grammar: S -> S a | a  (left-recursive, single terminal 'a')
/// Tests parsing various inputs without EOF. In sep1 the test checks `is_ok()`
/// (can-continue semantics); we test both acceptance and can-continue.
#[test]
fn test_repetition_no_eof_1() {
    // S -> S a | a
    // NT 0 = S
    // T 0 = a
    let gdef = make_grammar(
        vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(0)],
            }, // S -> S a
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }, // S -> a
        ],
        0,
        vec![tdef(0, "a")],
    );
    let parser = build_parser(&gdef);

    // "a" is a valid sentence
    assert!(
        accepts(&parser, &[0]),
        "\"a\" should be accepted (S -> a)"
    );

    // "aaa" is a valid sentence
    assert!(
        accepts(&parser, &[0, 0, 0]),
        "\"aaa\" should be accepted (S -> S a -> S a a -> a a a)"
    );

    // "" (empty) is NOT in the language (S is not nullable), but the parser can continue
    assert!(
        !accepts(&parser, &[]),
        "\"\" should NOT be accepted (S is not nullable)"
    );
    assert!(
        can_continue(&parser, &[]),
        "parser should be able to continue from initial state"
    );
}

/// Ported from `test_repetition_no_eof_2`.
///
/// Grammar: S -> S a | a, Other -> b
/// Tests that invalid token 'b' causes parse failure for the S language.
#[test]
fn test_repetition_no_eof_2() {
    // S -> S a | a, Other -> b
    // NT 0 = S, NT 1 = Other
    // T 0 = a, T 1 = b
    let gdef = make_grammar(
        vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(0)],
            }, // S -> S a
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }, // S -> a
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Terminal(1)],
            }, // Other -> b
        ],
        0,
        vec![tdef(0, "a"), tdef(1, "b")],
    );
    let parser = build_parser(&gdef);

    // "b" should fail (not part of S language)
    assert!(
        !accepts(&parser, &[1]),
        "\"b\" should fail for S language"
    );

    // "ab" should fail (b is not valid after a in S language)
    assert!(
        !accepts(&parser, &[0, 1]),
        "\"ab\" should fail for S language"
    );

    // Confirm "a" still works
    assert!(accepts(&parser, &[0]), "\"a\" should be accepted");
}

/// Ported from `test_super_simple_grammar`.
///
/// Grammar: S -> a eof
/// Tests the simplest possible grammar with explicit EOF terminal.
#[test]
fn test_super_simple_grammar() {
    // S -> a eof
    // NT 0 = S
    // T 0 = a, T 1 = $
    let gdef = make_grammar(
        vec![Rule {
            lhs: 0,
            rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
        }],
        0,
        vec![tdef(0, "a"), tdef(1, "$")],
    );
    let parser = build_parser(&gdef);

    // Valid: "a$"
    assert!(
        accepts(&parser, &[0, 1]),
        "\"a$\" should be accepted"
    );

    // Invalid: "$" alone
    assert!(
        !accepts(&parser, &[1]),
        "\"$\" should fail (wrong first token)"
    );

    // Invalid: "a" alone (missing $)
    assert!(
        !accepts(&parser, &[0]),
        "\"a\" alone should not be accepted (missing $)"
    );

    // Invalid: empty
    assert!(!accepts(&parser, &[]), "empty should not be accepted");
}

/// Ported from `test_simple_parse_table_generation_and_parse`.
///
/// Grammar: S -> A $, A -> A a | b
/// This grammar defines language b a* $ — any number of 'a's after a 'b', terminated by '$'.
#[test]
fn test_simple_parse_table_generation_and_parse() {
    // S -> A $
    // A -> A a
    // A -> b
    // NT 0 = S, NT 1 = A
    // T 0 = a, T 1 = b, T 2 = $
    let gdef = make_grammar(
        vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(2)],
            }, // S -> A $
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)],
            }, // A -> A a
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Terminal(1)],
            }, // A -> b
        ],
        0,
        vec![tdef(0, "a"), tdef(1, "b"), tdef(2, "$")],
    );
    let parser = build_parser(&gdef);

    // "b$" → accepted (A -> b, S -> A $)
    assert!(
        accepts(&parser, &[1, 2]),
        "\"b$\" should be accepted"
    );

    // "ba$" → accepted (A -> A a -> b a, S -> A $)
    assert!(
        accepts(&parser, &[1, 0, 2]),
        "\"ba$\" should be accepted"
    );

    // "baa$" → accepted
    assert!(
        accepts(&parser, &[1, 0, 0, 2]),
        "\"baa$\" should be accepted"
    );

    // "a$" → rejected (cannot start with 'a')
    assert!(
        !accepts(&parser, &[0, 2]),
        "\"a$\" should be rejected"
    );

    // "bb$" → rejected (two b's)
    assert!(
        !accepts(&parser, &[1, 1, 2]),
        "\"bb$\" should be rejected"
    );
}

/// Ported from `test_ambiguous_arithmetic`.
///
/// Grammar: E -> E + E | E * E | id
/// This is ambiguous: id + id * id has two parses.
/// GLR should accept it.
#[test]
fn test_ambiguous_arithmetic() {
    // E -> E + E | E * E | id
    // NT 0 = E
    // T 0 = id, T 1 = +, T 2 = *
    let gdef = make_grammar(
        vec![
            Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Nonterminal(0),
                    Symbol::Terminal(1),
                    Symbol::Nonterminal(0),
                ],
            }, // E -> E + E
            Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Nonterminal(0),
                    Symbol::Terminal(2),
                    Symbol::Nonterminal(0),
                ],
            }, // E -> E * E
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }, // E -> id
        ],
        0,
        vec![tdef(0, "id"), tdef(1, "+"), tdef(2, "*")],
    );
    let parser = build_parser(&gdef);

    // "id" → accepted
    assert!(accepts(&parser, &[0]), "\"id\" should be accepted");

    // "id + id * id" → accepted (ambiguous: (id+id)*id or id+(id*id))
    assert!(
        accepts(&parser, &[0, 1, 0, 2, 0]),
        "\"id+id*id\" should be accepted (ambiguous)"
    );

    // "id + id" → accepted
    assert!(
        accepts(&parser, &[0, 1, 0]),
        "\"id+id\" should be accepted"
    );

    // "id * id" → accepted
    assert!(
        accepts(&parser, &[0, 2, 0]),
        "\"id*id\" should be accepted"
    );

    // "id +" → rejected (incomplete)
    assert!(
        !accepts(&parser, &[0, 1]),
        "\"id+\" should be rejected (incomplete)"
    );

    // "id + + id" → rejected
    assert!(
        !accepts(&parser, &[0, 1, 1, 0]),
        "\"id++id\" should be rejected"
    );

    // "" → rejected
    assert!(!accepts(&parser, &[]), "empty should be rejected");

    // Determinism: same input produces same result
    let input = &[0, 1, 0, 2, 0];
    let r1 = accepts(&parser, input);
    let r2 = accepts(&parser, input);
    assert_eq!(r1, r2, "parser should be deterministic for same input");
}

/// Ported from `test_hidden_right_recursion`.
///
/// Grammar: S -> a S B | b, B -> epsilon
/// S -> a S B is effectively S -> a S (because B is nullable).
/// This is right-recursive with a hidden nullable suffix.
#[test]
fn test_hidden_right_recursion() {
    // S -> a S B | b
    // B -> epsilon
    // NT 0 = S, NT 1 = B
    // T 0 = a, T 1 = b
    let gdef = make_grammar(
        vec![
            Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Nonterminal(0),
                    Symbol::Nonterminal(1),
                ],
            }, // S -> a S B
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(1)],
            }, // S -> b
            Rule {
                lhs: 1,
                rhs: vec![],
            }, // B -> epsilon
        ],
        0,
        vec![tdef(0, "a"), tdef(1, "b")],
    );
    let parser = build_parser(&gdef);

    // "b" → accepted (S -> b)
    assert!(accepts(&parser, &[1]), "\"b\" should be accepted");

    // "ab" → accepted (S -> a S B -> a b ε)
    assert!(
        accepts(&parser, &[0, 1]),
        "\"ab\" should be accepted"
    );

    // "aab" → accepted
    assert!(
        accepts(&parser, &[0, 0, 1]),
        "\"aab\" should be accepted"
    );

    // "aaab" → accepted
    assert!(
        accepts(&parser, &[0, 0, 0, 1]),
        "\"aaab\" should be accepted"
    );

    // "a" → rejected (needs 'b')
    assert!(
        !accepts(&parser, &[0]),
        "\"a\" should be rejected (needs 'b')"
    );

    // "ba" → rejected
    assert!(
        !accepts(&parser, &[1, 0]),
        "\"ba\" should be rejected"
    );
}

/// Ported from `test_single_terminal_production`.
///
/// Grammar: S -> x
/// The simplest possible grammar — a single terminal production.
#[test]
fn test_single_terminal_production() {
    // S -> x
    // NT 0 = S
    // T 0 = x
    let gdef = make_grammar(
        vec![Rule {
            lhs: 0,
            rhs: vec![Symbol::Terminal(0)],
        }],
        0,
        vec![tdef(0, "x")],
    );
    let parser = build_parser(&gdef);

    // "x" → accepted
    assert!(accepts(&parser, &[0]), "\"x\" should be accepted");

    // "" → rejected
    assert!(!accepts(&parser, &[]), "empty should be rejected");
}
