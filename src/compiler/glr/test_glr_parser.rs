//! Additional GLR parser regression tests adapted from earlier internal test
//! suites.
//!
//! This file keeps the cases that map cleanly onto the current parser API.
//! Historical cases that depended on removed validation, reachability, or
//! stats helpers are intentionally omitted.

use super::analysis::AnalyzedGrammar;
use super::parser::{stacks_finished, GLRParser};
use super::table::{Action, GLRTable};
use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal, TerminalID};

fn literal_terminal(id: u32, name: &str) -> Terminal {
    Terminal::Literal {
        id,
        bytes: name.as_bytes().to_vec(),
    }
}

fn grammar_definition(rules: Vec<Rule>, start: u32, terminals: Vec<Terminal>) -> GrammarDef {
    GrammarDef {
        rules,
        start,
        terminals,
        ..Default::default()
    }
}

fn parser_for(grammar_def: &GrammarDef) -> GLRParser {
    let grammar = AnalyzedGrammar::from_grammar_def(grammar_def);
    let table = GLRTable::build(&grammar);
    GLRParser::new(table)
}

fn raw_o9788_n10_grammar() -> GrammarDef {
    const T_A: u32 = 0;
    const T_C: u32 = 1;
    const T_B: u32 = 2;
    const T_S: u32 = 3;

    const NT_ITEM: u32 = 0;
    const NT_OPT: u32 = 1;
    const NT_REPEAT2: u32 = 2;
    const NT_REPEAT4: u32 = 3;
    const NT_REPEAT8: u32 = 4;
    const NT_REPEAT10: u32 = 5;
    const NT_START: u32 = 6;

    grammar_definition(
        vec![
            Rule {
                lhs: NT_ITEM,
                rhs: vec![Symbol::Terminal(T_A), Symbol::Terminal(T_B)],
            },
            Rule {
                lhs: NT_ITEM,
                rhs: vec![
                    Symbol::Terminal(T_A),
                    Symbol::Nonterminal(NT_OPT),
                    Symbol::Terminal(T_B),
                ],
            },
            Rule {
                lhs: NT_OPT,
                rhs: vec![Symbol::Terminal(T_C)],
            },
            Rule {
                lhs: NT_REPEAT2,
                rhs: vec![Symbol::Terminal(T_S), Symbol::Nonterminal(NT_ITEM)],
            },
            Rule {
                lhs: NT_REPEAT2,
                rhs: vec![
                    Symbol::Terminal(T_S),
                    Symbol::Nonterminal(NT_ITEM),
                    Symbol::Terminal(T_S),
                    Symbol::Nonterminal(NT_ITEM),
                ],
            },
            Rule {
                lhs: NT_REPEAT4,
                rhs: vec![Symbol::Nonterminal(NT_REPEAT2)],
            },
            Rule {
                lhs: NT_REPEAT4,
                rhs: vec![
                    Symbol::Nonterminal(NT_REPEAT2),
                    Symbol::Nonterminal(NT_REPEAT2),
                ],
            },
            Rule {
                lhs: NT_REPEAT8,
                rhs: vec![Symbol::Nonterminal(NT_REPEAT4)],
            },
            Rule {
                lhs: NT_REPEAT8,
                rhs: vec![
                    Symbol::Nonterminal(NT_REPEAT4),
                    Symbol::Nonterminal(NT_REPEAT4),
                ],
            },
            Rule {
                lhs: NT_REPEAT10,
                rhs: vec![Symbol::Nonterminal(NT_REPEAT8)],
            },
            Rule {
                lhs: NT_REPEAT10,
                rhs: vec![Symbol::Nonterminal(NT_REPEAT2)],
            },
            Rule {
                lhs: NT_REPEAT10,
                rhs: vec![
                    Symbol::Nonterminal(NT_REPEAT8),
                    Symbol::Nonterminal(NT_REPEAT2),
                ],
            },
            Rule {
                lhs: NT_START,
                rhs: vec![Symbol::Nonterminal(NT_ITEM)],
            },
            Rule {
                lhs: NT_START,
                rhs: vec![
                    Symbol::Nonterminal(NT_ITEM),
                    Symbol::Nonterminal(NT_REPEAT10),
                ],
            },
        ],
        NT_START,
        vec![
            literal_terminal(T_A, "a"),
            literal_terminal(T_C, "c"),
            literal_terminal(T_B, "b"),
            literal_terminal(T_S, "s"),
        ],
    )
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

fn assert_accepts(parser: &GLRParser, input: &[TerminalID], message: &str) {
    assert!(accepts(parser, input), "{message}");
}

fn assert_rejects(parser: &GLRParser, input: &[TerminalID], message: &str) {
    assert!(!accepts(parser, input), "{message}");
}

/// Repetition without EOF, left-recursive form.
///
/// Grammar: S -> S a | a  (left-recursive, single terminal 'a')
/// Tests parsing various inputs without EOF using can-continue semantics.
#[test]
fn test_left_recursive_repetition_accepts_without_eof() {
    // S -> S a | a
    // NT 0 = S
    // T 0 = a
    let grammar = grammar_definition(
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
        vec![literal_terminal(0, "a")],
    );
    let parser = parser_for(&grammar);

    assert_accepts(&parser, &[0], "\"a\" should be accepted (S -> a)");
    assert_accepts(
        &parser,
        &[0, 0, 0],
        "\"aaa\" should be accepted (S -> S a -> S a a -> a a a)",
    );
    assert_rejects(&parser, &[], "\"\" should NOT be accepted (S is not nullable)");
    assert!(
        can_continue(&parser, &[]),
        "parser should be able to continue from initial state"
    );
}

/// Repetition without EOF, right-recursive form.
///
/// Grammar: S -> S a | a, Other -> b
/// Tests that invalid token 'b' causes parse failure for the S language.
#[test]
fn test_left_recursive_repetition_rejects_other_branch_tokens() {
    // S -> S a | a, Other -> b
    // NT 0 = S, NT 1 = Other
    // T 0 = a, T 1 = b
    let grammar = grammar_definition(
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
        vec![literal_terminal(0, "a"), literal_terminal(1, "b")],
    );
    let parser = parser_for(&grammar);

    assert_rejects(&parser, &[1], "\"b\" should fail for S language");
    assert_rejects(&parser, &[0, 1], "\"ab\" should fail for S language");
    assert_accepts(&parser, &[0], "\"a\" should be accepted");
}

/// Minimal single-production grammar.
///
/// Grammar: S -> a eof
/// Tests the simplest possible grammar with explicit EOF terminal.
#[test]
fn test_explicit_eof_single_rule_grammar() {
    // S -> a eof
    // NT 0 = S
    // T 0 = a, T 1 = $
    let grammar = grammar_definition(
        vec![Rule {
            lhs: 0,
            rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
        }],
        0,
        vec![literal_terminal(0, "a"), literal_terminal(1, "$")],
    );
    let parser = parser_for(&grammar);

    assert_accepts(&parser, &[0, 1], "\"a$\" should be accepted");
    assert_rejects(&parser, &[1], "\"$\" should fail (wrong first token)");
    assert_rejects(&parser, &[0], "\"a\" alone should not be accepted (missing $)");
    assert_rejects(&parser, &[], "empty should not be accepted");
}

#[test]
fn test_raw_o9788_n10_manual_grammar_smoke() {
    let grammar = raw_o9788_n10_grammar();
    let parser = parser_for(&grammar);

    assert_accepts(
        &parser,
        &[0, 2],
        "manual raw grammar should accept the base item without going through Lark or Constraint",
    );
}

#[test]
fn test_raw_o9788_n10_keeps_separator_shift_after_ten_items() {
    let grammar = raw_o9788_n10_grammar();
    let mut current = parser_for(&grammar);

    let mut prefix = vec![0u32, 2u32];
    for _ in 0..9 {
        prefix.extend_from_slice(&[3u32, 0u32, 2u32]);
    }

    for (index, token) in prefix.into_iter().enumerate() {
        let (next, progressed) = current.step(token);
        assert!(progressed, "raw parser repro stopped early at prefix token index {index}");
        current = next;
    }

    assert!(
        current.valid_terminals().contains(&3),
        "after 10 items, the raw parser should still allow one more separator 's' for the 11th item"
    );
}

/// Simple parse-table generation and parse.
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
    let grammar = grammar_definition(
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
        vec![
            literal_terminal(0, "a"),
            literal_terminal(1, "b"),
            literal_terminal(2, "$"),
        ],
    );
    let parser = parser_for(&grammar);

    // "b$" → accepted (A -> b, S -> A $)
    assert_accepts(&parser, &[1, 2], "\"b$\" should be accepted");

    // "ba$" → accepted (A -> A a -> b a, S -> A $)
    assert_accepts(&parser, &[1, 0, 2], "\"ba$\" should be accepted");

    // "baa$" → accepted
    assert_accepts(&parser, &[1, 0, 0, 2], "\"baa$\" should be accepted");

    // "a$" → rejected (cannot start with 'a')
    assert_rejects(&parser, &[0, 2], "\"a$\" should be rejected");

    // "bb$" → rejected (two b's)
    assert_rejects(&parser, &[1, 1, 2], "\"bb$\" should be rejected");
}

/// Ambiguous arithmetic grammar.
///
/// Grammar: E -> E + E | E * E | id
/// This is ambiguous: id + id * id has two parses.
/// GLR should accept it.
#[test]
fn test_ambiguous_arithmetic() {
    // E -> E + E | E * E | id
    // NT 0 = E
    // T 0 = id, T 1 = +, T 2 = *
    let grammar = grammar_definition(
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
        vec![
            literal_terminal(0, "id"),
            literal_terminal(1, "+"),
            literal_terminal(2, "*"),
        ],
    );
    let parser = parser_for(&grammar);

    // "id" → accepted
    assert_accepts(&parser, &[0], "\"id\" should be accepted");

    // "id + id * id" → accepted (ambiguous: (id+id)*id or id+(id*id))
    assert_accepts(
        &parser,
        &[0, 1, 0, 2, 0],
        "\"id+id*id\" should be accepted (ambiguous)",
    );

    // "id + id" → accepted
    assert_accepts(&parser, &[0, 1, 0], "\"id+id\" should be accepted");

    // "id * id" → accepted
    assert_accepts(&parser, &[0, 2, 0], "\"id*id\" should be accepted");

    // "id +" → rejected (incomplete)
    assert_rejects(&parser, &[0, 1], "\"id+\" should be rejected (incomplete)");

    // "id + + id" → rejected
    assert_rejects(&parser, &[0, 1, 1, 0], "\"id++id\" should be rejected");

    // "" → rejected
    assert_rejects(&parser, &[], "empty should be rejected");

    // Determinism: same input produces same result
    let input = &[0, 1, 0, 2, 0];
    let r1 = accepts(&parser, input);
    let r2 = accepts(&parser, input);
    assert_eq!(r1, r2, "parser should be deterministic for same input");
}

/// Hidden right-recursion grammar.
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
    let grammar = grammar_definition(
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
        vec![literal_terminal(0, "a"), literal_terminal(1, "b")],
    );
    let parser = parser_for(&grammar);

    // "b" → accepted (S -> b)
    assert_accepts(&parser, &[1], "\"b\" should be accepted");

    // "ab" → accepted (S -> a S B -> a b ε)
    assert_accepts(&parser, &[0, 1], "\"ab\" should be accepted");

    // "aab" → accepted
    assert_accepts(&parser, &[0, 0, 1], "\"aab\" should be accepted");

    // "aaab" → accepted
    assert_accepts(&parser, &[0, 0, 0, 1], "\"aaab\" should be accepted");

    // "a" → rejected (needs 'b')
    assert_rejects(&parser, &[0], "\"a\" should be rejected (needs 'b')");

    // "ba" → rejected
    assert_rejects(&parser, &[1, 0], "\"ba\" should be rejected");
}

/// Single-terminal production grammar.
///
/// Grammar: S -> x
/// The simplest possible grammar — a single terminal production.
#[test]
fn test_single_terminal_production() {
    // S -> x
    // NT 0 = S
    // T 0 = x
    let grammar = grammar_definition(
        vec![Rule {
            lhs: 0,
            rhs: vec![Symbol::Terminal(0)],
        }],
        0,
        vec![literal_terminal(0, "x")],
    );
    let parser = parser_for(&grammar);

    assert_accepts(&parser, &[0], "\"x\" should be accepted");
    assert_rejects(&parser, &[], "empty should be rejected");
}

fn assert_no_splits(table: &GLRTable) {
    for row in &table.action {
        for action in row.values() {
            if matches!(action, Action::Split { .. }) {
                panic!("Found split action in table: {:?}", action);
            }
        }
    }
}

#[test]
fn test_glrm_exact_repetition_determinism() {
    let grammar_str = "start S; nt S ::= \"a\"{16,16} \"$\";";
    let named = crate::grammar::glrm::from_glrm(grammar_str).unwrap();
    let factored = crate::grammar::factoring::factor_named_grammar(named);
    let gdef = crate::grammar::ast::lower(&factored).unwrap();
    let analyzed = AnalyzedGrammar::from_grammar_def(&gdef);
    let table = GLRTable::build(&analyzed);
    assert_no_splits(&table);
}

#[test]
fn test_glrm_up_to_repetition_determinism() {
    // Note: this test fails with the default RepeatTreeShape::Balanced because 
    // balanced up-to repetitions are inherently ambiguous in LR(1).
    // Use GLRMASK_REPEAT_TREE_SHAPE=Right to make it deterministic.
    let grammar_str = "start S; nt S ::= \"a\"{0,16} \"$\";";
    let named = crate::grammar::glrm::from_glrm(grammar_str).unwrap();
    let factored = crate::grammar::factoring::factor_named_grammar(named);
    let gdef = crate::grammar::ast::lower(&factored).unwrap();
    let analyzed = AnalyzedGrammar::from_grammar_def(&gdef);
    let table = GLRTable::build(&analyzed);
    assert_no_splits(&table);
}
