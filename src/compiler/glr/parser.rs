//! GLR parser.
//!
//! A Generalized LR parser that operates on the GLR parse table.
//! Uses a list-of-stacks representation for simplicity during compilation.
//! The runtime uses an efficient GSS (see `runtime/gss.rs`).
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::analysis::EOF;
use super::table::{Action, GLRTable};
use crate::compiler::grammar::ast::TerminalId;
use crate::ds::leveled_gss::{LeveledGSS, Merge};

/// Maps tokenizer state ID → set of disallowed terminal IDs.
pub type TerminalsDisallowed = BTreeMap<u32, BTreeSet<u32>>;

/// Create a fresh (empty) `TerminalsDisallowed`.
pub(crate) fn terminals_disallowed_fresh() -> TerminalsDisallowed {
    unimplemented!()
}

impl Merge for TerminalsDisallowed {
    fn merge(&self, other: &Self) -> Self {
        unimplemented!()
    }
}

/// A GSS (Graph-Structured Stack) for the parser stack state.
pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;

/// A live parser state paired with a single parser instance.
pub struct GLRParserState<'a> {
    pub parser: &'a GLRParser,
    pub stack: ParserGSS,
}

/// GLR parser backed by an SLR(1) table.
///
/// Handles ambiguous grammars by maintaining multiple parse stacks.
#[allow(dead_code)]
pub struct GLRParser {
    pub table: GLRTable,
}

#[allow(dead_code)]
impl GLRParser {
    /// Create a new parser from a table.
    pub fn new(table: GLRTable) -> Self {
        unimplemented!()
    }

    /// Create a fresh parser state at the start of the parse.
    pub fn start(&self) -> GLRParserState<'_> {
        unimplemented!()
    }

    /// Can the parser continue with this terminal from at least one stack?
    pub fn can_shift(&self, stacks: &[Vec<u32>], token: TerminalId) -> bool {
        unimplemented!()
    }

    /// Process one token across all active stacks.
    ///
    /// Returns (new_stacks_after_shift, did_any_accept).
    /// Reduces are processed exhaustively before shifts.
    pub fn step(&self, stacks: &[Vec<u32>], token: TerminalId) -> (Vec<Vec<u32>>, bool) {
        unimplemented!()
    }

    /// Enumerate all terminals that are valid continuations from the given stacks.
    pub fn valid_terminals(&self, stacks: &[Vec<u32>]) -> Vec<TerminalId> {
        unimplemented!()
    }
}

impl<'a> GLRParserState<'a> {
    /// Check whether this parser state accepts a full input sequence.
    pub fn accepts(&self, input: &[TerminalId]) -> bool {
        unimplemented!()
    }
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::GLRGrammar;
    use crate::compiler::grammar::ast::tests::*;
    use crate::compiler::grammar::ast::{GrammarDef, Rule, Symbol, TerminalDef};

    fn build_parser(gdef: &GrammarDef) -> GLRParser {
        let gg = GLRGrammar::from_grammar_def(gdef);
        let table = GLRTable::build(&gg);
        GLRParser::new(table)
    }

    fn accepts(parser: &GLRParser, input: &[TerminalId]) -> bool {
        parser.start().accepts(input)
    }

    #[test]
    fn test_parse_simple_ab() {
        let gdef = simple_ab_grammar(); // S → a b
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0, 1])); // "a b" — accepted
        assert!(!accepts(&parser, &[0])); // "a" alone — rejected
        assert!(!accepts(&parser, &[1, 0])); // "b a" — rejected
        assert!(!accepts(&parser, &[])); // empty — rejected
    }

    #[test]
    fn test_parse_choice() {
        let gdef = choice_grammar(); // S → a | b
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0])); // "a"
        assert!(accepts(&parser, &[1])); // "b"
        assert!(!accepts(&parser, &[0, 1])); // "a b" — too long
        assert!(!accepts(&parser, &[])); // empty
    }

    #[test]
    fn test_parse_two_nt() {
        let gdef = two_nt_grammar(); // S → A b, A → a
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0, 1])); // "a b"
        assert!(!accepts(&parser, &[0])); // "a" alone
        assert!(!accepts(&parser, &[1])); // "b" alone
    }

    #[test]
    fn test_parse_ambiguous() {
        // E → E + E | a
        let gdef = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Nonterminal(0),
                        Symbol::Terminal(1),
                        Symbol::Nonterminal(0),
                    ],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            start: 0,
            terminals: vec![
                TerminalDef {
                    id: 0,
                    name: "a".into(),
                    pattern: "a".into(),
                },
                TerminalDef {
                    id: 1,
                    name: "+".into(),
                    pattern: "\\+".into(),
                },
            ],
        };
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0])); // "a"
        assert!(accepts(&parser, &[0, 1, 0])); // "a + a"
        assert!(accepts(&parser, &[0, 1, 0, 1, 0])); // "a + a + a"
        assert!(!accepts(&parser, &[1])); // "+" alone
        assert!(!accepts(&parser, &[0, 1])); // "a +"
    }

    #[test]
    fn test_parse_nullable() {
        // S → A, A → a | ε
        let gdef = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![],
                }, // ε
            ],
            start: 0,
            terminals: vec![TerminalDef {
                id: 0,
                name: "a".into(),
                pattern: "a".into(),
            }],
        };
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[])); // empty (S → A → ε)
        assert!(accepts(&parser, &[0])); // "a" (S → A → a)
        assert!(!accepts(&parser, &[0, 0])); // "a a" — too long
    }

    #[test]
    fn test_valid_terminals() {
        let gdef = simple_ab_grammar(); // S → a b
        let parser = build_parser(&gdef);
        let stacks = vec![vec![0u32]];
        let valid = parser.valid_terminals(&stacks);
        assert!(valid.contains(&0)); // 'a' is valid from start
        assert!(!valid.contains(&1)); // 'b' is not valid from start
    }

    // ---------------------------------------------------------------------------
    // Ported tests from old grammars2024/src/glr/tests.rs
    // ---------------------------------------------------------------------------

    /// Shorthand for building a TerminalDef.
    fn tdef(id: u32, name: &str) -> TerminalDef {
        TerminalDef { id, name: name.into(), pattern: name.into() }
    }

    #[test]
    fn test_ported_glr_left_recursive() {
        // Ported from old test_simple_parse_table_generation_and_parse.
        // Grammar: A → A a | b  (left-recursive, language = b a*)
        // NT 0=A; Terminal 0='a', 1='b'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(0)] }, // A → A a
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(1)] },                          // A → b
            ],
            start: 0,
            terminals: vec![tdef(0, "a"), tdef(1, "b")],
        };
        let parser = build_parser(&gdef);
        // Accepted: b a*
        assert!(accepts(&parser, &[1]),       "\"b\" accepted");
        assert!(accepts(&parser, &[1, 0]),    "\"ba\" accepted");
        assert!(accepts(&parser, &[1, 0, 0]), "\"baa\" accepted");
        // Rejected
        assert!(!accepts(&parser, &[0]),    "\"a\" rejected (must start with 'b')");
        assert!(!accepts(&parser, &[1, 1]), "\"bb\" rejected (two 'b's)");
    }

    #[test]
    fn test_ported_glr_right_recursive() {
        // Ported from old test_right_recursive_grammar_parse.
        // Grammar: S → a S | b  (right-recursive, language = a* b)
        // NT 0=S; Terminal 0='a', 1='b'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(0)] }, // S → a S
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(1)] },                          // S → b
            ],
            start: 0,
            terminals: vec![tdef(0, "a"), tdef(1, "b")],
        };
        let parser = build_parser(&gdef);
        // Accepted: a* b
        assert!(accepts(&parser, &[1]),          "\"b\" accepted");
        assert!(accepts(&parser, &[0, 1]),       "\"ab\" accepted");
        assert!(accepts(&parser, &[0, 0, 1]),    "\"aab\" accepted");
        assert!(accepts(&parser, &[0, 0, 0, 1]), "\"aaab\" accepted");
        // Rejected
        assert!(!accepts(&parser, &[0]),     "\"a\" rejected (must end in 'b')");
        assert!(!accepts(&parser, &[1, 0]),  "\"ba\" rejected");
        assert!(!accepts(&parser, &[1, 1]),  "\"bb\" rejected");
    }

    #[test]
    fn test_ported_glr_expression_grammar() {
        // Ported from old test_expression_parse_table_generation_and_parse.
        // Grammar: E → E + T | T,  T → T * F | F,  F → ( E ) | i
        // NT 0=E, 1=T, 2=F; Terminal 0='i', 1='+', 2='*', 3='(', 4=')'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(1), Symbol::Nonterminal(1)] }, // E → E + T
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },                                               // E → T
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(2), Symbol::Nonterminal(2)] }, // T → T * F
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(2)] },                                               // T → F
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(3), Symbol::Nonterminal(0), Symbol::Terminal(4)] },    // F → ( E )
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },                                                  // F → i
            ],
            start: 0,
            terminals: vec![tdef(0, "i"), tdef(1, "+"), tdef(2, "*"), tdef(3, "("), tdef(4, ")")],
        };
        let parser = build_parser(&gdef);
        // Accepted
        assert!(accepts(&parser, &[0]),                   "\"i\" accepted");
        assert!(accepts(&parser, &[0, 1, 0]),             "\"i+i\" accepted");
        assert!(accepts(&parser, &[0, 2, 0]),             "\"i*i\" accepted");
        assert!(accepts(&parser, &[0, 1, 0, 2, 0]),       "\"i+i*i\" accepted");
        assert!(accepts(&parser, &[3, 0, 1, 0, 4, 2, 0]), "\"(i+i)*i\" accepted");
        // Rejected
        assert!(!accepts(&parser, &[0, 1]),       "\"i+\" rejected (incomplete)");
        assert!(!accepts(&parser, &[0, 1, 1, 0]), "\"i++i\" rejected (invalid)");
        assert!(!accepts(&parser, &[]),           "\"\" rejected (empty)");
        assert!(!accepts(&parser, &[4]),          "\")\" rejected");
        assert!(!accepts(&parser, &[3, 0]),       "\"(i\" rejected (unclosed paren)");
    }

    #[test]
    fn test_ported_glr_reduce_reduce_conflict() {
        // Ported from old test_reduce_reduce_conflict.
        // Grammar: S → A | B,  A → x,  B → x
        // GLR accepts "x" despite the A→x / B→x reduce/reduce conflict.
        // NT 0=S, 1=A, 2=B; Terminal 0='x'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] }, // S → A
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(2)] }, // S → B
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },    // A → x
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },    // B → x
            ],
            start: 0,
            terminals: vec![tdef(0, "x")],
        };
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0]),  "\"x\" accepted despite reduce/reduce conflict");
        assert!(!accepts(&parser, &[]), "\"\" rejected");
    }

    #[test]
    fn test_ported_glr_epsilon_ambiguity() {
        // Ported from old test_epsilon_rules_ambiguity.
        // Grammar: S → A B,  A → x | ε,  B → x | ε
        // Language: {ε, x, xx}  (A and B each consume zero or one 'x')
        // NT 0=S, 1=A, 2=B; Terminal 0='x'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Nonterminal(2)] }, // S → A B
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },  // A → x
                Rule { lhs: 1, rhs: vec![] },                     // A → ε
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },  // B → x
                Rule { lhs: 2, rhs: vec![] },                     // B → ε
            ],
            start: 0,
            terminals: vec![tdef(0, "x")],
        };
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[]),       "\"\" accepted (A→ε, B→ε)");
        assert!(accepts(&parser, &[0]),      "\"x\" accepted (A→x,B→ε or A→ε,B→x)");
        assert!(accepts(&parser, &[0, 0]),   "\"xx\" accepted (A→x, B→x)");
        assert!(!accepts(&parser, &[0, 0, 0]), "\"xxx\" rejected");
    }

    #[test]
    fn test_ported_glr_highly_ambiguous() {
        // Ported from old test_highly_ambiguous_potentially_slow.
        // Grammar: S → S S | a  (Catalan-number ambiguity)
        // NT 0=S; Terminal 0='a'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Nonterminal(0)] }, // S → S S
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },                             // S → a
            ],
            start: 0,
            terminals: vec![tdef(0, "a")],
        };
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0]),       "\"a\" accepted");
        assert!(accepts(&parser, &[0, 0]),    "\"aa\" accepted");
        assert!(accepts(&parser, &[0, 0, 0]), "\"aaa\" accepted (many parse trees)");
        assert!(!accepts(&parser, &[]),       "\"\" rejected (S not nullable)");
    }

    #[test]
    fn test_ported_glr_nullable_before_terminal() {
        // Ported from old test_nullable_nonterminal_before_terminal.
        // Grammar: A → B c,  B → d | ε
        // NT 0=A, 1=B; Terminal 0='c', 1='d'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)] }, // A → B c
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(1)] }, // B → d
                Rule { lhs: 1, rhs: vec![] },                    // B → ε
            ],
            start: 0,
            terminals: vec![tdef(0, "c"), tdef(1, "d")],
        };
        let parser = build_parser(&gdef);
        // Accepted
        assert!(accepts(&parser, &[1, 0]), "\"dc\" accepted (A → d c)");
        assert!(accepts(&parser, &[0]),    "\"c\" accepted (A → ε c via B→ε)");
        // Rejected
        assert!(!accepts(&parser, &[1]),   "\"d\" rejected (missing 'c')");
        assert!(!accepts(&parser, &[]),    "\"\" rejected (A always requires 'c')");
    }

    #[test]
    fn test_ported_glr_ambiguous_dangling_else() {
        // Ported from old test_ambiguous_dangling_else.
        // Grammar: Stmt → if id then Stmt
        //                | if id then Stmt else Stmt
        //                | other
        // The input "if id then if id then other else other" is ambiguous —
        // the else can attach to the inner or outer if.  GLR accepts both.
        // NT 0=Stmt; Terminal 0='if', 1='id', 2='then', 3='else', 4='other'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1), Symbol::Terminal(2), Symbol::Nonterminal(0)] }, // if id then Stmt
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1), Symbol::Terminal(2), Symbol::Nonterminal(0), Symbol::Terminal(3), Symbol::Nonterminal(0)] }, // if id then Stmt else Stmt
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(4)] }, // Stmt → other
            ],
            start: 0,
            terminals: vec![tdef(0, "if"), tdef(1, "id"), tdef(2, "then"), tdef(3, "else"), tdef(4, "other")],
        };
        let parser = build_parser(&gdef);
        // Ambiguous dangling-else input: accepted through GLR's parallel exploration
        assert!(accepts(&parser, &[0, 1, 2, 0, 1, 2, 4, 3, 4]),
            "ambiguous 'if id then if id then other else other' should be accepted");
        // Simpler cases
        assert!(accepts(&parser, &[4]),          "\"other\" accepted");
        assert!(accepts(&parser, &[0, 1, 2, 4]), "\"if id then other\" accepted");
        assert!(!accepts(&parser, &[0, 1, 2]),   "\"if id then\" rejected (incomplete)");
    }
}
