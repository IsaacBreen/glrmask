//! GLR parser.
//!
//! A Generalized LR parser that operates on the GLR parse table.
//! Uses a list-of-stacks representation for simplicity during compilation.
//! The runtime uses an efficient GSS (see `runtime/gss.rs`).

use std::collections::{BTreeSet, VecDeque};

use super::grammar::EOF;
use super::table::{Action, GlrTable};
use crate::compiler::grammar_def::TerminalId;

/// GLR parser backed by an SLR(1) table.
///
/// Handles ambiguous grammars by maintaining multiple parse stacks.
#[allow(dead_code)]
pub struct GlrParser {
    pub table: GlrTable,
}

#[allow(dead_code)]
impl GlrParser {
    /// Create a new parser from a table.
    pub fn new(table: GlrTable) -> Self {
        Self { table }
    }

    /// Parse a sequence of terminal IDs. Returns `true` if the input is accepted.
    pub fn parse(&self, input: &[TerminalId]) -> bool {
        let mut stacks: Vec<Vec<u32>> = vec![vec![0]];

        for &token in input {
            let (new_stacks, _accepted) = self.step(&stacks, token);
            stacks = new_stacks;
            if stacks.is_empty() {
                return false;
            }
        }

        // Check for accept on EOF.
        let (_new_stacks, accepted) = self.step(&stacks, EOF);
        accepted
    }

    /// Can the parser continue with this terminal from at least one stack?
    pub fn can_shift(&self, stacks: &[Vec<u32>], token: TerminalId) -> bool {
        let (new_stacks, accepted) = self.step(stacks, token);
        accepted || !new_stacks.is_empty()
    }

    /// Process one token across all active stacks.
    ///
    /// Returns (new_stacks_after_shift, did_any_accept).
    /// Reduces are processed exhaustively before shifts.
    pub fn step(&self, stacks: &[Vec<u32>], token: TerminalId) -> (Vec<Vec<u32>>, bool) {
        let mut shifted: Vec<Vec<u32>> = Vec::new();
        let mut accepted = false;

        // Worklist of stacks to process reduces on.
        let mut worklist: VecDeque<Vec<u32>> = stacks.iter().cloned().collect();
        let mut seen: BTreeSet<Vec<u32>> = stacks.iter().cloned().collect();

        while let Some(stack) = worklist.pop_front() {
            let state = *stack.last().unwrap();
            for action in self.table.actions(state, token) {
                match action {
                    Action::Shift(next) => {
                        let mut new_stack = stack.clone();
                        new_stack.push(*next);
                        shifted.push(new_stack);
                    }
                    Action::Reduce(rule_idx) => {
                        let rule = &self.table.rules[*rule_idx as usize];
                        let pop_count = rule.rhs.len();
                        if stack.len() <= pop_count {
                            continue; // Stack underflow — dead path.
                        }
                        let mut new_stack = stack[..stack.len() - pop_count].to_vec();
                        let exposed = *new_stack.last().unwrap();
                        if let Some(goto_state) = self.table.goto_target(exposed, rule.lhs) {
                            new_stack.push(goto_state);
                            if seen.insert(new_stack.clone()) {
                                worklist.push_back(new_stack);
                            }
                        }
                    }
                    Action::Accept => {
                        accepted = true;
                    }
                }
            }
        }

        (shifted, accepted)
    }

    /// Enumerate all terminals that are valid continuations from the given stacks.
    pub fn valid_terminals(&self, stacks: &[Vec<u32>]) -> Vec<TerminalId> {
        let mut result = BTreeSet::new();
        for t in 0..self.table.num_terminals {
            if self.can_shift(stacks, t) {
                result.insert(t);
            }
        }
        // Also check EOF.
        if self.can_shift(stacks, EOF) {
            result.insert(EOF);
        }
        result.into_iter().collect()
    }
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::grammar::GlrGrammar;
    use crate::compiler::grammar_def::tests::*;
    use crate::compiler::grammar_def::{GrammarDef, Rule, Symbol, TerminalDef};

    fn build_parser(gdef: &GrammarDef) -> GlrParser {
        let gg = GlrGrammar::from_grammar_def(gdef);
        let table = GlrTable::build(&gg);
        GlrParser::new(table)
    }

    #[test]
    fn test_parse_simple_ab() {
        let gdef = simple_ab_grammar(); // S → a b
        let parser = build_parser(&gdef);
        assert!(parser.parse(&[0, 1])); // "a b" — accepted
        assert!(!parser.parse(&[0])); // "a" alone — rejected
        assert!(!parser.parse(&[1, 0])); // "b a" — rejected
        assert!(!parser.parse(&[])); // empty — rejected
    }

    #[test]
    fn test_parse_choice() {
        let gdef = choice_grammar(); // S → a | b
        let parser = build_parser(&gdef);
        assert!(parser.parse(&[0])); // "a"
        assert!(parser.parse(&[1])); // "b"
        assert!(!parser.parse(&[0, 1])); // "a b" — too long
        assert!(!parser.parse(&[])); // empty
    }

    #[test]
    fn test_parse_two_nt() {
        let gdef = two_nt_grammar(); // S → A b, A → a
        let parser = build_parser(&gdef);
        assert!(parser.parse(&[0, 1])); // "a b"
        assert!(!parser.parse(&[0])); // "a" alone
        assert!(!parser.parse(&[1])); // "b" alone
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
        assert!(parser.parse(&[0])); // "a"
        assert!(parser.parse(&[0, 1, 0])); // "a + a"
        assert!(parser.parse(&[0, 1, 0, 1, 0])); // "a + a + a"
        assert!(!parser.parse(&[1])); // "+" alone
        assert!(!parser.parse(&[0, 1])); // "a +"
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
        assert!(parser.parse(&[])); // empty (S → A → ε)
        assert!(parser.parse(&[0])); // "a" (S → A → a)
        assert!(!parser.parse(&[0, 0])); // "a a" — too long
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
}
