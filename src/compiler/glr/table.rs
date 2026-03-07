//! GLR parse table construction via SLR(1).
//!
//! Builds LR(0) item sets, then derives SLR(1) actions using FOLLOW sets.
//! Shift/reduce and reduce/reduce conflicts are retained (GLR).
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::analysis::{EOF, GLRGrammar};
use crate::compiler::grammar_def::{NonterminalId, Rule, Symbol, TerminalID};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// An action in the GLR parse table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    /// Shift to state.
    Shift(u32),
    /// Reduce by rule index.
    Reduce(u32),
    /// Accept the input.
    Accept,
}

/// A GLR parse table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLRTable {
    /// `action[state]` = map from terminal → set of actions (GLR: multiple actions allowed).
    pub action: Vec<BTreeMap<TerminalID, Vec<Action>>>,
    /// `goto[state][nonterminal]` = target state.
    pub goto: Vec<BTreeMap<NonterminalId, u32>>,
    /// Number of states.
    pub num_states: u32,
    /// Number of terminals.
    pub num_terminals: u32,
    /// Number of rules.
    pub num_rules: u32,
    /// Copy of the rules (for RHS length and LHS during reduce).
    pub rules: Vec<Rule>,
}

impl GLRTable {
    /// Build SLR(1) parse tables from a [`GLRGrammar`].
    pub fn build(grammar: &GLRGrammar) -> Self {
        unimplemented!()
    }

    /// Get all actions for (state, terminal). Returns empty slice if none.
    pub fn actions(&self, state: u32, terminal: TerminalID) -> &[Action] {
        unimplemented!()
    }

    /// Get goto target for (state, nonterminal). Returns `None` for error.
    pub fn goto_target(&self, state: u32, nt: NonterminalId) -> Option<u32> {
        unimplemented!()
    }
}

// ---------------------------------------------------------------------------
// LR(0) items
// ---------------------------------------------------------------------------

/// An LR(0) item: (rule_index, dot_position).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Item {
    rule: u32,
    dot: u32,
}

impl Item {
    fn new(rule: u32, dot: u32) -> Self {
        unimplemented!()
    }

    /// The symbol just after the dot, or `None` if dot is at the end.
    fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {
        let rhs = &rules[self.rule as usize].rhs;
        rhs.get(self.dot as usize)
    }
}

// ---------------------------------------------------------------------------
// LR(0) item-set construction
// ---------------------------------------------------------------------------

/// Compute the closure of an item set.
fn closure(items: &BTreeSet<Item>, rules: &[Rule]) -> BTreeSet<Item> {
    let mut result = items.clone();
    let mut queue: VecDeque<Item> = items.iter().copied().collect();

    while let Some(item) = queue.pop_front() {
        if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
            // Add initial items for all rules with this NT as LHS.
            for (i, r) in rules.iter().enumerate() {
                if r.lhs == *nt {
                    let new_item = Item::new(i as u32, 0);
                    if result.insert(new_item) {
                        queue.push_back(new_item);
                    }
                }
            }
        }
    }
    result
}

/// Compute GOTO(item_set, symbol).
fn goto_set(items: &BTreeSet<Item>, sym: &Symbol, rules: &[Rule]) -> BTreeSet<Item> {
    let mut kernel = BTreeSet::new();
    for item in items {
        if item.next_symbol(rules) == Some(sym) {
            kernel.insert(Item::new(item.rule, item.dot + 1));
        }
    }
    closure(&kernel, rules)
}

/// Build all LR(0) canonical item sets and their transitions.
///
/// Returns:
/// - `item_sets[state_id]` = the canonical item set.
/// - `transitions[state_id]` = map from symbol → target state_id.
fn build_lr0_item_sets(grammar: &GLRGrammar) -> (Vec<BTreeSet<Item>>, Vec<BTreeMap<Symbol, u32>>) {
    let rules = &grammar.rules;

    // State 0: closure of [S' → · S].
    let initial = {
        let mut s = BTreeSet::new();
        s.insert(Item::new(0, 0)); // augmented start rule
        closure(&s, rules)
    };

    let mut item_sets: Vec<BTreeSet<Item>> = vec![initial.clone()];
    let mut transitions: Vec<BTreeMap<Symbol, u32>> = vec![BTreeMap::new()];
    let mut set_to_id: FxHashMap<Vec<Item>, u32> = FxHashMap::default();
    set_to_id.insert(initial.iter().copied().collect(), 0);

    let mut queue: VecDeque<u32> = VecDeque::new();
    queue.push_back(0);

    while let Some(sid) = queue.pop_front() {
        // Collect all symbols that appear after a dot.
        let symbols: BTreeSet<Symbol> = item_sets[sid as usize]
            .iter()
            .filter_map(|item| item.next_symbol(rules).cloned())
            .collect();

        for sym in &symbols {
            let target_set = goto_set(&item_sets[sid as usize], sym, rules);
            if target_set.is_empty() {
                continue;
            }

            let key: Vec<Item> = target_set.iter().copied().collect();
            let target_id = if let Some(&id) = set_to_id.get(&key) {
                id
            } else {
                let id = item_sets.len() as u32;
                set_to_id.insert(key, id);
                item_sets.push(target_set);
                transitions.push(BTreeMap::new());
                queue.push_back(id);
                id
            };

            transitions[sid as usize].insert(sym.clone(), target_id);
        }
    }

    (item_sets, transitions)
}

// ---------------------------------------------------------------------------
// SLR(1) table
// ---------------------------------------------------------------------------

fn build_slr1_table(
    grammar: &GLRGrammar,
    item_sets: &[BTreeSet<Item>],
    transitions: &[BTreeMap<Symbol, u32>],
) -> GLRTable {
        unimplemented!()
    }

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar_def::GrammarDef;
    use crate::compiler::grammar_def::tests::*;

    #[test]
    fn test_table_simple_ab() {
        // S → a b.  Should produce 4 states: s0, s1 (after 'a'), s2 (after 'b'), s3 (accept).
        let gdef = simple_ab_grammar();
        let gg = GLRGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        assert!(table.num_states >= 3);

        // State 0: shift on 'a' (terminal 0).
        let a0 = table.actions(0, 0);
        assert!(a0.iter().any(|a| matches!(a, Action::Shift(_))));

        // After shift on 'a', shift on 'b'.
        let shift_state = match &a0[0] {
            Action::Shift(s) => *s,
            _ => panic!("expected shift"),
        };
        let a1 = table.actions(shift_state, 1);
        assert!(a1.iter().any(|a| matches!(a, Action::Shift(_))));
    }

    #[test]
    fn test_table_choice() {
        // S → a | b.  State 0 should shift on both 'a' and 'b'.
        let gdef = choice_grammar();
        let gg = GLRGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        assert!(!table.actions(0, 0).is_empty()); // 'a'
        assert!(!table.actions(0, 1).is_empty()); // 'b'
    }

    #[test]
    fn test_table_accept() {
        // S → a.  After reading 'a', reduce to S, then accept on $.
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![crate::compiler::grammar_def::TerminalDef {
                id: 0,
                name: "a".into(),
                pattern: "a".into(),
            }],
        };
        let gg = GLRGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        // Walk: s0 --shift(a)--> s1 --reduce(S→a)--> s0 --goto(S)--> s2 --accept($)
        // Check that after shifting 'a', there's a reduce on $.
        let a0 = table.actions(0, 0);
        let s1 = match &a0[0] {
            Action::Shift(s) => *s,
            _ => panic!(),
        };
        let a1 = table.actions(s1, EOF);
        assert!(a1.iter().any(|a| matches!(a, Action::Reduce(_))));
    }

    #[test]
    fn test_table_two_nt() {
        // S → A b, A → a.
        let gdef = two_nt_grammar();
        let gg = GLRGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        // State 0 should have shift on 'a'.
        assert!(!table.actions(0, 0).is_empty());
    }

    #[test]
    fn test_table_ambiguous() {
        // E → E + E | a.  Should produce a table with shift/reduce conflicts.
        let gdef = GrammarDef {
            rules: vec![
                // r1: E → E + E
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Nonterminal(0),
                        Symbol::Terminal(1),
                        Symbol::Nonterminal(0),
                    ],
                },
                // r2: E → a
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            start: 0,
            terminals: vec![
                crate::compiler::grammar_def::TerminalDef {
                    id: 0,
                    name: "a".into(),
                    pattern: "a".into(),
                },
                crate::compiler::grammar_def::TerminalDef {
                    id: 1,
                    name: "+".into(),
                    pattern: "\\+".into(),
                },
            ],
        };
        let gg = GLRGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        // The table should have been built successfully (GLR handles conflicts).
        assert!(table.num_states > 0);

        // Find a state that has both shift and reduce for '+'.
        let has_conflict = (0..table.num_states).any(|s| {
            let acts = table.actions(s, 1); // '+'
            let has_shift = acts.iter().any(|a| matches!(a, Action::Shift(_)));
            let has_reduce = acts.iter().any(|a| matches!(a, Action::Reduce(_)));
            has_shift && has_reduce
        });
        assert!(
            has_conflict,
            "Expected shift/reduce conflict for ambiguous grammar"
        );
    }
}
