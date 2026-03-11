#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::analysis::{EOF, AnalyzedGrammar};
use crate::compiler::grammar_def::{NonterminalID, Rule, Symbol, TerminalID};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Shift(u32),
    Reduce(u32),
    Split {
        shift: Option<u32>,
        reduces: Vec<u32>,
        accept: bool,
    },
    Accept,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLRTable {
    pub action: Vec<BTreeMap<TerminalID, Action>>,
    pub goto: Vec<BTreeMap<NonterminalID, u32>>,
    pub num_states: u32,
    pub num_terminals: u32,
    pub num_rules: u32,
    pub rules: Vec<Rule>,
}

impl GLRTable {
    pub fn build(grammar: &AnalyzedGrammar) -> Self {
        let (item_sets, transitions) = build_lr0_item_sets(grammar);
        let mut table = build_slr1_table(grammar, &item_sets, &transitions);
        table.merge_identical_rows();
        table
    }

    pub fn action(&self, state: u32, terminal: TerminalID) -> Option<&Action> {
        self.action
            .get(state as usize)
            .and_then(|by_terminal| by_terminal.get(&terminal))
    }

    pub fn goto_target(&self, state: u32, nt: NonterminalID) -> Option<u32> {
        self.goto
            .get(state as usize)
            .and_then(|by_nt| by_nt.get(&nt).copied())
    }

    /// Merge states with identical (action, goto) rows.
    /// Iterates until no more merges are possible, since remapping targets
    /// can reveal new equivalences.
    fn merge_identical_rows(&mut self) {
        fn remap_action(action: &mut Action, mapping: &[u32]) {
            match action {
                Action::Shift(target) => *target = mapping[*target as usize],
                Action::Split { shift, .. } => {
                    if let Some(target) = shift {
                        *target = mapping[*target as usize];
                    }
                }
                _ => {}
            }
        }

        loop {
            // Build signature -> first representative state
            let mut sig_to_rep: FxHashMap<String, u32> = FxHashMap::default();
            let mut remap: Vec<u32> = (0..self.num_states).collect();
            let mut changed = false;

            for state in 0..self.num_states as usize {
                let sig = format!(
                    "{:?}|{:?}",
                    self.action[state], self.goto[state]
                );
                let rep = *sig_to_rep.entry(sig).or_insert(state as u32);
                if rep != state as u32 {
                    remap[state] = rep;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Build old_to_new: compose remap (merge) with sequential renumbering
            let mut new_id = 0u32;
            let mut rep_to_new: FxHashMap<u32, u32> = FxHashMap::default();
            let mut kept: Vec<u32> = Vec::new();
            for state in 0..self.num_states as usize {
                if remap[state] == state as u32 {
                    rep_to_new.insert(state as u32, new_id);
                    kept.push(state as u32);
                    new_id += 1;
                }
            }
            let mapping: Vec<u32> = (0..self.num_states as usize)
                .map(|s| rep_to_new[&remap[s]])
                .collect();

            // Extract representative rows and remap all state references
            let new_action: Vec<_> = kept
                .iter()
                .map(|&s| {
                    self.action[s as usize]
                        .iter()
                        .map(|(&tid, action)| {
                            let mut a = action.clone();
                            remap_action(&mut a, &mapping);
                            (tid, a)
                        })
                        .collect()
                })
                .collect();
            let new_goto: Vec<_> = kept
                .iter()
                .map(|&s| {
                    self.goto[s as usize]
                        .iter()
                        .map(|(&nt, &target)| (nt, mapping[target as usize]))
                        .collect()
                })
                .collect();

            self.action = new_action;
            self.goto = new_goto;
            self.num_states = kept.len() as u32;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Item {
    rule: u32,
    dot: u32,
}

impl Item {
    fn new(rule: u32, dot: u32) -> Self {
        Self { rule, dot }
    }

    fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {
        let rhs = &rules[self.rule as usize].rhs;
        rhs.get(self.dot as usize)
    }
}

fn closure(items: &BTreeSet<Item>, rules: &[Rule]) -> BTreeSet<Item> {
    let mut result = items.clone();
    let mut queue: VecDeque<Item> = items.iter().copied().collect();

    while let Some(item) = queue.pop_front() {
        if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
            
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

fn goto_set(items: &BTreeSet<Item>, sym: &Symbol, rules: &[Rule]) -> BTreeSet<Item> {
    let mut kernel = BTreeSet::new();
    for item in items {
        if item.next_symbol(rules) == Some(sym) {
            kernel.insert(Item::new(item.rule, item.dot + 1));
        }
    }
    closure(&kernel, rules)
}

fn build_lr0_item_sets(grammar: &AnalyzedGrammar) -> (Vec<BTreeSet<Item>>, Vec<BTreeMap<Symbol, u32>>) {
    let rules = &grammar.rules;

    let initial = {
        let mut s = BTreeSet::new();
        s.insert(Item::new(0, 0)); 
        closure(&s, rules)
    };

    let mut item_sets: Vec<BTreeSet<Item>> = vec![initial.clone()];
    let mut transitions: Vec<BTreeMap<Symbol, u32>> = vec![BTreeMap::new()];
    let mut set_to_id: FxHashMap<Vec<Item>, u32> = FxHashMap::default();
    set_to_id.insert(initial.iter().copied().collect(), 0);

    let mut queue: VecDeque<u32> = VecDeque::new();
    queue.push_back(0);

    while let Some(sid) = queue.pop_front() {
        
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

fn build_slr1_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[BTreeSet<Item>],
    transitions: &[BTreeMap<Symbol, u32>],
) -> GLRTable {
    #[derive(Default)]
    struct PendingAction {
        shift: Option<u32>,
        reduces: Vec<u32>,
        accept: bool,
    }

    impl PendingAction {
        fn push_shift(&mut self, target: u32) {
            match self.shift {
                Some(existing) => debug_assert_eq!(existing, target),
                None => self.shift = Some(target),
            }
        }

        fn push_reduce(&mut self, rule_id: u32) {
            self.reduces.push(rule_id);
        }

        fn push_accept(&mut self) {
            self.accept = true;
        }

        fn finish(mut self) -> Action {
            self.reduces.sort_unstable();
            self.reduces.dedup();
            match (self.shift, self.reduces.len(), self.accept) {
                (Some(target), 0, false) => Action::Shift(target),
                (None, 1, false) => Action::Reduce(self.reduces[0]),
                (None, 0, true) => Action::Accept,
                (shift, _, accept) => Action::Split {
                    shift,
                    reduces: self.reduces,
                    accept,
                },
            }
        }
    }

    let mut pending = std::iter::repeat_with(BTreeMap::<TerminalID, PendingAction>::new)
        .take(item_sets.len())
        .collect::<Vec<_>>();
    let mut goto = vec![BTreeMap::<NonterminalID, u32>::new(); item_sets.len()];

    for (state_id, items) in item_sets.iter().enumerate() {
        for (symbol, &target) in &transitions[state_id] {
            match symbol {
                Symbol::Terminal(terminal) => {
                    pending[state_id]
                        .entry(*terminal)
                        .or_default()
                        .push_shift(target);
                }
                Symbol::Nonterminal(nonterminal) => {
                    goto[state_id].insert(*nonterminal, target);
                }
            }
        }

        for item in items {
            let rule = &grammar.rules[item.rule as usize];
            if item.dot as usize != rule.rhs.len() {
                continue;
            }

            if item.rule == 0 {
                pending[state_id].entry(EOF).or_default().push_accept();
                continue;
            }

            for &lookahead in &grammar.follow[rule.lhs as usize] {
                pending[state_id]
                    .entry(lookahead)
                    .or_default()
                    .push_reduce(item.rule);
            }
        }
    }

    let action = pending
        .into_iter()
        .map(|by_terminal| {
            by_terminal
                .into_iter()
                .map(|(terminal, pending)| (terminal, pending.finish()))
                .collect()
        })
        .collect();

    GLRTable {
        action,
        goto,
        num_states: item_sets.len() as u32,
        num_terminals: grammar.num_terminals,
        num_rules: grammar.rules.len() as u32,
        rules: grammar.rules.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar_def::GrammarDef;
    use crate::compiler::grammar_def::tests::*;

    #[test]
    fn test_table_simple_ab() {
        
        let gdef = simple_ab_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        assert!(table.num_states >= 3);

        let a0 = table.action(0, 0);
        assert!(matches!(a0, Some(Action::Shift(_))));

        let shift_state = match a0 {
            Some(Action::Shift(s)) => *s,
            _ => panic!("expected shift"),
        };
        let a1 = table.action(shift_state, 1);
        assert!(matches!(a1, Some(Action::Shift(_))));
    }

    #[test]
    fn test_table_choice() {
        
        let gdef = choice_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        assert!(table.action(0, 0).is_some()); 
        assert!(table.action(0, 1).is_some()); 
    }

    #[test]
    fn test_table_accept() {
        
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![crate::compiler::grammar_def::Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        let a0 = table.action(0, 0);
        let s1 = match a0 {
            Some(Action::Shift(s)) => *s,
            _ => panic!(),
        };
        let a1 = table.action(s1, EOF);
        assert!(matches!(a1, Some(Action::Reduce(_))));
    }

    #[test]
    fn test_table_two_nt() {
        
        let gdef = two_nt_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        assert!(table.action(0, 0).is_some());
    }

    #[test]
    fn test_table_ambiguous() {
        
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
                crate::compiler::grammar_def::Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                crate::compiler::grammar_def::Terminal::Literal {
                    id: 1,
                    bytes: b"+".to_vec(),
                },
            ],
            ..Default::default()
        };
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        assert!(table.num_states > 0);

        let has_conflict = (0..table.num_states).any(|s| {
            matches!(table.action(s, 1), Some(Action::Split { shift: Some(_), .. }))
        });
        assert!(
            has_conflict,
            "Expected shift/reduce conflict for ambiguous grammar"
        );
    }
}
