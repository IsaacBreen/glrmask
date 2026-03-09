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
    Accept,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLRTable {
    pub action: Vec<BTreeMap<TerminalID, Vec<Action>>>,
    pub goto: Vec<BTreeMap<NonterminalID, u32>>,
    pub num_states: u32,
    pub num_terminals: u32,
    pub num_rules: u32,
    pub rules: Vec<Rule>,
}

impl GLRTable {
    pub fn build(grammar: &AnalyzedGrammar) -> Self {
        let (item_sets, transitions) = build_lr0_item_sets(grammar);
        build_slr1_table(grammar, &item_sets, &transitions)
    }

    pub fn actions(&self, state: u32, terminal: TerminalID) -> &[Action] {
        static EMPTY: [Action; 0] = [];
        self.action
            .get(state as usize)
            .and_then(|by_terminal| by_terminal.get(&terminal))
            .map(Vec::as_slice)
            .unwrap_or(&EMPTY)
    }

    pub fn goto_target(&self, state: u32, nt: NonterminalID) -> Option<u32> {
        self.goto
            .get(state as usize)
            .and_then(|by_nt| by_nt.get(&nt).copied())
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
    let mut action = vec![BTreeMap::<TerminalID, Vec<Action>>::new(); item_sets.len()];
    let mut goto = vec![BTreeMap::<NonterminalID, u32>::new(); item_sets.len()];

    for (state_id, items) in item_sets.iter().enumerate() {
        for (symbol, &target) in &transitions[state_id] {
            match symbol {
                Symbol::Terminal(terminal) => {
                    action[state_id]
                        .entry(*terminal)
                        .or_default()
                        .push(Action::Shift(target));
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
                action[state_id].entry(EOF).or_default().push(Action::Accept);
                continue;
            }

            for &lookahead in &grammar.follow[rule.lhs as usize] {
                action[state_id]
                    .entry(lookahead)
                    .or_default()
                    .push(Action::Reduce(item.rule));
            }
        }
    }

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

        let a0 = table.actions(0, 0);
        assert!(a0.iter().any(|a| matches!(a, Action::Shift(_))));

        let shift_state = match &a0[0] {
            Action::Shift(s) => *s,
            _ => panic!("expected shift"),
        };
        let a1 = table.actions(shift_state, 1);
        assert!(a1.iter().any(|a| matches!(a, Action::Shift(_))));
    }

    #[test]
    fn test_table_choice() {
        
        let gdef = choice_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        assert!(!table.actions(0, 0).is_empty()); 
        assert!(!table.actions(0, 1).is_empty()); 
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
        
        let gdef = two_nt_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        assert!(!table.actions(0, 0).is_empty());
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
            let acts = table.actions(s, 1); 
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
