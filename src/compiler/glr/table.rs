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
    pub action: Vec<FxHashMap<TerminalID, Action>>,
    pub goto: Vec<FxHashMap<NonterminalID, u32>>,
    pub num_states: u32,
    pub num_terminals: u32,
    pub num_rules: u32,
    pub rules: Vec<Rule>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TableRowKey {
    action: Vec<(TerminalID, Action)>,
    goto: Vec<(NonterminalID, u32)>,
}

impl GLRTable {
    pub fn build(grammar: &AnalyzedGrammar) -> Self {
        let t0 = std::time::Instant::now();
        let (item_sets, transitions) = build_lr1_item_sets(grammar);
        let lr1_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        let mut table = build_ielr_table(grammar, &item_sets, &transitions);
        let ielr_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let pre_merge_states = table.num_states;
        let t2 = std::time::Instant::now();
        table.merge_identical_rows();
        let merge_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let debug_profile = std::env::var("GLRMASK_DEBUG_PROFILE")
            .map(|v| { let n = v.trim().to_ascii_lowercase(); !matches!(n.as_str(), "" | "0" | "false" | "no" | "off") })
            .unwrap_or(false);
        if debug_profile {
            let max_items = item_sets.iter().map(|s| s.len()).max().unwrap_or(0);
            let total_items: usize = item_sets.iter().map(|s| s.len()).sum();
            eprintln!(
                "[glrmask/debug][glr_table] lr1_states={} lr1_ms={:.3} ielr_ms={:.3} pre_merge_states={} merge_ms={:.3} final_states={} max_items_per_state={} total_items={}",
                item_sets.len(), lr1_ms, ielr_ms, pre_merge_states, merge_ms, table.num_states, max_items, total_items,
            );
        }

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
        loop {
            let mut sig_to_rep: FxHashMap<TableRowKey, u32> = FxHashMap::default();
            let mut remap: Vec<u32> = (0..self.num_states).collect();
            let mut changed = false;

            for state in 0..self.num_states as usize {
                let row_key = row_key(&self.action[state], &self.goto[state]);
                let rep = *sig_to_rep.entry(row_key).or_insert(state as u32);
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
                        .map(|(&tid, action)| (tid, remap_action_targets(action, &mapping)))
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

fn row_key(
    action_row: &FxHashMap<TerminalID, Action>,
    goto_row: &FxHashMap<NonterminalID, u32>,
) -> TableRowKey {
    TableRowKey {
        action: action_row
            .iter()
            .map(|(&terminal, action)| (terminal, action.clone()))
            .collect(),
        goto: goto_row
            .iter()
            .map(|(&nonterminal, &target)| (nonterminal, target))
            .collect(),
    }
}

fn remap_action_targets(action: &Action, mapping: &[u32]) -> Action {
    match action {
        Action::Shift(target) => Action::Shift(mapping[*target as usize]),
        Action::Reduce(rule) => Action::Reduce(*rule),
        Action::Split {
            shift,
            reduces,
            accept,
        } => Action::Split {
            shift: shift.map(|target| mapping[target as usize]),
            reduces: reduces.clone(),
            accept: *accept,
        },
        Action::Accept => Action::Accept,
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

fn lr0_closure(items: &BTreeSet<Item>, rules: &[Rule]) -> BTreeSet<Item> {
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

fn lr0_goto_set(items: &BTreeSet<Item>, sym: &Symbol, rules: &[Rule]) -> BTreeSet<Item> {
    let mut kernel = BTreeSet::new();
    for item in items {
        if item.next_symbol(rules) == Some(sym) {
            kernel.insert(Item::new(item.rule, item.dot + 1));
        }
    }
    lr0_closure(&kernel, rules)
}

fn build_item_sets<ItemT, NextSymbol, GotoSet>(
    initial: BTreeSet<ItemT>,
    next_symbol: NextSymbol,
    goto_set: GotoSet,
) -> (Vec<BTreeSet<ItemT>>, Vec<BTreeMap<Symbol, u32>>)
where
    ItemT: Copy + Ord + std::hash::Hash,
    NextSymbol: Fn(&ItemT) -> Option<Symbol>,
    GotoSet: Fn(&BTreeSet<ItemT>, &Symbol) -> BTreeSet<ItemT>,
{
    let mut item_sets = vec![initial.clone()];
    let mut transitions = vec![BTreeMap::new()];
    let mut set_to_id: FxHashMap<Vec<ItemT>, u32> = FxHashMap::default();
    set_to_id.insert(initial.iter().copied().collect(), 0);

    let mut queue = VecDeque::from([0u32]);
    while let Some(state_id) = queue.pop_front() {
        let symbols: BTreeSet<Symbol> = item_sets[state_id as usize]
            .iter()
            .filter_map(&next_symbol)
            .collect();

        for symbol in &symbols {
            let target_items = goto_set(&item_sets[state_id as usize], symbol);
            if target_items.is_empty() {
                continue;
            }

            let key: Vec<ItemT> = target_items.iter().copied().collect();
            let target_id = if let Some(&existing_id) = set_to_id.get(&key) {
                existing_id
            } else {
                let new_id = item_sets.len() as u32;
                set_to_id.insert(key, new_id);
                item_sets.push(target_items);
                transitions.push(BTreeMap::new());
                queue.push_back(new_id);
                new_id
            };

            transitions[state_id as usize].insert(symbol.clone(), target_id);
        }
    }

    (item_sets, transitions)
}

#[allow(dead_code)]
fn build_lr0_item_sets(grammar: &AnalyzedGrammar) -> (Vec<BTreeSet<Item>>, Vec<BTreeMap<Symbol, u32>>) {
    let rules = &grammar.rules;

    let initial = {
        let mut s = BTreeSet::new();
        s.insert(Item::new(0, 0)); 
        lr0_closure(&s, rules)
    };

    build_item_sets(
        initial,
        |item| item.next_symbol(rules).cloned(),
        |items, sym| lr0_goto_set(items, sym, rules),
    )
}

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

fn initialize_pending_and_goto(
    transitions: &[BTreeMap<Symbol, u32>],
) -> (
    Vec<BTreeMap<TerminalID, PendingAction>>,
    Vec<FxHashMap<NonterminalID, u32>>,
) {
    let mut pending = std::iter::repeat_with(BTreeMap::<TerminalID, PendingAction>::new)
        .take(transitions.len())
        .collect::<Vec<_>>();
    let mut goto: Vec<FxHashMap<NonterminalID, u32>> = (0..transitions.len()).map(|_| FxHashMap::default()).collect();

    for (state_id, by_symbol) in transitions.iter().enumerate() {
        for (symbol, &target) in by_symbol {
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
    }

    (pending, goto)
}

fn finish_table(
    grammar: &AnalyzedGrammar,
    pending: Vec<BTreeMap<TerminalID, PendingAction>>,
    goto: Vec<FxHashMap<NonterminalID, u32>>,
) -> GLRTable {
    let action: Vec<FxHashMap<TerminalID, Action>> = pending
        .into_iter()
        .map(|by_terminal| {
            by_terminal
                .into_iter()
                .map(|(terminal, pending)| (terminal, pending.finish()))
                .collect()
        })
        .collect();
    let num_states = action.len() as u32;

    GLRTable {
        action,
        goto,
        num_states,
        num_terminals: grammar.num_terminals,
        num_rules: grammar.rules.len() as u32,
        rules: grammar.rules.clone(),
    }
}

#[allow(dead_code)]
fn build_slr1_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[BTreeSet<Item>],
    transitions: &[BTreeMap<Symbol, u32>],
) -> GLRTable {
    let (mut pending, goto) = initialize_pending_and_goto(transitions);

    for (state_id, items) in item_sets.iter().enumerate() {

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

    finish_table(grammar, pending, goto)
}

// LR(1) item set construction.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct LR1Item {
    rule: u32,
    dot: u32,
    lookahead: TerminalID,
}

impl LR1Item {
    fn new(rule: u32, dot: u32, lookahead: TerminalID) -> Self {
        Self { rule, dot, lookahead }
    }

    fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {
        let rhs = &rules[self.rule as usize].rhs;
        rhs.get(self.dot as usize)
    }
}

/// Compute FIRST set for a sequence of symbols followed by a lookahead terminal.
fn first_of_sequence(
    symbols: &[Symbol],
    lookahead: TerminalID,
    first: &[BTreeSet<TerminalID>],
    nullable: &BTreeSet<NonterminalID>,
) -> BTreeSet<TerminalID> {
    let mut result = BTreeSet::new();
    let mut all_nullable = true;
    for sym in symbols {
        match sym {
            Symbol::Terminal(t) => {
                result.insert(*t);
                all_nullable = false;
                break;
            }
            Symbol::Nonterminal(nt) => {
                result.extend(&first[*nt as usize]);
                if !nullable.contains(nt) {
                    all_nullable = false;
                    break;
                }
            }
        }
    }
    if all_nullable {
        result.insert(lookahead);
    }
    result
}

fn lr1_closure(
    items: &BTreeSet<LR1Item>,
    grammar: &AnalyzedGrammar,
) -> BTreeSet<LR1Item> {
    let rules = &grammar.rules;
    let mut result = items.clone();
    let mut queue: VecDeque<LR1Item> = items.iter().copied().collect();

    while let Some(item) = queue.pop_front() {
        if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
            let rhs = &rules[item.rule as usize].rhs;
            let beta = &rhs[(item.dot as usize + 1)..];

            let lookaheads = first_of_sequence(beta, item.lookahead, &grammar.first, &grammar.nullable);

            for &i in &grammar.rules_by_lhs[*nt as usize] {
                for &la in &lookaheads {
                    let new_item = LR1Item::new(i, 0, la);
                    if result.insert(new_item) {
                        queue.push_back(new_item);
                    }
                }
            }
        }
    }
    result
}

fn lr1_goto_set(
    items: &BTreeSet<LR1Item>,
    sym: &Symbol,
    grammar: &AnalyzedGrammar,
) -> BTreeSet<LR1Item> {
    let rules = &grammar.rules;
    let mut kernel = BTreeSet::new();
    for item in items {
        if item.next_symbol(rules) == Some(sym) {
            kernel.insert(LR1Item::new(item.rule, item.dot + 1, item.lookahead));
        }
    }
    lr1_closure(&kernel, grammar)
}

fn build_lr1_item_sets(
    grammar: &AnalyzedGrammar,
) -> (Vec<BTreeSet<LR1Item>>, Vec<BTreeMap<Symbol, u32>>) {
    let rules = &grammar.rules;

    let initial = {
        let mut s = BTreeSet::new();
        s.insert(LR1Item::new(0, 0, EOF));
        lr1_closure(&s, grammar)
    };

    let mut item_sets = vec![initial.clone()];
    let mut transitions = vec![BTreeMap::new()];
    let mut set_to_id: FxHashMap<Vec<LR1Item>, u32> = FxHashMap::default();
    set_to_id.insert(initial.iter().copied().collect(), 0);

    let mut queue = VecDeque::from([0u32]);
    while let Some(state_id) = queue.pop_front() {
        // Build all goto kernels in a single pass over items.
        let mut kernels: BTreeMap<Symbol, BTreeSet<LR1Item>> = BTreeMap::new();
        for item in &item_sets[state_id as usize] {
            if let Some(sym) = item.next_symbol(rules) {
                kernels
                    .entry(sym.clone())
                    .or_default()
                    .insert(LR1Item::new(item.rule, item.dot + 1, item.lookahead));
            }
        }

        for (symbol, kernel) in &kernels {
            let target_items = lr1_closure(kernel, grammar);
            if target_items.is_empty() {
                continue;
            }

            let key: Vec<LR1Item> = target_items.iter().copied().collect();
            let target_id = if let Some(&existing_id) = set_to_id.get(&key) {
                existing_id
            } else {
                let new_id = item_sets.len() as u32;
                set_to_id.insert(key, new_id);
                item_sets.push(target_items);
                transitions.push(BTreeMap::new());
                queue.push_back(new_id);
                new_id
            };

            transitions[state_id as usize].insert(symbol.clone(), target_id);
        }
    }

    (item_sets, transitions)
}

fn build_lr1_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[BTreeSet<LR1Item>],
    transitions: &[BTreeMap<Symbol, u32>],
) -> GLRTable {
    let (mut pending, goto) = initialize_pending_and_goto(transitions);

    for (state_id, items) in item_sets.iter().enumerate() {

        for item in items {
            let rule = &grammar.rules[item.rule as usize];
            if item.dot as usize != rule.rhs.len() {
                continue;
            }

            if item.rule == 0 {
                pending[state_id].entry(item.lookahead).or_default().push_accept();
                continue;
            }

            pending[state_id]
                .entry(item.lookahead)
                .or_default()
                .push_reduce(item.rule);
        }
    }

    finish_table(grammar, pending, goto)
}

// IELR-style merge.

fn lr1_core_key(items: &BTreeSet<LR1Item>) -> Vec<Item> {
    let mut core = BTreeSet::new();
    for item in items {
        core.insert(Item::new(item.rule, item.dot));
    }
    core.into_iter().collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ActionSig {
    Shift(u32),
    Reduce(u32),
    Split {
        shift: Option<u32>,
        reduces: Vec<u32>,
        accept: bool,
    },
    Accept,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowSignature {
    core_class: u32,
    action: Vec<(TerminalID, ActionSig)>,
    goto: Vec<(NonterminalID, u32)>,
}

fn remap_action_to_partition(action: &Action, partition: &[u32]) -> ActionSig {
    match action {
        Action::Shift(target) => ActionSig::Shift(partition[*target as usize]),
        Action::Reduce(rule) => ActionSig::Reduce(*rule),
        Action::Split {
            shift,
            reduces,
            accept,
        } => ActionSig::Split {
            shift: shift.map(|target| partition[target as usize]),
            reduces: reduces.clone(),
            accept: *accept,
        },
        Action::Accept => ActionSig::Accept,
    }
}

fn core_classes(core_keys: &[Vec<Item>]) -> Vec<u32> {
    let mut class_of = vec![0; core_keys.len()];
    let mut key_to_class: FxHashMap<Vec<Item>, u32> = FxHashMap::default();
    let mut next = 0u32;

    for (state, key) in core_keys.iter().enumerate() {
        let class = *key_to_class.entry(key.clone()).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        class_of[state] = class;
    }

    class_of
}

fn refine_same_core_partition(table: &GLRTable, core_keys: &[Vec<Item>]) -> Vec<u32> {
    let nstates = table.num_states as usize;
    let core_class_of = core_classes(core_keys);
    let mut partition = core_class_of.clone();

    loop {
        let mut sig_to_part: FxHashMap<RowSignature, u32> = FxHashMap::default();
        let mut next_partition = vec![0u32; nstates];
        let mut next_id = 0u32;

        for state in 0..nstates {
            let action = table.action[state]
                .iter()
                .map(|(&terminal, action)| {
                    (terminal, remap_action_to_partition(action, &partition))
                })
                .collect();
            let goto = table.goto[state]
                .iter()
                .map(|(&nt, &target)| (nt, partition[target as usize]))
                .collect();
            let signature = RowSignature {
                core_class: core_class_of[state],
                action,
                goto,
            };

            let class = *sig_to_part.entry(signature).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            next_partition[state] = class;
        }

        if next_partition == partition {
            return partition;
        }
        partition = next_partition;
    }
}

fn merge_same_core_lr1_states(table: GLRTable, core_keys: &[Vec<Item>]) -> GLRTable {
    let partition = refine_same_core_partition(&table, core_keys);
    let nstates = table.num_states as usize;
    let ngroups = partition.iter().copied().max().map(|x| x + 1).unwrap_or(0) as usize;

    let mut representatives = vec![u32::MAX; ngroups];
    for state in 0..nstates {
        let group = partition[state] as usize;
        if representatives[group] == u32::MAX {
            representatives[group] = state as u32;
        }
    }

    let action = representatives
        .iter()
        .map(|&rep| {
            table.action[rep as usize]
                .iter()
                .map(|(&terminal, action)| (terminal, remap_action_targets(action, &partition)))
                .collect()
        })
        .collect();
    let goto = representatives
        .iter()
        .map(|&rep| {
            table.goto[rep as usize]
                .iter()
                .map(|(&nt, &target)| (nt, partition[target as usize]))
                .collect()
        })
        .collect();

    GLRTable {
        action,
        goto,
        num_states: ngroups as u32,
        num_terminals: table.num_terminals,
        num_rules: table.num_rules,
        rules: table.rules,
    }
}

fn build_ielr_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[BTreeSet<LR1Item>],
    transitions: &[BTreeMap<Symbol, u32>],
) -> GLRTable {
    let canonical = build_lr1_table(grammar, item_sets, transitions);
    let core_keys = item_sets.iter().map(lr1_core_key).collect::<Vec<_>>();
    merge_same_core_lr1_states(canonical, &core_keys)
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
