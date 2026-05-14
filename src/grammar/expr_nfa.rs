use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::automata::unweighted_u32::dfa::{DFA, Label};
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic;
use crate::automata::unweighted_u32::minimize_cyclic::minimize_cyclic;
use crate::automata::unweighted_u32::nfa::NFA;

use super::ast::GrammarExpr;

/// An NFA whose transition labels are indices into `symbols`.
///
/// This keeps the transition graph compact while allowing each transition
/// symbol to be an arbitrary [`GrammarExpr`]. A transition label is valid when
/// it is non-negative and less than `symbols.len()`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExprNFA {
    pub nfa: NFA,
    pub symbols: Vec<GrammarExpr>,
}

impl ExprNFA {
    pub fn new(nfa: NFA, symbols: Vec<GrammarExpr>) -> Self {
        Self { nfa, symbols }
    }

    pub fn into_determinized_and_minimized(self) -> Self {
        let dfa = self.determinize_and_minimize();
        let symbols = self.symbols;
        let mut nfa = NFA::new_empty();
        for _ in &dfa.states {
            nfa.add_state();
        }
        if !dfa.states.is_empty() {
            nfa.start_states.push(dfa.start_state);
        }
        for (state_id, state) in dfa.states.iter().enumerate() {
            if state.is_accepting {
                nfa.set_accepting(state_id as u32);
            }
            for (&label, &target) in &state.transitions {
                nfa.add_transition(state_id as u32, label, target);
            }
        }
        Self::new(nfa, symbols)
    }

    pub fn determinize(&self) -> DFA {
        determinize_nfa(&self.nfa)
    }

    pub fn determinize_and_minimize(&self) -> DFA {
        minimize_dfa(&self.determinize())
    }

    pub fn symbol_for_label(&self, label: Label) -> Option<&GrammarExpr> {
        usize::try_from(label).ok().and_then(|index| self.symbols.get(index))
    }
}

/// Builder for an [`ExprNFA`] through an intermediate NFA.
///
/// Transitions are labeled by arbitrary [`GrammarExpr`] symbols. Equal symbols
/// are automatically interned to the same label, so callers can construct paths
/// directly without managing the side table by hand.
#[derive(Debug, Clone)]
pub struct ExprNfaBuilder {
    nfa: NFA,
    symbols: Vec<GrammarExpr>,
    symbol_labels: HashMap<GrammarExpr, Label>,
}

impl Default for ExprNfaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprNfaBuilder {
    pub fn new() -> Self {
        Self {
            nfa: NFA::new(),
            symbols: Vec::new(),
            symbol_labels: HashMap::new(),
        }
    }

    pub fn add_state(&mut self) -> u32 {
        self.nfa.add_state()
    }

    pub fn start_state(&self) -> u32 {
        self.nfa.start_states.first().copied().unwrap_or(0)
    }

    pub fn add_start_state(&mut self, state: u32) {
        if !self.nfa.start_states.contains(&state) {
            self.nfa.start_states.push(state);
        }
    }

    pub fn set_accepting(&mut self, state: u32) {
        self.nfa.set_accepting(state);
    }

    pub fn add_epsilon(&mut self, from: u32, to: u32) {
        self.nfa.add_epsilon(from, to);
    }

    pub fn add_symbol(&mut self, symbol: GrammarExpr) -> Label {
        if let Some(&label) = self.symbol_labels.get(&symbol) {
            return label;
        }
        let label = i32::try_from(self.symbols.len())
            .expect("ExprNFA symbol table exceeded i32 labels");
        self.symbols.push(symbol.clone());
        self.symbol_labels.insert(symbol, label);
        label
    }

    pub fn add_transition(&mut self, from: u32, symbol: GrammarExpr, to: u32) -> Label {
        let label = self.add_symbol(symbol);
        self.add_labeled_transition(from, label, to);
        label
    }

    pub fn add_labeled_transition(&mut self, from: u32, label: Label, to: u32) {
        self.nfa.add_transition(from, label, to);
    }

    pub fn into_nfa_and_symbols(self) -> (NFA, Vec<GrammarExpr>) {
        (self.nfa, self.symbols)
    }

    pub fn build(self) -> ExprNFA {
        let (nfa, symbols) = self.into_nfa_and_symbols();
        ExprNFA::new(nfa, symbols)
    }
}

pub fn minimize_dfa(dfa: &DFA) -> DFA {
    if dfa.is_acyclic() {
        minimize_acyclic(dfa)
    } else {
        minimize_cyclic(dfa)
    }
}

fn subset_is_accepting(nfa: &NFA, subset: &[u32]) -> bool {
    subset.iter().any(|&state| nfa.states[state as usize].is_accepting)
}

fn epsilon_closure(nfa: &NFA, seeds: &[u32]) -> BTreeSet<u32> {
    let mut closed = BTreeSet::new();
    let mut queue: VecDeque<u32> = seeds.iter().copied().collect();
    while let Some(state) = queue.pop_front() {
        if !closed.insert(state) {
            continue;
        }
        let Some(nfa_state) = nfa.states.get(state as usize) else {
            continue;
        };
        for &target in &nfa_state.epsilons {
            if !closed.contains(&target) {
                queue.push_back(target);
            }
        }
    }
    closed
}

fn gather_label_targets(nfa: &NFA, subset: &[u32]) -> BTreeMap<Label, BTreeSet<u32>> {
    let mut label_targets = BTreeMap::<Label, BTreeSet<u32>>::new();
    for &state in subset {
        let Some(nfa_state) = nfa.states.get(state as usize) else {
            continue;
        };
        for (&label, targets) in &nfa_state.transitions {
            label_targets
                .entry(label)
                .or_default()
                .extend(targets.iter().copied());
        }
    }
    label_targets
}

fn get_or_create_subset_state(
    dfa: &mut DFA,
    subset_map: &mut HashMap<Vec<u32>, u32>,
    worklist: &mut VecDeque<Vec<u32>>,
    subset: Vec<u32>,
) -> u32 {
    if let Some(&state) = subset_map.get(&subset) {
        return state;
    }
    let state = dfa.add_state();
    subset_map.insert(subset.clone(), state);
    worklist.push_back(subset);
    state
}

pub fn determinize_nfa(nfa: &NFA) -> DFA {
    if nfa.states.is_empty() || nfa.start_states.is_empty() {
        return DFA::new();
    }

    let mut dfa = DFA {
        states: Vec::new(),
        start_state: 0,
    };
    let mut subset_map = HashMap::<Vec<u32>, u32>::new();
    let mut worklist = VecDeque::<Vec<u32>>::new();

    let start_closure = epsilon_closure(nfa, &nfa.start_states);
    let start_key = start_closure.iter().copied().collect::<Vec<_>>();
    let start_id = dfa.add_state();
    dfa.start_state = start_id;
    subset_map.insert(start_key.clone(), start_id);
    worklist.push_back(start_key);

    while let Some(subset_key) = worklist.pop_front() {
        let dfa_state = subset_map[&subset_key];
        if subset_is_accepting(nfa, &subset_key) {
            dfa.set_accepting(dfa_state, true);
        }

        for (label, raw_targets) in gather_label_targets(nfa, &subset_key) {
            let seeds = raw_targets.iter().copied().collect::<Vec<_>>();
            let next_key = epsilon_closure(nfa, &seeds).into_iter().collect::<Vec<_>>();
            if next_key.is_empty() {
                continue;
            }
            let next_state =
                get_or_create_subset_state(&mut dfa, &mut subset_map, &mut worklist, next_key);
            dfa.add_transition(dfa_state, label, next_state);
        }
    }

    dfa
}

#[cfg(test)]
mod tests {
    use crate::grammar::ast::{lower, NamedGrammar, NamedRule};
    use crate::grammar::flat::Symbol;

    use super::*;

    #[test]
    fn lowers_expr_nfa_transition_symbols() {
        let mut nfa = NFA::new();
        let accept = nfa.add_state();
        nfa.add_transition(0, 0, accept);
        nfa.set_accepting(accept);

        let grammar = NamedGrammar {
            rules: vec![NamedRule {
                name: "start".into(),
                expr: GrammarExpr::ExprNFA(Box::new(ExprNFA::new(nfa, vec![
                    GrammarExpr::Literal(b"a".to_vec()),
                ]))),
                is_terminal: false,
                is_internal: false,
            }],
            start: "start".into(),
            ignore: None,
        };

        let lowered = lower(&grammar).expect("ExprNFA should lower");
        assert_eq!(lowered.terminals.len(), 1);
        assert!(lowered
            .rules
            .iter()
            .any(|rule| matches!(rule.rhs.as_slice(), [Symbol::Nonterminal(_), Symbol::Terminal(_)])));
    }

    #[test]
    fn builder_preserves_nfa_and_exposes_determinize_minimize() {
        let mut builder = ExprNfaBuilder::new();
        let start = builder.start_state();
        let loop_state = builder.add_state();
        let accept = builder.add_state();

        builder.add_epsilon(start, loop_state);
        builder.add_transition(loop_state, GrammarExpr::Literal(b"a".to_vec()), loop_state);
        builder.add_transition(loop_state, GrammarExpr::Literal(b"b".to_vec()), accept);
        builder.set_accepting(accept);

        let expr_nfa = builder.build();
        assert_eq!(expr_nfa.symbols.len(), 2);
        assert_eq!(expr_nfa.nfa.states[start as usize].epsilons, vec![loop_state]);

        let dfa = expr_nfa.determinize_and_minimize();
        assert!(dfa.states[dfa.start_state as usize]
            .transitions
            .values()
            .any(|&target| target == dfa.start_state));
        assert!(dfa.states.iter().any(|state| state.is_accepting));

        let minimized_expr_nfa = expr_nfa.into_determinized_and_minimized();
        assert_eq!(minimized_expr_nfa.symbols.len(), 2);
        assert!(minimized_expr_nfa
            .nfa
            .states
            .iter()
            .all(|state| state.epsilons.is_empty()));
    }
}
