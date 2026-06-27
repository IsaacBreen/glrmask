use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use crate::automata::unweighted_u32::dfa::{DFA, Label};
use crate::automata::unweighted_u32::minimize_acyclic::{
    minimize_acyclic, reindex_minimized_acyclic_dfa,
};
use crate::automata::unweighted_u32::minimize_cyclic::minimize_cyclic;
use crate::automata::unweighted_u32::nfa::NFA;

use super::ast::GrammarExpr;

/// An NFA whose transition labels are indices into `symbols`.
///
/// This keeps the transition graph compact while allowing each transition
/// symbol to be an arbitrary [`GrammarExpr`]. A transition label is valid when
/// it is non-negative and less than `symbols.len()`.
#[derive(Debug, Clone)]
pub struct ExprNFA {
    pub nfa: NFA,
    pub symbols: Vec<GrammarExpr>,
    /// `true` only when `nfa` is the exact minimized deterministic DFA for its
    /// current label graph.  AST lowering can then avoid repeating the same
    /// minimization before emitting left-linear rules.
    pub is_determinized_and_minimized: bool,
    /// The exact DFA produced by the first minimization, when available.
    /// This is performance metadata only: equality and hashing deliberately
    /// ignore it, and any graph or symbol rewrite must clear it.
    pub(crate) canonical_dfa: Option<DFA>,
}

impl PartialEq for ExprNFA {
    fn eq(&self, other: &Self) -> bool {
        self.nfa == other.nfa && self.symbols == other.symbols
    }
}

impl Eq for ExprNFA {}

impl Hash for ExprNFA {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.nfa.hash(state);
        self.symbols.hash(state);
    }
}

impl ExprNFA {
    pub fn new(nfa: NFA, symbols: Vec<GrammarExpr>) -> Self {
        Self {
            nfa,
            symbols,
            is_determinized_and_minimized: false,
            canonical_dfa: None,
        }
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
        Self {
            nfa,
            symbols,
            is_determinized_and_minimized: true,
            canonical_dfa: Some(dfa),
        }
    }

    pub fn determinize(&self) -> DFA {
        determinize_nfa(&self.nfa)
    }

    pub fn determinize_and_minimize(&self) -> DFA {
        if self.is_determinized_and_minimized
            && let Some(dfa) = &self.canonical_dfa
            && dfa.is_acyclic()
        {
            return reindex_minimized_acyclic_dfa(dfa);
        }
        let dfa = self.determinize();
        if self.is_determinized_and_minimized && dfa.is_acyclic() {
            reindex_minimized_acyclic_dfa(&dfa)
        } else {
            minimize_dfa(&dfa)
        }
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

fn gather_label_targets(nfa: &NFA, subset: &[u32]) -> BTreeMap<Label, Vec<u32>> {
    let mut label_targets = BTreeMap::<Label, Vec<u32>>::new();
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

/// Lazily cache single-state epsilon closures.  Determinization repeatedly
/// computes closures of unions of NFA targets; expanding the constituent
/// closures once and deduplicating through a generation-mark array avoids a
/// B-tree allocation for every labeled edge while retaining sorted subset keys.
struct EpsilonClosureCache<'a> {
    nfa: &'a NFA,
    closures: Vec<Option<Vec<u32>>>,
    marks: Vec<u32>,
    generation: u32,
}

impl<'a> EpsilonClosureCache<'a> {
    fn new(nfa: &'a NFA) -> Self {
        Self {
            nfa,
            closures: vec![None; nfa.states.len()],
            marks: vec![0; nfa.states.len()],
            generation: 0,
        }
    }

    fn ensure_closure_for_state(&mut self, state: u32) -> Option<usize> {
        let index = state as usize;
        if index >= self.closures.len() {
            return None;
        }
        if self.closures[index].is_none() {
            self.closures[index] = Some(epsilon_closure(self.nfa, &[state]).into_iter().collect());
        }
        Some(index)
    }

    fn union<I>(&mut self, states: I) -> Vec<u32>
    where
        I: IntoIterator<Item = u32>,
    {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.fill(0);
            self.generation = 1;
        }
        let generation = self.generation;
        let mut result = Vec::new();
        for state in states {
            let Some(closure_index) = self.ensure_closure_for_state(state) else {
                continue;
            };
            let (closures, marks) = (&self.closures, &mut self.marks);
            let closure = closures[closure_index]
                .as_deref()
                .expect("ensured epsilon closure is present");
            for &target in closure {
                let mark = &mut marks[target as usize];
                if *mark != generation {
                    *mark = generation;
                    result.push(target);
                }
            }
        }
        result.sort_unstable();
        result
    }
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
    if nfa.start_states.len() == 1
        && nfa.states.iter().all(|state| {
            state.epsilons.is_empty() && state.transitions.values().all(|targets| targets.len() <= 1)
        })
    {
        return determinize_already_deterministic_nfa(nfa);
    }

    determinize_general_nfa(nfa)
}

/// Preserve the ordinary subset construction's breadth-first state numbering
/// when the input is already a deterministic epsilon-free automaton.  This is
/// common for schema-generated discriminator automata and avoids allocating a
/// sorted subset, closure, and target set for every edge.
fn determinize_already_deterministic_nfa(nfa: &NFA) -> DFA {
    if nfa.states.is_empty() || nfa.start_states.is_empty() {
        return DFA::new();
    }

    let start = nfa.start_states[0] as usize;
    if start >= nfa.states.len() {
        return determinize_general_nfa(nfa);
    }

    let mut dfa = DFA {
        states: Vec::new(),
        start_state: 0,
    };
    let mut state_map = vec![None; nfa.states.len()];
    let mut worklist = VecDeque::new();

    let start_id = dfa.add_state();
    dfa.start_state = start_id;
    state_map[start] = Some(start_id);
    worklist.push_back(start);

    while let Some(nfa_state_id) = worklist.pop_front() {
        let dfa_state_id = state_map[nfa_state_id].expect("queued deterministic state has a DFA id");
        let nfa_state = &nfa.states[nfa_state_id];
        if nfa_state.is_accepting {
            dfa.set_accepting(dfa_state_id, true);
        }

        for (&label, targets) in &nfa_state.transitions {
            let Some(&target) = targets.first() else {
                continue;
            };
            let target = target as usize;
            if target >= nfa.states.len() {
                continue;
            }
            let target_id = if let Some(target_id) = state_map[target] {
                target_id
            } else {
                let target_id = dfa.add_state();
                state_map[target] = Some(target_id);
                worklist.push_back(target);
                target_id
            };
            dfa.add_transition(dfa_state_id, label, target_id);
        }
    }

    dfa
}

fn determinize_general_nfa(nfa: &NFA) -> DFA {
    if nfa.states.is_empty() || nfa.start_states.is_empty() {
        return DFA::new();
    }

    let mut dfa = DFA {
        states: Vec::new(),
        start_state: 0,
    };
    let mut subset_map = HashMap::<Vec<u32>, u32>::new();
    let mut worklist = VecDeque::<Vec<u32>>::new();
    let mut epsilon_closures = EpsilonClosureCache::new(nfa);

    let start_key = epsilon_closures.union(nfa.start_states.iter().copied());
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
            let next_key = epsilon_closures.union(raw_targets.into_iter());
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
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

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

    #[test]
    fn deterministic_fast_path_matches_general_subset_construction() {
        let mut nfa = NFA::new();
        let first = nfa.add_state();
        let second = nfa.add_state();
        let accept = nfa.add_state();
        nfa.add_transition(0, 2, second);
        nfa.add_transition(0, 1, first);
        nfa.add_transition(first, 3, accept);
        nfa.add_transition(second, 4, accept);
        nfa.set_accepting(accept);

        assert_eq!(determinize_nfa(&nfa), determinize_general_nfa(&nfa));
    }

    #[test]
    fn canonical_marker_does_not_affect_expression_identity() {
        let mut builder = ExprNfaBuilder::new();
        let accept = builder.add_state();
        builder.add_transition(
            builder.start_state(),
            GrammarExpr::Literal(b"a".to_vec()),
            accept,
        );
        builder.set_accepting(accept);

        let canonical = builder.build().into_determinized_and_minimized();
        let unmarked = ExprNFA {
            nfa: canonical.nfa.clone(),
            symbols: canonical.symbols.clone(),
            is_determinized_and_minimized: false,
            canonical_dfa: None,
        };
        assert_eq!(canonical, unmarked);
        let mut canonical_hasher = DefaultHasher::new();
        canonical.hash(&mut canonical_hasher);
        let mut unmarked_hasher = DefaultHasher::new();
        unmarked.hash(&mut unmarked_hasher);
        assert_eq!(canonical_hasher.finish(), unmarked_hasher.finish());
        assert_eq!(
            canonical.determinize_and_minimize(),
            unmarked.determinize_and_minimize(),
        );
    }
}
