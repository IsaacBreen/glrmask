//! Minimal acyclic determinization for template NFAs with DEFAULT semantics.
//!
//! Template NFAs use a distinguished `DEFAULT` label on a stack matcher to
//! mean "any concrete stack symbol". After determinization the runtime needs
//! an ordinary deterministic automaton in which a concrete nonnegative label
//! takes the union of its explicit edges and the DEFAULT fallback, while
//! negative PUSH labels never inherit the fallback.
//!
//! Expanding DEFAULT by subset construction (the historical specialization
//! step) enumerates every reachable *set* of automaton states. On wide LR
//! tables that is exponential even when the resulting language has only a few
//! thousand distinct residual languages. This module builds that minimal
//! deterministic automaton directly, with the same canonical residual-language
//! hashing used by the ordinary minimal determinizer.
//!
//! Canonical nodes carry `(accepting, default_target, explicit edges)`. A
//! nonnegative explicit edge is redundant when it equals the default target,
//! so it is dropped; this makes the node a true function of its residual
//! language and lets equivalent states intern to one node.

use std::collections::{BTreeSet, VecDeque};

use glrmask_invariant::__private::charge_template_build_budget as charge_build_budget;
use rustc_hash::FxHashMap;

use super::dfa::{DFA, Label};
use super::nfa::NFA;

pub const DEFAULT_NODE_BUDGET: usize = 1_000_000;
pub const DEFAULT_DEPTH_BUDGET: usize = 4_096;

#[derive(Debug, Clone)]
struct Node {
    accepting: bool,
    /// Target for nonnegative labels without an explicit edge.
    default: u32,
    /// Explicit edges, sorted by label. Negative labels are always explicit.
    transitions: Vec<(Label, u32)>,
}

impl Node {
    fn target(&self, label: Label) -> u32 {
        if label >= 0 {
            self.transitions
                .binary_search_by_key(&label, |&(candidate, _)| candidate)
                .map_or(self.default, |index| self.transitions[index].1)
        } else {
            self.transitions
                .binary_search_by_key(&label, |&(candidate, _)| candidate)
                .map_or(0, |index| self.transitions[index].1)
        }
    }
}

struct Builder {
    nodes: Vec<Node>,
    intern: FxHashMap<(bool, u32, Vec<(Label, u32)>), u32>,
    unions: FxHashMap<Vec<u32>, u32>,
    node_budget: usize,
    depth_budget: usize,
    accept: Option<u32>,
}

impl Builder {
    fn new(node_budget: usize, depth_budget: usize) -> Self {
        Self {
            nodes: vec![Node { accepting: false, default: 0, transitions: Vec::new() }],
            intern: FxHashMap::default(),
            unions: FxHashMap::default(),
            node_budget,
            depth_budget,
            accept: None,
        }
    }

    fn intern(
        &mut self,
        accepting: bool,
        default: u32,
        mut transitions: Vec<(Label, u32)>,
    ) -> Option<u32> {
        charge_build_budget("template_determinization", transitions.len() as u64);
        transitions.sort_unstable_by_key(|&(label, _)| label);
        transitions.dedup_by(|a, b| a.0 == b.0);
        transitions.retain(|&(label, target)| {
            if label >= 0 {
                target != default
            } else {
                target != 0
            }
        });
        if transitions.is_empty() && !accepting && default == 0 {
            return Some(0);
        }
        let key = (accepting, default, transitions);
        if let Some(&node) = self.intern.get(&key) {
            return Some(node);
        }
        if self.nodes.len() >= self.node_budget {
            return None;
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(Node {
            accepting: key.0,
            default: key.1,
            transitions: key.2.clone(),
        });
        self.intern.insert(key, id);
        Some(id)
    }

    fn accept_node(&mut self) -> Option<u32> {
        if let Some(node) = self.accept {
            return Some(node);
        }
        let node = self.intern(true, 0, Vec::new())?;
        self.accept = Some(node);
        Some(node)
    }

    fn union(&mut self, ids: &mut Vec<u32>, depth: usize) -> Option<u32> {
        charge_build_budget("template_determinization", ids.len() as u64);
        ids.sort_unstable();
        ids.dedup();
        ids.retain(|&id| id != 0);
        if ids.is_empty() {
            return Some(0);
        }
        if ids.len() == 1 {
            return Some(ids[0]);
        }
        if let Some(&node) = self.unions.get(ids.as_slice()) {
            return Some(node);
        }
        if depth >= self.depth_budget {
            return None;
        }

        let accepting = ids.iter().any(|&id| self.nodes[id as usize].accepting);
        let mut defaults = Vec::new();
        let mut labels = BTreeSet::new();
        for &id in ids.iter() {
            let node = &self.nodes[id as usize];
            if node.default != 0 {
                defaults.push(node.default);
            }
            for &(label, _) in &node.transitions {
                labels.insert(label);
            }
        }
        let default = self.union(&mut defaults, depth + 1)?;

        let mut transitions = Vec::with_capacity(labels.len());
        for label in labels {
            let mut children = Vec::new();
            for &id in ids.iter() {
                let child = self.nodes[id as usize].target(label);
                if child != 0 {
                    children.push(child);
                }
            }
            let child = self.union(&mut children, depth + 1)?;
            let keep = if label >= 0 { child != default } else { child != 0 };
            if keep {
                transitions.push((label, child));
            }
        }
        let node = self.intern(accepting, default, transitions)?;
        self.unions.insert(ids.clone(), node);
        Some(node)
    }
}

fn reverse_topological_order(nfa: &NFA) -> Option<Vec<u32>> {
    let states = nfa.states.len();
    let mut successors = vec![0usize; states];
    let mut predecessors: Vec<Vec<u32>> = vec![Vec::new(); states];
    for (source, state) in nfa.states.iter().enumerate() {
        for &target in state.transitions.values().flatten().chain(state.epsilons.iter()) {
            let target = target as usize;
            if target >= states {
                return None;
            }
            successors[source] += 1;
            predecessors[target].push(source as u32);
        }
    }
    let mut queue: VecDeque<u32> = successors
        .iter()
        .enumerate()
        .filter_map(|(state, &count)| (count == 0).then_some(state as u32))
        .collect();
    let mut order = Vec::with_capacity(states);
    while let Some(state) = queue.pop_front() {
        order.push(state);
        for &predecessor in &predecessors[state as usize] {
            let remaining = &mut successors[predecessor as usize];
            *remaining -= 1;
            if *remaining == 0 {
                queue.push_back(predecessor);
            }
        }
    }
    (order.len() == states).then_some(order)
}

fn materialize(builder: &Builder, start: u32, default_label: Label) -> DFA {
    if start == 0 {
        return DFA::new();
    }
    let mut dfa = DFA::new();
    let mut id_of: FxHashMap<u32, u32> = FxHashMap::default();
    let mut queue: VecDeque<u32> = VecDeque::new();
    id_of.insert(start, 0);
    queue.push_back(start);
    while let Some(node) = queue.pop_front() {
        let state = id_of[&node];
        let data = &builder.nodes[node as usize];
        charge_build_budget(
            "template_determinization",
            1u64.saturating_add(data.transitions.len() as u64),
        );
        dfa.states[state as usize].is_accepting = data.accepting;
        let mut transitions: Vec<(Label, u32)> = Vec::with_capacity(data.transitions.len() + 1);
        if data.default != 0 {
            transitions.push((default_label, data.default));
        }
        transitions.extend(data.transitions.iter().copied());
        for (label, child) in transitions {
            let target = if let Some(&target) = id_of.get(&child) {
                target
            } else {
                let target = dfa.add_state();
                id_of.insert(child, target);
                queue.push_back(child);
                target
            };
            dfa.states[state as usize].transitions.insert(label, target);
        }
    }
    dfa
}

/// Determinize an acyclic template NFA into the minimal deterministic
/// automaton under template DEFAULT semantics.
///
/// Returns `None` when the input is cyclic, contains out-of-range references,
/// or exceeds the node/depth budget.
pub fn try_determinize_minimal_defaults(nfa: &NFA, default_label: Label) -> Option<DFA> {
    try_determinize_minimal_defaults_with_budget(
        nfa,
        default_label,
        DEFAULT_NODE_BUDGET,
        DEFAULT_DEPTH_BUDGET,
    )
}

pub fn try_determinize_minimal_defaults_with_budget(
    nfa: &NFA,
    default_label: Label,
    node_budget: usize,
    depth_budget: usize,
) -> Option<DFA> {
    if nfa.states.is_empty() || nfa.start_states.is_empty() {
        return Some(DFA::new());
    }
    let order = reverse_topological_order(nfa)?;
    let mut builder = Builder::new(node_budget, depth_budget);
    let mut canonical = vec![0u32; nfa.states.len()];
    for state in order {
        let data = &nfa.states[state as usize];
        // Every outgoing transition materializes one child vector, and every
        // target in it is copied. Charge the whole row before those vectors
        // are allocated.
        let row_work = data
            .transitions
            .values()
            .map(|targets| targets.len().saturating_add(1))
            .sum::<usize>();
        charge_build_budget(
            "template_determinization",
            1u64.saturating_add(row_work as u64),
        );
        let mut default_targets = Vec::new();
        let mut explicit: Vec<(Label, Vec<u32>)> = Vec::new();
        for (&label, targets) in &data.transitions {
            if targets.is_empty() {
                continue;
            }
            let children: Vec<u32> = targets
                .iter()
                .map(|&target| canonical[target as usize])
                .collect();
            if label == default_label {
                default_targets.extend(children);
            } else {
                explicit.push((label, children));
            }
        }
        let default = builder.union(&mut default_targets, 0)?;
        let mut transitions = Vec::with_capacity(explicit.len());
        for (label, mut children) in explicit {
            if label >= 0 && default != 0 {
                children.push(default);
            }
            let child = builder.union(&mut children, 0)?;
            let keep = if label >= 0 { child != default } else { child != 0 };
            if keep {
                transitions.push((label, child));
            }
        }
        let labels_node = builder.intern(false, default, transitions)?;
        let mut parts: Vec<u32> = Vec::new();
        if data.is_accepting {
            parts.push(builder.accept_node()?);
        }
        for &target in &data.epsilons {
            parts.push(canonical[target as usize]);
        }
        if labels_node != 0 {
            parts.push(labels_node);
        }
        canonical[state as usize] = builder.union(&mut parts, 0)?;
    }
    let mut starts: Vec<u32> = nfa
        .start_states
        .iter()
        .map(|&state| canonical[state as usize])
        .collect();
    let start = builder.union(&mut starts, 0)?;
    Some(materialize(&builder, start, default_label))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    // Matches the GLR template wildcard coordinate. Kept local so this crate
    // stays independent of the grammar/GLR crates.
    const DEFAULT: Label = i32::MAX - 1;

    /// Reference semantics: expand DEFAULT onto every concrete nonnegative
    /// label in `alphabet`, determinize, then minimize the acyclic result.
    fn reference(nfa: &NFA, alphabet: &[Label]) -> DFA {
        let mut expanded = nfa.clone();
        for state in &mut expanded.states {
            if let Some(targets) = state.transitions.remove(&DEFAULT) {
                for &label in alphabet {
                    if label >= 0 {
                        state
                            .transitions
                            .entry(label)
                            .or_default()
                            .extend(targets.iter().copied());
                    }
                }
            }
        }
        let dfa = super::super::determinize::determinize(&expanded);
        super::super::minimize_acyclic::minimize_acyclic(&dfa)
    }

    /// Follow explicit edges, and let nonnegative labels fall back to DEFAULT.
    fn step(row: &super::super::dfa::DFAState, label: Label) -> Option<u32> {
        row.transitions.get(&label).copied().or_else(|| {
            (label >= 0)
                .then(|| row.transitions.get(&DEFAULT).copied())
                .flatten()
        })
    }

    fn words(dfa: &DFA, alphabet: &[Label], bound: usize) -> BTreeMap<Vec<Label>, ()> {
        let mut accepted = BTreeMap::new();
        let mut frontier = vec![(dfa.start_state, Vec::new())];
        if dfa.states[dfa.start_state as usize].is_accepting {
            accepted.insert(Vec::new(), ());
        }
        for _ in 0..bound {
            let mut next = Vec::new();
            for (state, word) in frontier {
                let row = &dfa.states[state as usize];
                for &label in alphabet {
                    let Some(target) = step(row, label) else { continue };
                    let mut updated = word.clone();
                    updated.push(label);
                    if dfa.states[target as usize].is_accepting {
                        accepted.insert(updated.clone(), ());
                    }
                    next.push((target, updated));
                }
            }
            frontier = next;
        }
        accepted
    }

    #[test]
    fn matches_expanded_subset_reference_on_random_acyclic_nfas() {
        let mut seed = 0x9e37_79b9u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let alphabet = [0, 1, 2, 3];
        for trial in 0..48 {
            let states = 2 + (next() % 8) as usize;
            let mut nfa = NFA::new_empty();
            for _ in 0..states {
                nfa.add_state();
            }
            nfa.start_states = vec![(next() % 2) as u32];
            for source in 0..states as u32 {
                if next() % 5 == 0 {
                    nfa.set_accepting(source);
                }
                let first = source as usize + 1;
                if first >= states {
                    continue;
                }
                let candidates = (first..states).collect::<Vec<_>>();
                for label in [DEFAULT, 0, 1, 2, -2, -3] {
                    if next() % 3 != 0 {
                        continue;
                    }
                    let target = candidates[(next() as usize) % candidates.len()] as u32;
                    nfa.add_transition(source, label, target);
                }
                if next() % 3 == 0 {
                    let target = candidates[(next() as usize) % candidates.len()] as u32;
                    nfa.add_epsilon(source, target);
                }
            }
            let minimal = try_determinize_minimal_defaults(&nfa, DEFAULT)
                .unwrap_or_else(|| panic!("trial {trial} exceeded budget"));
            let reference = reference(&nfa, &alphabet);
            assert_eq!(
                words(&minimal, &alphabet, states),
                words(&reference, &alphabet, states),
                "trial {trial}: language differs from expanded subset reference",
            );
            assert!(
                minimal.states.len() <= reference.states.len(),
                "trial {trial}: canonical construction is not minimal ({} > {})",
                minimal.states.len(),
                reference.states.len(),
            );
        }
    }

    #[test]
    fn default_never_applies_to_push_labels() {
        // DEFAULT accepts any concrete stack symbol, but PUSH labels must match
        // only explicitly. The reference expansion never duplicates a DEFAULT
        // edge onto negative labels, so agreement pins this down.
        let mut nfa = NFA::new_empty();
        let start = nfa.add_state();
        let tail = nfa.add_state();
        let accept = nfa.add_state();
        nfa.start_states = vec![start];
        nfa.add_transition(start, DEFAULT, tail);
        nfa.add_transition(tail, -7, accept);
        nfa.set_accepting(accept);
        let dfa = try_determinize_minimal_defaults(&nfa, DEFAULT).unwrap();
        let reference = reference(&nfa, &[0, 1]);
        assert_eq!(
            words(&dfa, &[0, 1, -7], 3),
            words(&reference, &[0, 1, -7], 3),
        );
        // A row's DEFAULT edge must not be reachable through a negative label.
        assert!(!dfa.states[0].transitions.contains_key(&-7));
    }
}
