//! Subset-construction determinization: NFA → DFA (unweighted).
//!
//! Implements the classical powerset / subset construction algorithm to convert
//! an `NFA` (with epsilon transitions) into a deterministic `DFA`.
//!
//! The caller is responsible for asserting acyclicity of the input NFA when
//! that invariant is required.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use super::dfa::{DFA, Label};
use super::nfa::NFA;

fn subset_is_accepting(nfa: &NFA, subset: &[u32]) -> bool {
    subset.iter().any(|&state| nfa.states[state as usize].is_accepting)
}

fn gather_label_targets(nfa: &NFA, subset: &[u32]) -> BTreeMap<Label, BTreeSet<u32>> {
    let mut label_targets: BTreeMap<Label, BTreeSet<u32>> = BTreeMap::new();
    for &nfa_state in subset {
        for (&label, targets) in &nfa.states[nfa_state as usize].transitions {
            let entry = label_targets.entry(label).or_default();
            for &target in targets {
                entry.insert(target);
            }
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
    if let Some(&existing) = subset_map.get(&subset) {
        return existing;
    }
    let new_id = dfa.add_state();
    subset_map.insert(subset.clone(), new_id);
    worklist.push_back(subset);
    new_id
}

/// Compute the epsilon closure of a set of NFA states.
fn epsilon_closure(nfa: &NFA, seeds: &[u32]) -> BTreeSet<u32> {
    let mut closed = BTreeSet::new();
    let mut queue: VecDeque<u32> = seeds.iter().copied().collect();
    while let Some(s) = queue.pop_front() {
        if closed.insert(s) {
            for &dst in &nfa.states[s as usize].epsilons {
                if !closed.contains(&dst) {
                    queue.push_back(dst);
                }
            }
        }
    }
    closed
}

/// Determinize an NFA into a DFA using subset construction.
///
/// Every DFA state corresponds to a set of NFA states (its "subset").
pub fn determinize(nfa: &NFA) -> DFA {
    assert!(nfa.is_acyclic(), "determinize: input NFA is cyclic");

    if nfa.states.is_empty() || nfa.start_states.is_empty() {
        return DFA::new();
    }

    let mut dfa = DFA {
        states: Vec::new(),
        start_state: 0,
    };

    // Map from NFA-state-subset (as sorted Vec) → DFA state ID.
    let mut subset_map: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut worklist: VecDeque<Vec<u32>> = VecDeque::new();

    let start_closure = epsilon_closure(nfa, &nfa.start_states);
    let start_key: Vec<u32> = start_closure.iter().copied().collect();
    let start_id = dfa.add_state();
    dfa.start_state = start_id;
    subset_map.insert(start_key.clone(), start_id);
    worklist.push_back(start_key);

    while let Some(subset_key) = worklist.pop_front() {
        let dfa_state = subset_map[&subset_key];

        if subset_is_accepting(nfa, &subset_key) {
            dfa.set_accepting(dfa_state, true);
        }

        let label_targets = gather_label_targets(nfa, &subset_key);

        for (label, raw_targets) in label_targets {
            let closed = epsilon_closure(
                nfa,
                &raw_targets.iter().copied().collect::<Vec<_>>(),
            );
            let next_key: Vec<u32> = closed.iter().copied().collect();
            if next_key.is_empty() {
                continue;
            }

            let next_dfa_state =
                get_or_create_subset_state(&mut dfa, &mut subset_map, &mut worklist, next_key);
            dfa.add_transition(dfa_state, label, next_dfa_state);
        }
    }

    dfa
}
