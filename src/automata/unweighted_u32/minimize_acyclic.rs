//! Minimization for acyclic unweighted DFAs.
//!
//! Uses reverse-topological signature-based merging under the crate's
//! partial-DFA semantics: missing transitions are treated as transitions to
//! a shared implicit rejecting sink. Processing in reverse-topological order
//! guarantees that children are classified before their parents.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::dfa::DFA;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StateSignature {
    is_accepting: bool,
    /// (label, equivalence-class of target)
    transitions: Vec<(i32, usize)>,
}

/// Minimize an acyclic unweighted DFA by merging states with identical
/// signatures (acceptance + transition map modulo equivalence class).
///
/// Panics (debug) if the input is cyclic.
pub fn minimize_acyclic(dfa: &DFA) -> DFA {
    debug_assert!(
        dfa.is_acyclic(),
        "minimize_acyclic: input DFA is cyclic"
    );

    if dfa.states.is_empty() {
        return dfa.clone();
    }

    // ---- Reverse-topological order via post-order DFS ----
    fn dfs(state_id: usize, dfa: &DFA, visited: &mut [bool], order: &mut Vec<usize>) {
        if visited[state_id] {
            return;
        }
        visited[state_id] = true;
        for &target in dfa.states[state_id].transitions.values() {
            let target = target as usize;
            if target < dfa.states.len() {
                dfs(target, dfa, visited, order);
            }
        }
        order.push(state_id);
    }

    let mut visited = vec![false; dfa.states.len()];
    let mut topo = Vec::new();
    dfs(dfa.start_state as usize, dfa, &mut visited, &mut topo);
    // `topo` is in reverse-topological order (leaves first).

    let reachable: HashSet<usize> = topo.iter().copied().collect();
    let labels: Vec<i32> = topo
        .iter()
        .flat_map(|&state_id| dfa.states[state_id].transitions.keys().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    const DEAD_CLASS: usize = 0;
    let dead_signature = StateSignature {
        is_accepting: false,
        transitions: labels.iter().map(|&label| (label, DEAD_CLASS)).collect(),
    };

    let mut signature_to_class = HashMap::<StateSignature, usize>::new();
    signature_to_class.insert(dead_signature.clone(), DEAD_CLASS);
    let mut class_of_state = vec![usize::MAX; dfa.states.len()];
    let mut class_representatives = HashMap::<usize, usize>::new();
    let mut next_class_id = 1usize;

    // Process leaves before parents.
    for &state_id in &topo {
        let state = &dfa.states[state_id];
        let transitions: Vec<(i32, usize)> = labels
            .iter()
            .map(|&label| {
                let target_class = state
                    .transitions
                    .get(&label)
                    .and_then(|&target| {
                        let target = target as usize;
                        reachable.contains(&target).then_some(class_of_state[target])
                    })
                    .unwrap_or(DEAD_CLASS);
                (label, target_class)
            })
            .collect();

        let signature = StateSignature {
            is_accepting: state.is_accepting,
            transitions,
        };

        let class_id = if let Some(&existing) = signature_to_class.get(&signature) {
            existing
        } else {
            let new_id = next_class_id;
            next_class_id += 1;
            signature_to_class.insert(signature, new_id);
            class_representatives.insert(new_id, state_id);
            new_id
        };
        class_of_state[state_id] = class_id;
    }

    // ---- Rebuild minimized DFA ----
    if class_of_state[dfa.start_state as usize] == DEAD_CLASS {
        return DFA::new();
    }

    let mut class_ids: Vec<usize> = class_representatives.keys().copied().collect();
    class_ids.sort_unstable();
    let class_to_state: HashMap<usize, u32> = class_ids
        .iter()
        .enumerate()
        .map(|(new_state, &class_id)| (class_id, new_state as u32))
        .collect();

    let mut minimized = DFA::new();
    // Replace the default first state.
    minimized.states = vec![super::dfa::DFAState::default(); class_ids.len()];
    minimized.start_state = class_to_state[&class_of_state[dfa.start_state as usize]];

    for &class_id in &class_ids {
        let repr_state_id = class_representatives[&class_id];
        let repr = &dfa.states[repr_state_id];
        let out_state = class_to_state[&class_id] as usize;
        minimized.states[out_state].is_accepting = repr.is_accepting;
        minimized.states[out_state].transitions = repr
            .transitions
            .iter()
            .filter_map(|(&label, &target)| {
                let target = target as usize;
                if !reachable.contains(&target) {
                    return None;
                }
                let target_class = class_of_state[target];
                if target_class == DEAD_CLASS {
                    None
                } else {
                    Some((label, class_to_state[&target_class]))
                }
            })
            .collect();
    }

    minimized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimize_single_state() {
        let dfa = DFA::new();
        let minimized = minimize_acyclic(&dfa);
        assert_eq!(minimized.num_states(), 1);
    }

    #[test]
    fn test_minimize_merges_equivalent_branches() {
        //  0 --1--> 1 (accept)
        //  0 --2--> 2 (accept)
        // States 1 and 2 are equivalent → should merge.
        let mut dfa = DFA::new();
        let s1 = dfa.add_state();
        let s2 = dfa.add_state();
        dfa.add_transition(0, 1, s1);
        dfa.add_transition(0, 2, s2);
        dfa.set_accepting(s1, true);
        dfa.set_accepting(s2, true);

        let minimized = minimize_acyclic(&dfa);
        // 2 classes: start (non-accept) + merged accept.
        assert_eq!(minimized.num_states(), 2);
    }

    #[test]
    fn test_minimize_preserves_distinct_states() {
        //  0 --1--> 1 (accept)
        //  0 --2--> 2 (non-accept, equivalent to implicit dead)
        let mut dfa = DFA::new();
        let s1 = dfa.add_state();
        let s2 = dfa.add_state();
        dfa.add_transition(0, 1, s1);
        dfa.add_transition(0, 2, s2);
        dfa.set_accepting(s1, true);

        let minimized = minimize_acyclic(&dfa);
        assert_eq!(minimized.num_states(), 2);
    }

    #[test]
    fn test_minimize_empty_dfa() {
        let dfa = DFA {
            states: Vec::new(),
            start_state: 0,
        };
        let minimized = minimize_acyclic(&dfa);
        assert_eq!(minimized.num_states(), 0);
    }

    #[test]
    fn test_minimize_collapses_all_rejecting_partial_dfa() {
        let mut dfa = DFA::new();
        let s1 = dfa.add_state();
        let s2 = dfa.add_state();
        dfa.add_transition(s1, 0, 0);
        dfa.add_transition(s1, 1, 0);
        dfa.add_transition(s2, 0, 0);
        dfa.add_transition(s2, 1, s1);
        dfa.start_state = s2;

        let minimized = minimize_acyclic(&dfa);
        assert_eq!(minimized.num_states(), 1);
        assert!(!minimized.states[0].is_accepting);
        assert!(minimized.states[0].transitions.is_empty());
    }

    #[test]
    fn test_minimize_omits_transition_to_dead_class() {
        let mut dfa = DFA::new();
        let reject = dfa.add_state();
        dfa.set_accepting(0, true);
        dfa.add_transition(0, 0, reject);

        let minimized = minimize_acyclic(&dfa);
        assert_eq!(minimized.num_states(), 1);
        assert!(minimized.states[0].is_accepting);
        assert!(minimized.states[0].transitions.is_empty());
    }
}
