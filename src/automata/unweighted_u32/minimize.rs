//! Minimization for acyclic unweighted DFAs.
//!
//! Uses reverse-topological signature-based merging: two states are
//! equivalent when they have the same acceptance flag and identical
//! transition maps (after class substitution).  Processing in
//! reverse-topological order guarantees that children are classified
//! before their parents.

use std::collections::{HashMap, HashSet};

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
/// # Panics
///
/// Panics (via `debug_assert!`) if the input DFA is cyclic.
pub fn minimize(dfa: &DFA) -> DFA {
    debug_assert!(dfa.is_acyclic(), "minimize: input DFA must be acyclic");

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
    let mut signature_to_class = HashMap::<StateSignature, usize>::new();
    let mut class_of_state = vec![usize::MAX; dfa.states.len()];
    let mut class_representatives = Vec::<usize>::new();

    // Process leaves before parents.
    for &state_id in &topo {
        let state = &dfa.states[state_id];
        let mut transitions: Vec<(i32, usize)> = state
            .transitions
            .iter()
            .filter_map(|(&label, &target)| {
                let target = target as usize;
                reachable.contains(&target).then_some((label, class_of_state[target]))
            })
            .collect();
        transitions.sort_unstable();

        let signature = StateSignature {
            is_accepting: state.is_accepting,
            transitions,
        };

        let class_id = if let Some(&existing) = signature_to_class.get(&signature) {
            existing
        } else {
            let new_id = class_representatives.len();
            signature_to_class.insert(signature, new_id);
            class_representatives.push(state_id);
            new_id
        };
        class_of_state[state_id] = class_id;
    }

    // ---- Rebuild minimized DFA ----
    let mut minimized = DFA::new();
    // Replace the default first state.
    minimized.states = vec![super::dfa::DFAState::default(); class_representatives.len()];
    minimized.start_state = class_of_state[dfa.start_state as usize] as u32;

    for (class_id, &repr_state_id) in class_representatives.iter().enumerate() {
        let repr = &dfa.states[repr_state_id];
        minimized.states[class_id].is_accepting = repr.is_accepting;
        minimized.states[class_id].transitions = repr
            .transitions
            .iter()
            .filter_map(|(&label, &target)| {
                let target = target as usize;
                reachable.contains(&target).then_some((label, class_of_state[target] as u32))
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
        let minimized = minimize(&dfa);
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

        let minimized = minimize(&dfa);
        // 2 classes: start (non-accept) + merged accept.
        assert_eq!(minimized.num_states(), 2);
    }

    #[test]
    fn test_minimize_preserves_distinct_states() {
        //  0 --1--> 1 (accept)
        //  0 --2--> 2 (non-accept)
        let mut dfa = DFA::new();
        let s1 = dfa.add_state();
        let s2 = dfa.add_state();
        dfa.add_transition(0, 1, s1);
        dfa.add_transition(0, 2, s2);
        dfa.set_accepting(s1, true);

        let minimized = minimize(&dfa);
        assert_eq!(minimized.num_states(), 3);
    }

    #[test]
    fn test_minimize_empty_dfa() {
        let dfa = DFA {
            states: Vec::new(),
            start_state: 0,
        };
        let minimized = minimize(&dfa);
        assert_eq!(minimized.num_states(), 0);
    }
}
