//! Minimization for acyclic unweighted DFAs.
//!
//! Uses reverse-topological signature-based merging under the crate's
//! partial-DFA semantics: missing transitions are treated as transitions to
//! a shared implicit rejecting sink. Processing in reverse-topological order
//! guarantees that children are classified before their parents.

use std::collections::HashMap;

use super::dfa::DFA;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StateSignature {
    is_accepting: bool,
    /// (label, equivalence-class of target)
    transitions: Vec<(i32, usize)>,
}

fn reverse_topological_order(dfa: &DFA) -> Vec<usize> {
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
    let mut order = Vec::new();
    dfs(dfa.start_state as usize, dfa, &mut visited, &mut order);
    order
}

fn state_signature(
    state_id: usize,
    dfa: &DFA,
    class_of_state: &[usize],
    dead_class: usize,
) -> StateSignature {
    let state = &dfa.states[state_id];
    let transitions = state
        .transitions
        .iter()
        .filter_map(|(&label, &target)| {
            let target_class = class_of_state
                .get(target as usize)
                .copied()
                .unwrap_or(dead_class);
            (target_class != dead_class).then_some((label, target_class))
        })
        .collect();

    StateSignature {
        is_accepting: state.is_accepting,
        transitions,
    }
}

fn build_minimized_acyclic_dfa(
    dfa: &DFA,
    class_of_state: &[usize],
    class_representatives: &HashMap<usize, usize>,
    dead_class: usize,
) -> DFA {
    if class_of_state[dfa.start_state as usize] == dead_class {
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
                let target_class = class_of_state
                    .get(target as usize)
                    .copied()
                    .unwrap_or(dead_class);
                if target_class == dead_class {
                    None
                } else {
                    Some((label, class_to_state[&target_class]))
                }
            })
            .collect();
    }

    minimized
}

/// Reindex an already-minimized acyclic DFA into the exact state order used by
/// [`minimize_acyclic`].
///
/// `minimize_acyclic` assigns class IDs in reverse DFS topological order.  When
/// every reachable state is already a distinct non-dead equivalence class, the
/// signature/interner pass is unnecessary: this reindexing is its only effect.
/// Callers must guarantee that the input is minimal and has no explicit dead
/// state (the normal output of `minimize_acyclic` satisfies both conditions).
pub fn reindex_minimized_acyclic_dfa(dfa: &DFA) -> DFA {
    assert!(
        dfa.is_acyclic(),
        "reindex_minimized_acyclic_dfa: input DFA is cyclic"
    );
    if dfa.states.is_empty() {
        return dfa.clone();
    }

    debug_assert!(dfa
        .states
        .iter()
        .all(|state| state.is_accepting || !state.transitions.is_empty()));

    let topo = reverse_topological_order(dfa);
    let mut state_map = vec![u32::MAX; dfa.states.len()];
    for (new_state, &old_state) in topo.iter().enumerate() {
        state_map[old_state] = new_state as u32;
    }

    let mut reindexed = DFA::new();
    reindexed.states = vec![super::dfa::DFAState::default(); topo.len()];
    reindexed.start_state = state_map[dfa.start_state as usize];
    for (new_state, &old_state) in topo.iter().enumerate() {
        let old = &dfa.states[old_state];
        reindexed.states[new_state].is_accepting = old.is_accepting;
        reindexed.states[new_state].transitions = old
            .transitions
            .iter()
            .filter_map(|(&label, &target)| {
                let mapped_target = state_map.get(target as usize).copied().unwrap_or(u32::MAX);
                (mapped_target != u32::MAX).then_some((label, mapped_target))
            })
            .collect();
    }
    reindexed
}

/// Minimize an acyclic unweighted DFA by merging states with identical
/// signatures (acceptance + transition map modulo equivalence class).
///
/// Panics (debug) if the input is cyclic.
pub fn minimize_acyclic(dfa: &DFA) -> DFA {
    assert!(
        dfa.is_acyclic(),
        "minimize_acyclic: input DFA is cyclic"
    );

    if dfa.states.is_empty() {
        return dfa.clone();
    }

    let topo = reverse_topological_order(dfa);

    const DEAD_CLASS: usize = 0;
    let dead_signature = StateSignature {
        is_accepting: false,
        transitions: Vec::new(),
    };

    let mut signature_to_class = HashMap::<StateSignature, usize>::new();
    signature_to_class.insert(dead_signature.clone(), DEAD_CLASS);
    let mut class_of_state = vec![DEAD_CLASS; dfa.states.len()];
    let mut class_representatives = HashMap::<usize, usize>::new();
    let mut next_class_id = 1usize;

    for &state_id in &topo {
        let signature = state_signature(
            state_id,
            dfa,
            &class_of_state,
            DEAD_CLASS,
        );

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

    build_minimized_acyclic_dfa(
        dfa,
        &class_of_state,
        &class_representatives,
        DEAD_CLASS,
    )
}
