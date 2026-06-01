//! Minimization for possibly-cyclic unweighted DFAs.
//!
//! Implements Hopcroft-style partition refinement. Unlike
//! `minimize_acyclic`, this works on DFAs with cycles (self-loops,
//! back-edges, etc.) at the cost of slightly higher constant factors.

use std::collections::{BTreeSet, HashMap};

use super::dfa::{DFA, DFAState, Label};

fn collect_reachable_alphabet(dfa: &DFA, reachable: &[usize]) -> Vec<Label> {
    let mut labels = BTreeSet::new();
    for &state_id in reachable {
        for &label in dfa.states[state_id].transitions.keys() {
            labels.insert(label);
        }
    }
    labels.into_iter().collect()
}

fn dense_reachable_states(reachable: &[usize]) -> (HashMap<usize, usize>, Vec<usize>) {
    let mut state_to_dense = HashMap::new();
    let mut dense_to_state = Vec::with_capacity(reachable.len());
    for &state_id in reachable {
        state_to_dense.insert(state_id, dense_to_state.len());
        dense_to_state.push(state_id);
    }
    (state_to_dense, dense_to_state)
}

fn initial_partition(dfa: &DFA, dense_to_state: &[usize], dead: usize) -> Vec<usize> {
    let mut class_of = vec![0usize; dense_to_state.len() + 1];
    class_of[dead] = 0;

    let mut next_class = 1usize;
    let reject_class = (!dense_to_state.is_empty()).then(|| {
        let class = next_class;
        next_class += 1;
        class
    });
    let accept_class = (!dense_to_state.is_empty()).then_some(next_class);

    for (dense_id, &state_id) in dense_to_state.iter().enumerate() {
        let class = if dfa.states[state_id].is_accepting {
            accept_class.unwrap_or(0)
        } else {
            reject_class.unwrap_or(0)
        };
        class_of[dense_id] = class;
    }

    class_of
}

fn refine_partition(
    class_of: &[usize],
    dead: usize,
    alphabet: &[Label],
    dense_to_state: &[usize],
    target_dense: impl Fn(usize, Label) -> usize,
) -> Vec<usize> {
    let mut signature_to_class = HashMap::<Vec<usize>, usize>::new();
    let mut new_class_of = vec![0usize; dense_to_state.len() + 1];
    let mut next_class = 0usize;

    for dense_id in 0..=dead {
        let mut signature = Vec::with_capacity(1 + alphabet.len());
        signature.push(class_of[dense_id]);
        if dense_id == dead {
            signature.extend(std::iter::repeat(class_of[dead]).take(alphabet.len()));
        } else {
            for &label in alphabet {
                signature.push(class_of[target_dense(dense_to_state[dense_id], label)]);
            }
        }

        let class = if let Some(&existing) = signature_to_class.get(&signature) {
            existing
        } else {
            let class = next_class;
            next_class += 1;
            signature_to_class.insert(signature, class);
            class
        };
        new_class_of[dense_id] = class;
    }

    new_class_of
}

fn build_minimized_cyclic_dfa(
    dfa: &DFA,
    state_to_dense: &HashMap<usize, usize>,
    dense_to_state: &[usize],
    class_of: &[usize],
    dead_class: usize,
) -> DFA {
    let start_class = class_of[state_to_dense[&(dfa.start_state as usize)]];
    let mut class_to_new_state = HashMap::<usize, u32>::new();
    let mut new_states = vec![DFAState::default()];
    class_to_new_state.insert(start_class, 0);

    for (dense_id, _) in dense_to_state.iter().enumerate() {
        let class = class_of[dense_id];
        if class == dead_class || class_to_new_state.contains_key(&class) {
            continue;
        }
        let new_state = new_states.len() as u32;
        class_to_new_state.insert(class, new_state);
        new_states.push(DFAState::default());
    }

    let mut filled = vec![false; new_states.len()];
    for (dense_id, &state_id) in dense_to_state.iter().enumerate() {
        let class = class_of[dense_id];
        if class == dead_class {
            continue;
        }

        let new_state = class_to_new_state[&class] as usize;
        if filled[new_state] {
            continue;
        }
        filled[new_state] = true;

        let original = &dfa.states[state_id];
        new_states[new_state].is_accepting = original.is_accepting;
        for (&label, &target) in &original.transitions {
            let Some(&target_dense) = state_to_dense.get(&(target as usize)) else {
                continue;
            };
            let target_class = class_of[target_dense];
            if target_class == dead_class {
                continue;
            }
            if let Some(&new_target) = class_to_new_state.get(&target_class) {
                new_states[new_state].transitions.insert(label, new_target);
            }
        }
    }

    DFA {
        states: new_states,
        start_state: 0,
    }
}

/// Minimize a (possibly cyclic) unweighted DFA via partition refinement.
///
/// Unreachable states are pruned. The returned DFA is language-equivalent
/// to the input with the fewest possible states.
pub fn minimize_cyclic(dfa: &DFA) -> DFA {
    if dfa.states.is_empty() {
        return dfa.clone();
    }

    // Collect reachable states.
    let reachable = reachable_states(dfa);
    if reachable.is_empty() {
        return DFA::new();
    }

    let alphabet = collect_reachable_alphabet(dfa, &reachable);

    let (state_to_dense, dense_to_state) = dense_reachable_states(&reachable);
    let n = dense_to_state.len();

    // Use an implicit DEAD sink for missing transitions (dense id = n).
    let dead = n;

    // Resolve a transition target to a dense id (DEAD if missing or unreachable).
    let target_dense = |state_idx: usize, label: Label| -> usize {
        dfa.states[state_idx]
            .transitions
            .get(&label)
            .and_then(|&t| state_to_dense.get(&(t as usize)).copied())
            .unwrap_or(dead)
    };

    let mut class_of = initial_partition(dfa, &dense_to_state, dead);

    // Refine partitions with composite signatures.
    loop {
        let new_class_of = refine_partition(&class_of, dead, &alphabet, &dense_to_state, target_dense);

        if new_class_of == class_of {
            break;
        }
        class_of = new_class_of;
    }
    let dead_class = class_of[dead];

    build_minimized_cyclic_dfa(dfa, &state_to_dense, &dense_to_state, &class_of, dead_class)
}

/// Collect indices of states reachable from `start_state`.
fn reachable_states(dfa: &DFA) -> Vec<usize> {
    let mut visited = vec![false; dfa.states.len()];
    let mut stack = vec![dfa.start_state as usize];
    let mut result = Vec::new();
    while let Some(s) = stack.pop() {
        if s >= dfa.states.len() || visited[s] {
            continue;
        }
        visited[s] = true;
        result.push(s);
        for &target in dfa.states[s].transitions.values() {
            let t = target as usize;
            if t < dfa.states.len() && !visited[t] {
                stack.push(t);
            }
        }
    }
    result.sort_unstable();
    result
}
