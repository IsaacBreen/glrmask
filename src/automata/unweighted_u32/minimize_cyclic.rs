//! Minimization for possibly-cyclic unweighted DFAs.
//!
//! Implements Hopcroft-style partition refinement. Unlike
//! `minimize_acyclic`, this works on DFAs with cycles (self-loops,
//! back-edges, etc.) at the cost of slightly higher constant factors.

use std::collections::{BTreeSet, HashMap};

use super::dfa::{DFA, DFAState, Label};

/// Minimize a (possibly cyclic) unweighted DFA via partition refinement.
///
/// Unreachable states are pruned. The returned DFA is language-equivalent
/// to the input with the fewest possible states.
pub fn minimize_cyclic(dfa: &DFA) -> DFA {
    if dfa.states.is_empty() {
        return dfa.clone();
    }

    // ---- 1. Collect reachable states ----
    let reachable = reachable_states(dfa);
    if reachable.is_empty() {
        return DFA::new();
    }

    // ---- 2. Collect the alphabet (set of all labels) ----
    let alphabet: Vec<Label> = {
        let mut labels = BTreeSet::new();
        for &sid in &reachable {
            for &label in dfa.states[sid].transitions.keys() {
                labels.insert(label);
            }
        }
        labels.into_iter().collect()
    };

    // Map reachable state indices to dense ids for the partition.
    let mut state_to_dense: HashMap<usize, usize> = HashMap::new();
    let mut dense_to_state: Vec<usize> = Vec::new();
    for &sid in &reachable {
        state_to_dense.insert(sid, dense_to_state.len());
        dense_to_state.push(sid);
    }
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

    // ---- 3. Initial partition: {accepting ∩ reachable, non-accepting ∩ reachable} ----
    // We also include DEAD as a separate class if any transition targets it.
    let mut class_of = vec![0usize; n + 1]; // +1 for DEAD
    let dead_class: usize = 0;
    class_of[dead] = dead_class;

    let mut accept_ids = Vec::new();
    let mut reject_ids = Vec::new();
    for i in 0..n {
        if dfa.states[dense_to_state[i]].is_accepting {
            accept_ids.push(i);
        } else {
            reject_ids.push(i);
        }
    }

    // Assign initial class ids.
    // Class 0 = dead, class 1 = non-accepting (or first non-empty), class 2 = accepting.
    let mut next_class = 1usize;
    let reject_class = if reject_ids.is_empty() {
        dead_class // will be unused
    } else {
        let c = next_class;
        next_class += 1;
        c
    };
    let accept_class = if accept_ids.is_empty() {
        dead_class
    } else {
        let c = next_class;
        let _ = c; // suppress unused warning
        c
    };
    for &i in &reject_ids {
        class_of[i] = reject_class;
    }
    for &i in &accept_ids {
        class_of[i] = accept_class;
    }

    // ---- 4. Partition refinement via composite signatures ----
    // Each iteration computes a full signature per state:
    //   (current_class, target_class_for_label_0, target_class_for_label_1, ...)
    // States sharing the same signature stay in the same class.
    loop {
        let mut signature_to_class: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut new_class_of = vec![0usize; n + 1];
        let mut new_next_class = 0usize;

        for i in 0..=dead {
            let sig: Vec<usize> = if i == dead {
                // DEAD: unique signature = (dead_class, dead_class for all labels)
                let mut s = Vec::with_capacity(1 + alphabet.len());
                s.push(class_of[dead]);
                for _ in &alphabet {
                    s.push(class_of[dead]);
                }
                s
            } else {
                let mut s = Vec::with_capacity(1 + alphabet.len());
                s.push(class_of[i]);
                for &label in &alphabet {
                    s.push(class_of[target_dense(dense_to_state[i], label)]);
                }
                s
            };

            let class = if let Some(&c) = signature_to_class.get(&sig) {
                c
            } else {
                let c = new_next_class;
                new_next_class += 1;
                signature_to_class.insert(sig, c);
                c
            };
            new_class_of[i] = class;
        }

        if new_class_of == class_of {
            break;
        }
        class_of = new_class_of;
        // Update dead_class tracking.
    }
    let dead_class = class_of[dead];

    // ---- 5. Build minimized DFA ----
    // Map class ids to new state ids.
    let mut class_to_new_state: HashMap<usize, u32> = HashMap::new();
    let mut new_states: Vec<DFAState> = Vec::new();

    // Determine which classes are used (excluding dead).
    let start_class = class_of[state_to_dense[&(dfa.start_state as usize)]];
    // Ensure start class gets state 0 in the new DFA.
    class_to_new_state.insert(start_class, 0);
    new_states.push(DFAState::default());

    for i in 0..n {
        let c = class_of[i];
        if c == dead_class {
            continue;
        }
        if !class_to_new_state.contains_key(&c) {
            let id = new_states.len() as u32;
            class_to_new_state.insert(c, id);
            new_states.push(DFAState::default());
        }
    }

    // Fill in transitions and acceptance for each class representative.
    let mut filled = vec![false; new_states.len()];
    for i in 0..n {
        let c = class_of[i];
        if c == dead_class {
            continue;
        }
        let new_id = class_to_new_state[&c] as usize;
        if filled[new_id] {
            continue;
        }
        filled[new_id] = true;
        let orig = &dfa.states[dense_to_state[i]];
        new_states[new_id].is_accepting = orig.is_accepting;
        for (&label, &target) in &orig.transitions {
            let target_usize = target as usize;
            if let Some(&dense) = state_to_dense.get(&target_usize) {
                let target_class = class_of[dense];
                if target_class != dead_class {
                    if let Some(&new_target) = class_to_new_state.get(&target_class) {
                        new_states[new_id].transitions.insert(label, new_target);
                    }
                }
            }
        }
    }

    DFA {
        states: new_states,
        start_state: 0,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimize_single_state() {
        let dfa = DFA::new();
        let minimized = minimize_cyclic(&dfa);
        assert_eq!(minimized.num_states(), 1);
        assert!(!minimized.states[0].is_accepting);
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

        let minimized = minimize_cyclic(&dfa);
        assert_eq!(minimized.num_states(), 2);
    }

    #[test]
    fn test_minimize_self_loop_cycle() {
        // 0 --1--> 1 (accept), 1 --1--> 1 (self-loop)
        let mut dfa = DFA::new();
        let s1 = dfa.add_state();
        dfa.add_transition(0, 1, s1);
        dfa.add_transition(s1, 1, s1);
        dfa.set_accepting(s1, true);

        let minimized = minimize_cyclic(&dfa);
        assert_eq!(minimized.num_states(), 2);
        // start is non-accepting, s1-class is accepting with self-loop
        let accept_state = if minimized.states[0].is_accepting { 0 } else { 1 };
        assert!(minimized.states[accept_state].is_accepting);
        assert_eq!(
            minimized.states[accept_state].transitions.get(&1),
            Some(&(accept_state as u32))
        );
    }

    #[test]
    fn test_minimize_mutual_cycle() {
        // 0 --1--> 1, 1 --1--> 0, both accepting → should merge into 1 state
        let mut dfa = DFA::new();
        let s1 = dfa.add_state();
        dfa.add_transition(0, 1, s1);
        dfa.add_transition(s1, 1, 0);
        dfa.set_accepting(0, true);
        dfa.set_accepting(s1, true);

        let minimized = minimize_cyclic(&dfa);
        assert_eq!(minimized.num_states(), 1);
        assert!(minimized.states[0].is_accepting);
        assert_eq!(minimized.states[0].transitions.get(&1), Some(&0));
    }

    #[test]
    fn test_minimize_preserves_distinct_cycle_states() {
        // 0 --1--> 1 (accept), 1 --1--> 0 (non-accept) → should NOT merge
        let mut dfa = DFA::new();
        let s1 = dfa.add_state();
        dfa.add_transition(0, 1, s1);
        dfa.add_transition(s1, 1, 0);
        dfa.set_accepting(s1, true);

        let minimized = minimize_cyclic(&dfa);
        assert_eq!(minimized.num_states(), 2);
    }

    #[test]
    fn test_minimize_unreachable_pruned() {
        let mut dfa = DFA::new();
        let _unreachable = dfa.add_state();
        dfa.set_accepting(0, true);

        let minimized = minimize_cyclic(&dfa);
        assert_eq!(minimized.num_states(), 1);
        assert!(minimized.states[0].is_accepting);
    }
}
