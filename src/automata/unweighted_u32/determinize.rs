//! Subset-construction determinization: NFA → DFA (unweighted).
//!
//! Implements the classical powerset / subset construction algorithm to convert
//! an `NFA` (with epsilon transitions) into a deterministic `DFA`.
//!
//! The caller is responsible for asserting acyclicity of the input NFA when
//! that invariant is required.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use super::dfa::{DFA, Label};
use super::nfa::NFA;

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
/// Epsilon transitions are resolved via epsilon-closure.
pub fn determinize(nfa: &NFA) -> DFA {
    assert!(
        nfa.is_acyclic(),
        "determinize: input NFA is cyclic"
    );

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

        // Mark accepting if any NFA state in the subset is accepting.
        if subset_key
            .iter()
            .any(|&s| nfa.states[s as usize].is_accepting)
        {
            dfa.set_accepting(dfa_state, true);
        }

        // Gather all labeled transitions reachable from this subset.
        let mut label_targets: BTreeMap<Label, BTreeSet<u32>> = BTreeMap::new();
        for &nfa_state in &subset_key {
            for (&label, targets) in &nfa.states[nfa_state as usize].transitions {
                let entry = label_targets.entry(label).or_default();
                for &t in targets {
                    entry.insert(t);
                }
            }
        }

        // For each label, compute the epsilon-closed target subset.
        for (label, raw_targets) in label_targets {
            let closed = epsilon_closure(
                nfa,
                &raw_targets.iter().copied().collect::<Vec<_>>(),
            );
            let next_key: Vec<u32> = closed.iter().copied().collect();
            if next_key.is_empty() {
                continue;
            }

            let next_dfa_state = if let Some(&existing) = subset_map.get(&next_key) {
                existing
            } else {
                let new_id = dfa.add_state();
                subset_map.insert(next_key.clone(), new_id);
                worklist.push_back(next_key);
                new_id
            };
            dfa.add_transition(dfa_state, label, next_dfa_state);
        }
    }

    dfa
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinize_simple_chain() {
        // NFA: 0 --(1)--> 1 --(2)--> 2 [accept]
        let mut nfa = NFA::new();
        let s1 = nfa.add_state();
        let s2 = nfa.add_state();
        nfa.add_transition(0, 1, s1);
        nfa.add_transition(s1, 2, s2);
        nfa.set_accepting(s2);

        let dfa = determinize(&nfa);
        assert!(dfa.is_acyclic());
        // Start → s1 → s2 → 3 states in DFA
        assert_eq!(dfa.num_states(), 3);
        assert!(!dfa.states[dfa.start_state as usize].is_accepting);
    }

    #[test]
    fn test_determinize_nondeterminism() {
        // NFA: 0 --(1)--> 1, 0 --(1)--> 2, 1 --(2)--> 3 [accept], 2 --(3)--> 3 [accept]
        let mut nfa = NFA::new();
        let s1 = nfa.add_state();
        let s2 = nfa.add_state();
        let s3 = nfa.add_state();
        nfa.add_transition(0, 1, s1);
        nfa.add_transition(0, 1, s2); // non-deterministic
        nfa.add_transition(s1, 2, s3);
        nfa.add_transition(s2, 3, s3);
        nfa.set_accepting(s3);

        let dfa = determinize(&nfa);
        assert!(dfa.is_acyclic());

        // After subset construction:
        //   DFA start = {0}
        //   {0} --(1)--> {1,2}
        //   {1,2} --(2)--> {3}
        //   {1,2} --(3)--> {3}
        // So: 3 DFA states, the merged {1,2} state has outgoing labels 2 and 3.
        assert_eq!(dfa.num_states(), 3);

        // start is not accepting; follow label 1 to get to {1,2}
        let start = dfa.start_state as usize;
        assert!(!dfa.states[start].is_accepting);
        let merged = *dfa.states[start].transitions.get(&1).expect("label 1");
        assert!(!dfa.states[merged as usize].is_accepting);
        assert_eq!(dfa.states[merged as usize].transitions.len(), 2);

        // Both labels 2 and 3 from {1,2} reach the single accepting state {3}
        let via_2 = dfa.states[merged as usize].transitions[&2];
        let via_3 = dfa.states[merged as usize].transitions[&3];
        assert_eq!(via_2, via_3, "both labels should reach the same accept state");
        assert!(dfa.states[via_2 as usize].is_accepting);
    }

    #[test]
    fn test_determinize_with_epsilon() {
        // NFA: 0 --ε--> 1 --(5)--> 2 [accept]
        let mut nfa = NFA::new();
        let s1 = nfa.add_state();
        let s2 = nfa.add_state();
        nfa.add_epsilon(0, s1);
        nfa.add_transition(s1, 5, s2);
        nfa.set_accepting(s2);

        let dfa = determinize(&nfa);
        assert!(dfa.is_acyclic());
        // Start state's epsilon closure includes {0, 1}.
        // From {0,1} on label 5 → {2} [accept].
        assert_eq!(dfa.num_states(), 2);
        assert!(!dfa.states[dfa.start_state as usize].is_accepting);
    }

    #[test]
    fn test_determinize_empty_nfa() {
        let nfa = NFA::new_empty();
        let dfa = determinize(&nfa);
        // Should produce a trivial DFA
        assert!(dfa.is_acyclic());
    }

    #[test]
    fn test_determinize_shared_target_different_paths() {
        // Models the template-DFA scenario: two reduces to the same NT node
        // with different pop counts (0 and 1).
        //
        // NFA:
        //   0 --ε--> 1 --(+4)--> nt_A       (reduce pop_count=0)
        //   0 --ε--> 2 --(+4)--> 3 --(D)--> nt_A  (reduce pop_count=1)
        //
        // nt_A = state 4 (shared target)
        // D = some default label
        let mut nfa = NFA::new();
        let s1 = nfa.add_state(); // 1
        let s2 = nfa.add_state(); // 2
        let s3 = nfa.add_state(); // 3
        let nt_a = nfa.add_state(); // 4 = nt_A

        let label_plus4 = 4i32;
        let label_default = i32::MAX - 1;

        nfa.add_epsilon(0, s1);
        nfa.add_epsilon(0, s2);
        nfa.add_transition(s1, label_plus4, nt_a);
        nfa.add_transition(s2, label_plus4, s3);
        nfa.add_transition(s3, label_default, nt_a);

        // The NFA should be acyclic (no self-loop; nt_A is just a sink).
        assert!(nfa.is_acyclic());

        let dfa = determinize(&nfa);
        // The DFA should also be acyclic — no self-loop at nt_A.
        assert!(
            dfa.is_acyclic(),
            "DFA should be acyclic for shared-target template scenario"
        );
    }
}
