//! # Tuple Merger Problem
//!
//! This module solves an abstract problem of merging state tuples from a product of several
//! component state machines. The goal is to find a minimal set of "merged states" that
//! covers all reachable product states, where merging is based on a compatibility relation.
//!
//! ## Problem Definition
//!
//! - We have `K` **components**. Each component `i` is a finite automaton with a set of states
//!   `S_i`, a start state `s_i_start`, and a transition function `d_i(state, symbol)`.
//!   Each component may have a designated **sink state**.
//!
//! - A **product state** is a tuple `(s_0, s_1, ..., s_{K-1})` where `s_i` is a state from
//!   component `i`. We represent this as `Vec<Option<usize>>`, where `None` indicates that the
//!   component is in its sink state.
//!
//! - Two product states (tuples) `T1` and `T2` are **compatible** if for every component `i`,
//!   either `T1[i] == T2[i]`, or at least one of them is `None` (sink).
//!
//! - A **merged state** is a set of mutually compatible product states. It can be uniquely
//!   represented by a **representative tuple**, which is the pointwise unification of all
//!   tuples in the set. Unification of `Some(s)` and `None` is `Some(s)`.
//!
//! - The task is to, given a `start_tuple`, explore all reachable product states and partition
//!   them into a minimal set of merged states. The output is a new automaton where states
//!   correspond to these merged states.
//!
//! ## Algorithm
//!
//! 1. Start with the `start_tuple`. Find or create a merged state for it.
//! 2. Maintain a worklist of merged states whose transitions have not been computed.
//! 3. For each merged state `M` on the worklist, take its representative tuple `R`.
//! 4. For each symbol in the alphabet, compute the successor tuple `R_succ` by applying the
//!    transition functions of all components to `R`.
//! 5. Find or create a merged state for `R_succ`. This defines the transition from `M` on that symbol.
//! 6. If a successor tuple is merged into an existing state `M'`, and this changes `M'`'s
//!    representative, `M'` must be added back to the worklist.
//! 7. Repeat until the worklist is empty.

#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap, VecDeque};

/// Represents one of the component automata in the product.
#[derive(Clone, Debug)]
pub struct Component {
    /// Total number of states in this component.
    pub num_states: usize,
    /// The start state ID.
    pub start_state: usize,
    /// A designated sink state, if one exists.
    pub sink_state: Option<usize>,
    /// Transition table: `transitions[state][symbol] -> next_state`.
    pub transitions: Vec<Vec<usize>>,
}

/// A state in the final merged automaton. It corresponds to a set of compatible product tuples.
#[derive(Clone, Debug)]
pub struct MergedState {
    /// The unique ID of this merged state.
    pub id: usize,
    /// The most specific tuple that represents all product tuples in this merged state.
    pub representative_tuple: ProductTuple,
    /// Transitions to other merged states: `transitions[symbol] -> merged_state_id`.
    pub transitions: Vec<usize>,
}

/// The final automaton built from merged states.
#[derive(Clone, Debug)]
pub struct MergedAutomaton {
    pub states: Vec<MergedState>,
    pub start_state_id: usize,
}

pub type ProductTuple = Vec<Option<usize>>;

/// Unifies two tuples pointwise. Returns `None` if they are incompatible.
/// Compatibility: for each position, either values are equal or one is `None`.
fn unify_tuples(a: &ProductTuple, b: &ProductTuple) -> Option<ProductTuple> {
    if a.len() != b.len() {
        return None;
    }
    let mut out = a.clone();
    for i in 0..a.len() {
        match (out[i], b[i]) {
            (Some(x), Some(y)) if x != y => return None,
            (None, Some(y)) => out[i] = Some(y),
            _ => {}
        }
    }
    Some(out)
}

/// Given a product tuple and a symbol, compute the successor tuple.
pub fn successor_tuple(
    tuple: &ProductTuple,
    symbol: usize,
    components: &[Component],
) -> ProductTuple {
    let k = components.len();
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        match tuple[i] {
            Some(s) => {
                let v = components[i].transitions[s][symbol];
                if components[i].sink_state == Some(v) {
                    out.push(None);
                } else {
                    out.push(Some(v));
                }
            }
            None => out.push(None),
        }
    }
    out
}

/// Internal state representation during the merging process.
#[derive(Debug)]
struct MergingState {
    representative_tuple: ProductTuple,
    // For debugging/verification, not strictly needed for the algorithm.
    contained_tuples: BTreeSet<ProductTuple>,
}

pub fn merge_and_build_automaton(
    start_tuple: ProductTuple,
    components: &[Component],
    alphabet_size: usize,
) -> MergedAutomaton {
    let mut merging_states: Vec<MergingState> = Vec::new();
    let mut tuple_to_state_id: HashMap<ProductTuple, usize> = HashMap::new();
    let mut worklist: VecDeque<usize> = VecDeque::new();

    // Create the initial state for the start_tuple.
    {
        let start_id = 0;
        merging_states.push(MergingState {
            representative_tuple: start_tuple.clone(),
            contained_tuples: BTreeSet::from([start_tuple.clone()]),
        });
        tuple_to_state_id.insert(start_tuple, start_id);
        worklist.push_back(start_id);
    }

    while let Some(state_id) = worklist.pop_front() {
        let representative = merging_states[state_id].representative_tuple.clone();

        for symbol in 0..alphabet_size {
            let succ_tuple = successor_tuple(&representative, symbol, components);

            if tuple_to_state_id.contains_key(&succ_tuple) {
                continue;
            }

            // Find a compatible existing state or create a new one.
            let mut placed = false;
            for existing_id in 0..merging_states.len() {
                let old_rep = &merging_states[existing_id].representative_tuple;
                if let Some(new_rep) = unify_tuples(old_rep, &succ_tuple) {
                    if new_rep != *old_rep {
                        merging_states[existing_id].representative_tuple = new_rep;
                        if !worklist.contains(&existing_id) {
                            worklist.push_back(existing_id);
                        }
                    }
                    tuple_to_state_id.insert(succ_tuple.clone(), existing_id);
                    merging_states[existing_id].contained_tuples.insert(succ_tuple.clone());
                    placed = true;
                    break;
                }
            }

            if !placed {
                let new_id = merging_states.len();
                merging_states.push(MergingState {
                    representative_tuple: succ_tuple.clone(),
                    contained_tuples: BTreeSet::from([succ_tuple.clone()]),
                });
                tuple_to_state_id.insert(succ_tuple, new_id);
                worklist.push_back(new_id);
            }
        }
    }

    // Finalize: build the MergedAutomaton with computed transitions.
    let mut final_states = Vec::with_capacity(merging_states.len());
    for (id, state) in merging_states.iter().enumerate() {
        let mut transitions = Vec::with_capacity(alphabet_size);
        for symbol in 0..alphabet_size {
            let succ_tuple = successor_tuple(&state.representative_tuple, symbol, components);
            // After the main loop, every reachable tuple must have an assigned state.
            let dest_id = *tuple_to_state_id.get(&succ_tuple).unwrap();
            transitions.push(dest_id);
        }
        final_states.push(MergedState {
            id,
            representative_tuple: state.representative_tuple.clone(),
            transitions,
        });
    }

    MergedAutomaton {
        states: final_states,
        start_state_id: 0, // By construction, the start state is always ID 0.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unify_tuples() {
        assert_eq!(unify_tuples(&vec![Some(1), None], &vec![None, Some(2)]), Some(vec![Some(1), Some(2)]));
        assert_eq!(unify_tuples(&vec![Some(1), Some(2)], &vec![Some(1), Some(2)]), Some(vec![Some(1), Some(2)]));
        assert_eq!(unify_tuples(&vec![Some(1), None], &vec![Some(1), Some(3)]), Some(vec![Some(1), Some(3)]));
        assert_eq!(unify_tuples(&vec![Some(1), Some(2)], &vec![Some(1), Some(3)]), None);
        assert_eq!(unify_tuples(&vec![None, None], &vec![Some(1), Some(2)]), Some(vec![Some(1), Some(2)]));
    }

    #[test]
    fn test_simple_merge() {
        // Component 0: 2 states (0=start, 1=sink). 0 -> 0 on 'a', 0 -> 1 on 'b'.
        let comp0 = Component {
            num_states: 2,
            start_state: 0,
            sink_state: Some(1),
            transitions: vec![vec![0, 1], vec![1, 1]], // s0: a->s0, b->s1; s1: a->s1, b->s1
        };
        // Component 1: 2 states (0=start, 1=sink). 0 -> 1 on 'a', 0 -> 0 on 'b'.
        let comp1 = Component {
            num_states: 2,
            start_state: 0,
            sink_state: Some(1),
            transitions: vec![vec![1, 0], vec![1, 1]], // s0: a->s1, b->s0; s1: a->s1, b->s1
        };
        let components = vec![comp0, comp1];
        let alphabet_size = 2; // 'a', 'b'

        // Start tuple: [Some(0), Some(0)]
        let start_tuple = vec![Some(0), Some(0)];

        let automaton = merge_and_build_automaton(start_tuple, &components, alphabet_size);

        // Expected states:
        // S0 (start): rep=[0,0]. a -> [0,None], b -> [None,0]
        // S1: rep=[0,None]. a -> [0,None], b -> [None,None]
        // S2: rep=[None,0]. a -> [None,None], b -> [None,0]
        // S3: rep=[None,None]. a -> [None,None], b -> [None,None]
        //
        // Let's trace:
        // 1. Start with S0=[0,0].
        //    - on 'a', succ is [0, None]. Create S1 for it. S0.trans[0] = S1.
        //    - on 'b', succ is [None, 0]. Create S2 for it. S0.trans[1] = S2.
        // 2. Process S1=[0,None].
        //    - on 'a', succ is [0, None]. Already have S1. S1.trans[0] = S1.
        //    - on 'b', succ is [None, None]. Create S3 for it. S1.trans[1] = S3.
        // 3. Process S2=[None,0].
        //    - on 'a', succ is [None, None]. Already have S3. S2.trans[0] = S3.
        //    - on 'b', succ is [None, 0]. Already have S2. S2.trans[1] = S2.
        // 4. Process S3=[None,None].
        //    - on 'a', succ is [None, None]. Already have S3. S3.trans[0] = S3.
        //    - on 'b', succ is [None, None]. Already have S3. S3.trans[1] = S3.
        // Total 4 states.

        assert_eq!(automaton.states.len(), 4);

        let s0_id = automaton.start_state_id;
        assert_eq!(automaton.states[s0_id].representative_tuple, vec![Some(0), Some(0)]);

        let s1_id = automaton.states[s0_id].transitions[0];
        assert_eq!(automaton.states[s1_id].representative_tuple, vec![Some(0), None]);

        let s2_id = automaton.states[s0_id].transitions[1];
        assert_eq!(automaton.states[s2_id].representative_tuple, vec![None, Some(0)]);

        let s3_id = automaton.states[s1_id].transitions[1];
        assert_eq!(automaton.states[s3_id].representative_tuple, vec![None, None]);

        // Check other transitions
        assert_eq!(automaton.states[s1_id].transitions[0], s1_id);
        assert_eq!(automaton.states[s2_id].transitions[0], s3_id);
        assert_eq!(automaton.states[s2_id].transitions[1], s2_id);
        assert_eq!(automaton.states[s3_id].transitions[0], s3_id);
        assert_eq!(automaton.states[s3_id].transitions[1], s3_id);
    }
}
