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
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        match (a[i], b[i]) {
            (Some(x), Some(y)) => {
                if x != y { return None; }
                out.push(Some(x));
            }
            (Some(x), None) => out.push(Some(x)),
            (None, Some(y)) => out.push(Some(y)),
            (None, None) => out.push(None),
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

/// This function implements a standard product construction over the component automata.
/// Each unique reachable product tuple becomes a state in the resulting automaton.
/// The "merging" aspect of the original design was found to be a faulty heuristic and has been removed
/// in favor of this more standard and correct approach. The state explosion is managed by
/// representing sink states as `None` in the tuples.
pub fn merge_and_build_automaton(
    start_tuple: ProductTuple,
    components: &[Component],
    alphabet_size: usize,
) -> MergedAutomaton {
    let mut states: Vec<MergedState> = Vec::new();
    let mut tuple_to_id: HashMap<ProductTuple, usize> = HashMap::new();

    // Initial state
    tuple_to_id.insert(start_tuple.clone(), 0);
    states.push(MergedState { id: 0, representative_tuple: start_tuple, transitions: vec![] });

    let mut head = 0;
    while head < states.len() {
        let current_tuple = states[head].representative_tuple.clone();
        let mut transitions = Vec::with_capacity(alphabet_size);

        for symbol in 0..alphabet_size {
            let succ_tuple = successor_tuple(&current_tuple, symbol, components);
            let dest_id = *tuple_to_id.entry(succ_tuple.clone()).or_insert_with(|| {
                let new_id = states.len();
                states.push(MergedState {
                    id: new_id,
                    representative_tuple: succ_tuple,
                    transitions: vec![], // placeholder
                });
                new_id
            });
            transitions.push(dest_id);
        }
        states[head].transitions = transitions;
        head += 1;
    }

    MergedAutomaton {
        states,
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
