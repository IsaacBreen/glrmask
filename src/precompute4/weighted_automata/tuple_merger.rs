//! # The Point Compatibility Partitioning Problem
//!
//! This module solves an abstract problem of partitioning a set of points (tuples) into a
//! minimal number of "compatible" sets. This is used in automata determinization to merge
//! states from a product construction, but the problem itself is self-contained.
//!
//! ## Problem Definition
//!
//! Let `K` be a number of components. For each component `i ∈ {0, ..., K-1}`, let `S_i` be a
//! finite set of component-states. We define an augmented state space `S'_i = S_i ∪ {⊥}`, where
//! `⊥` is a special "sink" or "bottom" element.
//!
//! A **point** is an element of the product space `P = S'_0 × S'_1 × ... × S'_{K-1}`.
//! A point is thus a tuple of length `K`, where each element is either a component-state or `⊥`.
//! In this implementation, `Point` is `Vec<Option<usize>>`, where `None` represents `⊥`.
//!
//! ### Compatibility and Joining
//!
//! Two points `p` and `q` are **compatible** if for every coordinate `i`, either `p[i] == q[i]`,
//! or at least one of them is `⊥`.
//!
//! The **join** (`∨`) of two compatible points `p` and `q` is their pointwise unification. The
//! result `r = p ∨ q` is defined as:
//! - `r[i] = p[i]` if `q[i] == ⊥`
//! - `r[i] = q[i]` if `p[i] == ⊥`
//! - `r[i] = p[i]` (which equals `q[i]`) otherwise.
//!
//! A set of points `C` is a **compatible set** if all pairs of points in `C` are compatible.
//! For any compatible set `C`, we can define its **representative** as `rep(C) = ⋁_{p ∈ C} p`.
//!
//! ### System Dynamics
//!
//! We are given a finite alphabet `Σ` and a **successor function** `Succ(point, symbol) -> Point`
//! which defines the system's dynamics.
//!
//! The **reachable set** `R` is the set of all points reachable from a given `start_point` by
//! applying the successor function with any sequence of symbols.
//!
//! ### The Goal
//!
//! The problem is to find a partition `Π = {C_1, C_2, ..., C_M}` of the reachable set `R` such that:
//! 1. Each `C_j` is a compatible set.
//! 2. The number of sets `M` is minimized.
//!
//! The output is a description of this partition and the dynamics between the sets, represented
//! as a new, smaller state machine where each state corresponds to a set `C_j`.
//!
//! ## Complexity and The Implemented Heuristic
//!
//! This partitioning problem is equivalent to the **graph coloring problem**. Consider an
//! "incompatibility graph" `G = (R, E)` where an edge `(p, q)` exists if `p` and `q` are
//! incompatible. A compatible set `C_j` is an **independent set** in `G`. A partition of `R`
//! into compatible sets is a valid coloring of `G`. Minimizing `M` is equivalent to finding the
//! chromatic number of `G`, a classic NP-hard problem.
//!
//! Given this, we implement a **greedy, online heuristic** to find a valid, and hopefully small,
//! partition. The algorithm explores the reachable set and assigns each new point to the first
//! compatible partition set it finds. If no such set exists, a new one is created.
//!
//! ### Correctness of the Heuristic
//!
//! The algorithm is guaranteed to produce a valid partition. A partition requires that every
//! point belongs to exactly one set, and that every set is compatible.
//! - **Partition Property**: The algorithm maintains a map from each discovered point to a
//!   single set ID, ensuring each point has exactly one home. The exploration process guarantees
//!   all reachable points are discovered and assigned.
//! - **Compatibility Property**: A point `p` is added to a set `C` only if it is compatible
//!   with `rep(C)`. Since `rep(C)` is the join of all points already in `C`, compatibility with
//!   the representative implies compatibility with all individual points that formed it.
//!   Thus, the new set `C ∪ {p}` remains a compatible set.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, VecDeque};

/// Represents one of the component systems in the product.
#[derive(Clone, Debug)]
pub struct Component {
    /// Sparse transition table: `transitions[state]` is a map from `symbol` to `next_state`.
    /// Any symbol not in the map is assumed to transition to the sink state (`None`).
    pub transitions: Vec<BTreeMap<usize, usize>>,
}

/// A state in the final merged automaton, corresponding to a set of compatible product tuples.
#[derive(Clone, Debug)]
pub struct MergedState {
    /// The unique ID of this merged state.
    pub id: usize,
    /// The most specific tuple that represents all product tuples in this merged state.
    pub representative_tuple: ProductTuple,
    /// Transitions to other merged states: `transitions[symbol] -> merged_state_id`.
    pub transitions: Vec<usize>,
}

/// The final automaton built from merged states, representing the partitioned system.
#[derive(Clone, Debug)]
pub struct MergedAutomaton {
    pub states: Vec<MergedState>,
    pub start_state_id: usize,
}

/// A point in the product space, `Vec<Option<usize>>`, where `None` is the sink state.
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
                if x == y {
                    out.push(Some(x));
                } else {
                    return None; // Incompatible
                }
            }
            (Some(x), None) => out.push(Some(x)),
            (None, Some(y)) => out.push(Some(y)),
            (None, None) => out.push(None),
        }
    }
    Some(out)
}

/// Given a product tuple and a symbol, compute the successor tuple.
pub fn successor_tuple(tuple: &ProductTuple, symbol: usize, components: &[Component]) -> ProductTuple {
    let k = components.len();
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        match tuple[i] {
            Some(s) => {
                // Look up in sparse map. If not found, it's a transition to the sink.
                if let Some(&v) = components[i].transitions[s].get(&symbol) {
                    out.push(Some(v));
                } else {
                    out.push(None);
                }
            }
            None => {
                out.push(None); // Sink state is absorbing.
            }
        }
    }
    out
}

/// Finds a compatible partition of the reachable state space using a greedy heuristic.
pub fn merge_and_build_automaton(
    start_tuple: ProductTuple,
    components: &[Component],
    alphabet_size: usize,
) -> MergedAutomaton {
    // `states` stores the representative tuple for each merged state ID.
    let mut states: Vec<ProductTuple> = Vec::new();
    // `point_map` maps a discovered point to the ID of the merged state it belongs to.
    let mut point_map: HashMap<ProductTuple, usize> = HashMap::new();
    // `worklist` contains IDs of merged states whose representatives have changed and need reprocessing.
    let mut worklist: VecDeque<usize> = VecDeque::new();

    // Create the initial state for the start_tuple.
    let start_id = 0;
    states.push(start_tuple.clone());
    point_map.insert(start_tuple, start_id);
    worklist.push_back(start_id);

    // Pass 1: Discover all merged states and their final representatives.
    while let Some(state_id) = worklist.pop_front() {
        let representative = states[state_id].clone();

        for symbol in 0..alphabet_size {
            let successor = successor_tuple(&representative, symbol, components);

            if point_map.contains_key(&successor) {
                continue; // This point has already been assigned to a merged state.
            }

            // Find a home for the new successor point by finding the first compatible state.
            let mut assigned_id = None;
            for id in 0..states.len() {
                if let Some(new_rep) = unify_tuples(&states[id], &successor) {
                    // This state is compatible. Merge the point in.
                    if new_rep != states[id] {
                        // The representative has become more specific.
                        states[id] = new_rep;
                        // This state must be re-processed as its successors may change.
                        if !worklist.contains(&id) {
                            worklist.push_back(id);
                        }
                    }
                    assigned_id = Some(id);
                    break; // Greedy choice: commit to the first compatible state found.
                }
            }

            // If no existing state was compatible, create a new one.
            let home_id = assigned_id.unwrap_or_else(|| {
                let new_id = states.len();
                states.push(successor.clone());
                worklist.push_back(new_id);
                new_id
            });

            point_map.insert(successor, home_id);
        }
    }

    // Pass 2: Build the final automaton structure with computed transitions.
    let mut final_states = Vec::with_capacity(states.len());
    for (id, rep) in states.iter().enumerate() {
        let transitions = (0..alphabet_size)
            .map(|symbol| {
                let succ = successor_tuple(rep, symbol, components);
                // This lookup must succeed. Every reachable successor from a final
                // representative must have been found and assigned a home in Pass 1.
                *point_map.get(&succ).expect("Successor point must have an assigned state")
            })
            .collect();

        final_states.push(MergedState { id, representative_tuple: rep.clone(), transitions });
    }

    MergedAutomaton { states: final_states, start_state_id: 0 }
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
        // Component 0: 2 states (0=start). 0 -> 0 on symbol 0. Transitions to sink on symbol 1.
        let comp0 = Component { transitions: vec![BTreeMap::from([(0, 0)])] };
        // Component 1: 2 states (0=start). 0 -> 0 on symbol 1. Transitions to sink on symbol 0.
        let comp1 = Component { transitions: vec![BTreeMap::from([(1, 0)])] };
        let components = vec![comp0, comp1];
        let alphabet_size = 2;

        // Start tuple: [Some(0), Some(0)]
        let start_tuple = vec![Some(0), Some(0)];

        let automaton = merge_and_build_automaton(start_tuple, &components, alphabet_size);

        // The reachable points are [0,0], [0,None], [None,0], and [None,None] (from sink).
        // All these points are mutually compatible. Their join is [0,0].
        // The greedy algorithm should therefore find a single merged state.
        assert_eq!(automaton.states.len(), 1);

        // Check that the single state has the correct representative and transitions.
        let s0_id = automaton.start_state_id;
        assert_eq!(automaton.states[s0_id].representative_tuple, vec![Some(0), Some(0)]);

        // Transition on symbol 0 from rep [0,0] gives succ [0,None].
        // [0,None] is compatible and merges into state 0. The transition should be a self-loop.
        assert_eq!(automaton.states[s0_id].transitions[0], s0_id);

        // Transition on symbol 1 from rep [0,0] gives succ [None,0].
        // [None,0] is compatible and merges into state 0. The transition should be a self-loop.
        assert_eq!(automaton.states[s0_id].transitions[1], s0_id);
    }
}
