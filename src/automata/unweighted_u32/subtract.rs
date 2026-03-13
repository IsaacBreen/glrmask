//! DFA subtraction: compute `left \ right` for unweighted DFAs.
//!
//! The resulting DFA accepts exactly the words accepted by `left` and not by
//! `right`. Missing transitions in `right` are treated as transitions to an
//! implicit non-accepting sink.

use std::collections::{HashMap, VecDeque};

use super::dfa::DFA;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ProductState {
    left: u32,
    right: Option<u32>,
}

pub fn subtract(left: &DFA, right: &DFA) -> DFA {
    if left.states.is_empty() {
        return left.clone();
    }

    let right_start = (!right.states.is_empty()).then_some(right.start_state);
    let start = ProductState {
        left: left.start_state,
        right: right_start,
    };

    let mut result = DFA {
        states: Vec::new(),
        start_state: 0,
    };
    let mut state_ids = HashMap::<ProductState, u32>::new();
    let mut worklist = VecDeque::<ProductState>::new();

    let start_id = result.add_state();
    result.start_state = start_id;
    state_ids.insert(start, start_id);
    worklist.push_back(start);

    while let Some(product) = worklist.pop_front() {
        let result_state = state_ids[&product];
        let left_state = &left.states[product.left as usize];
        let right_state = product.right.map(|state| &right.states[state as usize]);

        result.states[result_state as usize].is_accepting = left_state.is_accepting
            && !right_state.is_some_and(|state| state.is_accepting);

        for (&label, &left_next) in &left_state.transitions {
            let right_next = right_state.and_then(|state| state.transitions.get(&label).copied());
            let next_product = ProductState {
                left: left_next,
                right: right_next,
            };
            let next_result_state = if let Some(&existing) = state_ids.get(&next_product) {
                existing
            } else {
                let new_state = result.add_state();
                state_ids.insert(next_product, new_state);
                worklist.push_back(next_product);
                new_state
            };
            result.add_transition(result_state, label, next_result_state);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accepts(dfa: &DFA, word: &[i32]) -> bool {
        let mut state = dfa.start_state;
        for &label in word {
            let Some(&next) = dfa.states[state as usize].transitions.get(&label) else {
                return false;
            };
            state = next;
        }
        dfa.states[state as usize].is_accepting
    }

    #[test]
    fn test_subtract_removes_right_language() {
        let mut left = DFA::new();
        let left_mid = left.add_state();
        let left_accept_12 = left.add_state();
        let left_accept_13 = left.add_state();
        left.add_transition(0, 1, left_mid);
        left.add_transition(left_mid, 2, left_accept_12);
        left.add_transition(left_mid, 3, left_accept_13);
        left.set_accepting(left_accept_12, true);
        left.set_accepting(left_accept_13, true);

        let mut right = DFA::new();
        let right_mid = right.add_state();
        let right_accept = right.add_state();
        right.add_transition(0, 1, right_mid);
        right.add_transition(right_mid, 2, right_accept);
        right.set_accepting(right_accept, true);

        let result = subtract(&left, &right);
        assert!(!accepts(&result, &[1, 2]));
        assert!(accepts(&result, &[1, 3]));
    }

    #[test]
    fn test_subtract_treats_missing_right_transitions_as_rejecting() {
        let mut left = DFA::new();
        let accept = left.add_state();
        left.add_transition(0, 7, accept);
        left.set_accepting(accept, true);

        let right = DFA::new();

        let result = subtract(&left, &right);
        assert!(accepts(&result, &[7]));
    }
}