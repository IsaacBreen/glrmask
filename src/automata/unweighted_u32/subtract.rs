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

fn get_or_create_product_state(
    result: &mut DFA,
    state_ids: &mut HashMap<ProductState, u32>,
    worklist: &mut VecDeque<ProductState>,
    product_state: ProductState,
) -> u32 {
    if let Some(&existing) = state_ids.get(&product_state) {
        return existing;
    }
    let new_state = result.add_state();
    state_ids.insert(product_state, new_state);
    worklist.push_back(product_state);
    new_state
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
            let next_result_state = get_or_create_product_state(
                &mut result,
                &mut state_ids,
                &mut worklist,
                next_product,
            );
            result.add_transition(result_state, label, next_result_state);
        }
    }

    result
}
