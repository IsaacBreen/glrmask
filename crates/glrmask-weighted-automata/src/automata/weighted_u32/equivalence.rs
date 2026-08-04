//! Exact equivalence checking for acyclic weighted DWAs.
//!
//! The product state retains both accumulated path weights. This is necessary:
//! two paths can reach the same pair of topology states while admitting
//! different token sets. Acyclicity bounds the exploration depth, so the
//! product is finite.

use std::collections::{BTreeSet, VecDeque};

use rustc_hash::FxHashSet;

use super::dwa::DWA;
use super::nwa::Label;
use crate::ds::weight::Weight;
use crate::GlrMaskError;

#[derive(Clone, Eq, PartialEq, Hash)]
struct ProductState {
    left_state: Option<u32>,
    right_state: Option<u32>,
    left_weight: Weight,
    right_weight: Weight,
}

fn final_weight(dwa: &DWA, state: Option<u32>, path_weight: &Weight) -> Weight {
    state
        .and_then(|state| dwa.states().get(state as usize))
        .and_then(|state| state.final_weight.as_ref())
        .map(|final_weight| path_weight.intersection(final_weight))
        .unwrap_or_else(Weight::empty)
}

fn advance(
    dwa: &DWA,
    state: Option<u32>,
    path_weight: &Weight,
    label: Label,
) -> (Option<u32>, Weight) {
    let Some(state) = state.and_then(|state| dwa.states().get(state as usize)) else {
        return (None, Weight::empty());
    };
    let Some((next_state, edge_weight)) = state.transitions.get(&label) else {
        return (None, Weight::empty());
    };
    let next_weight = path_weight.intersection(edge_weight);
    ((!next_weight.is_empty()).then_some(*next_state), next_weight)
}

fn outgoing_labels(dwa: &DWA, state: Option<u32>, labels: &mut BTreeSet<Label>) {
    let Some(state) = state.and_then(|state| dwa.states().get(state as usize)) else {
        return;
    };
    labels.extend(state.transitions.keys().copied());
}

/// Return a shortest distinguishing word, or `None` when the two acyclic DWAs
/// compute exactly the same weight for every word.
pub fn find_difference(left: &DWA, right: &DWA) -> Result<Option<Vec<Label>>, GlrMaskError> {
    if !left.is_acyclic() || !right.is_acyclic() {
        return Err(GlrMaskError::Compilation(
            "exact weighted DWA equivalence currently supports only acyclic inputs".into(),
        ));
    }

    let start = ProductState {
        left_state: Some(left.start_state()),
        right_state: Some(right.start_state()),
        left_weight: Weight::all(),
        right_weight: Weight::all(),
    };
    let mut seen = FxHashSet::default();
    seen.insert(start.clone());
    let mut queue = VecDeque::from([(start, Vec::<Label>::new())]);

    while let Some((current, word)) = queue.pop_front() {
        if final_weight(left, current.left_state, &current.left_weight)
            != final_weight(right, current.right_state, &current.right_weight)
        {
            return Ok(Some(word));
        }

        let mut labels = BTreeSet::new();
        outgoing_labels(left, current.left_state, &mut labels);
        outgoing_labels(right, current.right_state, &mut labels);
        for label in labels {
            let (left_state, left_weight) =
                advance(left, current.left_state, &current.left_weight, label);
            let (right_state, right_weight) =
                advance(right, current.right_state, &current.right_weight, label);
            if left_weight.is_empty() && right_weight.is_empty() {
                continue;
            }
            let next = ProductState {
                left_state,
                right_state,
                left_weight,
                right_weight,
            };
            if seen.insert(next.clone()) {
                let mut next_word = word.clone();
                next_word.push(label);
                queue.push_back((next, next_word));
            }
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use range_set_blaze::RangeSetBlaze;

    use super::*;

    fn tokens(tokens: &[u32]) -> Weight {
        Weight::from_per_tsid_token_sets(std::iter::once((
            0,
            RangeSetBlaze::from_iter(tokens.iter().copied().map(|token| token..=token)),
        )))
    }

    #[test]
    fn finds_difference_after_shared_prefix() {
        let mut left = DWA::new(1, 2);
        let left_end = left.add_state();
        left.add_transition(0, 7, left_end, tokens(&[0]));
        left.set_final_weight(left_end, tokens(&[0]));

        let mut right = DWA::new(1, 2);
        let right_end = right.add_state();
        right.add_transition(0, 7, right_end, tokens(&[1]));
        right.set_final_weight(right_end, tokens(&[1]));

        assert_eq!(find_difference(&left, &right).unwrap(), Some(vec![7]));
    }

    #[test]
    fn accepts_structurally_different_equivalent_dwas() {
        let weight = tokens(&[0, 1]);
        let mut left = DWA::new(1, 2);
        let left_end = left.add_state();
        left.add_transition(0, 7, left_end, weight.clone());
        left.set_final_weight(left_end, weight.clone());

        let mut right = DWA::new(1, 2);
        let right_mid = right.add_state();
        let right_end = right.add_state();
        right.add_transition(0, 7, right_mid, weight.clone());
        right.add_transition(right_mid, 8, right_end, weight.clone());
        right.set_final_weight(right_mid, weight.clone());
        right.set_final_weight(right_end, weight);

        let mut left_extended = left.clone();
        let left_extended_end = left_extended.add_state();
        left_extended.add_transition(left_end, 8, left_extended_end, tokens(&[0, 1]));
        left_extended.set_final_weight(left_extended_end, tokens(&[0, 1]));

        assert_eq!(find_difference(&left_extended, &right).unwrap(), None);
    }
}
