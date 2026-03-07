//! Resolve negative parser-state labels in weighted NWAs.
//!
//! This is the `glrmask` analogue of `sep1/precompute4/resolve_negatives.rs`,
//! kept as a separate compiler module so cancellation semantics are not baked
//! into `parser_dwa.rs`.

use std::collections::{BTreeSet, HashMap, VecDeque};

use crate::automata::weighted::nwa::Nwa;
use crate::automata::weighted::weight::Weight;
use crate::compiler::labels::{DEFAULT_LABEL, is_negative_label, negative_to_positive_label};

pub(crate) fn compute_cancellations(nwa: &Nwa) -> Vec<(u32, u32, Weight)> {
    type QueryKey = (u32, i32);

    let mut queries: HashMap<u32, HashMap<QueryKey, Weight>> = HashMap::new();
    let mut worklist: VecDeque<(u32, u32, i32)> = VecDeque::new();
    let mut in_queue: BTreeSet<(u32, u32, i32)> = BTreeSet::new();
    let mut new_eps_from: HashMap<u32, HashMap<u32, Weight>> = HashMap::new();

    let enqueue = |worklist: &mut VecDeque<(u32, u32, i32)>,
                   in_queue: &mut BTreeSet<(u32, u32, i32)>,
                   s: u32,
                   a: u32,
                   c: i32| {
        if in_queue.insert((s, a, c)) {
            worklist.push_back((s, a, c));
        }
    };

    for (a, state) in nwa.states.iter().enumerate() {
        let a = a as u32;
        for (&label, targets) in &state.transitions {
            if !is_negative_label(label) {
                continue;
            }
            let c = negative_to_positive_label(label);
            for (b, weight) in targets {
                let query_weight = queries
                    .entry(*b)
                    .or_default()
                    .entry((a, c))
                    .or_insert_with(Weight::empty);
                if !weight.is_subset(query_weight) {
                    *query_weight = query_weight.union(weight);
                    enqueue(&mut worklist, &mut in_queue, *b, a, c);
                }
            }
        }
    }

    while let Some((s, a, c)) = worklist.pop_front() {
        in_queue.remove(&(s, a, c));
        let Some(w_as) = queries.get(&s).and_then(|m| m.get(&(a, c))).cloned() else {
            continue;
        };

        if let Some(existing_eps) = new_eps_from.get(&s) {
            let propagations: Vec<(u32, Weight)> = existing_eps
                .iter()
                .filter_map(|(&target, eps_w)| {
                    let prop_w = w_as.intersection(eps_w);
                    (!prop_w.is_empty()).then_some((target, prop_w))
                })
                .collect();
            for (target, prop_w) in propagations {
                let query_weight = queries
                    .entry(target)
                    .or_default()
                    .entry((a, c))
                    .or_insert_with(Weight::empty);
                if !prop_w.is_subset(query_weight) {
                    *query_weight = query_weight.union(&prop_w);
                    enqueue(&mut worklist, &mut in_queue, target, a, c);
                }
            }
        }

        let mut cancellation_updates: Vec<(u32, Weight)> = Vec::new();
        if let Some(pos_targets) = nwa.states[s as usize].transitions.get(&c) {
            for (target, weight) in pos_targets {
                let new_eps_w = w_as.intersection(weight);
                if !new_eps_w.is_empty() {
                    cancellation_updates.push((*target, new_eps_w));
                }
            }
        }
        if let Some(default_targets) = nwa.states[s as usize].transitions.get(&DEFAULT_LABEL) {
            for (target, weight) in default_targets {
                let new_eps_w = w_as.intersection(weight);
                if !new_eps_w.is_empty() {
                    cancellation_updates.push((*target, new_eps_w));
                }
            }
        }

        for (target, new_eps_w) in cancellation_updates {
            let eps_from_a = new_eps_from.entry(a).or_default();
            let combined_eps_w = {
                let eps_weight = eps_from_a
                    .entry(target)
                    .or_insert_with(Weight::empty);
                if new_eps_w.is_subset(eps_weight) {
                    eps_weight.clone()
                } else {
                    *eps_weight = eps_weight.union(&new_eps_w);
                    eps_weight.clone()
                }
            };

            let queries_at_a: Vec<((u32, i32), Weight)> = queries
                .get(&a)
                .map(|m| m.iter().map(|(k, v)| (*k, v.clone())).collect())
                .unwrap_or_default();
            for ((a_prime, c_prime), w_a_prime_a) in queries_at_a {
                let prop_w = w_a_prime_a.intersection(&combined_eps_w);
                if prop_w.is_empty() {
                    continue;
                }
                let query_weight = queries
                    .entry(target)
                    .or_default()
                    .entry((a_prime, c_prime))
                    .or_insert_with(Weight::empty);
                if !prop_w.is_subset(query_weight) {
                    *query_weight = query_weight.union(&prop_w);
                    enqueue(&mut worklist, &mut in_queue, target, a_prime, c_prime);
                }
            }
        }

        for (target, weight) in &nwa.states[s as usize].epsilons {
            let prop_w = w_as.intersection(weight);
            if prop_w.is_empty() {
                continue;
            }
            let query_weight = queries
                .entry(*target)
                .or_default()
                .entry((a, c))
                .or_insert_with(Weight::empty);
            if !prop_w.is_subset(query_weight) {
                *query_weight = query_weight.union(&prop_w);
                enqueue(&mut worklist, &mut in_queue, *target, a, c);
            }
        }
    }

    new_eps_from
        .into_iter()
        .flat_map(|(from, targets)| {
            targets
                .into_iter()
                .map(move |(to, weight)| (from, to, weight))
        })
        .collect()
}

pub(crate) fn apply_cancellations(nwa: &mut Nwa) {
    for (from, to, weight) in compute_cancellations(nwa) {
        nwa.add_epsilon(from, to, weight);
    }
}

pub(crate) fn apply_finality_fixpoint(nwa: &mut Nwa) {
    loop {
        let mut changed = false;
        let existing_finals: Vec<Option<Weight>> = nwa
            .states
            .iter()
            .map(|state| state.final_weight.clone())
            .collect();

        for sid in 0..nwa.states.len() {
            let mut propagated: Option<Weight> = None;

            for (&label, targets) in &nwa.states[sid].transitions {
                if label >= 0 && label != DEFAULT_LABEL {
                    continue;
                }
                for (target, weight) in targets {
                    if let Some(final_weight) = &existing_finals[*target as usize] {
                        let add = weight.intersection(final_weight);
                        if add.is_empty() {
                            continue;
                        }
                        propagated = Some(match propagated {
                            Some(current) => current.union(&add),
                            None => add,
                        });
                    }
                }
            }

            for (target, weight) in &nwa.states[sid].epsilons {
                if let Some(final_weight) = &existing_finals[*target as usize] {
                    let add = weight.intersection(final_weight);
                    if add.is_empty() {
                        continue;
                    }
                    propagated = Some(match propagated {
                        Some(current) => current.union(&add),
                        None => add,
                    });
                }
            }

            if let Some(add) = propagated {
                let state = &mut nwa.states[sid];
                let new_final = match &state.final_weight {
                    Some(current) => current.union(&add),
                    None => add,
                };
                if state
                    .final_weight
                    .as_ref()
                    .is_none_or(|current| !new_final.is_subset(current))
                {
                    state.final_weight = Some(new_final);
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }
}

pub(crate) fn remove_negative_transitions(nwa: &mut Nwa) {
    for state in &mut nwa.states {
        state.transitions.retain(|label, _| !is_negative_label(*label));
    }
}

pub(crate) fn remove_redundant_default_transitions(nwa: &mut Nwa) {
    let num_states = nwa.states.len();
    if num_states == 0 {
        return;
    }

    let mut is_terminal = vec![false; num_states];

    for (sid, state) in nwa.states.iter().enumerate() {
        let has_non_default_transitions = state
            .transitions
            .iter()
            .any(|(&label, targets)| label != DEFAULT_LABEL && !targets.is_empty());
        let has_epsilons = !state.epsilons.is_empty();
        let is_final = state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty());

        if !has_non_default_transitions && !has_epsilons && is_final {
            is_terminal[sid] = true;
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for (sid, state) in nwa.states.iter().enumerate() {
            if is_terminal[sid] {
                continue;
            }

            let has_non_default_transitions = state
                .transitions
                .iter()
                .any(|(&label, targets)| label != DEFAULT_LABEL && !targets.is_empty());
            let has_epsilons = !state.epsilons.is_empty();
            let is_final = state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty());

            if has_non_default_transitions || has_epsilons || !is_final {
                continue;
            }

            let all_default_targets_terminal = state
                .transitions
                .get(&DEFAULT_LABEL)
                .is_none_or(|targets| {
                    targets
                        .iter()
                        .all(|(target, _)| (*target as usize) < num_states && is_terminal[*target as usize])
                });

            if all_default_targets_terminal {
                is_terminal[sid] = true;
                changed = true;
            }
        }
    }

    for state in &mut nwa.states {
        if let Some(default_targets) = state.transitions.get_mut(&DEFAULT_LABEL) {
            default_targets.retain(|(target, _)| {
                (*target as usize) >= num_states || !is_terminal[*target as usize]
            });
        }
        state.transitions.retain(|_, targets| !targets.is_empty());
    }
}

pub(crate) fn resolve_negative_codes_in_nwa(nwa: &mut Nwa) {
    apply_cancellations(nwa);
    apply_finality_fixpoint(nwa);
    remove_negative_transitions(nwa);
    remove_redundant_default_transitions(nwa);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::weighted::weight::Weight;
    use crate::automata::weighted::weight::TokenSet;

    fn singleton_weight(token: u32) -> Weight {
        Weight::from_entries(vec![(0, 0, TokenSet::from_iter([token..=token]))])
    }

    #[test]
    fn removes_default_transition_to_terminal_final_state() {
        let mut nwa = Nwa::new(1, 3);
        let start = nwa.add_state();
        let end = nwa.add_state();
        nwa.start_states.push(start);

        let weight = singleton_weight(1);
        nwa.add_transition(start, DEFAULT_LABEL, end, weight.clone());
        nwa.set_final_weight(end, weight.clone());

        resolve_negative_codes_in_nwa(&mut nwa);

        assert_eq!(nwa.states[start as usize].final_weight.as_ref(), Some(&weight));
        assert!(!nwa.states[start as usize].transitions.contains_key(&DEFAULT_LABEL));
    }

    #[test]
    fn removes_default_only_chain_after_finality_propagation() {
        let mut nwa = Nwa::new(1, 3);
        let start = nwa.add_state();
        let mid = nwa.add_state();
        let end = nwa.add_state();
        nwa.start_states.push(start);

        let weight = singleton_weight(2);
        nwa.add_transition(start, DEFAULT_LABEL, mid, weight.clone());
        nwa.add_transition(mid, DEFAULT_LABEL, end, weight.clone());
        nwa.set_final_weight(end, weight.clone());

        resolve_negative_codes_in_nwa(&mut nwa);

        assert_eq!(nwa.states[start as usize].final_weight.as_ref(), Some(&weight));
        assert_eq!(nwa.states[mid as usize].final_weight.as_ref(), Some(&weight));
        assert!(!nwa.states[start as usize].transitions.contains_key(&DEFAULT_LABEL));
        assert!(!nwa.states[mid as usize].transitions.contains_key(&DEFAULT_LABEL));
    }

    #[test]
    fn keeps_default_transition_when_target_is_not_terminal() {
        let mut nwa = Nwa::new(1, 3);
        let start = nwa.add_state();
        let mid = nwa.add_state();
        let end = nwa.add_state();
        nwa.start_states.push(start);

        let weight = singleton_weight(0);
        nwa.add_transition(start, DEFAULT_LABEL, mid, weight.clone());
        nwa.add_transition(mid, 7, end, weight.clone());
        nwa.set_final_weight(end, weight.clone());

        resolve_negative_codes_in_nwa(&mut nwa);

        assert!(nwa.states[start as usize].transitions.contains_key(&DEFAULT_LABEL));
    }
}
