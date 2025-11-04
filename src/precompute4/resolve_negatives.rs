use crate::precompute4::full_dwa::Precomputed4;
use crate::precompute4::weighted_automata::{DWA, DWAState, DWAStates, StateID, Weight};
use std::collections::BTreeMap;

pub fn resolve_negative_codes_for_all(precomputed4: &mut Precomputed4) {
    for (_sid, dwa) in precomputed4.iter_mut() {
        resolve_negative_codes_in_dwa(dwa);
    }
}

fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    loop {
        let mut changed_in_pass = false;
        let mut all_new_states = Vec::with_capacity(dwa.states.len());

        for state_id in 0..dwa.states.len() {
            let (new_state, changed) =
                resolve_negative_codes_in_dwa_internal(state_id, &dwa.states);
            if changed {
                changed_in_pass = true;
            }
            all_new_states.push(new_state);
        }

        if !changed_in_pass {
            break;
        }

        dwa.states = DWAStates(all_new_states);
    }
    dwa.simplify();
}

fn resolve_negative_codes_in_dwa_internal(
    state_id: StateID,
    states: &DWAStates,
) -> (DWAState, bool) {
    let state_a = &states[state_id];

    let negative_edges: Vec<_> = state_a
        .transitions
        .exceptions
        .iter()
        .filter(|(k, _)| **k < 0)
        .map(|(k, v)| (*k, *v))
        .collect();

    if negative_edges.is_empty() {
        return (state_a.clone(), false);
    }

    // Create an empty state.
    let mut resolved_state = DWAState::default();
    resolved_state.weight = state_a.weight.clone();

    // Add all positive outgoing edges to it.
    // Add the default edge data and exceptions.
    resolved_state.transitions.default = state_a.transitions.default;
    resolved_state.trans_weight_default = state_a.trans_weight_default.clone();
    for (&on, &to) in &state_a.transitions.exceptions {
        if on >= 0 {
            resolved_state.transitions.exceptions.insert(on, to);
            if let Some(w) = state_a.trans_weights_exceptions.get(&on) {
                resolved_state.trans_weights_exceptions.insert(on, w.clone());
            }
        }
    }

    // Add its final weight.
    resolved_state.final_weight = state_a.final_weight.clone();

    // Collect inherited negative transitions to handle non-determinism.
    let mut inherited_negative_transitions: BTreeMap<i16, (StateID, Weight)> = BTreeMap::new();

    // Loop through negative edges A -> B
    for (neg_code, b_id) in negative_edges {
        let w_ab = state_a.trans_weights_exceptions.get(&neg_code).unwrap();
        let state_b = &states[b_id];

        // If the dst has final weight... intersect with the edge's weight, and put it in A.
        if let Some(fw_b) = &state_b.final_weight {
            let inherited_fw = fw_b & w_ab;
            if !inherited_fw.is_empty() {
                if let Some(fw_a) = &mut resolved_state.final_weight {
                    *fw_a |= &inherited_fw;
                } else {
                    resolved_state.final_weight = Some(inherited_fw);
                }
            }
        }

        // Keep any outgoing negative edges from B in-place.
        for (&b_neg_code, &b_target_id) in &state_b.transitions.exceptions {
            if b_neg_code < 0 {
                let w_bc = state_b.trans_weights_exceptions.get(&b_neg_code).unwrap();
                let combined_weight = w_ab & w_bc;

                if let Some((existing_target, existing_weight)) = inherited_negative_transitions.get_mut(&b_neg_code) {
                    if *existing_target != b_target_id {
                        panic!("Non-determinism introduced during negative code resolution: state {} receives conflicting transitions on code {} to states {} and {}", state_id, b_neg_code, *existing_target, b_target_id);
                    }
                    *existing_weight |= &combined_weight;
                } else {
                    inherited_negative_transitions.insert(b_neg_code, (b_target_id, combined_weight));
                }
            }
        }

        // Now for positive edges... B -> C
        let pos_code = neg_code.wrapping_sub(i16::MIN);

        // For the one that does match... merge it into this node.
        if let Some(&c_id) = state_b.transitions.get(pos_code) {
            let w_bc = state_b.trans_weights_exceptions.get(&pos_code)
                .or(state_b.trans_weight_default.as_ref())
                .unwrap();
            let state_c = &states[c_id];
            let combined_weight = w_ab & w_bc;

            let mut c_copy = state_c.clone();
            c_copy.apply_weight(&combined_weight);
            resolved_state.merge_union(&c_copy);
        }
    }

    for (code, (target, weight)) in inherited_negative_transitions {
        resolved_state.transitions.exceptions.insert(code, target);
        resolved_state.trans_weights_exceptions.insert(code, weight);
    }

    (resolved_state, true)
}
