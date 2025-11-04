use crate::precompute4::full_dwa::Precomputed4;
use crate::precompute4::weighted_automata::{DWA, DWAState, DWAStates, StateID, Weight};

pub fn resolve_negative_codes_for_all(precomputed4: &mut Precomputed4) {
    for (_sid, dwa) in precomputed4.iter_mut() {
        resolve_negative_codes_in_dwa(dwa);
    }
}

fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    // The core idea is to iteratively replace transitions on negative codes `-k`
    // with the behavior of their target state upon receiving the positive code `k`.
    // This is repeated until no negative-code transitions remain.

    let mut changed = true;
    while changed {
        // Find all states that have at least one outgoing negative transition.
        let states_with_neg_trans: Vec<StateID> = (0..dwa.states.len())
            .filter(|&id| dwa.states[id].transitions.exceptions.keys().any(|&k| k < 0))
            .collect();

        if states_with_neg_trans.is_empty() {
            changed = false;
            continue;
        }

        // We calculate all modifications for this pass based on the DWA's current state,
        // and then apply them all at once to avoid race conditions.
        let mut all_modifications: Vec<(StateID, DWAState)> = Vec::new();

        for id_a in states_with_neg_trans {
            let state_a = &dwa.states[id_a];

            // Collect all negative transitions from the current state.
            let neg_transitions: Vec<(i16, StateID, Weight)> = state_a
                .transitions
                .exceptions
                .iter()
                .filter(|(&on, _)| on < 0)
                .map(|(&on, &id_b)| {
                    let weight = state_a
                        .trans_weights_exceptions
                        .get(&on)
                        .cloned()
                        .unwrap_or_else(Weight::zeros);
                    (on, id_b, weight)
                })
                .collect();

            // Start building the new state for `A` by cloning it and removing the
            // negative transitions that we are about to resolve.
            let mut new_state_a = state_a.clone();
            new_state_a.transitions.exceptions.retain(|&on, _| on >= 0);
            new_state_a.trans_weights_exceptions.retain(|&on, _| on >= 0);

            // For each negative transition, expand it.
            for (on, id_b, w_ab) in neg_transitions {
                let k = -on;
                let state_b = &dwa.states[id_b];

                // If state B is final, its finality (gated by the transition weight)
                // is transferred to state A.
                if let Some(fw_b) = &state_b.final_weight {
                    let new_fw = fw_b & &w_ab;
                    if !new_fw.is_empty() {
                        if let Some(fw_a) = &mut new_state_a.final_weight {
                            *fw_a |= &new_fw;
                        } else {
                            new_state_a.final_weight = Some(new_fw);
                        }
                    }
                }

                // Find the matching positive transition from B on code `k`.
                if let Some(&id_c) = state_b.transitions.get(k) {
                    // Get the weight of the B->C transition.
                    let w_bc = state_b
                        .trans_weights_exceptions
                        .get(&k)
                        .or(state_b.trans_weight_default.as_ref())
                        .cloned()
                        .unwrap_or_else(Weight::zeros);

                    let state_c = &dwa.states[id_c];
                    let path_weight = &w_ab & &w_bc;

                    // Create a temporary copy of state C, with all its weights
                    // (state, final, and outgoing transitions) gated by the path weight.
                    let mut weighted_state_c = state_c.clone();
                    weighted_state_c.apply_weight(&path_weight);

                    // Merge the behavior of this weighted state C into our new state A.
                    new_state_a.merge_union(&weighted_state_c);
                }
            }
            all_modifications.push((id_a, new_state_a));
        }

        // Apply all computed modifications to the DWA.
        for (id, new_state) in all_modifications {
            dwa.states[id] = new_state;
        }
    }
}

