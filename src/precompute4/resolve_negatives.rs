use crate::precompute4::full_dwa::Precomputed4;
use crate::precompute4::weighted_automata::{DWA, DWAState, StateID, Weight};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub fn resolve_negative_codes_for_all(precomputed4: &mut Precomputed4) {
    for (_sid, dwa) in precomputed4.iter_mut() {
        resolve_negative_codes_in_dwa(dwa);
    }
}

fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    // Stage 1: Isolate final states targeted by negative transitions to simplify later logic.
    isolate_mixed_final_states(dwa);

    // Stage 2: Iteratively resolve A -(-x)-> B -(x)-> C patterns by merging C's behavior into A.
    propagate_cancellations(dwa);

    // Stage 3: Remove remaining internal negative transitions that don't lead to a final state.
    remove_internal_negatives(dwa);

    // Finally, simplify the automaton to merge equivalent states and remove unreachable ones.
    dwa.simplify();
}

/// If a negative transition `A -(-x)-> B` exists where B is final but also has outgoing
/// transitions, this is ambiguous. We resolve this by creating a new, "pure" final state `B'`
/// with B's final properties but no outgoing transitions, and retargeting the edge to `A -(-x)-> B'`.
fn isolate_mixed_final_states(dwa: &mut DWA) {
    let mut updates = Vec::new(); // (from_state, neg_code, new_target)

    for from_id in 0..dwa.states.len() {
        let from_state = dwa.states[from_id].clone();
        for (&neg_code, &target_id) in &from_state.transitions.exceptions {
            if neg_code >= 0 {
                continue;
            }

            let target_state = &dwa.states[target_id];
            let is_final = target_state.final_weight.is_some();
            let has_outgoing = target_state.transitions.default.is_some() || !target_state.transitions.exceptions.is_empty();

            if is_final && has_outgoing {
                let new_state = DWAState {
                    weight: target_state.weight.clone(),
                    final_weight: target_state.final_weight.clone(),
                    ..DWAState::default()
                };
                let new_id = dwa.states.add_state();
                dwa.states[new_id] = new_state;
                updates.push((from_id, neg_code, new_id));
            }
        }
    }

    for (from_id, neg_code, new_target_id) in updates {
        let from_state = &mut dwa.states[from_id];
        from_state.transitions.exceptions.insert(neg_code, new_target_id);
    }
}

/// Iteratively processes `A -(-x)-> B -(x)-> C` patterns. This is interpreted as an
/// effective epsilon-transition from A to C. We realize this by merging C's behaviors
/// (weights, finality, transitions) into A, gated by the weight of the `A -> B -> C` path.
/// This process is repeated until no more changes occur.
fn propagate_cancellations(dwa: &mut DWA) {
    let mut changed = true;
    while changed {
        changed = false;

        let mut weight_updates: BTreeMap<StateID, Weight> = BTreeMap::new();
        let mut final_weight_updates: BTreeMap<StateID, Weight> = BTreeMap::new();
        let mut transition_additions = Vec::new();

        for a_id in 0..dwa.states.len() {
            let a_state = dwa.states[a_id].clone();
            for (&neg_code, &b_id) in &a_state.transitions.exceptions {
                if neg_code >= 0 {
                    continue;
                }
                let pos_code = -neg_code;

                if let Some(&c_id) = dwa.states[b_id].transitions.get(pos_code) {
                    let w_ab = a_state.trans_weights_exceptions.get(&neg_code).unwrap();
                    let w_bc = dwa.states[b_id]
                        .trans_weights_exceptions
                        .get(&pos_code)
                        .or(dwa.states[b_id].trans_weight_default.as_ref())
                        .unwrap();
                    let path_weight = w_ab & w_bc;

                    let c_state = dwa.states[c_id].clone();

                    // Aggregate weight updates
                    let weight_to_add = &path_weight & &c_state.weight;
                    if !weight_to_add.is_empty() {
                        *weight_updates.entry(a_id).or_default() |= &weight_to_add;
                    }

                    if let Some(fw) = &c_state.final_weight {
                        let final_weight_to_add = &path_weight & fw;
                        if !final_weight_to_add.is_empty() {
                            *final_weight_updates.entry(a_id).or_default() |= &final_weight_to_add;
                        }
                    }

                    // Aggregate transition updates
                    if let Some(def_target) = c_state.transitions.default {
                        let w = c_state.trans_weight_default.as_ref().unwrap();
                        transition_additions.push((a_id, None, def_target, &path_weight & w));
                    }
                    for (&on, &target) in &c_state.transitions.exceptions {
                        let w = c_state.trans_weights_exceptions.get(&on).unwrap();
                        transition_additions.push((a_id, Some(on), target, &path_weight & w));
                    }
                }
            }
        }

        // Apply updates
        for (id, w) in weight_updates {
            let old_len = dwa.states[id].weight.len();
            dwa.states[id].weight |= &w;
            if dwa.states[id].weight.len() != old_len {
                changed = true;
            }
        }
        for (id, fw) in final_weight_updates {
            let state = &mut dwa.states[id];
            let old_len = state.final_weight.as_ref().map_or(0, |w| w.len());
            if let Some(existing_fw) = &mut state.final_weight {
                *existing_fw |= &fw;
            } else {
                state.final_weight = Some(fw);
            }
            let new_len = state.final_weight.as_ref().map_or(0, |w| w.len());
            if old_len != new_len {
                changed = true;
            }
        }

        if !transition_additions.is_empty() {
            changed = true;
        }
    }
}

/// After propagation, any remaining negative transitions that do not point to a final state
/// are considered invalid paths and are removed.
fn remove_internal_negatives(dwa: &mut DWA) {
    let is_final: Vec<bool> = dwa.states.iter().map(|s| s.final_weight.is_some()).collect();

    for state in &mut dwa.states.0 {
        let to_remove: Vec<i16> = state
            .transitions
            .exceptions
            .iter()
            .filter(|(&k, &v)| k < 0 && !is_final[v])
            .map(|(k, _)| *k)
            .collect();

        for k in to_remove {
            state.transitions.exceptions.remove(&k);
            state.trans_weights_exceptions.remove(&k);
        }
    }
}
