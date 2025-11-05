use crate::precompute4::full_dwa::Precomputed4;
use crate::precompute4::weighted_automata::{DWA, DWAStates, StateID, Weight};
use std::collections::BTreeMap;

pub fn resolve_negative_codes_for_all(precomputed4: &mut Precomputed4) {
    for (_sid, dwa) in precomputed4.iter_mut() {
        resolve_negative_codes_in_dwa(dwa);
    }
}

fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    crate::debug!(5, "Initial DWA:\n{}", dwa);
    loop {
        let mut changed_in_pass = false;

        for state_id in 0..dwa.states.len() {
            let changed =
                resolve_negative_codes_in_dwa_internal(state_id, &mut dwa.states);
            if changed {
                changed_in_pass = true;
            }
        }

        if !changed_in_pass {
            break;
        }
    }
    dwa.simplify();
    crate::debug!(5, "Resolved DWA:\n{}", dwa);
}

fn resolve_negative_codes_in_dwa_internal(
    state_id: StateID,
    states: &mut DWAStates,
) -> bool {
    let mut changed = false;
    // We need to collect the negative transitions first because we'll be modifying the state's transitions.
    let state_a_clone = states[state_id].clone();
    let negative_transitions: Vec<(i16, StateID)> = state_a_clone
        .transitions
        .exceptions
        .iter()
        .filter(|(k, _)| **k < 0)
        .map(|(k, v)| (*k, *v))
        .collect();

    if negative_transitions.is_empty() {
        return false;
    }

    for (neg_code, b_orig_id) in negative_transitions {
        changed = true;
        let p = neg_code.wrapping_sub(i16::MIN);
        let w_neg = state_a_clone.trans_weights_exceptions.get(&neg_code).unwrap().clone();

        // Step 1: Copy B
        let b_copy_id = states.copy_state(b_orig_id);

        // Step 2: Handle final weight from B
        if let Some(b_final_weight) = states[b_copy_id].final_weight.take() {
            let new_a_final_weight = b_final_weight & &w_neg;
            if !new_a_final_weight.is_empty() {
                let a_state = &mut states[state_id];
                if let Some(a_fw) = a_state.final_weight.as_mut() {
                    *a_fw |= &new_a_final_weight;
                } else {
                    a_state.final_weight = Some(new_a_final_weight);
                }
            }
        }

        // Step 3: Handle matching positive edge (cancellation)
        let b_orig_state_clone = states[b_orig_id].clone();
        if let Some(&c_orig_id) = b_orig_state_clone.transitions.exceptions.get(&p) {
            let c_copy_id = states.copy_state(c_orig_id);
            states.apply_weight(c_copy_id, &w_neg);
            let c_copy_state = states[c_copy_id].clone();
            let a_state = &mut states[state_id];

            // Merge C's weights and transitions into A
            a_state.weight |= &c_copy_state.weight;
            if let Some(c_fw) = c_copy_state.final_weight {
                if !c_fw.is_empty() {
                    if let Some(a_fw) = a_state.final_weight.as_mut() {
                        *a_fw |= &c_fw;
                    } else {
                        a_state.final_weight = Some(c_fw);
                    }
                }
            }

            if let Some(c_default_target) = c_copy_state.transitions.default {
                let a_def_w = a_state.trans_weight_default.get_or_insert_with(Weight::zeros);
                if let Some(c_def_w) = c_copy_state.trans_weight_default {
                    *a_def_w |= &c_def_w;
                }
                a_state.transitions.default = Some(c_default_target);
            }

            for (on, c_target) in c_copy_state.transitions.exceptions {
                let c_weight = c_copy_state.trans_weights_exceptions.get(&on).unwrap().clone();
                if let Some(a_weight) = a_state.trans_weights_exceptions.get_mut(&on) {
                    if a_state.transitions.exceptions.get(&on) == Some(&c_target) {
                        *a_weight |= &c_weight;
                    } else {
                        a_state.transitions.exceptions.insert(on, c_target);
                        *a_weight = c_weight;
                    }
                } else {
                    a_state.transitions.exceptions.insert(on, c_target);
                    a_state.trans_weights_exceptions.insert(on, c_weight);
                }
            }
        }

        // Step 4: Discard all positive edges from B_copy
        let b_copy_state = &mut states[b_copy_id];
        b_copy_state.transitions.exceptions.retain(|k, _| *k < 0);
        b_copy_state.trans_weights_exceptions.retain(|k, _| *k < 0);
        b_copy_state.transitions.default = None;
        b_copy_state.trans_weight_default = None;

        // Step 5: Replace A -> B with A -> B_copy
        states[state_id].transitions.exceptions.insert(neg_code, b_copy_id);
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precompute4::weighted_automata::{assert_dwa_equivalent, DWA, Weight};

    #[test]
    fn test_resolve_negatives_simple_cancellation() {
        // Corresponds to a sequence like `a, neg(a)`.
        // The DWA should resolve to an automaton that accepts `a` and is final.
        let mut d = DWA::new();
        let s1 = d.add_state();
        let s2 = d.add_state();

        let code_a = 7;
        let neg_code_a = i16::MIN + code_a;

        // 0 --a--> 1
        d.add_transition(d.body.start_state, code_a, s1, Weight::from_item(2)).unwrap();
        // 1 --neg(a)--> 2
        d.add_transition(s1, neg_code_a, s2, Weight::all()).unwrap();
        // 2 is final
        d.set_final_weight(s2, Weight::all()).unwrap();

        resolve_negative_codes_in_dwa(&mut d);

        // Expected: 0 --a--> final_state
        let mut expected = DWA::new();
        let s_final = expected.add_state();
        expected.add_transition(expected.body.start_state, code_a, s_final, Weight::from_item(2)).unwrap();
        // The final weight is the intersection of the edge weight (2) and the final weight (all),
        // constrained by reachability during simplification.
        expected.set_final_weight(s_final, Weight::from_item(2)).unwrap();

        assert_dwa_equivalent(d, expected);
    }

    #[test]
    fn test_resolve_negatives_long_cancellation_chain() {
        // This test models a path from a real-world DWA that involves multiple cancellations.
        // The path is: 0 --7--> 5 --neg(7)--> 10 --neg(3)--> 13 --3--> 15 --7--> 22 --neg(7)--> 27 --neg(1)--> 29 --neg(2)--> 30(final)
        let mut d = DWA::new();
        let s5 = d.add_state();
        let s10 = d.add_state();
        let s13 = d.add_state();
        let s15 = d.add_state();
        let s22 = d.add_state();
        let s27 = d.add_state();
        let s29 = d.add_state();
        let s30 = d.add_state();

        let code7 = 7;
        let neg_code7 = i16::MIN + code7;
        let code3 = 3;
        let neg_code3 = i16::MIN + code3;
        let code1 = 1;
        let neg_code1 = i16::MIN + code1;
        let code2 = 2;
        let neg_code2 = i16::MIN + code2;

        d.add_transition(d.body.start_state, code7, s5, Weight::all()).unwrap();
        d.add_transition(s5, neg_code7, s10, Weight::all()).unwrap();
        d.add_transition(s10, neg_code3, s13, Weight::all()).unwrap();
        d.add_transition(s13, code3, s15, Weight::all()).unwrap();
        d.add_transition(s15, code7, s22, Weight::all()).unwrap();
        d.add_transition(s22, neg_code7, s27, Weight::all()).unwrap();
        d.add_transition(s27, neg_code1, s29, Weight::all()).unwrap();
        d.add_transition(s29, neg_code2, s30, Weight::all()).unwrap();
        d.set_final_weight(s30, Weight::all()).unwrap();

        resolve_negative_codes_in_dwa(&mut d);

        // After resolution and simplification, the chain of cancellations should result
        // in an automaton that accepts the "7".
        let mut expected = DWA::new();
        let s_final = expected.add_state();
        expected.add_transition(expected.body.start_state, code7, s_final, Weight::all()).unwrap();
        expected.set_final_weight(s_final, Weight::all()).unwrap();

        assert_dwa_equivalent(d, expected);
    }
}
