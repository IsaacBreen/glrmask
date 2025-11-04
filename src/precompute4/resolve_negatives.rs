use crate::precompute4::full_dwa::Precomputed4;
use crate::precompute4::weighted_automata::{DWA, DWAState, DWAStates, StateID, Weight};
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
    crate::debug!(5, "Resolved DWA:\n{}", dwa);
}

fn resolve_negative_codes_in_dwa_internal(
    state_id: StateID,
    states: &DWAStates,
) -> (DWAState, bool) {
    let state_a = &states[state_id];

    // Check if there are any negative transitions to process.
    let has_negative_edges = state_a.transitions.exceptions.keys().any(|&on| on < 0);
    if !has_negative_edges {
        return (state_a.clone(), false);
    }

    // A has negative edges, so we are building a new state for it.
    let mut new_state = DWAState::default();

    // 1. Start with A's own properties and positive transitions.
    new_state.weight = state_a.weight.clone();
    new_state.final_weight = state_a.final_weight.clone();
    new_state.transitions.default = state_a.transitions.default;
    new_state.trans_weight_default = state_a.trans_weight_default.clone();

    for (&on, &to) in &state_a.transitions.exceptions {
        if on >= 0 {
            new_state.transitions.exceptions.insert(on, to);
            if let Some(w) = state_a.trans_weights_exceptions.get(&on) {
                new_state.trans_weights_exceptions.insert(on, w.clone());
            }
        }
    }

    // 2. For each negative edge A --neg(c)--> B, merge properties from B.
    for (&neg_c, &state_b_id) in &state_a.transitions.exceptions {
        if neg_c >= 0 { continue; }

        let c = neg_c.wrapping_sub(i16::MIN);
        let weight_a_to_b = state_a.trans_weights_exceptions.get(&neg_c)
            .cloned()
            .unwrap_or_else(Weight::all);

        if state_b_id >= states.len() { continue; }
        let state_b = &states[state_b_id];

        // Merge B's final weight into A.
        if let Some(fw_b) = &state_b.final_weight {
            let inherited_fw = &weight_a_to_b & fw_b;
            if !inherited_fw.is_empty() {
                if let Some(fw_a) = &mut new_state.final_weight {
                    *fw_a |= &inherited_fw;
                } else {
                    new_state.final_weight = Some(inherited_fw);
                }
            }
        }

        // Merge B's negative edges into A.
        for (&b_neg_c, &b_to) in &state_b.transitions.exceptions {
            if b_neg_c < 0 {
                let weight_b_to_target = state_b.trans_weights_exceptions.get(&b_neg_c)
                    .cloned()
                    .unwrap_or_else(Weight::all);
                let combined_weight = &weight_a_to_b & &weight_b_to_target;

                if let Some(&existing_to) = new_state.transitions.exceptions.get(&b_neg_c) {
                    if existing_to != b_to {
                        panic!("Cannot merge states: conflicting negative transition targets for code {}", b_neg_c);
                    }
                } else {
                    new_state.transitions.exceptions.insert(b_neg_c, b_to);
                }
                
                new_state.trans_weights_exceptions.entry(b_neg_c)
                    .and_modify(|w| *w |= &combined_weight)
                    .or_insert(combined_weight);
            }
        }

        // 3. Handle cancellation: if B --c--> C, merge C's properties into A.
        if let Some(&state_c_id) = state_b.transitions.get(c) {
            let is_exception = state_b.transitions.exceptions.contains_key(&c);
            let weight_b_to_c = if is_exception {
                state_b.trans_weights_exceptions.get(&c).cloned().unwrap_or_else(Weight::all)
            } else {
                state_b.trans_weight_default.as_ref().cloned().unwrap_or_else(Weight::all)
            };

            if state_c_id >= states.len() { continue; }
            let state_c = &states[state_c_id];
            let path_weight = &weight_a_to_b & &weight_b_to_c;

            let mut temp_c = state_c.clone();
            temp_c.apply_weight(&path_weight);
            new_state.merge_union(&temp_c);
        }
    }

    (new_state, true)
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
