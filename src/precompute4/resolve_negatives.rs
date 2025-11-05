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
    todo!()
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
