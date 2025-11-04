use crate::precompute4::full_dwa::Precomputed4;
use crate::precompute4::weighted_automata::{DWA, DWAState, DWAStates, StateID, Weight};

pub fn resolve_negative_codes_for_all(precomputed4: &mut Precomputed4) {
    for (_sid, dwa) in precomputed4.iter_mut() {
        resolve_negative_codes_in_dwa(dwa);
    }
}

fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precompute4::weighted_automata::{assert_dwa_equivalent, DWA, Weight};

    #[test]
    fn test_resolve_negatives_complex_cancellation() {
        let mut d = DWA::new();
        // State 0 is start
        let s1 = d.add_state();
        let s2 = d.add_state();
        let s3 = d.add_state();
        let s4 = d.add_state();
        let s5 = d.add_state();
        let s6 = d.add_state();
        let s7 = d.add_state();
        let s8 = d.add_state();
        let s9 = d.add_state();

        // State 0
        d.add_transition(0, 0, s1, Weight::from_item(1)).unwrap();
        d.add_transition(0, 1, s2, Weight::from_iter(0..=1)).unwrap();
        d.add_transition(0, 2, s3, Weight::from_item(0)).unwrap();
        d.add_transition(0, 3, s4, Weight::from_iter(0..=1)).unwrap();
        // State 1
        d.add_transition(s1, -1, s5, Weight::all()).unwrap();
        // State 2
        d.set_default_transition(s2, s6, Weight::all()).unwrap();
        // State 3
        d.add_transition(s3, -2, s7, Weight::all()).unwrap();
        // State 4 is a sink
        // State 5
        d.add_transition(s5, -1, s8, Weight::all()).unwrap();
        // State 6 is a sink
        // State 7
        d.add_transition(s7, -1, s9, Weight::all()).unwrap();
        // State 8
        d.set_final_weight(s8, Weight::all()).unwrap();
        // State 9
        d.set_final_weight(s9, Weight::all()).unwrap();

        resolve_negative_codes_in_dwa(&mut d);

        let mut expected = DWA::new(); // state 0
        let s1_exp = expected.add_state(); // state 1
        let s_final = expected.add_state(); // state 2

        // After resolution, the negative paths leading to final states effectively make
        // their predecessor states final. The final weight propagates backwards as ALL
        // because all intermediate edge weights and original final weights are ALL.
        expected.set_final_weight(s_final, Weight::all()).unwrap();

        expected.add_transition(0, 0, s_final, Weight::from_item(1)).unwrap();
        expected.add_transition(0, 2, s_final, Weight::from_item(0)).unwrap();

        assert_dwa_equivalent(d, expected);
    }

    #[test]
    fn test_resolve_negatives_isolated_path_from_tokenizer_1() {
        let mut d = DWA::new();
        let s1 = d.add_state();
        let s2 = d.add_state();

        d.add_transition(d.body.start_state, 7, s1, Weight::from_item(2)).unwrap();
        d.add_transition(s1, -7, s2, Weight::all()).unwrap();
        d.set_final_weight(s2, Weight::all()).unwrap();

        resolve_negative_codes_in_dwa(&mut d);

        let mut expected = DWA::new();
        let s_final = expected.add_state();
        expected.add_transition(expected.body.start_state, 7, s_final, Weight::from_item(2)).unwrap();
        expected.set_final_weight(s_final, Weight::from_item(2)).unwrap();

        assert_dwa_equivalent(d, expected);
    }
}
