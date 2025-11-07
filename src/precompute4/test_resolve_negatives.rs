use crate::precompute4::resolve_negatives::resolve_negative_codes_in_dwa;
use crate::precompute4::test_weighted_automata::stochastic_equivalence_test;
use crate::precompute4::weighted_automata::{Weight, DWA};

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
    expected.set_final_weight(s_final, Weight::from_item(2)).unwrap();

    stochastic_equivalence_test(d, expected);
}

#[test]
fn test_resolve_negatives_long_cancellation_chain() {
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

    let mut expected = DWA::new();
    let s_final = expected.add_state();
    expected.add_transition(expected.body.start_state, code7, s_final, Weight::all()).unwrap();
    expected.set_final_weight(s_final, Weight::all()).unwrap();

    stochastic_equivalence_test(d, expected);
}

#[test]
fn test_resolve_negatives_from_debug_log() {
    let mut d = DWA::new();
    let s1 = d.add_state();
    let s2 = d.add_state();
    let s3 = d.add_state();
    let s4 = d.add_state();
    let s5 = d.add_state();
    let s6 = d.add_state();

    let code0 = 0;
    let neg_code0 = i16::MIN + code0;
    let neg_code1 = i16::MIN + 1;
    let code2 = 2;
    let neg_code2 = i16::MIN + code2;

    d.add_transition(d.body.start_state, code0, s1, Weight::all()).unwrap();
    d.add_transition(d.body.start_state, code2, s2, Weight::all()).unwrap();
    d.add_transition(s1, neg_code0, s3, Weight::all()).unwrap();
    d.add_transition(s2, neg_code2, s4, Weight::all()).unwrap();
    d.add_transition(s3, neg_code1, s5, Weight::all()).unwrap();
    d.add_transition(s4, neg_code0, s6, Weight::all()).unwrap();
    d.set_final_weight(s5, Weight::from_item(1)).unwrap();
    d.set_final_weight(s6, Weight::from_item(0)).unwrap();

    resolve_negative_codes_in_dwa(&mut d);

    let mut expected = DWA::new();
    let s_final1 = expected.add_state();
    let s_final2 = expected.add_state();
    expected.add_transition(expected.body.start_state, code0, s_final1, Weight::all()).unwrap();
    expected.set_final_weight(s_final1, Weight::from_item(1)).unwrap();
    expected.add_transition(expected.body.start_state, code2, s_final2, Weight::all()).unwrap();
    expected.set_final_weight(s_final2, Weight::from_item(0)).unwrap();

    stochastic_equivalence_test(d, expected);
}

#[test]
fn test_resolve_negatives_from_intermediate_debug_log() {
    let mut d = DWA::new();
    let s1 = d.add_state();
    let s2 = d.add_state();
    let s3 = d.add_state();
    let s4 = d.add_state();
    let s5 = d.add_state();
    let s6 = d.add_state();
    let s7 = d.add_state();

    let neg_code1 = i16::MIN + 1;
    let neg_code7 = i16::MIN + 7;

    // State 0 (start)
    d.add_transition(d.body.start_state, 1, s1, Weight::all()).unwrap();
    d.add_transition(d.body.start_state, 3, s2, Weight::all()).unwrap();
    d.add_transition(d.body.start_state, 4, s3, Weight::all()).unwrap();
    d.add_transition(d.body.start_state, 7, s4, Weight::all()).unwrap();

    // State 1
    d.add_transition(s1, neg_code1, s5, Weight::all()).unwrap();
    d.add_transition(s1, 7, s4, Weight::all()).unwrap();

    // State 2
    d.add_transition(s2, 7, s6, Weight::all()).unwrap();

    // State 3
    d.set_default_transition(s3, s2, Weight::all()).unwrap();

    // State 4
    d.set_final_weight(s4, Weight::from_item(2)).unwrap();

    // State 5
    d.set_default_transition(s5, s7, Weight::all()).unwrap();

    // State 6
    d.set_final_weight(s6, Weight::from_item(2)).unwrap();
    d.add_transition(s6, neg_code7, s1, Weight::all()).unwrap();

    // State 7
    d.add_transition(s7, 7, s4, Weight::all()).unwrap();

    resolve_negative_codes_in_dwa(&mut d);

    let mut expected = DWA::new();
    let exp_s1 = expected.add_state();
    let exp_s2 = expected.add_state();
    let exp_s3 = expected.add_state();
    let exp_s_after_4 = expected.add_state();
    let exp_s_final = expected.add_state();

    expected.set_final_weight(exp_s_final, Weight::from_item(2)).unwrap();

    // Paths leading to final state
    expected.add_transition(expected.body.start_state, 7, exp_s_final, Weight::all()).unwrap();

    expected.add_transition(expected.body.start_state, 1, exp_s1, Weight::all()).unwrap();
    expected.add_transition(exp_s1, 7, exp_s_final, Weight::all()).unwrap();

    expected.add_transition(expected.body.start_state, 3, exp_s2, Weight::all()).unwrap();
    expected.add_transition(exp_s2, 7, exp_s_final, Weight::all()).unwrap();

    expected.add_transition(expected.body.start_state, 4, exp_s3, Weight::all()).unwrap();
    expected.set_default_transition(exp_s3, exp_s_after_4, Weight::all()).unwrap();
    expected.add_transition(exp_s_after_4, 7, exp_s_final, Weight::all()).unwrap();

    stochastic_equivalence_test(d, expected);
}

#[test]
fn test_resolve_negatives_minimal_loop_with_default() {
    let mut d = DWA::new();
    let s1 = d.add_state();
    let s2 = d.add_state();

    let neg_code1 = i16::MIN + 1;

    // 0 --neg(1)--> 1
    d.add_transition(d.body.start_state, neg_code1, s1, Weight::all()).unwrap();
    // 1 --*--> 2 (default)
    d.set_default_transition(s1, s2, Weight::all()).unwrap();
    // 2 is final
    d.set_final_weight(s2, Weight::all()).unwrap();

    resolve_negative_codes_in_dwa(&mut d);

    let mut expected = DWA::new();
    expected.set_final_weight(expected.body.start_state, Weight::all()).unwrap();

    stochastic_equivalence_test(d, expected);
}