use crate::precompute4::resolve_negatives::resolve_negative_codes_in_dwa;
use crate::precompute4::test_weighted_automata::stochastic_equivalence_test;
use crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL;
use crate::precompute4::weighted_automata::{DWA, Weight};

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
fn test_resolve_negatives_from_large_nwa_log() {
    let mut d = DWA::new();
    let mut states = vec![d.body.start_state];
    for _ in 0..50 {
        states.push(d.add_state());
    }

    // State 0:
    d.add_transition(states[0], 0, states[1], Weight::all()).unwrap();
    d.add_transition(states[0], 161, states[2], Weight::all()).unwrap();
    d.add_transition(states[0], 165, states[2], Weight::all()).unwrap();
    d.add_transition(states[0], 166, states[2], Weight::all()).unwrap();
    // State 1:
    for code in [0, 69, 79, 101, 131, 151, 161, 165, 166, 279, 280, 286, 300, 310, 371, 400, 429, 476] {
        d.add_transition(states[1], code, states[3], Weight::all()).unwrap();
    }
    d.add_transition(states[1], 422, states[4], Weight::all()).unwrap();
    d.add_transition(states[1], 436, states[5], Weight::all()).unwrap();
    d.add_transition(states[1], 437, states[6], Weight::all()).unwrap();
    d.add_transition(states[1], 438, states[7], Weight::all()).unwrap();
    d.add_transition(states[1], 458, states[8], Weight::all()).unwrap();
    d.add_transition(states[1], 459, states[9], Weight::all()).unwrap();
    // State 2:
    d.set_final_weight(states[2], Weight::from_ranges(&[(0, 5)])).unwrap();
    // State 3:
    d.add_transition(states[3], 422, states[10], Weight::all()).unwrap();
    d.add_transition(states[3], 436, states[11], Weight::all()).unwrap();
    d.add_transition(states[3], 437, states[7], Weight::all()).unwrap();
    // State 4:
    d.add_transition(states[4], i16::MIN + 422, states[12], Weight::all()).unwrap();
    // State 5:
    d.add_transition(states[5], i16::MIN + 436, states[12], Weight::all()).unwrap();
    // State 6:
    d.add_transition(states[6], i16::MIN + 437, states[12], Weight::all()).unwrap();
    // State 7:
    d.add_transition(states[7], 436, states[11], Weight::all()).unwrap();
    // State 8:
    d.add_transition(states[8], i16::MIN + 458, states[13], Weight::all()).unwrap();
    // State 9:
    d.add_transition(states[9], DEFAULT_TRANSITION_SYMBOL, states[3], Weight::all()).unwrap();
    // State 10:
    d.add_transition(states[10], i16::MIN + 422, states[5], Weight::all()).unwrap();
    // State 11:
    d.add_transition(states[11], i16::MIN + 436, states[6], Weight::all()).unwrap();
    // State 12:
    d.add_transition(states[12], i16::MIN + 458, states[14], Weight::all()).unwrap();
    // State 13:
    d.add_transition(states[13], i16::MIN + 459, states[14], Weight::all()).unwrap();
    // State 14:
    for code in [0, 69, 79, 101, 131, 151, 161, 165, 166, 279, 280, 286, 300, 310, 371, 400, 429, 476] {
        d.add_transition(states[14], code, states[15], Weight::from_item(1)).unwrap();
    }
    d.add_transition(states[14], 422, states[16], Weight::from_item(1)).unwrap();
    d.add_transition(states[14], 436, states[17], Weight::from_item(1)).unwrap();
    d.add_transition(states[14], 437, states[18], Weight::from_item(1)).unwrap();
    d.add_transition(states[14], 438, states[19], Weight::from_item(1)).unwrap();
    d.add_transition(states[14], 458, states[20], Weight::from_item(1)).unwrap();
    d.add_transition(states[14], 459, states[21], Weight::from_item(1)).unwrap();
    // State 15:
    d.add_transition(states[15], 422, states[22], Weight::from_item(1)).unwrap();
    d.add_transition(states[15], 436, states[23], Weight::from_item(1)).unwrap();
    d.add_transition(states[15], 437, states[19], Weight::from_item(1)).unwrap();
    // State 16:
    d.add_transition(states[16], i16::MIN + 422, states[24], Weight::from_item(1)).unwrap();
    // State 17:
    d.add_transition(states[17], i16::MIN + 436, states[24], Weight::from_item(1)).unwrap();
    // State 18:
    d.add_transition(states[18], i16::MIN + 437, states[24], Weight::from_item(1)).unwrap();
    // State 19:
    d.add_transition(states[19], 436, states[23], Weight::from_item(1)).unwrap();
    // State 20:
    d.add_transition(states[20], i16::MIN + 458, states[25], Weight::from_item(1)).unwrap();
    // State 21:
    d.add_transition(states[21], DEFAULT_TRANSITION_SYMBOL, states[15], Weight::from_item(1)).unwrap();
    // State 22:
    d.add_transition(states[22], i16::MIN + 422, states[17], Weight::from_item(1)).unwrap();
    // State 23:
    d.add_transition(states[23], i16::MIN + 436, states[18], Weight::from_item(1)).unwrap();
    // State 24:
    d.add_transition(states[24], i16::MIN + 458, states[26], Weight::from_item(1)).unwrap();
    // State 25:
    d.add_transition(states[25], i16::MIN + 459, states[26], Weight::from_item(1)).unwrap();
    // State 26:
    for code in [0, 69, 79, 101, 131, 151, 161, 165, 166, 279, 280, 286, 300, 310, 371, 400, 429, 476] {
        d.add_transition(states[26], code, states[27], Weight::from_item(1)).unwrap();
    }
    d.add_transition(states[26], 422, states[28], Weight::from_item(1)).unwrap();
    d.add_transition(states[26], 436, states[29], Weight::from_item(1)).unwrap();
    d.add_transition(states[26], 437, states[30], Weight::from_item(1)).unwrap();
    d.add_transition(states[26], 438, states[31], Weight::from_item(1)).unwrap();
    d.add_transition(states[26], 458, states[32], Weight::from_item(1)).unwrap();
    d.add_transition(states[26], 459, states[33], Weight::from_item(1)).unwrap();
    // State 27:
    d.add_transition(states[27], 422, states[34], Weight::from_item(1)).unwrap();
    d.add_transition(states[27], 436, states[35], Weight::from_item(1)).unwrap();
    d.add_transition(states[27], 437, states[31], Weight::from_item(1)).unwrap();
    // State 28:
    d.add_transition(states[28], i16::MIN + 422, states[36], Weight::from_item(1)).unwrap();
    // State 29:
    d.add_transition(states[29], i16::MIN + 436, states[36], Weight::from_item(1)).unwrap();
    // State 30:
    d.add_transition(states[30], i16::MIN + 437, states[36], Weight::from_item(1)).unwrap();
    // State 31:
    d.add_transition(states[31], 436, states[35], Weight::from_item(1)).unwrap();
    // State 32:
    d.add_transition(states[32], i16::MIN + 458, states[37], Weight::from_item(1)).unwrap();
    // State 33:
    d.add_transition(states[33], DEFAULT_TRANSITION_SYMBOL, states[27], Weight::from_item(1)).unwrap();
    // State 34:
    d.add_transition(states[34], i16::MIN + 422, states[29], Weight::from_item(1)).unwrap();
    // State 35:
    d.add_transition(states[35], i16::MIN + 436, states[30], Weight::from_item(1)).unwrap();
    // State 36:
    d.add_transition(states[36], i16::MIN + 458, states[38], Weight::from_item(1)).unwrap();
    // State 37:
    d.add_transition(states[37], i16::MIN + 459, states[38], Weight::from_item(1)).unwrap();
    // State 38:
    for code in [0, 69, 79, 101, 131, 151, 161, 165, 166, 279, 280, 286, 300, 310, 371, 400, 429, 476] {
        d.add_transition(states[38], code, states[39], Weight::from_item(1)).unwrap();
    }
    d.add_transition(states[38], 422, states[40], Weight::from_item(1)).unwrap();
    d.add_transition(states[38], 436, states[41], Weight::from_item(1)).unwrap();
    d.add_transition(states[38], 437, states[42], Weight::from_item(1)).unwrap();
    d.add_transition(states[38], 438, states[43], Weight::from_item(1)).unwrap();
    d.add_transition(states[38], 458, states[44], Weight::from_item(1)).unwrap();
    d.add_transition(states[38], 459, states[45], Weight::from_item(1)).unwrap();
    // State 39:
    d.add_transition(states[39], 422, states[46], Weight::from_item(1)).unwrap();
    d.add_transition(states[39], 436, states[47], Weight::from_item(1)).unwrap();
    d.add_transition(states[39], 437, states[43], Weight::from_item(1)).unwrap();
    // State 40:
    d.add_transition(states[40], i16::MIN + 422, states[48], Weight::from_item(1)).unwrap();
    // State 41:
    d.add_transition(states[41], i16::MIN + 436, states[48], Weight::from_item(1)).unwrap();
    // State 42:
    d.add_transition(states[42], i16::MIN + 437, states[48], Weight::from_item(1)).unwrap();
    // State 43:
    d.add_transition(states[43], 436, states[47], Weight::from_item(1)).unwrap();
    // State 44:
    d.add_transition(states[44], i16::MIN + 458, states[49], Weight::from_item(1)).unwrap();
    // State 45:
    d.add_transition(states[45], DEFAULT_TRANSITION_SYMBOL, states[39], Weight::from_item(1)).unwrap();
    // State 46:
    d.add_transition(states[46], i16::MIN + 422, states[41], Weight::from_item(1)).unwrap();
    // State 47:
    d.add_transition(states[47], i16::MIN + 436, states[42], Weight::from_item(1)).unwrap();
    // State 48:
    d.add_transition(states[48], i16::MIN + 458, states[50], Weight::from_item(1)).unwrap();
    // State 49:
    d.add_transition(states[49], i16::MIN + 459, states[50], Weight::from_item(1)).unwrap();
    // State 50:
    d.set_final_weight(states[50], Weight::from_item(1)).unwrap();

    d.simplify();
    println!("{}", d);
    resolve_negative_codes_in_dwa(&mut d);

    // Assert [0, 422] is accepted with weight in [1].
    assert_eq!(
        d.eval_word_weight(&[0, 422]),
        Weight::from_item(1),
        "DWA did not accept [0, 422] with expected weight after resolving negatives."
    );
}

#[test]
fn test_resolve_negatives_from_nwa_log_2() {
    let mut d = DWA::new();
    let mut states = vec![d.body.start_state];
    for _ in 0..16 {
        states.push(d.add_state());
    }

    // State 0: 0 -> 1 (weight: ALL)
    d.add_transition(states[0], 0, states[1], Weight::all()).unwrap();
    // State 1: 422 -> 2 (weight: ALL)
    d.add_transition(states[1], 422, states[2], Weight::all()).unwrap();
    // State 2: neg(422) -> 3 (weight: ALL)
    d.add_transition(states[2], i16::MIN + 422, states[3], Weight::all()).unwrap();
    // State 3: neg(458) -> 4 (weight: [1])
    d.add_transition(states[3], i16::MIN + 458, states[4], Weight::from_item(1)).unwrap();
    // State 4: 458 -> 5 (weight: ALL)
    d.add_transition(states[4], 458, states[5], Weight::all()).unwrap();
    // State 5: neg(458) -> 6 (weight: ALL)
    d.add_transition(states[5], i16::MIN + 458, states[6], Weight::all()).unwrap();
    // State 6: neg(459) -> 7 (weight: [1])
    d.add_transition(states[6], i16::MIN + 459, states[7], Weight::from_item(1)).unwrap();
    // State 7: 459 -> 8 (weight: ALL)
    d.add_transition(states[7], 459, states[8], Weight::all()).unwrap();
    // State 8: 32767 -> 9 (weight: ALL)
    d.add_transition(states[8], DEFAULT_TRANSITION_SYMBOL, states[9], Weight::all()).unwrap();
    // State 9: 422 -> 10 (weight: ALL)
    d.add_transition(states[9], 422, states[10], Weight::all()).unwrap();
    // State 10: neg(422) -> 11 (weight: ALL)
    d.add_transition(states[10], i16::MIN + 422, states[11], Weight::all()).unwrap();
    // State 11: neg(436) -> 12 (weight: ALL)
    d.add_transition(states[11], i16::MIN + 436, states[12], Weight::all()).unwrap();
    // State 12: neg(458) -> 13 (weight: [1])
    d.add_transition(states[12], i16::MIN + 458, states[13], Weight::from_item(1)).unwrap();
    // State 13: 458 -> 14 (weight: ALL)
    d.add_transition(states[13], 458, states[14], Weight::all()).unwrap();
    // State 14: neg(458) -> 15 (weight: ALL)
    d.add_transition(states[14], i16::MIN + 458, states[15], Weight::all()).unwrap();
    // State 15: neg(459) -> 16 (weight: [1])
    d.add_transition(states[15], i16::MIN + 459, states[16], Weight::from_item(1)).unwrap();
    // State 16: final_weight: ALL
    d.set_final_weight(states[16], Weight::all()).unwrap();

    resolve_negative_codes_in_dwa(&mut d);

    let mut expected = DWA::new();
    let s1 = expected.add_state();
    let s2 = expected.add_state();
    expected.add_transition(expected.body.start_state, 0, s1, Weight::all()).unwrap();
    expected.add_transition(s1, 422, s2, Weight::all()).unwrap();
    expected.set_final_weight(s2, Weight::from_item(1)).unwrap();

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
    d.add_transition(s3, DEFAULT_TRANSITION_SYMBOL, s2, Weight::all()).unwrap();

    // State 4
    d.set_final_weight(s4, Weight::from_item(2)).unwrap();

    // State 5
    d.add_transition(s5, DEFAULT_TRANSITION_SYMBOL, s7, Weight::all()).unwrap();

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
    expected.add_transition(exp_s3, DEFAULT_TRANSITION_SYMBOL, exp_s_after_4, Weight::all()).unwrap();
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
    d.add_transition(s1, DEFAULT_TRANSITION_SYMBOL, s2, Weight::all()).unwrap();
    // 2 is final
    d.set_final_weight(s2, Weight::all()).unwrap();

    resolve_negative_codes_in_dwa(&mut d);

    let mut expected = DWA::new();
    expected.set_final_weight(expected.body.start_state, Weight::all()).unwrap();

    stochastic_equivalence_test(d, expected);
}