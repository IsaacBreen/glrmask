use crate::precompute4::resolve_negatives::{apply_finality_fixpoint, remove_negative_transitions, resolve_negative_codes_in_dwa};
use crate::precompute4::test_weighted_automata::stochastic_equivalence_test;
use crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL;
use crate::precompute4::weighted_automata::{DWA, NWA, Weight};
use crate::precompute4::weighted_automata::common::Label;

#[test]
fn test_resolve_negatives_simple_cancellation() {
    // Corresponds to a sequence like `a, neg(a)`.
    // The DWA should resolve to an automaton that accepts `a` and is final.
    let mut d = DWA::new();
    let s1 = d.add_state();
    let s2 = d.add_state();

    let code_a = 7;
    let neg_code_a = Label::MIN + code_a;

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
    use crate::precompute4::resolve_negatives::resolve_negative_codes_in_nwa;
    let mut nwa = NWA::new();
    let mut states = Vec::new();
    for _ in 0..69 {
        states.push(nwa.add_state());
    }
    nwa.body.start_state = states[68];

    // State 0:
    nwa.states[states[0]].final_weight = Some(Weight::all());
    // State 1:
    nwa.add_epsilon(states[1], states[0], Weight::all());
    // State 2:
    nwa.add_epsilon(states[2], states[1], Weight::all());
    // State 3:
    nwa.add_epsilon(states[3], states[4], Weight::all());
    // State 4:
    nwa.add_epsilon(states[4], states[5], Weight::all());
    // State 5:
    nwa.add_transition(states[5], 0, states[6], Weight::all());
    nwa.add_transition(states[5], 69, states[6], Weight::all());
    nwa.add_transition(states[5], 79, states[6], Weight::all());
    nwa.add_transition(states[5], 101, states[6], Weight::all());
    nwa.add_transition(states[5], 131, states[6], Weight::all());
    nwa.add_transition(states[5], 151, states[6], Weight::all());
    nwa.add_transition(states[5], 161, states[6], Weight::all());
    nwa.add_transition(states[5], 165, states[6], Weight::all());
    nwa.add_transition(states[5], 166, states[6], Weight::all());
    nwa.add_transition(states[5], 279, states[6], Weight::all());
    nwa.add_transition(states[5], 280, states[6], Weight::all());
    nwa.add_transition(states[5], 286, states[6], Weight::all());
    nwa.add_transition(states[5], 300, states[6], Weight::all());
    nwa.add_transition(states[5], 310, states[6], Weight::all());
    nwa.add_transition(states[5], 371, states[6], Weight::all());
    nwa.add_transition(states[5], 400, states[6], Weight::all());
    nwa.add_transition(states[5], 422, states[8], Weight::all());
    nwa.add_transition(states[5], 429, states[6], Weight::all());
    nwa.add_transition(states[5], 436, states[9], Weight::all());
    nwa.add_transition(states[5], 437, states[10], Weight::all());
    nwa.add_transition(states[5], 438, states[11], Weight::all());
    nwa.add_transition(states[5], 458, states[12], Weight::all());
    nwa.add_transition(states[5], 459, states[7], Weight::all());
    nwa.add_transition(states[5], 476, states[6], Weight::all());
    // State 6:
    nwa.add_transition(states[6], 422, states[13], Weight::all());
    nwa.add_transition(states[6], 436, states[14], Weight::all());
    nwa.add_transition(states[6], 437, states[11], Weight::all());
    // State 7:
    nwa.add_transition(states[7], DEFAULT_TRANSITION_SYMBOL, states[6], Weight::all());
    // State 8:
    nwa.add_transition(states[8], Label::MIN + 422, states[15], Weight::all());
    // State 9:
    nwa.add_transition(states[9], Label::MIN + 436, states[15], Weight::all());
    // State 10:
    nwa.add_transition(states[10], Label::MIN + 437, states[15], Weight::all());
    // State 11:
    nwa.add_transition(states[11], 436, states[14], Weight::all());
    // State 12:
    nwa.add_transition(states[12], Label::MIN + 458, states[16], Weight::all());
    // State 13:
    nwa.add_transition(states[13], Label::MIN + 422, states[9], Weight::all());
    // State 14:
    nwa.add_transition(states[14], Label::MIN + 436, states[10], Weight::all());
    // State 15:
    nwa.add_transition(states[15], Label::MIN + 458, states[17], Weight::all());
    // State 16:
    nwa.add_transition(states[16], Label::MIN + 459, states[17], Weight::all());
    // State 17:
    nwa.add_epsilon(states[17], states[2], Weight::from_item(1));
    // State 18:
    nwa.add_epsilon(states[18], states[3], Weight::all());
    // State 19:
    nwa.add_epsilon(states[19], states[2], Weight::from_ranges(&[(0, 5)]));
    // State 20:
    nwa.add_epsilon(states[20], states[19], Weight::all());
    // State 21:
    nwa.add_epsilon(states[21], states[22], Weight::all());
    // State 22:
    nwa.add_transition(states[22], 0, states[23], Weight::all());
    nwa.add_transition(states[22], 69, states[23], Weight::all());
    nwa.add_transition(states[22], 79, states[23], Weight::all());
    nwa.add_transition(states[22], 101, states[23], Weight::all());
    nwa.add_transition(states[22], 131, states[23], Weight::all());
    nwa.add_transition(states[22], 151, states[23], Weight::all());
    nwa.add_transition(states[22], 161, states[23], Weight::all());
    nwa.add_transition(states[22], 165, states[23], Weight::all());
    nwa.add_transition(states[22], 166, states[23], Weight::all());
    nwa.add_transition(states[22], 279, states[23], Weight::all());
    nwa.add_transition(states[22], 280, states[23], Weight::all());
    nwa.add_transition(states[22], 286, states[23], Weight::all());
    nwa.add_transition(states[22], 300, states[23], Weight::all());
    nwa.add_transition(states[22], 310, states[23], Weight::all());
    nwa.add_transition(states[22], 371, states[23], Weight::all());
    nwa.add_transition(states[22], 400, states[23], Weight::all());
    nwa.add_transition(states[22], 422, states[25], Weight::all());
    nwa.add_transition(states[22], 429, states[23], Weight::all());
    nwa.add_transition(states[22], 436, states[26], Weight::all());
    nwa.add_transition(states[22], 437, states[27], Weight::all());
    nwa.add_transition(states[22], 438, states[28], Weight::all());
    nwa.add_transition(states[22], 458, states[29], Weight::all());
    nwa.add_transition(states[22], 459, states[24], Weight::all());
    nwa.add_transition(states[22], 476, states[23], Weight::all());
    // State 23:
    nwa.add_transition(states[23], 422, states[30], Weight::all());
    nwa.add_transition(states[23], 436, states[31], Weight::all());
    nwa.add_transition(states[23], 437, states[28], Weight::all());
    // State 24:
    nwa.add_transition(states[24], DEFAULT_TRANSITION_SYMBOL, states[23], Weight::all());
    // State 25:
    nwa.add_transition(states[25], Label::MIN + 422, states[32], Weight::all());
    // State 26:
    nwa.add_transition(states[26], Label::MIN + 436, states[32], Weight::all());
    // State 27:
    nwa.add_transition(states[27], Label::MIN + 437, states[32], Weight::all());
    // State 28:
    nwa.add_transition(states[28], 436, states[31], Weight::all());
    // State 29:
    nwa.add_transition(states[29], Label::MIN + 458, states[33], Weight::all());
    // State 30:
    nwa.add_transition(states[30], Label::MIN + 422, states[26], Weight::all());
    // State 31:
    nwa.add_transition(states[31], Label::MIN + 436, states[27], Weight::all());
    // State 32:
    nwa.add_transition(states[32], Label::MIN + 458, states[34], Weight::all());
    // State 33:
    nwa.add_transition(states[33], Label::MIN + 459, states[34], Weight::all());
    // State 34:
    nwa.add_epsilon(states[34], states[18], Weight::from_item(1));
    // State 35:
    nwa.add_epsilon(states[35], states[21], Weight::all());
    // State 36:
    nwa.add_epsilon(states[36], states[37], Weight::all());
    // State 37:
    nwa.add_epsilon(states[37], states[38], Weight::all());
    // State 38:
    nwa.add_transition(states[38], 0, states[39], Weight::all());
    nwa.add_transition(states[38], 69, states[39], Weight::all());
    nwa.add_transition(states[38], 79, states[39], Weight::all());
    nwa.add_transition(states[38], 101, states[39], Weight::all());
    nwa.add_transition(states[38], 131, states[39], Weight::all());
    nwa.add_transition(states[38], 151, states[39], Weight::all());
    nwa.add_transition(states[38], 161, states[39], Weight::all());
    nwa.add_transition(states[38], 165, states[39], Weight::all());
    nwa.add_transition(states[38], 166, states[39], Weight::all());
    nwa.add_transition(states[38], 279, states[39], Weight::all());
    nwa.add_transition(states[38], 280, states[39], Weight::all());
    nwa.add_transition(states[38], 286, states[39], Weight::all());
    nwa.add_transition(states[38], 300, states[39], Weight::all());
    nwa.add_transition(states[38], 310, states[39], Weight::all());
    nwa.add_transition(states[38], 371, states[39], Weight::all());
    nwa.add_transition(states[38], 400, states[39], Weight::all());
    nwa.add_transition(states[38], 422, states[41], Weight::all());
    nwa.add_transition(states[38], 429, states[39], Weight::all());
    nwa.add_transition(states[38], 436, states[42], Weight::all());
    nwa.add_transition(states[38], 437, states[43], Weight::all());
    nwa.add_transition(states[38], 438, states[44], Weight::all());
    nwa.add_transition(states[38], 458, states[45], Weight::all());
    nwa.add_transition(states[38], 459, states[40], Weight::all());
    nwa.add_transition(states[38], 476, states[39], Weight::all());
    // State 39:
    nwa.add_transition(states[39], 422, states[46], Weight::all());
    nwa.add_transition(states[39], 436, states[47], Weight::all());
    nwa.add_transition(states[39], 437, states[44], Weight::all());
    // State 40:
    nwa.add_transition(states[40], DEFAULT_TRANSITION_SYMBOL, states[39], Weight::all());
    // State 41:
    nwa.add_transition(states[41], Label::MIN + 422, states[48], Weight::all());
    // State 42:
    nwa.add_transition(states[42], Label::MIN + 436, states[48], Weight::all());
    // State 43:
    nwa.add_transition(states[43], Label::MIN + 437, states[48], Weight::all());
    // State 44:
    nwa.add_transition(states[44], 436, states[47], Weight::all());
    // State 45:
    nwa.add_transition(states[45], Label::MIN + 458, states[49], Weight::all());
    // State 46:
    nwa.add_transition(states[46], Label::MIN + 422, states[42], Weight::all());
    // State 47:
    nwa.add_transition(states[47], Label::MIN + 436, states[43], Weight::all());
    // State 48:
    nwa.add_transition(states[48], Label::MIN + 458, states[50], Weight::all());
    // State 49:
    nwa.add_transition(states[49], Label::MIN + 459, states[50], Weight::all());
    // State 50:
    nwa.add_epsilon(states[50], states[35], Weight::from_item(1));
    // State 51:
    nwa.add_epsilon(states[51], states[36], Weight::all());
    // State 52:
    nwa.add_epsilon(states[52], states[53], Weight::all());
    nwa.add_epsilon(states[52], states[54], Weight::all());
    // State 53:
    nwa.add_epsilon(states[53], states[51], Weight::from_ranges(&[(2, 2), (4, 5)]));
    // State 54:
    nwa.add_transition(states[54], 0, states[55], Weight::all());
    nwa.add_transition(states[54], 69, states[55], Weight::all());
    nwa.add_transition(states[54], 79, states[55], Weight::all());
    nwa.add_transition(states[54], 101, states[55], Weight::all());
    nwa.add_transition(states[54], 131, states[55], Weight::all());
    nwa.add_transition(states[54], 151, states[55], Weight::all());
    nwa.add_transition(states[54], 161, states[55], Weight::all());
    nwa.add_transition(states[54], 165, states[55], Weight::all());
    nwa.add_transition(states[54], 166, states[55], Weight::all());
    nwa.add_transition(states[54], 279, states[55], Weight::all());
    nwa.add_transition(states[54], 280, states[55], Weight::all());
    nwa.add_transition(states[54], 286, states[55], Weight::all());
    nwa.add_transition(states[54], 300, states[55], Weight::all());
    nwa.add_transition(states[54], 310, states[55], Weight::all());
    nwa.add_transition(states[54], 371, states[55], Weight::all());
    nwa.add_transition(states[54], 400, states[55], Weight::all());
    nwa.add_transition(states[54], 422, states[57], Weight::all());
    nwa.add_transition(states[54], 429, states[55], Weight::all());
    nwa.add_transition(states[54], 436, states[58], Weight::all());
    nwa.add_transition(states[54], 437, states[59], Weight::all());
    nwa.add_transition(states[54], 438, states[60], Weight::all());
    nwa.add_transition(states[54], 458, states[61], Weight::all());
    nwa.add_transition(states[54], 459, states[56], Weight::all());
    nwa.add_transition(states[54], 476, states[55], Weight::all());
    // State 55:
    nwa.add_transition(states[55], 422, states[62], Weight::all());
    nwa.add_transition(states[55], 436, states[63], Weight::all());
    nwa.add_transition(states[55], 437, states[60], Weight::all());
    // State 56:
    nwa.add_transition(states[56], DEFAULT_TRANSITION_SYMBOL, states[55], Weight::all());
    // State 57:
    nwa.add_transition(states[57], Label::MIN + 422, states[64], Weight::all());
    // State 58:
    nwa.add_transition(states[58], Label::MIN + 436, states[64], Weight::all());
    // State 59:
    nwa.add_transition(states[59], Label::MIN + 437, states[64], Weight::all());
    // State 60:
    nwa.add_transition(states[60], 436, states[63], Weight::all());
    // State 61:
    nwa.add_transition(states[61], Label::MIN + 458, states[65], Weight::all());
    // State 62:
    nwa.add_transition(states[62], Label::MIN + 422, states[58], Weight::all());
    // State 63:
    nwa.add_transition(states[63], Label::MIN + 436, states[59], Weight::all());
    // State 64:
    nwa.add_transition(states[64], Label::MIN + 458, states[66], Weight::all());
    // State 65:
    nwa.add_transition(states[65], Label::MIN + 459, states[66], Weight::all());
    // State 66:
    nwa.add_epsilon(states[66], states[51], Weight::from_item(1));
    // State 67:
    nwa.add_epsilon(states[67], states[52], Weight::all());
    // State 68:
    nwa.add_transition(states[68], 0, states[67], Weight::all());

    nwa.simplify();
    println!("Before negative resolution:\n{}", nwa);

    let mut d = nwa.determinize_to_dwa();
    d.simplify();
    println!("DWA before negative resolution:\n{}", d);
    assert_eq!(
        d.eval_word_weight(&[0, 422, Label::MIN + 422, Label::MIN + 458, 458, Label::MIN + 458, Label::MIN + 459, 459, DEFAULT_TRANSITION_SYMBOL, 422, Label::MIN + 422, Label::MIN + 436, Label::MIN + 458, 458, Label::MIN + 458, Label::MIN + 459]),
        Weight::from_item(1),
        "DWA did not accept [0, 422] with expected weight after resolving negatives."
    );
    let mut nwa2 = NWA::from_dwa(&d);
    resolve_negative_codes_in_nwa(&mut nwa2);
    println!("After negative resolution (from DWA):\n{}", nwa2);

    resolve_negative_codes_in_nwa(&mut nwa);
    println!("After negative resolution (from NWA):\n{}", nwa);
    let mut d = nwa.determinize_to_dwa();
    println!("DWA after negative resolution:\n{}", d);

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
    d.add_transition(states[2], Label::MIN + 422, states[3], Weight::all()).unwrap();
    // State 3: neg(458) -> 4 (weight: [1])
    d.add_transition(states[3], Label::MIN + 458, states[4], Weight::from_item(1)).unwrap();
    // State 4: 458 -> 5 (weight: ALL)
    d.add_transition(states[4], 458, states[5], Weight::all()).unwrap();
    // State 5: neg(458) -> 6 (weight: ALL)
    d.add_transition(states[5], Label::MIN + 458, states[6], Weight::all()).unwrap();
    // State 6: neg(459) -> 7 (weight: [1])
    d.add_transition(states[6], Label::MIN + 459, states[7], Weight::from_item(1)).unwrap();
    // State 7: 459 -> 8 (weight: ALL)
    d.add_transition(states[7], 459, states[8], Weight::all()).unwrap();
    // State 8: DEFAULT_TRANSITION_SYMBOL -> 9 (weight: ALL)
    d.add_transition(states[8], DEFAULT_TRANSITION_SYMBOL, states[9], Weight::all()).unwrap();
    // State 9: 422 -> 10 (weight: ALL)
    d.add_transition(states[9], 422, states[10], Weight::all()).unwrap();
    // State 10: neg(422) -> 11 (weight: ALL)
    d.add_transition(states[10], Label::MIN + 422, states[11], Weight::all()).unwrap();
    // State 11: neg(436) -> 12 (weight: ALL)
    d.add_transition(states[11], Label::MIN + 436, states[12], Weight::all()).unwrap();
    // State 12: neg(458) -> 13 (weight: [1])
    d.add_transition(states[12], Label::MIN + 458, states[13], Weight::from_item(1)).unwrap();
    // State 13: 458 -> 14 (weight: ALL)
    d.add_transition(states[13], 458, states[14], Weight::all()).unwrap();
    // State 14: neg(458) -> 15 (weight: ALL)
    d.add_transition(states[14], Label::MIN + 458, states[15], Weight::all()).unwrap();
    // State 15: neg(459) -> 16 (weight: [1])
    d.add_transition(states[15], Label::MIN + 459, states[16], Weight::from_item(1)).unwrap();
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
    let neg_code7 = Label::MIN + code7;
    let code3 = 3;
    let neg_code3 = Label::MIN + code3;
    let code1 = 1;
    let neg_code1 = Label::MIN + code1;
    let code2 = 2;
    let neg_code2 = Label::MIN + code2;

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
    let neg_code0 = Label::MIN + code0;
    let neg_code1 = Label::MIN + 1;
    let code2 = 2;
    let neg_code2 = Label::MIN + code2;

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

    let neg_code1 = Label::MIN + 1;
    let neg_code7 = Label::MIN + 7;

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

    let neg_code1 = Label::MIN + 1;

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