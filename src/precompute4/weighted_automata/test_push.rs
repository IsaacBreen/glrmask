#![cfg(test)]
//! Tests for weight pushing optimization.
//!
//! Each test defines:
//! - An input DWA
//! - An expected equivalent but simpler DWA
//!
//! The shared test harness `run_push_optimization_test` then:
//! 1. Sanity checks: input ≡ expected
//! 2. Gets stats for expected DWA
//! 3. Minimizes input WITHOUT pushing → asserts it does NOT achieve those stats
//! 4. Pushes + minimizes input → asserts it DOES achieve those stats

use crate::precompute4::weighted_automata::*;
use crate::precompute4::weighted_automata::common::Label;
use crate::precompute4::weighted_automata::dwa::{DWABody, DWAStates};
use crate::precompute4::weighted_automata::test_weighted_automata::stochastic_equivalence_test;

/// Get DWA stats (states, transitions)
fn dwa_stats(dwa: &DWA) -> (usize, usize) {
    (dwa.states.len(), dwa.states.num_transitions())
}

/// Shared test harness for push optimization tests.
///
/// 1. Sanity check: input ≡ expected
/// 2. Get expected stats
/// 3. Minimize only → assert does NOT match expected stats
/// 4. Push + minimize → assert DOES match expected stats
fn run_push_optimization_test(input: DWA, expected: DWA) {
    // 1. Sanity check equivalence
    stochastic_equivalence_test(input.clone(), expected.clone());

    // 2. Get expected stats
    let (exp_states, exp_trans) = dwa_stats(&expected);

    // 3. Minimize without pushing
    let mut minimized_only = input.clone();
    minimized_only.minimize_with_rustfst();
    let (min_states, min_trans) = dwa_stats(&minimized_only);

    assert!(
        min_states > exp_states || min_trans > exp_trans,
        "Test setup error: minimize-only should NOT achieve optimal stats.\n\
         Expected: {} states, {} trans\n\
         Got (minimize only): {} states, {} trans\n\
         If they match, this test is not demonstrating a push optimization opportunity.",
        exp_states, exp_trans, min_states, min_trans
    );

    // 4. Push + minimize
    let mut pushed = input.clone();
    pushed.residuated_push();
    pushed.minimize_with_rustfst();
    let (push_states, push_trans) = dwa_stats(&pushed);

    // Verify equivalence still holds after push
    stochastic_equivalence_test(pushed.clone(), expected.clone());

    assert_eq!(
        (push_states, push_trans), (exp_states, exp_trans),
        "Push + minimize should achieve optimal stats.\n\
         Expected: {} states, {} trans\n\
         Got (push + minimize): {} states, {} trans",
        exp_states, exp_trans, push_states, push_trans
    );
}

// =============================================================================
// TEST 1: Two states with different final weights, shared outgoing
// =============================================================================
//
// Input:
//   0 ──[a]──> 1 ──[:]──> 3 (fw={100}) ──[x]──> 5 (fw=ALL)
//   0 ──[b]──> 2 ──[:]──> 4 (fw={200}) ──[x]──> 5 (fw=ALL)
//
// States 3 and 4 have identical outgoing but different final weights.
//
// Expected (merged):
//   0 ──[a]──> 1 ──[:, w={100}]──> 34 (fw={100,200}) ──[x]──> 5
//   0 ──[b]──> 2 ──[:, w={200}]──> 34

#[test]
#[ignore = "Weight pushing algorithm not yet correct"]
fn test_merge_states_with_different_final_weights() {
    let a: Label = 97;
    let b: Label = 98;
    let colon: Label = 58;
    let x: Label = 120;

    let all = Weight::all();
    let w100 = Weight::from_item(100);
    let w200 = Weight::from_item(200);
    let w100_200 = &w100 | &w200;

    // Input DWA
    let input = {
        let mut nwa = NWA::new();
        nwa.states.0.clear();

        let s0 = nwa.states.add_state();
        let s1 = nwa.states.add_state();
        let s2 = nwa.states.add_state();
        let s3 = nwa.states.add_state();
        let s4 = nwa.states.add_state();
        let s5 = nwa.states.add_state();

        nwa.body.start_states = vec![s0];

        nwa.add_transition(s0, a, s1, all.clone()).unwrap();
        nwa.add_transition(s0, b, s2, all.clone()).unwrap();
        nwa.add_transition(s1, colon, s3, all.clone()).unwrap();
        nwa.add_transition(s2, colon, s4, all.clone()).unwrap();
        nwa.add_transition(s3, x, s5, all.clone()).unwrap();
        nwa.add_transition(s4, x, s5, all.clone()).unwrap();

        nwa.states[s3].final_weight = Some(w100.clone());
        nwa.states[s4].final_weight = Some(w200.clone());
        nwa.states[s5].final_weight = Some(all.clone());

        nwa.determinize()
    };

    // Expected DWA (optimal)
    let expected = {
        let mut states = DWAStates::default();

        let s0 = states.add_state();
        let s1 = states.add_state();
        let s2 = states.add_state();
        let s34 = states.add_state();
        let s5 = states.add_state();

        states[s0].transitions.insert(a, s1);
        states[s0].trans_weights.insert(a, all.clone());
        states[s0].transitions.insert(b, s2);
        states[s0].trans_weights.insert(b, all.clone());

        states[s1].transitions.insert(colon, s34);
        states[s1].trans_weights.insert(colon, w100.clone());

        states[s2].transitions.insert(colon, s34);
        states[s2].trans_weights.insert(colon, w200.clone());

        states[s34].transitions.insert(x, s5);
        states[s34].trans_weights.insert(x, all.clone());
        states[s34].final_weight = Some(w100_200.clone());

        states[s5].final_weight = Some(all.clone());

        DWA { body: DWABody { start_state: s0 }, states }
    };

    run_push_optimization_test(input, expected);
}

// =============================================================================
// TEST 2: Multiple keys with shared sink (field name pattern)
// =============================================================================
//
// Input:
//   0 ──[key_i]──> i ──[:]──> S_i (fw={i}) ──[value]──> SINK (fw=ALL)
//
// All S_i have identical outgoing but different fw.
//
// Expected: All S_i merged into one state.

#[test]
#[ignore = "Weight pushing algorithm not yet correct"]
fn test_field_name_pattern() {
    let num_fields = 3;
    let colon: Label = 58;
    let value: Label = 200;

    let all = Weight::all();

    // Input DWA
    let input = {
        let mut nwa = NWA::new();
        nwa.states.0.clear();

        let start = nwa.states.add_state();
        nwa.body.start_states = vec![start];

        let sink = nwa.states.add_state();
        nwa.states[sink].final_weight = Some(all.clone());

        for i in 0..num_fields {
            let field_state = nwa.states.add_state();
            let colon_state = nwa.states.add_state();

            nwa.add_transition(start, (100 + i) as Label, field_state, all.clone()).unwrap();
            nwa.add_transition(field_state, colon, colon_state, all.clone()).unwrap();
            nwa.states[colon_state].final_weight = Some(Weight::from_item(i));
            nwa.add_transition(colon_state, value, sink, all.clone()).unwrap();
        }

        nwa.determinize()
    };

    // Expected DWA
    // Structure: start -> field_i -> merged_colon -> sink
    // The merged_colon has fw = {0, 1, 2} and incoming edges have specific weights
    let expected = {
        let mut states = DWAStates::default();

        let start = states.add_state();
        let merged_colon = states.add_state();
        let sink = states.add_state();

        let mut union_fw = Weight::zeros();
        for i in 0..num_fields {
            union_fw = &union_fw | &Weight::from_item(i);
        }

        for i in 0..num_fields {
            let field_state = states.add_state();

            states[start].transitions.insert((100 + i) as Label, field_state);
            states[start].trans_weights.insert((100 + i) as Label, all.clone());

            states[field_state].transitions.insert(colon, merged_colon);
            states[field_state].trans_weights.insert(colon, Weight::from_item(i));
        }

        states[merged_colon].transitions.insert(value, sink);
        states[merged_colon].trans_weights.insert(value, all.clone());
        states[merged_colon].final_weight = Some(union_fw);

        states[sink].final_weight = Some(all.clone());

        DWA { body: DWABody { start_state: start }, states }
    };

    run_push_optimization_test(input, expected);
}

// =============================================================================
// TEST 3: Chain with tightenable weights (simpler, for basic correctness)
// =============================================================================
//
// Input:  A --[{1,2}]--> B --[{2}]--> C (fw={2})
//         Token 1 can never reach acceptance, so A->B can be tightened.
//
// Expected: A --[{2}]--> B --[{2}]--> C (fw={2})
//           Same structure, tighter weights.

#[test]
#[ignore = "Weight pushing algorithm not yet correct"]
fn test_chain_tightening() {
    // Input DWA
    let input = {
        let mut states = DWAStates::default();
        let a = states.add_state();
        let b = states.add_state();
        let c = states.add_state();

        states[a].transitions.insert(0, b);
        states[a].trans_weights.insert(0, Weight::from_iter([1, 2]));

        states[b].transitions.insert(1, c);
        states[b].trans_weights.insert(1, Weight::from_item(2));

        states[c].final_weight = Some(Weight::from_item(2));

        DWA { body: DWABody { start_state: a }, states }
    };

    // Expected DWA (tightened)
    let expected = {
        let mut states = DWAStates::default();
        let a = states.add_state();
        let b = states.add_state();
        let c = states.add_state();

        states[a].transitions.insert(0, b);
        states[a].trans_weights.insert(0, Weight::from_item(2)); // Tightened

        states[b].transitions.insert(1, c);
        states[b].trans_weights.insert(1, Weight::from_item(2));

        states[c].final_weight = Some(Weight::from_item(2));

        DWA { body: DWABody { start_state: a }, states }
    };

    run_push_optimization_test(input, expected);
}
