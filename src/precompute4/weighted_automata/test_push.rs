#![cfg(test)]
//! Tests for weight pushing optimization.
//!
//! Focus: Verifying that `residuated_push` correctly computes potentials and moves weights
//! to earlier edges in the graph.

use crate::precompute4::weighted_automata::*;
use crate::precompute4::weighted_automata::common::Label;
use crate::precompute4::weighted_automata::dwa::{DWABody, DWAStates};
use crate::precompute4::weighted_automata::test_weighted_automata::stochastic_equivalence_test;

// =============================================================================
// TEST 1: Chain Tightening (Basic Correctness)
// =============================================================================
//
// Input:  A --[{1,2}]--> B --[{2}]--> C (fw={2})
//
// ρ(C) = {2}
// ρ(B) = {2} ∩ {2} = {2}
//
// After push: A->B should be tightened to {2}.

#[test]
fn test_chain_tightening() {
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

    let mut pushed = input.clone();
    pushed.residuated_push();

    // Verify equivalence
    stochastic_equivalence_test(pushed.clone(), input);

    // Verify A->B tightened
    let start = pushed.body.start_state;
    // Note: residuated_push might change start state ID if it optimizes? Usually preserves.
    
    // Find transition 0 from start
    let target_b = *pushed.states[start].transitions.get(&0).expect("Edge 0 missing");
    let weight_ab = pushed.states[start].trans_weights.get(&0).expect("Weight 0 missing");
    
    assert_eq!(weight_ab, &Weight::from_item(2), "A->B should be tightened to {{2}}");
}

// =============================================================================
// TEST 2: Weight Pushing Enables Merging (Manual Check)
// =============================================================================
//
// Input:  1->3 (ALL), 2->4 (ALL)
//         3->5 ({100}), 4->5 ({200})
//         3,4 have fw={100}, {200} respectively.
//
// Check: After push, 1->3 should get {100}, 2->4 should get {200}.
//        3->5 and 4->5 should become ALL (loosened).

#[test]
fn test_weight_movement() {
    let a: Label = 97;
    let b: Label = 98;
    let colon: Label = 58;
    let x: Label = 120;

    let all = Weight::all();
    let w100 = Weight::from_item(100);
    let w200 = Weight::from_item(200);

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
        nwa.add_transition(s1, colon, s3, all.clone()).unwrap(); // 1->3 ALL
        nwa.add_transition(s2, colon, s4, all.clone()).unwrap(); // 2->4 ALL
        
        nwa.add_transition(s3, x, s5, w100.clone()).unwrap(); // 3->5 {100}
        nwa.add_transition(s4, x, s5, w200.clone()).unwrap(); // 4->5 {200}

        nwa.states[s3].final_weight = Some(w100.clone());
        nwa.states[s4].final_weight = Some(w200.clone());
        nwa.states[s5].final_weight = Some(all.clone());

        nwa.determinize()
    };

    let mut pushed = input.clone();
    pushed.residuated_push();

    stochastic_equivalence_test(pushed.clone(), input);

    // Inspect weights in pushed DWA
    // Since determinize might reorder states, we traverse to find them
    let s0 = pushed.body.start_state;
    // Edges from start: 0->1 (a) and 0->2 (b)
    let w_0_1 = pushed.states[s0].trans_weights.get(&a).unwrap();
    let w_0_2 = pushed.states[s0].trans_weights.get(&b).unwrap();
    
    // Weights should be pushed all the way to start edges
    assert_eq!(w_0_1, &w100, "0->1 should receive pushed weight {{100}}");
    assert_eq!(w_0_2, &w200, "0->2 should receive pushed weight {{200}}");

    let s1 = *pushed.states[s0].transitions.get(&a).unwrap();
    let s2 = *pushed.states[s0].transitions.get(&b).unwrap();
    
    // Intermediate edges should be loosened to ALL (relative to context)
    // 1->3 (colon)
    let w_1_3 = pushed.states[s1].trans_weights.get(&colon).unwrap();
    let w_2_4 = pushed.states[s2].trans_weights.get(&colon).unwrap();

    assert_eq!(w_1_3, &all, "1->3 should be loosened to ALL");
    assert_eq!(w_2_4, &all, "2->4 should be loosened to ALL");

    let s3 = *pushed.states[s1].transitions.get(&colon).unwrap();
    let s4 = *pushed.states[s2].transitions.get(&colon).unwrap();

    // Check loosening
    // outgoing from 3 on x
    let w_3_5 = pushed.states[s3].trans_weights.get(&x).unwrap();
    // outgoing from 4 on x
    let w_4_5 = pushed.states[s4].trans_weights.get(&x).unwrap();

    assert_eq!(w_3_5, &all, "3->5 should be loosened to ALL");
    assert_eq!(w_4_5, &all, "4->5 should be loosened to ALL");

    // Check final weights on 3 and 4
    let fw_3 = pushed.states[s3].final_weight.as_ref().unwrap();
    let fw_4 = pushed.states[s4].final_weight.as_ref().unwrap();
    
    // fw' = ¬ρ ∪ fw. ρ={100}. fw={100}. ¬{100} ∪ {100} = ALL.
    assert!(fw_3.is_all_fast() || fw_3 == &all, "fw(3) should be loosened to ALL");
    assert!(fw_4.is_all_fast() || fw_4 == &all, "fw(4) should be loosened to ALL");
}

// =============================================================================
// TEST 3: Field Name Pattern (Multi-step push)
// =============================================================================
// 
// start -> field -> colon -> sink
// colon->sink has weight {i}
// Push should move {i} to field->colon, then to start->field.

#[test]
fn test_field_name_push() {
    let colon: Label = 58;
    let value: Label = 200;
    let all = Weight::all();
    let w_field0 = Weight::from_item(0);

    let input = {
        let mut nwa = NWA::new();
        nwa.states.0.clear();
        let start = nwa.states.add_state();
        nwa.body.start_states = vec![start];
        let sink = nwa.states.add_state();
        nwa.states[sink].final_weight = Some(all.clone());

        // Field 0
        let field = nwa.states.add_state();
        let col = nwa.states.add_state();

        nwa.add_transition(start, 100, field, all.clone()).unwrap();
        nwa.add_transition(field, colon, col, all.clone()).unwrap();
        nwa.states[col].final_weight = Some(w_field0.clone());
        nwa.add_transition(col, value, sink, w_field0.clone()).unwrap(); // {0}

        nwa.determinize()
    };

    let mut pushed = input.clone();
    pushed.residuated_push();
    stochastic_equivalence_test(pushed.clone(), input);

    // Traverse
    let s0 = pushed.body.start_state;
    let s_field = *pushed.states[s0].transitions.get(&100).unwrap();
    let s_col = *pushed.states[s_field].transitions.get(&colon).unwrap();

    // Check weights
    let w_start_field = pushed.states[s0].trans_weights.get(&100).unwrap();
    let w_field_col = pushed.states[s_field].trans_weights.get(&colon).unwrap();
    let w_col_sink = pushed.states[s_col].trans_weights.get(&value).unwrap();

    assert_eq!(w_start_field, &w_field0, "Weight {{0}} should be pushed to start->field");
    assert_eq!(w_field_col, &all, "field->col should be loosened to ALL (relative to context)"); 
    // Wait: field->col gets pushed input {0}. Output Pushed to start->field.
    // w'(field->col) = ¬ρ(field) ∪ (w ∩ ρ(col)).
    // ρ(col) = {0}. w = ALL. w_tight = {0}.
    // ρ(field) = {0}.
    // w' = ¬{0} ∪ {0} = ALL.
    // Correct.
    
    // w'(col->sink) = ¬ρ(col) ∪ (w ∩ ρ(sink)).
    // ρ(col) = {0}. w = {0}. ρ(sink) = ALL. w_tight = {0}.
    // w' = ¬{0} ∪ {0} = ALL.
    
    assert_eq!(w_col_sink, &all, "col->sink should be loosened to ALL");
}
