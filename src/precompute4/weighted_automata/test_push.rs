#![cfg(test)]
//! Tests for weight pushing optimization.
//!
//! STRICT STRUCTURE:
//! 1. Define Input DWA.
//! 2. Define Expected Optimized DWA.
//! 3. Run `run_push_optimization_test(input, expected)`.
//!
//! The runner enforces:
//! - Sanity Check: Input ≡ Expected (must pass).
//! - Optimization Check: (Push + Minimize) Stats == Expected Stats.
//!
//! Since the implementation is currently TODO, all tests MUST FAIL at the final assertion.

use crate::precompute4::weighted_automata::*;
use crate::precompute4::weighted_automata::common::Label;
use crate::precompute4::weighted_automata::dwa::{DWABody, DWAStates};
use crate::precompute4::weighted_automata::test_weighted_automata::stochastic_equivalence_test;

fn dwa_stats(dwa: &DWA) -> (usize, usize) {
    (dwa.states.len(), dwa.states.num_transitions())
}

fn run_push_optimization_test(input: DWA, expected: DWA) {
    // 1. Sanity Check: Input must be semantically equivalent to Expected
    println!("Sanity check: validating Input equivalent to Expected...");
    stochastic_equivalence_test(input.clone(), expected.clone());
    println!("Sanity check passed.");

    // 2. Optimization Potential Check
    let (input_states, input_trans) = dwa_stats(&input);
    let (exp_states, exp_trans) = dwa_stats(&expected);
    
    assert!(
        input_states > exp_states || input_trans > exp_trans,
        "FAULTY TEST: Expected DWA is not smaller than Input DWA.\n\
         Input:    {} states, {} trans\n\
         Expected: {} states, {} trans\n\
         The test must demonstrate an optimization opportunity.",
        input_states, input_trans, exp_states, exp_trans
    );

    // 3. Optimization Check

    let mut pushed = input.clone();
    pushed.residuated_push();
    pushed.simplify();
    let (push_states, push_trans) = dwa_stats(&pushed);

    // This assertion MUST FAIL if weight pushing is not working/implemented
    // (because standard minimization won't achieve the merged state count)
    assert_eq!(
        (push_states, push_trans), (exp_states, exp_trans),
        "Optimization Failed!\n\
         Expected: {} states, {} trans\n\
         Got:      {} states, {} trans\n\
         (Standard minimization failed to merge states that require weight pushing)",
        exp_states, exp_trans, push_states, push_trans
    );
}

// =============================================================================
// TEST 1: Merge Branches (Simple)
// =============================================================================
// Input:
//   0->1 (a), 0->2 (b)
//   1->3 (:), 2->4 (:)
//   3->5 ({100}), 4->5 ({200})
//   3,4 distinct because outgoing weights diff.
//
// Expected:
//   0->1 ({100}), 0->2 ({200})
//   1->34 (:), 2->34 (:)  <-- Merged!
//   34->5 (ALL)           <-- Loosened!

#[test]
fn test_merge_branches() {
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
        let s1 = nwa.states.add_state(); // a
        let s2 = nwa.states.add_state(); // b
        let s3 = nwa.states.add_state(); // : from 1
        let s4 = nwa.states.add_state(); // : from 2
        let s5 = nwa.states.add_state(); // sink
        
        nwa.body.start_states = vec![s0];
        nwa.states[s5].final_weight = Some(all.clone());

        nwa.add_transition(s0, a, s1, all.clone()).unwrap();
        nwa.add_transition(s0, b, s2, all.clone()).unwrap();
        
        nwa.add_transition(s1, colon, s3, all.clone()).unwrap();
        nwa.add_transition(s2, colon, s4, all.clone()).unwrap();

        // Distinct weights on outgoing edges prevent s3/s4 merge in standard min
        nwa.add_transition(s3, x, s5, w100.clone()).unwrap();
        nwa.add_transition(s4, x, s5, w200.clone()).unwrap();
        
        // Final weights must match potential (or be None if not final)
        // Here we just use transitions to define behavior. 
        // s3, s4 are not final. s5 is final.

        nwa.determinize()
    };

    let expected = {
        let mut states = DWAStates::default();
        let s0 = states.add_state();
        let s12 = states.add_state(); // merged 1,2
        let s34 = states.add_state(); // merged 3,4
        let s5 = states.add_state(); // sink

        // 0 -> 12 on a ({100})
        states[s0].transitions.insert(a, s12);
        states[s0].trans_weights.insert(a, w100.clone());
        // 0 -> 12 on b ({200})
        states[s0].transitions.insert(b, s12);
        states[s0].trans_weights.insert(b, w200.clone());

        // 12 -> 34 on :
        states[s12].transitions.insert(colon, s34);
        states[s12].trans_weights.insert(colon, all.clone());

        // 34 -> 5 on x
        states[s34].transitions.insert(x, s5);
        states[s34].trans_weights.insert(x, all.clone());

        states[s5].final_weight = Some(all.clone());

        DWA { body: DWABody { start_state: s0 }, states }
    };

    run_push_optimization_test(input, expected);
}

// =============================================================================
// TEST 2: Field Name Pattern (Scale)
// =============================================================================
// Input:
//   start -> field_i -> colon_i -> sink
//   colon_i -> sink has weight {i}
//   All colon_i states are distinct in Input.
// Expected:
//   All colon_i states merged into ONE state.
//   Weight {i} pushed back to start->field_i.

#[test]
fn test_field_name_optimization() {
    let num_fields = 5;
    let colon: Label = 58;
    let value: Label = 200;
    let all = Weight::all();

    let input = {
        let mut nwa = NWA::new();
        nwa.states.0.clear();
        let start = nwa.states.add_state();
        nwa.body.start_states = vec![start];
        let sink = nwa.states.add_state();
        nwa.states[sink].final_weight = Some(all.clone());

        for i in 0..num_fields {
            let field = nwa.states.add_state();
            let col = nwa.states.add_state();
            
            nwa.add_transition(start, (100+i) as Label, field, all.clone()).unwrap();
            
            // Loose intermediate
            nwa.add_transition(field, colon, col, all.clone()).unwrap();
            
            // Strict outgoing - prevents merging
            nwa.add_transition(col, value, sink, Weight::from_item(i)).unwrap();
        }
        nwa.determinize()
    };

    let expected = {
        let mut states = DWAStates::default();
        let start = states.add_state();
        let merged_field = states.add_state();
        let merged_colon = states.add_state();
        let sink = states.add_state();
        
        // start -> merged_field for all i
        for i in 0..num_fields {
             states[start].transitions.insert((100+i) as Label, merged_field);
             states[start].trans_weights.insert((100+i) as Label, Weight::from_item(i));
        }

        // merged_field -> merged_colon
        states[merged_field].transitions.insert(colon, merged_colon);
        states[merged_field].trans_weights.insert(colon, all.clone());

        // merged_colon -> sink
        states[merged_colon].transitions.insert(value, sink);
        states[merged_colon].trans_weights.insert(value, all.clone());

        states[sink].final_weight = Some(all.clone());

        DWA { body: DWABody { start_state: start }, states }
    };

    run_push_optimization_test(input, expected);
}

// =============================================================================
// TEST 3: Diamond Structure (User Requested)
// =============================================================================
// Input:
//   START -[0, w=ALL]-> A (fw={0}) -[0, w=ALL]-> END (fw={3})
//   START -[1, w=ALL]-> B (fw={1}) -[0, w=ALL]-> END (fw={3})
//   START -[2, w=ALL]-> C (fw={2}) -[0, w=ALL]-> END (fw={3})
//
// States A, B, C are distinct due to fw.
//
// Expected:
//   START -[0, w={0}]-> ABC
//   START -[1, w={1}]-> ABC
//   START -[2, w={2}]-> ABC
//   ABC (fw={0,1,2}) -[0, w=ALL]-> END (fw={3})

#[test]
fn test_diamond_structure() {
    // Labels
    let l0: Label = 0;
    let l1: Label = 1;
    let l2: Label = 2; // For transition inputs
    
    // For weights, we use standard simple weights
    let all = Weight::all();
    let w0 = Weight::from_item(0);
    let w1 = Weight::from_item(1);
    let w2 = Weight::from_item(2);
    let w3 = Weight::from_item(3);
    
    // In expected, ABC final weight is union of {0}, {1}, {2}
    let w012 = &(&w0 | &w1) | &w2;

    let input = {
        let mut nwa = NWA::new();
        nwa.states.0.clear();
        let start = nwa.states.add_state();
        let a = nwa.states.add_state();
        let b = nwa.states.add_state();
        let c = nwa.states.add_state();
        let end = nwa.states.add_state();
        
        nwa.body.start_states = vec![start];

        // START -> A/B/C
        nwa.add_transition(start, l0, a, all.clone()).unwrap(); // Use label 0
        nwa.add_transition(start, l1, b, all.clone()).unwrap(); // Use label 1
        nwa.add_transition(start, l2, c, all.clone()).unwrap(); // Use label 2

        // A/B/C -> END (on label 0)
        nwa.add_transition(a, l0, end, all.clone()).unwrap();
        nwa.add_transition(b, l0, end, all.clone()).unwrap();
        nwa.add_transition(c, l0, end, all.clone()).unwrap();

        nwa.states[a].final_weight = Some(w0.clone());
        nwa.states[b].final_weight = Some(w1.clone());
        nwa.states[c].final_weight = Some(w2.clone());
        
        nwa.states[end].final_weight = Some(w3.clone());

        nwa.determinize()
    };

    let expected = {
        let mut states = DWAStates::default();
        let start = states.add_state();
        let abc = states.add_state(); // Merged A,B,C
        let end = states.add_state();

        // START -> ABC (pushed weights)
        states[start].transitions.insert(l0, abc);
        states[start].trans_weights.insert(l0, w0.clone());

        states[start].transitions.insert(l1, abc);
        states[start].trans_weights.insert(l1, w1.clone());

        states[start].transitions.insert(l2, abc);
        states[start].trans_weights.insert(l2, w2.clone());

        // ABC -> END (loosened A,B,C all had transitions to end with ALL)
        states[abc].transitions.insert(l0, end);
        states[abc].trans_weights.insert(l0, all.clone());
        
        // Final weight of merged state is union of pushed-out components
        states[abc].final_weight = Some(w012.clone());

        states[end].final_weight = Some(w3.clone());

        DWA { body: DWABody { start_state: start }, states }
    };

    run_push_optimization_test(input, expected);
}
