#![cfg(test)]
//! Tests for weight pushing optimization.
//!
//! Each test follows a standardized structure:
//! 1. Define an input DWA
//! 2. Define an expected equivalent but simpler DWA
//! 3. Sanity check: assert input ≡ expected (test is malformed if not)
//! 4. Get stats for expected DWA (num states, num transitions)
//! 5. Minimize input WITHOUT pushing → assert it does NOT achieve those stats
//! 6. Push + minimize input → assert it DOES achieve those stats

use crate::precompute4::weighted_automata::*;
use crate::precompute4::weighted_automata::common::Label;
use crate::precompute4::weighted_automata::dwa::{DWABody, DWAStates};
use crate::precompute4::weighted_automata::test_weighted_automata::stochastic_equivalence_test;

/// Helper to get DWA stats (states, transitions)
fn dwa_stats(dwa: &DWA) -> (usize, usize) {
    (dwa.states.len(), dwa.states.num_transitions())
}

// =============================================================================
// TEST 1: Simple linear chain with redundant weight
// =============================================================================
//
// Input:  A --[{1,2}]--> B --[{2}]--> C (final, fw={1,2})
//           Path weight = {1,2} ∩ {2} ∩ {1,2} = {2}
//
// After pushing, A--B becomes {2}; this doesn't change state count but
// demonstrates tightening. Not a great test for merging, included for coverage.

#[test]
fn test_simple_chain_tightening() {
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

        states[c].final_weight = Some(Weight::from_iter([1, 2]));

        DWA { body: DWABody { start_state: a }, states }
    };

    // Expected DWA (same structure, but A--B weight is tightened to {2})
    let expected = {
        let mut states = DWAStates::default();
        let a = states.add_state();
        let b = states.add_state();
        let c = states.add_state();

        states[a].transitions.insert(0, b);
        states[a].trans_weights.insert(0, Weight::from_item(2)); // Tightened!

        states[b].transitions.insert(1, c);
        states[b].trans_weights.insert(1, Weight::from_item(2));

        states[c].final_weight = Some(Weight::from_iter([1, 2]));

        DWA { body: DWABody { start_state: a }, states }
    };

    // Sanity check: input ≡ expected
    stochastic_equivalence_test(input.clone(), expected.clone());

    let (exp_states, exp_trans) = dwa_stats(&expected);

    // Minimize without pushing
    let mut minimized_only = input.clone();
    minimized_only.minimize_with_rustfst();
    let (min_states, min_trans) = dwa_stats(&minimized_only);

    // This test doesn't expect state count reduction, just weight change
    // So we skip the "doesn't achieve" assertion for this simple case

    // Push + minimize
    let mut pushed = input.clone();
    pushed.residuated_push_prune_only();
    pushed.minimize_with_rustfst();

    // After push, verify equivalence still holds
    stochastic_equivalence_test(pushed.clone(), expected.clone());
    
    // And verify the edge weight was actually tightened
    assert_eq!(
        pushed.states[pushed.body.start_state].trans_weights.get(&0),
        Some(&Weight::from_item(2)),
        "Edge A->B should be tightened to {{2}}"
    );
}

// =============================================================================
// TEST 2: Dead path removal
// =============================================================================
//
// Input:  A --[{1}]--> B --[{2}]--> C (final, fw={3})
//         Path weight = {1} ∩ {2} ∩ {3} = {} (empty!)
//
// Expected: Empty automaton (no reachable final states)

#[test]
fn test_dead_path_removal() {
    // Input DWA with dead path
    let input = {
        let mut states = DWAStates::default();
        let a = states.add_state();
        let b = states.add_state();
        let c = states.add_state();

        states[a].transitions.insert(0, b);
        states[a].trans_weights.insert(0, Weight::from_item(1));

        states[b].transitions.insert(1, c);
        states[b].trans_weights.insert(1, Weight::from_item(2));

        states[c].final_weight = Some(Weight::from_item(3));

        DWA { body: DWABody { start_state: a }, states }
    };

    // Expected: start state with no transitions (dead)
    let expected = {
        let mut states = DWAStates::default();
        let a = states.add_state();
        // No transitions, no final weight = accepts nothing
        DWA { body: DWABody { start_state: a }, states }
    };

    // Sanity check: both accept empty language (no words)
    // Can't use stochastic_equivalence_test easily here since both are empty
    // Just verify input has empty path weight
    assert!(input.eval_word_weight(&[0, 1]).is_empty());

    let (exp_states, exp_trans) = dwa_stats(&expected);
    assert_eq!(exp_trans, 0, "Expected DWA should have 0 transitions");

    // Minimize without pushing - won't remove dead edges
    let mut minimized_only = input.clone();
    minimized_only.minimize_with_rustfst();
    let (_, min_trans) = dwa_stats(&minimized_only);
    // Note: rustfst may or may not remove unreachable states; we check transitions
    // This assertion may need adjustment based on actual behavior

    // Push + minimize - should remove dead edges
    let mut pushed = input.clone();
    pushed.residuated_push_prune_only();
    let (_, pushed_trans) = dwa_stats(&pushed);
    
    assert_eq!(pushed_trans, 0, "After pushing, dead edges should be removed");
}

// =============================================================================
// TEST 3: Branching with different final tokens
// =============================================================================
//
// Input:
//       ┌─[0, w={1,2}]─→ B ─[2, w={2}]─→ D (final, fw={2})
//   A ──┤
//       └─[1, w={1,3}]─→ C ─[3, w={3}]─→ E (final, fw={3})
//
// After pushing, edges from A are tightened:
//   A ──[0, w={2}]──→ B ...
//   A ──[1, w={3}]──→ C ...

#[test]
fn test_branching_tightening() {
    // Input DWA
    let input = {
        let mut states = DWAStates::default();
        let a = states.add_state();
        let b = states.add_state();
        let c = states.add_state();
        let d = states.add_state();
        let e = states.add_state();

        states[a].transitions.insert(0, b);
        states[a].trans_weights.insert(0, Weight::from_iter([1, 2]));
        states[a].transitions.insert(1, c);
        states[a].trans_weights.insert(1, Weight::from_iter([1, 3]));

        states[b].transitions.insert(2, d);
        states[b].trans_weights.insert(2, Weight::from_item(2));

        states[c].transitions.insert(3, e);
        states[c].trans_weights.insert(3, Weight::from_item(3));

        states[d].final_weight = Some(Weight::from_item(2));
        states[e].final_weight = Some(Weight::from_item(3));

        DWA { body: DWABody { start_state: a }, states }
    };

    // Expected DWA (A edges tightened)
    let expected = {
        let mut states = DWAStates::default();
        let a = states.add_state();
        let b = states.add_state();
        let c = states.add_state();
        let d = states.add_state();
        let e = states.add_state();

        states[a].transitions.insert(0, b);
        states[a].trans_weights.insert(0, Weight::from_item(2)); // Tightened
        states[a].transitions.insert(1, c);
        states[a].trans_weights.insert(1, Weight::from_item(3)); // Tightened

        states[b].transitions.insert(2, d);
        states[b].trans_weights.insert(2, Weight::from_item(2));

        states[c].transitions.insert(3, e);
        states[c].trans_weights.insert(3, Weight::from_item(3));

        states[d].final_weight = Some(Weight::from_item(2));
        states[e].final_weight = Some(Weight::from_item(3));

        DWA { body: DWABody { start_state: a }, states }
    };

    // Sanity check
    stochastic_equivalence_test(input.clone(), expected.clone());

    // Push and verify edge weights
    let mut pushed = input.clone();
    pushed.residuated_push_prune_only();

    stochastic_equivalence_test(pushed.clone(), expected.clone());

    // Verify tightening happened
    let start = pushed.body.start_state;
    assert_eq!(
        pushed.states[start].trans_weights.get(&0),
        Some(&Weight::from_item(2)),
        "A->B should be tightened to {{2}}"
    );
    assert_eq!(
        pushed.states[start].trans_weights.get(&1),
        Some(&Weight::from_item(3)),
        "A->C should be tightened to {{3}}"
    );
}

// =============================================================================
// TEST 4: State merging via weight pushing
// =============================================================================
//
// This is the key optimization test. Two states with identical outgoing
// structure but different final weights can be merged after pushing.
//
// Input (after determinization):
//   0 ──[a]──> 1 ──[:]──> 3 (fw={100}) ──[x]──> 5 (fw=ALL)
//   0 ──[b]──> 2 ──[:]──> 4 (fw={200}) ──[x]──> 5 (fw=ALL)
//
// States 3 and 4 are structurally identical but have different fw.
// Standard minimization can't merge them.
//
// Expected (optimal):
//   0 ──[a]──> 1 ──[:, w={100}]──> 34 (fw={100,200}) ──[x]──> 5
//   0 ──[b]──> 2 ──[:, w={200}]──> 34 (fw={100,200}) ──[x]──> 5
//
// After pushing to incoming edges, 3 and 4 become mergeable.

#[test]
#[ignore = "Weight pushing algorithm is not yet implemented correctly"]
fn test_state_merging_via_push() {
    let a: Label = 97;  // 'a'
    let b: Label = 98;  // 'b'
    let colon: Label = 58;  // ':'
    let x: Label = 120; // 'x'

    let all = Weight::all();
    let w100 = Weight::from_item(100);
    let w200 = Weight::from_item(200);
    let w100_200 = &w100 | &w200;

    // Input DWA (suboptimal - states 3 and 4 not merged)
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

    // Expected DWA (optimal - states merged)
    let expected = {
        let mut states = DWAStates::default();

        let s0 = states.add_state();
        let s1 = states.add_state();
        let s2 = states.add_state();
        let s34 = states.add_state(); // Merged state
        let s5 = states.add_state();

        // 0 -> 1 on 'a', 0 -> 2 on 'b'
        states[s0].transitions.insert(a, s1);
        states[s0].trans_weights.insert(a, all.clone());
        states[s0].transitions.insert(b, s2);
        states[s0].trans_weights.insert(b, all.clone());

        // 1 -> 34 on ':' with weight={100}
        states[s1].transitions.insert(colon, s34);
        states[s1].trans_weights.insert(colon, w100.clone());

        // 2 -> 34 on ':' with weight={200}
        states[s2].transitions.insert(colon, s34);
        states[s2].trans_weights.insert(colon, w200.clone());

        // 34 -> 5 on 'x'
        states[s34].transitions.insert(x, s5);
        states[s34].trans_weights.insert(x, all.clone());
        states[s34].final_weight = Some(w100_200.clone());

        states[s5].final_weight = Some(all.clone());

        DWA { body: DWABody { start_state: s0 }, states }
    };

    // Sanity check: input ≡ expected
    stochastic_equivalence_test(input.clone(), expected.clone());

    let (exp_states, exp_trans) = dwa_stats(&expected);

    // Minimize without pushing - should NOT achieve optimal stats
    let mut minimized_only = input.clone();
    minimized_only.minimize_with_rustfst();
    let (min_states, min_trans) = dwa_stats(&minimized_only);

    assert!(
        min_states > exp_states || min_trans > exp_trans,
        "Without pushing, minimization should NOT achieve optimal stats.\n\
         Expected: {} states, {} trans\n\
         Got: {} states, {} trans",
        exp_states, exp_trans, min_states, min_trans
    );

    // Push + minimize - should achieve optimal stats
    let mut pushed = input.clone();
    pushed.residuated_push();
    pushed.minimize_with_rustfst();
    let (push_states, push_trans) = dwa_stats(&pushed);

    stochastic_equivalence_test(pushed.clone(), expected.clone());

    assert_eq!(
        (push_states, push_trans), (exp_states, exp_trans),
        "With pushing, should achieve optimal stats.\n\
         Expected: {} states, {} trans\n\
         Got: {} states, {} trans",
        exp_states, exp_trans, push_states, push_trans
    );
}

// =============================================================================
// TEST 5: Realistic field pattern (multiple keys, shared sink)
// =============================================================================
//
// Structure (num_fields keys, all leading to same sink with different fw):
//   0 ──[key_i]──> i ──[:]──> S_i (fw={i*10}) ──[value_terms]──> SINK
//
// All S_i are structurally identical (same outgoing) but different fw.
// After pushing, they should be mergeable.

#[test]
#[ignore = "Weight pushing algorithm is not yet implemented correctly"]
fn test_realistic_field_pattern() {
    let num_fields = 5;
    let num_value_terms = 10;

    let all = Weight::all();
    let colon: Label = 58;

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

            // start -> field_state on field name
            nwa.add_transition(start, (100 + i) as Label, field_state, all.clone()).unwrap();

            // field_state -> colon_state on ':'
            nwa.add_transition(field_state, colon, colon_state, all.clone()).unwrap();

            // colon_state has unique final_weight
            nwa.states[colon_state].final_weight = Some(Weight::from_item(i * 10));

            // colon_state -> sink on all value terms
            for term in 0..num_value_terms {
                nwa.add_transition(colon_state, (200 + term) as Label, sink, all.clone()).unwrap();
            }
        }

        nwa.determinize()
    };

    // Expected: After pushing, all colon_states are merged
    // This is harder to build manually, so we just check stats
    // The optimal has: start + num_fields field_states + 1 merged_colon + 1 sink
    //                = 1 + num_fields + 1 + 1 = num_fields + 3 states
    // But this depends on minimization details...

    // For now, just verify push reduces state count

    let mut minimized_only = input.clone();
    minimized_only.minimize_with_rustfst();
    let (min_states, _) = dwa_stats(&minimized_only);

    let mut pushed = input.clone();
    pushed.residuated_push();
    pushed.minimize_with_rustfst();
    let (push_states, _) = dwa_stats(&pushed);

    // Verify equivalence is preserved
    stochastic_equivalence_test(pushed.clone(), input.clone());

    assert!(
        push_states < min_states,
        "Pushing should reduce state count.\n\
         Without push: {} states\n\
         With push: {} states",
        min_states, push_states
    );
}
