//! Tests ported from sep1's `dwa_i32/tests/test_determinization.rs`.
//!
//! Focuses on determinization behavior: edge cases, epsilon-explosion analysis,
//! and weight-inflation regression tests.

use super::determinize;
use super::dwa::DWA;
use super::nwa::{Label, NWA};
use crate::ds::weight::Weight;

// Re-use helpers from the weighted automata test module.
// Since we can't import from sibling test modules, we duplicate the minimal set.

fn weight_from_item(n: u32) -> Weight {
    Weight::from_compact_ranges(vec![(n..=n, vec![0..=0])])
}

fn weight_from_iter<I: IntoIterator<Item = u32>>(items: I) -> Weight {
    let mut sorted: Vec<u32> = items.into_iter().collect();
    if sorted.is_empty() {
        return Weight::empty();
    }
    sorted.sort_unstable();
    sorted.dedup();
    let mut ranges = Vec::new();
    let mut start = sorted[0];
    let mut end = sorted[0];
    for &item in &sorted[1..] {
        if item == end + 1 {
            end = item;
        } else {
            ranges.push((start..=end, vec![0..=0]));
            start = item;
            end = item;
        }
    }
    ranges.push((start..=end, vec![0..=0]));
    Weight::from_compact_ranges(ranges)
}

fn weight_contains(w: &Weight, item: u32) -> bool {
    !w.intersection(&weight_from_item(item)).is_empty()
}

fn count_transitions(dwa: &DWA) -> usize {
    dwa.states.iter().map(|s| s.transitions.len()).sum()
}

// ============================================================================
// Determinize Edge Cases
// ============================================================================

/// Ported from sep1's `test_determinize_simple_divergence`.
/// In sep1 this is `#[should_panic]` — the assertions on state count fail.
/// In glrmask the determinize returns Result, so we keep `should_panic`
/// to preserve the original test intent. If glrmask handles this correctly
/// (no panic), the test will fail, exposing the behavioral difference.
#[should_panic]
#[test]
fn test_determinize_simple_divergence() {
    let mut nwa = NWA::new(0, 0);
    let s0 = nwa.add_state();
    let s1 = nwa.add_state();
    let s2 = nwa.add_state();
    nwa.add_transition(s0, 'a' as Label, s1, Weight::all());
    nwa.add_transition(s1, 'c' as Label, s2, Weight::all());
    nwa.set_final_weight(s2, weight_from_item(0));

    let s3 = nwa.add_state();
    let s4 = nwa.add_state();
    let s5 = nwa.add_state();
    nwa.add_transition(s3, 'b' as Label, s4, Weight::all());
    nwa.add_transition(s4, 'c' as Label, s5, Weight::all());
    nwa.set_final_weight(s5, weight_from_item(1));

    let start = nwa.add_state();
    nwa.add_epsilon(start, s0, Weight::all());
    nwa.add_epsilon(start, s3, Weight::all());
    nwa.start_states = vec![start];

    let dwa = determinize::determinize(&nwa).expect("determinize failed");
    assert_eq!(dwa.eval_word(&['a' as Label, 'c' as Label]), weight_from_item(0));
    assert_eq!(dwa.eval_word(&['b' as Label, 'c' as Label]), weight_from_item(1));
    assert!(dwa.states.len() <= 4);
}

// ============================================================================
// Epsilon Explosion Analysis Tests
// ============================================================================
//
// These are analytical tests that compare labeled-transition vs epsilon-transition
// NWA structures to study determinization cost differences. They print statistics
// but have no assertions — they exercise the determinize path on various structures.

#[test]
fn test_epsilon_explosion_minimal() {
    const N: usize = 4;
    let char_label: Label = 'x' as Label;

    // LABELED: start --i--> intermediate --char--> final
    let mut nwa_labeled = NWA::new(0, 0);
    let start_labeled = nwa_labeled.add_state();
    nwa_labeled.start_states = vec![start_labeled];

    for i in 0..N {
        let intermediate = nwa_labeled.add_state();
        let final_state = nwa_labeled.add_state();
        nwa_labeled.add_transition(start_labeled, i as Label, intermediate, Weight::all());
        nwa_labeled.add_transition(intermediate, char_label, final_state, Weight::all());
        nwa_labeled.set_final_weight(final_state, weight_from_item(i as u32));
    }

    let dwa_labeled = determinize::determinize(&nwa_labeled).expect("determinize labeled");
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = count_transitions(&dwa_labeled);

    // EPSILON: start --eps--> intermediate --char--> final
    let mut nwa_epsilon = NWA::new(0, 0);
    let start_eps = nwa_epsilon.add_state();
    nwa_epsilon.start_states = vec![start_eps];

    for i in 0..N {
        let intermediate = nwa_epsilon.add_state();
        let final_state = nwa_epsilon.add_state();
        nwa_epsilon.add_epsilon(start_eps, intermediate, Weight::all());
        nwa_epsilon.add_transition(intermediate, char_label, final_state, Weight::all());
        nwa_epsilon.set_final_weight(final_state, weight_from_item(i as u32));
    }

    let dwa_epsilon = determinize::determinize(&nwa_epsilon).expect("determinize epsilon");
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = count_transitions(&dwa_epsilon);

}

#[test]
fn test_epsilon_explosion_diverging_patterns() {
    const N: usize = 4;
    let shared_char: Label = 'a' as Label;

    // LABELED
    let mut nwa_labeled = NWA::new(0, 0);
    let start_labeled = nwa_labeled.add_state();
    nwa_labeled.start_states = vec![start_labeled];

    for i in 0..N {
        let q_i = nwa_labeled.add_state();
        let q_i_a = nwa_labeled.add_state();
        let f_i = nwa_labeled.add_state();
        nwa_labeled.add_transition(start_labeled, i as Label, q_i, Weight::all());
        nwa_labeled.add_transition(q_i, shared_char, q_i_a, Weight::all());
        nwa_labeled.add_transition(q_i_a, (i as i32 + 100) as Label, f_i, Weight::all());
        nwa_labeled.set_final_weight(f_i, weight_from_item(i as u32));
    }

    let dwa_labeled = determinize::determinize(&nwa_labeled).expect("determinize labeled");
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = count_transitions(&dwa_labeled);

    // EPSILON
    let mut nwa_epsilon = NWA::new(0, 0);
    let start_eps = nwa_epsilon.add_state();
    nwa_epsilon.start_states = vec![start_eps];

    for i in 0..N {
        let q_i = nwa_epsilon.add_state();
        let q_i_a = nwa_epsilon.add_state();
        let f_i = nwa_epsilon.add_state();
        nwa_epsilon.add_epsilon(start_eps, q_i, Weight::all());
        nwa_epsilon.add_transition(q_i, shared_char, q_i_a, Weight::all());
        nwa_epsilon.add_transition(q_i_a, (i as i32 + 100) as Label, f_i, Weight::all());
        nwa_epsilon.set_final_weight(f_i, weight_from_item(i as u32));
    }

    let dwa_epsilon = determinize::determinize(&nwa_epsilon).expect("determinize epsilon");
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = count_transitions(&dwa_epsilon);

}

#[test]
fn test_epsilon_explosion_overlapping_alphabet() {
    const N: usize = 4;
    let char_a: Label = 'a' as Label;

    // LABELED
    let mut nwa_labeled = NWA::new(0, 0);
    let start_labeled = nwa_labeled.add_state();
    nwa_labeled.start_states = vec![start_labeled];

    for i in 0..N {
        let mut prev = nwa_labeled.add_state();
        nwa_labeled.add_transition(start_labeled, i as Label, prev, Weight::all());
        for _ in 0..i {
            let next = nwa_labeled.add_state();
            nwa_labeled.add_transition(prev, char_a, next, Weight::all());
            prev = next;
        }
        let final_state = nwa_labeled.add_state();
        nwa_labeled.add_transition(prev, char_a, final_state, Weight::all());
        nwa_labeled.set_final_weight(final_state, weight_from_item(i as u32));
    }

    let dwa_labeled = determinize::determinize(&nwa_labeled).expect("determinize labeled");
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = count_transitions(&dwa_labeled);

    // EPSILON
    let mut nwa_epsilon = NWA::new(0, 0);
    let start_eps = nwa_epsilon.add_state();
    nwa_epsilon.start_states = vec![start_eps];

    for i in 0..N {
        let mut prev = nwa_epsilon.add_state();
        nwa_epsilon.add_epsilon(start_eps, prev, Weight::all());
        for _ in 0..i {
            let next = nwa_epsilon.add_state();
            nwa_epsilon.add_transition(prev, char_a, next, Weight::all());
            prev = next;
        }
        let final_state = nwa_epsilon.add_state();
        nwa_epsilon.add_transition(prev, char_a, final_state, Weight::all());
        nwa_epsilon.set_final_weight(final_state, weight_from_item(i as u32));
    }

    let dwa_epsilon = determinize::determinize(&nwa_epsilon).expect("determinize epsilon");
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = count_transitions(&dwa_epsilon);

}

#[test]
fn test_epsilon_explosion_shared_second_hop() {
    const N: usize = 6;
    let char_x: Label = 'x' as Label;

    // LABELED
    let mut nwa_labeled = NWA::new(0, 0);
    let start_labeled = nwa_labeled.add_state();
    nwa_labeled.start_states = vec![start_labeled];
    let shared_state = nwa_labeled.add_state();
    nwa_labeled.set_final_weight(shared_state, Weight::all());
    for i in 0..N {
        let first_hop = nwa_labeled.add_state();
        nwa_labeled.add_transition(start_labeled, i as Label, first_hop, Weight::all());
        nwa_labeled.add_transition(first_hop, char_x, shared_state, weight_from_item(i as u32));
    }

    let dwa_labeled = determinize::determinize(&nwa_labeled).expect("determinize labeled");
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = count_transitions(&dwa_labeled);

    // EPSILON
    let mut nwa_epsilon = NWA::new(0, 0);
    let start_eps = nwa_epsilon.add_state();
    nwa_epsilon.start_states = vec![start_eps];
    let shared_state_eps = nwa_epsilon.add_state();
    nwa_epsilon.set_final_weight(shared_state_eps, Weight::all());
    for i in 0..N {
        let first_hop = nwa_epsilon.add_state();
        nwa_epsilon.add_epsilon(start_eps, first_hop, Weight::all());
        nwa_epsilon.add_transition(first_hop, char_x, shared_state_eps, weight_from_item(i as u32));
    }

    let dwa_epsilon = determinize::determinize(&nwa_epsilon).expect("determinize epsilon");
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = count_transitions(&dwa_epsilon);

}

#[test]
fn test_epsilon_explosion_shared_then_diverge() {
    const N: usize = 5;

    // LABELED
    let mut nwa_labeled = NWA::new(0, 0);
    let start_labeled = nwa_labeled.add_state();
    nwa_labeled.start_states = vec![start_labeled];
    let shared_second = nwa_labeled.add_state();
    nwa_labeled.set_final_weight(shared_second, weight_from_item(999));
    for i in 0..N {
        let first_hop = nwa_labeled.add_state();
        nwa_labeled.add_transition(start_labeled, i as Label, first_hop, Weight::all());
        nwa_labeled.add_transition(first_hop, 'a' as Label, shared_second, weight_from_item(i as u32));
        let unique_second = nwa_labeled.add_state();
        nwa_labeled.add_transition(first_hop, 'b' as Label, unique_second, weight_from_item(i as u32));
        nwa_labeled.set_final_weight(unique_second, weight_from_item(i as u32));
    }

    let dwa_labeled = determinize::determinize(&nwa_labeled).expect("determinize labeled");
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = count_transitions(&dwa_labeled);

    // EPSILON
    let mut nwa_epsilon = NWA::new(0, 0);
    let start_eps = nwa_epsilon.add_state();
    nwa_epsilon.start_states = vec![start_eps];
    let shared_second_eps = nwa_epsilon.add_state();
    nwa_epsilon.set_final_weight(shared_second_eps, weight_from_item(999));
    for i in 0..N {
        let first_hop = nwa_epsilon.add_state();
        nwa_epsilon.add_epsilon(start_eps, first_hop, Weight::all());
        nwa_epsilon.add_transition(first_hop, 'a' as Label, shared_second_eps, weight_from_item(i as u32));
        let unique_second = nwa_epsilon.add_state();
        nwa_epsilon.add_transition(first_hop, 'b' as Label, unique_second, weight_from_item(i as u32));
        nwa_epsilon.set_final_weight(unique_second, weight_from_item(i as u32));
    }

    let dwa_epsilon = determinize::determinize(&nwa_epsilon).expect("determinize epsilon");
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = count_transitions(&dwa_epsilon);

}

#[test]
fn test_epsilon_explosion_paths_through_shared() {
    const N: usize = 6;

    // LABELED
    let mut nwa_labeled = NWA::new(0, 0);
    let start_labeled = nwa_labeled.add_state();
    nwa_labeled.start_states = vec![start_labeled];
    let final_state = nwa_labeled.add_state();
    nwa_labeled.set_final_weight(final_state, Weight::all());

    let mut first_hops = vec![];
    for i in 0..N {
        let fh = nwa_labeled.add_state();
        nwa_labeled.add_transition(start_labeled, i as Label, fh, Weight::all());
        first_hops.push(fh);
    }
    for i in 0..N - 1 {
        let sh = nwa_labeled.add_state();
        let label = (100 + i) as Label;
        nwa_labeled.add_transition(first_hops[i], label, sh, weight_from_item(i as u32));
        nwa_labeled.add_transition(first_hops[i + 1], label, sh, weight_from_item((i + 1) as u32));
        nwa_labeled.add_transition(sh, 'f' as Label, final_state, Weight::all());
    }

    let dwa_labeled = determinize::determinize(&nwa_labeled).expect("determinize labeled");
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = count_transitions(&dwa_labeled);

    // EPSILON
    let mut nwa_epsilon = NWA::new(0, 0);
    let start_eps = nwa_epsilon.add_state();
    nwa_epsilon.start_states = vec![start_eps];
    let final_state_eps = nwa_epsilon.add_state();
    nwa_epsilon.set_final_weight(final_state_eps, Weight::all());

    let mut first_hops_eps = vec![];
    for _ in 0..N {
        let fh = nwa_epsilon.add_state();
        nwa_epsilon.add_epsilon(start_eps, fh, Weight::all());
        first_hops_eps.push(fh);
    }
    for i in 0..N - 1 {
        let sh = nwa_epsilon.add_state();
        let label = (100 + i) as Label;
        nwa_epsilon.add_transition(first_hops_eps[i], label, sh, weight_from_item(i as u32));
        nwa_epsilon.add_transition(first_hops_eps[i + 1], label, sh, weight_from_item((i + 1) as u32));
        nwa_epsilon.add_transition(sh, 'f' as Label, final_state_eps, Weight::all());
    }

    let dwa_epsilon = determinize::determinize(&nwa_epsilon).expect("determinize epsilon");
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = count_transitions(&dwa_epsilon);

}

#[test]
fn test_epsilon_explosion_many_sources_same_label() {
    const N: usize = 10;
    let shared_label: Label = 10;

    // LABELED
    let mut nwa_labeled = NWA::new(0, 0);
    let start_labeled = nwa_labeled.add_state();
    nwa_labeled.start_states = vec![start_labeled];
    for i in 0..N {
        let first_hop = nwa_labeled.add_state();
        nwa_labeled.add_transition(start_labeled, i as Label, first_hop, Weight::all());
        let target = nwa_labeled.add_state();
        nwa_labeled.add_transition(first_hop, shared_label, target, weight_from_item(i as u32));
        nwa_labeled.set_final_weight(target, weight_from_item(i as u32));
    }

    let dwa_labeled = determinize::determinize(&nwa_labeled).expect("determinize labeled");
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = count_transitions(&dwa_labeled);

    // EPSILON
    let mut nwa_epsilon = NWA::new(0, 0);
    let start_eps = nwa_epsilon.add_state();
    nwa_epsilon.start_states = vec![start_eps];
    for i in 0..N {
        let first_hop = nwa_epsilon.add_state();
        nwa_epsilon.add_epsilon(start_eps, first_hop, Weight::all());
        let target = nwa_epsilon.add_state();
        nwa_epsilon.add_transition(first_hop, shared_label, target, weight_from_item(i as u32));
        nwa_epsilon.set_final_weight(target, weight_from_item(i as u32));
    }

    let dwa_epsilon = determinize::determinize(&nwa_epsilon).expect("determinize epsilon");
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = count_transitions(&dwa_epsilon);

}

#[test]
fn test_epsilon_explosion_many_sources_with_continuation() {
    const N: usize = 20;
    const K: usize = 15;
    let shared_label: Label = 'L' as Label;

    // LABELED
    let mut nwa_labeled = NWA::new(0, 0);
    let start_labeled = nwa_labeled.add_state();
    nwa_labeled.start_states = vec![start_labeled];
    let shared_target = nwa_labeled.add_state();
    let after_shared = nwa_labeled.add_state();
    nwa_labeled.add_transition(shared_target, 'X' as Label, after_shared, Weight::all());
    nwa_labeled.set_final_weight(after_shared, Weight::all());
    for i in 0..N {
        let first_hop = nwa_labeled.add_state();
        nwa_labeled.add_transition(start_labeled, i as Label, first_hop, Weight::all());
        if i < K {
            nwa_labeled.add_transition(first_hop, shared_label, shared_target, weight_from_item(i as u32));
        } else {
            let unique_target = nwa_labeled.add_state();
            nwa_labeled.add_transition(first_hop, 'U' as Label, unique_target, weight_from_item(i as u32));
            nwa_labeled.set_final_weight(unique_target, weight_from_item(i as u32));
        }
    }

    let dwa_labeled = determinize::determinize(&nwa_labeled).expect("determinize labeled");
    let labeled_states = dwa_labeled.states.len();
    let labeled_trans = count_transitions(&dwa_labeled);

    // EPSILON
    let mut nwa_epsilon = NWA::new(0, 0);
    let start_eps = nwa_epsilon.add_state();
    nwa_epsilon.start_states = vec![start_eps];
    let shared_target_eps = nwa_epsilon.add_state();
    let after_shared_eps = nwa_epsilon.add_state();
    nwa_epsilon.add_transition(shared_target_eps, 'X' as Label, after_shared_eps, Weight::all());
    nwa_epsilon.set_final_weight(after_shared_eps, Weight::all());
    for i in 0..N {
        let first_hop = nwa_epsilon.add_state();
        nwa_epsilon.add_epsilon(start_eps, first_hop, Weight::all());
        if i < K {
            nwa_epsilon.add_transition(first_hop, shared_label, shared_target_eps, weight_from_item(i as u32));
        } else {
            let unique_target = nwa_epsilon.add_state();
            nwa_epsilon.add_transition(first_hop, 'U' as Label, unique_target, weight_from_item(i as u32));
            nwa_epsilon.set_final_weight(unique_target, weight_from_item(i as u32));
        }
    }

    let dwa_epsilon = determinize::determinize(&nwa_epsilon).expect("determinize epsilon");
    let epsilon_states = dwa_epsilon.states.len();
    let epsilon_trans = count_transitions(&dwa_epsilon);

}

// ============================================================================
// Weight Inflation Regression Test
// ============================================================================

/// Regression test for acyclic determinization weight inflation bug.
///
/// When two labels reach the same destination powerset (containing both a final
/// and a non-final NWA state), the acyclic determinizer's normalization step
/// can inflate residual weights. If those inflated residuals are then unioned
/// (because the destination DWA state is shared), items from the non-final NWA
/// state leak into the DWA final weight.
///
/// Minimal NWA: 3 states, 4 transitions, 2 labels, 4 items (distinct per label).
///   A --label0--> B (item 0)    A --label0--> C (item 1)
///   A --label1--> B (item 2)    A --label1--> C (item 3)
///   B is final, C is dead.
///
/// Correct:  eval(label 0) = {item 0},  eval(label 1) = {item 2}
/// Buggy:    eval(label 0) = {item 0, item 1}  (item 1 leaks from dead C)
#[test]
fn test_acyclic_determinize_shared_dest_no_weight_inflation() {
    let mut nwa = NWA::new(0, 0);
    let a = nwa.add_state(); // start
    let b = nwa.add_state(); // final
    let c = nwa.add_state(); // dead
    nwa.start_states = vec![a];

    nwa.add_transition(a, 0 as Label, b, weight_from_item(0));
    nwa.add_transition(a, 0 as Label, c, weight_from_item(1));
    nwa.add_transition(a, 1 as Label, b, weight_from_item(2));
    nwa.add_transition(a, 1 as Label, c, weight_from_item(3));

    nwa.set_final_weight(b, Weight::all());

    let dwa = determinize::determinize(&nwa).expect("determinize failed");

    let w0 = dwa.eval_word(&[0 as Label]);
    assert!(weight_contains(&w0, 0), "item 0 should be accepted for label 0 (A→B, final)");
    assert!(!weight_contains(&w0, 1), "item 1 should NOT be accepted for label 0 (A→C, dead)");

    let w1 = dwa.eval_word(&[1 as Label]);
    assert!(weight_contains(&w1, 2), "item 2 should be accepted for label 1 (A→B, final)");
    assert!(!weight_contains(&w1, 3), "item 3 should NOT be accepted for label 1 (A→C, dead)");
}
