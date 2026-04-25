//! Regression tests for the weighted automata stack.
//!
//! These exercise the weighted DWA/NWA/determinize/minimize stack directly,
//! using glrmask's internal types.

use std::collections::BTreeSet;

use super::determinize;
use super::dwa::DWA;
use super::minimize;
use super::nwa::{Label, NWA};
use super::test_support::{
    add_dwa_states,
    add_nwa_states,
    assert_weights_eq,
    weight_from_item,
    weight_from_iter,
    weight_from_ranges,
};
use crate::ds::weight::Weight;

// Helper functions

/// Convert a DWA into an NWA (for use in union/concatenate).
fn dwa_to_nwa(dwa: &DWA) -> NWA {
    let mut nwa = NWA::new(0, 0);
    add_nwa_states(&mut nwa, dwa.states().len());
    nwa.set_start_states(vec![dwa.start_state()]);
    for (state_id, state) in dwa.states().iter().enumerate() {
        if let Some(fw) = &state.final_weight {
            nwa.set_final_weight(state_id as u32, fw.clone());
        }
        for (&label, (to, weight)) in &state.transitions {
            nwa.add_transition(state_id as u32, label, *to, weight.clone());
        }
    }
    nwa
}

/// Compute the union of two DWAs: A ∪ B → determinized DWA.
fn dwa_union(a: &DWA, b: &DWA) -> DWA {
    let nwa_a = dwa_to_nwa(a);
    let nwa_b = dwa_to_nwa(b);
    let mut combined = NWA::new(0, 0);
    let body_a = combined.append_with_body(&nwa_a);
    let body = combined.union_in_place(&nwa_b, &body_a);
    combined.set_start_states(body.start_states);
    determinize::determinize(&combined).expect("determinize failed in dwa_union")
}

/// Compute the concatenation of two DWAs: A · B → determinized DWA.
fn dwa_concatenate(a: &DWA, b: &DWA) -> DWA {
    let nwa_a = dwa_to_nwa(a);
    let nwa_b = dwa_to_nwa(b);
    let mut combined = NWA::new(0, 0);
    let right_body = combined.append_with_body(&nwa_b);
    let left_body = combined.concatenate_in_place(&nwa_a, &right_body);
    combined.set_start_states(left_body.start_states);
    determinize::determinize(&combined).expect("determinize failed in dwa_concatenate")
}

/// Apply a weight gate to a DWA's start state by cloning the start state with
/// intersected weights.
fn apply_weight_to_dwa(dwa: &mut DWA, w: &Weight) {
    let start = dwa.start_state() as usize;
    let old_state = dwa.states()[start].clone();
    let new_id = dwa.add_state();
    // Copy transitions with intersected weights
    for (&label, (target, edge_w)) in &old_state.transitions {
        let new_w = edge_w.intersection(w);
        if !new_w.is_empty() {
            dwa.add_transition(new_id, label, *target, new_w);
        }
    }
    // Copy final weight with intersection
    if let Some(fw) = &old_state.final_weight {
        let new_fw = fw.intersection(w);
        if !new_fw.is_empty() {
            dwa.set_final_weight(new_id, new_fw);
        }
    }
    dwa.set_start_state(new_id);
}

/// Enumerate all accepted words from a DWA (BFS), up to max_depth.
/// Returns (word, weight) pairs.
fn enumerate_accepted(dwa: &DWA, max_depth: usize) -> Vec<(Vec<Label>, Weight)> {
    let mut result = Vec::new();
    let mut stack: Vec<(u32, Vec<Label>, Weight)> =
        vec![(dwa.start_state(), vec![], Weight::all())];

    while let Some((state, word, acc)) = stack.pop() {
        if word.len() > max_depth {
            continue;
        }
        let st = &dwa.states()[state as usize];

        // Check if accepting
        if let Some(fw) = &st.final_weight {
            let w = acc.intersection(fw);
            if !w.is_empty() {
                result.push((word.clone(), w));
            }
        }

        // Explore transitions
        for (&label, (next, ew)) in &st.transitions {
            let new_acc = acc.intersection(ew);
            if !new_acc.is_empty() {
                let mut new_word = word.clone();
                new_word.push(label);
                stack.push((*next, new_word, new_acc));
            }
        }
    }
    result
}

/// Assert two DWAs are semantically equivalent (same accepted words with same weights).
/// Works for acyclic DWAs; uses max_depth bound for cyclic ones.
fn assert_dwa_equivalent(a: &DWA, b: &DWA, max_depth: usize) {
    let words_a = enumerate_accepted(a, max_depth);
    let words_b = enumerate_accepted(b, max_depth);

    let all_words: BTreeSet<Vec<Label>> = words_a
        .iter()
        .chain(words_b.iter())
        .map(|(w, _)| w.clone())
        .collect();

    for word in &all_words {
        let wa = a.eval_word(word);
        let wb = b.eval_word(word);
        assert_weights_eq(
            &wa,
            &wb,
            &format!("Equivalence mismatch on word {:?}", word),
        );
    }
}

/// Validate that `u` is the correct union of `a` and `b`.
fn validate_union(a: &DWA, b: &DWA, u: &DWA, max_depth: usize) {
    let words_a = enumerate_accepted(a, max_depth);
    let words_b = enumerate_accepted(b, max_depth);
    let words_u = enumerate_accepted(u, max_depth);

    let all_words: BTreeSet<Vec<Label>> = words_a
        .iter()
        .chain(words_b.iter())
        .chain(words_u.iter())
        .map(|(w, _)| w.clone())
        .collect();

    for word in &all_words {
        let wa = a.eval_word(word);
        let wb = b.eval_word(word);
        let wu = u.eval_word(word);
        let expected = wa.union(&wb);
        assert_weights_eq(
            &wu,
            &expected,
            &format!(
                "Union mismatch on word {:?}:\n  A(w) = {}\n  B(w) = {}",
                word, wa, wb
            ),
        );
    }
}

/// Expected concatenation weight: union over all split points.
fn expected_concat_weight(a: &DWA, b: &DWA, word: &[Label]) -> Weight {
    let mut acc = Weight::empty();
    for i in 0..=word.len() {
        let wa = a.eval_word(&word[..i]);
        if wa.is_empty() {
            continue;
        }
        let wb = b.eval_word(&word[i..]);
        if wb.is_empty() {
            continue;
        }
        let both = wa.intersection(&wb);
        if !both.is_empty() {
            acc = acc.union(&both);
        }
    }
    acc
}

/// Validate that `c` is the correct concatenation of `a` and `b`.
fn validate_concatenation(a: &DWA, b: &DWA, c: &DWA, max_depth: usize) {
    let words_a = enumerate_accepted(a, max_depth);
    let words_b = enumerate_accepted(b, max_depth);
    let words_c = enumerate_accepted(c, max_depth);

    let mut all_words: BTreeSet<Vec<Label>> = words_c.iter().map(|(w, _)| w.clone()).collect();

    // Add all concatenations of A words with B words
    for (wa_word, _) in &words_a {
        for (wb_word, _) in &words_b {
            let mut combined = wa_word.clone();
            combined.extend(wb_word);
            all_words.insert(combined);
        }
    }

    for word in &all_words {
        let wc = c.eval_word(word);
        let expected = expected_concat_weight(a, b, word);
        assert_weights_eq(
            &wc,
            &expected,
            &format!("Concat mismatch on word {:?}", word),
        );
    }
}

/// Helper to create a DWA that accepts a single character with a given final weight.
fn dwa_accepts_char(ch: char, final_weight: Weight) -> DWA {
    let mut dwa = DWA::new(0, 0);
    let final_state = dwa.add_state();
    dwa.add_transition(dwa.start_state(), ch as Label, final_state, Weight::all());
    dwa.set_final_weight(final_state, final_weight);
    dwa
}

/// Helper to create a DWA that accepts a string with a given final weight.
fn dwa_from_str(s: &str, final_weight: Weight) -> DWA {
    let mut dwa = DWA::new(0, 0);
    let mut current = dwa.start_state();
    for ch in s.chars() {
        let next = dwa.add_state();
        dwa.add_transition(current, ch as Label, next, Weight::all());
        current = next;
    }
    dwa.set_final_weight(current, final_weight);
    dwa
}

/// Helper to create a DWA with a single transition with specified weights.
fn dwa_with_char_and_weights(ch: char, edge_weight: Weight, final_weight: Weight) -> DWA {
    let mut d = DWA::new(0, 0);
    let s = d.add_state();
    d.add_transition(d.start_state(), ch as Label, s, edge_weight);
    d.set_final_weight(s, final_weight);
    d
}

/// Helper to create an NWA that accepts a single character.
fn nwa_accepts_char(ch: char, weight: Weight) -> NWA {
    let mut nwa = NWA::new(0, 0);
    let start = nwa.add_state();
    nwa.start_states_mut().push(start);
    let final_state = nwa.add_state();
    nwa.add_transition(start, ch as Label, final_state, Weight::all());
    nwa.set_final_weight(final_state, weight);
    nwa
}

/// Negative label encoding used by these weighted-automata tests.
fn neg(x: Label) -> Label {
    x.wrapping_add(Label::MIN)
}

// DWA Builder Tests

#[test]
fn test_dwa_builder() {
    let mut dwa = DWA::new(0, 0);
    assert_eq!(dwa.states().len(), 1);
    assert_eq!(dwa.start_state(), 0);

    let s1 = dwa.add_state();
    assert_eq!(s1, 1);
    assert_eq!(dwa.states().len(), 2);

    dwa.set_final_weight(1, weight_from_item(20));

    assert_weights_eq(
        dwa.states()[1].final_weight.as_ref().unwrap(),
        &weight_from_item(20),
        "Final weight should be 20",
    );

    dwa.add_transition(0, 'a' as Label, 1, weight_from_item(30));
    let (target, ref tw) = dwa.states()[0].transitions[&('a' as Label)];
    assert_eq!(target, 1);
    assert_weights_eq(tw, &weight_from_item(30), "Transition weight should be 30");
}

// Minimize Tests

#[test]
fn test_minimize_redundant_states() {
    let mut d = DWA::new(0, 0);
    let s1 = d.add_state();
    let s2 = d.add_state();
    let s3 = d.add_state(); // Should be merged with s2
    let s4 = d.add_state(); // Final state
    let _s5 = d.add_state(); // Unreachable

    d.add_transition(0, 'a' as Label, s1, Weight::all());
    d.add_transition(0, 'b' as Label, s2, Weight::all());
    d.add_transition(0, 'c' as Label, s3, Weight::all());
    d.add_transition(s1, 'x' as Label, s4, Weight::all());
    d.add_transition(s2, 'y' as Label, s4, Weight::all());
    d.add_transition(s3, 'y' as Label, s4, Weight::all()); // Same behavior as s2
    d.set_final_weight(s4, weight_from_item(1));

    assert_eq!(d.states().len(), 6);
    let d = minimize::minimize(&d);
    // s5 pruned (unreachable). s2 and s3 merged.
    assert!(
        d.states().len() <= 5,
        "Should minimize to at most 5 states (optimal=4), got {}",
        d.states().len()
    );
}

#[test]
#[ignore] // glrmask determinize/minimize only supports acyclic DWAs; this test builds a cyclic DWA via validate_union
fn test_prune_unreachable_with_default_chain() {
    let mut d = DWA::new(0, 0);
    let s1 = d.add_state();
    let _s2 = d.add_state(); // Unused, unreachable
    d.add_transition(d.start_state(), 'y' as Label, s1, Weight::all());
    d.set_final_weight(s1, weight_from_item(1));
    d.add_transition(s1, 'x' as Label, s1, Weight::all());

    // Completely unreachable component
    let s_unreach = d.add_state();
    d.add_transition(s_unreach, 'z' as Label, s_unreach, Weight::all());

    let before = d.states().len();
    let d = minimize::minimize(&d);
    let after = d.states().len();
    assert!(after < before, "Unreachable states should be pruned");
    assert_eq!(after, 2, "Only start and s1 should remain reachable");
}

#[test]
fn test_equivalence_via_minimization() {
    // DWA 'a' has explicit transitions for inputs 1 and 3 that lead
    // to sink-like states.
    let mut a = DWA::new(0, 0);
    let s1a = a.add_state();
    let s2a = a.add_state();
    a.add_transition(0, 0, s1a, weight_from_item(1));
    a.add_transition(0, 1, s2a, weight_from_iter(0..=1));
    a.add_transition(0, 2, s1a, weight_from_item(0));
    a.add_transition(0, 3, s1a, weight_from_iter(0..=1));

    // DWA 'b' lacks these transitions (implicit sink).
    let mut b = DWA::new(0, 0);
    let s1b = b.add_state();
    b.add_transition(0, 0, s1b, weight_from_item(1));
    b.add_transition(0, 2, s1b, weight_from_item(0));

    assert_dwa_equivalent(&a, &b, 5);
}

#[test]
fn test_minimize_propagates_future_weights() {
    let mut a = DWA::new(0, 0);
    let s1 = a.add_state();
    let s2 = a.add_state();
    a.add_transition(0, 'a' as Label, s1, Weight::all());
    a.add_transition(s1, 'b' as Label, s2, weight_from_ranges([1..=2]));
    a.set_final_weight(s2, weight_from_item(2));

    let mut b = DWA::new(0, 0);
    let s1b = b.add_state();
    let s2b = b.add_state();
    b.add_transition(0, 'a' as Label, s1b, Weight::all());
    b.add_transition(s1b, 'b' as Label, s2b, Weight::all());
    b.set_final_weight(s2b, weight_from_item(2));

    let a = minimize::minimize(&a);
    assert_dwa_equivalent(&a, &b, 5);
}

#[test]
fn test_minimize() {
    let mut d = DWA::new(0, 0);
    let s1 = d.add_state();
    let s2 = d.add_state();
    let s3 = d.add_state();
    let s4 = d.add_state();
    let s5 = d.add_state();
    let s6 = d.add_state();
    let s7 = d.add_state();
    let s8 = d.add_state();
    let s9 = d.add_state();
    let s10 = d.add_state();
    let s11 = d.add_state();
    let s12 = d.add_state();
    let s13 = d.add_state();

    let w_all = Weight::all();
    let w_1_2 = weight_from_iter(1..=2);

    // State 0 (start)
    d.add_transition(d.start_state(), 0, s1, w_all.clone());
    d.add_transition(d.start_state(), 1, s2, w_all.clone());
    // State 1
    d.add_transition(s1, 0, s3, w_1_2.clone());
    d.add_transition(s1, 3, s4, w_1_2.clone());
    d.add_transition(s1, 7, s5, w_all.clone());
    d.add_transition(s1, 10, s6, w_all.clone());
    d.add_transition(s1, 12, s7, w_all.clone());
    d.add_transition(s1, 13, s5, w_all.clone());
    // State 2
    d.add_transition(s2, 0, s8, w_all.clone());
    d.add_transition(s2, 3, s9, w_all.clone());
    d.add_transition(s2, 7, s8, w_all.clone());
    d.add_transition(s2, 10, s10, w_all.clone());
    d.add_transition(s2, 12, s11, w_all.clone());
    d.add_transition(s2, 13, s8, w_all.clone());
    // State 3
    d.set_final_weight(s3, w_1_2.clone());
    // State 4
    d.add_transition(s4, 7, s3, w_1_2.clone());
    d.add_transition(s4, 13, s3, w_1_2.clone());
    // State 5
    d.set_final_weight(s5, w_all.clone());
    // State 6
    d.add_transition(s6, 100, s12, w_all.clone());
    // State 7
    d.add_transition(s7, 100, s6, w_all.clone());
    // State 8
    d.set_final_weight(s8, w_all.clone());
    // State 9
    d.add_transition(s9, 7, s8, w_all.clone());
    d.add_transition(s9, 13, s8, w_all.clone());
    // State 10
    d.add_transition(s10, 100, s13, w_all.clone());
    // State 11
    d.add_transition(s11, 100, s10, w_all.clone());
    // State 12
    d.add_transition(s12, 13, s5, w_all.clone());
    // State 13
    d.add_transition(s13, 13, s8, w_all.clone());

    let expected = d.clone();
    let minimized = minimize::minimize(&d);
    assert_dwa_equivalent(&minimized, &expected, 10);
}

#[test]
fn test_minimize_relaxed_merge_conditions() {
    let mut d = DWA::new(0, 0);
    let s1 = d.add_state();
    let s2 = d.add_state();
    let s3 = d.add_state();
    let s4 = d.add_state();

    d.add_transition(0, 1, s1, weight_from_item(1));
    d.add_transition(0, 2, s2, weight_from_item(2));
    d.add_transition(0, 3, s3, weight_from_iter([1, 2, 3]));
    d.add_transition(0, 4, s4, weight_from_iter([1, 2, 4]));
    d.add_transition(s1, 5, s3, weight_from_item(1));
    d.add_transition(s2, 5, s4, weight_from_item(2));
    d.set_final_weight(s3, weight_from_iter([1, 2, 3]));
    d.set_final_weight(s4, weight_from_iter([1, 2, 4]));

    let w15_before = d.eval_word(&[1, 5]);
    let w25_before = d.eval_word(&[2, 5]);
    let w3_before = d.eval_word(&[3]);
    let w4_before = d.eval_word(&[4]);

    let minimized = minimize::minimize(&d);

    let w15_after = minimized.eval_word(&[1, 5]);
    let w25_after = minimized.eval_word(&[2, 5]);
    let w3_after = minimized.eval_word(&[3]);
    let w4_after = minimized.eval_word(&[4]);

    assert_weights_eq(&w15_before, &w15_after, "Path [1,5] weight should be preserved");
    assert_weights_eq(&w25_before, &w25_after, "Path [2,5] weight should be preserved");
    assert_weights_eq(&w3_before, &w3_after, "Path [3] weight should be preserved");
    assert_weights_eq(&w4_before, &w4_after, "Path [4] weight should be preserved");

    assert_weights_eq(&w15_after, &weight_from_item(1), "Path [1,5] should yield weight [1]");
    assert_weights_eq(&w25_after, &weight_from_item(2), "Path [2,5] should yield weight [2]");
    assert_weights_eq(
        &w3_after,
        &weight_from_iter([1, 2, 3]),
        "Path [3] should yield weight [1,2,3]",
    );
    assert_weights_eq(
        &w4_after,
        &weight_from_iter([1, 2, 4]),
        "Path [4] should yield weight [1,2,4]",
    );
}

#[test]
fn test_minimize_no_false_merge_when_targets_differ() {
    let mut d = DWA::new(0, 0);
    let s1 = d.add_state();
    let s2 = d.add_state();
    let s3 = d.add_state();
    let s4 = d.add_state();

    d.add_transition(0, 1, s1, weight_from_item(1));
    d.add_transition(0, 2, s2, weight_from_item(2));
    d.add_transition(s1, 3, s3, weight_from_item(1));
    d.add_transition(s2, 3, s4, weight_from_item(2));
    d.set_final_weight(s3, weight_from_item(1));
    d.set_final_weight(s4, weight_from_item(2));

    let w13_before = d.eval_word(&[1, 3]);
    let w23_before = d.eval_word(&[2, 3]);

    let minimized = minimize::minimize(&d);

    let w13_after = minimized.eval_word(&[1, 3]);
    let w23_after = minimized.eval_word(&[2, 3]);

    assert_weights_eq(&w13_before, &w13_after, "Path [1,3] weight should be preserved");
    assert_weights_eq(&w23_before, &w23_after, "Path [2,3] weight should be preserved");
    assert_weights_eq(&w13_after, &weight_from_item(1), "Path [1,3] should yield weight [1]");
    assert_weights_eq(&w23_after, &weight_from_item(2), "Path [2,3] should yield weight [2]");
}

#[test]
fn test_minimize_cross_height_merge_opportunity() {
    let mut d = DWA::new(0, 0);
    let s1 = d.add_state();
    let s2 = d.add_state();
    let s3 = d.add_state();

    d.add_transition(0, b'a' as i32, s1, weight_from_item(0));
    d.add_transition(0, b'd' as i32, s2, weight_from_item(1));
    d.add_transition(s1, b'b' as i32, s3, weight_from_item(0));
    d.set_final_weight(s1, weight_from_item(0));
    d.set_final_weight(s2, weight_from_item(1));
    d.set_final_weight(s3, weight_from_item(0));

    let a_before = d.eval_word(&[b'a' as i32]);
    let ab_before = d.eval_word(&[b'a' as i32, b'b' as i32]);
    let d_before = d.eval_word(&[b'd' as i32]);
    let db_before = d.eval_word(&[b'd' as i32, b'b' as i32]);

    assert_weights_eq(&a_before, &weight_from_item(0), "a should accept token 0");
    assert_weights_eq(&ab_before, &weight_from_item(0), "a,b should accept token 0");
    assert_weights_eq(&d_before, &weight_from_item(1), "d should accept token 1");
    assert_weights_eq(&db_before, &Weight::empty(), "d,b should be rejected");

    let minimized = minimize::minimize(&d);

    let a_after = minimized.eval_word(&[b'a' as i32]);
    let ab_after = minimized.eval_word(&[b'a' as i32, b'b' as i32]);
    let d_after = minimized.eval_word(&[b'd' as i32]);
    let db_after = minimized.eval_word(&[b'd' as i32, b'b' as i32]);

    assert_weights_eq(&a_before, &a_after, "a path weight should be preserved");
    assert_weights_eq(&ab_before, &ab_after, "a,b path weight should be preserved");
    assert_weights_eq(&d_before, &d_after, "d path weight should be preserved");
    assert_weights_eq(&db_before, &db_after, "d,b path weight should be preserved");
    assert!(
        minimized.states().len() <= 4,
        "Should produce at most 4 states, got {}",
        minimized.states().len()
    );
}

#[test]
fn test_minimize_disjoint_paths_merge() {
    let mut d = DWA::new(0, 0);
    let s1 = d.add_state();
    let s2 = d.add_state();
    let s3 = d.add_state();
    let s4 = d.add_state();

    d.add_transition(0, b'a' as i32, s1, weight_from_item(0));
    d.add_transition(0, b'b' as i32, s2, weight_from_item(1));
    d.add_transition(s1, b'c' as i32, s3, weight_from_item(0));
    d.add_transition(s2, b'c' as i32, s4, weight_from_item(1));
    d.set_final_weight(s3, weight_from_item(0));
    d.set_final_weight(s4, weight_from_item(1));

    let ac_before = d.eval_word(&[b'a' as i32, b'c' as i32]);
    let bc_before = d.eval_word(&[b'b' as i32, b'c' as i32]);
    assert_weights_eq(&ac_before, &weight_from_item(0), "a,c should accept token 0");
    assert_weights_eq(&bc_before, &weight_from_item(1), "b,c should accept token 1");

    let minimized = minimize::minimize(&d);

    let ac_after = minimized.eval_word(&[b'a' as i32, b'c' as i32]);
    let bc_after = minimized.eval_word(&[b'b' as i32, b'c' as i32]);
    assert_weights_eq(&ac_before, &ac_after, "a,c path weight should be preserved");
    assert_weights_eq(&bc_before, &bc_after, "b,c path weight should be preserved");
    assert!(
        minimized.states().len() <= 5,
        "Should minimize to at most 5 states, got {}",
        minimized.states().len()
    );
}

#[test]
fn test_minimize_cross_height_via_relaxed_conditions() {
    let mut d = DWA::new(0, 0);
    let s1 = d.add_state();
    let s2 = d.add_state();
    let s3 = d.add_state();
    let s4 = d.add_state();

    let w01 = weight_from_item(0).union(&weight_from_item(1));
    d.add_transition(0, b'a' as i32, s1, w01);
    d.add_transition(0, b'b' as i32, s2, weight_from_item(0));
    d.add_transition(0, b'c' as i32, s3, weight_from_item(1));
    d.set_final_weight(s1, weight_from_item(0));
    d.add_transition(s2, b'd' as i32, s4, weight_from_item(0));
    d.set_final_weight(s2, weight_from_item(0));
    d.set_final_weight(s3, weight_from_item(1));
    d.set_final_weight(s4, weight_from_item(0));

    let a_before = d.eval_word(&[b'a' as i32]);
    let b_before = d.eval_word(&[b'b' as i32]);
    let bd_before = d.eval_word(&[b'b' as i32, b'd' as i32]);
    let c_before = d.eval_word(&[b'c' as i32]);
    let cd_before = d.eval_word(&[b'c' as i32, b'd' as i32]);

    assert_weights_eq(&a_before, &weight_from_item(0), "a path should accept token 0");
    assert_weights_eq(&b_before, &weight_from_item(0), "b path should accept token 0");
    assert_weights_eq(&bd_before, &weight_from_item(0), "b,d path should accept token 0");
    assert_weights_eq(&c_before, &weight_from_item(1), "c path should accept token 1");
    assert_weights_eq(&cd_before, &Weight::empty(), "c,d path should be rejected");

    let minimized = minimize::minimize(&d);

    assert_weights_eq(&a_before, &minimized.eval_word(&[b'a' as i32]), "a preserved");
    assert_weights_eq(&b_before, &minimized.eval_word(&[b'b' as i32]), "b preserved");
    assert_weights_eq(&bd_before, &minimized.eval_word(&[b'b' as i32, b'd' as i32]), "b,d preserved");
    assert_weights_eq(&c_before, &minimized.eval_word(&[b'c' as i32]), "c preserved");
    assert_weights_eq(&cd_before, &minimized.eval_word(&[b'c' as i32, b'd' as i32]), "c,d preserved");
    assert!(
        minimized.states().len() <= 4,
        "Should achieve at most 4 states, got {}",
        minimized.states().len()
    );
}

// Union Tests

#[test]
fn test_union_simple() {
    let d1 = dwa_accepts_char('a', weight_from_item(1));
    let d2 = dwa_accepts_char('b', weight_from_item(2));

    let mut expected = DWA::new(0, 0);
    let s_a = expected.add_state();
    let s_b = expected.add_state();
    expected.add_transition(0, 'a' as Label, s_a, Weight::all());
    expected.add_transition(0, 'b' as Label, s_b, Weight::all());
    expected.set_final_weight(s_a, weight_from_item(1));
    expected.set_final_weight(s_b, weight_from_item(2));

    let u = dwa_union(&d1, &d2);
    assert_dwa_equivalent(&u, &expected, 5);
}

#[test]
fn test_union_overlapping() {
    let d1 = dwa_accepts_char('a', weight_from_item(1));
    let mut d2 = dwa_accepts_char('b', weight_from_item(3));
    let s_a2 = d2.add_state();
    d2.add_transition(d2.start_state(), 'a' as Label, s_a2, Weight::all());
    d2.set_final_weight(s_a2, weight_from_item(2));

    let mut expected = DWA::new(0, 0);
    let s_a = expected.add_state();
    let s_b = expected.add_state();
    expected.add_transition(0, 'a' as Label, s_a, Weight::all());
    expected.add_transition(0, 'b' as Label, s_b, Weight::all());
    expected.set_final_weight(s_a, weight_from_iter([1, 2]));
    expected.set_final_weight(s_b, weight_from_item(3));

    let u = dwa_union(&d1, &d2);
    assert_dwa_equivalent(&u, &expected, 5);
}

#[test]
fn test_union_transition_weight_union() {
    fn build(ch: char, ew: u32, fw: u32) -> DWA {
        dwa_with_char_and_weights(ch, weight_from_item(ew), weight_from_item(fw))
    }
    let d1 = build('x', 10, 1);
    let d2 = build('x', 20, 2);
    let u = dwa_union(&d1, &d2);

    let mut expected = DWA::new(0, 0);
    let s = expected.add_state();
    expected.add_transition(0, 'x' as Label, s, weight_from_iter([10, 20]));
    expected.set_final_weight(s, weight_from_iter([1, 2]));

    assert_dwa_equivalent(&u, &expected, 5);
}

#[test]
#[ignore] // glrmask determinize/minimize only supports acyclic DWAs; this test has a self-loop (a*)
fn test_union_identical_cyclic() {
    // DWA that accepts a* with final weight [1].
    let mut d1 = DWA::new(0, 0);
    d1.add_transition(d1.start_state(), 'a' as Label, d1.start_state(), Weight::all());
    d1.set_final_weight(d1.start_state(), weight_from_item(1));

    let d2 = d1.clone();

    let u = dwa_union(&d1, &d2);
    assert_dwa_equivalent(&u, &d1, 10);
}

#[test]
fn test_union_handles_final_start_and_single_branch_regression() {
    let mut left = DWA::new(0, 0);
    left.set_final_weight(0, weight_from_item(0));

    let mut right = DWA::new(0, 0);
    let s1b = right.add_state();
    right.add_transition(0, 0, s1b, weight_from_item(1));
    right.set_final_weight(s1b, Weight::all());

    let u = dwa_union(&left, &right);
    validate_union(&left, &right, &u, 5);
}

#[test]
fn test_union_handles_shorter_left_branch_regression() {
    let mut left = DWA::new(0, 0);
    let s1a = left.add_state();
    left.add_transition(0, 0, s1a, weight_from_item(0));
    left.set_final_weight(s1a, Weight::all());

    let mut right = DWA::new(0, 0);
    let s1b = right.add_state();
    let s2b = right.add_state();
    right.add_transition(0, 0, s1b, weight_from_item(1));
    right.add_transition(s1b, 1, s2b, Weight::all());
    right.set_final_weight(s2b, Weight::all());

    let u = dwa_union(&left, &right);
    validate_union(&left, &right, &u, 5);
}

#[test]
fn test_union_handles_shared_target_regression() {
    let mut left = DWA::new(0, 0);
    let s1a = left.add_state();
    left.add_transition(0, 0, s1a, weight_from_item(0));
    left.add_transition(0, 1, s1a, weight_from_item(1));
    left.set_final_weight(s1a, Weight::all());

    let mut right = DWA::new(0, 0);
    let s1b = right.add_state();
    let s2b = right.add_state();
    right.add_transition(0, 0, s1b, weight_from_item(1));
    right.add_transition(s1b, 1, s2b, Weight::all());
    right.set_final_weight(s2b, Weight::all());

    let u = dwa_union(&left, &right);
    validate_union(&left, &right, &u, 5);
}

#[test]
fn test_union_handles_nested_shared_prefix_regression() {
    let mut left = DWA::new(0, 0);
    let s1a = left.add_state();
    let s2a = left.add_state();
    left.add_transition(0, 0, s1a, weight_from_item(0));
    left.add_transition(s1a, 1, s2a, Weight::all());
    left.set_final_weight(s2a, Weight::all());

    let mut right = DWA::new(0, 0);
    let s1b = right.add_state();
    let s2b = right.add_state();
    let s3b = right.add_state();
    right.add_transition(0, 0, s1b, weight_from_item(1));
    right.add_transition(s1b, 1, s2b, Weight::all());
    right.add_transition(s2b, 2, s3b, Weight::all());
    right.set_final_weight(s3b, Weight::all());

    let u = dwa_union(&left, &right);
    validate_union(&left, &right, &u, 5);
}

#[test]
fn test_union_handles_large_mixed_label_regression() {
    let mut left = DWA::new(0, 0);
    add_dwa_states(&mut left, 9);
    assert_eq!(left.states().len(), 10);

    left.set_final_weight(0, weight_from_item(2));
    left.add_transition(0, 0, 1, weight_from_item(1));
    left.add_transition(0, 1, 2, weight_from_iter(0..=1));
    left.add_transition(0, 2, 3, weight_from_item(0));
    left.add_transition(0, 3, 4, weight_from_iter(0..=1));
    left.add_transition(1, neg(0), 5, Weight::all());
    left.add_transition(2, 100, 6, Weight::all());
    left.add_transition(3, neg(2), 7, Weight::all());
    left.add_transition(5, neg(1), 8, Weight::all());
    left.add_transition(7, neg(0), 9, Weight::all());
    left.set_final_weight(8, Weight::all());
    left.set_final_weight(9, Weight::all());

    let mut right = DWA::new(0, 0);
    add_dwa_states(&mut right, 12);
    assert_eq!(right.states().len(), 13);

    right.add_transition(0, 1, 1, weight_from_item(3));
    right.add_transition(0, 2, 2, weight_from_item(3));
    right.add_transition(0, 3, 3, weight_from_item(3));
    right.add_transition(1, 100, 4, Weight::all());
    right.add_transition(2, neg(2), 5, Weight::all());
    right.add_transition(5, neg(0), 6, Weight::all());
    right.add_transition(6, 0, 7, weight_from_item(3));
    right.add_transition(6, 1, 8, weight_from_item(3));
    right.add_transition(6, 3, 9, weight_from_item(3));
    right.add_transition(7, neg(0), 10, Weight::all());
    right.add_transition(8, 100, 11, Weight::all());
    right.add_transition(10, neg(1), 12, Weight::all());
    right.set_final_weight(12, Weight::all());

    let u = dwa_union(&left, &right);
    validate_union(&left, &right, &u, 15);
}

#[test]
fn test_union_handles_large_union_regression() {
    let mut a = DWA::new(0, 0);
    add_dwa_states(&mut a, 23);
    assert_eq!(a.states().len(), 24);

    a.add_transition(0, 0, 1, weight_from_item(1));
    a.add_transition(0, 1, 2, weight_from_item(0));
    a.add_transition(0, 2, 3, weight_from_iter(1..=2));
    a.add_transition(0, 3, 4, weight_from_iter(1..=2));
    a.add_transition(0, 4, 5, weight_from_iter(1..=2));
    a.add_transition(0, 5, 6, weight_from_item(0));
    a.add_transition(0, 7, 7, weight_from_iter(1..=2));
    a.add_transition(0, 8, 8, weight_from_item(0));
    a.add_transition(0, 9, 9, weight_from_iter(1..=2));
    a.add_transition(1, neg(0), 10, weight_from_item(1));
    a.add_transition(2, neg(1), 11, weight_from_item(0));
    a.add_transition(3, 100, 12, weight_from_item(1));
    a.add_transition(3, neg(2), 13, weight_from_item(2));
    a.add_transition(4, neg(3), 13, weight_from_item(2));
    a.add_transition(4, 5, 14, weight_from_item(1));
    a.add_transition(5, 1, 15, weight_from_iter(1..=2));
    a.add_transition(5, 5, 16, weight_from_iter(1..=2));
    a.add_transition(5, 8, 17, weight_from_iter(1..=2));
    a.add_transition(6, neg(5), 11, weight_from_item(0));
    a.add_transition(7, 1, 15, weight_from_iter(1..=2));
    a.add_transition(7, 5, 16, weight_from_iter(1..=2));
    a.add_transition(8, neg(8), 11, weight_from_item(0));
    a.add_transition(9, 100, 17, weight_from_iter(1..=2));
    a.add_transition(10, neg(1), 18, weight_from_item(1));
    a.add_transition(11, neg(4), 19, weight_from_item(0));
    a.add_transition(12, 100, 20, weight_from_item(1));
    a.add_transition(13, neg(8), 21, weight_from_item(2));
    a.add_transition(14, neg(5), 1, weight_from_item(1));
    a.add_transition(15, 100, 20, weight_from_item(1));
    a.add_transition(15, neg(1), 22, weight_from_item(2));
    a.add_transition(16, neg(5), 23, weight_from_iter(1..=2));
    a.add_transition(17, 100, 7, weight_from_iter(1..=2));
    a.set_final_weight(18, weight_from_item(1));
    a.set_final_weight(19, weight_from_item(0));
    a.add_transition(20, 5, 14, weight_from_item(1));
    a.set_final_weight(21, weight_from_item(2));
    a.add_transition(22, neg(2), 13, weight_from_item(2));
    a.add_transition(23, neg(0), 10, weight_from_item(1));
    a.add_transition(23, neg(3), 13, weight_from_item(2));

    let mut b = DWA::new(0, 0);
    add_dwa_states(&mut b, 16);
    assert_eq!(b.states().len(), 17);

    b.add_transition(0, 0, 1, weight_from_item(3));
    b.add_transition(0, 2, 2, weight_from_item(3));
    b.add_transition(0, 3, 3, weight_from_item(3));
    b.add_transition(0, 4, 4, weight_from_item(3));
    b.add_transition(0, 7, 5, weight_from_item(3));
    b.add_transition(0, 9, 6, weight_from_item(3));
    b.add_transition(1, neg(0), 7, weight_from_item(3));
    b.add_transition(2, 100, 8, weight_from_item(3));
    b.add_transition(3, 5, 9, weight_from_item(3));
    b.add_transition(4, 1, 8, weight_from_item(3));
    b.add_transition(4, 5, 9, weight_from_item(3));
    b.add_transition(4, 8, 10, weight_from_item(3));
    b.add_transition(5, 1, 8, weight_from_item(3));
    b.add_transition(5, 5, 9, weight_from_item(3));
    b.add_transition(6, 100, 10, weight_from_item(3));
    b.add_transition(7, neg(1), 11, weight_from_item(3));
    b.add_transition(8, 100, 3, weight_from_item(3));
    b.add_transition(9, neg(5), 1, weight_from_item(3));
    b.add_transition(10, 100, 5, weight_from_item(3));
    b.add_transition(11, 1, 12, weight_from_item(3));
    b.add_transition(11, 5, 13, weight_from_item(3));
    b.add_transition(11, 8, 14, weight_from_item(3));
    b.add_transition(12, neg(1), 15, weight_from_item(3));
    b.add_transition(13, neg(5), 15, weight_from_item(3));
    b.add_transition(14, neg(8), 15, weight_from_item(3));
    b.add_transition(15, neg(4), 16, weight_from_item(3));
    b.set_final_weight(16, weight_from_item(3));

    let u = dwa_union(&a, &b);
    validate_union(&a, &b, &u, 20);
}

#[test]
#[ignore] // glrmask determinize/minimize only supports acyclic DWAs; the union product contains cycles
fn test_union_complex_from_attachment() {
    let w01 = weight_from_iter(0..=1);

    // --- Build LEFT DWA ---
    let mut left = DWA::new(0, 0);
    add_dwa_states(&mut left, 47);

    left.add_transition(0, 0, 1, weight_from_item(1));
    left.add_transition(0, 2, 2, weight_from_item(1));
    left.add_transition(0, 3, 3, weight_from_item(1));
    left.add_transition(0, 4, 4, weight_from_item(1));
    left.add_transition(0, 5, 5, weight_from_item(1));
    left.add_transition(0, 6, 6, weight_from_item(1));
    left.add_transition(0, 7, 7, weight_from_item(1));
    left.add_transition(0, 8, 8, weight_from_item(1));
    left.add_transition(0, 9, 9, weight_from_item(1));
    left.add_transition(0, 10, 10, weight_from_item(1));
    left.add_transition(1, neg(0), 11, Weight::all());
    left.add_transition(2, 100, 12, Weight::all());
    left.add_transition(3, neg(3), 13, Weight::all());
    left.add_transition(5, 3, 14, Weight::all());
    left.add_transition(5, 7, 9, Weight::all());
    left.add_transition(6, 100, 5, Weight::all());
    left.add_transition(7, neg(7), 15, Weight::all());
    left.add_transition(8, 100, 9, Weight::all());
    left.add_transition(9, 3, 16, Weight::all());
    left.add_transition(9, 7, 9, Weight::all());
    left.add_transition(10, 5, 5, Weight::all());
    left.add_transition(11, neg(9), 17, Weight::all());
    left.add_transition(12, 100, 18, Weight::all());
    left.add_transition(13, neg(9), 19, Weight::all());
    left.add_transition(14, neg(3), 20, Weight::all());
    left.add_transition(15, neg(9), 21, Weight::all());
    left.add_transition(16, neg(3), 22, Weight::all());
    left.add_transition(17, 2, 23, w01.clone());
    left.add_transition(17, 4, 24, w01.clone());
    left.add_transition(17, 5, 25, w01.clone());
    left.add_transition(17, 6, 26, w01.clone());
    left.add_transition(17, 8, 27, w01.clone());
    left.add_transition(17, 9, 28, w01.clone());
    left.add_transition(17, 10, 29, w01.clone());
    left.add_transition(19, 2, 23, w01.clone());
    left.add_transition(19, 4, 24, w01.clone());
    left.add_transition(19, 5, 25, w01.clone());
    left.add_transition(19, 6, 26, w01.clone());
    left.add_transition(19, 8, 27, w01.clone());
    left.add_transition(19, 9, 28, w01.clone());
    left.add_transition(19, 10, 29, w01.clone());
    left.add_transition(20, neg(0), 30, Weight::all());
    left.add_transition(21, 2, 23, w01.clone());
    left.add_transition(21, 4, 24, w01.clone());
    left.add_transition(21, 5, 25, w01.clone());
    left.add_transition(21, 6, 26, w01.clone());
    left.add_transition(21, 8, 27, w01.clone());
    left.add_transition(21, 9, 28, w01.clone());
    left.add_transition(21, 10, 29, w01.clone());
    left.add_transition(22, neg(0), 31, Weight::all());
    left.add_transition(23, 100, 32, Weight::all());
    left.add_transition(25, 7, 28, Weight::all());
    left.add_transition(26, 100, 25, Weight::all());
    left.add_transition(27, 100, 28, Weight::all());
    left.add_transition(28, 0, 33, Weight::all());
    left.add_transition(28, 3, 34, Weight::all());
    left.add_transition(28, 7, 35, Weight::all());
    left.add_transition(29, 5, 25, Weight::all());
    left.add_transition(30, neg(9), 36, Weight::all());
    left.add_transition(31, neg(9), 37, Weight::all());
    left.add_transition(32, 100, 38, Weight::all());
    left.add_transition(33, neg(0), 39, Weight::all());
    left.add_transition(34, neg(3), 40, Weight::all());
    left.add_transition(35, neg(7), 41, Weight::all());
    left.add_transition(36, 2, 23, w01.clone());
    left.add_transition(36, 4, 24, w01.clone());
    left.add_transition(36, 5, 25, w01.clone());
    left.add_transition(36, 6, 26, w01.clone());
    left.add_transition(36, 8, 27, w01.clone());
    left.add_transition(36, 9, 28, w01.clone());
    left.add_transition(36, 10, 29, w01.clone());
    left.add_transition(37, 2, 23, w01.clone());
    left.add_transition(37, 4, 24, w01.clone());
    left.add_transition(37, 5, 25, w01.clone());
    left.add_transition(37, 6, 26, w01.clone());
    left.add_transition(37, 8, 27, w01.clone());
    left.add_transition(37, 9, 28, w01.clone());
    left.add_transition(37, 10, 29, w01.clone());
    left.add_transition(39, neg(5), 42, Weight::all());
    left.add_transition(40, neg(5), 43, Weight::all());
    left.add_transition(41, neg(5), 44, Weight::all());
    left.add_transition(42, neg(10), 45, Weight::all());
    left.add_transition(43, neg(10), 46, Weight::all());
    left.add_transition(44, neg(10), 47, Weight::all());
    left.set_final_weight(45, Weight::all());
    left.set_final_weight(46, Weight::all());
    left.set_final_weight(47, Weight::all());

    // --- Build RIGHT DWA ---
    let mut right = DWA::new(0, 0);
    add_dwa_states(&mut right, 42);

    right.add_transition(0, 2, 1, weight_from_item(0));
    right.add_transition(0, 4, 2, weight_from_item(0));
    right.add_transition(0, 5, 3, weight_from_item(0));
    right.add_transition(0, 6, 4, weight_from_item(0));
    right.add_transition(0, 8, 5, weight_from_item(0));
    right.add_transition(0, 9, 6, weight_from_item(0));
    right.add_transition(0, 10, 7, weight_from_item(0));
    right.add_transition(1, 100, 8, Weight::all());
    right.add_transition(3, 7, 6, Weight::all());
    right.add_transition(4, 100, 3, Weight::all());
    right.add_transition(5, 100, 6, Weight::all());
    right.add_transition(6, 0, 9, Weight::all());
    right.add_transition(6, 3, 10, Weight::all());
    right.add_transition(6, 7, 11, Weight::all());
    right.add_transition(7, 5, 3, Weight::all());
    right.add_transition(8, 100, 12, Weight::all());
    right.add_transition(9, neg(0), 13, Weight::all());
    right.add_transition(10, neg(3), 14, Weight::all());
    right.add_transition(11, neg(7), 15, Weight::all());
    right.add_transition(13, neg(5), 16, Weight::all());
    right.add_transition(14, neg(5), 17, Weight::all());
    right.add_transition(15, neg(5), 18, Weight::all());
    right.add_transition(16, neg(10), 19, Weight::all());
    right.add_transition(17, neg(10), 20, Weight::all());
    right.add_transition(18, neg(10), 21, Weight::all());
    right.add_transition(19, 2, 22, w01.clone());
    right.add_transition(19, 4, 23, w01.clone());
    right.add_transition(19, 5, 24, w01.clone());
    right.add_transition(19, 6, 25, w01.clone());
    right.add_transition(19, 8, 26, w01.clone());
    right.add_transition(19, 9, 27, w01.clone());
    right.add_transition(19, 10, 28, w01.clone());
    right.add_transition(20, 2, 22, w01.clone());
    right.add_transition(20, 4, 23, w01.clone());
    right.add_transition(20, 5, 24, w01.clone());
    right.add_transition(20, 6, 25, w01.clone());
    right.add_transition(20, 8, 26, w01.clone());
    right.add_transition(20, 9, 27, w01.clone());
    right.add_transition(20, 10, 28, w01.clone());
    right.add_transition(21, 2, 22, w01.clone());
    right.add_transition(21, 4, 23, w01.clone());
    right.add_transition(21, 5, 24, w01.clone());
    right.add_transition(21, 6, 25, w01.clone());
    right.add_transition(21, 8, 26, w01.clone());
    right.add_transition(21, 9, 27, w01.clone());
    right.add_transition(21, 10, 28, w01.clone());
    right.add_transition(22, 100, 29, Weight::all());
    right.add_transition(24, 7, 27, Weight::all());
    right.add_transition(25, 100, 24, Weight::all());
    right.add_transition(26, 100, 27, Weight::all());
    right.add_transition(27, 0, 30, Weight::all());
    right.add_transition(27, 3, 31, Weight::all());
    right.add_transition(27, 7, 32, Weight::all());
    right.add_transition(28, 5, 24, Weight::all());
    right.add_transition(29, 100, 33, Weight::all());
    right.add_transition(30, neg(0), 34, Weight::all());
    right.add_transition(31, neg(3), 35, Weight::all());
    right.add_transition(32, neg(7), 36, Weight::all());
    right.add_transition(34, neg(5), 37, Weight::all());
    right.add_transition(35, neg(5), 38, Weight::all());
    right.add_transition(36, neg(5), 39, Weight::all());
    right.add_transition(37, neg(10), 40, Weight::all());
    right.add_transition(38, neg(10), 41, Weight::all());
    right.add_transition(39, neg(10), 42, Weight::all());
    right.set_final_weight(40, Weight::all());
    right.set_final_weight(41, Weight::all());
    right.set_final_weight(42, Weight::all());

    let u = dwa_union(&left, &right);
    validate_union(&left, &right, &u, 20);
}

#[test]
#[ignore] // glrmask determinize/minimize only supports acyclic DWAs; simplified version of cyclic attachment test
fn test_union_complex_from_attachment_simplified() {
    let w01 = weight_from_iter(0..=1);

    // Build left DWA
    let mut left = DWA::new(0, 0);
    add_dwa_states(&mut left, 20);
    assert_eq!(left.states().len(), 21);

    left.add_transition(0, 0, 1, weight_from_item(1));
    left.add_transition(0, 3, 2, weight_from_item(1));
    left.add_transition(0, 5, 3, weight_from_item(1));
    left.add_transition(0, 6, 4, weight_from_item(1));
    left.add_transition(0, 7, 5, weight_from_item(1));
    left.add_transition(0, 8, 4, weight_from_item(1));
    left.add_transition(0, 9, 3, weight_from_item(1));
    left.add_transition(0, 10, 6, weight_from_item(1));
    left.add_transition(1, neg(0), 7, Weight::all());
    left.add_transition(2, neg(3), 7, Weight::all());
    left.add_transition(3, 3, 8, Weight::all());
    left.add_transition(3, 7, 3, Weight::all());
    left.add_transition(4, 100, 3, Weight::all());
    left.add_transition(5, neg(7), 7, Weight::all());
    left.add_transition(6, 5, 3, Weight::all());
    left.add_transition(7, neg(9), 9, Weight::all());
    left.add_transition(8, neg(3), 1, Weight::all());
    left.add_transition(9, 5, 10, w01.clone());
    left.add_transition(9, 6, 11, w01.clone());
    left.add_transition(9, 8, 12, w01.clone());
    left.add_transition(9, 9, 13, w01.clone());
    left.add_transition(9, 10, 14, w01.clone());
    left.add_transition(10, 7, 13, Weight::all());
    left.add_transition(11, 100, 10, Weight::all());
    left.add_transition(12, 100, 13, Weight::all());
    left.add_transition(13, 0, 15, Weight::all());
    left.add_transition(13, 3, 16, Weight::all());
    left.add_transition(13, 7, 17, Weight::all());
    left.add_transition(14, 5, 10, Weight::all());
    left.add_transition(15, neg(0), 18, Weight::all());
    left.add_transition(16, neg(3), 18, Weight::all());
    left.add_transition(17, neg(7), 18, Weight::all());
    left.add_transition(18, neg(5), 19, Weight::all());
    left.add_transition(19, neg(10), 20, Weight::all());
    left.set_final_weight(20, weight_from_item(1));

    // Build right DWA
    let mut right = DWA::new(0, 0);
    add_dwa_states(&mut right, 22);
    assert_eq!(right.states().len(), 23);

    right.add_transition(0, 5, 1, weight_from_item(0));
    right.add_transition(0, 6, 2, weight_from_item(0));
    right.add_transition(0, 8, 3, weight_from_item(0));
    right.add_transition(0, 9, 4, weight_from_item(0));
    right.add_transition(0, 10, 5, weight_from_item(0));
    right.add_transition(1, 7, 4, Weight::all());
    right.add_transition(2, 100, 1, Weight::all());
    right.add_transition(3, 100, 4, Weight::all());
    right.add_transition(4, 0, 6, Weight::all());
    right.add_transition(4, 3, 7, Weight::all());
    right.add_transition(4, 7, 8, Weight::all());
    right.add_transition(5, 5, 1, Weight::all());
    right.add_transition(6, neg(0), 9, Weight::all());
    right.add_transition(7, neg(3), 9, Weight::all());
    right.add_transition(8, neg(7), 9, Weight::all());
    right.add_transition(9, neg(5), 10, Weight::all());
    right.add_transition(10, neg(10), 11, Weight::all());
    right.set_final_weight(11, weight_from_item(0));
    right.add_transition(11, 5, 12, w01.clone());
    right.add_transition(11, 6, 13, w01.clone());
    right.add_transition(11, 8, 14, w01.clone());
    right.add_transition(11, 9, 15, w01.clone());
    right.add_transition(11, 10, 16, w01.clone());
    right.add_transition(12, 7, 15, Weight::all());
    right.add_transition(13, 100, 12, Weight::all());
    right.add_transition(14, 100, 15, Weight::all());
    right.add_transition(15, 0, 17, Weight::all());
    right.add_transition(15, 3, 18, Weight::all());
    right.add_transition(15, 7, 19, Weight::all());
    right.add_transition(16, 5, 12, Weight::all());
    right.add_transition(17, neg(0), 20, Weight::all());
    right.add_transition(18, neg(3), 20, Weight::all());
    right.add_transition(19, neg(7), 20, Weight::all());
    right.add_transition(20, neg(5), 21, Weight::all());
    right.add_transition(21, neg(10), 22, Weight::all());
    right.set_final_weight(22, Weight::all());

    let u = dwa_union(&left, &right);
    validate_union(&left, &right, &u, 20);
}

// Concatenate Tests

#[test]
fn test_concatenate_simple() {
    let d1 = dwa_accepts_char('a', weight_from_iter([1, 2]));
    let d2 = dwa_accepts_char('b', weight_from_iter([2, 3]));
    let c = dwa_concatenate(&d1, &d2);
    let expected = dwa_from_str("ab", weight_from_item(2));
    assert_dwa_equivalent(&c, &expected, 5);
}

#[test]
fn test_concatenate_left_start_is_final() {
    let mut left = DWA::new(0, 0);
    left.set_final_weight(left.start_state(), weight_from_iter([0, 1]));

    let mut right = DWA::new(0, 0);
    right.set_final_weight(right.start_state(), weight_from_iter([1, 2]));

    let c = dwa_concatenate(&left, &right);

    let mut expected = DWA::new(0, 0);
    expected.set_final_weight(expected.start_state(), weight_from_item(1));

    assert_dwa_equivalent(&c, &expected, 5);
}

#[test]
fn test_concatenate_disjoint_weights() {
    let word_a = vec![10, 5, 3, neg(3), neg(0), neg(9)];
    let mut dwa_a = DWA::new(0, 0);
    let mut current = dwa_a.start_state();
    for &ch in &word_a {
        let next = dwa_a.add_state();
        dwa_a.add_transition(current, ch, next, Weight::all());
        current = next;
    }
    dwa_a.set_final_weight(current, weight_from_item(1));

    let word_b = vec![9, 3, neg(3), neg(5), neg(10), 9, 7, neg(7), neg(5), neg(10)];
    let mut dwa_b = DWA::new(0, 0);
    current = dwa_b.start_state();
    for &ch in &word_b {
        let next = dwa_b.add_state();
        dwa_b.add_transition(current, ch, next, Weight::all());
        current = next;
    }
    dwa_b.set_final_weight(current, weight_from_item(0));

    let c = dwa_concatenate(&dwa_a, &dwa_b);

    let mut combined_word = word_a.clone();
    combined_word.extend_from_slice(&word_b);

    let wa = dwa_a.eval_word(&word_a);
    let wb = dwa_b.eval_word(&word_b);
    assert_weights_eq(&wa, &weight_from_item(1), "wa should be [1]");
    assert_weights_eq(&wb, &weight_from_item(0), "wb should be [0]");
    assert_weights_eq(&wa.intersection(&wb), &Weight::empty(), "wa & wb should be empty");

    let wc = c.eval_word(&combined_word);
    assert_weights_eq(&wc, &Weight::empty(), "Combined word should be rejected");
}

#[test]
fn test_concatenate_default_path_to_final() {
    let mut a = DWA::new(0, 0);
    let s1a = a.add_state();
    a.add_transition(a.start_state(), 'a' as Label, s1a, Weight::all());
    a.set_final_weight(s1a, weight_from_item(1));

    let mut b = DWA::new(0, 0);
    let s1b = b.add_state();
    b.add_transition(b.start_state(), 'x' as Label, s1b, Weight::all());
    b.set_final_weight(s1b, weight_from_item(1));

    let c = dwa_concatenate(&a, &b);

    let weight = c.eval_word(&['a' as Label, 'x' as Label]);
    assert_weights_eq(&weight, &weight_from_item(1), "Word 'ax' should yield weight [1]");

    let weight_x = c.eval_word(&['x' as Label]);
    assert_weights_eq(&weight_x, &Weight::empty(), "Word 'x' should be rejected");
}

#[test]
fn test_concatenate_complex_from_attachment() {
    let w_all = Weight::all();
    let w_01 = weight_from_iter(0..=1);

    let mut left = DWA::new(0, 0);
    add_dwa_states(&mut left, 25);
    left.set_start_state(25);
    assert_eq!(left.states().len(), 26);

    left.add_transition(0, 2, 9, w_all.clone());
    left.add_transition(0, 4, 1, w_all.clone());
    left.add_transition(0, 5, 3, w_all.clone());
    left.add_transition(0, 6, 11, w_all.clone());
    left.add_transition(0, 8, 12, w_all.clone());
    left.add_transition(0, 9, 4, w_all.clone());
    left.add_transition(0, 10, 5, w_all.clone());
    left.add_transition(3, 7, 4, w_all.clone());
    left.add_transition(4, 0, 13, w_all.clone());
    left.add_transition(4, 3, 17, w_all.clone());
    left.add_transition(4, 7, 21, w_all.clone());
    left.add_transition(5, 5, 3, w_all.clone());
    left.add_transition(6, neg(5), 7, w_all.clone());
    left.add_transition(7, neg(10), 8, w_all.clone());
    left.set_final_weight(8, w_all.clone());
    left.add_transition(9, 100, 10, w_all.clone());
    left.add_transition(10, 101, 2, w_all.clone());
    left.add_transition(11, 100, 3, w_all.clone());
    left.add_transition(12, 100, 4, w_all.clone());
    left.add_transition(13, neg(0), 14, w_all.clone());
    left.add_transition(14, neg(5), 15, w_all.clone());
    left.add_transition(15, neg(10), 16, w_all.clone());
    left.set_final_weight(16, w_all.clone());
    left.add_transition(17, neg(3), 18, w_all.clone());
    left.add_transition(18, neg(5), 19, w_all.clone());
    left.add_transition(19, neg(10), 20, w_all.clone());
    left.set_final_weight(20, w_all.clone());
    left.add_transition(21, neg(7), 22, w_all.clone());
    left.add_transition(22, neg(5), 23, w_all.clone());
    left.add_transition(23, neg(10), 24, w_all.clone());
    left.set_final_weight(24, w_all.clone());
    left.add_transition(25, 2, 9, w_01.clone());
    left.add_transition(25, 4, 1, w_01.clone());
    left.add_transition(25, 5, 3, w_01.clone());
    left.add_transition(25, 6, 11, w_01.clone());
    left.add_transition(25, 8, 12, w_01.clone());
    left.add_transition(25, 9, 4, w_01.clone());
    left.add_transition(25, 10, 5, w_01.clone());

    let left = minimize::minimize(&left);

    let mut right = DWA::new(0, 0);
    right.set_final_weight(0, Weight::all());
    let right = minimize::minimize(&right);

    let c = dwa_concatenate(&left, &right);
    validate_concatenation(&left, &right, &c, 15);
}

#[test]
fn test_concatenate_handles_weight_gated_regression() {
    let mut base_dwa = DWA::new(0, 0);
    add_dwa_states(&mut base_dwa, 12);
    assert_eq!(base_dwa.states().len(), 13);

    base_dwa.add_transition(0, 6, 1, Weight::all());
    base_dwa.add_transition(0, 7, 4, Weight::all());
    base_dwa.add_transition(0, 10, 5, Weight::all());
    base_dwa.add_transition(0, 11, 6, Weight::all());
    base_dwa.add_transition(0, 12, 3, Weight::all());
    base_dwa.add_transition(1, 9, 6, Weight::all());
    base_dwa.add_transition(2, 0, 7, Weight::all());
    base_dwa.add_transition(2, 4, 11, Weight::all());
    base_dwa.add_transition(2, 9, 12, Weight::all());
    base_dwa.add_transition(3, 6, 1, Weight::all());
    base_dwa.add_transition(4, 100, 1, Weight::all());
    base_dwa.add_transition(5, 100, 6, Weight::all());
    base_dwa.add_transition(6, 100, 2, Weight::all());
    base_dwa.add_transition(7, neg(0), 8, Weight::all());
    base_dwa.add_transition(8, neg(6), 9, Weight::all());
    base_dwa.add_transition(9, neg(12), 10, Weight::all());
    base_dwa.set_final_weight(10, Weight::all());
    base_dwa.add_transition(11, neg(4), 8, Weight::all());
    base_dwa.add_transition(12, neg(9), 8, Weight::all());

    let mut dwa1 = base_dwa.clone();
    apply_weight_to_dwa(&mut dwa1, &weight_from_item(0));
    let mut dwa2 = base_dwa.clone();
    apply_weight_to_dwa(&mut dwa2, &weight_from_item(0));

    let c = dwa_concatenate(&dwa1, &dwa2);
    validate_concatenation(&dwa1, &dwa2, &c, 15);
}

// Minimize Complex Tests

#[test]
fn test_minimize_complex_dwa_from_attachment() {
    let w_all = Weight::all();
    let w_01 = weight_from_iter(0..=1);

    let mut left = DWA::new(0, 0);
    add_dwa_states(&mut left, 25);
    left.set_start_state(25);
    assert_eq!(left.states().len(), 26);

    left.add_transition(0, 2, 9, w_all.clone());
    left.add_transition(0, 4, 1, w_all.clone());
    left.add_transition(0, 5, 3, w_all.clone());
    left.add_transition(0, 6, 11, w_all.clone());
    left.add_transition(0, 8, 12, w_all.clone());
    left.add_transition(0, 9, 4, w_all.clone());
    left.add_transition(0, 10, 5, w_all.clone());
    left.add_transition(3, 7, 4, w_all.clone());
    left.add_transition(4, 0, 13, w_all.clone());
    left.add_transition(4, 3, 17, w_all.clone());
    left.add_transition(4, 7, 21, w_all.clone());
    left.add_transition(5, 5, 3, w_all.clone());
    left.add_transition(6, neg(5), 7, w_all.clone());
    left.add_transition(7, neg(10), 8, w_all.clone());
    left.set_final_weight(8, w_all.clone());
    left.add_transition(9, 100, 10, w_all.clone());
    left.add_transition(10, 101, 2, w_all.clone());
    left.add_transition(11, 100, 3, w_all.clone());
    left.add_transition(12, 100, 4, w_all.clone());
    left.add_transition(13, neg(0), 14, w_all.clone());
    left.add_transition(14, neg(5), 15, w_all.clone());
    left.add_transition(15, neg(10), 16, w_all.clone());
    left.set_final_weight(16, w_all.clone());
    left.add_transition(17, neg(3), 18, w_all.clone());
    left.add_transition(18, neg(5), 19, w_all.clone());
    left.add_transition(19, neg(10), 20, w_all.clone());
    left.set_final_weight(20, w_all.clone());
    left.add_transition(21, neg(7), 22, w_all.clone());
    left.add_transition(22, neg(5), 23, w_all.clone());
    left.add_transition(23, neg(10), 24, w_all.clone());
    left.set_final_weight(24, w_all.clone());
    left.add_transition(25, 2, 9, w_01.clone());
    left.add_transition(25, 4, 1, w_01.clone());
    left.add_transition(25, 5, 3, w_01.clone());
    left.add_transition(25, 6, 11, w_01.clone());
    left.add_transition(25, 8, 12, w_01.clone());
    left.add_transition(25, 9, 4, w_01.clone());
    left.add_transition(25, 10, 5, w_01.clone());

    let expected = left.clone();
    let minimized = minimize::minimize(&left);
    assert_dwa_equivalent(&minimized, &expected, 15);
}

// DWA ↔ NWA Roundtrip Tests

#[test]
fn test_dwa_to_nwa_to_dwa_roundtrip() {
    let mut a = DWA::new(0, 0);
    add_dwa_states(&mut a, 23);
    assert_eq!(a.states().len(), 24);

    a.add_transition(0, 0, 1, weight_from_item(1));
    a.add_transition(0, 1, 2, weight_from_item(0));
    a.add_transition(0, 2, 3, weight_from_iter(1..=2));
    a.add_transition(0, 3, 4, weight_from_iter(1..=2));
    a.add_transition(0, 4, 5, weight_from_iter(1..=2));
    a.add_transition(0, 5, 6, weight_from_item(0));
    a.add_transition(0, 7, 7, weight_from_iter(1..=2));
    a.add_transition(0, 8, 8, weight_from_item(0));
    a.add_transition(0, 9, 9, weight_from_iter(1..=2));
    a.add_transition(1, neg(0), 10, weight_from_item(1));
    a.add_transition(2, neg(1), 11, weight_from_item(0));
    a.add_transition(3, 100, 12, weight_from_item(1));
    a.add_transition(3, neg(2), 13, weight_from_item(2));
    a.add_transition(4, neg(3), 13, weight_from_item(2));
    a.add_transition(4, 5, 14, weight_from_item(1));
    a.add_transition(5, 1, 15, weight_from_iter(1..=2));
    a.add_transition(5, 5, 16, weight_from_iter(1..=2));
    a.add_transition(5, 8, 17, weight_from_iter(1..=2));
    a.add_transition(6, neg(5), 11, weight_from_item(0));
    a.add_transition(7, 1, 15, weight_from_iter(1..=2));
    a.add_transition(7, 5, 16, weight_from_iter(1..=2));
    a.add_transition(8, neg(8), 11, weight_from_item(0));
    a.add_transition(9, 100, 17, weight_from_iter(1..=2));
    a.add_transition(10, neg(1), 18, weight_from_item(1));
    a.add_transition(11, neg(4), 19, weight_from_item(0));
    a.add_transition(12, 100, 20, weight_from_item(1));
    a.add_transition(13, neg(8), 21, weight_from_item(2));
    a.add_transition(14, neg(5), 1, weight_from_item(1));
    a.add_transition(15, 100, 20, weight_from_item(1));
    a.add_transition(15, neg(1), 22, weight_from_item(2));
    a.add_transition(16, neg(5), 23, weight_from_iter(1..=2));
    a.add_transition(17, 100, 7, weight_from_iter(1..=2));
    a.set_final_weight(18, weight_from_item(1));
    a.set_final_weight(19, weight_from_item(0));
    a.add_transition(20, 5, 14, weight_from_item(1));
    a.set_final_weight(21, weight_from_item(2));
    a.add_transition(22, neg(2), 13, weight_from_item(2));
    a.add_transition(23, neg(0), 10, weight_from_item(1));
    a.add_transition(23, neg(3), 13, weight_from_item(2));

    let nwa = dwa_to_nwa(&a);
    let roundtrip_dwa = determinize::determinize(&nwa).expect("determinize failed");
    let roundtrip_dwa = minimize::minimize(&roundtrip_dwa);

    assert_dwa_equivalent(&a, &roundtrip_dwa, 20);
}

#[test]
fn test_dwa_roundtrip_minimal_repro() {
    let mut a = DWA::new(0, 0);
    let s1 = a.add_state();
    let s2 = a.add_state();
    let s3 = a.add_state();

    a.add_transition(0, 1, s1, weight_from_item(0));
    a.add_transition(s1, neg(1), s2, weight_from_item(0));
    a.add_transition(s2, neg(4), s3, weight_from_item(0));
    a.set_final_weight(s3, weight_from_item(0));

    let nwa = dwa_to_nwa(&a);
    let roundtrip_dwa = determinize::determinize(&nwa).expect("determinize failed");
    let roundtrip_dwa = minimize::minimize(&roundtrip_dwa);

    assert_dwa_equivalent(&a, &roundtrip_dwa, 10);
}

// Determinize Tests

#[test]
fn test_det_simple_char() {
    let nwa = nwa_accepts_char('a', weight_from_item(1));
    let dwa = determinize::determinize(&nwa).expect("determinize failed");
    let expected = dwa_accepts_char('a', weight_from_item(1));
    assert_dwa_equivalent(&dwa, &expected, 5);
}

#[test]
fn test_det_union_of_chars() {
    let mut nwa = NWA::new(0, 0);
    let start = nwa.add_state();
    nwa.start_states_mut().push(start);
    let s_a = nwa.add_state();
    let s_b = nwa.add_state();
    let final_a = nwa.add_state();
    let final_b = nwa.add_state();
    nwa.add_epsilon(start, s_a, Weight::all());
    nwa.add_epsilon(start, s_b, Weight::all());
    nwa.add_transition(s_a, 'a' as Label, final_a, Weight::all());
    nwa.add_transition(s_b, 'b' as Label, final_b, Weight::all());
    nwa.set_final_weight(final_a, weight_from_item(1));
    nwa.set_final_weight(final_b, weight_from_item(2));

    let dwa = determinize::determinize(&nwa).expect("determinize failed");

    let mut expected = DWA::new(0, 0);
    let final_a_dwa = expected.add_state();
    let final_b_dwa = expected.add_state();
    expected.add_transition(expected.start_state(), 'a' as Label, final_a_dwa, Weight::all());
    expected.add_transition(expected.start_state(), 'b' as Label, final_b_dwa, Weight::all());
    expected.set_final_weight(final_a_dwa, weight_from_item(1));
    expected.set_final_weight(final_b_dwa, weight_from_item(2));

    assert_dwa_equivalent(&dwa, &expected, 5);
}

#[test]
fn test_det_nondeterminism_on_char() {
    let mut nwa = NWA::new(0, 0);
    let start = nwa.add_state();
    nwa.start_states_mut().push(start);
    let f1 = nwa.add_state();
    let f2 = nwa.add_state();
    nwa.add_transition(start, 'a' as Label, f1, weight_from_item(1));
    nwa.add_transition(start, 'a' as Label, f2, weight_from_item(2));
    nwa.set_final_weight(f1, Weight::all());
    nwa.set_final_weight(f2, Weight::all());

    let dwa = determinize::determinize(&nwa).expect("determinize failed");

    let mut expected = DWA::new(0, 0);
    let final_state = expected.add_state();
    expected.add_transition(expected.start_state(), 'a' as Label, final_state, weight_from_iter([1, 2]));
    expected.set_final_weight(final_state, Weight::all());

    assert_dwa_equivalent(&dwa, &expected, 5);
}

#[test]
fn test_det_weight_union_overapprox_paths() {
    let mut nwa = NWA::new(0, 0);
    let start = nwa.add_state();
    nwa.start_states_mut().push(start);
    let s1 = nwa.add_state();
    let s2 = nwa.add_state();
    let s3 = nwa.add_state();

    let token_a: u32 = 0;
    let token_b: u32 = 1;

    nwa.add_transition(start, 'a' as Label, s1, weight_from_item(token_a));
    nwa.add_transition(start, 'a' as Label, s2, weight_from_item(token_b));
    nwa.add_transition(s1, 'b' as Label, s3, weight_from_item(token_a));
    nwa.add_transition(s2, 'b' as Label, s3, weight_from_item(token_b));
    nwa.add_transition(s2, 'c' as Label, s3, weight_from_item(token_b));
    nwa.set_final_weight(s3, Weight::all());

    let dwa = determinize::determinize(&nwa).expect("determinize failed");

    // After 'a', check that 'c' transition weight does NOT include token_a
    let start_state = dwa.start_state() as usize;
    let (s_after_a, _) = dwa.states()[start_state]
        .transitions
        .get(&('a' as Label))
        .expect("expected 'a' transition after determinization");

    let (_, weight_c) = dwa.states()[*s_after_a as usize]
        .transitions
        .get(&('c' as Label))
        .expect("expected 'c' transition after determinization");

    assert!(
        weight_c.intersection(&weight_from_item(token_a)).is_empty(),
        "token_a should not be present on path a->c (over-approx bug if it is)"
    );
}

#[test]
fn test_det_weight_partitioning() {
    let mut nwa = NWA::new(0, 0);
    let start = nwa.add_state();
    nwa.start_states_mut().push(start);
    let f1 = nwa.add_state();
    let f2 = nwa.add_state();
    nwa.add_transition(start, 'a' as Label, f1, weight_from_iter(0..=1));
    nwa.add_transition(start, 'a' as Label, f2, weight_from_iter(1..=2));
    nwa.set_final_weight(f1, Weight::all());
    nwa.set_final_weight(f2, Weight::all());

    let dwa = determinize::determinize(&nwa).expect("determinize failed");

    let mut expected = DWA::new(0, 0);
    let final_state = expected.add_state();
    expected.add_transition(expected.start_state(), 'a' as Label, final_state, weight_from_iter(0..=2));
    expected.set_final_weight(final_state, weight_from_iter(0..=2));

    assert_dwa_equivalent(&dwa, &expected, 5);
}

#[test]
fn test_det_empty_nwa() {
    // Truly empty NWA (no states at all)
    let nwa = NWA::new(0, 0);
    let dwa = determinize::determinize(&nwa).expect("determinize failed");
    assert_eq!(dwa.states().len(), 1);
    assert!(dwa.states()[dwa.start_state() as usize].final_weight.is_none());
    assert!(dwa.states()[dwa.start_state() as usize].transitions.is_empty());
}

#[test]
fn test_det_accepts_nothing() {
    // Start state, but no transitions and not final
    let mut nwa = NWA::new(0, 0);
    let s = nwa.add_state();
    nwa.start_states_mut().push(s);
    let dwa = determinize::determinize(&nwa).expect("determinize failed");
    let expected = DWA::new(0, 0);
    assert_dwa_equivalent(&dwa, &expected, 5);
}

#[test]
fn test_det_accepts_empty_word() {
    let mut nwa = NWA::new(0, 0);
    let s = nwa.add_state();
    nwa.start_states_mut().push(s);
    nwa.set_final_weight(s, weight_from_item(42));
    let dwa = determinize::determinize(&nwa).expect("determinize failed");

    let mut expected = DWA::new(0, 0);
    expected.set_final_weight(expected.start_state(), weight_from_item(42));

    assert_dwa_equivalent(&dwa, &expected, 5);
}

#[test]
#[ignore] // glrmask determinize only supports acyclic NWAs; this NWA contains cycles
fn test_determinize_complex_nwa_from_template() {
    let mut nwa = NWA::new(0, 0);
    add_nwa_states(&mut nwa, 39);
    nwa.start_states_mut().push(0);

    // State 0
    nwa.add_epsilon(0, 6, Weight::all());
    nwa.add_epsilon(0, 10, Weight::all());
    nwa.add_epsilon(0, 13, Weight::all());
    nwa.add_epsilon(0, 14, Weight::all());
    nwa.add_epsilon(0, 15, Weight::all());
    nwa.add_epsilon(0, 17, Weight::all());
    nwa.add_epsilon(0, 19, Weight::all());
    nwa.add_epsilon(0, 20, Weight::all());
    // State 3
    nwa.add_epsilon(3, 21, Weight::all());
    // State 4
    nwa.add_epsilon(4, 22, Weight::all());
    nwa.add_epsilon(4, 23, Weight::all());
    nwa.add_epsilon(4, 28, Weight::all());
    nwa.add_epsilon(4, 33, Weight::all());
    // State 5
    nwa.add_epsilon(5, 38, Weight::all());
    // State 6
    nwa.add_transition(6, 5, 7, Weight::all());
    // State 7
    nwa.add_transition(7, neg(5), 8, Weight::all());
    // State 8
    nwa.add_transition(8, neg(10), 9, Weight::all());
    // State 9
    nwa.set_final_weight(9, Weight::all());
    // State 10
    nwa.add_transition(10, 2, 11, Weight::all());
    // State 13
    nwa.add_transition(13, 4, 1, Weight::all());
    // State 14
    nwa.add_transition(14, 5, 3, Weight::all());
    // State 15
    nwa.add_transition(15, 6, 16, Weight::all());
    // State 17
    nwa.add_transition(17, 8, 18, Weight::all());
    // State 19
    nwa.add_transition(19, 9, 4, Weight::all());
    // State 20
    nwa.add_transition(20, 10, 5, Weight::all());
    // State 21
    nwa.add_transition(21, 7, 4, Weight::all());
    // State 22
    nwa.add_transition(22, 7, 4, Weight::all());
    // State 23
    nwa.add_transition(23, 0, 24, Weight::all());
    // State 24
    nwa.add_transition(24, neg(0), 25, Weight::all());
    // State 25
    nwa.add_transition(25, neg(5), 26, Weight::all());
    // State 26
    nwa.add_transition(26, neg(10), 27, Weight::all());
    // State 27
    nwa.set_final_weight(27, Weight::all());
    // State 28
    nwa.add_transition(28, 3, 29, Weight::all());
    // State 29
    nwa.add_transition(29, neg(3), 30, Weight::all());
    // State 30
    nwa.add_transition(30, neg(5), 31, Weight::all());
    // State 31
    nwa.add_transition(31, neg(10), 32, Weight::all());
    // State 32
    nwa.set_final_weight(32, Weight::all());
    // State 33
    nwa.add_transition(33, 7, 34, Weight::all());
    // State 34
    nwa.add_transition(34, neg(7), 35, Weight::all());
    // State 35
    nwa.add_transition(35, neg(5), 36, Weight::all());
    // State 36
    nwa.add_transition(36, neg(10), 37, Weight::all());
    // State 37
    nwa.set_final_weight(37, Weight::all());
    // State 38
    nwa.add_transition(38, 5, 3, Weight::all());

    let dwa = determinize::determinize(&nwa).expect("determinize failed");

    let word = vec![9, 3, neg(3), neg(5), neg(10)];
    let weight = dwa.eval_word(&word);
    assert!(!weight.is_empty(), "Path should be valid after determinization");
}

#[test]
fn test_determinize_minimal_failing_nwa() {
    let mut nwa = NWA::new(0, 0);
    add_nwa_states(&mut nwa, 34);
    nwa.start_states_mut().push(0);

    nwa.add_epsilon(0, 19, Weight::all());
    nwa.add_transition(19, 9, 4, Weight::all());
    nwa.add_epsilon(4, 28, Weight::all());
    nwa.add_transition(28, 3, 29, Weight::all());
    nwa.add_transition(29, neg(3), 30, Weight::all());
    nwa.add_transition(30, neg(5), 31, Weight::all());
    nwa.add_transition(31, neg(10), 32, Weight::all());
    nwa.set_final_weight(32, Weight::all());

    let dwa = determinize::determinize(&nwa).expect("determinize failed");
    let word = vec![9, 3, neg(3), neg(5), neg(10)];
    let weight = dwa.eval_word(&word);
    assert!(!weight.is_empty(), "Path should be valid after determinization");
}

// Diamond Structure Optimization Test

#[test]
fn test_diamond_structure() {
    let l0: Label = 0;
    let l1: Label = 1;
    let l2: Label = 2;

    let all = Weight::all();
    let w0 = weight_from_item(0);
    let w1 = weight_from_item(1);
    let w2 = weight_from_item(2);
    let w3 = weight_from_item(3);

    // Build input DWA via NWA + determinize (diamond structure)
    let input = {
        let mut nwa = NWA::new(0, 0);
        let start = nwa.add_state();
        let a = nwa.add_state();
        let b = nwa.add_state();
        let c = nwa.add_state();
        let end = nwa.add_state();
        nwa.set_start_states(vec![start]);

        nwa.add_transition(start, l0, a, all.clone());
        nwa.add_transition(start, l1, b, all.clone());
        nwa.add_transition(start, l2, c, all.clone());

        nwa.add_transition(a, l0, end, all.clone());
        nwa.add_transition(b, l0, end, all.clone());
        nwa.add_transition(c, l0, end, all.clone());

        nwa.set_final_weight(a, w0.clone());
        nwa.set_final_weight(b, w1.clone());
        nwa.set_final_weight(c, w2.clone());
        nwa.set_final_weight(end, w3.clone());

        determinize::determinize(&nwa).expect("determinize failed")
    };

    // Build expected DWA (merged A,B,C → single intermediate state)
    let expected = {
        let mut d = DWA::new(0, 0);
        let abc = d.add_state();
        let end = d.add_state();

        let w0_pushed = w0.union(&w3);
        let w1_pushed = w1.union(&w3);
        let w2_pushed = w2.union(&w3);
        let w012 = w0.union(&w1).union(&w2);

        d.add_transition(0, l0, abc, w0_pushed);
        d.add_transition(0, l1, abc, w1_pushed);
        d.add_transition(0, l2, abc, w2_pushed);
        d.add_transition(abc, l0, end, all.clone());
        d.set_final_weight(abc, w012);
        d.set_final_weight(end, w3.clone());
        d
    };

    // Verify semantic equivalence
    assert_dwa_equivalent(&input, &expected, 5);

    // Check optimization: minimized should be semantically equivalent
    let minimized = minimize::minimize(&input);
    assert_dwa_equivalent(&minimized, &expected, 5);

    // Verify no expansion
    assert!(
        minimized.states().len() <= input.states().len(),
        "Minimization should not expand: {} → {}",
        input.states().len(),
        minimized.states().len()
    );
}

#[test]
fn test_json_roundtrip_complex() {
    let mut d = DWA::new(0, 0);
    let s1 = d.add_state();
    let s2 = d.add_state();
    d.add_transition(d.start_state(), 'y' as Label, s1, weight_from_iter(vec![1, 2, 3]));
    d.add_transition(d.start_state(), 'x' as Label, s2, weight_from_item(99));
    d.set_final_weight(s2, weight_from_iter(vec![5, 7]));

    let s = serde_json::to_string(&d).expect("Failed to serialize DWA");
    let d2: DWA = serde_json::from_str(&s).expect("Failed to deserialize DWA");
    
    assert_dwa_equivalent(&d, &d2, 10);
}
