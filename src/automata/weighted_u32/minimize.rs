//! Dwa minimization.
//!
//! Signature-based partition minimization for acyclic [`Dwa`]s.
//!
//! States are processed bottom-up (by height).  Two states are merged iff
//! they agree on:
//! - final weight (or both non-accepting),
//! - the set of outgoing labels,
//! - for each label: the equivalence class of the target state *and* the
//!   transition weight.
//!
//! Because the DWA is acyclic, a single bottom-up pass suffices.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use rustc_hash::FxHashMap;

use super::dwa::{Dwa, DwaState};
use super::nwa::Label;
use crate::ds::rangeset2d::Weight;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Minimize a [`Dwa`] by merging states with identical behaviour.
pub fn minimize(dwa: &Dwa) -> Dwa {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// Heights
// ---------------------------------------------------------------------------

fn compute_heights(dwa: &Dwa) -> Vec<u32> {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// State signatures
// ---------------------------------------------------------------------------

/// A hashable signature that identifies a state's behaviour.
///
/// Two states with the same `Sig` can be merged.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Sig {
    final_weight: Option<Weight>,
    /// Sorted (label, target_class, weight) tuples.
    transitions: Vec<(Label, u32, Weight)>,
}

fn state_signature(st: &DwaState, class: &[u32]) -> Sig {
    unimplemented!()
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;

    #[test]
    fn test_minimize_identity() {
        // Already-minimal 2-state DWA.
        let nt = 1u32;
        let max_tok = 5u32;
        let mut dwa = CompDwa::new(nt, max_tok);
        let s1 = dwa.add_state();
        let w_all = Weight::all();
        dwa.add_transition(0, 0, s1, w_all.clone());
        dwa.set_final_weight(s1, w_all);

        let min = minimize(&dwa);
        assert_eq!(min.num_states(), 2);
    }

    #[test]
    fn test_minimize_merges_identical() {
        // s0 --0--> s1, s0 --1--> s2. s1 and s2 have identical behaviour.
        let nt = 1u32;
        let max_tok = 5u32;
        let mut dwa = CompDwa::new(nt, max_tok);
        let s1 = dwa.add_state();
        let s2 = dwa.add_state();
        let w = Weight::all();
        dwa.add_transition(0, 0, s1, w.clone());
        dwa.add_transition(0, 1, s2, w.clone());
        dwa.set_final_weight(s1, w.clone());
        dwa.set_final_weight(s2, w);

        assert_eq!(dwa.num_states(), 3);
        let min = minimize(&dwa);
        // s1 and s2 should merge.
        assert_eq!(min.num_states(), 2);

        // Both words should still accept.
        assert!(!min.eval_word(&[0]).is_empty());
        assert!(!min.eval_word(&[1]).is_empty());
    }

    #[test]
    fn test_minimize_no_merge_different_weight() {
        // s0 --0--> s1, s0 --1--> s2.  s1 and s2 have different final weights.
        let nt = 1u32;
        let max_tok = 5u32;
        let mut dwa = CompDwa::new(nt, max_tok);
        let s1 = dwa.add_state();
        let s2 = dwa.add_state();
        let w = Weight::all();
        let w1 = Weight::empty();
        let w2 = Weight::all();
        dwa.add_transition(0, 0, s1, w.clone());
        dwa.add_transition(0, 1, s2, w);
        dwa.set_final_weight(s1, w1);
        dwa.set_final_weight(s2, w2);

        let min = minimize(&dwa);
        // Cannot merge: different final weights.
        assert_eq!(min.num_states(), 3);
    }

    #[test]
    fn test_minimize_single_state() {
        let dwa = CompDwa::new(1, 5);
        let min = minimize(&dwa);
        assert_eq!(min.num_states(), 1);
    }

    #[test]
    fn test_minimize_chain() {
        // s0 --0--> s1 --1--> s2 (accepting).  All unique, no merges.
        let nt = 1u32;
        let max_tok = 5u32;
        let mut dwa = CompDwa::new(nt, max_tok);
        let s1 = dwa.add_state();
        let s2 = dwa.add_state();
        let w = Weight::all();
        dwa.add_transition(0, 0, s1, w.clone());
        dwa.add_transition(s1, 1, s2, w.clone());
        dwa.set_final_weight(s2, w);

        let min = minimize(&dwa);
        assert_eq!(min.num_states(), 3);
        assert!(!min.eval_word(&[0, 1]).is_empty());
    }

    #[test]
    fn test_minimize_deep_merge() {
        // Two identical sub-trees:
        //   s0 --0--> s1 --2--> s3 (final)
        //   s0 --1--> s2 --2--> s4 (final)
        // s3 ≡ s4, then s1 ≡ s2.
        let nt = 1u32;
        let max_tok = 5u32;
        let mut dwa = CompDwa::new(nt, max_tok);
        let s1 = dwa.add_state();
        let s2 = dwa.add_state();
        let s3 = dwa.add_state();
        let s4 = dwa.add_state();
        let w = Weight::all();
        dwa.add_transition(0, 0, s1, w.clone());
        dwa.add_transition(0, 1, s2, w.clone());
        dwa.add_transition(s1, 2, s3, w.clone());
        dwa.add_transition(s2, 2, s4, w.clone());
        dwa.set_final_weight(s3, w.clone());
        dwa.set_final_weight(s4, w);

        assert_eq!(dwa.num_states(), 5);
        let min = minimize(&dwa);
        // s3 ≡ s4, s1 ≡ s2 → 3 states.
        assert_eq!(min.num_states(), 3);
        assert!(!min.eval_word(&[0, 2]).is_empty());
        assert!(!min.eval_word(&[1, 2]).is_empty());
    }

    #[test]
    fn test_ported_min_redundant_states() {
        // Ported from old test_minimize_redundant_states.
        //
        // DWA structure:
        //   s0 --'a'--> s1 --'x'--> s4 (final, w1)
        //   s0 --'b'--> s2 --'y'--> s4   ← s2 and s3 have identical behaviour
        //   s0 --'c'--> s3 --'y'--> s4
        //   s5: added but unreachable (no incoming transitions)
        //
        // minimize merges s2 ≡ s3 (identical signatures).
        // s5 remains because minimize does not prune unreachable states.
        // Net: 6 → 5 states.
        let nt = 1u32;
        let max_tok = 200u32;
        let mut d = CompDwa::new(nt, max_tok);
        let s1 = d.add_state();
        let s2 = d.add_state();
        let s3 = d.add_state();
        let s4 = d.add_state();
        let _s5 = d.add_state(); // Unreachable
        let w_all = Weight::all();
        let w1 = Weight::empty();

        d.add_transition(0, b'a' as i32, s1, w_all.clone());
        d.add_transition(0, b'b' as i32, s2, w_all.clone());
        d.add_transition(0, b'c' as i32, s3, w_all.clone());
        d.add_transition(s1, b'x' as i32, s4, w_all.clone());
        d.add_transition(s2, b'y' as i32, s4, w_all.clone());
        d.add_transition(s3, b'y' as i32, s4, w_all.clone()); // Same behaviour as s2
        d.set_final_weight(s4, w1);

        assert_eq!(d.num_states(), 6);
        let min = minimize(&d);
        // s2 ≡ s3 merged; s5 stays (unreachable but distinct from final s4).
        assert_eq!(min.num_states(), 5, "s2≡s3 should merge; expect 5 states, got {}", min.num_states());

        // Accepted words still work after minimisation
        assert!(!min.eval_word(&[b'a' as i32, b'x' as i32]).is_empty(), "'ax' accepted");
        assert!(!min.eval_word(&[b'b' as i32, b'y' as i32]).is_empty(), "'by' accepted");
        assert!(!min.eval_word(&[b'c' as i32, b'y' as i32]).is_empty(), "'cy' accepted");

        // Rejected words remain rejected
        assert!(min.eval_word(&[b'a' as i32, b'y' as i32]).is_empty(), "'ay' rejected");
        assert!(min.eval_word(&[b'b' as i32, b'x' as i32]).is_empty(), "'bx' rejected");
    }

    #[test]
    fn test_ported_min_chain_narrowing_equivalence() {
        // Ported from old test_minimize_propagates_future_weights.
        //
        // Two DWAs that produce identical eval_word outputs:
        //   A: s0 --'a'(w_all)--> s1 --'b'(w_{1,2})--> s2  (final=w_2)
        //   B: s0 --'a'(w_all)--> s1 --'b'(w_all)  --> s2  (final=w_2)
        //
        // For word "ab":
        //   A: w_all ∩ w_{1..2} ∩ w_2 = {2}
        //   B: w_all ∩ w_all ∩ w_2 = {2}
        //
        // The narrow edge weight in A is redundant because the final weight is
        // the binding constraint.  Both DWAs semantically agree on all inputs.
        // After minimisation this equivalence is preserved.
        let nt = 1u32;
        let max_tok = 5u32;
        let w_all = Weight::all();
        let w_1_2 = Weight::empty();
        let w_2 = Weight::all();

        // DWA A: narrow edge weight on s1 → s2
        let mut a = CompDwa::new(nt, max_tok);
        let s1a = a.add_state();
        let s2a = a.add_state();
        a.add_transition(0, b'a' as i32, s1a, w_all.clone());
        a.add_transition(s1a, b'b' as i32, s2a, w_1_2);
        a.set_final_weight(s2a, w_2.clone());

        // DWA B: all weight on every edge
        let mut b = CompDwa::new(nt, max_tok);
        let s1b = b.add_state();
        let s2b = b.add_state();
        b.add_transition(0, b'a' as i32, s1b, w_all.clone());
        b.add_transition(s1b, b'b' as i32, s2b, w_all.clone());
        b.set_final_weight(s2b, w_2.clone());

        let word = [b'a' as i32, b'b' as i32];
        // Both produce the same result for "ab" before minimisation
        assert_eq!(a.eval_word(&word), b.eval_word(&word), "A and B must agree on 'ab' before minimisation");
        assert!(!a.eval_word(&word).is_empty(), "'ab' must be accepted");

        // After minimisation the equivalence is still preserved
        let min_a = minimize(&a);
        let min_b = minimize(&b);
        assert_eq!(min_a.eval_word(&word), min_b.eval_word(&word), "minimised A and B must still agree on 'ab'");
        assert_eq!(min_a.num_states(), min_b.num_states(), "A and B have the same state count after minimisation");

        // Rejected words remain empty
        assert!(min_a.eval_word(&[b'a' as i32]).is_empty(), "'a' alone rejected");
        assert!(min_a.eval_word(&[b'b' as i32, b'a' as i32]).is_empty(), "'ba' rejected");
    }

    #[test]
    fn test_ported_min_sink_state_collapse() {
        // Ported from old test_equivalence_via_minimization.
        //
        // DWA `a` has explicit non-accepting transitions on labels 1 and 3 that lead to
        // dead-end states (s1a, s2a).  Neither state has a final weight, so reaching
        // either one produces an empty result.
        //
        // Both s1a and s2a are non-accepting dead sinks with no outgoing transitions —
        // their signatures are identical, so minimize should merge them.
        // Result: 3 states → 2 states.
        let nt = 1u32;
        let max_tok = 10u32;
        let w1 = Weight::empty();
        let w_0_1 = Weight::all();
        let w0 = Weight::empty();

        // DWA a: explicit transitions to dead-end sinks on all four labels
        let mut a = CompDwa::new(nt, max_tok);
        let s1a = a.add_state();
        let s2a = a.add_state();
        a.add_transition(0, 0, s1a, w1.clone());
        a.add_transition(0, 1, s2a, w_0_1.clone());
        a.add_transition(0, 2, s1a, w0.clone());
        a.add_transition(0, 3, s1a, w_0_1.clone());
        // s1a and s2a intentionally have no final weight → both are dead sinks

        // Both reject every 1-step word (no final states reachable)
        for label in 0_i32..=3 {
            assert!(a.eval_word(&[label]).is_empty(), "label {label}: a rejects all 1-step words");
        }

        // After minimisation: s1a ≡ s2a (both non-accepting with no transitions) → merge.
        // s0 remains distinct (has outgoing transitions).  Result: 2 states.
        let min_a = minimize(&a);
        assert_eq!(min_a.num_states(), 2, "s1a≡s2a should merge: expect 2 states, got {}", min_a.num_states());

        // Minimised a still rejects everything
        for label in 0_i32..=3 {
            assert!(min_a.eval_word(&[label]).is_empty(), "label {label}: minimised a still rejects all 1-step words");
        }
    }
}
