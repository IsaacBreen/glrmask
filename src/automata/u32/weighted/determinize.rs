//! NWA → CompDwa determinization.
//!
//! Provides two flavors:
//!
//! - [`determinize`] – general-purpose weighted subset construction that
//!   handles arbitrary NWAs (including those with cycles).
//! - [`determinize_acyclic`] – optimised two-phase algorithm for acyclic
//!   NWAs.  Returns an error if the NWA contains cycles.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};

use rustc_hash::FxHashMap;

use super::dwa::{CompDwa, CompDwaState};
use super::nwa::{Label, Nwa};
use super::weight::{TokenSet, Weight};
use crate::GlrMaskError;

type SubsetTransitions = (Vec<BTreeSet<u32>>, Vec<Vec<(Label, u32)>>);

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Determinize an NWA into a compilation-time DWA.
///
/// Works for arbitrary NWAs (acyclic or cyclic).  Uses a worklist-based
/// weighted subset construction with fixed-point ε-closures.
pub fn determinize(nwa: &Nwa) -> CompDwa {
    unimplemented!()
}

/// Determinize an acyclic NWA into a compilation-time DWA.
///
/// Returns an error if the NWA contains cycles.
pub fn determinize_acyclic(nwa: &Nwa) -> Result<CompDwa, GlrMaskError> {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// Topological sort  (Kahn's algorithm)
// ---------------------------------------------------------------------------

fn topo_sort(nwa: &Nwa) -> Result<Vec<u32>, GlrMaskError> {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// Unweighted ε-closures
// ---------------------------------------------------------------------------

/// For each NWA state, compute the set of states reachable via ε-transitions.
fn unweighted_epsilon_closures(nwa: &Nwa, topo: &[u32]) -> Vec<BTreeSet<u32>> {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// Unweighted subset construction
// ---------------------------------------------------------------------------

/// Explore the DWA state space without weights.
///
/// Returns:
/// - `subsets[dwa_id]` = set of NWA states forming that DWA state.
/// - `transitions[dwa_id]` = vec of (label, target_dwa_id).
fn unweighted_subset_construction(nwa: &Nwa, eps_uw: &[BTreeSet<u32>]) -> SubsetTransitions {
    unimplemented!()
}

/// Intern a subset: if already seen return its id, otherwise register it.
fn intern_subset(
    subset: &BTreeSet<u32>,
    subsets: &mut Vec<BTreeSet<u32>>,
    transitions: &mut Vec<Vec<(Label, u32)>>,
    seen: &mut FxHashMap<Vec<u32>, u32>,
    queue: &mut VecDeque<u32>,
) -> u32 {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// Weighted ε-closures
// ---------------------------------------------------------------------------

/// For each NWA state `u`, compute:
///   closure[u] = { (v, w) | v reachable from u via ε, w = ∩ of edge-weights }
///
/// Multiple paths to the same state v are combined with ∪.
fn weighted_epsilon_closures(nwa: &Nwa, topo: &[u32]) -> Vec<BTreeMap<u32, Weight>> {
    unimplemented!()
}

// ---------------------------------------------------------------------------
// Build CompDwa with weights
// ---------------------------------------------------------------------------

fn build_comp_dwa(
    nwa: &Nwa,
    subsets: &[BTreeSet<u32>],
    uw_transitions: &[Vec<(Label, u32)>],
    eps_w: &[BTreeMap<u32, Weight>],
) -> Result<CompDwa, GlrMaskError> {
    unimplemented!()
}

/// Merge weighted ε-closures for all NWA states in a subset.
fn merge_weighted_closures(
    subset: &BTreeSet<u32>,
    eps_w: &[BTreeMap<u32, Weight>],
) -> BTreeMap<u32, Weight> {
    unimplemented!()
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::weighted::weight::TokenSet;

    #[test]
    fn test_determinize_trivial_accepting() {
        // Single-state accepting NWA → single-state accepting DWA.
        let mut nwa = Nwa::new(1, 5);
        let s = nwa.add_state();
        nwa.start_states.push(s);
        nwa.set_final_weight(s, Weight::full());

        let dwa = determinize_acyclic(&nwa).unwrap();
        assert_eq!(dwa.num_states(), 1);
        assert!(dwa.states[0].final_weight.is_some());
    }

    #[test]
    fn test_determinize_linear() {
        // s0 --label 0--> s1 (accepting)
        let nt = 1u32;
        let max_tok = 5u32;
        let mut nwa = Nwa::new(nt, max_tok);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        nwa.start_states.push(s0);

        let w_all = Weight::full();
        nwa.add_transition(s0, 0, s1, w_all.clone());
        nwa.set_final_weight(s1, w_all);

        let dwa = determinize_acyclic(&nwa).unwrap();
        assert_eq!(dwa.num_states(), 2);
        assert!(dwa.states[0].final_weight.is_none());
        assert!(dwa.states[1].final_weight.is_some());

        // eval_word([0]) should be non-empty
        assert!(!dwa.eval_word(&[0]).is_empty());
        // eval_word([1]) should be empty (no transition for label 1)
        assert!(dwa.eval_word(&[1]).is_empty());
    }

    #[test]
    fn test_determinize_nondeterminism() {
        // Two transitions on the same label with disjoint weights.
        // s0 --0,w1--> s1 (accepting)
        // s0 --0,w2--> s2 (accepting)
        let nt = 1u32;
        let max_tok = 5u32;
        let mut nwa = Nwa::new(nt, max_tok);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.start_states.push(s0);

        let w1 = Weight::from_positions(&TokenSet::from_iter([0..=2]), nt);
        let w2 = Weight::from_positions(&TokenSet::from_iter([3..=5]), nt);
        nwa.add_transition(s0, 0, s1, w1);
        nwa.add_transition(s0, 0, s2, w2);
        nwa.set_final_weight(s1, Weight::full());
        nwa.set_final_weight(s2, Weight::full());

        let dwa = determinize_acyclic(&nwa).unwrap();
        let result = dwa.eval_word(&[0]);
        assert!(!result.is_empty());
        // Should see positions from both w1 and w2 (0..=5 → 6 positions).
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_determinize_epsilon() {
        // s0 --ε--> s1 --label 0--> s2 (accepting)
        let nt = 1u32;
        let max_tok = 5u32;
        let mut nwa = Nwa::new(nt, max_tok);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.start_states.push(s0);

        let w_all = Weight::full();
        nwa.add_epsilon(s0, s1, w_all.clone());
        nwa.add_transition(s1, 0, s2, w_all.clone());
        nwa.set_final_weight(s2, w_all);

        let dwa = determinize_acyclic(&nwa).unwrap();
        assert!(!dwa.eval_word(&[0]).is_empty());
    }

    #[test]
    fn test_determinize_cycle_rejected() {
        let mut nwa = Nwa::new(1, 5);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        nwa.start_states.push(s0);
        let w = Weight::full();
        nwa.add_epsilon(s0, s1, w.clone());
        nwa.add_epsilon(s1, s0, w);

        assert!(determinize_acyclic(&nwa).is_err());
    }

    #[test]
    fn test_determinize_empty_nwa() {
        let nwa = Nwa::new(1, 5);
        let dwa = determinize_acyclic(&nwa).unwrap();
        // CompDwa::new creates a single dead start state.
        assert_eq!(dwa.num_states(), 1);
        assert!(dwa.states[0].final_weight.is_none());
    }

    #[test]
    fn test_determinize_no_start_states() {
        // NWA with states but no start states → start subset = ∅ → 1 dead DWA state.
        let mut nwa = Nwa::new(1, 5);
        let s0 = nwa.add_state();
        nwa.set_final_weight(s0, Weight::full());
        // No start_states pushed.
        let dwa = determinize_acyclic(&nwa).unwrap();
        assert_eq!(dwa.num_states(), 1);
        assert!(dwa.states[0].final_weight.is_none());
    }

    #[test]
    fn test_determinize_chain_with_epsilon() {
        // s0 --0,w_all--> s1 --ε,w_all--> s2 --1,w_all--> s3 (accepting)
        let nt = 1u32;
        let max_tok = 5u32;
        let mut nwa = Nwa::new(nt, max_tok);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        let s3 = nwa.add_state();
        nwa.start_states.push(s0);

        let w_all = Weight::full();
        nwa.add_transition(s0, 0, s1, w_all.clone());
        nwa.add_epsilon(s1, s2, w_all.clone());
        nwa.add_transition(s2, 1, s3, w_all.clone());
        nwa.set_final_weight(s3, w_all);

        let dwa = determinize_acyclic(&nwa).unwrap();
        // Word [0, 1] should reach the accepting state.
        assert!(!dwa.eval_word(&[0, 1]).is_empty());
        // Word [0] alone should NOT be accepting.
        assert!(dwa.eval_word(&[0]).is_empty());
        // Word [1] alone should NOT have a transition from start.
        assert!(dwa.eval_word(&[1]).is_empty());
    }

    #[test]
    fn test_determinize_weight_filtering() {
        // s0 --0,w_small--> s1 (accepting with w_all)
        // Only positions in w_small should survive.
        let nt = 1u32;
        let max_tok = 10u32;
        let mut nwa = Nwa::new(nt, max_tok);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        nwa.start_states.push(s0);

        let w_small = Weight::from_positions(&TokenSet::from_iter([2..=5]), nt);
        let w_all = Weight::full();
        nwa.add_transition(s0, 0, s1, w_small);
        nwa.set_final_weight(s1, w_all);

        let dwa = determinize_acyclic(&nwa).unwrap();
        let result = dwa.eval_word(&[0]);
        assert!(!result.is_empty());
        // Only positions 2..=5 survive the intersection.
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_ported_det_diverging_epsilon_paths() {
        // Ported from old test_determinize_simple_divergence (was #[should_panic] in the
        // old codebase — old code panicked on this; new code handles it correctly).
        //
        // Two NWA paths joined by epsilon from a super-start:
        //   super_start --eps--> s0 --'a'--> s1 --'c'--> s2  (final, pos 0)
        //   super_start --eps--> s3 --'b'--> s4 --'c'--> s5  (final, pos 1)
        //
        // After determinisation:
        //   eval("ac") contains pos 0 but NOT pos 1
        //   eval("bc") contains pos 1 but NOT pos 0
        let nt = 1u32;
        let max_tok = 200u32; // Must cover 'a'=97, 'b'=98, 'c'=99
        let mut nwa = Nwa::new(nt, max_tok);
        let w_all = Weight::full();
        let w0 = Weight::from_positions(&TokenSet::from_iter([0..=0]), nt);
        let w1 = Weight::from_positions(&TokenSet::from_iter([1..=1]), nt);

        // Path 1: s0 --'a'--> s1 --'c'--> s2 (final, w0)
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.add_transition(s0, b'a' as i32, s1, w_all.clone());
        nwa.add_transition(s1, b'c' as i32, s2, w_all.clone());
        nwa.set_final_weight(s2, w0.clone());

        // Path 2: s3 --'b'--> s4 --'c'--> s5 (final, w1)
        let s3 = nwa.add_state();
        let s4 = nwa.add_state();
        let s5 = nwa.add_state();
        nwa.add_transition(s3, b'b' as i32, s4, w_all.clone());
        nwa.add_transition(s4, b'c' as i32, s5, w_all.clone());
        nwa.set_final_weight(s5, w1.clone());

        // Super-start with epsilon transitions to both paths
        let super_start = nwa.add_state();
        nwa.add_epsilon(super_start, s0, w_all.clone());
        nwa.add_epsilon(super_start, s3, w_all.clone());
        nwa.start_states.push(super_start);

        let dwa = determinize_acyclic(&nwa).unwrap();

        // eval("ac") should contain pos 0 only
        let r_ac = dwa.eval_word(&[b'a' as i32, b'c' as i32]);
        assert!(!r_ac.is_empty(), "'ac' should be accepted");
        assert!(r_ac.contains(0, nt), "'ac' should yield weight pos 0");
        assert!(!r_ac.contains(1, nt), "'ac' should NOT yield weight pos 1");

        // eval("bc") should contain pos 1 only
        let r_bc = dwa.eval_word(&[b'b' as i32, b'c' as i32]);
        assert!(!r_bc.is_empty(), "'bc' should be accepted");
        assert!(r_bc.contains(1, nt), "'bc' should yield weight pos 1");
        assert!(!r_bc.contains(0, nt), "'bc' should NOT yield weight pos 0");

        // eval("a") alone is empty (no final state after one step)
        assert!(dwa.eval_word(&[b'a' as i32]).is_empty(), "'a' alone should be empty");

        // DWA should have a compact number of states
        assert!(dwa.num_states() <= 5, "DWA should have ≤5 states, got {}", dwa.num_states());
    }

    #[test]
    fn test_ported_det_epsilon_convergence() {
        // Ported from test_epsilon_explosion_minimal (correctness aspect).
        //
        // Two epsilon branches share the same terminal label and converge to one DFA state:
        //   super_start --eps--> s0  (s0: 'x' -> s_final with w0)
        //   super_start --eps--> s1  (s1: 'x' -> s_final with w1)
        //   s_final: final weight w_all
        //
        // On reading 'x', both branches arrive at s_final with their respective
        // per-transition weights; the resulting weight is the union: w0 ∪ w1.
        let nt = 1u32;
        let max_tok = 200u32;
        let mut nwa = Nwa::new(nt, max_tok);
        let w_all = Weight::full();
        let w0 = Weight::from_positions(&TokenSet::from_iter([0..=0]), nt);
        let w1 = Weight::from_positions(&TokenSet::from_iter([1..=1]), nt);

        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        let s_final = nwa.add_state();
        nwa.add_transition(s0, b'x' as i32, s_final, w0.clone());
        nwa.add_transition(s1, b'x' as i32, s_final, w1.clone());
        nwa.set_final_weight(s_final, w_all.clone());

        let super_start = nwa.add_state();
        nwa.add_epsilon(super_start, s0, w_all.clone());
        nwa.add_epsilon(super_start, s1, w_all.clone());
        nwa.start_states.push(super_start);

        let dwa = determinize_acyclic(&nwa).unwrap();

        // eval("x") should contain both pos 0 (from s0 branch) and pos 1 (from s1 branch)
        let r = dwa.eval_word(&[b'x' as i32]);
        assert!(!r.is_empty(), "'x' should be accepted");
        assert!(r.contains(0, nt), "'x' should yield pos 0 (from s0 branch)");
        assert!(r.contains(1, nt), "'x' should yield pos 1 (from s1 branch)");

        // eval("y") should be empty (no transition on 'y')
        assert!(dwa.eval_word(&[b'y' as i32]).is_empty(), "'y' should be rejected");
    }
}
