//! CompDwa minimization.
//!
//! Signature-based partition minimization for acyclic [`CompDwa`]s.
//!
//! States are processed bottom-up (by height).  Two states are merged iff
//! they agree on:
//! - final weight (or both non-accepting),
//! - the set of outgoing labels,
//! - for each label: the equivalence class of the target state *and* the
//!   transition weight.
//!
//! Because the DWA is acyclic, a single bottom-up pass suffices.

use std::collections::BTreeMap;

use rustc_hash::FxHashMap;

use super::dwa::{CompDwa, CompDwaState};
use super::nwa::Label;
use super::weight::Weight;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Minimize a [`CompDwa`] by merging states with identical behaviour.
pub fn minimize_acyclic(dwa: &CompDwa) -> CompDwa {
    let n = dwa.states.len();
    if n <= 1 {
        return dwa.clone();
    }

    // 1. Compute heights (leaf = 0).
    let heights = compute_heights(dwa);

    // 2. Sort states by height (ascending).
    let mut by_height: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (i, &h) in heights.iter().enumerate() {
        by_height.entry(h).or_default().push(i as u32);
    }

    // 3. Bottom-up signature computation.
    //    `class[old_state]` = new canonical state id.
    let mut class: Vec<u32> = vec![u32::MAX; n];
    let mut canon_states: Vec<CompDwaState> = Vec::new();
    // Signature → canonical id.
    let mut sig_map: FxHashMap<Sig, u32> = FxHashMap::default();

    for states_at_h in by_height.values() {
        for &sid in states_at_h {
            let st = &dwa.states[sid as usize];
            let sig = state_signature(st, &class);

            if let Some(&canon) = sig_map.get(&sig) {
                // Merge: map this state to the existing canonical state.
                class[sid as usize] = canon;
            } else {
                // New canonical state.
                let cid = canon_states.len() as u32;
                sig_map.insert(sig, cid);
                class[sid as usize] = cid;

                // Build the canonical CompDwaState with remapped targets.
                let mut transitions = BTreeMap::new();
                for (&label, (target, w)) in &st.transitions {
                    transitions.insert(label, (class[*target as usize], w.clone()));
                }
                canon_states.push(CompDwaState {
                    final_weight: st.final_weight.clone(),
                    transitions,
                });
            }
        }
    }

    CompDwa {
        states: canon_states,
        start_state: class[dwa.start_state as usize],
        num_tsids: dwa.num_tsids,
        max_token: dwa.max_token,
    }
}

// ---------------------------------------------------------------------------
// Heights
// ---------------------------------------------------------------------------

fn compute_heights(dwa: &CompDwa) -> Vec<u32> {
    let n = dwa.states.len();
    let mut heights = vec![0u32; n];

    // Topological sort via Kahn's algorithm.
    let mut indegree = vec![0u32; n];
    for st in &dwa.states {
        for (target, _) in st.transitions.values() {
            indegree[*target as usize] += 1;
        }
    }
    let mut queue = std::collections::VecDeque::new();
    for (i, &d) in indegree.iter().enumerate() {
        if d == 0 {
            queue.push_back(i as u32);
        }
    }
    let mut topo = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        topo.push(u);
        for (target, _) in dwa.states[u as usize].transitions.values() {
            let d = &mut indegree[*target as usize];
            *d -= 1;
            if *d == 0 {
                queue.push_back(*target);
            }
        }
    }

    // Reverse topo: compute heights bottom-up.
    for &u in topo.iter().rev() {
        let h = dwa.states[u as usize]
            .transitions
            .values()
            .map(|(t, _)| heights[*t as usize] + 1)
            .max()
            .unwrap_or(0);
        heights[u as usize] = h;
    }

    heights
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

fn state_signature(st: &CompDwaState, class: &[u32]) -> Sig {
    let transitions: Vec<(Label, u32, Weight)> = st
        .transitions
        .iter()
        .map(|(&label, (target, w))| (label, class[*target as usize], w.clone()))
        .collect();
    Sig {
        final_weight: st.final_weight.clone(),
        transitions,
    }
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ds::RangeSet;

    #[test]
    fn test_minimize_identity() {
        // Already-minimal 2-state DWA.
        let nt = 1u32;
        let max_tok = 5u32;
        let mut dwa = CompDwa::new(nt, max_tok);
        let s1 = dwa.add_state();
        let w_all = Weight::all(5, nt);
        dwa.add_transition(0, 0, s1, w_all.clone());
        dwa.set_final_weight(s1, w_all);

        let min = minimize_acyclic(&dwa);
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
        let w = Weight::all(5, nt);
        dwa.add_transition(0, 0, s1, w.clone());
        dwa.add_transition(0, 1, s2, w.clone());
        dwa.set_final_weight(s1, w.clone());
        dwa.set_final_weight(s2, w);

        assert_eq!(dwa.num_states(), 3);
        let min = minimize_acyclic(&dwa);
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
        let w = Weight::all(5, nt);
        let w1 = Weight::from_positions(&RangeSet::from_range(0, 2), nt);
        let w2 = Weight::from_positions(&RangeSet::from_range(3, 5), nt);
        dwa.add_transition(0, 0, s1, w.clone());
        dwa.add_transition(0, 1, s2, w);
        dwa.set_final_weight(s1, w1);
        dwa.set_final_weight(s2, w2);

        let min = minimize_acyclic(&dwa);
        // Cannot merge: different final weights.
        assert_eq!(min.num_states(), 3);
    }

    #[test]
    fn test_minimize_single_state() {
        let dwa = CompDwa::new(1, 5);
        let min = minimize_acyclic(&dwa);
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
        let w = Weight::all(5, nt);
        dwa.add_transition(0, 0, s1, w.clone());
        dwa.add_transition(s1, 1, s2, w.clone());
        dwa.set_final_weight(s2, w);

        let min = minimize_acyclic(&dwa);
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
        let w = Weight::all(5, nt);
        dwa.add_transition(0, 0, s1, w.clone());
        dwa.add_transition(0, 1, s2, w.clone());
        dwa.add_transition(s1, 2, s3, w.clone());
        dwa.add_transition(s2, 2, s4, w.clone());
        dwa.set_final_weight(s3, w.clone());
        dwa.set_final_weight(s4, w);

        assert_eq!(dwa.num_states(), 5);
        let min = minimize_acyclic(&dwa);
        // s3 ≡ s4, s1 ≡ s2 → 3 states.
        assert_eq!(min.num_states(), 3);
        assert!(!min.eval_word(&[0, 2]).is_empty());
        assert!(!min.eval_word(&[1, 2]).is_empty());
    }
}
