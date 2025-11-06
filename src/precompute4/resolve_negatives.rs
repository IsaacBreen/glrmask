//! High-performance negative-code resolution
//!
//! Semantics preserved from the previous implementation, but executed with a single
//! monotone, worklist-based pass over an NWA and a final determinization + simplify.
//!
//! Key properties preserved (for every negative edge A --neg(x,w_neg)-> B):
//!   1) Final propagation: A.final |= (w_neg & B.final).
//!   2) Cancellation: if B has a positive/default transition on x to C with weight w_bx,
//!      then add an epsilon edge A --eps,(w_neg & w_bx)--> C.
//!   3) Target isolation: if B has any positive/default/final behavior, redirect the
//!      negative edge to a canonical "negative-only" copy B' (final cleared, default removed,
//!      only negative-labeled transitions retained; epsilons preserved).
//!
//! The algorithm processes every negative-labeled edge exactly once (per originating state
//! and label), recursively including edges from any canonical negative-only copies it creates.
//! This guarantees that after the worklist empties, every negative edge points to a
//! negative-only target, and all final/cancellation contributions have been added exactly once.
//!
//! Performance guarantees:
//! - Each (state, negative-label) pair is processed at most once: O(E_neg).
//! - Each original target that needs isolation is copied at most once and shared across
//!   all incoming negative edges: O(#states_needing_isolation + E_neg).
//! - Cancellation lookups are cached per (state, positive-code) to avoid repeated map lookups.
//!
//! Correctness sketch (matches prior version):
//! - Step (1) adds acceptance at empty suffix after consuming a single neg symbol; this is
//!   independent per (A,neg-label) and unaffected by later rewrites.
//! - Step (2) cancellation is purely local to (B,x): we add ε to C with weight intersection;
//!   repeatedly adding the same ε is idempotent, so one-time processing suffices.
//! - Step (3) target isolation preserves languages: redirecting to B' only changes the
//!   state seen after the neg-step to one without positive/default/final behavior;
//!   this is the same effect as repeatedly splitting in successive passes. A single, canonical
//!   copy shared across all incoming negative edges is sufficient, as the copy's content is
//!   independent of the incoming weight and label.
//! - Recursively processing negative edges from B' (and deeper copies) is exactly what multiple
//!   outer passes achieved previously; we do it eagerly in one worklist pass.
//!
//! After the transformation stabilizes, we determinize to a DWA and simplify, as before.

use crate::precompute4::weighted_automata::{DWA, NWA, NWAStateID, NWAStates, Weight};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet, VecDeque};

pub fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    let pb = if PROGRESS_BAR_ENABLED {
        let p = ProgressBar::new(1);
        p.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [Resolving negative codes (fast): {elapsed_precise}] \
                           [{wide_bar:.cyan/blue}] {pos}/{len} edges ({msg})")
                .expect("progress-bar"),
        );
        Some(p)
    } else {
        None
    };

    crate::debug!(3, "Initial DWA: {}", dwa);

    // Convert once to NWA
    let mut nwa = NWA::from_dwa(dwa);

    // Fast in-place resolution in the NWA arena
    fast_resolve_negatives_in_nwa(&mut nwa.states, &pb);

    // Final determinization to DWA and simplify (as before)
    let mut result = nwa.determinize_to_dwa();
    result.simplify();

    crate::debug!(3, "Final DWA: {}", result);
    *dwa = result;
}

fn fast_resolve_negatives_in_nwa(states: &mut NWAStates, pb: &Option<ProgressBar>) {
    let n0 = states.len();
    if n0 == 0 {
        if let Some(p) = pb {
            p.finish_with_message("empty");
        }
        return;
    }

    // Worklist of negative edges, identified by (from_state, neg_label).
    // Each pair is processed at most once.
    let mut work_q: VecDeque<(NWAStateID, i16)> = VecDeque::new();
    let mut seen: HashSet<(NWAStateID, i16)> = HashSet::new();

    // Cache for cancellation lookups: (state, pos_code) -> Option<(to, weight)>
    let mut cancel_cache: HashMap<(NWAStateID, i16), Option<(NWAStateID, Weight)>> = HashMap::new();

    // Canonical "negative-only" copy per original state that needs isolation.
    // If a state doesn't need isolation, we never cache it and use it as-is.
    let mut neg_only_copy_of: HashMap<NWAStateID, NWAStateID> = HashMap::new();

    // Initialize the worklist with all existing negative-labeled edges.
    // Also count edges for initial progress-bar setup.
    let mut initial_neg_edges = 0usize;
    for s in 0..states.len() {
        enqueue_neg_edges_from(s, states, &mut work_q, &mut seen, &mut initial_neg_edges);
    }
    if let Some(p) = pb.as_ref() {
        p.set_length(initial_neg_edges.max(1) as u64);
        p.set_position(0);
        p.set_message("scanning");
    }

    // Process the queue
    let mut processed_edges = 0usize;
    while let Some((from, neg_lbl)) = work_q.pop_front() {
        // The queue may contain stale entries if the edge was removed; check existence.
        let (to, w_neg) = match states.0.get(from).and_then(|st| st.transitions.get(&neg_lbl)).cloned() {
            Some(pair) => pair,
            None => continue,
        };

        processed_edges += 1;
        if let Some(p) = pb.as_ref() {
            // Keep the bar moving; expand its length conservatively as new edges appear.
            if processed_edges as u64 > p.length().unwrap_or(1) {
                p.set_length(processed_edges as u64);
            }
            p.set_position(processed_edges as u64);
            p.set_message("resolving");
        }

        // Step (1): propagate final weight from 'to' into 'from' gated by w_neg
        if let Some(b_final) = states[to].final_weight.clone() {
            let new_a_final = &w_neg & &b_final;
            if !new_a_final.is_empty() {
                let before = states[from].final_weight.clone();
                if let Some(a_fw) = states[from].final_weight.as_mut() {
                    *a_fw |= &new_a_final;
                } else {
                    states[from].final_weight = Some(new_a_final);
                }
                if states[from].final_weight != before {
                    // No need to re-enqueue anything; finals are output-only for this pass.
                }
            }
        }

        // Step (2): cancellation via positive 'p' for this neg label
        let pcode = neg_to_pos(neg_lbl);
        if pcode >= 0 {
            let key = (to, pcode);
            let to_pos = if let Some(res) = cancel_cache.get(&key) {
                res.clone()
            } else {
                let found = states[to].get_transition(pcode).cloned();
                cancel_cache.insert(key, found.clone());
                found
            };
            if let Some((c, w_bp)) = to_pos {
                let w_eps = &w_neg & &w_bp;
                if !w_eps.is_empty() {
                    states.add_epsilon(from, c, w_eps);
                }
            }
        }

        // Step (3): redirect to negative-only copy if target has positive/default/final behavior
        if state_needs_isolation(&states.0[to]) {
            let neg_id = *neg_only_copy_of.entry(to).or_insert_with(|| {
                let copy_id = states.copy_state(to);
                {
                    let st = &mut states.0[copy_id];
                    // "Negative-only" copy: clear final, remove default, keep epsilons,
                    // keep only negative-labeled transitions.
                    st.final_weight = None;
                    st.default = None;
                    st.transitions.retain(|k, _| *k < 0);
                }
                // Newly created state's negative edges must also be processed eventually.
                enqueue_neg_edges_from(copy_id, states, &mut work_q, &mut seen, &mut initial_neg_edges);
                copy_id
            });
            if let Some((ref mut tgt, _w)) = states.0[from].transitions.get_mut(&neg_lbl) {
                *tgt = neg_id;
            }
        }
    }

    if let Some(p) = pb {
        p.finish_with_message(format!("processed {} negative edges", processed_edges));
    }
}

#[inline]
fn neg_to_pos(neg_code: i16) -> i16 {
    // Matches prior semantics: map neg(x) encoded as i16 to its positive counterpart 'x'.
    // Invariant: neg_code < 0.
    neg_code.wrapping_sub(i16::MIN)
}

#[inline]
fn state_needs_isolation(st: &crate::precompute4::weighted_automata::nwa::NWAState) -> bool {
    st.final_weight.is_some()
        || st.default.is_some()
        || st.transitions.keys().any(|&k| k >= 0)
}

fn enqueue_neg_edges_from(
    from: NWAStateID,
    states: &NWAStates,
    q: &mut VecDeque<(NWAStateID, i16)>,
    seen: &mut HashSet<(NWAStateID, i16)>,
    counter: &mut usize,
) {
    if from >= states.len() {
        return;
    }
    let st = &states.0[from];
    for (&lbl, _) in &st.transitions {
        if lbl < 0 {
            if seen.insert((from, lbl)) {
                q.push_back((from, lbl));
                *counter += 1;
            }
        }
    }
}
