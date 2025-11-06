use crate::precompute4::weighted_automata::{
    DWA, NWA, NWAState, NWAStateID, NWAStates, Weight,
};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::BTreeMap;

/// High-level, faster strategy:
/// - Convert DWA -> NWA (default transitions stay as defaults).
/// - Single fixpoint-like in-place pass (queue over growing arena) that:
///   For each negative edge A -neg(x)-> B with weight w_neg:
///     1) Propagates B.final_weight gated by w_neg into A.final_weight.
///     2) Adds ε: A --eps--> C with weight (w_neg & w_BxC) if B has a positive 'x' (or default) to C.
///     3) Retargets A -neg(x)-> B' where B' is a memoized "neg-only" clone of B:
///        - final_weight cleared
///        - default removed
///        - epsilons cleared
///        - only negative-labeled transitions retained
/// - We process all states, including newly created neg-only clones (by iterating
///   with a dynamically increasing upper bound).
/// - Determinize back to DWA and simplify once at the end.
///
/// This removes the need for multiple passes and intermediate determinization, making it drastically faster.
pub fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    let pb = if PROGRESS_BAR_ENABLED {
        let p = ProgressBar::new(1);
        p.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [Resolving negative codes: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} states ({msg})")
                .expect("progress-bar"),
        );
        Some(p)
    } else {
        None
    };

    // Convert to NWA
    crate::debug!(3, "Initial DWA: {}", dwa);
    let mut nwa = NWA::from_dwa(dwa);

    // Fast in-place resolution
    resolve_negative_codes_in_nwa_fast(&mut nwa, pb.as_ref());

    // Final determinization to DWA
    let mut result = nwa.determinize_to_dwa();
    result.simplify();
    crate::debug!(3, "Final DWA: {}", result);
    *dwa = result;
}

/// Returns true if the state already conforms to the "neg-only" shape:
/// - No final_weight
/// - No default
/// - No epsilons
/// - All labeled transitions are on negative labels
fn is_neg_only(st: &NWAState) -> bool {
    if st.final_weight.is_some() || st.default.is_some() || !st.epsilons.is_empty() {
        return false;
    }
    st.transitions.keys().all(|k| *k < 0)
}

/// Ensure we have a "neg-only" clone for the given state `orig`.
/// If `orig` already is neg-only, return `orig`.
/// Otherwise, memoize and return a (single) neg-only clone:
/// - final_weight cleared
/// - default removed
/// - epsilons cleared
/// - only negative-labeled transitions retained
fn ensure_neg_only_clone(
    states: &mut NWAStates,
    neg_clone_of: &mut Vec<Option<NWAStateID>>,
    orig: NWAStateID,
) -> NWAStateID {
    // Ensure mapping grows with the arena
    if neg_clone_of.len() <= orig {
        neg_clone_of.resize(states.len(), None);
    }

    // If the state is already neg-only, reuse it directly (no extra clone needed)
    if is_neg_only(&states[orig]) {
        return orig;
    }

    // If we already created a neg-only clone for `orig`, reuse it
    if let Some(id) = neg_clone_of[orig] {
        return id;
    }

    // Otherwise, create a clone and strip it
    let new_id = states.copy_state(orig);
    {
        let st = &mut states[new_id];
        st.final_weight = None;
        st.default = None;
        st.epsilons.clear();
        st.transitions.retain(|k, _| *k < 0);
    }
    // Ensure mapping can index this new state if the arena just grew
    if neg_clone_of.len() <= new_id {
        neg_clone_of.resize(new_id + 1, None);
    }
    neg_clone_of[orig] = Some(new_id);
    new_id
}

/// Merge (or add) an epsilon edge from `from` to `to` with weight `w_add`.
/// If an epsilon to the same destination already exists, OR the weight into it.
fn merge_or_add_epsilon(states: &mut NWAStates, from: NWAStateID, to: NWAStateID, w_add: Weight) {
    if w_add.is_empty() {
        return;
    }
    let eps = &mut states[from].epsilons;
    if let Some((_, w_existing)) = eps.iter_mut().find(|(t, _)| *t == to) {
        *w_existing |= &w_add;
    } else {
        eps.push((to, w_add));
    }
}

/// Perform the fast, single-pass (with dynamic arena) negative-code resolution.
fn resolve_negative_codes_in_nwa_fast(nwa: &mut NWA, pb: Option<&ProgressBar>) {
    let states = &mut nwa.states;

    // Memoization: for each original state ID, optional "neg-only" clone ID
    let mut neg_clone_of: Vec<Option<NWAStateID>> = vec![None; states.len()];

    // We'll process states in a queue-like manner by iterating over indices while allowing growth.
    let mut i: usize = 0;

    // Initialize progress
    if let Some(p) = pb {
        p.set_length(states.len() as u64);
        p.set_position(0);
        p.set_message("pre-scan");
    }

    while i < states.len() {
        // Keep progress updated
        if let Some(p) = pb {
            if p.length().unwrap_or(0) != states.len() as u64 {
                p.set_length(states.len() as u64);
            }
            p.set_position(i as u64);
            p.set_message("processing");
        }

        // Collect negative transitions for state `i` to avoid borrow conflicts
        let negatives: Vec<(i16, NWAStateID, Weight)> = {
            let st = &states[i];
            st.transitions
                .iter()
                .filter(|(lbl, _)| **lbl < 0)
                .map(|(lbl, (to, w))| (*lbl, *to, w.clone()))
                .collect()
        };

        if !negatives.is_empty() {
            // Accumulate final-weight additions and cancellation epsilons per destination
            let mut final_acc = Weight::zeros();
            let mut eps_acc: BTreeMap<NWAStateID, Weight> = BTreeMap::new();

            for (neg_code, b_orig, w_neg) in negatives {
                // Step 1: Propagate final weight from B into A
                if let Some(b_final) = states[b_orig].final_weight.as_ref() {
                    let add = &w_neg & b_final;
                    if !add.is_empty() {
                        final_acc |= &add;
                    }
                }

                // Step 2: Cancellation if B has a positive edge on p (or a default)
                let p_code = neg_code.wrapping_sub(i16::MIN);
                if let Some((c_orig, w_b_c)) = states[b_orig].get_transition(p_code).cloned() {
                    let add = &w_neg & &w_b_c;
                    if !add.is_empty() {
                        let entry = eps_acc.entry(c_orig).or_insert_with(Weight::zeros);
                        *entry |= &add;
                    }
                }

                // Step 3: Retarget neg edge to a neg-only clone of B
                let b_neg_id = ensure_neg_only_clone(states, &mut neg_clone_of, b_orig);
                if let Some((ref mut tgt, _)) = states[i].transitions.get_mut(&neg_code) {
                    *tgt = b_neg_id;
                }
            }

            // Apply accumulated final additions to A
            if !final_acc.is_empty() {
                if let Some(a_fw) = states[i].final_weight.as_mut() {
                    *a_fw |= &final_acc;
                } else {
                    states[i].final_weight = Some(final_acc);
                }
            }

            // Merge accumulated epsilons to A
            for (to, w) in eps_acc {
                merge_or_add_epsilon(states, i, to, w);
            }
        }

        // Ensure memoization vector keeps up with arena growth
        if neg_clone_of.len() < states.len() {
            neg_clone_of.resize(states.len(), None);
        }

        i += 1;
    }

    if let Some(p) = pb {
        p.set_position(states.len() as u64);
        p.set_message("done");
        p.finish_and_clear();
    }
}
