use crate::precompute4::weighted_automata::{DWA, NWA, NWAStateID, NWAStates, Weight};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};

/// High-level strategy:
/// - Convert DWA -> NWA.
/// - Iteratively perform local rewrites to resolve negative codes:
///   For each A -neg(x)-> B:
///     1) Propagate B's final weight gated by the neg-edge weight into A's final.
///     2) If B has a positive edge 'x' to C (cancellation), add epsilon A --eps--> C with the combined weight.
///     3) Replace the edge target with a copy B' that has only negative-labeled transitions (and no epsilons), and with its final weight cleared.
/// - After each full pass of rewrites, simplify the NWA to collapse epsilons and prune the graph.
/// - Repeat until a pass makes no changes, then determinize back to a DWA and simplify.
pub fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    let pb = if PROGRESS_BAR_ENABLED {
        let p = ProgressBar::new(1);
        p.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [Resolving negative codes: {elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} passes ({msg})")
                .expect("progress-bar"),
        );
        Some(p)
    } else {
        None
    };

    let mut nwa = NWA::from_dwa(dwa);
    let mut passes = 0;
    loop {
        passes += 1;
        if let Some(p) = &pb {
            p.set_length(passes as u64);
            p.set_position(passes as u64);
        }

        let mut changed_in_pass = false;
        let n = nwa.states.len();
        for state_id in 0..n {
            if resolve_negative_codes_in_nwa_internal(state_id, &mut nwa.states) {
                changed_in_pass = true;
            }
        }

        if let Some(p) = &pb {
            p.set_message(if changed_in_pass { "changes made" } else { "stable" });
        }

        if !changed_in_pass {
            break;
        }

        // Cheaper normalization step than full determinization
        nwa.simplify();
    }

    let mut result = nwa.determinize_to_dwa();
    result.simplify();
    *dwa = result;

    if let Some(p) = &pb {
        p.finish_with_message("Done");
    }
}

fn resolve_negative_codes_in_nwa_internal(
    state_id: NWAStateID,
    states: &mut NWAStates,
) -> bool {
    let mut changed = false;

    // Collect negative transitions first to avoid borrow issues
    let negatives: Vec<(i16, NWAStateID, Weight)> = {
        let st = &states[state_id];
        st.transitions
            .iter()
            .filter(|(k, _)| **k < 0)
            .map(|(k, (t, w))| (*k, *t, w.clone()))
            .collect()
    };

    if negatives.is_empty() {
        return false;
    }

    for (neg_code, b_orig_id, w_neg) in negatives {
        let p = neg_code.wrapping_sub(i16::MIN);

        // Step 1: Propagate final weight from B into A
        if let Some(b_final) = states[b_orig_id].final_weight.clone() {
            let new_a_final = &w_neg & &b_final;
            if !new_a_final.is_empty() {
                let a_fw_before = states[state_id].final_weight.clone();
                if let Some(a_fw) = states[state_id].final_weight.as_mut() {
                    *a_fw |= &new_a_final;
                } else {
                    states[state_id].final_weight = Some(new_a_final);
                }
                if states[state_id].final_weight != a_fw_before {
                    changed = true;
                }
            }
        }

        // Handle cancellation if B has a positive edge on p
        if let Some((c_orig_id, w_b_c)) = states[b_orig_id].get_transition(p).cloned() {
            let w = &w_neg & &w_b_c;
            if !w.is_empty() {
                states.add_epsilon(state_id, c_orig_id, w);
                changed = true;
            }
        }

        // Check if B needs to be split: if it has positive behavior that should not be triggered by the neg path.
        let b_needs_splitting = {
            let b_orig = &states[b_orig_id];
            b_orig.final_weight.is_some()
                || b_orig.transitions.keys().any(|k| *k >= 0)
                || b_orig.default.is_some()
        };

        if b_needs_splitting {
            // Step 2: Copy B and strip positive transitions; also clear final weight in the copy.
            let b_copy_id = states.copy_state(b_orig_id);
            {
                let b_copy = &mut states[b_copy_id];
                b_copy.final_weight = None;
                b_copy.transitions.retain(|k, _| *k < 0);
                b_copy.default = None;
            }

            // Step 3: Replace edge A -(neg)-> B with A -(neg)-> B_copy
            if let Some((ref mut tgt, _)) = states[state_id].transitions.get_mut(&neg_code) {
                if *tgt != b_copy_id {
                    *tgt = b_copy_id;
                    changed = true;
                }
            }
        }
    }

    changed
}
