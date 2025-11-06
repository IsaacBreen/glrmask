use crate::precompute4::full_dwa::Precomputed4;
use crate::precompute4::weighted_automata::{DWA, NWA, NWAStateID, NWAStates, NWABody, StateID, Weight};
use std::collections::BTreeMap;

pub fn resolve_negative_codes_for_all(precomputed4: &mut Precomputed4) {
    for (_sid, dwa) in precomputed4.iter_mut() {
        resolve_negative_codes_in_dwa(dwa);
    }
}

/// High-level strategy:
/// - Convert DWA -> NWA (default transitions become epsilons).
/// - Iteratively perform local rewrites to resolve negative codes:
///   For each A -neg(x)-> B:
///     1) Propagate B's final weight gated by the neg-edge weight into A's final.
///     2) Replace the edge target with a copy B' that has only negative-labeled transitions (and no epsilons), and with its final weight cleared.
///     3) If B has a positive edge 'x' to C (cancellation), add epsilon A --eps--> C with weight (w_neg & w_BxC).
/// - Repeat until a pass makes no change; determinize back to DWA and simplify.
/// This is correctness-first; performance will be improved later.
pub fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    // Convert to NWA
    crate::debug!(3, "Initial DWA: {}", dwa);
    let mut nwa = NWA::from_dwa(dwa);
    loop {
        let mut changed_in_pass = false;

        let n = nwa.states.len();
        for state_id in 0..n {
            let changed = resolve_negative_codes_in_nwa_internal(state_id, &mut nwa.states);
            if changed {
                changed_in_pass = true;
            }
        }

        if !changed_in_pass {
            break;
        }
        // Determinize to DWA then back to NWA to normalize the graph, which helps subsequent passes.
        let mut tmp_dwa = nwa.determinize_to_dwa();
        tmp_dwa.simplify();
        crate::debug!(3, "Intermediate DWA: {}", tmp_dwa);
        nwa = NWA::from_dwa(&tmp_dwa);
    }
    // Final determinization to DWA
    let mut result = nwa.determinize_to_dwa();
    result.simplify();
    crate::debug!(3, "Final DWA: {}", result);
    *dwa = result;
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
            let (ref mut tgt, _) = states[state_id].transitions.get_mut(&neg_code).unwrap();
            *tgt = b_copy_id;
            changed = true;
        }
    }

    changed
}
