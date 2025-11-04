use crate::precompute4::full_dwa::Precomputed4;
use crate::precompute4::weighted_automata::{DWA, DWAState, DWAStates, StateID, Weight};

pub fn resolve_negative_codes_for_all(precomputed4: &mut Precomputed4) {
    for (_sid, dwa) in precomputed4.iter_mut() {
        resolve_negative_codes_in_dwa(dwa);
    }
}

fn is_negative_code(code: i16) -> bool {
    code < 0
}

fn is_positive_code(code: i16) -> bool {
    code >= 0
}

fn is_final_nonempty(st: &DWAState) -> bool {
    st.final_weight.as_ref().map_or(false, |w| !w.is_empty())
}

fn has_any_positive_outgoing(st: &DWAState) -> bool {
    st.transitions.default.is_some() || st.transitions.exceptions.keys().any(|&k| is_positive_code(k))
}

// Clone a state keeping: weight, final_weight, and only negative-coded outgoing transitions (and their weights).
// Drops default and all non-negative exceptions.
fn clone_state_with_only_negative_outgoing(dwa: &mut DWA, original_id: StateID) -> StateID {
    let orig = dwa.states[original_id].clone();
    let new_id = dwa.add_state();
    let mut st = DWAState::default();
    st.weight = orig.weight.clone();
    st.final_weight = orig.final_weight.clone();
    // Keep only negative-coded exceptions (and their weights).
    for (ch, tgt) in orig.transitions.exceptions {
        if is_negative_code(ch) {
            st.transitions.exceptions.insert(ch, tgt);
            let w = orig
                .trans_weights_exceptions
                .get(&ch)
                .cloned()
                .unwrap_or_else(Weight::all);
            st.trans_weights_exceptions.insert(ch, w);
        }
    }
    // No default from the clone.
    st.transitions.default = None;
    st.trans_weight_default = None;
    dwa.states[new_id] = st;
    new_id
}

// Union a weight into an Option<Weight>. Returns true if changed.
fn union_into_option(dst: &mut Option<Weight>, inc: &Weight) -> bool {
    if inc.is_empty() {
        return false;
    }
    match dst {
        Some(w) => {
            let before = w.clone();
            *w |= inc;
            *w != before
        }
        None => {
            *dst = Some(inc.clone());
            true
        }
    }
}

// Insert an exception edge (code -> tgt) into state s_id with given weight (add_w).
// - If the exception exists and points to the same tgt: union weights.
// - If it exists but points elsewhere: conservatively union the weight into the existing edge's weight,
//   leaving the target unchanged (keeps determinism).
// - If it does not exist: insert it with the provided weight.
// Returns true if a change was made.
fn add_or_union_exception(dwa: &mut DWA, s_id: StateID, ch: i16, tgt: StateID, add_w: Weight) -> bool {
    if add_w.is_empty() {
        return false;
    }
    let st = &mut dwa.states[s_id];
    if let Some(&existing_tgt) = st.transitions.exceptions.get(&ch) {
        // Already exists.
        if existing_tgt == tgt {
            let entry = st
                .trans_weights_exceptions
                .entry(ch)
                .or_insert_with(Weight::zeros);
            let before = entry.clone();
            *entry |= &add_w;
            return *entry != before;
        } else {
            // Determinism requirement: we cannot have two different targets for the same code.
            // Conservatively union the additional weight into the existing edge's weight.
            let entry = st
                .trans_weights_exceptions
                .entry(ch)
                .or_insert_with(Weight::zeros);
            let before = entry.clone();
            *entry |= &add_w;
            return *entry != before;
        }
    } else {
        // Insert new exception.
        st.transitions.exceptions.insert(ch, tgt);
        st.trans_weights_exceptions.insert(ch, add_w);
        return true;
    }
}

// Merge src state (src_id) into dst state (dst_id), gating everything by `gate`.
// - dst.weight |= (gate & src.weight)
// - dst.final_weight |= (gate & src.final_weight)
// - For each outgoing edge e of src:
//     - If default: fold into dst.default (union weights if same target; otherwise, keep target and union weights)
//     - If exception: add_or_union_exception with weight (gate & edge_weight)
// Returns true if any change was made to dst.
fn merge_state_into(dwa: &mut DWA, dst_id: StateID, src_id: StateID, gate: &Weight) -> bool {
    if gate.is_empty() {
        return false;
    }
    let src = dwa.states[src_id].clone(); // prevent borrow conflicts
    let dst = &mut dwa.states[dst_id];
    let mut changed = false;

    // State weights
    let gated_state_w = &src.weight & gate;
    let before_w = dst.weight.clone();
    dst.weight |= &gated_state_w;
    if dst.weight != before_w {
        changed = true;
    }

    // Final weights
    if let Some(fw) = &src.final_weight {
        let gated_fw = fw & gate;
        if !gated_fw.is_empty() {
            if union_into_option(&mut dst.final_weight, &gated_fw) {
                changed = true;
            }
        }
    }

    // Default transition
    if let Some(def_tgt) = src.transitions.default {
        // Default edge weight on src's default; treat missing weight as ALL.
        let def_w = src.trans_weight_default.as_ref().cloned().unwrap_or_else(Weight::all);
        let gated_def_w = &def_w & gate;
        if !gated_def_w.is_empty() {
            if let Some(dst_def_tgt) = dst.transitions.default {
                if dst_def_tgt == def_tgt {
                    // Merge weight
                    let entry = dst
                        .trans_weight_default
                        .get_or_insert_with(Weight::zeros);
                    let before = entry.clone();
                    *entry |= &gated_def_w;
                    if *entry != before {
                        changed = true;
                    }
                } else {
                    // Keep deterministic target; merge weights conservatively.
                    let entry = dst
                        .trans_weight_default
                        .get_or_insert_with(Weight::zeros);
                    let before = entry.clone();
                    *entry |= &gated_def_w;
                    if *entry != before {
                        changed = true;
                    }
                }
            } else {
                // No default yet: set it.
                dst.transitions.default = Some(def_tgt);
                dst.trans_weight_default = Some(gated_def_w);
                changed = true;
            }
        }
    }

    // Exception transitions
    for (ch, tgt) in src.transitions.exceptions {
        // Treat missing per-edge weight as ALL
        let edge_w = src
            .trans_weights_exceptions
            .get(&ch)
            .cloned()
            .unwrap_or_else(Weight::all);
        let gated = &edge_w & gate;
        if !gated.is_empty() {
            if add_or_union_exception(dwa, dst_id, ch, tgt, gated) {
                changed = true;
            }
        }
    }

    changed
}

fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    if dwa.states.len() == 0 {
        return;
    }

    // Stage A: Preprocess negative edges that go to final nodes with positive outgoing transitions.
    // Clone a new destination with only negative outgoing edges (and final weight) and redirect.
    {
        use std::collections::BTreeMap;
        let n0 = dwa.states.len();

        // 1. Find all edges that need redirection without mutating the DWA.
        let mut edges_to_redirect: Vec<(StateID, i16, StateID)> = Vec::new();
        for sid in 0..n0 {
            let st = &dwa.states[sid];
            for (&ch, &tgt) in st.transitions.exceptions.iter() {
                if is_negative_code(ch) {
                    // Check bounds; invalid tgt could exist before prune_unreachable.
                    if tgt < n0 {
                        let dst = &dwa.states[tgt];
                        if is_final_nonempty(dst) && has_any_positive_outgoing(dst) {
                            edges_to_redirect.push((sid, ch, tgt));
                        }
                    }
                }
            }
        }

        // 2. For each unique target, clone it once. This mutates the DWA.
        let mut cloned_tgts: BTreeMap<StateID, StateID> = BTreeMap::new();
        for &(_, _, tgt) in &edges_to_redirect {
            cloned_tgts.entry(tgt).or_insert_with(|| {
                clone_state_with_only_negative_outgoing(dwa, tgt)
            });
        }

        // 3. Apply redirects using the map of cloned targets. This also mutates.
        for (sid, ch, tgt) in edges_to_redirect {
            if let Some(new_tgt) = cloned_tgts.get(&tgt) {
                if let Some(entry) = dwa.states[sid].transitions.exceptions.get_mut(&ch) {
                    *entry = *new_tgt;
                }
            }
        }
    }

    // Stage B: Internal negative cancellation/merging fixpoint.
    // For each A -(-x)-> B and B -(x)-> C, merge C into A with the weight along the two edges (meet).
    {
        let mut changed_any = true;
        let mut guard_rounds = 0usize;
        while changed_any && guard_rounds < 64 {
            changed_any = false;
            guard_rounds += 1;
            let n = dwa.states.len();
            for a_id in 0..n {
                // Collect negative edges out of a_id to avoid borrow issues.
                let neg_edges: Vec<(i16, StateID)> = dwa.states[a_id]
                    .transitions
                    .exceptions
                    .iter()
                    .filter_map(|(&ch, &b_id)| if is_negative_code(ch) { Some((ch, b_id)) } else { None })
                    .collect();
                for (neg_ch, b_id) in neg_edges {
                    let x = -neg_ch; // matching positive code
                    // Edge weights: treat missing as ALL
                    let w_neg = dwa.states[a_id]
                        .trans_weights_exceptions
                        .get(&neg_ch)
                        .cloned()
                        .unwrap_or_else(Weight::all);
                    if w_neg.is_empty() {
                        continue;
                    }
                    if let Some(&c_id) = dwa.states[b_id].transitions.exceptions.get(&x) {
                        let w_pos = dwa.states[b_id]
                            .trans_weights_exceptions
                            .get(&x)
                            .cloned()
                            .unwrap_or_else(Weight::all);
                        if w_pos.is_empty() {
                            continue;
                        }
                        let w_path = &w_neg & &w_pos;
                        if !w_path.is_empty() {
                            if merge_state_into(dwa, a_id, c_id, &w_path) {
                                changed_any = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Stage D: Remove negative edges to final states ("epsilon replacement").
    // For each A -(-x)-> B (B final), remove the edge and add (edge_weight ∧ B.final_weight) into A.final_weight.
    // This needs to be a fixpoint iteration to propagate finality backwards.
    {
        let mut changed_any = true;
        let mut guard_rounds = 0usize;
        while changed_any && guard_rounds < 64 {
            changed_any = false;
            guard_rounds += 1;

            let n = dwa.states.len();
            for a_id in 0..n {
                // Capture all negatives-to-final to process.
                let neg_to_final: Vec<(i16, StateID)> = dwa.states[a_id]
                    .transitions
                    .exceptions
                    .iter()
                    .filter_map(|(&ch, &tgt)| {
                        if is_negative_code(ch) && is_final_nonempty(&dwa.states[tgt]) {
                            Some((ch, tgt))
                        } else {
                            None
                        }
                    })
                    .collect();

                if neg_to_final.is_empty() {
                    continue;
                }

                for (ch, tgt) in neg_to_final {
                    let edge_w = dwa.states[a_id]
                        .trans_weights_exceptions
                        .get(&ch)
                        .cloned()
                        .unwrap_or_else(Weight::all);
                    if let Some(fw) = &dwa.states[tgt].final_weight {
                        let inc = &edge_w & fw;
                        if !inc.is_empty() {
                            if union_into_option(&mut dwa.states[a_id].final_weight, &inc) {
                                changed_any = true;
                            }
                        }
                    }
                    // Remove the negative edge itself.
                    dwa.states[a_id].transitions.exceptions.remove(&ch);
                    dwa.states[a_id].trans_weights_exceptions.remove(&ch);
                }
            }
        }
    }

    // Stage C: Remove any remaining negative edges. After the fixpoint above, any
    // negative edge that could lead to a final state has been processed and removed.
    // Any that remain are part of cycles or lead to non-final sinks, and can be pruned.
    {
        let n = dwa.states.len();
        for a_id in 0..n {
            let to_remove: Vec<i16> = dwa.states[a_id]
                .transitions
                .exceptions
                .iter()
                .filter_map(|(&ch, _)| if is_negative_code(ch) { Some(ch) } else { None })
                .collect();
            for ch in to_remove {
                dwa.states[a_id].transitions.exceptions.remove(&ch);
                dwa.states[a_id].trans_weights_exceptions.remove(&ch);
            }
        }
    }

    // Clean up and simplify.
    DWA::normalize_edges_inplace(&mut dwa.states);
    dwa.simplify();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precompute4::weighted_automata::{assert_dwa_equivalent, DWA, Weight};

    #[test]
    fn test_resolve_negatives_complex_cancellation() {
        let mut d = DWA::new();
        // State 0 is start
        let s1 = d.add_state();
        let s2 = d.add_state();
        let s3 = d.add_state();
        let s4 = d.add_state();
        let s5 = d.add_state();
        let s6 = d.add_state();
        let s7 = d.add_state();
        let s8 = d.add_state();
        let s9 = d.add_state();

        // State 0
        d.add_transition(0, 0, s1, Weight::from_item(1)).unwrap();
        d.add_transition(0, 1, s2, Weight::from_iter(0..=1)).unwrap();
        d.add_transition(0, 2, s3, Weight::from_item(0)).unwrap();
        d.add_transition(0, 3, s4, Weight::from_iter(0..=1)).unwrap();
        // State 1
        d.add_transition(s1, -1, s5, Weight::all()).unwrap();
        // State 2
        d.set_default_transition(s2, s6, Weight::all()).unwrap();
        // State 3
        d.add_transition(s3, -2, s7, Weight::all()).unwrap();
        // State 4 is a sink
        // State 5
        d.add_transition(s5, -1, s8, Weight::all()).unwrap();
        // State 6 is a sink
        // State 7
        d.add_transition(s7, -1, s9, Weight::all()).unwrap();
        // State 8
        d.set_final_weight(s8, Weight::all()).unwrap();
        // State 9
        d.set_final_weight(s9, Weight::all()).unwrap();

        resolve_negative_codes_in_dwa(&mut d);

        let mut expected = DWA::new(); // state 0
        let s1_exp = expected.add_state(); // state 1
        let s_final = expected.add_state(); // state 2

        // After resolution, the negative paths leading to final states effectively make
        // their predecessor states final. The final weight propagates backwards as ALL
        // because all intermediate edge weights and original final weights are ALL.
        expected.set_final_weight(s_final, Weight::all()).unwrap();

        expected.add_transition(0, 0, s_final, Weight::from_item(1)).unwrap();
        expected.add_transition(0, 2, s_final, Weight::from_item(0)).unwrap();

        assert_dwa_equivalent(d, expected);
    }
}
