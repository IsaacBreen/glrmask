use crate::precompute4::full_dwa::Precomputed4;
use crate::precompute4::weighted_automata::{DWA, DWAState, DWAStates, StateID, Weight};
use std::collections::BTreeMap;

pub fn resolve_negative_codes_for_all(precomputed4: &mut Precomputed4) {
    for (_sid, dwa) in precomputed4.iter_mut() {
        resolve_negative_codes_in_dwa(dwa);
    }
}

fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    crate::debug!(5, "Initial DWA:\n{}", dwa);
    loop {
        let mut changed_in_pass = false;
        let mut all_new_states = Vec::with_capacity(dwa.states.len());

        for state_id in 0..dwa.states.len() {
            let (new_state, changed) =
                resolve_negative_codes_in_dwa_internal(state_id, &dwa.states);
            if changed {
                changed_in_pass = true;
            }
            all_new_states.push(new_state);
        }

        if !changed_in_pass {
            break;
        }

        dwa.states = DWAStates(all_new_states);
    }
    dwa.simplify();
    crate::debug!(5, "Resolved DWA:\n{}", dwa);
}

fn resolve_negative_codes_in_dwa_internal(
    state_id: StateID,
    states: &DWAStates,
) -> (DWAState, bool) {
    let orig = &states[state_id];

    // Create a fresh state:
    // - preserve state weight and existing final_weight
    // - copy default transition (and weight)
    // - copy only positive-code exception transitions (and their weights)
    let mut new_state = DWAState::default();
    new_state.weight = orig.weight.clone();
    new_state.final_weight = orig.final_weight.clone();
    new_state.transitions.default = orig.transitions.default;
    new_state.trans_weight_default = orig.trans_weight_default.clone();

    for (&ch, &tgt) in orig.transitions.exceptions.iter() {
        if ch >= 0 {
            new_state.transitions.exceptions.insert(ch, tgt);
            if let Some(w) = orig.trans_weights_exceptions.get(&ch) {
                new_state.trans_weights_exceptions.insert(ch, w.clone());
            }
        }
    }

    // Gather negative exception edges to process.
    let mut neg_edges: Vec<(i16, StateID)> = Vec::new();
    for (&ch, &tgt) in orig.transitions.exceptions.iter() {
        if ch < 0 {
            neg_edges.push((ch, tgt));
        }
    }

    for (neg_ch, b_id) in neg_edges {
        let edge_w = orig
            .trans_weights_exceptions
            .get(&neg_ch)
            .cloned()
            .unwrap_or_else(Weight::zeros);

        let b_state = &states[b_id];

        // 1) Pull back finality across the negative edge.
        if let Some(b_fw) = b_state.final_weight.clone() {
            let mut pulled = b_fw;
            pulled &= &edge_w;
            if !pulled.is_empty() {
                if let Some(ref mut s_fw) = new_state.final_weight {
                    *s_fw |= &pulled;
                } else {
                    new_state.final_weight = Some(pulled);
                }
            }
        }

        // 2) Try to cancel with a matching positive edge from B on the decoded positive code.
        //    Negative codes are encoded as i16::MIN + code; decode with wrapping_sub.
        let pos_ch: i16 = neg_ch.wrapping_sub(i16::MIN);
        let (c_opt, w_pos_opt) = if let Some(&cid) = b_state.transitions.exceptions.get(&pos_ch) {
            let w_pos = b_state
                .trans_weights_exceptions
                .get(&pos_ch)
                .cloned()
                .unwrap_or_else(Weight::zeros);
            (Some(cid), Some(w_pos))
        } else if let Some(cid) = b_state.transitions.default {
            let w_pos = b_state
                .trans_weight_default
                .clone()
                .unwrap_or_else(Weight::zeros);
            (Some(cid), Some(w_pos))
        } else {
            (None, None)
        };

        if let (Some(c_id), Some(w_pos)) = (c_opt, w_pos_opt) {
            // Compose the two edge weights (neg followed by matching pos).
            let gate = &edge_w & &w_pos;
            let mut c_copy = states[c_id].clone();
            c_copy.apply_weight(&gate);

            // Merge C's behavior into the current state's new version.
            // Shallow merge (preserving previous determinism assumption). If targets conflict,
            // we assert as before. The non-panicking, conflict-resolving merge lives in
            // DWA::union_into_state, which requires a mutable states arena.
            //
            // 1) Union state and final weights
            new_state.weight |= &c_copy.weight;
            if let Some(rfw) = &c_copy.final_weight {
                if let Some(lfw) = &mut new_state.final_weight {
                    *lfw |= rfw;
                } else {
                    new_state.final_weight = Some(rfw.clone());
                }
            }
            // 2) Merge default transitions and weights
            if let Some(rd) = c_copy.transitions.default {
                if let Some(ld) = new_state.transitions.default {
                    assert!(
                        ld == rd,
                        "Cannot merge negative-resolution results with conflicting default transitions"
                    );
                } else {
                    new_state.transitions.default = Some(rd);
                }
            }
            if let Some(rdw) = &c_copy.trans_weight_default {
                if let Some(ldw) = &mut new_state.trans_weight_default {
                    *ldw |= rdw;
                } else {
                    new_state.trans_weight_default = Some(rdw.clone());
                }
            }
            // 3) Merge exception transitions and weights
            for (&ch, &rt) in c_copy.transitions.exceptions.iter() {
                if let Some(&lt) = new_state.transitions.exceptions.get(&ch) {
                    assert!(lt == rt, "Cannot merge negative-resolution results with conflicting exception transitions on char {}", ch);
                } else {
                    new_state.transitions.exceptions.insert(ch, rt);
                }
                let rw = c_copy.trans_weights_exceptions.get(&ch).cloned().unwrap_or_else(Weight::zeros);
                new_state.trans_weights_exceptions.entry(ch).and_modify(|w| *w |= &rw).or_insert(rw);
            }
        } else {
            // No positive match. Keep the negative edge only if B is not final.
            let b_is_final = b_state
                .final_weight
                .as_ref()
                .map_or(false, |w| !w.is_empty());
            if !b_is_final {
                new_state.transitions.exceptions.insert(neg_ch, b_id);
                new_state.trans_weights_exceptions.insert(neg_ch, edge_w);
            }
        }
    }

    let changed = &new_state != orig;
    (new_state, changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precompute4::weighted_automata::{assert_dwa_equivalent, DWA, Weight};

    #[test]
    fn test_resolve_negatives_simple_cancellation() {
        // Corresponds to a sequence like `a, neg(a)`.
        // The DWA should resolve to an automaton that accepts `a` and is final.
        let mut d = DWA::new();
        let s1 = d.add_state();
        let s2 = d.add_state();

        let code_a = 7;
        let neg_code_a = i16::MIN + code_a;

        // 0 --a--> 1
        d.add_transition(d.body.start_state, code_a, s1, Weight::from_item(2)).unwrap();
        // 1 --neg(a)--> 2
        d.add_transition(s1, neg_code_a, s2, Weight::all()).unwrap();
        // 2 is final
        d.set_final_weight(s2, Weight::all()).unwrap();

        resolve_negative_codes_in_dwa(&mut d);

        // Expected: 0 --a--> final_state
        let mut expected = DWA::new();
        let s_final = expected.add_state();
        expected.add_transition(expected.body.start_state, code_a, s_final, Weight::from_item(2)).unwrap();
        // The final weight is the intersection of the edge weight (2) and the final weight (all),
        // constrained by reachability during simplification.
        expected.set_final_weight(s_final, Weight::from_item(2)).unwrap();

        assert_dwa_equivalent(d, expected);
    }

    #[test]
    fn test_resolve_negatives_long_cancellation_chain() {
        // This test models a path from a real-world DWA that involves multiple cancellations.
        // The path is: 0 --7--> 5 --neg(7)--> 10 --neg(3)--> 13 --3--> 15 --7--> 22 --neg(7)--> 27 --neg(1)--> 29 --neg(2)--> 30(final)
        let mut d = DWA::new();
        let s5 = d.add_state();
        let s10 = d.add_state();
        let s13 = d.add_state();
        let s15 = d.add_state();
        let s22 = d.add_state();
        let s27 = d.add_state();
        let s29 = d.add_state();
        let s30 = d.add_state();

        let code7 = 7;
        let neg_code7 = i16::MIN + code7;
        let code3 = 3;
        let neg_code3 = i16::MIN + code3;
        let code1 = 1;
        let neg_code1 = i16::MIN + code1;
        let code2 = 2;
        let neg_code2 = i16::MIN + code2;

        d.add_transition(d.body.start_state, code7, s5, Weight::all()).unwrap();
        d.add_transition(s5, neg_code7, s10, Weight::all()).unwrap();
        d.add_transition(s10, neg_code3, s13, Weight::all()).unwrap();
        d.add_transition(s13, code3, s15, Weight::all()).unwrap();
        d.add_transition(s15, code7, s22, Weight::all()).unwrap();
        d.add_transition(s22, neg_code7, s27, Weight::all()).unwrap();
        d.add_transition(s27, neg_code1, s29, Weight::all()).unwrap();
        d.add_transition(s29, neg_code2, s30, Weight::all()).unwrap();
        d.set_final_weight(s30, Weight::all()).unwrap();

        resolve_negative_codes_in_dwa(&mut d);

        // After resolution and simplification, the chain of cancellations should result
        // in an automaton that accepts the "7".
        let mut expected = DWA::new();
        let s_final = expected.add_state();
        expected.add_transition(expected.body.start_state, code7, s_final, Weight::all()).unwrap();
        expected.set_final_weight(s_final, Weight::all()).unwrap();

        assert_dwa_equivalent(d, expected);
    }
}
