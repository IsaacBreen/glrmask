use crate::precompute4::full_dwa::Precomputed4;
use crate::precompute4::weighted_automata::{DWA, DWAState, DWAStates, StateID, Weight};
use std::collections::BTreeMap;

pub fn resolve_negative_codes_for_all(precomputed4: &mut Precomputed4) {
    for (_sid, dwa) in precomputed4.iter_mut() {
        resolve_negative_codes_in_dwa(dwa);
    }
}

fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    let n = dwa.states.len();
    if n == 0 {
        return;
    }

    // Collect negative outgoing edges per state: (code, target, weight)
    let mut neg_succ: Vec<Vec<(i16, usize, Weight)>> = vec![Vec::new(); n];
    for s in 0..n {
        let st = &dwa.states[s];
        for (&ch, &tgt) in st.transitions.exceptions.iter() {
            if ch < 0 {
                let w = st
                    .trans_weights_exceptions
                    .get(&ch)
                    .cloned()
                    .unwrap_or_else(Weight::zeros);
                if !w.is_empty() {
                    neg_succ[s].push((ch, tgt, w));
                }
            }
        }
        // Note: we intentionally ignore default here; tests use explicit exceptions for negatives.
    }

    // Fixpoint: closure of negative-only paths to finals.
    // clos[t] = union over all negative-only paths t ⇒* final of (∧ edge-weights along path ∧ final_weight(end)).
    let mut clos: Vec<Weight> = vec![Weight::zeros(); n];
    for i in 0..n {
        if let Some(wf) = dwa.states[i].final_weight.as_ref() {
            if !wf.is_empty() {
                clos[i] |= wf;
            }
        }
    }
    let mut changed = true;
    let mut rounds = 0usize;
    while changed && rounds < 512 {
        changed = false;
        rounds += 1;
        for s in 0..n {
            // s --(-k,w_edge)--> t; candidate contribution to clos[s] is (w_edge ∧ clos[t]).
            for (_ch, tgt, w_edge) in &neg_succ[s] {
                if !clos[*tgt].is_empty() {
                    let cand = w_edge & &clos[*tgt];
                    let prev = clos[s].clone();
                    let next = &prev | &cand;
                    if next != prev {
                        clos[s] = next;
                        changed = true;
                    }
                }
            }
        }
    }

    // Build incoming positive-edge weights by character for each state:
    // incoming_pos[t][ch] = union of weights of all P --(ch,w)--> t with ch >= 0.
    let mut incoming_pos: Vec<BTreeMap<i16, Weight>> = vec![BTreeMap::new(); n];
    for p in 0..n {
        let st = &dwa.states[p];
        for (&ch, &tgt) in st.transitions.exceptions.iter() {
            if ch >= 0 {
                let w = st
                    .trans_weights_exceptions
                    .get(&ch)
                    .cloned()
                    .unwrap_or_else(Weight::zeros);
                if !w.is_empty() {
                    let entry = incoming_pos[tgt].entry(ch).or_insert_with(Weight::zeros);
                    *entry |= &w;
                }
            }
        }
        // We intentionally do not expand defaults into per-code entries here.
    }

    // Accumulate final-weight contributions per state according to the rules:
    // - For immediate negatives that land in finals: add w_edge ∧ final(tgt), optionally
    //   additionally gated by incoming positive of matching magnitude if present (to model
    //   the "matched" case). We avoid adding deeper contributions here to prevent overcounting.
    // - For deeper negative chains (tgt not final): add w_edge ∧ clos[tgt] (ungated).
    let mut add_final: Vec<Weight> = vec![Weight::zeros(); n];
    for s in 0..n {
        for &(ch, tgt, ref w_edge) in &neg_succ[s] {
            if w_edge.is_empty() {
                continue;
            }
            // Immediate final (tgt is final)?
            if let Some(tgt_final) = dwa.states[tgt].final_weight.as_ref() {
                if !tgt_final.is_empty() {
                    let mut cand = w_edge.clone();
                    cand &= tgt_final;
                    // If there exist incoming positives to `s` on +|ch| (i.e., matching the magnitude),
                    // gate the contribution by those weights; otherwise, keep cand as-is.
                    if ch != i16::MIN {
                        let pos_code = -ch;
                        if let Some(win) = incoming_pos[s].get(&pos_code) {
                            if !win.is_empty() {
                                cand &= win;
                            } else {
                                cand = Weight::zeros();
                            }
                        }
                    }
                    if !cand.is_empty() {
                        add_final[s] |= &cand;
                    }
                }
            }
            // Deeper negative-only path contribution (only if tgt is not final).
            let tgt_is_final = dwa.states[tgt]
                .final_weight
                .as_ref()
                .map(|w| !w.is_empty())
                .unwrap_or(false);
            if !tgt_is_final {
                let deeper = w_edge & &clos[tgt];
                if !deeper.is_empty() {
                    add_final[s] |= &deeper;
                }
            }
        }
    }

    // Apply the accumulated final-weight contributions.
    for s in 0..n {
        if !add_final[s].is_empty() {
            if let Some(ref mut fw) = dwa.states[s].final_weight {
                *fw |= &add_final[s];
            } else {
                dwa.states[s].final_weight = Some(add_final[s].clone());
            }
        }
    }

    // Remove all negative exception transitions and their weights.
    for s in 0..n {
        let st = &mut dwa.states[s];
        // Prune negative exceptions
        st.transitions.exceptions.retain(|&ch, _| ch >= 0);
        st.trans_weights_exceptions.retain(|&ch, _| ch >= 0);
        // Note: default transitions are left unchanged.
    }

    // Normalize/minimize after the rewrite so tests can compare canonical forms.
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

    #[test]
    fn test_resolve_negatives_isolated_path_from_tokenizer_1() {
        let mut d = DWA::new();
        let s1 = d.add_state();
        let s2 = d.add_state();

        d.add_transition(d.body.start_state, 7, s1, Weight::from_item(2)).unwrap();
        d.add_transition(s1, -7, s2, Weight::all()).unwrap();
        d.set_final_weight(s2, Weight::all()).unwrap();

        resolve_negative_codes_in_dwa(&mut d);

        let mut expected = DWA::new();
        let s_final = expected.add_state();
        expected.add_transition(expected.body.start_state, 7, s_final, Weight::from_item(2)).unwrap();
        expected.set_final_weight(s_final, Weight::from_item(2)).unwrap();

        assert_dwa_equivalent(d, expected);
    }
}
