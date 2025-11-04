use crate::precompute4::full_dwa::Precomputed4;
use crate::precompute4::weighted_automata::{DWA, DWAState, DWAStates, StateID, Weight};
use std::collections::BTreeMap;

pub fn resolve_negative_codes_for_all(precomputed4: &mut Precomputed4) {
    for (_sid, dwa) in precomputed4.iter_mut() {
        resolve_negative_codes_in_dwa(dwa);
    }
}

fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    if dwa.states.is_empty() {
        return;
    }

    // 1. Build reverse adjacency list for positive edges to find potential cancellations.
    let mut pos_rev_adj: Vec<Vec<(i16, Weight)>> = vec![vec![]; dwa.states.len()];
    for (from_id, from_state) in dwa.states.0.iter().enumerate() {
        for (&ch, &to_id) in &from_state.transitions.exceptions {
            if ch >= 0 && to_id < dwa.states.len() {
                let weight = from_state
                    .trans_weights_exceptions
                    .get(&ch)
                    .cloned()
                    .unwrap_or_else(Weight::all);
                pos_rev_adj[to_id].push((ch, weight));
            }
        }
    }

    // 2. Propagate finality backwards over negative edges until a fixed point is reached.
    loop {
        let mut changed = false;
        let current_final_weights = dwa
            .states
            .0
            .iter()
            .map(|s| s.final_weight.clone().unwrap_or_else(Weight::zeros))
            .collect::<Vec<_>>();

        for s_id in 0..dwa.states.len() {
            let s = &dwa.states[s_id].clone(); // Clone to avoid borrow checker issues
            for (&neg_code, &b_id) in s.transitions.exceptions.iter().filter(|(&ch, _)| ch < 0) {
                if b_id >= dwa.states.len() {
                    continue;
                }

                let b_final_weight = &current_final_weights[b_id];
                if b_final_weight.is_empty() {
                    continue;
                }

                let w_neg = s
                    .trans_weights_exceptions
                    .get(&neg_code)
                    .cloned()
                    .unwrap_or_else(Weight::all);

                let pos_code = if neg_code != i16::MIN { -neg_code } else { i16::MAX };

                let mut incoming_pos_weights = Weight::zeros();
                let mut has_matching_pos_edge = false;
                for &(ch, ref w_pos) in &pos_rev_adj[s_id] {
                    if ch == pos_code {
                        incoming_pos_weights |= w_pos;
                        has_matching_pos_edge = true;
                    }
                }

                let w_prop = if has_matching_pos_edge {
                    b_final_weight & &w_neg & &incoming_pos_weights
                } else {
                    b_final_weight & &w_neg
                };

                if !w_prop.is_empty() {
                    let old_fw = &dwa.states[s_id].final_weight.clone().unwrap_or_else(Weight::zeros);
                    let mut new_fw = old_fw.clone();
                    new_fw |= &w_prop;
                    if &new_fw != old_fw {
                        dwa.states[s_id].final_weight = Some(new_fw);
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    // 4. Remove all negative transitions.
    for state in &mut dwa.states.0 {
        state.transitions.exceptions.retain(|&ch, _| ch >= 0);
        state.trans_weights_exceptions.retain(|&ch, _| ch >= 0);
    }

    // 5. Simplify to remove newly unreachable/redundant states.
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
        d.add_transition(s1, i16::MIN + 1, s5, Weight::all()).unwrap();
        // State 2
        d.set_default_transition(s2, s6, Weight::all()).unwrap();
        // State 3
        d.add_transition(s3, i16::MIN + 2, s7, Weight::all()).unwrap();
        // State 4 is a sink
        // State 5
        d.add_transition(s5, i16::MIN + 1, s8, Weight::all()).unwrap();
        // State 6 is a sink
        // State 7
        d.add_transition(s7, i16::MIN + 1, s9, Weight::all()).unwrap();
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
        d.add_transition(s1, i16::MIN + 7, s2, Weight::all()).unwrap();
        d.set_final_weight(s2, Weight::all()).unwrap();

        resolve_negative_codes_in_dwa(&mut d);

        let mut expected = DWA::new();
        let s_final = expected.add_state();
        expected.add_transition(expected.body.start_state, 7, s_final, Weight::from_item(2)).unwrap();
        expected.set_final_weight(s_final, Weight::from_item(2)).unwrap();

        assert_dwa_equivalent(d, expected);
    }
}
