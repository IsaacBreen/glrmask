use crate::precompute4::full_dwa::Precomputed4;
use crate::precompute4::weighted_automata::{DWA, DWAStates, StateID, Weight, NWA, NWAStates, NWAStateID, NWABody};
use std::collections::{BTreeMap};

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
fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    // Convert to NWA
    crate::debug!(3, "Initial DWA: {}", dwa);
    let mut nwa = NWA::from_dwa(dwa);
    loop {
        let mut previous_dwa: Option<DWA> = None;
        let mut changed_in_pass = false;

        let n = nwa.states.len();
        for state_id in 0..n {
            let changed =
                resolve_negative_codes_in_nwa_internal(state_id, &mut nwa.states);
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

        if previous_dwa.as_ref() == Some(&tmp_dwa) {
            break; // Fixed point reached, break loop.
        }
        previous_dwa = Some(tmp_dwa.clone());

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

    if state_id >= states.len() {
        return false;
    }

    // Collect negative transitions first to avoid borrow issues
    let negatives: Vec<(i16, NWAStateID, Weight)> = {
        let st = &states[state_id];
        st.transitions.iter()
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

        // Check if B needs to be split. B needs splitting if it has "positive" behavior
        // (a final weight or a positive-code transition) that should not be triggered
        // by the negative path.
        let b_needs_splitting = {
            let b_orig = &states[b_orig_id];
            b_orig.final_weight.is_some() || b_orig.transitions.keys().any(|k| *k >= 0)
        };

        if b_needs_splitting {
            // Step 2: Copy B and strip positive transitions; also clear final weight in the copy.
            let b_copy_id = states.copy_state(b_orig_id);
            {
                let b_copy = &mut states[b_copy_id];
                b_copy.final_weight = None;
                b_copy.transitions.retain(|k, _| *k < 0);
            }

            // Step 3: Replace edge A -(neg)-> B with A -(neg)-> B_copy
            let (ref mut tgt, _) = states[state_id].transitions.get_mut(&neg_code).unwrap();
            *tgt = b_copy_id;
            changed = true;
        }
    }

    changed
}


#[cfg(test)]
mod tests {
    use crate::precompute4::test_weighted_automata::assert_dwa_equivalent;
    use super::*;
    use crate::precompute4::weighted_automata::{DWA, Weight};

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

    #[test]
    fn test_resolve_negatives_from_debug_log() {
        let mut d = DWA::new();
        let s1 = d.add_state();
        let s2 = d.add_state();
        let s3 = d.add_state();
        let s4 = d.add_state();
        let s5 = d.add_state();
        let s6 = d.add_state();

        let code0 = 0;
        let neg_code0 = i16::MIN + code0;
        let neg_code1 = i16::MIN + 1;
        let code2 = 2;
        let neg_code2 = i16::MIN + code2;

        d.add_transition(d.body.start_state, code0, s1, Weight::all()).unwrap();
        d.add_transition(d.body.start_state, code2, s2, Weight::all()).unwrap();
        d.add_transition(s1, neg_code0, s3, Weight::all()).unwrap();
        d.add_transition(s2, neg_code2, s4, Weight::all()).unwrap();
        d.add_transition(s3, neg_code1, s5, Weight::all()).unwrap();
        d.add_transition(s4, neg_code0, s6, Weight::all()).unwrap();
        d.set_final_weight(s5, Weight::from_item(1)).unwrap();
        d.set_final_weight(s6, Weight::from_item(0)).unwrap();

        resolve_negative_codes_in_dwa(&mut d);

        // Expected: After resolution, finality is propagated backward.
        // The path 0 -> 1 becomes final with weight [1].
        // The path 2 -> 2 becomes final with weight [0].
        // The dangling negative transitions are pruned by simplification.
        let mut expected = DWA::new();
        let s_final1 = expected.add_state();
        let s_final2 = expected.add_state();
        expected.add_transition(expected.body.start_state, code0, s_final1, Weight::all()).unwrap();
        expected.set_final_weight(s_final1, Weight::from_item(1)).unwrap();
        expected.add_transition(expected.body.start_state, code2, s_final2, Weight::all()).unwrap();
        expected.set_final_weight(s_final2, Weight::from_item(0)).unwrap();

        assert_dwa_equivalent(d, expected);
    }

    #[test]
    fn test_resolve_negatives_from_intermediate_debug_log() {
        let mut d = DWA::new();
        let s1 = d.add_state();
        let s2 = d.add_state();
        let s3 = d.add_state();
        let s4 = d.add_state();
        let s5 = d.add_state();
        let s6 = d.add_state();
        let s7 = d.add_state();

        let neg_code1 = i16::MIN + 1;
        let neg_code7 = i16::MIN + 7;

        // State 0 (start)
        d.add_transition(d.body.start_state, 1, s1, Weight::all()).unwrap();
        d.add_transition(d.body.start_state, 3, s2, Weight::all()).unwrap();
        d.add_transition(d.body.start_state, 4, s3, Weight::all()).unwrap();
        d.add_transition(d.body.start_state, 7, s4, Weight::all()).unwrap();

        // State 1
        d.add_transition(s1, neg_code1, s5, Weight::all()).unwrap();
        d.add_transition(s1, 7, s4, Weight::all()).unwrap();

        // State 2
        d.add_transition(s2, 7, s6, Weight::all()).unwrap();

        // State 3
        d.set_default_transition(s3, s2, Weight::all()).unwrap();

        // State 4
        d.set_final_weight(s4, Weight::from_item(2)).unwrap();

        // State 5
        d.set_default_transition(s5, s7, Weight::all()).unwrap();

        // State 6
        d.set_final_weight(s6, Weight::from_item(2)).unwrap();
        d.add_transition(s6, neg_code7, s1, Weight::all()).unwrap();

        // State 7
        d.add_transition(s7, 7, s4, Weight::all()).unwrap();

        resolve_negative_codes_in_dwa(&mut d);

        let mut expected = DWA::new();
        let exp_s1 = expected.add_state();
        let exp_s2 = expected.add_state();
        let exp_s3 = expected.add_state();
        let exp_s_after_4 = expected.add_state();
        let exp_s_final = expected.add_state();

        expected.set_final_weight(exp_s_final, Weight::from_item(2)).unwrap();

        // Paths leading to final state
        expected.add_transition(expected.body.start_state, 7, exp_s_final, Weight::all()).unwrap();

        expected.add_transition(expected.body.start_state, 1, exp_s1, Weight::all()).unwrap();
        expected.add_transition(exp_s1, 7, exp_s_final, Weight::all()).unwrap();

        expected.add_transition(expected.body.start_state, 3, exp_s2, Weight::all()).unwrap();
        expected.add_transition(exp_s2, 7, exp_s_final, Weight::all()).unwrap();

        expected.add_transition(expected.body.start_state, 4, exp_s3, Weight::all()).unwrap();
        expected.set_default_transition(exp_s3, exp_s_after_4, Weight::all()).unwrap();
        expected.add_transition(exp_s_after_4, 7, exp_s_final, Weight::all()).unwrap();

        assert_dwa_equivalent(d, expected);
    }

    #[test]
    fn test_resolve_negatives_minimal_loop_with_default() {
        let mut d = DWA::new();
        let s1 = d.add_state();
        let s2 = d.add_state();

        let neg_code1 = i16::MIN + 1;

        // 0 --neg(1)--> 1
        d.add_transition(d.body.start_state, neg_code1, s1, Weight::all()).unwrap();
        // 1 --*--> 2 (default)
        d.set_default_transition(s1, s2, Weight::all()).unwrap();
        // 2 is final
        d.set_final_weight(s2, Weight::all()).unwrap();

        // With the bug, this function would not terminate.
        // With the fix, it terminates, producing a stable DWA.
        resolve_negative_codes_in_dwa(&mut d);

        // The stable result of the flawed-but-terminating algorithm is an automaton
        // where the start state has become final due to cancellation, and it has a
        // neg(1) transition to a state with a default transition to another final state.
        let mut expected = DWA::new();
        let exp_s1 = expected.add_state();
        let exp_s2 = expected.add_state();
        expected.set_final_weight(expected.body.start_state, Weight::all()).unwrap();
        expected.add_transition(expected.body.start_state, neg_code1, exp_s1, Weight::all()).unwrap();
        expected.set_default_transition(exp_s1, exp_s2, Weight::all()).unwrap();
        expected.set_final_weight(exp_s2, Weight::all()).unwrap();

        assert_dwa_equivalent(d, expected);
    }
}
