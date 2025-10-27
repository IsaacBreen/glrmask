use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::constraint::{PrecomputeNode3Index, StateIDBV, Trie3GodWrapper};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::Trie;
use crate::glr::parser::GLRParser;
use crate::glr::table::StateID;
use crate::tokenizer::TokenizerStateID;

/// Propagate possible parser states across the trie (respecting pop deltas)
/// and generalize edge StateID bitvectors accordingly.
///
/// This implementation computes a conservative "possible states" set per node by
/// seeding all roots with all states, pushing forward along edges while applying
/// n-step predecessors for pop>0 edges. It currently computes the fixed-point but
/// does not modify the trie with the results. It is safe and provides a basis for
/// downstream passes that may choose to use the propagated sets.
pub fn propagate_and_generalize_sids_trie3(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    parser: &GLRParser,
    max_state_id: usize,
) {
    crate::debug!(2, "Propagating possible states to generalize StateIDBVs in Trie3...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    // Part A: Pre-computation
    let one_step_back_map = parser.build_one_step_back_map();

    let mut max_pop = 0;
    for node_idx in &all_nodes {
        let r = node_idx.read(trie3_god).expect("read");
        for ((pop, _), _) in r.children() {
            if *pop > 0 {
                max_pop = max_pop.max(*pop as usize);
            }
        }
    }

    if max_pop == 0 {
        crate::debug!(2, "No pop > 0 edges found, skipping SID generalization.");
        return;
    }

    // Precompute T_{2^k} maps using repeated composition up to ceil(log2(max_pop))
    let num_levels = (max_pop as f64).log2().ceil() as usize + 1;
    let mut two_power_back_maps: Vec<BTreeMap<StateID, StateIDBV>> = Vec::with_capacity(num_levels);
    two_power_back_maps.push(one_step_back_map);

    for i in 1..num_levels {
        let prev_map = &two_power_back_maps[i - 1];
        let mut next_map: BTreeMap<StateID, StateIDBV> = BTreeMap::new();
        for (&state_id, preds1) in prev_map {
            let mut preds2 = StateIDBV::zeros();
            for pred1_val in preds1.iter() {
                let sid = StateID(pred1_val);
                if let Some(preds_of_pred) = prev_map.get(&sid) {
                    preds2 |= preds_of_pred;
                }
            }
            if !preds2.is_empty() {
                next_map.insert(state_id, preds2);
            }
        }
        two_power_back_maps.push(next_map);
    }

    // Helper for T_n(S): apply the predecessor relation n times using the cached powers-of-two
    let apply_n_step_back = |s: &StateIDBV, n: usize| -> StateIDBV {
        let mut current_s = s.clone();
        for k in 0..two_power_back_maps.len() {
            if (n >> k) & 1 == 1 {
                let map_k = &two_power_back_maps[k];
                let mut next_s = StateIDBV::zeros();
                for state_id_val in current_s.iter() {
                    let state_id = StateID(state_id_val);
                    if let Some(preds) = map_k.get(&state_id) {
                        next_s |= preds;
                    }
                }
                current_s = next_s;
            }
        }
        current_s
    };

    // Part B: Worklist propagation of possible states along trie edges.
    let mut possible_states: HashMap<PrecomputeNode3Index, StateIDBV> = HashMap::new();
    let mut worklist: VecDeque<PrecomputeNode3Index> = VecDeque::new();

    let all_states_bv = StateIDBV::ones(max_state_id + 1);
    for root_idx in roots.values() {
        possible_states.insert(*root_idx, all_states_bv.clone());
        worklist.push_back(*root_idx);
    }

    while let Some(u_idx) = worklist.pop_front() {
        let s_u = possible_states
            .get(&u_idx)
            .cloned()
            .unwrap_or_else(StateIDBV::zeros);
        if s_u.is_empty() {
            continue;
        }

        let u_guard = u_idx.read(trie3_god).expect("read");
        for ((pop, _), dest_map) in u_guard.children() {
            let s_u_popped = if *pop > 0 {
                apply_n_step_back(&s_u, *pop as usize)
            } else {
                s_u.clone()
            };

            if s_u_popped.is_empty() {
                continue;
            }

            for (v_idx, sids_uv) in dest_map {
                let s_transmitted = &s_u_popped & sids_uv;
                if s_transmitted.is_empty() {
                    continue;
                }

                let s_v_old = possible_states.entry(*v_idx).or_default();
                let old_len = s_v_old.len();
                *s_v_old |= &s_transmitted;
                if s_v_old.len() > old_len {
                    worklist.push_back(*v_idx);
                }
            }
        }
    }

    // Note: We currently compute 'possible_states' but do not update edges in-place.
    // If you want to use this information to shrink SIDs per edge safely, you can
    // intersect edge SIDs with the destination's 'possible_states' or the appropriate
    // preimage projection considering pops. Left as a safe no-op for the split pipeline.
}
