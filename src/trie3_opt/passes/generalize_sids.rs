use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::glr::{parser::GLRParser, table::StateID};
use crate::trie3_opt::{
    context::OptimizationContext,
    core::{MiniTrie, NodeId, SortedSet},
    passes::OptimizationPass,
};

pub struct GeneralizeSidsPass;

impl OptimizationPass for GeneralizeSidsPass {
    fn name(&self) -> &'static str {
        "GeneralizeSids"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
        let parser = if let Some(p) = &ctx.parser {
            p.borrow()
        } else {
            return;
        };

        let one_step_back_map = parser.build_one_step_back_map();
        let mut max_pop = 0;
        for node in &trie.nodes {
            for (ek, _) in &node.children {
                if ek.pop > 0 {
                    max_pop = max_pop.max(ek.pop as usize);
                }
            }
        }
        if max_pop == 0 {
            return;
        }

        let num_levels = (max_pop as f64).log2().ceil() as usize + 1;
        let mut two_power_back_maps: Vec<BTreeMap<StateID, SortedSet>> =
            Vec::with_capacity(num_levels);
        two_power_back_maps.push(
            one_step_back_map
                .into_iter()
                .map(|(k, v)| (k, SortedSet::from_iter(v.iter())))
                .collect(),
        );

        for i in 1..num_levels {
            let prev_map = &two_power_back_maps[i - 1];
            let mut next_map: BTreeMap<StateID, SortedSet> = BTreeMap::new();
            for (&state_id, preds1) in prev_map {
                let mut preds2 = SortedSet::new();
                for pred1_val in preds1.iter() {
                    let sid = StateID(pred1_val);
                    if let Some(preds_of_pred) = prev_map.get(&sid) {
                        preds2.union_inplace(preds_of_pred);
                    }
                }
                if !preds2.is_empty() {
                    next_map.insert(state_id, preds2);
                }
            }
            two_power_back_maps.push(next_map);
        }

        let apply_n_step_back = |s: &SortedSet, n: usize| -> SortedSet {
            let mut current_s = s.clone();
            for k in 0..two_power_back_maps.len() {
                if (n >> k) & 1 == 1 {
                    let map_k = &two_power_back_maps[k];
                    let mut next_s = SortedSet::new();
                    for state_id_val in current_s.iter() {
                        let state_id = StateID(state_id_val);
                        if let Some(preds) = map_k.get(&state_id) {
                            next_s.union_inplace(preds);
                        }
                    }
                    current_s = next_s;
                }
            }
            current_s
        };

        let mut possible_states: HashMap<NodeId, SortedSet> = HashMap::new();
        let mut worklist: VecDeque<NodeId> = VecDeque::new();
        let all_states = SortedSet::from_iter(0..=ctx.max_state_id);

        for root_id in &trie.root_ids {
            possible_states.insert(*root_id, all_states.clone());
            worklist.push_back(*root_id);
        }

        while let Some(u_id) = worklist.pop_front() {
            let s_u = possible_states.get(&u_id).cloned().unwrap_or_default();
            if s_u.is_empty() {
                continue;
            }
            let u_node = &trie.nodes[u_id as usize];
            for (ek, dm) in &u_node.children {
                let s_u_popped = if ek.pop > 0 {
                    apply_n_step_back(&s_u, ek.pop as usize)
                } else {
                    s_u.clone()
                };
                if s_u_popped.is_empty() {
                    continue;
                }
                for (v_id, sids_uv) in dm {
                    let s_transmitted = s_u_popped.intersect(sids_uv);
                    if s_transmitted.is_empty() {
                        continue;
                    }
                    let s_v_old = possible_states.entry(*v_id).or_default();
                    let old_len = s_v_old.len();
                    s_v_old.union_inplace(&s_transmitted);
                    if s_v_old.len() > old_len {
                        worklist.push_back(*v_id);
                    }
                }
            }
        }

        for node in &mut trie.nodes {
            for (_ek, dm) in &mut node.children {
                dm.retain(|v_id, sids| {
                    if let Some(s_v) = possible_states.get(v_id) {
                        let intersection = sids.intersect(s_v);
                        if !intersection.is_empty() {
                            *sids = intersection;
                            true
                        } else {
                            false
                        }
                    } else {
                        // v is not reachable with any valid state, so this edge is dead.
                        false
                    }
                });
            }
            node.children.retain(|_, dm| !dm.is_empty());
        }
    }
}
