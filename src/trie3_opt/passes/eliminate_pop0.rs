use std::collections::{BTreeMap, HashMap};

use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;
use crate::glr::table::StateID;

/// Eliminate pop=0 edges whose source is not a root by composing predecessors:
/// For each B --(0, L_bc, S_bc)--> C and each A --(p_ab, L_ab, S_ab)--> B,
/// add A --(p_ab, L_ab∧L_bc, S_ab∧pre^{p_ab}(S_bc))--> C and remove the B->C entry.
/// Here pre^{p}(S) is the p-step state preimage under parser "pop" transitions.
/// Iterate to a fixed point. This is semantics-preserving under the model where a path's
/// token set is the intersection across edges, and the admissible source states for a path
/// are the intersection across edges of preimages of the edge state-sets by the cumulative pop.
pub struct EliminatePop0ExceptRootsPass;

impl OptimizationPass for EliminatePop0ExceptRootsPass {
    fn name(&self) -> &'static str {
        "EliminatePop0ExceptRoots"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
        let root_set = trie.root_ids.clone();
        let n = trie.num_nodes();
        if n == 0 {
            return;
        }

        // Compute the maximum pop seen. If > 0, we need the parser to compute state preimages.
        let mut max_pop = 0usize;
        for node in trie.nodes() {
            for (ek, _) in node.children() {
                if ek.pop > 0 {
                    max_pop = max_pop.max(ek.pop as usize);
                }
            }
        }

        // Build 2^k preimage tables (as in GeneralizeSidsPass) if needed.
        let mut two_power_back_maps: Vec<BTreeMap<StateID, SortedSet>> = Vec::new();
        if max_pop > 0 {
            let parser_ref = if let Some(p) = &ctx.parser {
                p.borrow()
            } else {
                // Without a parser we cannot correctly compose states when p_ab > 0.
                // To avoid unsound rewrites, skip the pass.
                return;
            };
            let one_step_back_map = parser_ref.build_one_step_back_map();
            let num_levels = (max_pop as f64).log2().ceil() as usize + 1;
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
        }

        // Closure to apply n-step preimage. If n == 0 (or we have no tables), it's identity.
        let apply_n_step_back = |s: &SortedSet, n_steps: usize| -> SortedSet {
            if n_steps == 0 || two_power_back_maps.is_empty() {
                return s.clone();
            }
            let mut current_s = s.clone();
            for k in 0..two_power_back_maps.len() {
                if ((n_steps >> k) & 1) == 1 {
                    let map_k = &two_power_back_maps[k];
                    let mut next_s = SortedSet::new();
                    for state_id_val in current_s.iter() {
                        let sid = StateID(state_id_val);
                        if let Some(preds) = map_k.get(&sid) {
                            next_s.union_inplace(preds);
                        }
                    }
                    current_s = next_s;
                }
            }
            current_s
        };

        loop {
            // Collect pop=0 edges by cloning to avoid borrow issues.
            let mut zero_edges: Vec<(NodeId, SortedSet, NodeId, SortedSet)> = Vec::new();

            for node in trie.nodes() {
                for (ek, dm) in node.children() {
                    if ek.pop == 0 {
                        for (dst, sids) in dm {
                            zero_edges.push((node.id(), ek.tokens.clone(), *dst, sids.clone()));
                        }
                    }
                }
            }

            if zero_edges.is_empty() { break; }

            let mut removed_this_iter = 0usize;

            for (b, llm_bc, c, s_bc) in zero_edges {
                if root_set.contains(&b) { continue; }

                if let Some(b_node) = trie.get_node(b) {
                    let preds = b_node.parents().clone(); // clone to avoid borrow issues
                    for (a, edges_from_a) in &preds {
                        for (key_ab, s_ab) in edges_from_a {
                            let p_ab = key_ab.pop;
                            let llm_ab = &key_ab.tokens;
                            let new_tokens = llm_ab.intersect(&llm_bc);
                            if new_tokens.is_empty() { continue; }
                            let p_usize = if p_ab <= 0 { 0usize } else { p_ab as usize };
                            // Compose states with the correct p-step preimage.
                            // new_sids = S_ab ∩ pre^{p_ab}(S_bc)
                            let s_bc_pre = apply_n_step_back(&s_bc, p_usize);
                            let new_sids = s_ab.intersect(&s_bc_pre);
                            if new_sids.is_empty() { continue; }

                            let key = crate::trie3_opt::core::EdgeKey::new(p_ab, new_tokens.clone());
                            trie.add_edge(*a, key, c, new_sids);
                        }
                    }
                }

                // Remove B -> C under (0, llm_bc)
                let key = crate::trie3_opt::core::EdgeKey::new(0, llm_bc.clone());
                if trie.remove_edge_dest(b, &key, c).is_some() {
                    removed_this_iter += 1;
                }
            }

            if removed_this_iter == 0 {
                break;
            }
        }
    }
}
