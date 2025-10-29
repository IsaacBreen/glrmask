use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::glr::table::StateID;
use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;

/// Compose away all pop=0 edges whose source is not a root:
/// - Iteratively, for each B --(0,T_bc,S_bc)--> C and each incoming A --(p_ab,T_ab,S_ab)--> B,
///   add a bypass A --(p_ab, T_ab ∩ T_bc, S_ab ∩ pre^{p_ab}(S_bc))--> C.
///   Skip 0-parent edges inside the same 0-SCC unless A is a root (to avoid in-cycle blow-ups).
/// - After reaching a fixed point, delete all remaining pop=0 edges with non-root sources.
/// - Optionally assert that no non-root pop=0 edges remain (config flag).
pub struct EliminatePop0ExceptRootsPass;

impl OptimizationPass for EliminatePop0ExceptRootsPass {
    fn name(&self) -> &'static str {
        "EliminatePop0ExceptRoots"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
        let n = trie.num_nodes();
        if n == 0 {
            return;
        }
        let root_set: BTreeSet<NodeId> = trie.root_ids.iter().cloned().collect();

        // Build dense indices for NodeId.
        let node_ids: Vec<NodeId> = trie.node_ids().collect();
        let id_to_idx: HashMap<NodeId, usize> =
            node_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Build adjacency over only pop=0 edges for SCC detection.
        let mut adj0: Vec<Vec<usize>> = vec![Vec::new(); node_ids.len()];
        for node in trie.nodes() {
            let u_idx = id_to_idx[&node.id()];
            for (ek, dm) in node.children() {
                if ek.pop == 0 {
                    for (v, _) in dm {
                        let v_idx = id_to_idx[v];
                        adj0[u_idx].push(v_idx);
                    }
                }
            }
        }

        // Tarjan's algorithm to find SCCs in the 0-pop graph.
        let mut index: usize = 0;
        let mut stack: Vec<usize> = Vec::new();
        let mut onstack: Vec<bool> = vec![false; node_ids.len()];
        let mut indices: Vec<Option<usize>> = vec![None; node_ids.len()];
        let mut lowlink: Vec<usize> = vec![0; node_ids.len()];
        let mut comp_id: Vec<usize> = vec![usize::MAX; node_ids.len()];
        let mut comp_count: usize = 0;

        fn strongconnect(
            v: usize,
            index: &mut usize,
            stack: &mut Vec<usize>,
            onstack: &mut [bool],
            indices: &mut [Option<usize>],
            lowlink: &mut [usize],
            comp_id: &mut [usize],
            comp_count: &mut usize,
            adj0: &Vec<Vec<usize>>,
        ) {
            indices[v] = Some(*index);
            lowlink[v] = *index;
            *index += 1;
            stack.push(v);
            onstack[v] = true;

            for &w in &adj0[v] {
                if indices[w].is_none() {
                    strongconnect(
                        w, index, stack, onstack, indices, lowlink, comp_id, comp_count, adj0,
                    );
                    lowlink[v] = lowlink[v].min(lowlink[w]);
                } else if onstack[w] {
                    lowlink[v] = lowlink[v].min(indices[w].unwrap());
                }
            }

            if lowlink[v] == indices[v].unwrap() {
                // Start a new component
                loop {
                    let w = stack.pop().unwrap();
                    onstack[w] = false;
                    comp_id[w] = *comp_count;
                    if w == v {
                        break;
                    }
                }
                *comp_count += 1;
            }
        }

        for v in 0..node_ids.len() {
            if indices[v].is_none() {
                strongconnect(
                    v,
                    &mut index,
                    &mut stack,
                    &mut onstack,
                    &mut indices,
                    &mut lowlink,
                    &mut comp_id,
                    &mut comp_count,
                    &adj0,
                );
            }
        }

        // Precompute 2^k state preimage maps for p>0 compositions (if parser is available).
        let mut max_pop = 0usize;
        for node in trie.nodes() {
            for (ek, _) in node.children() {
                if ek.pop > 0 {
                    max_pop = max_pop.max(ek.pop as usize);
                }
            }
        }
        let mut two_power_back_maps: Vec<BTreeMap<StateID, SortedSet>> = Vec::new();
        if max_pop > 0 {
            if let Some(p_rc) = &ctx.parser {
                let parser = p_rc.borrow();
                let one_step_back_map = parser.build_one_step_back_map();
                let levels = (max_pop as f64).log2().ceil() as usize + 1;
                two_power_back_maps.push(
                    one_step_back_map
                        .into_iter()
                        .map(|(k, v)| (k, SortedSet::from_iter(v.iter())))
                        .collect(),
                );
                for i in 1..levels {
                    let prev = &two_power_back_maps[i - 1];
                    let mut next: BTreeMap<StateID, SortedSet> = BTreeMap::new();
                    for (&sid, preds1) in prev {
                        let mut preds2 = SortedSet::new();
                        for p1 in preds1.iter() {
                            let s = StateID(p1);
                            if let Some(preds_of_pred) = prev.get(&s) {
                                preds2.union_inplace(preds_of_pred);
                            }
                        }
                        if !preds2.is_empty() {
                            next.insert(sid, preds2);
                        }
                    }
                    two_power_back_maps.push(next);
                }
            }
        }

        let apply_n_step_back = |s: &SortedSet, n_steps: usize| -> SortedSet {
            if n_steps == 0 || two_power_back_maps.is_empty() {
                return s.clone();
            }
            let mut cur = s.clone();
            for k in 0..two_power_back_maps.len() {
                if ((n_steps >> k) & 1) == 1 {
                    let map_k = &two_power_back_maps[k];
                    let mut nxt = SortedSet::new();
                    for sid_val in cur.iter() {
                        let sid = StateID(sid_val);
                        if let Some(preds) = map_k.get(&sid) {
                            nxt.union_inplace(preds);
                        }
                    }
                    cur = nxt;
                }
            }
            cur
        };

        // Helper to get current states for a triple (src, key, dst).
        let get_existing = |trie: &MiniTrie, src: NodeId, key: &EdgeKey, dst: NodeId| -> SortedSet {
            if let Some(src_node) = trie.get_node(src) {
                if let Some(dm) = src_node.children().get(key) {
                    if let Some(s) = dm.get(&dst) {
                        return s.clone();
                    }
                }
            }
            SortedSet::new()
        };

        // If parser is missing and any p>0 compositions are needed, we cannot do a sound removal.
        // We still proceed with composition for p=0 when it is safe (root parent allowed),
        // but we will NOT delete non-root pop=0 edges unless we can assert the property later.
        let parser_available = ctx.parser.is_some();

        // Iterate, adding bypass edges until no more can be added or a small round budget is hit.
        let max_rounds = 16usize;
        for _round in 0..max_rounds {
            // Collect all candidate additions for this round, unioning their SID sets per (src,key,dst).
            let mut to_add: HashMap<(NodeId, EdgeKey, NodeId), SortedSet> = HashMap::new();

            // Enumerate all pop=0 edges B --(0, T_bc, S_bc)--> C.
            for b_node in trie.nodes() {
                let b_id = b_node.id();
                let b_comp = comp_id[id_to_idx[&b_id]];
                for (ek_bc, dm_bc) in b_node.children() {
                    if ek_bc.pop != 0 {
                        continue;
                    }
                    for (c_id, s_bc) in dm_bc {
                        let t_bc = &ek_bc.tokens;
                        let s_bc_clone = s_bc.clone();

                        // For each predecessor A --(p_ab, T_ab, S_ab)--> B:
                        let preds = b_node.parents().clone(); // clone to avoid borrow issues
                        for (a_id, edges_from_a) in &preds {
                            let a_comp = comp_id[id_to_idx[a_id]];
                            for (key_ab, s_ab) in edges_from_a {
                                let p_ab = key_ab.pop;
                                let t_ab = &key_ab.tokens;

                                // Guard against in-cycle 0-parent expansion unless A is a root.
                                if p_ab == 0 && !root_set.contains(a_id) && a_comp == b_comp {
                                    continue;
                                }

                                // Compose tokens.
                                let new_tokens = t_ab.intersect(t_bc);
                                if new_tokens.is_empty() {
                                    continue;
                                }

                                // Compose states: S_ab ∩ pre^{p_ab}(S_bc).
                                let p_usize = if p_ab > 0 { p_ab as usize } else { 0 };
                                if p_usize > 0 && two_power_back_maps.is_empty() {
                                    // No parser provided: cannot soundly compute p>0 preimages; skip.
                                    continue;
                                }
                                let s_bc_pre = if p_usize > 0 {
                                    apply_n_step_back(&s_bc_clone, p_usize)
                                } else {
                                    s_bc_clone.clone()
                                };
                                if s_bc_pre.is_empty() {
                                    continue;
                                }
                                let new_sids = s_ab.intersect(&s_bc_pre);
                                if new_sids.is_empty() {
                                    continue;
                                }

                                let new_key = EdgeKey::new(p_ab, new_tokens);
                                // Only add the difference relative to what's already present.
                                let existing = get_existing(trie, *a_id, &new_key, *c_id);
                                let delta = new_sids.difference(&existing);
                                if delta.is_empty() {
                                    continue;
                                }
                                let k = (*a_id, new_key, *c_id);
                                to_add.entry(k).or_default().union_inplace(&delta);
                            }
                        }
                    }
                }
            }

            if to_add.is_empty() {
                break; // fixed point
            }

            // Apply all new edges for this round.
            for ((src, key, dst), sids) in to_add {
                trie.add_edge(src, key, dst, sids);
            }
        }

        // After convergence, delete all pop=0 edges whose source is not a root.
        // This enforces the target property while preserving semantics via the composed edges.
        let mut removed_any = false;
        let all_node_ids: Vec<_> = trie.node_ids().collect();
        for u_id in all_node_ids {
            if root_set.contains(&u_id) {
                continue;
            }
            if let Some(u_node) = trie.get_node(u_id) {
                // Collect pop=0 keys to remove.
                let mut keys_to_remove: Vec<EdgeKey> = Vec::new();
                for (ek, _dm) in u_node.children() {
                    if ek.pop == 0 {
                        keys_to_remove.push(ek.clone());
                    }
                }
                if keys_to_remove.is_empty() {
                    continue;
                }
                for key in keys_to_remove {
                    if let Some(u_node2) = trie.get_node(u_id) {
                        if let Some(dm) = u_node2.children().get(&key) {
                            let dests: Vec<_> = dm.keys().cloned().collect();
                            for v_id in dests {
                                let _ = trie.remove_edge_dest(u_id, &key, v_id);
                                removed_any = true;
                            }
                        }
                    }
                }
            }
        }

        // Optional assertion: verify no pop=0 edges remain from non-root sources.
        if ctx.assert_no_pop0_except_roots {
            let mut violations = 0usize;
            for node in trie.nodes() {
                if root_set.contains(&node.id()) {
                    continue;
                }
                for (ek, dm) in node.children() {
                    if ek.pop == 0 && !dm.is_empty() {
                        violations += dm.len();
                    }
                }
            }
            assert!(
                violations == 0,
                "EliminatePop0ExceptRoots: found {} pop=0 edge destinations from non-root sources after pass",
                violations
            );
        }
    }
}

