use std::collections::{BTreeMap, BTreeSet, VecDeque};
use crate::datastructures::trie::Trie;
use crate::trie3_opt::context::OptimizationContext;
use crate::trie3_opt::core::{EdgeKey, MiniTrie, Node, NodeId, SortedSet};
use crate::trie3_opt::passes::OptimizationPass;
use crate::weighted_automata::{NWA, SimpleBitset};

/// A roundtrip pass that:
/// - Converts the MiniTrie subgraph for each root into a weighted NWA,
/// - Determinizes it into a DWA,
/// - Converts the DWA back into a MiniTrie,
/// - Merges all per-root results into a single MiniTrie, preserving root order.
///
/// Encoding:
/// - `pop = m` is modeled as a chain of `m-1` default transitions followed by a SID-specific transition.
/// - State IDs from the `MiniTrie` are used as alphabet symbols in the NWA.
/// - LLM token sets from the `MiniTrie` are used as weights in the NWA.
pub struct NwaDwaRoundtripPass;

impl NwaDwaRoundtripPass {
    fn reachable_from_one(trie: &MiniTrie, start: NodeId) -> BTreeSet<NodeId> {
        let mut seen: BTreeSet<NodeId> = BTreeSet::new();
        let mut q: VecDeque<NodeId> = VecDeque::new();
        q.push_back(start);
        while let Some(u) = q.pop_front() {
            if !seen.insert(u) {
                continue;
            }
            if let Some(node) = trie.get_node(u) {
                for (_ek, dm) in node.children() {
                    for (v, _) in dm {
                        if !seen.contains(v) {
                            q.push_back(*v);
                        }
                    }
                }
            }
        }
        seen
    }

    fn build_nwa_for_root(
        mini: &MiniTrie,
        root: NodeId,
        _ctx: &OptimizationContext,
    ) -> (NWA, BTreeMap<NodeId, usize>) {
        let mut nwa = NWA::new();
        // Collect subgraph nodes reachable from this root.
        let sub_nodes = Self::reachable_from_one(mini, root);
        // Map MiniTrie node -> NWA state
        let mut map_mt_to_nwa: BTreeMap<NodeId, usize> = BTreeMap::new();
        for n_id in sub_nodes.iter() {
            let s = nwa.add_state();
            map_mt_to_nwa.insert(*n_id, s);
        }
        // Root state is the NWA start
        if let Some(&start_id) = map_mt_to_nwa.get(&root) {
            nwa.start_state = start_id;
        }
        // Mark final weights for ends
        for (mt_id, nwa_id) in map_mt_to_nwa.iter() {
            if let Some(node) = mini.get_node(*mt_id) {
                if node.is_end() {
                    nwa.set_final_weight(*nwa_id, SimpleBitset::all());
                }
            }
        }

        for (mt_id, &nwa_src) in map_mt_to_nwa.iter() {
            let node = mini.get_node(*mt_id).unwrap();
            for (ek, dm) in node.children() {
                if ek.pop < 0 { continue; }
                let pop = ek.pop as usize;
                let mut cur = nwa_src;
                // Create pop chain using default transitions. A pop `m` operation has `m-1` intermediate
                // steps that consume any SID, followed by one SID-specific step.
                for _ in 0..pop.saturating_sub(1) {
                    let inter = nwa.add_state();
                    // Pop steps consume any SID and don't constrain tokens.
                    nwa.add_default_transition(cur, inter, SimpleBitset::all());
                    cur = inter;
                }
                for (&mt_dst, sids) in dm {
                    // The last step is not a default transition, but specific to the SIDs.
                    let nwa_dst = map_mt_to_nwa[&mt_dst];
                    let weight = SimpleBitset::from_iter(ek.tokens.iter());
                    for sid in sids.iter() {
                        // State IDs are the alphabet.
                        nwa.add_transition(cur, sid as u16, nwa_dst, weight.clone());
                    }
                }
            }
        }
        (nwa, map_mt_to_nwa)
    }
    fn convert_dwa_to_minitrie(
        dwa: crate::weighted_automata::DWA,
        _ctx: &OptimizationContext,
    ) -> (MiniTrie, NodeId) {
        let mut mini = MiniTrie::new();
        if dwa.states.is_empty() {
            // Should not happen if NWA was non-empty, as DWA gets at least a start state.
            let root_id = mini.add_node(false);
            return (mini, root_id);
        }

        // 1. Create MiniTrie nodes for each DWA state.
        let mut dwa_to_mt_map: BTreeMap<usize, NodeId> = BTreeMap::new();
        let end_node_id = mini.add_node(true);
        for (dwa_id, _dwa_state) in dwa.states.iter().enumerate() {
            // DWA states are not end nodes themselves. Finality is modeled by an edge to a canonical end node.
            let mt_id = mini.add_node(false);
            dwa_to_mt_map.insert(dwa_id, mt_id);
        }

        let mt_root_id = dwa_to_mt_map[&dwa.start_state];

        // 2. Create MiniTrie edges.
        for (dwa_src_id, dwa_src_state) in dwa.states.iter().enumerate() {
            let mt_src_id = dwa_to_mt_map[&dwa_src_id];

            let pop = if mt_src_id == mt_root_id { 0isize } else { 1isize };

            // Group by (tokens, dst) to build MiniTrie edges, since pop is fixed for this src node.
            let mut edge_groups: BTreeMap<(SortedSet, NodeId), SortedSet> = BTreeMap::new();

            // Handle exception transitions
            for (&sid, &dwa_dst_id) in &dwa_src_state.transitions.exceptions {
                if let Some(tokens) = dwa_src_state.trans_weights_exceptions.get(&sid) {
                    if tokens.is_empty() {
                        continue;
                    }

                    let tokens_set = SortedSet::from_iter(tokens.iter());
                    let mt_dst_id = dwa_to_mt_map[&dwa_dst_id];

                    let key = (tokens_set, mt_dst_id);
                    edge_groups.entry(key).or_default().insert(sid as usize);
                }
            }

            // Handle default transition
            if let Some(dwa_dst_id) = dwa_src_state.transitions.default {
                if let Some(tokens) = &dwa_src_state.trans_weight_default {
                    if !tokens.is_empty() {
                        let tokens_set = SortedSet::from_iter(tokens.iter());
                        let mt_dst_id = dwa_to_mt_map[&dwa_dst_id];
                        let key = (tokens_set, mt_dst_id);

                        let exception_sids: BTreeSet<u16> =
                            dwa_src_state.transitions.exceptions.keys().cloned().collect();

                        let default_sids_entry = edge_groups.entry(key).or_default();
                        for sid in 0..=_ctx.max_state_id {
                            if !exception_sids.contains(&(sid as u16)) {
                                default_sids_entry.insert(sid);
                            }
                        }
                    }
                }
            }

            // Now build the new children for the MiniTrie node
            let mut new_children = BTreeMap::<EdgeKey, BTreeMap<NodeId, SortedSet>>::new();
            for ((tokens, dst), sids) in edge_groups {
                if sids.is_empty() {
                    continue;
                }
                let ek = EdgeKey::new(pop, tokens);
                new_children.entry(ek).or_default().insert(dst, sids);
            }

            // If the DWA state has a final weight, create an edge to a canonical end node.
            // This encodes the conditional finality.
            if let Some(final_weight) = &dwa_src_state.final_weight {
                if !final_weight.is_empty() {
                    let tokens = SortedSet::from_iter(final_weight.iter());
                    let sids = SortedSet::from_iter(0..=_ctx.max_state_id);  // TODO: VERY INEFFICIENT
                    new_children.entry(EdgeKey::new(0, tokens)).or_default().insert(end_node_id, sids);
                }
            }
            mini.set_children(mt_src_id, new_children);
        }

        (mini, mt_root_id)
    }
}

impl OptimizationPass for NwaDwaRoundtripPass {
    fn name(&self) -> &'static str {
        "NwaDwaRoundtrip"
    }

    fn run(&self, trie: &mut MiniTrie, ctx: &mut OptimizationContext) {
        if trie.num_nodes() == 0 {
            return;
        }

        // println!("{}", trie);

        // Eliminate pop=0 edges on non-root nodes by merging them into predecessors.
        // This is a prerequisite for NWA conversion, which has simplified handling for pop>0.
        // After this loop, only root nodes may have pop=0 edges.
        let mut changed = true;
        while changed {
            changed = false;
            let node_ids: Vec<_> = trie.node_ids().collect();
            let root_set: BTreeSet<_> = trie.root_ids.iter().copied().collect();

            for b_id in node_ids {
                if root_set.contains(&b_id) {
                    continue;
                }

                // Must clone, as we will be modifying the trie.
                let b_node = if let Some(n) = trie.get_node(b_id) {
                    n.clone()
                } else {
                    continue;
                };

                let mut pop0_edges = BTreeMap::new();
                let mut non_pop0_edges = BTreeMap::new();

                for (ek, dm) in b_node.children() {
                    if ek.pop == 0 {
                        pop0_edges.insert(ek.clone(), dm.clone());
                    } else {
                        non_pop0_edges.insert(ek.clone(), dm.clone());
                    }
                }

                if pop0_edges.is_empty() {
                    continue;
                }

                changed = true;
                let parents_of_b = b_node.parents().clone();

                for (pop0_ek, pop0_dm) in &pop0_edges {
                    for (c_id, sids_bc) in pop0_dm {
                        for (a_id, edges_from_a_to_b) in &parents_of_b {
                            for (ek_ab, sids_ab) in edges_from_a_to_b {
                                let new_tokens = ek_ab.tokens.intersect(&pop0_ek.tokens);
                                let new_sids = sids_ab.intersect(sids_bc);
                                let new_ek = EdgeKey::new(ek_ab.pop, new_tokens);
                                trie.add_edge(*a_id, new_ek, *c_id, new_sids);
                            }
                        }
                    }
                }
                trie.set_children(b_id, non_pop0_edges);
            }
        }

        // Prune edges that don't lead to an end node.
        let productive_nodes = trie.can_reach_end();
        let all_node_ids: Vec<_> = trie.node_ids().collect();
        for node_id in all_node_ids {
            let node = if let Some(n) = trie.get_node(node_id) {
                n.clone()
            } else {
                continue;
            };
            let mut to_remove = Vec::new();
            for (ek, dm) in node.children() {
                for dst in dm.keys() {
                    if !productive_nodes.contains(dst) {
                        to_remove.push((ek.clone(), *dst));
                    }
                }
            }
            for (ek, dst) in to_remove {
                trie.remove_edge_dest(node_id, &ek, dst);
            }
        }

        let original_roots = trie.root_ids.clone();
        // Build per-root roundtrip and merge results into a single MiniTrie.
        let mut merged = MiniTrie::new();
        let mut new_roots: Vec<NodeId> = Vec::with_capacity(original_roots.len());

        // println!("{}", trie);
        for &root in &original_roots {
            dbg!(&root);
            let (nwa, _map_mt_to_nwa) = Self::build_nwa_for_root(trie, root, ctx);
            // println!("NWA for root {}: {}", root, nwa);
            let dwa = nwa.determinize();
            // println!("DWA for root {}: {}", root, dwa);
            let (partial, partial_root) = Self::convert_dwa_to_minitrie(dwa, ctx);

            // Merge partial into merged
            // Build map: partial NodeId -> merged NodeId
            let mut id_map: BTreeMap<NodeId, NodeId> = BTreeMap::new();
            for node in partial.nodes() {
                let nid = merged.add_node(node.is_end());
                id_map.insert(node.id(), nid);
            }
            // Edges
            for node in partial.nodes() {
                let src_new = *id_map.get(&node.id()).unwrap();
                for (ek, dm) in node.children() {
                    let mut new_dm: BTreeMap<NodeId, SortedSet> = BTreeMap::new();
                    for (dst, sids) in dm {
                        let dst_new = *id_map.get(dst).unwrap();
                        new_dm.insert(dst_new, sids.clone());
                    }
                    for (dst_new, sids) in new_dm {
                        merged.add_edge(src_new, ek.clone(), dst_new, sids);
                    }
                }
            }
            // Root
            new_roots.push(*id_map.get(&partial_root).unwrap());
        }

        merged.root_ids = new_roots;
        // Replace input trie with merged result
        *trie = merged;
        // println!("{}", trie);
    }
}
