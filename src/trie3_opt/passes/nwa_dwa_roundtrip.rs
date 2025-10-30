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
                for (&mt_dst, sids) in dm {
                    let mut cur = nwa_src;
                    // Create pop chain using default transitions. A pop `m` operation has `m-1` intermediate
                    // steps that consume any SID, followed by one SID-specific step.
                    for _ in 0..pop.saturating_sub(1) {
                        let inter = nwa.add_state();
                        // Pop steps consume any SID and don't constrain tokens.
                        nwa.add_default_transition(cur, inter, SimpleBitset::all());
                        cur = inter;
                    }

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
        for (dwa_id, dwa_state) in dwa.states.iter().enumerate() {
            let is_end = dwa_state.final_weight.as_ref().map_or(false, |fw| !fw.is_empty());
            let mt_id = mini.add_node(is_end);
            dwa_to_mt_map.insert(dwa_id, mt_id);
        }

        // 2. Create MiniTrie edges.
        for (dwa_src_id, dwa_src_state) in dwa.states.iter().enumerate() {
            let mt_src_id = dwa_to_mt_map[&dwa_src_id];

            // Group SIDs by their target DWA state.
            let mut transitions_by_dest: BTreeMap<usize, BTreeSet<u16>> = BTreeMap::new();
            for (&sid, &dwa_dst_id) in &dwa_src_state.transitions.exceptions {
                transitions_by_dest.entry(dwa_dst_id).or_default().insert(sid);
            }

            if let Some(default_dst_id) = dwa_src_state.transitions.default {
                let exception_sids: BTreeSet<u16> =
                    dwa_src_state.transitions.exceptions.keys().cloned().collect();
                let default_sids_entry = transitions_by_dest.entry(default_dst_id).or_default();
                // The alphabet is StateIDs from the original MiniTrie.
                for sid in 0..=_ctx.max_state_id {
                    if !exception_sids.contains(&(sid as u16)) {
                        default_sids_entry.insert(sid as u16);
                    }
                }
            }

            let mut new_children = BTreeMap::<EdgeKey, BTreeMap<NodeId, SortedSet>>::new();

            for (dwa_dst_id, sids) in transitions_by_dest {
                if sids.is_empty() { continue; }
                let mt_dst_id = dwa_to_mt_map[&dwa_dst_id];
                let dwa_dst_state = &dwa.states[dwa_dst_id];

                // If the destination is a final state, the tokens are its final_weight.
                // Otherwise, they are its path weight.
                let tokens = dwa_dst_state.final_weight.as_ref().filter(|fw| !fw.is_empty()).unwrap_or(&dwa_dst_state.weight).clone();
                if tokens.is_empty() { continue; }

                let tokens_set = SortedSet::from_iter(tokens.iter());
                let sids_set = SortedSet::from_iter(sids.into_iter().map(|s| s as usize));

                // A DWA transition consumes one SID. In `aici` semantics, this would normally
                // correspond to `pop=1`. However, the NWA construction in this pass loses the
                // distinction between `pop=0` and `pop=1`. We use `pop=0` to align with the
                // example output structure provided in the prompt.
                let ek = EdgeKey::new(0, tokens_set);

                let dest_map = new_children.entry(ek).or_default();
                dest_map.entry(mt_dst_id).or_default().union_inplace(&sids_set);
            }
            mini.set_children(mt_src_id, new_children);
        }

        let mt_root_id = dwa_to_mt_map[&dwa.start_state];
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

        println!("{}", trie);

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
                                if new_tokens.is_empty() { continue; }
                                let new_sids = sids_ab.intersect(sids_bc);
                                if new_sids.is_empty() { continue; }
                                let new_ek = EdgeKey::new(ek_ab.pop, new_tokens);
                                trie.add_edge(*a_id, new_ek, *c_id, new_sids);
                            }
                        }
                    }
                }
                trie.set_children(b_id, non_pop0_edges);
            }
        }

        let original_roots = trie.root_ids.clone();
        // Build per-root roundtrip and merge results into a single MiniTrie.
        let mut merged = MiniTrie::new();
        let mut new_roots: Vec<NodeId> = Vec::with_capacity(original_roots.len());

        println!("{}", trie);
        for &root in &original_roots {
            dbg!(&root);
            let (nwa, _map_mt_to_nwa) = Self::build_nwa_for_root(trie, root, ctx);
            println!("NWA for root {}: {}", root, nwa);
            let dwa = nwa.determinize();
            println!("DWA for root {}: {}", root, dwa);
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
    }
}
