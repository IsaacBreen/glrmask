use std::collections::{BTreeMap, BTreeSet, VecDeque};

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
/// - We use a single "token consumption" symbol (CHAR_TOKEN) for actual token edges.
/// - pop = m is modeled as m default transitions followed by one CHAR_TOKEN exception.
/// - Weights encode both tokens and StateIDs in a single set:
///   [0..=max_llm_token_id]      -> tokens
///   [offset..=offset+max_sid]   -> state ids, where offset = max_llm_token_id + 1
pub struct NwaDwaRoundtripPass;

impl NwaDwaRoundtripPass {
    const CHAR_TOKEN: u16 = b'T' as u16;

    fn offset(max_llm_token_id: usize) -> usize {
        max_llm_token_id.saturating_add(1)
    }

    fn encode_weight(tokens: &SortedSet, sids: &SortedSet, offset: usize) -> SimpleBitset {
        let mut items: Vec<usize> = Vec::with_capacity(tokens.len() + sids.len());
        for t in tokens.iter() {
            items.push(t);
        }
        for s in sids.iter() {
            items.push(offset + s);
        }
        SimpleBitset::from_iter(items)
    }

    fn decode_weight(
        w: &SimpleBitset,
        offset: usize,
        max_llm_token_id: usize,
        max_state_id: usize,
    ) -> (SortedSet, SortedSet) {
        // Defensive path for ALL; not expected for exception edges, but included for safety.
        if *w == SimpleBitset::all() {
            let toks = SortedSet::from_iter(0..=max_llm_token_id);
            let sids = SortedSet::from_iter(0..=max_state_id);
            return (toks, sids);
        }
        let mut toks = SortedSet::new();
        let mut sids = SortedSet::new();
        for idx in w.iter() {
            if idx <= max_llm_token_id {
                toks.insert(idx);
            } else if idx >= offset && idx <= offset + max_state_id {
                sids.insert(idx - offset);
            }
        }
        (toks, sids)
    }

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
        ctx: &OptimizationContext,
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
        // Build transitions: pop m => m default steps, then CHAR_TOKEN with weight(tokens ∪ offset+sids)
        let offset = Self::offset(ctx.max_llm_token_id);
        for (mt_id, &nwa_src) in map_mt_to_nwa.iter() {
            let node = mini.get_node(*mt_id).unwrap();
            for (ek, dm) in node.children() {
                // After pre-processing, pop=0 edges are only expected from root nodes.
                // Negative pops are not expected in valid graphs.
                if ek.pop < 0 { continue; }
                let pop = ek.pop as usize;
                for (&mt_dst, sids) in dm {
                    let mut cur = nwa_src;
                    // Create pop default chain
                    for _ in 0..pop.saturating_sub(1) {
                        let inter = nwa.add_state();
                        nwa.add_default_transition(cur, inter, SimpleBitset::all());
                        cur = inter;
                    }
                    // Final CHAR_TOKEN edge carrying combined weight
                    if let Some(&nwa_dst) = map_mt_to_nwa.get(&mt_dst) {
                        let w = Self::encode_weight(&ek.tokens, sids, offset);
                        nwa.add_transition(cur, Self::CHAR_TOKEN, nwa_dst, w);
                    }
                }
            }
        }
        (nwa, map_mt_to_nwa)
    }

    fn convert_dwa_to_minitrie(
        dwa: crate::weighted_automata::DWA,
        ctx: &OptimizationContext,
    ) -> (MiniTrie, NodeId) {
        let mut out = MiniTrie::new();
        let mut map_dwa_to_mt: BTreeMap<usize, NodeId> = BTreeMap::new();
        let offset = Self::offset(ctx.max_llm_token_id);
        // Create nodes
        for (state_id, st) in dwa.states.iter().enumerate() {
            let is_end = st.final_weight.is_some();
            let nid = out.add_node(is_end);
            map_dwa_to_mt.insert(state_id, nid);
        }
        // Add edges
        for (state_id, st) in dwa.states.iter().enumerate() {
            let src_mt = *map_dwa_to_mt.get(&state_id).unwrap();
            // Walk default chain; at each step, if there's a CHAR_TOKEN exception, emit edge with pop = steps
            let mut steps = 0usize;
            let mut cur = state_id;
            let mut seen: BTreeSet<usize> = BTreeSet::new();
            loop {
                if seen.contains(&cur) {
                    break;
                }
                seen.insert(cur);
                // If this state has a CHAR_TOKEN exception, add an edge
                if let Some(&to_id) = dwa.states[cur].transitions.get(Self::CHAR_TOKEN) {
                    if let Some(w) = dwa.states[cur].trans_weights_exceptions.get(&Self::CHAR_TOKEN) {
                        let (toks, sids) = Self::decode_weight(w, offset, ctx.max_llm_token_id, ctx.max_state_id);
                        if !toks.is_empty() && !sids.is_empty() {
                            let ek = EdgeKey::new(steps as isize, toks.clone());
                            let dst_mt = *map_dwa_to_mt.get(&to_id).unwrap();
                            out.add_edge(src_mt, ek, dst_mt, sids);
                        }
                    }
                }
                if let Some(next) = dwa.states[cur].transitions.default {
                    cur = next;
                    steps += 1;
                } else {
                    break;
                }
            }
        }
        // Return constructed trie and the start node for use as this root's root
        let root_mt = *map_dwa_to_mt.get(&dwa.start_state).unwrap();
        (out, root_mt)
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

        for &root in &original_roots {
            let (nwa, _map_mt_to_nwa) = Self::build_nwa_for_root(trie, root, ctx);
            let dwa = nwa.determinize();
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
