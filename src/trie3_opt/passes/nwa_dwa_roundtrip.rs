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
/// - `pop = m` is modeled as a chain of `m` transitions on a special `POP_SYMBOL`.
/// - State IDs from the `MiniTrie` are used as alphabet symbols in the NWA.
/// - LLM token sets from the `MiniTrie` are used as weights in the NWA.
pub struct NwaDwaRoundtripPass;

impl NwaDwaRoundtripPass {
    const POP_SYMBOL: u16 = u16::MAX;

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
                    // Create pop chain
                    for _ in 0..pop {
                        let inter = nwa.add_state();
                        // Pop transitions don't constrain tokens.
                        nwa.add_transition(cur, Self::POP_SYMBOL, inter, SimpleBitset::all());
                        cur = inter;
                    }

                    if let Some(&nwa_dst) = map_mt_to_nwa.get(&mt_dst) {
                        let weight = SimpleBitset::from_iter(ek.tokens.iter());
                        if weight.is_empty() { continue; }
                        for sid in sids.iter() {
                            // State IDs are the alphabet.
                            nwa.add_transition(cur, sid as u16, nwa_dst, weight.clone());
                        }
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
        let mut out = MiniTrie::new();
        let mut map_dwa_to_mt: BTreeMap<usize, NodeId> = BTreeMap::new();
        // Create nodes
        for (state_id, st) in dwa.states.iter().enumerate() {
            let is_end = st.final_weight.is_some();
            let nid = out.add_node(is_end);
            map_dwa_to_mt.insert(state_id, nid);
        }
        // Add edges
        // Collect all potential MiniTrie edges before adding them to group SIDs.
        // Key: (src_mt, pop, tokens, dst_mt), Value: sids
        let mut new_edges: BTreeMap<(NodeId, isize, SortedSet, NodeId), SortedSet> = BTreeMap::new();

        for (state_id, st) in dwa.states.iter().enumerate() {
            let src_mt = *map_dwa_to_mt.get(&state_id).unwrap();
            // Walk POP_SYMBOL chain; at each step, check for SID exceptions.
            let mut pop = 0;
            let mut cur = state_id;
            let mut seen: BTreeSet<usize> = BTreeSet::new();
            loop {
                if !seen.insert(cur) { break; } // Cycle in POP chain

                let current_dwa_state = &dwa.states[cur];

                // Check for SID transitions from this point in the pop chain.
                for (&char_code, &next_dwa_id) in current_dwa_state.transitions.iter_exceptions() {
                    if char_code == Self::POP_SYMBOL { continue; }

                    let sid = char_code as usize;
                    let weight = current_dwa_state.trans_weights_exceptions.get(&char_code).unwrap();
                    let tokens = SortedSet::from_iter(weight.iter());
                    if tokens.is_empty() { continue; }

                    let dst_mt = *map_dwa_to_mt.get(&next_dwa_id).unwrap();

                    let key = (src_mt, pop, tokens, dst_mt);
                    new_edges.entry(key).or_default().insert(sid);
                }

                // Follow POP transition to continue the chain.
                if let Some(&next) = current_dwa_state.transitions.get(Self::POP_SYMBOL) {
                    cur = next;
                    pop += 1;
                } else {
                    break;
                }
            }
        }
        for ((src_mt, pop, tokens, dst_mt), sids) in new_edges {
            out.add_edge(src_mt, EdgeKey::new(pop, tokens), dst_mt, sids);
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

        let original_roots = trie.root_ids.clone();
        // Build per-root roundtrip and merge results into a single MiniTrie.
        let mut merged = MiniTrie::new();
        let mut new_roots: Vec<NodeId> = Vec::with_capacity(original_roots.len());

        for &root in &original_roots {
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
