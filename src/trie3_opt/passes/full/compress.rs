use std::collections::{BTreeMap, HashMap};

use ordered_hash_map::OrderedHashMap;

use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, StateIDBV, Trie3GodWrapper};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::{Trie, Trie2Index};
use crate::profiler::PROGRESS_BAR_ENABLED;

/// Compress edges by grouping identical destination maps (for a given pop) and unioning token masks.
/// Adopt the rewrite only if a local cost metric improves.
pub fn compress_trie3_edges(
    roots: &mut BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    max_llm_token_id: usize,
    max_state_id: usize,
) {
    crate::debug!(2, "Compressing Trie3 edges (group identical dest-maps and union token masks)...");

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    #[cfg(not(rustrover))]
    let it = kdam::tqdm!(
        all_nodes.iter(),
        desc = "Trie3 Compress Edges (cost-aware)",
        total = all_nodes.len(),
        disable = !PROGRESS_BAR_ENABLED,
        leave = true
    );
    #[cfg(rustrover)]
    let it = all_nodes.iter();
    for node_idx in it {
        let mut w = node_idx.write(trie3_god).expect("write");
        if w.children().is_empty() {
            continue;
        }

        // Work from a snapshot so we can compute both old/new costs and adopt only if better.
        let old_snapshot = w.children().clone();

        // Local helpers for bounded range-count and cost metric
        #[inline]
        fn ranges_len_bounded(m: &LLMTokenBV, max_id: usize) -> usize {
            let mut c = m.clone();
            c.constrain(max_id);
            c.inner().ranges_len()
        }
        use std::collections::BTreeMap as BTM;
        #[inline]
        fn children_cost(
            children: &BTM<(isize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>>,
            max_llm_token_id: usize,
        ) -> (usize, usize, usize) {
            let mut ranges_sum = 0usize;
            let mut dests_sum = 0usize;
            for ((_, l), dm) in children {
                ranges_sum += ranges_len_bounded(l, max_llm_token_id);
                dests_sum += dm.len();
            }
            (ranges_sum + dests_sum, ranges_sum, dests_sum)
        }

        // Compute old cost
        let old_cost = children_cost(&old_snapshot, max_llm_token_id).0;

        // First stage: group by (pop, canonicalized dest-map), unioning L masks for identical dest-maps.
        use std::collections::HashMap as HM;
        let mut by_pop: BTM<isize, HM<Vec<(Trie2Index, StateIDBV)>, LLMTokenBV>> = BTM::new();
        for ((pop, llm_bv_orig), dest_map_orig) in &old_snapshot {
            // Constrain token bitset to bound and skip empties.
            let mut llm_bv = llm_bv_orig.clone();
            llm_bv.constrain(max_llm_token_id);
            if llm_bv.is_empty() {
                continue;
            }

            // Aggregate destinations by unioning SIDs and removing empties.
            let mut dest_agg: BTM<Trie2Index, StateIDBV> = BTM::new();
            for (dst, sids0) in dest_map_orig {
                let mut sids = sids0.clone();
                sids.constrain(max_state_id);
                if sids.is_empty() {
                    continue;
                }
                dest_agg
                    .entry(*dst)
                    .and_modify(|e| *e |= &sids)
                    .or_insert(sids);
            }
            if dest_agg.is_empty() {
                continue;
            }

            // Canonical destination vector (sorted by dst) for use as a key.
            let dest_vec: Vec<(Trie2Index, StateIDBV)> = dest_agg.into_iter().collect();

            // Merge edges that have the same pop and identical destination map by unioning their LLM masks.
            let entry = by_pop
                .entry(*pop)
                .or_default()
                .entry(dest_vec)
                .or_insert_with(LLMTokenBV::zeros);
            *entry |= &llm_bv;
        }

        // Second stage: for each pop, combine groups that yield identical final L masks
        // by unioning their destination maps (one entry per (pop, llm_bv)).
        let mut new_children: BTM<(isize, LLMTokenBV), OrderedHashMap<Trie2Index, StateIDBV>> =
            BTM::new();
        for (pop, groups) in by_pop {
            // llm_bv -> aggregated destination map (as BTreeMap for determinism)
            let mut out_by_llm: BTM<LLMTokenBV, BTM<Trie2Index, StateIDBV>> = BTM::new();

            for (dest_vec, llm_bv) in groups {
                if llm_bv.is_empty() {
                    continue;
                }
                let dest_out = out_by_llm.entry(llm_bv).or_insert_with(BTM::new);
                for (dst, sids) in dest_vec {
                    dest_out
                        .entry(dst)
                        .and_modify(|e| *e |= &sids)
                        .or_insert(sids);
                }
            }

            // Emit final edges: (pop, llm_bv) -> OrderedHashMap<dst, StateIDBV>
            for (llm_bv, dest_btree) in out_by_llm {
                if llm_bv.is_empty() {
                    continue;
                }
                let mut ordered_dm: OrderedHashMap<Trie2Index, StateIDBV> =
                    OrderedHashMap::new();
                for (dst, sids) in dest_btree {
                    if !sids.is_empty() {
                        ordered_dm.insert(dst, sids);
                    }
                }
                if !ordered_dm.is_empty() {
                    let entry = new_children
                        .entry((pop, llm_bv))
                        .or_insert_with(OrderedHashMap::new);
                    for (dst, sids) in ordered_dm {
                        entry
                            .entry(dst)
                            .and_modify(|e| *e |= &sids)
                            .or_insert(sids);
                    }
                }
            }
        }

        // Compare costs and adopt only if better.
        let new_cost = children_cost(&new_children, max_llm_token_id).0;
        if new_cost < old_cost {
            let mut new_live = LLMTokenBV::zeros();
            for ((_, llm_bv), _) in &new_children {
                new_live |= llm_bv;
            }
            w.value.live_tokens = new_live;
            *w.children_mut() = new_children;
        } else {
            // Keep old structure; normalize live_tokens from old children.
            let mut live = LLMTokenBV::zeros();
            for ((_, llm_bv), _) in &old_snapshot {
                live |= llm_bv;
            }
            w.value.live_tokens = live;
            // w.children() already equals old_snapshot
        }
    }

    crate::debug!(2, "Finished compressing Trie3 edges.");
}
