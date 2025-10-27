use std::collections::{BTreeMap, HashSet};

use ordered_hash_map::OrderedHashMap;

use crate::constraint::{
    LLMTokenBV, PrecomputeNode3Index, StateIDBV, Trie3GodWrapper,
};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::{Trie, Trie2Index};

/// Canonicalize all END nodes to a single representative.
/// - Picks the first reachable END node as canonical
/// - Clears its children and sets live_tokens to 0
/// - Rewrites every edge targeting any END node to target the canonical END
/// - Remaps END roots to the canonical END
/// - GC and recompute depths
pub fn canonicalize_end_nodes_trie3(
    roots: &mut BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    crate::debug!(2, "Canonicalizing END nodes in Trie3...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    let mut end_nodes: Vec<PrecomputeNode3Index> = Vec::new();
    for n in &all_nodes {
        let r = n.read(trie3_god).expect("read");
        if r.value.end {
            end_nodes.push(*n);
        }
    }
    if end_nodes.len() <= 1 {
        crate::debug!(
            3,
            "END canonicalization skipped: {} END node(s).",
            end_nodes.len()
        );
        return;
    }

    let canonical = end_nodes[0];
    // Ensure canonical is a clean terminal.
    if let Some(mut w) = canonical.write(trie3_god) {
        *w.children_mut() = BTreeMap::new();
        w.value.live_tokens = LLMTokenBV::zeros();
    }
    let end_set: HashSet<PrecomputeNode3Index> = end_nodes.into_iter().collect();

    // Rewire all edges that point to any END node -> canonical END
    for n in &all_nodes {
        let mut w = n.write(trie3_god).expect("write");
        let old_children = std::mem::take(w.children_mut());
        let mut new_children: BTreeMap<
            (isize, LLMTokenBV),
            OrderedHashMap<Trie2Index, StateIDBV>,
        > = BTreeMap::new();
        for (ek, dm) in old_children {
            let mut new_dm: OrderedHashMap<Trie2Index, StateIDBV> = OrderedHashMap::new();
            for (dst, sids) in dm {
                let new_dst = if end_set.contains(&dst) { canonical } else { dst };
                new_dm
                    .entry(new_dst)
                    .and_modify(|e| *e |= &sids)
                    .or_insert(sids);
            }
            if !new_dm.is_empty() {
                new_children.insert(ek, new_dm);
            }
        }
        let mut new_live = LLMTokenBV::zeros();
        for ((_, llm_bv), _) in &new_children {
            new_live |= llm_bv;
        }
        w.value.live_tokens = new_live;
        *w.children_mut() = new_children;
    }

    // Remap END roots to canonical
    for r in roots.values_mut() {
        if end_set.contains(r) {
            *r = canonical;
        }
    }
    let roots_vec2: Vec<_> = roots.values().cloned().collect();
    Trie::gc(trie3_god, &roots_vec2);
    Trie::recompute_all_max_depths(trie3_god, &roots_vec2);
    crate::debug!(2, "Canonicalized END nodes to representative {}.", canonical);
}
