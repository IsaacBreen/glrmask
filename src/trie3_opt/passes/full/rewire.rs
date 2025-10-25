use std::collections::BTreeMap;

use ordered_hash_map::OrderedHashMap;

use crate::constraint::{LLMTokenBV, StateIDBV, Trie3GodWrapper};
use crate::datastructures::trie::Trie2Index;
use crate::datastructures::EntryApi;

/// Rewire every destination in the graph to point to its class representative.
/// This is the crucial step that actually collapses structurally equivalent nodes:
/// once all edges point to representatives, non-representatives become unreachable
/// (after roots are remapped) and can be GC'd.
pub fn rewire_all_edges_to_representatives(
    trie3_god: &Trie3GodWrapper,
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, crate::constraint::PrecomputeNode3Index>,
    node_to_rep: &std::collections::HashMap<Trie2Index, Trie2Index>,
) {
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = crate::datastructures::trie::Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

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
                let rep_dst = *node_to_rep.get(&dst).unwrap_or(&dst);
                new_dm
                    .entry(rep_dst)
                    .and_modify(|e| *e |= &sids)
                    .or_insert(sids);
            }
            if !new_dm.is_empty() {
                new_children.insert(ek, new_dm);
            }
        }
        // Recompute live tokens as the union of outgoing LLM masks.
        let mut new_live = LLMTokenBV::zeros();
        for ((_, llm_bv), _) in &new_children {
            new_live |= llm_bv;
        }
        w.value.live_tokens = new_live;
        *w.children_mut() = new_children;
    }
}
