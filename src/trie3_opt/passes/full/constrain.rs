use std::collections::BTreeMap;

use ordered_hash_map::OrderedHashMap;

use crate::constraint::{LLMTokenBV, StateIDBV, Trie3GodWrapper};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::{Trie, Trie2Index};

/// Constrain bitvectors for tokens and states to given bounds; drop empties and merge duplicates.
pub fn constrain_bitvecs_trie3(
    trie3_god: &Trie3GodWrapper,
    roots_vec: &[crate::constraint::PrecomputeNode3Index],
    max_state_id: usize,
    max_llm_token_id: usize,
) {
    crate::debug!(2, "Constraining bitvectors in Trie 3...");
    let all_nodes = Trie::all_nodes(trie3_god, roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    for node_arc in all_nodes {
        let mut guard = node_arc.write(trie3_god).unwrap();

        // Constrain live_tokens on the node value
        guard.value.live_tokens.constrain(max_llm_token_id);

        let old_children = std::mem::take(guard.children_mut());
        let mut new_children = BTreeMap::new();

        for ((pop, mut llm_bv), dest_map) in old_children {
            llm_bv.constrain(max_llm_token_id);

            let mut new_dest_map = OrderedHashMap::new();
            for (dest_wrapper, mut sids_bv) in dest_map {
                sids_bv.constrain(max_state_id);
                if !sids_bv.is_empty() {
                    new_dest_map.insert(dest_wrapper, sids_bv);
                }
            }

            if !llm_bv.is_empty() && !new_dest_map.is_empty() {
                // Need to merge if the key (with constrained llm_bv) already exists
                let entry = new_children
                    .entry((pop, llm_bv))
                    .or_insert_with(OrderedHashMap::new);
                for (dest, sids) in new_dest_map {
                    entry
                        .entry(dest)
                        .and_modify(|existing_sids| *existing_sids |= &sids)
                        .or_insert(sids);
                }
            }
        }
        *guard.children_mut() = new_children;
    }
    crate::debug!(2, "Finished constraining bitvectors.");
}
