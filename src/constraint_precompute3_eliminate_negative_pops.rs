// src/constraint_precompute3_eliminate_negative_pops.rs
use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, StateIDBV, Trie3GodWrapper};
use crate::datastructures::trie::Trie;
use std::collections::BTreeMap;
use crate::datastructures::EntryApi;

/// Traverses the Trie3 graph and replaces all edges with negative pop values
/// with their absolute value counterparts. If both `(k, ...)` and `(-k, ...)` edges
/// exist from the same source node, their destinations and state bitvectors are merged.
pub fn eliminate_negative_pops_trie3(
    trie3_god: &Trie3GodWrapper,
    roots: &[PrecomputeNode3Index],
) {
    let all_nodes = Trie::all_nodes(trie3_god, roots);
    for node_idx in all_nodes {
        let mut node_write_guard = node_idx
            .write(trie3_god)
            .expect("Failed to get write lock on Trie3 node");

        // Take ownership of the children map to rebuild it.
        let old_children = std::mem::take(node_write_guard.children_mut());

        for (edge_key, dest_map) in old_children {
            let (pop, llm_bv) = edge_key;
            let new_key = (pop.abs(), llm_bv);

            // Get or create the entry for the normalized key in the new children map.
            let entry = node_write_guard.children_mut().entry(new_key).or_default();

            // Merge destinations and state BVs.
            for (dest_idx, state_bv) in dest_map {
                entry
                    .entry(dest_idx)
                    .and_modify(|existing_bv: &mut StateIDBV| *existing_bv |= &state_bv)
                    .or_insert(state_bv);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::{PrecomputedNodeContents, StateIDBV};
    use crate::datastructures::trie::Trie2Index;

    #[test]
    fn test_eliminate_negative_pops() {
        let god = Trie3GodWrapper::new();

        // Create nodes
        let root = Trie2Index::new(god.insert(Trie::new(PrecomputedNodeContents::internal())));
        let n1 = Trie2Index::new(god.insert(Trie::new(PrecomputedNodeContents::internal())));
        let n2 = Trie2Index::new(god.insert(Trie::new(PrecomputedNodeContents::internal())));
        let n3 = Trie2Index::new(god.insert(Trie::new(PrecomputedNodeContents::internal())));

        let llm_bv1 = LLMTokenBV::from_iter(0..10);
        let llm_bv2 = LLMTokenBV::from_iter(10..20);
        let llm_bv3 = LLMTokenBV::from_iter(20..30);

        let mut state_bv1 = StateIDBV::zeros();
        state_bv1.insert(1);
        let mut state_bv2 = StateIDBV::zeros();
        state_bv2.insert(2);
        let mut state_bv3 = StateIDBV::zeros();
        state_bv3.insert(3);
        let mut state_bv4 = StateIDBV::zeros();
        state_bv4.insert(4);

        // Add edges with negative and positive pops
        {
            let mut root_w = root.write(&god).unwrap();
            // root --(-1, llm_bv1)--> n1
            root_w
                .children_mut()
                .entry((-1, llm_bv1.clone()))
                .or_default()
                .insert(n1, state_bv1.clone());
            // root --(1, llm_bv3)--> n3
            root_w
                .children_mut()
                .entry((1, llm_bv3.clone()))
                .or_default()
                .insert(n3, state_bv3.clone());
            // root --(-1, llm_bv3)--> n3 (will merge with above)
            root_w
                .children_mut()
                .entry((-1, llm_bv3.clone()))
                .or_default()
                .insert(n3, state_bv4.clone());
        }
        {
            let mut n1_w = n1.write(&god).unwrap();
            // n1 --(-2, llm_bv2)--> n2
            n1_w.children_mut()
                .entry((-2, llm_bv2.clone()))
                .or_default()
                .insert(n2, state_bv2.clone());
        }

        // Run the function
        eliminate_negative_pops_trie3(&god, &[root]);

        // --- Assertions ---
        let root_r = root.read(&god).unwrap();
        let n1_r = n1.read(&god).unwrap();

        // Check root's edges
        assert_eq!(root_r.children().len(), 2); // (-1,bv1) became (1,bv1). (-1,bv3) merged into (1,bv3).

        // Check edge to n1
        let dests_to_n1 = root_r.children().get(&(1, llm_bv1)).unwrap();
        assert_eq!(dests_to_n1.len(), 1);
        assert_eq!(dests_to_n1.get(&n1).unwrap(), &state_bv1);

        // Check edge to n3 (merged)
        let dests_to_n3 = root_r.children().get(&(1, llm_bv3)).unwrap();
        let mut expected_merged_bv = state_bv3.clone();
        expected_merged_bv |= &state_bv4;
        assert_eq!(dests_to_n3.len(), 1);
        assert_eq!(dests_to_n3.get(&n3).unwrap(), &expected_merged_bv);

        // Check n1's edge
        assert_eq!(n1_r.children().len(), 1);
        let dests_to_n2 = n1_r.children().get(&(2, llm_bv2)).unwrap();
        assert_eq!(dests_to_n2.len(), 1);
        assert_eq!(dests_to_n2.get(&n2).unwrap(), &state_bv2);
    }
}
