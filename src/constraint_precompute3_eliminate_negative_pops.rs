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
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::{PrecomputedNodeContents, StateIDBV, LLMTokenBV, PrecomputeNode3Index};
    use crate::datastructures::trie::{Trie, Trie2Index};
    use ordered_hash_map::OrderedHashMap;

    // --- Test Helpers ---

    /// Helper to create a new internal node in the arena.
    fn new_node(god: &Trie3GodWrapper) -> PrecomputeNode3Index {
        Trie2Index::new(god.insert(Trie::new(PrecomputedNodeContents::internal())))
    }

    /// Helper to create a StateIDBV from a slice of indices.
    fn state_bv(indices: &[usize]) -> StateIDBV {
        let mut bv = StateIDBV::zeros();
        for &i in indices {
            bv.insert(i);
        }
        bv
    }

    /// Helper to add an edge to the graph, encapsulating the write lock logic.
    fn add_edge(
        god: &Trie3GodWrapper,
        from: PrecomputeNode3Index,
        key: (i32, LLMTokenBV),
        to: PrecomputeNode3Index,
        val: StateIDBV,
    ) {
        let mut from_w = from.write(god).unwrap();
        from_w.children_mut().entry(key).or_default().insert(to, val);
    }

    // --- Tests ---

    #[test]
    fn test_eliminate_negative_pops() {
        // 1. ARRANGE: Set up the graph and test data
        let god = Trie3GodWrapper::new();

        let root = new_node(&god);
        let n1 = new_node(&god);
        let n2 = new_node(&god);
        let n3 = new_node(&god);

        let llm_bv1 = LLMTokenBV::from_iter(0..10);
        let llm_bv2 = LLMTokenBV::from_iter(10..20);
        let llm_bv3 = LLMTokenBV::from_iter(20..30);

        // Build the initial graph with a mix of positive and negative pops
        // root --(-1, bv1)--> n1
        add_edge(&god, root, (-1, llm_bv1.clone()), n1, state_bv(&[1]));
        // n1 --(-2, bv2)--> n2
        add_edge(&god, n1, (-2, llm_bv2.clone()), n2, state_bv(&[2]));
        // root --(1, bv3)--> n3
        add_edge(&god, root, (1, llm_bv3.clone()), n3, state_bv(&[3]));
        // root --(-1, bv3)--> n3 (this edge will be merged with the one above)
        add_edge(&god, root, (-1, llm_bv3.clone()), n3, state_bv(&[4]));

        // 2. ACT: Run the transformation
        eliminate_negative_pops_trie3(&god, &[root]);

        // 3. ASSERT: Verify the graph structure is correct
        let root_r = root.read(&god).unwrap();
        let n1_r = n1.read(&god).unwrap();

        // Assert root's children are correct after transformation and merging
        let expected_root_children = BTreeMap::from([
            // Edge to n1: pop was -1, is now 1.
            ((1, llm_bv1), OrderedHashMap::from([(n1, state_bv(&[1]))])),
            // Edge to n3: merged from pop 1 and pop -1. State BVs are OR'd.
            ((1, llm_bv3), OrderedHashMap::from([(n3, state_bv(&[3, 4]))])),
        ]);
        assert_eq!(root_r.children(), &expected_root_children);

        // Assert n1's children are correct
        let expected_n1_children = BTreeMap::from([
            // Edge to n2: pop was -2, is now 2.
            ((2, llm_bv2), OrderedHashMap::from([(n2, state_bv(&[2]))])),
        ]);
        assert_eq!(n1_r.children(), &expected_n1_children);
    }
}