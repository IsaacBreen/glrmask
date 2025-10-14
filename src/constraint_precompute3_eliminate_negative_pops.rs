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
    use std::collections::{BTreeSet, HashSet};

    // --- Test Helpers ---

    fn new_node(god: &Trie3GodWrapper) -> PrecomputeNode3Index {
        Trie2Index::new(god.insert(Trie::new(PrecomputedNodeContents::internal())))
    }

    fn state_bv(indices: &[usize]) -> StateIDBV {
        let mut bv = StateIDBV::zeros();
        for &i in indices {
            bv.insert(i);
        }
        bv
    }

    fn add_edge(
        god: &Trie3GodWrapper,
        from: PrecomputeNode3Index,
        key: (isize, LLMTokenBV),
        to: PrecomputeNode3Index,
        val: StateIDBV,
    ) {
        let mut from_w = from.write(god).unwrap();
        from_w.children_mut().entry(key).or_default().insert(to, val);
    }

    // --- Type Aliases for Readability ---
    type Path = Vec<(isize, LLMTokenBV)>;
    type PathSet = BTreeSet<Path>;

    // --- Tests ---

    #[test]
    fn test_eliminate_negative_pops_path_based() {
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
        add_edge(&god, root, (-1, llm_bv1.clone()), n1, state_bv(&[1]));
        add_edge(&god, n1, (-2, llm_bv2.clone()), n2, state_bv(&[2]));
        add_edge(&god, root, (1, llm_bv3.clone()), n3, state_bv(&[3]));
        add_edge(&god, root, (-1, llm_bv3.clone()), n3, state_bv(&[4]));

        // 2. ACT: Run the transformation
        eliminate_negative_pops_trie3(&god, &[root]);

        // 3. ASSERT: Verify the set of possible paths is correct
        let actual_paths = Trie::get_all_paths(&god, &[root]);

        let expected_paths = PathSet::from([
            // The root path is always present.
            vec![],
            // Path to n1: pop was -1, is now 1.
            vec![(1, llm_bv1.clone())],
            // Path to n2: pops were -1, -2, are now 1, 2.
            vec![(1, llm_bv1.clone()), (2, llm_bv2.clone())],
            // Path to n3: both (1, bv3) and (-1, bv3) now map to this single path.
            vec![(1, llm_bv3.clone())],
        ]);

        assert_eq!(actual_paths, expected_paths);
    }
}