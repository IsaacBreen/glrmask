use super::*;
use crate::constraint::{LLMTokenBV, PrecomputedNodeContents};
use crate::datastructures::gss::{allow_only_llm_tokens_and_prune_arc, get_roots, popn_collect_isolated_parents, print_gss_forest, process_predecessors, sample_path, Acc, GSSInternal, GSSNode, GSSPopper, GSSPrintConfig, GSSRoot, NodeMap, NodeSet, PruneAndTransformRecursiveMemo, StoredPrecomputeNode, StoredPrecomputeNodeIndex, StoredTrieGodWrapper};
use crate::datastructures::gss_pruning::prune_and_transform_recursive;
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use crate::glr::parser::ParseStateEdgeContent;
use crate::glr::table::StateID;
use bimap::BiBTreeMap;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

// Helper to create a local Acc that disallows a single token.
fn mock_acc(val: usize) -> Acc {
    let mut disallowed_bv = LLMTokenBV::zeros();
    disallowed_bv.insert(val);
    let allowed_bv = HybridBitset::max_ones() - disallowed_bv;
    Acc::new_with_local_constraints(allowed_bv, HybridL2Bitset::all())
}

fn empty_acc() -> Acc {
    Acc::new_fresh()
}

fn mock_edge(id: usize) -> ParseStateEdgeContent {
    ParseStateEdgeContent { state_id: StateID(id) }
}

#[test]
fn test_gss_new_node() {
    let acc = mock_acc(1);
    let node = GSSNode::new(acc.clone());
    assert_eq!(node.acc().llm_tokens_union, acc.llm_tokens_union);
    assert!(node.predecessors().is_empty());
    assert_eq!(node.max_depth(), 0);
}

#[test]
fn test_gss_push() {
    let root = Arc::new(GSSNode::new(mock_acc(1))); // Allows all but 1
    let pushed = root.push(mock_edge(10));

    assert_eq!(pushed.max_depth(), 1);

    // The new logic for `push` is to inherit the predecessor's acc, as the local acc is fresh.
    assert_eq!(*pushed.acc(), *root.acc());
}

#[test]
fn test_gss_pop() {
    let root = Arc::new(GSSNode::new(mock_acc(1))); // Allows all but 1
    let pushed = Arc::new(root.push(mock_edge(10))); // Now inherits root's acc.

    // Pop 1 level from `pushed`. The initial_acc is "fresh" (all allowed), so it doesn't constrain the path.
    let pop_result = pushed.popn(1);
    // We should not keep root nodes in paths.
    assert_eq!(pop_result.paths.len(), 0);
    assert_eq!(pop_result.below_bottom.len(), 1);
    // We reached the bottom exactly (depth 0).
    let combined_acc_map = pop_result.below_bottom.get(&1).unwrap(); // Depth 1 entry holds last-edge grouped map
    // The map should contain the edge 10 leading to the root
    let combined_acc = combined_acc_map.get(&mock_edge(10)).unwrap();

    // `pushed.acc` (same as `root.acc`) allows all but 1.
    // The narrowed union should allow all but 1.
    let mut disallowed = HybridBitset::zeros();
    disallowed.insert(1);
    let expected_allowed = HybridBitset::max_ones() - disallowed;
    assert_eq!(combined_acc.llm_tokens_union, expected_allowed);
}

#[test]
fn test_gss_merge() {
    let n0 = Arc::new(GSSNode::new(empty_acc()));
    let n1 = Arc::new(n0.push(mock_edge(0)));
    let n2 = Arc::new(n0.push(mock_edge(0)));

    let mut merged = (*n1).clone();
    merged.merge_with_depth(1, &n2);

    assert_eq!(merged.acc().llm_tokens_union, HybridBitset::max_ones());

    assert_eq!(merged.num_predecessors(), 1);
}

#[test]
fn test_popper_new_from_root_and_shift() {
    let root = Arc::new(GSSNode::new(mock_acc(1)));
    let mut popper = GSSPopper::new_from_node(root.clone(), Arc::new(Acc::new_fresh()));
    // Should not store roots in paths.
    assert!(popper.paths.is_empty());
    // Now below_bottom has an empty map at depth 0
    assert_eq!(popper.below_bottom.len(), 1);
    assert!(popper.below_bottom.get(&0).unwrap().is_empty());
    // Pop once; it shifts down since no edges are present
    popper.popn(1);
    assert!(popper.below_bottom.get(&0).is_none());
    assert!(popper.below_bottom.get(&1).unwrap().is_empty());
    // Pop two more steps; now it should be at 3 (still empty maps)
    popper.popn(2);
    assert!(popper.below_bottom.get(&1).is_none());
    assert!(popper.below_bottom.get(&3).unwrap().is_empty());
}

#[test]
fn test_popper_below_bottom_shifts_from_non_root() {
    let root = Arc::new(GSSNode::new(mock_acc(1)));
    let pushed = Arc::new(root.push(mock_edge(10)));
    let mut popper = pushed.popn(1); // Reaches bottom via edge 10

    assert!(popper.paths.is_empty());
    let by_edge_1 = popper.below_bottom.get(&1).expect("depth 1 entry missing");
    assert_eq!(by_edge_1.len(), 1);
    let acc0 = by_edge_1.get(&mock_edge(10)).expect("edge 10 missing at depth 1").clone();
    // Shift down by 2 more pops.
    popper.popn(2);
    assert!(popper.below_bottom.get(&1).is_none());
    let by_edge_3 = popper.below_bottom.get(&3).expect("depth 3 entry missing");
    let acc2 = by_edge_3.get(&mock_edge(10)).expect("edge 10 missing at depth 3").clone();
    assert_eq!(*acc0, *acc2);
}

#[test]
fn test_popper_merges_below_bottom_accs() {
    // Build a node that has two root predecessors with different disallowed tokens.
    let root1 = Arc::new(GSSNode::new(mock_acc(1))); // disallow token 1
    let root2 = Arc::new(GSSNode::new(mock_acc(2))); // disallow token 2
    let mut preds = NodeSet::new();
    preds.insert((root1.clone(), mock_edge(100)));
    preds.insert((root2.clone(), mock_edge(200)));
    let preds_map = process_predecessors(&preds);
    let parent = Arc::new(GSSNode::new_with_map(Arc::new(empty_acc()), preds_map));

    let (s, _) = print_gss_forest(
        &[parent.clone()], //
        &BiBTreeMap::new(),
        &GSSPrintConfig::default(),
    );
    println!("GSS Forest:\n{}", s);

    let popper = parent.popn(1);
    assert!(popper.paths.is_empty());
    let by_edge = popper.below_bottom.get(&1).expect("depth 1 entry missing");
    assert_eq!(by_edge.len(), 2);

    // Edge 100 (root1)
    {
        let acc_below_100 = by_edge.get(&mock_edge(100)).expect("edge 100 missing at depth 1");
        // Union should disallow token 1.
        let mut disallowed = HybridBitset::zeros();
        disallowed.insert(1);
        let expected_intersection = HybridBitset::max_ones() - disallowed;
        assert_eq!(acc_below_100.llm_tokens_union, expected_intersection);
    }

    // Edge 200 (root2)
    {
        let acc_below_200 = by_edge.get(&mock_edge(200)).expect("edge 200 missing at depth 1");
        let mut disallowed = HybridBitset::zeros();
        disallowed.insert(2);
        let expected_intersection = HybridBitset::max_ones() - disallowed;
        assert_eq!(acc_below_200.llm_tokens_union, expected_intersection);
    }
}

#[test]
fn test_gss_fuse_predecessors() {
    let leaf1 = Arc::new(GSSNode::new(mock_acc(1)));
    let leaf2 = Arc::new(GSSNode::new(mock_acc(2)));
    let b = Arc::new(leaf1.push(mock_edge(1)));
    let c_tmp = Arc::new(leaf2.push(mock_edge(2)));
    let c_tmp2 = Arc::new(c_tmp.push(mock_edge(3)));
    let c = Arc::new(c_tmp2.push(mock_edge(4)));

    assert_eq!(b.max_depth(), 1);
    assert_eq!(c.max_depth(), 3);

    let mut preds_map = NodeMap::new();
    preds_map.entry(mock_edge(100)).or_default().insert(b.dest_key(), vec![b.clone()]);
    preds_map.entry(mock_edge(100)).or_default().insert(c.dest_key(), vec![c.clone()]);

    let mut root = GSSNode::new_with_map(Arc::new(empty_acc()), preds_map);
    assert_eq!(root.num_predecessors(), 2);

    root.fuse_predecessors(1);

    assert_eq!(root.num_predecessors(), 1);
    let fused_pred_arc = root
        .predecessors()
        .values()
        .next()
        .unwrap()
        .values()
        .next()
        .unwrap()[0]
        .clone();

    assert_eq!(fused_pred_arc.acc().llm_tokens_union, HybridBitset::max_ones());
    assert_eq!(fused_pred_arc.num_predecessors(), 2);
}

#[test]
fn test_sample_path() {
    let d = Arc::new(GSSNode::new(empty_acc()));
    let e = Arc::new(GSSNode::new(empty_acc()));

    let mut c_preds = NodeSet::new();
    c_preds.insert((d, mock_edge(30)));
    c_preds.insert((e, mock_edge(40)));
    let c_preds_map = process_predecessors(&c_preds);
    let c = Arc::new(GSSNode::new_with_map(Arc::new(empty_acc()), c_preds_map));

    let b = Arc::new(c.push(mock_edge(20)));
    let root = b.push(mock_edge(10));

    let path1 = sample_path(&[&root], 0).unwrap();
    // let path2 = sample_path(&[&root], 1).unwrap();

    assert_eq!(path1.len(), 3);
    assert_eq!(path1[0], mock_edge(10));
    assert_eq!(path1[1], mock_edge(20));
    assert!(path1[2] == mock_edge(30) || path1[2] == mock_edge(40));

    let path1_again = sample_path(&[&root], 0).unwrap();
    assert_eq!(path1, path1_again);
}

#[test]
fn test_prune_and_transform_noop_does_not_merge_distinct_predecessors() {
    // This test checks for a bug where prune_and_transform_recursive with a no-op
    // closure would still modify the GSS by merging structurally distinct predecessor
    // nodes that happen to share the same edge value and depth.

    // 1. Create two distinct leaf nodes.
    let leaf1 = Arc::new(GSSNode::new(mock_acc(1)));
    let leaf2 = Arc::new(GSSNode::new(mock_acc(2)));

    // 2. Create two intermediate nodes that are structurally different because they
    // have different predecessors.
    let intermediate1 = Arc::new(leaf1.push(mock_edge(10)));
    let intermediate2 = Arc::new(leaf2.push(mock_edge(10)));
    assert_ne!(*intermediate1, *intermediate2, "Intermediates should be structurally different");
    assert_eq!(intermediate1.max_depth(), 1);
    assert_eq!(intermediate2.max_depth(), 1);

    // 3. Manually construct a root node that has both intermediates as predecessors
    // under the same edge value and at the same depth. This structure is key to
    // reproducing the bug. The `Vec` in the NodeMap contains multiple distinct nodes.
    let mut root_preds = NodeMap::new();
    root_preds
        .entry(mock_edge(100))
        .or_default()
        .insert(1, vec![intermediate1.clone(), intermediate2.clone()]);

    let root = Arc::new(GSSNode::new_with_map(Arc::new(empty_acc()), root_preds));
    assert_eq!(root.num_predecessors(), 2);

    // 4. Run prune_and_transform_recursive with a no-op closure.
    // This should not change the structure of the GSS at all.
    let mut memo = PruneAndTransformRecursiveMemo::new();
    let new_root_opt = prune_and_transform_recursive(
        &root,
        &mut |internal: &GSSInternal| Some((internal.acc().clone(), true)), // don't prune internal nodes
        &mut |root_node: &GSSRoot| Some(root_node.acc().clone()), // No-op: keep root
        &mut memo,
    );

    // 5. Assert that the structure is unchanged.
    let new_root = new_root_opt.expect("Root should not be pruned");

    // Check full equality for good measure. This is the most important check.
    // With the bug, this fails because the new_root will have its predecessors merged.
    assert_eq!(*root, *new_root, "The GSS structure should be identical after a no-op transform");
    assert_eq!(new_root.num_predecessors(), 2, "Should still have 2 predecessors");
}

#[test]
fn test_merge_preserves_stored_trie_nodes() {
    // This test reproduces a bug where merging GSS nodes would cause
    // stored_trie_nodes from leaf predecessors to be lost due to incorrect
    // constraint propagation (narrowing).

    // --- GSS 1 Setup ---
    let stored_trie_god = StoredTrieGodWrapper::new();
    let stored_trie_node1 = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));
    let stored_trie_node2 = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));
    let stored_trie_node3 = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));

    let mut acc_l1 = empty_acc();
    acc_l1.stored_trie_nodes_mut().insert(stored_trie_node1.clone());
    let l1 = Arc::new(GSSNode::new(acc_l1));

    let mut acc_l2 = empty_acc();
    acc_l2.stored_trie_nodes_mut().insert(stored_trie_node2.clone());
    let l2 = Arc::new(GSSNode::new(acc_l2));

    let mut acc_l3 = empty_acc();
    acc_l3.stored_trie_nodes_mut().insert(stored_trie_node3.clone());
    let l3 = Arc::new(GSSNode::new(acc_l3));

    let mut gss1_preds = NodeMap::new();
    gss1_preds.entry(mock_edge(0)).or_default().insert(l1.dest_key(), vec![l1.clone()]);
    gss1_preds.entry(mock_edge(1)).or_default().insert(l2.dest_key(), vec![l2.clone()]);
    gss1_preds.entry(mock_edge(2)).or_default().insert(l3.dest_key(), vec![l3.clone()]);

    let mut gss1 = GSSNode::new_with_map(Arc::new(mock_acc(0)), gss1_preds); // mock_acc(0) restricts token 0

    // --- GSS 2 Setup ---
    let mut acc_l4 = empty_acc();
    acc_l4.stored_trie_nodes_mut().insert(stored_trie_node1.clone()); // Shared stored_trie_node
    let l4 = Arc::new(GSSNode::new(acc_l4));
    let i1 = Arc::new(l4.push(mock_edge(0)));
    let gss2 = i1.push(mock_edge(1));

    // --- Merge ---
    gss1.merge_with_depth(1, &gss2);

    // --- Assertions ---
    // Traverse the merged GSS and collect all stored_trie_nodes from all leaf nodes.
    let mut q = VecDeque::new();
    q.push_back(Arc::new(gss1));
    let mut visited = HashSet::new();
    let mut final_leaf_stored_trie_nodes = BTreeSet::new();

    while let Some(node) = q.pop_front() {
        if !visited.insert(Arc::as_ptr(&node)) { continue; }
        if node.is_root() { final_leaf_stored_trie_nodes.extend(node.acc().stored_trie_nodes().clone()); }
        for p in node.predecessors().values().flat_map(|m| m.values()).flatten() { q.push_back(p.clone()); }
    }

    assert!(final_leaf_stored_trie_nodes.contains(&stored_trie_node1), "stored_trie_node1 missing");
    assert!(final_leaf_stored_trie_nodes.contains(&stored_trie_node2), "stored_trie_node2 missing");
    assert!(final_leaf_stored_trie_nodes.contains(&stored_trie_node3), "stored_trie_node3 missing");
    assert_eq!(final_leaf_stored_trie_nodes.len(), 3, "Should have 3 unique stored_trie nodes in the leaves");
}

#[test]
fn test_merge_does_not_incorrectly_collapse_branches() {
    // This test reproduces a bug where merging two GSSs with a common edge value
    // but different sub-structures would incorrectly collapse the distinct sub-structures.

    // --- Shared Nodes ---
    let stored_trie_god = StoredTrieGodWrapper::new();
    let stored_trie_node1 = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));
    let mut acc1 = empty_acc();
    acc1.stored_trie_nodes_mut().insert(stored_trie_node1.clone());
    let leaf1 = Arc::new(GSSNode::new(acc1)); // This is "Node 2" with trie ...6f0

    let stored_trie_node2 = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));
    let mut acc2 = empty_acc();
    acc2.stored_trie_nodes_mut().insert(stored_trie_node2.clone());
    let leaf2 = Arc::new(GSSNode::new(acc2)); // This is "Node 2" with trie ...560

    // --- GSS A ---
    // Root -> (edge 1) -> leaf1
    let gss_a = GSSNode::new_with_single_predecessor(
        leaf1.clone(),
        mock_edge(1),
        empty_acc(),
    );

    // --- GSS B ---
    // intermediate -> (edge 0) -> leaf2
    let intermediate_b = Arc::new(GSSNode::new_with_single_predecessor(
        leaf2.clone(),
        mock_edge(0),
        empty_acc(),
    ));
    // Root -> (edge 1) -> leaf1
    //      -> (edge 1) -> intermediate
    let mut gss_b_preds = NodeMap::new();
    gss_b_preds.entry(mock_edge(1)).or_default().insert(leaf1.dest_key(), vec![leaf1.clone()]);
    gss_b_preds.entry(mock_edge(1)).or_default().insert(intermediate_b.dest_key(), vec![intermediate_b.clone()]);
    let gss_b = GSSNode::new_with_map(Arc::new(empty_acc()), gss_b_preds);

    // --- Merge ---
    let mut merged_gss = gss_a.clone();
    merged_gss.merge_with_depth(usize::MAX, &gss_b);

    // --- Assertions ---
    // The merged GSS should have two distinct predecessors under edge 1, because
    // they have different depths and structures. The incorrect behavior collapses them into one.
    assert_eq!(merged_gss.num_predecessors(), 2, "Merged GSS should have two predecessors");

    let preds_for_edge1 = merged_gss.predecessors().get(&mock_edge(1)).expect("Edge 1 should exist");
    assert_eq!(preds_for_edge1.len(), 2, "Edge 1 should have predecessors at two different depths");
}

#[test]
fn test_merge_with_different_depth_predecessors() {
    // This test reproduces a bug where merging two GSSs with a common edge value
    // but different sub-structures would incorrectly collapse the distinct sub-structures.
    // GSS A: Root -> (edge 1) -> leaf_a
    // GSS B: Root -> (edge 1) -> intermediate_b -> (edge 0) -> leaf_b
    // Merged should have two predecessors from root via edge 1, at different depths.

    // --- GSS A setup ---
    let stored_trie_god = StoredTrieGodWrapper::new();
    let stored_trie_node_a = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));
    let mut acc_a = empty_acc();
    acc_a.stored_trie_nodes_mut().insert(stored_trie_node_a.clone());
    let leaf_a = Arc::new(GSSNode::new(acc_a));

    let gss_a = GSSNode::new_with_single_predecessor(
        leaf_a.clone(),
        mock_edge(1),
        empty_acc(),
    );

    // --- GSS B setup ---
    let stored_trie_node_b = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));
    let mut acc_b = empty_acc();
    acc_b.stored_trie_nodes_mut().insert(stored_trie_node_b.clone());
    let leaf_b = Arc::new(GSSNode::new(acc_b));

    let intermediate_b = Arc::new(GSSNode::new_with_single_predecessor(
        leaf_b.clone(),
        mock_edge(0),
        empty_acc(),
    ));
    let gss_b = GSSNode::new_with_single_predecessor(intermediate_b.clone(), mock_edge(1), empty_acc());

    // --- Merge ---
    let mut merged_gss = gss_a.clone();
    merged_gss.merge_with_depth(usize::MAX, &gss_b);

    // --- Assertions ---
    // The merged GSS should have two distinct predecessors under edge 1, because
    // they have different depths and structures. The incorrect behavior collapses them into one.
    assert_eq!(merged_gss.num_predecessors(), 2, "Merged GSS should have two predecessors");

    let preds_for_edge1 = merged_gss.predecessors().get(&mock_edge(1)).expect("Edge 1 should exist");
    assert_eq!(preds_for_edge1.len(), 2, "Edge 1 should have predecessors at two different depths");
}

#[test]
fn test_merge_unions_stored_trie_nodes_across_identical_towers() {
    // This test reproduces a bug where merging multiple identical towers (same edges and structure)
    // but with different stored_trie_nodes at the leaf results in the leaf keeping only one
    // of the stored_trie_nodes instead of the union of all of them.
    //
    // Structure for each tower:
    // Root -> (edge 2) -> ... -> Leaf [Trie={unique}]
    //
    // After merging two such towers, the single leaf should contain the union of the two distinct
    // of the stored_trie_nodes instead of the union of all of them.

    // --- Build two distinct stored_trie nodes ---
    let stored_trie_god = StoredTrieGodWrapper::new();
    let t1 = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));
    let t2 = StoredPrecomputeNodeIndex::new(stored_trie_god.insert(StoredPrecomputeNode::new(PrecomputedNodeContents::internal())));

    // Helper to build one tower given a leaf with a unique stored_trie node.
    let build_tower_from_leaf = |leaf: Arc<GSSNode>| -> GSSNode {
        let n5 = Arc::new(GSSNode::new_with_single_predecessor(leaf, mock_edge(5), empty_acc()));
        let n1 = Arc::new(n5.push(mock_edge(1)));
        n1.push(mock_edge(2))
    };

    // --- Leaf 1 with stored_trie_node t1 ---
    let mut acc1 = empty_acc();
    acc1.stored_trie_nodes_mut().insert(t1.clone());
    let leaf1 = Arc::new(GSSNode::new(acc1.clone()));
    let tower1 = build_tower_from_leaf(leaf1);

    // --- Leaf 2 with stored_trie_node t2 ---
    let mut acc2 = empty_acc();
    acc2.stored_trie_nodes_mut().insert(t2.clone());
    let leaf2 = Arc::new(GSSNode::new(acc2.clone()));
    let tower2 = build_tower_from_leaf(leaf2);

    // --- Merge the two identical towers ---
    let mut merged = tower1.clone();
    merged.merge_with_depth(usize::MAX, &tower2);

    // --- Assertions ---
    // With the new hoisting logic, the merged acc from the leaves should be hoisted
    // all the way to the top-level node of the merged tower.
    let final_acc = Acc::merge(&acc1, &acc2);
    let stored_trie_nodes = final_acc.stored_trie_nodes();

    assert_eq!(stored_trie_nodes.len(), 2, "Merged tower root should contain the union of stored_trie nodes from the leaves");
    assert!(stored_trie_nodes.contains(&t1), "Merged acc missing stored_trie node 1");
    assert!(stored_trie_nodes.contains(&t2), "Merged acc missing stored_trie node 2");

    // --- New assertions ---
    // 1. Check get_roots
    let roots_map = get_roots(std::iter::once(&merged));
    assert_eq!(roots_map.len(), 1, "get_roots should find one root path");
    let (last_edge, acc_set) = roots_map.iter().next().unwrap();
    assert_eq!(*last_edge, mock_edge(5));
    assert_eq!(acc_set.len(), 1, "There should be one unique path acc");
    let path_acc = acc_set.iter().next().unwrap();
    assert_eq!(**path_acc, final_acc, "Path acc from get_roots should match the hoisted acc");

    // 2. Check popping
    let tower_depth = 3;
    let popper = merged.popn(tower_depth + 1); // Pop one level past the bottom
    assert!(popper.paths.is_empty(), "Popper paths should be empty after popping past bottom");
    assert_eq!(popper.below_bottom.len(), 1, "Should have one entry in below_bottom");
    let (depth, by_edge) = popper.below_bottom.iter().next().unwrap();
    assert_eq!(*depth, 2, "Popping 1 level past bottom should result in depth key 2");
    assert_eq!(by_edge.len(), 1, "Should be one edge leading to bottom");
    let (edge, acc) = by_edge.iter().next().unwrap();
    assert_eq!(*edge, mock_edge(5));
    assert_eq!(**acc, final_acc, "Acc from popping past bottom should match hoisted acc");
}

#[test]
fn test_allow_only_llm_tokens_and_prune_arc_simple_tower() {
    // This test is based on a real-world bug where filtering did not seem to apply.
    // Structure: Root -> (edge 2) -> Node 1 -> (edge 0) -> Node 2 (leaf)

    // 1. Build the GSS tower.
    let leaf = Arc::new(GSSNode::new(empty_acc())); // Node 2
    let intermediate = Arc::new(leaf.push(mock_edge(0))); // Node 1
    let mut root_arc = Arc::new(intermediate.push(mock_edge(2))); // Root 0

    // 2. Check initial state.
    assert_eq!(
        root_arc.allowed_llm_tokens(),
        HybridBitset::max_ones(),
        "Initial allowed tokens should be everything"
    );

    // 3. Filter to allow only token 0.
    let mut allowed_tokens = LLMTokenBV::zeros();
    allowed_tokens.insert(0);
    allow_only_llm_tokens_and_prune_arc(&mut root_arc, &allowed_tokens, &mut HashMap::new());

    // 4. Assert that the allowed tokens for the whole GSS have been updated.
    assert_eq!(
        root_arc.allowed_llm_tokens(),
        allowed_tokens,
        "Allowed tokens should be restricted to only token 0 after filtering"
    );
}

#[test]
fn test_popn_collect_isolated_parents_preserves_acc() {
    // Setup: Root(acc42) -> Intermediate(empty) -> Leaf(empty)
    let leaf = Arc::new(GSSNode::new(empty_acc()));
    let intermediate = Arc::new(leaf.push(mock_edge(10)));
    assert!(intermediate.local_acc().is_default());

    let root_acc = mock_acc(42);
    let root = Arc::new(GSSNode::new_with_single_predecessor(
        intermediate.clone(),
        mock_edge(20),
        root_acc.clone(),
    ));
    assert_eq!(*root.local_acc(), root_acc);

    // Action: pop 1 level. We expect to get back a node representing the `intermediate`
    // node, but with the path constraint from the root applied.
    let result = popn_collect_isolated_parents(&root, 1);

    assert_eq!(result.len(), 1);
    let (_state_id, isolated_parent) = &result[0];

    // Validation: The `isolated_parent` is a reconstruction of the `intermediate` node's
    // path to its predecessor (`leaf`). The overall `acc()` of this new structure
    // should reflect the constraint from the popped `root`.
    let final_acc = isolated_parent.acc();

    // The final acc should be the intersection of the root's acc and the rest of the path.
    // Since the rest of the path is empty, it should just be the root's acc.
    assert_eq!(*final_acc, root_acc);
    assert!(!final_acc.llm_tokens_union.contains(42));
}