// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(false)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::collections::{HashSet, HashMap};
    use crate::datastructures::hybrid_bitset::HybridBitset; // Import HybridBitset for tests
    use std::iter::FromIterator; // For collect

    // Use concrete types for merge tests
    type TestTrieMerge = Trie<&'static str, Vec<i32>, String>;
    type TestNodeMerge = Arc<RwLock<TestTrieMerge>>;
    // Use simpler types for basic tests
    type TestTrieBasic = Trie<&'static str, &'static str, i32>;
    type TestNodeBasic = Arc<RwLock<TestTrieBasic>>;

    // Use concrete types for EdgeInserter tests
    type TestTrieEI = Trie<&'static str, HybridBitset, String>; // Use HybridBitset here
    type TestNodeEI = Arc<RwLock<TestTrieEI>>;

    // Helper to get Arc pointer for tests
    fn arc_ptr<N>(arc: &Arc<RwLock<N>>) -> *const RwLock<N> {
        Arc::as_ptr(arc)
    }

    #[test]
    fn test_try_insertion_and_retrieval() {
        let root_node: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let child3: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3))); // Another child for 'a'

        { // Scope for mutable borrow of root
            let mut root = root_node.write().unwrap();
            root.try_insert("a", &mut Some("edge_a1"), child1.clone()).expect("Insert failed");
            root.try_insert("b", &mut Some( "edge_b"), child2.clone()).expect("Insert failed");
            root.try_insert("a", &mut Some("edge_a3"), child3.clone()).expect("Insert failed"); // Insert second child for 'a'
        } // root lock released

        // Scope for read-only borrow of root
        let root = root_node.read().unwrap();

        // Test get for 'a'
        let retrieved_children_a = root.get(&"a").expect("Failed to get children for 'a'"); // Now a &BTreeMap<ArcPtrWrapper<Mutex<...>>, EV>
        assert_eq!(retrieved_children_a.len(), 2);
        // Use Arc pointers for comparison
        let retrieved_data_a: HashSet<(&str, *const RwLock<TestTrieBasic>)> = retrieved_children_a
            .iter() // Iterates yielding (&ArcPtrWrapper<...>, &&str)
            .map(|(node_ptr, ev_ref)| (*ev_ref, arc_ptr(node_ptr.as_arc()))) // Dereference ev_ref twice
            .collect();
        assert!(retrieved_data_a.contains(&("edge_a1", arc_ptr(&child1))));
        assert!(retrieved_data_a.contains(&("edge_a3", arc_ptr(&child3))));

        // Test get for 'b'
        let retrieved_children_b = root.children().get(&"b").expect("Failed to get child 'b'"); // Now a &BTreeMap
        assert_eq!(retrieved_children_b.len(), 1);
        let (node_ptr, ev_ref) = retrieved_children_b.iter().next().unwrap(); // Get the single entry
        assert_eq!(*ev_ref, "edge_b"); // Check edge value
        assert!(Arc::ptr_eq(node_ptr.as_arc(), &child2)); // Check Arc pointer equality

        assert!(root.get(&"c").is_none());

        // Test children iterator order (BTreeMap ensures sorted order of keys 'a', 'b')
        let children_keys: Vec<_> = root.children().keys().cloned().collect();
        assert_eq!(children_keys, vec!["a", "b"]);
        assert_eq!(root.children().get("a").unwrap().len(), 2);
        assert_eq!(root.children().get("b").unwrap().len(), 1);

        // Test is_leaf
        assert!(!root.is_leaf());
        // Drop root lock before locking children
        drop(root);
        assert!(child1.read().unwrap().is_leaf());
        assert!(child2.read().unwrap().is_leaf());
        assert!(child3.read().unwrap().is_leaf());
    }

    #[test]
    fn test_multiple_children_same_edge_key() {
        // Structure:
        //      root (0) --"edge", "val1"--> child1 (1)
        //           |
        //            -----"edge", "val2"--> child2 (2)
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));

        {
            let mut r = root.write().unwrap();
            r.try_insert("edge", &mut Some("val1"), child1.clone()).unwrap();
            r.try_insert("edge", &mut Some("val2"), child2.clone()).unwrap();
        } // root lock released

        // Check retrieval - lock root again
        {
            let binding = root.read().unwrap();
            let children_map = binding.get(&"edge").unwrap(); // Now a &BTreeMap<ArcPtrWrapper<Mutex<...>>, EV>
            assert_eq!(children_map.len(), 2);
            let child_data: HashSet<(&str, *const RwLock<TestTrieBasic>)> = children_map
                .iter() // Iterating over (&ArcPtrWrapper<...>, &EV)
                .map(|(node_ptr, ev_ref)| (*ev_ref, arc_ptr(node_ptr.as_arc())))
                .collect();
            assert!(child_data.contains(&("val1", arc_ptr(&child1))));
            assert!(child_data.contains(&("val2", arc_ptr(&child2))));
        } // root lock released

        // Check all_nodes - call *after* releasing lock
        let all = Trie::all_nodes(&[root.clone()]);
        assert_eq!(all.len(), 3); // root, child1, child2
        let all_ptrs: HashSet<_> = all.iter().map(arc_ptr).collect();
        assert!(all_ptrs.contains(&arc_ptr(&root)));
        assert!(all_ptrs.contains(&arc_ptr(&child1)));
        assert!(all_ptrs.contains(&arc_ptr(&child2)));

        // Check special_map
        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();
        let mut edge_info_at_step = Vec::new(); // Store (EK, EV) seen by step

        Trie::special_map(
            vec![(root.clone(), 100)],
            // step: add one, ignore edge info
            |parent_val, ek, ev, _child_node| {
                 edge_info_at_step.push((ek.clone(), ev.clone()));
                 Some(parent_val + 1)
            },
            |current, new| *current = new, // merge: replace
            |node, computed_val| { // process: always continue
                processed_node_values.push(node.value);
                computed_values.push(*computed_val);
                true
            },
        );

        // Expected processing order: 0, then (1, 2) in some order based on depth.
        assert_eq!(processed_node_values.len(), 3);
        assert!(processed_node_values.contains(&0));
        assert!(processed_node_values.contains(&1));
        assert!(processed_node_values.contains(&2));
        // Depth 0 nodes processed first
        assert_eq!(processed_node_values[0], 0);
        // Depth 1 nodes processed next (order not guaranteed for equal depth)
        let depth1_nodes: HashSet<_> = processed_node_values[1..].iter().cloned().collect();
        assert!(depth1_nodes.contains(&1));
        assert!(depth1_nodes.contains(&2));


        // Expected computed values: root = 100, child1 = 101, child2 = 101.
        assert_eq!(computed_values.len(), 3);
        assert_eq!(computed_values[0], 100);
        let results_map: HashMap<i32, i32> = processed_node_values.iter().cloned().zip(computed_values.iter().cloned()).collect();
        assert_eq!(results_map.get(&0), Some(&100));
        assert_eq!(results_map.get(&1), Some(&101));
        assert_eq!(results_map.get(&2), Some(&101));

        // Check edge info captured by step
        assert_eq!(edge_info_at_step.len(), 2); // 2 edges traversed from root
        assert!(edge_info_at_step.contains(&("edge", "val1")));
        assert!(edge_info_at_step.contains(&("edge", "val2")));
    }


    #[test]
    fn test_special_map_bfs_order_with_edges() {
        // Structure:
        //      root (0)
        //       /       \
        // ("r->c1","e1") ("r->c2","e2")
        //     /           \
        //   c1 (1)       c2 (2)
        //      |
        // ("c1->gc","e3")
        //      |
        //   gc (3)
        //
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));

        {
            let mut r = root.write().unwrap();
            r.try_insert("r->c1", &mut Some("e1"), child1.clone()).unwrap();
            r.try_insert("r->c2", &mut Some("e2"), child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.write().unwrap();
            c1.try_insert("c1->gc", &mut Some("e3"), grandchild.clone()).unwrap();
        }
         // No edge from c2 to grandchild in this test setup, removed the line below
        // {
        //     let mut c2 = child2.lock().unwrap();
        //     c2.try_insert("c2->gc", &mut Some("e4"), grandchild.clone()).unwrap();
        // }


        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();
        let mut edge_info_at_step = Vec::new(); // Store (EK, EV) seen by step

        Trie::special_map(
            vec![(root.clone(), 100)],
            // step: add one, record edge info
            |parent_val, ek, ev, _child_node| {
                edge_info_at_step.push((ek.clone(), ev.clone()));
                Some(parent_val + 1)
            },
            // merge: replace
            |current, new| { *current = new; },
            // process: always continue
            |node, computed_val| {
                processed_node_values.push(node.value);
                computed_values.push(*computed_val);
                true
            },
        );

        // Check processing order (by depth)
        // Depth 0: root (0)
        // Depth 1: child1 (1), child2 (2) - order depends on heap
        // Depth 2: grandchild (3)
        assert_eq!(processed_node_values.len(), 4);
        assert_eq!(processed_node_values[0], 0); // Root (depth 0) is first
        let depth1_nodes: HashSet<_> = processed_node_values[1..3].iter().cloned().collect();
        assert!(depth1_nodes.contains(&1));
        assert!(depth1_nodes.contains(&2));
        assert_eq!(processed_node_values[3], 3); // Grandchild (depth 2) is last


        // Check computed values
        let results_map: HashMap<i32, i32> = processed_node_values.iter().cloned()
            .zip(computed_values.iter().cloned()).collect();
        assert_eq!(results_map.get(&0), Some(&100));
        assert_eq!(results_map.get(&1), Some(&101));
        assert_eq!(results_map.get(&2), Some(&101));
        assert_eq!(results_map.get(&3), Some(&102)); // Reached from c1 (101+1)

        // Check edge info captured by step
        assert_eq!(edge_info_at_step.len(), 3); // 3 edges traversed (r->c1, r->c2, c1->gc)
        assert!(edge_info_at_step.contains(&("r->c1", "e1")));
        assert!(edge_info_at_step.contains(&("r->c2", "e2")));
        assert!(edge_info_at_step.contains(&("c1->gc", "e3")));
    }

    #[test]
    fn test_all_nodes_diamond() {
        // Diamond structure:
        //       root
        //      /    \
        // ("r1","e1") ("r2","e2")
        //    /        \
        // child1    child2
        //    \        /
        // ("c1","e3") ("c2","e4")
        //      \    /
        //    grandchild
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));

        {
            let mut r = root.write().unwrap();
            r.try_insert("r1", &mut Some("e1"), child1.clone()).unwrap();
            r.try_insert("r2", &mut Some("e2"), child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.write().unwrap();
            c1.try_insert("c1", &mut Some("e3"), grandchild.clone()).unwrap();
        }
        {
            let mut c2 = child2.write().unwrap();
            c2.try_insert("c2", &mut Some("e4"), grandchild.clone()).unwrap(); // Diamond
        }

        let all_nodes = Trie::all_nodes(&[root.clone()]);

        // Should find 4 unique nodes.
        assert_eq!(all_nodes.len(), 4);
        let node_ptrs: HashSet<_> = all_nodes.iter().map(arc_ptr).collect(); // Use arc_ptr
        assert_eq!(node_ptrs.len(), 4);
        assert!(node_ptrs.contains(&arc_ptr(&root)));
        assert!(node_ptrs.contains(&arc_ptr(&child1)));
        assert!(node_ptrs.contains(&arc_ptr(&child2)));
        assert!(node_ptrs.contains(&arc_ptr(&grandchild)));
    }

    #[test]
    fn test_special_map_diamond_merge_max() {
        // Diamond structure
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));

        // Build the structure
        {
            let mut r = root.write().unwrap();
            r.try_insert("r->c1", &mut Some("edge1"), child1.clone()).unwrap();
            r.try_insert("r->c2", &mut Some("edge2"), child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.write().unwrap();
            c1.try_insert("c1->gc", &mut Some("edge3"), grandchild.clone()).unwrap();
        }
        {
            let mut c2 = child2.write().unwrap();
            c2.try_insert("c2->gc", &mut Some("edge4"), grandchild.clone()).unwrap();
        }

        // Check max_depths after insertion
        assert_eq!(root.read().unwrap().max_depth, 0);
        assert_eq!(child1.read().unwrap().max_depth, 1);
        assert_eq!(child2.read().unwrap().max_depth, 1);
        assert_eq!(grandchild.read().unwrap().max_depth, 2);

        let processed_nodes = Arc::new(RwLock::new(HashMap::<i32, i32>::new()));
        let process_count = Arc::new(AtomicUsize::new(0));

        Trie::special_map(
            vec![(root.clone(), 100)], // Start at root
            // step: increment value, ignore edges
            |p_val, _ek, _ev, _child_node| Some(p_val + 1),
            // merge: take max value
            |current_v, new_v| *current_v = (*current_v).max(new_v),
            { // process: always continue
                let processed_nodes = processed_nodes.clone();
                let process_count = process_count.clone();
                move |node, final_v| {
                    let mut map = processed_nodes.write().unwrap();
                    map.insert(node.value, *final_v);
                    process_count.fetch_add(1, Ordering::SeqCst);
                    true
                }
            }
        );

        // Assertions
        let final_results = processed_nodes.read().unwrap();
        assert_eq!(process_count.load(Ordering::SeqCst), 4, "Should process 4 unique nodes");
        assert_eq!(final_results.get(&0), Some(&100));
        assert_eq!(final_results.get(&1), Some(&101));
        assert_eq!(final_results.get(&2), Some(&101));
        assert_eq!(final_results.get(&3), Some(&102)); // gc gets max(101+1, 101+1) = 102
    }


    #[test]
    fn test_empty_trie() {
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(42)));
        let nodes = Trie::all_nodes(&[root.clone()]);
        assert_eq!(nodes.len(), 1);
        assert!(Arc::ptr_eq(&nodes[0], &root));
        assert!(root.read().unwrap().is_leaf()); // Lock needed here

        let mut processed = false;
        Trie::special_map(
            vec![(root.clone(), 100)],
            |_p, _ek, _ev, _n| panic!("Step should not be called for leaf"),
            |_cur, _new| {},
            |node, v| { // process: always continue
                assert_eq!(node.value, 42);
                assert_eq!(*v, 100);
                processed = true;
                true
            },
        );
        assert!(processed);
    }

    #[test]
    fn test_cycle_detection_on_try_insert() {
        // Cycle:  root -> child -> root
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));

        // Insert root -> child
        let insert1_result = {
            let mut r = root.write().unwrap();
            r.try_insert("r->c", &mut Some("e1"), child.clone())
        };
        assert!(insert1_result.is_ok());
        assert_eq!(child.read().unwrap().max_depth, 1);
        assert_eq!(root.read().unwrap().max_depth, 0);

        // Attempt insert child -> root
        let insert2_result = {
            let mut c = child.write().unwrap();
            // This insert should call detect_cycle(child_ptr, &root), which should detect the cycle.
            c.try_insert("c->r", &mut Some("e2"), root.clone())
        };

        // Assert that cycle detection returned an error
        assert!(insert2_result.is_err());
        assert_eq!(insert2_result.err(), Some(CycleDetectedError));

        // Check state after failed insertion:
        // - The edge must *not* be present because the insertion was rejected.
        let child_locked = child.read().unwrap();
        let has_edge_to_root = if let Some(dest_map) = child_locked.children.get("c->r") {
            let lookup_key = ArcPtrWrapper::new(root.clone()); // Use ArcPtrWrapper
            dest_map.contains_key(&lookup_key)
         } else {
             false
         };
        assert!(!has_edge_to_root, "Edge that would introduce a cycle should NOT be present");

        // - Max depths should be unchanged from before the failed insertion attempt.
        assert_eq!(root.read().unwrap().max_depth, 0);
        assert_eq!(child_locked.max_depth, 1);

        println!("Done testing cycle detection on try_insert");
    }


    #[test]
    fn test_cycle_all_nodes_no_panic() {
        // Cycle:  root -> child -> root.
        // Manually create cycle without insert's propagation.
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));

        // Manually create links
        root.write().unwrap().force_insert_to_node("r->c", "e1", &child);
        child.write().unwrap().force_insert_to_node("c->r", "e2", &root);
        // Manually set depths (optional for all_nodes logic)
        root.write().unwrap().max_depth = 0;
        child.write().unwrap().max_depth = 1;

        let all_nodes = Trie::all_nodes(&[root.clone()]);

        // Should detect both nodes exactly once.
        assert_eq!(all_nodes.len(), 2);
        let node_ptrs: HashSet<_> = all_nodes.iter().map(arc_ptr).collect(); // Use arc_ptr
        assert_eq!(node_ptrs.len(), 2);
        assert!(node_ptrs.contains(&arc_ptr(&root)));
        assert!(node_ptrs.contains(&arc_ptr(&child)));
    }

     #[test]
    fn test_has_any_cycle() {
        // No cycle
        let root1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));
        root1.write().unwrap().force_insert_to_node("a", "e1", &child1);
        root1.write().unwrap().force_insert_to_node("b", "e2", &child2);
        child1.write().unwrap().force_insert_to_node("c", "e3", &grandchild);
        child2.write().unwrap().force_insert_to_node("d", "e4", &grandchild); // Diamond
        assert!(!Trie::has_any_cycle(root1.clone()));

        // Simple cycle: root2 -> child3 -> root2
        let root2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(10)));
        let child3: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(11)));
        root2.write().unwrap().force_insert_to_node("x", "e5", &child3);
        child3.write().unwrap().force_insert_to_node("y", "e6", &root2);
        assert!(Trie::has_any_cycle(root2.clone()));

        // Larger cycle: root3 -> A -> B -> C -> A
        let root3: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(20)));
        let node_a: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(21)));
        let node_b: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(22)));
        let node_c: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(23)));
        root3.write().unwrap().force_insert_to_node("r->a", "e7", &node_a);
        node_a.write().unwrap().force_insert_to_node("a->b", "e8", &node_b);
        node_b.write().unwrap().force_insert_to_node("b->c", "e9", &node_c);
        node_c.write().unwrap().force_insert_to_node("c->a", "e10", &node_a); // Cycle C -> A
        assert!(Trie::has_any_cycle(root3.clone()));

        // Cycle with unconnected node: root4 -> A -> B -> A; C (unconnected)
        let root4: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(30)));
        let node_a2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(31)));
        let node_b2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(32)));
        let node_c2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(33))); // Unconnected to root4
        root4.write().unwrap().force_insert_to_node("r->a", "e11", &node_a2);
        node_a2.write().unwrap().force_insert_to_node("a->b", "e12", &node_b2);
        node_b2.write().unwrap().force_insert_to_node("b->a", "e13", &node_a2); // Cycle B -> A
        assert!(Trie::has_any_cycle(root4.clone()));

        // Disconnected graph with a cycle: root5 (linear chain), root6 (cycle)
        let root5: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(40)));
        let node_d: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(41)));
        root5.write().unwrap().force_insert_to_node("r->d", "e14", &node_d);
        // Separately, a cycle structure
        let root6_in_cycle: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(50)));
        let node_e: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(51)));
        root6_in_cycle.write().unwrap().force_insert_to_node("c1->e", "e15", &node_e);
        node_e.write().unwrap().force_insert_to_node("e->c1", "e16", &root6_in_cycle); // Cycle
        // Checking from root5 should NOT find the cycle
        assert!(!Trie::has_any_cycle(root5.clone()));
        // Checking from root6_in_cycle SHOULD find the cycle
        assert!(Trie::has_any_cycle(root6_in_cycle.clone()));
    }


    #[test]
    fn test_cycle_special_map_no_panic_limited_processing() {
        // Cycle: root -> child -> root.
        // Manually create cycle.
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));

        // Manually create links
        root.write().unwrap().force_insert_to_node("r->c", "e1", &child);
        child.write().unwrap().force_insert_to_node("c->r", "e2", &root);
        // Manually set depths. These are crucial for special_map's readiness check.
        root.write().unwrap().max_depth = 0; // Initial node, depth 0
        child.write().unwrap().max_depth = 1; // Child reachable at depth 1

        let mut processed_vals = Vec::new();
        let mut computed_vals = Vec::new();

        Trie::special_map(
            vec![(root.clone(), 100)], // Start at root
            |p, _ek, _ev, _n| Some(p + 1), // Step: increment
            |cur, new| *cur = (*cur).max(new), // Merge: max
            |node, v| { // process: always continue
                processed_vals.push(node.value);
                computed_vals.push(*v);
                true
            },
        );

        // Expected behavior: Root processed (V=100), Child processed (V=101).
        // The cycle back to root doesn't re-process root because root is in `done`.
        // The new depth-based scheduler should handle this gracefully.
        assert_eq!(processed_vals.len(), 2);
        assert!(processed_vals.contains(&0));
        assert!(processed_vals.contains(&1));

        let results_map: HashMap<i32, i32> = processed_vals.iter().cloned()
            .zip(computed_vals.iter().cloned()).collect();
        assert_eq!(results_map.get(&0), Some(&100));
        assert_eq!(results_map.get(&1), Some(&101));
    }

    #[test]
    fn test_special_map_stop_processing() {
        // Structure:
        //      root (0) --e1,e2--> c1(1), c2(2)
        //      c1(1) --e3--> gc1(3)
        //      c2(2) --e4--> gc2(4)
        // Process returns false for c1, true otherwise.
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));
        let grandchild2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(4)));

        {
            let mut r = root.write().unwrap();
            r.try_insert("r->c1", &mut Some("edge1"), child1.clone()).unwrap();
            r.try_insert("r->c2", &mut Some("edge2"), child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.write().unwrap();
            c1.try_insert("c1->gc", &mut Some("edge3"), grandchild1.clone()).unwrap();
        }
        {
            let mut c2 = child2.write().unwrap();
            c2.try_insert("c2->gc", &mut Some("edge4"), grandchild2.clone()).unwrap();
        }

        let processed_nodes = Arc::new(RwLock::new(HashSet::<i32>::new()));
        let computed_values = Arc::new(RwLock::new(HashMap::<i32, i32>::new()));

        Trie::special_map(
            vec![(root.clone(), 100)],
            |p_val, _ek, _ev, _child_node| Some(p_val + 1), // step: increment value
            |current_v, new_v| *current_v = new_v, // merge: replace
            {
                let processed_nodes = processed_nodes.clone();
                let computed_values = computed_values.clone();
                move |node, final_v| {
                    processed_nodes.write().unwrap().insert(node.value);
                    computed_values.write().unwrap().insert(node.value, *final_v);
                    if node.value == 1 { // Stop processing children if node value is 1 (child1)
                        false
                    } else {
                        true
                    }
                }
            }
        );

        let final_processed = processed_nodes.read().unwrap();
        let final_values = computed_values.read().unwrap();

        // Expected processed nodes: 0, 1, 2, 4. Node 3 should be skipped because propagation stopped at node 1.
        assert_eq!(final_processed.len(), 4);
        assert!(final_processed.contains(&0));
        assert!(final_processed.contains(&1)); // Processed, but stopped propagation
        assert!(final_processed.contains(&2)); // Processed, continued propagation
        assert!(!final_processed.contains(&3)); // gc1 should NOT be processed
        assert!(final_processed.contains(&4)); // gc2 should be processed

        // Check computed values
        assert_eq!(final_values.get(&0), Some(&100));
        assert_eq!(final_values.get(&1), Some(&101));
        assert_eq!(final_values.get(&2), Some(&101));
        assert_eq!(final_values.get(&3), None);      // Not processed
        assert_eq!(final_values.get(&4), Some(&102)); // Processed via child2
    }

    #[test]
    fn test_special_map_step_returns_none() {
        // Structure:
        //      root (0) --"keep"--> c1(1)
        //           |
        //           --"skip"--> c2(2) --"keep"--> gc2(3)
        // Step returns None if edge key is "skip".
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));

        {
            let mut r = root.write().unwrap();
            r.try_insert("keep", &mut Some("e1"), child1.clone()).unwrap();
            r.try_insert("skip", &mut Some("e2"), child2.clone()).unwrap();
        }
        {
            let mut c2 = child2.write().unwrap();
            c2.try_insert("keep", &mut Some("e3"), grandchild2.clone()).unwrap();
        }

        let processed_nodes = Arc::new(RwLock::new(HashSet::<i32>::new()));
        let computed_values = Arc::new(RwLock::new(HashMap::<i32, i32>::new()));

        Trie::special_map(
            vec![(root.clone(), 100)],
            // step: increment value only if edge key is "keep"
            |p_val, ek, _ev, _child_node| {
                if *ek == "keep" {
                    Some(p_val + 1)
                } else {
                    None // Skip this edge
                }
            },
            |current_v, new_v| *current_v = new_v, // merge: replace
            {
                let processed_nodes = processed_nodes.clone();
                let computed_values = computed_values.clone();
                move |node, final_v| {
                    processed_nodes.write().unwrap().insert(node.value);
                    computed_values.write().unwrap().insert(node.value, *final_v);
                    true // Always continue processing if node is reached
                }
            }
        );

        let final_processed = processed_nodes.read().unwrap();
        let final_values = computed_values.read().unwrap();

        // Expected processed nodes: 0, 1. Node 2 is skipped because step for root->child2 returns None. Node 3 is not reached as its parent (node 2) is not processed.
        assert_eq!(final_processed.len(), 2);
        assert!(final_processed.contains(&0));
        assert!(final_processed.contains(&1));

        // Check computed values
        assert_eq!(final_values.get(&0), Some(&100));
        assert_eq!(final_values.get(&1), Some(&101));
        assert_eq!(final_values.get(&2), None); // Not processed
        assert_eq!(final_values.get(&3), None); // Not reached, as c2 is not processed
    }


    // --- Tests for insert_or_merge_edge ---

    // Helper merge functions for tests
    // Merge edge value (Vec<i32>): Append new vec to existing if existing is not empty
    fn merge_ev_append(existing_ev: &mut Vec<i32>, new_ev: Vec<i32>) { // Changed existing_ev to &Vec<i32>
        existing_ev.extend(new_ev.iter().copied()); // Use iter().copied()
    }

    // Merge node value (String): Append new string if existing contains "mergeable"
    //
    // NOTE:
    // The sentinel strings used throughout the tests include both
    // “…_mergeable” (should merge)  and “…_not_mergeable” (should NOT merge).
    // The original helper simply checked `contains("mergeable")`, which means
    // `"child_not_mergeable"` was (incorrectly) considered merge-able because
    // it still contains the substring `"mergeable"`.
    //
    // To align the helper’s behaviour with the test‐case expectations we now:
    //   1. Require that the value contains `"mergeable"`, *and*
    //   2. Explicitly reject any value that contains `"not_mergeable"`.
    //
    // This makes values like `"child_mergeable"` merge, while
    // `"child_not_mergeable"` (and similar) do NOT merge.
    fn merge_nv_append_if_flag(existing_nv: &String, new_nv: String) -> Option<String> {
        if existing_nv.contains("mergeable") && !existing_nv.contains("not_mergeable") {
            Some(format!("{}|{}", existing_nv, new_nv))
        } else {
            None
        }
    }

    // test_insert_or_merge_edge_detects_cycle removed as try_insert_or_merge_edge
    // doesn't attempt to re-insert an existing node in a way that would trigger
    // cycle detection based on the node itself being passed again. Cycle detection
    // relies on the try_insert call in Pass 3 when creating a *new* edge/node.

    // --- Tests for EdgeInserter ---

    // Helper merge function for EdgeInserter tests: Union HybridBitset
    fn merge_bitset_union(existing: &mut HybridBitset, new: HybridBitset) {
        *existing |= new // Use reference for the OR operation
    }

    #[test]
    fn test_ei_try_destination_success_new_edge() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest".to_string())));
        let edge_val: HybridBitset = vec![1].into_iter().collect();


        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "key", edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node = inserter.try_destination(dest.clone()).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest));
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap<ArcPtrWrapper<Mutex<...>>, EV>
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, edge_val);
        assert!(Arc::ptr_eq(node_ptr.as_arc(), &dest));
        assert_eq!(dest.read().unwrap().max_depth, 1); // Depth updated by try_insert
    }

    #[test]
    fn test_ei_try_destination_success_merge_ev() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest".to_string())));
        let initial_edge_val: HybridBitset = vec![10].into_iter().collect();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();
        let merged_edge_val: HybridBitset = vec![1, 10].into_iter().collect();

        // Pre-insert edge
        source.write().unwrap().try_insert("key", &mut Some(initial_edge_val), dest.clone()).unwrap();
        assert_eq!(dest.read().unwrap().max_depth, 1); // Check initial depth

        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node = inserter.try_destination(dest.clone()).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest));
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1); // Still one edge
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, merged_edge_val); // Merged value
        assert!(Arc::ptr_eq(node_ptr.as_arc(), &dest));
        assert_eq!(dest.read().unwrap().max_depth, 1); // Depth should remain 1
    }

    #[test]
    fn test_ei_try_destination_fail_merge_ev() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest".to_string())));
        // Pre-insert edge with empty HybridBitset
        let initial_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        source.write().unwrap().try_insert("key", &mut Some(initial_edge_val), dest.clone()).unwrap();

        // In this case, merge_bitset_union will always return Some, so merge should succeed.
        // To test a failing merge, we'd need a different merge function or EV type.
        // Let's repurpose this to test a successful merge where existing is empty.
        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_opt = inserter.try_destination(dest.clone()).into_option();

        assert!(result_opt.is_some()); // Merge succeeded
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        // The result of merge_bitset_union(&empty, &new_edge_val) is new_edge_val
        assert_eq!(*ev, new_edge_val);
        assert!(Arc::ptr_eq(node_ptr.as_arc(), &dest));
    }

    #[test]
    fn test_ei_try_destination_fail_cycle() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest".to_string())));
         let dummy_edge_val = HybridBitset::zeros();

        // Create cycle manually for test setup
        dest.write().unwrap().force_insert_to_node("dest_to_src", dummy_edge_val.clone(), &source); // dest -> source edge
        //source.lock().unwrap().force_insert_to_node("src_to_dest", dummy_edge_val.clone(), &dest); // source -> dest edge - this is what we are trying to insert

        // Now try inserting source -> dest again using EdgeInserter
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();
        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "src_to_dest", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        // This will call try_insert which should detect the cycle
        let result_opt = inserter.try_destination(dest.clone()).into_option();

        assert!(result_opt.is_none()); // Cycle detected, insert failed
    }


    #[test]
    fn test_ei_try_slice_success() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dest2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest2".to_string())));
        let dest3: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest3".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: dest2 -> source creates a cycle if we try source -> dest2
        dest2.write().unwrap().force_insert_to_node("d2->s", dummy_edge_val.clone(), &source);

        let destinations = [dest1.clone(), dest2.clone(), dest3.clone()];

        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        // try(dest1) -> OK
        // try(dest2) -> Cycle Error (skipped because dest1 succeeded)
        // try(dest3) -> Skipped
        let result_node = inserter.try_destinations(&destinations).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest1)); // Should succeed with dest1
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(node_ptr.as_arc(), &dest1));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_try_slice_success_later() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dest2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest2".to_string())));
        let dest3: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest3".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        // Setup: dest1 -> source creates a cycle if we try source -> dest1
        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let destinations = [dest1.clone(), dest2.clone(), dest3.clone()];

        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        // try(dest1) -> Cycle Error
        // try(dest2) -> OK
        // try(dest3) -> Skipped
        let result_node = inserter.try_destinations(&destinations).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest2)); // Should succeed with dest2
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(node_ptr.as_arc(), &dest2));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_try_slice_fail_all() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dest2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest2".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: Both destinations cause cycles
        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);
        dest2.write().unwrap().force_insert_to_node("d2->s", dummy_edge_val.clone(), &source);

        let destinations = [dest1.clone(), dest2.clone()];

        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_opt = inserter.try_destinations(&destinations).into_option();

        assert!(result_opt.is_none()); // All attempts failed
        assert!(source.read().unwrap().get(&"key").is_none()); // No edge added
    }

    #[test]
    fn test_ei_try_children_success_merge() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let child1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child1".to_string())));
        let child2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child2".to_string())));
        let child_other_key: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child_other_key".to_string())));

        let edge_key = "target_key";
        let initial_ev_c1: HybridBitset = vec![10].into_iter().collect();
        let initial_ev_c2: HybridBitset = vec![20].into_iter().collect();
        let new_ev_for_inserter: HybridBitset = vec![1].into_iter().collect();
        let merged_ev_c1: HybridBitset = vec![1, 10].into_iter().collect(); // Expected merge with child1

        // Setup:
        // source --(target_key, initial_ev_c1)--> child1
        // source --(target_key, initial_ev_c2)--> child2
        // source --("other_key", dummy_ev)--> child_other_key
        {
            let mut s = source.write().unwrap();
            s.try_insert(edge_key, &mut Some(initial_ev_c1), child1.clone()).unwrap();
            s.try_insert(edge_key, &mut Some(initial_ev_c2.clone()), child2.clone()).unwrap();
            s.try_insert("other_key", &mut Some(HybridBitset::zeros()), child_other_key.clone()).unwrap();
        }

        // 1. Test successful merge with the first child under the key.
        //    EdgeInserter is created with source, target_key, and new_ev_for_inserter.
        //    merge_bitset_union should merge new_ev_for_inserter into initial_ev_c1.
        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), edge_key, new_ev_for_inserter.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node_opt = inserter.try_children().into_option();

        assert!(result_node_opt.is_some(), "Should find and merge with child1");
        let result_node = result_node_opt.unwrap();
        assert!(Arc::ptr_eq(&result_node, &child1), "Result should be child1");

        // Check edge values:
        // Edge to child1 should be merged.
        // Edge to child2 should be unchanged (because merge with child1 succeeded first).
        // Edge to child_other_key should be unchanged.
        {
            let s_guard = source.read().unwrap();
            let children_map_target_key = s_guard.get(&edge_key).expect("Target key should exist");

            let ev_c1 = children_map_target_key.get(&ArcPtrWrapper::new(child1.clone())).expect("Child1 should be under target_key");
            assert_eq!(*ev_c1, merged_ev_c1, "Edge value for child1 should be merged");

            let ev_c2 = children_map_target_key.get(&ArcPtrWrapper::new(child2.clone())).expect("Child2 should be under target_key");
            assert_eq!(*ev_c2, initial_ev_c2, "Edge value for child2 should be unchanged");

            let children_map_other_key = s_guard.get(&"other_key").expect("Other key should exist");
            assert_eq!(children_map_other_key.len(), 1, "Should be one child under other_key");
            // You could also check the value of the edge to child_other_key if necessary.
        }

        // 2. Test when merge_edge_value fails for all children under the key.
        //    (This test needs a merge function that can fail or a different EV type,
        //     merge_bitset_union always succeeds by design).
        //    Re-using this section to verify the initial state for part 3 is correct.
        let source_nm: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_nm".to_string())));
        let child1_nm: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child1_nm".to_string())));
        let edge_key_nm = "nm_key"; // "nm" for "no merge"
        let initial_ev_nm: HybridBitset = vec![50].into_iter().collect();
        let new_ev_inserter_nm: HybridBitset = vec![5].into_iter().collect();

        source_nm.write().unwrap().try_insert(edge_key_nm, &mut Some(initial_ev_nm.clone()), child1_nm.clone()).unwrap();

        // Check edge value for child1_nm is unchanged - this is now done in part 3.

        // 3. Test when no children exist under the specified edge_key.
        let source_empty: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_empty".to_string())));
        let edge_key_empty = "empty_key"; // This key has no children in source_empty
        let new_ev_inserter_empty: HybridBitset = vec![7].into_iter().collect();

        let mut god = God::new();
        let inserter_empty = EdgeInserter::new(&mut god, source_empty.clone(), edge_key_empty, new_ev_inserter_empty.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node_empty_opt = inserter_empty.try_children().into_option();
        assert!(result_node_empty_opt.is_none(), "try_children should return None if no children under the key");

        // 4. Test chaining with else_create: try_children (no children under key) -> else_create
        let source_chain: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_chain".to_string())));
        let edge_key_chain = "chain_key"; // No children under this key initially in source_chain
        let new_ev_chain: HybridBitset = vec![8].into_iter().collect();
        let created_val = "created_node_via_fallback".to_string();

        let mut god = God::new();
        let inserter_chain = EdgeInserter::new(&mut god, source_chain.clone(), edge_key_chain, new_ev_chain.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node_chain = inserter_chain
            .try_children() // Will do nothing as no children under "chain_key"
            .else_create_destination_with_value(created_val.clone()) // This should execute
            .unwrap();

        assert_eq!(result_node_chain.read().unwrap().value, created_val, "Fallback node should be created with correct value");
        // Check that an edge was created to this new node
        let s_chain_guard = source_chain.read().unwrap();
        let children_map_chain = s_chain_guard.get(&edge_key_chain).expect("Chain key should now exist in source_chain");
        assert_eq!(children_map_chain.len(), 1, "One edge should be created under chain_key");
        let (node_ptr_chain, ev_chain) = children_map_chain.iter().next().unwrap();
        assert!(Arc::ptr_eq(node_ptr_chain.as_arc(), &result_node_chain), "Edge should point to the newly created node");
        assert_eq!(*ev_chain, new_ev_chain, "Edge should have the new_ev_chain value");
    }

    #[test]
    fn test_ei_else_create_with_value() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        // No try calls, should go straight to else_create
        let result_node = inserter.else_create_destination_with_value("created".to_string()).unwrap();

        assert_eq!(result_node.read().unwrap().value, "created");
        assert_eq!(result_node.read().unwrap().max_depth, 1); // Depth updated
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(node_ptr.as_arc(), &result_node));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_else_create_with() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let created_flag = Arc::new(AtomicUsize::new(0));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let flag_clone = created_flag.clone();
        let result_node = inserter.else_create_destination_with(|| {
            flag_clone.fetch_add(1, Ordering::SeqCst);
            "created_via_fn".to_string()
        }).unwrap();

        assert_eq!(created_flag.load(Ordering::SeqCst), 1); // Closure was called
        assert_eq!(result_node.read().unwrap().value, "created_via_fn");
        assert_eq!(result_node.read().unwrap().max_depth, 1);
    }

    #[test]
    fn test_ei_else_create_default() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        // String::default() is ""
        let result_node = inserter.else_create_destination().unwrap();

        assert_eq!(result_node.read().unwrap().value, ""); // Default value
        assert_eq!(result_node.read().unwrap().max_depth, 1);
    }

    #[test]
    fn test_ei_chaining_try_then_else() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: dest1 causes cycle
        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node = inserter
            .try_destination(dest1.clone()) // Fails (cycle)
            .else_create_destination_with_value("fallback".to_string()) // Executes
            .unwrap();

        assert_eq!(result_node.read().unwrap().value, "fallback"); // Fallback was created
        assert!(!Arc::ptr_eq(&result_node, &dest1));
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(node_ptr.as_arc(), &result_node));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_chaining_try_success_skips_else() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node = inserter
            .try_destination(dest1.clone()) // Succeeds
            .else_create_destination_with_value("fallback".to_string()) // Should be skipped
            .unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest1)); // Original dest1 was used
        assert_eq!(result_node.read().unwrap().value, "dest1");
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(node_ptr.as_arc(), &dest1));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    #[should_panic(expected = "EdgeInserter::unwrap() called but no destination was found or created")]
    fn test_ei_unwrap_panic() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: dest1 causes cycle
        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        // Try fails, no else_create called
        inserter.try_destination(dest1.clone()).unwrap(); // Panic here
    }

    #[test]
    fn test_ei_get() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: dest1 causes cycle
        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});

        // Try fails
        let inserter_after_try = inserter.try_destination(dest1.clone());
        assert!(inserter_after_try.clone_into_option().is_none());

        // Now use else_create
        let inserter_after_else = inserter_after_try.else_create_destination_with_value("fallback".to_string());
        let result_opt = inserter_after_else.into_option();
        assert!(result_opt.is_some());
        assert_eq!(result_opt.unwrap().read().unwrap().value, "fallback");
    }

    #[test]
    fn test_ei_chaining_stops_after_success() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let child1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child1".to_string()))); // This one succeeds
        let child2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child2".to_string())));
        let new_node_val_if_created = "new_node_val".to_string();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let destinations_for_slice = vec![child2.clone()];

        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node = inserter
            .try_destination(child1.clone()) // This succeeds, result is set to child1
            // try_slice, else_create_with_value should now have no effect
            .try_destinations(&destinations_for_slice) // Should be skipped
            .else_create_destination_with_value(new_node_val_if_created.clone()) // Should be skipped
            .unwrap();

        assert!(Arc::ptr_eq(&result_node, &child1), "Chain should stop after first success (try_insert)");

        // Check only the edge to child1 was added
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(node_ptr.as_arc(), &child1));
        assert_eq!(*ev, new_edge_val);

        // Ensure the value for the skipped else_create was not used
        assert_ne!(result_node.read().unwrap().value, new_node_val_if_created);
    }

     #[test]
    fn test_ei_try_children_new_logic() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let child1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child1".to_string())));
        let child2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child2".to_string())));
        let child_other_key: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child_other_key".to_string())));

        let edge_key = "target_key";
        let initial_ev_c1: HybridBitset = vec![10].into_iter().collect();
        let initial_ev_c2: HybridBitset = vec![20].into_iter().collect();
        let new_ev_for_inserter: HybridBitset = vec![1].into_iter().collect();
        let merged_ev_c1: HybridBitset = vec![1, 10].into_iter().collect(); // Expected merge with child1

        // Setup:
        // source --(target_key, initial_ev_c1)--> child1
        // source --(target_key, initial_ev_c2)--> child2
        // source --("other_key", dummy_ev)--> child_other_key
        {
            let mut s = source.write().unwrap();
            s.try_insert(edge_key, &mut Some(initial_ev_c1), child1.clone()).unwrap();
            s.try_insert(edge_key, &mut Some(initial_ev_c2.clone()), child2.clone()).unwrap();
            s.try_insert("other_key", &mut Some(HybridBitset::zeros()), child_other_key.clone()).unwrap();
        }

        // 1. Test successful merge with the first child under the key.
        //    EdgeInserter is created with source, target_key, and new_ev_for_inserter.
        //    merge_bitset_union should merge new_ev_for_inserter into initial_ev_c1.
        let mut god = God::new();
        let inserter = EdgeInserter::new(&mut god, source.clone(), edge_key, new_ev_for_inserter.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node_opt = inserter.try_children().into_option();

        assert!(result_node_opt.is_some(), "Should find and merge with child1");
        let result_node = result_node_opt.unwrap();
        assert!(Arc::ptr_eq(&result_node, &child1), "Result should be child1, got {:?} and {:?}", result_node, child1);

        // Check edge values:
        // Edge to child1 should be merged.
        // Edge to child2 should be unchanged (because merge with child1 succeeded first).
        // Edge to child_other_key should be unchanged.
         #[cfg(false)]
        {
            let s_guard = source.read().unwrap();
            let children_map_target_key = s_guard.get(&edge_key).expect("Target key should exist");

            let ev_c1 = children_map_target_key.get(&ArcPtrWrapper::Strong(ArcPtrWrapper::new(child1.clone()))).expect("Child1 should be under target_key");
            assert_eq!(*ev_c1, merged_ev_c1, "Edge value for child1 should be merged");

            let ev_c2 = children_map_target_key.get(&ArcPtrWrapper::Strong(ArcPtrWrapper::new(child2.clone()))).expect("Child2 should be under target_key");
            assert_eq!(*ev_c2, initial_ev_c2, "Edge value for child2 should be unchanged");

            let children_map_other_key = s_guard.get(&"other_key").expect("Other key should exist");
            assert_eq!(children_map_other_key.len(), 1, "Should be one child under other_key");
            // You could also check the value of the edge to child_other_key if necessary.
        }

        // 2. Test when merge_edge_value fails for all children under the key.
        //    (This test needs a merge function that can fail or a different EV type,
        //     merge_bitset_union always succeeds by design).
        //    Re-using this section to verify the initial state for part 3 is correct.
        let source_nm: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_nm".to_string())));
        let child1_nm: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child1_nm".to_string())));
        let edge_key_nm = "nm_key"; // "nm" for "no merge"
        let initial_ev_nm: HybridBitset = vec![50].into_iter().collect();
        let new_ev_inserter_nm: HybridBitset = vec![5].into_iter().collect();

        source_nm.write().unwrap().try_insert(edge_key_nm, &mut Some(initial_ev_nm.clone()), child1_nm.clone()).unwrap();

        // Check edge value for child1_nm is unchanged - this is now done in part 3.

        // 3. Test when no children exist under the specified edge_key.
        let source_empty: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_empty".to_string())));
        let edge_key_empty = "empty_key"; // This key has no children in source_empty
        let new_ev_inserter_empty: HybridBitset = vec![7].into_iter().collect();

        let mut god = God::new();
        let inserter_empty = EdgeInserter::new(&mut god, source_empty.clone(), edge_key_empty, new_ev_inserter_empty.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node_empty_opt = inserter_empty.try_children().into_option();
        assert!(result_node_empty_opt.is_none(), "try_children should return None if no children under the key");

        // 4. Test chaining with else_create: try_children (no children under key) -> else_create
        let source_chain: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_chain".to_string())));
        let edge_key_chain = "chain_key"; // No children under this key initially in source_chain
        let new_ev_chain: HybridBitset = vec![8].into_iter().collect();
        let created_val = "created_node_via_fallback".to_string();

        let mut god = God::new();
        let inserter_chain = EdgeInserter::new(&mut god, source_chain.clone(), edge_key_chain, new_ev_chain.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node_chain = inserter_chain
            .try_children() // Will do nothing as no children under "chain_key"
            .else_create_destination_with_value(created_val.clone()) // This should execute
            .unwrap();

        assert_eq!(result_node_chain.read().unwrap().value, created_val, "Fallback node should be created with correct value");
        // Check that an edge was created to this new node
        let s_chain_guard = source_chain.read().unwrap();
        let children_map_chain = s_chain_guard.get(&edge_key_chain).expect("Chain key should now exist in source_chain");
        assert_eq!(children_map_chain.len(), 1, "One edge should be created under chain_key");
        let (node_ptr_chain, ev_chain) = children_map_chain.iter().next().unwrap();
        assert!(Arc::ptr_eq(node_ptr_chain.as_arc(), &result_node_chain), "Edge should point to the newly created node");
        assert_eq!(*ev_chain, new_ev_chain, "Edge should have the new_ev_chain value");
    }
}
