use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
// Removed: Ordering, BinaryHeap, HashMap, InsertError

#[derive(Debug, Clone)]
pub struct TrieNode<EV, E, T> {
    pub value: T,
    // Using BTreeMap for ordered iteration over children, if needed.
    children: BTreeMap<E, (EV, Arc<Mutex<TrieNode<EV, E, T>>>)>,
    // Removed: max_depth
}

// Removed: InsertError enum

impl<EV: Clone, T, E: Ord> TrieNode<EV, E, T> {
    pub fn new(value: T) -> TrieNode<EV, E, T> {
        TrieNode {
            value,
            children: BTreeMap::new(),
            // Removed: max_depth initialization
        }
    }

    // Removed: would_create_cycle method
    // Removed: update_max_depths method

    /// Inserts a child node associated with the given edge.
    /// Assumes that the insertion does not create a cycle.
    pub fn insert(&mut self, edge: E, child: Arc<Mutex<TrieNode<EV, E, T>>>, ev: EV) {
        // Simplified insertion: directly insert into the children map.
        self.children.insert(edge, (ev, child));
        // Removed: Cycle check and max_depth update.
    }

    /// Gets the edge value and child node associated with the given edge.
    pub fn get(&self, edge: &E) -> Option<(EV, Arc<Mutex<TrieNode<EV, E, T>>>)> {
        self.children.get(edge).cloned()
    }

    /// Returns a reference to the map of children nodes.
    pub fn children(&self) -> &BTreeMap<E, (EV, Arc<Mutex<TrieNode<EV, E, T>>>)> {
        &self.children
    }

    /// Checks if the node has any children.
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    // Removed: max_depth() method

    /// Collects all unique nodes reachable from the given root using BFS.
    /// Nodes are identified by their memory address.
    pub fn all_nodes(root: Arc<Mutex<TrieNode<EV, E, T>>>) -> Vec<Arc<Mutex<TrieNode<EV, E, T>>>> {
        // Use a set for tracking visited nodes by the raw pointer to the TrieNode data.
        let mut visited_ptrs: HashSet<*const TrieNode<EV, E, T>> = HashSet::new();
        let mut result: Vec<Arc<Mutex<TrieNode<EV, E, T>>>> = Vec::new();
        let mut queue: VecDeque<Arc<Mutex<TrieNode<EV, E, T>>>> = VecDeque::new();

        // Lock the root node to get its pointer and mark as visited
        {
            let root_node = root.try_lock().expect("Failed to lock root node in all_nodes");
            let root_ptr = &*root_node as *const TrieNode<EV, E, T>;
            // Insert returns true if the value was not already present.
            if visited_ptrs.insert(root_ptr) {
                 queue.push_back(root.clone()); // Use the original Arc here
            }
            // root_node lock is released here
        }


        while let Some(node_arc) = queue.pop_front() {
            result.push(node_arc.clone()); // Add the Arc to the result list
            // Lock the current node to access its children
            let node = node_arc.try_lock().expect("Failed to lock node in all_nodes");

            for (_, child_arc) in node.children.values() {
                // Lock the child node *only* to get its pointer for the visited check
                let child_node = child_arc.try_lock().expect("Failed to lock child node in all_nodes");
                let child_ptr = &*child_node as *const TrieNode<EV, E, T>;
                // Release lock immediately after getting pointer
                drop(child_node);

                // Insert the *pointer* into visited_ptrs. If it wasn't there before...
                if visited_ptrs.insert(child_ptr) {
                    // ...enqueue the Arc for processing.
                    queue.push_back(child_arc.clone());
                }
            }
            // node lock is released here
        }
        result
    }
}

// Removed: QueueItem struct and its impls (PartialEq, Eq, PartialOrd, Ord)

impl<EV: Clone, T: Clone, E: Ord + Clone> TrieNode<EV, E, T> {
    /// Performs a breadth-first traversal and applies functions.
    ///
    /// Traverses the trie starting from `initial_nodes_and_values` using BFS.
    /// For each node visited:
    /// 1. Calls `process` with the node's value and the computed value `V`.
    /// 2. For each child, computes a new value using `step` and enqueues the child if not visited.
    ///
    /// Note: The `merge` function parameter is kept for signature compatibility
    /// but is not used in this simplified BFS implementation.
    pub fn special_map<V>(
        initial_nodes_and_values: Vec<(Arc<Mutex<TrieNode<EV, E, T>>>, V)>,
        mut step: impl FnMut(&V, &E, &EV, &TrieNode<EV, E, T>) -> V,
        _merge: impl FnMut(Vec<V>) -> V, // Kept in signature, but unused
        mut process: impl FnMut(&T, &V), // Simplified: process doesn't return bool
    ) where
        V: Clone,
        E: Ord, // E needs Ord because children are in BTreeMap
    {
        // Use VecDeque for BFS queue
        let mut queue: VecDeque<(Arc<Mutex<TrieNode<EV, E, T>>>, V)> = VecDeque::new();
        // Use HashSet to track visited nodes by pointer to the TrieNode data
        let mut visited: HashSet<*const TrieNode<EV, E, T>> = HashSet::new();

        // Initialize queue and visited set
        for (node_arc, value) in initial_nodes_and_values {
            // Lock the node to get its pointer
            let node = node_arc.try_lock().expect("Failed to lock initial node in special_map");
            let node_ptr = &*node as *const TrieNode<EV, E, T>;
            // Release lock
            drop(node);

            // If the node hasn't been visited yet, add it to the queue
            if visited.insert(node_ptr) {
                queue.push_back((node_arc.clone(), value.clone()));
            }
            // Note: If multiple initial nodes point to the same actual node,
            // only the first one encountered here will be added to the queue.
        }

        while let Some((node_arc, value)) = queue.pop_front() {
            // Lock the node to access its data and children
            let node = node_arc.try_lock().expect("Failed to lock node in special_map");
            // Note: We don't need the pointer here again because it was checked before adding to queue

            // Process the current node
            process(&node.value, &value);

            // Prepare children for the next level of BFS
            for (edge, (ev, child_arc)) in &node.children {
                 // Lock the child to pass to `step` and to get its pointer
                let child_node = child_arc.try_lock().expect("Failed to lock child node in special_map");
                let child_ptr = &*child_node as *const TrieNode<EV, E, T>;

                // Calculate the value for the child node *before* releasing lock
                let new_child_value = step(&value, edge, ev, &child_node);
                // Release lock
                drop(child_node);

                // Enqueue child only if it hasn't been visited yet
                // Insert the *pointer* into visited set. If it wasn't there before...
                if visited.insert(child_ptr) {
                     // ...enqueue the Arc and its computed value.
                    queue.push_back((child_arc.clone(), new_child_value));
                }
                // If the child was already visited, we don't re-enqueue or merge.
            }
            // node lock is released here
        }
    }
}

/// Helper function to dump the structure for debugging.
pub(crate) fn dump_structure<EV, E, T>(root: Arc<Mutex<TrieNode<EV, E, T>>>) where E: Debug, T: Debug {
    let mut queue = VecDeque::new(); // Use VecDeque for BFS dump
    // Use HashSet to track visited nodes by pointer to the TrieNode data
    let mut seen: HashSet<*const TrieNode<EV, E, T>> = HashSet::new();

    // Lock root to get pointer and add to queue if not seen
    {
        let root_node = root.try_lock().unwrap();
        let root_ptr = &*root_node as *const TrieNode<EV, E, T>;
        if seen.insert(root_ptr) {
            queue.push_back(root.clone());
        }
        // root_node lock released here
    }


    println!("Dumping Trie Structure (BFS):");
    while let Some(node_arc) = queue.pop_front() {
        // Lock node to print its info and access children
        let node = node_arc.try_lock().unwrap();
        let node_ptr = &*node as *const TrieNode<EV, E, T>;
        // Removed max_depth printout
        println!("{:?}: Value: {:?}", node_ptr, node.value);

        for (edge, (_, child_arc)) in &node.children {
            // Lock child to get its pointer for printing and visited check
            let child_node = child_arc.try_lock().unwrap();
            let child_ptr = &*child_node as *const TrieNode<EV, E, T>;
            println!("  - Edge: {:?} -> Child: {:?}", edge, child_ptr);
            // Release lock immediately after getting pointer and printing
            drop(child_node);

            // Insert the *pointer* into seen set. If it wasn't there before...
            if seen.insert(child_ptr) {
                // ...enqueue the Arc for processing.
                queue.push_back(child_arc.clone());
            }
        }
        // node lock released here
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet; // Import HashSet for test_all_nodes

    // Removed: test_cycle_detection
    // Removed: test_max_depth_updates

    #[test]
    fn test_insertion_and_retrieval() {
        let mut root = TrieNode::<(), &str, i32>::new(0);
        let child1 = Arc::new(Mutex::new(TrieNode::new(1)));
        let child2 = Arc::new(Mutex::new(TrieNode::new(2)));

        root.insert("a", child1.clone(), ());
        root.insert("b", child2.clone(), ());

        // Test get
        let (ev1, retrieved_child1) = root.get(&"a").unwrap();
        assert!(Arc::ptr_eq(&retrieved_child1, &child1));
        assert_eq!(ev1, ());

        let (ev2, retrieved_child2) = root.get(&"b").unwrap();
        assert!(Arc::ptr_eq(&retrieved_child2, &child2));
        assert_eq!(ev2, ());

        assert!(root.get(&"c").is_none());

        // Test children iterator
        let children: Vec<_> = root.children().keys().cloned().collect();
        assert_eq!(children, vec!["a", "b"]); // BTreeMap ensures order

        // Test is_empty
        assert!(!root.is_empty());
        assert!(child1.try_lock().unwrap().is_empty());
    }


    #[test]
    fn test_special_map_bfs_order() {
        // Structure:
        //    root(0)
        //   /     \
        // c1(1)   c2(2)
        //  |
        // gc(3)
        let root = Arc::new(Mutex::new(TrieNode::<(), &str, i32>::new(0)));
        let child1 = Arc::new(Mutex::new(TrieNode::new(1)));
        let child2 = Arc::new(Mutex::new(TrieNode::new(2)));
        let grandchild = Arc::new(Mutex::new(TrieNode::new(3)));

        // Build the trie (manually locking)
        root.try_lock().unwrap().insert("r->c1", child1.clone(), ());
        root.try_lock().unwrap().insert("r->c2", child2.clone(), ());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone(), ());

        let mut processed_order = Vec::new();
        let mut processed_values = Vec::new();

        TrieNode::special_map(
            vec![(root.clone(), 100)], // Start BFS from root with initial value 100
            |parent_v, _edge, _ev, _child_node| {
                // Simple step: child value is parent value + 1
                parent_v + 1
            },
            |_| panic!("Merge should not be called in this BFS implementation"), // Merge not used
            |node_t_val, computed_v| {
                // Process: record the node's T value and the computed V value
                processed_order.push(*node_t_val);
                processed_values.push(*computed_v);
            }
        );

        // Verify nodes are processed in BFS order
        // Expected order: root, child1, child2, grandchild
        assert_eq!(processed_order, vec![0, 1, 2, 3]);

        // Verify computed values (V) based on BFS traversal
        // root: initial value 100
        // c1: step(100) = 101
        // c2: step(100) = 101
        // gc: step(value_of_c1) = step(101) = 102
        assert_eq!(processed_values, vec![100, 101, 101, 102]);
    }

    #[test]
    fn test_all_nodes() {
        let root = Arc::new(Mutex::new(TrieNode::<(), &str, &str>::new("root")));
        let child1 = Arc::new(Mutex::new(TrieNode::new("child1")));
        let child2 = Arc::new(Mutex::new(TrieNode::new("child2")));
        let grandchild = Arc::new(Mutex::new(TrieNode::new("grandchild")));

        // Build the trie
        root.try_lock().unwrap().insert("r->c1", child1.clone(), ());
        root.try_lock().unwrap().insert("r->c2", child2.clone(), ());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone(), ());
        // Add a shared child (diamond shape) - c2 also points to grandchild
        child2.try_lock().unwrap().insert("c2->gc", grandchild.clone(), ());


        let all_nodes = TrieNode::all_nodes(root.clone());

        // Check that all unique nodes are present (count should be 4)
        assert_eq!(all_nodes.len(), 4);

        // Use HashSet of pointers to verify uniqueness and presence
        // Get pointers *after* locking each node in the result
        let mut node_ptrs = HashSet::new();
        for node_arc in &all_nodes {
             let node = node_arc.try_lock().unwrap();
             let node_ptr = &*node as *const TrieNode<_, _, _>;
             node_ptrs.insert(node_ptr);
        }

        assert_eq!(node_ptrs.len(), 4); // Ensure collected nodes are unique

        // Verify presence by checking pointers obtained after locking
        let root_ptr = &*root.try_lock().unwrap() as *const _;
        let c1_ptr = &*child1.try_lock().unwrap() as *const _;
        let c2_ptr = &*child2.try_lock().unwrap() as *const _;
        let gc_ptr = &*grandchild.try_lock().unwrap() as *const _;

        assert!(node_ptrs.contains(&root_ptr));
        assert!(node_ptrs.contains(&c1_ptr));
        assert!(node_ptrs.contains(&c2_ptr));
        assert!(node_ptrs.contains(&gc_ptr));
    }

     #[test]
    fn test_special_map_diamond() {
        // Structure:
        //      root(0)
        //     /     \
        //   c1(1)   c2(2)
        //     \     /
        //      gc(3)
        let root = Arc::new(Mutex::new(TrieNode::<(), &str, i32>::new(0)));
        let child1 = Arc::new(Mutex::new(TrieNode::new(1)));
        let child2 = Arc::new(Mutex::new(TrieNode::new(2)));
        let grandchild = Arc::new(Mutex::new(TrieNode::new(3)));

        // Build the trie
        root.try_lock().unwrap().insert("r->c1", child1.clone(), ());
        root.try_lock().unwrap().insert("r->c2", child2.clone(), ());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone(), ());
        child2.try_lock().unwrap().insert("c2->gc", grandchild.clone(), ()); // Diamond

        let mut processed_order = Vec::new();
        let mut processed_values = Vec::new();

        TrieNode::special_map(
            vec![(root.clone(), 100)],
            |parent_v, _edge, _ev, _child_node| parent_v + 1,
            |_| panic!("Merge should not be called"),
            |node_t_val, computed_v| {
                processed_order.push(*node_t_val);
                processed_values.push(*computed_v);
            }
        );

        // BFS order: root, c1, c2, gc (gc is visited only once)
        // The exact order between c1 and c2 might vary in a simple VecDeque BFS,
        // but gc should appear after both. Let's check the content regardless of
        // the c1/c2 order.
        assert_eq!(processed_order.len(), 4);
        assert!(processed_order.contains(&0));
        assert!(processed_order.contains(&1));
        assert!(processed_order.contains(&2));
        assert!(processed_order.contains(&3));
        assert_eq!(processed_order[0], 0); // Root is always first
        assert_eq!(processed_order[3], 3); // Grandchild is last in this structure


        // Values: root=100, c1=101, c2=101
        // gc is processed when visited first (e.g., from c1), value = step(value_c1) = 102
        // When reached from c2 later, it's already 'visited', so not enqueued again.
        // Check values corresponding to the processed order.
        let mut expected_values = vec![100]; // Root value
        if processed_order[1] == 1 { // If c1 was processed second
            expected_values.push(101); // c1 value
            expected_values.push(101); // c2 value
        } else { // If c2 was processed second
            expected_values.push(101); // c2 value
            expected_values.push(101); // c1 value
        }
        expected_values.push(102); // gc value (derived from the first parent processed)

        assert_eq!(processed_values, expected_values);
    }
}