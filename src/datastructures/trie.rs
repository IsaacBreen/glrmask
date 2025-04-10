use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

/// Represents a node in a Trie-like structure (allowing shared subtrees and DAGs).
///
/// `EV`: Type of the value associated with an edge.
/// `E`: Type of the edge label (must be comparable).
/// `T`: Type of the value stored within the node.
#[derive(Debug, Clone)]
pub struct TrieNode<EV, E, T> {
    pub value: T,
    children: BTreeMap<E, (EV, Arc<Mutex<TrieNode<EV, E, T>>>)>,
}

impl<EV: Clone, T, E: Ord> TrieNode<EV, E, T> {
    /// Creates a new TrieNode with the given value and no children.
    pub fn new(value: T) -> TrieNode<EV, E, T> {
        TrieNode {
            value,
            children: BTreeMap::new(),
        }
    }

    /// Inserts a child node associated with the given edge and edge value.
    ///
    /// Note: This implementation does *not* perform cycle detection. Adding an edge
    /// that creates a cycle may lead to infinite loops in traversal algorithms.
    pub fn insert(&mut self, edge: E, child: Arc<Mutex<TrieNode<EV, E, T>>>, ev: EV) {
        self.children.insert(edge, (ev, child));
    }

    /// Gets the edge value and child node associated with the given edge, if it exists.
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

    /// Collects all unique nodes reachable from the given root using Breadth-First Search (BFS).
    /// Node uniqueness is determined by the memory address of the `TrieNode` data.
    pub fn all_nodes(root: Arc<Mutex<TrieNode<EV, E, T>>>) -> Vec<Arc<Mutex<TrieNode<EV, E, T>>>> {
        let mut visited_ptrs: HashSet<*const TrieNode<EV, E, T>> = HashSet::new();
        let mut result: Vec<Arc<Mutex<TrieNode<EV, E, T>>>> = Vec::new();
        let mut queue: VecDeque<Arc<Mutex<TrieNode<EV, E, T>>>> = VecDeque::new();

        // Helper to check if a node (via its Arc) has been visited and add it to queue if not.
        // Returns true if the node was added to the queue, false otherwise.
        let mut visit_and_enqueue =
            |node_arc: Arc<Mutex<TrieNode<EV, E, T>>>,
             q: &mut VecDeque<Arc<Mutex<TrieNode<EV, E, T>>>>,
             visited: &mut HashSet<*const TrieNode<EV, E, T>>|
             -> bool {
                let ptr = {
                    // Lock briefly to get the pointer
                    let node_guard = node_arc.try_lock().expect("Failed to lock node");
                    &*node_guard as *const TrieNode<EV, E, T>
                };
                if visited.insert(ptr) {
                    q.push_back(node_arc);
                    true
                } else {
                    false
                }
            };

        // Initialize queue with the root node
        visit_and_enqueue(root, &mut queue, &mut visited_ptrs);

        while let Some(node_arc) = queue.pop_front() {
            result.push(node_arc.clone()); // Add the visited node Arc to the result

            // Lock the current node to access its children
            let node_guard = node_arc.try_lock().expect("Failed to lock node");

            for child_arc in node_guard.children.values().map(|(_, arc)| arc) {
                visit_and_enqueue(child_arc.clone(), &mut queue, &mut visited_ptrs);
            }
            // node_guard lock is released here
        }
        result
    }
}

impl<EV: Clone, T: Clone, E: Ord + Clone> TrieNode<EV, E, T> {
    /// Performs a Breadth-First Search (BFS) traversal applying functions at each step.
    ///
    /// Starts from `initial_nodes_and_values`. For each visited node:
    /// 1. Calls `process` with the node's internal value (`T`) and the computed value (`V`).
    /// 2. For each child, computes a new value `V` using `step` and enqueues the child if not already visited.
    ///
    /// Node uniqueness for visitation is determined by the memory address of the `TrieNode` data.
    ///
    /// The `merge` function parameter is unused in this BFS implementation but kept for potential
    /// compatibility with algorithms that might need to merge values arriving at a node via different paths.
    pub fn special_map<V>(
        initial_nodes_and_values: Vec<(Arc<Mutex<TrieNode<EV, E, T>>>, V)>,
        mut step: impl FnMut(
            &V,                           /* parent computed value */
            &E,                           /* edge label */
            &EV,                          /* edge value */
            &TrieNode<EV, E, T>, /* child node (locked) */
        ) -> V,
        _merge: impl FnMut(Vec<V>) -> V, // Unused in this BFS implementation
        mut process: impl FnMut(
            &T, /* node's internal value */
            &V, /* node's computed value */
        ),
    ) where
        V: Clone,
    {
        let mut queue: VecDeque<(Arc<Mutex<TrieNode<EV, E, T>>>, V)> = VecDeque::new();
        let mut visited: HashSet<*const TrieNode<EV, E, T>> = HashSet::new();

        // Initialize queue and visited set
        for (node_arc, value) in initial_nodes_and_values {
            let ptr = {
                let node_guard = node_arc.try_lock().expect("Failed to lock initial node");
                &*node_guard as *const TrieNode<EV, E, T>
            };
            if visited.insert(ptr) {
                queue.push_back((node_arc, value));
            }
        }

        while let Some((node_arc, current_value)) = queue.pop_front() {
            // Lock the node to access its data and children
            let node_guard = node_arc.try_lock().expect("Failed to lock node");

            // Process the current node
            process(&node_guard.value, &current_value);

            // Prepare children for the next level of BFS
            for (edge, (ev, child_arc)) in &node_guard.children {
                let child_ptr = {
                    // Lock child briefly to get pointer and pass to step function
                    let child_guard = child_arc.try_lock().expect("Failed to lock child node");
                    let ptr = &*child_guard as *const TrieNode<EV, E, T>;

                    // Enqueue child only if it hasn't been visited yet
                    if visited.insert(ptr) {
                        // Calculate the value for the child node *before* releasing lock
                        let next_value = step(&current_value, edge, ev, &child_guard);
                        queue.push_back((child_arc.clone(), next_value));
                    }
                    // child_guard lock is released here
                    ptr // Return ptr for potential use (though not strictly needed now)
                };
                 // child_ptr is available here if needed, but visited check already done
            }
            // node_guard lock is released here
        }
    }
}

/// Helper function to print the structure of the Trie/DAG starting from root (BFS).
pub(crate) fn dump_structure<EV, E, T>(root: Arc<Mutex<TrieNode<EV, E, T>>>)
where
    E: Debug,
    T: Debug,
{
    let mut queue = VecDeque::new();
    let mut seen: HashSet<*const TrieNode<EV, E, T>> = HashSet::new();

    println!("Dumping Trie Structure (BFS):");

    // Helper to visit and enqueue
     let mut visit_and_enqueue =
            |node_arc: Arc<Mutex<TrieNode<EV, E, T>>>,
             q: &mut VecDeque<Arc<Mutex<TrieNode<EV, E, T>>>>,
             visited: &mut HashSet<*const TrieNode<EV, E, T>>| {
                let ptr = {
                    let node_guard = node_arc.try_lock().expect("Failed to lock node for dump");
                    &*node_guard as *const TrieNode<EV, E, T>
                };
                if visited.insert(ptr) {
                    q.push_back(node_arc);
                }
            };

    visit_and_enqueue(root, &mut queue, &mut seen);


    while let Some(node_arc) = queue.pop_front() {
        // Lock node to print its info and access children
        let node_guard = node_arc.try_lock().expect("Failed to lock node for dump");
        let node_ptr = &*node_guard as *const TrieNode<EV, E, T>;

        println!("{:?}: Value: {:?}", node_ptr, node_guard.value);

        for (edge, (_, child_arc)) in &node_guard.children {
            let child_ptr = {
                 let child_guard = child_arc.try_lock().expect("Failed to lock child for dump");
                 &*child_guard as *const TrieNode<EV, E, T>
            };
            println!("  - Edge: {:?} -> Child: {:?}", edge, child_ptr);
            // Enqueue child if not seen
            visit_and_enqueue(child_arc.clone(), &mut queue, &mut seen);
        }
        // node_guard lock released here
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet; // Required for pointer sets in tests

    #[test]
    fn test_insertion_and_retrieval() {
        let mut root = TrieNode::<(), &str, i32>::new(0);
        let child1 = Arc::new(Mutex::new(TrieNode::new(1)));
        let child2 = Arc::new(Mutex::new(TrieNode::new(2)));

        root.insert("a", child1.clone(), ());
        root.insert("b", child2.clone(), ());

        // Test get
        let (ev1, retrieved_child1) = root.get(&"a").expect("Failed to get child 'a'");
        assert!(Arc::ptr_eq(&retrieved_child1, &child1));
        assert_eq!(ev1, ());

        let (ev2, retrieved_child2) = root.get(&"b").expect("Failed to get child 'b'");
        assert!(Arc::ptr_eq(&retrieved_child2, &child2));
        assert_eq!(ev2, ());

        assert!(root.get(&"c").is_none());

        // Test children iterator order (BTreeMap)
        let children_keys: Vec<_> = root.children().keys().cloned().collect();
        assert_eq!(children_keys, vec!["a", "b"]);

        // Test is_empty
        assert!(!root.is_empty());
        assert!(child1.try_lock().unwrap().is_empty());
    }

    #[test]
    fn test_special_map_bfs_order() {
        // Structure: root(0) -> c1(1) -> gc(3), root(0) -> c2(2)
        let root = Arc::new(Mutex::new(TrieNode::<(), &str, i32>::new(0)));
        let child1 = Arc::new(Mutex::new(TrieNode::new(1)));
        let child2 = Arc::new(Mutex::new(TrieNode::new(2)));
        let grandchild = Arc::new(Mutex::new(TrieNode::new(3)));

        root.try_lock().unwrap().insert("r->c1", child1.clone(), ());
        root.try_lock().unwrap().insert("r->c2", child2.clone(), ());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone(), ());

        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();

        TrieNode::special_map(
            vec![(root.clone(), 100)], // Start BFS from root with initial value 100
            |parent_v, _edge, _ev, _child_node| parent_v + 1, // Step: increment value
            |_| panic!("Merge should not be called"),
            |node_t_val, computed_v| {
                processed_node_values.push(*node_t_val);
                computed_values.push(*computed_v);
            },
        );

        // Expected BFS processing order: 0, 1, 2, 3
        assert_eq!(processed_node_values, vec![0, 1, 2, 3]);
        // Expected computed values: root=100, c1=101, c2=101, gc=102 (from c1)
        assert_eq!(computed_values, vec![100, 101, 101, 102]);
    }

    #[test]
    fn test_all_nodes_diamond() {
        // Structure: root -> c1 -> gc, root -> c2 -> gc (diamond)
        let root = Arc::new(Mutex::new(TrieNode::<(), &str, &str>::new("root")));
        let child1 = Arc::new(Mutex::new(TrieNode::new("child1")));
        let child2 = Arc::new(Mutex::new(TrieNode::new("child2")));
        let grandchild = Arc::new(Mutex::new(TrieNode::new("grandchild")));

        root.try_lock().unwrap().insert("r->c1", child1.clone(), ());
        root.try_lock().unwrap().insert("r->c2", child2.clone(), ());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone(), ());
        child2.try_lock().unwrap().insert("c2->gc", grandchild.clone(), ()); // Diamond

        let all_nodes = TrieNode::all_nodes(root.clone());

        // Should find 4 unique nodes
        assert_eq!(all_nodes.len(), 4);

        // Verify uniqueness using pointers
        let mut node_ptrs = HashSet::new();
        for node_arc in &all_nodes {
            let node_guard = node_arc.try_lock().unwrap();
            let ptr = &*node_guard as *const TrieNode<_, _, _>;
            node_ptrs.insert(ptr);
        }
        assert_eq!(node_ptrs.len(), 4); // Confirm unique pointers collected

        // Verify presence of all nodes (by comparing pointers)
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
        // Structure: root(0) -> c1(1) -> gc(3), root(0) -> c2(2) -> gc(3)
        let root = Arc::new(Mutex::new(TrieNode::<(), &str, i32>::new(0)));
        let child1 = Arc::new(Mutex::new(TrieNode::new(1)));
        let child2 = Arc::new(Mutex::new(TrieNode::new(2)));
        let grandchild = Arc::new(Mutex::new(TrieNode::new(3)));

        root.try_lock().unwrap().insert("r->c1", child1.clone(), ());
        root.try_lock().unwrap().insert("r->c2", child2.clone(), ());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone(), ());
        child2.try_lock().unwrap().insert("c2->gc", grandchild.clone(), ()); // Diamond

        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();

        TrieNode::special_map(
            vec![(root.clone(), 100)],
            |parent_v, _edge, _ev, _child_node| parent_v + 1,
            |_| panic!("Merge should not be called"),
            |node_t_val, computed_v| {
                processed_node_values.push(*node_t_val);
                computed_values.push(*computed_v);
            },
        );

        // BFS ensures each node is processed exactly once.
        // Expected order: 0, then {1, 2} in some order, then 3.
        assert_eq!(processed_node_values.len(), 4);
        assert!(processed_node_values.contains(&0));
        assert!(processed_node_values.contains(&1));
        assert!(processed_node_values.contains(&2));
        assert!(processed_node_values.contains(&3));
        assert_eq!(processed_node_values[0], 0); // Root first
        assert_eq!(processed_node_values[3], 3); // Grandchild last

        // Values: root=100, c1=101, c2=101.
        // gc=102 (processed once, value derived from the first parent path reaching it in BFS).
        assert_eq!(computed_values.len(), 4);
        assert_eq!(computed_values[0], 100); // Root value
        assert!(computed_values[1..3].contains(&101)); // c1 and c2 values
        assert_eq!(computed_values[1..3].iter().filter(|&&v| v == 101).count(), 2);
        assert_eq!(computed_values[3], 102); // Grandchild value
    }
}