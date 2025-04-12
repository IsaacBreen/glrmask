use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

/// Represents a node in a Trie-like structure (allowing shared subtrees and DAGs).
///
/// `E`: Type of the edge label (must be comparable).
/// `T`: Type of the value stored within the node.
#[derive(Debug, Clone)]
pub struct Trie<E, T> {
    pub value: T,
    children: BTreeMap<E, Arc<Mutex<Trie<E, T>>>>,
    max_depth: usize, // New field to track maximum depth
}

// Helper to get the raw pointer of the node inside an Arc<Mutex<Trie>>.
// Panics if the mutex is poisoned.
fn node_ptr<E, T>(node_arc: &Arc<Mutex<Trie<E, T>>>) -> *const Trie<E, T> {
    let guard = node_arc.try_lock().expect("Mutex poisoned");
    &*guard as *const _
}

impl<T, E: Ord> Trie<E, T> {
    pub fn new(value: T) -> Self {
        Trie {
            value,
            children: BTreeMap::new(),
            max_depth: 0, // Initially 0 for source nodes
        }
    }

    pub fn insert(&mut self, edge: E, child: Arc<Mutex<Trie<E, T>>>) {
        // Update child's max_depth if this path increases it
        let new_depth = self.max_depth + 1;
        {
            let mut child_guard = child.try_lock().expect("Mutex poisoned during insert");
            if new_depth > child_guard.max_depth {
                child_guard.max_depth = new_depth;
            }
        }

        // Ensure we don't overwrite an existing edge.
        assert!(self.children.insert(edge, child).is_none());
    }

    /// Gets the child node associated with the given edge, if it exists.
    pub fn get(&self, edge: &E) -> Option<Arc<Mutex<Trie<E, T>>>> {
        self.children.get(edge).cloned()
    }

    /// Returns a reference to the map of children nodes.
    pub fn children(&self) -> &BTreeMap<E, Arc<Mutex<Trie<E, T>>>> {
        &self.children
    }

    /// Checks if the node has any children (i.e., is a leaf).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Collects all unique nodes reachable from the given root using Breadth-First Search (BFS).
    /// Node uniqueness is determined by the memory address of the `Trie` data.
    pub fn all_nodes(root: Arc<Mutex<Trie<E, T>>>) -> Vec<Arc<Mutex<Trie<E, T>>>> {
        let mut visited_ptrs: HashSet<*const Trie<E, T>> = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        let root_ptr = node_ptr(&root);
        if visited_ptrs.insert(root_ptr) {
            queue.push_back(root);
        }

        while let Some(node_arc) = queue.pop_front() {
            result.push(node_arc.clone());

            let node_guard = node_arc.try_lock().expect("Mutex poisoned during BFS");
            for child_arc in node_guard.children.values() {
                let child_ptr = node_ptr(child_arc);
                if visited_ptrs.insert(child_ptr) {
                    queue.push_back(child_arc.clone());
                }
            }
        }
        result
    }
}

impl<T: Clone + std::cmp::Eq, E: Ord + Clone> Trie<E, T> {
    /// Performs a modified Breadth-First Search (BFS) traversal with value merging and depth ordering.
    ///
    /// - Nodes are processed in order of their max_depth (highest first)
    /// - A node is only processed after all its incoming edges (reachable from starting nodes) are traversed
    /// - Values for nodes with multiple incoming paths are merged using the provided merge function
    pub fn special_map<V: Clone + std::cmp::Eq>(
        initial_nodes_and_values: Vec<(Arc<Mutex<Trie<E, T>>>, V)>,
        mut step: impl FnMut(&V, &E, &Trie<E, T>) -> V,
        mut process: impl FnMut(&T, &V),
        mut merge: impl FnMut(&mut V, V),
    ) {
        // Maps node pointers to their current value
        let mut node_values: HashMap<*const Trie<E, T>, V> = HashMap::new();

        // Maps node pointers to their remaining in-degree
        let mut in_degree: HashMap<*const Trie<E, T>, usize> = HashMap::new();

        // Maps node pointers to their Arc<Mutex<Trie>> for easy lookup
        let mut node_arcs: HashMap<*const Trie<E, T>, Arc<Mutex<Trie<E, T>>>> = HashMap::new();

        // First pass: discover all reachable nodes and count in-degrees
        let mut discovery_queue = VecDeque::new();

        // Initialize with starting nodes
        for (node_arc, value) in &initial_nodes_and_values {
            let ptr = node_ptr(node_arc);
            node_values.insert(ptr, value.clone());
            in_degree.insert(ptr, 0); // No incoming edges for start nodes
            node_arcs.insert(ptr, node_arc.clone());
            discovery_queue.push_back(node_arc.clone());
        }

        // BFS to count in-degrees for all reachable nodes
        while let Some(node_arc) = discovery_queue.pop_front() {
            let node_guard = node_arc.try_lock().expect("Mutex poisoned during discovery");

            for (_, child_arc) in &node_guard.children {
                let child_ptr = node_ptr(&child_arc);

                // Increment in-degree for child
                *in_degree.entry(child_ptr).or_insert(0) += 1;

                // Add child to node_arcs and discovery_queue if not already discovered
                if !node_arcs.contains_key(&child_ptr) {
                    node_arcs.insert(child_ptr, child_arc.clone());
                    discovery_queue.push_back(child_arc.clone());
                }
            }
        }

        // Second pass: process nodes in order of max_depth
        // Custom struct for the priority queue
        struct QueueEntry<E: Ord + Clone, T: Clone + Eq, V: Clone + Eq> {
            max_depth: usize,
            node_ptr: *const Trie<E, T>,
            node_arc: Arc<Mutex<Trie<E, T>>>,
            value: V,
        }

        impl<E: Ord + Clone, T: Clone + Eq, V: Clone + Eq> PartialEq for QueueEntry<E, T, V> {
            fn eq(&self, other: &Self) -> bool {
                self.max_depth == other.max_depth && self.node_ptr == other.node_ptr
            }
        }
        
        impl<E: Ord + Clone, T: Clone + Eq, V: Clone + Eq> Eq for QueueEntry<E, T, V> {}

        // Implement ordering for the priority queue (max-heap by depth)
        impl<E: Ord + Clone, T: Clone + Eq, V: Clone + Eq> Ord for QueueEntry<E, T, V> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.max_depth.cmp(&other.max_depth).reverse() // Reverse for max-heap
            }
        }

        impl<E: Ord + Clone, T: Clone + PartialEq + std::cmp::Eq, V: Clone + PartialEq + std::cmp::Eq> PartialOrd for QueueEntry<E, T, V> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        // Priority queue to process nodes in order of max_depth
        let mut processing_queue = std::collections::BinaryHeap::new();

        // Add initial nodes with in-degree 0 to processing queue
        for (node_arc, value) in initial_nodes_and_values {
            let ptr = node_ptr(&node_arc);
            if in_degree[&ptr] == 0 {
                let depth = node_arc.try_lock().expect("Mutex poisoned").max_depth;
                processing_queue.push(QueueEntry {
                    max_depth: depth,
                    node_ptr: ptr,
                    node_arc,
                    value: node_values[&ptr].clone(),
                });
            }
        }

        // Process nodes in order of max_depth
        while let Some(entry) = processing_queue.pop() {
            let node_arc = entry.node_arc;
            let node_guard = node_arc.try_lock().expect("Mutex poisoned during processing");

            // Process this node with its final merged value
            process(&node_guard.value, &entry.value);

            // Process children and update their in-degrees
            for (edge, child_arc) in &node_guard.children {
                let child_ptr = node_ptr(child_arc);

                // Compute new value for child
                let child_guard = child_arc.try_lock().expect("Mutex poisoned");
                let next_value = step(&entry.value, edge, &child_guard);
                let child_depth = child_guard.max_depth;
                drop(child_guard); // Release lock early

                // Update child's value (merge if already exists)
                if let Some(existing_value) = node_values.get_mut(&child_ptr) {
                    // Merge the new value with existing value
                    merge(existing_value, next_value);
                } else {
                    node_values.insert(child_ptr, next_value);
                }

                // Decrement child's in-degree
                if let Some(degree) = in_degree.get_mut(&child_ptr) {
                    *degree -= 1;

                    // If all incoming edges processed, add to queue
                    if *degree == 0 {
                        let child_value = node_values[&child_ptr].clone();
                        processing_queue.push(QueueEntry {
                            max_depth: child_depth,
                            node_ptr: child_ptr,
                            node_arc: child_arc.clone(),
                            value: child_value,
                        });
                    }
                }
            }
        }
    }
}

/// Helper function to print the structure of the Trie/DAG starting from root (BFS).
pub(crate) fn dump_structure<E: Debug, T: Debug>(root: Arc<Mutex<Trie<E, T>>>) {
    let mut queue = VecDeque::new();
    let mut seen: HashSet<*const Trie<E, T>> = HashSet::new();

    println!("Dumping Trie Structure (BFS):");

    let root_ptr = node_ptr(&root);
    if seen.insert(root_ptr) {
        queue.push_back(root);
    }

    while let Some(node_arc) = queue.pop_front() {
        let node_guard = node_arc.try_lock().expect("Mutex poisoned during dump");
        let node_ptr_val = &*node_guard as *const _; // Get pointer again for printing

        println!("{:?}: Value: {:?}", node_ptr_val, node_guard.value);

        for (edge, child_arc) in &node_guard.children {
            let child_ptr_val = node_ptr(child_arc); // Get pointer for printing/checking
            println!("  - Edge: {:?} -> Child: {:?}", edge, child_ptr_val);
            if seen.insert(child_ptr_val) {
                queue.push_back(child_arc.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insertion_and_retrieval() {
        let mut root = Trie::<&str, i32>::new(0);
        let child1 = Arc::new(Mutex::new(Trie::new(1)));
        let child2 = Arc::new(Mutex::new(Trie::new(2)));

        root.insert("a", child1.clone());
        root.insert("b", child2.clone());

        // Test get
        let retrieved_child1 = root.get(&"a").expect("Failed to get child 'a'");
        assert!(Arc::ptr_eq(&retrieved_child1, &child1));

        let retrieved_child2 = root.get(&"b").expect("Failed to get child 'b'");
        assert!(Arc::ptr_eq(&retrieved_child2, &child2));

        assert!(root.get(&"c").is_none());

        // Test children iterator order (BTreeMap)
        let children_keys: Vec<_> = root.children().keys().cloned().collect();
        assert_eq!(children_keys, vec!["a", "b"]);

        // Test is_leaf
        assert!(!root.is_leaf());
        assert!(child1.try_lock().unwrap().is_leaf());
    }

    #[test]
    fn test_special_map_bfs_order() {
        // Structure: root(0) -> c1(1) -> gc(3), root(0) -> c2(2)
        let root = Arc::new(Mutex::new(Trie::<&str, i32>::new(0)));
        let child1 = Arc::new(Mutex::new(Trie::new(1)));
        let child2 = Arc::new(Mutex::new(Trie::new(2)));
        let grandchild = Arc::new(Mutex::new(Trie::new(3)));

        root.try_lock().unwrap().insert("r->c1", child1.clone());
        root.try_lock().unwrap().insert("r->c2", child2.clone());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone());

        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();

        Trie::special_map(
            vec![(root.clone(), 100)], // Start BFS from root with initial value 100
            |parent_v, _edge, _child_node| parent_v + 1, // Step: increment value
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
        let root = Arc::new(Mutex::new(Trie::<&str, &str>::new("root")));
        let child1 = Arc::new(Mutex::new(Trie::new("child1")));
        let child2 = Arc::new(Mutex::new(Trie::new("child2")));
        let grandchild = Arc::new(Mutex::new(Trie::new("grandchild")));

        root.try_lock().unwrap().insert("r->c1", child1.clone());
        root.try_lock().unwrap().insert("r->c2", child2.clone());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone());
        child2.try_lock().unwrap().insert("c2->gc", grandchild.clone()); // Diamond

        let all_nodes = Trie::all_nodes(root.clone());

        // Should find 4 unique nodes
        assert_eq!(all_nodes.len(), 4);

        // Verify uniqueness using pointers
        let node_ptrs: HashSet<_> = all_nodes.iter().map(|arc| node_ptr(arc)).collect();
        assert_eq!(node_ptrs.len(), 4); // Confirm unique pointers collected

        // Verify presence of all nodes (by comparing pointers)
        assert!(node_ptrs.contains(&node_ptr(&root)));
        assert!(node_ptrs.contains(&node_ptr(&child1)));
        assert!(node_ptrs.contains(&node_ptr(&child2)));
        assert!(node_ptrs.contains(&node_ptr(&grandchild)));
    }

    #[test]
    fn test_special_map_diamond() {
        // Structure: root(0) -> c1(1) -> gc(3), root(0) -> c2(2) -> gc(3)
        let root = Arc::new(Mutex::new(Trie::<&str, i32>::new(0)));
        let child1 = Arc::new(Mutex::new(Trie::new(1)));
        let child2 = Arc::new(Mutex::new(Trie::new(2)));
        let grandchild = Arc::new(Mutex::new(Trie::new(3)));

        root.try_lock().unwrap().insert("r->c1", child1.clone());
        root.try_lock().unwrap().insert("r->c2", child2.clone());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone());
        child2.try_lock().unwrap().insert("c2->gc", grandchild.clone()); // Diamond

        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();

        Trie::special_map(
            vec![(root.clone(), 100)],
            |parent_v, _edge, _child_node| parent_v + 1,
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

     #[test]
    fn test_empty_trie() {
        let root = Arc::new(Mutex::new(Trie::<&str, i32>::new(42)));
        let nodes = Trie::all_nodes(root.clone());
        assert_eq!(nodes.len(), 1);
        assert!(Arc::ptr_eq(&nodes[0], &root));
        assert!(root.try_lock().unwrap().is_leaf());

        let mut processed = false;
        Trie::special_map(
            vec![(root.clone(), 100)],
            |_v, _e, _n| panic!("Step should not be called for leaf"),
            |t, v| {
                assert_eq!(*t, 42);
                assert_eq!(*v, 100);
                processed = true;
            },
        );
        assert!(processed);
    }

    #[test]
    fn test_cycle_all_nodes() {
        // root -> child -> root (cycle)
        let root = Arc::new(Mutex::new(Trie::<&str, i32>::new(0)));
        let child = Arc::new(Mutex::new(Trie::new(1)));

        root.try_lock().unwrap().insert("r->c", child.clone());
        child.try_lock().unwrap().insert("c->r", root.clone());

        let nodes = Trie::all_nodes(root.clone());
        // Should detect both nodes despite the cycle
        assert_eq!(nodes.len(), 2);
        let node_ptrs: HashSet<_> = nodes.iter().map(|arc| node_ptr(arc)).collect();
        assert_eq!(node_ptrs.len(), 2);
        assert!(node_ptrs.contains(&node_ptr(&root)));
        assert!(node_ptrs.contains(&node_ptr(&child)));
    }

     #[test]
    fn test_cycle_special_map() {
        // root -> child -> root (cycle)
        let root = Arc::new(Mutex::new(Trie::<&str, i32>::new(0)));
        let child = Arc::new(Mutex::new(Trie::new(1)));

        root.try_lock().unwrap().insert("r->c", child.clone());
        child.try_lock().unwrap().insert("c->r", root.clone());

        let mut processed_values = Vec::new();
        let mut computed_vals = Vec::new();

        Trie::special_map(
            vec![(root.clone(), 100)],
            |v, _e, _n| v + 1,
            |t, v| {
                processed_values.push(*t);
                computed_vals.push(*v);
            },
        );

        // Should process each node exactly once due to visited set
        assert_eq!(processed_values.len(), 2);
        assert!(processed_values.contains(&0));
        assert!(processed_values.contains(&1));
        assert_eq!(processed_values[0], 0); // Root processed first

        assert_eq!(computed_vals.len(), 2);
        assert_eq!(computed_vals[0], 100); // Root value
        assert_eq!(computed_vals[1], 101); // Child value (from root)
    }
}