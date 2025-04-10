use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

/// Represents a node in a Trie-like structure (allowing shared subtrees and DAGs).
///
/// `E`: Type of the edge label (must be comparable).
/// `T`: Type of the value stored within the node.
#[derive(Debug, Clone)]
pub struct TrieNode<E, T> {
    pub value: T,
    children: BTreeMap<E, Arc<Mutex<TrieNode<E, T>>>>,
}

// Helper to get the raw pointer of the node inside an Arc<Mutex<TrieNode>>.
// Panics if the mutex is poisoned.
fn node_ptr<E, T>(node_arc: &Arc<Mutex<TrieNode<E, T>>>) -> *const TrieNode<E, T> {
    let guard = node_arc.try_lock().expect("Mutex poisoned");
    &*guard as *const _
}

impl<T, E: Ord> TrieNode<E, T> {
    /// Creates a new TrieNode with the given value and no children.
    pub fn new(value: T) -> Self {
        TrieNode {
            value,
            children: BTreeMap::new(),
        }
    }

    /// Inserts a child node associated with the given edge.
    ///
    /// Note: This implementation does *not* perform cycle detection. Adding an edge
    /// that creates a cycle may lead to infinite loops in traversal algorithms.
    pub fn insert(&mut self, edge: E, child: Arc<Mutex<TrieNode<E, T>>>) {
        self.children.insert(edge, child);
    }

    /// Gets the child node associated with the given edge, if it exists.
    pub fn get(&self, edge: &E) -> Option<Arc<Mutex<TrieNode<E, T>>>> {
        self.children.get(edge).cloned()
    }

    /// Returns a reference to the map of children nodes.
    pub fn children(&self) -> &BTreeMap<E, Arc<Mutex<TrieNode<E, T>>>> {
        &self.children
    }

    /// Checks if the node has any children.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Collects all unique nodes reachable from the given root using Breadth-First Search (BFS).
    /// Node uniqueness is determined by the memory address of the `TrieNode` data.
    pub fn all_nodes(root: Arc<Mutex<TrieNode<E, T>>>) -> Vec<Arc<Mutex<TrieNode<E, T>>>> {
        let mut visited_ptrs: HashSet<*const TrieNode<E, T>> = HashSet::new();
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

impl<T: Clone, E: Ord + Clone> TrieNode<E, T> {
    /// Performs a Breadth-First Search (BFS) traversal applying functions at each step.
    ///
    /// Starts from `initial_nodes_and_values`. For each visited node:
    /// 1. Calls `process` with the node's internal value (`T`) and the computed value (`V`).
    /// 2. For each child, computes a new value `V` using `step` and enqueues the child if not already visited.
    ///
    /// Node uniqueness for visitation is determined by the memory address of the `TrieNode` data.
    pub fn special_map<V: Clone>(
        initial_nodes_and_values: Vec<(Arc<Mutex<TrieNode<E, T>>>, V)>,
        mut step: impl FnMut(&V, &E, &TrieNode<E, T>) -> V,
        mut process: impl FnMut(&T, &V),
    ) {
        let mut queue: VecDeque<(Arc<Mutex<TrieNode<E, T>>>, V)> = VecDeque::new();
        let mut visited: HashSet<*const TrieNode<E, T>> = HashSet::new();

        // Initialize queue and visited set
        for (node_arc, value) in initial_nodes_and_values {
            let ptr = node_ptr(&node_arc);
            if visited.insert(ptr) {
                queue.push_back((node_arc, value));
            }
        }

        while let Some((node_arc, current_value)) = queue.pop_front() {
            // Lock the node to access its data and children
            let node_guard = node_arc.try_lock().expect("Mutex poisoned during special_map");

            // Process the current node
            process(&node_guard.value, &current_value);

            // Prepare children for the next level of BFS
            for (edge, child_arc) in &node_guard.children {
                // Lock child briefly only if it hasn't been visited yet
                let child_ptr = node_ptr(child_arc);
                if visited.insert(child_ptr) {
                    let child_guard = child_arc.try_lock().expect("Mutex poisoned for child");
                    let next_value = step(&current_value, edge, &child_guard);
                    // Release child lock implicitly here
                    queue.push_back((child_arc.clone(), next_value));
                }
            }
            // Release node lock implicitly here
        }
    }
}

/// Helper function to print the structure of the Trie/DAG starting from root (BFS).
pub(crate) fn dump_structure<E: Debug, T: Debug>(root: Arc<Mutex<TrieNode<E, T>>>) {
    let mut queue = VecDeque::new();
    let mut seen: HashSet<*const TrieNode<E, T>> = HashSet::new();

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
        let mut root = TrieNode::<&str, i32>::new(0);
        let child1 = Arc::new(Mutex::new(TrieNode::new(1)));
        let child2 = Arc::new(Mutex::new(TrieNode::new(2)));

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
        let root = Arc::new(Mutex::new(TrieNode::<&str, i32>::new(0)));
        let child1 = Arc::new(Mutex::new(TrieNode::new(1)));
        let child2 = Arc::new(Mutex::new(TrieNode::new(2)));
        let grandchild = Arc::new(Mutex::new(TrieNode::new(3)));

        root.try_lock().unwrap().insert("r->c1", child1.clone());
        root.try_lock().unwrap().insert("r->c2", child2.clone());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone());

        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();

        TrieNode::special_map(
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
        let root = Arc::new(Mutex::new(TrieNode::<&str, &str>::new("root")));
        let child1 = Arc::new(Mutex::new(TrieNode::new("child1")));
        let child2 = Arc::new(Mutex::new(TrieNode::new("child2")));
        let grandchild = Arc::new(Mutex::new(TrieNode::new("grandchild")));

        root.try_lock().unwrap().insert("r->c1", child1.clone());
        root.try_lock().unwrap().insert("r->c2", child2.clone());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone());
        child2.try_lock().unwrap().insert("c2->gc", grandchild.clone()); // Diamond

        let all_nodes = TrieNode::all_nodes(root.clone());

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
        let root = Arc::new(Mutex::new(TrieNode::<&str, i32>::new(0)));
        let child1 = Arc::new(Mutex::new(TrieNode::new(1)));
        let child2 = Arc::new(Mutex::new(TrieNode::new(2)));
        let grandchild = Arc::new(Mutex::new(TrieNode::new(3)));

        root.try_lock().unwrap().insert("r->c1", child1.clone());
        root.try_lock().unwrap().insert("r->c2", child2.clone());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone());
        child2.try_lock().unwrap().insert("c2->gc", grandchild.clone()); // Diamond

        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();

        TrieNode::special_map(
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
        let root = Arc::new(Mutex::new(TrieNode::<&str, i32>::new(42)));
        let nodes = TrieNode::all_nodes(root.clone());
        assert_eq!(nodes.len(), 1);
        assert!(Arc::ptr_eq(&nodes[0], &root));
        assert!(root.try_lock().unwrap().is_leaf());

        let mut processed = false;
        TrieNode::special_map(
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
        let root = Arc::new(Mutex::new(TrieNode::<&str, i32>::new(0)));
        let child = Arc::new(Mutex::new(TrieNode::new(1)));

        root.try_lock().unwrap().insert("r->c", child.clone());
        child.try_lock().unwrap().insert("c->r", root.clone());

        let nodes = TrieNode::all_nodes(root.clone());
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
        let root = Arc::new(Mutex::new(TrieNode::<&str, i32>::new(0)));
        let child = Arc::new(Mutex::new(TrieNode::new(1)));

        root.try_lock().unwrap().insert("r->c", child.clone());
        child.try_lock().unwrap().insert("c->r", root.clone());

        let mut processed_values = Vec::new();
        let mut computed_vals = Vec::new();

        TrieNode::special_map(
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