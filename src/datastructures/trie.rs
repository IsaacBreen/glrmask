use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::{Hash, Hasher}; // Needed for HashMap keys
use std::sync::{Arc, Mutex};

/// Represents a node in a Trie-like structure (allowing shared subtrees and DAGs).
///
/// `E`: Type of the edge label (must be comparable).
/// `T`: Type of the value stored within the node.
#[derive(Debug, Clone)]
pub struct Trie<E, T> {
    pub value: T,
    children: BTreeMap<E, Arc<Mutex<Trie<E, T>>>>,
    // Note: max_depth field is omitted as the in-degree approach is more suitable
    // for the merging requirement.
}

// Helper type alias for pointer representation used as map keys
type NodePtr<E, T> = *const Trie<E, T>;

// Helper to get the raw pointer of the node inside an Arc<Mutex<Trie>>.
// Panics if the mutex is poisoned.
fn node_ptr<E, T>(node_arc: &Arc<Mutex<Trie<E, T>>>) -> NodePtr<E, T> {
    // Using data_ptr() is potentially safer if the Mutex implementation details change,
    // but requires careful handling as it bypasses the lock temporarily.
    // For simplicity and consistency with the original code, we'll lock briefly.
    // If performance becomes critical, alternatives could be explored.
    let guard = node_arc.try_lock().expect("Mutex poisoned");
    &*guard as *const _
}

// Implement Hash and Eq for Arc<Mutex<Trie>> based on pointer equality
// This allows using Arc<Mutex<Trie>> directly in HashMaps/HashSets if needed,
// although using the raw pointer NodePtr is often more explicit.
impl<E, T> Hash for Trie<E, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self as *const Self).hash(state);
    }
}

impl<E, T> PartialEq for Trie<E, T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self as *const _, other as *const _)
    }
}
impl<E, T> Eq for Trie<E, T> {}


impl<T, E: Ord> Trie<E, T> {
    /// Creates a new Trie node with the given value and no children.
    pub fn new(value: T) -> Self {
        Trie {
            value,
            children: BTreeMap::new(),
        }
    }

    /// Inserts a child node associated with the given edge.
    ///
    /// Note: This implementation does *not* perform cycle detection. Adding an edge
    /// that creates a cycle may lead to infinite loops in *some* traversal algorithms
    /// if not handled carefully (like the `all_nodes` and `special_map_merge` below).
    pub fn insert(&mut self, edge: E, child: Arc<Mutex<Trie<E, T>>>) {
        // Allow overwriting for flexibility, or keep assert if needed.
        // assert!(self.children.insert(edge, child).is_none());
        self.children.insert(edge, child);
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
        let mut visited_ptrs: HashSet<NodePtr<E, T>> = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        let root_ptr = node_ptr(&root);
        if visited_ptrs.insert(root_ptr) {
            queue.push_back(root);
        }

        while let Some(node_arc) = queue.pop_front() {
            result.push(node_arc.clone()); // Collect the Arc itself

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


impl<T: Clone, E: Ord + Clone> Trie<E, T> {
    /// Performs a Breadth-First Search (BFS) traversal applying functions at each step,
    /// merging values for nodes reached via multiple paths before processing.
    ///
    /// Starts from `initial_nodes_and_values`. For each node encountered:
    /// 1. Computes contributions to its children's values using `step`.
    /// 2. Merges incoming contributions using `merge`.
    /// 3. Once all incoming edges *reachable from the initial set* have been processed,
    ///    the node is dequeued and `process` is called with its final merged value.
    ///
    /// Node uniqueness for visitation and state tracking is determined by the memory address.
    /// Handles DAGs and cycles correctly by processing nodes only when their dependencies are met.
    ///
    /// `V`: The type of the value being computed and merged during traversal. Must be Clone.
    /// `step`: `FnMut(&V, &E, &Trie<E, T>) -> V` - Computes the value contribution to a child.
    ///         Takes the parent's merged value, the edge label, and the locked child node.
    /// `merge`: `FnMut(&mut V, V)` - Merges a new value contribution into the node's current value.
    /// `process`: `FnMut(&T, &V)` - Processes the node's internal value (`T`) and its final merged value (`V`).
    pub fn special_map_merge<V: Clone>(
        initial_nodes_and_values: Vec<(Arc<Mutex<Trie<E, T>>>, V)>,
        mut step: impl FnMut(&V, &E, &Trie<E, T>) -> V,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&T, &V),
    ) {
        if initial_nodes_and_values.is_empty() {
            return; // Nothing to process
        }

        // --- Pass 1: Calculate In-degrees within the Reachable Subgraph ---
        let mut in_degree: HashMap<NodePtr<E, T>, usize> = HashMap::new();
        let mut visited_for_indegree: HashSet<NodePtr<E, T>> = HashSet::new();
        let mut queue_for_indegree: VecDeque<Arc<Mutex<Trie<E, T>>>> = VecDeque::new();

        // Initialize queue for in-degree calculation
        for (node_arc, _) in &initial_nodes_and_values {
            let ptr = node_ptr(node_arc);
            if visited_for_indegree.insert(ptr) {
                queue_for_indegree.push_back(node_arc.clone());
                in_degree.entry(ptr).or_insert(0); // Ensure initial nodes are in the map
            }
            // Don't increment in_degree here, only for target nodes of edges
        }

        // BFS for in-degree calculation
        let mut bfs_idx = 0; // Use index to iterate queue_for_indegree to avoid borrow checker issues
        while bfs_idx < queue_for_indegree.len() {
             let node_arc = queue_for_indegree[bfs_idx].clone();
             bfs_idx += 1;

            let node_guard = node_arc.try_lock().expect("Mutex poisoned during in-degree calc");
            for child_arc in node_guard.children.values() {
                let child_ptr = node_ptr(child_arc);
                // Increment in-degree for the child *within the reachable subgraph*
                *in_degree.entry(child_ptr).or_insert(0) += 1;
                if visited_for_indegree.insert(child_ptr) {
                    queue_for_indegree.push_back(child_arc.clone());
                }
            }
        }
        // `in_degree` now holds the count of incoming edges *from nodes reachable from the start set*.
        // Nodes not reachable from the start set will not be in `in_degree`.

        // --- Pass 2: Perform Merging BFS ---
        let mut process_queue: VecDeque<Arc<Mutex<Trie<E, T>>>> = VecDeque::new(); // Nodes ready to process
        let mut merged_values: HashMap<NodePtr<E, T>, V> = HashMap::new();
        let mut in_degree_remaining = in_degree; // Use the calculated in-degrees

        // Initialize merged_values and the processing queue
        for (node_arc, initial_value) in initial_nodes_and_values {
            let ptr = node_ptr(&node_arc);

            // Ensure the node is reachable (it must be, as it's an initial node)
            if !in_degree_remaining.contains_key(&ptr) {
                 // This case should ideally not happen if in-degree calculation was correct
                 // for initial nodes, but handle defensively.
                 in_degree_remaining.insert(ptr, 0);
            }

            match merged_values.entry(ptr) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    // Node provided multiple times in initial list, merge initial values
                    merge(entry.get_mut(), initial_value);
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    // First time seeing this initial node
                    entry.insert(initial_value);
                    // If a node has 0 *reachable* in-degree, it's ready to process immediately.
                    if in_degree_remaining.get(&ptr).cloned().unwrap_or(0) == 0 {
                         process_queue.push_back(node_arc.clone());
                    }
                }
            }
        }

        // Main merging loop (Topological Sort-like BFS)
        while let Some(node_arc) = process_queue.pop_front() {
            let ptr = node_ptr(&node_arc);

            // Retrieve the final merged value for processing. Panic if not found (logic error).
            let current_merged_value = merged_values.get(&ptr)
                .expect("Node in process queue must have a merged value")
                .clone(); // Clone needed for immutable borrow in step

            // --- Process the node ---
            { // Scope for node_guard
                let node_guard = node_arc.try_lock().expect("Mutex poisoned during process");
                process(&node_guard.value, &current_merged_value); // Use the final merged value

                // --- Propagate value to children ---
                for (edge, child_arc) in &node_guard.children {
                    let child_ptr = node_ptr(child_arc);

                    // Only consider children that are part of the reachable subgraph
                    // (i.e., they were visited during the in-degree calculation pass)
                    if let Some(remaining_count) = in_degree_remaining.get_mut(&child_ptr) {
                        // Calculate the value contribution from *this* parent
                        let next_value_contribution = { // Scope for child_guard lock
                            let child_guard = child_arc.try_lock().expect("Mutex poisoned for child step");
                            step(&current_merged_value, edge, &*child_guard)
                        }; // child_guard lock released

                        // Merge the contribution into the child's accumulating value
                        match merged_values.entry(child_ptr) {
                            std::collections::hash_map::Entry::Occupied(mut entry) => {
                                merge(entry.get_mut(), next_value_contribution);
                            }
                            std::collections::hash_map::Entry::Vacant(entry) => {
                                // First contribution received for this child
                                entry.insert(next_value_contribution);
                            }
                        }

                        // Decrement the count of pending incoming edges for the child
                        *remaining_count -= 1;
                        if *remaining_count == 0 {
                            // All reachable incoming edges processed, child is ready
                            process_queue.push_back(child_arc.clone());
                        }
                    }
                    // If child_ptr is not in in_degree_remaining, it means it wasn't reachable
                    // from the initial set during the first pass, so we ignore it for this traversal.
                }
            } // node_guard lock released
        }

        // After the loop, nodes remaining in `in_degree_remaining` with counts > 0
        // are part of cycles within the reachable subgraph or their predecessors
        // were part of such cycles, preventing them from being processed.
        // This is the expected behavior for this algorithm.
    }
}


/// Helper function to print the structure of the Trie/DAG starting from root (BFS).
pub(crate) fn dump_structure<E: Debug, T: Debug>(root: Arc<Mutex<Trie<E, T>>>) {
    let mut queue = VecDeque::new();
    let mut seen: HashSet<NodePtr<E, T>> = HashSet::new();

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
    use std::cell::RefCell; // For tracking calls in tests

    #[test]
    fn test_insertion_and_retrieval() {
        let mut root = Trie::<&str, i32>::new(0);
        let child1 = Arc::new(Mutex::new(Trie::new(1)));
        let child2 = Arc::new(Mutex::new(Trie::new(2)));

        root.insert("a", child1.clone());
        root.insert("b", child2.clone());

        let retrieved_child1 = root.get(&"a").expect("Failed to get child 'a'");
        assert!(Arc::ptr_eq(&retrieved_child1, &child1));
        let retrieved_child2 = root.get(&"b").expect("Failed to get child 'b'");
        assert!(Arc::ptr_eq(&retrieved_child2, &child2));
        assert!(root.get(&"c").is_none());
        let children_keys: Vec<_> = root.children().keys().cloned().collect();
        assert_eq!(children_keys, vec!["a", "b"]);
        assert!(!root.is_leaf());
        assert!(child1.try_lock().unwrap().is_leaf());
    }

    #[test]
    fn test_all_nodes_diamond() {
        let root = Arc::new(Mutex::new(Trie::<&str, &str>::new("root")));
        let child1 = Arc::new(Mutex::new(Trie::new("child1")));
        let child2 = Arc::new(Mutex::new(Trie::new("child2")));
        let grandchild = Arc::new(Mutex::new(Trie::new("grandchild")));

        root.try_lock().unwrap().insert("r->c1", child1.clone());
        root.try_lock().unwrap().insert("r->c2", child2.clone());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone());
        child2.try_lock().unwrap().insert("c2->gc", grandchild.clone()); // Diamond

        let all_nodes = Trie::all_nodes(root.clone());
        assert_eq!(all_nodes.len(), 4);
        let node_ptrs: HashSet<_> = all_nodes.iter().map(|arc| node_ptr(arc)).collect();
        assert_eq!(node_ptrs.len(), 4);
        assert!(node_ptrs.contains(&node_ptr(&root)));
        assert!(node_ptrs.contains(&node_ptr(&child1)));
        assert!(node_ptrs.contains(&node_ptr(&child2)));
        assert!(node_ptrs.contains(&node_ptr(&grandchild)));
    }

     #[test]
    fn test_empty_trie_all_nodes() {
        let root = Arc::new(Mutex::new(Trie::<&str, i32>::new(42)));
        let nodes = Trie::all_nodes(root.clone());
        assert_eq!(nodes.len(), 1);
        assert!(Arc::ptr_eq(&nodes[0], &root));
        assert!(root.try_lock().unwrap().is_leaf());
    }

    #[test]
    fn test_cycle_all_nodes() {
        let root = Arc::new(Mutex::new(Trie::<&str, i32>::new(0)));
        let child = Arc::new(Mutex::new(Trie::new(1)));

        root.try_lock().unwrap().insert("r->c", child.clone());
        child.try_lock().unwrap().insert("c->r", root.clone());

        let nodes = Trie::all_nodes(root.clone());
        assert_eq!(nodes.len(), 2);
        let node_ptrs: HashSet<_> = nodes.iter().map(|arc| node_ptr(arc)).collect();
        assert_eq!(node_ptrs.len(), 2);
        assert!(node_ptrs.contains(&node_ptr(&root)));
        assert!(node_ptrs.contains(&node_ptr(&child)));
    }

    // --- Tests for special_map_merge ---

    #[test]
    fn test_special_map_merge_simple_bfs() {
        // Structure: root(0) -> c1(1) -> gc(3), root(0) -> c2(2)
        // Use merge function that just overwrites (simulates old behavior for simple tree)
        let root = Arc::new(Mutex::new(Trie::<&str, i32>::new(0)));
        let child1 = Arc::new(Mutex::new(Trie::new(1)));
        let child2 = Arc::new(Mutex::new(Trie::new(2)));
        let grandchild = Arc::new(Mutex::new(Trie::new(3)));

        root.try_lock().unwrap().insert("r->c1", child1.clone());
        root.try_lock().unwrap().insert("r->c2", child2.clone());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone());

        let processed_order = RefCell::new(Vec::new()); // Track processing order (node T value)
        let processed_values = RefCell::new(HashMap::new()); // Track final merged V value per node T

        Trie::special_map_merge(
            vec![(root.clone(), 100)], // Start BFS from root with initial value 100
            |parent_v, _edge, _child_node| parent_v + 1, // Step: increment value
            |current_v, new_v| *current_v = new_v, // Merge: Overwrite (for this test)
            |node_t_val, final_v| {
                processed_order.borrow_mut().push(*node_t_val);
                processed_values.borrow_mut().insert(*node_t_val, *final_v);
            },
        );

        // Expected BFS processing order: 0, 1, 2, 3 (order of 1 and 2 might swap)
        let order = processed_order.borrow();
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], 0); // Root first
        assert!(order[1..3].contains(&1));
        assert!(order[1..3].contains(&2));
        assert_eq!(order[3], 3); // Grandchild last

        // Expected final values: root=100, c1=101, c2=101, gc=102 (from c1)
        let values = processed_values.borrow();
        assert_eq!(values.len(), 4);
        assert_eq!(values[&0], 100);
        assert_eq!(values[&1], 101);
        assert_eq!(values[&2], 101);
        assert_eq!(values[&3], 102);
    }

    #[test]
    fn test_special_map_merge_diamond_summing() {
        // Structure: root(0) -> c1(1) -> gc(3), root(0) -> c2(2) -> gc(3)
        // Merge by summing contributions.
        let root = Arc::new(Mutex::new(Trie::<&str, i32>::new(0)));
        let child1 = Arc::new(Mutex::new(Trie::new(1)));
        let child2 = Arc::new(Mutex::new(Trie::new(2)));
        let grandchild = Arc::new(Mutex::new(Trie::new(3)));

        root.try_lock().unwrap().insert("r->c1", child1.clone());
        root.try_lock().unwrap().insert("r->c2", child2.clone());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone());
        child2.try_lock().unwrap().insert("c2->gc", grandchild.clone()); // Diamond

        let processed_order = RefCell::new(Vec::new());
        let processed_values = RefCell::new(HashMap::new());

        Trie::special_map_merge(
            vec![(root.clone(), 100)], // Start value
            |parent_v, _edge, _child_node| parent_v + 1, // Step: contribution is parent_v + 1
            |current_v, new_v| *current_v += new_v, // Merge: Sum contributions
            |node_t_val, final_v| {
                processed_order.borrow_mut().push(*node_t_val);
                processed_values.borrow_mut().insert(*node_t_val, *final_v);
            },
        );

        // Expected processing order: 0, {1, 2}, 3
        let order = processed_order.borrow();
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], 0);
        assert!(order[1..3].contains(&1));
        assert!(order[1..3].contains(&2));
        assert_eq!(order[3], 3); // Grandchild processed last

        // Expected final values:
        // root=100 (initial)
        // c1 = step(100) = 101
        // c2 = step(100) = 101
        // gc = merge(step(c1_val), step(c2_val)) = merge(step(101), step(101))
        //    = merge(102, 102) = 102 + 102 = 204
        let values = processed_values.borrow();
        assert_eq!(values.len(), 4);
        assert_eq!(values[&0], 100);
        assert_eq!(values[&1], 101);
        assert_eq!(values[&2], 101);
        assert_eq!(values[&3], 204); // Sum of contributions from c1 and c2
    }

     #[test]
    fn test_special_map_merge_cycle() {
        // root(0) -> child(1) -> root(0) (cycle)
        let root = Arc::new(Mutex::new(Trie::<&str, i32>::new(0)));
        let child = Arc::new(Mutex::new(Trie::new(1)));

        root.try_lock().unwrap().insert("r->c", child.clone());
        child.try_lock().unwrap().insert("c->r", root.clone());

        let processed_order = RefCell::new(Vec::new());
        let processed_values = RefCell::new(HashMap::new());

        Trie::special_map_merge(
            vec![(root.clone(), 100)], // Start at root
            |v, _e, _n| v + 1,        // Step: increment
            |current_v, new_v| *current_v = new_v.max(*current_v), // Merge: take max
            |t, v| {
                processed_order.borrow_mut().push(*t);
                processed_values.borrow_mut().insert(*t, *v);
            },
        );

        // Because of the cycle and the in-degree counting:
        // 1. In-degree pass: root reachable, child reachable. in_degree[root]=1, in_degree[child]=1.
        // 2. Init: merged_values[root]=100. in_degree_remaining[root]=1, in_degree_remaining[child]=1.
        //    Neither root nor child has 0 remaining in-degree initially.
        // 3. Nothing is added to the process_queue initially.
        // 4. The while loop `while let Some(node_arc) = process_queue.pop_front()` never runs.
        // Result: Nothing should be processed.

        assert!(processed_order.borrow().is_empty());
        assert!(processed_values.borrow().is_empty());
    }

     #[test]
    fn test_special_map_merge_cycle_with_entry_point() {
        // entry(10) -> root(0) -> child(1) -> root(0)
        // Start the process from 'entry'.
        let entry = Arc::new(Mutex::new(Trie::<&str, i32>::new(10)));
        let root = Arc::new(Mutex::new(Trie::<&str, i32>::new(0)));
        let child = Arc::new(Mutex::new(Trie::new(1)));

        entry.try_lock().unwrap().insert("e->r", root.clone());
        root.try_lock().unwrap().insert("r->c", child.clone());
        child.try_lock().unwrap().insert("c->r", root.clone()); // Cycle back to root

        let processed_order = RefCell::new(Vec::new());
        let processed_values = RefCell::new(HashMap::new());

        Trie::special_map_merge(
            vec![(entry.clone(), 50)], // Start at entry
            |v, _e, _n| v + 1,        // Step: increment
            |current_v, new_v| *current_v = new_v.max(*current_v), // Merge: take max
            |t, v| {
                processed_order.borrow_mut().push(*t);
                processed_values.borrow_mut().insert(*t, *v);
            },
        );

        // Expected processing:
        // 1. In-degree pass: entry, root, child reachable.
        //    in_degree[entry]=0, in_degree[root]=2 (from entry, from child), in_degree[child]=1 (from root).
        // 2. Init: merged_values[entry]=50. in_degree_remaining = {entry:0, root:2, child:1}.
        //    process_queue = [entry] (since in_degree is 0).
        // 3. Dequeue 'entry'. Process entry(10) with value 50.
        //    Propagate to 'root': contribution = step(50) = 51.
        //    merged_values[root] = 51. in_degree_remaining[root] = 1. (Not ready)
        // 4. Queue empty. Loop ends.
        // Result: Only 'entry' should be processed. 'root' and 'child' are stuck in the cycle dependency.

        assert_eq!(*processed_order.borrow(), vec![10]);
        let values = processed_values.borrow();
        assert_eq!(values.len(), 1);
        assert_eq!(values[&10], 50);
    }

    #[test]
    fn test_special_map_merge_multiple_initial_nodes() {
        // Structure: c1(1), c2(2). Both -> gc(3)
        let child1 = Arc::new(Mutex::new(Trie::new(1)));
        let child2 = Arc::new(Mutex::new(Trie::new(2)));
        let grandchild = Arc::new(Mutex::new(Trie::new(3)));

        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone());
        child2.try_lock().unwrap().insert("c2->gc", grandchild.clone());

        let processed_order = RefCell::new(Vec::new());
        let processed_values = RefCell::new(HashMap::new());

        Trie::special_map_merge(
            vec![(child1.clone(), 10), (child2.clone(), 20)], // Start from c1 and c2
            |v, _e, _n| v + 5, // Step: add 5
            |current_v, new_v| *current_v += new_v, // Merge: sum
            |t, v| {
                processed_order.borrow_mut().push(*t);
                processed_values.borrow_mut().insert(*t, *v);
            },
        );

        // Expected processing:
        // 1. In-degree: c1, c2, gc reachable. in_degree[c1]=0, in_degree[c2]=0, in_degree[gc]=2.
        // 2. Init: merged[c1]=10, merged[c2]=20. remaining = {c1:0, c2:0, gc:2}.
        //    queue = [c1, c2] (order might vary).
        // 3. Dequeue c1. Process c1(1) with value 10.
        //    Propagate to gc: contribution = step(10) = 15.
        //    merged[gc] = 15. remaining[gc] = 1.
        // 4. Dequeue c2. Process c2(2) with value 20.
        //    Propagate to gc: contribution = step(20) = 25.
        //    Merge into gc: merged[gc] = 15 + 25 = 40. remaining[gc] = 0.
        //    Enqueue gc.
        // 5. Dequeue gc. Process gc(3) with value 40.
        //    No children.
        // 6. Queue empty.

        let order = processed_order.borrow();
        assert_eq!(order.len(), 3);
        assert!(order[0..2].contains(&1)); // c1 processed
        assert!(order[0..2].contains(&2)); // c2 processed
        assert_eq!(order[2], 3);          // gc processed last

        let values = processed_values.borrow();
        assert_eq!(values.len(), 3);
        assert_eq!(values[&1], 10);
        assert_eq!(values[&2], 20);
        assert_eq!(values[&3], 40); // 15 (from c1) + 25 (from c2)
    }

     #[test]
    fn test_special_map_merge_unreachable_part() {
        // root -> child1 -> gc
        // isolated_node
        let root = Arc::new(Mutex::new(Trie::<&str, i32>::new(0)));
        let child1 = Arc::new(Mutex::new(Trie::new(1)));
        let grandchild = Arc::new(Mutex::new(Trie::new(3)));
        let _isolated_node = Arc::new(Mutex::new(Trie::<&str, i32>::new(99))); // Not connected

        root.try_lock().unwrap().insert("r->c1", child1.clone());
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone());

        let processed_order = RefCell::new(Vec::new());
        let processed_values = RefCell::new(HashMap::new());

        Trie::special_map_merge(
            vec![(root.clone(), 100)], // Start only from root
            |v, _e, _n| v + 1,
            |curr, new| *curr = new, // Merge: overwrite
            |t, v| {
                 processed_order.borrow_mut().push(*t);
                 processed_values.borrow_mut().insert(*t, *v);
            },
        );

        // Only root, child1, grandchild should be processed
        assert_eq!(*processed_order.borrow(), vec![0, 1, 3]); // BFS order
        let values = processed_values.borrow();
        assert_eq!(values.len(), 3);
        assert_eq!(values[&0], 100);
        assert_eq!(values[&1], 101);
        assert_eq!(values[&3], 102);
        assert!(!values.contains_key(&99)); // Isolated node not processed
    }

     #[test]
    fn test_special_map_merge_empty_initial() {
        let root = Arc::new(Mutex::new(Trie::<&str, i32>::new(0)));
        let child1 = Arc::new(Mutex::new(Trie::new(1)));
        root.try_lock().unwrap().insert("r->c1", child1.clone());

        let processed_order = RefCell::new(Vec::new());

        Trie::special_map_merge::<i32>( // Explicit type V needed if vec is empty
            vec![], // Empty initial set
            |v, _e: &String, _n| v + 1,
            |curr, new| *curr = new,
            |t: &i32, _v| {
                 processed_order.borrow_mut().push(*t);
            },
        );

        assert!(processed_order.borrow().is_empty()); // Nothing processed
    }
}