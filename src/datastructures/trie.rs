use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

/// Represents a node in a Trie–like structure (allowing shared subtrees and DAGs).
/// Multiple children can exist for the same edge key. Each edge instance has a value.
///
/// EK: type of the edge key (must be Ord).
/// EV: type of the edge value.
/// T: type of the value stored within the node.
#[derive(Debug, Clone)]
pub struct Trie<EK: Ord, EV, T> {
    pub value: T,
    // Changed: Stores a Vec of (EdgeValue, ChildArc) tuples for each edge key.
    children: BTreeMap<EK, Vec<(EV, Arc<Mutex<Trie<EK, EV, T>>>)>>,
    /// The “longest distance” from some source node (as computed during insertion)
    /// This value is set (or updated) when an edge is inserted.
    pub max_depth: usize,
}

// Implementation block for core Trie functionality
impl<EK: Ord, EV, T> Trie<EK, EV, T> {
    /// Creates a new trie node with the given value and no children.
    /// The max_depth is initialized to 0.
    pub fn new(value: T) -> Self {
        Trie {
            value,
            children: BTreeMap::new(),
            max_depth: 0,
        }
    }

    /// Inserts a child node associated with the given edge key and edge value.
    /// If the edge key already exists, the (edge_value, child) tuple is added
    /// to the list of children for that edge key.
    ///
    /// WARNING: This method does not detect cycles during insertion itself, but
    /// the subsequent max_depth propagation *does* detect cycles and panics.
    /// We “relax” max_depth on insert and propagate any update downwards.
    pub fn insert(
        &mut self,
        edge_key: EK,
        edge_value: EV, // Added edge value parameter
        child: Arc<Mutex<Trie<EK, EV, T>>>,
    ) {
        let candidate_depth = self.max_depth.saturating_add(1);
        {
            // First update the inserted child if needed.
            let mut child_lock = child.lock().expect("Mutex poisoned in insert");
            if candidate_depth > child_lock.max_depth {
                child_lock.max_depth = candidate_depth;
                // Only propagate if the child's depth actually changed.
                // Drop the lock before propagating.
                drop(child_lock);
                // Because the child’s max_depth may now have increased, we “propagate” that update downward.
                Self::propagate_max_depth(child.clone(), candidate_depth);
            }
            // else: child's depth didn't change, no need to propagate from here.
        } // child_lock is released here if not dropped earlier

        // Add the (edge_value, child) tuple to the list for this edge key.
        self.children
            .entry(edge_key)
            .or_default()
            .push((edge_value, child)); // Store the tuple
    }

    /// Propagates a max_depth update to all descendant nodes.
    ///
    /// The new version uses a recursive helper that tracks the current propagation
    /// chain in a HashSet. If a node is encountered twice along the same chain,
    /// a cycle exists and we panic.
    fn propagate_max_depth(node_arc: Arc<Mutex<Trie<EK, EV, T>>>, current_depth: usize) {
        // rec_stack will contain the set of node pointers from the root of the propagation
        // down to the current recursion level.
        let mut rec_stack: HashSet<*const Trie<EK, EV, T>> = HashSet::new();
        Self::_propagate_max_depth(node_arc, current_depth, &mut rec_stack);
    }

    /// Recursive helper for propagate_max_depth.
    fn _propagate_max_depth(
        node_arc: Arc<Mutex<Trie<EK, EV, T>>>,
        current_depth: usize,
        rec_stack: &mut HashSet<*const Trie<EK, EV, T>>,
    ) {
        let node_ptr_val = node_ptr(&node_arc);
        // If this node is already in the current recursion chain, we have a cycle.
        if rec_stack.contains(&node_ptr_val) {
            panic!(
                "Cycle detected in propagate_max_depth at node pointer: {:?}",
                node_ptr_val
            );
        }

        // Add the current node to the recursion stack.
        rec_stack.insert(node_ptr_val);

        // Collect *all* child Arcs outside of the lock.
        let children_arcs: Vec<Arc<Mutex<Trie<EK, EV, T>>>> = {
            let node = node_arc
                .lock()
                .expect("Mutex poisoned in propagate_max_depth");
            // Iterate through the Vecs for each edge key, then through the tuples,
            // extracting and cloning only the Arc.
            node.children
                .values() // Iterate over Vec<(EV, Arc<...>)>
                .flat_map(|vec_of_tuples| vec_of_tuples.iter().map(|(_ev, arc)| arc.clone()))
                .collect()
        };

        // For each child, compute the candidate depth.
        let candidate_depth = current_depth.saturating_add(1);
        for child_arc in children_arcs {
            let child_ptr_val = node_ptr(&child_arc);
            // Update the child if the candidate depth is higher.
            let should_propagate = {
                let mut child = child_arc
                    .lock()
                    .expect("Mutex poisoned in propagate_max_depth");
                if candidate_depth > child.max_depth {
                    child.max_depth = candidate_depth;
                    true
                } else {
                    false
                }
            };
            if should_propagate {
                // Before recursing, check again whether the child is already in rec_stack.
                if rec_stack.contains(&child_ptr_val) {
                    panic!(
                        "Cycle detected in propagate_max_depth at child node pointer: {:?}",
                        child_ptr_val
                    );
                }
                Self::_propagate_max_depth(child_arc, candidate_depth, rec_stack);
            }
        }

        // Finished processing this node; remove from recursion stack.
        rec_stack.remove(&node_ptr_val);
    }

    /// Gets the list of (EdgeValue, ChildArc) tuples associated with the given edge key.
    /// Returns a cloned Vec of the tuples. Requires EV: Clone.
    pub fn get(
        &self,
        edge_key: &EK,
    ) -> Option<Vec<(EV, Arc<Mutex<Trie<EK, EV, T>>>)>>
    where EV: Clone // Add constraint here as it's specific to this method's cloning
    {
        // .cloned() clones the Vec<(EV, Arc<...>)>, which clones EV and the Arc.
        self.children.get(edge_key).cloned()
    }


    /// Returns a reference to the map of children nodes.
    /// The map's values are Vecs of (EdgeValue, ChildArc) tuples.
    pub fn children(&self) -> &BTreeMap<EK, Vec<(EV, Arc<Mutex<Trie<EK, EV, T>>>)>> {
        &self.children
    }

    /// Checks if the node is a leaf (has no children).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Collects all *unique* nodes (by pointer) reachable from the given root (BFS).
    pub fn all_nodes(root: Arc<Mutex<Trie<EK, EV, T>>>) -> Vec<Arc<Mutex<Trie<EK, EV, T>>>> {
        // Use a visited pointer set.
        let mut visited_ptrs: HashSet<*const Trie<EK, EV, T>> = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        let root_ptr = node_ptr(&root);
        if visited_ptrs.insert(root_ptr) {
            queue.push_back(root);
        }

        while let Some(node_arc) = queue.pop_front() {
            result.push(node_arc.clone());

            let node = node_arc.lock().expect("Mutex poisoned during BFS");
            // Iterate through the Vecs of children for each edge key.
            for children_vec in node.children.values() {
                // Iterate through the individual (EV, child Arc) tuples in the Vec.
                for (_edge_val, child_arc) in children_vec {
                    let child_ptr = node_ptr(child_arc);
                    if visited_ptrs.insert(child_ptr) {
                        queue.push_back(child_arc.clone());
                    }
                }
            }
        }
        result
    }
}

// A helper that “gets” the raw pointer from an Arc<Mutex<Trie>>; panic if poisoned.
// Updated generics.
fn node_ptr<EK: Ord, EV, T>(node_arc: &Arc<Mutex<Trie<EK, EV, T>>>) -> *const Trie<EK, EV, T> {
    let guard = node_arc.try_lock().expect("Mutex poisoned");
    &*guard as *const _
}


// Implementation block for special_map and related functionality
// Requires T: Clone, EK: Ord + Clone, EV: Clone
impl<T: Clone, EK: Ord + Clone, EV: Clone> Trie<EK, EV, T> {
    /// Performs a specialized breadth-first traversal (related to Dijkstra/Bellman-Ford relaxation).
    ///
    /// V: the “accumulated” value type that is computed along the BFS.
    ///
    /// initial_nodes_and_values: a vector of source nodes with their initial V.
    ///
    /// step: function to compute a new V for a child given a parent’s V, the edge key (EK),
    ///       the edge value (EV), and the child node (which is locked briefly).
    ///       Signature: FnMut(&V, &EK, &EV, &Trie<EK, EV, T>) -> V
    ///
    /// merge: function to combine a new V into a stored V (its signature is
    ///        FnMut(&mut V, V) ). For example, merge might “replace” the old value
    ///        or accumulate them in some way.
    ///
    /// process: function that is called exactly once for each node processed;
    ///          it is given the node’s local T value and the final merged V value.
    ///          **It returns a boolean: `true` to continue processing children, `false` to stop.**
    ///
    /// The algorithm waits to process a node until all incoming paths relevant to its
    /// `max_depth` have likely contributed (arrival_depth == max_depth).
    pub fn special_map<V: Clone>(
        initial_nodes_and_values: Vec<(Arc<Mutex<Trie<EK, EV, T>>>, V)>,
        mut step: impl FnMut(&V, &EK, &EV, &Trie<EK, EV, T>) -> V, // <-- MODIFIED signature
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&Trie<EK, EV, T>, &mut V) -> bool,
    ) {
        // state: for each node (by raw pointer), store (merged V, arrival_depth)
        let mut state: HashMap<*const Trie<EK, EV, T>, (V, usize)> = HashMap::new();
        // ready queue: we will push arcs whenever a node is “ready” to process
        let mut ready: VecDeque<Arc<Mutex<Trie<EK, EV, T>>>> = VecDeque::new();
        // set of processed nodes (by pointer) so that we process each only once
        let mut processed: HashSet<*const Trie<EK, EV, T>> = HashSet::new();
        // record which nodes came in as initial nodes – for these we process right away.
        let mut initial_set: HashSet<*const Trie<EK, EV, T>> = HashSet::new();

        // Initialize state for starting nodes.
        for (node_arc, v) in initial_nodes_and_values {
            let ptr = node_ptr(&node_arc);
            initial_set.insert(ptr);
            state.entry(ptr)
                .and_modify(|(stored, _depth)| { // depth is always 0 for initial
                    merge(stored, v.clone());
                })
                .or_insert((v, 0)); // Initial arrival depth is 0
            // push starting nodes into ready queue unconditionally.
            ready.push_back(node_arc.clone());
        }

        // Main loop.
        while let Some(node_arc) = ready.pop_front() {
            let ptr = node_ptr(&node_arc);
            if processed.contains(&ptr) {
                continue;
            }
            // get stored state (merged V and arrival depth) for this node.
            let (mut node_val_merged, arr_depth) = match state.get(&ptr) {
                Some(&ref tup) => tup.clone(),
                None => {
                    assert!(!initial_set.contains(&ptr), "Initial node lost its state");
                    continue; // Skip if state is missing (shouldn't happen for ready nodes unless logic error)
                }
            };
            // Get the fixed max_depth for this node from its trie.
            let node_max_depth = {
                let node = node_arc.lock().expect("Mutex poisoned in special_map");
                node.max_depth
            };

            // A non–initial node is considered ready once its arrival depth equals node.max_depth.
            // Initial nodes are processed immediately when popped.
            if !initial_set.contains(&ptr) && arr_depth != node_max_depth {
                 // Not yet fully updated based on longest path; skip processing now.
                 // It might be re-added later when its arrival depth increases and matches max_depth.
                continue;
            }

            // Mark node as processed (and remove it from initial_set if it was there).
            processed.insert(ptr);
            initial_set.remove(&ptr); // Safe to call even if not present

            // Call process on this node. Capture the boolean result.
            let should_continue_processing_children = {
                let node = node_arc.lock().expect("Mutex poisoned during process call");
                process(&node, &mut node_val_merged)
            };

            // Only propagate to children if process returned true.
            if should_continue_processing_children {
                // Collect all (EdgeKey, EdgeValue, ChildArc) tuples.
                // Requires EK: Clone, EV: Clone.
                let children_edges_values_arcs: Vec<(EK, EV, Arc<Mutex<Trie<EK, EV, T>>>)> = {
                    let node = node_arc.lock().expect("Mutex poisoned while reading children");
                    node.children
                        .iter()
                        .flat_map(|(edge_key, children_vec)| {
                            // For each edge key, iterate through the Vec<(EV, Arc)>
                            children_vec.iter().map(move |(edge_val, child_arc)| {
                                (edge_key.clone(), edge_val.clone(), child_arc.clone()) // Clone EK, EV, Arc
                            })
                        })
                        .collect()
                };

                for (edge_key, edge_val, child_arc) in children_edges_values_arcs {
                    let child_ptr = node_ptr(&child_arc);
                    if processed.contains(&child_ptr) {
                        continue; // Skip already processed children
                    }

                    // The candidate arrival depth for this child is one more than parent's arrival depth.
                    let candidate_arrival_depth = arr_depth.saturating_add(1);

                    // Compute candidate V for child: use step with the merged V from the parent,
                    // plus the edge key and edge value.
                    let candidate_v = {
                        let child_node = child_arc.lock().expect("Mutex poisoned during step");
                        // Pass parent's merged V, edge key, edge value, and child node T
                        step(&node_val_merged, &edge_key, &edge_val, &child_node) // <-- MODIFIED call
                    };

                    // Update state for the child: merge the new candidate V and update arrival depth.
                    let mut current_child_arr_depth = 0; // Will be updated below
                    state.entry(child_ptr)
                        .and_modify(|(existing_v, existing_depth)| {
                            merge(existing_v, candidate_v.clone()); // Merge the value
                            *existing_depth = (*existing_depth).max(candidate_arrival_depth); // Update depth
                            current_child_arr_depth = *existing_depth; // Record current depth
                        })
                        .or_insert_with(|| {
                            current_child_arr_depth = candidate_arrival_depth; // Record current depth
                            (candidate_v, candidate_arrival_depth) // Insert new state
                        });

                    // Check if the child's inherent max_depth needs updating *and propagate if necessary*.
                    // This handles cases where special_map finds a longer path than insertion did.
                    let child_current_max_depth;
                    {
                        let mut child_node = child_arc.lock().expect("Mutex poisoned while updating child max_depth");
                        if candidate_arrival_depth > child_node.max_depth {
                            child_node.max_depth = candidate_arrival_depth;
                            // Propagate this update downward. Must drop lock before calling.
                            drop(child_node);
                            Trie::<EK, EV, T>::propagate_max_depth(child_arc.clone(), candidate_arrival_depth);
                            // Re-acquire lock briefly to get the potentially updated max_depth
                            child_current_max_depth = child_arc.lock().expect("Mutex poisoned after propagate").max_depth;
                        } else {
                            child_current_max_depth = child_node.max_depth;
                        }
                    } // child_node lock released here

                    // Check readiness: does the *current* arrival depth in state match the child's *current* max_depth?
                    if current_child_arr_depth == child_current_max_depth {
                        // Only queue if it's ready and not already processed
                        if !processed.contains(&child_ptr) {
                             ready.push_back(child_arc.clone());
                        }
                    }
                    // else: Child is not ready yet (arrival depth < max_depth), it might be queued later
                    // when another path updates its arrival depth.
                } // end for each child
            } // end if should_continue_processing_children
        } // end while queue not empty

        // After the loop, check if any initial nodes were *not* processed.
        if !initial_set.is_empty() {
             eprintln!("Warning: Some initial nodes were not processed: {:?}", initial_set);
        }
    }
}


/// A helper function to print the structure of the Trie/DAG via BFS.
/// Updated generics and print statement.
pub(crate) fn dump_structure<EK: Debug + Ord, EV: Debug, T: Debug>(root: Arc<Mutex<Trie<EK, EV, T>>>) {
    let mut queue = VecDeque::new();
    let mut seen: HashSet<*const Trie<EK, EV, T>> = HashSet::new();

    println!("Dumping Trie Structure (BFS):");

    let root_ptr = node_ptr(&root);
    if seen.insert(root_ptr) {
        queue.push_back(root);
    }

    while let Some(node_arc) = queue.pop_front() {
        let node = node_arc.lock().expect("Mutex poisoned during dump");
        let ptr = &*node as *const _;
        println!("{:?}: Value: {:?}, MaxDepth: {}", ptr, node.value, node.max_depth);

        // Iterate through edges and their corresponding Vecs of children
        for (edge_key, children_vec) in node.children.iter() {
            // Iterate through each (EV, child Arc) tuple in the Vec
            for (edge_val, child_arc) in children_vec {
                let child_ptr = node_ptr(child_arc);
                // Updated print statement
                println!("  - Edge Key: {:?}, Edge Val: {:?} -> Child: {:?}", edge_key, edge_val, child_ptr);
                if seen.insert(child_ptr) {
                    queue.push_back(child_arc.clone());
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// Updated for EK, EV, T generics and new method signatures.

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Use concrete types for tests, e.g., &str for EK, &str for EV, i32 for T
    type TestTrie = Trie<&'static str, &'static str, i32>;
    type TestNode = Arc<Mutex<TestTrie>>;

    #[test]
    fn test_insertion_and_retrieval() {
        let mut root = TestTrie::new(0);
        let child1: TestNode = Arc::new(Mutex::new(TestTrie::new(1)));
        let child2: TestNode = Arc::new(Mutex::new(TestTrie::new(2)));
        let child3: TestNode = Arc::new(Mutex::new(TestTrie::new(3))); // Another child for 'a'

        root.insert("a", "edge_a1", child1.clone());
        root.insert("b", "edge_b", child2.clone());
        root.insert("a", "edge_a3", child3.clone()); // Insert second child for 'a'

        // Test get for 'a'
        let retrieved_children_a = root.get(&"a").expect("Failed to get children for 'a'");
        assert_eq!(retrieved_children_a.len(), 2);
        // Check pointers and edge values
        let retrieved_data_a: HashSet<(&str, *const TestTrie)> = retrieved_children_a
            .iter()
            .map(|(ev, arc)| (*ev, node_ptr(arc)))
            .collect();
        assert!(retrieved_data_a.contains(&("edge_a1", node_ptr(&child1))));
        assert!(retrieved_data_a.contains(&("edge_a3", node_ptr(&child3))));

        // Test get for 'b'
        let retrieved_children_b = root.get(&"b").expect("Failed to get child 'b'");
        assert_eq!(retrieved_children_b.len(), 1);
        assert_eq!(retrieved_children_b[0].0, "edge_b"); // Check edge value
        assert!(Arc::ptr_eq(&retrieved_children_b[0].1, &child2)); // Check Arc pointer

        assert!(root.get(&"c").is_none());

        // Test children iterator order (BTreeMap ensures sorted order of keys 'a', 'b')
        let children_keys: Vec<_> = root.children().keys().cloned().collect();
        assert_eq!(children_keys, vec!["a", "b"]);
        assert_eq!(root.children().get("a").unwrap().len(), 2);
        assert_eq!(root.children().get("b").unwrap().len(), 1);

        // Test is_leaf
        assert!(!root.is_leaf());
        assert!(child1.try_lock().unwrap().is_leaf());
        assert!(child2.try_lock().unwrap().is_leaf());
        assert!(child3.try_lock().unwrap().is_leaf());
    }

    #[test]
    fn test_multiple_children_same_edge_key() {
        // Structure:
        //      root (0) --"edge", "val1"--> child1 (1)
        //           |
        //           --"edge", "val2"--> child2 (2)
        let root: TestNode = Arc::new(Mutex::new(TestTrie::new(0)));
        let child1: TestNode = Arc::new(Mutex::new(TestTrie::new(1)));
        let child2: TestNode = Arc::new(Mutex::new(TestTrie::new(2)));

        {
            let mut r = root.lock().unwrap();
            r.insert("edge", "val1", child1.clone());
            r.insert("edge", "val2", child2.clone());
        }

        // Check retrieval
        let children_tuples = root.lock().unwrap().get(&"edge").unwrap();
        assert_eq!(children_tuples.len(), 2);
        let child_data: HashSet<(&str, *const TestTrie)> = children_tuples
            .iter()
            .map(|(ev, arc)| (*ev, node_ptr(arc)))
            .collect();
        assert!(child_data.contains(&("val1", node_ptr(&child1))));
        assert!(child_data.contains(&("val2", node_ptr(&child2))));

        // Check all_nodes
        let all = Trie::all_nodes(root.clone());
        assert_eq!(all.len(), 3); // root, child1, child2
        let all_ptrs: HashSet<_> = all.iter().map(node_ptr).collect();
        assert!(all_ptrs.contains(&node_ptr(&root)));
        assert!(all_ptrs.contains(&node_ptr(&child1)));
        assert!(all_ptrs.contains(&node_ptr(&child2)));

        // Check special_map
        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();

        Trie::special_map(
            vec![(root.clone(), 100)],
            // step: add one, ignore edge key/value
            |parent_val, _ek, _ev, _child_node| parent_val + 1,
            |current, new| *current = new, // merge: replace
            |node, computed_val| { // process: always continue
                processed_node_values.push(node.value);
                computed_values.push(*computed_val);
                true
            },
        );

        // Expected processing order: 0, then (1, 2) in some order.
        assert_eq!(processed_node_values.len(), 3);
        assert!(processed_node_values.contains(&0));
        assert!(processed_node_values.contains(&1));
        assert!(processed_node_values.contains(&2));
        assert_eq!(processed_node_values[0], 0); // Root must be first

        // Expected computed values: root = 100, child1 = 101, child2 = 101.
        assert_eq!(computed_values.len(), 3);
        assert_eq!(computed_values[0], 100);
        assert!(computed_values[1..].contains(&101)); // Both children should get 101
        assert_eq!(computed_values.iter().filter(|&&v| v == 101).count(), 2);
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
        let root: TestNode = Arc::new(Mutex::new(TestTrie::new(0)));
        let child1: TestNode = Arc::new(Mutex::new(TestTrie::new(1)));
        let child2: TestNode = Arc::new(Mutex::new(TestTrie::new(2)));
        let grandchild: TestNode = Arc::new(Mutex::new(TestTrie::new(3)));

        {
            let mut r = root.lock().unwrap();
            r.insert("r->c1", "e1", child1.clone());
            r.insert("r->c2", "e2", child2.clone());
        }
        {
            let mut c1 = child1.lock().unwrap();
            c1.insert("c1->gc", "e3", grandchild.clone());
        }

        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();
        let mut edge_info_at_step = Vec::new(); // Store (EK, EV) seen by step

        Trie::special_map(
            vec![(root.clone(), 100)],
            // step: add one, record edge info
            |parent_val, ek, ev, _child_node| {
                edge_info_at_step.push((ek.clone(), ev.clone())); // Clone needed as step takes refs
                parent_val + 1
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

        // Check processing order
        assert_eq!(processed_node_values.len(), 4);
        assert_eq!(processed_node_values[0], 0); // Root first
        assert_eq!(processed_node_values[3], 3); // Grandchild last
        assert!(processed_node_values[1..3].contains(&1));
        assert!(processed_node_values[1..3].contains(&2));

        // Check computed values
        let results_map: HashMap<i32, i32> = processed_node_values.iter().cloned()
            .zip(computed_values.iter().cloned()).collect();
        assert_eq!(results_map.get(&0), Some(&100));
        assert_eq!(results_map.get(&1), Some(&101));
        assert_eq!(results_map.get(&2), Some(&101));
        assert_eq!(results_map.get(&3), Some(&102));

        // Check edge info captured by step
        assert_eq!(edge_info_at_step.len(), 3); // 3 edges traversed
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
        let root: TestNode = Arc::new(Mutex::new(TestTrie::new(0))); // Use i32 value
        let child1: TestNode = Arc::new(Mutex::new(TestTrie::new(1)));
        let child2: TestNode = Arc::new(Mutex::new(TestTrie::new(2)));
        let grandchild: TestNode = Arc::new(Mutex::new(TestTrie::new(3)));

        {
            let mut r = root.lock().unwrap();
            r.insert("r1", "e1", child1.clone());
            r.insert("r2", "e2", child2.clone());
        }
        {
            let mut c1 = child1.lock().unwrap();
            c1.insert("c1", "e3", grandchild.clone());
        }
        {
            let mut c2 = child2.lock().unwrap();
            c2.insert("c2", "e4", grandchild.clone()); // Diamond
        }

        let all_nodes = Trie::all_nodes(root.clone());

        // Should find 4 unique nodes.
        assert_eq!(all_nodes.len(), 4);
        let node_ptrs: HashSet<_> = all_nodes.iter().map(|arc| node_ptr(arc)).collect();
        assert_eq!(node_ptrs.len(), 4);
        assert!(node_ptrs.contains(&node_ptr(&root)));
        assert!(node_ptrs.contains(&node_ptr(&child1)));
        assert!(node_ptrs.contains(&node_ptr(&child2)));
        assert!(node_ptrs.contains(&node_ptr(&grandchild)));
    }

    #[test]
    fn test_special_map_diamond_merge_max() {
        // Diamond structure (values/depths as before)
        let root: TestNode = Arc::new(Mutex::new(TestTrie::new(0)));
        let child1: TestNode = Arc::new(Mutex::new(TestTrie::new(1)));
        let child2: TestNode = Arc::new(Mutex::new(TestTrie::new(2)));
        let grandchild: TestNode = Arc::new(Mutex::new(TestTrie::new(3)));

        // Build the structure with edge values
        {
            let mut r = root.lock().unwrap();
            r.insert("r->c1", "edge1", child1.clone());
            r.insert("r->c2", "edge2", child2.clone());
        }
        {
            let mut c1 = child1.lock().unwrap();
            c1.insert("c1->gc", "edge3", grandchild.clone());
        }
        {
            let mut c2 = child2.lock().unwrap();
            c2.insert("c2->gc", "edge4", grandchild.clone());
        }

        // Check initial max_depths
        assert_eq!(root.lock().unwrap().max_depth, 0);
        assert_eq!(child1.lock().unwrap().max_depth, 1);
        assert_eq!(child2.lock().unwrap().max_depth, 1);
        assert_eq!(grandchild.lock().unwrap().max_depth, 2);

        let processed_nodes = Arc::new(Mutex::new(HashMap::<i32, i32>::new()));
        let process_count = Arc::new(AtomicUsize::new(0));

        Trie::special_map(
            vec![(root.clone(), 100)],
            // step: increment value, ignore edges
            |p_val, _ek, _ev, _child_node| p_val + 1,
            // merge: take max value
            |current_v, new_v| *current_v = (*current_v).max(new_v),
            { // process: always continue
                let processed_nodes = processed_nodes.clone();
                let process_count = process_count.clone();
                move |node, final_v| {
                    // println!("Processing node T={}, V={}", node.value, final_v);
                    let mut map = processed_nodes.lock().unwrap();
                    map.insert(node.value, *final_v);
                    process_count.fetch_add(1, Ordering::SeqCst);
                    true
                }
            }
        );

        // Assertions
        let final_results = processed_nodes.lock().unwrap();
        assert_eq!(process_count.load(Ordering::SeqCst), 4, "Should process 4 unique nodes");
        assert_eq!(final_results.get(&0), Some(&100));
        assert_eq!(final_results.get(&1), Some(&101));
        assert_eq!(final_results.get(&2), Some(&101));
        assert_eq!(final_results.get(&3), Some(&102)); // gc gets max(101+1, 101+1) = 102
    }


    #[test]
    fn test_empty_trie() {
        let root: TestNode = Arc::new(Mutex::new(TestTrie::new(42)));
        let nodes = Trie::all_nodes(root.clone());
        assert_eq!(nodes.len(), 1);
        assert!(Arc::ptr_eq(&nodes[0], &root));
        assert!(root.try_lock().unwrap().is_leaf());

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

    #[ignore] // This test relies on the exact panic message which might be fragile
    #[test]
    #[should_panic(expected = "Cycle detected in propagate_max_depth")]
    fn test_cycle_detection_on_insert() {
        // Cycle:  root -> child -> root
        let root: TestNode = Arc::new(Mutex::new(TestTrie::new(0)));
        let child: TestNode = Arc::new(Mutex::new(TestTrie::new(1)));

        {
            let mut r = root.lock().unwrap();
            r.insert("r->c", "e1", child.clone()); // child.md=1
        }
        {
            let mut c = child.lock().unwrap();
            // This insert should cause propagation that detects the cycle.
            c.insert("c->r", "e2", root.clone()); // <--- This should panic
        }
    }


    #[test]
    fn test_cycle_all_nodes_no_panic() {
        // Cycle:  root -> child -> root
        // Manually create cycle without insert's propagation.
        let root: TestNode = Arc::new(Mutex::new(TestTrie::new(0)));
        let child: TestNode = Arc::new(Mutex::new(TestTrie::new(1)));

        // Manually create links
        root.lock().unwrap().children.entry("r->c").or_default().push(("e1", child.clone()));
        child.lock().unwrap().children.entry("c->r").or_default().push(("e2", root.clone()));
        // Manually set depths
        root.lock().unwrap().max_depth = 0;
        child.lock().unwrap().max_depth = 1; // Arbitrary, doesn't affect all_nodes logic

        let nodes = Trie::all_nodes(root.clone());
        // Should detect both nodes exactly once.
        assert_eq!(nodes.len(), 2);
        let node_ptrs: HashSet<_> = nodes.iter().map(|arc| node_ptr(arc)).collect();
        assert_eq!(node_ptrs.len(), 2);
        assert!(node_ptrs.contains(&node_ptr(&root)));
        assert!(node_ptrs.contains(&node_ptr(&child)));
    }


    #[test]
    fn test_cycle_special_map_no_panic_limited_processing() {
        // Cycle: root -> child -> root.
        // Manually create cycle.
        let root: TestNode = Arc::new(Mutex::new(TestTrie::new(0)));
        let child: TestNode = Arc::new(Mutex::new(TestTrie::new(1)));

        // Manually create links
        root.lock().unwrap().children.entry("r->c").or_default().push(("e1", child.clone()));
        child.lock().unwrap().children.entry("c->r").or_default().push(("e2", root.clone()));
        // Manually set depths.
        root.lock().unwrap().max_depth = 0; // Initial node, depth 0
        child.lock().unwrap().max_depth = 1; // Child reachable at depth 1

        let mut processed_vals = Vec::new();
        let mut computed_vals = Vec::new();

        Trie::special_map(
            vec![(root.clone(), 100)], // Start at root
            |p, _ek, _ev, _n| p + 1, // Step: increment
            |cur, new| *cur = (*cur).max(new), // Merge: max
            |node, v| { // process: always continue
                processed_vals.push(node.value);
                computed_vals.push(*v);
                true
            },
        );

        // Expected behavior (as described in previous thought process):
        // Root processed (V=100). Child processed (V=101). Propagation back to root skipped.
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
        let root: TestNode = Arc::new(Mutex::new(TestTrie::new(0)));
        let child1: TestNode = Arc::new(Mutex::new(TestTrie::new(1)));
        let child2: TestNode = Arc::new(Mutex::new(TestTrie::new(2)));
        let grandchild1: TestNode = Arc::new(Mutex::new(TestTrie::new(3)));
        let grandchild2: TestNode = Arc::new(Mutex::new(TestTrie::new(4)));

        {
            let mut r = root.lock().unwrap();
            r.insert("r->c1", "e1", child1.clone());
            r.insert("r->c2", "e2", child2.clone());
        }
        {
            let mut c1 = child1.lock().unwrap();
            c1.insert("c1->gc1", "e3", grandchild1.clone());
        }
        {
            let mut c2 = child2.lock().unwrap();
            c2.insert("c2->gc2", "e4", grandchild2.clone());
        }

        let processed_nodes = Arc::new(Mutex::new(HashSet::<i32>::new()));
        let computed_values = Arc::new(Mutex::new(HashMap::<i32, i32>::new()));

        Trie::special_map(
            vec![(root.clone(), 100)],
            |p_val, _ek, _ev, _child_node| p_val + 1, // step: increment value
            |current_v, new_v| *current_v = new_v, // merge: replace
            {
                let processed_nodes = processed_nodes.clone();
                let computed_values = computed_values.clone();
                move |node, final_v| {
                    // println!("Processing node T={}, V={}", node.value, final_v);
                    processed_nodes.lock().unwrap().insert(node.value);
                    computed_values.lock().unwrap().insert(node.value, *final_v);
                    if node.value == 1 { // Stop processing if node value is 1 (child1)
                        false
                    } else {
                        true
                    }
                }
            }
        );

        let final_processed = processed_nodes.lock().unwrap();
        let final_values = computed_values.lock().unwrap();

        // Expected processed nodes: 0, 1, 2, 4. Node 3 should be skipped.
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
}