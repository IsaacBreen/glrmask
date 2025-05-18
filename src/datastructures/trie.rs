use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{self, Debug};
// Import TryLockError explicitly for matching
use std::sync::{Arc, Mutex, TryLockError, MutexGuard};
use std::sync::atomic::{AtomicUsize, Ordering}; // Added for tests
use std::cmp::Reverse;          // min-heap helper
use std::collections::BinaryHeap;


use crate::datastructures::hybrid_bitset::HybridBitset; // Import HybridBitset
use crate::datastructures::ArcPtrWrapper; // Import ArcPtrWrapper
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added
use std::collections::BTreeMap as StdMap; // Added for derive macro pattern


/// Error type indicating that a cycle was detected during an operation
/// that updates graph structure or properties like max_depth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleDetectedError;

impl fmt::Display for CycleDetectedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cycle detected in Trie structure")
    }
}

impl Error for CycleDetectedError {}


/// Represents a node in a Trie–like structure (allowing shared subtrees and DAGs).
/// Multiple children can exist for the same edge key. Each edge instance has a value.
///
/// EK: type of the edge key (must be Ord).
/// EV: type of the edge value.
/// T: type of the value stored within the node.
#[derive(Debug, Clone)]
pub struct Trie<EK: Ord, EV, T> {
    pub value: T,
    /// Stores a map from EdgeKey to (a map from ChildArc (wrapped) to EdgeValue).
    children: BTreeMap<EK, BTreeMap<ArcPtrWrapper<Mutex<Trie<EK, EV, T>>>, EV>>,
    /// The “longest distance” from some source node (as computed during insertion).
    /// This value is set (or updated) when an edge is inserted.
    /// If A -> B, then A.max_depth < B.max_depth.
    pub max_depth: usize,
}

impl<EK, EV, T> JSONConvertible for Trie<EK, EV, T>
where
    EK: Ord + JSONConvertible,
    EV: JSONConvertible,
    T: JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        // WARNING: This is a naive serialization that does NOT handle cycles or shared structure.
        // It will likely lead to infinite recursion for cyclic Tries or excessive data duplication.
        // A proper graph serialization strategy is needed for robust Trie serialization.
        todo!("Trie to_json: Complex graph structure, requires advanced serialization strategy.")
    }

    fn from_json(_node: JSONNode) -> Result<Self, String> {
        // WARNING: Deserializing a Trie with shared structure or cycles from a simple JSON
        // representation is non-trivial and not implemented here.
        todo!("Trie from_json: Complex graph structure, requires advanced deserialization strategy.")
    }
}


// Implementation block for core Trie functionality
// Added Clone bound for EK needed in try_insert_or_merge_edge and others
impl<EK: Ord + Clone, EV, T> Trie<EK, EV, T> {
    /// Creates a new trie node with the given value and no children.
    /// The max_depth is initialized to 0.
    pub fn new(value: T) -> Self {
        Trie {
            value,
            children: BTreeMap::new(),
            max_depth: 0,
        }
    }

    // force_insert remains unchanged
    pub fn force_insert_to_new_node(&mut self, edge_key: EK, edge_value: EV, value: T) -> Arc<Mutex<Trie<EK, EV, T>>> {
        let new_node = Arc::new(Mutex::new(Trie::new(value)));
        let new_node_comparable = ArcPtrWrapper::new(new_node.clone());
        self.children.entry(edge_key).or_default().insert(new_node_comparable, edge_value);
        // Note: force_insert does NOT update max_depth or check for cycles. Use with caution.
        new_node.clone()
    }

    pub fn force_insert_to_node(&mut self, edge_key: EK, edge_value: EV, dst: &Arc<Mutex<Trie<EK, EV, T>>>) {
        let dst_comparable = ArcPtrWrapper::new(dst.clone());
        self.children.entry(edge_key).or_default().insert(dst_comparable, edge_value);
    }

    // already_has_dst remains unchanged
    pub fn already_has_dst(&self, edge_key: EK, dst: &Arc<Mutex<Trie<EK, EV, T>>>) -> bool {
        let lookup_key = ArcPtrWrapper::new(dst.clone()); // Clone Arc for temporary ownership in key
        self.children.get(&edge_key).map_or(false, |dest_map| dest_map.contains_key(&lookup_key))
    }

    // get_edge_value remains unchanged
    pub fn get_edge_value(&self, edge_key: EK, dst: &Arc<Mutex<Trie<EK, EV, T>>>) -> Option<&EV> {
        let lookup_key = ArcPtrWrapper::new(dst.clone());
        self.children.get(&edge_key).and_then(|dest_map| dest_map.get(&lookup_key))
    }

    // get_edge_value_mut remains unchanged
    pub fn get_edge_value_mut(&mut self, edge_key: EK, dst: &Arc<Mutex<Trie<EK, EV, T>>>) -> Option<&mut EV> {
        let lookup_key = ArcPtrWrapper::new(dst.clone());
        self.children.get_mut(&edge_key).and_then(|dest_map| dest_map.get_mut(&lookup_key))
    }

    pub fn try_insert(
        &mut self,
        edge_key: EK,
        edge_value: &mut Option<EV>, // Changed to allow taking the value
        child: Arc<Mutex<Trie<EK, EV, T>>>,
    ) -> Result<(), CycleDetectedError> {
        // ------------------------------------------------------------------
        // 1. Detect whether adding the edge would introduce a cycle.
        //    A cycle exists iff `self` is reachable from `child`.
        // ------------------------------------------------------------------
        let self_ptr = self as *const Trie<EK, EV, T>;
        if Self::detect_cycle(self_ptr, &child) {
            return Err(CycleDetectedError);
        }

        // ------------------------------------------------------------------
        // 2. Update the child's max-depth *before* the edge is inserted.
        //    This lets us rollback cleanly if `propagate_max_depth` fails
        //    (because no structural change has been committed yet).
        // ------------------------------------------------------------------
        let candidate_depth = self.max_depth.saturating_add(1);
        let previous_child_depth; // Store previous depth for potential rollback
        let needs_depth_update;

        // Scope for child lock
        {
            let mut child_guard = child
                .lock()
                .expect("Mutex poisoned while updating child's max_depth");
            previous_child_depth = child_guard.max_depth;
            needs_depth_update = candidate_depth > previous_child_depth;
            if needs_depth_update {
                child_guard.max_depth = candidate_depth;
            }
        } // child_guard lock released here

        // If the child's depth actually changed we must propagate.
        if needs_depth_update {
            // Propagate the update. If it fails (cycle detected during propagation),
            // roll back the change we just made to the child's depth.
            if let Err(e) = Self::propagate_max_depth(child.clone(), candidate_depth) {
                // Roll-back the depth change made above
                let mut child_guard = child
                    .lock()
                    .expect("Mutex poisoned while rolling back max_depth");
                // Only roll back if the depth is still what we set it to.
                // (Another thread might have increased it further, which is fine).
                if child_guard.max_depth == candidate_depth {
                     child_guard.max_depth = previous_child_depth;
                }
                // We should still return the error, as a cycle was detected somewhere.
                return Err(e);
            }
        }

        // ------------------------------------------------------------------
        // 3. All checks have passed – perform the real structural mutation.
        // ------------------------------------------------------------------
        let child_comparable = ArcPtrWrapper::new(child.clone()); // child is an Arc, clone it
        self.children
            .entry(edge_key)
            .or_default()
            .insert(child_comparable, edge_value.take().unwrap()); // Take the value


        Ok(())
    }

    /// Returns `true` if `target_ptr` (pointer to the Trie data) is reachable from `start_arc`.
    /// This function handles the case where `target_ptr` points to a node that is currently locked
    /// by the calling thread (e.g., `self` in `try_insert`).
    fn detect_cycle(
        target_ptr: *const Trie<EK, EV, T>,
        start_arc: &Arc<Mutex<Trie<EK, EV, T>>>,
    ) -> bool {
        // Use Arc::as_ptr to get stable pointers to the Mutex itself for visited tracking.
        let mut visited_arcs: HashSet<*const Mutex<Trie<EK, EV, T>>> = HashSet::new();
        let mut queue: VecDeque<Arc<Mutex<Trie<EK, EV, T>>>> = VecDeque::new();

        let start_arc_ptr = Arc::as_ptr(start_arc);
        if visited_arcs.insert(start_arc_ptr) {
            queue.push_back(start_arc.clone());
        }

        while let Some(node_arc) = queue.pop_front() {
            // Attempt to lock the node to get its data pointer and children.
            let lock_result = node_arc.try_lock();

            match lock_result {
                Ok(node_guard) => {
                    // Successfully locked the node.
                    let current_data_ptr = &*node_guard as *const Trie<EK, EV, T>;

                    // Check if this node's data pointer matches the target pointer.
                    if current_data_ptr == target_ptr {
                        // We reached the target node. Cycle detected.
                        return true;
                    }

                    // Get children while holding the lock.
                    let children_arcs: Vec<Arc<Mutex<Trie<EK, EV, T>>>> = node_guard.children
                        .values() // Iterates over BTreeMap<ArcPtrWrapper<Mutex<...>>, EV>
                        .flat_map(|dest_map| dest_map.keys().map(|wrapper_arc| wrapper_arc.as_arc().clone()))
                        .collect();

                    // Explicitly drop the guard before potentially long operations (queueing).
                    drop(node_guard);

                    // Enqueue unvisited children.
                    for child_arc_val in children_arcs { // Renamed child_arc
                        let child_arc_ptr = Arc::as_ptr(&child_arc_val); // Use child_arc_val
                        if visited_arcs.insert(child_arc_ptr) {
                            queue.push_back(child_arc_val); // Use child_arc_val
                        }
                    }
                }
                Err(TryLockError::WouldBlock) => {
                    // Failed to lock because it's held elsewhere (potentially by the thread calling try_insert).
                    // Assume this means we've reached the target node in the context of try_insert.
                    // If detect_cycle were used elsewhere, this assumption might need revisiting.
                    return true;
                }
                Err(TryLockError::Poisoned(p)) => {
                    // A mutex was poisoned. Propagate the panic.
                    panic!("Mutex poisoned during cycle detection: {:?}", p);
                }
            }
        }

        // BFS completed without finding the target pointer. No cycle detected.
        false
    }


    /// Propagates a max_depth update to all descendant nodes, detecting cycles.
    ///
    /// Returns `Ok(())` if propagation completes successfully.
    /// Returns `Err(CycleDetectedError)`.
    fn propagate_max_depth(node_arc: Arc<Mutex<Trie<EK, EV, T>>>, current_depth: usize) -> Result<(), CycleDetectedError> {
        // rec_stack will contain the set of node pointers from the root of the propagation
        // down to the current recursion level. Use Arc::as_ptr for stable pointers.
        let mut rec_stack: HashSet<*const Mutex<Trie<EK, EV, T>>> = HashSet::new();
        Self::_propagate_max_depth(node_arc, current_depth, &mut rec_stack)
    }

    /// Recursive helper for propagate_max_depth, detecting cycles using Arc pointers.
    /// Returns `Ok(())` or `Err(CycleDetectedError)`.
    fn _propagate_max_depth(
        node_arc: Arc<Mutex<Trie<EK, EV, T>>>,
        current_depth: usize,
        rec_stack: &mut HashSet<*const Mutex<Trie<EK, EV, T>>>,
    ) -> Result<(), CycleDetectedError> {
        let node_arc_ptr = Arc::as_ptr(&node_arc);

        // If this node (identified by its Arc pointer) is already in the current recursion chain, we have a cycle.
        if rec_stack.contains(&node_arc_ptr) {
            return Err(CycleDetectedError);
        }

        // Add the current node to the recursion stack.
        rec_stack.insert(node_arc_ptr);

        // Collect *all* child Arcs outside of the lock to avoid holding lock during recursion.
        let children_arcs: Vec<Arc<Mutex<Trie<EK, EV, T>>>> = {
            let node = node_arc
                .lock()
                .expect("Mutex poisoned in _propagate_max_depth (getting children)");
            node.children
                .values() // Iterates over BTreeMap<ArcPtrWrapper<Mutex<...>>, EV>
                .flat_map(|dest_map| dest_map.keys().map(|wrapper_arc| wrapper_arc.as_arc().clone()))
                .collect()
        }; // child_guard lock released here

        // For each child, compute the candidate depth.
        let candidate_depth_val = current_depth.saturating_add(1); // Renamed candidate_depth
        for child_arc in children_arcs {
            // Check if the child needs updating *before* recursing.
            let should_propagate;
            { // Scope for child lock
                let mut child_guard = child_arc
                    .lock()
                    .expect("Mutex poisoned in _propagate_max_depth (checking child depth)");
                if candidate_depth_val > child_guard.max_depth { // Use candidate_depth_val
                    child_guard.max_depth = candidate_depth_val; // Use candidate_depth_val
                    should_propagate = true;
                } else {
                    should_propagate = false;
                }
            } // child_guard lock released here

            if should_propagate {
                // Recurse. Propagate the error up if recursion detects a cycle.
                Self::_propagate_max_depth(child_arc, candidate_depth_val, rec_stack)?; // Use candidate_depth_val
            }
        }

        // Finished processing this node; remove from recursion stack.
        rec_stack.remove(&node_arc_ptr);
        Ok(()) // Success for this branch
    }

    // get remains unchanged
    pub fn get(
        &self,
        edge_key: &EK,
    ) -> Option<&BTreeMap<ArcPtrWrapper<Mutex<Trie<EK, EV, T>>>, EV>>
    {
        self.children.get(edge_key)
    }

    // get_mut remains unchanged
    pub fn get_mut(
        &mut self,
        edge_key: &EK,
    ) -> Option<&mut BTreeMap<ArcPtrWrapper<Mutex<Trie<EK, EV, T>>>, EV>>
    {
        self.children.get_mut(edge_key)
    }

    // children remains unchanged
    pub fn children(&self) -> &BTreeMap<EK, BTreeMap<ArcPtrWrapper<Mutex<Trie<EK, EV, T>>>, EV>> {
        &self.children
    }

    // is_leaf remains unchanged
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Collects all *unique* nodes (by pointer) reachable from the given root (BFS).
    /// This method does not panic on cycles, it simply avoids revisiting nodes.
    pub fn all_nodes(root: Arc<Mutex<Trie<EK, EV, T>>>) -> Vec<Arc<Mutex<Trie<EK, EV, T>>>> {
        // Use Arc::as_ptr for visited tracking
        let mut visited_arcs: HashSet<*const Mutex<Trie<EK, EV, T>>> = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        let root_arc_ptr = Arc::as_ptr(&root);
        if visited_arcs.insert(root_arc_ptr) {
            queue.push_back(root);
        }

        while let Some(node_arc) = queue.pop_front() {
            result.push(node_arc.clone()); // Add the node itself to the result

            // Lock the node to get its children
            let node_guard = node_arc.lock().expect("Mutex poisoned during BFS"); // Renamed node to node_guard
            for children_map in node_guard.children.values() { // Use node_guard
                for child_wrapper_arc in children_map.keys() { // Iterate over ArcPtrWrapper keys
                    let child_arc = child_wrapper_arc.as_arc();
                    let child_arc_ptr = Arc::as_ptr(child_arc);
                    if visited_arcs.insert(child_arc_ptr) {
                        queue.push_back(child_arc.clone());
                    }
                }
            }
            // node_guard lock is released here
        }
        result
    }

    /// Checks if there are any cycles reachable from the given `root_arc`.
    /// Returns `true` if a cycle is detected, `false` otherwise.
    /// This method is useful for verifying graph integrity after complex build processes.
    pub fn has_any_cycle(root_arc: Arc<Mutex<Trie<EK, EV, T>>>) -> bool {
        let mut global_visited_arcs: HashSet<*const Mutex<Trie<EK, EV, T>>> = HashSet::new();
        let mut recursion_stack_arcs: HashSet<*const Mutex<Trie<EK, EV, T>>> = HashSet::new();
        // Call the recursive helper starting with the root node.
        Self::_has_any_cycle_recursive(root_arc, &mut global_visited_arcs, &mut recursion_stack_arcs)
    }

    /// Recursive helper function for `has_any_cycle`.
    ///
    /// `global_visited_arcs`: Tracks all nodes that have been visited and processed across
    ///                        all recursion branches. This prevents re-processing subgraphs
    ///                        that are already known to be cycle-free (or whose cycles
    ///                        would have been detected via another path).
    /// `recursion_stack_arcs`: Tracks nodes currently in the recursion stack for the *current*
    ///                         DFS path. A cycle is detected if we try to visit a node
    ///                         that is already in this set.
    fn _has_any_cycle_recursive(
        node_arc: Arc<Mutex<Trie<EK, EV, T>>>,
        global_visited_arcs: &mut HashSet<*const Mutex<Trie<EK, EV, T>>>,
        recursion_stack_arcs: &mut HashSet<*const Mutex<Trie<EK, EV, T>>>,
    ) -> bool {
        let node_arc_ptr = Arc::as_ptr(&node_arc);

        // If the node is already in the current recursion stack, we've found a back-edge (a cycle).
        if recursion_stack_arcs.contains(&node_arc_ptr) {
            return true; // Cycle detected
        }

        // If the node has been globally visited AND is not in the current recursion stack,
        // it means this node was fully processed via another path and no cycles were found
        // originating from it *then*. We can safely return false to avoid re-exploring.
        if global_visited_arcs.contains(&node_arc_ptr) {
            return false; // Already processed and known to be part of a cycle-free subgraph (from its perspective)
        }

        // Add the current node to both sets:
        // - To recursion_stack_arcs to mark it as part of the current DFS path.
        // - To global_visited_arcs to mark it as processed, so we don't re-explore it unnecessarily
        //   if reached from another path later.
        recursion_stack_arcs.insert(node_arc_ptr);
        global_visited_arcs.insert(node_arc_ptr);

        // Lock the node to get its children.
        // It's important to collect children Arcs first and then release the lock before recursing.
        let children_arcs: Vec<Arc<Mutex<Trie<EK, EV, T>>>> = {
            let node_guard_val = node_arc.lock().expect("Mutex poisoned during has_any_cycle traversal"); // Renamed node_guard
            node_guard_val.children // Use node_guard_val
                .values() // Iterate over BTreeMap<ArcPtrWrapper<Mutex<...>>, EV>
                .flat_map(|dest_map| dest_map.keys().map(|wrapper_arc| wrapper_arc.as_arc().clone()))
                .collect()
        }; // node_guard_val lock is released here.

        // Recursively check each child.
        for child_arc in children_arcs {
            if Self::_has_any_cycle_recursive(child_arc, global_visited_arcs, recursion_stack_arcs) {
                return true; // Cycle detected in a descendant path
            }
        }

        // If we've processed all children of this node and found no cycles,
        // remove it from the recursion stack (as we are "returning" up the DFS path).
        // It remains in global_visited_arcs.
        recursion_stack_arcs.remove(&node_arc_ptr);

        false // No cycle found originating from this node or its descendants along this path
    }
}

/// Helper to get the raw pointer to the Trie data from an Arc<Mutex<Trie>>.
/// Panics if the mutex is poisoned. Returns None if lock fails (WouldBlock).
/// **Use with caution:** Only use when you know a failed lock means the current thread holds it.
/// Consider using `Arc::as_ptr` for identity checks instead if possible.
#[allow(dead_code)] // Keep available, but node_ptr is preferred generally
pub(crate) fn try_get_node_data_ptr<EK: Ord, EV, T>(node_arc: &Arc<Mutex<Trie<EK, EV, T>>>) -> Option<*const Trie<EK, EV, T>> {
    match node_arc.try_lock() {
        Ok(guard) => {
            let ptr = &*guard as *const Trie<EK, EV, T>;
            Some(ptr)
            // Guard is dropped here, lock released
        }
        Err(TryLockError::Poisoned(p)) => {
            panic!("Mutex poisoned when trying to get node data pointer: {:?}", p);
        }
        Err(TryLockError::WouldBlock) => {
            // Lock is held, likely by the current thread in specific scenarios (like cycle check).
            None
        }
    }
}

/// Helper to get the raw pointer to the Trie data from an Arc<Mutex<Trie>>.
/// Panics if the mutex is poisoned or if locking fails (blocking lock).
/// **Use when you need the pointer and expect the lock to succeed.**
#[allow(dead_code)] // Keep available, but Arc::as_ptr is often better for identity
pub(crate) fn node_ptr<EK: Ord, EV, T>(node_arc: &Arc<Mutex<Trie<EK, EV, T>>>) -> *const Trie<EK, EV, T> {
    let guard = node_arc.lock().expect("Mutex poisoned or lock failed when getting node pointer");
    &*guard as *const _
}


// Implementation block for special_map and related functionality
// Requires T: Clone, EK: Ord + Clone, EV: Clone
impl<T: Clone, EK: Ord + Clone, EV: Clone> Trie<EK, EV, T> {
    /// Performs a specialized breadth-first traversal (related to Dijkstra/Bellman-Ford relaxation).
    /// (special_map implementation remains unchanged)
    pub fn special_map<V: Clone>(
        initial_nodes_and_values: Vec<(Arc<Mutex<Trie<EK, EV, T>>>, V)>,
        mut step: impl FnMut(&V, &EK, &EV, &Trie<EK, EV, T>) -> Option<V>, // Changed Trie<...> to &Trie<...>
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&Trie<EK, EV, T>, &mut V) -> bool, // Changed Trie<...> to &Trie<...>
    ) {
        // ------------------------------------------------------------------
        //  Simple depth-driven scheduler.
        //
        //  The key observation is:
        //      parent.max_depth  <  child.max_depth
        //  for every edge (parent → child).
        //  Therefore processing nodes strictly in ascending `max_depth`
        //  guarantees every parent is handled before each of its children.
        //
        //  • `values`  – accumulated V for every discovered node
        //  • `done`    – nodes that have already been processed
        //  • `todo`    – min-heap keyed by max_depth
        // ------------------------------------------------------------------
        let mut values   : HashMap<*const Mutex<Self>, V> = HashMap::new();
        let mut done     : HashSet <*const Mutex<Self>>   = HashSet ::new();
        let mut todo     : BTreeMap<usize, HashSet<ArcPtrWrapper<Mutex<Self>>>> = BTreeMap::new();

        // Seed with the user-supplied starting set
        for (node_arc, v0) in initial_nodes_and_values {
            let ptr = Arc::as_ptr(&node_arc);
            values
                .entry(ptr)
                .and_modify(|old| merge(old, v0.clone()))
                .or_insert(v0);
            let depth = node_arc.lock().expect("poison").max_depth;
            todo.entry(depth).or_default().insert(ArcPtrWrapper::new(node_arc.clone()));
        }

        // Main loop ---------------------------------------------------------
        while let Some((_depth, node_arc_ptr_wrappers)) = todo.pop_first() {
            for node_arc_ptr_wrapper in &node_arc_ptr_wrappers {
                let ptr = Arc::as_ptr(node_arc_ptr_wrapper.as_arc());
                if done.contains(&ptr) { continue; }               // already processed

                // Pull the merged value that all parents contributed
                let mut agg_v = match values.remove(&ptr) {
                    Some(v) => v,
                    None => continue,                            // can happen if every parent’s `step` returned None
                };

                // ---------- user ‘process’ callback ----------
                let proceed = {
                    let guard = node_arc_ptr_wrapper.as_arc().lock().expect("poison");
                    process(&guard, &mut agg_v)
                };
                done.insert(ptr);

                if !proceed { continue; }                           // user stopped traversal at this node

                // ---------- propagate to children -------------
                // We read children once, outside any long-lived locks
                let edges: Vec<(EK, EV, Arc<Mutex<Self>>)> = {
                    let guard = node_arc_ptr_wrapper.as_arc().lock().expect("poison");
                    guard.children
                        .iter()
                        .flat_map(|(ek, dst_map)| {
                            dst_map.iter().map(move |(wrap, ev)| (ek.clone(), ev.clone(), wrap.as_arc().clone()))
                        })
                        .collect()
                };

                for (ek, ev, child_arc) in edges {
                    let child_ptr = Arc::as_ptr(&child_arc);

                    // user ‘step’ callback
                    let maybe_v = {
                        let child_guard = child_arc.lock().expect("poison");
                        step(&agg_v, &ek, &ev, &child_guard) // Pass &child_guard
                    };
                    if let Some(new_v) = maybe_v {
                        values
                            .entry(child_ptr)
                            .and_modify(|old| merge(old, new_v.clone()))
                            .or_insert(new_v);

                        // Queue child by its declared depth
                        let child_depth = child_arc.lock().expect("poison").max_depth;
                        todo.entry(child_depth).or_default().insert(ArcPtrWrapper::new(child_arc));
                    }
                }
            }
        }
    }
}


/// A helper struct to facilitate inserting an edge into a Trie,
/// trying multiple potential destinations and optionally creating a new node.
/// Provides a chainable interface.
pub struct EdgeInserter<EK, EV, T, FMergeEV>
where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone, // T needs to be Clone for else_create_destination_with_value -> Trie::new(value)
    FMergeEV: FnMut(&mut EV, EV), // Closure to merge edge values if edge exists - Changed signature
{
    source_arc: Arc<Mutex<Trie<EK, EV, T>>>, // The source node for the edge
    edge_key: EK,                            // The key for the edge to be inserted
    edge_value: Option<EV>,                          // The value for the edge to be inserted
    merge_edge_value: FMergeEV,              // The function to merge edge values
    result: Option<Arc<Mutex<Trie<EK, EV, T>>>>, // Stores the successful destination node
}

impl<EK, EV, T, FMergeEV> EdgeInserter<EK, EV, T, FMergeEV>
where
    EK: Ord + Clone + Debug,
    EV: Clone,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV), // Changed signature
{
    /// Creates a new `EdgeInserter`.
    ///
    /// # Arguments
    ///
    /// * `source_arc`: The source node where the edge originates.
    /// * `edge_key`: The key for the new edge.
    /// * `edge_value`: The value for the new edge.
    /// * `merge_edge_value`: A closure that takes the existing edge value and the new edge value,
    ///   both by value, returning a merged value. This is only called if an edge with the same `edge_key` already
    ///   points to the `destination` being tried.
    pub fn new(
        source_arc: Arc<Mutex<Trie<EK, EV, T>>>,
        edge_key: EK,
        edge_value: EV,
        merge_edge_value: FMergeEV,
    ) -> Self {
        EdgeInserter {
            source_arc,
            edge_key,
            edge_value: Some(edge_value),
            merge_edge_value,
            result: None,
        }
    }

    /// Tries to establish an edge to the given `destination`.
    ///
    /// If an edge with the same `edge_key` already exists pointing to `destination`,
    /// it attempts to merge the `edge_value` using the `merge_edge_value` closure.
    /// If no such edge exists, it attempts to insert a new edge using `try_insert`.
    ///
    /// This operation only proceeds if a successful destination hasn't already been found.
    /// Returns `self` to allow chaining.
    pub fn try_destination(mut self, destination: Arc<Mutex<Trie<EK, EV, T>>>) -> Self {
        if self.result.is_some() {
            return self; // Already found a destination
        }

        let mut source_guard = self.source_arc.lock().expect("Mutex poisoned while locking source in try_destination"); // Renamed source to source_guard
        let destination_wrapper = ArcPtrWrapper::new(destination.clone()); // Use ArcPtrWrapper

        // Check if edge already exists and try merging EV
        if let Some(existing_ev_mut) = source_guard.children.get_mut(&self.edge_key).and_then(|dest_map| dest_map.get_mut(&destination_wrapper)) {
            (self.merge_edge_value)(existing_ev_mut, self.edge_value.take().unwrap());
            self.result = Some(destination);
        } else {
            // Edge doesn't exist, try inserting. try_insert expects the value by move.
            match source_guard.try_insert(self.edge_key.clone(), &mut self.edge_value, destination.clone()) { // Clone for insert
                Ok(()) => {
                    self.result = Some(destination); // Insert successful, destination found
                }
                Err(CycleDetectedError) => {
                    // Cycle detected, insert failed, result remains None
                    crate::debug!(4, "Cycle detected trying to insert edge {:?} to node {:p}", self.edge_key, Arc::as_ptr(&destination));
                }
            }
        }
        drop(source_guard); // Use source_guard
        self
    }

    /// Tries to establish an edge to any destination in the provided slice.
    ///
    /// Iterates through `destinations` and calls `try_destination` for each until one succeeds.
    /// Returns `self` to allow chaining.
    pub fn try_destinations(mut self, destinations: &[Arc<Mutex<Trie<EK, EV, T>>>]) -> Self {
        for destination in destinations {
            if self.result.is_some() {
                break; // Stop trying once a destination is found
            }
            // Need to consume and reassign self because try_destination takes self
            self = self.try_destination(destination.clone());
        }
        self
    }

    pub fn try_destinations_iter(mut self, destinations: impl Iterator<Item = Arc<Mutex<Trie<EK, EV, T>>>>) -> Self {
        for destination in destinations {
            if self.result.is_some() {
                break; // Stop trying once a destination is found
            }
            // Need to consume and reassign self because try_destination takes self
            self = self.try_destination(destination.clone()); // destination is already Arc, clone it
        }
        self
    }

    pub fn try_destinations_iter_with<F, R>(mut self, destinations: F) -> Self
    where
        F: Fn() -> R,
        R: Iterator<Item = Arc<Mutex<Trie<EK, EV, T>>>>,
    {
        for destination in destinations() {
            if self.result.is_some() { // Check before calling try_destination
                break;
            }
            self = self.try_destination(destination.clone()); // destination is already Arc, clone it
        }
        self
    }


    /// Tries to merge the edge with existing children under `self.edge_key`.
    ///
    /// This method identifies all children of the source node that are already
    /// destinations for an edge with `self.edge_key`. For each such child, it
    /// attempts to merge `self.edge_value` into the existing edge's value using
    /// the `merge_edge_value` closure provided when the `EdgeInserter` was created.
    /// This is done by calling `try_destination` for each identified child.
    ///
    /// The first child for which `try_destination` is successful (typically by merging
    /// the edge value, as the edge `(self.edge_key, child)` already exists)
    /// will be set as `self.result`, and no further children under this key will be tried.
    ///
    /// If `merge_edge_value` returns `None` for a child (causing `try_destination` to not
    /// set a result for that child), or if no children exist under `self.edge_key`,
    /// `self.result` remains unchanged by this method with respect to those children.
    /// This method focuses on updating existing edges associated with `self.edge_key`.
    ///
    /// Returns `self` to allow chaining.
    pub fn try_children(mut self) -> Self {
        if self.result.is_some() {
            return self;
        }

        // Collect children arcs that are specifically under self.edge_key.
        let children_for_this_key: Vec<Arc<Mutex<Trie<EK, EV, T>>>> = {
            let source_guard = self.source_arc.lock().expect("Mutex poisoned while locking source in try_children");
            if let Some(dest_map) = source_guard.children.get(&self.edge_key) {
                dest_map.keys().map(|wrapper_arc| wrapper_arc.as_arc().clone()).collect()
            } else {
                Vec::new() // No children under this specific edge key
            }
        }; // Lock is dropped here

        // If there are children under this key, try them.
        // self.try_destinations will attempt to merge the edge value.
        if !children_for_this_key.is_empty() {
            self = self.try_destinations(&children_for_this_key);
        }
        // Return self, which may or may not have found a result.
        self
    }


    /// If no destination has been found yet, creates a new node with the given `value`,
    /// inserts an edge to it from the source, and sets it as the result.
    ///
    /// Returns `self` to allow chaining.
    pub fn else_create_destination_with_value(mut self, value: T) -> Self {
        if self.result.is_some() {
            return self;
        }

        let new_node_arc = Arc::new(Mutex::new(Trie::new(value)));
        let mut source_guard = self.source_arc.lock().expect("Mutex poisoned while locking source in else_create_with_value"); // Renamed source

        // try_insert expects the value by move, so clone here
        match source_guard.try_insert(self.edge_key.clone(), &mut self.edge_value, new_node_arc.clone()) { // Clone for try_insert
            Ok(()) => {
                self.result = Some(new_node_arc);
            }
            Err(CycleDetectedError) => {
                // Insert failed (e.g., cycle detected even with new node - unusual)
                crate::debug!(1, "Cycle detected trying to insert edge {:?} to NEW node {:p}. Creation failed.", self.edge_key, Arc::as_ptr(&new_node_arc));
                // result remains None
            }
        }
        drop(source_guard); // Use source_guard
        self
    }

    /// If no destination has been found yet, creates a new node by calling `value_fn`,
    /// inserts an edge to it from the source, and sets it as the result.
    ///
    /// Returns `self` to allow chaining.
    pub fn else_create_destination_with(self, value_fn: impl FnOnce() -> T) -> Self {
        if self.result.is_some() {
            return self;
        }
        self.else_create_destination_with_value(value_fn())
    }

    /// If no destination has been found yet, creates a new node with the default value (`T::default()`),
    /// inserts an edge to it from the source, and sets it as the result.
    ///
    /// Requires `T: Default`.
    /// Returns `self` to allow chaining.
    pub fn else_create_destination(self) -> Self
    where
        T: Default,
    {
        if self.result.is_some() {
            return self;
        }
        self.else_create_destination_with_value(T::default())
    }


    /// Returns the resulting destination node, if one was found or created.
    pub fn into_option(self) -> Option<Arc<Mutex<Trie<EK, EV, T>>>> {
        self.result
    }

    pub fn clone_into_option(&self) -> Option<Arc<Mutex<Trie<EK, EV, T>>>> {
        self.result.clone()
    }

    /// Returns the resulting destination node, panicking if none was found or created.
    pub fn unwrap(self) -> Arc<Mutex<Trie<EK, EV, T>>> {
        self.result.expect("EdgeInserter::unwrap() called but no destination was found or created")
    }

    /// Returns the resulting destination node, panicking with the given message if none was found or created.
    pub fn expect(self, msg: &str) -> Arc<Mutex<Trie<EK, EV, T>>> {
        self.result.expect(msg)
    }
}


// Optional: Add a convenience method to Trie to create an EdgeInserter easily.
impl<EK: Ord + Clone + Debug, EV: Clone, T: Clone> Trie<EK, EV, T> {
    /// Creates an `EdgeInserter` to help add an edge starting from this node.
    ///
    /// This provides a convenient entry point for the chainable insertion pattern.
    ///
    /// # Arguments
    /// * `edge_key`: The key for the new edge.
    /// * `edge_value`: The value for the new edge.
    /// * `merge_edge_value`: A closure that takes the existing edge value and the new edge value,
    ///   both by reference, returning `Some(merged_value)` if merging is possible/desired,
    ///   or `None` otherwise. This is only called by `EdgeInserter::try_destination` if an edge
    ///   with the same `edge_key` already points to the destination being tried.
    ///
    /// # Example
    ///
    /// 
