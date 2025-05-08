use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{self, Debug};
// Import TryLockError explicitly for matching
use std::sync::{Arc, Mutex, TryLockError, MutexGuard};
use std::sync::atomic::{AtomicUsize, Ordering}; // Added for tests

use crate::datastructures::hybrid_bitset::HybridBitset; // Import HybridBitset

// Wrapper for Arc<Mutex<N>> to make it Ord and Hashable based on its pointer.
#[derive(Debug)]
pub struct ComparableArc<N>(Arc<Mutex<N>>);

impl<N> ComparableArc<N> {
    fn new(arc: Arc<Mutex<N>>) -> Self {
        ComparableArc(arc)
    }

    pub fn as_arc(&self) -> &Arc<Mutex<N>> {
        &self.0
    }

    #[allow(dead_code)] // Potentially useful later
    fn into_arc(self) -> Arc<Mutex<N>> {
        self.0
    }
}

impl<N> Clone for ComparableArc<N> {
    fn clone(&self) -> Self {
        ComparableArc(self.0.clone())
    }
}

impl<N> PartialEq for ComparableArc<N> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<N> Eq for ComparableArc<N> {}

impl<N> PartialOrd for ComparableArc<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<N> Ord for ComparableArc<N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (Arc::as_ptr(&self.0) as usize).cmp(&(Arc::as_ptr(&other.0) as usize))
    }
}

impl<N> std::hash::Hash for ComparableArc<N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.0) as usize).hash(state);
    }
}


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
    children: BTreeMap<EK, BTreeMap<ComparableArc<Trie<EK, EV, T>>, EV>>,
    /// The “longest distance” from some source node (as computed during insertion).
    /// This value is set (or updated) when an edge is inserted.
    pub max_depth: usize,
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
        let new_node_comparable = ComparableArc::new(new_node.clone());
        self.children.entry(edge_key).or_default().insert(new_node_comparable, edge_value);
        // Note: force_insert does NOT update max_depth or check for cycles. Use with caution.
        new_node.clone()
    }

    pub fn force_insert_to_node(&mut self, edge_key: EK, edge_value: EV, dst: &Arc<Mutex<Trie<EK, EV, T>>>) {
        let dst_comparable = ComparableArc::new(dst.clone());
        self.children.entry(edge_key).or_default().insert(dst_comparable, edge_value);
    }

    // already_has_dst remains unchanged
    pub fn already_has_dst(&self, edge_key: EK, dst: &Arc<Mutex<Trie<EK, EV, T>>>) -> bool {
        let lookup_key = ComparableArc::new(dst.clone()); // Clone Arc for temporary ownership in key
        self.children.get(&edge_key).map_or(false, |dest_map| dest_map.contains_key(&lookup_key))
    }

    // get_edge_value remains unchanged
    pub fn get_edge_value(&self, edge_key: EK, dst: &Arc<Mutex<Trie<EK, EV, T>>>) -> Option<&EV> {
        let lookup_key = ComparableArc::new(dst.clone());
        self.children.get(&edge_key).and_then(|dest_map| dest_map.get(&lookup_key))
    }

    // get_edge_value_mut remains unchanged
    pub fn get_edge_value_mut(&mut self, edge_key: EK, dst: &Arc<Mutex<Trie<EK, EV, T>>>) -> Option<&mut EV> {
        let lookup_key = ComparableArc::new(dst.clone());
        self.children.get_mut(&edge_key).and_then(|dest_map| dest_map.get_mut(&lookup_key))
    }

    pub fn try_insert(
        &mut self,
        edge_key: EK,
        edge_value: EV,
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
        let child_comparable = ComparableArc::new(child.clone()); // child is an Arc, clone it
        self.children
            .entry(edge_key)
            .or_default()
            .insert(child_comparable, edge_value);


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
                        .values() // Iterates over BTreeMap<ComparableArc, EV>
                        .flat_map(|dest_map| dest_map.keys().map(|comparable_arc| comparable_arc.as_arc().clone()))
                        .collect();

                    // Explicitly drop the guard before potentially long operations (queueing).
                    drop(node_guard);

                    // Enqueue unvisited children.
                    for child_arc in children_arcs {
                        let child_arc_ptr = Arc::as_ptr(&child_arc);
                        if visited_arcs.insert(child_arc_ptr) {
                            queue.push_back(child_arc);
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
                .values() // Iterates over BTreeMap<ComparableArc, EV>
                .flat_map(|dest_map| dest_map.keys().map(|ca| ca.as_arc().clone()))
                .collect()
        };

        // For each child, compute the candidate depth.
        let candidate_depth = current_depth.saturating_add(1);
        for child_arc in children_arcs {
            // Check if the child needs updating *before* recursing.
            let should_propagate;
            { // Scope for child lock
                let mut child_guard = child_arc
                    .lock()
                    .expect("Mutex poisoned in _propagate_max_depth (checking child depth)");
                if candidate_depth > child_guard.max_depth {
                    child_guard.max_depth = candidate_depth;
                    should_propagate = true;
                } else {
                    should_propagate = false;
                }
            } // child_guard lock released here

            if should_propagate {
                // Recurse. Propagate the error up if recursion detects a cycle.
                Self::_propagate_max_depth(child_arc, candidate_depth, rec_stack)?;
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
    ) -> Option<&BTreeMap<ComparableArc<Trie<EK, EV, T>>, EV>>
    {
        self.children.get(edge_key)
    }

    // get_mut remains unchanged
    pub fn get_mut(
        &mut self,
        edge_key: &EK,
    ) -> Option<&mut BTreeMap<ComparableArc<Trie<EK, EV, T>>, EV>>
    {
        self.children.get_mut(edge_key)
    }

    // children remains unchanged
    pub fn children(&self) -> &BTreeMap<EK, BTreeMap<ComparableArc<Trie<EK, EV, T>>, EV>> {
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
            let node = node_arc.lock().expect("Mutex poisoned during BFS");
            for children_map in node.children.values() { // children_map is BTreeMap<ComparableArc, EV>
                for child_comparable_arc in children_map.keys() { // Iterate over ComparableArc keys
                    let child_arc = child_comparable_arc.as_arc();
                    let child_arc_ptr = Arc::as_ptr(child_arc);
                    if visited_arcs.insert(child_arc_ptr) {
                        queue.push_back(child_arc.clone());
                    }
                }
            }
            // node lock is released here
        }
        result
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
        mut step: impl FnMut(&V, &EK, &EV, &Trie<EK, EV, T>) -> Option<V>,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&Trie<EK, EV, T>, &mut V) -> bool,
    ) {
        // state: for each node (by Arc pointer), store (merged V, arrival_depth)
        // Using Arc::as_ptr for HashMap key for stable pointers
        let mut state: HashMap<*const Mutex<Trie<EK, EV, T>>, (V, usize)> = HashMap::new();
        let mut ready: VecDeque<Arc<Mutex<Trie<EK, EV, T>>>> = VecDeque::new();
        // set of processed nodes (by Arc pointer)
        let mut processed: HashSet<*const Mutex<Trie<EK, EV, T>>> = HashSet::new();
        // record which nodes came in as initial nodes
        let mut initial_set: HashSet<*const Mutex<Trie<EK, EV, T>>> = HashSet::new();

        // Initialize state for starting nodes.
        for (node_arc, v) in initial_nodes_and_values {
            let arc_ptr = Arc::as_ptr(&node_arc);
            initial_set.insert(arc_ptr);
            state.entry(arc_ptr)
                .and_modify(|(stored, _depth)| { // depth is always 0 for initial
                    merge(stored, v.clone());
                })
                .or_insert((v, 0)); // Initial arrival depth is 0
            ready.push_back(node_arc.clone());
        }

        // Main loop.
        while let Some(node_arc) = ready.pop_front() {
            let arc_ptr = Arc::as_ptr(&node_arc);
            if processed.contains(&arc_ptr) {
                continue;
            }
            // get stored state (merged V and arrival depth) for this node.
            let (mut node_val_merged, arr_depth) = match state.get(&arc_ptr) {
                Some(tup) => tup.clone(), // Clone the tuple (V, usize)
                None => {
                    // Node might not be in state if path was pruned by step returning None or process returning false
                    continue;
                }
            };
            // Get the fixed max_depth for this node from its trie.
            let node_max_depth = {
                let node = node_arc.lock().expect("Mutex poisoned in special_map getting max_depth");
                node.max_depth
            };

            // A non–initial node is considered ready once its arrival depth equals node.max_depth.
            // Initial nodes are processed immediately when popped.
            if !initial_set.contains(&arc_ptr) && arr_depth < node_max_depth {
                 // Not yet fully updated based on longest path; skip processing now.
                 // It might be re-added later when its arrival depth increases and matches max_depth.
                continue;
            }
            // If arr_depth > node_max_depth, something is inconsistent. Warn?
            if arr_depth > node_max_depth {
                 // This can happen if max_depth was updated concurrently or if graph has cycles not caught by insertion
                 crate::debug!(3, "Warning: Node {:p} has arrival depth {} > max_depth {}. Processing anyway.", arc_ptr, arr_depth, node_max_depth);
            }


            // Mark node as processed (and remove it from initial_set if it was there).
            processed.insert(arc_ptr);
            initial_set.remove(&arc_ptr); // Safe to call even if not present

            // Call process on this node. Capture the boolean result.
            let should_continue_processing_children = {
                let node = node_arc.lock().expect("Mutex poisoned during process call");
                process(&node, &mut node_val_merged)
            };

            // Only propagate to children if process returned true.
            if should_continue_processing_children {
                // Collect all (EdgeKey, EdgeValue, ChildArc) tuples. Lock briefly.
                let children_edges_values_arcs: Vec<(EK, EV, Arc<Mutex<Trie<EK, EV, T>>>)> = {
                    let node = node_arc.lock().expect("Mutex poisoned while reading children");
                    node.children
                        .iter()
                        .flat_map(|(edge_key, dest_map)| { // dest_map is &BTreeMap<ComparableArc, EV>
                            dest_map.iter().map(move |(child_comparable_arc, edge_val)| {
                                (edge_key.clone(), edge_val.clone(), child_comparable_arc.as_arc().clone())
                            })
                        })
                        .collect()
                }; // node lock released here

                for (edge_key, edge_val, child_arc) in children_edges_values_arcs {
                    let child_arc_ptr = Arc::as_ptr(&child_arc);
                    if processed.contains(&child_arc_ptr) {
                        continue; // Skip already processed children
                    }

                    // The candidate arrival depth for this child is one more than parent's arrival depth.
                    let candidate_arrival_depth = arr_depth.saturating_add(1);

                    // Compute candidate V for child using the potentially failing step function. Lock briefly.
                    let candidate_v_opt = {
                        let child_node = child_arc.lock().expect("Mutex poisoned during step");
                        step(&node_val_merged, &edge_key, &edge_val, &child_node)
                    }; // child_node lock released here

                    if let Some(candidate_v) = candidate_v_opt {
                        // Update state for the child: merge the new candidate V and update arrival depth.
                        let mut current_child_arr_depth = 0; // Will be updated by entry API
                        state.entry(child_arc_ptr)
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
                        let mut propagation_result = Ok(()); // Track result of propagation
                        { // Scope for child lock
                            let mut child_node = child_arc.lock().expect("Mutex poisoned while updating child max_depth");
                            if candidate_arrival_depth > child_node.max_depth {
                                child_node.max_depth = candidate_arrival_depth;
                                // Need to propagate this update downward. Must drop lock before calling.
                                drop(child_node); // Explicit drop before propagation call

                                // Handle the Result from propagate_max_depth. Currently panics on cycle.
                                // TODO: Consider handling this error more gracefully if needed.
                                propagation_result = Trie::<EK, EV, T>::propagate_max_depth(child_arc.clone(), candidate_arrival_depth);
                                if propagation_result.is_err() {
                                    // Panic, as per the function documentation note.
                                    propagation_result.expect("Cycle detected during max_depth propagation within special_map");
                                }

                                // Re-acquire lock briefly to get the potentially updated max_depth
                                child_current_max_depth = child_arc.lock().expect("Mutex poisoned after propagate").max_depth;
                            } else {
                                child_current_max_depth = child_node.max_depth;
                            }
                        } // child_node lock released here


                        // Check readiness: does the *current* arrival depth in state match the child's *current* max_depth?
                        // Use >= to handle potential inconsistencies.
                        if current_child_arr_depth >= child_current_max_depth {
                            // Only queue if it's ready and not already processed
                            if !processed.contains(&child_arc_ptr) {
                                 ready.push_back(child_arc.clone());
                            }
                        }
                        // else: Child is not ready yet (arrival depth < max_depth), it might be queued later.
                    }
                    // If step returned None, we implicitly do nothing for this edge.
                } // end for each child edge
            } // end if should_continue_processing_children
        } // end while queue not empty

        // After the loop, check for unprocessed nodes (optional debug info)
        // Check initial nodes first
        let mut unprocessed_initial = false;
        for initial_arc_ptr in initial_set {
            if !processed.contains(&initial_arc_ptr) {
                if !unprocessed_initial {
                     crate::debug!(3, "Warning: Some initial nodes were not processed (Arc Ptrs):");
                     unprocessed_initial = true;
                }
                crate::debug!(3, "  - {:p}", initial_arc_ptr);
            }
        }
        // Check nodes remaining in state
        let mut unprocessed_in_state = false;
        for (arc_ptr, (_v, arr_depth)) in state.iter() {
            if !processed.contains(arc_ptr) {
                 if !unprocessed_in_state {
                     crate::debug!(3, "Warning: Nodes remaining in state but not processed (Arc Ptr, arrival_depth):");
                     unprocessed_in_state = true;
                 }
                crate::debug!(3, "  - ({:p}, {})", arc_ptr, arr_depth);
            }
        }
    }


    /// Attempts to insert an edge, potentially merging with existing edges/nodes based on provided functions.
    /// Uses a two-phase approach: check for merges immutably, then apply changes mutably.
    pub fn try_insert_or_merge_edge<FMergeEV, FMergeNV>(
        &mut self,
        edge_key: EK,
        edge_value: EV, // The NEW edge value to potentially merge/insert
        new_node_value: T, // The NEW node value if creating node
        mut merge_edge_value: FMergeEV, // FnMut(&EV, EV) -> Option<EV> (existing, new)
        mut merge_node_value: FMergeNV, // FnMut(&T, T) -> Option<T> (existing, new)
    ) -> Result<Arc<Mutex<Trie<EK, EV, T>>>, CycleDetectedError>

    where
        FMergeEV: FnMut(&EV, EV) -> Option<EV>,
        FMergeNV: FnMut(&T, T) -> Option<T>,
        EV: Clone,
        T: Clone,
    {
        // --- Check Phase 1: Node Merge Possibility ---
        // Find the first ComparableArc key `ca` where node merge IS POSSIBLE.
        let target_ca_for_node_merge: Option<ComparableArc<Trie<EK, EV, T>>> =
            if let Some(children_map) = self.children.get(&edge_key) { // children_map is &BTreeMap<ComparableArc, EV>
                children_map.iter().find_map(|(ca, _current_edge_val)| { // ca is &ComparableArc, _current_edge_val is &EV
                    let node_arc = ca.as_arc();
                    let can_merge = {
                        let node_guard = node_arc.lock().expect("Lock failed during node merge check");
                        // Check if merge IS POSSIBLE without calculating the value yet
                        merge_node_value(&node_guard.value, new_node_value.clone()).is_some()
                    };
                    if can_merge {
                        Some(ca.clone()) // Clone the ComparableArc key
                    } else {
                        None
                    }
                })
            } else {
                None
            };


        // --- Apply Phase 1: Node Merge ---
        if let Some(target_ca) = target_ca_for_node_merge { // target_ca is ComparableArc
            let node_arc = target_ca.as_arc().clone(); // Clone Arc for return
            {
                let mut node_guard = target_ca.as_arc().lock().expect("Lock failed for node update");
                // Calculate the merged value NOW and update
                // Pass the original new_node_value by value, consuming it here.
                if let Some(merged_val) = merge_node_value(&node_guard.value, new_node_value) {
                     node_guard.value = merged_val;
                } else {
                    // This should not happen if the check phase logic is correct and merge_node_value is deterministic
                    panic!("Node merge check succeeded but merge failed during apply phase");
                }
            } // Lock released

            // Try update edge value for the found target_ca
            let edge_val_mut = self.children.get_mut(&edge_key)
                .expect("Children map disappeared for edge key")
                .get_mut(&target_ca)
                .expect("Child entry disappeared from map");

            // Pass the original edge_value by value, consuming it here if merge succeeds.
            if let Some(merged_ev) = merge_edge_value(edge_val_mut, edge_value) {
                 *edge_val_mut = merged_ev;
            }
            // Return the Arc corresponding to the merged node
            return Ok(node_arc);
        }
        // Note: new_node_value and edge_value are potentially consumed above if Phase 1 runs.
        // If Phase 1 did not run, they are still available for Phase 2/3.

        // --- Check Phase 2: Edge Merge Possibility ---
        let target_ca_for_edge_merge: Option<ComparableArc<Trie<EK, EV, T>>> =
            if let Some(children_map) = self.children.get(&edge_key) {
                 children_map.iter().find_map(|(ca, existing_ev)| { // ca is &ComparableArc, existing_ev is &EV
                    // Check if merge IS POSSIBLE
                    if merge_edge_value(existing_ev, edge_value.clone()).is_some() { // Clone edge_value for check
                        Some(ca.clone()) // Clone the ComparableArc key
                    } else {
                        None
                    }
                 })
            } else {
                None
            };


        // --- Apply Phase 2: Edge Merge ---
        if let Some(target_ca) = target_ca_for_edge_merge { // target_ca is ComparableArc
            let node_arc = target_ca.as_arc().clone(); // Clone Arc for return
            let edge_val_mut = self.children.get_mut(&edge_key)
                .expect("Children map disappeared for edge key")
                .get_mut(&target_ca)
                .expect("Child entry disappeared from map");
            // Re-calculate merged edge value and update
            // Pass the original edge_value by value, consuming it here.
            if let Some(merged_ev) = merge_edge_value(edge_val_mut, edge_value) {
                 *edge_val_mut = merged_ev;
            } else {
                 // This should not happen if the check phase logic is correct and merge_edge_value is deterministic
                 panic!("Edge merge check succeeded but merge failed during apply phase");
            }
            // Return the Arc corresponding to the existing node with the merged edge
            return Ok(node_arc);
        }

        // --- Apply Phase 3: Create New Node and Edge ---
        // No suitable node/edge found for merging.
        // Use the original new_node_value and edge_value (which were not consumed if Phase 1/2 didn't run)
        let new_node = Arc::new(Mutex::new(Trie::new(new_node_value))); // Consumes new_node_value
        // Use try_insert which handles adding to children vec (creating if needed) and cycle checks/depth updates
        self.try_insert(edge_key, edge_value, new_node.clone())?; // Consumes edge_value
        Ok(new_node)
    }
}


/// A helper function to print the structure of the Trie/DAG via BFS.
/// Uses Arc::as_ptr for node identity.
pub(crate) fn dump_structure<EK: Debug + Ord, EV: Debug, T: Debug>(root: Arc<Mutex<Trie<EK, EV, T>>>) {
    let mut queue = VecDeque::new();
    let mut seen: HashSet<*const Mutex<Trie<EK, EV, T>>> = HashSet::new(); // Use Arc pointer for seen set

    println!("Dumping Trie Structure (BFS):");
    println!("(Node identity shown as {:p} which corresponds to the Mutex Arc pointer)", std::ptr::null::<Mutex<Trie<EK, EV, T>>>()); // Print format hint

    let root_arc_ptr = Arc::as_ptr(&root);
    if seen.insert(root_arc_ptr) {
        queue.push_back(root);
    }

    while let Some(node_arc) = queue.pop_front() {
        let node_arc_ptr = Arc::as_ptr(&node_arc); // Get pointer for current node
        let node = node_arc.lock().expect("Mutex poisoned during dump");
        // Use node_arc_ptr for printing identity
        println!("{:p}: Value: {:?}, MaxDepth: {}", node_arc_ptr, node.value, node.max_depth);

        // Iterate through edges and their corresponding maps of children
        for (edge_key, children_map) in node.children.iter() { // children_map is &BTreeMap<ComparableArc, EV>
            // Iterate through each (ComparableArc, EV) tuple in the map
            for (child_comparable_arc, edge_val) in children_map.iter() { // Iterate over map entries
                let child_arc = child_comparable_arc.as_arc();
                let child_arc_ptr = Arc::as_ptr(child_arc); // Get pointer for child node
                // Use child_arc_ptr for printing identity
                println!("  - Edge Key: {:?}, Edge Val: {:?} -> Child: {:p}", edge_key, edge_val, child_arc_ptr);
                if seen.insert(child_arc_ptr) {
                    queue.push_back(child_arc.clone());
                }
            }
        }
        // node lock released here
    }
}

/// A helper struct to facilitate inserting an edge into a Trie,
/// trying multiple potential destinations and optionally creating a new node.
/// Provides a chainable interface.
pub struct EdgeInserter<EK, EV, T, FMergeEV>
where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FMergeEV: FnMut(&EV, EV) -> Option<EV>, // Closure to merge edge values if edge exists
{
    source_arc: Arc<Mutex<Trie<EK, EV, T>>>, // The source node for the edge
    edge_key: EK,                            // The key for the edge to be inserted
    edge_value: EV,                          // The value for the edge to be inserted
    merge_edge_value: FMergeEV,              // The function to merge edge values
    result: Option<Arc<Mutex<Trie<EK, EV, T>>>>, // Stores the successful destination node
}

impl<EK, EV, T, FMergeEV> EdgeInserter<EK, EV, T, FMergeEV>
where
    EK: Ord + Clone + Debug,
    EV: Clone,
    T: Clone,
    FMergeEV: FnMut(&EV, EV) -> Option<EV>,
{
    /// Creates a new `EdgeInserter`.
    ///
    /// # Arguments
    ///
    /// * `source_arc`: The source node where the edge originates.
    /// * `edge_key`: The key for the new edge.
    /// * `edge_value`: The value for the new edge.
    /// * `merge_edge_value`: A closure that takes the existing edge value and the new edge value,
    ///   returning `Some(merged_value)` if merging is possible/desired, or `None` otherwise.
    ///   This is only called if an edge with the same `edge_key` already points to the
    ///   `destination` being tried.
    pub fn new(
        source_arc: Arc<Mutex<Trie<EK, EV, T>>>,
        edge_key: EK,
        edge_value: EV,
        merge_edge_value: FMergeEV,
    ) -> Self {
        EdgeInserter {
            source_arc,
            edge_key,
            edge_value,
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

        let mut source = self.source_arc.lock().expect("Mutex poisoned while locking source in try_destination");
        let destination_comparable = ComparableArc::new(destination.clone());

        // Check if edge already exists and try merging EV
        if let Some(existing_ev_mut) = source.children.get_mut(&self.edge_key).and_then(|dest_map| dest_map.get_mut(&destination_comparable)) {
            if let Some(merged_ev) = (self.merge_edge_value)(existing_ev_mut, self.edge_value.clone()) {
                *existing_ev_mut = merged_ev;
                self.result = Some(destination); // Merge successful, destination found
            }
            // If merge fails, result remains None, but we don't try inserting again.
        } else {
            // Edge doesn't exist, try inserting
            match source.try_insert(self.edge_key.clone(), self.edge_value.clone(), destination.clone()) {
                Ok(()) => {
                    self.result = Some(destination); // Insert successful, destination found
                }
                Err(CycleDetectedError) => {
                    // Cycle detected, insert failed, result remains None
                    crate::debug!(4, "Cycle detected trying to insert edge {:?} to node {:p}", self.edge_key, Arc::as_ptr(&destination));
                }
            }
        }
        drop(source);
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
            self = self.try_destination(destination.clone());
        }
        self
    }

    pub fn try_destinations_iter_with<F, R>(mut self, destinations: F) -> Self
    where
        F: Fn() -> R,
        R: Iterator<Item = Arc<Mutex<Trie<EK, EV, T>>>>,
    {
        for destination in destinations() {
            self = self.try_destination(destination.clone());
        }
        self
    }

    /// Tries to establish an edge to any existing child node of the source node,
    /// regardless of the edge key under which the child was originally added.
    ///
    /// Iterates through *all* children of the source node. For each child, it calls
    /// `try_destination` to attempt adding the new edge (`self.edge_key`, `self.edge_value`)
    /// pointing to that child. The first successful attempt (either merging an existing
    /// edge with the same key to that child, or inserting a new edge if no cycle occurs)
    /// sets the result and stops further attempts.
    ///
    /// Returns `self` to allow chaining.
    pub fn try_children(mut self) -> Self {
        if self.result.is_some() {
            return self;
        }

        // Collect ALL children arcs, regardless of the edge key they are under.
        let all_children_arcs: Vec<Arc<Mutex<Trie<EK, EV, T>>>> = {
            let source = self.source_arc.lock().expect("Mutex poisoned while locking source in try_children");
            source.children
                .values() // Iterates over BTreeMap<ComparableArc, EV> for each edge key
                .flat_map(|dest_map| dest_map.keys().map(|ca| ca.as_arc().clone())) // Extract Arcs
                .collect()
        }; // Lock is dropped here

        // Call try_slice with the collected children
        self.try_destinations(&all_children_arcs)
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
        let mut source = self.source_arc.lock().expect("Mutex poisoned while locking source in else_create_with_value");

        match source.try_insert(self.edge_key.clone(), self.edge_value.clone(), new_node_arc.clone()) {
            Ok(()) => {
                self.result = Some(new_node_arc);
            }
            Err(CycleDetectedError) => {
                // Insert failed (e.g., cycle detected even with new node - unusual)
                crate::debug!(1, "Cycle detected trying to insert edge {:?} to NEW node {:p}. Creation failed.", self.edge_key, Arc::as_ptr(&new_node_arc));
                // result remains None
            }
        }
        drop(source);
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
    ///   returning `Some(merged_value)` if merging is possible/desired, or `None` otherwise.
    ///   This is only called by `EdgeInserter::try_destination` if an edge with the same
    ///   `edge_key` already points to the destination being tried.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::sync::{Arc, Mutex};
    /// use crate::datastructures::trie::Trie; // Assuming Trie is in this module
    /// use crate::datastructures::trie::EdgeInserter; // Also need EdgeInserter
    /// use crate::datastructures::hybrid_bitset::HybridBitset; // Need HybridBitset
    /// use std::iter::FromIterator; // For collect
    ///
    /// #[derive(Debug, Clone, Default)] // Need Default for else_create
    /// struct NodeValue { /* ... */ }
    ///
    /// // Example merge function for edge values (e.g., HybridBitset)
    /// fn merge_bitset_union(existing: &HybridBitset, new: HybridBitset) -> Option<HybridBitset> {
    ///     // Union is always possible/desired
    ///     Some(existing | &new) // Use reference for the OR operation
    /// }
    ///
    /// // Assuming root_node is Arc<Mutex<Trie<String, HybridBitset, NodeValue>>>
    /// let root_node: Arc<Mutex<Trie<String, HybridBitset, NodeValue>>> = Arc::new(Mutex::new(Trie::new(NodeValue::default())));
    ///
    /// // Create a HybridBitset to use as edge value
    /// let new_edge_value: HybridBitset = vec![].into_iter().collect();
    ///
    /// let potential_destinations: Vec<Arc<Mutex<Trie<String, HybridBitset, NodeValue>>>> = vec![/* ... */];
    ///
    /// let new_or_existing_node = { // Use a block to drop the temporary mutex guard
    ///     let root_guard = root_node.lock().unwrap(); // Get a guard to call insert_edge
    ///     root_guard.insert_edge("key".to_string(), new_edge_value, merge_bitset_union)
    ///         .try_destinations(&potential_destinations) // potential_destinations is &[Arc<Mutex<...>>]
    ///         .else_create() // Or else_create_with(...) or else_create_with_value(...)
    ///         .unwrap()
    /// };
    /// // root_node (Arc<Mutex>) is an Arc<Mutex> and can be used further.
    /// ```
    pub fn insert_edge<FMergeEV>(
        &self, // Note: This method takes &self, not &mut self. The EdgeInserter handles the mutation via Arc<Mutex>.
        edge_key: EK,
        edge_value: EV,
        merge_edge_value: FMergeEV,
    ) -> EdgeInserter<EK, EV, T, FMergeEV>
    where
         FMergeEV: FnMut(&EV, EV) -> Option<EV>,
    {
            EdgeInserter::new(Arc::new(Mutex::new(self.clone())), edge_key, edge_value, merge_edge_value)
        }
    }

/// Attempts to establish an edge from `source` to a single `destination`,
/// optionally merging edge values if an edge already exists.
/// Returns `Some(Arc<Mutex<Trie<...>>>)` if merge or insert succeeded,
/// or `None` if merge failed or a cycle was detected.
pub fn try_destination<EK, EV, T, FMergeEV>(
    source: Arc<Mutex<Trie<EK, EV, T>>>,
    edge_key: EK,
    edge_value: EV,
    destination: Arc<Mutex<Trie<EK, EV, T>>>,
    merge_edge_value: FMergeEV,
) -> Option<Arc<Mutex<Trie<EK, EV, T>>>>
where
    EK: Ord + Clone + Debug,
    EV: Clone,
    T: Clone,
    FMergeEV: FnMut(&EV, EV) -> Option<EV>,
{
    EdgeInserter::new(source, edge_key, edge_value, merge_edge_value)
        .try_destination(destination)
        .into_option()
}

/// Attempts to establish an edge from `source` to any of the provided `destinations`,
/// returning the first successful one (merge or insert), or `None` if all attempts failed.
pub fn try_destination_with<EK, EV, T, FMergeEV>(
    source: Arc<Mutex<Trie<EK, EV, T>>>,
    edge_key: EK,
    edge_value: EV,
    destinations: &[Arc<Mutex<Trie<EK, EV, T>>>],
    merge_edge_value: FMergeEV,
) -> Option<Arc<Mutex<Trie<EK, EV, T>>>>
where
    EK: Ord + Clone + Debug,
    EV: Clone,
    T: Clone,
    FMergeEV: FnMut(&EV, EV) -> Option<EV>,
{
    EdgeInserter::new(source, edge_key, edge_value, merge_edge_value)
        .try_destinations(destinations)
        .into_option()
}


// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::collections::{HashSet, HashMap};
    use crate::datastructures::hybrid_bitset::HybridBitset; // Import HybridBitset for tests
    use std::iter::FromIterator; // For collect

    // Use concrete types for merge tests
    type TestTrieMerge = Trie<&'static str, Vec<i32>, String>;
    type TestNodeMerge = Arc<Mutex<TestTrieMerge>>;
    // Use simpler types for basic tests
    type TestTrieBasic = Trie<&'static str, &'static str, i32>;
    type TestNodeBasic = Arc<Mutex<TestTrieBasic>>;

    // Use concrete types for EdgeInserter tests
    type TestTrieEI = Trie<&'static str, HybridBitset, String>; // Use HybridBitset here
    type TestNodeEI = Arc<Mutex<TestTrieEI>>;

    // Helper to get Arc pointer for tests
    fn arc_ptr<N>(arc: &Arc<Mutex<N>>) -> *const Mutex<N> {
        Arc::as_ptr(arc)
    }

    #[test]
    fn test_try_insertion_and_retrieval() {
        let root_node: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(2)));
        let child3: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(3))); // Another child for 'a'

        { // Scope for mutable borrow of root
            let mut root = root_node.lock().unwrap();
            root.try_insert("a", "edge_a1", child1.clone()).expect("Insert failed");
            root.try_insert("b", "edge_b", child2.clone()).expect("Insert failed");
            root.try_insert("a", "edge_a3", child3.clone()).expect("Insert failed"); // Insert second child for 'a'
        } // root lock released

        // Scope for read-only borrow of root
        let root = root_node.lock().unwrap();

        // Test get for 'a'
        let retrieved_children_a = root.get(&"a").expect("Failed to get children for 'a'"); // Now a &BTreeMap
        assert_eq!(retrieved_children_a.len(), 2);
        // Use Arc pointers for comparison
        let retrieved_data_a: HashSet<(&str, *const Mutex<TestTrieBasic>)> = retrieved_children_a
            .iter() // Iterates yielding (&ComparableArc, &&str)
            .map(|(ca, ev_ref)| (*ev_ref, arc_ptr(ca.as_arc()))) // Dereference ev_ref twice
            .collect();
        assert!(retrieved_data_a.contains(&("edge_a1", arc_ptr(&child1))));
        assert!(retrieved_data_a.contains(&("edge_a3", arc_ptr(&child3))));

        // Test get for 'b'
        let retrieved_children_b = root.get(&"b").expect("Failed to get child 'b'"); // Now a &BTreeMap
        assert_eq!(retrieved_children_b.len(), 1);
        let (ca, ev_ref) = retrieved_children_b.iter().next().unwrap(); // Get the single entry
        assert_eq!(*ev_ref, "edge_b"); // Check edge value
        assert!(Arc::ptr_eq(ca.as_arc(), &child2)); // Check Arc pointer equality

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
        assert!(child1.lock().unwrap().is_leaf());
        assert!(child2.lock().unwrap().is_leaf());
        assert!(child3.lock().unwrap().is_leaf());
    }

    #[test]
    fn test_multiple_children_same_edge_key() {
        // Structure:
        //      root (0) --"edge", "val1"--> child1 (1)
        //           |
        //            -----"edge", "val2"--> child2 (2)
        let root: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(2)));

        {
            let mut r = root.lock().unwrap();
            r.try_insert("edge", "val1", child1.clone()).unwrap();
            r.try_insert("edge", "val2", child2.clone()).unwrap();
        } // root lock released

        // Check retrieval - lock root again
        {
            let binding = root.lock().unwrap();
            let children_map = binding.get(&"edge").unwrap(); // Now a &BTreeMap
            assert_eq!(children_map.len(), 2);
            let child_data: HashSet<(&str, *const Mutex<TestTrieBasic>)> = children_map
                .iter()
                .map(|(ca, ev_ref)| (*ev_ref, arc_ptr(ca.as_arc())))
                .collect();
            assert!(child_data.contains(&("val1", arc_ptr(&child1))));
            assert!(child_data.contains(&("val2", arc_ptr(&child2))));
        } // root lock released

        // Check all_nodes - call *after* releasing lock
        let all = Trie::all_nodes(root.clone());
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

        // Check edge info captured by step
        assert_eq!(edge_info_at_step.len(), 2); // 2 edges traversed from root
        assert!(edge_info_at_step.contains(&("edge", "val1")) || edge_info_at_step.contains(&("edge", "val2")));
         assert!(edge_info_at_step.contains(&("edge", "val2")) || edge_info_at_step.contains(&("edge", "val1")));
         assert_ne!(edge_info_at_step[0], edge_info_at_step[1]); // Must be different children
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
        let root: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(3)));

        {
            let mut r = root.lock().unwrap();
            r.try_insert("r->c1", "e1", child1.clone()).unwrap();
            r.try_insert("r->c2", "e2", child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.lock().unwrap();
            c1.try_insert("c1->gc", "e3", grandchild.clone()).unwrap();
        }

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

        // Check processing order
        assert_eq!(processed_node_values.len(), 4);
        assert_eq!(processed_node_values[0], 0); // Root first
        let pos1 = processed_node_values.iter().position(|&v| v == 1).unwrap();
        let pos2 = processed_node_values.iter().position(|&v| v == 2).unwrap();
        let pos3 = processed_node_values.iter().position(|&v| v == 3).unwrap();
        assert!(pos1 > 0 && pos1 < 3); // c1 processed after root, before gc
        assert!(pos2 > 0 && pos2 < 4); // c2 processed after root
        assert!(pos3 > pos1);          // gc processed after c1


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
        let root: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(3)));

        {
            let mut r = root.lock().unwrap();
            r.try_insert("r1", "e1", child1.clone()).unwrap();
            r.try_insert("r2", "e2", child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.lock().unwrap();
            c1.try_insert("c1", "e3", grandchild.clone()).unwrap();
        }
        {
            let mut c2 = child2.lock().unwrap();
            c2.try_insert("c2", "e4", grandchild.clone()).unwrap(); // Diamond
        }

        let all_nodes = Trie::all_nodes(root.clone());

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
        let root: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(3)));

        // Build the structure
        {
            let mut r = root.lock().unwrap();
            r.try_insert("r->c1", "edge1", child1.clone()).unwrap();
            r.try_insert("r->c2", "edge2", child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.lock().unwrap();
            c1.try_insert("c1->gc", "edge3", grandchild.clone()).unwrap();
        }
        {
            let mut c2 = child2.lock().unwrap();
            c2.try_insert("c2->gc", "edge4", grandchild.clone()).unwrap();
        }

        // Check max_depths after insertion
        assert_eq!(root.lock().unwrap().max_depth, 0);
        assert_eq!(child1.lock().unwrap().max_depth, 1);
        assert_eq!(child2.lock().unwrap().max_depth, 1);
        assert_eq!(grandchild.lock().unwrap().max_depth, 2);

        let processed_nodes = Arc::new(Mutex::new(HashMap::<i32, i32>::new()));
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
        let root: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(42)));
        let nodes = Trie::all_nodes(root.clone());
        assert_eq!(nodes.len(), 1);
        assert!(Arc::ptr_eq(&nodes[0], &root));
        assert!(root.lock().unwrap().is_leaf()); // Lock needed here

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
        let root: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(0)));
        let child: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(1)));

        // Insert root -> child
        let insert1_result = {
            let mut r = root.lock().unwrap();
            r.try_insert("r->c", "e1", child.clone())
        };
        assert!(insert1_result.is_ok());
        assert_eq!(child.lock().unwrap().max_depth, 1);
        assert_eq!(root.lock().unwrap().max_depth, 0);

        // Attempt insert child -> root
        let insert2_result = {
            let mut c = child.lock().unwrap();
            // This insert should call detect_cycle(child_ptr, &root), which should detect the cycle.
            c.try_insert("c->r", "e2", root.clone())
        };

        // Assert that cycle detection returned an error
        assert!(insert2_result.is_err());
        assert_eq!(insert2_result.err(), Some(CycleDetectedError));

        // Check state after failed insertion:
        // - The edge must *not* be present because the insertion was rejected.
        let child_locked = child.lock().unwrap();
        let has_edge_to_root = if let Some(dest_map) = child_locked.children.get("c->r") {
             let lookup_key = ComparableArc::new(root.clone());
             dest_map.contains_key(&lookup_key)
         } else {
             false
         };
        assert!(!has_edge_to_root, "Edge that would introduce a cycle should NOT be present");

        // - Max depths should be unchanged from before the failed insertion attempt.
        assert_eq!(root.lock().unwrap().max_depth, 0);
        assert_eq!(child_locked.max_depth, 1);

        println!("Done testing cycle detection on try_insert");
    }


    #[test]
    fn test_cycle_all_nodes_no_panic() {
        // Cycle:  root -> child -> root
        // Manually create cycle without insert's propagation.
        let root: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(0)));
        let child: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(1)));

        // Manually create links
        root.lock().unwrap().force_insert_to_node("r->c", "e1", &child);
        child.lock().unwrap().force_insert_to_node("c->r", "e2", &root);
        // Manually set depths (optional for all_nodes logic)
        root.lock().unwrap().max_depth = 0;
        child.lock().unwrap().max_depth = 1;

        let all_nodes = Trie::all_nodes(root.clone());

        // Should detect both nodes exactly once.
        assert_eq!(all_nodes.len(), 2);
        let node_ptrs: HashSet<_> = all_nodes.iter().map(arc_ptr).collect(); // Use arc_ptr
        assert_eq!(node_ptrs.len(), 2);
        assert!(node_ptrs.contains(&arc_ptr(&root)));
        assert!(node_ptrs.contains(&arc_ptr(&child)));
    }


    #[test]
    fn test_cycle_special_map_no_panic_limited_processing() {
        // Cycle: root -> child -> root.
        // Manually create cycle.
        let root: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(0)));
        let child: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(1)));

        // Manually create links
        root.lock().unwrap().force_insert_to_node("r->c", "e1", &child);
        child.lock().unwrap().force_insert_to_node("c->r", "e2", &root);
        // Manually set depths. These are crucial for special_map's readiness check.
        root.lock().unwrap().max_depth = 0; // Initial node, depth 0
        child.lock().unwrap().max_depth = 1; // Child reachable at depth 1

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

        // Expected behavior: Root processed (V=100). Child processed (V=101).
        // Propagation back to root skipped because root is already processed.
        // The max_depth update inside special_map might trigger, but propagate_max_depth
        // inside it should detect the cycle and panic (as documented).
        // Let's assume the `processed` check prevents the panic path for this test.
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
        let root: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(2)));
        let grandchild1: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(3)));
        let grandchild2: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(4)));

        {
            let mut r = root.lock().unwrap();
            r.try_insert("r->c1", "e1", child1.clone()).unwrap();
            r.try_insert("r->c2", "e2", child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.lock().unwrap();
            c1.try_insert("c1->gc1", "e3", grandchild1.clone()).unwrap();
        }
        {
            let mut c2 = child2.lock().unwrap();
            c2.try_insert("c2->gc2", "e4", grandchild2.clone()).unwrap();
        }

        let processed_nodes = Arc::new(Mutex::new(HashSet::<i32>::new()));
        let computed_values = Arc::new(Mutex::new(HashMap::<i32, i32>::new()));

        Trie::special_map(
            vec![(root.clone(), 100)],
            |p_val, _ek, _ev, _child_node| Some(p_val + 1), // step: increment value
            |current_v, new_v| *current_v = new_v, // merge: replace
            {
                let processed_nodes = processed_nodes.clone();
                let computed_values = computed_values.clone();
                move |node, final_v| {
                    processed_nodes.lock().unwrap().insert(node.value);
                    computed_values.lock().unwrap().insert(node.value, *final_v);
                    if node.value == 1 { // Stop processing children if node value is 1 (child1)
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

    #[test]
    fn test_special_map_step_returns_none() {
        // Structure:
        //      root (0) --"keep"--> c1(1)
        //           |
        //           --"skip"--> c2(2) --"keep"--> gc2(3)
        // Step returns None if edge key is "skip".
        let root: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(2)));
        let grandchild2: TestNodeBasic = Arc::new(Mutex::new(TestTrieBasic::new(3)));

        {
            let mut r = root.lock().unwrap();
            r.try_insert("keep", "e1", child1.clone()).unwrap();
            r.try_insert("skip", "e2", child2.clone()).unwrap();
        }
        {
            let mut c2 = child2.lock().unwrap();
            c2.try_insert("keep", "e3", grandchild2.clone()).unwrap();
        }

        let processed_nodes = Arc::new(Mutex::new(HashSet::<i32>::new()));
        let computed_values = Arc::new(Mutex::new(HashMap::<i32, i32>::new()));

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
                    processed_nodes.lock().unwrap().insert(node.value);
                    computed_values.lock().unwrap().insert(node.value, *final_v);
                    true // Always continue processing if node is reached
                }
            }
        );

        let final_processed = processed_nodes.lock().unwrap();
        let final_values = computed_values.lock().unwrap();

        // Expected processed nodes: 0, 1. Nodes 2 and 3 should be skipped.
        assert_eq!(final_processed.len(), 2);
        assert!(final_processed.contains(&0));
        assert!(final_processed.contains(&1)); // Reached via "keep" edge
        assert!(!final_processed.contains(&2)); // Skipped via "skip" edge
        assert!(!final_processed.contains(&3)); // Never reached because c2 was skipped

        // Check computed values
        assert_eq!(final_values.get(&0), Some(&100));
        assert_eq!(final_values.get(&1), Some(&101));
        assert_eq!(final_values.get(&2), None); // Not processed
        assert_eq!(final_values.get(&3), None); // Not processed
    }


    // --- Tests for insert_or_merge_edge ---

    // Helper merge functions for tests
    // Merge edge value (Vec<i32>): Append new vec to existing if existing is not empty
    fn merge_ev_append(existing_ev: &Vec<i32>, new_ev: Vec<i32>) -> Option<Vec<i32>> {
        if !existing_ev.is_empty() {
            let mut merged = existing_ev.clone();
            merged.extend(new_ev);
            Some(merged)
        } else {
            None // Don't merge into an empty vec
        }
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

    #[test]
    fn test_insert_or_merge_no_existing_key() {
        let root_node: TestNodeMerge = Arc::new(Mutex::new(TestTrieMerge::new("root".to_string())));
        let edge_key = "new_key";
        let edge_val = vec![1];
        let node_val = "new_node".to_string();

        let returned_node_res = { // Scope for mutable borrow
            let mut root = root_node.lock().unwrap();
            root.try_insert_or_merge_edge(
                edge_key,
                edge_val.clone(),
                node_val.clone(),
                merge_ev_append,
                merge_nv_append_if_flag,
            )
        };
        assert!(returned_node_res.is_ok());
        let returned_node = returned_node_res.unwrap();

        // Check that a new node was created
        assert_eq!(returned_node.lock().unwrap().value, node_val);
        assert_eq!(returned_node.lock().unwrap().max_depth, 1); // Depth updated by try_insert

        // Check that the edge was added to the root
        let root = root_node.lock().unwrap(); // Re-lock read-only
        let children_map = root.children.get(edge_key).unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap(); // Get the single entry
        assert_eq!(*ev, edge_val); // Original edge value
        assert!(Arc::ptr_eq(ca.as_arc(), &returned_node));
        assert_eq!(root.max_depth, 0); // Root depth unchanged
    }

    #[test]
    fn test_insert_or_merge_node_merge_success_edge_merge_success() {
        let root_node: TestNodeMerge = Arc::new(Mutex::new(TestTrieMerge::new("root".to_string())));
        let existing_node: TestNodeMerge = Arc::new(Mutex::new(TestTrieMerge::new("child_mergeable".to_string())));
        { // Initial insert
            let mut root = root_node.lock().unwrap();
            root.try_insert("key", vec![10], existing_node.clone()).unwrap();
        }

        let edge_key = "key";
        let edge_val = vec![1]; // New edge value
        let node_val = "data".to_string(); // New node value data

        let returned_node_res = { // Scope for mutable borrow
            let mut root = root_node.lock().unwrap();
            root.try_insert_or_merge_edge(
                edge_key,
                edge_val.clone(),
                node_val.clone(),
                merge_ev_append, // Should succeed: [10] + [1] -> [10, 1]
                merge_nv_append_if_flag, // Should succeed: "child_mergeable" + "data" -> "child_mergeable|data"
            )
        };
        assert!(returned_node_res.is_ok());
        let returned_node = returned_node_res.unwrap();

        // Check that the existing node was returned and updated
        assert!(Arc::ptr_eq(&returned_node, &existing_node));
        assert_eq!(returned_node.lock().unwrap().value, "child_mergeable|data");
        assert_eq!(returned_node.lock().unwrap().max_depth, 1); // Depth unchanged

        // Check that the edge value was updated in the root
        let root = root_node.lock().unwrap(); // Re-lock read-only
        let children_map = root.children.get(edge_key).unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, vec![10, 1]); // Merged edge value
        assert!(Arc::ptr_eq(ca.as_arc(), &existing_node));
    }

     #[test]
    fn test_insert_or_merge_node_merge_success_edge_merge_fail() {
        let root_node: TestNodeMerge = Arc::new(Mutex::new(TestTrieMerge::new("root".to_string())));
        // Edge value is empty, so merge_ev_append will fail
        let existing_node: TestNodeMerge = Arc::new(Mutex::new(TestTrieMerge::new("child_mergeable".to_string())));
        { // Initial insert
            let mut root = root_node.lock().unwrap();
            root.try_insert("key", vec![], existing_node.clone()).unwrap();
        }

        let edge_key = "key";
        let edge_val = vec![1];
        let node_val = "data".to_string();

        let returned_node_res = { // Scope for mutable borrow
            let mut root = root_node.lock().unwrap();
            root.try_insert_or_merge_edge(
                edge_key,
                edge_val.clone(),
                node_val.clone(),
                merge_ev_append, // Should fail: existing is empty
                merge_nv_append_if_flag, // Should succeed
            )
        };
        assert!(returned_node_res.is_ok());
        let returned_node = returned_node_res.unwrap();

        // Check existing node returned and updated (due to node merge success)
        assert!(Arc::ptr_eq(&returned_node, &existing_node));
        assert_eq!(returned_node.lock().unwrap().value, "child_mergeable|data");

        // Check edge value was *not* updated (because edge merge failed)
        let root = root_node.lock().unwrap(); // Re-lock read-only
        let children_map = root.children.get(edge_key).unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, Vec::<i32>::new()); // Original value remains
        assert!(Arc::ptr_eq(ca.as_arc(), &existing_node));
    }

    #[test]
    fn test_insert_or_merge_node_merge_fail_edge_merge_success() {
        let root_node: TestNodeMerge = Arc::new(Mutex::new(TestTrieMerge::new("root".to_string())));
        // Node value does not contain "mergeable", so merge_nv will fail
        let existing_node: TestNodeMerge = Arc::new(Mutex::new(TestTrieMerge::new("child_not_mergeable".to_string())));
        { // Initial insert
            let mut root = root_node.lock().unwrap();
            root.try_insert("key", vec![10], existing_node.clone()).unwrap();
        }

        let edge_key = "key";
        let edge_val = vec![1];
        let node_val = "data".to_string();

        let returned_node_res = { // Scope for mutable borrow
            let mut root = root_node.lock().unwrap();
            root.try_insert_or_merge_edge(
                edge_key,
                edge_val.clone(),
                node_val.clone(),
                merge_ev_append, // Should succeed
                merge_nv_append_if_flag, // Should fail
            )
        };
        assert!(returned_node_res.is_ok());
        let returned_node = returned_node_res.unwrap();

        // Check existing node returned, but *not* updated (node merge failed)
        assert!(Arc::ptr_eq(&returned_node, &existing_node));
        assert_eq!(returned_node.lock().unwrap().value, "child_not_mergeable"); // Original value

        // Check edge value *was* updated (edge merge succeeded in Pass 2)
        let root = root_node.lock().unwrap(); // Re-lock read-only
        let children_map = root.children.get(edge_key).unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, vec![10, 1]); // Merged edge value
        assert!(Arc::ptr_eq(ca.as_arc(), &existing_node));
    }

    #[test]
    fn test_insert_or_merge_both_merge_fail_creates_new() {
        let root_node: TestNodeMerge = Arc::new(Mutex::new(TestTrieMerge::new("root".to_string())));
        // Node value not mergeable, edge value empty -> both merges fail
        let existing_node: TestNodeMerge = Arc::new(Mutex::new(TestTrieMerge::new("child_not_mergeable".to_string())));
        { // Initial insert
            let mut root = root_node.lock().unwrap();
            root.try_insert("key", vec![], existing_node.clone()).unwrap();
            // Add assertion: check the edge value is indeed empty
            let children_map = root.children.get("key").unwrap();
            let (ca, ev) = children_map.iter().find(|(ca, _)| Arc::ptr_eq(ca.as_arc(), &existing_node)).unwrap();
             assert_eq!(*ev, Vec::<i32>::new());
        }

        let edge_key = "key";
        let edge_val = vec![1]; // New edge value
        let node_val = "new_data".to_string(); // New node value

        let returned_node_res = { // Scope for mutable borrow
            let mut root = root_node.lock().unwrap();
            root.try_insert_or_merge_edge(
                edge_key,
                edge_val.clone(),
                node_val.clone(),
                merge_ev_append, // Fails (existing empty)
                merge_nv_append_if_flag, // Fails (existing doesn't contain "mergeable")
            )
        };
        assert!(returned_node_res.is_ok());
        let returned_node = returned_node_res.unwrap();

        // Check a *new* node was returned (Pass 3 executed)
        assert!(!Arc::ptr_eq(&returned_node, &existing_node));
        assert_eq!(returned_node.lock().unwrap().value, node_val);
        assert_eq!(returned_node.lock().unwrap().max_depth, 1); // New node depth

        // Check root now has *two* children for "key"
        let root = root_node.lock().unwrap(); // Re-lock read-only
        let children_map = root.children.get(edge_key).unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 2);

        // Find the original edge/node
        let original_edge_entry = children_map.iter().find(|(ca, _)| Arc::ptr_eq(ca.as_arc(), &existing_node)).unwrap();
        assert_eq!(*original_edge_entry.1, Vec::<i32>::new()); // Original edge value unchanged
        assert_eq!(existing_node.lock().unwrap().value, "child_not_mergeable"); // Original node value unchanged

        // Find the new edge/node
        let new_edge_entry = children_map.iter().find(|(ca, _)| Arc::ptr_eq(ca.as_arc(), &returned_node)).unwrap();
        assert_eq!(*new_edge_entry.1, edge_val); // New edge value used
    }

     #[test]
    fn test_insert_or_merge_multiple_edges_picks_first_match() {
        let root_node: TestNodeMerge = Arc::new(Mutex::new(TestTrieMerge::new("root".to_string())));

        // Edge 1: Node merge fails, Edge merge succeeds
        let node1: TestNodeMerge = Arc::new(Mutex::new(TestTrieMerge::new("node1_not_mergeable".to_string())));
        // Edge 2: Node merge succeeds, Edge merge fails
        let node2: TestNodeMerge = Arc::new(Mutex::new(TestTrieMerge::new("node2_mergeable".to_string())));

        { // Initial inserts
            let mut root = root_node.lock().unwrap();
            // Insert in specific order to test iteration
            root.try_insert("key", vec![10], node1.clone()).unwrap(); // index 0
            root.try_insert("key", vec![], node2.clone()).unwrap();   // index 1
        }

        let edge_key = "key";
        let edge_val = vec![1]; // New EV
        let node_val = "data".to_string(); // New T

        // Since node merge is checked first (Pass 1), node2 (at index 1) should be selected.
        let returned_node_res = { // Scope for mutable borrow
            let mut root = root_node.lock().unwrap();
            root.try_insert_or_merge_edge(
                edge_key,
                edge_val.clone(), // Pass vec![1] as new EV
                node_val.clone(), // Pass "data" as new T
                merge_ev_append, // Fn(&Vec<i32>, Vec<i32>) -> Option<Vec<i32>>
                merge_nv_append_if_flag, // Fn(&String, String) -> Option<String>
            )
        };
        assert!(returned_node_res.is_ok());
        let returned_node = returned_node_res.unwrap();

        // Check node2 was returned and updated (Pass 1 succeeded for index 1)
        assert!(Arc::ptr_eq(&returned_node, &node2), "Returned node should be node2"); // Check pointer equality
        assert_eq!(returned_node.lock().unwrap().value, "node2_mergeable|data", "Node2 value should be merged");

        // Check root's children: node1 unchanged, node2's edge unchanged (because edge merge failed for node2)
        let root = root_node.lock().unwrap(); // Re-lock read-only
        let children_map = root.children.get(edge_key).unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 2);

        // Check node1 state
        let edge1_info_entry = children_map.iter().find(|(ca, _ev)| Arc::ptr_eq(ca.as_arc(), &node1)).unwrap();
        assert_eq!(*edge1_info_entry.1, vec![10]); // Unchanged
        assert_eq!(node1.lock().unwrap().value, "node1_not_mergeable"); // Unchanged

        // Check node2 state
        let edge2_info_entry = children_map.iter().find(|(ca, _ev)| Arc::ptr_eq(ca.as_arc(), &node2)).unwrap();
        assert!(Arc::ptr_eq(edge2_info_entry.0.as_arc(), &node2));
        assert_eq!(*edge2_info_entry.1, Vec::<i32>::new()); // Unchanged (edge merge failed for this edge in Pass 1)
        // Node value was updated (checked above).
    }

    // test_insert_or_merge_edge_detects_cycle removed as try_insert_or_merge_edge
    // doesn't attempt to re-insert an existing node in a way that would trigger
    // cycle detection based on the node itself being passed again. Cycle detection
    // relies on the try_insert call in Pass 3 when creating a *new* edge/node.

    // --- Tests for EdgeInserter ---

    // Helper merge function for EdgeInserter tests: Union HybridBitset
    fn merge_bitset_union(existing: &HybridBitset, new: HybridBitset) -> Option<HybridBitset> {
        Some(existing | &new) // Use reference for the OR operation
    }

    #[test]
    fn test_ei_try_destination_success_new_edge() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest".to_string())));
        let edge_val: HybridBitset = vec![1].into_iter().collect();


        let inserter = EdgeInserter::new(source.clone(), "key", edge_val.clone(), merge_bitset_union);
        let result_node = inserter.try_destination(dest.clone()).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest));
        let s = source.lock().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, edge_val);
        assert!(Arc::ptr_eq(ca.as_arc(), &dest));
        assert_eq!(dest.lock().unwrap().max_depth, 1); // Depth updated by try_insert
    }

    #[test]
    fn test_ei_try_destination_success_merge_ev() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest".to_string())));
        let initial_edge_val: HybridBitset = vec![10].into_iter().collect();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();
        let merged_edge_val: HybridBitset = vec![1, 10].into_iter().collect();

        // Pre-insert edge
        source.lock().unwrap().try_insert("key", initial_edge_val.clone(), dest.clone()).unwrap();
        assert_eq!(dest.lock().unwrap().max_depth, 1); // Check initial depth

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_node = inserter.try_destination(dest.clone()).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest));
        let s = source.lock().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1); // Still one edge
        let (ca, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, merged_edge_val); // Merged value
        assert!(Arc::ptr_eq(ca.as_arc(), &dest));
        assert_eq!(dest.lock().unwrap().max_depth, 1); // Depth should remain 1
    }

    #[test]
    fn test_ei_try_destination_fail_merge_ev() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest".to_string())));
        // Pre-insert edge with empty HybridBitset
        let initial_edge_val = HybridBitset::new();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        source.lock().unwrap().try_insert("key", initial_edge_val.clone(), dest.clone()).unwrap();

        // In this case, merge_bitset_union will always return Some, so merge should succeed.
        // To test a failing merge, we'd need a different merge function or EV type.
        // Let's repurpose this to test a successful merge where existing is empty.
        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_opt = inserter.try_destination(dest.clone()).into_option();

        assert!(result_opt.is_some()); // Merge succeeded
        let s = source.lock().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        // The result of merge_bitset_union(&empty, &new_edge_val) is new_edge_val
        assert_eq!(*ev, new_edge_val);
        assert!(Arc::ptr_eq(ca.as_arc(), &dest));
    }

    #[test]
    fn test_ei_try_destination_fail_cycle() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest".to_string())));
         let dummy_edge_val = HybridBitset::new();

        // Create cycle manually for test setup
        dest.lock().unwrap().force_insert_to_node("dest_to_src", dummy_edge_val.clone(), &source); // dest -> source edge
        //source.lock().unwrap().force_insert_to_node("src_to_dest", dummy_edge_val.clone(), &dest); // source -> dest edge - this is what we are trying to insert

        // Now try inserting source -> dest again using EdgeInserter
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();
        let inserter = EdgeInserter::new(source.clone(), "src_to_dest", new_edge_val.clone(), merge_bitset_union);
        // This will call try_insert which should detect the cycle
        let result_opt = inserter.try_destination(dest.clone()).into_option();

        assert!(result_opt.is_none()); // Cycle detected, insert failed
    }


    #[test]
    fn test_ei_try_slice_success() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest1".to_string())));
        let dest2: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest2".to_string())));
        let dest3: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest3".to_string())));
        let dummy_edge_val = HybridBitset::new();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: dest2 -> source creates a cycle if we try source -> dest2
        dest2.lock().unwrap().force_insert_to_node("d2->s", dummy_edge_val.clone(), &source);

        let destinations = [dest1.clone(), dest2.clone(), dest3.clone()];

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        // try(dest1) -> OK
        // try(dest2) -> Cycle Error (skipped because dest1 succeeded)
        // try(dest3) -> Skipped
        let result_node = inserter.try_destinations(&destinations).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest1)); // Should succeed with dest1
        let s = source.lock().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(ca.as_arc(), &dest1));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_try_slice_success_later() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest1".to_string())));
        let dest2: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest2".to_string())));
        let dest3: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest3".to_string())));
        let dummy_edge_val = HybridBitset::new();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        // Setup: dest1 -> source creates a cycle if we try source -> dest1
        dest1.lock().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let destinations = [dest1.clone(), dest2.clone(), dest3.clone()];

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        // try(dest1) -> Cycle Error
        // try(dest2) -> OK
        // try(dest3) -> Skipped
        let result_node = inserter.try_destinations(&destinations).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest2)); // Should succeed with dest2
        let s = source.lock().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(ca.as_arc(), &dest2));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_try_slice_fail_all() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest1".to_string())));
        let dest2: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest2".to_string())));
        let dummy_edge_val = HybridBitset::new();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: Both destinations cause cycles
        dest1.lock().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);
        dest2.lock().unwrap().force_insert_to_node("d2->s", dummy_edge_val.clone(), &source);

        let destinations = [dest1.clone(), dest2.clone()];

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_opt = inserter.try_destinations(&destinations).into_option();

        assert!(result_opt.is_none()); // All attempts failed
        assert!(source.lock().unwrap().get(&"key").is_none()); // No edge added
    }

    #[test]
    fn test_ei_try_children_success_merge() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let child1: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("child1".to_string())));
        let child2: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("child2".to_string())));
        let initial_edge_val1: HybridBitset = vec![10].into_iter().collect();
        let initial_edge_val2: HybridBitset = vec![20].into_iter().collect();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();
        let merged_edge_val1: HybridBitset = vec![1, 10].into_iter().collect();

        // Pre-insert edges with the target key
        {
            let mut s = source.lock().unwrap();
            s.try_insert("key", initial_edge_val1.clone(), child1.clone()).unwrap();
            s.try_insert("key", initial_edge_val2.clone(), child2.clone()).unwrap();
        }

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        // Should try child1, merge succeeds. Should not try child2.
        let result_node = inserter.try_children().unwrap();

        assert!(Arc::ptr_eq(&result_node, &child1)); // Merged with child1
        let s = source.lock().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 2); // Still two children

        let edge1_entry = children_map.iter().find(|(ca, _)| Arc::ptr_eq(ca.as_arc(), &child1)).unwrap();
        assert_eq!(*edge1_entry.1, merged_edge_val1); // Merged value

        let edge2_entry = children_map.iter().find(|(ca, _)| Arc::ptr_eq(ca.as_arc(), &child2)).unwrap();
        assert_eq!(*edge2_entry.1, initial_edge_val2); // Unchanged
    }

    #[test]
    fn test_ei_else_create_with_value() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        // No try calls, should go straight to else_create
        let result_node = inserter.else_create_destination_with_value("created".to_string()).unwrap();

        assert_eq!(result_node.lock().unwrap().value, "created");
        assert_eq!(result_node.lock().unwrap().max_depth, 1); // Depth updated
        let s = source.lock().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, new_edge_val);
        assert!(Arc::ptr_eq(ca.as_arc(), &result_node));
    }

    #[test]
    fn test_ei_else_create_with() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let created_flag = Arc::new(AtomicUsize::new(0));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let flag_clone = created_flag.clone();
        let result_node = inserter.else_create_destination_with(|| {
            flag_clone.fetch_add(1, Ordering::SeqCst);
            "created_via_fn".to_string()
        }).unwrap();

        assert_eq!(created_flag.load(Ordering::SeqCst), 1); // Closure was called
        assert_eq!(result_node.lock().unwrap().value, "created_via_fn");
        assert_eq!(result_node.lock().unwrap().max_depth, 1);
    }

    #[test]
    fn test_ei_else_create_default() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        // String::default() is ""
        let result_node = inserter.else_create_destination().unwrap();

        assert_eq!(result_node.lock().unwrap().value, ""); // Default value
        assert_eq!(result_node.lock().unwrap().max_depth, 1);
    }

    #[test]
    fn test_ei_chaining_try_then_else() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest1".to_string())));
        let dummy_edge_val = HybridBitset::new();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: dest1 causes cycle
        dest1.lock().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_node = inserter
            .try_destination(dest1.clone()) // Fails (cycle)
            .else_create_destination_with_value("fallback".to_string()) // Executes
            .unwrap();

        assert_eq!(result_node.lock().unwrap().value, "fallback"); // Fallback was created
        assert!(!Arc::ptr_eq(&result_node, &dest1));
        let s = source.lock().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(ca.as_arc(), &result_node));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_chaining_try_success_skips_else() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest1".to_string())));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_node = inserter
            .try_destination(dest1.clone()) // Succeeds
            .else_create_destination_with_value("fallback".to_string()) // Should be skipped
            .unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest1)); // Original dest1 was used
        assert_eq!(result_node.lock().unwrap().value, "dest1");
        let s = source.lock().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(ca.as_arc(), &dest1));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    #[should_panic(expected = "EdgeInserter::unwrap() called but no destination was found or created")]
    fn test_ei_unwrap_panic() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest1".to_string())));
        let dummy_edge_val = HybridBitset::new();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: dest1 causes cycle
        dest1.lock().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        // Try fails, no else_create called
        inserter.try_destination(dest1.clone()).unwrap(); // Panic here
    }

    #[test]
    fn test_ei_get() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("dest1".to_string())));
        let dummy_edge_val = HybridBitset::new();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: dest1 causes cycle
        dest1.lock().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);

        // Try fails
        let inserter_after_try = inserter.try_destination(dest1.clone());
        assert!(inserter_after_try.clone_into_option().is_none());

        // Now use else_create
        let inserter_after_else = inserter_after_try.else_create_destination_with_value("fallback".to_string());
        let result_opt = inserter_after_else.into_option();
        assert!(result_opt.is_some());
        assert_eq!(result_opt.unwrap().lock().unwrap().value, "fallback");
    }

    #[test]
    fn test_ei_chaining_stops_after_success() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let child1: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("child1".to_string()))); // This one succeeds
        let child2: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("child2".to_string())));
        let new_node_val_if_created = "new_node_val".to_string();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let destinations_for_slice = vec![child2.clone()];

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_node = inserter
            .try_destination(child1.clone()) // This succeeds, result is set to child1
            // try_slice, else_create_with_value should now have no effect
            .try_destinations(&destinations_for_slice) // Should be skipped
            .else_create_destination_with_value(new_node_val_if_created.clone()) // Should be skipped
            .unwrap();

        assert!(Arc::ptr_eq(&result_node, &child1), "Chain should stop after first success (try_insert)");

        // Check only the edge to child1 was added
        let s = source.lock().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(ca.as_arc(), &child1));
        assert_eq!(*ev, new_edge_val);

        // Ensure the value for the skipped else_create was not used
        assert_ne!(result_node.lock().unwrap().value, new_node_val_if_created);
    }

     #[test]
    fn test_ei_try_children() {
        let source: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source".to_string())));
        let child1: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("child1".to_string())));
        let child2: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("child2".to_string())));
        let initial_edge_val1: HybridBitset = vec![10].into_iter().collect();
        let initial_edge_val2: HybridBitset = vec![20].into_iter().collect();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();
        let dummy_edge_val = HybridBitset::new();


        // Add child1 and child2 as children of source under different keys initially
        {
            let mut s = source.lock().unwrap();
            s.try_insert("other_key_c1", initial_edge_val1.clone(), child1.clone()).unwrap();
            s.try_insert("other_key_c2", initial_edge_val2.clone(), child2.clone()).unwrap();
        }

        // Now use EdgeInserter from source to try inserting an edge *to* its *existing children* under a NEW key
        let inserter = EdgeInserter::new(source.clone(), "new_edge_key", new_edge_val.clone(), merge_bitset_union);
        // try_children will look for existing children of 'source' under 'new_edge_key', find none, then try
        // inserting to child1 and child2 using try_slice. Assuming no cycles exist *now* to child1 or child2
        // via the new 'new_edge_key', the first attempt (to child1) should succeed.
        let result_node = inserter.try_children().unwrap();

        assert!(Arc::ptr_eq(&result_node, &child1), "try_children should succeed with the first child it tries if no cycle");

        // Check the new edge was added to source, pointing to child1
        let s = source.lock().unwrap();
        let children_map_new_key = s.get(&"new_edge_key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map_new_key.len(), 1);
        let (ca, ev) = children_map_new_key.iter().next().unwrap();
        assert!(Arc::ptr_eq(ca.as_arc(), &child1));
        assert_eq!(*ev, new_edge_val);


        // Check original edges are still there
        assert_eq!(s.get(&"other_key_c1").unwrap().len(), 1);
         assert_eq!(s.get(&"other_key_c2").unwrap().len(), 1);

        // Test with fallback: Try children (should fail if cycles exist), then create a new node
         let source_for_fb: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("source_fb".to_string())));
         let child1_fb: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("child1_fb".to_string())));
         let child2_fb: TestNodeEI = Arc::new(Mutex::new(TestTrieEI::new("child2_fb".to_string())));

         // Make try_children fail by creating cycles
         {
            let mut s = source_for_fb.lock().unwrap();
            s.try_insert("fb_key_c1", initial_edge_val1.clone(), child1_fb.clone()).unwrap(); // Link source -> child1
            child1_fb.lock().unwrap().force_insert_to_node("c1->s", dummy_edge_val.clone(), &source_for_fb); // Link child1 -> source (cycle)

            s.try_insert("fb_key_c2", initial_edge_val2.clone(), child2_fb.clone()).unwrap(); // Link source -> child2
            child2_fb.lock().unwrap().force_insert_to_node("c2->s", dummy_edge_val.clone(), &source_for_fb); // Link child2 -> source (cycle)
         }

        let new_node_val_if_created = "fallback_node".to_string();
        let fallback_edge_val: HybridBitset = vec![99].into_iter().collect();
         let inserter_fb = EdgeInserter::new(source_for_fb.clone(), "fallback_key", fallback_edge_val.clone(), merge_bitset_union);
         let result_node_fb = inserter_fb
            .try_children() // Try inserting to child1_fb or child2_fb, both should fail cycle detection
            .else_create_destination_with_value(new_node_val_if_created.clone()) // Succeeds
            .unwrap();

        assert_eq!(result_node_fb.lock().unwrap().value, new_node_val_if_created); // Fallback node was created
        let source_guard_fb = source_for_fb.lock().unwrap();
        let children_map_fb = source_guard_fb.get(&"fallback_key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map_fb.len(), 1); // New edge created
        let (ca_fb, ev_fb) = children_map_fb.iter().next().unwrap();
        assert!(Arc::ptr_eq(ca_fb.as_arc(), &result_node_fb)); // Edge points to new node
        assert_eq!(*ev_fb, fallback_edge_val);
     }
}
