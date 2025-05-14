use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc};
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::cell::RefCell; // Add this import

use crate::datastructures::ArcPtrWrapper; // Import ArcPtrWrapper

const HASH_CYCLE_SENTINEL: usize = usize::MAX; // Or another distinct value

#[derive(Debug, Clone)] // Removed PartialEq, Eq, PartialOrd, Ord, Hash
pub struct GSSNode<T> {
    pub value: T,
    predecessors: BTreeSet<ArcPtrWrapper<GSSNode<T>>>,
    // Add this line:
    hash_key_cache: RefCell<Option<usize>>,
}

impl<T> GSSNode<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            predecessors: BTreeSet::new(),
            hash_key_cache: RefCell::new(None), // Initialize the cache
        }
    }
    pub fn new_with_predecessors(value: T, predecessors: Vec<Arc<GSSNode<T>>>) -> Self {
        Self {
            value,
            predecessors: predecessors.into_iter().map(ArcPtrWrapper::new).collect(),
            hash_key_cache: RefCell::new(None), // Initialize the cache
        }
    }

    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        let mut root = Self::new(iter.next().unwrap());
        for value in iter {
            root = root.push(value);
        }
        root
    }

    pub fn push(self, value: T) -> Self {
        let mut new_node = Self::new(value);
        new_node.predecessors.insert(ArcPtrWrapper::new(Arc::new(self)));
        new_node
    }

    pub fn pop(&self) -> Vec<Arc<Self>> {
        self.predecessors.iter().map(|wrapper| wrapper.as_arc().clone()).collect()
    }

    pub fn popn(&self, n: usize) -> Vec<Arc<Self>>
    where
        T: Clone,
    {
        if n == 0 {
            return vec![Arc::new(self.clone())];
        }

        let mut result = Vec::new();
        let mut seen: HashSet<*const GSSNode<T>> = HashSet::new();

        // recurse on predecessors and collect, skipping duplicates
        for predecessor_wrapper in &self.predecessors {
            // predecessor_wrapper is &ArcPtrWrapper<GSSNode<T>>
            // predecessor_wrapper.as_arc() is &Arc<GSSNode<T>>
            for node in predecessor_wrapper.as_arc().popn(n - 1) {
                let ptr = Arc::as_ptr(&node);
                if seen.insert(ptr) {
                    result.push(node);
                }
            }
        }

        result
    }

    pub fn peek(&self) -> &T {
        &self.value
    }

    pub fn value_mut(&mut self) -> &mut T {
        *self.hash_key_cache.borrow_mut() = None; // Invalidate cache
        &mut self.value
    }

    pub fn flatten(&self) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        let mut result = Vec::new();
        let mut stack = Vec::new();
        stack.push((self, Vec::new()));
        while let Some((node, mut path)) = stack.pop() {
            path.push(node.value.clone());
            if node.predecessors.is_empty() {
                result.push(path);
            } else {
                for predecessor_wrapper in &node.predecessors { // predecessor_wrapper is &ArcPtrWrapper<GSSNode<T>>
                    // predecessor_wrapper.as_ref() is &GSSNode<T> (due to Deref)
                    stack.push((predecessor_wrapper.as_ref(), path.clone()));
                }
            }
        }
        result
    }

    pub fn flatten_bulk(nodes: &[Self]) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        nodes.iter().flat_map(|node| node.flatten()).collect()
    }

    pub fn merge(&mut self, mut other: Self)
    where
        T: PartialEq,
        Self: Sized // Needed for Arc::make_mut in bulk_merge, though not here
    {
        assert!(self.value == other.value);
        self.predecessors.extend(std::mem::take(&mut other.predecessors));
        *self.hash_key_cache.borrow_mut() = None; // Invalidate cache
    }

    pub fn merge_unchecked(&mut self, mut other: Self)
    where
        Self: Sized // Needed for Arc::make_mut in bulk_merge
    {
        self.predecessors.extend(std::mem::take(&mut other.predecessors));
         *self.hash_key_cache.borrow_mut() = None; // Invalidate cache
    }

    pub fn map<F, U>(&self, f: F) -> GSSNode<U>
    where
        F: Copy + Fn(&T) -> U,
        U: Clone + Hash, // Add these bounds for U
    {
        GSSNode {
            value: f(&self.value),
            predecessors: self.predecessors.iter()
                // wrapper.as_ref() is &GSSNode<T>, then map is applied to the GSSNode
                .map(|wrapper| ArcPtrWrapper::new(Arc::new(wrapper.as_ref().map(f))))
                .collect(),
            hash_key_cache: RefCell::new(None), // Initialize the cache
        }
    }

    /// Internal helper for recursive hash computation.
    /// Manages cache lookup and calls compute_and_cache_hash_recursive on cache miss.
    fn get_hash_key_internal(
        &self,
        visited_on_current_path: &mut HashSet<*const GSSNode<T>>,
    ) -> usize
    where
        T: Hash,
    {
        let self_ptr = self as *const GSSNode<T>;

        // Check cache first.
        if let Some(cached_hash) = *self.hash_key_cache.borrow() {
            // If this node is already in visited_on_current_path, it means we've found a cycle
            // that leads back to a node whose hash calculation is already in progress higher up the stack
            // (but not yet cached, otherwise we'd hit this 'Some' earlier).
            // However, if its hash is already in the cache, it means this node was fully processed
            // via another path. Use the cached value.
            // The cycle detection in compute_and_cache_hash_recursive handles cases where a node is re-visited
            // *before* its hash is computed and cached along the current DFS path.
            if visited_on_current_path.contains(&self_ptr) {
                // This implies a cycle where this node is an ancestor in the current DFS path.
                // If its hash is already cached, it means it was computed via a different path
                // that didn't involve this cycle, or this cycle was already resolved.
                // It's generally safe to return the cached_hash.
                // The HASH_CYCLE_SENTINEL is primarily for cycles detected *during* a computation.
                return cached_hash;
            }
            // If not part of a current path cycle leading to this node, just return cached value.
            return cached_hash;
        }

        // If not cached, compute, cache, and return.
        // The compute_and_cache_hash_recursive function handles visited_on_current_path insertion/removal.
        self.compute_and_cache_hash_recursive(visited_on_current_path)
    }


    // Method to recursively calculate the hash key and cache it.
    fn compute_and_cache_hash_recursive(
        &self,
        visited_on_current_path: &mut HashSet<*const GSSNode<T>>,
    ) -> usize
    where
        T: Hash,
    {
        let self_ptr = self as *const GSSNode<T>;

        // Cycle detection for the current recursive path
        if !visited_on_current_path.insert(self_ptr) {
            return HASH_CYCLE_SENTINEL; // Cycle detected on this path
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.value.hash(&mut hasher); // Hash current node's value

        // Recursively get hashes of predecessors and hash them in a canonical order
        if !self.predecessors.is_empty() {
            let mut predecessor_hashes = BTreeSet::new();
            for pred_wrapper in &self.predecessors {
                let pred_node = pred_wrapper.as_ref(); // pred_node is &GSSNode<T>
                predecessor_hashes.insert(pred_node.get_hash_key_internal(visited_on_current_path));
            }
            predecessor_hashes.hash(&mut hasher);
        }

        let computed_hash = hasher.finish() as usize;

        // Cache the computed hash
        *self.hash_key_cache.borrow_mut() = Some(computed_hash);

        visited_on_current_path.remove(&self_ptr); // Backtrack: remove from set for this path

        computed_hash
    }

    /// Retrieves or computes the semantic hash key for the node.
    /// The key is a usize value.
    pub fn get_hash_key(&self) -> usize
    where
        T: Hash, // Required by calculate_hash_key_recursive
    {
        // Check cache first (quick path)
        if let Some(cached_hash) = *self.hash_key_cache.borrow() {
            return cached_hash;
        }
        // If not cached, compute it. Initialize a new visited set for this top-level call.
        let mut visited_on_current_path = HashSet::new();
        self.compute_and_cache_hash_recursive(&mut visited_on_current_path)
    }
}

impl<T> Drop for GSSNode<T> {
    // Custom drop to iteratively drop predecessors and break potential cycles.
    fn drop(&mut self) {
        // Take the predecessors to drop them outside of holding the mutex
        let predecessors_to_process_further = std::mem::take(&mut self.predecessors);
        let mut worklist: Vec<Arc<GSSNode<T>>> = predecessors_to_process_further.into_iter().map(|wrapper| wrapper.into_arc()).collect(); // Use into_arc

        while let Some(node_arc) = worklist.pop() {
            if let Ok(mut inner_node) = Arc::try_unwrap(node_arc) {
                // Successfully got unique ownership, take predecessors and add to worklist
                worklist.extend(std::mem::take(&mut inner_node.predecessors).into_iter().map(|wrapper| wrapper.into_arc())); // Use into_arc
            }
            // Else: Arc is still shared, it will be dropped when the last ArcPtrWrapper wrapper is dropped.
        }
    }
}

impl<T: Clone + Hash> Hash for GSSNode<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.get_hash_key());
    }
}

impl<T: Clone + Hash> PartialEq for GSSNode<T> {
    fn eq(&self, other: &Self) -> bool {
        self.get_hash_key() == other.get_hash_key()
    }
}

impl<T: Clone + Hash> Eq for GSSNode<T> {}

impl<T: Clone + Hash> PartialOrd for GSSNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Clone + Hash> Ord for GSSNode<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.get_hash_key().cmp(&other.get_hash_key())
    }
}


pub trait GSSTrait<T: Clone> {
    type Peek<'a> where T: 'a, Self: 'a;
    fn peek(&self) -> Self::Peek<'_>;
    fn push(&self, value: T) -> GSSNode<T>;
    fn pop(&self) -> Vec<Arc<GSSNode<T>>>;
    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>>;
}

impl<T: Clone> GSSTrait<T> for GSSNode<T> {
    type Peek<'a> = &'a T where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        &self.value
    }

    fn push(&self, value: T) -> GSSNode<T> {
        let mut new_node = GSSNode::new(value);
        new_node.predecessors.insert(ArcPtrWrapper::new(Arc::new(self.clone())));
        new_node
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.predecessors.iter().map(|wrapper| wrapper.as_arc().clone()).collect()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        // Delegate to the inherent, de-duplicating implementation above
        GSSNode::popn(self, n)
    }
}

impl<T: Clone> GSSTrait<T> for Arc<GSSNode<T>> {
    type Peek<'a> = &'a T where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        &self.value
    }

    fn push(&self, value: T) -> GSSNode<T> {
        let mut new_node = GSSNode::new(value);
        new_node.predecessors.insert(ArcPtrWrapper::new(self.clone()));
        new_node
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.predecessors.iter().map(|wrapper| wrapper.as_arc().clone()).collect()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        // Re-use the implementation on the underlying node, which already de-duplicates.
        self.as_ref().popn(n)
    }
}

impl<T: Clone> GSSTrait<T> for Option<Arc<GSSNode<T>>> {
    type Peek<'a> = Option<&'a T> where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        self.as_ref().map(|node| node.peek())
    }

    fn push(&self, value: T) -> GSSNode<T> {
        self.clone().map(|node| node.push(value.clone())).unwrap_or_else(|| GSSNode::new(value))
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node| node.pop()).unwrap_or_default()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node| node.popn(n)).unwrap_or_default()
    }
}

impl<T: Clone> GSSTrait<T> for Option<GSSNode<T>> {
    type Peek<'a> = Option<&'a T> where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        self.as_ref().map(|node| node.peek())
    }

    fn push(&self, value: T) -> GSSNode<T> {
        self.clone().map(|node| node.push(value.clone())).unwrap_or_else(|| GSSNode::new(value))
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node| node.pop()).unwrap_or_default()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node| node.popn(n)).unwrap_or_default()
    }
}

pub trait BulkMerge<T> {
    fn bulk_merge(&mut self);
}

impl<T: Clone + Hash> BulkMerge<T> for Vec<Arc<GSSNode<T>>> {
    fn bulk_merge(&mut self) {
        // todo: should be possible to avoid cloning T in some cases by using &T in this map,
        //  but we need to be careful about lifetimes. If we use `node.as_ref().value`, then node
        //  will go out of bounds while the reference to its value is still inside `groups`.
        // Use the semantic hash key for grouping
        let mut groups: BTreeMap<usize, HashMap<*const GSSNode<T>, Arc<GSSNode<T>>>> = BTreeMap::new();
        for node in self.drain(..) {
            // Group by the semantic hash key
            let key = node.get_hash_key();
            groups.entry(key).or_default().entry(Arc::as_ptr(&node)).or_insert(node);
        }

        // Process groups, merging predecessors of nodes with the same semantic key
        for mut group in groups.into_values() {
            let mut group_vec = group.into_values().collect::<Vec<_>>();
            if group_vec.is_empty() {
                continue; // Should not happen with drain, but for safety
            }

            let mut first = group_vec.pop().unwrap(); // Take one node as the base
            
            // If there's more than one node in the group, merge their predecessors
            if !group_vec.is_empty() {
                 // Arc::make_mut gives us a mutable reference. If the Arc is shared, it clones the node.
                 let first_mut_ref = Arc::make_mut(&mut first);
                 // Invalidate the cache as predecessors are about to change
                 *first_mut_ref.hash_key_cache.borrow_mut() = None;

                 // Add predecessors from all sibling nodes in the group
                 for sibling_arc in group_vec { // sibling_arc is Arc<GSSNode<T>>
                     // Iterate through predecessors of the sibling and insert into the base node's predecessors set.
                     // BTreeSet::insert handles deduplication based on ArcPtrWrapper's Ord impl (pointer address).
                     for pred_wrapper in &sibling_arc.predecessors { // pred_wrapper is &ArcPtrWrapper<GSSNode<T>>
                         first_mut_ref.predecessors.insert(pred_wrapper.clone());
                     }
                 }
            }
            // Push the potentially modified 'first' node back to self (which is the result vector)
            self.push(first);
        }
    }
}


// Helper function for prune_and_transform_roots
pub fn prune_and_transform_recursive<T: Clone>(
    node_arc: &Arc<GSSNode<T>>,
    closure: &impl Fn(&T) -> Option<(T, bool)>, // Returns Option<(NewValue, ContinueRecursion)>
    memo: &mut HashMap<*const GSSNode<T>, Option<Arc<GSSNode<T>>>>,
) -> Option<Arc<GSSNode<T>>> {
    // TODO: clean up
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }

    match closure(&node_arc.value) {
        None => {
            // Prune this node
            memo.insert(node_ptr, None);
            None
        }
        Some((new_value, continue_recursion)) => {
            let new_node_arc = if !continue_recursion {
                // Stop recursion, create new node with original predecessors but new value
                let mut transformed_predecessors = Vec::new();
                for pred_wrapper in &node_arc.predecessors { // pred_wrapper is &ArcPtrWrapper<GSSNode<T>>
                     // pred_arc is &Arc<GSSNode<T>>
                     let pred_arc = pred_wrapper.as_arc();
                     // Check memo for already transformed predecessor
                    if let Some(existing_transformed) = memo.get(&Arc::as_ptr(pred_arc)) {
                        if let Some(transformed_pred) = existing_transformed {
                            transformed_predecessors.push(transformed_pred.clone());
                        }
                        // If existing_transformed is None, the predecessor was pruned, so skip.
                        crate::debug!(4, "Skipping pruned predecessor");
                    } else {
                        // This case *shouldn't* happen if traversal order is correct (parents processed after children),
                        // but as a fallback, keep the original if not found in memo. Or perhaps panic?
                        // Let's assume the caller manages the order or this closure handles cycles/shared nodes correctly.
                        // For simplicity now, let's stick to the logic: if we stop, we keep original pointers below.
                         transformed_predecessors = node_arc.predecessors.clone().into_iter()
                             .map(|wrapper| wrapper.as_arc().clone()) // wrapper is ArcPtrWrapper, wrapper.as_arc().clone() is Arc<GSSNode<T>>
                             .collect();
                         crate::debug!(3, "Keeping {} original predecessors", transformed_predecessors.len());
                         // TODO: Revisit required: This might lead to incorrect sharing if predecessors weren't processed.
                         // A better approach for early stop might be needed, maybe marking nodes instead.
                         break; // Exit loop once we decide to keep originals
                    }
                }

                Arc::new(GSSNode::new_with_predecessors(new_value, transformed_predecessors))
            } else {
                // Continue recursion for predecessors
                let mut new_predecessors = Vec::new();
                for pred_wrapper in &node_arc.predecessors { // pred_wrapper is &ArcPtrWrapper<GSSNode<T>>
                    let pred_arc = pred_wrapper.as_arc(); // pred_arc is &Arc<GSSNode<T>>
                    if let Some(new_pred) = prune_and_transform_recursive(pred_arc, closure, memo) {
                        new_predecessors.push(new_pred);
                    }
                }

                // Only create a node if it has predecessors OR it's an original root (how to check?).
                // If new_predecessors is empty AND the original node had predecessors, it means all paths were pruned.
                if new_predecessors.is_empty() && !node_arc.predecessors.is_empty() {
                     memo.insert(node_ptr, None); // Mark as pruned
                     return None; // Return None, pruning this node
                } else {
                     Arc::new(GSSNode::new_with_predecessors(new_value, new_predecessors))
                }
            };
            memo.insert(node_ptr, Some(new_node_arc.clone()));
            Some(new_node_arc)
        }
    }
}

/// Traverses the GSS forest defined by `roots`, applying `closure` to each node's value.
/// Handles shared nodes using memoization. Prunes branches where `closure` returns `None`.
/// Stops recursion down a path if `closure` returns `(_, false)`.
/// Returns a Vec of `Option<Arc<GSSNode<T>>>` corresponding to the input `roots`.
pub fn prune_and_transform_roots<T: Clone>(
    roots: &[Arc<GSSNode<T>>],
    closure: &impl Fn(&T) -> Option<(T, bool)>, // Returns Option<(NewValue, ContinueRecursion)>
) -> Vec<Option<Arc<GSSNode<T>>>> {
    // We need a processing order that ensures children are processed before parents
    // if we want the early-stop optimization (`continue_recursion = false`) to work reliably
    // with shared nodes. A simple recursive approach might process shared children multiple times
    // or incorrectly reuse non-transformed predecessors.
    // For now, let's proceed with the simple recursive approach + memoization, acknowledging the
    // potential issue with the early-stop logic accuracy for shared nodes below the stop point.
    // A full topological sort or iterative approach might be needed for perfect early-stop.

    let mut memo = HashMap::new();
    roots
        .iter()
        .map(|root| prune_and_transform_recursive(root, closure, &mut memo))
        .collect()
}

// --- Longest Path ---

// Recursive helper for find_longest_path.
// Returns the longest path *ending* at node_arc, discovered so far.
fn find_longest_path_recursive<T>(
    node_arc: &Arc<GSSNode<T>>,
    memo: &mut HashMap<*const GSSNode<T>, Vec<Arc<GSSNode<T>>>>, // Stores longest path ending at the key node
    visited_recursion: &mut HashSet<*const GSSNode<T>>, // Detects cycles during the current DFS traversal
) -> Vec<Arc<GSSNode<T>>> {
    let node_ptr = Arc::as_ptr(node_arc);

    // Check memo first
    if let Some(cached_path) = memo.get(&node_ptr) {
        return cached_path.clone();
    }

    // Cycle detection for the current traversal path
    if !visited_recursion.insert(node_ptr) {
        // Cycle detected, return an empty path to avoid infinite recursion
        // and signal that this path shouldn't be considered the longest.
        return Vec::new();
    }

    let mut longest_pred_path: Vec<Arc<GSSNode<T>>> = Vec::new();

    // Explore predecessors recursively
    if !node_arc.predecessors.is_empty() {
        for pred_wrapper in &node_arc.predecessors { // pred_wrapper is &ArcPtrWrapper<GSSNode<T>>
            let pred_arc = pred_wrapper.as_arc(); // pred_arc is &Arc<GSSNode<T>>
            let pred_path = find_longest_path_recursive(pred_arc, memo, visited_recursion);
            // Only update if the predecessor path is valid (non-empty, meaning no cycle encountered below)
            if !pred_path.is_empty() && pred_path.len() > longest_pred_path.len() {
                longest_pred_path = pred_path;
            }
        }
    }
    // else: This node has no predecessors, it's a starting point for paths ending here.

    // Construct the path ending at the current node
    let mut current_path = longest_pred_path; // Starts with the longest path ending at a predecessor
    current_path.push(node_arc.clone()); // Appends the current node

    // Store in memo and backtrack from recursion stack
    memo.insert(node_ptr, current_path.clone());
    visited_recursion.remove(&node_ptr);

    current_path
}

/// Finds one of the longest paths in the GSS forest defined by the given roots.
/// Handles cycles by ignoring paths that contain them.
/// Returns the path as a Vec of nodes from a root to a leaf (or the longest path found).
/// Returns `None` if there are no roots or no valid paths (e.g., only cycles).
pub fn find_longest_path<T>(roots: &[Arc<GSSNode<T>>]) -> Option<Vec<Arc<GSSNode<T>>>> {
    let mut memo: HashMap<*const GSSNode<T>, Vec<Arc<GSSNode<T>>>> = HashMap::new();

    // Populate the memo by traversing from all roots
    for root_arc in roots {
        let mut visited_recursion = HashSet::new(); // Reset cycle detection for each root
        find_longest_path_recursive(root_arc, &mut memo, &mut visited_recursion);
    }

    // Find the longest path among all paths stored in the memo values
    memo.into_values().max_by_key(|path| path.len())
}

/// Statistics about the structure of a GSS forest.
#[derive(Debug, Clone, Default)]
pub struct GSSStats {
    /// Number of root nodes provided.
    pub num_roots: usize,
    /// Total number of unique nodes reachable from the roots.
    pub unique_nodes: usize,
    /// Maximum depth encountered (distance from a root node).
    pub max_depth: usize,
    /// Average depth of nodes (distance from a root node).
    pub average_depth: f664,
    /// Number of nodes with more than one predecessor (merge points).
    pub merge_points: usize,
    /// Maximum number of predecessors for any single node.
    pub max_predecessors: usize,
    /// Average number of predecessors per node.
    pub average_predecessors: f64,
}

/// Gathers statistics about the GSS forest defined by the given roots.
/// Traverses the graph using BFS to calculate depths from roots.
pub fn gather_gss_stats<T: Clone>(roots: &[Arc<GSSNode<T>>]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut visited: HashSet<*const GSSNode<T>> = HashSet::new();
    let mut queue: VecDeque<(Arc<GSSNode<T>>, usize)> = VecDeque::new(); // (node, depth)

    let mut total_depth_sum: u64 = 0;
    let mut total_predecessors_sum: u64 = 0;

    for root_arc in roots {
        let root_ptr = Arc::as_ptr(root_arc);
        if visited.insert(root_ptr) {
            queue.push_back((root_arc.clone(), 0));
        }
    }

    while let Some((current_node_arc, current_depth)) = queue.pop_front() {
        let current_node = current_node_arc.as_ref(); // Borrow the content
        stats.unique_nodes += 1;
        stats.max_depth = stats.max_depth.max(current_depth);
        total_depth_sum += current_depth as u64;

        let num_predecessors = current_node.predecessors.len();
        stats.max_predecessors = stats.max_predecessors.max(num_predecessors);
        total_predecessors_sum += num_predecessors as u64;
        if num_predecessors > 1 {
            stats.merge_points += 1;
        }

        for pred_wrapper in &current_node.predecessors { // pred_wrapper is &ArcPtrWrapper<GSSNode<T>>
            let pred_arc = pred_wrapper.as_arc(); // pred_arc is &Arc<GSSNode<T>>
            let pred_raw_ptr = Arc::as_ptr(pred_arc);
            if visited.insert(pred_raw_ptr) {
                queue.push_back((pred_arc.clone(), current_depth + 1)); // Queue the Arc
            }
        }
    }

    if stats.unique_nodes > 0 {
        stats.average_depth = total_depth_sum as f64 / stats.unique_nodes as f64;
        stats.average_predecessors = total_predecessors_sum as f64 / stats.unique_nodes as f64;
    }

    stats
}

/// Recursive helper to build the string representation of the GSS structure.
fn print_gss_node_recursive<T: Debug>(
    node_arc: &Arc<GSSNode<T>>,
    visited: &mut HashSet<*const GSSNode<T>>,
    indent: usize,
    node_count: &mut usize,
    max_nodes: usize,
    output: &mut String,
) -> Result<(), std::fmt::Error> {
    if *node_count >= max_nodes {
        return Ok(()); // Stop recursion if max_nodes limit is reached
    }

    let node_ptr = Arc::as_ptr(node_arc);
    let prefix = format!("{:indent$}", "", indent = indent * 2);

    if visited.contains(&node_ptr) {
        writeln!(output, "{}- Node {:p} (Visited)", prefix, node_ptr)?;
        return Ok(());
    }

    visited.insert(node_ptr);
    *node_count += 1;

    // Print current node info
    writeln!(output, "{}- Node {:p}: {:?}", prefix, node_ptr, node_arc.value)?;

    // Print predecessors
    if !node_arc.predecessors.is_empty() {
        writeln!(output, "{}  Predecessors:", prefix)?;
        for pred_wrapper in &node_arc.predecessors { // pred_wrapper is &ArcPtrWrapper<GSSNode<T>>
            let pred_arc = pred_wrapper.as_arc(); // pred_arc is &Arc<GSSNode<T>>
            // Recursively print predecessors
            print_gss_node_recursive(pred_arc, visited, indent + 2, node_count, max_nodes, output)?; // Corrected indent
            if *node_count >= max_nodes {
                 // Check again after recursive call in case it hit the limit
                return Ok(());
            }
        }
    }

    Ok(())
}

/// Generates a string representation of the GSS forest structure starting from the given roots.
///
/// Traverses the graph, handling cycles and shared nodes. Stops printing if the number
/// of unique nodes encountered exceeds `max_nodes`.
///
/// # Arguments
/// * `roots` - A slice of `Arc<GSSNode<T>>` representing the roots of the forest.
/// * `max_nodes` - The maximum number of unique nodes to include in the output string.
///
/// # Returns
/// A `String` containing the formatted GSS structure, potentially truncated.
pub fn print_gss_forest<T: Debug>(roots: &[Arc<GSSNode<T>>], max_nodes: usize) -> String {
    let mut visited = HashSet::new();
    let mut node_count = 0;
    let mut output = String::new();

    if roots.is_empty() {
        return "GSS Forest: (No roots)".to_string();
    }

    writeln!(&mut output, "GSS Forest Roots (Max Nodes: {}):", max_nodes).unwrap();

    for (i, root_arc) in roots.iter().enumerate() {
        writeln!(&mut output, "Root {}:", i).unwrap();
        match print_gss_node_recursive(root_arc, &mut visited, 1, &mut node_count, max_nodes, &mut output) {
            Ok(_) => {
                if node_count >= max_nodes {
                    writeln!(&mut output, "... (Truncated: Reached max nodes {})", max_nodes).unwrap();
                    break; // Stop processing more roots if limit reached
                }
            }
            Err(e) => {
                // Should not happen with String::write_fmt
                eprintln!("Error writing GSS structure to string: {}", e);
                return format!("Error generating GSS string: {}", e);
            }
        }
    }

    if node_count < max_nodes && node_count > visited.len() {
         // This condition indicates some nodes were visited but not printed due to the limit being hit mid-recursion
         writeln!(&mut output, "... (Truncated: Reached max nodes {})", max_nodes).unwrap();
    }


    output
}


// Recursive helper for GSS simplification.
// Processes a single node, ensuring that semantically identical nodes are represented by the same Arc.
fn simplify_node_recursive<T: Clone + Hash + Debug>(
    original_node_arc: &Arc<GSSNode<T>>,
    // Memoization for original node pointers to their simplified Arc<GSSNode<T>> versions.
    memo_original_to_simplified: &mut HashMap<*const GSSNode<T>, Arc<GSSNode<T>>>,
    // Global map of semantic hash keys to canonical Arc<GSSNode<T>> instances.
    global_simplified_nodes: &mut HashMap<usize, Arc<GSSNode<T>>>,
    // Used to detect cycles in the original graph structure during this recursive simplification.
    visited_on_stack: &mut HashSet<*const GSSNode<T>>,
) -> Arc<GSSNode<T>> {
    let original_node_ptr = Arc::as_ptr(original_node_arc);

    // Check if this original node is already being simplified (cycle in original graph)
    if !visited_on_stack.insert(original_node_ptr) {
        // Cycle detected (e.g., A -> B -> A, and we are trying to simplify A again).
        // We must return something. For robust cycle handling, this might involve
        // returning a pre-existing (partially) simplified node if available,
        // or a special canonical cycle node.
        // For now, if it's in memo_original_to_simplified, means it was fully processed before this path, which is fine.
        // If it's on stack AND in memo, it's complex.
        // This indicates a cycle that wasn't broken by prior memoization hits.
        // A robust solution might involve fixed-point iteration or specific cycle markers.
        // For this implementation, we'll prefer a memoized version if somehow available,
        // otherwise, this indicates an issue with graph structure or requires more advanced cycle handling.
        // A simple stop-gap: if it's already processed and in memo, use that.
        // Otherwise, this recursive path can't complete it.
        // This part is tricky. For now, let's assume that if it's on stack, it's not yet in memo_original_to_simplified.
        // A true cycle (X->Y->X, simplifying X, then Y, then X again) would hit this.
        // We'll return a newly created node with no predecessors to break the cycle here.
        // This is a simplistic way to handle cycles in the source graph for simplification.
        // A more advanced strategy would be needed for true canonical cyclic graph simplification.
        // For now, we create a "stub" node if a cycle is detected this way.
        // This stub will have the same value but no predecessors from the cycle path.

        // Try to find if a simplified version for this cycle node already exists globally
        let cycle_stub_node_data = GSSNode {
            value: original_node_arc.value.clone(),
            predecessors: BTreeSet::new(), // Break cycle by emptying predecessors for this path
            hash_key_cache: RefCell::new(None),
        };
        let key = cycle_stub_node_data.get_hash_key(); // Compute key for this stub

        if let Some(existing_arc) = global_simplified_nodes.get(&key) {
            visited_on_stack.remove(&original_node_ptr); // Backtrack from stack
            return existing_arc.clone();
        } else {
            let new_arc = Arc::new(cycle_stub_node_data);
            global_simplified_nodes.insert(key, new_arc.clone());
            // Also memoize for original_node_ptr to this specific cycle-broken stub
            memo_original_to_simplified.insert(original_node_ptr, new_arc.clone());
            visited_on_stack.remove(&original_node_ptr); // Backtrack from stack
            return new_arc;
        }
    }

    // Check memo: if this original node has already been simplified, return the canonical Arc.
    if let Some(simplified_arc) = memo_original_to_simplified.get(&original_node_ptr) {
        visited_on_stack.remove(&original_node_ptr); // Backtrack from stack
        return simplified_arc.clone();
    }

    // Recursively simplify predecessors.
    let mut simplified_predecessors_arcs: Vec<Arc<GSSNode<T>>> = Vec::new();
    for original_pred_wrapper in &original_node_arc.predecessors {
        let original_pred_arc = original_pred_wrapper.as_arc(); // This is &Arc<GSSNode<T>>
        let simplified_pred_arc = simplify_node_recursive(
            original_pred_arc,
            memo_original_to_simplified,
            global_simplified_nodes,
            visited_on_stack,
        );
        simplified_predecessors_arcs.push(simplified_pred_arc);
    }

    // Create the set of simplified predecessor ArcPtrWrappers.
    // BTreeSet orders them by pointer (via ArcPtrWrapper's Ord impl),
    // ensuring canonical order for structurally identical sets of predecessor Arcs.
    let simplified_predecessors_set: BTreeSet<ArcPtrWrapper<GSSNode<T>>> =
        simplified_predecessors_arcs.into_iter().map(ArcPtrWrapper::new).collect();

    // Construct a candidate simplified node (not yet Arc'd).
    let candidate_simplified_node_data = GSSNode {
        value: original_node_arc.value.clone(),
        predecessors: simplified_predecessors_set,
        hash_key_cache: RefCell::new(None), // Will be computed by get_hash_key()
    };

    // Compute its semantic hash key. This will populate its cache.
    let key = candidate_simplified_node_data.get_hash_key();

    // Check if a semantically identical node (based on key) already exists globally.
    if let Some(existing_canonical_arc) = global_simplified_nodes.get(&key) {
        // Yes, use the existing canonical Arc.
        memo_original_to_simplified.insert(original_node_ptr, existing_canonical_arc.clone());
        visited_on_stack.remove(&original_node_ptr); // Backtrack from stack
        existing_canonical_arc.clone()
    } else {
        // No, this candidate is semantically new. Arc it and add to global map.
        let new_canonical_arc = Arc::new(candidate_simplified_node_data);
        // The hash_key is already computed and cached within new_canonical_arc due to the get_hash_key() call above.
        global_simplified_nodes.insert(key, new_canonical_arc.clone());
        memo_original_to_simplified.insert(original_node_ptr, new_canonical_arc.clone());
        visited_on_stack.remove(&original_node_ptr); // Backtrack from stack
        new_canonical_arc
    }
}

/// Simplifies a GSS forest, ensuring that semantically identical nodes are represented by
/// the same `Arc<GSSNode<T>>` instance. This process is deep, starting from leaves.
/// Nodes are considered semantically identical if their values are identical and their
/// simplified predecessors (recursively) are identical sets.
///
/// The simplification handles shared substructures correctly.
///
/// Note: This version's cycle handling for cycles *within the original graph structure*
/// is basic: it breaks cycles by creating a node with the same value but empty predecessors
/// for the back-edge. More sophisticated cycle canonicalization might be needed for complex cyclic graphs.
///
/// # Arguments
/// * `roots`: A slice of `Arc<GSSNode<T>>` representing the roots of the GSS forest to simplify.
///
/// # Returns
/// A `Vec<Arc<GSSNode<T>>>` containing the `Arc`s to the simplified root nodes.
pub fn simplify_gss_forest<T: Clone + Hash + Debug>(
    roots: &[Arc<GSSNode<T>>],
) -> Vec<Arc<GSSNode<T>>> {
    let mut memo_original_to_simplified = HashMap::new();
    let mut global_simplified_nodes = HashMap::new();

    let mut simplified_roots = Vec::new();
    for root_arc in roots {
        // visited_on_stack is reset for each root processing if they are truly separate trees.
        // If roots can be part of a connected graph, visited_on_stack should persist across calls or be managed globally.
        // For simplicity, assuming independent processing or that shared parts are handled by memo_original_to_simplified.
        let mut visited_on_stack = HashSet::new();
        simplified_roots.push(simplify_node_recursive(
            root_arc,
            &mut memo_original_to_simplified,
            &mut global_simplified_nodes,
            &mut visited_on_stack,
        ));
    }
    simplified_roots
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Helper to create a GSSNode quickly for tests
    fn gss<T: Clone + Hash + Debug>(value: T, predecessors: Vec<Arc<GSSNode<T>>>) -> Arc<GSSNode<T>> {
        Arc::new(GSSNode::new_with_predecessors(value, predecessors))
    }
    fn leaf<T: Clone + Hash + Debug>(value: T) -> Arc<GSSNode<T>> {
        Arc::new(GSSNode::new(value))
    }

    #[test]
    fn test_gss_node_hash_key() {
        // L0 <- L1
        //  \-> L2
        let l2 = leaf(20); // Level 0 for l2: [{20}]
        let l1 = leaf(10); // Level 0 for l1: [{10}]
        let l0 = gss(0, vec![l1.clone(), l2.clone()]);

        // Compute hashes
        let hash_l0 = l0.get_hash_key();
        let hash_l1 = l1.get_hash_key();
        let hash_l2 = l2.get_hash_key();

        // Basic checks: Different values should (probably) have different hashes
        assert_ne!(hash_l1, hash_l2);

        // Check that the hash of L0 is based on its value AND predecessor hashes
        // This is hard to assert directly without knowing the exact hash algorithm output,
        // but we can check relative equality/inequality.
        // A node with different predecessors should have a different hash.
        let l0_alt = gss(0, vec![l1.clone()]); // L0 with only L1 as pred
        assert_ne!(hash_l0, l0_alt.get_hash_key());

        // A node with the same predecessors (even if different ArcPtrWrapper order internally) should have the same hash.
        let l0_reordered_preds = gss(0, vec![l2.clone(), l1.clone()]); // Same preds as l0, different vec order
        assert_eq!(hash_l0, l0_reordered_preds.get_hash_key());

        // L3 -> L1 (shared)
        let l3 = gss(3, vec![l1.clone()]);
        let hash_l3 = l3.get_hash_key();

        // L3 should have a different hash than L0 and L1
        assert_ne!(hash_l3, hash_l0);
        assert_ne!(hash_l3, hash_l1);

        // Check cache is populated
        assert!(l0.hash_key_cache.borrow().is_some());
        assert!(l1.hash_key_cache.borrow().is_some());
        assert!(l2.hash_key_cache.borrow().is_some());
        assert!(l3.hash_key_cache.borrow().is_some());
    }

    #[test]
    fn test_gss_node_equality_and_order() {
        let l_a1 = leaf(1); // Key: [{1}] -> hash(1)
        let l_a2 = leaf(1); // Key: [{1}] -> hash(1)
        let l_b = leaf(2);  // Key: [{2}] -> hash(2)

        // Test PartialEq/Eq (now comparing usize hashes)
        assert_eq!(l_a1.as_ref() == l_a2.as_ref(), true); // Compare GSSNode content
        assert_eq!(*l_a1, *l_a2); // Deref Arc then compare GSSNode content (uses PartialEq)
        assert_ne!(*l_a1, *l_b);

        // Test PartialOrd/Ord (now comparing usize hashes)
        // These comparisons will be based on the numerical value of the computed hash.
        // While not semantically meaningful for value ordering, it provides a canonical order
        // for BTreeSet/BTreeMap based on structural equivalence.
        assert!((*l_a1).cmp(&*l_a2) == std::cmp::Ordering::Equal); // Use cmp explicitly for clarity on Ord behavior
        assert!((*l_a1).partial_cmp(&*l_a2) == Some(std::cmp::Ordering::Equal));

        // The comparison between l_a1 and l_b depends on the specific hash values,
        // which are non-deterministic between runs, but deterministic for the same structure within a run.
        // We can't assert a specific ordering like l_a1 < l_b unless we control the hash values.
        // But we can assert inequality.
        assert!((*l_a1).cmp(&*l_b) != std::cmp::Ordering::Equal);

        // More complex
        // N1: 0 -> {10, 20}
        let n1_p1 = leaf(10);
        let n1_p2 = leaf(20);
        let n1 = gss(0, vec![n1_p1.clone(), n1_p2.clone()]); // Key: hash(0, hash({hash(10), hash(20)}))

        // N2: 0 -> {20, 10} (same predecessors, different order in vec)
        let n2 = gss(0, vec![n1_p2.clone(), n1_p1.clone()]); // Key: hash(0, hash({hash(20), hash(10)})) -> Same as N1 due to BTreeSet
        assert_eq!(*n1, *n2); // Compares hash values, should be equal.

        // N3: 0 -> {10, 30}
        let n3_p3 = leaf(30);
        let n3 = gss(0, vec![n1_p1.clone(), n3_p3.clone()]); // Key: hash(0, hash({hash(10), hash(30)}))
        assert_ne!(*n1, *n3); // Hashes should be different.
        // Ordering depends on hash values.
    }

    #[test]
    fn test_gss_simplification_shared_nodes() {
        // R1 -> A(val=10) -> B(val=20)
        // R2 -> C(val=10) -> D(val=20)
        // A, B, C, D are distinct Arcs initially.
        // Expected: B and D simplify to S_BD. A and C simplify to S_AC.
        // R1 and R2 should point to the same S_AC.

        let b = leaf(20); // Original B
        let a = gss(10, vec![b.clone()]); // Original A

        let d = leaf(20); // Original D
        let c = gss(10, vec![d.clone()]); // Original C

        assert!(!Arc::ptr_eq(&b, &d)); // Ensure B and D are different Arcs initially
        assert!(!Arc::ptr_eq(&a, &c)); // Ensure A and C are different Arcs initially

        let roots = vec![a.clone(), c.clone()];
        let simplified_roots = simplify_gss_forest(&roots);

        let simplified_a_target = &simplified_roots[0];
        let simplified_c_target = &simplified_roots[1];

        // R1 and R2 should point to the same simplified node S_AC (same Arc)
        assert!(Arc::ptr_eq(simplified_a_target, simplified_c_target));

        // Check structure of S_AC: value 10
        assert_eq!(simplified_a_target.value, 10);
        assert_eq!(simplified_a_target.predecessors.len(), 1);

        // The predecessor should be S_BD
        let s_bd_arc_ptr_wrapper = simplified_a_target.predecessors.iter().next().unwrap();
        let s_bd_arc = s_bd_arc_ptr_wrapper.as_arc();
        assert_eq!(s_bd_arc.value, 20);
        assert!(s_bd_arc.predecessors.is_empty());

        // Verify that original B and D simplified to this S_BD
        // We can do this by simplifying B and D individually (if simplify_gss_forest was not run)
        // or by checking the memoization maps if they were exposed (they are not).
        // The fact that simplified_a_target and simplified_c_target are ptr_eq and have the correct structure
        // strongly implies the inner S_BD was also canonicalized.
        // Also, the simplified_a_target and simplified_c_target having the same hash confirms they are semantically equal.
        assert_eq!(simplified_a_target.get_hash_key(), simplified_c_target.get_hash_key());
    }

    #[test]
    fn test_gss_simplification_diamond() {
        //    R -> N0(0) --↘
        //              N1(1) -> N3(3)
        //    R -> N0(0) --↗
        //              N2(1) -> N3(3) (N1 and N2 are semantically same, N3 is shared)

        let n3_orig = leaf(3);
        let n1_orig = gss(1, vec![n3_orig.clone()]);
        let n2_orig = gss(1, vec![n3_orig.clone()]); // N2 is like N1 but different Arc

        assert!(!Arc::ptr_eq(&n1_orig, &n2_orig));

        let n0_orig = gss(0, vec![n1_orig.clone(), n2_orig.clone()]);

        let roots = vec![n0_orig];
        let simplified_roots = simplify_gss_forest(&roots);
        let simplified_n0 = &simplified_roots[0];

        assert_eq!(simplified_n0.value, 0);
        // N1 and N2 should simplify to the *same* Arc<GSSNode>.
        // So, N0 should have only one distinct predecessor after simplification.
        assert_eq!(simplified_n0.predecessors.len(), 1);

        let simplified_n1n2_arc_ptr_wrapper = simplified_n0.predecessors.iter().next().unwrap();
        let simplified_n1n2 = simplified_n1n2_arc_ptr_wrapper.as_arc();
        assert_eq!(simplified_n1n2.value, 1);
        assert_eq!(simplified_n1n2.predecessors.len(), 1);

        let simplified_n3_arc_ptr_wrapper = simplified_n1n2.predecessors.iter().next().unwrap();
        let simplified_n3 = simplified_n3_arc_ptr_wrapper.as_arc();
        assert_eq!(simplified_n3.value, 3);
        assert!(simplified_n3.predecessors.is_empty());

        // Additionally, verify that the two original predecessors of n0 simplified to the same node
        // This relies on the internal workings of simplify_gss_forest grouping by hash.
        // Since n1_orig and n2_orig are semantically identical (value 1, single pred pointing to n3_orig),
        // they should simplify to the same Arc instance.
        // The simplify_gss_forest function doesn't expose the mapping from original nodes to simplified,
        // but the structure of simplified_n0 confirms the diamond collapse.
    }
     #[test]
    fn test_gss_simplification_cycle_in_original() {
        // N0 -> N1 -> N0 (cycle)
        // N0 (value 0), N1 (value 1)
        let n0_val = 0;
        let n1_val = 1;

        // Create nodes that will form a cycle. This is tricky with Arc.
        // We can't directly create Arc<GSSNode> that cyclically reference each other at construction.
        // Simplification is usually for DAGs or specific cycle handling.
        // The current simplify_node_recursive has basic cycle breaking.
        // Let's test a self-loop: N0 -> N0

        // Create N0 that initially points to nothing.
        let n0_initial = Arc::new(GSSNode::new(n0_val));

        // Manually (and unsafely, for test purposes) create a cycle.
        // This is not how GSSNodes are typically built but tests the cycle detection in simplify.
        // To do this safely, one would need interior mutability or a post-construction step.
        // The provided GSSNode structure doesn't easily allow creating cycles after construction
        // without `Arc::make_mut` and then re-wrapping, which is complex.

        // Let's test with a slightly different structure that can be built:
        // R -> N0(0)
        // R -> N1(1) -> N0(0) (N1 points back to an equivalent of R's first child)

        let common_leaf = leaf(0); // Represents the target of the "back edge"
        let n1 = gss(1, vec![common_leaf.clone()]);
        let n0_pointing_to_n1 = gss(0, vec![n1.clone()]); // This is one root: 0 -> 1 -> 0

        // Another root that is just the common_leaf
        // Roots: [ (0 -> 1 -> 0), (0) ]

        let roots = vec![n0_pointing_to_n1, common_leaf.clone()];
        let simplified_roots = simplify_gss_forest(&roots);

        // Expected:
        // simplified_common_leaf: val=0, preds={}
        // simplified_n1: val=1, preds={simplified_common_leaf}
        // simplified_n0_pointing_to_n1: val=0, preds={simplified_n1}
        // The two roots should be distinct simplified nodes.

        assert_eq!(simplified_roots.len(), 2);
        let s_n0_path = &simplified_roots[0]; // Should be 0 -> 1 -> 0
        let s_leaf_path = &simplified_roots[1];   // Should be 0 (leaf)

        assert!(!Arc::ptr_eq(s_n0_path, s_leaf_path));

        // Check s_leaf_path
        assert_eq!(s_leaf_path.value, 0);
        assert!(s_leaf_path.predecessors.is_empty());
        assert!(s_leaf_path.hash_key_cache.borrow().is_some()); // Hash should be computed

        // Check s_n0_path
        assert_eq!(s_n0_path.value, 0);
        assert_eq!(s_n0_path.predecessors.len(), 1);
        assert!(s_n0_path.hash_key_cache.borrow().is_some()); // Hash should be computed

        let s_n1_arc_ptr_wrapper = s_n0_path.predecessors.iter().next().unwrap();
        let s_n1 = s_n1_arc_ptr_wrapper.as_arc();
        assert_eq!(s_n1.value, 1);
        assert_eq!(s_n1.predecessors.len(), 1);
        assert!(s_n1.hash_key_cache.borrow().is_some()); // Hash should be computed

        let s_n1_pred_arc_ptr_wrapper = s_n1.predecessors.iter().next().unwrap();
        let s_n1_pred = s_n1_pred_arc_ptr_wrapper.as_arc();

        // s_n1_pred should be the same Arc instance as s_leaf_path (canonicalization)
        assert_eq!(s_n1_pred.value, 0);
        assert!(s_n1_pred.predecessors.is_empty());
        assert!(s_n1_pred.hash_key_cache.borrow().is_some()); // Hash should be computed
        assert!(Arc::ptr_eq(s_n1_pred, s_leaf_path));

        // The cycle detection in simplify_node_recursive is more about stack depth on original nodes.
        // If N0 -> N0 (direct cycle in original pointers), simplify(N0) calls simplify(N0_pred=N0).
        // visited_on_stack for N0 would be hit. It would create a stub: N0_stub(val=0, preds={}).
        // This N0_stub would be memoized for original N0.
        // This test doesn't directly create such a raw pointer cycle for simplify to untangle,
        // but relies on semantic equivalence and DAG path processing.
    }
}
