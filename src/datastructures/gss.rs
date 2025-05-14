use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc};
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use crate::datastructures::ArcPtrWrapper; // Import ArcPtrWrapper

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GSSNode<T> {
    pub value: T,
    predecessors: BTreeSet<ArcPtrWrapper<GSSNode<T>>>,
    cached_deep_hash: Option<Arc<Vec<Vec<T>>>>, // New field
}

impl<T> GSSNode<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            predecessors: BTreeSet::new(),
            cached_deep_hash: None, // Add this line
        }
    }
    pub fn new_with_predecessors(value: T, predecessors: Vec<Arc<GSSNode<T>>>) -> Self {
        Self {
            value,
            predecessors: predecessors.into_iter().map(ArcPtrWrapper::new).collect(),
            cached_deep_hash: None, // Add this line
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
    {
        assert!(self.value == other.value);
        self.predecessors.extend(std::mem::take(&mut other.predecessors));
    }

    pub fn merge_unchecked(&mut self, mut other: Self)
    {
        self.predecessors.extend(std::mem::take(&mut other.predecessors));
    }

    pub fn map<F, U>(&self, f: F) -> GSSNode<U>
    where
        F: Copy + Fn(&T) -> U,
    {
        GSSNode {
            value: f(&self.value),
            predecessors: self.predecessors.iter()
                // wrapper.as_ref() is &GSSNode<T>, then map is applied to the GSSNode
                .map(|wrapper| ArcPtrWrapper::new(Arc::new(wrapper.as_ref().map(f))))
                .collect(),
            cached_deep_hash: None, // Add this line
        }
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

impl<T: Clone + Ord> BulkMerge<T> for Vec<Arc<GSSNode<T>>> {
    fn bulk_merge(&mut self) {
        // todo: should be possible to avoid cloning T in some cases by using &T in this map,
        //  but we need to be careful about lifetimes. If we use `node.as_ref().value`, then node
        //  will go out of bounds while the reference to its value is still inside `groups`.
        let mut groups: BTreeMap<T, HashMap<_, Arc<GSSNode<T>>>> = BTreeMap::new();
        for node in self.drain(..) {
            groups.entry(node.value.clone()).or_default().entry(Arc::as_ptr(&node)).or_insert(node);
        }
        for mut group in groups.into_values() {
            let mut group = group.into_values().collect::<Vec<_>>();
            let mut first = group.pop().unwrap();
            if group.is_empty() {
                self.push(first);
            } else {
                // Arc::make_mut clones the GSSNode if `first` is shared.
                // The new `first_mut_ref` will have its predecessors modified.
                let first_mut_ref = Arc::make_mut(&mut first);
                // The original predecessors of `first` are already in `first_mut_ref.predecessors`.
                // Add predecessors from all siblings.
                // `BTreeSet::insert` handles deduplication based on ArcPtrWrapper's Ord impl (pointer address).
                for sibling_arc in group { // sibling_arc is Arc<GSSNode<T>>
                    for pred_wrapper in &sibling_arc.predecessors { // pred_wrapper is &ArcPtrWrapper<GSSNode<T>>
                        first_mut_ref.predecessors.insert(pred_wrapper.clone());
                    }
                }
                self.push(first);
            }
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
                // If new_predecessors.is_empty() AND the original node had predecessors, it means all paths were pruned.
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
    pub average_depth: f64,
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


// Helper for GSS simplification.
#[derive(Clone)]
enum SimplificationState<T: Clone + Ord + Hash + Eq + Debug> {
    Processing,
    Done(Arc<GSSNode<T>>),
}

/// Recursive helper for GSS simplification.
fn simplify_recursive<T: Clone + Ord + Hash + Eq + Debug>(
    original_node_arc: &Arc<GSSNode<T>>,
    processed_nodes_memo: &mut HashMap<*const GSSNode<T>, SimplificationState<T>>,
    structural_memo: &mut HashMap<(T, Arc<Vec<Vec<T>>>), Arc<GSSNode<T>>>,
) -> Arc<GSSNode<T>> {
    let original_node_ptr = Arc::as_ptr(original_node_arc);

    // 1. Check memo for already processed or currently processing node
    if let Some(state) = processed_nodes_memo.get(&original_node_ptr) {
        match state {
            SimplificationState::Processing => {
                // Cycle detected (a node is an ancestor of itself in the recursion path).
                // This simplification approach relies on "leaves-up" (post-order traversal)
                // and is designed for DAGs. Deep hashing of cyclic structures is non-trivial.
                panic!("Cycle detected during GSS simplification. This method assumes a DAG.");
            }
            SimplificationState::Done(simplified_node_arc) => {
                return simplified_node_arc.clone();
            }
        }
    }

    // Mark current node as "Processing" to detect cycles.
    processed_nodes_memo.insert(original_node_ptr, SimplificationState::Processing);

    // 2. Recursively simplify predecessors
    let mut simplified_predecessors_set = BTreeSet::new();
    for pred_wrapper in original_node_arc.predecessors.iter() {
        let original_pred_arc = pred_wrapper.as_arc();
        let simplified_pred_arc =
            simplify_recursive(original_pred_arc, processed_nodes_memo, structural_memo);
        simplified_predecessors_set.insert(ArcPtrWrapper::new(simplified_pred_arc));
    }

    // 3. Compute the deep hash for the current node based on its value and its simplified predecessors' hashes.
    let current_value = original_node_arc.value.clone();

    // `levels_of_sets` will temporarily store BTreeSets for each level to easily merge values.
    let mut levels_of_sets: Vec<BTreeSet<T>> = Vec::new();

    // Level 0: current node's value
    let mut level0_set = BTreeSet::new();
    level0_set.insert(current_value.clone());
    levels_of_sets.push(level0_set);

    // Aggregate hashes from simplified predecessors
    for simpl_pred_wrapper in &simplified_predecessors_set {
        let simpl_pred_arc = simpl_pred_wrapper.as_arc();

        // Simplified predecessors must have their cached_deep_hash populated by previous recursive calls.
        let pred_cached_hash_arc = simpl_pred_arc.cached_deep_hash.as_ref()
            .expect("Internal error: Simplified predecessor must have its cached_deep_hash populated.");

        let pred_hash_levels_vec_vec_t: &Vec<Vec<T>> = pred_cached_hash_arc.as_ref();

        for (i, pred_level_values_vec) in pred_hash_levels_vec_vec_t.iter().enumerate() {
            // Ensure levels_of_sets has BTreeSet for level i+1 (from predecessor's level i)
            while levels_of_sets.len() <= i + 1 {
                levels_of_sets.push(BTreeSet::new());
            }
            // Add all values from the predecessor's i-th level to current node's (i+1)-th level set.
            levels_of_sets[i + 1].extend(pred_level_values_vec.iter().cloned());
        }
    }

    // Convert Vec<BTreeSet<T>> to Vec<Vec<T>> (where inner Vec<T> are sorted and unique)
    // This is the canonical, hashable representation of the deep hash.
    let current_node_deep_hash_vec_vec_t: Vec<Vec<T>> = levels_of_sets
        .into_iter()
        .map(|set| {
            // BTreeSet iteration is already sorted, so just collect.
            set.into_iter().collect::<Vec<T>>()
        })
        .collect();
    let current_node_deep_hash_arc = Arc::new(current_node_deep_hash_vec_vec_t);

    // 4. Check structural_memo for an existing, structurally identical simplified node.
    // The key combines the node's own value and its computed deep hash.
    let structural_key = (current_value.clone(), current_node_deep_hash_arc.clone());

    let final_simplified_node_arc =
        if let Some(existing_structurally_identical_node) = structural_memo.get(&structural_key) {
            // Found an existing node that is structurally identical. Reuse it.
            existing_structurally_identical_node.clone()
        } else {
            // 5. Create a new simplified GSSNode instance.
            // No structurally identical node found, so create, cache, and return a new one.
            let new_simplified_node = GSSNode {
                value: current_value, // Already cloned for structural_key
                predecessors: simplified_predecessors_set,
                cached_deep_hash: Some(current_node_deep_hash_arc.clone()), // Store the computed hash
            };
            let new_arc = Arc::new(new_simplified_node);

            // Store this new node in structural_memo for potential reuse by other branches.
            structural_memo.insert(structural_key, new_arc.clone());
            new_arc
        };

    // 6. Update processed_nodes_memo to mark this original node as "Done" and store its simplified version.
    processed_nodes_memo.insert(original_node_ptr, SimplificationState::Done(final_simplified_node_arc.clone()));

    // 7. Return the (potentially shared) simplified node.
    final_simplified_node_arc
}


/// Simplifies a GSS forest deeply, starting from leaves and going up.
/// Nodes with identical values and identical deep structural hashes (based on values at each level below)
/// will be deduplicated to share the same `Arc<GSSNode<T>>` instance.
/// The deep hash (a vector of sets of T values by level) is cached in the simplified nodes.
/// Predecessor order is normalized by using `BTreeSet` internally for simplified predecessors.
///
/// This function assumes the GSS structure is a Directed Acyclic Graph (DAG).
/// If cycles are present, it will panic.
///
/// # Arguments
/// * `roots` - A slice of `Arc<GSSNode<T>>` representing the roots of the forest to simplify.
///
/// # Returns
/// A `Vec<Arc<GSSNode<T>>>` containing the simplified root nodes.
///
/// # Type Constraints
/// * `T` must implement `Clone + Ord + Hash + Eq + Debug`.
pub fn simplify_gss_forest_deep<T: Clone + Ord + Hash + Eq + Debug>(
    roots: &[Arc<GSSNode<T>>],
) -> Vec<Arc<GSSNode<T>>> {
    // Memoization table for nodes already processed/simplified (maps original node ptr to its simplified Arc).
    let mut processed_nodes_memo: HashMap<*const GSSNode<T>, SimplificationState<T>> = HashMap::new();

    // Memoization table for structural deduplication (maps (value, deep_hash) to a shared simplified Arc).
    // The deep_hash is an Arc<Vec<Vec<T>>>, where each inner Vec<T> is sorted and represents a set of values at a level.
    let mut structural_memo: HashMap<(T, Arc<Vec<Vec<T>>>), Arc<GSSNode<T>>> = HashMap::new();

    roots
        .iter()
        .map(|root_arc| {
            simplify_recursive(root_arc, &mut processed_nodes_memo, &mut structural_memo)
        })
        .collect()
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
