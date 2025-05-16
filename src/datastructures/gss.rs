use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;

// No longer importing ArcPtrWrapper

#[derive(Debug, Clone)]
pub struct GSSNode<T> {
    pub value: T,
    predecessors: BTreeSet<Arc<GSSNode<T>>>, // Changed to BTreeSet<Arc<GSSNode<T>>>
    hash_key_cache: u64,
}

// Type alias for the canonicalization cache key
// T must be Ord for BTreeSet key.
type NodeCacheKey<T> = (T, BTreeSet<Arc<GSSNode<T>>>);
// Type alias for the canonicalization cache
pub type NodeCache<T> = HashMap<NodeCacheKey<T>, Arc<GSSNode<T>>>;

// Helper function to compute a node's hash.
// This will be used to populate `hash_key_cache`.
// T must be Hash for value.hash(), predecessors Arcs must point to GSSNodes with valid hash_key_cache.
fn compute_node_hash<T: Hash>(value: &T, predecessors: &BTreeSet<Arc<GSSNode<T>>>) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    // The BTreeSet ensures predecessors are iterated in a canonical order (by Arc pointer address, due to Arc's Ord impl).
    // We hash the predecessors' *cached* hashes, not their full content, to avoid infinite recursion
    // and leverage the bottom-up nature of canonicalization.
    for pred_arc in predecessors {
        pred_arc.hash_key_cache.hash(&mut hasher);
    }
    hasher.finish()
}


// Add Clone + Ord + Debug bounds for canonicalization
impl<T: Clone + Ord + Hash + Debug> GSSNode<T> {
    // Removed old new and new_with_predecessors

    // Canonical way to create or retrieve a GSSNode<T> wrapped in an Arc.
    // Ensures that structurally identical nodes (same value, same set of canonical predecessors)
    // are represented by the same Arc instance.
    pub fn get_canonical(
        value: T,
        predecessors: BTreeSet<Arc<Self>>, // Use Arc directly
        cache: &mut NodeCache<T>,
    ) -> Arc<Self> {
        let key = (value, predecessors); // value and predecessors are moved into key

        // Use entry API to avoid double lookup and handle cloning more explicitly
        cache.entry(key)
            .or_insert_with(|| {
                // If not found, key.0 (value) and key.1 (predecessors) are used to create the new node.
                // Then this key is inserted by the entry API.

                // Value for the new node (will be moved from key.0)
                // Predecessors for the new node (will be moved from key.1)
                let node_value_for_struct = key.0.clone();
                let node_predecessors_for_struct = key.1.clone();

                let hash_key_cache = compute_node_hash(&node_value_for_struct, &node_predecessors_for_struct);

                Arc::new(GSSNode {
                    value: node_value_for_struct,
                    predecessors: node_predecessors_for_struct,
                    hash_key_cache,
                })
            })
            .clone() // Return a clone of the Arc from the cache
    }

    // Convenience method to create a canonical root node (a node with no predecessors).
    pub fn new_empty_canonical(value: T, cache: &mut NodeCache<T>) -> Arc<Self> {
        Self::get_canonical(value, BTreeSet::new(), cache)
    }


    // Removed compute_hash_key_cache


    // Modified from_iter to use canonical creation
    pub fn from_iter<I>(iter: I, cache: &mut NodeCache<T>) -> Arc<Self>
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        let mut root = Self::new_empty_canonical(iter.next().unwrap(), cache);
        for value in iter {
            root = Self::push_onto_canonical(root, value, cache);
        }
        root
    }

    // Removed the consuming push method. Use GSSTrait push or push_onto_canonical instead.

    // Static method to push a new value onto an existing canonical node,
    // returning the new canonical node.
    pub fn push_onto_canonical(
        current_stack_top: Arc<Self>,
        value: T,
        cache: &mut NodeCache<T>,
    ) -> Arc<Self> {
        let mut predecessors = BTreeSet::new();
        predecessors.insert(current_stack_top);
        Self::get_canonical(value, predecessors, cache)
    }


    // Modified pop to iterate directly over Arcs
    pub fn pop(&self) -> Vec<Arc<Self>> {
        self.predecessors.iter().cloned().collect()
    }

    // Removed inherent popn. Implementation moved to GSSTrait for Arc<GSSNode<T>>.


    pub fn peek(&self) -> &T {
        &self.value
    }

    // value_mut() is not compatible with Arc<GSSNode<T>> as it requires exclusive access.
    // Removed value_mut()


    // Modified flatten to take Arc<Self> and iterate directly over Arcs
    pub fn flatten(self_arc: Arc<Self>) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        let mut result = Vec::new();
        // Stack stores (node_arc, current_path)
        let mut stack: Vec<(Arc<Self>, Vec<T>)> = Vec::new();
        stack.push((self_arc, Vec::new()));

        let mut visited_paths: HashSet<(Arc<Self>, Vec<T>)> = HashSet::new(); // Optional: helps with cycles if paths repeat

        while let Some((node_arc, mut path)) = stack.pop() {
             // Optional: Cycle detection on paths if needed, but Arc handles node cycles.
             // If you push a path with a cycle back onto the stack, it might loop infinitely
             // unless you track (node, path) combinations or path hashes.
             // For now, basic DFS assumes non-cyclic *paths*, relying on Arc cycle handling for node dropping.

            path.push(node_arc.value.clone());

            if node_arc.predecessors.is_empty() {
                result.push(path);
            } else {
                for pred_arc in &node_arc.predecessors { // pred_arc is &Arc<GSSNode<T>>
                    // Use pred_arc.clone() to put Arc onto the stack
                    stack.push((pred_arc.clone(), path.clone()));
                }
            }
        }
        // The DFS above builds paths in reverse (leaf to root). Reverse them.
        for path in &mut result {
            path.reverse();
        }
        result
    }

    // Modified flatten_bulk to call the new flatten signature
    pub fn flatten_bulk(nodes: &[Arc<Self>]) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        nodes.iter().flat_map(|arc_node| GSSNode::flatten(arc_node.clone())).collect()
    }

    // Removed merge and merge_unchecked. Use merge_canonical instead.

    // Canonical way to merge two GSSNodes (represented by Arcs).
    // Returns a new canonical Arc<GSSNode<T>> that represents the merged node.
    // Assumes node1_arc and node2_arc point to canonical nodes.
    pub fn merge_canonical(
        node1_arc: Arc<Self>,
        node2_arc: Arc<Self>,
        cache: &mut NodeCache<T>,
    ) -> Result<Arc<Self>, &'static str> // Return Result to signal incompatible merge
    where
        T: PartialEq, // Keep this bound for value comparison
    {
        if node1_arc.value != node2_arc.value {
            // Or handle this as an error/panic, depending on desired behavior
            return Err("Cannot merge nodes with different values");
        }
        // The predecessors of node1_arc and node2_arc should already be canonical Arcs.
        // We merge their sets of predecessors. BTreeSet handles duplicates (based on Arc pointer address).
        let mut merged_predecessors = node1_arc.predecessors.clone();
        for pred_arc in &node2_arc.predecessors {
            merged_predecessors.insert(pred_arc.clone());
        }
        // Value can be cloned from either node1_arc or node2_arc
        Ok(Self::get_canonical(node1_arc.value.clone(), merged_predecessors, cache))
    }


    // Modified map to map_canonical, takes Arc<Self> and cache, returns Arc<GSSNode<U>>
    pub fn map_canonical<F, U>(
        self_arc: Arc<Self>, // Takes Arc<Self>
        f: F,
        cache_u: &mut NodeCache<U>, // Cache for GSSNode<U>
    ) -> Arc<GSSNode<U>>
    where
        F: Copy + Fn(&T) -> U,
        // T bounds already on GSSNode impl block: Clone + Ord + Hash + Debug
        U: Clone + Ord + Hash + Debug, // Bounds for the new node type U
    {
        let new_value = f(&self_arc.value);
        let new_predecessors: BTreeSet<Arc<GSSNode<U>>> = self_arc.predecessors.iter()
            .map(|pred_arc_t| {
                // Recursive call to map_canonical for predecessors
                GSSNode::map_canonical(pred_arc_t.clone(), f, cache_u)
            })
            .collect();
        GSSNode::<U>::get_canonical(new_value, new_predecessors, cache_u)
    }

    // Getter for hash_key_cache (useful for external comparisons/debugging)
    pub fn get_hash_key_cache(&self) -> u64 {
        self.hash_key_cache
    }
}

impl<T> Drop for GSSNode<T> {
    // Custom drop to iteratively drop predecessors and break potential cycles.
    fn drop(&mut self) {
        // Take the predecessors to drop them outside of holding the mutex
        // Since predecessors are now BTreeSet<Arc<GSSNode<T>>>, just take the set.
        let predecessors_to_process_further = std::mem::take(&mut self.predecessors);
        let mut worklist: Vec<Arc<GSSNode<T>>> = predecessors_to_process_further.into_iter().collect(); // Directly collect Arcs

        while let Some(node_arc) = worklist.pop() {
            if let Ok(mut inner_node) = Arc::try_unwrap(node_arc) {
                // Successfully got unique ownership, take predecessors and add to worklist
                worklist.extend(std::mem::take(&mut inner_node.predecessors).into_iter()); // Extend with Arcs
            }
            // Else: Arc is still shared, it will be dropped when the last Arc instance is dropped.
            // If a cycle exists, the nodes in the cycle will eventually be dropped when their Arc counts drop to 0
            // outside of this explicit dropping process, relying on Arc's cycle detection mechanisms.
            // The iterative drop helps with trees or DAGs, and doesn't infinite loop on simple cycles.
        }
    }
}

// Using hash_key_cache computed by compute_node_hash which includes value hash.
impl<T: Hash> Hash for GSSNode<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_key_cache.hash(state);
        // Value hashing is included in compute_node_hash
    }
}

// PartialEq requires T: PartialEq. Comparison includes value, hash, and predecessors set.
impl<T: Hash + PartialEq> PartialEq for GSSNode<T> {
    fn eq(&self, other: &Self) -> bool {
        // If hash_key_cache is different, they are likely different.
        // If T: Eq, then value equality is definitive.
        // If T: PartialEq, value equality might be partial.
        // The hash_key_cache should be a strong distinguisher, but we double check value and predecessors.
        // Predecessors set comparison is by Arc pointer (Ord for Arc).
        self.hash_key_cache == other.hash_key_cache && self.value == other.value && self.predecessors == other.predecessors
    }
}

// Eq requires T: Eq and self.predecessors == other.predecessors implies Eq.
// BTreeSet<Arc<T>>::eq requires Arc<T>::eq. Arc<T>::eq requires T::eq and pointer equality.
// Since Arc pointer equality is used by BTreeSet::eq, and our canonicalization ensures
// structural equality implies pointer equality for canonical nodes, this holds.
// Eq requires T: Eq.
impl<T: Hash + Eq> Eq for GSSNode<T> {}


// PartialOrd requires T: PartialOrd. Comparison includes hash, value, and predecessors set.
impl<T: Hash + PartialOrd> PartialOrd for GSSNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // T must be PartialOrd for self.value.partial_cmp
        // T must be Ord for BTreeSet in NodeCacheKey, so T is also PartialOrd.
        match self.hash_key_cache.partial_cmp(&other.hash_key_cache) {
            Some(Ordering::Equal) => {
                match self.value.partial_cmp(&other.value) {
                    Some(Ordering::Equal) => self.predecessors.partial_cmp(&other.predecessors), // Compares sets of Arc pointers
                    other_ordering => other_ordering,
                }
            }
            other_ordering => other_ordering,
        }
    }
}

// Ord requires T: Ord. Comparison includes hash, value, and predecessors set.
impl<T: Hash + Ord> Ord for GSSNode<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.hash_key_cache.cmp(&other.hash_key_cache)
            .then_with(|| self.value.cmp(&other.value))
            .then_with(|| self.predecessors.cmp(&other.predecessors)) // Compares sets of Arc pointers
    }
}


// Updated GSSTrait bounds and push signature
pub trait GSSTrait<T: Clone + Ord + Hash + Debug> {
    type Peek<'a> where T: 'a, Self: 'a;
    fn peek(&self) -> Self::Peek<'_>;
    // Push now takes a cache and returns an Arc<GSSNode<T>>
    fn push(&self, value: T, cache: &mut NodeCache<T>) -> Arc<GSSNode<T>>;
    fn pop(&self) -> Vec<Arc<GSSNode<T>>>;
    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>>;
}

// Removed impl GSSTrait for GSSNode<T>

// Updated impl GSSTrait for Arc<GSSNode<T>> bounds and methods
impl<T: Clone + Ord + Hash + Debug> GSSTrait<T> for Arc<GSSNode<T>> {
    type Peek<'a> = &'a T where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        &self.value
    }

    fn push(&self, value: T, cache: &mut NodeCache<T>) -> Arc<GSSNode<T>> {
        // self here is Arc<GSSNode<T>>
        GSSNode::push_onto_canonical(self.clone(), value, cache)
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.predecessors.iter().cloned().collect()
    }

    // Implemented popn directly here for Arc<GSSNode<T>>
    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        if n == 0 {
            return vec![self.clone()]; // self is Arc<GSSNode<T>>
        }

        let mut result = Vec::new();
        // Use a HashSet to track Arcs added to result in this specific call to popn,
        // to avoid duplicates if multiple paths lead to the same Arc at the same level.
        let mut seen_arcs_for_this_call: HashSet<*const GSSNode<T>> = HashSet::new();

        for predecessor_arc in &self.predecessors {
            // Recursively call popn on the Arc predecessor (which implements GSSTrait)
            for node_arc_from_popn in predecessor_arc.popn(n - 1) {
                let ptr = Arc::as_ptr(&node_arc_from_popn);
                if seen_arcs_for_this_call.insert(ptr) {
                    result.push(node_arc_from_popn);
                }
            }
        }
        result
    }
}

// Updated impl GSSTrait for Option<Arc<GSSNode<T>>> bounds and methods
impl<T: Clone + Ord + Hash + Debug> GSSTrait<T> for Option<Arc<GSSNode<T>>> {
    type Peek<'a> = Option<&'a T> where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        self.as_ref().map(|node| node.peek()) // Calls GSSTrait::peek on Arc
    }

    fn push(&self, value: T, cache: &mut NodeCache<T>) -> Arc<GSSNode<T>> {
        match self {
            Some(arc_node) => arc_node.push(value, cache), // Calls GSSTrait::push on Arc
            None => GSSNode::new_empty_canonical(value, cache),
        }
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node| node.pop()).unwrap_or_default() // Calls GSSTrait::pop on Arc
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node| node.popn(n)).unwrap_or_default() // Calls GSSTrait::popn on Arc
    }
}

// Removed impl GSSTrait for Option<GSSNode<T>>


// Updated BulkMerge trait bounds
pub trait BulkMerge<T: Clone + Ord + Hash + Debug> { // Added Hash + Debug
    fn bulk_merge(&mut self, cache: &mut NodeCache<T>); // Added cache
}

// Updated impl BulkMerge for Vec<Arc<GSSNode<T>>> bounds and method
impl<T: Clone + Ord + Hash + Debug> BulkMerge<T> for Vec<Arc<GSSNode<T>>> {
    fn bulk_merge(&mut self, cache: &mut NodeCache<T>) {
        // Groups nodes by their value. Because we are working with canonical Arcs,
        // multiple Arcs in the input `self` vector with the same value might represent
        // structurally distinct nodes (different predecessor sets). The goal of bulk_merge
        // is to merge nodes that arrived *together* for processing and have the same value,
        // even if they came from different predecessor paths. This is a form of local merge.
        // The canonicalization (`get_canonical`) handles the global merge of structurally
        // identical results from this local merge step.

        let mut groups_by_value: BTreeMap<T, Vec<Arc<GSSNode<T>>>> = BTreeMap::new();
        for node_arc in self.drain(..) {
            groups_by_value.entry(node_arc.value.clone()).or_default().push(node_arc);
        }

        let mut new_merged_nodes = Vec::new();
        for (value, group_arcs) in groups_by_value {
            if group_arcs.is_empty() { continue; }

            // Merge the predecessors from all Arcs in this value group.
            // The resulting set of predecessors for the merged node will be the union
            // of the predecessor sets of all original nodes in this group.
            let mut merged_predecessors: BTreeSet<Arc<GSSNode<T>>> = BTreeSet::new();
            for node_arc_in_group in group_arcs {
                for pred_arc in &node_arc_in_group.predecessors {
                    merged_predecessors.insert(pred_arc.clone());
                }
            }
            // Create a new canonical node with the merged predecessors.
            // get_canonical will handle finding an existing node with this structure
            // or creating a new one and adding it to the cache.
            let merged_node = GSSNode::get_canonical(value, merged_predecessors, cache);
            new_merged_nodes.push(merged_node);
        }
        *self = new_merged_nodes;
    }
}

// Helper function for prune_and_transform_roots - updated to take cache
pub fn prune_and_transform_recursive<T: Clone + Ord + Hash + Debug>( // Added Ord + Debug
    node_arc: &Arc<GSSNode<T>>,
    closure: &impl Fn(&T) -> Option<(T, bool)>, // Returns Option<(NewValue, ContinueRecursion)>
    memo: &mut HashMap<*const GSSNode<T>, Option<Arc<GSSNode<T>>>>,
    cache: &mut NodeCache<T>, // Add cache
) -> Option<Arc<GSSNode<T>>> {
    // TODO: clean up - Keep existing TODO
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
            let mut new_predecessors;
            if continue_recursion {
                // Continue recursion for predecessors
                new_predecessors = BTreeSet::new();
                for pred_arc in &node_arc.predecessors { // pred_arc is &Arc<GSSNode<T>>
                    // Recursive call with the cache
                    if let Some(new_pred) = prune_and_transform_recursive(pred_arc, closure, memo, cache) {
                        new_predecessors.insert(new_pred); // Insert Arc directly
                    }
                }
            } else {
                // Stop recursion, create new node with original predecessors but new value
                // The original predecessors should already be canonical Arcs if the input GSS is canonical.
                new_predecessors = node_arc.predecessors.clone();
            };
            // Create the new node using the canonical method
            let new_node_arc = GSSNode::get_canonical(new_value, new_predecessors, cache);
            memo.insert(node_ptr, Some(new_node_arc.clone()));
            Some(new_node_arc)
        }
    }
}

/// Traverses the GSS forest defined by `roots`, applying `closure` to each node's value.
/// Handles shared nodes using memoization. Prunes branches where `closure` returns `None`.
/// Stops recursion down a path if `closure` returns `(_, false)`.
/// Returns a Vec of `Option<Arc<GSSNode<T>>>` corresponding to the input `roots`.
/// The returned GSS forest is canonicalized.
pub fn prune_and_transform_roots<T: Clone + Ord + Hash + Debug>( // Added Ord + Debug
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
    // Keep existing comment

    let mut memo = HashMap::new();
    let mut cache = NodeCache::new(); // Create cache for this transformation pass
    roots
        .iter()
        .map(|root| prune_and_transform_recursive(root, closure, &mut memo, &mut cache))
        .collect()
}

// --- Longest Path ---

// Recursive helper for find_longest_path.
// Returns the longest path *ending* at node_arc, discovered so far.
// Updated to iterate directly over Arcs.
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
        for pred_arc in &node_arc.predecessors { // pred_arc is &Arc<GSSNode<T>>
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
    pub average_predecessors: f64, // Corrected type from f664
}

/// Gathers statistics about the GSS forest defined by the given roots.
/// Traverses the graph using BFS to calculate depths from roots.
// Updated to iterate directly over Arcs.
pub fn gather_gss_stats<T: Clone>(roots: &[Arc<GSSNode<T>>]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut visited: HashSet<*const GSSNode<T>> = HashSet::new();
    let mut queue: VecDeque<(Arc<GSSNode<T>>, usize)> = VecDeque::new(); // (node_arc, depth)

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

        for pred_arc in &current_node.predecessors { // pred_arc is &Arc<GSSNode<T>>
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
// Updated to iterate directly over Arcs.
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
        for pred_arc in &node_arc.predecessors { // pred_arc is &Arc<GSSNode<T>>
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

// Removed simplify_node_recursive
// Removed simplify_gss_forest


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
    use std::fmt::Debug;
    use std::ptr;


    // Define Mock Types (Keep these as they are needed for tests)
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct MockLLMTokenInfo {
        active: String,
        intersection: String,
    }

    // Manual Debug impl to match log format closely
    impl Debug for MockLLMTokenInfo {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("LLMTokenInfo")
             .field("active", &self.active)
             .field("intersection", &self.intersection)
             .finish()
        }
    }


    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct MockParseStateNodeContent {
        state_id: usize, // Using usize to match StateID(0) etc.
        t: MockLLMTokenInfo,
    }

    // Manual Debug impl to match log format closely
    impl Debug for MockParseStateNodeContent { // Overwrite previous Debug for MockParseStateNodeContent
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!(
                "ParseStateNodeContent {{ state_id: StateID({}), t: {:?} }}",
                self.state_id, self.t
            ))
        }
    }


    // Type alias for GSSNode with MockParseStateNodeContent for brevity in the test
    type MockGSSNode = GSSNode<MockParseStateNodeContent>;

    // Type alias for the cache used in tests
    type MockNodeCache = NodeCache<MockParseStateNodeContent>;


    // Helper to create a *canonical* node for tests
    // Now takes a cache and returns Arc<MockGSSNode>
    fn node_canonical(
        value: MockParseStateNodeContent,
        predecessors: Vec<Arc<MockGSSNode>>,
        cache: &mut MockNodeCache,
    ) -> Arc<MockGSSNode> {
        let pred_set: BTreeSet<Arc<MockGSSNode>> = predecessors.into_iter().collect();
        GSSNode::get_canonical(value, pred_set, cache)
    }


    // Helper to get a stable representation of a GSS node for comparison purposes in tests.
    // Returns (value, Vec<pred_hashes_sorted>)
    type NodeRepr<T> = (T, Vec<u64>);

    fn get_node_repr<T: Clone + Hash>(node_arc: &Arc<GSSNode<T>>) -> NodeRepr<T> { // Added Hash bound
        let mut pred_hashes: Vec<u64> = node_arc.predecessors.iter()
            .map(|p_arc| p_arc.hash_key_cache) // Use hash_key_cache directly from Arc
            .collect();
        pred_hashes.sort_unstable(); // Sort hashes for canonical representation
        (node_arc.value.clone(), pred_hashes)
    }

    // Helper to recursively collect all unique node *pointers* and their representations
    fn collect_all_nodes_recursive<T: Clone + Hash>(
        node_arc: &Arc<GSSNode<T>>,
        visited: &mut HashSet<*const GSSNode<T>>,
        collected_nodes: &mut HashMap<*const GSSNode<T>, NodeRepr<T>>,
    ) {
        let ptr = Arc::as_ptr(node_arc);
        if !visited.insert(ptr) {
            return;
        }
        collected_nodes.insert(ptr, get_node_repr(node_arc));
        for pred_arc in &node_arc.predecessors { // pred_arc is &Arc<GSSNode<T>>
            collect_all_nodes_recursive(pred_arc, visited, collected_nodes);
        }
    }

    // Helper to recursively collect all unique Arcs in a GSS forest
    fn collect_arcs_recursive<T: Clone + Hash>(
        node_arc: &Arc<GSSNode<T>>,
        // Output map: raw pointer to GSSNode -> Arc pointing to that GSSNode
        collected_arcs: &mut HashMap<*const GSSNode<T>, Arc<GSSNode<T>>>,
    ) {
        let ptr = Arc::as_ptr(node_arc);
        if collected_arcs.contains_key(&ptr) {
            return; // Already visited and collected this Arc
        }
        collected_arcs.insert(ptr, node_arc.clone());
        for pred_arc in &node_arc.predecessors { // pred_arc is &Arc<GSSNode<T>>
            collect_arcs_recursive(pred_arc, collected_arcs);
        }
    }


    #[test]
    fn test_gss_canonicalization_basic() {
        // Helper for i32 nodes
        fn node_canonical_i32(
            value: i32,
            predecessors: Vec<Arc<GSSNode<i32>>>,
            cache: &mut NodeCache<i32>,
        ) -> Arc<GSSNode<i32>> {
            let pred_set: BTreeSet<Arc<GSSNode<i32>>> = predecessors.into_iter().collect();
            GSSNode::get_canonical(value, pred_set, cache)
        }

        let mut cache: NodeCache<i32> = HashMap::new();

        // Construct the graph using the canonical constructor
        // D1 (40, [])
        let d1 = node_canonical_i32(40, vec![], &mut cache);
        // C1 (30, [D1])
        let c1 = node_canonical_i32(30, vec![d1.clone()], &mut cache);
        // B1 (20, [C1])
        let b1 = node_canonical_i32(20, vec![c1.clone()], &mut cache);

        // D2 (40, []) - structurally identical to D1
        let d2 = node_canonical_i32(40, vec![], &mut cache);

        // A1 (10, [B1, D2])
        let a1 = node_canonical_i32(10, vec![b1.clone(), d2.clone()], &mut cache);

        // The graph is now built with canonical nodes.
        // The roots are just the entry points, the graph structure is in the Arcs.
        let roots = vec![a1.clone()];

        // Verify structure and hash caching after canonicalization
        // D1 and D2 should be the *same* Arc instance because they are structurally identical leaves.
        assert!(Arc::ptr_eq(&d1, &d2), "Structurally identical D nodes should be canonicalized to 1 Arc instance");

        let mut visited_check = HashSet::new();
        let mut collected_check = HashMap::new();
        collect_all_nodes_recursive(&a1, &mut visited_check, &mut collected_check);

        // We expect 4 unique node *structures* (and thus 4 unique Arcs) in this graph:
        // Value 40 (D nodes)
        // Value 30 (C1 node)
        // Value 20 (B1 node)
        // Value 10 (A1 node)
        assert_eq!(collected_check.len(), 4, "Expected 4 unique node structures (Arc instances) after canonicalization");

        // Find the canonical nodes by value
        let mut canonical_nodes_by_val = HashMap::new();
        let mut all_collected_arcs = HashMap::new();
        collect_arcs_recursive(&a1, &mut all_collected_arcs);

        for s_node_arc in all_collected_arcs.values() {
             canonical_nodes_by_val.entry(s_node_arc.value).or_insert_with(Vec::new).push(s_node_arc.clone());
        }

        let s_d_nodes = canonical_nodes_by_val.get(&40).unwrap();
        assert_eq!(s_d_nodes.len(), 1, "Canonicalization should result in 1 Arc for value 40");
        let s_d_arc = &s_d_nodes[0];
        assert_ne!(s_d_arc.hash_key_cache, 0, "D node hash should be computed");
        assert_eq!(s_d_arc.predecessors.len(), 0, "Canonical D node should have no predecessors");


        let s_c1_nodes = canonical_nodes_by_val.get(&30).unwrap();
        assert_eq!(s_c1_nodes.len(), 1);
        let s_c1_arc = &s_c1_nodes[0];
        assert_ne!(s_c1_arc.hash_key_cache, 0, "C1 node hash should be computed");
        assert_eq!(s_c1_arc.predecessors.len(), 1, "Canonical C1 should have 1 predecessor");
        assert_eq!(s_c1_arc.predecessors.iter().next().unwrap().value, 40, "C1 predecessor should be a D node");
        assert_eq!(s_c1_arc.predecessors.iter().next().unwrap().hash_key_cache, s_d_arc.hash_key_cache, "C1 predecessor hash should match D's hash");
         // Check that the predecessor is the canonical D node Arc
         assert!(Arc::ptr_eq(s_c1_arc.predecessors.iter().next().unwrap(), s_d_arc), "C1's predecessor should be the canonical D node Arc");


        let s_b1_nodes = canonical_nodes_by_val.get(&20).unwrap();
        assert_eq!(s_b1_nodes.len(), 1);
        let s_b1_arc = &s_b1_nodes[0];
        assert_ne!(s_b1_arc.hash_key_cache, 0, "B1 node hash should be computed");
        assert_eq!(s_b1_arc.predecessors.len(), 1, "Canonical B1 should have 1 predecessor");
        assert_eq!(s_b1_arc.predecessors.iter().next().unwrap().value, 30, "B1 predecessor should be C1 node");
        assert_eq!(s_b1_arc.predecessors.iter().next().unwrap().hash_key_cache, s_c1_arc.hash_key_cache, "B1 predecessor hash should match C1's hash");
         // Check that the predecessor is the canonical C1 node Arc
         assert!(Arc::ptr_eq(s_b1_arc.predecessors.iter().next().unwrap(), s_c1_arc), "B1's predecessor should be the canonical C1 node Arc");


        let s_a1_nodes = canonical_nodes_by_val.get(&10).unwrap();
        assert_eq!(s_a1_nodes.len(), 1);
        let s_a1_arc = &s_a1_nodes[0];
        assert_ne!(s_a1_arc.hash_key_cache, 0, "A1 node hash should be computed");
        assert_eq!(s_a1_arc.predecessors.len(), 2, "Canonical A1 should have 2 predecessors");

        let a1_pred_hashes: Vec<u64> = s_a1_arc.predecessors.iter().map(|p_arc| p_arc.hash_key_cache).collect();
        let expected_a1_pred_hashes = vec![s_b1_arc.hash_key_cache, s_d_arc.hash_key_cache]; // Order might vary, so compare as sets or sort

        let mut sorted_a1_pred_hashes = a1_pred_hashes;
        sorted_a1_pred_hashes.sort_unstable();
        let mut sorted_expected_a1_pred_hashes = expected_a1_pred_hashes;
        sorted_expected_a1_pred_hashes.sort_unstable();

        assert_eq!(sorted_a1_pred_hashes, sorted_expected_a1_pred_hashes, "A1's canonical predecessors' hashes do not match expected");

         // Check that A1's predecessors are the canonical B1 and D nodes
         let a1_preds_ptrs: HashSet<*const GSSNode<i32>> = s_a1_arc.predecessors.iter().map(|p_arc| Arc::as_ptr(p_arc)).collect();
         let expected_a1_preds_ptrs: HashSet<*const GSSNode<i32>> = vec![Arc::as_ptr(s_b1_arc), Arc::as_ptr(s_d_arc)].into_iter().collect();
         assert_eq!(a1_preds_ptrs, expected_a1_preds_ptrs, "A1's predecessors should be the canonical B1 and D nodes");


        // Test shared node reuse from original structure
        let mut shared_cache: NodeCache<i32> = HashMap::new();
        // E (500, [])
        let e = node_canonical_i32(500, vec![], &mut shared_cache);
        // F (600, [E])
        let f = node_canonical_i32(600, vec![e.clone()], &mut shared_cache);
        // G (700, [E]) - E is shared predecessor
        let g = node_canonical_i32(700, vec![e.clone()], &mut shared_cache);

        let simplified_shared_roots = vec![f.clone(), g.clone()]; // These are already canonical

        assert_eq!(simplified_shared_roots.len(), 2);
        let s_f = &simplified_shared_roots[0];
        let s_g = &simplified_shared_roots[1];

        assert_ne!(Arc::as_ptr(s_f), Arc::as_ptr(s_g), "Canonical F and G should be different Arcs as they are different roots with different values");

        let s_f_pred = s_f.predecessors.iter().next().unwrap();
        let s_g_pred = s_g.predecessors.iter().next().unwrap();

        assert_eq!(s_f_pred.value, 500);
        assert_eq!(s_g_pred.value, 500);

        // The canonical E node should be the same Arc instance for F and G's predecessors.
        assert!(Arc::ptr_eq(s_f_pred, s_g_pred), "Shared canonical node E should be the same Arc instance for F and G's predecessors");
        assert!(Arc::ptr_eq(s_f_pred, &e), "F's predecessor should be the canonical E node");
        assert!(Arc::ptr_eq(s_g_pred, &e), "G's predecessor should be the canonical E node");


        // Test predecessor order normalization and global canonicalization
        let mut norm_cache: NodeCache<i32> = HashMap::new();
        // I (80, [])
        let i = node_canonical_i32(80, vec![], &mut norm_cache);
        // J (90, [])
        let j = node_canonical_i32(90, vec![], &mut norm_cache);

        // H1 (100, [I, J])
        let h1 = node_canonical_i32(100, vec![i.clone(), j.clone()], &mut norm_cache);
        // H2 (100, [J, I]) - Different pred order, but structurally identical after canonicalization of I and J
        let h2 = node_canonical_i32(100, vec![j.clone(), i.clone()], &mut norm_cache);

        let simplified_norm_roots = vec![h1.clone(), h2.clone()]; // These are already canonical

        assert_eq!(simplified_norm_roots.len(), 2, "Expected 2 roots in canonical forest");
        let s_h1 = &simplified_norm_roots[0];
        let s_h2 = &simplified_norm_roots[1];

        // With global canonicalization via get_canonical, s_h1 and s_h2 should point to the same GSSNode content
        // AND be the same Arc instance because their original roots had the same value (100)
        // and the same set of predecessors after canonicalization (the canonical I and J).
        assert_eq!(*s_h1, *s_h2, "s_h1 and s_h2 should have identical GSSNode content after canonicalization");
        assert_eq!(s_h1.hash_key_cache, s_h2.hash_key_cache, "s_h1 and s_h2 should have the same hash after canonicalization");
        assert_eq!(s_h1.value, s_h2.value);

        // This is the key check for global canonicalization of structurally identical nodes
        assert!(Arc::ptr_eq(s_h1, s_h2), "Structurally identical nodes originating from different roots should be canonicalized to the same Arc instance");


        // Check that their predecessor sets, after canonicalization, are identical
        let s_h1_pred_hashes: BTreeSet<u64> = s_h1.predecessors.iter().map(|p_arc| p_arc.hash_key_cache).collect();
        let s_h2_pred_hashes: BTreeSet<u64> = s_h2.predecessors.iter().map(|p_arc| p_arc.hash_key_cache).collect();
        assert_eq!(s_h1_pred_hashes, s_h2_pred_hashes, "Predecessor hashes of s_h1 and s_h2 should be identical after canonicalization");

        // Check that the actual predecessor Arcs are the same (due to I and J simplifying consistently)
        let s_i_arc_h1 = s_h1.predecessors.iter().find(|p_arc| p_arc.value == 80).unwrap();
        let s_j_arc_h1 = s_h1.predecessors.iter().find(|p_arc| p_arc.value == 90).unwrap();
        let s_i_arc_h2 = s_h2.predecessors.iter().find(|p_arc| p_arc.value == 80).unwrap();
        let s_j_arc_h2 = s_h2.predecessors.iter().find(|p_arc| p_arc.value == 90).unwrap();

        assert!(Arc::ptr_eq(s_i_arc_h1, s_i_arc_h2), "Canonical I-node should be the same Arc instance for H1 and H2");
        assert!(Arc::ptr_eq(s_j_arc_h1, s_j_arc_h2), "Canonical J-node should be the same Arc instance for H1 and H2");
        assert!(Arc::ptr_eq(s_i_arc_h1, &i), "H1/H2's I predecessor should be the canonical I node");
        assert!(Arc::ptr_eq(s_j_arc_h1, &j), "H1/H2's J predecessor should be the canonical J node");
    }

    #[test]
    fn test_global_canonicalization_with_distinct_initial_arcs() {
        // Helper for i32 nodes
        fn node_canonical_i32(
            value: i32,
            predecessors: Vec<Arc<GSSNode<i32>>>,
            cache: &mut NodeCache<i32>,
        ) -> Arc<GSSNode<i32>> {
            let pred_set: BTreeSet<Arc<GSSNode<i32>>> = predecessors.into_iter().collect();
            GSSNode::get_canonical(value, pred_set, cache)
        }

        let mut cache: NodeCache<i32> = HashMap::new();

        // L1, L2, L3 are initially distinct Arcs, but structurally identical (value 0, no preds).
        // get_canonical will ensure they resolve to the *same* canonical Arc.
        let l1 = node_canonical_i32(0, vec![], &mut cache);
        let l2 = node_canonical_i32(0, vec![], &mut cache);
        let l3 = node_canonical_i32(0, vec![], &mut cache);

        // Verify they are the same Arc instance due to canonicalization
        assert!(Arc::ptr_eq(&l1, &l2));
        assert!(Arc::ptr_eq(&l1, &l3));
        assert!(Arc::ptr_eq(&l2, &l3));

        // M1, M2, M3 have the same value (1).
        // Their predecessors (L1, L2, L3 respectively) were canonicalized to a single Arc (let's call it canonical_L).
        // M1, M2, M3 will all be created with value 1 and predecessor set containing only canonical_L.
        // get_canonical will ensure M1, M2, M3 resolve to the *same* canonical Arc.
        let m1 = node_canonical_i32(1, vec![l1.clone()], &mut cache); // Use clone of canonical_L
        let m2 = node_canonical_i32(1, vec![l2.clone()], &mut cache); // Use clone of canonical_L
        let m3 = node_canonical_i32(1, vec![l3.clone()], &mut cache); // Use clone of canonical_L

        // Verify they are the same Arc instance due to canonicalization
        assert!(Arc::ptr_eq(&m1, &m2));
        assert!(Arc::ptr_eq(&m1, &m3));
        assert!(Arc::ptr_eq(&m2, &m3));

        // R1 has M1, M2, M3 as predecessors. Since M1, M2, M3 canonicalized to a single Arc (canonical_M),
        // R1 will be created with value 2 and predecessor set containing only canonical_M.
        // get_canonical will ensure R1 resolves to a single canonical Arc.
        let r1 = node_canonical_i32(2, vec![m1.clone(), m2.clone(), m3.clone()], &mut cache);

        // The forest is now built and canonicalized. The root is just r1.
        let roots = vec![r1.clone()];

        let mut collected_arcs_map = HashMap::new();
        collect_arcs_recursive(&r1, &mut collected_arcs_map); // Use recursive collector on the root

        // Expected unique Arcs with GLOBAL canonicalization:
        // One canonical L-level node (from l1, l2, l3) -> 1 Arc.
        // One canonical M-level node (from m1, m2, m3) -> 1 Arc.
        // One canonical R-level node (from r1) -> 1 Arc.
        // Total = 1 + 1 + 1 = 3 unique Arcs.
        assert_eq!(collected_arcs_map.len(), 3, "Expected 3 unique Arcs in the canonical GSS forest");

        // Detailed verification of the structure:
        let s_r1_node = r1.as_ref();
        assert_eq!(s_r1_node.value, 2);
        // With global canonicalization, R1's predecessors (which were M1, M2, M3)
        // should all be the *same* canonical M-level Arc.
        // The BTreeSet will only contain one entry for this repeated Arc.
        assert_eq!(s_r1_node.predecessors.len(), 1, "Canonical R1 should have 1 predecessor Arc (the canonical M node)");

        // Get the single canonical M-level node
        let s_m_level_arc = s_r1_node.predecessors.iter().next().unwrap().clone();
        assert_eq!(s_m_level_arc.value, 1);
        // The canonical M-level node's predecessors should be the single canonical L-level node.
        assert_eq!(s_m_level_arc.predecessors.len(), 1, "The canonical M node should have 1 predecessor Arc (the canonical L node)");

        // Get the single canonical L-level node
        let s_l_level_arc = s_m_level_arc.predecessors.iter().next().unwrap().clone();
        assert_eq!(s_l_level_arc.value, 0);
        assert_eq!(s_l_level_arc.predecessors.len(), 0, "The canonical L node should have no predecessors");

        // Verify the Arcs are canonical (pointers are equal for structurally identical nodes)
        let canonical_r_arc = r1.clone();
        let canonical_m_arc = s_m_level_arc;
        let canonical_l_arc = s_l_level_arc;

        // Find the arcs in the collected map by value to confirm they are the same instances
        let found_r_arc = collected_arcs_map.values().find(|a| a.value == 2).unwrap();
        let found_m_arc = collected_arcs_map.values().find(|a| a.value == 1).unwrap();
        let found_l_arc = collected_arcs_map.values().find(|a| a.value == 0).unwrap();

        assert!(Arc::ptr_eq(&canonical_r_arc, found_r_arc), "Root R1 should be the canonical R node");
        assert!(Arc::ptr_eq(&canonical_m_arc, found_m_arc), "The single predecessor of canonical R should be the canonical M node");
        assert!(Arc::ptr_eq(&canonical_l_arc, found_l_arc), "The single predecessor of canonical M should be the canonical L node");

        // Also check pointer equality between constructed nodes and the canonical ones
        assert!(Arc::ptr_eq(&l1, &canonical_l_arc));
        assert!(Arc::ptr_eq(&m1, &canonical_m_arc));
        assert!(Arc::ptr_eq(&r1, &canonical_r_arc));

    }


    #[test]
    fn test_gss_canonicalization_reproduces_logged_structure() {
        let mut cache: MockNodeCache = HashMap::new();

        // Values for the nodes, mimicking the log's StateID and LLMTokenInfo
        let token_info = MockLLMTokenInfo {
            active: "[0]".to_string(),
            intersection: "[0]".to_string(),
        };

        let val0 = MockParseStateNodeContent { state_id: 0, t: token_info.clone() };
        let val1 = MockParseStateNodeContent { state_id: 1, t: token_info.clone() };
        let val2 = MockParseStateNodeContent { state_id: 2, t: token_info.clone() };

        // --- Constructing the canonical GSS ---

        // State 0 Leaf Nodes: All with val0 and no predecessors will be the *same* canonical Arc
        let node_a_val0 = node_canonical(val0.clone(), vec![], &mut cache); // Root 0
        let node_c_val0 = node_canonical(val0.clone(), vec![], &mut cache); // Shared predecessor
        let node_g_val0 = node_canonical(val0.clone(), vec![], &mut cache);
        let node_i_val0 = node_canonical(val0.clone(), vec![], &mut cache);
        let node_k_val0 = node_canonical(val0.clone(), vec![], &mut cache);
        let node_m_val0 = node_canonical(val0.clone(), vec![], &mut cache);
        let node_o_val0 = node_canonical(val0.clone(), vec![], &mut cache);
        let node_q_val0 = node_canonical(val0.clone(), vec![], &mut cache);
        let node_s_val0 = node_canonical(val0.clone(), vec![], &mut cache);
        let node_u_val0 = node_canonical(val0.clone(), vec![], &mut cache);

        // Assert that all StateID(0) nodes are the same canonical Arc
        let canonical_s0_node = node_a_val0.clone(); // Pick one as the reference
        assert!(Arc::ptr_eq(&node_c_val0, &canonical_s0_node));
        assert!(Arc::ptr_eq(&node_g_val0, &canonical_s0_node));
        assert!(Arc::ptr_eq(&node_i_val0, &canonical_s0_node));
        assert!(Arc::ptr_eq(&node_k_val0, &canonical_s0_node));
        assert!(Arc::ptr_eq(&node_m_val0, &canonical_s0_node));
        assert!(Arc::ptr_eq(&node_o_val0, &canonical_s0_node));
        assert!(Arc::ptr_eq(&node_q_val0, &canonical_s0_node));
        assert!(Arc::ptr_eq(&node_s_val0, &canonical_s0_node));
        assert!(Arc::ptr_eq(&node_u_val0, &canonical_s0_node));


        // State 1 Intermediate Nodes: All with val1 and predecessor set {canonical_s0_node} will be the *same* canonical Arc
        let node_b_val1 = node_canonical(val1.clone(), vec![node_c_val0.clone()], &mut cache); // Root 1, predecessor is canonical_s0_node

        // Node E (pred: Node C, which is canonical_s0_node)
        let node_e_val1 = node_canonical(val1.clone(), vec![node_c_val0.clone()], &mut cache); // Shares node_c_val0 (canonical_s0_node)

        // 8 more intermediate nodes with val1, each pointing to one of the other_s0_leaves.
        // Since all other_s0_leaves canonicalized to canonical_s0_node, all these StateID(1)
        // nodes will have the same value (val1) and the same predecessor set ({canonical_s0_node}).
        // Thus, they will all canonicalize to the *same* Arc.
        let mut other_s1_nodes: Vec<Arc<MockGSSNode>> = Vec::new();
        let other_s0_leaves = vec![ // Put these in a vec for easy iteration
            node_g_val0.clone(), node_i_val0.clone(), node_k_val0.clone(),
            node_m_val0.clone(), node_o_val0.clone(), node_q_val0.clone(),
            node_s_val0.clone(), node_u_val0.clone(),
        ];
        for leaf_s0_node in &other_s0_leaves {
            // The predecessor here is always the canonical_s0_node
            other_s1_nodes.push(node_canonical(val1.clone(), vec![leaf_s0_node.clone()], &mut cache));
        }

        // Assert that node_b_val1, node_e_val1, and all other_s1_nodes are the same canonical Arc
        let canonical_s1_node = node_b_val1.clone(); // Pick one as the reference
        assert!(Arc::ptr_eq(&node_e_val1, &canonical_s1_node));
        for other_s1 in &other_s1_nodes {
            assert!(Arc::ptr_eq(other_s1, &canonical_s1_node));
        }


        // State 2 Node D: Has val2. Its predecessors are node_e_val1 and the 8 other_s1_nodes.
        // Since node_e_val1 and all other_s1_nodes canonicalized to canonical_s1_node,
        // the predecessor set for node_d_val2 will be {canonical_s1_node}.
        let mut preds_for_d = vec![
            node_e_val1.clone(), // canonical_s1_node
        ];
        preds_for_d.extend(other_s1_nodes.iter().cloned()); // Extend with clones of canonical_s1_node
        // The vec `preds_for_d` now contains 9 clones of the *same* Arc (canonical_s1_node).
        // When passed to node_canonical, it will become a BTreeSet containing only *one* Arc.
        assert_eq!(preds_for_d.len(), 9, "Vector of predecessors before BTreeSet should have 9 items (clones)");


        let node_d_val2 = node_canonical(val2.clone(), preds_for_d, &mut cache); // Root 2

        // Assert that node_d_val2 is a unique canonical Arc (value 2, pred set {canonical_s1_node})
        let canonical_s2_node = node_d_val2.clone();


        // --- Collect roots and print ---
        // The roots are now the canonical Arcs representing the entry points
        let roots = vec![
            node_a_val0.clone(), // Root 0 -> canonical_s0_node
            node_b_val1.clone(), // Root 1 -> canonical_s1_node
            node_d_val2.clone(), // Root 2 -> canonical_s2_node
        ];

        // Assert the roots are the canonical nodes we identified
        assert!(Arc::ptr_eq(&roots[0], &canonical_s0_node));
        assert!(Arc::ptr_eq(&roots[1], &canonical_s1_node));
        assert!(Arc::ptr_eq(&roots[2], &canonical_s2_node));


        let max_nodes_to_print = 30; // Match the log's max_nodes
        // Print the canonical GSS (no separate simplification step needed)
        let canonical_gss_string_representation = print_gss_forest(&roots, max_nodes_to_print);

        println!("\n--- Canonical GSS Structure for Visual Comparison ---\n");
        println!("{}", canonical_gss_string_representation);
        println!("--- End of Canonical GSS Structure ---\n");


        // Collect unique Arcs in the canonical GSS forest by traversing from roots
        let mut collected_arcs_map: HashMap<*const MockGSSNode, Arc<MockGSSNode>> = HashMap::new();
        for root_arc in &roots {
            collect_arcs_recursive(root_arc, &mut collected_arcs_map);
        }

        // Assert the number of unique Arcs (nodes) in the canonical graph
        // Expected:
        // One canonical StateID(0) node -> 1 Arc
        // One canonical StateID(1) node -> 1 Arc
        // One canonical StateID(2) node -> 1 Arc
        // Total = 1 + 1 + 1 = 3 unique Arcs.
        assert_eq!(collected_arcs_map.len(), 3, "The canonical GSS should contain 3 unique Arcs");

        // Further checks on the canonical structure:
        assert_eq!(roots.len(), 3, "Should still have 3 roots (representing the entry points)");

        // Get the canonical nodes from the collected map by value (should only be one of each value)
        let mut canonical_nodes_by_value: HashMap<usize, Arc<MockGSSNode>> = HashMap::new();
        for canonical_arc in collected_arcs_map.values() {
            canonical_nodes_by_value.insert(canonical_arc.value.state_id, canonical_arc.clone());
        }
        assert_eq!(canonical_nodes_by_value.len(), 3, "Should find one canonical node for each StateID (0, 1, 2)");

        let s_node_0 = canonical_nodes_by_value.get(&0).unwrap();
        let s_node_1 = canonical_nodes_by_value.get(&1).unwrap();
        let s_node_2 = canonical_nodes_by_value.get(&2).unwrap();

        // Verify structure of canonical nodes
        assert_eq!(s_node_0.value.state_id, 0);
        assert_eq!(s_node_0.predecessors.len(), 0, "Canonical StateID 0 node should have no predecessors");

        assert_eq!(s_node_1.value.state_id, 1);
        assert_eq!(s_node_1.predecessors.len(), 1, "Canonical StateID 1 node should have 1 predecessor");
        assert!(Arc::ptr_eq(s_node_1.predecessors.iter().next().unwrap(), s_node_0), "Canonical StateID 1 node's predecessor should be the canonical StateID 0 node");

        assert_eq!(s_node_2.value.state_id, 2);
        assert_eq!(s_node_2.predecessors.len(), 1, "Canonical StateID 2 node should have 1 predecessor");
        assert!(Arc::ptr_eq(s_node_2.predecessors.iter().next().unwrap(), s_node_1), "Canonical StateID 2 node's predecessor should be the canonical StateID 1 node");

        // Verify that the roots are the correct canonical nodes
        // Note: The order of roots in the output vector corresponds to the order of roots in the input vector.
        assert!(Arc::ptr_eq(&roots[0], s_node_0), "Root 0 should be the canonical StateID 0 node");
        assert!(Arc::ptr_eq(&roots[1], s_node_1), "Root 1 should be the canonical StateID 1 node");
        assert!(Arc::ptr_eq(&roots[2], s_node_2), "Root 2 should be the canonical StateID 2 node");
    }
}
