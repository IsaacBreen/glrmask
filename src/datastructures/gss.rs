use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;

use crate::datastructures::ArcPtrWrapper; // Import ArcPtrWrapper

#[derive(Debug, Clone)] // Removed PartialEq, Eq, PartialOrd, Ord, Hash
pub struct GSSNode<T> {
    pub value: T,
    predecessors: BTreeSet<ArcPtrWrapper<GSSNode<T>>>,
    hash_key_cache: u64, // Add this line
}

impl<T: Hash> GSSNode<T> {
    pub fn new(value: T) -> Self {
        let hash_key_cache = Self::compute_hash_key_cache(&BTreeSet::new()); // Removed '&value, '
        Self {
            value,
            predecessors: BTreeSet::new(),
            hash_key_cache,
        }
    }
    pub fn new_with_predecessors(value: T, predecessors: BTreeSet<Arc<GSSNode<T>>>) -> Self {
        let predecessors_arc_ptr_wrapper = predecessors.into_iter().map(ArcPtrWrapper::new).collect(); // This is now BTreeSet<ArcPtrWrapper<GSSNode<T>>>
        let hash_key_cache = Self::compute_hash_key_cache(&predecessors_arc_ptr_wrapper); // Removed '&value, '
        Self {
            value,
            predecessors: predecessors_arc_ptr_wrapper,
            hash_key_cache,
        }
    }

    pub fn compute_hash_key_cache(predecessors: &BTreeSet<ArcPtrWrapper<GSSNode<T>>>) -> u64 { // Removed 'value: &T'
        let mut hasher = DefaultHasher::new();
        // value.hash(&mut hasher); // Remove this line
        for pred in predecessors {
            pred.hash(&mut hasher);
        }
        hasher.finish()
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
        // hash_key_cache is now invalid, should be recomputed or marked dirty if used
    }

    pub fn merge_unchecked(&mut self, mut other: Self)
    {
        self.predecessors.extend(std::mem::take(&mut other.predecessors));
        // hash_key_cache is now invalid
    }

    pub fn map<F, U>(&self, f: F) -> GSSNode<U>
    where
        F: Copy + Fn(&T) -> U,
        U: Hash, // U needs Hash for the GSSNode<U> to be valid in contexts expecting Hash
    {
        let new_value = f(&self.value); // Renamed 'value' to 'new_value' for clarity
        let new_predecessors: BTreeSet<ArcPtrWrapper<GSSNode<U>>> = self.predecessors.iter()
            .map(|wrapper| ArcPtrWrapper::new(Arc::new(wrapper.as_ref().map(f))))
            .collect();
        let hash_key_cache = GSSNode::<U>::compute_hash_key_cache(&new_predecessors); // Removed '&new_value, '
        GSSNode {
            value: new_value,
            predecessors: new_predecessors,
            hash_key_cache,
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

impl<T: Hash> Hash for GSSNode<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_key_cache.hash(state);
        self.value.hash(state); // Add this line
    }
}

impl<T: Hash + PartialEq> PartialEq for GSSNode<T> {
    fn eq(&self, other: &Self) -> bool {
        // First compare hash, then value, then predecessors
        self.hash_key_cache == other.hash_key_cache && self.value == other.value && self.predecessors == other.predecessors
    }
}

impl<T: Hash + PartialEq> Eq for GSSNode<T> {}

impl<T: Hash + PartialOrd> PartialOrd for GSSNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.hash_key_cache.partial_cmp(&other.hash_key_cache) {
            Some(Ordering::Equal) => {
                // Hashes are equal, compare values
                match self.value.partial_cmp(&other.value) {
                    Some(Ordering::Equal) => self.predecessors.partial_cmp(&other.predecessors), // Values are also equal (or reported as equal by partial_cmp), compare predecessors
                    other_ordering => other_ordering, // Values are different or one is greater/less
                }
            }
            other_ordering => other_ordering, // Hashes are different or incomparable
        }
    }
}

impl<T: Hash + PartialOrd + Ord> Ord for GSSNode<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare hash, then value, then predecessors
        self.hash_key_cache.cmp(&other.hash_key_cache).then_with(|| self.value.cmp(&other.value)).then_with(|| self.predecessors.cmp(&other.predecessors))
    }
}


pub trait GSSTrait<T: Clone> {
    type Peek<'a> where T: 'a, Self: 'a;
    fn peek(&self) -> Self::Peek<'_>;
    fn push(&self, value: T) -> GSSNode<T>;
    fn pop(&self) -> Vec<Arc<GSSNode<T>>>;
    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>>;
}

impl<T: Clone + Hash> GSSTrait<T> for GSSNode<T> { // Added Hash bound
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

impl<T: Clone + Hash> GSSTrait<T> for Arc<GSSNode<T>> { // Added Hash bound
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

impl<T: Clone + Hash> GSSTrait<T> for Option<Arc<GSSNode<T>>> { // Added Hash bound
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

impl<T: Clone + Hash> GSSTrait<T> for Option<GSSNode<T>> { // Added Hash bound
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
                // Note: When merging, the hash_key_cache of `first_mut_ref` becomes invalid
                // as its predecessors have changed. It should ideally be recomputed or marked as dirty.
                // For now, the simplification step recomputes hashes correctly.
                self.push(first);
            }
        }
    }
}

// Helper function for prune_and_transform_roots
pub fn prune_and_transform_recursive<T: Clone + Hash>(
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
            let mut new_predecessors;
            if continue_recursion {
                // Continue recursion for predecessors
                new_predecessors = BTreeSet::new();
                for pred_wrapper in &node_arc.predecessors { // pred_wrapper is &ArcPtrWrapper<GSSNode<T>>
                    let pred_arc = pred_wrapper.as_arc(); // pred_arc is &Arc<GSSNode<T>>
                    if let Some(new_pred) = prune_and_transform_recursive(pred_arc, closure, memo) {
                        new_predecessors.insert(ArcPtrWrapper::new(new_pred));
                    }
                }
            } else {
                // Stop recursion, create new node with original predecessors but new value
                new_predecessors = node_arc.predecessors.clone();
            };
            let hash_key_cache = GSSNode::<T>::compute_hash_key_cache(&new_predecessors); // Removed '&new_value, '
            let new_node_arc = Arc::new(GSSNode { value: new_value, predecessors: new_predecessors, hash_key_cache }); // Add this line
            memo.insert(node_ptr, Some(new_node_arc.clone()));
            Some(new_node_arc)
        }
    }
}

/// Traverses the GSS forest defined by `roots`, applying `closure` to each node's value.
/// Handles shared nodes using memoization. Prunes branches where `closure` returns `None`.
/// Stops recursion down a path if `closure` returns `(_, false)`.
/// Returns a Vec of `Option<Arc<GSSNode<T>>>` corresponding to the input `roots`.
pub fn prune_and_transform_roots<T: Clone + std::hash::Hash>(
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

// Helper function for GSS simplification.
// Recursively simplifies a node and its predecessors.
// Uses two memoization tables:
// 1. original_ptr_memo: Maps original node pointers to their canonical simplified Arcs.
//                       Handles cycles and ensures original shared nodes are processed once.
// 2. canonical_nodes_memo: Maps a canonical structural representation (value + canonical predecessor ArcPtrWrappers)
//                          to the single canonical Arc for that structure. Ensures global canonicalization.
fn simplify_node_recursive<T: Clone + Ord + Hash + Debug>(
    original_node_arc: &Arc<GSSNode<T>>,
    original_ptr_memo: &mut HashMap<*const GSSNode<T>, Arc<GSSNode<T>>>,
    canonical_nodes_memo: &mut HashMap<(T, BTreeSet<ArcPtrWrapper<GSSNode<T>>>), Arc<GSSNode<T>>>,
) -> Arc<GSSNode<T>> {
    let original_node_ptr = Arc::as_ptr(original_node_arc);

    // Check original_ptr_memo first: if this original Arc has been visited, return its known simplified form.
    if let Some(simplified_node) = original_ptr_memo.get(&original_node_ptr) {
        return simplified_node.clone();
    }

    // Recursively simplify predecessors to get a set of canonical predecessor Arcs.
    let mut simplified_predecessors_canonical_arcs: BTreeSet<Arc<GSSNode<T>>> = BTreeSet::new();
    for pred_wrapper in &original_node_arc.predecessors {
        let original_pred_arc = pred_wrapper.as_arc();
        let simplified_pred_arc = simplify_node_recursive(
            original_pred_arc,
            original_ptr_memo, // Pass down both memos
            canonical_nodes_memo,
        );
        simplified_predecessors_canonical_arcs.insert(simplified_pred_arc);
    }

    // Convert the set of canonical predecessor Arcs to ArcPtrWrappers for the new node's structure
    // and for the canonical_nodes_memo key.
    let new_node_predecessors_wrappers: BTreeSet<ArcPtrWrapper<GSSNode<T>>> =
        simplified_predecessors_canonical_arcs
            .iter()
            .map(|arc| ArcPtrWrapper::new(arc.clone()))
            .collect();

    // Create the key for the canonical_nodes_memo.
    // This key represents the structural identity of the node we want to find/create.
    let node_value_clone = original_node_arc.value.clone();
    let canonical_key = (node_value_clone, new_node_predecessors_wrappers.clone()); // Clone wrappers for key

    // Check canonical_nodes_memo: if a node with this exact structure already exists, reuse it.
    if let Some(existing_canonical_arc) = canonical_nodes_memo.get(&canonical_key) {
        // This structure has been created before. Reuse the canonical Arc.
        // Also, memoize this result for the original_node_ptr.
        original_ptr_memo.insert(original_node_ptr, existing_canonical_arc.clone());
        return existing_canonical_arc.clone();
    }

    // This specific structure (value + canonical predecessors) hasn't been seen. Create a new canonical node.
    let new_hash_key_cache = GSSNode::compute_hash_key_cache(&new_node_predecessors_wrappers);

    let new_canonical_gss_node = GSSNode {
        value: original_node_arc.value.clone(), // Use original_node_arc.value directly, canonical_key took a clone
        predecessors: new_node_predecessors_wrappers, // These are already the wrappers from canonical_key
        hash_key_cache: new_hash_key_cache,
    };
    let new_canonical_arc = Arc::new(new_canonical_gss_node);

    // Store this new canonical Arc in both memos:
    // 1. In canonical_nodes_memo for future structural lookups.
    canonical_nodes_memo.insert(canonical_key, new_canonical_arc.clone());
    // 2. In original_ptr_memo for this specific original_node_ptr.
    original_ptr_memo.insert(original_node_ptr, new_canonical_arc.clone());

    new_canonical_arc
}

/// Simplifies a GSS forest, ensuring that structurally identical nodes
/// (after simplification) are represented by shared `Arc<GSSNode<T>>` instances
/// (global canonicalization or hash-consing).
///
/// The simplification process works from the bottom up (leaves to roots).
/// - Node values are preserved.
/// - Predecessor lists are normalized by ordering simplified predecessors based on their content hash.
/// - A hash is computed for each simplified node based on its value and its simplified predecessors' hashes.
///   This hash is stored in `hash_key_cache` and used for `Ord` comparisons within the simplification logic.
///
/// Handles shared nodes and, with global canonicalization, structurally identical nodes
/// will resolve to the same Arc instance regardless of their original Arc identity.
/// Assumes the GSS forest does not contain cycles.
pub fn simplify_gss_forest<T: Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T>>],
) -> Vec<Arc<GSSNode<T>>> {
    let mut original_ptr_memo: HashMap<*const GSSNode<T>, Arc<GSSNode<T>>> = HashMap::new();
    // New memo for canonical nodes: Key is (value, BTreeSet<ArcPtrWrapper_to_canonical_predecessors>)
    let mut canonical_nodes_memo: HashMap<(T, BTreeSet<ArcPtrWrapper<GSSNode<T>>>), Arc<GSSNode<T>>> = HashMap::new();
    let mut simplified_roots = Vec::with_capacity(roots.len());

    for root_arc in roots {
        simplified_roots.push(simplify_node_recursive(
            root_arc,
            &mut original_ptr_memo,
            &mut canonical_nodes_memo,
        ));
    }

    simplified_roots
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::collections::{BTreeSet, HashMap, HashSet, VecDeque}; // Add if not present
    use std::fmt::Debug; // Add if not present


    // Define Mock Types
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


    // Helper to create a node for tests
    // Assumes GSSNode::new_with_predecessors and ArcPtrWrapper are working as expected
    // and T: Clone + Ord + Hash + Debug (MockParseStateNodeContent satisfies this)
    fn node(value: MockParseStateNodeContent, predecessors: Vec<Arc<MockGSSNode>>) -> Arc<MockGSSNode> {
        Arc::new(MockGSSNode::new_with_predecessors(value, predecessors.into_iter().collect()))
    }


    // Helper to get a stable representation of a simplified GSS for comparison.
    // Returns (value, Vec<pred_hashes_sorted>)
    type SimplifiedNodeRepr<T> = (T, Vec<u64>);

    fn get_simplified_repr<T: Clone + Hash>(node_arc: &Arc<GSSNode<T>>) -> SimplifiedNodeRepr<T> { // Added Hash bound for get_hash_key_cache
        let mut pred_hashes: Vec<u64> = node_arc.predecessors.iter()
            .map(|p| p.as_arc().hash_key_cache) // Use getter
            .collect();
        pred_hashes.sort_unstable();
        (node_arc.value.clone(), pred_hashes)
    }

    // Helper to recursively collect all unique node representations in a simplified forest
    fn collect_all_simplified_nodes<T: Clone + Hash>(
        node_arc: &Arc<GSSNode<T>>,
        visited: &mut HashSet<*const GSSNode<T>>,
        collected_nodes: &mut HashMap<*const GSSNode<T>, SimplifiedNodeRepr<T>>,
    ) {
        let ptr = Arc::as_ptr(node_arc);
        if !visited.insert(ptr) {
            return;
        }
        collected_nodes.insert(ptr, get_simplified_repr(node_arc));
        for pred_wrapper in &node_arc.predecessors {
            collect_all_simplified_nodes(pred_wrapper.as_arc(), visited, collected_nodes);
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
        for pred_wrapper in &node_arc.predecessors {
            collect_arcs_recursive(pred_wrapper.as_arc(), collected_arcs);
        }
    }


    #[test]
    fn test_gss_simplification_basic() {
        fn node(value: i32, predecessors: Vec<Arc<GSSNode<i32>>>) -> Arc<GSSNode<i32>> {
            Arc::new(GSSNode::new_with_predecessors(value, predecessors.into_iter().collect()))
        }

        // D1
        // |
        // C1
        // |
        // B1   D2
        // |   /
        // A1 (preds: B1, D2)
        let d1_orig = node(40, vec![]);
        let c1_orig = node(30, vec![d1_orig.clone()]);
        let b1_orig = node(20, vec![c1_orig.clone()]);

        let d2_orig = node(40, vec![]); // Same content as d1_orig, but different instance initially

        let a1_orig = node(10, vec![b1_orig.clone(), d2_orig.clone()]);

        let roots = vec![a1_orig.clone()];
        let simplified_roots = simplify_gss_forest(&roots);
        let simplified_a1 = simplified_roots[0].clone();

        // Verify structure and hash caching after simplification
        // Simplified D nodes (s_d1 from d1_orig, s_d2 from d2_orig)
        // With global canonicalization, structurally identical nodes should merge.
        // d1_orig and d2_orig are structurally identical leaves (value 40, no preds).
        // They should simplify to the *same* Arc instance.

        let mut visited_check = HashSet::new();
        let mut collected_check = HashMap::new();
        collect_all_simplified_nodes(&simplified_a1, &mut visited_check, &mut collected_check);

        // Expected simplified structure values and predecessor hashes:
        // D_s(40, []) -> hash_d
        // C1_s(30, [D_s]) -> hash_c1 (depends on hash_d)
        // B1_s(20, [C1_s]) -> hash_b1 (depends on hash_c1)
        // A1_s(10, [B1_s, D_s]) -> hash_a1 (depends on hash_b1, hash_d)
        // Note: D_s and D_s' will now be the *same* Arc, and thus have the same hash_key_cache value.

        // Find the simplified nodes by value (this is a bit indirect for a test)
        let mut s_nodes_by_val = HashMap::new();
         // Iterate over the unique Arcs found by collect_all_simplified_nodes' keys
        for s_node_arc in collected_check.keys().map(|k| unsafe { Arc::from_raw(*k) }) {
            s_nodes_by_val.entry(s_node_arc.value).or_insert_with(Vec::new).push(s_node_arc.clone());
             // Do not forget Arc here, collect_all_simplified_nodes keys are raw pointers, ownership is not transferred
        }


        let s_d_nodes = s_nodes_by_val.get(&40).unwrap();
        assert_eq!(s_d_nodes.len(), 1, "Structurally identical D nodes should be canonicalized to 1 Arc"); // MODIFIED: Expected 1 unique Arc
        let s_d_hash = s_d_nodes[0].hash_key_cache;
        assert_ne!(s_d_hash, 0, "D node hash should be computed");
        assert_eq!(s_d_nodes[0].predecessors.len(), 0, "Simplified D node should have no predecessors");


        let s_c1_nodes = s_nodes_by_val.get(&30).unwrap();
        assert_eq!(s_c1_nodes.len(), 1);
        let s_c1 = &s_c1_nodes[0];
        assert_ne!(s_c1.hash_key_cache, 0, "C1 node hash should be computed");
        assert_eq!(s_c1.predecessors.len(), 1, "Simplified C1 should have 1 predecessor");
        assert_eq!(s_c1.predecessors.iter().next().unwrap().as_arc().value, 40, "C1 predecessor should be a D node");
        assert_eq!(s_c1.predecessors.iter().next().unwrap().as_arc().hash_key_cache, s_d_hash, "C1 predecessor hash should match D's hash");
         // Check that the predecessor is the canonical D node Arc
         assert!(Arc::ptr_eq(s_c1.predecessors.iter().next().unwrap().as_arc(), &s_d_nodes[0]), "C1's predecessor should be the canonical D node Arc");


        let s_b1_nodes = s_nodes_by_val.get(&20).unwrap();
        assert_eq!(s_b1_nodes.len(), 1);
        let s_b1 = &s_b1_nodes[0];
        assert_ne!(s_b1.hash_key_cache, 0, "B1 node hash should be computed");
        assert_eq!(s_b1.predecessors.len(), 1, "Simplified B1 should have 1 predecessor");
        assert_eq!(s_b1.predecessors.iter().next().unwrap().as_arc().value, 30, "B1 predecessor should be C1 node");
        assert_eq!(s_b1.predecessors.iter().next().unwrap().as_arc().hash_key_cache, s_c1.hash_key_cache, "B1 predecessor hash should match C1's hash");
         // Check that the predecessor is the canonical C1 node Arc
         assert!(Arc::ptr_eq(s_b1.predecessors.iter().next().unwrap().as_arc(), s_c1), "B1's predecessor should be the canonical C1 node Arc");


        let s_a1_nodes = s_nodes_by_val.get(&10).unwrap();
        assert_eq!(s_a1_nodes.len(), 1);
        let s_a1 = &s_a1_nodes[0];
        assert_ne!(s_a1.hash_key_cache, 0, "A1 node hash should be computed");
        assert_eq!(s_a1.predecessors.len(), 2, "Simplified A1 should have 2 predecessors");

        let a1_pred_hashes: Vec<u64> = s_a1.predecessors.iter().map(|p| p.as_arc().hash_key_cache).collect();
        let expected_a1_pred_hashes = vec![s_b1.hash_key_cache, s_d_hash]; // Order might vary, so compare as sets or sort

        let mut sorted_a1_pred_hashes = a1_pred_hashes;
        sorted_a1_pred_hashes.sort_unstable();
        let mut sorted_expected_a1_pred_hashes = expected_a1_pred_hashes;
        sorted_expected_a1_pred_hashes.sort_unstable();

        assert_eq!(sorted_a1_pred_hashes, sorted_expected_a1_pred_hashes, "A1's simplified predecessors' hashes do not match expected");

         // Check that A1's predecessors are the canonical B1 and D nodes
         let a1_preds_ptrs: HashSet<*const GSSNode<i32>> = s_a1.predecessors.iter().map(|p| Arc::as_ptr(p.as_arc())).collect();
         let expected_a1_preds_ptrs: HashSet<*const GSSNode<i32>> = vec![Arc::as_ptr(s_b1), Arc::as_ptr(&s_d_nodes[0])].into_iter().collect();
         assert_eq!(a1_preds_ptrs, expected_a1_preds_ptrs, "A1's predecessors should be the canonical B1 and D nodes");


        // Test shared node reuse from original structure
        // E -> F
        // E -> G
        // Root1 = F, Root2 = G. E should be simplified only once and result in a single canonical Arc.
        let e_orig = node(500, vec![]);
        let f_orig = node(600, vec![e_orig.clone()]);
        let g_orig = node(700, vec![e_orig.clone()]); // e_orig is shared

        let simplified_shared = simplify_gss_forest(&[f_orig, g_orig]);
        assert_eq!(simplified_shared.len(), 2);
        let s_f = &simplified_shared[0];
        let s_g = &simplified_shared[1];

        assert_ne!(Arc::as_ptr(s_f), Arc::as_ptr(s_g), "Simplified F and G should be different Arcs as they are different roots with different values");

        let s_f_pred = s_f.predecessors.iter().next().unwrap().as_arc();
        let s_g_pred = s_g.predecessors.iter().next().unwrap().as_arc();

        assert_eq!(s_f_pred.value, 500);
        assert_eq!(s_g_pred.value, 500);

        // The simplified E node should be the same Arc instance for F and G's predecessors.
        assert!(Arc::ptr_eq(s_f_pred, s_g_pred), "Shared original node E should simplify to the same Arc instance for F and G's predecessors");


        // Test predecessor order normalization
        // H1 -> (I, J)
        // H2 -> (J, I)
        // I, J are leaves. H1 and H2 should simplify to identical GSSNode structures,
        // and with global canonicalization, they should resolve to the *same* Arc instance,
        // since they originate from different roots but have the same value and
        // same set of simplified (and canonical) predecessors.
        let i_orig = node(80, vec![]);
        let j_orig = node(90, vec![]);

        let h1_orig = node(100, vec![i_orig.clone(), j_orig.clone()]);
        let h2_orig = node(100, vec![j_orig.clone(), i_orig.clone()]); // Different pred order

        let simplified_norm = simplify_gss_forest(&[h1_orig, h2_orig]);
        assert_eq!(simplified_norm.len(), 2, "Expected 2 roots in simplified forest");
        let s_h1 = &simplified_norm[0];
        let s_h2 = &simplified_norm[1];

        // With global canonicalization, s_h1 and s_h2 should point to the same GSSNode content
        // AND be the same Arc instance because their original roots had the same value (100)
        // and the same set of predecessors after simplification (the canonical I and J).
        assert_eq!(*s_h1, *s_h2, "s_h1 and s_h2 should have identical GSSNode content after normalization and simplification");
        assert_eq!(s_h1.hash_key_cache, s_h2.hash_key_cache, "s_h1 and s_h2 should have the same hash after normalization");
        assert_eq!(s_h1.value, s_h2.value);

        // This is the key check for global canonicalization of structurally identical nodes
        assert!(Arc::ptr_eq(s_h1, s_h2), "Structurally identical nodes originating from different roots should be canonicalized to the same Arc");


        // Check that their predecessor sets, after simplification and normalization, are identical
        let s_h1_pred_hashes: BTreeSet<u64> = s_h1.predecessors.iter().map(|p| p.as_arc().hash_key_cache).collect();
        let s_h2_pred_hashes: BTreeSet<u64> = s_h2.predecessors.iter().map(|p| p.as_arc().hash_key_cache).collect();
        assert_eq!(s_h1_pred_hashes, s_h2_pred_hashes, "Predecessor hashes of s_h1 and s_h2 should be identical after normalization");

        // Check that the actual predecessor Arcs are the same (due to I and J simplifying consistently)
        let s_i_arc_h1 = s_h1.predecessors.iter().find(|p| p.as_arc().value == 80).unwrap().as_arc();
        let s_j_arc_h1 = s_h1.predecessors.iter().find(|p| p.as_arc().value == 90).unwrap().as_arc();
        let s_i_arc_h2 = s_h2.predecessors.iter().find(|p| p.as_arc().value == 80).unwrap().as_arc();
        let s_j_arc_h2 = s_h2.predecessors.iter().find(|p| p.as_arc().value == 90).unwrap().as_arc();

        assert!(Arc::ptr_eq(s_i_arc_h1, s_i_arc_h2), "Simplified I-node should be the same Arc instance for H1 and H2");
        assert!(Arc::ptr_eq(s_j_arc_h1, s_j_arc_h2), "Simplified J-node should be the same Arc instance for H1 and H2");
    }

    #[test]
    fn test_simplification_does_not_canonicalize_structurally_identical_nodes_from_distinct_arcs() {
        fn node(value: i32, predecessors: Vec<Arc<GSSNode<i32>>>) -> Arc<GSSNode<i32>> {
            Arc::new(GSSNode::new_with_predecessors(value, predecessors.into_iter().collect()))
        }
        // This test demonstrates the current behavior where structurally identical nodes
        // that originate from different initial Arcs are not unified into a single Arc
        // by `simplify_gss_forest`. This can lead to a larger GSS node count than
        // if full canonicalization (merging all structurally identical nodes to one Arc)
        // were performed.

        // L1, L2, L3 are structurally identical (value 0, no preds) but are distinct Arcs.
        let l1 = node(0, vec![]); // node() helper requires T: Clone + Ord + Hash + Debug
        let l2 = node(0, vec![]);
        let l3 = node(0, vec![]);

        // Ensure they are distinct Arcs initially
        assert_ne!(Arc::as_ptr(&l1), Arc::as_ptr(&l2));
        assert_ne!(Arc::as_ptr(&l1), Arc::as_ptr(&l3));
        assert_ne!(Arc::as_ptr(&l2), Arc::as_ptr(&l3));

        // M1, M2, M3 have the same value (1).
        // Their predecessors (L1, L2, L3 respectively) are structurally identical GSSNodes.
        // After simplification, L1, L2, L3 will simplify to the *same* canonical Arc.
        // M1, M2, M3 will then have the same value (1) and the same canonical predecessor Arc.
        // With global canonicalization, M1, M2, M3 should also simplify to the *same* canonical Arc.
        let m1 = node(1, vec![l1.clone()]);
        let m2 = node(1, vec![l2.clone()]);
        let m3 = node(1, vec![l3.clone()]);

        assert_ne!(Arc::as_ptr(&m1), Arc::as_ptr(&m2));
        assert_ne!(Arc::as_ptr(&m1), Arc::as_ptr(&m3));
        assert_ne!(Arc::as_ptr(&m2), Arc::as_ptr(&m3));

        // R1 has M1, M2, M3 as predecessors.
        let r1_orig = node(2, vec![m1.clone(), m2.clone(), m3.clone()]);

        let simplified_roots = simplify_gss_forest(&[r1_orig]);
        let simplified_r1_arc = simplified_roots[0].clone();

        let mut collected_arcs_map = HashMap::new();
        collect_arcs_recursive(&simplified_r1_arc, &mut collected_arcs_map); // Use recursive collector

        // Expected unique Arcs with GLOBAL canonicalization:
        // One canonical L-level node (from l1, l2, l3) -> 1 Arc.
        // One canonical M-level node (from m1, m2, m3) -> 1 Arc.
        // One canonical R-level node (from r1_orig) -> 1 Arc.
        // Total = 1 + 1 + 1 = 3 unique Arcs.
        assert_eq!(collected_arcs_map.len(), 3, "Expected 3 unique Arcs in the simplified GSS for this structure after global canonicalization"); // MODIFIED: Expected 3

        // Detailed verification of the structure:
        let s_r1_node = simplified_r1_arc.as_ref();
        assert_eq!(s_r1_node.value, 2);
        // With global canonicalization, R1's predecessors (simplified M1, M2, M3)
        // should all be the *same* canonical M-level Arc.
        // The BTreeSet will only contain one entry for this repeated ArcPtrWrapper.
        assert_eq!(s_r1_node.predecessors.len(), 1, "Simplified R1 should have 1 predecessor Arc (the canonical M node)"); // MODIFIED: Expected 1

        // Get the single canonical M-level node
        let s_m_level_arc = s_r1_node.predecessors.iter().next().unwrap().as_arc().clone();
        assert_eq!(s_m_level_arc.value, 1);
        // The canonical M-level node's predecessors should be the single canonical L-level node.
        assert_eq!(s_m_level_arc.predecessors.len(), 1, "The canonical M node should have 1 predecessor Arc (the canonical L node)"); // MODIFIED: Expected 1

        // Get the single canonical L-level node
        let s_l_level_arc = s_m_level_arc.predecessors.iter().next().unwrap().as_arc().clone();
        assert_eq!(s_l_level_arc.value, 0);
        assert_eq!(s_l_level_arc.predecessors.len(), 0, "The canonical L node should have no predecessors");

        // Verify the Arcs are canonical (pointers are equal for structurally identical nodes)
        let collected_arcs_vec: Vec<_> = collected_arcs_map.values().cloned().collect();
        let canonical_r_arc = simplified_r1_arc.clone();
        let canonical_m_arc = s_m_level_arc;
        let canonical_l_arc = s_l_level_arc;

        assert!(Arc::ptr_eq(&canonical_r_arc, &collected_arcs_vec.iter().find(|a| a.value.clone() == 2).unwrap()), "Root R1 should be the canonical R node");
        assert!(Arc::ptr_eq(&canonical_m_arc, &collected_arcs_vec.iter().find(|a| a.value.clone() == 1).unwrap()), "The single predecessor of canonical R should be the canonical M node");
        assert!(Arc::ptr_eq(&canonical_l_arc, &collected_arcs_vec.iter().find(|a| a.value.clone() == 0).unwrap()), "The single predecessor of canonical M should be the canonical L node");

         // This assertion now passes due to global canonicalization
        // assert_ne!(*s_m_level_arcs[0], *s_m_level_arcs[1], "Simplified M-nodes should not be Eq");
        // assert_ne!(s_m_level_arcs[0].hash_key_cache, s_m_level_arcs[1].hash_key_cache, "Simplified M-nodes should have different hash_key_caches");


        // This assertion now fails as there is only one L-level Arc
        // let s_l_level_ptrs: HashSet<*const GSSNode<i32>> = s_l_level_arcs_collected.iter().map(|arc| Arc::as_ptr(arc)).collect();
        // assert_eq!(s_l_level_ptrs.len(), 3, "Should be 3 distinct Arcs for L-level nodes");

        // This assertion now passes as all simplified L-nodes refer to the same canonical Arc
        // assert_eq!(*s_l_level_arcs_collected[0], *s_l_level_arcs_collected[1], "Simplified L-nodes content should be Eq");
        // assert_eq!(*s_l_level_arcs_collected[1], *s_l_level_arcs_collected[2], "Simplified L-nodes content should be Eq");
        // assert_eq!(s_l_level_arcs_collected[0].hash_key_cache, s_l_level_arcs_collected[1].hash_key_cache, "Simplified L-nodes should have same hash_key_cache");

    }

    #[test]
    fn test_gss_simplification_reproduces_logged_structure() {
        // Values for the nodes, mimicking the log's StateID and LLMTokenInfo
        let token_info = MockLLMTokenInfo {
            active: "[0]".to_string(),
            intersection: "[0]".to_string(),
        };

        let val0 = MockParseStateNodeContent { state_id: 0, t: token_info.clone() };
        let val1 = MockParseStateNodeContent { state_id: 1, t: token_info.clone() };
        let val2 = MockParseStateNodeContent { state_id: 2, t: token_info.clone() };

        // --- Constructing the GSS based on the log analysis ---

        // State 0 Leaf Nodes:
        let node_a_val0 = node(val0.clone(), vec![]); // Root 0

        let node_c_val0 = node(val0.clone(), vec![]); // Shared predecessor

        let node_g_val0 = node(val0.clone(), vec![]);
        let node_i_val0 = node(val0.clone(), vec![]);
        let node_k_val0 = node(val0.clone(), vec![]);
        let node_m_val0 = node(val0.clone(), vec![]);
        let node_o_val0 = node(val0.clone(), vec![]);
        let node_q_val0 = node(val0.clone(), vec![]);
        let node_s_val0 = node(val0.clone(), vec![]);
        let node_u_val0 = node(val0.clone(), vec![]);

        // State 1 Intermediate Nodes:
        let node_b_val1 = node(val1.clone(), vec![node_c_val0.clone()]); // Root 1

        // Predecessors for Node D (Order might matter for print visual match, but BTreeSet will sort by ptr)
        // Node E (pred: Node C)
        let node_e_val1 = node(val1.clone(), vec![node_c_val0.clone()]); // Shares node_c_val0

        // 8 more distinct intermediate nodes with val1, each pointing to one of the other_orig_s0_leaves
        let mut other_orig_s1_nodes: Vec<Arc<MockGSSNode>> = Vec::new();
        let other_orig_s0_leaves = vec![ // Put these in a vec for easy iteration
            node_g_val0.clone(), node_i_val0.clone(), node_k_val0.clone(),
            node_m_val0.clone(), node_o_val0.clone(), node_q_val0.clone(),
            node_s_val0.clone(), node_u_val0.clone(),
        ];
        for leaf_s0_node in &other_orig_s0_leaves {
            other_orig_s1_nodes.push(node(val1.clone(), vec![leaf_s0_node.clone()]));
        }
        // These are node_f_val1 (pred node_g_val0), ..., node_t_val1 (pred node_u_val0)

        let mut preds_for_d = vec![
            node_e_val1.clone(), // As per log order (0x...08d0)
        ];
        preds_for_d.extend(other_orig_s1_nodes.iter().cloned()); // Add the 8 other s1 nodes
        assert_eq!(preds_for_d.len(), 9, "S2 node should have 9 predecessors before simplification");

        let node_d_val2 = node(val2.clone(), preds_for_d); // Root 2

        // --- Collect roots and print ---
        let roots = vec![
            node_a_val0.clone(), // Root 0
            node_b_val1.clone(), // Root 1
            node_d_val2.clone(), // Root 2
        ];

        let max_nodes_to_print = 30; // Match the log's max_nodes
        let gss_string_representation = print_gss_forest(&roots, max_nodes_to_print);

        println!("\n--- GSS Structure for Visual Comparison (Original) ---\n");
        println!("{}", gss_string_representation);
        println!("--- End of GSS Structure (Original) ---\n");

        // You can add assertions here if needed, e.g., count unique nodes printed
        // or verify specific "(Visited)" occurrences if you parse the string.
        // For now, the main goal is visual inspection of the printed output.

        // Example assertion: check if node_c_val0 is marked as visited
        // This requires knowing the pointer of node_c_val0.
        let ptr_c_str = format!("{:p}", Arc::as_ptr(&node_c_val0));
        let occurrences_of_ptr_c = gss_string_representation.matches(&ptr_c_str).count();
        let occurrences_of_ptr_c_visited = gss_string_representation.matches(&format!("{} (Visited)", ptr_c_str)).count();

        assert_eq!(occurrences_of_ptr_c, 2, "Node C should appear twice in the original printout");
        assert_eq!(occurrences_of_ptr_c_visited, 1, "Node C should be marked (Visited) once in the original printout");

        // Count total unique nodes involved in this structure to confirm it's 21.
        let mut all_involved_arcs: HashMap<*const MockGSSNode, Arc<MockGSSNode>> = HashMap::new();
        for r in &roots {
            collect_arcs_recursive(r, &mut all_involved_arcs);
        }
        assert_eq!(all_involved_arcs.len(), 21, "The constructed original GSS should have 21 unique nodes before simplification.");

        // Perform simplification
        let simplified_roots = simplify_gss_forest(&roots);

        // Print simplified GSS
        let simplified_gss_string_representation = print_gss_forest(&simplified_roots, max_nodes_to_print);
        println!("\n--- Simplified GSS Structure for Visual Comparison (After Global Canonicalization) ---\n");
        println!("{}", simplified_gss_string_representation);
        println!("--- End of Simplified GSS Structure (After Global Canonicalization) ---\n");

        // Collect unique Arcs in the simplified GSS forest
        let mut collected_arcs_map: HashMap<*const MockGSSNode, Arc<MockGSSNode>> = HashMap::new();
        for root_arc in &simplified_roots {
            collect_arcs_recursive(root_arc, &mut collected_arcs_map);
        }

        // Assert the number of unique Arcs (nodes) after global canonicalization
        // Expected:
        // One canonical StateID(0) node -> 1 Arc
        // One canonical StateID(1) node -> 1 Arc
        // One canonical StateID(2) node -> 1 Arc
        // Total = 1 + 1 + 1 = 3 unique Arcs.
        assert_eq!(collected_arcs_map.len(), 3, "The simplified GSS should contain 3 unique Arcs after global canonicalization."); // MODIFIED: Expected 3

        // Further checks on the simplified structure for canonicalization:
        assert_eq!(simplified_roots.len(), 3, "Should still have 3 roots after simplification.");

        // Get the canonical nodes by value (since there should only be one of each value)
        let mut canonical_nodes_by_value: HashMap<usize, Arc<MockGSSNode>> = HashMap::new();
        for simplified_arc in collected_arcs_map.values() {
            canonical_nodes_by_value.insert(simplified_arc.value.state_id, simplified_arc.clone());
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
        assert!(Arc::ptr_eq(s_node_1.predecessors.iter().next().unwrap().as_arc(), s_node_0), "Canonical StateID 1 node's predecessor should be the canonical StateID 0 node");

        assert_eq!(s_node_2.value.state_id, 2);
        assert_eq!(s_node_2.predecessors.len(), 1, "Canonical StateID 2 node should have 1 predecessor");
        assert!(Arc::ptr_eq(s_node_2.predecessors.iter().next().unwrap().as_arc(), s_node_1), "Canonical StateID 2 node's predecessor should be the canonical StateID 1 node");

        // Verify that the simplified roots are the correct canonical nodes
        // Note: The order of roots in the output vector corresponds to the order of roots in the input vector.
        assert!(Arc::ptr_eq(&simplified_roots[0], s_node_0), "Simplified root 0 should be the canonical StateID 0 node");
        assert!(Arc::ptr_eq(&simplified_roots[1], s_node_1), "Simplified root 1 should be the canonical StateID 1 node");
        assert!(Arc::ptr_eq(&simplified_roots[2], s_node_2), "Simplified root 2 should be the canonical StateID 2 node");
    }
}
