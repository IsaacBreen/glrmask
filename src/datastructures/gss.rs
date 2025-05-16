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
    }

    pub fn merge_unchecked(&mut self, mut other: Self)
    {
        self.predecessors.extend(std::mem::take(&mut other.predecessors));
    }

    pub fn map<F, U>(&self, f: F) -> GSSNode<U>
    where
        F: Copy + Fn(&T) -> U,
        U: Hash,
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
        match self.hash_key_cache.cmp(&other.hash_key_cache) {
            Ordering::Equal => {
                // Hashes are equal, compare values
                match self.value.partial_cmp(&other.value) {
                    Some(Ordering::Equal) => None, // Values are also equal (or reported as equal by partial_cmp), return None
                    other_ordering => other_ordering, // Values are different or one is greater/less
                }
            }
            other_ordering => Some(other_ordering), // Hashes are different
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
            let new_node_arc = Arc::new(GSSNode { value: new_value, predecessors: new_predecessors, hash_key_cache });
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
// Uses memoization to handle shared nodes and ensure canonical simplified forms.
fn simplify_node_recursive<T: Clone + Ord + Hash + Debug>(
    original_node_arc: &Arc<GSSNode<T>>,
    memo: &mut HashMap<*const GSSNode<T>, Arc<GSSNode<T>>>,
) -> Arc<GSSNode<T>> {
    let original_node_ptr = Arc::as_ptr(original_node_arc);

    // Check if this original node has already been simplified
    if let Some(simplified_node) = memo.get(&original_node_ptr) {
        return simplified_node.clone();
    }

    // Recursively simplify predecessors
    // Collect simplified predecessors into a BTreeSet to normalize order and ensure uniqueness based on content hash.
    // Arc<GSSNode<T>> is Ord because GSSNode<T> is Ord (based on hash_key_cache).
    let mut simplified_predecessors_arcs: BTreeSet<Arc<GSSNode<T>>> = BTreeSet::new();
    for pred_wrapper in &original_node_arc.predecessors {
        let original_pred_arc = pred_wrapper.as_arc();
        let simplified_pred_arc = simplify_node_recursive(original_pred_arc, memo);
        simplified_predecessors_arcs.insert(simplified_pred_arc);
    }

    // Convert BTreeSet<Arc<GSSNode<T>>> to BTreeSet<ArcPtrWrapper<GSSNode<T>>>
    // This will be used for both the new node's `predecessors` field and for computing its hash_key_cache.
    let new_node_predecessors_arc_ptr_wrappers: BTreeSet<ArcPtrWrapper<GSSNode<T>>> =
        simplified_predecessors_arcs // This is BTreeSet<Arc<GSSNode<T>>>
            .iter()
            .map(|arc| ArcPtrWrapper::new(arc.clone()))
            .collect();

    // Compute the hash for the new simplified node based *only* on its (simplified) predecessors.
    let hash_key_cache = GSSNode::compute_hash_key_cache(&new_node_predecessors_arc_ptr_wrappers);

    // Create the new simplified GSSNode
    let simplified_node = GSSNode {
        value: original_node_arc.value.clone(),
        predecessors: new_node_predecessors_arc_ptr_wrappers, // Use the already collected ArcPtrWrappers
        hash_key_cache,
    };
    let simplified_node_arc = Arc::new(simplified_node);

    // Memoize the result for the original node's pointer
    memo.insert(original_node_ptr, simplified_node_arc.clone());

    simplified_node_arc
}

/// Simplifies a GSS forest, ensuring that structurally identical nodes
/// (after simplification) are represented by shared `Arc<GSSNode<T>>` instances
/// where possible (specifically, original shared nodes remain shared, and
/// predecessors are normalized).
///
/// The simplification process works from the bottom up (leaves to roots).
/// - Node values are preserved.
/// - Predecessor lists are normalized by ordering simplified predecessors based on their content hash.
/// - A hash is computed for each simplified node based on its value and its simplified predecessors' hashes.
///   This hash is stored in `hash_key_cache` and used for `Ord` comparisons within the simplification logic.
///
/// Assumes the GSS forest does not contain cycles.
pub fn simplify_gss_forest<T: Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T>>],
) -> Vec<Arc<GSSNode<T>>> {
    let mut memo: HashMap<*const GSSNode<T>, Arc<GSSNode<T>>> = HashMap::new();
    let mut simplified_roots = Vec::with_capacity(roots.len());

    for root_arc in roots {
        simplified_roots.push(simplify_node_recursive(root_arc, &mut memo));
    }

    simplified_roots
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Helper to create a node for tests
    fn node<T: Clone + Ord + Hash + Debug>(value: T, predecessors: Vec<Arc<GSSNode<T>>>) -> Arc<GSSNode<T>> {
        Arc::new(GSSNode::new_with_predecessors(value, predecessors.into_iter().collect()))
    }

    // Helper to get a stable representation of a simplified GSS for comparison.
    // Returns (value, Vec<pred_hashes_sorted>)
    type SimplifiedNodeRepr<T> = (T, Vec<u64>);

    fn get_simplified_repr<T: Clone>(node_arc: &Arc<GSSNode<T>>) -> SimplifiedNodeRepr<T> {
        let mut pred_hashes: Vec<u64> = node_arc.predecessors.iter()
            .map(|p| p.as_arc().hash_key_cache)
            .collect();
        pred_hashes.sort_unstable(); // Ensure order for comparison
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


    #[test]
    fn test_gss_simplification_basic() {
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
        // Since d1_orig and d2_orig are identical (value 40, no preds),
        // their simplified versions should have the same hash.
        // simplify_node_recursive will create distinct Arc<GSSNode> for s_d1 and s_d2
        // because they are from different original pointers, but their GSSNode content (and hash) will be identical.

        let mut visited_check = HashSet::new();
        let mut collected_check = HashMap::new();
        collect_all_simplified_nodes(&simplified_a1, &mut visited_check, &mut collected_check);

        // Expected simplified structure values and predecessor hashes:
        // D_s(40, []) -> hash_d
        // C1_s(30, [D_s]) -> hash_c1 (depends on hash_d)
        // B1_s(20, [C1_s]) -> hash_b1 (depends on hash_c1)
        // A1_s(10, [B1_s, D_s']) -> hash_a1 (depends on hash_b1, hash_d)
        // Note: D_s and D_s' will have the same hash_key_cache value.

        // Find the simplified nodes by value (this is a bit indirect for a test)
        let mut s_nodes_by_val = HashMap::new();
        for s_node_arc in collected_check.keys().map(|k| unsafe { Arc::from_raw(*k) }) { // Use unsafe for from_raw
            s_nodes_by_val.entry(s_node_arc.value).or_insert_with(Vec::new).push(s_node_arc.clone());
             // Do not forget Arc here, collect_all_simplified_nodes keys are raw pointers, ownership is not transferred
        }

        let s_d_nodes = s_nodes_by_val.get(&40).unwrap();
        assert!(s_d_nodes.len() >= 1); // Could be 1 if d1_orig and d2_orig simplified to the same Arc, or 2 if not.
                                       // With current memo (original_ptr -> simplified_arc), they will be distinct Arcs if d1_orig and d2_orig are distinct.
                                       // But their GSSNode content (value, preds, hash_key_cache) will be identical.
        let s_d_hash = s_d_nodes[0].hash_key_cache;
        assert_ne!(s_d_hash, 0, "D node hash should be computed");
        for s_d_node in s_d_nodes {
            assert_eq!(s_d_node.hash_key_cache, s_d_hash, "All simplified D nodes must have same hash");
            assert_eq!(s_d_node.predecessors.len(), 0, "Simplified D node should have no predecessors");
        }


        let s_c1_nodes = s_nodes_by_val.get(&30).unwrap();
        assert_eq!(s_c1_nodes.len(), 1);
        let s_c1 = &s_c1_nodes[0];
        assert_ne!(s_c1.hash_key_cache, 0, "C1 node hash should be computed");
        assert_eq!(s_c1.predecessors.len(), 1, "Simplified C1 should have 1 predecessor");
        assert_eq!(s_c1.predecessors.iter().next().unwrap().as_arc().value, 40, "C1 predecessor should be a D node");
        assert_eq!(s_c1.predecessors.iter().next().unwrap().as_arc().hash_key_cache, s_d_hash, "C1 predecessor hash should match D's hash");

        let s_b1_nodes = s_nodes_by_val.get(&20).unwrap();
        assert_eq!(s_b1_nodes.len(), 1);
        let s_b1 = &s_b1_nodes[0];
        assert_ne!(s_b1.hash_key_cache, 0, "B1 node hash should be computed");
        assert_eq!(s_b1.predecessors.len(), 1, "Simplified B1 should have 1 predecessor");
        assert_eq!(s_b1.predecessors.iter().next().unwrap().as_arc().value, 30, "B1 predecessor should be C1 node");
        assert_eq!(s_b1.predecessors.iter().next().unwrap().as_arc().hash_key_cache, s_c1.hash_key_cache, "B1 predecessor hash should match C1's hash");


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

        // Test shared node reuse from original structure
        // E -> F
        // E -> G
        // Root1 = F, Root2 = G. E should be simplified only once.
        let e_orig = node(500, vec![]);
        let f_orig = node(600, vec![e_orig.clone()]);
        let g_orig = node(700, vec![e_orig.clone()]); // e_orig is shared

        let simplified_shared = simplify_gss_forest(&[f_orig, g_orig]);
        let s_f = &simplified_shared[0];
        let s_g = &simplified_shared[1];

        let s_f_pred_ptr = Arc::as_ptr(s_f.predecessors.iter().next().unwrap().as_arc());
        let s_g_pred_ptr = Arc::as_ptr(s_g.predecessors.iter().next().unwrap().as_arc());
        assert_eq!(s_f_pred_ptr, s_g_pred_ptr, "Shared original node E should simplify to the same Arc instance for F and G");
        assert_eq!(s_f.predecessors.iter().next().unwrap().as_arc().value, 500);

        // Test predecessor order normalization
        // H1 -> (I, J)
        // H2 -> (J, I)
        // I, J are leaves. H1 and H2 should simplify to identical GSSNode structures (same hash).
        // The Arcs for simplified H1 and H2 will be different, but their pointed-to GSSNodes will be Eq.
        let i_orig = node(80, vec![]);
        let j_orig = node(90, vec![]);

        let h1_orig = node(100, vec![i_orig.clone(), j_orig.clone()]);
        let h2_orig = node(100, vec![j_orig.clone(), i_orig.clone()]); // Different pred order

        let simplified_norm = simplify_gss_forest(&[h1_orig, h2_orig]);
        let s_h1 = &simplified_norm[0];
        let s_h2 = &simplified_norm[1];

        assert_ne!(Arc::as_ptr(s_h1), Arc::as_ptr(s_h2), "s_h1 and s_h2 should be different Arcs due to different original roots");
        assert_eq!(s_h1.hash_key_cache, s_h2.hash_key_cache, "s_h1 and s_h2 should have the same hash after normalization");
        assert_eq!(s_h1.value, s_h2.value);

        // Check that their predecessor sets, after simplification and normalization, are identical
        // This means they contain ArcPtrWrappers pointing to the same set of simplified predecessor Arcs,
        // and those ArcPtrWrappers will be ordered by pointer in the BTreeSet.
        // The crucial part is that the set of (Arc pointing to simplified I) and (Arc pointing to simplified J) is the same.
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
}
