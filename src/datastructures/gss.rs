use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added
use std::collections::BTreeMap as StdMap; // Added for derive macro pattern


// Type alias for the canonicalization cache key
type NodeCacheKey<T> = (T, BTreeSet<Arc<GSSNode<T>>>);
// Type alias for the canonicalization cache
pub type NodeCache<T> = HashMap<NodeCacheKey<T>, Arc<GSSNode<T>>>;

// Helper function to compute a node's hash_key_cache.
// This is used by both canonical and non-canonical node creation.
fn compute_internal_hash_key<T: Hash>(value: &T, predecessors: &BTreeSet<Arc<GSSNode<T>>>) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    // The BTreeSet ensures predecessors are iterated in a canonical order (by Arc pointer address).
    for pred_arc in predecessors {
        // Use the predecessor's own hash_key_cache.
        // This makes the hash recursive and dependent on the structure.
        pred_arc.hash_key_cache.hash(&mut hasher);
    }
    hasher.finish()
}

#[derive(Debug, Clone)] // Removed PartialEq, Eq, PartialOrd, Ord, Hash for manual impl
pub struct GSSNode<T> {
    pub value: T,
    predecessors: BTreeSet<Arc<GSSNode<T>>>, // Changed from ArcPtrWrapper
    hash_key_cache: u64,
}

impl<T: JSONConvertible> JSONConvertible for GSSNode<T> {
    fn to_json(&self) -> JSONNode {
        // WARNING: Naive serialization. Does not handle cycles or shared structure well.
        // Will lead to infinite recursion for cycles.
        todo!("GSSNode to_json: Complex graph structure, requires advanced serialization strategy.")
    }

    fn from_json(_node: JSONNode) -> Result<Self, String> {
        todo!("GSSNode from_json: Complex graph structure, requires advanced deserialization strategy.")
    }
}


// Methods for creating non-canonical GSSNode instances (original API style)
impl<T: Hash> GSSNode<T> {
    pub fn new(value: T) -> Self {
        let predecessors = BTreeSet::new();
        let hash_key_cache = compute_internal_hash_key(&value, &predecessors);
        Self {
            value,
            predecessors,
            hash_key_cache,
        }
    }

    pub fn new_with_predecessors(value: T, predecessors: BTreeSet<Arc<Self>>) -> Self {
        let hash_key_cache = compute_internal_hash_key(&value, &predecessors);
        Self {
            value,
            predecessors,
            hash_key_cache,
        }
    }
}

// Methods involving canonicalization or requiring more bounds (Ord, Clone, Debug for cache keys)
impl<T: Clone + Ord + Hash + Debug> GSSNode<T> {
    /// Internal method to get/create a canonical Arc<GSSNode<T>>.
    fn get_canonical(
        value: T,
        predecessors: BTreeSet<Arc<Self>>,
        cache: &mut NodeCache<T>,
    ) -> Arc<Self> {
        let key = (value, predecessors); // value and predecessors are moved into key

        if let Some(existing_node) = cache.get(&key) {
            return existing_node.clone();
        }

        // Not found, create new. Key owns value and predecessors now.
        let node_value_for_struct = key.0.clone();
        let node_predecessors_for_struct = key.1.clone();

        // hash_key_cache for the new canonical node.
        let hash_key_cache = compute_internal_hash_key(&node_value_for_struct, &node_predecessors_for_struct);

        let new_node_arc = Arc::new(GSSNode {
            value: node_value_for_struct,
            predecessors: node_predecessors_for_struct,
            hash_key_cache,
        });

        // Insert the original key (which moved its components) and the new Arc into the cache.
        cache.insert(key, new_node_arc.clone());
        new_node_arc
    }

    pub fn new_canonical(value: T, cache: &mut NodeCache<T>) -> Arc<Self> {
        Self::get_canonical(value, BTreeSet::new(), cache)
    }

    pub fn new_with_predecessors_canonical(value: T, predecessors: BTreeSet<Arc<Self>>, cache: &mut NodeCache<T>) -> Arc<Self> {
        Self::get_canonical(value, predecessors, cache)
    }

    pub fn from_iter_canonical<I>(iter: I, cache: &mut NodeCache<T>) -> Arc<Self>
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter_val = iter.into_iter(); // Renamed iter
        let first_val = iter_val.next().expect("from_iter_canonical requires at least one element"); // Use iter_val
        let mut root = Self::new_canonical(first_val, cache);
        for value in iter_val { // Use iter_val
            root = Self::push_onto_canonical(root, value, cache);
        }
        root
    }

    pub fn push_onto_canonical(
        current_stack_top: Arc<Self>,
        value: T,
        cache: &mut NodeCache<T>,
    ) -> Arc<Self> {
        let mut predecessors = BTreeSet::new();
        predecessors.insert(current_stack_top);
        Self::get_canonical(value, predecessors, cache)
    }

    pub fn merge_canonical(
        node1_arc: Arc<Self>,
        node2_arc: Arc<Self>,
        cache: &mut NodeCache<T>,
    ) -> Result<Arc<Self>, &'static str>
    where
        T: PartialEq, // Existing bound on GSSNode<T> is Clone + Ord + Hash + Debug
    {
        if node1_arc.value != node2_arc.value {
            return Err("Cannot merge nodes with different values");
        }
        let mut merged_predecessors = node1_arc.predecessors.clone();
        for pred_arc in &node2_arc.predecessors {
            merged_predecessors.insert(pred_arc.clone());
        }
        Ok(Self::get_canonical(node1_arc.value.clone(), merged_predecessors, cache))
    }

    pub fn map_canonical<F, U>(
        self_arc: Arc<Self>,
        f: F,
        cache_u: &mut NodeCache<U>, // Cache for GSSNode<U>
    ) -> Arc<GSSNode<U>>
    where
        F: Copy + Fn(&T) -> U,
        U: Clone + Ord + Hash + Debug, // Bounds for the new node type U
    {
        let new_value = f(&self_arc.value);
        let new_predecessors: BTreeSet<Arc<GSSNode<U>>> = self_arc.predecessors.iter()
            .map(|pred_arc_t| {
                GSSNode::map_canonical(pred_arc_t.clone(), f, cache_u)
            })
            .collect();
        GSSNode::<U>::get_canonical(new_value, new_predecessors, cache_u)
    }
}

// Public methods consistent with the original API (mostly non-canonical)
impl<T> GSSNode<T> {
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Ord + Hash, // Needed for push -> new -> compute_internal_hash_key
    {
        let mut iter_val = iter.into_iter(); // Renamed iter
        let mut root = Self::new(iter_val.next().unwrap()); // Uses non-canonical new // Use iter_val
        for value in iter_val { // Use iter_val
            root = root.push(value); // Uses non-canonical push
        }
        root
    }

    pub fn push(self, value: T) -> Self
    where T: Ord + Hash
    {
        let mut new_node_predecessors = BTreeSet::new();
        // Arc::new(self) creates a new Arc, does not use canonical cache.
        new_node_predecessors.insert(Arc::new(self));
        Self::new_with_predecessors(value, new_node_predecessors)
    }

    pub fn pop(&self) -> Vec<Arc<Self>> {
        self.predecessors.iter().cloned().collect()
    }

    pub fn popn(&self, n: usize) -> Vec<Arc<Self>>
    where
        T: Clone + Hash, // Clone for self.clone(), Hash for recursive popn if it were on GSSNode
    {
        if n == 0 {
            // To return Vec<Arc<Self>>, we need to wrap self in an Arc.
            // This creates a new Arc, not necessarily canonical.
            return vec![Arc::new(self.clone())];
        }

        let mut result = Vec::new();
        // To avoid infinite loops in cyclic graphs if popn were to be called on non-Arc GSSNode directly
        // and to de-duplicate paths, we'd need a seen set.
        // Since predecessors are Arcs, we operate on them.
        let mut seen_arcs_for_this_call: HashSet<*const GSSNode<T>> = HashSet::new();

        for predecessor_arc in &self.predecessors {
            // predecessor_arc is &Arc<GSSNode<T>>.
            // Call popn on the GSSNode content of the Arc.
            for node_arc_from_popn in predecessor_arc.as_ref().popn(n - 1) {
                 let ptr = Arc::as_ptr(&node_arc_from_popn);
                 if seen_arcs_for_this_call.insert(ptr) {
                    result.push(node_arc_from_popn);
                }
            }
        }
        result
    }

    pub fn peek(&self) -> &T {
        &self.value
    }

    pub fn value_mut(&mut self) -> &mut T {
        // Warning: Mutating value will invalidate hash_key_cache and break canonicalization
        // if this node was part of a canonical set. User must re-calculate hash or re-intern.
        &mut self.value
    }

    pub fn flatten(&self) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        let mut result = Vec::new();
        // let mut stack: Vec<T> = Vec::new(); // Unused
        // For flatten on &self, we need to Arc self to put on stack if it's a root of a path.
        // However, flatten explores predecessors which are already Arcs.
        // The initial call path starts from `self`.

        // To handle the initial `self` correctly without Arcing it unnecessarily if it's never a predecessor itself:
        // let initial_path = vec![self.value.clone()]; // Unused

        if self.predecessors.is_empty() {
            result.push(vec![self.value.clone()]); // Path for a root node
        } else {
            // Each predecessor_arc starts a new exploration from that point.
            // The path passed down should be the path *to* that predecessor.
            // This is more complex than the Arc-based flatten.
            // Let's use a structure that tracks the GSSNode reference and current path.
            // (node_ref, path_to_node_ref_value)
            let mut q: VecDeque<(&GSSNode<T>, Vec<T>)> = VecDeque::new();
            // For the initial call on `self`, the path leading to it is empty.
            // The path will be built by prepending current node's value.
            q.push_back((self, Vec::new()));

            while let Some((current_node, path_so_far)) = q.pop_front() { // Renamed current_path_values
                let mut new_path = vec![current_node.value.clone()]; // Start new path with current value
                new_path.extend(path_so_far); // Prepend current value to path from successor

                if current_node.predecessors.is_empty() {
                    result.push(new_path); // This is a full path from a leaf up to `self`
                } else {
                    for pred_arc in &current_node.predecessors {
                        q.push_back((pred_arc.as_ref(), new_path.clone()));
                    }
                }
            }
            // Paths are built in reverse (leaf to root), so reverse them at the end.
            for path in &mut result {
                path.reverse();
            }
        }
        result
    }


    pub fn flatten_bulk(nodes: &[Self]) -> Vec<Vec<T>>
    where
        T: Clone + Hash, // Hash needed for GSSNode methods called by flatten
    {
        nodes.iter().flat_map(|node| node.flatten()).collect()
    }

    pub fn merge(&mut self, mut other: Self)
    where
        T: Ord + Hash, // Hash for re-calculating hash_key_cache
    {
        assert!(self.value == other.value); // Requires T: PartialEq, which is implied by Ord
        self.predecessors.append(&mut other.predecessors);
        self.hash_key_cache = compute_internal_hash_key(&self.value, &self.predecessors);
    }

    pub fn merge_unchecked(&mut self, mut other: Self)
    where T: Ord + Hash // Hash for re-calculating hash_key_cache
    {
        self.predecessors.append(&mut other.predecessors);
        self.hash_key_cache = compute_internal_hash_key(&self.value, &self.predecessors);
    }

    pub fn map<F, U>(&self, f: F) -> GSSNode<U>
    where
        F: Copy + Fn(&T) -> U,
        U: Ord + Hash, // For GSSNode<U> and compute_internal_hash_key
        T: Hash, // For self
    {
        let new_value = f(&self.value);
        let new_predecessors_arcs: BTreeSet<Arc<GSSNode<U>>> = self.predecessors.iter()
            .map(|pred_arc_t| { // pred_arc_t is &Arc<GSSNode<T>>
                // Recursively call map on the GSSNode content. Result is GSSNode<U>.
                // Wrap it in an Arc for the new predecessor set.
                Arc::new(pred_arc_t.as_ref().map(f))
            })
            .collect();
        // Create a new non-canonical GSSNode<U>.
        GSSNode::<U>::new_with_predecessors(new_value, new_predecessors_arcs)
    }
}

impl<T> Drop for GSSNode<T> {
    fn drop(&mut self) {
        let predecessors_to_process_further = std::mem::take(&mut self.predecessors);
        let mut worklist: Vec<Arc<GSSNode<T>>> = predecessors_to_process_further.into_iter().collect();

        while let Some(node_arc) = worklist.pop() {
            if Arc::strong_count(&node_arc) == 1 { // Check if we are the last owner before try_unwrap
                if let Ok(mut inner_node) = Arc::try_unwrap(node_arc) {
                    worklist.extend(std::mem::take(&mut inner_node.predecessors).into_iter());
                }
            }
            // Else: Arc is still shared, it will be dropped when its last Arc reference is dropped.
        }
    }
}

impl<T: Hash> Hash for GSSNode<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_key_cache.hash(state);
        // self.value.hash(state); // This was in user's last version.
                                // It's redundant if hash_key_cache already includes value's hash.
                                // compute_internal_hash_key *does* include value.hash().
    }
}

impl<T: Hash + PartialEq> PartialEq for GSSNode<T> {
    fn eq(&self, other: &Self) -> bool {
        if std::ptr::eq(self, other) { return true; } // Optimization for same instance
        self.hash_key_cache == other.hash_key_cache &&
        self.value == other.value &&
        self.predecessors == other.predecessors // Compares BTreeSet<Arc<_>> by Arc pointers
    }
}

impl<T: Hash + Eq> Eq for GSSNode<T> {}

impl<T: Hash + PartialOrd> PartialOrd for GSSNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if std::ptr::eq(self, other) { return Some(Ordering::Equal); } // Optimization for same instance
        match self.hash_key_cache.partial_cmp(&other.hash_key_cache) {
            Some(Ordering::Equal) => {
                match self.value.partial_cmp(&other.value) {
                    Some(Ordering::Equal) => self.predecessors.partial_cmp(&other.predecessors),
                    other_ordering => other_ordering,
                }
            }
            other_ordering => other_ordering,
        }
    }
}

impl<T: Hash + Ord> Ord for GSSNode<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        if std::ptr::eq(self, other) { return Ordering::Equal; } // Optimization for same instance
        self.hash_key_cache.cmp(&other.hash_key_cache)
            .then_with(|| self.value.cmp(&other.value))
            .then_with(|| self.predecessors.cmp(&other.predecessors))
    }
}


// GSSTrait remains largely as per original public API
// T: Clone + Hash (original bounds)
pub trait GSSTrait<T: Clone + Hash> {
    type Peek<'a> where T: 'a, Self: 'a;
    fn peek(&self) -> Self::Peek<'_>;
    // push returns GSSNode<T>, so it's non-canonical by default from trait.
    fn push(&self, value: T) -> GSSNode<T> where T: Ord;
    fn pop(&self) -> Vec<Arc<GSSNode<T>>>;
    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>>;
}

impl<T: Clone + Hash> GSSTrait<T> for GSSNode<T> {
    type Peek<'a> = &'a T where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        &self.value
    }

    fn push(&self, value: T) -> GSSNode<T> where T: Ord {
        // self is &GSSNode<T>. We need to clone it to own it for Arc::new.
        let self_owned_clone = self.clone();
        GSSNode::push(self_owned_clone, value) // Calls the inherent GSSNode::push
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        GSSNode::pop(self) // Calls inherent GSSNode::pop
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        GSSNode::popn(self, n) // Calls inherent GSSNode::popn
    }
}

impl<T: Clone + Hash> GSSTrait<T> for Arc<GSSNode<T>> {
    type Peek<'a> = &'a T where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        &self.value
    }

    fn push(&self, value: T) -> GSSNode<T> where T: Ord {
        // self is &Arc<GSSNode<T>>.
        let mut new_node = GSSNode::new(value); // Non-canonical new
        new_node.predecessors.insert(self.clone()); // Insert the Arc
        // Recompute hash_key_cache after modifying predecessors
        new_node.hash_key_cache = compute_internal_hash_key(&new_node.value, &new_node.predecessors);
        new_node
    }


    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().pop()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().popn(n)
    }
}

impl<T: Clone + Hash> GSSTrait<T> for Option<Arc<GSSNode<T>>> {
    type Peek<'a> = Option<&'a T> where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        self.as_ref().map(|node_arc| node_arc.peek())
    }

    fn push(&self, value: T) -> GSSNode<T> where T: Ord {
        match self {
            Some(arc_node) => arc_node.push(value), // Arc's GSSTrait push
            None => GSSNode::new(value), // Non-canonical new
        }
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node_arc| node_arc.pop()).unwrap_or_default()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node_arc| node_arc.popn(n)).unwrap_or_default()
    }
}

impl<T: Clone + Hash> GSSTrait<T> for Option<GSSNode<T>> {
    type Peek<'a> = Option<&'a T> where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        self.as_ref().map(|node| node.peek())
    }

    fn push(&self, value: T) -> GSSNode<T> where T: Ord {
        match self {
            Some(node) => node.push(value), // GSSNode's GSSTrait push
            None => GSSNode::new(value),
        }
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node| node.pop()).unwrap_or_default()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node| node.popn(n)).unwrap_or_default()
    }
}


// BulkMerge uses Ord for BTreeMap key, Hash for GSSNode::new_with_predecessors
pub trait BulkMerge<T> {
    fn bulk_merge(&mut self);
}

impl<T: Clone + Ord + Hash> BulkMerge<T> for Vec<Arc<GSSNode<T>>> {
    fn bulk_merge(&mut self) {
        let mut groups: BTreeMap<T, Vec<Arc<GSSNode<T>>>> = BTreeMap::new();
        for node_arc in self.drain(..) {
            groups.entry(node_arc.value.clone()).or_default().push(node_arc);
        }

        let mut new_merged_nodes = Vec::new();
        for (value, group_arcs) in groups {
            if group_arcs.is_empty() { continue; }
            if group_arcs.len() == 1 {
                new_merged_nodes.push(group_arcs.into_iter().next().unwrap());
                continue;
            }

            let mut merged_predecessors: BTreeSet<Arc<GSSNode<T>>> = BTreeSet::new();
            for node_arc_in_group in group_arcs {
                for pred_arc in &node_arc_in_group.predecessors {
                    merged_predecessors.insert(pred_arc.clone());
                }
            }
            // Create a new non-canonical node for the merged result.
            let merged_node = GSSNode::new_with_predecessors(value, merged_predecessors);
            new_merged_nodes.push(Arc::new(merged_node));
        }
        *self = new_merged_nodes;
    }
}

// Canonical version of bulk_merge
pub fn bulk_merge_canonical<T: Clone + Ord + Hash + Debug>(
    nodes: &mut Vec<Arc<GSSNode<T>>>,
    cache: &mut NodeCache<T>
) {
    let mut groups: BTreeMap<T, Vec<Arc<GSSNode<T>>>> = BTreeMap::new();
    for node_arc in nodes.drain(..) {
        groups.entry(node_arc.value.clone()).or_default().push(node_arc);
    }

    let mut new_merged_nodes = Vec::new();
    for (value, group_arcs) in groups {
        if group_arcs.is_empty() { continue; }

        let mut merged_predecessors: BTreeSet<Arc<GSSNode<T>>> = BTreeSet::new();
        for node_arc_in_group in group_arcs {
            for pred_arc in &node_arc_in_group.predecessors {
                merged_predecessors.insert(pred_arc.clone());
            }
        }
        let merged_node_arc = GSSNode::get_canonical(value, merged_predecessors, cache);
        new_merged_nodes.push(merged_node_arc);
    }
    *nodes = new_merged_nodes;
}


// prune_and_transform_roots: Assumed to produce canonical nodes as per last discussion.
// Requires T: Clone + Ord + Hash + Debug for GSSNode::get_canonical.
pub fn prune_and_transform_recursive_canonical<T: Clone + Ord + Hash + Debug>(
    node_arc: &Arc<GSSNode<T>>,
    closure: &impl Fn(&T) -> Option<(T, bool)>,
    memo: &mut HashMap<*const GSSNode<T>, Option<Arc<GSSNode<T>>>>,
    cache: &mut NodeCache<T>, // Cache for canonical node creation
) -> Option<Arc<GSSNode<T>>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }

    match closure(&node_arc.value) {
        None => {
            memo.insert(node_ptr, None);
            None
        }
        Some((new_value, continue_recursion)) => {
            let new_predecessors: BTreeSet<Arc<GSSNode<T>>>;
            if continue_recursion {
                let mut current_new_predecessors = BTreeSet::new();
                for pred_arc in &node_arc.predecessors { // pred_arc is &Arc<GSSNode<T>>
                    if let Some(new_pred_arc) = prune_and_transform_recursive_canonical(pred_arc, closure, memo, cache) {
                        current_new_predecessors.insert(new_pred_arc);
                    }
                }
                new_predecessors = current_new_predecessors;
            } else {
                // Stop recursion, keep original predecessors (which should be canonical if input was canonical)
                // or simplified predecessors if this is part of a larger simplification.
                // For safety, ensure predecessors are also passed through a canonicalization step if they weren't already.
                // However, if `continue_recursion` is false, the expectation is often to reuse existing structure below.
                // The current logic reuses the direct predecessor Arcs from node_arc.
                new_predecessors = node_arc.predecessors.clone();
            };
            // Create a canonical node with the new value and (potentially transformed) predecessors.
            let new_node_arc = GSSNode::get_canonical(new_value, new_predecessors, cache);
            memo.insert(node_ptr, Some(new_node_arc.clone()));
            Some(new_node_arc)
        }
    }
}

pub fn prune_and_transform_roots_canonical<T: Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T>>],
    closure: &impl Fn(&T) -> Option<(T, bool)>,
    cache: &mut NodeCache<T>, // Cache for canonical node creation
) -> Vec<Option<Arc<GSSNode<T>>>> {
    let mut memo = HashMap::new();
    roots
        .iter()
        .map(|root| prune_and_transform_recursive_canonical(root, closure, &mut memo, cache))
        .collect()
}

pub fn prune_and_transform_recursive<T: Clone + Ord + Hash + Debug>(
    node_arc: &Arc<GSSNode<T>>,
    closure: &impl Fn(&T) -> Option<(T, bool)>,
    memo: &mut HashMap<*const GSSNode<T>, Option<Arc<GSSNode<T>>>>,
) -> Option<Arc<GSSNode<T>>> {
    // TODO: The NodeCache will still be checked/populated by prune_and_transform_recursive_canonical, which is a bit wasteful.
    //       Need some way of 'simplifying out' this behaviour.
    prune_and_transform_recursive_canonical(node_arc, closure, memo, &mut NodeCache::new())
}

pub fn prune_and_transform_roots<T: Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T>>],
    closure: &impl Fn(&T) -> Option<(T, bool)>,
) -> Vec<Option<Arc<GSSNode<T>>>> {
    // TODO: Same issue as above: NodeCache still checked/populated.
    prune_and_transform_roots_canonical(roots, closure, &mut NodeCache::new())
}

// Read-only functions: find_longest_path, gather_gss_stats, print_gss_forest
// These primarily change due to ArcPtrWrapper removal.

fn find_longest_path_recursive<T>(
    node_arc: &Arc<GSSNode<T>>,
    memo: &mut HashMap<*const GSSNode<T>, Vec<Arc<GSSNode<T>>>>,
    visited_recursion: &mut HashSet<*const GSSNode<T>>,
) -> Vec<Arc<GSSNode<T>>> {
    let node_ptr = Arc::as_ptr(node_arc);

    if let Some(cached_path) = memo.get(&node_ptr) {
        return cached_path.clone();
    }

    if !visited_recursion.insert(node_ptr) {
        return Vec::new();
    }

    let mut longest_pred_path: Vec<Arc<GSSNode<T>>> = Vec::new();

    if !node_arc.predecessors.is_empty() {
        for pred_arc in &node_arc.predecessors { // pred_arc is &Arc<GSSNode<T>>
            let pred_path = find_longest_path_recursive(pred_arc, memo, visited_recursion);
            if pred_path.len() > longest_pred_path.len() { // Removed !pred_path.is_empty() as len check covers it
                longest_pred_path = pred_path;
            }
        }
    }

    let mut current_path = longest_pred_path;
    current_path.push(node_arc.clone());

    memo.insert(node_ptr, current_path.clone());
    visited_recursion.remove(&node_ptr);

    current_path
}

pub fn find_longest_path<T>(roots: &[Arc<GSSNode<T>>]) -> Option<Vec<Arc<GSSNode<T>>>> {
    let mut memo: HashMap<*const GSSNode<T>, Vec<Arc<GSSNode<T>>>> = HashMap::new();

    for root_arc in roots {
        let mut visited_recursion = HashSet::new();
        find_longest_path_recursive(root_arc, &mut memo, &mut visited_recursion);
    }

    memo.into_values().max_by_key(|path| path.len())
}

#[derive(Debug, Clone, Default)]
pub struct GSSStats {
    pub num_roots: usize,
    pub unique_nodes: usize,
    pub max_depth: usize,
    pub average_depth: f64,
    pub merge_points: usize,
    pub max_predecessors: usize,
    pub average_predecessors: f64,
}

// Manual impl for GSSStats
impl JSONConvertible for GSSStats {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("num_roots".to_string(), self.num_roots.to_json());
        obj.insert("unique_nodes".to_string(), self.unique_nodes.to_json());
        obj.insert("max_depth".to_string(), self.max_depth.to_json());
        obj.insert("average_depth".to_string(), self.average_depth.to_json());
        obj.insert("merge_points".to_string(), self.merge_points.to_json());
        obj.insert("max_predecessors".to_string(), self.max_predecessors.to_json());
        obj.insert("average_predecessors".to_string(), self.average_predecessors.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let num_roots = obj.remove("num_roots").ok_or_else(|| "Missing field num_roots".to_string())
                                   .and_then(usize::from_json)?;
                let unique_nodes = obj.remove("unique_nodes").ok_or_else(|| "Missing field unique_nodes".to_string())
                                      .and_then(usize::from_json)?;
                let max_depth = obj.remove("max_depth").ok_or_else(|| "Missing field max_depth".to_string())
                                   .and_then(usize::from_json)?;
                let average_depth = obj.remove("average_depth").ok_or_else(|| "Missing field average_depth".to_string())
                                       .and_then(f64::from_json)?;
                let merge_points = obj.remove("merge_points").ok_or_else(|| "Missing field merge_points".to_string())
                                      .and_then(usize::from_json)?;
                let max_predecessors = obj.remove("max_predecessors").ok_or_else(|| "Missing field max_predecessors".to_string())
                                          .and_then(usize::from_json)?;
                let average_predecessors = obj.remove("average_predecessors").ok_or_else(|| "Missing field average_predecessors".to_string())
                                              .and_then(f64::from_json)?;
                Ok(GSSStats {
                    num_roots,
                    unique_nodes,
                    max_depth,
                    average_depth,
                    merge_points,
                    max_predecessors,
                    average_predecessors,
                })
            }
            _ => Err("Expected JSONNode::Object for GSSStats".to_string()),
        }
    }
}


pub fn gather_gss_stats<T>(roots: &[Arc<GSSNode<T>>]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut visited: HashSet<*const GSSNode<T>> = HashSet::new();
    let mut queue: VecDeque<(Arc<GSSNode<T>>, usize)> = VecDeque::new();

    let mut total_depth_sum: u64 = 0;
    let mut total_predecessors_sum: u64 = 0;

    for root_arc in roots {
        let root_ptr = Arc::as_ptr(root_arc);
        if visited.insert(root_ptr) {
            queue.push_back((root_arc.clone(), 0));
        }
    }

    while let Some((current_node_arc, current_depth)) = queue.pop_front() {
        stats.unique_nodes += 1;
        stats.max_depth = stats.max_depth.max(current_depth);
        total_depth_sum += current_depth as u64;

        let num_predecessors = current_node_arc.predecessors.len();
        stats.max_predecessors = stats.max_predecessors.max(num_predecessors);
        total_predecessors_sum += num_predecessors as u64;
        if num_predecessors > 1 {
            stats.merge_points += 1;
        }

        for pred_arc in &current_node_arc.predecessors { // pred_arc is &Arc<GSSNode<T>>
            let pred_raw_ptr = Arc::as_ptr(pred_arc);
            if visited.insert(pred_raw_ptr) {
                queue.push_back((pred_arc.clone(), current_depth + 1));
            }
        }
    }

    if stats.unique_nodes > 0 {
        stats.average_depth = total_depth_sum as f64 / stats.unique_nodes as f64;
        stats.average_predecessors = total_predecessors_sum as f64 / stats.unique_nodes as f64;
    }
    stats
}

fn print_gss_node_recursive<T: Debug>(
    node_arc: &Arc<GSSNode<T>>,
    visited: &mut HashSet<*const GSSNode<T>>,
    indent: usize,
    node_count: &mut usize,
    max_nodes: usize,
    output: &mut String,
) -> Result<(), std::fmt::Error> {
    if *node_count >= max_nodes {
        return Ok(());
    }

    let node_ptr = Arc::as_ptr(node_arc);
    let prefix = format!("{:indent$}", "", indent = indent * 2);

    if visited.contains(&node_ptr) {
        writeln!(output, "{}- Node {:p} (Visited)", prefix, node_ptr)?;
        return Ok(());
    }

    visited.insert(node_ptr);
    *node_count += 1;

    writeln!(output, "{}- Node {:p}: {:?}", prefix, node_ptr, node_arc.value)?;

    if !node_arc.predecessors.is_empty() {
        writeln!(output, "{}  Predecessors:", prefix)?;
        for pred_arc in &node_arc.predecessors { // pred_arc is &Arc<GSSNode<T>>
            print_gss_node_recursive(pred_arc, visited, indent + 2, node_count, max_nodes, output)?;
            if *node_count >= max_nodes {
                return Ok(());
            }
        }
    }
    Ok(())
}

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
                if node_count >= max_nodes && i < roots.len() -1 { // Check if max_nodes reached and not the last root
                    writeln!(&mut output, "... (Truncated: Reached max nodes {})", max_nodes).unwrap();
                    break;
                }
            }
            Err(e) => {
                eprintln!("Error writing GSS structure to string: {}", e);
                return format!("Error generating GSS string: {}", e);
            }
        }
    }
    // This check was slightly off, if max_nodes is hit on the last root, it might not print truncated.
    // The check inside the loop is better.
    // if node_count < max_nodes && visited.len() > node_count { // Corrected condition
    //      writeln!(&mut output, "... (Truncated: Reached max nodes {})", max_nodes).unwrap();
    // }
    if node_count >= max_nodes && visited.len() > node_count { // If we printed max_nodes but there were more visited (due to shared structure not printed)
         writeln!(&mut output, "... (More nodes exist but not printed due to max_nodes limit)").unwrap();
    }

    output
}

// Simplification functions remain to canonicalize an existing GSS.
// They use GSSNode::get_canonical internally.
fn simplify_node_recursive<T: Clone + Ord + Hash + Debug>(
    original_node_arc: &Arc<GSSNode<T>>,
    original_ptr_memo: &mut HashMap<*const GSSNode<T>, Arc<GSSNode<T>>>,
    canonicalization_cache: &mut NodeCache<T>,
) -> Arc<GSSNode<T>> {
    let original_node_ptr = Arc::as_ptr(original_node_arc);

    if let Some(canonical_arc) = original_ptr_memo.get(&original_node_ptr) {
        return canonical_arc.clone();
    }

    let mut canonical_predecessor_arcs: BTreeSet<Arc<GSSNode<T>>> = BTreeSet::new();
    for original_pred_arc in &original_node_arc.predecessors { // original_pred_arc is &Arc<GSSNode<T>>
        let simplified_pred_arc = simplify_node_recursive(
            original_pred_arc,
            original_ptr_memo,
            canonicalization_cache,
        );
        canonical_predecessor_arcs.insert(simplified_pred_arc);
    }

    let canonical_arc_for_current_node = GSSNode::get_canonical(
        original_node_arc.value.clone(),
        canonical_predecessor_arcs,
        canonicalization_cache,
    );

    original_ptr_memo.insert(original_node_ptr, canonical_arc_for_current_node.clone());
    canonical_arc_for_current_node
}

pub fn simplify_gss_forest<T: Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T>>],
) -> Vec<Arc<GSSNode<T>>> {
    let mut original_ptr_memo: HashMap<*const GSSNode<T>, Arc<GSSNode<T>>> = HashMap::new();
    let mut canonicalization_cache_for_this_run: NodeCache<T> = NodeCache::new();
    let mut simplified_roots = Vec::with_capacity(roots.len());

    for root_arc in roots {
        simplified_roots.push(simplify_node_recursive(
            root_arc,
            &mut original_ptr_memo,
            &mut canonicalization_cache_for_this_run,
        ));
    }
    simplified_roots
}


#[cfg(test)]
mod tests {
    use super::*;
    // std::collections::{BTreeSet, HashMap, HashSet, VecDeque} are already imported by super::*;
    // std::sync::Arc is also imported
    // std::fmt::Debug is also imported

    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct MockLLMTokenInfo {
        active: String,
        intersection: String,
    }

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
        state_id: usize,
        t: MockLLMTokenInfo,
    }

    impl Debug for MockParseStateNodeContent {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!(
                "ParseStateNodeContent {{ state_id: StateID({}), t: {:?} }}",
                self.state_id, self.t
            ))
        }
    }

    type MockGSSNode = GSSNode<MockParseStateNodeContent>;
    type MockNodeCache = NodeCache<MockParseStateNodeContent>; // For canonical mock nodes
    type IntNodeCache = NodeCache<i32>; // For canonical i32 nodes

    // Helper to create a non-canonical GSSNode Arc for tests
    fn nc_node_arc(value: i32, predecessors: Vec<Arc<GSSNode<i32>>>) -> Arc<GSSNode<i32>> {
        let pred_set: BTreeSet<Arc<GSSNode<i32>>> = predecessors.into_iter().collect();
        Arc::new(GSSNode::new_with_predecessors(value, pred_set))
    }

    // Helper to create a canonical GSSNode Arc for tests
    fn c_node_arc(value: i32, predecessors: Vec<Arc<GSSNode<i32>>>, cache: &mut IntNodeCache) -> Arc<GSSNode<i32>> {
        let pred_set: BTreeSet<Arc<GSSNode<i32>>> = predecessors.into_iter().collect();
        GSSNode::new_with_predecessors_canonical(value, pred_set, cache)
    }

    // Helper for mock content nodes (canonical)
    fn c_mock_node_arc(
        value: MockParseStateNodeContent,
        predecessors: Vec<Arc<MockGSSNode>>,
        cache: &mut MockNodeCache,
    ) -> Arc<MockGSSNode> {
        let pred_set: BTreeSet<Arc<MockGSSNode>> = predecessors.into_iter().collect();
        GSSNode::new_with_predecessors_canonical(value, pred_set, cache)
    }


    fn collect_arcs_recursive<T>(
        node_arc: &Arc<GSSNode<T>>,
        collected_arcs: &mut HashMap<*const GSSNode<T>, Arc<GSSNode<T>>>,
    ) {
        let ptr = Arc::as_ptr(node_arc);
        if collected_arcs.contains_key(&ptr) {
            return;
        }
        collected_arcs.insert(ptr, node_arc.clone());
        for pred_arc in &node_arc.predecessors {
            collect_arcs_recursive(pred_arc, collected_arcs);
        }
    }

    #[test]
    fn test_gss_simplification_basic() {
        // Create a non-canonical graph first
        let d1_orig_nc = nc_node_arc(40, vec![]);
        let c1_preds_nc = vec![d1_orig_nc.clone()];
        let c1_orig_nc = nc_node_arc(30, c1_preds_nc);
        let b1_preds_nc = vec![c1_orig_nc.clone()];
        let b1_orig_nc = nc_node_arc(20, b1_preds_nc);

        let d2_orig_nc = nc_node_arc(40, vec![]); // Structurally same as d1_orig_nc, but different Arc
        assert_ne!(Arc::as_ptr(&d1_orig_nc), Arc::as_ptr(&d2_orig_nc));

        let a1_preds_nc = vec![b1_orig_nc.clone(), d2_orig_nc.clone()];
        let a1_orig_nc = nc_node_arc(10, a1_preds_nc);

        let roots_nc = vec![a1_orig_nc.clone()];
        let simplified_roots = simplify_gss_forest(&roots_nc);
        let simplified_a1 = simplified_roots[0].clone();

        let mut collected_arcs = HashMap::new();
        collect_arcs_recursive(&simplified_a1, &mut collected_arcs);
        assert_eq!(collected_arcs.len(), 4, "Expected 4 unique Arcs after simplification (A1, B1, C1, D_canonical)");

        let s_a1 = simplified_a1;
        let s_b1 = s_a1.predecessors.iter().find(|n| n.value == 20).expect("B1 node").clone();
        let s_d_from_a1 = s_a1.predecessors.iter().find(|n| n.value == 40).expect("D node from A1").clone();

        let s_c1 = s_b1.predecessors.iter().find(|n| n.value == 30).expect("C1 node").clone();
        let s_d_from_c1 = s_c1.predecessors.iter().find(|n| n.value == 40).expect("D node from C1").clone();

        assert!(Arc::ptr_eq(&s_d_from_a1, &s_d_from_c1), "D nodes should be canonicalized to the same Arc");
        let s_d_canonical = s_d_from_a1;

        assert_eq!(s_d_canonical.predecessors.len(), 0);
        assert!(Arc::ptr_eq(s_c1.predecessors.iter().next().unwrap(), &s_d_canonical));
        assert!(Arc::ptr_eq(s_b1.predecessors.iter().next().unwrap(), &s_c1));
        assert!(s_a1.predecessors.contains(&s_b1));
        assert!(s_a1.predecessors.contains(&s_d_canonical));
    }

    #[test]
    fn test_simplification_canonicalizes_structurally_identical_nodes() {
        // Simulating non-canonical input for simplify_gss_forest
        let l_val = 0; let m_val = 1; let r_val = 2;

        let l1_nc = nc_node_arc(l_val, vec![]);
        let l2_nc = nc_node_arc(l_val, vec![]);
        let l3_nc = nc_node_arc(l_val, vec![]);

        let m1_nc = nc_node_arc(m_val, vec![l1_nc.clone()]);
        let m2_nc = nc_node_arc(m_val, vec![l2_nc.clone()]);
        let m3_nc = nc_node_arc(m_val, vec![l3_nc.clone()]);

        let r1_orig_non_canonical = nc_node_arc(r_val, vec![m1_nc.clone(), m2_nc.clone(), m3_nc.clone()]);

        let simplified_roots = simplify_gss_forest(&[r1_orig_non_canonical]);
        let simplified_r1_arc = simplified_roots[0].clone();

        let mut collected_arcs_map = HashMap::new();
        collect_arcs_recursive(&simplified_r1_arc, &mut collected_arcs_map);
        assert_eq!(collected_arcs_map.len(), 3, "Expected 3 unique Arcs after simplify_gss_forest");

        let s_r1_node = simplified_r1_arc.as_ref();
        assert_eq!(s_r1_node.value, 2);
        assert_eq!(s_r1_node.predecessors.len(), 1); // Canonical M node

        let s_m_level_arc = s_r1_node.predecessors.iter().next().unwrap().clone();
        assert_eq!(s_m_level_arc.value, 1);
        assert_eq!(s_m_level_arc.predecessors.len(), 1); // Canonical L node

        let s_l_level_arc = s_m_level_arc.predecessors.iter().next().unwrap().clone();
        assert_eq!(s_l_level_arc.value, 0);
        assert_eq!(s_l_level_arc.predecessors.len(), 0);
    }

    #[test]
    fn test_gss_simplification_reproduces_logged_structure() {
        // This test will now use simplify_gss_forest on a non-canonical construction
        // to verify its behavior, similar to how it might be used in practice.
        let token_info = MockLLMTokenInfo { active: "[0]".to_string(), intersection: "[0]".to_string() };
        let val0 = MockParseStateNodeContent { state_id: 0, t: token_info.clone() };
        let val1 = MockParseStateNodeContent { state_id: 1, t: token_info.clone() };
        let val2 = MockParseStateNodeContent { state_id: 2, t: token_info.clone() };

        // Non-canonical construction
        let nc_node = |v, p: Vec<Arc<MockGSSNode>>| Arc::new(MockGSSNode::new_with_predecessors(v, p.into_iter().collect()));

        let node_a_val0_nc = nc_node(val0.clone(), vec![]);
        let node_c_val0_nc = nc_node(val0.clone(), vec![]); // Distinct Arc from a_val0_nc

        let mut other_s0_leaves_nc = Vec::new();
        for _ in 0..8 { other_s0_leaves_nc.push(nc_node(val0.clone(), vec![])); }

        let node_b_val1_nc = nc_node(val1.clone(), vec![node_c_val0_nc.clone()]);
        let node_e_val1_nc = nc_node(val1.clone(), vec![node_c_val0_nc.clone()]); // Shares c_val0_nc, but distinct from b_val1_nc if c_val0_nc is same

        let mut other_s1_nodes_nc = Vec::new();
        for leaf_s0_nc in &other_s0_leaves_nc {
            other_s1_nodes_nc.push(nc_node(val1.clone(), vec![leaf_s0_nc.clone()]));
        }

        let mut preds_for_d_nc = vec![node_e_val1_nc.clone()];
        preds_for_d_nc.extend(other_s1_nodes_nc.iter().cloned());
        let node_d_val2_nc = nc_node(val2.clone(), preds_for_d_nc);

        let roots_non_canonical = vec![node_a_val0_nc.clone(), node_b_val1_nc.clone(), node_d_val2_nc.clone()];

        // Count unique nodes in non-canonical input to see the reduction
        let mut original_arcs_map = HashMap::new();
        for r_nc in &roots_non_canonical { collect_arcs_recursive(r_nc, &mut original_arcs_map); }
        println!("Number of unique Arcs in non-canonical input: {}", original_arcs_map.len());
        // Expected: 1 (a0) + 1 (c0) + 8 (other s0) + 1 (b1) + 1 (e1) + 8 (other s1) + 1 (d2) = 21

        let simplified_roots = simplify_gss_forest(&roots_non_canonical);

        let max_nodes_to_print = 30;
        let simplified_gss_string_representation = print_gss_forest(&simplified_roots, max_nodes_to_print);
        println!("\n--- Simplified GSS Structure (from non-canonical log input) ---\n");
        println!("{}", simplified_gss_string_representation);
        println!("--- End of Simplified GSS Structure ---\n");

        let mut collected_arcs_map: HashMap<*const MockGSSNode, Arc<MockGSSNode>> = HashMap::new();
        for root_arc in &simplified_roots { collect_arcs_recursive(root_arc, &mut collected_arcs_map); }
        assert_eq!(collected_arcs_map.len(), 3, "The simplified GSS should contain 3 unique Arcs.");

        let s_root0 = simplified_roots.iter().find(|r| r.value.state_id == 0).unwrap();
        let s_root1 = simplified_roots.iter().find(|r| r.value.state_id == 1).unwrap();
        let s_root2 = simplified_roots.iter().find(|r| r.value.state_id == 2).unwrap();

        assert_eq!(s_root0.predecessors.len(), 0);
        assert_eq!(s_root1.predecessors.len(), 1);
        assert!(Arc::ptr_eq(s_root1.predecessors.iter().next().unwrap(), s_root0));
        assert_eq!(s_root2.predecessors.len(), 1);
        assert!(Arc::ptr_eq(s_root2.predecessors.iter().next().unwrap(), s_root1));
    }
}
