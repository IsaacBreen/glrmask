use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc};
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::collections::hash_map::DefaultHasher;

use crate::datastructures::ArcPtrWrapper; // Import ArcPtrWrapper

// Helper function to compute level_sets
fn compute_gss_level_sets<T: Clone + Ord>(
    value: &T,
    predecessor_arcs: &BTreeSet<Arc<GSSNode<T>>>,
) -> Vec<BTreeSet<T>> {
    let mut result_sets = Vec::new();
    result_sets.push(BTreeSet::from([value.clone()])); // Level 0: current node's value

    if predecessor_arcs.is_empty() {
        return result_sets;
    }

    let mut current_pred_level_idx = 0;
    loop {
        let mut combined_set_for_this_level = BTreeSet::new();
        let mut any_pred_contributed_to_this_level = false;

        for pred_arc in predecessor_arcs {
            if let Some(pred_level_values) = pred_arc.level_sets.get(current_pred_level_idx) {
                combined_set_for_this_level.extend(pred_level_values.iter().cloned());
                any_pred_contributed_to_this_level = true;
            }
        }

        if !any_pred_contributed_to_this_level {
            break;
        }
        
        result_sets.push(combined_set_for_this_level);
        current_pred_level_idx += 1;

        if current_pred_level_idx > 512 { // Safety break
            // Consider logging a warning if a logging facade is available
            // e.g., log::warn!("Exceeded max depth in compute_gss_level_sets for node value (debug): {:?}", value);
            break;
        }
    }
    result_sets
}

// Helper function to compute hash_key_cache
fn compute_gss_hash<T: Ord + Hash>(value: &T, level_sets: &Vec<BTreeSet<T>>) -> usize {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    level_sets.len().hash(&mut hasher); 
    for set in level_sets {
        set.len().hash(&mut hasher); 
        for item in set { // BTreeSet iterates in sorted order
            item.hash(&mut hasher);
        }
    }
    hasher.finish() as usize
}


// Replace this line:
// #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
// pub struct GSSNode<T> {
// With this:
#[derive(Debug, Clone)]
pub struct GSSNode<T: Ord> { // Add T: Ord here
    pub value: T,
    predecessors: BTreeSet<ArcPtrWrapper<GSSNode<T>>>,
    level_sets: Vec<BTreeSet<T>>, // New field
    hash_key_cache: usize,        // New field
}


impl<T: Ord> GSSNode<T> { // T: Ord is from struct
    // Change signature and body of new:
    // pub fn new(value: T) -> Self {
    // Becomes:
    pub fn new(value: T) -> Self where T: Clone + Hash { // Add Clone + Hash here
        let level_sets = vec![BTreeSet::from([value.clone()])];
        let hash_key_cache = compute_gss_hash(&value, &level_sets);
        Self {
            value,
            predecessors: BTreeSet::new(),
            level_sets,
            hash_key_cache,
        }
    }

    // Change signature and body of new_with_predecessors:
    // pub fn new_with_predecessors(value: T, predecessors: Vec<Arc<GSSNode<T>>>) -> Self {
    // Becomes:
    pub fn new_with_predecessors(value: T, predecessors_arcs_vec: Vec<Arc<GSSNode<T>>>) -> Self
    where
        T: Clone + Hash, // T: Ord is from struct. Add Clone + Hash here.
    {
        let unique_predecessor_arcs: BTreeSet<Arc<GSSNode<T>>> = predecessors_arcs_vec.into_iter().collect();

        let level_sets = compute_gss_level_sets(&value, &unique_predecessor_arcs);
        let hash_key_cache = compute_gss_hash(&value, &level_sets);
        
        let predecessors_arcptr: BTreeSet<ArcPtrWrapper<GSSNode<T>>> = unique_predecessor_arcs
            .into_iter()
            .map(ArcPtrWrapper::new)
            .collect();

        Self {
            value,
            predecessors: predecessors_arcptr,
            level_sets,
            hash_key_cache,
        }
    }

    // Change signature of from_iter:
    // pub fn from_iter<I>(iter: I) -> Self
    // where
    //     I: IntoIterator<Item = T>,
    // Becomes:
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Clone + Ord + Hash, // Add this bound
    {
        let mut iter = iter.into_iter();
        let mut root = Self::new(iter.next().unwrap());
        for value in iter {
            root = root.push(value);
        }
        root
    }

    // Change signature and body of push:
    // pub fn push(self, value: T) -> Self {
    //     let mut new_node = Self::new(value);
    //     new_node.predecessors.insert(ArcPtrWrapper::new(Arc::new(self)));
    //     new_node
    // }
    // Becomes:
    pub fn push(self, value: T) -> Self where T: Clone + Ord + Hash { // Add bounds
        let predecessors_arcs = vec![Arc::new(self)];
        GSSNode::new_with_predecessors(value, predecessors_arcs)
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

    // Change signature of value_mut (add bounds, no body change):
    // pub fn value_mut(&mut self) -> &mut T {
    // Becomes:
    pub fn value_mut(&mut self) -> &mut T where T: Clone + Hash { // T: Ord from struct. Add Clone + Hash here.
        // WARNING: Modifying value through this reference will invalidate
        // level_sets[0] and hash_key_cache. This node may behave incorrectly
        // in comparisons or hash-based collections afterwards if not handled by caller.
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

    // Change signature and body of merge:
    // pub fn merge(&mut self, mut other: Self)
    // where
    //     T: PartialEq,
    // Becomes:
    pub fn merge(&mut self, mut other: Self)
    where
        T: Clone + Hash + PartialEq, // T: Ord from struct. Add Clone + Hash.
    {
        assert!(self.value == other.value);
        self.predecessors.extend(std::mem::take(&mut other.predecessors));
        
        let self_predecessor_arcs: BTreeSet<Arc<GSSNode<T>>> = self.predecessors.iter()
            .map(|apw| apw.as_arc().clone())
            .collect();
        self.level_sets = compute_gss_level_sets(&self.value, &self_predecessor_arcs);
        self.hash_key_cache = compute_gss_hash(&self.value, &self.level_sets);
    }

    // Change signature and body of merge_unchecked:
    // pub fn merge_unchecked(&mut self, mut other: Self)
    // Becomes:
    pub fn merge_unchecked(&mut self, mut other: Self)
    where
        T: Clone + Hash, // T: Ord from struct. Add Clone + Hash.
    {
        self.predecessors.extend(std::mem::take(&mut other.predecessors));

        let self_predecessor_arcs: BTreeSet<Arc<GSSNode<T>>> = self.predecessors.iter()
            .map(|apw| apw.as_arc().clone())
            .collect();
        self.level_sets = compute_gss_level_sets(&self.value, &self_predecessor_arcs);
        self.hash_key_cache = compute_gss_hash(&self.value, &self.level_sets);
    }

    // Change signature and body of map:
    // pub fn map<F, U>(&self, f: F) -> GSSNode<U>
    // where
    //     F: Copy + Fn(&T) -> U,
    // {
    //     GSSNode {
    //         value: f(&self.value),
    //         predecessors: self.predecessors.iter()
    //             .map(|wrapper| ArcPtrWrapper::new(Arc::new(wrapper.as_ref().map(f))))
    //             .collect(),
    //     }
    // }
    // Becomes:
    pub fn map<F, U>(&self, f: F) -> GSSNode<U>
    where
        F: Copy + Fn(&T) -> U,
        U: Clone + Ord + Hash, // U must meet GSSNode<U>'s requirements
        // T: Ord is from self's GSSNode<T>
    {
        let new_value = f(&self.value);
        let new_predecessors_arcs: Vec<Arc<GSSNode<U>>> = self.predecessors.iter()
            .map(|wrapper| {
                Arc::new(wrapper.as_ref().map(f)) // wrapper.as_ref() is &GSSNode<T>
            })
            .collect();
        GSSNode::new_with_predecessors(new_value, new_predecessors_arcs)
    }

    // Add Accessors
    pub fn get_level_sets(&self) -> &Vec<BTreeSet<T>> {
        &self.level_sets
    }

    pub fn get_hash_key_cache(&self) -> usize {
        self.hash_key_cache
    }

    pub fn get_predecessors_arcs(&self) -> Vec<Arc<GSSNode<T>>> {
        self.predecessors.iter().map(|apw| apw.as_arc().clone()).collect()
    }
}

// Implement Hash, PartialEq, Eq, PartialOrd, Ord for GSSNode<T>
impl<T: Ord + Hash> Hash for GSSNode<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_key_cache.hash(state);
    }
}

impl<T: Ord + PartialEq> PartialEq for GSSNode<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.hash_key_cache != other.hash_key_cache {
            // Hash mismatch implies inequality, but the converse is not true.
            // However, our hash is specifically designed based on canonical representation (value + sorted level_sets),
            // so equal hash *should* imply equality in this structure unless hash collision occurs.
            // For safety and strict PartialEq contract, we check full content.
            return false;
        }
        // The canonicalization in new/push/new_with_predecessors ensures predecessors are canonicalized within ArcPtrWrapper BTreeSet,
        // and level_sets are computed deterministically.
        // Comparing level_sets captures the structure below this node up to a certain depth (implicitly, fully if no cycles).
        // Comparing value and level_sets is sufficient for structural equality given the canonical representation.
        self.value == other.value && self.level_sets == other.level_sets
    }
}

impl<T: Ord + Eq> Eq for GSSNode<T> {}

impl<T: Ord + PartialOrd> PartialOrd for GSSNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> Ord for GSSNode<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Canonical representation is value then level_sets
        self.value.cmp(&other.value)
            .then_with(|| self.level_sets.cmp(&other.level_sets))
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

impl<T: Clone + Ord + Hash> GSSTrait<T> for GSSNode<T> { // Add Ord + Hash bound
    type Peek<'a> = &'a T where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        &self.value
    }

    fn push(&self, value: T) -> GSSNode<T> {
        let mut new_node = GSSNode::new(value); // Uses GSSNode::new
        new_node.predecessors.insert(ArcPtrWrapper::new(Arc::new(self.clone()))); // This clone might not be ideal, push(self) is better
        
        // Recompute hash/level_sets after predecessor added
         let current_preds_arcs: BTreeSet<Arc<GSSNode<T>>> = new_node.predecessors.iter()
            .map(|apw| apw.as_arc().clone())
            .collect();
        new_node.level_sets = compute_gss_level_sets(&new_node.value, &current_preds_arcs);
        new_node.hash_key_cache = compute_gss_hash(&new_node.value, &new_node.level_sets);

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

impl<T: Clone + Ord + Hash> GSSTrait<T> for Arc<GSSNode<T>> { // Add Ord + Hash bound
    type Peek<'a> = &'a T where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        &self.value
    }

    fn push(&self, value: T) -> GSSNode<T> {
        let mut new_node = GSSNode::new(value); // Uses GSSNode::new
        new_node.predecessors.insert(ArcPtrWrapper::new(self.clone())); // Uses the Arc directly

        // Recompute hash/level_sets after predecessor added
        let current_preds_arcs: BTreeSet<Arc<GSSNode<T>>> = new_node.predecessors.iter()
            .map(|apw| apw.as_arc().clone())
            .collect();
        new_node.level_sets = compute_gss_level_sets(&new_node.value, &current_preds_arcs);
        new_node.hash_key_cache = compute_gss_hash(&new_node.value, &new_node.level_sets);

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

impl<T: Clone + Ord + Hash> GSSTrait<T> for Option<Arc<GSSNode<T>>> { // Add Ord + Hash bound
    type Peek<'a> = Option<&'a T> where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        self.as_ref().map(|node| node.peek())
    }

    fn push(&self, value: T) -> GSSNode<T> {
        self.clone().map(|node| node.push(value.clone())).unwrap_or_else(|| GSSNode::new(value)) // Uses GSSNode::new
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node| node.pop()).unwrap_or_default()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node| node.popn(n)).unwrap_or_default()
    }
}

impl<T: Clone + Ord + Hash> GSSTrait<T> for Option<GSSNode<T>> { // Add Ord + Hash bound
    type Peek<'a> = Option<&'a T> where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        self.as_ref().map(|node| node.peek())
    }

    fn push(&self, value: T) -> GSSNode<T> {
        self.clone().map(|node| node.push(value.clone())).unwrap_or_else(|| GSSNode::new(value)) // Uses GSSNode::new
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

// Change the impl line:
// impl<T: Clone + Ord> BulkMerge<T> for Vec<Arc<GSSNode<T>>> {
// Becomes:
impl<T: Clone + Ord + Hash> BulkMerge<T> for Vec<Arc<GSSNode<T>>> { // Add T: Hash
    fn bulk_merge(&mut self) {
        // todo: should be possible to avoid cloning T in some cases by using &T in this map,
        //  but we need to be careful about lifetimes. If we use `node.as_ref().value`, then node
        //  will go out of bounds while the reference to its value is still inside `groups`.
        let mut groups: BTreeMap<T, HashMap<*const GSSNode<T>, Arc<GSSNode<T>>>> = BTreeMap::new();
        for node in self.drain(..) {
             // Use the actual node's value for grouping, not the pointer key
             groups.entry(node.value.clone()).or_default().entry(Arc::as_ptr(&node)).or_insert(node);
        }
        for mut group in groups.into_values() {
            // Convert HashMap values (Arc<GSSNode<T>>) to a Vec for processing
            let mut group_vec = group.into_values().collect::<Vec<_>>();
            if group_vec.is_empty() {
                 continue; // Should not happen if groups was not empty, but belt and suspenders
            }
            
            // Take the first node to merge others into
            let mut first = group_vec.remove(0);
            
            if group_vec.is_empty() {
                // Only one node in this group, no merging needed
                self.push(first);
            } else {
                // Arc::make_mut clones the GSSNode if `first` is shared.
                // The new `first_mut_ref` will have its predecessors modified.
                let first_mut_ref = Arc::make_mut(&mut first);
                // The original predecessors of `first` are already in `first_mut_ref.predecessors`.
                // Add predecessors from all siblings.
                // `BTreeSet::insert` handles deduplication based on ArcPtrWrapper's Ord impl (pointer address).
                for sibling_arc in group_vec { // sibling_arc is Arc<GSSNode<T>>
                    for pred_wrapper in &sibling_arc.predecessors { // pred_wrapper is &ArcPtrWrapper<GSSNode<T>>
                        first_mut_ref.predecessors.insert(pred_wrapper.clone());
                    }
                }
                // Insert the recomputation logic BEFORE self.push(first) inside that `else` block:
                // After the loop `for sibling_arc in group_vec { ... }`:
                let current_preds_arcs: BTreeSet<Arc<GSSNode<T>>> = first_mut_ref.predecessors.iter()
                    .map(|apw| apw.as_arc().clone())
                    .collect();
                first_mut_ref.level_sets = compute_gss_level_sets(&first_mut_ref.value, &current_preds_arcs);
                first_mut_ref.hash_key_cache = compute_gss_hash(&first_mut_ref.value, &first_mut_ref.level_sets);

                self.push(first);
            }
        }
    }
}


// Helper function for prune_and_transform_roots
pub fn prune_and_transform_recursive<T: Clone + Ord + Hash>( // Add Ord + Hash bound
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
                let mut keep_originals = false; // Flag to decide if we keep original predecessors or try lookup

                for pred_wrapper in &node_arc.predecessors { // pred_wrapper is &ArcPtrWrapper<GSSNode<T>>
                     // pred_arc is &Arc<GSSNode<T>>
                     let pred_arc = pred_wrapper.as_arc();
                     // Check memo for already transformed predecessor
                    if let Some(existing_transformed) = memo.get(&Arc::as_ptr(pred_arc)) {
                        if let Some(transformed_pred) = existing_transformed {
                            transformed_predecessors.push(transformed_pred.clone());
                        }
                        // If existing_transformed is None, the predecessor was pruned, so skip.
                        // crate::debug!(4, "Skipping pruned predecessor"); // Use actual debug macro if available
                        #[cfg(debug_assertions)]
                        println!("Debug(4): Skipping pruned predecessor");
                    } else {
                        // This case *shouldn't* happen if traversal order is correct (parents processed after children),
                        // but as a fallback, keep the original if not found in memo. Or perhaps panic?
                        // Let's assume the caller manages the order or this closure handles cycles/shared nodes correctly.
                        // For simplicity now, let's stick to the logic: if we stop, we keep original pointers below.
                         transformed_predecessors = node_arc.predecessors.clone().into_iter()
                             .map(|wrapper| wrapper.as_arc().clone()) // wrapper is ArcPtrWrapper, wrapper.as_arc().clone() is Arc<GSSNode<T>>
                             .collect();
                         // crate::debug!(3, "Keeping {} original predecessors", transformed_predecessors.len()); // Use actual debug macro if available
                         #[cfg(debug_assertions)]
                         println!("Debug(3): Keeping {} original predecessors", transformed_predecessors.len());
                         keep_originals = true;
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
pub fn prune_and_transform_roots<T: Clone + Ord + Hash>( // Add Ord + Hash bound
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
fn find_longest_path_recursive<T: Ord>( // Add Ord bound
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
pub fn find_longest_path<T: Ord>(roots: &[Arc<GSSNode<T>>]) -> Option<Vec<Arc<GSSNode<T>>>> { // Add Ord bound
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
pub fn gather_gss_stats<T: Clone + Ord>(roots: &[Arc<GSSNode<T>>]) -> GSSStats { // Add Ord bound
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
fn print_gss_node_recursive<T: Debug + Ord>( // Add Ord bound
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
pub fn print_gss_forest<T: Debug + Ord>(roots: &[Arc<GSSNode<T>>], max_nodes: usize) -> String { // Add Ord bound
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
         // Re-check visited.len() vs node_count isn't the right way to check for truncation,
         // The primary check is if node_count >= max_nodes inside the recursive function.
         // This extra check here is likely redundant or incorrect. Removing this complex check.
         // writeln!(&mut output, "... (Truncated: Reached max nodes {})", max_nodes).unwrap();
    }


    output
}

// --- Simplification ---

// Recursive helper for simplify_gss_forest.
fn simplify_node_recursive<T: Clone + Ord + Hash + Debug>(
    original_node_arc: &Arc<GSSNode<T>>,
    memo: &mut HashMap<*const GSSNode<T>, Arc<GSSNode<T>>>,
    canonical_nodes: &mut BTreeMap<GSSNode<T>, Arc<GSSNode<T>>>,
) -> Arc<GSSNode<T>> {
    let original_node_ptr = Arc::as_ptr(original_node_arc);
    if let Some(simplified_arc) = memo.get(&original_node_ptr) {
        return simplified_arc.clone();
    }

    let mut simplified_predecessors_vec: Vec<Arc<GSSNode<T>>> =
        Vec::with_capacity(original_node_arc.predecessors.len());
    for pred_wrapper in &original_node_arc.predecessors {
        let original_pred_arc = pred_wrapper.as_arc();
        let simplified_pred_arc =
            simplify_node_recursive(original_pred_arc, memo, canonical_nodes);
        simplified_predecessors_vec.push(simplified_pred_arc);
    }

    // Create a candidate node. new_with_predecessors will canonicalize predecessors internally
    // by putting them into a BTreeSet<Arc<GSSNode<T>>> and computing level_sets/hash based on that.
    let candidate_key_node = GSSNode::new_with_predecessors(
        original_node_arc.value.clone(),
        simplified_predecessors_vec,
    );

    use std::collections::btree_map::Entry;
    let final_arc = match canonical_nodes.entry(candidate_key_node) {
        Entry::Occupied(occupied_entry) => {
            // A canonical node with this structure already exists, use it.
            occupied_entry.get().clone()
        }
        Entry::Vacant(vacant_entry) => {
            // This structure is new, make this node the canonical one for this structure.
            // Take ownership of the canonical_key_node built above.
            let key_node_for_arc = vacant_entry.key().clone(); // Clone the node itself
            let new_canonical_arc = Arc::new(key_node_for_arc);
            vacant_entry.insert(new_canonical_arc.clone());
            new_canonical_arc
        }
    };

    // Memoize the mapping from the original node's pointer to the simplified, canonical Arc.
    memo.insert(original_node_ptr, final_arc.clone());
    final_arc
}

/// Simplifies the GSS forest defined by the given roots.
///
/// This process canonicalizes the GSS structure from the leaves upwards, ensuring that
/// identical subgraphs are represented by a single, shared `Arc<GSSNode<T>>` instance.
/// This reduces memory usage and allows for efficient structural equality checks.
///
/// The simplification is based on the `value` of the node and the canonical representation
/// of its predecessors (captured by `level_sets` and `hash_key_cache`).
///
/// Handles shared nodes using memoization. Assumes no cycles for simplicity in this implementation.
///
/// Returns a Vec of `Arc<GSSNode<T>>>` corresponding to the input `roots`, where each
/// returned Arc points to the root of the simplified subgraph.
pub fn simplify_gss_forest<T: Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T>>],
) -> Vec<Arc<GSSNode<T>>> {
    // memo: Maps original node pointer -> simplified canonical Arc for that node
    let mut memo: HashMap<*const GSSNode<T>, Arc<GSSNode<T>>> = HashMap::new();
    // canonical_nodes: Maps simplified GSSNode structure -> single canonical Arc for that structure
    let mut canonical_nodes: BTreeMap<GSSNode<T>, Arc<GSSNode<T>>> = BTreeMap::new();

    roots
        .iter()
        .map(|root_arc| {
            simplify_node_recursive(root_arc, &mut memo, &mut canonical_nodes)
        })
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*; // Imports items from the parent module (your gss.rs content)
    use std::sync::Arc;

    // Helper to create a GSSNode with String value for tests
    fn new_node(s: &str) -> GSSNode<String> {
        GSSNode::new(s.to_string())
    }

    fn new_node_with_preds(s: &str, preds: Vec<Arc<GSSNode<String>>>) -> GSSNode<String> {
        GSSNode::new_with_predecessors(s.to_string(), preds)
    }

    #[test]
    fn test_gss_node_creation_and_fields() {
        let node_a = new_node("A");
        assert_eq!(node_a.value, "A".to_string());
        assert_eq!(node_a.level_sets.len(), 1);
        assert_eq!(node_a.level_sets[0], BTreeSet::from(["A".to_string()]));
        assert_ne!(node_a.hash_key_cache, 0); // Basic check

        let node_b = new_node_with_preds("B", vec![Arc::new(node_a.clone())]);
        assert_eq!(node_b.value, "B".to_string());
        assert_eq!(node_b.level_sets.len(), 2);
        assert_eq!(node_b.level_sets[0], BTreeSet::from(["B".to_string()]));
        assert_eq!(node_b.level_sets[1], BTreeSet::from(["A".to_string()]));
    }
    
    #[test]
    fn test_gss_node_equality_and_ordering() {
        let node_a1 = new_node("A");
        let node_a2 = new_node("A");
        let node_b = new_node("B");

        assert_eq!(node_a1, node_a2, "Nodes with same value and no preds should be equal");
        assert_eq!(node_a1.hash_key_cache, node_a2.hash_key_cache, "Hashes should be equal for equal nodes");
        assert_ne!(node_a1, node_b, "Nodes with different values should not be equal");
        assert!(node_a1.cmp(&node_b) == std::cmp::Ordering::Less, "Node A should be less than Node B");

        let leaf1 = Arc::new(new_node("L1"));
        let leaf2 = Arc::new(new_node("L2"));

        // Ensure L1 < L2 for predictable ordering in BTreeSet<Arc<GSSNode<String>>>
        assert!(leaf1.as_ref().cmp(leaf2.as_ref()) == std::cmp::Ordering::Less);


        let node_c1 = new_node_with_preds("C", vec![leaf1.clone(), leaf2.clone()]);
        // Create with different Arc instances for predecessors but same logical content
        let leaf1_alt = Arc::new(new_node("L1")); 
        let leaf2_alt = Arc::new(new_node("L2"));
        let node_c2 = new_node_with_preds("C", vec![leaf2_alt.clone(), leaf1_alt.clone()]);
        
        assert_eq!(node_c1.level_sets, node_c2.level_sets, "Level sets should be identical regardless of initial pred order");
        assert_eq!(node_c1, node_c2, "Nodes C1 and C2 should be equal due to content canonicalization");
        assert_eq!(node_c1.hash_key_cache, node_c2.hash_key_cache, "Hashes for C1 and C2 should be equal");

        // Test that the internal `predecessors` field (ArcPtrWrapper set) is also consistent
        // This relies on `new_with_predecessors` using `BTreeSet<Arc<GSSNode<T>>>` internally before creating ArcPtrWrappers
        let c1_pred_ptrs: Vec<*const GSSNode<String>> = node_c1.predecessors.iter().map(|apw| Arc::as_ptr(apw.as_arc())).collect();
        let c2_pred_ptrs: Vec<*const GSSNode<String>> = node_c2.predecessors.iter().map(|apw| Arc::as_ptr(apw.as_arc())).collect();
        // Convert to BTreeSet for comparison that ignores order, then convert back to Vec for assertion
        let c1_pred_ptrs_set: BTreeSet<*const GSSNode<String>> = c1_pred_ptrs.into_iter().collect();
        let c2_pred_ptrs_set: BTreeSet<*const GSSNode<String>> = c2_pred_ptrs.into_iter().collect();

        let c1_pred_arcs: BTreeSet<Arc<GSSNode<String>>> = node_c1.predecessors.iter().map(|apw| apw.as_arc().clone()).collect();
        let c2_pred_arcs: BTreeSet<Arc<GSSNode<String>>> = node_c2.predecessors.iter().map(|apw| apw.as_arc().clone()).collect();

        // Check that the Arc contents pointed to are the same canonical Arcs
        assert_eq!(c1_pred_arcs.len(), c2_pred_arcs.len());
        for (arc1, arc2) in c1_pred_arcs.iter().zip(c2_pred_arcs.iter()) {
            assert!(Arc::ptr_eq(arc1, arc2), "Predecessor Arcs should be pointer-equal after canonicalization");
        }
    }

    #[test]
    fn test_simplify_gss_forest_shared_leaf() {
        let leaf_a_orig1 = Arc::new(new_node("A"));
        let leaf_a_orig2 = Arc::new(new_node("A")); // Different Arc, same content

        let root1 = Arc::new(new_node_with_preds("R1", vec![leaf_a_orig1.clone()]));
        let root2 = Arc::new(new_node_with_preds("R2", vec![leaf_a_orig2.clone()]));
        
        let roots = vec![root1, root2];
        let simplified_roots = simplify_gss_forest(&roots);

        assert_eq!(simplified_roots.len(), 2);
        let simplified_r1_preds = simplified_roots[0].get_predecessors_arcs();
        let simplified_r2_preds = simplified_roots[1].get_predecessors_arcs();

        assert_eq!(simplified_r1_preds.len(), 1);
        assert_eq!(simplified_r2_preds.len(), 1);
        let simplified_a_from_r1 = &simplified_r1_preds[0];
        let simplified_a_from_r2 = &simplified_r2_preds[0];

        assert_eq!(simplified_a_from_r1.value, "A".to_string());
        assert!(Arc::ptr_eq(simplified_a_from_r1, simplified_a_from_r2), "Simplified 'A' nodes should be the same Arc instance");
    }

    #[test]
    fn test_simplify_gss_forest_diamond() {
        let leaf_orig_s1 = Arc::new(new_node("Leaf"));
        let leaf_orig_s2 = Arc::new(new_node("Leaf"));

        let mid1 = Arc::new(new_node_with_preds("Mid1", vec![leaf_orig_s1]));
        let mid2 = Arc::new(new_node_with_preds("Mid2", vec![leaf_orig_s2]));
        
        let top1 = Arc::new(new_node_with_preds("Top1", vec![mid1.clone()]));
        let top2 = Arc::new(new_node_with_preds("Top2", vec![mid2.clone()]));

        let simplified_roots = simplify_gss_forest(&[top1.clone(), top2.clone()]); // Clone roots for the function
        assert_eq!(simplified_roots.len(), 2);

        let s_mid1 = simplified_roots[0].get_predecessors_arcs()[0].clone();
        let s_mid2 = simplified_roots[1].get_predecessors_arcs()[0].clone();

        assert_eq!(s_mid1.value, "Mid1".to_string());
        assert_eq!(s_mid2.value, "Mid2".to_string());
        assert!(!Arc::ptr_eq(&s_mid1, &s_mid2), "Mid1 and Mid2 should be different nodes");

        let s_leaf1 = s_mid1.get_predecessors_arcs()[0].clone();
        let s_leaf2 = s_mid2.get_predecessors_arcs()[0].clone();

        assert_eq!(s_leaf1.value, "Leaf".to_string());
        assert!(Arc::ptr_eq(&s_leaf1, &s_leaf2), "Simplified 'Leaf' nodes should be the same Arc instance");

        // Also check that the roots themselves were not canonicalized (they are different)
        assert!(!Arc::ptr_eq(&simplified_roots[0], &simplified_roots[1]));

        // Check that the original and simplified roots for Top1 are structurally equal
        assert_eq!(top1.as_ref(), simplified_roots[0].as_ref());
        assert_eq!(top2.as_ref(), simplified_roots[1].as_ref());
    }

     #[test]
    fn test_simplify_identical_branches() {
        let l1_v1 = Arc::new(new_node("L1"));
        let l2_v1 = Arc::new(new_node("L2"));
        let l1_v2 = Arc::new(new_node("L1"));
        let l2_v2 = Arc::new(new_node("L2"));

        let n1 = Arc::new(new_node_with_preds("N_val_A", vec![l1_v1.clone(), l2_v1.clone()]));
        let n2 = Arc::new(new_node_with_preds("N_val_A", vec![l1_v2.clone(), l2_v2.clone()]));
        
        // The GSSNodes themselves should be equal due to internal canonicalization on creation
        assert_eq!(n1.as_ref(), n2.as_ref(), "Original N1 and N2 GSSNodes should be content-equal");
        // But the Arcs are distinct initially
        assert!(!Arc::ptr_eq(&n1, &n2), "Original N1 and N2 Arcs should be different instances");


        let r1 = Arc::new(new_node_with_preds("R1", vec![n1.clone()])); // Clone n1 Arc
        let r2 = Arc::new(new_node_with_preds("R2", vec![n2.clone()])); // Clone n2 Arc

        let roots_to_simplify = vec![r1.clone(), r2.clone()]; // Clone roots for the function
        let simplified_roots = simplify_gss_forest(&roots_to_simplify);
        assert_eq!(simplified_roots.len(), 2);

        let sr1_n = simplified_roots[0].get_predecessors_arcs()[0].clone();
        let sr2_n = simplified_roots[1].get_predecessors_arcs()[0].clone();

        assert_eq!(sr1_n.value, "N_val_A".to_string());
        assert!(Arc::ptr_eq(&sr1_n, &sr2_n), "Nodes N under R1 and R2 should simplify to the same Arc instance.");

        let n_preds_arcs = sr1_n.get_predecessors_arcs();
        assert_eq!(n_preds_arcs.len(), 2);
        let mut simplified_leaf_values: Vec<&str> = n_preds_arcs.iter().map(|arc| arc.value.as_str()).collect();
        simplified_leaf_values.sort_unstable(); // Order from BTreeSet depends on String Ord
        assert_eq!(simplified_leaf_values, vec!["L1", "L2"]);
        
        // Check that L1 and L2 are themselves canonicalized
        let s_l1_from_n = n_preds_arcs.iter().find(|n| n.value == "L1").unwrap().clone();
        let s_l2_from_n = n_preds_arcs.iter().find(|n| n.value == "L2").unwrap().clone();

        // To verify L1 and L2 are canonical, simplify them directly as well.
        let initial_leaves = vec![l1_v1.clone(), l1_v2.clone(), l2_v1.clone(), l2_v2.clone()]; // Clone for simplify
        let simplified_leaves = simplify_gss_forest(&initial_leaves);
        assert_eq!(simplified_leaves.len(), 4);

        let s_l1_direct1 = simplified_leaves[0].clone();
        let s_l1_direct2 = simplified_leaves[1].clone();
        let s_l2_direct1 = simplified_leaves[2].clone();
        let s_l2_direct2 = simplified_leaves[3].clone();

        assert!(Arc::ptr_eq(&s_l1_from_n, &s_l1_direct1), "Simplified L1 from N should be pointer-equal to directly simplified L1");
        assert!(Arc::ptr_eq(&s_l1_direct1, &s_l1_direct2), "Directly simplified L1s should be pointer-equal");
        assert!(Arc::ptr_eq(&s_l2_from_n, &s_l2_direct1), "Simplified L2 from N should be pointer-equal to directly simplified L2");
        assert!(Arc::ptr_eq(&s_l2_direct1, &s_l2_direct2), "Directly simplified L2s should be pointer-equal");

        // Also check that the root nodes R1 and R2 were not canonicalized (they are different)
        assert!(!Arc::ptr_eq(&simplified_roots[0], &simplified_roots[1]));

         // Check that the original and simplified roots are structurally equal
        assert_eq!(r1.as_ref(), simplified_roots[0].as_ref());
        assert_eq!(r2.as_ref(), simplified_roots[1].as_ref());
    }
}
