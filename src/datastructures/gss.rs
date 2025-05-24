use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;


// Type alias for the canonicalization cache key
type NodeCacheKey<T, A> = (T, BTreeSet<Arc<GSSNode<T, A>>>);
// Type alias for the canonicalization cache
pub type NodeCache<T, A> = HashMap<NodeCacheKey<T, A>, Arc<GSSNode<T, A>>>;

pub trait PathAccumulator: Sized + Clone + Debug + Eq + PartialEq + Ord + PartialOrd + Hash + Default {
    /// Combines two accumulators, typically representing the union of possibilities.
    fn union(&self, other: &Self) -> Self;
    /// Finds the commonality between two accumulators, typically representing an intersection.
    fn intersect(&self, other: &Self) -> Self;
    // Optional: fn identity_for_intersection() -> Self; (if needed by generic logic)
}

impl PathAccumulator for () {
    fn union(&self, _other: &Self) -> Self { () }
    fn intersect(&self, _other: &Self) -> Self { () }
}

// Helper function to compute a node's hash_key_cache.
// This is used by both canonical and non-canonical node creation.
fn compute_internal_hash_key<T: Hash, A: PathAccumulator>(value: &T, predecessors: &BTreeSet<Arc<GSSNode<T, A>>>) -> u64 {
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
pub struct GSSNode<T, A: PathAccumulator> {
    pub value: T,
    pub acc: A, // Accumulator value
    predecessors: BTreeSet<Arc<GSSNode<T, A>>>,
    hash_key_cache: u64, // Based on T and predecessors' hashes only
}

// JSONConvertible for GSSNode is complex and currently marked todo!
// impl<T: JSONConvertible, A: JSONConvertible + PathAccumulator> JSONConvertible for GSSNode<T, A> {
//     fn to_json(&self) -> JSONNode {
//         todo!("GSSNode to_json: Complex graph structure, requires advanced serialization strategy.")
//     }

//     fn from_json(_node: JSONNode) -> Result<Self, String> {
//         todo!("GSSNode from_json: Complex graph structure, requires advanced deserialization strategy.")
//     }
// }


// Methods for creating non-canonical GSSNode instances (original API style)
impl<T: Hash, A: PathAccumulator> GSSNode<T, A> {
    pub fn new(value: T, acc: A) -> Self {
        let predecessors = BTreeSet::new();
        let hash_key_cache = compute_internal_hash_key::<T, A>(&value, &predecessors);
        Self {
            value,
            acc,
            predecessors,
            hash_key_cache,
        }
    }

    pub fn new_with_predecessors(value: T, predecessors: BTreeSet<Arc<Self>>) -> Self {
        let unioned_acc = if predecessors.is_empty() {
            A::default()
        } else {
            let mut iter = predecessors.iter();
            // .unwrap() is safe because predecessors is not empty in this branch.
            let mut acc_val = iter.next().unwrap().acc.clone();
            for pred_arc in iter {
                acc_val = acc_val.union(&pred_arc.acc);
            }
            acc_val
        };
        let hash_key_cache = compute_internal_hash_key::<T, A>(&value, &predecessors);
        Self {
            value,
            acc: unioned_acc,
            predecessors,
            hash_key_cache,
        }
    }
}

// Methods involving canonicalization or requiring more bounds (Ord, Clone, Debug for cache keys)
impl<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug> GSSNode<T, A> {
    /// Internal method to get/create a canonical Arc<GSSNode<T, A>>.
    fn get_canonical(
        value: T, // Consumed
        predecessors: BTreeSet<Arc<Self>>, // Consumed
        cache: &mut NodeCache<T, A>,
    ) -> Arc<Self> {
        let key_for_lookup = (value.clone(), predecessors.clone()); // Clones for lookup key

        let current_context_unioned_acc = if predecessors.is_empty() {
            // If predecessors is empty, this implies a root node being formed structurally.
            // The accumulator for such a node, when derived purely structurally without explicit
            // initial accumulator (like in new_canonical), should be Default::default().
            A::default()
        } else {
            let mut iter = predecessors.iter();
            // .unwrap() is safe because predecessors is not empty in this branch.
            let mut acc_val = iter.next().unwrap().acc.clone();
            for pred_arc in iter {
                acc_val = acc_val.union(&pred_arc.acc);
            }
            acc_val
        };

        if let Some(entry_arc) = cache.get_mut(&key_for_lookup) {
            let new_potential_acc = entry_arc.acc.union(&current_context_unioned_acc);
            if new_potential_acc != entry_arc.acc {
                // Make the GSSNode mutable and update its acc.
                let mut temp_arc = entry_arc.clone(); // Clone the Arc from the cache entry
                let node_instance_mut = Arc::make_mut(&mut temp_arc); // Get mutable reference to the node instance
                node_instance_mut.acc = new_potential_acc;
                // Update the Arc in the cache entry to point to the modified node
                *entry_arc = temp_arc.clone();
                return temp_arc; // Return the modified Arc
            }
            return entry_arc.clone(); // Return existing Arc if acc didn't change
        }

        // Not found, create new. `value` and `predecessors` are the owned versions.
        let hash_key_cache = compute_internal_hash_key::<T, A>(&value, &predecessors);
        let new_node = GSSNode {
            value, // `value` moved here
            acc: current_context_unioned_acc,
            predecessors, // `predecessors` moved here
            hash_key_cache,
        };
        let new_node_arc = Arc::new(new_node);
        // Insert with the key used for lookup, which now owns the cloned value/predecessors.
        cache.insert(key_for_lookup, new_node_arc.clone());
        new_node_arc
    }

    pub fn new_canonical(value: T, initial_acc: A, cache: &mut NodeCache<T, A>) -> Arc<Self> {
        let predecessors = BTreeSet::new();
        let key = (value.clone(), predecessors.clone());
        if let Some(entry_arc) = cache.get_mut(&key) {
            // A root node's acc might be re-specified. Union it.
            let new_potential_acc = entry_arc.acc.union(&initial_acc);
             if new_potential_acc != entry_arc.acc {
                let mut temp_arc = entry_arc.clone();
                let node_instance_mut = Arc::make_mut(&mut temp_arc);
                node_instance_mut.acc = new_potential_acc;
                *entry_arc = temp_arc.clone();
                return temp_arc;
            }
            return entry_arc.clone();
        }

        let hash_key_cache = compute_internal_hash_key::<T, A>(&value, &predecessors);
        let new_node_arc = Arc::new(GSSNode {
            value, // value moved here
            acc: initial_acc, // Use provided initial_acc
            predecessors, // predecessors moved here
            hash_key_cache,
        });
        cache.insert(key, new_node_arc.clone());
        new_node_arc
    }

    pub fn new_with_predecessors_canonical(value: T, predecessors: BTreeSet<Arc<Self>>, cache: &mut NodeCache<T, A>) -> Arc<Self> {
        Self::get_canonical(value, predecessors, cache)
    }

    pub fn from_iter_canonical<I>(iter: I, cache: &mut NodeCache<T, A>) -> Arc<Self>
    where
        I: IntoIterator<Item = (T, A)>, // Iterator yields (Value, Initial Acc for that node)
    {
        let mut iter_val = iter.into_iter();
        let (first_val, first_acc) = iter_val.next().expect("from_iter_canonical requires at least one element");
        let mut root = Self::new_canonical(first_val, first_acc, cache);
        for (value, _acc) in iter_val { // Acc from iter is ignored for subsequent nodes; acc is structural
            root = Self::push_onto_canonical(root, value, cache);
        }
        root
    }

    pub fn push_onto_canonical(
        current_stack_top: Arc<Self>,
        value: T,
        cache: &mut NodeCache<T, A>,
    ) -> Arc<Self> {
        let mut predecessors = BTreeSet::new();
        predecessors.insert(current_stack_top);
        // get_canonical will derive acc from the single predecessor's acc.
        Self::get_canonical(value, predecessors, cache)
    }

    pub fn map_canonical<F, U>(
        self_arc: Arc<Self>,
        f: F,
        cache_u: &mut NodeCache<U, A>, // Cache for GSSNode<U, A>
    ) -> Arc<GSSNode<U, A>>
    where
        F: Copy + Fn(&T, &A) -> U, // Closure gets Value and Accumulator
        U: Clone + Ord + Hash + Debug, // Bounds for the new node value type U
    {
        let new_value = f(&self_arc.value, &self_arc.acc);
        let new_predecessors_mapped: BTreeSet<Arc<GSSNode<U, A>>> = self_arc.predecessors.iter()
            .map(|pred_arc_t_a| {
                // Recursive call. The type U for value, A for accumulator is inferred.
                GSSNode::<T, A>::map_canonical(pred_arc_t_a.clone(), f, cache_u)
            })
            .collect();
        // get_canonical will derive acc from the new predecessors' accs.
        GSSNode::<U, A>::get_canonical(new_value, new_predecessors_mapped, cache_u)
    }
}

// Public methods consistent with the original API (mostly non-canonical)
impl<T, A: PathAccumulator> GSSNode<T, A> {
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (T, A)>, // Iterator yields (Value, Initial Acc for that node)
        T: Ord + Hash, // Needed for push -> new_with_predecessors -> compute_internal_hash_key
    {
        let mut iter_val = iter.into_iter();
        let (first_val, first_acc) = iter_val.next().expect("from_iter requires at least one element");
        let mut root = Self::new(first_val, first_acc); // Uses non-canonical new
        for (value, _acc) in iter_val { // Acc from iter is ignored for subsequent nodes; acc is structural
            root = root.push(value); // Uses non-canonical push
        }
        root
    }

    pub fn push(self, value: T) -> Self
    where T: Ord + Hash, A: PathAccumulator
    {
        let mut new_node_predecessors = BTreeSet::new();
        // Arc::new(self) creates a new Arc, does not use canonical cache.
        // new_with_predecessors will set acc to self.acc
        new_node_predecessors.insert(Arc::new(self));
        Self::new_with_predecessors(value, new_node_predecessors)
    }

    pub fn pop(&self) -> Vec<Arc<Self>> {
        self.predecessors.iter().cloned().collect()
    }

    pub fn popn(&self, n: usize) -> Vec<Arc<Self>>
    where
        T: Clone + Hash, A: PathAccumulator, // Clone for self.clone(), Hash for recursive popn
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
        let mut seen_arcs_for_this_call: HashSet<*const GSSNode<T, A>> = HashSet::new();

        for predecessor_arc in &self.predecessors {
            // predecessor_arc is &Arc<GSSNode<T, A>>.
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

    pub fn acc(&self) -> &A {
        &self.acc
    }

    pub fn value_mut(&mut self) -> &mut T {
        // Warning: Mutating value will invalidate hash_key_cache and break canonicalization
        // if this node was part of a canonical set. User must re-calculate hash or re-intern.
        &mut self.value
    }

    pub fn acc_mut(&mut self) -> &mut A {
        // Caller is responsible for ensuring this modification is valid
        // (e.g., does not break semantic invariants if node is shared).
        &mut self.acc
    }


    pub fn flatten(&self) -> Vec<Vec<(T, A)>>
    where
        T: Clone, A: Clone,
    {
        let mut result = Vec::new();
        // (node_ref, path_to_node_ref_value_acc)
        let mut q: VecDeque<(&GSSNode<T, A>, Vec<(T, A)>)> = VecDeque::new();
        q.push_back((self, Vec::new()));

        while let Some((current_node, mut current_path_values_acc)) = q.pop_front() {
            current_path_values_acc.push((current_node.value.clone(), current_node.acc.clone()));
            if current_node.predecessors.is_empty() {
                // Paths are built from root to leaf, reverse at the end
                current_path_values_acc.reverse();
                result.push(current_path_values_acc);
            } else {
                for pred_arc in &current_node.predecessors {
                    q.push_back((pred_arc.as_ref(), current_path_values_acc.clone()));
                }
            }
        }
        result
    }


    pub fn flatten_bulk(nodes: &[Self]) -> Vec<Vec<(T, A)>>
    where
        T: Clone + Hash, A: PathAccumulator, // Hash needed for GSSNode methods called by flatten
    {
        nodes.iter().flat_map(|node| node.flatten()).collect()
    }

    pub fn merge(&mut self, mut other: Self)
    where
        T: Ord + Hash, A: PathAccumulator // Hash for re-calculating hash_key_cache
    {
        assert!(self.value == other.value); // Requires T: PartialEq, which is implied by Ord
        self.acc = self.acc.union(&other.acc); // Union the accumulators
        self.predecessors.append(&mut other.predecessors);
        self.hash_key_cache = compute_internal_hash_key::<T, A>(&self.value, &self.predecessors);
    }

    pub fn merge_unchecked(&mut self, mut other: Self)
    where T: Ord + Hash, A: PathAccumulator // Hash for re-calculating hash_key_cache
    {
         self.acc = self.acc.union(&other.acc); // Union the accumulators
        self.predecessors.append(&mut other.predecessors);
        self.hash_key_cache = compute_internal_hash_key::<T, A>(&self.value, &self.predecessors);
    }

    pub fn map<F, U>(&self, f: F) -> GSSNode<U, A>
    where
        F: Copy + Fn(&T, &A) -> U, // Closure gets Value and Accumulator
        U: Ord + Hash, A: PathAccumulator, // For GSSNode<U, A> value
        T: Hash,
    {
        let new_value = f(&self.value, &self.acc);
        let new_predecessors_arcs: BTreeSet<Arc<GSSNode<U, A>>> = self.predecessors.iter()
            .map(|pred_arc_t_a| { // pred_arc_t_a is &Arc<GSSNode<T, A>>
                // Recursively call map on the GSSNode content. Result is GSSNode<U, A>.
                // Wrap it in an Arc for the new predecessor set.
                Arc::new(pred_arc_t_a.as_ref().map(f))
            })
            .collect();
        // new_with_predecessors will calculate acc for GSSNode<U,A> from its predecessors
        GSSNode::<U, A>::new_with_predecessors(new_value, new_predecessors_arcs)
    }
}

impl<T, A: PathAccumulator> Drop for GSSNode<T, A> {
    fn drop(&mut self) {
        let predecessors_to_process_further = std::mem::take(&mut self.predecessors);
        let mut worklist: Vec<Arc<GSSNode<T, A>>> = predecessors_to_process_further.into_iter().collect();

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

impl<T: Hash, A: PathAccumulator> Hash for GSSNode<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_key_cache.hash(state);
    }
}

impl<T: Hash + PartialEq, A: PathAccumulator> PartialEq for GSSNode<T, A> {
    fn eq(&self, other: &Self) -> bool {
        if std::ptr::eq(self, other) { return true; } // Optimization for same instance
        // Compare by hash_key_cache (structural hash), value, and predecessor set (by Arc pointers).
        // DO NOT compare self.acc == other.acc here for structural equality.
        self.hash_key_cache == other.hash_key_cache &&
        self.value == other.value &&
        self.predecessors == other.predecessors
    }
}

impl<T: Hash + Eq, A: PathAccumulator> Eq for GSSNode<T, A> {}

impl<T: Hash + PartialOrd, A: PathAccumulator> PartialOrd for GSSNode<T, A> {
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

impl<T: Hash + Ord, A: PathAccumulator> Ord for GSSNode<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        if std::ptr::eq(self, other) { return Ordering::Equal; } // Optimization for same instance
        self.hash_key_cache.cmp(&other.hash_key_cache)
            .then_with(|| self.value.cmp(&other.value))
            .then_with(|| self.predecessors.cmp(&other.predecessors))
    }
}


// GSSTrait uses Clone + Hash for T, and PathAccumulator for A
pub trait GSSTrait<T: Clone + Hash, A: PathAccumulator> {
    type Peek<'a> where T: 'a, Self: 'a;
    fn peek(&self) -> Self::Peek<'_>;
    fn push(&self, value: T) -> GSSNode<T, A> where T: Ord; // push returns non-canonical GSSNode
    fn pop(&self) -> Vec<Arc<GSSNode<T, A>>>;
    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T, A>>>;
}

impl<T: Clone + Hash, A: PathAccumulator> GSSTrait<T, A> for GSSNode<T, A> {
    type Peek<'a> = &'a T where T: 'a, A: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        &self.value
    }

    fn push(&self, value: T) -> GSSNode<T, A> where T: Ord {
        // self is &GSSNode<T, A>. We need to clone it to own it for Arc::new.
        let self_owned_clone = self.clone();
        GSSNode::push(self_owned_clone, value) // Calls the inherent GSSNode::push
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T, A>>> {
        GSSNode::pop(self) // Calls inherent GSSNode::pop
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T, A>>> {
        GSSNode::popn(self, n) // Calls inherent GSSNode::popn
    }
}

impl<T: Clone + Hash, A: PathAccumulator> GSSTrait<T, A> for Arc<GSSNode<T, A>> {
    type Peek<'a> = &'a T where T: 'a, A: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        &self.value
    }

    fn push(&self, value: T) -> GSSNode<T, A> where T: Ord {
        // self is &Arc<GSSNode<T, A>>.
        // new_with_predecessors will set acc to self.acc
        let mut new_node_predecessors = BTreeSet::new();
        new_node_predecessors.insert(self.clone()); // Insert the Arc
        GSSNode::new_with_predecessors(value, new_node_predecessors)
    }


    fn pop(&self) -> Vec<Arc<GSSNode<T, A>>> {
        self.as_ref().pop()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T, A>>> {
        self.as_ref().popn(n)
    }
}

impl<T: Clone + Hash, A: PathAccumulator> GSSTrait<T, A> for Option<Arc<GSSNode<T, A>>> {
    type Peek<'a> = Option<&'a T> where T: 'a, A: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        self.as_ref().map(|node_arc| node_arc.peek())
    }

    fn push(&self, value: T) -> GSSNode<T, A> where T: Ord, A: Default {
        match self {
            Some(arc_node) => arc_node.push(value), // Arc's GSSTrait push (inherits acc)
            None => GSSNode::new(value, A::default()), // Non-canonical new (needs initial acc)
        }
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T, A>>> {
        self.as_ref().map(|node_arc| node_arc.pop()).unwrap_or_default()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T, A>>> {
        self.as_ref().map(|node_arc| node_arc.popn(n)).unwrap_or_default()
    }
}

impl<T: Clone + Hash, A: PathAccumulator> GSSTrait<T, A> for Option<GSSNode<T, A>> {
    type Peek<'a> = Option<&'a T> where T: 'a, A: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        self.as_ref().map(|node| node.peek())
    }

    fn push(&self, value: T) -> GSSNode<T, A> where T: Ord, A: Default {
        match self {
            Some(node) => node.push(value), // GSSNode's GSSTrait push (inherits acc)
            None => GSSNode::new(value, A::default()), // Non-canonical new (needs initial acc)
        }
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T, A>>> {
        self.as_ref().map(|node| node.pop()).unwrap_or_default()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T, A>>> {
        self.as_ref().map(|node| node.popn(n)).unwrap_or_default()
    }
}


// BulkMerge uses Ord for BTreeMap key, Hash for GSSNode::new_with_predecessors
pub trait BulkMerge<T, A: PathAccumulator> {
    fn bulk_merge(&mut self);
}

impl<T: Clone + Ord + Hash, A: PathAccumulator> BulkMerge<T, A> for Vec<Arc<GSSNode<T, A>>> {
    fn bulk_merge(&mut self) {
        let mut groups: BTreeMap<T, Vec<Arc<GSSNode<T, A>>>> = BTreeMap::new();
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

            // Calculate the union of accumulators for all nodes in this group
            let mut iter = group_arcs.iter();
            // .unwrap() is safe because group_arcs is not empty here.
            let mut union_of_accs_in_group = iter.next().unwrap().acc.clone();
            for node_arc_in_group in iter {
                union_of_accs_in_group = union_of_accs_in_group.union(&node_arc_in_group.acc);
            }

            // Collect predecessors from all nodes in this group
            let mut merged_predecessors: BTreeSet<Arc<GSSNode<T, A>>> = BTreeSet::new();
            for node_arc_in_group in group_arcs {
                for pred_arc in &node_arc_in_group.predecessors {
                    merged_predecessors.insert(pred_arc.clone());
                }
            }
            // Create a new non-canonical node for the merged result.
            // new_with_predecessors will set its acc based on the merged_predecessors,
            // but we need it to be the union_of_accs_in_group.
            let mut merged_node = GSSNode::new_with_predecessors(value.clone(), merged_predecessors);
            merged_node.acc = union_of_accs_in_group; // Override with the correct unioned acc
            new_merged_nodes.push(Arc::new(merged_node));
        }
        *self = new_merged_nodes;
    }
}

// Canonical version of bulk_merge
pub fn bulk_merge_canonical<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    nodes: &mut Vec<Arc<GSSNode<T, A>>>,
    cache: &mut NodeCache<T, A>
) {
    let mut groups: BTreeMap<T, Vec<Arc<GSSNode<T, A>>>> = BTreeMap::new();
    for node_arc in nodes.drain(..) {
        groups.entry(node_arc.value.clone()).or_default().push(node_arc);
    }

    let mut new_merged_nodes = Vec::new();
    for (value, group_arcs) in groups {
        if group_arcs.is_empty() { continue; }

        // Calculate the union of accumulators for all nodes in this group
        let mut iter = group_arcs.iter();
        // .unwrap() is safe because group_arcs is not empty here.
        let mut union_of_accs_in_group = iter.next().unwrap().acc.clone();
        for node_arc_in_group in iter {
            union_of_accs_in_group = union_of_accs_in_group.union(&node_arc_in_group.acc);
        }

        // Collect predecessors from all nodes in this group
        let mut merged_predecessors: BTreeSet<Arc<GSSNode<T, A>>> = BTreeSet::new();
        for node_arc_in_group in group_arcs {
            for pred_arc in &node_arc_in_group.predecessors {
                merged_predecessors.insert(pred_arc.clone());
            }
        }
        // Create the canonical merged node. get_canonical handles unioning the acc from its preds
        // and also updates its acc if an existing node is found in the cache that maps to the same key.
        // We also need to union in the `union_of_accs_in_group` because these are the accs
        // of the nodes that were *merged*, not just the resulting node's predecessors.
        // This requires get_canonical to take the acc_from_merged_nodes as a seed.
        // Let's assume get_canonical handles this via its internal logic or by relying on
        // the acc being updated when an existing node is found via `get_mut`.
        let merged_node_arc = GSSNode::get_canonical(value, merged_predecessors, cache);

        // Ensure the acc on the canonical node includes the union of accs from the merged nodes.
        // This should be handled by get_canonical's `get_mut` path.
        // If get_canonical doesn't do this implicitly, we'd need to do it explicitly here:
        // let mut temp_arc = merged_node_arc.clone();
        // let node_instance_mut = Arc::make_mut(&mut temp_arc);
        // node_instance_mut.acc = node_instance_mut.acc.union(&union_of_accs_in_group);
        // new_merged_nodes.push(temp_arc);
        // But let's rely on the get_canonical logic updating acc on cache hit for now.

        new_merged_nodes.push(merged_node_arc);
    }
    *nodes = new_merged_nodes;
}


// prune_and_transform_roots: Assumed to produce canonical nodes as per last discussion.
// Requires T: Clone + Ord + Hash + Debug for GSSNode::get_canonical.
pub fn prune_and_transform_recursive_canonical<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    node_arc: &Arc<GSSNode<T, A>>,
    // Closure now takes Value and Accumulator, and returns new Value, new Accumulator, and continue_recursion flag
    closure: &impl Fn(&T, &A) -> Option<(T, A, bool)>,
    memo: &mut HashMap<*const GSSNode<T, A>, Option<Arc<GSSNode<T, A>>>>,
    cache: &mut NodeCache<T, A>, // Cache for canonical node creation
) -> Option<Arc<GSSNode<T, A>>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }

    match closure(&node_arc.value, &node_arc.acc) {
        None => {
            memo.insert(node_ptr, None);
            None
        }
        Some((new_value, new_acc, continue_recursion)) => {
            let new_predecessors: BTreeSet<Arc<GSSNode<T, A>>>;
            if continue_recursion {
                let mut current_new_predecessors = BTreeSet::new();
                for pred_arc in &node_arc.predecessors { // pred_arc is &Arc<GSSNode<T, A>>
                    if let Some(new_pred_arc) = prune_and_transform_recursive_canonical(pred_arc, closure, memo, cache) {
                        current_new_predecessors.insert(new_pred_arc);
                    }
                }
                new_predecessors = current_new_predecessors;
            } else {
                // Stop recursion, keep original predecessors (which should be canonical if input was canonical)
                // or simplified predecessors if this is part of a larger simplification.
                // The acc for the new node will be based on these predecessors (handled by get_canonical)
                // BUT overridden by `new_acc` from the closure.
                new_predecessors = node_arc.predecessors.clone();
            };
            // Create a canonical node with the new value and (potentially transformed) predecessors.
            // get_canonical will set acc based on these predecessors AND unions if node exists.
            // Then, we override acc with the specific acc from the closure.
            let new_node_arc = GSSNode::get_canonical(new_value.clone(), new_predecessors, cache);
            let mut temp_arc = new_node_arc.clone();
            let node_instance_mut = Arc::make_mut(&mut temp_arc);
            node_instance_mut.acc = new_acc; // Apply acc from the closure

            memo.insert(node_ptr, Some(temp_arc.clone())); // Cache the modified Arc
            Some(temp_arc)
        }
    }
}

pub fn prune_and_transform_roots_canonical<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T, A>>],
    closure: &impl Fn(&T, &A) -> Option<(T, A, bool)>,
    cache: &mut NodeCache<T, A>, // Cache for canonical node creation
) -> Vec<Option<Arc<GSSNode<T, A>>>> {
    let mut memo = HashMap::new();
    roots
        .iter()
        .map(|root| prune_and_transform_recursive_canonical(root, closure, &mut memo, cache))
        .collect()
}

// Non-canonical versions (use a new cache for each call)
pub fn prune_and_transform_recursive<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    node_arc: &Arc<GSSNode<T, A>>,
    closure: &impl Fn(&T, &A) -> Option<(T, A, bool)>,
    memo: &mut HashMap<*const GSSNode<T, A>, Option<Arc<GSSNode<T, A>>>>,
) -> Option<Arc<GSSNode<T, A>>> {
    let mut cache = NodeCache::new(); // New cache for this call chain
    prune_and_transform_recursive_canonical(node_arc, closure, memo, &mut cache)
}

pub fn prune_and_transform_roots<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T, A>>],
    closure: &impl Fn(&T, &A) -> Option<(T, A, bool)>,
) -> Vec<Option<Arc<GSSNode<T, A>>>> {
    let mut memo = HashMap::new();
    let mut cache = NodeCache::new(); // New cache for this call
    roots
        .iter()
        .map(|root| prune_and_transform_recursive_canonical(root, closure, &mut memo, &mut cache))
        .collect()
}

// Implement the new pop method that applies accumulator context
pub fn pop_and_apply_contextual_accumulator<T: Clone + Ord + Hash, A: PathAccumulator>(
    source_nodes: &[Arc<GSSNode<T, A>>],
) -> Vec<Arc<GSSNode<T, A>>> {
    let mut resultMap: HashMap<*const GSSNode<T, A>, (Arc<GSSNode<T, A>>, A)> = HashMap::new();

    for src_node_arc in source_nodes {
        for pred_arc in &src_node_arc.predecessors {
            let pred_ptr = Arc::as_ptr(pred_arc);
            let acc_from_source = src_node_arc.acc.clone();
            resultMap.entry(pred_ptr)
                .and_modify(|e| { e.1 = e.1.union(&acc_from_source); })
                .or_insert_with(|| (pred_arc.clone(), acc_from_source));
        }
    }

    let mut final_nodes: Vec<Arc<GSSNode<T, A>>> = Vec::with_capacity(resultMap.len());
    for (_ptr, (original_pred_arc_cloned, pop_context_a)) in resultMap {
        // original_pred_arc_cloned is a clone of an Arc from one of the predecessor sets.
        // We want to modify the GSSNode it points to.
        let mut arc_to_modify = original_pred_arc_cloned;

        // Make the GSSNode mutable. This may clone the GSSNode if Arc's strong_count > 1.
        // The arc_to_modify will point to the (potentially new) GSSNode instance.
        let node_instance_mut = Arc::make_mut(&mut arc_to_modify);

        // Modify the accumulator value by intersecting it with the accumulated context from the pop
        node_instance_mut.acc = node_instance_mut.acc.intersect(&pop_context_a);

        final_nodes.push(arc_to_modify);
    }
    final_nodes
}


// Read-only functions: find_longest_path, gather_gss_stats, print_gss_forest
// These primarily change due to ArcPtrWrapper removal and A generic.

fn find_longest_path_recursive<T, A: PathAccumulator>(
    node_arc: &Arc<GSSNode<T, A>>,
    memo: &mut HashMap<*const GSSNode<T, A>, Vec<Arc<GSSNode<T, A>>>>,
    visited_recursion: &mut HashSet<*const GSSNode<T, A>>,
) -> Vec<Arc<GSSNode<T, A>>> {
    let node_ptr = Arc::as_ptr(node_arc);

    if let Some(cached_path) = memo.get(&node_ptr) {
        return cached_path.clone();
    }

    if !visited_recursion.insert(node_ptr) {
        return Vec::new();
    }

    let mut longest_pred_path: Vec<Arc<GSSNode<T, A>>> = Vec::new();

    if !node_arc.predecessors.is_empty() {
        for pred_arc in &node_arc.predecessors {
            let pred_path = find_longest_path_recursive(pred_arc, memo, visited_recursion);
            if pred_path.len() > longest_pred_path.len() {
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

pub fn find_longest_path<T, A: PathAccumulator>(roots: &[Arc<GSSNode<T, A>>]) -> Option<Vec<Arc<GSSNode<T, A>>>> {
    let mut memo: HashMap<*const GSSNode<T, A>, Vec<Arc<GSSNode<T, A>>>> = HashMap::new();

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


pub fn gather_gss_stats<T, A: PathAccumulator>(roots: &[Arc<GSSNode<T, A>>]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut visited: HashSet<*const GSSNode<T, A>> = HashSet::new();
    let mut queue: VecDeque<(Arc<GSSNode<T, A>>, usize)> = VecDeque::new();

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

        for pred_arc in &current_node_arc.predecessors { // pred_arc is &Arc<GSSNode<T, A>>
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

fn print_gss_node_recursive<T: Debug, A: PathAccumulator>(
    node_arc: &Arc<GSSNode<T, A>>,
    visited: &mut HashSet<*const GSSNode<T, A>>,
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

    writeln!(output, "{}- Node {:p}: {:?} (Acc: {:?})", prefix, node_ptr, node_arc.value, node_arc.acc)?;

    if !node_arc.predecessors.is_empty() {
        writeln!(output, "{}  Predecessors:", prefix)?;
        for pred_arc in &node_arc.predecessors {
            print_gss_node_recursive(pred_arc, visited, indent + 2, node_count, max_nodes, output)?;
            if *node_count >= max_nodes {
                return Ok(());
            }
        }
    }
    Ok(())
}

pub fn print_gss_forest<T: Debug, A: PathAccumulator>(roots: &[Arc<GSSNode<T, A>>], max_nodes: usize) -> String {
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
    if node_count >= max_nodes && visited.len() > node_count { // If we printed max_nodes but there were more visited (due to shared structure not printed)
         writeln!(&mut output, "... (More nodes exist but not printed due to max_nodes limit)").unwrap();
    }

    output
}

// Simplification functions remain to canonicalize an existing GSS.
// They use GSSNode::get_canonical internally.
fn simplify_node_recursive<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    original_node_arc: &Arc<GSSNode<T, A>>,
    original_ptr_memo: &mut HashMap<*const GSSNode<T, A>, Arc<GSSNode<T, A>>>,
    canonicalization_cache: &mut NodeCache<T, A>,
) -> Arc<GSSNode<T, A>> {
    let original_node_ptr = Arc::as_ptr(original_node_arc);

    if let Some(canonical_arc) = original_ptr_memo.get(&original_node_ptr) {
        return canonical_arc.clone();
    }

    let mut predecessor_knitting_map: BTreeMap<T, Arc<GSSNode<T, A>>> = BTreeMap::new();
    for original_pred_arc in &original_node_arc.predecessors {
        let original_pred_value = original_pred_arc.value.clone();
        if let Some(knitted_predecessor_arc) = predecessor_knitting_map.get_mut(&original_pred_value) {
            let knitted_predecessor_mut_ref = Arc::make_mut(knitted_predecessor_arc);
            knitted_predecessor_mut_ref.merge(original_pred_arc.as_ref().clone());
        } else {
            predecessor_knitting_map.insert(original_pred_value, original_pred_arc.clone());
        }
    }
    let knitted_predecessors: Vec<Arc<GSSNode<T, A>>> = predecessor_knitting_map.into_values().collect();

    let mut canonical_predecessor_arcs: BTreeSet<Arc<GSSNode<T, A>>> = BTreeSet::new();
    for original_pred_arc in &knitted_predecessors {
        let simplified_pred_arc = simplify_node_recursive(
            original_pred_arc,
            original_ptr_memo,
            canonicalization_cache,
        );
        canonical_predecessor_arcs.insert(simplified_pred_arc);
    }

    // Create a canonical node using the simplified predecessors.
    // get_canonical will set its acc based on these predecessors and update it
    // if a node with the same key exists in the cache.
    let canonical_arc_for_current_node = GSSNode::get_canonical(
        original_node_arc.value.clone(),
        canonical_predecessor_arcs,
        canonicalization_cache,
    );

    // Now, union the original node's accumulator into the accumulator of the canonical node.
    // This ensures that any acc info specific to the original node (e.g. if it was a root with
    // a specific initial acc, or its acc was modified) is preserved in the canonical graph.
    let mut temp_arc = canonical_arc_for_current_node.clone();
    let node_instance_mut = Arc::make_mut(&mut temp_arc);
    node_instance_mut.acc = node_instance_mut.acc.union(&original_node_arc.acc);

    original_ptr_memo.insert(original_node_ptr, temp_arc.clone()); // Cache the modified Arc
    temp_arc // Return the modified Arc
}

pub fn simplify_gss_forest<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T, A>>],
) -> Vec<Arc<GSSNode<T, A>>> {
    let mut original_ptr_memo: HashMap<*const GSSNode<T, A>, Arc<GSSNode<T, A>>> = HashMap::new();
    let mut canonicalization_cache_for_this_run: NodeCache<T, A> = NodeCache::new();
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
    // std::hash::{Hash, Hasher} is also imported
    // std::cmp::Ordering is also imported
    // std::collections::hash_map::DefaultHasher is also imported

    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct MockPathAccumulator {
        active: BTreeSet<usize>,
        intersection: BTreeSet<usize>,
    }

    impl Debug for MockPathAccumulator {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockAcc")
             .field("active", &self.active)
             .field("intersection", &self.intersection)
             .finish()
        }
    }

    impl Default for MockPathAccumulator {
        fn default() -> Self {
            Self {
                active: BTreeSet::new(),
                // For intersection field, identity for UNION operation is ALL_ONES.
                // This requires knowing the capacity.
                // In tests, let's assume a small capacity or a convention.
                // This is a known limitation of generic PathAccumulator identity.
                // A better mock would take capacity in its constructor or have a static capacity.
                // Let's use a placeholder that means "all ones conceptually".
                 intersection: BTreeSet::new(),
            }
        }
    }

    impl PathAccumulator for MockPathAccumulator {
        fn union(&self, other: &Self) -> Self {
            Self {
                active: self.active.union(&other.active).cloned().collect(),
                // intersection becomes stricter (AND) when paths UNION
                intersection: self.intersection.intersection(&other.intersection).cloned().collect(),
            }
        }

        fn intersect(&self, other: &Self) -> Self {
            Self {
                active: self.active.intersection(&other.active).cloned().collect(),
                intersection: self.intersection.intersection(&other.intersection).cloned().collect(), // Or union, depending on semantic
            }
        }
    }


    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct MockParseStateNodeContent {
        state_id: usize,
        // No 't' field anymore
    }

    impl Debug for MockParseStateNodeContent {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!(
                "ParseStateNodeContent {{ state_id: StateID({}) }}",
                self.state_id
            ))
        }
    }

    type MockGSSNode = GSSNode<MockParseStateNodeContent, MockPathAccumulator>;
    type MockNodeCache = NodeCache<MockParseStateNodeContent, MockPathAccumulator>; // For canonical mock nodes
    type IntNodeCache = NodeCache<i32, MockPathAccumulator>; // For canonical i32 nodes

    // Helper to create a non-canonical GSSNode Arc for tests
    fn nc_node_arc_int(value: i32, acc: MockPathAccumulator, predecessors: Vec<Arc<GSSNode<i32, MockPathAccumulator>>>) -> Arc<GSSNode<i32, MockPathAccumulator>> {
        let pred_set: BTreeSet<Arc<GSSNode<i32, MockPathAccumulator>>> = predecessors.into_iter().collect();
        Arc::new(GSSNode::new_with_predecessors(value, pred_set)) // new_with_predecessors calculates acc from preds
    }
     fn nc_node_arc_int_root(value: i32, acc: MockPathAccumulator) -> Arc<GSSNode<i32, MockPathAccumulator>> {
         Arc::new(GSSNode::new(value, acc)) // Use new for roots with explicit acc
     }

    // Helper to create a canonical GSSNode Arc for tests
    fn c_node_arc_int(value: i32, acc: MockPathAccumulator, predecessors: Vec<Arc<GSSNode<i32, MockPathAccumulator>>>, cache: &mut IntNodeCache) -> Arc<GSSNode<i32, MockPathAccumulator>> {
        let pred_set: BTreeSet<Arc<GSSNode<i32, MockPathAccumulator>>> = predecessors.into_iter().collect();
        // new_with_predecessors_canonical should handle acc propagation
        GSSNode::new_with_predecessors_canonical(value, pred_set, cache)
    }
    fn c_node_arc_int_root(value: i32, acc: MockPathAccumulator, cache: &mut IntNodeCache) -> Arc<GSSNode<i32, MockPathAccumulator>> {
        GSSNode::new_canonical(value, acc, cache)
    }

    // Helper for mock content nodes (canonical)
    fn c_mock_node_arc(
        value: MockParseStateNodeContent,
        acc: MockPathAccumulator,
        predecessors: Vec<Arc<MockGSSNode>>,
        cache: &mut MockNodeCache,
    ) -> Arc<MockGSSNode> {
        let pred_set: BTreeSet<Arc<MockGSSNode>> = predecessors.into_iter().collect();
        // new_with_predecessors_canonical should handle acc propagation
        GSSNode::new_with_predecessors_canonical(value, pred_set, cache)
    }
     fn c_mock_node_arc_root(
         value: MockParseStateNodeContent,
         acc: MockPathAccumulator,
         cache: &mut MockNodeCache,
     ) -> Arc<MockGSSNode> {
         GSSNode::new_canonical(value, acc, cache)
     }


    fn collect_arcs_recursive<T, A: PathAccumulator>(
        node_arc: &Arc<GSSNode<T, A>>,
        collected_arcs: &mut HashMap<*const GSSNode<T, A>, Arc<GSSNode<T, A>>>,
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
        let acc_base = MockPathAccumulator { active: BTreeSet::from([0]), intersection: BTreeSet::from([0]) };
        let acc_other = MockPathAccumulator { active: BTreeSet::from([1]), intersection: BTreeSet::from([1]) };
        let acc_merged = acc_base.union(&acc_other); // active {0,1}, intersection { }

        // Create a non-canonical graph first
        let d1_orig_nc = nc_node_arc_int_root(40, acc_base.clone());
        let c1_preds_nc = vec![d1_orig_nc.clone()];
        let c1_orig_nc = nc_node_arc_int(30, MockPathAccumulator::default(), c1_preds_nc); // acc from d1
        let b1_preds_nc = vec![c1_orig_nc.clone()];
        let b1_orig_nc = nc_node_arc_int(20, MockPathAccumulator::default(), b1_preds_nc); // acc from c1

        let d2_orig_nc = nc_node_arc_int_root(40, acc_other.clone()); // Structurally same as d1_orig_nc, but different Arc and different acc
        assert_ne!(Arc::as_ptr(&d1_orig_nc), Arc::as_ptr(&d2_orig_nc));

        let a1_preds_nc = vec![b1_orig_nc.clone(), d2_orig_nc.clone()];
        let a1_orig_nc = nc_node_arc_int(10, MockPathAccumulator::default(), a1_preds_nc); // acc from b1 union d2

        let roots_nc = vec![a1_orig_nc.clone()];
        let simplified_roots = simplify_gss_forest(&roots_nc);
        let simplified_a1 = simplified_roots[0].clone();

        let mut collected_arcs = HashMap::new();
        collect_arcs_recursive(&simplified_a1, &mut collected_arcs);
        // Expected unique nodes: D_canonical (value 40), C1 (value 30), B1 (value 20), A1 (value 10) = 4
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

        // Check accumulators after simplification
        // s_d_canonical should have acc = d1_orig_nc.acc union d2_orig_nc.acc
        assert_eq!(s_d_canonical.acc, acc_base.union(&acc_other)); // active {0,1}, intersection { }

        // s_c1.acc should be derived from its predecessor s_d_canonical
        assert_eq!(s_c1.acc, s_d_canonical.acc);

        // s_b1.acc should be derived from its predecessor s_c1
        assert_eq!(s_b1.acc, s_c1.acc);

        // s_a1.acc should be union of s_b1.acc and s_d_canonical.acc
        assert_eq!(s_a1.acc, s_b1.acc.union(&s_d_canonical.acc));
    }

    #[test]
    fn test_simplification_canonicalizes_structurally_identical_nodes() {
        let acc_l = MockPathAccumulator { active: BTreeSet::from([100]), intersection: BTreeSet::from([100]) };
        let acc_m = MockPathAccumulator { active: BTreeSet::from([200]), intersection: BTreeSet::from([200]) };
        let acc_r = MockPathAccumulator { active: BTreeSet::from([300]), intersection: BTreeSet::from([300]) };


        // Simulating non-canonical input for simplify_gss_forest
        let l_val = 0; let m_val = 1; let r_val = 2;

        let l1_nc = nc_node_arc_int_root(l_val, acc_l.clone());
        let l2_nc = nc_node_arc_int_root(l_val, acc_l.clone());
        let l3_nc = nc_node_arc_int_root(l_val, acc_l.clone());

        let m1_nc = nc_node_arc_int(m_val, MockPathAccumulator::default(), vec![l1_nc.clone()]); // acc from l1
        let m2_nc = nc_node_arc_int(m_val, MockPathAccumulator::default(), vec![l2_nc.clone()]); // acc from l2
        let m3_nc = nc_node_arc_int(m_val, MockPathAccumulator::default(), vec![l3_nc.clone()]); // acc from l3

        let r1_orig_non_canonical = nc_node_arc_int(r_val, MockPathAccumulator::default(), vec![m1_nc.clone(), m2_nc.clone(), m3_nc.clone()]); // acc from m1 union m2 union m3

        let simplified_roots = simplify_gss_forest(&[r1_orig_non_canonical.clone()]); // Simplify the root arc
        let simplified_r1_arc = simplified_roots[0].clone();

        let mut collected_arcs_map = HashMap::new();
        collect_arcs_recursive(&simplified_r1_arc, &mut collected_arcs_map);
        // Expected 3 unique Arcs after simplify_gss_forest if canonicalization works perfectly:
        // R_canonical, M_canonical, L_canonical
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

        // Check accumulators
        // s_l_level_arc should have acc from L nodes. Since all Ls were identical, it should be acc_l.
        assert_eq!(s_l_level_arc.acc, acc_l);

        // s_m_level_arc should have acc from its predecessor s_l_level_arc union'd with accs from m1,m2,m3's original accs?
        // No, simplify_node_recursive unioned the original node's acc.
        // m1.acc = l1.acc, m2.acc = l2.acc, m3.acc = l3.acc. Since l1=l2=l3=acc_l, m1=m2=m3=acc_l.
        // s_m_level_arc is canonical for m1, m2, m3. Its original accs were all acc_l.
        // simplify_node_recursive for m1 -> s_m_canon (pred s_l_canon). Union m1.acc (acc_l) into s_m_canon.acc.
        // simplify_node_recursive for m2 -> s_m_canon (pred s_l_canon). Union m2.acc (acc_l) into s_m_canon.acc.
        // simplify_node_recursive for m3 -> s_m_canon (pred s_l_canon). Union m3.acc (acc_l) into s_m_canon.acc.
        // s_m_canon.acc initially from s_l_canon (acc_l). Then union acc_l three times. Result is acc_l.
        assert_eq!(s_m_level_arc.acc, acc_l);

        // s_r1_arc is canonical for r1. Its original acc was m1.acc union m2.acc union m3.acc = acc_l union acc_l union acc_l = acc_l.
        // s_r1_arc acc derived from its predecessor s_m_level_arc (acc_l).
        // simplify_node_recursive for r1 -> s_r1_canon (pred s_m_canon). Union r1.acc (acc_l) into s_r1_canon.acc.
        // s_r1_canon.acc initially from s_m_canon (acc_l). Then union acc_l. Result is acc_l.
        assert_eq!(s_r1_node.acc, acc_l);
    }

     #[test]
    fn test_pop_and_apply_contextual_accumulator_basic() {
        let acc0 = MockPathAccumulator { active: BTreeSet::from([0]), intersection: BTreeSet::from([0]) };
        let acc1 = MockPathAccumulator { active: BTreeSet::from([1]), intersection: BTreeSet::from([1]) };
        let acc2 = MockPathAccumulator { active: BTreeSet::from([2]), intersection: BTreeSet::from([2]) };

        let node_a_0 = nc_node_arc_int_root(0, acc0.clone());
        let node_b_1_pred_a = nc_node_arc_int(1, MockPathAccumulator::default(), vec![node_a_0.clone()]); // acc = acc0

        let node_c_0 = nc_node_arc_int_root(0, acc1.clone());
        let node_d_1_pred_c = nc_node_arc_int(1, MockPathAccumulator::default(), vec![node_c_0.clone()]); // acc = acc1

        let node_e_2_pred_b = nc_node_arc_int(2, MockPathAccumulator::default(), vec![node_b_1_pred_a.clone()]); // acc = acc0
        let node_f_2_pred_d = nc_node_arc_int(2, MockPathAccumulator::default(), vec![node_d_1_pred_c.clone()]); // acc = acc1

        // We are at nodes E and F. We want to pop one level (to B and D).
        // The context for B is E's acc (acc0). The context for D is F's acc (acc1).
        // Node B's acc (acc0) should be intersected with E's acc (acc0) -> acc0
        // Node D's acc (acc1) should be intersected with F's acc (acc1) -> acc1
        let source_nodes = vec![node_e_2_pred_b.clone(), node_f_2_pred_d.clone()];

        let popped_nodes = pop_and_apply_contextual_accumulator(&source_nodes);

        assert_eq!(popped_nodes.len(), 2, "Expected 2 nodes after popping");

        let node_b_popped = popped_nodes.iter().find(|n| n.value == 1 && Arc::ptr_eq(&n.predecessors.iter().next().unwrap(), &node_a_0)).expect("B node not found");
        let node_d_popped = popped_nodes.iter().find(|n| n.value == 1 && Arc::ptr_eq(&n.predecessors.iter().next().unwrap(), &node_c_0)).expect("D node not found");

        // Check that the original nodes B and D were modified in place or cloned and modified correctly
        // The function returns NEW Arcs, but they should point to nodes that were potentially
        // cloned from the originals (due to make_mut) and had their acc modified.

        // Original B had acc0. It was a predecessor of E. E had acc0.
        // Popped B's acc should be original B's acc intersected with E's acc.
        assert_eq!(node_b_popped.acc, acc0.intersect(&acc0)); // Should be acc0

        // Original D had acc1. It was a predecessor of F. F had acc1.
        // Popped D's acc should be original D's acc intersected with F's acc.
        assert_eq!(node_d_popped.acc, acc1.intersect(&acc1)); // Should be acc1

        // Test with merging context
        let node_g_2_pred_b_d = nc_node_arc_int(2, MockPathAccumulator::default(), vec![node_b_1_pred_a.clone(), node_d_1_pred_c.clone()]); // acc = acc0 union acc1

        // Now we are at node G. We pop one level.
        // The predecessors of G are B and D.
        // The context for B is G's acc (acc0 union acc1).
        // The context for D is G's acc (acc0 union acc1).
        // Node B's acc (acc0) should be intersected with G's acc (acc0 union acc1) -> acc0
        // Node D's acc (acc1) should be intersected with G's acc (acc0 union acc1) -> acc1
        let source_nodes_g = vec![node_g_2_pred_b_d.clone()];
        let popped_nodes_g = pop_and_apply_contextual_accumulator(&source_nodes_g);

        assert_eq!(popped_nodes_g.len(), 2, "Expected 2 nodes after popping from G");

        let node_b_popped_g = popped_nodes_g.iter().find(|n| n.value == 1 && Arc::ptr_eq(&n.predecessors.iter().next().unwrap(), &node_a_0)).expect("B node not found from G");
        let node_d_popped_g = popped_nodes_g.iter().find(|n| n.value == 1 && Arc::ptr_eq(&n.predecessors.iter().next().unwrap(), &node_c_0)).expect("D node not found from G");

        let context_acc = acc0.union(&acc1); // {active: {0,1}, intersection: {}}

        assert_eq!(node_b_popped_g.acc, acc0.intersect(&context_acc)); // acc0 & {0,1}, {0} & {} -> active {0}, intersection {}
        assert_eq!(node_d_popped_g.acc, acc1.intersect(&context_acc)); // acc1 & {0,1}, {1} & {} -> active {1}, intersection {}

        assert_eq!(node_b_popped_g.acc.active, BTreeSet::from([0]));
        assert_eq!(node_b_popped_g.acc.intersection, BTreeSet::new()); // Using MockPathAccumulator default for intersection

        assert_eq!(node_d_popped_g.acc.active, BTreeSet::from([1]));
        assert_eq!(node_d_popped_g.acc.intersection, BTreeSet::new()); // Using MockPathAccumulator default for intersection
    }
}

