use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
// use std::ops::Deref; // Not explicitly used after review, can be removed if not needed by GSSTrait macro expansion
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use crate::json_serialization::{JSONConvertible, JSONNode}; // Assuming this exists from context
use std::collections::BTreeMap as StdMap; // For JSON


// Type alias for the canonicalization cache key
type NodeCacheKey<T, A> = (T, BTreeSet<Arc<GSSNode<T, A>>>);
// Type alias for the canonicalization cache
pub type NodeCache<T, A> = HashMap<NodeCacheKey<T, A>, Arc<GSSNode<T, A>>>;

pub trait PathAccumulator: Sized + Clone + Debug + Eq + PartialEq + Ord + PartialOrd + Hash + Default {
    /// Combines two accumulators, typically representing the union of possibilities.
    fn union(&self, other: &Self) -> Self;
    /// Finds the commonality between two accumulators, typically representing an intersection.
    fn intersect(&self, other: &Self) -> Self;
}

impl PathAccumulator for () {
    fn union(&self, _other: &Self) -> Self { () }
    fn intersect(&self, _other: &Self) -> Self { () }
}

// Helper function to compute a node's hash_key_cache.
fn compute_internal_hash_key<T: Hash, A: PathAccumulator>(value: &T, predecessors: &BTreeSet<Arc<GSSNode<T, A>>>) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    // The BTreeSet ensures predecessors are iterated in a canonical order (by Arc pointer address).
    for pred_arc in predecessors {
        pred_arc.hash_key_cache.hash(&mut hasher);
    }
    hasher.finish()
}

// Helper to calculate unioned accumulator from predecessors
fn calculate_unioned_acc_from_predecessors<T, A: PathAccumulator>(
    predecessors: &BTreeSet<Arc<GSSNode<T, A>>>,
) -> A {
    predecessors.iter()
        .map(|p| p.acc.clone())
        .reduce(|acc, item| acc.union(&item))
        .unwrap_or_else(A::default)
}

#[derive(Debug, Clone)]
pub struct GSSNode<T, A: PathAccumulator> {
    pub value: T,
    pub acc: A,
    predecessors: BTreeSet<Arc<GSSNode<T, A>>>,
    hash_key_cache: u64, // Based on T and predecessors' hashes only
}

// JSONConvertible for GSSNode is complex and currently marked todo!
// impl<T: JSONConvertible, A: JSONConvertible + PathAccumulator> JSONConvertible for GSSNode<T, A> {
//     fn to_json(&self) -> JSONNode { todo!("GSSNode to_json") }
//     fn from_json(_node: JSONNode) -> Result<Self, String> { todo!("GSSNode from_json") }
// }


// Methods for creating non-canonical GSSNode instances
impl<T: Hash, A: PathAccumulator> GSSNode<T, A> {
    pub fn new(value: T, acc: A) -> Self {
        let predecessors = BTreeSet::new();
        let hash_key_cache = compute_internal_hash_key::<T, A>(&value, &predecessors);
        Self { value, acc, predecessors, hash_key_cache }
    }

    pub fn new_with_predecessors(value: T, predecessors: BTreeSet<Arc<Self>>) -> Self {
        let unioned_acc = calculate_unioned_acc_from_predecessors(&predecessors);
        let hash_key_cache = compute_internal_hash_key::<T, A>(&value, &predecessors);
        Self { value, acc: unioned_acc, predecessors, hash_key_cache }
    }
}

// Methods involving canonicalization
impl<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug> GSSNode<T, A> {
    /// Internal method to get/create a canonical Arc<GSSNode<T, A>>.
    /// Accumulator is derived from predecessors. If node exists in cache, its acc is unioned with this derived acc.
    fn get_canonical(
        value: T, // Consumed
        predecessors: BTreeSet<Arc<Self>>, // Consumed
        cache: &mut NodeCache<T, A>,
    ) -> Arc<Self> {
        let key_for_lookup = (value.clone(), predecessors.clone()); // Clones for lookup key
        let acc_from_predecessors = calculate_unioned_acc_from_predecessors(&predecessors);

        if let Some(entry_arc) = cache.get_mut(&key_for_lookup) { // entry_arc is &mut Arc<GSSNode<T,A>>
            let new_potential_acc = entry_arc.acc.union(&acc_from_predecessors);
            if new_potential_acc != entry_arc.acc {
                // Arc::make_mut updates entry_arc if GSSNode is cloned
                Arc::make_mut(entry_arc).acc = new_potential_acc;
            }
            return entry_arc.clone(); // Return a clone of the Arc from the cache.
        }

        // Not found, create new.
        let hash_key_cache = compute_internal_hash_key::<T, A>(&value, &predecessors);
        let new_node = GSSNode {
            value, // `value` moved here
            acc: acc_from_predecessors,
            predecessors, // `predecessors` moved here
            hash_key_cache,
        };
        let new_node_arc = Arc::new(new_node);
        cache.insert(key_for_lookup, new_node_arc.clone());
        new_node_arc
    }

    pub fn new_canonical(value: T, initial_acc: A, cache: &mut NodeCache<T, A>) -> Arc<Self> {
        let predecessors = BTreeSet::new();
        let key = (value.clone(), predecessors.clone()); // Key for cache

        if let Some(entry_arc) = cache.get_mut(&key) {
            let new_potential_acc = entry_arc.acc.union(&initial_acc);
            if new_potential_acc != entry_arc.acc {
                Arc::make_mut(entry_arc).acc = new_potential_acc;
            }
            return entry_arc.clone();
        }

        let hash_key_cache = compute_internal_hash_key::<T, A>(&value, &predecessors);
        let new_node_arc = Arc::new(GSSNode {
            value,
            acc: initial_acc,
            predecessors,
            hash_key_cache,
        });
        cache.insert(key, new_node_arc.clone());
        new_node_arc
    }

    pub fn new_with_predecessors_canonical(value: T, predecessors: BTreeSet<Arc<Self>>, cache: &mut NodeCache<T, A>) -> Arc<Self> {
        Self::get_canonical(value, predecessors, cache)
    }

    pub fn from_iter_canonical<I>(iter: I, cache: &mut NodeCache<T, A>) -> Arc<Self>
    where I: IntoIterator<Item = (T, A)>,
    {
        let mut iter_val = iter.into_iter();
        let (first_val, first_acc) = iter_val.next().expect("from_iter_canonical requires at least one element");
        let mut root = Self::new_canonical(first_val, first_acc, cache);
        for (value, _acc) in iter_val { // Acc from iter is ignored for subsequent nodes; acc is structural
            root = Self::push_onto_canonical(root, value, cache);
        }
        root
    }

    pub fn push_onto_canonical(current_stack_top: Arc<Self>, value: T, cache: &mut NodeCache<T, A>) -> Arc<Self> {
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
            .map(|pred_arc_t_a| GSSNode::<T, A>::map_canonical(pred_arc_t_a.clone(), f, cache_u))
            .collect();
        // get_canonical will derive acc from the new predecessors' accs.
        GSSNode::<U, A>::get_canonical(new_value, new_predecessors_mapped, cache_u)
    }
}

// Public methods (mostly non-canonical or general utilities)
impl<T, A: PathAccumulator> GSSNode<T, A> {
    pub fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = (T, A)>, T: Ord + Hash,
    {
        let mut iter_val = iter.into_iter();
        let (first_val, first_acc) = iter_val.next().expect("from_iter requires at least one element");
        let mut root = Self::new(first_val, first_acc);
        for (value, _acc) in iter_val {
            root = root.push(value);
        }
        root
    }

    pub fn push(self, value: T) -> Self where T: Ord + Hash {
        let mut predecessors = BTreeSet::new();
        predecessors.insert(Arc::new(self)); // Creates a new Arc, non-canonical
        Self::new_with_predecessors(value, predecessors)
    }

    pub fn pop(&self) -> Vec<Arc<Self>> {
        self.predecessors.iter().cloned().collect()
    }

    pub fn popn(&self, n: usize) -> Vec<Arc<Self>>
    where T: Clone + Hash,
    {
        if n == 0 { return vec![Arc::new(self.clone())]; } // New Arc, non-canonical

        let mut result = Vec::new();
        let mut seen_arcs_for_this_call: HashSet<*const GSSNode<T, A>> = HashSet::new();
        for predecessor_arc in &self.predecessors {
            for node_arc_from_popn in predecessor_arc.as_ref().popn(n - 1) {
                 if seen_arcs_for_this_call.insert(Arc::as_ptr(&node_arc_from_popn)) {
                    result.push(node_arc_from_popn);
                }
            }
        }
        result
    }

    pub fn peek(&self) -> &T { &self.value }
    pub fn acc(&self) -> &A { &self.acc }

    // Mutable accessors - caller beware of invalidating invariants (e.g., hash_key_cache)
    pub fn value_mut(&mut self) -> &mut T { &mut self.value }
    pub fn acc_mut(&mut self) -> &mut A { &mut self.acc }


    pub fn flatten(&self) -> Vec<Vec<(T, A)>> where T: Clone, A: Clone {
        let mut result = Vec::new();
        let mut q: VecDeque<(&GSSNode<T, A>, Vec<(T, A)>)> = VecDeque::new();
        q.push_back((self, Vec::new()));

        while let Some((current_node, mut current_path)) = q.pop_front() {
            current_path.push((current_node.value.clone(), current_node.acc.clone()));
            if current_node.predecessors.is_empty() {
                current_path.reverse(); // Paths are built root-to-leaf, reverse for leaf-to-root
                result.push(current_path);
            } else {
                for pred_arc in &current_node.predecessors {
                    q.push_back((pred_arc.as_ref(), current_path.clone()));
                }
            }
        }
        result
    }

    pub fn flatten_bulk(nodes: &[Self]) -> Vec<Vec<(T, A)>>
    where T: Clone + Hash, // Hash needed for GSSNode methods called by flatten
    {
        nodes.iter().flat_map(Self::flatten).collect()
    }

    pub fn merge(&mut self, other: Self) where T: Ord + Hash {
        assert!(self.value == other.value); // T: PartialEq implied by Ord
        self.merge_unchecked(other);
    }

    pub fn merge_unchecked(&mut self, mut other: Self) where T: Ord + Hash {
        self.acc = self.acc.union(&other.acc);
        self.predecessors.append(&mut other.predecessors);
        self.hash_key_cache = compute_internal_hash_key::<T, A>(&self.value, &self.predecessors);
    }

    pub fn map<F, U>(&self, f: F) -> GSSNode<U, A>
    where
        F: Copy + Fn(&T, &A) -> U,
        U: Ord + Hash, T: Hash,
    {
        let new_value = f(&self.value, &self.acc);
        let new_predecessors_arcs: BTreeSet<Arc<GSSNode<U, A>>> = self.predecessors.iter()
            .map(|pred_arc| Arc::new(pred_arc.as_ref().map(f))) // Non-canonical map
            .collect();
        GSSNode::<U, A>::new_with_predecessors(new_value, new_predecessors_arcs)
    }
}

impl<T, A: PathAccumulator> Drop for GSSNode<T, A> {
    fn drop(&mut self) {
        // Custom drop to iteratively break down the structure, potentially handling cycles/deep stacks.
        let mut worklist: Vec<Arc<GSSNode<T, A>>> = std::mem::take(&mut self.predecessors).into_iter().collect();
        while let Some(node_arc) = worklist.pop() {
            if Arc::strong_count(&node_arc) == 1 { // If we are the last owner
                if let Ok(mut inner_node) = Arc::try_unwrap(node_arc) {
                    worklist.extend(std::mem::take(&mut inner_node.predecessors).into_iter());
                    // inner_node (and its value, acc) drops here, its predecessors are empty.
                }
            }
        }
    }
}

impl<T: Hash, A: PathAccumulator> Hash for GSSNode<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) { self.hash_key_cache.hash(state); }
}

impl<T: Hash + PartialEq, A: PathAccumulator> PartialEq for GSSNode<T, A> {
    fn eq(&self, other: &Self) -> bool {
        if std::ptr::eq(self, other) { return true; }
        // Structural equality: hash, value, and predecessor Arcs (compared by pointer via BTreeSet order)
        // Accumulator `acc` is NOT part of structural equality for canonicalization purposes.
        self.hash_key_cache == other.hash_key_cache &&
        self.value == other.value &&
        self.predecessors == other.predecessors
    }
}
impl<T: Hash + Eq, A: PathAccumulator> Eq for GSSNode<T, A> {}

impl<T: Hash + PartialOrd, A: PathAccumulator> PartialOrd for GSSNode<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if std::ptr::eq(self, other) { return Some(Ordering::Equal); }
        match self.hash_key_cache.partial_cmp(&other.hash_key_cache) {
            Some(Ordering::Equal) => match self.value.partial_cmp(&other.value) {
                Some(Ordering::Equal) => self.predecessors.partial_cmp(&other.predecessors),
                other_ord => other_ord,
            },
            other_ord => other_ord,
        }
    }
}

impl<T: Hash + Ord, A: PathAccumulator> Ord for GSSNode<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        if std::ptr::eq(self, other) { return Ordering::Equal; }
        self.hash_key_cache.cmp(&other.hash_key_cache)
            .then_with(|| self.value.cmp(&other.value))
            .then_with(|| self.predecessors.cmp(&other.predecessors))
    }
}


pub trait GSSTrait<T: Clone + Hash, A: PathAccumulator> {
    type Peek<'a> where T: 'a, Self: 'a;
    fn peek(&self) -> Self::Peek<'_>;
    fn push(&self, value: T) -> GSSNode<T, A> where T: Ord;
    fn pop(&self) -> Vec<Arc<GSSNode<T, A>>>;
    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T, A>>>;
}

impl<T: Clone + Hash, A: PathAccumulator> GSSTrait<T, A> for GSSNode<T, A> {
    type Peek<'a> = &'a T where T: 'a, A: 'a;
    fn peek(&self) -> Self::Peek<'_> { GSSNode::peek(self) }
    fn push(&self, value: T) -> GSSNode<T, A> where T: Ord { GSSNode::push(self.clone(), value) }
    fn pop(&self) -> Vec<Arc<GSSNode<T, A>>> { GSSNode::pop(self) }
    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T, A>>> { GSSNode::popn(self, n) }
}

impl<T: Clone + Hash, A: PathAccumulator> GSSTrait<T, A> for Arc<GSSNode<T, A>> {
    type Peek<'a> = &'a T where T: 'a, A: 'a;
    fn peek(&self) -> Self::Peek<'_> { self.as_ref().peek() }
    fn pop(&self) -> Vec<Arc<GSSNode<T, A>>> { self.as_ref().pop() }
    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T, A>>> { self.as_ref().popn(n) }
    fn push(&self, value: T) -> GSSNode<T, A> where T: Ord {
        let mut predecessors = BTreeSet::new();
        predecessors.insert(self.clone()); // Clone the Arc for the new node's predecessors
        GSSNode::new_with_predecessors(value, predecessors) // acc derived from self.acc
    }
}

macro_rules! impl_gss_trait_for_option_wrapper {
    ($wrapper_type:ty) => {
        impl<T: Clone + Hash, A: PathAccumulator> GSSTrait<T, A> for Option<$wrapper_type> {
            type Peek<'a> = Option<&'a T> where T: 'a, A: 'a, $wrapper_type: 'a;

            fn peek(&self) -> Self::Peek<'_> {
                self.as_ref().map(|inner_val| GSSTrait::peek(inner_val))
            }

            fn push(&self, value: T) -> GSSNode<T, A> where T: Ord, A: Default {
                match self.as_ref() {
                    Some(inner_val) => GSSTrait::push(inner_val, value),
                    None => GSSNode::new(value, A::default()),
                }
            }

            fn pop(&self) -> Vec<Arc<GSSNode<T, A>>> {
                self.as_ref().map_or_else(Vec::new, |inner_val| GSSTrait::pop(inner_val))
            }

            fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T, A>>> {
                self.as_ref().map_or_else(Vec::new, |inner_val| GSSTrait::popn(inner_val, n))
            }
        }
    };
}
impl_gss_trait_for_option_wrapper!(Arc<GSSNode<T, A>>);
impl_gss_trait_for_option_wrapper!(GSSNode<T, A>);


pub trait BulkMerge<T, A: PathAccumulator> { fn bulk_merge(&mut self); }

impl<T: Clone + Ord + Hash, A: PathAccumulator> BulkMerge<T, A> for Vec<Arc<GSSNode<T, A>>> {
    fn bulk_merge(&mut self) {
        if self.len() <= 1 { return; } // Optimization
        let mut groups: BTreeMap<T, Vec<Arc<GSSNode<T, A>>>> = BTreeMap::new();
        for node_arc in self.drain(..) { // drain to take ownership
            groups.entry(node_arc.value.clone()).or_default().push(node_arc);
        }

        let mut new_merged_nodes = Vec::with_capacity(groups.len());
        for (value, group_arcs) in groups {
            // group_arcs is guaranteed not empty by BTreeMap construction if value is a key
            if group_arcs.len() == 1 {
                new_merged_nodes.push(group_arcs.into_iter().next().unwrap());
                continue;
            }

            let union_of_accs_in_group = group_arcs.iter()
                .map(|arc_node| arc_node.acc.clone())
                .reduce(|acc, item| acc.union(&item))
                .expect("group_arcs is not empty, so reduce will yield Some");

            let merged_predecessors: BTreeSet<Arc<GSSNode<T, A>>> = group_arcs.iter()
                .flat_map(|node_arc| node_arc.predecessors.iter().cloned())
                .collect();

            let mut merged_node = GSSNode::new_with_predecessors(value, merged_predecessors);
            merged_node.acc = union_of_accs_in_group; // Override acc with the group's unioned acc
            new_merged_nodes.push(Arc::new(merged_node));
        }
        *self = new_merged_nodes;
    }
}

pub fn bulk_merge_canonical<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    nodes: &mut Vec<Arc<GSSNode<T, A>>>,
    cache: &mut NodeCache<T, A>
) {
    if nodes.len() <= 1 { return; } // Optimization
    let mut groups: BTreeMap<T, Vec<Arc<GSSNode<T, A>>>> = BTreeMap::new();
    for node_arc in nodes.drain(..) {
        groups.entry(node_arc.value.clone()).or_default().push(node_arc);
    }

    let mut new_merged_nodes = Vec::with_capacity(groups.len());
    for (value, group_arcs) in groups {
        // group_arcs is guaranteed not empty
        let union_of_accs_in_group = group_arcs.iter()
            .map(|arc_node| arc_node.acc.clone())
            .reduce(|acc, item| acc.union(&item))
            .expect("group_arcs is not empty, so reduce will yield Some");

        let merged_predecessors: BTreeSet<Arc<GSSNode<T, A>>> = group_arcs.iter()
            .flat_map(|node_arc| node_arc.predecessors.iter().cloned())
            .collect();

        // Key for cache operations. Must clone value and predecessors for the key.
        let cache_key = (value.clone(), merged_predecessors.clone());

        // Step 1: Ensure structural node is in cache; its acc reflects its predecessors.
        // GSSNode::get_canonical uses the cloned value & merged_predecessors from cache_key.
        let _ = GSSNode::get_canonical(cache_key.0.clone(), cache_key.1.clone(), cache);

        // Step 2: Union the specific accumulator from the merged group into the cached node.
        let cached_arc_ref_mut = cache.get_mut(&cache_key)
            .expect("Node must be in cache after get_canonical call");

        let final_acc = cached_arc_ref_mut.acc.union(&union_of_accs_in_group);
        if final_acc != cached_arc_ref_mut.acc {
            Arc::make_mut(cached_arc_ref_mut).acc = final_acc;
        }
        new_merged_nodes.push(cached_arc_ref_mut.clone());
    }
    *nodes = new_merged_nodes;
}

pub fn prune_and_transform_recursive_canonical<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    node_arc: &Arc<GSSNode<T, A>>,
    closure: &impl Fn(&T, &A) -> Option<(T, A, bool)>, // Returns (NewVal, NewAcc, ContinueRecursion)
    memo: &mut HashMap<*const GSSNode<T, A>, Option<Arc<GSSNode<T, A>>>>,
    cache: &mut NodeCache<T, A>,
) -> Option<Arc<GSSNode<T, A>>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }

    match closure(&node_arc.value, &node_arc.acc) {
        None => { // Prune this node
            memo.insert(node_ptr, None);
            None
        }
        Some((new_value, new_acc_from_closure, continue_recursion)) => {
            let transformed_predecessors: BTreeSet<Arc<GSSNode<T, A>>> = if continue_recursion {
                node_arc.predecessors.iter()
                    .filter_map(|pred_arc| prune_and_transform_recursive_canonical(pred_arc, closure, memo, cache))
                    .collect()
            } else {
                // Stop recursion, keep original predecessors (assumed to be/become canonical via this process if roots are simplified)
                node_arc.predecessors.clone()
            };

            // Key for cache operations. Must clone new_value and transformed_predecessors.
            let cache_key = (new_value.clone(), transformed_predecessors.clone());

            // Step 1: Ensure structural node is in cache; its acc reflects its new predecessors.
            let _ = GSSNode::get_canonical(cache_key.0.clone(), cache_key.1.clone(), cache);

            // Step 2: Set the accumulator of the cached node to new_acc_from_closure.
            // This overrides any accumulator derived from predecessors by get_canonical.
            let cached_arc_ref_mut = cache.get_mut(&cache_key)
                .expect("Node must be in cache after get_canonical for prune_and_transform");

            if new_acc_from_closure != cached_arc_ref_mut.acc {
                 Arc::make_mut(cached_arc_ref_mut).acc = new_acc_from_closure; // new_acc_from_closure is cloned
            }

            let result_arc = cached_arc_ref_mut.clone();
            memo.insert(node_ptr, Some(result_arc.clone()));
            Some(result_arc)
        }
    }
}

pub fn prune_and_transform_roots_canonical<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T, A>>],
    closure: &impl Fn(&T, &A) -> Option<(T, A, bool)>,
    cache: &mut NodeCache<T, A>,
) -> Vec<Option<Arc<GSSNode<T, A>>>> {
    let mut memo = HashMap::new(); // Memoization for the current transformation pass
    roots.iter().map(|root| prune_and_transform_recursive_canonical(root, closure, &mut memo, cache)).collect()
}

// Non-canonical versions (create a new cache for each top-level call)
pub fn prune_and_transform_recursive<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    node_arc: &Arc<GSSNode<T, A>>,
    closure: &impl Fn(&T, &A) -> Option<(T, A, bool)>,
    memo: &mut HashMap<*const GSSNode<T, A>, Option<Arc<GSSNode<T, A>>>>,
) -> Option<Arc<GSSNode<T, A>>> {
    prune_and_transform_recursive_canonical(node_arc, closure, memo, &mut NodeCache::new())
}

pub fn prune_and_transform_roots<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T, A>>],
    closure: &impl Fn(&T, &A) -> Option<(T, A, bool)>,
) -> Vec<Option<Arc<GSSNode<T, A>>>> {
    prune_and_transform_roots_canonical(roots, closure, &mut NodeCache::new())
}


pub fn pop_and_apply_contextual_accumulator<T: Clone + Ord + Hash, A: PathAccumulator>(
    source_nodes: &[Arc<GSSNode<T, A>>],
) -> Vec<Arc<GSSNode<T, A>>> {
    let mut resultMap: HashMap<*const GSSNode<T, A>, (Arc<GSSNode<T, A>>, A)> = HashMap::new();

    for src_node_arc in source_nodes {
        for pred_arc in &src_node_arc.predecessors {
            let pred_ptr = Arc::as_ptr(pred_arc);
            let acc_from_source = src_node_arc.acc.clone(); // Context from the "popping" source
            resultMap.entry(pred_ptr)
                .and_modify(|e| e.1 = e.1.union(&acc_from_source)) // Union contexts if multiple paths pop to same pred
                .or_insert_with(|| (pred_arc.clone(), acc_from_source));
        }
    }

    resultMap.into_values().map(|(mut original_pred_arc, pop_context_a)| {
        // Arc::make_mut may clone GSSNode if original_pred_arc is shared.
        // The modified (or new) GSSNode's acc is updated.
        let node_instance_mut = Arc::make_mut(&mut original_pred_arc);
        node_instance_mut.acc = node_instance_mut.acc.intersect(&pop_context_a);
        original_pred_arc // Return the Arc, now pointing to the modified GSSNode
    }).collect()
}


// Read-only functions: find_longest_path, gather_gss_stats, print_gss_forest

fn find_longest_path_recursive<T, A: PathAccumulator>(
    node_arc: &Arc<GSSNode<T, A>>,
    memo: &mut HashMap<*const GSSNode<T, A>, Vec<Arc<GSSNode<T, A>>>>,
    visited_in_current_path: &mut HashSet<*const GSSNode<T, A>>, // For cycle detection in current DFS path
) -> Vec<Arc<GSSNode<T, A>>> {
    let node_ptr = Arc::as_ptr(node_arc);

    if let Some(cached_path) = memo.get(&node_ptr) { return cached_path.clone(); }
    if !visited_in_current_path.insert(node_ptr) { return Vec::new(); } // Cycle detected in current path

    let mut longest_path_from_preds = node_arc.predecessors.iter()
        .map(|pred_arc| find_longest_path_recursive(pred_arc, memo, visited_in_current_path))
        .max_by_key(|path| path.len())
        .unwrap_or_default(); // If no predecessors or all lead to cycles/empty paths

    longest_path_from_preds.push(node_arc.clone()); // Append current node to the path from predecessors

    memo.insert(node_ptr, longest_path_from_preds.clone());
    visited_in_current_path.remove(&node_ptr); // Backtrack: remove from current path visited set
    longest_path_from_preds
}

pub fn find_longest_path<T, A: PathAccumulator>(roots: &[Arc<GSSNode<T, A>>]) -> Option<Vec<Arc<GSSNode<T, A>>>> {
    let mut memo = HashMap::new(); // Memoizes results for all nodes across all root traversals
    roots.iter()
        .map(|root_arc| find_longest_path_recursive(root_arc, &mut memo, &mut HashSet::new()))
        .max_by_key(|path| path.len())
        .filter(|path| !path.is_empty()) // Return None if all paths are empty (e.g., empty roots or all cyclic)
}

#[derive(Debug, Clone, Default)]
pub struct GSSStats {
    pub num_roots: usize, pub unique_nodes: usize, pub max_depth: usize, pub average_depth: f64,
    pub merge_points: usize, pub max_predecessors: usize, pub average_predecessors: f64,
}

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
            JSONNode::Object(mut obj) => Ok(GSSStats {
                num_roots: usize::from_json(obj.remove("num_roots").ok_or("Missing field num_roots")?)?,
                unique_nodes: usize::from_json(obj.remove("unique_nodes").ok_or("Missing field unique_nodes")?)?,
                max_depth: usize::from_json(obj.remove("max_depth").ok_or("Missing field max_depth")?)?,
                average_depth: f64::from_json(obj.remove("average_depth").ok_or("Missing field average_depth")?)?,
                merge_points: usize::from_json(obj.remove("merge_points").ok_or("Missing field merge_points")?)?,
                max_predecessors: usize::from_json(obj.remove("max_predecessors").ok_or("Missing field max_predecessors")?)?,
                average_predecessors: f64::from_json(obj.remove("average_predecessors").ok_or("Missing field average_predecessors")?)?,
            }),
            _ => Err("Expected JSONNode::Object for GSSStats".to_string()),
        }
    }
}

pub fn gather_gss_stats<T, A: PathAccumulator>(roots: &[Arc<GSSNode<T, A>>]) -> GSSStats {
    let mut stats = GSSStats { num_roots: roots.len(), ..Default::default() };
    let mut visited_nodes: HashSet<*const GSSNode<T, A>> = HashSet::new(); // Tracks all unique nodes visited
    let mut queue: VecDeque<(Arc<GSSNode<T, A>>, usize)> = VecDeque::new(); // (Node, Depth) for BFS
    let mut total_depth_sum: u64 = 0;
    let mut total_predecessors_sum: u64 = 0;

    for root_arc in roots {
        if visited_nodes.insert(Arc::as_ptr(root_arc)) {
            queue.push_back((root_arc.clone(), 0)); // Depth of root is 0
        }
    }

    while let Some((current_node_arc, depth)) = queue.pop_front() {
        stats.unique_nodes += 1;
        stats.max_depth = stats.max_depth.max(depth);
        total_depth_sum += depth as u64;

        let num_preds = current_node_arc.predecessors.len();
        stats.max_predecessors = stats.max_predecessors.max(num_preds);
        total_predecessors_sum += num_preds as u64;
        if num_preds > 1 { stats.merge_points += 1; } // Node is a merge point if it has >1 predecessors

        for pred_arc in &current_node_arc.predecessors {
            if visited_nodes.insert(Arc::as_ptr(pred_arc)) {
                queue.push_back((pred_arc.clone(), depth + 1)); // Predecessors are at next depth level
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
    node_arc: &Arc<GSSNode<T, A>>, visited: &mut HashSet<*const GSSNode<T, A>>, indent_level: usize,
    nodes_printed_count: &mut usize, max_nodes_to_print: usize, output_string: &mut String,
) -> std::fmt::Result {
    if *nodes_printed_count >= max_nodes_to_print { return Ok(()); }

    let indent_str = "  ".repeat(indent_level);
    let node_ptr = Arc::as_ptr(node_arc);

    if !visited.insert(node_ptr) { // If already visited (and printed), mark as (Visited)
        return writeln!(output_string, "{}- Node {:p} (Visited)", indent_str, node_ptr);
    }

    *nodes_printed_count += 1;
    writeln!(output_string, "{}- Node {:p}: {:?} (Acc: {:?})", indent_str, node_ptr, node_arc.value, node_arc.acc)?;

    if !node_arc.predecessors.is_empty() {
        writeln!(output_string, "{}  Predecessors:", indent_str)?;
        for pred_arc in &node_arc.predecessors {
            print_gss_node_recursive(pred_arc, visited, indent_level + 1, nodes_printed_count, max_nodes_to_print, output_string)?;
            if *nodes_printed_count >= max_nodes_to_print { return Ok(()); } // Check after each recursive call
        }
    }
    Ok(())
}

pub fn print_gss_forest<T: Debug, A: PathAccumulator>(roots: &[Arc<GSSNode<T, A>>], max_nodes_to_print: usize) -> String {
    let mut output_string = String::new();
    if roots.is_empty() { return "GSS Forest: (No roots)".to_string(); }

    writeln!(&mut output_string, "GSS Forest Roots (Max Nodes: {}):", max_nodes_to_print).unwrap();
    let mut visited_nodes_for_printing = HashSet::new(); // Tracks nodes printed in this call
    let mut nodes_printed_count = 0;

    for (i, root_arc) in roots.iter().enumerate() {
        if nodes_printed_count >= max_nodes_to_print {
             writeln!(&mut output_string, "... (Truncated: Reached max nodes to print before processing all roots)").unwrap();
             break;
        }
        writeln!(&mut output_string, "Root {}:", i).unwrap();
        if print_gss_node_recursive(root_arc, &mut visited_nodes_for_printing, 1, &mut nodes_printed_count, max_nodes_to_print, &mut output_string).is_err() {
            // Simplified error handling for the example; consider more robust error propagation if needed.
            return format!("Error generating GSS string representation for root {}", i);
        }
    }
    if nodes_printed_count >= max_nodes_to_print && visited_nodes_for_printing.len() == nodes_printed_count {
        // This condition might be true if the last node printed hit the max_nodes limit.
        // A more general truncation message if limit is hit mid-recursion is handled inside print_gss_node_recursive.
        // This final message can indicate if not all roots were even started.
    }
    output_string
}


// Simplification functions (canonicalize an existing GSS)
fn simplify_node_recursive<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    original_node_arc: &Arc<GSSNode<T, A>>,
    original_ptr_memo: &mut HashMap<*const GSSNode<T, A>, Arc<GSSNode<T, A>>>, // Memoizes original_ptr -> canonical_Arc
    canonicalization_cache: &mut NodeCache<T, A>, // Global cache for (value, preds_arcs) -> canonical_Arc
) -> Arc<GSSNode<T, A>> {
    let original_node_ptr = Arc::as_ptr(original_node_arc);
    if let Some(canonical_arc) = original_ptr_memo.get(&original_node_ptr) {
        return canonical_arc.clone();
    }

    // Predecessor Knitting: Merge direct predecessors with the same value before recursive simplification.
    // This is a specific semantic choice for how to simplify.
    let mut predecessor_knitting_map: BTreeMap<T, Arc<GSSNode<T, A>>> = BTreeMap::new();
    for pred_arc in &original_node_arc.predecessors {
        let pred_value = pred_arc.value.clone();
        match predecessor_knitting_map.entry(pred_value) {
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                // Arc::make_mut gets a mutable ref to GSSNode, cloning if necessary.
                // The Arc in the map (`entry.get_mut()`) is updated if GSSNode is cloned.
                Arc::make_mut(entry.get_mut()).merge(pred_arc.as_ref().clone());
            }
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(pred_arc.clone()); // Clone Arc for the map
            }
        }
    }
    // Recursively simplify these (potentially merged) knitted predecessors.
    let canonical_predecessor_arcs: BTreeSet<Arc<GSSNode<T, A>>> = predecessor_knitting_map.into_values()
        .map(|knitted_pred_arc| simplify_node_recursive(&knitted_pred_arc, original_ptr_memo, canonicalization_cache))
        .collect();

    // Key for cache operations. Must clone value and canonical_predecessor_arcs.
    let cache_key = (original_node_arc.value.clone(), canonical_predecessor_arcs.clone());

    // Step 1: Ensure structural node is in cache; its acc reflects its canonical predecessors.
    let _ = GSSNode::get_canonical(cache_key.0.clone(), cache_key.1.clone(), canonicalization_cache);

    // Step 2: Union the original node's accumulator into the cached node.
    let cached_arc_ref_mut = canonicalization_cache.get_mut(&cache_key)
        .expect("Node must be in cache after get_canonical call for simplify_node_recursive");

    let final_acc = cached_arc_ref_mut.acc.union(&original_node_arc.acc);
    if final_acc != cached_arc_ref_mut.acc {
        Arc::make_mut(cached_arc_ref_mut).acc = final_acc;
    }

    let result_arc = cached_arc_ref_mut.clone();
    original_ptr_memo.insert(original_node_ptr, result_arc.clone()); // Memoize result for original_node_ptr
    result_arc
}

pub fn simplify_gss_forest<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T, A>>],
) -> Vec<Arc<GSSNode<T, A>>> {
    let mut original_ptr_memo = HashMap::new(); // Memoization for this simplification run
    let mut canonicalization_cache_for_this_run = NodeCache::new(); // Fresh cache for this run
    roots.iter()
        .map(|root_arc| simplify_node_recursive(root_arc, &mut original_ptr_memo, &mut canonicalization_cache_for_this_run))
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct MockPathAccumulator {
        active: BTreeSet<usize>,
        intersection: BTreeSet<usize>, // Example field
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
            Self { active: BTreeSet::new(), intersection: BTreeSet::new() } // Simplified default
        }
    }

    impl PathAccumulator for MockPathAccumulator {
        fn union(&self, other: &Self) -> Self {
            Self {
                active: self.active.union(&other.active).cloned().collect(),
                intersection: self.intersection.intersection(&other.intersection).cloned().collect(), // Example logic
            }
        }

        fn intersect(&self, other: &Self) -> Self {
            // Example: intersection of active, union of 'intersection' field (depends on semantics)
            Self {
                active: self.active.intersection(&other.active).cloned().collect(),
                intersection: self.intersection.union(&other.intersection).cloned().collect(),
            }
        }
    }

    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct MockParseStateNodeContent {
        state_id: usize,
    }

    impl Debug for MockParseStateNodeContent {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!("StateID({})", self.state_id))
        }
    }

    type MockGSSNode = GSSNode<MockParseStateNodeContent, MockPathAccumulator>;
    type MockNodeCache = NodeCache<MockParseStateNodeContent, MockPathAccumulator>;
    type IntGSSNode = GSSNode<i32, MockPathAccumulator>;
    type IntNodeCache = NodeCache<i32, MockPathAccumulator>;

    // Helper to create a non-canonical GSSNode Arc for tests
    fn nc_node_arc_int(value: i32, _acc_override: Option<MockPathAccumulator>, predecessors: Vec<Arc<IntGSSNode>>) -> Arc<IntGSSNode> {
        let pred_set: BTreeSet<Arc<IntGSSNode>> = predecessors.into_iter().collect();
        let mut node = IntGSSNode::new_with_predecessors(value, pred_set);
        if let Some(acc) = _acc_override { node.acc = acc; } // Override if needed, new_with_predecessors sets from preds
        Arc::new(node)
    }
    fn nc_node_arc_int_root(value: i32, acc: MockPathAccumulator) -> Arc<IntGSSNode> {
         Arc::new(IntGSSNode::new(value, acc))
    }

    // Helper to collect all unique Arcs in a GSS structure starting from a root
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
        // let acc_merged = acc_base.union(&acc_other);

        // Create a non-canonical graph first
        let d1_orig_nc = nc_node_arc_int_root(40, acc_base.clone());
        let c1_orig_nc = nc_node_arc_int(30, None, vec![d1_orig_nc.clone()]);
        let b1_orig_nc = nc_node_arc_int(20, None, vec![c1_orig_nc.clone()]);

        let d2_orig_nc = nc_node_arc_int_root(40, acc_other.clone());
        assert_ne!(Arc::as_ptr(&d1_orig_nc), Arc::as_ptr(&d2_orig_nc));

        let a1_orig_nc = nc_node_arc_int(10, None, vec![b1_orig_nc.clone(), d2_orig_nc.clone()]);

        let roots_nc = vec![a1_orig_nc.clone()];
        let simplified_roots = simplify_gss_forest(&roots_nc);
        assert_eq!(simplified_roots.len(), 1);
        let simplified_a1 = simplified_roots[0].clone();

        let mut collected_arcs = HashMap::new();
        collect_arcs_recursive(&simplified_a1, &mut collected_arcs);
        // Expected unique nodes: D_canonical (value 40), C1 (value 30), B1 (value 20), A1 (value 10) = 4
        assert_eq!(collected_arcs.len(), 4, "Expected 4 unique Arcs after simplification");

        let s_a1 = simplified_a1;
        let s_b1_opt = s_a1.predecessors.iter().find(|n| n.value == 20);
        assert!(s_b1_opt.is_some(), "B1 node missing");
        let s_b1 = s_b1_opt.unwrap().clone();

        let s_d_from_a1_opt = s_a1.predecessors.iter().find(|n| n.value == 40);
        assert!(s_d_from_a1_opt.is_some(), "D node from A1 missing");
        let s_d_from_a1 = s_d_from_a1_opt.unwrap().clone();

        let s_c1_opt = s_b1.predecessors.iter().find(|n| n.value == 30);
        assert!(s_c1_opt.is_some(), "C1 node missing");
        let s_c1 = s_c1_opt.unwrap().clone();

        let s_d_from_c1_opt = s_c1.predecessors.iter().find(|n| n.value == 40);
        assert!(s_d_from_c1_opt.is_some(), "D node from C1 missing");
        let s_d_from_c1 = s_d_from_c1_opt.unwrap().clone();

        assert!(Arc::ptr_eq(&s_d_from_a1, &s_d_from_c1), "D nodes should be canonicalized to the same Arc");
        let s_d_canonical = s_d_from_a1;

        // Check structure
        assert_eq!(s_d_canonical.predecessors.len(), 0);
        assert_eq!(s_c1.predecessors.len(), 1);
        assert!(Arc::ptr_eq(s_c1.predecessors.iter().next().unwrap(), &s_d_canonical));
        assert_eq!(s_b1.predecessors.len(), 1);
        assert!(Arc::ptr_eq(s_b1.predecessors.iter().next().unwrap(), &s_c1));
        assert_eq!(s_a1.predecessors.len(), 2); // B1 and D_canonical
        assert!(s_a1.predecessors.contains(&s_b1));
        assert!(s_a1.predecessors.contains(&s_d_canonical));

        // Check accumulators after simplification
        // s_d_canonical.acc should be d1_orig_nc.acc.union(d2_orig_nc.acc) due to simplify_node_recursive's unioning logic.
        assert_eq!(s_d_canonical.acc, acc_base.union(&acc_other));

        // s_c1.acc should be derived from s_d_canonical.acc (from get_canonical) AND unioned with c1_orig_nc.acc.
        // c1_orig_nc.acc was from d1_orig_nc.acc (which is acc_base).
        // So, s_c1.acc = s_d_canonical.acc.union(acc_base) = (acc_base.union(acc_other)).union(acc_base) = acc_base.union(acc_other)
        assert_eq!(s_c1.acc, s_d_canonical.acc.union(&d1_orig_nc.acc)); // More precisely, c1_orig_nc.acc
                                                                        // c1_orig_nc.acc is d1_orig_nc.acc
        assert_eq!(s_c1.acc, acc_base.union(&acc_other));


        // s_b1.acc = (acc from s_c1 in get_canonical).union(b1_orig_nc.acc)
        // b1_orig_nc.acc was from c1_orig_nc.acc (which is acc_base)
        assert_eq!(s_b1.acc, s_c1.acc.union(&c1_orig_nc.acc));
        assert_eq!(s_b1.acc, acc_base.union(&acc_other));


        // s_a1.acc = (acc from {s_b1, s_d_canonical} in get_canonical).union(a1_orig_nc.acc)
        // a1_orig_nc.acc was from b1_orig_nc.acc.union(d2_orig_nc.acc) = acc_base.union(acc_other)
        let acc_from_preds_for_a1 = s_b1.acc.union(&s_d_canonical.acc);
        assert_eq!(s_a1.acc, acc_from_preds_for_a1.union(&a1_orig_nc.acc));
        assert_eq!(s_a1.acc, acc_base.union(&acc_other));
    }

    #[test]
    fn test_pop_and_apply_contextual_accumulator_basic() {
        let acc0 = MockPathAccumulator { active: BTreeSet::from([0]), intersection: BTreeSet::new() };
        let acc1 = MockPathAccumulator { active: BTreeSet::from([1]), intersection: BTreeSet::new() };
        // let acc2 = MockPathAccumulator { active: BTreeSet::from([2]), intersection: BTreeSet::new() };

        // Original nodes (non-canonical for test setup simplicity)
        let node_a_orig = nc_node_arc_int_root(0, acc0.clone()); // A(0, acc0)
        let node_b_orig = nc_node_arc_int(1, Some(acc0.clone()), vec![node_a_orig.clone()]); // B(1, acc0) -> A

        let node_c_orig = nc_node_arc_int_root(0, acc1.clone()); // C(0, acc1)
        let node_d_orig = nc_node_arc_int(1, Some(acc1.clone()), vec![node_c_orig.clone()]); // D(1, acc1) -> C

        // Nodes from which we pop
        let node_e = nc_node_arc_int(2, Some(acc0.clone()), vec![node_b_orig.clone()]); // E(2, acc0) -> B
        let node_f = nc_node_arc_int(2, Some(acc1.clone()), vec![node_d_orig.clone()]); // F(2, acc1) -> D

        let source_nodes = vec![node_e.clone(), node_f.clone()];
        let popped_nodes = pop_and_apply_contextual_accumulator(&source_nodes);

        assert_eq!(popped_nodes.len(), 2);

        // Find B and D in popped_nodes. They might be new Arcs if make_mut cloned.
        // We check by structure and original accumulator before intersection.
        let popped_b = popped_nodes.iter().find(|n|
            n.value == 1 &&
            n.predecessors.iter().any(|p| Arc::ptr_eq(p, &node_a_orig))
        ).expect("Popped B not found");

        let popped_d = popped_nodes.iter().find(|n|
            n.value == 1 &&
            n.predecessors.iter().any(|p| Arc::ptr_eq(p, &node_c_orig))
        ).expect("Popped D not found");

        // B's original acc was acc0. Context from E was acc0. Intersection is acc0.
        assert_eq!(popped_b.acc, acc0.intersect(&acc0));
        // D's original acc was acc1. Context from F was acc1. Intersection is acc1.
        assert_eq!(popped_d.acc, acc1.intersect(&acc1));

        // Test with merging context
        // G(2, acc0 U acc1) -> B, D
        let acc_g = acc0.union(&acc1);
        let node_g = nc_node_arc_int(2, Some(acc_g.clone()), vec![node_b_orig.clone(), node_d_orig.clone()]);

        let source_nodes_g = vec![node_g.clone()];
        let popped_nodes_g = pop_and_apply_contextual_accumulator(&source_nodes_g);
        assert_eq!(popped_nodes_g.len(), 2);

        let popped_b_from_g = popped_nodes_g.iter().find(|n|
            n.value == 1 &&
            n.predecessors.iter().any(|p| Arc::ptr_eq(p, &node_a_orig))
        ).expect("Popped B from G not found");

        let popped_d_from_g = popped_nodes_g.iter().find(|n|
            n.value == 1 &&
            n.predecessors.iter().any(|p| Arc::ptr_eq(p, &node_c_orig))
        ).expect("Popped D from G not found");

        // B's original acc was acc0. Context from G was acc_g (acc0 U acc1).
        // Intersection: acc0.intersect(acc0 U acc1)
        assert_eq!(popped_b_from_g.acc, acc0.intersect(&acc_g));
        // D's original acc was acc1. Context from G was acc_g (acc0 U acc1).
        // Intersection: acc1.intersect(acc0 U acc1)
        assert_eq!(popped_d_from_g.acc, acc1.intersect(&acc_g));
    }
    // Add more tests as needed, especially for canonical operations and edge cases.
}