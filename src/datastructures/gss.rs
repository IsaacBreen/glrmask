use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use deterministic_hash::DeterministicHasher;

pub trait PathAccumulator: Sized + Clone + Debug + Eq + PartialEq + Ord + PartialOrd + Hash + Default {
    fn union(&self, other: &Self) -> Self;
    fn pop(&self, right: &Self) -> Self;
}

impl PathAccumulator for () {
    fn union(&self, _other: &Self) -> Self { () }
    fn pop(&self, _right: &Self) -> Self { () }
}

// Helper function to compute a node's hash_key_cache.
// This is used by both canonical and non-canonical node creation.
// T is the type of the value on the edge.
fn compute_internal_hash_key<T: Hash, A: PathAccumulator>(
    predecessors_with_values: &BTreeSet<(Arc<GSSNode<T, A>>, T)>
) -> u64 {
    // TODO: delete this
    // return 0;
    let mut hasher = DeterministicHasher::new(DefaultHasher::new());
    // The BTreeSet ensures predecessors_with_values are iterated in a canonical order.
    // Order depends on Arc pointer addresses and T values.
    for (pred_arc, edge_val) in predecessors_with_values {
        edge_val.hash(&mut hasher);
        pred_arc.hash_key_cache.hash(&mut hasher); // Hash predecessor's hash
    }
    hasher.finish()
}

#[derive(Debug, Clone)]
pub struct GSSNode<T, A: PathAccumulator> {
    // T is the type of value on edges leading to this node.
    // Nodes themselves do not store a singular T value.
    acc: A, // Accumulator value
    predecessors_with_values: BTreeSet<(Arc<GSSNode<T, A>>, T)>,
    hash_key_cache: u64, // Based on predecessors_with_values' (arcs and T values) hashes only
}

/// An iterator over all paths leading to a GSS node.
/// Paths are represented as `Vec<T>`, where `T` is the edge value type.
/// The iteration proceeds from the target node upwards to its roots.
#[derive(Clone)]
pub struct PathsIter<'a, T: Clone, A: PathAccumulator> {
    // The queue stores tuples of (node_to_visit, path_suffix_from_original_node_to_current_node).
    // The path_suffix is built in reverse order during traversal and corrected when a full path is yielded.
    // We use &'a GSSNode to avoid Arc cloning if not necessary for the iterator's logic itself,
    // and to tie the iterator's lifetime to the GSSNode it's iterating over.
    queue: VecDeque<(&'a GSSNode<T, A>, Vec<T>)>,
}

impl<'a, T: Clone, A: PathAccumulator> Iterator for PathsIter<'a, T, A> {
    type Item = Vec<T>; // Each path is a Vec of edge values.

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((current_node_ref, mut path_suffix_reversed)) = self.queue.pop_front() {
            if current_node_ref.predecessors_with_values.is_empty() {
                // This node is a root for the current path.
                // The path is complete. Reverse it to get the correct order.
                path_suffix_reversed.reverse();
                return Some(path_suffix_reversed);
            } else {
                // This node has predecessors. Add them to the queue for further exploration.
                for (pred_arc, edge_val) in &current_node_ref.predecessors_with_values {
                    let mut new_path_suffix_reversed = path_suffix_reversed.clone();
                    // The edge_val leads from pred_arc to current_node_ref.
                    // So, it's the next segment when tracing the path "upwards".
                    new_path_suffix_reversed.push(edge_val.clone());
                    // Add the predecessor node and the extended path suffix to the queue.
                    // pred_arc.as_ref() gets a &GSSNode<T, A> from Arc<GSSNode<T, A>>.
                    self.queue.push_back((pred_arc.as_ref(), new_path_suffix_reversed));
                }
            }
        }
        None // The queue is empty, so no more paths can be found.
    }
}


// Methods for creating non-canonical GSSNode instances
impl<T: Ord + Hash, A: PathAccumulator> GSSNode<T, A> { // T needs Ord for BTreeSet key part, Hash for hash_key_cache
    /// Creates a new root GSSNode (no predecessors).
    pub fn new(acc: A) -> Self {
        let predecessors_with_values = BTreeSet::new();
        let hash_key_cache = compute_internal_hash_key::<T, A>(&predecessors_with_values);
        Self {
            acc,
            predecessors_with_values,
            hash_key_cache,
        }
    }

    pub fn new_default() -> Self {
        Self::new(A::default())
    }

    /// Creates a new GSSNode with specified predecessors and edge values.
    /// The accumulator `acc` is derived from the union of predecessor accumulators.
    pub fn new_with_predecessors(predecessors_with_values: BTreeSet<(Arc<Self>, T)>) -> Self {
        let unioned_acc = if predecessors_with_values.is_empty() {
            A::default()
        } else {
            let mut iter = predecessors_with_values.iter();
            // .unwrap() is safe because predecessors_with_values is not empty in this branch.
            let mut acc_val = iter.next().unwrap().0.acc.clone(); // .0 accesses the Arc<GSSNode>
            for (pred_arc, _) in iter {
                acc_val = acc_val.union(&pred_arc.acc);
            }
            acc_val
        };
        let hash_key_cache = compute_internal_hash_key::<T, A>(&predecessors_with_values);
        Self {
            acc: unioned_acc,
            predecessors_with_values,
            hash_key_cache,
        }
    }

    pub fn predecessors_with_values(&self) -> &BTreeSet<(Arc<Self>, T)> {
        &self.predecessors_with_values
    }

    pub fn is_empty(&self) -> bool {
        self.predecessors_with_values.is_empty()
    }
}

// Methods involving canonicalization
impl<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug> GSSNode<T, A> {
    /// Internal method to get/create a canonical Arc<GSSNode<T, A>>.
    /// A node is defined by its set of (predecessor_node, edge_value_T) pairs.
    fn get_canonical(
        predecessors_with_values: BTreeSet<(Arc<Self>, T)>, // Consumed
        cache: &mut HashMap<BTreeSet<(Arc<GSSNode<T, A>>, T)>, Arc<GSSNode<T, A>>>,
    ) -> Arc<Self> {
        let key_for_lookup = predecessors_with_values.clone();

        let current_context_unioned_acc = if predecessors_with_values.is_empty() {
            A::default()
        } else {
            let mut iter = predecessors_with_values.iter();
            let mut acc_val = iter.next().unwrap().0.acc.clone();
            for (pred_arc, _) in iter {
                acc_val = acc_val.union(&pred_arc.acc);
            }
            acc_val
        };

        if let Some(entry_arc) = cache.get_mut(&key_for_lookup) {
            let new_potential_acc = entry_arc.acc.union(&current_context_unioned_acc);
            if new_potential_acc != entry_arc.acc {
                let mut temp_arc = entry_arc.clone();
                let node_instance_mut = Arc::make_mut(&mut temp_arc);
                node_instance_mut.acc = new_potential_acc;
                *entry_arc = temp_arc.clone();
                return temp_arc;
            }
            return entry_arc.clone();
        }

        let hash_key_cache = compute_internal_hash_key::<T, A>(&predecessors_with_values);
        let new_node = GSSNode {
            acc: current_context_unioned_acc,
            predecessors_with_values,
            hash_key_cache,
        };
        let new_node_arc = Arc::new(new_node);
        cache.insert(key_for_lookup, new_node_arc.clone());
        new_node_arc
    }

    /// Creates a new canonical root GSSNode (no predecessors) with a specific initial accumulator.
    pub fn new_canonical(initial_acc: A, cache: &mut HashMap<BTreeSet<(Arc<GSSNode<T, A>>, T)>, Arc<GSSNode<T, A>>>) -> Arc<Self> {
        let predecessors_with_values = BTreeSet::new();
        let key = predecessors_with_values.clone();
        if let Some(entry_arc) = cache.get_mut(&key) {
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

        let hash_key_cache = compute_internal_hash_key::<T, A>(&predecessors_with_values);
        let new_node_arc = Arc::new(GSSNode {
            acc: initial_acc,
            predecessors_with_values,
            hash_key_cache,
        });
        cache.insert(key, new_node_arc.clone());
        new_node_arc
    }
}

// Public methods (mostly non-canonical)
impl<T: Ord + Hash + Clone, A: PathAccumulator + Clone> GSSNode<T, A> { // Added Clone to T and A for self.clone() in push
    pub fn push(self, edge_value: T) -> Self {
        let mut new_node_predecessors_with_values = BTreeSet::new();
        new_node_predecessors_with_values.insert((Arc::new(self), edge_value));
        Self::new_with_predecessors(new_node_predecessors_with_values)
    }

    pub fn pop_into(&self, mut result: GSSNode<T, A>) -> GSSNode<T, A> {
        for (pred_arc, edge_val) in self.predecessors_with_values() {
            result.merge(pred_arc.as_ref().clone());
        }
        result
    }

    pub fn pop(&self) -> GSSNode<T, A> {
        // self.clone()
        self.pop_into(GSSNode::new_default())
    }

    pub fn popn(&self, n: usize) -> GSSNode<T, A> {
        if n == 0 {
            self.clone()
        } else {
            self.pop().popn(n - 1)
        }
    }

    pub fn acc(&self) -> &A {
        &self.acc
    }

    pub fn acc_mut(&mut self) -> &mut A {
        &mut self.acc
    }

    pub fn flatten(&self) -> Vec<Vec<(T, A)>> {
        let mut result_paths = Vec::new();
        let mut q_flatten: VecDeque<(&GSSNode<T,A>, Vec<(T,A)>)> = VecDeque::new();
        q_flatten.push_back((self, Vec::new()));

        while let Some((current_node_ref, mut path_suffix)) = q_flatten.pop_front() {
            if current_node_ref.predecessors_with_values.is_empty() {
                path_suffix.reverse();
                result_paths.push(path_suffix);
            } else {
                for (pred_arc, edge_val) in &current_node_ref.predecessors_with_values {
                    let mut new_path_suffix = path_suffix.clone();
                    new_path_suffix.push((edge_val.clone(), current_node_ref.acc.clone()));
                    q_flatten.push_back((pred_arc.as_ref(), new_path_suffix));
                }
            }
        }
        result_paths
    }

    pub fn flatten_bulk(nodes: &[Arc<GSSNode<T, A>>]) -> Vec<Vec<(T, A)>> {
        nodes.iter().flat_map(|node_arc| node_arc.as_ref().flatten()).collect()
    }

    /// Returns an iterator over all paths that terminate at this GSS node.
    ///
    /// Each path is a `Vec<T>` containing the sequence of edge values from a root
    /// of the GSS graph to this node. The paths are yielded as they are found by
    /// a breadth-first search-like traversal upwards from this node.
    ///
    /// Example: If node C has a predecessor B with edge value `val_bc`, and B has
    /// a predecessor A (a root) with edge value `val_ab`, then one path yielded
    /// for `C.iter_paths()` would be `vec![val_ab, val_bc]`.
    pub fn iter_paths(&self) -> PathsIter<'_, T, A> {
        let mut queue = VecDeque::new();
        // Start traversal from `self` with an empty path suffix.
        queue.push_back((self, Vec::new()));
        PathsIter { queue }
    }

    pub fn merge(&mut self, mut other: Self) {
        self.acc = self.acc.union(&other.acc);
        self.predecessors_with_values.append(&mut other.predecessors_with_values);
        self.hash_key_cache = compute_internal_hash_key::<T, A>(&self.predecessors_with_values);
    }

    pub fn merged(self, other: Self) -> Self {
        let mut merged = self.clone();
        merged.merge(other);
        merged
    }

    pub fn map<F_edge, U_edge>(&self, f_edge: F_edge) -> GSSNode<U_edge, A>
    where
        F_edge: Copy + Fn(&T) -> U_edge,
        U_edge: Ord + Hash + Clone, // Clone for BTreeSet if U_edge is part of key
        A: PathAccumulator + Clone,
    {
        let new_predecessors_with_values: BTreeSet<(Arc<GSSNode<U_edge, A>>, U_edge)> =
            self.predecessors_with_values.iter()
            .map(|(pred_arc_t_a, edge_val_t)| {
                let mapped_pred_arc = Arc::new(pred_arc_t_a.as_ref().map(f_edge));
                let new_edge_val_u = f_edge(edge_val_t);
                (mapped_pred_arc, new_edge_val_u)
            })
            .collect();

        GSSNode::<U_edge, A>::new_with_predecessors(new_predecessors_with_values)
    }
}

impl<T, A: PathAccumulator> Drop for GSSNode<T, A> {
    fn drop(&mut self) {
        let predecessors_to_process_further = std::mem::take(&mut self.predecessors_with_values);
        let mut worklist: Vec<Arc<GSSNode<T, A>>> = predecessors_to_process_further
            .into_iter()
            .map(|(p_arc, _)| p_arc)
            .collect();

        while let Some(node_arc) = worklist.pop() {
            if Arc::strong_count(&node_arc) == 1 {
                if let Ok(mut inner_node) = Arc::try_unwrap(node_arc) {
                    let inner_preds = std::mem::take(&mut inner_node.predecessors_with_values);
                    worklist.extend(inner_preds.into_iter().map(|(p,_)| p));
                }
            }
        }
    }
}

impl<T: Hash, A: PathAccumulator> Hash for GSSNode<T, A> {
    fn hash<H_hasher: Hasher>(&self, state: &mut H_hasher) {
        self.hash_key_cache.hash(state);
    }
}

impl<T: Ord + Hash + PartialEq, A: PathAccumulator + PartialEq> PartialEq for GSSNode<T, A> { // T needs Ord for BTreeSet
    fn eq(&self, other: &Self) -> bool {
        if std::ptr::eq(self, other) { return true; }
        self.hash_key_cache == other.hash_key_cache &&
        self.acc == other.acc &&
        self.predecessors_with_values == other.predecessors_with_values
    }
}

impl<T: Ord + Hash + Eq, A: PathAccumulator + Eq> Eq for GSSNode<T, A> {} // T needs Ord for BTreeSet

impl<T: Ord + Hash + PartialOrd, A: PathAccumulator + PartialOrd> PartialOrd for GSSNode<T, A> { // T needs Ord for BTreeSet
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if std::ptr::eq(self, other) { return Some(Ordering::Equal); }
        match self.hash_key_cache.partial_cmp(&other.hash_key_cache) {
            Some(Ordering::Equal) => {
                match self.acc.partial_cmp(&other.acc) {
                    Some(Ordering::Equal) => self.predecessors_with_values.partial_cmp(&other.predecessors_with_values),
                    other_ordering => other_ordering,
                }
            }
            other_ordering => other_ordering,
        }
    }
}

impl<T: Ord + Hash, A: PathAccumulator + Ord> Ord for GSSNode<T, A> { // T needs Ord for BTreeSet
    fn cmp(&self, other: &Self) -> Ordering {
        if std::ptr::eq(self, other) { return Ordering::Equal; }
        self.hash_key_cache.cmp(&other.hash_key_cache)
            .then_with(|| self.acc.cmp(&other.acc))
            .then_with(|| self.predecessors_with_values.cmp(&other.predecessors_with_values))
    }
}

pub trait GSSTrait<T: Clone + Hash, A: PathAccumulator> {
    // type Peek<'a> where A: 'a, Self: 'a;
    // fn peek(&self) -> Self::Peek<'_>;
    fn push(&self, edge_value: T) -> GSSNode<T, A> where T: Ord + Clone, A: Clone; // Added Clone for GSSNode::push(self_owned_clone)
    fn push_to(&self, edge_value: T, dest: &mut GSSNode<T, A>) where T: Ord + Clone, A: Clone; // Added Clone for GSSNode::push(self_owned_clone)
    fn pop(&self) -> GSSNode<T, A> where T: Ord + Clone, A: Clone; // Added Clone for popn's GSSNode::popn
    fn popn(&self, n: usize) -> GSSNode<T, A> where T: Ord + Clone, A: Clone; // Added Clone for popn's GSSNode::popn
}

impl<T: Clone + Ord + Hash, A: PathAccumulator + Clone> GSSTrait<T, A> for GSSNode<T, A> {
    // type Peek<'a> = &'a A where A: 'a, T: 'a;

    // fn peek(&self) -> Self::Peek<'_> {
    //     &self.acc
    // }

    fn push(&self, edge_value: T) -> GSSNode<T, A> {
        let self_owned_clone = self.clone();
        GSSNode::push(self_owned_clone, edge_value)
    }

    fn push_to(&self, edge_value: T, dest: &mut GSSNode<T, A>) {
        GSSNode::push_to(&self, edge_value, dest)
    }

    fn pop(&self) -> GSSNode<T, A> {
        GSSNode::pop(self)
    }

    fn popn(&self, n: usize) -> GSSNode<T, A> {
        GSSNode::popn(self, n)
    }
}

impl<T: Clone + Ord + Hash, A: PathAccumulator + Clone> GSSTrait<T, A> for Arc<GSSNode<T, A>> {
    // type Peek<'a> = &'a A where A: 'a, T: 'a;
    //
    // fn peek(&self) -> Self::Peek<'_> {
    //     &self.acc
    // }

    fn push(&self, edge_value: T) -> GSSNode<T, A> {
        let mut new_preds_with_values = BTreeSet::new();
        new_preds_with_values.insert((self.clone(), edge_value));
        GSSNode::new_with_predecessors(new_preds_with_values)
    }

    fn push_to(&self, edge_value: T, dest: &mut GSSNode<T, A>) {
        dest.merge(self.as_ref().clone());
        dest.acc.pop(&self.acc);
    }

    fn pop(&self) -> GSSNode<T, A> {
        self.as_ref().pop()
    }

    fn popn(&self, n: usize) -> GSSNode<T, A> {
        self.as_ref().popn(n)
    }
}

impl<T: Clone + Ord + Hash, A: PathAccumulator + Clone + Default> GSSTrait<T, A> for Option<Arc<GSSNode<T, A>>> {
    // type Peek<'a> = Option<&'a A> where A: 'a, T: 'a;
    //
    // fn peek(&self) -> Self::Peek<'_> {
    //     self.as_ref().map(|node_arc| node_arc.peek())
    // }

    fn push(&self, edge_value: T) -> GSSNode<T, A> {
        match self {
            Some(arc_node) => arc_node.push(edge_value),
            None => {
                let root_state = GSSNode::new(A::default());
                root_state.push(edge_value)
            }
        }
    }

    fn push_to(&self, edge_value: T, dest: &mut GSSNode<T, A>) {
        match self {
            Some(arc_node) => arc_node.push_to(edge_value, dest),
            None => {
                let root_state = GSSNode::new(A::default());
                root_state.push_to(edge_value, dest)
            }
        }
    }

    fn pop(&self) -> GSSNode<T, A> {
        self.as_ref().map(|node_arc| node_arc.pop()).unwrap_or_else(GSSNode::new_default)
    }

    fn popn(&self, n: usize) -> GSSNode<T, A> {
        self.as_ref().map(|node_arc| node_arc.popn(n)).unwrap_or_else(GSSNode::new_default)
    }
}

impl<T: Clone + Ord + Hash, A: PathAccumulator + Clone + Default> GSSTrait<T, A> for Option<GSSNode<T, A>> {
    //  type Peek<'a> = Option<&'a A> where A: 'a, T: 'a;
    //
    // fn peek(&self) -> Self::Peek<'_> {
    //     self.as_ref().map(|node| node.peek())
    // }

    fn push(&self, edge_value: T) -> GSSNode<T, A> {
        match self {
            Some(node) => node.clone().push(edge_value),
            None => {
                let root_state = GSSNode::new(A::default());
                root_state.push(edge_value)
            }
        }
    }

    fn push_to(&self, edge_value: T, dest: &mut GSSNode<T, A>) {
        match self {
            Some(node) => node.push_to(edge_value, dest),
            None => {
                let root_state = GSSNode::new(A::default());
                root_state.push_to(edge_value, dest)
            }
        }
    }

    fn pop(&self) -> GSSNode<T, A> {
        self.as_ref().map(|node| node.pop()).unwrap_or_else(GSSNode::new_default)
    }

    fn popn(&self, n: usize) -> GSSNode<T, A> {
        self.as_ref().map(|node| node.popn(n)).unwrap_or_else(GSSNode::new_default)
    }
}

/*
// BulkMerge trait and its implementation are commented out as they relied on `node.value: T`.
// A redesign is needed, specifying new grouping criteria suitable for edge-valued GSS.
pub trait BulkMerge<T_NodeVal, T_EdgeVal, A: PathAccumulator> {
    fn bulk_merge(&mut self);
}
// ... implementations ...
pub fn bulk_merge_canonical ...
*/

pub fn prune_and_transform_recursive_canonical<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    node_arc: &Arc<GSSNode<T, A>>,
    closure: &impl Fn(&A) -> Option<(A, bool)>,
    memo: &mut HashMap<*const GSSNode<T, A>, Option<Arc<GSSNode<T, A>>>>,
    cache: &mut HashMap<BTreeSet<(Arc<GSSNode<T, A>>, T)>, Arc<GSSNode<T, A>>>,
) -> Option<Arc<GSSNode<T, A>>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }

    match closure(&node_arc.acc) {
        None => {
            memo.insert(node_ptr, None);
            None
        }
        Some((new_acc_for_this_node, continue_recursion)) => {
            let new_predecessors_with_values: BTreeSet<(Arc<GSSNode<T, A>>, T)>;
            if continue_recursion {
                let mut current_new_preds_w_vals = BTreeSet::new();
                for (pred_arc, edge_val) in &node_arc.predecessors_with_values {
                    if let Some(new_pred_arc) = prune_and_transform_recursive_canonical(pred_arc, closure, memo, cache) {
                        current_new_preds_w_vals.insert((new_pred_arc, edge_val.clone()));
                    }
                }
                new_predecessors_with_values = current_new_preds_w_vals;
            } else {
                new_predecessors_with_values = node_arc.predecessors_with_values.clone();
            };

            let new_node_arc = GSSNode::get_canonical(new_predecessors_with_values, cache);

            let mut temp_arc = new_node_arc.clone();
            let node_instance_mut = Arc::make_mut(&mut temp_arc);
            node_instance_mut.acc = node_instance_mut.acc.union(&new_acc_for_this_node);

            memo.insert(node_ptr, Some(temp_arc.clone()));
            Some(temp_arc)
        }
    }
}

pub fn prune_and_transform_roots_canonical<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T, A>>],
    closure: &impl Fn(&A) -> Option<(A, bool)>,
    cache: &mut HashMap<BTreeSet<(Arc<GSSNode<T, A>>, T)>, Arc<GSSNode<T, A>>>,
) -> Vec<Option<Arc<GSSNode<T, A>>>> {
    let mut memo = HashMap::new();
    roots
        .iter()
        .map(|root| prune_and_transform_recursive_canonical(root, closure, &mut memo, cache))
        .collect()
}

pub fn prune_and_transform_recursive<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    node_arc: &Arc<GSSNode<T, A>>,
    closure: &impl Fn(&A) -> Option<(A, bool)>,
    memo: &mut HashMap<*const GSSNode<T, A>, Option<Arc<GSSNode<T, A>>>>,
) -> Option<Arc<GSSNode<T, A>>> {
    let mut cache = HashMap::<BTreeSet<(Arc<GSSNode<T, A>>, T)>, Arc<GSSNode<T, A>>>::new();
    prune_and_transform_recursive_canonical(node_arc, closure, memo, &mut cache)
}

pub fn prune_and_transform_roots<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T, A>>],
    closure: &impl Fn(&A) -> Option<(A, bool)>,
) -> Vec<Option<Arc<GSSNode<T, A>>>> {
    let mut memo = HashMap::new();
    let mut cache = HashMap::<BTreeSet<(Arc<GSSNode<T, A>>, T)>, Arc<GSSNode<T, A>>>::new();
    roots
        .iter()
        .map(|root| prune_and_transform_recursive_canonical(root, closure, &mut memo, &mut cache))
        .collect()
}

// Helper recursive function for find_longest_path.
// It computes the longest path consisting of (edge_value, node_reached_by_edge) pairs,
// ending with `node_arc` being the node reached by the last edge in the path.
// T: Edge value type, must be Cloneable (for path construction), Ord + Hash (GSSNode requirements).
// A: PathAccumulator type.
fn find_longest_path_ending_at_node_recursive<T: Clone + Ord + Hash, A: PathAccumulator>(
    node_arc: &Arc<GSSNode<T, A>>,
    memo: &mut HashMap<*const GSSNode<T, A>, Vec<(T, Arc<GSSNode<T, A>>)>>,
    visited_recursion: &mut HashSet<*const GSSNode<T, A>>,
) -> Vec<(T, Arc<GSSNode<T, A>>)> {
    let node_ptr = Arc::as_ptr(node_arc);

    if let Some(cached_path) = memo.get(&node_ptr) {
        return cached_path.clone();
    }
    if !visited_recursion.insert(node_ptr) {
        return Vec::new(); // Cycle detected, return empty path for this branch
    }

    // Base case: If node_arc has no predecessors, it's a "root" in the GSS structure.
    // No edge leads to it, so the path of (edge, node) pairs ending here is empty.
    if node_arc.predecessors_with_values.is_empty() {
        visited_recursion.remove(&node_ptr);
        memo.insert(node_ptr, Vec::new());
        return Vec::new();
    }

    let mut longest_path_found: Vec<(T, Arc<GSSNode<T, A>>)> = Vec::new();

    // Iterate over all predecessors to find the one that contributes to the longest path to current node_arc
    for (pred_arc, edge_val_to_current_node) in &node_arc.predecessors_with_values {
        let path_from_pred_recursive = find_longest_path_ending_at_node_recursive(
            pred_arc,
            memo,
            visited_recursion,
        );

        // Construct the candidate path: path to predecessor + (edge_to_current, current_node)
        let mut current_candidate_path = path_from_pred_recursive;
        current_candidate_path.push((edge_val_to_current_node.clone(), node_arc.clone()));

        if current_candidate_path.len() > longest_path_found.len() {
            longest_path_found = current_candidate_path;
        }
    }

    memo.insert(node_ptr, longest_path_found.clone());
    visited_recursion.remove(&node_ptr);
    longest_path_found
}

/// Finds the longest path leading to the predecessors of the given `root_node`.
/// The path is a sequence of (edge_value, node_reached_by_edge) tuples.
/// The `root_node` itself is not part of the returned path.
/// If `root_node` has no predecessors, or if all paths are empty, returns `None` or `Some(Vec::new())`.
pub fn find_longest_path<T: Clone + Ord + Hash, A: PathAccumulator>(root_node: &GSSNode<T, A>) -> Option<Vec<(T, Arc<GSSNode<T, A>>)>> {
    if root_node.predecessors_with_values.is_empty() {
        return None; // No predecessors, so no path leading to it.
    }

    let mut memo: HashMap<*const GSSNode<T, A>, Vec<(T, Arc<GSSNode<T, A>>)>> = HashMap::new();
    let mut longest_overall_path: Option<Vec<(T, Arc<GSSNode<T, A>>)>> = None;

    // Iterate over direct predecessors of root_node. The path we seek ends at one of these predecessors.
    for (pred_arc, _edge_val_to_pred) in root_node.predecessors_with_values() {
        let mut visited_recursion = HashSet::new(); // Fresh for each DFS traversal from a direct predecessor
        let path_ending_at_pred = find_longest_path_ending_at_node_recursive(pred_arc, &mut memo, &mut visited_recursion);

        // If this path is longer than any found so far, update longest_overall_path.
        // This handles the first path found (longest_overall_path is None) and subsequent longer paths.
        // An empty path_ending_at_pred can become the longest_overall_path if it's the first one considered.
        if longest_overall_path.as_ref().map_or(true, |current_longest| path_ending_at_pred.len() > current_longest.len()) {
            longest_overall_path = Some(path_ending_at_pred);
        }
    }
    longest_overall_path
}

#[derive(Debug, Clone, Default)]
pub struct GSSStats {
    pub num_roots: usize,
    pub unique_nodes: usize,
    pub max_depth: usize,
    pub average_depth: f64,
    pub merge_points: usize,
    pub max_predecessors_with_values: usize,
    pub average_predecessors_with_values: f64,
}

pub fn gather_gss_stats<T, A: PathAccumulator>(roots: &[Arc<GSSNode<T, A>>]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut q_visited: HashSet<*const GSSNode<T, A>> = HashSet::new(); // Tracks nodes added to queue
    let mut processed_nodes: HashSet<*const GSSNode<T, A>> = HashSet::new(); // Tracks nodes fully processed by BFS
    let mut queue: VecDeque<(Arc<GSSNode<T, A>>, usize)> = VecDeque::new();

    let mut total_depth_sum: u64 = 0;
    let mut total_preds_w_vals_sum: u64 = 0;

    for root_arc in roots {
        if q_visited.insert(Arc::as_ptr(root_arc)) {
            queue.push_back((root_arc.clone(), 0));
        }
    }

    while let Some((current_node_arc, current_depth)) = queue.pop_front() {
        if !processed_nodes.insert(Arc::as_ptr(&current_node_arc)) {
            continue;
        }

        stats.unique_nodes += 1;
        stats.max_depth = stats.max_depth.max(current_depth);
        total_depth_sum += current_depth as u64;

        let num_preds_w_vals = current_node_arc.predecessors_with_values.len();
        stats.max_predecessors_with_values = stats.max_predecessors_with_values.max(num_preds_w_vals);
        total_preds_w_vals_sum += num_preds_w_vals as u64;

        let unique_pred_nodes: HashSet<*const GSSNode<T,A>> = current_node_arc.predecessors_with_values.iter()
            .map(|(p, _)| Arc::as_ptr(p)).collect();
        if unique_pred_nodes.len() > 1 {
            stats.merge_points += 1;
        }

        for (pred_arc, _edge_val) in &current_node_arc.predecessors_with_values {
            if q_visited.insert(Arc::as_ptr(pred_arc)) {
                queue.push_back((pred_arc.clone(), current_depth + 1));
            }
        }
    }

    if stats.unique_nodes > 0 {
        stats.average_depth = total_depth_sum as f64 / stats.unique_nodes as f64;
        stats.average_predecessors_with_values = total_preds_w_vals_sum as f64 / stats.unique_nodes as f64;
    }
    stats
}

fn print_gss_node_recursive<T: Debug, A: PathAccumulator>(
    node_arc: &Arc<GSSNode<T, A>>,
    visited: &mut HashSet<*const GSSNode<T, A>>,
    indent: usize,
    node_count: &mut usize,
    max_nodes_to_print: usize,
    output: &mut String,
) -> Result<(), std::fmt::Error> {
    if *node_count >= max_nodes_to_print {
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

    writeln!(output, "{}- Node {:p}: (Acc: {:?})", prefix, node_ptr, node_arc.acc)?;

    if !node_arc.predecessors_with_values.is_empty() {
        writeln!(output, "{}  Predecessors (Edge Value, Pred Node Ptr):", prefix)?;
        for (pred_arc, edge_val) in &node_arc.predecessors_with_values {
            // Limit recursion depth for printing if necessary, or rely on max_nodes_to_print
            writeln!(output, "{}    - Edge: {:?}, Pred_Node: {:p}", prefix, edge_val, Arc::as_ptr(pred_arc))?;
            if *node_count < max_nodes_to_print { // Check before recursive call
                 print_gss_node_recursive(pred_arc, visited, indent + 2, node_count, max_nodes_to_print, output)?;
            }
            if *node_count >= max_nodes_to_print { // Check after, to stop further iteration if limit reached in recursion
                return Ok(());
            }
        }
    }
    Ok(())
}

pub fn print_gss_forest<T: Debug, A: PathAccumulator>(roots: &[Arc<GSSNode<T, A>>], max_nodes_to_print: usize) -> String {
    let mut visited = HashSet::new();
    let mut node_count = 0;
    let mut output = String::new();

    if roots.is_empty() {
        return "GSS Forest: (No roots)".to_string();
    }
    writeln!(&mut output, "GSS Forest Roots (Max Nodes to Print: {}):", max_nodes_to_print).unwrap();

    for (i, root_arc) in roots.iter().enumerate() {
        writeln!(&mut output, "Root {}: {:p}", i, Arc::as_ptr(root_arc)).unwrap();
        match print_gss_node_recursive(root_arc, &mut visited, 1, &mut node_count, max_nodes_to_print, &mut output) {
            Ok(_) => {
                if node_count >= max_nodes_to_print && i < roots.len() -1 {
                    writeln!(&mut output, "... (Truncated: Reached max nodes {})", max_nodes_to_print).unwrap();
                    break;
                }
            }
            Err(e) => {
                eprintln!("Error writing GSS structure to string: {}", e);
                return format!("Error generating GSS string: {}", e);
            }
        }
    }
    if node_count >= max_nodes_to_print && visited.len() > node_count {
         writeln!(&mut output, "... (More nodes exist but not printed due to max_nodes_to_print limit)").unwrap();
    }
    output
}

fn simplify_node_recursive<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    original_node_arc: &Arc<GSSNode<T, A>>,
    memo: &mut HashMap<*const GSSNode<T, A>, Arc<GSSNode<T, A>>>,
    canonicalization_cache: &mut HashMap<BTreeSet<(Arc<GSSNode<T, A>>, T)>, Arc<GSSNode<T, A>>>,
) -> Arc<GSSNode<T, A>> {
    // TODO: delete this
    // return original_node_arc.clone();
    let original_node_ptr = Arc::as_ptr(original_node_arc);
    if let Some(canonical_arc) = memo.get(&original_node_ptr) {
        return canonical_arc.clone();
    }

    let mut new_predecessors_with_values: BTreeSet<(Arc<GSSNode<T, A>>, T)> = BTreeSet::new();
    for (original_pred_arc, edge_val) in &original_node_arc.predecessors_with_values {
        let simplified_pred_arc = simplify_node_recursive(
            original_pred_arc,
            memo,
            canonicalization_cache,
        );
        new_predecessors_with_values.insert((simplified_pred_arc, edge_val.clone()));
    }

    let mut predecessors_grouped: BTreeMap<T, Arc<GSSNode<T, A>>> = BTreeMap::new();
    for (pred_arc, edge_val) in &new_predecessors_with_values {
        // Key by everything except the predecessor's acc
        if let Some(existing) = predecessors_grouped.get_mut(edge_val) {
            Arc::make_mut(existing).merge(pred_arc.as_ref().clone());
        } else {
            predecessors_grouped.insert(edge_val.clone(), pred_arc.clone());
        }
    }
    let mut new_predecessors_with_values: BTreeSet<(Arc<GSSNode<T, A>>, T)> = BTreeSet::new();
    for (edge_val, pred_arc) in predecessors_grouped {
        new_predecessors_with_values.insert((pred_arc, edge_val));
    }

    let canonical_arc = GSSNode::get_canonical(
        new_predecessors_with_values,
        canonicalization_cache,
    );

    let mut temp_arc = canonical_arc.clone();
    let node_instance_mut = Arc::make_mut(&mut temp_arc);
    node_instance_mut.acc = node_instance_mut.acc.union(&original_node_arc.acc);

    memo.insert(original_node_ptr, temp_arc.clone());
    temp_arc
}

fn simplify_gss_forest<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T, A>>],
) -> Vec<Arc<GSSNode<T, A>>> {
    // TODO: delete this
    // return roots.to_vec();
    let mut memo: HashMap<*const GSSNode<T, A>, Arc<GSSNode<T, A>>> = HashMap::new();
    let mut canonicalization_cache: HashMap<BTreeSet<(Arc<GSSNode<T, A>>, T)>, Arc<GSSNode<T, A>>> = HashMap::<BTreeSet<(Arc<GSSNode<T, A>>, T)>, Arc<GSSNode<T, A>>>::new();
    let mut simplified_roots_vec = Vec::with_capacity(roots.len());

    for root_arc in roots {
        simplified_roots_vec.push(simplify_node_recursive(
            root_arc,
            &mut memo,
            &mut canonicalization_cache,
        ));
    }

    // Deduplicate root Arcs if multiple original roots simplify to the same canonical Arc
    let mut unique_simplified_roots_map: HashMap<*const GSSNode<T,A>, Arc<GSSNode<T,A>>> = HashMap::new();
    for r_arc in simplified_roots_vec.iter_mut() { // Use _vec to avoid conflict with original roots name
        let unique_r_arc =unique_simplified_roots_map.entry(Arc::as_ptr(&r_arc)).or_insert(r_arc.clone());
        *r_arc = unique_r_arc.clone();

    }
    assert_eq!(roots.len(), simplified_roots_vec.len());
    simplified_roots_vec
}

impl<T: Ord + Hash + Clone + Debug, A: PathAccumulator + Clone> GSSNode<T, A> {
    pub fn simplify(&mut self) {
        let simplified_roots = simplify_gss_forest(&[Arc::new(self.clone())]);
        *self = simplified_roots[0].as_ref().clone();
    }

    pub fn simplify_recursive(
        this: &mut Arc<GSSNode<T, A>>,
        memo: &mut HashMap<*const GSSNode<T,A>, Arc<GSSNode<T,A>>>,
        canonicalization_cache: &mut HashMap<BTreeSet<(Arc<GSSNode<T, A>>, T)>, Arc<GSSNode<T, A>>>,
    ) {
        *this = simplify_node_recursive(&Arc::new(this.as_ref().clone()), memo, canonicalization_cache);
    }

    pub fn simplify_together(nodes: &mut [Arc<GSSNode<T, A>>]) {
        let mut memo: HashMap<*const GSSNode<T, A>, Arc<GSSNode<T, A>>> = HashMap::new();
        let mut canonicalization_cache: HashMap<BTreeSet<(Arc<GSSNode<T, A>>, T)>, Arc<GSSNode<T, A>>> = HashMap::<BTreeSet<(Arc<GSSNode<T, A>>, T)>, Arc<GSSNode<T, A>>>::new();
        for node in nodes {
            *node = simplify_node_recursive(node, &mut memo, &mut canonicalization_cache);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        fn default() -> Self { Self { active: BTreeSet::new(), intersection: BTreeSet::new() } }
    }
    impl PathAccumulator for MockPathAccumulator {
        fn union(&self, other: &Self) -> Self {
            Self {
                active: self.active.union(&other.active).cloned().collect(),
                intersection: self.intersection.union(&other.intersection).cloned().collect(),
            }
        }

        fn pop(&self, right: &Self) -> Self {
            Self {
                active: self.active.intersection(&right.active).cloned().collect(),
                intersection: self.intersection.intersection(&right.intersection).cloned().collect(),
            }
        }
    }

    type MockGSSNode = GSSNode<i32, MockPathAccumulator>;
    type MockNodeCache = HashMap<BTreeSet<(Arc<GSSNode<i32, MockPathAccumulator>>, i32)>, Arc<GSSNode<i32, MockPathAccumulator>>>;

    fn nc_node_arc(
        _initial_acc_for_new_node: MockPathAccumulator,
        preds_with_vals_vec: Vec<(Arc<MockGSSNode>, i32)>
    ) -> Arc<MockGSSNode> {
        let pred_set: BTreeSet<(Arc<MockGSSNode>, i32)> = preds_with_vals_vec.into_iter().collect();
        let mut node = MockGSSNode::new_with_predecessors(pred_set);
        // If _initial_acc_for_new_node is meant to override derived, do it here:
        // node.acc = _initial_acc_for_new_node; // Or node.acc = node.acc.union(&_initial_acc_for_new_node);
        Arc::new(node)
    }

    fn nc_root_node_arc(acc: MockPathAccumulator) -> Arc<MockGSSNode> {
         Arc::new(MockGSSNode::new(acc))
    }

    fn c_root_node_arc(acc: MockPathAccumulator, cache: &mut MockNodeCache) -> Arc<MockGSSNode> { // Canonical root
        MockGSSNode::new_canonical(acc, cache)
    }

    fn collect_arcs_recursive(
        node_arc: &Arc<MockGSSNode>,
        collected_arcs: &mut HashMap<*const MockGSSNode, Arc<MockGSSNode>>,
    ) {
        let ptr = Arc::as_ptr(node_arc);
        if collected_arcs.contains_key(&ptr) {
            return;
        }
        collected_arcs.insert(ptr, node_arc.clone());
        for (pred_arc, _edge_val) in &node_arc.predecessors_with_values {
            collect_arcs_recursive(pred_arc, collected_arcs);
        }
    }

    #[test]
    fn test_gss_simplification_basic() {
        let acc_base = MockPathAccumulator { active: BTreeSet::from([0]), intersection: BTreeSet::from([0]) };
        let acc_other = MockPathAccumulator { active: BTreeSet::from([1]), intersection: BTreeSet::from([1]) };

        let n4_base_nc = nc_root_node_arc(acc_base.clone());
        let d1_orig_nc = nc_node_arc(MockPathAccumulator::default(), vec![(n4_base_nc.clone(), 40)]);

        let n4_other_nc = nc_root_node_arc(acc_other.clone());
        let d2_orig_nc = nc_node_arc(MockPathAccumulator::default(), vec![(n4_other_nc.clone(), 40)]);

        let c1_orig_nc = nc_node_arc(MockPathAccumulator::default(), vec![(d1_orig_nc.clone(), 30)]);
        let b1_orig_nc = nc_node_arc(MockPathAccumulator::default(), vec![(c1_orig_nc.clone(), 20)]);

        let a1_preds_nc = vec![
            (b1_orig_nc.clone(), 10),
            (d2_orig_nc.clone(), 10)
        ];
        let a1_orig_nc = nc_node_arc(MockPathAccumulator::default(), a1_preds_nc);

        let roots_nc = vec![a1_orig_nc.clone()];
        println!("Before simplifying GSS forest: {}", print_gss_forest(&roots_nc, usize::MAX));
        let simplified_roots = simplify_gss_forest(&roots_nc);
        println!("After simplifying GSS forest: {}", print_gss_forest(&simplified_roots, usize::MAX));
        assert_eq!(simplified_roots.len(), 1);
        let s_a1 = simplified_roots[0].clone();

        let mut collected_arcs = HashMap::new();
        collect_arcs_recursive(&s_a1, &mut collected_arcs);
        assert_eq!(collected_arcs.len(), 7, "Expected 7 unique Arcs after simplification");

        assert_eq!(s_a1.predecessors_with_values.len(), 2);
        let acc_expected_a1 = d1_orig_nc.acc.union(&d2_orig_nc.acc); // More precisely, B1.acc U D2.acc
                                                                    // B1.acc is from C1.acc, from D1.acc, from N4_base.acc
                                                                    // D2.acc is from N4_other.acc
        let expected_a1_acc_manual = acc_base.union(&acc_other);
        assert_eq!(s_a1.acc, expected_a1_acc_manual);


        let s_b1_arc_opt = s_a1.predecessors_with_values.iter()
            .find_map(|(p, edge_val)| {
                // B1 is pred of A1 via edge 10. B1's pred is C1. C1's pred is D1. D1's pred is N4_base.
                // Check if this path has acc_base at its root.
                if *edge_val == 10 && p.predecessors_with_values.len() == 1 { // p is B1
                    let (c1_arc, _) = p.predecessors_with_values.iter().next().unwrap();
                    if c1_arc.predecessors_with_values.len() == 1 { // c1_arc is C1
                        let (d1_arc, _) = c1_arc.predecessors_with_values.iter().next().unwrap();
                        if d1_arc.predecessors_with_values.len() == 1 { // d1_arc is D1
                             let (n4_arc, _) = d1_arc.predecessors_with_values.iter().next().unwrap();
                             if n4_arc.acc == acc_base { Some(p.clone())} else {None}
                        } else {None}
                    } else {None}
                } else { None }
            });
        let s_b1_arc = s_b1_arc_opt.expect("Simplified B1 node not found");
        assert_eq!(s_b1_arc.acc, acc_base);

        let s_d2_arc_opt = s_a1.predecessors_with_values.iter()
            .find_map(|(p, edge_val)| {
                // D2 is pred of A1 via edge 10. D2's pred is N4_other.
                if *edge_val == 10 && p.predecessors_with_values.len() == 1 { // p is D2
                    let (n4_arc, _) = p.predecessors_with_values.iter().next().unwrap();
                    if n4_arc.acc == acc_other { Some(p.clone()) } else { None }
                } else { None }
            });
        let s_d2_arc = s_d2_arc_opt.expect("Simplified D2 node not found");
        assert_eq!(s_d2_arc.acc, acc_other);

        assert_eq!(s_b1_arc.predecessors_with_values.len(), 1);
        let (s_c1_arc, edge_val_c1) = s_b1_arc.predecessors_with_values.iter().next().unwrap();
        assert_eq!(*edge_val_c1, 20);
        assert_eq!(s_c1_arc.acc, acc_base);

        assert_eq!(s_c1_arc.predecessors_with_values.len(), 1);
        let (s_d1_arc, edge_val_d1) = s_c1_arc.predecessors_with_values.iter().next().unwrap();
        assert_eq!(*edge_val_d1, 30);
        assert_eq!(s_d1_arc.acc, acc_base);

        assert_eq!(s_d1_arc.predecessors_with_values.len(), 1);
        let (s_n4_base_arc, edge_val_n4_base) = s_d1_arc.predecessors_with_values.iter().next().unwrap();
        assert_eq!(*edge_val_n4_base, 40);
        assert!(s_n4_base_arc.predecessors_with_values.is_empty());
        assert_eq!(s_n4_base_arc.acc, acc_base);

        assert_eq!(s_d2_arc.predecessors_with_values.len(), 1);
        let (s_n4_other_arc, edge_val_n4_other) = s_d2_arc.predecessors_with_values.iter().next().unwrap();
        assert_eq!(*edge_val_n4_other, 40);
        assert!(s_n4_other_arc.predecessors_with_values.is_empty());
        assert_eq!(s_n4_other_arc.acc, acc_other);

        assert_ne!(Arc::as_ptr(s_n4_base_arc), Arc::as_ptr(s_n4_other_arc));
        assert_ne!(Arc::as_ptr(s_d1_arc), Arc::as_ptr(&s_d2_arc));
    }
}
