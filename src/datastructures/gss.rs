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
// T is the type of the value on the edge.
fn compute_internal_hash_key<T: Hash, A: PathAccumulator>(
    predecessors: &BTreeMap<T, Arc<GSSNode<T, A>>>
) -> u64 {
    let mut hasher = DeterministicHasher::new(DefaultHasher::new());
    // The BTreeMap ensures predecessors are iterated in a canonical order based on T.
    for (edge_val, pred_arc) in predecessors {
        edge_val.hash(&mut hasher);
        pred_arc.hash_key_cache.hash(&mut hasher); // Hash predecessor's hash
    }
    hasher.finish()
}

#[derive(Debug, Clone)]
pub struct GSSNode<T, A: PathAccumulator> {
    acc: A, // Accumulator value
    // Predecessors are stored as a map from edge value to the predecessor node.
    // For a given edge value T, there's one (potentially merged) predecessor node.
    predecessors: BTreeMap<T, Arc<GSSNode<T, A>>>,
    hash_key_cache: u64, // Based on predecessors' (T values and Arcs) hashes only
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
            if current_node_ref.predecessors.is_empty() {
                // This node is a root for the current path.
                // The path is complete. Reverse it to get the correct order.
                path_suffix_reversed.reverse();
                return Some(path_suffix_reversed);
            } else {
                // This node has predecessors. Add them to the queue for further exploration.
                // Iterate over BTreeMap: (edge_val, pred_arc)
                for (edge_val, pred_arc) in &current_node_ref.predecessors {
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

// Helper function to process incoming predecessors from a BTreeSet format
// into the internal BTreeMap format, merging nodes if multiple predecessors
// arrive with the same edge value.
// T needs Ord for BTreeMap key, Hash for GSSNode (indirectly), Clone for map storage.
// A needs PathAccumulator + Clone for GSSNode::merge and Arc::new.
fn process_incoming_predecessors<T: Ord + Hash + Clone, A: PathAccumulator + Clone>(
    incoming_preds_set: &BTreeSet<(Arc<GSSNode<T, A>>, T)>
) -> BTreeMap<T, Arc<GSSNode<T, A>>> {
    let mut grouped_by_edge_val: BTreeMap<T, Vec<Arc<GSSNode<T, A>>>> = BTreeMap::new();
    for (pred_arc, edge_val) in incoming_preds_set {
        // Clone edge_val for key, pred_arc for vec storage
        grouped_by_edge_val.entry(edge_val.clone()).or_default().push(pred_arc.clone());
    }

    let mut final_predecessors_map: BTreeMap<T, Arc<GSSNode<T, A>>> = BTreeMap::new();
    for (edge_val, pred_arcs_list) in grouped_by_edge_val {
        if pred_arcs_list.is_empty() { continue; } // Should not happen if built from BTreeSet

        let mut iter = pred_arcs_list.into_iter();
        let first_arc = iter.next().unwrap(); // Safe due to is_empty check and or_default

        if iter.len() == 0 { // Only one predecessor Arc for this edge_val
            final_predecessors_map.insert(edge_val, first_arc);
        } else { // Multiple predecessor Arcs for this edge_val, their GSSNodes need to be merged
            // Start with the GSSNode from the first_arc (cloned, as we'll modify it via merge)
            let mut merged_node_owned = (*first_arc).clone();
            // Merge GSSNodes from subsequent Arcs into it
            for other_arc in iter {
                // GSSNode::merge takes other: Self (an owned GSSNode).
                // We get GSSNode by dereferencing other_arc and cloning.
                merged_node_owned.merge((*other_arc).clone());
            }
            // Store the Arc to the newly created/merged GSSNode.
            final_predecessors_map.insert(edge_val, Arc::new(merged_node_owned));
        }
    }
    final_predecessors_map
}


// Methods for creating non-canonical GSSNode instances
// T needs Ord + Hash + Clone for BTreeMap key and GSSNode requirements.
// A needs PathAccumulator + Clone for GSSNode storage and operations.
impl<T: Ord + Hash + Clone, A: PathAccumulator + Clone> GSSNode<T, A> {
    /// Creates a new root GSSNode (no predecessors).
    pub fn new(acc: A) -> Self {
        let predecessors = BTreeMap::new();
        let hash_key_cache = compute_internal_hash_key::<T, A>(&predecessors);
        Self {
            acc,
            predecessors,
            hash_key_cache,
        }
    }

    pub fn new_default() -> Self {
        Self::new(A::default())
    }

    /// Creates a new GSSNode. Predecessors are provided as a BTreeSet of (Arc<Node>, EdgeValue).
    /// If multiple predecessors in the set have the same edge value, their pointed-to GSSNodes are merged.
    /// The accumulator `acc` is derived from the union of (merged) predecessor accumulators.
    pub fn new_with_predecessors(predecessors_with_values_set: BTreeSet<(Arc<Self>, T)>) -> Self {
        // Process the input set into the BTreeMap structure, merging nodes as needed.
        let processed_predecessors_map = process_incoming_predecessors(&predecessors_with_values_set);

        let unioned_acc = if processed_predecessors_map.is_empty() {
            A::default()
        } else {
            let mut iter = processed_predecessors_map.values();
            // .unwrap() is safe because processed_predecessors_map is not empty in this branch.
            let mut acc_val = iter.next().unwrap().acc.clone(); // .acc from Arc<GSSNode>
            for pred_arc in iter {
                acc_val = acc_val.union(&pred_arc.acc);
            }
            acc_val
        };
        let hash_key_cache = compute_internal_hash_key::<T, A>(&processed_predecessors_map);
        Self {
            acc: unioned_acc,
            predecessors: processed_predecessors_map,
            hash_key_cache,
        }
    }

    /// For compatibility: returns predecessors in the BTreeSet<(Arc<Node>, EdgeValue)> format.
    pub fn predecessors_with_values(&self) -> BTreeSet<(Arc<Self>, T)> {
        self.predecessors.iter().map(|(edge_val, pred_arc)| (pred_arc.clone(), edge_val.clone())).collect()
    }
    
    /// Direct access to the internal predecessor map: BTreeMap<EdgeValue, Arc<PredecessorNode>>.
    pub fn predecessors(&self) -> &BTreeMap<T, Arc<Self>> {
        &self.predecessors
    }

    pub fn is_empty(&self) -> bool {
        self.predecessors.is_empty()
    }
}

// Methods involving canonicalization
// T needs Clone + Ord + Hash + Debug. A needs PathAccumulator + Clone + Ord + Hash + Debug.
impl<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug> GSSNode<T, A> {
    /// Internal method to get/create a canonical Arc<GSSNode<T, A>>.
    /// A node is defined by its set of (edge_value_T, predecessor_node) pairs.
    /// Input `predecessors_with_values_set` defines the structure.
    fn get_canonical(
        predecessors_with_values_set: BTreeSet<(Arc<Self>, T)>, // Consumed (effectively, as processed)
        cache: &mut HashMap<BTreeMap<T, Arc<GSSNode<T, A>>>, Arc<GSSNode<T, A>>>, // Cache key is BTreeMap
    ) -> Arc<Self> {
        // Process the input BTreeSet into the canonical BTreeMap structure.
        // This map will be used as the cache key.
        let key_for_lookup_map = process_incoming_predecessors(&predecessors_with_values_set);

        // Calculate the accumulator based on the structure defined by key_for_lookup_map.
        // This accumulator represents the combined state flowing from predecessors.
        let current_context_unioned_acc = if key_for_lookup_map.is_empty() {
            A::default()
        } else {
            let mut iter = key_for_lookup_map.values();
            let mut acc_val = iter.next().unwrap().acc.clone();
            for pred_arc in iter {
                acc_val = acc_val.union(&pred_arc.acc);
            }
            acc_val
        };

        if let Some(entry_arc) = cache.get_mut(&key_for_lookup_map) {
            // Node with this structure already exists. Union accumulator.
            let new_potential_acc = entry_arc.acc.union(&current_context_unioned_acc);
            if new_potential_acc != entry_arc.acc {
                // Accumulator changed. Need to update the GSSNode inside the Arc.
                // Arc::make_mut will clone the GSSNode if entry_arc is shared.
                let mut temp_arc = entry_arc.clone(); 
                let node_instance_mut = Arc::make_mut(&mut temp_arc);
                node_instance_mut.acc = new_potential_acc;
                *entry_arc = temp_arc.clone(); // Update cache with the (potentially new) Arc
                return temp_arc;
            }
            return entry_arc.clone(); // Return existing Arc if acc didn't change
        }

        // Node not in cache, create a new canonical one.
        let hash_key_cache = compute_internal_hash_key::<T, A>(&key_for_lookup_map);
        let new_node = GSSNode {
            acc: current_context_unioned_acc, // Use the just-computed unioned acc
            predecessors: key_for_lookup_map.clone(), // Store the processed map
            hash_key_cache,
        };
        let new_node_arc = Arc::new(new_node);
        cache.insert(key_for_lookup_map, new_node_arc.clone());
        new_node_arc
    }

    /// Creates a new canonical root GSSNode (no predecessors) with a specific initial accumulator.
    pub fn new_canonical(
        initial_acc: A, 
        cache: &mut HashMap<BTreeMap<T, Arc<GSSNode<T, A>>>, Arc<GSSNode<T, A>>> // Cache key is BTreeMap
    ) -> Arc<Self> {
        let predecessors_map_key = BTreeMap::new(); // Empty map for root node key
        
        if let Some(entry_arc) = cache.get_mut(&predecessors_map_key) {
            // Root node (by structure) exists. Union with initial_acc.
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

        // Create new canonical root node.
        let hash_key_cache = compute_internal_hash_key::<T, A>(&predecessors_map_key);
        let new_node_arc = Arc::new(GSSNode {
            acc: initial_acc, // Use provided initial_acc
            predecessors: predecessors_map_key.clone(), // Empty map
            hash_key_cache,
        });
        cache.insert(predecessors_map_key, new_node_arc.clone());
        new_node_arc
    }
}

// Public methods (mostly non-canonical)
impl<T: Ord + Hash + Clone, A: PathAccumulator + Clone> GSSNode<T, A> {
    pub fn push(self, edge_value: T) -> Self {
        // Create a BTreeSet representing the new predecessor link.
        let mut new_node_predecessors_with_values = BTreeSet::new();
        new_node_predecessors_with_values.insert((Arc::new(self), edge_value));
        // new_with_predecessors handles processing this set into BTreeMap.
        Self::new_with_predecessors(new_node_predecessors_with_values)
    }

    pub fn pop_into(&self, mut result: GSSNode<T, A>) -> GSSNode<T, A> {
        // Iterates over (Arc<Node>, T) pairs using the compatibility method.
        for (pred_arc, _edge_val) in self.predecessors_with_values() {
            result.merge(pred_arc.as_ref().clone());
        }
        result
    }

    pub fn pop(&self) -> GSSNode<T, A> {
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
            if current_node_ref.predecessors.is_empty() {
                path_suffix.reverse(); // Path built in reverse, correct it.
                result_paths.push(path_suffix);
            } else {
                // Iterate BTreeMap: (edge_val, pred_arc)
                for (edge_val, pred_arc) in &current_node_ref.predecessors {
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

    pub fn iter_paths(&self) -> PathsIter<'_, T, A> {
        let mut queue = VecDeque::new();
        queue.push_back((self, Vec::new())); // Start with self, empty path suffix
        PathsIter { queue }
    }

    pub fn merge(&mut self, other: Self) {
        self.acc = self.acc.union(&other.acc);
        
        for (other_edge_val, other_pred_arc) in other.predecessors {
            match self.predecessors.entry(other_edge_val) { // other_edge_val is cloned if vacant
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(other_pred_arc);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    // Key collision: merge the GSSNode instances pointed to by the Arcs.
                    // Arc::make_mut ensures we have a mutable reference, cloning GSSNode if necessary.
                    let self_pred_node_mut = Arc::make_mut(entry.get_mut());
                    // other_pred_arc is Arc<GSSNode>. Its GSSNode is *other_pred_arc or other_pred_arc.deref().
                    // GSSNode::merge takes other: Self (an owned GSSNode).
                    self_pred_node_mut.merge((*other_pred_arc).clone());
                }
            }
        }
        // Recompute hash key cache as predecessors might have changed structurally,
        // Arcs might have been replaced, or GSSNodes inside Arcs might have changed.
        self.hash_key_cache = compute_internal_hash_key::<T, A>(&self.predecessors);
    }

    pub fn merged(self, other: Self) -> Self {
        let mut merged_node = self.clone(); // Clone self to start
        merged_node.merge(other); // Merge other into it
        merged_node
    }

    pub fn map<F_edge, U_edge>(&self, f_edge: F_edge) -> GSSNode<U_edge, A>
    where
        F_edge: Copy + Fn(&T) -> U_edge,
        U_edge: Ord + Hash + Clone, // U_edge needs these for BTreeMap key and GSSNode reqs
        A: PathAccumulator + Clone, // A remains the same type
    {
        // Produces a BTreeSet<(Arc<GSSNode<U_edge, A>>, U_edge)>
        // This set will be processed by new_with_predecessors into BTreeMap.
        let new_predecessors_as_set: BTreeSet<(Arc<GSSNode<U_edge, A>>, U_edge)> =
            self.predecessors.iter() // Iterate current BTreeMap<T, Arc<GSSNode<T,A>>>
            .map(|(edge_val_t, pred_arc_t_a)| { // Item is (&T, &Arc<GSSNode<T,A>>)
                // Recursively map the predecessor node.
                let mapped_pred_arc = Arc::new(pred_arc_t_a.as_ref().map(f_edge));
                // Transform the current edge value.
                let new_edge_val_u = f_edge(edge_val_t);
                // Create tuple for the BTreeSet: (Arc<GSSNode<U,A>>, U_edge)
                (mapped_pred_arc, new_edge_val_u)
            })
            .collect();

        // Create new GSSNode of type GSSNode<U_edge, A> using the set.
        GSSNode::<U_edge, A>::new_with_predecessors(new_predecessors_as_set)
    }
}

impl<T, A: PathAccumulator> Drop for GSSNode<T, A> {
    fn drop(&mut self) {
        // Take ownership of the predecessor Arcs from the BTreeMap
        let predecessors_map = std::mem::take(&mut self.predecessors);
        // Collect Arcs from the map's values into the worklist
        let mut worklist: Vec<Arc<GSSNode<T, A>>> = predecessors_map
            .into_iter()
            .map(|(_edge_val, pred_arc)| pred_arc) // Extract Arcs
            .collect();

        // Iteratively try to break cycles / drop unreferenced nodes
        while let Some(node_arc) = worklist.pop() {
            // If this Arc is the last strong reference to the GSSNode
            if Arc::strong_count(&node_arc) == 1 {
                // Try to obtain mutable ownership of the GSSNode.
                // This succeeds if node_arc was indeed the last strong reference.
                if let Ok(mut inner_node) = Arc::try_unwrap(node_arc) {
                    // Take inner node's predecessors (which is a BTreeMap)
                    let inner_preds_map = std::mem::take(&mut inner_node.predecessors);
                    // Extend worklist with Arcs from this map's values
                    worklist.extend(inner_preds_map.into_iter().map(|(_, p_arc)| p_arc));
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

// T needs Ord + Hash for BTreeMap keys and GSSNode requirements.
// PartialEq for T and Arc<GSSNode> (which means GSSNode needs PartialEq).
impl<T: Ord + Hash + PartialEq, A: PathAccumulator + PartialEq> PartialEq for GSSNode<T, A> {
    fn eq(&self, other: &Self) -> bool {
        if std::ptr::eq(self, other) { return true; } // Same instance
        // hash_key_cache is a primary distinguisher for structure.
        // acc and predecessors (BTreeMap) must also match.
        // BTreeMap's PartialEq compares keys and then values. Arc<GSSNode> values are compared by pointer.
        // This is suitable for canonical GSS where identical structures should yield pointer-equal Arcs.
        self.hash_key_cache == other.hash_key_cache &&
        self.acc == other.acc &&
        self.predecessors == other.predecessors
    }
}

impl<T: Ord + Hash + Eq, A: PathAccumulator + Eq> Eq for GSSNode<T, A> {} // Marker trait

// T needs Ord + Hash for BTreeMap keys. PartialOrd for T and Arc<GSSNode>.
impl<T: Ord + Hash + PartialOrd, A: PathAccumulator + PartialOrd> PartialOrd for GSSNode<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if std::ptr::eq(self, other) { return Some(Ordering::Equal); }
        match self.hash_key_cache.partial_cmp(&other.hash_key_cache) {
            Some(Ordering::Equal) => {
                match self.acc.partial_cmp(&other.acc) {
                    Some(Ordering::Equal) => self.predecessors.partial_cmp(&other.predecessors),
                    other_ordering => other_ordering,
                }
            }
            other_ordering => other_ordering,
        }
    }
}

// T needs Ord + Hash. A needs Ord. Arc<GSSNode> needs Ord (pointer comparison).
impl<T: Ord + Hash, A: PathAccumulator + Ord> Ord for GSSNode<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        if std::ptr::eq(self, other) { return Ordering::Equal; }
        self.hash_key_cache.cmp(&other.hash_key_cache)
            .then_with(|| self.acc.cmp(&other.acc))
            .then_with(|| self.predecessors.cmp(&other.predecessors)) // BTreeMap cmp
    }
}

pub trait GSSTrait<T: Clone + Hash, A: PathAccumulator> {
    fn push(&self, edge_value: T) -> GSSNode<T, A> where T: Ord + Clone, A: Clone;
    fn push_to(&self, edge_value: T, dest: &mut GSSNode<T, A>) where T: Ord + Clone, A: Clone;
    fn pop(&self) -> GSSNode<T, A> where T: Ord + Clone, A: Clone;
    fn popn(&self, n: usize) -> GSSNode<T, A> where T: Ord + Clone, A: Clone;
}

impl<T: Clone + Ord + Hash, A: PathAccumulator + Clone> GSSTrait<T, A> for GSSNode<T, A> {
    fn push(&self, edge_value: T) -> GSSNode<T, A> {
        let self_owned_clone = self.clone(); // GSSNode::push takes self by value
        GSSNode::push(self_owned_clone, edge_value)
    }

    fn push_to(&self, edge_value: T, dest: &mut GSSNode<T, A>) {
        // This interprets "push_to" as making `self` a predecessor of `dest` via `edge_value`.
        // A new node representing this link is created and merged into `dest`.
        let pred_arc = Arc::new(self.clone());
        let new_link_set = BTreeSet::from([(pred_arc, edge_value)]);
        let node_representing_new_link = GSSNode::new_with_predecessors(new_link_set);
        // This node_representing_new_link will have `self` as its predecessor via `edge_value`,
        // and its accumulator will be `self.acc`.
        // Merging this into `dest` adds this path and updates `dest.acc`.
        dest.merge(node_representing_new_link);
    }

    fn pop(&self) -> GSSNode<T, A> {
        GSSNode::pop(self) // GSSNode::pop takes &self
    }

    fn popn(&self, n: usize) -> GSSNode<T, A> {
        GSSNode::popn(self, n) // GSSNode::popn takes &self
    }
}

impl<T: Clone + Ord + Hash, A: PathAccumulator + Clone> GSSTrait<T, A> for Arc<GSSNode<T, A>> {
    fn push(&self, edge_value: T) -> GSSNode<T, A> {
        let mut new_preds_with_values_set = BTreeSet::new();
        new_preds_with_values_set.insert((self.clone(), edge_value)); // self is Arc, clone it.
        GSSNode::new_with_predecessors(new_preds_with_values_set)
    }

    fn push_to(&self, _edge_value: T, dest: &mut GSSNode<T, A>) {
        // This matches the original peculiar implementation for Arc<GSSNode>,
        // which ignores edge_value and does a specific acc adjustment.
        dest.merge(self.as_ref().clone()); // Merge the GSSNode data from self (Arc)
        dest.acc = dest.acc.pop(&self.acc); // Adjust accumulator
    }

    fn pop(&self) -> GSSNode<T, A> {
        self.as_ref().pop()
    }

    fn popn(&self, n: usize) -> GSSNode<T, A> {
        self.as_ref().popn(n)
    }
}

impl<T: Clone + Ord + Hash, A: PathAccumulator + Clone + Default> GSSTrait<T, A> for Option<Arc<GSSNode<T, A>>> {
    fn push(&self, edge_value: T) -> GSSNode<T, A> {
        match self {
            Some(arc_node) => arc_node.push(edge_value), // Arc's push
            None => {
                // If None, push from a new default root.
                // The resulting node will have this new root as its sole predecessor via edge_value.
                let root_state_arc = Arc::new(GSSNode::new(A::default()));
                let mut new_preds_set = BTreeSet::new();
                new_preds_set.insert((root_state_arc, edge_value));
                GSSNode::new_with_predecessors(new_preds_set)
            }
        }
    }

    fn push_to(&self, edge_value: T, dest: &mut GSSNode<T, A>) {
        match self {
            Some(arc_node) => arc_node.push_to(edge_value, dest), // Arc's push_to
            None => {
                // Consistent with Arc's push_to: merge a default node and pop its acc.
                // edge_value is unused here, as in Arc's push_to.
                let default_node = GSSNode::new(A::default());
                dest.merge(default_node.clone()); 
                dest.acc = dest.acc.pop(&default_node.acc); 
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
    fn push(&self, edge_value: T) -> GSSNode<T, A> {
        match self {
            Some(node) => node.clone().push(edge_value), // GSSNode's push
            None => {
                let root_state = GSSNode::new(A::default());
                root_state.push(edge_value)
            }
        }
    }

    fn push_to(&self, edge_value: T, dest: &mut GSSNode<T, A>) {
        match self {
            Some(node) => node.push_to(edge_value, dest), // GSSNode's push_to
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

pub fn prune_and_transform_recursive_canonical<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    node_arc: &Arc<GSSNode<T, A>>,
    closure: &impl Fn(&A) -> Option<(A, bool)>,
    memo: &mut HashMap<*const GSSNode<T, A>, Option<Arc<GSSNode<T, A>>>>,
    cache: &mut HashMap<BTreeMap<T, Arc<GSSNode<T, A>>>, Arc<GSSNode<T, A>>>, // Cache key type changed
) -> Option<Arc<GSSNode<T, A>>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }

    match closure(&node_arc.acc) {
        None => { // Prune this node
            memo.insert(node_ptr, None);
            None
        }
        Some((new_acc_for_this_node_part, continue_recursion)) => {
            // This will be a BTreeSet<(Arc<ProcessedPred>, T_edge)> to pass to get_canonical.
            let new_predecessors_set_for_canonical: BTreeSet<(Arc<GSSNode<T, A>>, T)>; 
            if continue_recursion {
                let mut current_new_preds_set = BTreeSet::new();
                // Iterate over BTreeMap<T, Arc<GSSNode>> of current node_arc
                for (edge_val, pred_arc) in &node_arc.predecessors {
                    if let Some(new_pred_arc) = prune_and_transform_recursive_canonical(pred_arc, closure, memo, cache) {
                        // Collect (Arc<SimplifiedPred>, EdgeVal)
                        current_new_preds_set.insert((new_pred_arc, edge_val.clone()));
                    }
                }
                new_predecessors_set_for_canonical = current_new_preds_set;
            } else { // Don't recurse, use original predecessors structure (converted to BTreeSet)
                new_predecessors_set_for_canonical = node_arc.predecessors_with_values();
            };

            // GSSNode::get_canonical takes BTreeSet and processes it to form its internal BTreeMap structure
            // and uses that BTreeMap as cache key.
            let canonical_new_node_arc = GSSNode::get_canonical(new_predecessors_set_for_canonical, cache);

            // The accumulator of canonical_new_node_arc is already based on its (new) structure.
            // We need to union the specific accumulator part from the closure for *this* transformation context.
            let mut temp_arc = canonical_new_node_arc.clone(); 
            let node_instance_mut = Arc::make_mut(&mut temp_arc);
            node_instance_mut.acc = node_instance_mut.acc.union(&new_acc_for_this_node_part);
            
            memo.insert(node_ptr, Some(temp_arc.clone()));
            Some(temp_arc)
        }
    }
}

pub fn prune_and_transform_recursive<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    node_arc: &Arc<GSSNode<T, A>>,
    closure: &impl Fn(&A) -> Option<(A, bool)>,
    memo: &mut HashMap<*const GSSNode<T, A>, Option<Arc<GSSNode<T, A>>>>,
) -> Option<Arc<GSSNode<T, A>>> {
    // Cache for GSSNode::get_canonical uses BTreeMap as key type
    let mut cache = HashMap::<BTreeMap<T, Arc<GSSNode<T, A>>>, Arc<GSSNode<T, A>>>::new();
    prune_and_transform_recursive_canonical(node_arc, closure, memo, &mut cache)
}

fn find_longest_path_ending_at_node_recursive<T: Clone + Ord + Hash, A: PathAccumulator>(
    node_arc: &Arc<GSSNode<T, A>>,
    memo: &mut HashMap<*const GSSNode<T, A>, Vec<(T, Arc<GSSNode<T, A>>)>>,
    visited_recursion: &mut HashSet<*const GSSNode<T, A>>,
) -> Vec<(T, Arc<GSSNode<T, A>>)> {
    let node_ptr = Arc::as_ptr(node_arc);

    if let Some(cached_path) = memo.get(&node_ptr) {
        return cached_path.clone();
    }
    if !visited_recursion.insert(node_ptr) { // Cycle detected
        return Vec::new(); 
    }

    // Base case: Node has no predecessors.
    if node_arc.predecessors.is_empty() {
        visited_recursion.remove(&node_ptr);
        memo.insert(node_ptr, Vec::new());
        return Vec::new();
    }

    let mut longest_path_found: Vec<(T, Arc<GSSNode<T, A>>)> = Vec::new();

    // Iterate over BTreeMap: (&EdgeValue, &Arc<PredecessorNode>)
    for (edge_val_to_current_node, pred_arc) in &node_arc.predecessors {
        let path_from_pred_recursive = find_longest_path_ending_at_node_recursive(
            pred_arc,
            memo,
            visited_recursion,
        );

        // Construct candidate path: path to predecessor + (edge_to_current, current_node_arc)
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

pub fn find_longest_path<T: Clone + Ord + Hash, A: PathAccumulator>(root_node: &GSSNode<T, A>) -> Option<Vec<(T, Arc<GSSNode<T, A>>)>> {
    if root_node.predecessors.is_empty() { // No predecessors, so no path leading to it.
        return None; 
    }

    let mut memo: HashMap<*const GSSNode<T, A>, Vec<(T, Arc<GSSNode<T, A>>)>> = HashMap::new();
    let mut longest_overall_path: Option<Vec<(T, Arc<GSSNode<T, A>>)>> = None;

    // Iterate over direct predecessors of root_node (BTreeMap: (&EdgeValue, &Arc<PredecessorNode>)).
    // The path we seek ends at one of these predecessors.
    for (_edge_val_to_pred, pred_arc) in root_node.predecessors() {
        let mut visited_recursion = HashSet::new(); // Fresh for each DFS from a direct predecessor
        let path_ending_at_pred = find_longest_path_ending_at_node_recursive(pred_arc, &mut memo, &mut visited_recursion);

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
    pub max_predecessors_with_values: usize, // Field name kept for compatibility
    pub average_predecessors_with_values: f64, // Field name kept for compatibility
}

pub fn gather_gss_stats<T, A: PathAccumulator>(roots: &[impl AsRef<GSSNode<T, A>>]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut q_visited: HashSet<*const GSSNode<T, A>> = HashSet::new(); // Tracks node *pointers* added to queue
    let mut processed_nodes: HashSet<*const GSSNode<T, A>> = HashSet::new(); // Tracks node *pointers* fully processed
    let mut queue: VecDeque<(&GSSNode<T, A>, usize)> = VecDeque::new(); // Stores (&Node, depth)

    let mut total_depth_sum: u64 = 0;
    let mut total_preds_sum: u64 = 0;

    for root_as_ref in roots {
        let root_node_ref = root_as_ref.as_ref();
        // Use raw pointer for HashSet keys
        if q_visited.insert(root_node_ref as *const GSSNode<T,A>) {
            queue.push_back((root_node_ref, 0));
        }
    }

    while let Some((current_node_ref, current_depth)) = queue.pop_front() {
        if !processed_nodes.insert(current_node_ref as *const GSSNode<T,A>) {
            continue; // Already processed this node
        }

        stats.unique_nodes += 1;
        stats.max_depth = stats.max_depth.max(current_depth);
        total_depth_sum += current_depth as u64;

        let num_preds = current_node_ref.predecessors.len(); // Use .predecessors (BTreeMap)
        stats.max_predecessors_with_values = stats.max_predecessors_with_values.max(num_preds);
        total_preds_sum += num_preds as u64;

        // A merge point has multiple distinct predecessor nodes.
        let unique_pred_node_ptrs: HashSet<*const GSSNode<T,A>> = current_node_ref.predecessors.values()
            .map(|p_arc| Arc::as_ptr(p_arc)).collect();
        if unique_pred_node_ptrs.len() > 1 {
            stats.merge_points += 1;
        }

        // Iterate BTreeMap: (&EdgeValue, &Arc<PredecessorNode>)
        for (_edge_val, pred_arc) in &current_node_ref.predecessors {
            let pred_node_ref = pred_arc.as_ref();
            if q_visited.insert(pred_node_ref as *const GSSNode<T,A>) {
                queue.push_back((pred_node_ref, current_depth + 1));
            }
        }
    }

    if stats.unique_nodes > 0 {
        stats.average_depth = total_depth_sum as f64 / stats.unique_nodes as f64;
        stats.average_predecessors_with_values = total_preds_sum as f64 / stats.unique_nodes as f64;
    }
    stats
}

fn print_gss_node_recursive<T: Debug, A: PathAccumulator>(
    node_arc: &Arc<GSSNode<T, A>>,
    visited: &mut HashSet<*const GSSNode<T, A>>, // Stores raw pointers
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

    if !node_arc.predecessors.is_empty() {
        writeln!(output, "{}  Predecessors (Edge Value -> Pred Node Ptr):", prefix)?;
        // Iterate BTreeMap: (&EdgeValue, &Arc<PredecessorNode>)
        for (edge_val, pred_arc) in &node_arc.predecessors {
            writeln!(output, "{}    - Edge: {:?} -> Pred_Node: {:p}", prefix, edge_val, Arc::as_ptr(pred_arc))?;
            if *node_count < max_nodes_to_print { 
                 print_gss_node_recursive(pred_arc, visited, indent + 2, node_count, max_nodes_to_print, output)?;
            }
            if *node_count >= max_nodes_to_print { 
                return Ok(()); // Stop further iteration if limit reached in recursion
            }
        }
    }
    Ok(())
}

pub fn print_gss_forest<T: Debug, A: PathAccumulator>(roots: &[Arc<GSSNode<T, A>>], max_nodes_to_print: usize) -> String {
    let mut visited = HashSet::new(); // Stores raw pointers
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
                if node_count >= max_nodes_to_print && i < roots.len() -1 { // If truncated and more roots exist
                    writeln!(&mut output, "... (Truncated: Reached max nodes {})", max_nodes_to_print).unwrap();
                    break; 
                }
            }
            Err(e) => {
                // eprintln is okay for auxiliary error info, but function should return error string.
                return format!("Error writing GSS structure to string: {}", e);
            }
        }
    }
    // Check if total unique nodes visited (even if not fully printed) exceeds count due to truncation.
    if node_count >= max_nodes_to_print && visited.len() > node_count {
         writeln!(&mut output, "... (More nodes exist but not printed due to max_nodes_to_print limit)").unwrap();
    }
    output
}

// T, A need full constraints for canonicalization and simplification.
fn simplify_node_recursive<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    original_node_arc: &Arc<GSSNode<T, A>>,
    memo: &mut HashMap<*const GSSNode<T, A>, Arc<GSSNode<T, A>>>, // Memoizes original_ptr -> simplified_Arc
    canonicalization_cache: &mut HashMap<BTreeMap<T, Arc<GSSNode<T, A>>>, Arc<GSSNode<T, A>>>, // Cache for get_canonical
) -> Arc<GSSNode<T, A>> {
    let original_node_ptr = Arc::as_ptr(original_node_arc);
    if let Some(simplified_arc) = memo.get(&original_node_ptr) {
        return simplified_arc.clone();
    }

    // Recursively simplify predecessors and collect them for GSSNode::get_canonical.
    // get_canonical expects a BTreeSet<(Arc<SimplifiedPred>, EdgeVal)>.
    let mut simplified_predecessors_as_set: BTreeSet<(Arc<GSSNode<T, A>>, T)> = BTreeSet::new();
    // Iterate original node's BTreeMap: (&EdgeVal, &Arc<OriginalPred>)
    for (edge_val, original_pred_arc) in &original_node_arc.predecessors {
        let simplified_pred_arc = simplify_node_recursive(
            original_pred_arc,
            memo,
            canonicalization_cache,
        );
        simplified_predecessors_as_set.insert((simplified_pred_arc, edge_val.clone()));
    }
    
    // The `predecessors_grouped` logic from the original `simplify_node_recursive` is effectively
    // handled by `GSSNode::get_canonical`'s internal call to `process_incoming_predecessors`
    // when it converts the `simplified_predecessors_as_set` into its BTreeMap cache key and internal structure.

    let canonical_arc = GSSNode::get_canonical(
        simplified_predecessors_as_set, // Pass the BTreeSet of (SimplifiedPredArc, EdgeVal)
        canonicalization_cache,
    );

    // The accumulator of `original_node_arc` needs to be unioned into the `canonical_arc`.
    // `GSSNode::get_canonical` sets acc based on the structure it's given.
    // We union `original_node_arc.acc` to preserve its specific accumulator state.
    let mut temp_arc = canonical_arc.clone(); // Clone Arc for make_mut
    let node_instance_mut = Arc::make_mut(&mut temp_arc);
    node_instance_mut.acc = node_instance_mut.acc.union(&original_node_arc.acc);

    memo.insert(original_node_ptr, temp_arc.clone());
    temp_arc
}

fn simplify_gss_forest<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T, A>>],
) -> Vec<Arc<GSSNode<T, A>>> {
    let mut memo: HashMap<*const GSSNode<T, A>, Arc<GSSNode<T, A>>> = HashMap::new();
    // Cache for GSSNode::get_canonical (used by simplify_node_recursive)
    let mut canonicalization_cache = HashMap::<BTreeMap<T, Arc<GSSNode<T, A>>>, Arc<GSSNode<T, A>>>::new();
    let mut simplified_roots_vec: Vec<Arc<GSSNode<T,A>>> = Vec::with_capacity(roots.len());

    for root_arc in roots {
        simplified_roots_vec.push(simplify_node_recursive(
            root_arc, // Pass &Arc<GSSNode>
            &mut memo,
            &mut canonicalization_cache,
        ));
    }
    
    // Deduplicate root Arcs: if multiple original roots simplify to the same canonical GSSNode structure,
    // ensure they all point to the *same* Arc instance in the returned Vec.
    let mut unique_simplified_roots_map: HashMap<*const GSSNode<T,A>, Arc<GSSNode<T,A>>> = HashMap::new();
    for r_arc_mut_ref in simplified_roots_vec.iter_mut() { // r_arc_mut_ref is &mut Arc<GSSNode<T,A>>
        // Arc::as_ptr(r_arc_mut_ref) gives the pointer to the GSSNode data.
        // Use this pointer to check if we've already stored an Arc for this GSSNode.
        let entry = unique_simplified_roots_map.entry(Arc::as_ptr(r_arc_mut_ref)).or_insert_with(|| r_arc_mut_ref.clone());
        // Update the Arc in simplified_roots_vec to be the one from the map.
        *r_arc_mut_ref = entry.clone(); 
    }
    // The number of roots should remain the same.
    assert_eq!(roots.len(), simplified_roots_vec.len());
    simplified_roots_vec
}

// T, A need full constraints for simplification.
impl<T: Ord + Hash + Clone + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug> GSSNode<T, A> {
    pub fn simplify(&mut self) {
        // To simplify `self` (a GSSNode), we need an Arc to it to pass to recursive functions.
        // Clone `self`, wrap in Arc, simplify, then replace `self`'s content.
        let self_clone_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        let mut canonical_cache = HashMap::new(); // BTreeMap key
        let simplified_arc = simplify_node_recursive(&self_clone_arc, &mut memo, &mut canonical_cache);
        
        // Replace self's content with the GSSNode data from the simplified_arc.
        *self = (*simplified_arc).clone(); 
    }

    pub fn simplify_recursive(
        this_arc: &mut Arc<GSSNode<T, A>>, // Input is a mutable reference to an Arc
        memo: &mut HashMap<*const GSSNode<T,A>, Arc<GSSNode<T,A>>>,
        canonicalization_cache: &mut HashMap<BTreeMap<T, Arc<GSSNode<T, A>>>, Arc<GSSNode<T, A>>>,
    ) {
        // simplify_node_recursive takes &Arc, returns simplified Arc.
        // Update the Arc pointed to by this_arc.
        *this_arc = simplify_node_recursive(this_arc, memo, canonicalization_cache);
    }

    pub fn simplify_together(nodes: &mut [&mut Arc<GSSNode<T, A>>]) {
        let mut memo: HashMap<*const GSSNode<T, A>, Arc<GSSNode<T, A>>> = HashMap::new();
        let mut canonicalization_cache = HashMap::<BTreeMap<T, Arc<GSSNode<T, A>>>, Arc<GSSNode<T, A>>>::new();
        for node_arc_mut_ref in nodes { // node_arc_mut_ref is &mut Arc<GSSNode<T,A>>
            // Pass the Arc itself (dereferenced from &mut Arc) to simplify_node_recursive.
            // Update the Arc through the mutable reference.
            **node_arc_mut_ref = simplify_node_recursive(*node_arc_mut_ref, &mut memo, &mut canonicalization_cache);
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
    // Cache key for canonical nodes is now BTreeMap<T, Arc<Node>>
    type MockNodeCache = HashMap<BTreeMap<i32, Arc<MockGSSNode>>, Arc<MockGSSNode>>;

    // nc_node_arc creates a non-canonical node.
    // Its input `preds_with_vals_vec` is Vec<(Arc<MockGSSNode>, i32)>.
    // This is converted to BTreeSet for new_with_predecessors.
    fn nc_node_arc(
        _initial_acc_for_new_node: MockPathAccumulator, // This param is effectively ignored if acc is always derived.
        preds_with_vals_vec: Vec<(Arc<MockGSSNode>, i32)>
    ) -> Arc<MockGSSNode> {
        let pred_set: BTreeSet<(Arc<MockGSSNode>, i32)> = preds_with_vals_vec.into_iter().collect();
        // new_with_predecessors derives accumulator from predecessors.
        // If _initial_acc_for_new_node was meant to augment, it would need explicit unioning here.
        // Assuming derived accumulator is intended for this helper.
        let node = MockGSSNode::new_with_predecessors(pred_set);
        Arc::new(node)
    }

    fn nc_root_node_arc(acc: MockPathAccumulator) -> Arc<MockGSSNode> {
         Arc::new(MockGSSNode::new(acc)) // new() sets acc directly.
    }

    // c_root_node_arc creates a canonical root node.
    fn c_root_node_arc(acc: MockPathAccumulator, cache: &mut MockNodeCache) -> Arc<MockGSSNode> {
        MockGSSNode::new_canonical(acc, cache)
    }

    fn collect_arcs_recursive(
        node_arc: &Arc<MockGSSNode>,
        collected_arcs: &mut HashMap<*const MockGSSNode, Arc<MockGSSNode>>, // Map raw_ptr to Arc
    ) {
        let ptr = Arc::as_ptr(node_arc);
        if collected_arcs.contains_key(&ptr) {
            return;
        }
        collected_arcs.insert(ptr, node_arc.clone());
        // Iterate BTreeMap: (&EdgeVal, &Arc<PredNode>)
        for (_edge_val, pred_arc) in &node_arc.predecessors { 
            collect_arcs_recursive(pred_arc, collected_arcs);
        }
    }

    #[test]
    fn test_gss_simplification_basic() {
        let acc_base = MockPathAccumulator { active: BTreeSet::from([0]), intersection: BTreeSet::from([0]) };
        let acc_other = MockPathAccumulator { active: BTreeSet::from([1]), intersection: BTreeSet::from([1]) };

        // Non-canonical graph construction
        let n4_base_nc = nc_root_node_arc(acc_base.clone()); 
        let d1_orig_nc = nc_node_arc(MockPathAccumulator::default(), vec![(n4_base_nc.clone(), 40)]);

        let n4_other_nc = nc_root_node_arc(acc_other.clone()); 
        let d2_orig_nc = nc_node_arc(MockPathAccumulator::default(), vec![(n4_other_nc.clone(), 40)]);
        
        let c1_orig_nc = nc_node_arc(MockPathAccumulator::default(), vec![(d1_orig_nc.clone(), 30)]);
        let b1_orig_nc = nc_node_arc(MockPathAccumulator::default(), vec![(c1_orig_nc.clone(), 20)]);

        // a1_orig_nc will have two (Arc, EdgeVal=10) items in the BTreeSet passed to new_with_predecessors.
        // process_incoming_predecessors will merge the GSSNodes pointed to by b1_orig_nc and d2_orig_nc
        // because they share the same edge value 10.
        // So, a1_orig_nc.predecessors will be {10: Arc<merged_b1_d2_node>}.
        let a1_preds_nc = vec![
            (b1_orig_nc.clone(), 10), 
            (d2_orig_nc.clone(), 10)  
        ];
        let a1_orig_nc = nc_node_arc(MockPathAccumulator::default(), a1_preds_nc);

        let roots_nc = vec![a1_orig_nc.clone()];
        println!("Before simplifying GSS forest (non-canonical, BTreeMap based): {}", print_gss_forest(&roots_nc, usize::MAX));
        
        let simplified_roots = simplify_gss_forest(&roots_nc);
        println!("After simplifying GSS forest: {}", print_gss_forest(&simplified_roots, usize::MAX));
        
        assert_eq!(simplified_roots.len(), 1);
        let s_a1 = simplified_roots[0].clone();

        // Expected structure after simplification with BTreeMap and merging:
        // s_a1 --(10)--> s_pred_of_a1 (this is a canonical node representing merged b1 and d2)
        //   s_pred_of_a1.predecessors = {20: s_c1, 40: s_n4_other} (from b1's pred and d2's pred)
        //     s_c1 --(30)--> s_d1 --(40)--> s_n4_base (root)
        //     s_n4_other (root)
        // Unique canonical nodes: s_a1, s_pred_of_a1, s_c1, s_d1, s_n4_base, s_n4_other. Total 6 nodes.
        
        let mut collected_arcs = HashMap::new();
        collect_arcs_recursive(&s_a1, &mut collected_arcs);
        assert_eq!(collected_arcs.len(), 6, "Expected 6 unique Arcs after simplification with BTreeMap structure.");

        // s_a1 has one predecessor entry (for edge 10), pointing to the merged node.
        assert_eq!(s_a1.predecessors.len(), 1); 
        let (edge_val_to_pred, s_merged_pred_arc) = s_a1.predecessors.iter().next().unwrap();
        assert_eq!(*edge_val_to_pred, 10);

        // Accumulator checks:
        // s_a1.acc should be acc of s_merged_pred_arc (due to simplify_node_recursive logic and get_canonical).
        // s_merged_pred_arc.acc should be union of original b1.acc and d2.acc,
        // plus acc from its own simplified structure.
        // Original b1.acc (derived from n4_base.acc = acc_base).
        // Original d2.acc (derived from n4_other.acc = acc_other).
        // The simplify_node_recursive ensures original node's acc is unioned into canonical form.
        // So, s_merged_pred_arc.acc should contain acc_base.union(acc_other).
        // And s_a1.acc should also contain this.
        let expected_merged_acc_component = acc_base.union(&acc_other);
        // Check if s_merged_pred_arc.acc contains this (it might be unioned with more)
        // For this test, it should be exactly this due to how nc_node_arc derives acc.
        assert_eq!(s_merged_pred_arc.acc, expected_merged_acc_component);
        assert_eq!(s_a1.acc, expected_merged_acc_component); 

        // Structure of s_merged_pred_arc:
        // Predecessors are { (s_c1, edge 20), (s_n4_other, edge 40) }
        assert_eq!(s_merged_pred_arc.predecessors.len(), 2);

        let s_c1_arc = s_merged_pred_arc.predecessors.get(&20).expect("Simplified C1 node not found as pred of merged node via edge 20");
        let s_n4_other_arc_as_pred = s_merged_pred_arc.predecessors.get(&40).expect("Simplified N4_other node not found as pred of merged node via edge 40");
        
        // s_c1 path leads to n4_base, so its acc should be acc_base.
        assert_eq!(s_c1_arc.acc, acc_base); 
        // s_n4_other_arc_as_pred is the canonical root node s_n4_other, its acc is acc_other.
        assert_eq!(s_n4_other_arc_as_pred.acc, acc_other);

        // Trace s_c1 path:
        assert_eq!(s_c1_arc.predecessors.len(), 1); // s_c1 --(30)--> s_d1
        let (edge_val_to_d1, s_d1_arc) = s_c1_arc.predecessors.iter().next().unwrap();
        assert_eq!(*edge_val_to_d1, 30);
        assert_eq!(s_d1_arc.acc, acc_base);

        assert_eq!(s_d1_arc.predecessors.len(), 1); // s_d1 --(40)--> s_n4_base
        let (edge_val_to_n4_base, s_n4_base_arc) = s_d1_arc.predecessors.iter().next().unwrap();
        assert_eq!(*edge_val_to_n4_base, 40);
        assert_eq!(s_n4_base_arc.acc, acc_base);
        assert!(s_n4_base_arc.predecessors.is_empty()); // s_n4_base is a root

        // Trace s_n4_other_arc_as_pred path (it's a root):
        assert!(s_n4_other_arc_as_pred.predecessors.is_empty());

        // Ensure distinct root nodes were preserved and are indeed different Arcs:
        assert_ne!(Arc::as_ptr(s_n4_base_arc), Arc::as_ptr(s_n4_other_arc_as_pred));
        // s_d1_arc and s_n4_other_arc_as_pred are structurally different.
        assert_ne!(Arc::as_ptr(s_d1_arc), Arc::as_ptr(s_n4_other_arc_as_pred));
    }
}

