use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;

// Type alias for the canonicalization cache key
type NodeCacheKey<T> = (T, BTreeSet<Arc<GSSNode<T>>>);
// Type alias for the canonicalization cache
pub type NodeCache<T> = HashMap<NodeCacheKey<T>, Arc<GSSNode<T>>>;

// Helper function to compute a node's hash.
// T must be Hash for value.hash(), predecessors Arcs must point to GSSNodes with valid hash_key_cache.
fn compute_node_hash<T: Hash>(value: &T, predecessors: &BTreeSet<Arc<GSSNode<T>>>) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    // The BTreeSet ensures predecessors are iterated in a canonical order (by Arc pointer address).
    for pred_arc in predecessors {
        pred_arc.hash_key_cache.hash(&mut hasher);
    }
    hasher.finish()
}

#[derive(Debug, Clone)]
pub struct GSSNode<T> {
    pub value: T,
    predecessors: BTreeSet<Arc<GSSNode<T>>>,
    hash_key_cache: u64,
}

impl<T: Clone + Ord + Hash + Debug> GSSNode<T> {
    pub fn get_canonical(
        value: T,
        predecessors: BTreeSet<Arc<Self>>,
        cache: &mut NodeCache<T>,
    ) -> Arc<Self> {
        let key = (value, predecessors); // value and predecessors are moved into key

        if let Some(existing_node) = cache.get(&key) {
            return existing_node.clone();
        }

        let node_value_for_struct = key.0.clone();
        let node_predecessors_for_struct = key.1.clone();

        let hash_key_cache = compute_node_hash(&node_value_for_struct, &node_predecessors_for_struct);

        let new_node_arc = Arc::new(GSSNode {
            value: node_value_for_struct,
            predecessors: node_predecessors_for_struct,
            hash_key_cache,
        });

        cache.insert(key, new_node_arc.clone());
        new_node_arc
    }

    pub fn new_with_predecessors(value: T, predecessors: BTreeSet<Arc<Self>>) -> Self {
        let hash_key_cache = compute_node_hash(&value, &predecessors);
        Self {
            value,
            predecessors,
            hash_key_cache,
        }
    }

    pub fn new_empty(value: T) -> Self {
        Self::new_with_predecessors(value, BTreeSet::new())
    }

    pub fn new_with_predecessors_canonical(value: T, predecessors: BTreeSet<Arc<Self>>, cache: &mut NodeCache<T>) -> Arc<Self> {
        Self::get_canonical(value, predecessors, cache)
    }

    pub fn new_empty_canonical(value: T, cache: &mut NodeCache<T>) -> Arc<Self> {
        Self::get_canonical(value, BTreeSet::new(), cache)
    }

    pub fn from_iter<I>(iter: I, cache: &mut NodeCache<T>) -> Arc<Self>
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        let first_val = iter.next().expect("from_iter requires at least one element");
        let mut root = Self::new_empty_canonical(first_val, cache);
        for value in iter {
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

    pub fn pop(self_arc: &Arc<Self>) -> Vec<Arc<Self>> {
        self_arc.predecessors.iter().cloned().collect()
    }

    pub fn peek(self_arc: &Arc<Self>) -> &T {
        &self_arc.value
    }

    // value_mut is tricky with Arc and canonicalization.
    // If you mutate a value, it might break canonicalization unless the node is re-interned.
    // Generally, GSSNodes should be treated as immutable after creation/interning.
    // pub fn value_mut(&mut self) -> &mut T {
    //     &mut self.value
    // }

    pub fn flatten(self_arc: Arc<Self>) -> Vec<Vec<T>>
    where
        T: Clone, // T: Clone + Ord + Hash + Debug from impl block
    {
        let mut result = Vec::new();
        let mut stack: Vec<(Arc<GSSNode<T>>, Vec<T>)> = Vec::new();
        stack.push((self_arc, Vec::new()));
        while let Some((node_arc, mut path)) = stack.pop() {
            path.push(node_arc.value.clone());
            if node_arc.predecessors.is_empty() {
                result.push(path);
            } else {
                for pred_arc in &node_arc.predecessors {
                    stack.push((pred_arc.clone(), path.clone()));
                }
            }
        }
        result
    }

    pub fn flatten_bulk(nodes: &[Arc<Self>]) -> Vec<Vec<T>>
    where
        T: Clone, // T: Clone + Ord + Hash + Debug from impl block
    {
        nodes.iter().flat_map(|arc_node| Self::flatten(arc_node.clone())).collect()
    }

    pub fn merge_canonical(
        node1_arc: Arc<Self>,
        node2_arc: Arc<Self>,
        cache: &mut NodeCache<T>,
    ) -> Result<Arc<Self>, &'static str>
    where
        T: PartialEq, // T: Clone + Ord + Hash + Debug from impl block
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

    pub fn merge_unchecked(node1_arc: Arc<Self>, node2_arc: Arc<Self>) -> Arc<Self> {
        let mut merged_predecessors = node1_arc.predecessors.clone();
        for pred_arc in &node2_arc.predecessors {
            merged_predecessors.insert(pred_arc.clone());
        }
        node1_arc.value.clone()
    }

    pub fn map_canonical<F, U>(
        self_arc: Arc<Self>,
        f: F,
        cache_u: &mut NodeCache<U>,
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

impl<T> Drop for GSSNode<T> {
    fn drop(&mut self) {
        let predecessors_to_process_further = std::mem::take(&mut self.predecessors);
        let mut worklist: Vec<Arc<GSSNode<T>>> = predecessors_to_process_further.into_iter().collect();

        while let Some(node_arc) = worklist.pop() {
            if let Ok(mut inner_node) = Arc::try_unwrap(node_arc) {
                worklist.extend(std::mem::take(&mut inner_node.predecessors).into_iter());
            }
        }
    }
}

impl<T: Hash> Hash for GSSNode<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_key_cache.hash(state);
    }
}

impl<T: Eq + Hash> Eq for GSSNode<T> {} // T: Ord implies T: Eq

impl<T: PartialEq + Hash> PartialEq for GSSNode<T> {
    fn eq(&self, other: &Self) -> bool {
        // For canonical nodes, pointer equality implies structural equality.
        // However, this PartialEq is for GSSNode<T> content.
        // If hash_key_cache is different, they are different.
        // If T: Eq, then value equality is definitive.
        // The hash_key_cache should be a strong distinguisher.
        if self.hash_key_cache != other.hash_key_cache {
            return false;
        }
        // If hashes are same, values must be same for equality.
        if self.value != other.value {
            return false;
        }
        // If hashes and values are same, predecessors must be same.
        // BTreeSet<Arc<GSSNode<T>>> compares based on Arc pointers.
        // This is correct if both self and other are part of a canonicalized graph.
        self.predecessors == other.predecessors
    }
}

impl<T: PartialOrd + Hash> PartialOrd for GSSNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
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

impl<T: Ord + Hash> Ord for GSSNode<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.hash_key_cache.cmp(&other.hash_key_cache)
            .then_with(|| self.value.cmp(&other.value))
            .then_with(|| self.predecessors.cmp(&other.predecessors))
    }
}


pub trait GSSTrait<T: Clone + Ord + Hash + Debug> {
    type Peek<'a> where T: 'a, Self: 'a;
    fn peek(&self) -> Self::Peek<'_>;
    fn push(&self, value: T) -> GSSNode<T>;
    fn push_canonical(&self, value: T, cache: &mut NodeCache<T>) -> Arc<GSSNode<T>>;
    fn pop(&self) -> Vec<Arc<GSSNode<T>>>;
    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>>;
}

impl<T: Clone + Ord + Hash + Debug> GSSTrait<T> for Arc<GSSNode<T>> {
    type Peek<'a> = &'a T where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        &self.value // Or GSSNode::peek(self)
    }

    fn push(&self, value: T) -> GSSNode<T> {
        Arc::new(GSSNode::new_with_predecessors(value, BTreeSet::from([self.clone()])))
    }

    fn push_canonical(&self, value: T, cache: &mut NodeCache<T>) -> Arc<GSSNode<T>> {
        GSSNode::new_with_predecessors_canonical(value, BTreeSet::from([self.clone()]), cache)
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.predecessors.iter().cloned().collect() // Or GSSNode::pop(self)
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        if n == 0 {
            return vec![self.clone()];
        }
        let mut result = Vec::new();
        let mut seen_arcs_for_this_call: HashSet<*const GSSNode<T>> = HashSet::new();

        for predecessor_arc in &self.predecessors {
            for node_arc_from_popn in predecessor_arc.popn(n - 1) { // Recursive call on Arc
                let ptr = Arc::as_ptr(&node_arc_from_popn);
                if seen_arcs_for_this_call.insert(ptr) {
                    result.push(node_arc_from_popn);
                }
            }
        }
        result
    }
}

impl<T: Clone + Ord + Hash + Debug> GSSTrait<T> for Option<Arc<GSSNode<T>>> {
    type Peek<'a> = Option<&'a T> where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        self.as_ref().map(|node_arc| node_arc.peek())
    }

    fn push(&self, value: T) -> GSSNode<T> {
        match self {
            Some(arc_node) => arc_node.push(value),
            None => GSSNode::new_empty(value),
        }
    }

    fn push_canonical(&self, value: T, cache: &mut NodeCache<T>) -> Arc<GSSNode<T>> {
        match self {
            Some(arc_node) => arc_node.push_canonical(value, cache),
            None => GSSNode::new_empty_canonical(value, cache),
        }
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node_arc| node_arc.pop()).unwrap_or_default()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node_arc| node_arc.popn(n)).unwrap_or_default()
    }
}


pub trait BulkMerge<T: Clone + Ord + Hash + Debug> {
    fn bulk_merge(&mut self);
    fn bulk_merge_canonical(&mut self, cache: &mut NodeCache<T>);
}

impl<T: Clone + Ord + Hash + Debug> BulkMerge<T> for Vec<Arc<GSSNode<T>>> {
    fn bulk_merge(&mut self) {
        self.bulk_merge_canonical(&mut NodeCache::new());
    }

    fn bulk_merge_canonical(&mut self, cache: &mut NodeCache<T>) {
        let mut groups_by_value: BTreeMap<T, Vec<Arc<GSSNode<T>>>> = BTreeMap::new();
        for node_arc in self.drain(..) {
            groups_by_value.entry(node_arc.value.clone()).or_default().push(node_arc);
        }

        let mut new_merged_nodes = Vec::new();
        for (value, group_arcs) in groups_by_value {
            if group_arcs.is_empty() { continue; }

            let mut merged_predecessors: BTreeSet<Arc<GSSNode<T>>> = BTreeSet::new();
            for node_arc_in_group in group_arcs {
                for pred_arc in &node_arc_in_group.predecessors {
                    merged_predecessors.insert(pred_arc.clone());
                }
            }
            let merged_node = GSSNode::get_canonical(value, merged_predecessors, cache);
            new_merged_nodes.push(merged_node);
        }
        *self = new_merged_nodes;
    }
}

pub fn prune_and_transform_recursive<T: Clone + Ord + Hash + Debug>(
    node_arc: &Arc<GSSNode<T>>,
    closure: &impl Fn(&T) -> Option<(T, bool)>,
    memo: &mut HashMap<*const GSSNode<T>, Option<Arc<GSSNode<T>>>>,
    cache: &mut NodeCache<T>,
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
                for pred_arc in &node_arc.predecessors {
                    if let Some(new_pred_arc) = prune_and_transform_recursive(pred_arc, closure, memo, cache) {
                        current_new_predecessors.insert(new_pred_arc);
                    }
                }
                new_predecessors = current_new_predecessors;
            } else {
                new_predecessors = node_arc.predecessors.clone();
            };
            let new_node_arc = GSSNode::get_canonical(new_value, new_predecessors, cache);
            memo.insert(node_ptr, Some(new_node_arc.clone()));
            Some(new_node_arc)
        }
    }
}

pub fn prune_and_transform_roots<T: Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T>>],
    closure: &impl Fn(&T) -> Option<(T, bool)>,
) -> Vec<Option<Arc<GSSNode<T>>>> {
    let mut memo = HashMap::new();
    let mut cache = NodeCache::new();
    roots
        .iter()
        .map(|root| prune_and_transform_recursive(root, closure, &mut memo, &mut cache))
        .collect()
}

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
        for pred_arc in &node_arc.predecessors {
            let pred_path = find_longest_path_recursive(pred_arc, memo, visited_recursion);
            if !pred_path.is_empty() && pred_path.len() > longest_pred_path.len() {
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
    pub average_predecessors: f64, // Corrected f664 to f64
}

pub fn gather_gss_stats<T>(roots: &[Arc<GSSNode<T>>]) -> GSSStats { // T: Clone removed, not needed
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

        for pred_arc in &current_node_arc.predecessors {
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
        for pred_arc in &node_arc.predecessors {
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
                if node_count >= max_nodes {
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

    if node_count < max_nodes && node_count > visited.len() {
         writeln!(&mut output, "... (Truncated: Reached max nodes {})", max_nodes).unwrap();
    }
    output
}

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
    for original_pred_arc in &original_node_arc.predecessors {
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
    use std::sync::Arc;
    use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
    use std::fmt::Debug;


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
    type MockNodeCache = NodeCache<MockParseStateNodeContent>;
    type IntNodeCache = NodeCache<i32>;


    fn node_canonical_mock(
        value: MockParseStateNodeContent,
        predecessors: Vec<Arc<MockGSSNode>>,
        cache: &mut MockNodeCache,
    ) -> Arc<MockGSSNode> {
        let pred_set: BTreeSet<Arc<MockGSSNode>> = predecessors.into_iter().collect();
        GSSNode::get_canonical(value, pred_set, cache)
    }

    fn node_canonical_i32(
        value: i32,
        predecessors: Vec<Arc<GSSNode<i32>>>,
        cache: &mut IntNodeCache,
    ) -> Arc<GSSNode<i32>> {
        let pred_set: BTreeSet<Arc<GSSNode<i32>>> = predecessors.into_iter().collect();
        GSSNode::get_canonical(value, pred_set, cache)
    }


    type SimplifiedNodeRepr<T> = (T, Vec<u64>);

    fn get_simplified_repr<T: Clone + Hash>(node_arc: &Arc<GSSNode<T>>) -> SimplifiedNodeRepr<T> {
        let mut pred_hashes: Vec<u64> = node_arc.predecessors.iter()
            .map(|p_arc| p_arc.hash_key_cache)
            .collect();
        pred_hashes.sort_unstable();
        (node_arc.value.clone(), pred_hashes)
    }

    fn collect_all_simplified_nodes_repr<T: Clone + Hash>(
        node_arc: &Arc<GSSNode<T>>,
        visited: &mut HashSet<*const GSSNode<T>>,
        collected_nodes: &mut HashMap<*const GSSNode<T>, SimplifiedNodeRepr<T>>,
    ) {
        let ptr = Arc::as_ptr(node_arc);
        if !visited.insert(ptr) {
            return;
        }
        collected_nodes.insert(ptr, get_simplified_repr(node_arc));
        for pred_arc in &node_arc.predecessors {
            collect_all_simplified_nodes_repr(pred_arc, visited, collected_nodes);
        }
    }

    fn collect_arcs_recursive<T>( // T no longer needs bounds here
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
        let mut cache: IntNodeCache = HashMap::new();

        // D1
        // |
        // C1
        // |
        // B1   D2
        // |   /
        // A1 (preds: B1, D2)
        let d1_orig = node_canonical_i32(40, vec![], &mut cache);
        let c1_orig = node_canonical_i32(30, vec![d1_orig.clone()], &mut cache);
        let b1_orig = node_canonical_i32(20, vec![c1_orig.clone()], &mut cache);

        // d2_orig will be canonicalized to d1_orig if created with the same cache
        let d2_orig = node_canonical_i32(40, vec![], &mut cache);
        assert!(Arc::ptr_eq(&d1_orig, &d2_orig), "d1 and d2 should be the same Arc due to canonicalization");

        let a1_orig = node_canonical_i32(10, vec![b1_orig.clone(), d2_orig.clone()], &mut cache);

        // Since nodes are canonicalized on creation, simplification of roots built this way is idempotent.
        let roots = vec![a1_orig.clone()];
        let simplified_roots = simplify_gss_forest(&roots); // Use a new cache for simplify_gss_forest
        let simplified_a1 = simplified_roots[0].clone();

        // Verify structure and hash caching after simplification
        let mut collected_arcs = HashMap::new();
        collect_arcs_recursive(&simplified_a1, &mut collected_arcs);

        // Expected unique Arcs: A1, B1, C1, D1 (D2 is same as D1)
        assert_eq!(collected_arcs.len(), 4, "Expected 4 unique Arcs in the simplified GSS");

        let s_a1 = simplified_a1; // Renaming for clarity
        let s_b1 = s_a1.predecessors.iter().find(|n| n.value == 20).unwrap().clone();
        let s_d_node_from_a1 = s_a1.predecessors.iter().find(|n| n.value == 40).unwrap().clone();

        let s_c1 = s_b1.predecessors.iter().find(|n| n.value == 30).unwrap().clone();
        let s_d_node_from_c1 = s_c1.predecessors.iter().find(|n| n.value == 40).unwrap().clone();

        assert!(Arc::ptr_eq(&s_d_node_from_a1, &s_d_node_from_c1), "Both paths to D node should point to the same canonical Arc");
        let s_d_canonical = s_d_node_from_a1;


        assert_eq!(s_d_canonical.value, 40);
        assert_eq!(s_d_canonical.predecessors.len(), 0);
        assert_ne!(s_d_canonical.hash_key_cache, 0);

        assert_eq!(s_c1.value, 30);
        assert_eq!(s_c1.predecessors.len(), 1);
        assert!(Arc::ptr_eq(s_c1.predecessors.iter().next().unwrap(), &s_d_canonical));
        assert_ne!(s_c1.hash_key_cache, 0);

        assert_eq!(s_b1.value, 20);
        assert_eq!(s_b1.predecessors.len(), 1);
        assert!(Arc::ptr_eq(s_b1.predecessors.iter().next().unwrap(), &s_c1));
        assert_ne!(s_b1.hash_key_cache, 0);

        assert_eq!(s_a1.value, 10);
        assert_eq!(s_a1.predecessors.len(), 2); // B1 and D_canonical
        assert!(s_a1.predecessors.contains(&s_b1));
        assert!(s_a1.predecessors.contains(&s_d_canonical));
        assert_ne!(s_a1.hash_key_cache, 0);


        // Test shared node reuse from original structure (if simplify_gss_forest is used on non-canonical input)
        // Create non-canonical graph first using separate caches or manual GSSNode construction
        let e_val = 500;
        let f_val = 600;
        let g_val = 700;

        // Simulating non-canonical construction for simplify_gss_forest input
        let e_orig_non_canon_1 = Arc::new(GSSNode { value: e_val, predecessors: BTreeSet::new(), hash_key_cache: compute_node_hash(&e_val, &BTreeSet::new()) });
        let e_orig_non_canon_2 = Arc::new(GSSNode { value: e_val, predecessors: BTreeSet::new(), hash_key_cache: compute_node_hash(&e_val, &BTreeSet::new()) });
        assert_ne!(Arc::as_ptr(&e_orig_non_canon_1), Arc::as_ptr(&e_orig_non_canon_2)); // Ensure they are different Arcs

        let f_preds: BTreeSet<Arc<GSSNode<i32>>> = [e_orig_non_canon_1.clone()].into_iter().collect();
        let f_orig_non_canon = Arc::new(GSSNode { value: f_val, predecessors: f_preds.clone(), hash_key_cache: compute_node_hash(&f_val, &f_preds) });

        let g_preds: BTreeSet<Arc<GSSNode<i32>>> = [e_orig_non_canon_2.clone()].into_iter().collect(); // Using the other E
        let g_orig_non_canon = Arc::new(GSSNode { value: g_val, predecessors: g_preds.clone(), hash_key_cache: compute_node_hash(&g_val, &g_preds) });

        let simplified_shared = simplify_gss_forest(&[f_orig_non_canon, g_orig_non_canon]);
        assert_eq!(simplified_shared.len(), 2);
        let s_f = simplified_shared.iter().find(|n| n.value == f_val).unwrap();
        let s_g = simplified_shared.iter().find(|n| n.value == g_val).unwrap();

        assert_ne!(Arc::as_ptr(s_f), Arc::as_ptr(s_g));

        let s_f_pred = s_f.predecessors.iter().next().unwrap().clone();
        let s_g_pred = s_g.predecessors.iter().next().unwrap().clone();

        assert_eq!(s_f_pred.value, e_val);
        assert_eq!(s_g_pred.value, e_val);
        assert!(Arc::ptr_eq(&s_f_pred, &s_g_pred), "Shared original node E (structurally) should simplify to the same Arc instance");


        // Test predecessor order normalization with simplify_gss_forest
        let i_val = 80;
        let j_val = 90;
        let h_val = 100;

        let i_orig_nc = Arc::new(GSSNode { value: i_val, predecessors: BTreeSet::new(), hash_key_cache: compute_node_hash(&i_val, &BTreeSet::new()) });
        let j_orig_nc = Arc::new(GSSNode { value: j_val, predecessors: BTreeSet::new(), hash_key_cache: compute_node_hash(&j_val, &BTreeSet::new()) });

        let h1_preds_vec = vec![i_orig_nc.clone(), j_orig_nc.clone()];
        let h1_preds_set: BTreeSet<Arc<GSSNode<i32>>> = h1_preds_vec.into_iter().collect();
        let h1_orig_nc = Arc::new(GSSNode { value: h_val, predecessors: h1_preds_set.clone(), hash_key_cache: compute_node_hash(&h_val, &h1_preds_set) });

        let h2_preds_vec = vec![j_orig_nc.clone(), i_orig_nc.clone()]; // Different order
        let h2_preds_set: BTreeSet<Arc<GSSNode<i32>>> = h2_preds_vec.into_iter().collect();
        let h2_orig_nc = Arc::new(GSSNode { value: h_val, predecessors: h2_preds_set.clone(), hash_key_cache: compute_node_hash(&h_val, &h2_preds_set) });

        assert_ne!(Arc::as_ptr(&h1_orig_nc), Arc::as_ptr(&h2_orig_nc));

        let simplified_norm = simplify_gss_forest(&[h1_orig_nc, h2_orig_nc]);
        assert_eq!(simplified_norm.len(), 2);
        let s_h1 = &simplified_norm[0];
        let s_h2 = &simplified_norm[1];

        assert!(Arc::ptr_eq(s_h1, s_h2), "Structurally identical nodes H1 and H2 should be canonicalized to the same Arc by simplify_gss_forest");
    }

    #[test]
    fn test_simplification_canonicalizes_structurally_identical_nodes() {
        // This test demonstrates that simplify_gss_forest *does* canonicalize
        // structurally identical nodes, even if they originate from distinct Arcs.
        let mut cache_build: IntNodeCache = HashMap::new();

        // L1, L2, L3 are structurally identical (value 0, no preds)
        // When built with the same cache, they will be the same Arc.
        let l1 = node_canonical_i32(0, vec![], &mut cache_build);
        let l2 = node_canonical_i32(0, vec![], &mut cache_build);
        let l3 = node_canonical_i32(0, vec![], &mut cache_build);
        assert!(Arc::ptr_eq(&l1, &l2) && Arc::ptr_eq(&l1, &l3));

        // M1, M2, M3 have the same value (1) and same predecessor (the canonical L node).
        // They will also be the same Arc.
        let m1 = node_canonical_i32(1, vec![l1.clone()], &mut cache_build);
        let m2 = node_canonical_i32(1, vec![l2.clone()], &mut cache_build);
        let m3 = node_canonical_i32(1, vec![l3.clone()], &mut cache_build);
        assert!(Arc::ptr_eq(&m1, &m2) && Arc::ptr_eq(&m1, &m3));

        // R1 has M1, M2, M3 as predecessors. Since M1,M2,M3 are the same Arc,
        // R1 will have one unique predecessor.
        let r1_built_canonically = node_canonical_i32(2, vec![m1.clone(), m2.clone(), m3.clone()], &mut cache_build);
        assert_eq!(r1_built_canonically.predecessors.len(), 1, "R1 built canonically should have 1 unique predecessor Arc");

        // Now, let's simulate a non-canonical input for simplify_gss_forest
        let l_val = 0;
        let m_val = 1;
        let r_val = 2;

        let l1_nc = Arc::new(GSSNode { value: l_val, predecessors: BTreeSet::new(), hash_key_cache: compute_node_hash(&l_val, &BTreeSet::new())});
        let l2_nc = Arc::new(GSSNode { value: l_val, predecessors: BTreeSet::new(), hash_key_cache: compute_node_hash(&l_val, &BTreeSet::new())});
        let l3_nc = Arc::new(GSSNode { value: l_val, predecessors: BTreeSet::new(), hash_key_cache: compute_node_hash(&l_val, &BTreeSet::new())});

        let m1_preds: BTreeSet<_> = [l1_nc.clone()].into_iter().collect();
        let m1_nc = Arc::new(GSSNode { value: m_val, predecessors: m1_preds.clone(), hash_key_cache: compute_node_hash(&m_val, &m1_preds)});
        let m2_preds: BTreeSet<_> = [l2_nc.clone()].into_iter().collect();
        let m2_nc = Arc::new(GSSNode { value: m_val, predecessors: m2_preds.clone(), hash_key_cache: compute_node_hash(&m_val, &m2_preds)});
        let m3_preds: BTreeSet<_> = [l3_nc.clone()].into_iter().collect();
        let m3_nc = Arc::new(GSSNode { value: m_val, predecessors: m3_preds.clone(), hash_key_cache: compute_node_hash(&m_val, &m3_preds)});

        let r1_preds_vec = vec![m1_nc.clone(), m2_nc.clone(), m3_nc.clone()];
        let r1_preds_set: BTreeSet<_> = r1_preds_vec.into_iter().collect();
        let r1_orig_non_canonical = Arc::new(GSSNode { value: r_val, predecessors: r1_preds_set.clone(), hash_key_cache: compute_node_hash(&r_val, &r1_preds_set)});

        let simplified_roots = simplify_gss_forest(&[r1_orig_non_canonical]);
        let simplified_r1_arc = simplified_roots[0].clone();

        let mut collected_arcs_map = HashMap::new();
        collect_arcs_recursive(&simplified_r1_arc, &mut collected_arcs_map);

        // Expected unique Arcs with GLOBAL canonicalization by simplify_gss_forest:
        // One canonical L-level node (from l1_nc, l2_nc, l3_nc) -> 1 Arc.
        // One canonical M-level node (from m1_nc, m2_nc, m3_nc) -> 1 Arc.
        // One canonical R-level node (from r1_orig_non_canonical) -> 1 Arc.
        // Total = 1 + 1 + 1 = 3 unique Arcs.
        assert_eq!(collected_arcs_map.len(), 3, "Expected 3 unique Arcs in the simplified GSS after simplify_gss_forest");

        let s_r1_node = simplified_r1_arc.as_ref();
        assert_eq!(s_r1_node.value, 2);
        assert_eq!(s_r1_node.predecessors.len(), 1, "Simplified R1 should have 1 predecessor Arc (the canonical M node)");

        let s_m_level_arc = s_r1_node.predecessors.iter().next().unwrap().clone();
        assert_eq!(s_m_level_arc.value, 1);
        assert_eq!(s_m_level_arc.predecessors.len(), 1, "The canonical M node should have 1 predecessor Arc (the canonical L node)");

        let s_l_level_arc = s_m_level_arc.predecessors.iter().next().unwrap().clone();
        assert_eq!(s_l_level_arc.value, 0);
        assert_eq!(s_l_level_arc.predecessors.len(), 0, "The canonical L node should have no predecessors");
    }

    #[test]
    fn test_gss_simplification_reproduces_logged_structure() {
        let mut cache: MockNodeCache = HashMap::new();

        let token_info = MockLLMTokenInfo {
            active: "[0]".to_string(),
            intersection: "[0]".to_string(),
        };

        let val0 = MockParseStateNodeContent { state_id: 0, t: token_info.clone() };
        let val1 = MockParseStateNodeContent { state_id: 1, t: token_info.clone() };
        let val2 = MockParseStateNodeContent { state_id: 2, t: token_info.clone() };

        // --- Constructing the GSS using canonicalized creation ---
        let node_a_val0 = node_canonical_mock(val0.clone(), vec![], &mut cache); // Root 0
        let node_c_val0 = node_canonical_mock(val0.clone(), vec![], &mut cache); // Will be same Arc as node_a_val0
        assert!(Arc::ptr_eq(&node_a_val0, &node_c_val0));

        let node_g_val0 = node_canonical_mock(val0.clone(), vec![], &mut cache); // etc. all val0 leaves are same Arc
        // ... many other val0 leaves, all will point to node_a_val0/node_c_val0

        let node_b_val1 = node_canonical_mock(val1.clone(), vec![node_c_val0.clone()], &mut cache); // Root 1
        let node_e_val1 = node_canonical_mock(val1.clone(), vec![node_c_val0.clone()], &mut cache); // Will be same Arc as node_b_val1
        assert!(Arc::ptr_eq(&node_b_val1, &node_e_val1));

        // All other_orig_s1_nodes will also be the same Arc as node_b_val1/node_e_val1
        let mut other_orig_s1_nodes: Vec<Arc<MockGSSNode>> = Vec::new();
        for _ in 0..8 { // 8 other s1 nodes
            // Each points to the *same* canonical val0 node (node_c_val0)
            other_orig_s1_nodes.push(node_canonical_mock(val1.clone(), vec![node_c_val0.clone()], &mut cache));
        }
        for s1_node in &other_orig_s1_nodes {
            assert!(Arc::ptr_eq(s1_node, &node_b_val1));
        }

        let mut preds_for_d = vec![node_e_val1.clone()]; // This is node_b_val1
        preds_for_d.extend(other_orig_s1_nodes.iter().cloned()); // All are node_b_val1

        // BTreeSet will ensure only one instance of node_b_val1 is stored as predecessor
        let node_d_val2 = node_canonical_mock(val2.clone(), preds_for_d, &mut cache); // Root 2
        assert_eq!(node_d_val2.predecessors.len(), 1, "Node D (val2) should have 1 unique predecessor Arc (the canonical val1 node)");
        assert!(Arc::ptr_eq(node_d_val2.predecessors.iter().next().unwrap(), &node_b_val1));


        let roots_built_canonically = vec![
            node_a_val0.clone(),
            node_b_val1.clone(),
            node_d_val2.clone(),
        ];

        let max_nodes_to_print = 30;
        let gss_string_representation = print_gss_forest(&roots_built_canonically, max_nodes_to_print);
        println!("\n--- GSS Structure (Built Canonically) ---\n");
        println!("{}", gss_string_representation);
        println!("--- End of GSS Structure (Built Canonically) ---\n");

        let mut all_involved_arcs: HashMap<*const MockGSSNode, Arc<MockGSSNode>> = HashMap::new();
        for r in &roots_built_canonically {
            collect_arcs_recursive(r, &mut all_involved_arcs);
        }
        // Expected: 1 node for val0, 1 for val1, 1 for val2. Total 3.
        assert_eq!(all_involved_arcs.len(), 3, "The GSS built canonically should have 3 unique nodes.");

        // Now test simplify_gss_forest with a potentially non-canonical input structure
        // (mimicking the original test's intent before on-the-fly canonicalization)
        let nc_val0 = MockParseStateNodeContent { state_id: 0, t: token_info.clone() };
        let nc_val1 = MockParseStateNodeContent { state_id: 1, t: token_info.clone() };
        let nc_val2 = MockParseStateNodeContent { state_id: 2, t: token_info.clone() };

        // Create distinct Arcs for structurally identical nodes
        let nc_a0_1 = Arc::new(GSSNode { value: nc_val0.clone(), predecessors: BTreeSet::new(), hash_key_cache: compute_node_hash(&nc_val0, &BTreeSet::new()) });
        let nc_a0_2 = Arc::new(GSSNode { value: nc_val0.clone(), predecessors: BTreeSet::new(), hash_key_cache: compute_node_hash(&nc_val0, &BTreeSet::new()) });
        // ... and so on for all 10 original leaf nodes if we were to fully replicate non-canonical structure.
        // For brevity, let's use fewer for the simplify_gss_forest test part.

        let nc_b1_preds: BTreeSet<_> = [nc_a0_1.clone()].into_iter().collect();
        let nc_b1 = Arc::new(GSSNode { value: nc_val1.clone(), predecessors: nc_b1_preds.clone(), hash_key_cache: compute_node_hash(&nc_val1, &nc_b1_preds) });

        let nc_e1_preds: BTreeSet<_> = [nc_a0_2.clone()].into_iter().collect(); // Different val0 Arc
        let nc_e1 = Arc::new(GSSNode { value: nc_val1.clone(), predecessors: nc_e1_preds.clone(), hash_key_cache: compute_node_hash(&nc_val1, &nc_e1_preds) });

        let nc_d2_preds_vec = vec![nc_b1.clone(), nc_e1.clone()]; // Two distinct val1 Arcs
        let nc_d2_preds_set: BTreeSet<_> = nc_d2_preds_vec.into_iter().collect();
        let nc_d2 = Arc::new(GSSNode { value: nc_val2.clone(), predecessors: nc_d2_preds_set.clone(), hash_key_cache: compute_node_hash(&nc_val2, &nc_d2_preds_set) });

        let roots_non_canonical = vec![nc_a0_1.clone(), nc_b1.clone(), nc_d2.clone()];

        let simplified_roots = simplify_gss_forest(&roots_non_canonical);

        let simplified_gss_string_representation = print_gss_forest(&simplified_roots, max_nodes_to_print);
        println!("\n--- Simplified GSS Structure (After simplify_gss_forest on non-canonical input) ---\n");
        println!("{}", simplified_gss_string_representation);
        println!("--- End of Simplified GSS Structure ---\n");

        let mut collected_arcs_map: HashMap<*const MockGSSNode, Arc<MockGSSNode>> = HashMap::new();
        for root_arc in &simplified_roots {
            collect_arcs_recursive(root_arc, &mut collected_arcs_map);
        }
        assert_eq!(collected_arcs_map.len(), 3, "The simplified GSS should contain 3 unique Arcs after global canonicalization by simplify_gss_forest.");

        assert_eq!(simplified_roots.len(), 3);
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