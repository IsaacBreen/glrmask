use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use deterministic_hash::DeterministicHasher;

// Type aliases for cleaner signatures
type NodeCache<T, A> = HashMap<BTreeMap<T, Arc<GSSNode<T, A>>>, Arc<GSSNode<T, A>>>;
type NodeMap<T, A> = BTreeMap<T, Arc<GSSNode<T, A>>>;
type NodeSet<T, A> = BTreeSet<(Arc<GSSNode<T, A>>, T)>;

pub trait PathAccumulator: Sized + Clone + Debug + Eq + PartialEq + Ord + PartialOrd + Hash + Default {
    fn union(&self, other: &Self) -> Self;
    fn pop(&self, right: &Self) -> Self;
}

impl PathAccumulator for () {
    fn union(&self, _other: &Self) -> Self { () }
    fn pop(&self, _right: &Self) -> Self { () }
}

fn compute_hash_key<T: Hash, A: PathAccumulator>(predecessors: &NodeMap<T, A>) -> u64 {
    let mut hasher = DeterministicHasher::new(DefaultHasher::new());
    for (edge_val, pred_arc) in predecessors {
        edge_val.hash(&mut hasher);
        pred_arc.hash_key_cache.hash(&mut hasher);
    }
    hasher.finish()
}

#[derive(Debug, Clone)]
pub struct GSSNode<T, A: PathAccumulator> {
    acc: A,
    predecessors: NodeMap<T, A>,
    hash_key_cache: u64,
}

#[derive(Clone)]
pub struct PathsIter<'a, T: Clone, A: PathAccumulator> {
    queue: VecDeque<(&'a GSSNode<T, A>, Vec<T>)>,
}

impl<'a, T: Clone, A: PathAccumulator> Iterator for PathsIter<'a, T, A> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((current_node, mut path_suffix)) = self.queue.pop_front() {
            if current_node.predecessors.is_empty() {
                path_suffix.reverse();
                return Some(path_suffix);
            }

            for (edge_val, pred_arc) in &current_node.predecessors {
                let mut new_path = path_suffix.clone();
                new_path.push(edge_val.clone());
                self.queue.push_back((pred_arc.as_ref(), new_path));
            }
        }
        None
    }
}

fn process_predecessors<T: Ord + Hash + Clone, A: PathAccumulator + Clone>(
    incoming: &NodeSet<T, A>
) -> NodeMap<T, A> {
    let mut grouped: BTreeMap<T, Vec<Arc<GSSNode<T, A>>>> = BTreeMap::new();
    for (pred_arc, edge_val) in incoming {
        grouped.entry(edge_val.clone()).or_default().push(pred_arc.clone());
    }

    let mut result = NodeMap::new();
    for (edge_val, pred_arcs) in grouped {
        if pred_arcs.is_empty() { continue; }

        let mut iter = pred_arcs.into_iter();
        let first = iter.next().unwrap();

        if iter.len() == 0 {
            result.insert(edge_val, first);
        } else {
            let mut merged = (*first).clone();
            for other in iter {
                merged.merge(&other);
            }
            result.insert(edge_val, Arc::new(merged));
        }
    }
    result
}

// Basic node creation and manipulation
impl<T: Ord + Hash + Clone, A: PathAccumulator + Clone> GSSNode<T, A> {
    pub fn new(acc: A) -> Self {
        let predecessors = NodeMap::new();
        let hash_key_cache = compute_hash_key(&predecessors);
        Self { acc, predecessors, hash_key_cache }
    }

    pub fn new_default() -> Self {
        Self::new(A::default())
    }

    pub fn new_with_predecessors(predecessors_set: NodeSet<T, A>) -> Self {
        let predecessors = process_predecessors(&predecessors_set);

        let acc = if predecessors.is_empty() {
            A::default()
        } else {
            predecessors.values()
                .map(|arc| &arc.acc)
                .fold(A::default(), |acc, other| acc.union(other))
        };

        let hash_key_cache = compute_hash_key(&predecessors);
        Self { acc, predecessors, hash_key_cache }
    }

    pub fn predecessors_with_values(&self) -> impl ExactSizeIterator<Item = (&Arc<Self>, &T)> {
        self.predecessors.iter().map(|(edge_val, pred_arc)| (pred_arc, edge_val))
    }

    pub fn predecessors(&self) -> &NodeMap<T, A> {
        &self.predecessors
    }

    pub fn is_empty(&self) -> bool {
        self.predecessors.is_empty()
    }

    pub fn acc(&self) -> &A {
        &self.acc
    }

    pub fn acc_mut(&mut self) -> &mut A {
        &mut self.acc
    }
}

// Canonicalization methods
impl<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug> GSSNode<T, A> {
    fn get_canonical(predecessors_set: NodeSet<T, A>, cache: &mut NodeCache<T, A>) -> Arc<Self> {
        let key = process_predecessors(&predecessors_set);

        let current_acc = if key.is_empty() {
            A::default()
        } else {
            key.values()
                .map(|arc| &arc.acc)
                .fold(A::default(), |acc, other| acc.union(other))
        };

        if let Some(entry) = cache.get_mut(&key) {
            let new_acc = entry.acc.union(&current_acc);
            if new_acc != entry.acc {
                let mut temp_arc = entry.clone();
                Arc::make_mut(&mut temp_arc).acc = new_acc;
                *entry = temp_arc.clone();
                return temp_arc;
            }
            return entry.clone();
        }

        let hash_key_cache = compute_hash_key(&key);
        let new_node = Arc::new(GSSNode {
            acc: current_acc,
            predecessors: key.clone(),
            hash_key_cache,
        });
        cache.insert(key, new_node.clone());
        new_node
    }

    pub fn new_canonical(initial_acc: A, cache: &mut NodeCache<T, A>) -> Arc<Self> {
        let key = NodeMap::new();

        if let Some(entry) = cache.get_mut(&key) {
            let new_acc = entry.acc.union(&initial_acc);
            if new_acc != entry.acc {
                let mut temp_arc = entry.clone();
                Arc::make_mut(&mut temp_arc).acc = new_acc;
                *entry = temp_arc.clone();
                return temp_arc;
            }
            return entry.clone();
        }

        let hash_key_cache = compute_hash_key(&key);
        let new_node = Arc::new(GSSNode {
            acc: initial_acc,
            predecessors: key.clone(),
            hash_key_cache,
        });
        cache.insert(key, new_node.clone());
        new_node
    }
}

// Core manipulation methods
impl<T: Ord + Hash + Clone, A: PathAccumulator + Clone> GSSNode<T, A> {
    pub fn push(self, edge_value: T) -> Self {
        let predecessors_set = NodeSet::from([(Arc::new(self), edge_value)]);
        Self::new_with_predecessors(predecessors_set)
    }

    pub fn pop_into(&self, mut result: Self) -> Self {
        for (pred_arc, _) in self.predecessors_with_values() {
            result.merge(&pred_arc);
        }
        result
    }

    pub fn pop(&self) -> Self {
        self.pop_into(Self::new_default())
    }

    pub fn popn(&self, n: usize) -> Self {
        if n == 0 {
            self.clone()
        } else {
            self.pop().popn(n - 1)
        }
    }

    pub fn merge(&mut self, other: &Self) {
        if self == other { return; }
        self.acc = self.acc.union(&other.acc);

        for (edge_val, other_pred) in &other.predecessors {
            match self.predecessors.entry(edge_val.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(other_pred.clone());
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    Arc::make_mut(entry.get_mut()).merge(&other_pred);
                }
            }
        }
        self.hash_key_cache = compute_hash_key(&self.predecessors);
    }

    pub fn merged(mut self, other: Self) -> Self {
        self.merge(&other);
        self
    }

    pub fn iter_paths(&self) -> PathsIter<'_, T, A> {
        let mut queue = VecDeque::new();
        queue.push_back((self, Vec::new()));
        PathsIter { queue }
    }

    pub fn flatten(&self) -> Vec<Vec<(T, A)>> {
        let mut results = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((self, Vec::new()));

        while let Some((node, mut path)) = queue.pop_front() {
            if node.predecessors.is_empty() {
                path.reverse();
                results.push(path);
            } else {
                for (edge_val, pred_arc) in &node.predecessors {
                    let mut new_path = path.clone();
                    new_path.push((edge_val.clone(), node.acc.clone()));
                    queue.push_back((pred_arc.as_ref(), new_path));
                }
            }
        }
        results
    }

    pub fn flatten_bulk(nodes: &[Arc<Self>]) -> Vec<Vec<(T, A)>> {
        nodes.iter().flat_map(|node| node.flatten()).collect()
    }

    pub fn map<F, U>(&self, f: F) -> GSSNode<U, A>
    where
        F: Copy + Fn(&T) -> U,
        U: Ord + Hash + Clone,
    {
        let new_predecessors: NodeSet<U, A> = self.predecessors.iter()
            .map(|(edge_val, pred_arc)| {
                let mapped_pred = Arc::new(pred_arc.map(f));
                let new_edge_val = f(edge_val);
                (mapped_pred, new_edge_val)
            })
            .collect();

        GSSNode::<U, A>::new_with_predecessors(new_predecessors)
    }
}

// Trait implementations
impl<T: Hash, A: PathAccumulator> Hash for GSSNode<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_key_cache.hash(state);
    }
}

impl<T: Ord + Hash + PartialEq, A: PathAccumulator + PartialEq> PartialEq for GSSNode<T, A> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) || (
            self.hash_key_cache == other.hash_key_cache &&
            self.acc == other.acc &&
            self.predecessors == other.predecessors
        )
    }
}

impl<T: Ord + Hash + Eq, A: PathAccumulator + Eq> Eq for GSSNode<T, A> {}

impl<T: Ord + Hash + PartialOrd, A: PathAccumulator + PartialOrd> PartialOrd for GSSNode<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if std::ptr::eq(self, other) { return Some(Ordering::Equal); }

        self.hash_key_cache.partial_cmp(&other.hash_key_cache)
            .and_then(|ord| match ord {
                Ordering::Equal => self.acc.partial_cmp(&other.acc)
                    .and_then(|ord| match ord {
                        Ordering::Equal => self.predecessors.partial_cmp(&other.predecessors),
                        other => Some(other),
                    }),
                other => Some(other),
            })
    }
}

impl<T: Ord + Hash, A: PathAccumulator + Ord> Ord for GSSNode<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        if std::ptr::eq(self, other) { return Ordering::Equal; }

        self.hash_key_cache.cmp(&other.hash_key_cache)
            .then_with(|| self.acc.cmp(&other.acc))
            .then_with(|| self.predecessors.cmp(&other.predecessors))
    }
}

impl<T, A: PathAccumulator> Drop for GSSNode<T, A> {
    fn drop(&mut self) {
        let predecessors = std::mem::take(&mut self.predecessors);
        let mut worklist: Vec<Arc<GSSNode<T, A>>> = predecessors.into_values().collect();

        while let Some(node_arc) = worklist.pop() {
            if Arc::strong_count(&node_arc) == 1 {
                if let Ok(mut inner_node) = Arc::try_unwrap(node_arc) {
                    let inner_preds = std::mem::take(&mut inner_node.predecessors);
                    worklist.extend(inner_preds.into_values());
                }
            }
        }
    }
}

// Simplified trait for GSS operations
pub trait GSSTrait<T: Clone + Hash, A: PathAccumulator> {
    fn push(&self, edge_value: T) -> GSSNode<T, A> where T: Ord + Clone, A: Clone;
    fn push_to(&self, edge_value: T, dest: &mut GSSNode<T, A>) where T: Ord + Clone, A: Clone;
    fn pop(&self) -> GSSNode<T, A> where T: Ord + Clone, A: Clone;
    fn popn(&self, n: usize) -> GSSNode<T, A> where T: Ord + Clone, A: Clone;
}

impl<T: Clone + Ord + Hash, A: PathAccumulator + Clone> GSSTrait<T, A> for GSSNode<T, A> {
    fn push(&self, edge_value: T) -> GSSNode<T, A> {
        self.clone().push(edge_value)
    }

    fn push_to(&self, edge_value: T, dest: &mut GSSNode<T, A>) {
        let new_link = GSSNode::new_with_predecessors(
            NodeSet::from([(Arc::new(self.clone()), edge_value)])
        );
        dest.merge(&new_link);
    }

    fn pop(&self) -> GSSNode<T, A> {
        GSSNode::pop(self)
    }

    fn popn(&self, n: usize) -> GSSNode<T, A> {
        GSSNode::popn(self, n)
    }
}

impl<T: Clone + Ord + Hash, A: PathAccumulator + Clone> GSSTrait<T, A> for Arc<GSSNode<T, A>> {
    fn push(&self, edge_value: T) -> GSSNode<T, A> {
        GSSNode::new_with_predecessors(NodeSet::from([(self.clone(), edge_value)]))
    }

    fn push_to(&self, _edge_value: T, dest: &mut GSSNode<T, A>) {
        dest.merge(&self.as_ref().clone());
        dest.acc = dest.acc.pop(&self.acc);
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
            Some(arc) => arc.push(edge_value),
            None => {
                let root = Arc::new(GSSNode::new(A::default()));
                GSSNode::new_with_predecessors(NodeSet::from([(root, edge_value)]))
            }
        }
    }

    fn push_to(&self, edge_value: T, dest: &mut GSSNode<T, A>) {
        match self {
            Some(arc) => arc.push_to(edge_value, dest),
            None => {
                let default_node = GSSNode::new(A::default());
                dest.merge(&default_node);
                dest.acc = dest.acc.pop(&default_node.acc);
            }
        }
    }

    fn pop(&self) -> GSSNode<T, A> {
        self.as_ref().map(|arc| arc.pop()).unwrap_or_else(GSSNode::new_default)
    }

    fn popn(&self, n: usize) -> GSSNode<T, A> {
        self.as_ref().map(|arc| arc.popn(n)).unwrap_or_else(GSSNode::new_default)
    }
}

impl<T: Clone + Ord + Hash, A: PathAccumulator + Clone + Default> GSSTrait<T, A> for Option<GSSNode<T, A>> {
    fn push(&self, edge_value: T) -> GSSNode<T, A> {
        match self {
            Some(node) => node.clone().push(edge_value),
            None => GSSNode::new(A::default()).push(edge_value),
        }
    }

    fn push_to(&self, edge_value: T, dest: &mut GSSNode<T, A>) {
        match self {
            Some(node) => node.push_to(edge_value, dest),
            None => GSSNode::new(A::default()).push_to(edge_value, dest),
        }
    }

    fn pop(&self) -> GSSNode<T, A> {
        self.as_ref().map(|node| node.pop()).unwrap_or_else(GSSNode::new_default)
    }

    fn popn(&self, n: usize) -> GSSNode<T, A> {
        self.as_ref().map(|node| node.popn(n)).unwrap_or_else(GSSNode::new_default)
    }
}

// Utility functions
pub fn prune_and_transform_recursive_canonical<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    node_arc: &Arc<GSSNode<T, A>>,
    closure: &impl Fn(&A) -> Option<(A, bool)>,
    memo: &mut HashMap<*const GSSNode<T, A>, Option<Arc<GSSNode<T, A>>>>,
    cache: &mut NodeCache<T, A>,
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
        Some((new_acc, continue_recursion)) => {
            let new_predecessors_set = if continue_recursion {
                node_arc.predecessors.iter()
                    .filter_map(|(edge_val, pred_arc)| {
                        prune_and_transform_recursive_canonical(pred_arc, closure, memo, cache)
                            .map(|new_pred| (new_pred, edge_val.clone()))
                    })
                    .collect()
            } else {
                node_arc.predecessors_with_values()
                    .map(|(pred_arc, edge_val)| (pred_arc.clone(), edge_val.clone()))
                    .collect()
            };

            let canonical_arc = GSSNode::get_canonical(new_predecessors_set, cache);
            let mut temp_arc = canonical_arc.clone();
            Arc::make_mut(&mut temp_arc).acc = temp_arc.acc.union(&new_acc);

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
    let mut cache = NodeCache::new();
    prune_and_transform_recursive_canonical(node_arc, closure, memo, &mut cache)
}

pub fn find_longest_path<T: Clone + Ord + Hash, A: PathAccumulator>(
    root_node: &GSSNode<T, A>
) -> Option<Vec<(T, Arc<GSSNode<T, A>>)>> {
    if root_node.predecessors.is_empty() {
        return None;
    }

    fn find_longest_recursive<T: Clone + Ord + Hash, A: PathAccumulator>(
        node_arc: &Arc<GSSNode<T, A>>,
        memo: &mut HashMap<*const GSSNode<T, A>, Vec<(T, Arc<GSSNode<T, A>>)>>,
        visited: &mut HashSet<*const GSSNode<T, A>>,
    ) -> Vec<(T, Arc<GSSNode<T, A>>)> {
        let node_ptr = Arc::as_ptr(node_arc);

        if let Some(cached) = memo.get(&node_ptr) {
            return cached.clone();
        }
        if !visited.insert(node_ptr) {
            return Vec::new();
        }

        if node_arc.predecessors.is_empty() {
            visited.remove(&node_ptr);
            memo.insert(node_ptr, Vec::new());
            return Vec::new();
        }

        let mut longest = Vec::new();
        for (edge_val, pred_arc) in &node_arc.predecessors {
            let mut path = find_longest_recursive(pred_arc, memo, visited);
            path.push((edge_val.clone(), node_arc.clone()));
            if path.len() > longest.len() {
                longest = path;
            }
        }

        memo.insert(node_ptr, longest.clone());
        visited.remove(&node_ptr);
        longest
    }

    let mut memo = HashMap::new();
    let mut longest_overall = None;

    for (_, pred_arc) in root_node.predecessors() {
        let mut visited = HashSet::new();
        let path = find_longest_recursive(pred_arc, &mut memo, &mut visited);
        if longest_overall.as_ref().map_or(true, |current: &Vec<_>| path.len() > current.len()) {
            longest_overall = Some(path);
        }
    }
    longest_overall
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

pub fn gather_gss_stats<T, A: PathAccumulator>(roots: &[impl AsRef<GSSNode<T, A>>]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut visited = HashSet::new();
    let mut processed = HashSet::new();
    let mut queue = VecDeque::new();
    let mut total_depth = 0u64;
    let mut total_preds = 0u64;

    for root in roots {
        let node_ref = root.as_ref();
        let node_ptr = node_ref as *const GSSNode<T, A>;
        if visited.insert(node_ptr) {
            queue.push_back((node_ref, 0));
        }
    }

    while let Some((node, depth)) = queue.pop_front() {
        let node_ptr = node as *const GSSNode<T, A>;
        if !processed.insert(node_ptr) {
            continue;
        }

        stats.unique_nodes += 1;
        stats.max_depth = stats.max_depth.max(depth);
        total_depth += depth as u64;

        let num_preds = node.predecessors.len();
        stats.max_predecessors_with_values = stats.max_predecessors_with_values.max(num_preds);
        total_preds += num_preds as u64;

        let unique_preds: HashSet<_> = node.predecessors.values()
            .map(|arc| Arc::as_ptr(arc))
            .collect();
        if unique_preds.len() > 1 {
            stats.merge_points += 1;
        }

        for (_, pred_arc) in &node.predecessors {
            let pred_ptr = pred_arc.as_ref() as *const GSSNode<T, A>;
            if visited.insert(pred_ptr) {
                queue.push_back((pred_arc.as_ref(), depth + 1));
            }
        }
    }

    if stats.unique_nodes > 0 {
        stats.average_depth = total_depth as f64 / stats.unique_nodes as f64;
        stats.average_predecessors_with_values = total_preds as f64 / stats.unique_nodes as f64;
    }
    stats
}

pub fn print_gss_forest<T: Debug, A: PathAccumulator>(
    roots: &[Arc<GSSNode<T, A>>],
    max_nodes: usize
) -> String {
    fn print_node<T: Debug, A: PathAccumulator>(
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

        writeln!(output, "{}- Node {:p}: (Acc: {:?})", prefix, node_ptr, node_arc.acc)?;

        if !node_arc.predecessors.is_empty() {
            writeln!(output, "{}  Predecessors:", prefix)?;
            for (edge_val, pred_arc) in &node_arc.predecessors {
                writeln!(output, "{}    - Edge: {:?} -> {:p}", prefix, edge_val, Arc::as_ptr(pred_arc))?;
                if *node_count < max_nodes {
                    print_node(pred_arc, visited, indent + 2, node_count, max_nodes, output)?;
                }
                if *node_count >= max_nodes {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    let mut visited = HashSet::new();
    let mut node_count = 0;
    let mut output = String::new();

    if roots.is_empty() {
        return "GSS Forest: (No roots)".to_string();
    }

    writeln!(&mut output, "GSS Forest (Max Nodes: {}):", max_nodes).unwrap();

    for (i, root) in roots.iter().enumerate() {
        writeln!(&mut output, "Root {}: {:p}", i, Arc::as_ptr(root)).unwrap();
        if print_node(root, &mut visited, 1, &mut node_count, max_nodes, &mut output).is_err() {
            return format!("Error writing GSS structure");
        }
        if node_count >= max_nodes && i < roots.len() - 1 {
            writeln!(&mut output, "... (Truncated)").unwrap();
            break;
        }
    }

    output
}

// Simplification methods
fn simplify_node_recursive<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    node_arc: &Arc<GSSNode<T, A>>,
    memo: &mut HashMap<*const GSSNode<T, A>, Arc<GSSNode<T, A>>>,
    cache: &mut NodeCache<T, A>,
) -> Arc<GSSNode<T, A>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(simplified) = memo.get(&node_ptr) {
        return simplified.clone();
    }

    let simplified_predecessors: NodeSet<T, A> = node_arc.predecessors.iter()
        .map(|(edge_val, pred_arc)| {
            let simplified_pred = simplify_node_recursive(pred_arc, memo, cache);
            (simplified_pred, edge_val.clone())
        })
        .collect();

    let canonical_arc = GSSNode::get_canonical(simplified_predecessors, cache);
    let mut temp_arc = canonical_arc.clone();
    Arc::make_mut(&mut temp_arc).acc = temp_arc.acc.union(&node_arc.acc);

    memo.insert(node_ptr, temp_arc.clone());
    temp_arc
}

fn simplify_gss_forest<T: Clone + Ord + Hash + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug>(
    roots: &[Arc<GSSNode<T, A>>],
) -> Vec<Arc<GSSNode<T, A>>> {
    let mut memo = HashMap::new();
    let mut cache = NodeCache::new();

    let mut simplified: Vec<_> = roots.iter()
        .map(|root| simplify_node_recursive(root, &mut memo, &mut cache))
        .collect();

    // Deduplicate
    let mut unique_map = HashMap::new();
    for arc in &mut simplified {
        let ptr = Arc::as_ptr(arc);
        let canonical = unique_map.entry(ptr).or_insert_with(|| arc.clone());
        *arc = canonical.clone();
    }

    simplified
}

impl<T: Ord + Hash + Clone + Debug, A: PathAccumulator + Clone + Ord + Hash + Debug> GSSNode<T, A> {
    pub fn simplify(&mut self) {
        let self_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        let mut cache = NodeCache::new();
        let simplified = simplify_node_recursive(&self_arc, &mut memo, &mut cache);
        *self = (*simplified).clone();
    }

    pub fn simplify_recursive(
        this_arc: &mut Arc<Self>,
        memo: &mut HashMap<*const Self, Arc<Self>>,
        cache: &mut NodeCache<T, A>,
    ) {
        *this_arc = simplify_node_recursive(this_arc, memo, cache);
    }

    pub fn simplify_together(nodes: &mut [&mut Arc<Self>]) {
        let mut memo = HashMap::new();
        let mut cache = NodeCache::new();
        for node_arc in nodes {
            **node_arc = simplify_node_recursive(*node_arc, &mut memo, &mut cache);
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
        fn default() -> Self {
            Self { active: BTreeSet::new(), intersection: BTreeSet::new() }
        }
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

    #[test]
    fn test_gss_simplification_basic() {
        let acc_base = MockPathAccumulator {
            active: BTreeSet::from([0]),
            intersection: BTreeSet::from([0])
        };
        let acc_other = MockPathAccumulator {
            active: BTreeSet::from([1]),
            intersection: BTreeSet::from([1])
        };

        let n4_base = Arc::new(MockGSSNode::new(acc_base.clone()));
        let d1_orig = Arc::new(MockGSSNode::new_with_predecessors(
            NodeSet::from([(n4_base.clone(), 40)])
        ));

        let n4_other = Arc::new(MockGSSNode::new(acc_other.clone()));
        let d2_orig = Arc::new(MockGSSNode::new_with_predecessors(
            NodeSet::from([(n4_other.clone(), 40)])
        ));

        let c1_orig = Arc::new(MockGSSNode::new_with_predecessors(
            NodeSet::from([(d1_orig.clone(), 30)])
        ));
        let b1_orig = Arc::new(MockGSSNode::new_with_predecessors(
            NodeSet::from([(c1_orig.clone(), 20)])
        ));

        let a1_orig = Arc::new(MockGSSNode::new_with_predecessors(NodeSet::from([
            (b1_orig.clone(), 10),
            (d2_orig.clone(), 10)
        ])));

        let roots = vec![a1_orig.clone()];
        let simplified_roots = simplify_gss_forest(&roots);

        assert_eq!(simplified_roots.len(), 1);
        let s_a1 = &simplified_roots[0];

        // Collect all nodes to verify structure
        fn collect_nodes(node: &Arc<MockGSSNode>, collected: &mut HashMap<*const MockGSSNode, Arc<MockGSSNode>>) {
            let ptr = Arc::as_ptr(node);
            if collected.contains_key(&ptr) {
                return;
            }
            collected.insert(ptr, node.clone());
            for (_, pred) in &node.predecessors {
                collect_nodes(pred, collected);
            }
        }

        let mut collected = HashMap::new();
        collect_nodes(s_a1, &mut collected);
        assert_eq!(collected.len(), 6, "Expected 6 unique nodes after simplification");

        // Verify structure
        assert_eq!(s_a1.predecessors.len(), 1);
        let (edge_val, merged_pred) = s_a1.predecessors.iter().next().unwrap();
        assert_eq!(*edge_val, 10);

        let expected_merged_acc = acc_base.union(&acc_other);
        assert_eq!(merged_pred.acc, expected_merged_acc);
        assert_eq!(s_a1.acc, expected_merged_acc);

        assert_eq!(merged_pred.predecessors.len(), 2);
        assert!(merged_pred.predecessors.contains_key(&20));
        assert!(merged_pred.predecessors.contains_key(&40));
    }
}