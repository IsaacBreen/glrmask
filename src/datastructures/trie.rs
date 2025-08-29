use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Display};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::profiler::PROGRESS_BAR_ENABLED;
use deterministic_hash::DeterministicHasher;
use kdam::{tqdm, BarExt};
use ordered_hash_map::{OrderedHashMap, OrderedHashSet};
use profiler_macro::time_it;

/// Represents a node in a Trie–like structure (allowing shared subtrees and DAGs).
/// Multiple children can exist for the same edge key. Each edge instance has a value.
///
/// EK: type of the edge key (must be Ord).
/// EV: type of the edge value.
/// T: type of the value stored within the node.
///
/// NOTE: This node no longer stores Arc/RwLock pointers to other nodes. Instead,
/// children reference other nodes by their index (Trie2Index) in an Arena. Any access
/// to nodes (read/write of value, children traversal, depth recomputation, etc.) now
/// requires passing a reference to the Arena that owns the nodes.
#[derive(Debug, Clone)]
pub struct Trie<EK: Ord, EV, T> {
    pub value: T,
    /// Stores a map from EdgeKey to a map of destination node indices and edge values.
    children: BTreeMap<EK, OrderedHashMap<Trie2Index, EV>>,
    /// The “longest distance” from some source node (as computed by recompute_all_max_depths).
    /// Defaults to usize::MAX and is only updated when the user calls recompute_all_max_depths.
    pub max_depth: usize,
}

/// An index into the Arena for a Trie node.
/// This is the light-weight replacement for Arc<RwLock<Trie<...>>> in external code.
///
/// It provides `read(&arena)` and `write(&arena)` methods that mimic RwLock's API:
/// both return Option<Guard>. In most code you'll immediately call `.expect(...)` or `.unwrap()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Trie2Index {
    index: Index,
}

impl Display for Trie2Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Trie2Index({})", self.index.as_usize())
    }
}

impl Trie2Index {
    pub(crate) fn as_arc(&self) -> &Self {
        self
    }

    pub fn new(index: Index) -> Self {
        Trie2Index { index }
    }

    pub fn as_usize(self) -> usize {
        self.index.as_usize()
    }

    pub fn as_index(self) -> Index {
        self.index
    }

    /// Read-locks the Arena and returns a guard that derefs to &Trie at this index.
    /// Returns None if the index does not exist.
    pub fn read<'a, EK: Ord, EV, T>(
        self,
        arena: &'a Arena<Trie<EK, EV, T>>,
    ) -> Option<Trie2ReadGuard<'a, EK, EV, T>> {
        let guard = arena.values.read().ok()?;
        if !guard.contains_key(&self.index.as_usize()) {
            return None;
        }
        Some(Trie2ReadGuard {
            guard,
            index: self.index.as_usize(),
        })
    }

    /// Write-locks the Arena and returns a guard that derefs to &mut Trie at this index.
    /// Returns None if the index does not exist.
    pub fn write<'a, EK: Ord, EV, T>(
        self,
        arena: &'a Arena<Trie<EK, EV, T>>,
    ) -> Option<Trie2WriteGuard<'a, EK, EV, T>> {
        let guard = arena.values.write().ok()?;
        if !guard.contains_key(&self.index.as_usize()) {
            return None;
        }
        Some(Trie2WriteGuard {
            guard,
            index: self.index.as_usize(),
        })
    }

    /// Convenience constructor from usize.
    pub fn from_usize(i: usize) -> Self {
        Trie2Index { index: Index::from(i) }
    }
}

impl From<Index> for Trie2Index {
    fn from(i: Index) -> Self {
        Trie2Index { index: i }
    }
}

impl From<Trie2Index> for Index {
    fn from(ti: Trie2Index) -> Self {
        ti.index
    }
}

impl From<usize> for Trie2Index {
    fn from(u: usize) -> Self {
        Trie2Index { index: Index::from(u) }
    }
}

impl From<Trie2Index> for usize {
    fn from(ti: Trie2Index) -> usize {
        ti.index.as_usize()
    }
}

/// A read guard that keeps the Arena's internal RwLockReadGuard alive and provides
/// immutable access to a Trie node at a given index via Deref.
pub struct Trie2ReadGuard<'a, EK: Ord, EV, T> {
    guard: RwLockReadGuard<'a, BTreeMap<usize, Trie<EK, EV, T>>>,
    index: usize,
}

impl<'a, EK: Ord, EV, T> Deref for Trie2ReadGuard<'a, EK, EV, T> {
    type Target = Trie<EK, EV, T>;
    fn deref(&self) -> &Self::Target {
        self.guard
            .get(&self.index)
            .expect("Trie2ReadGuard: index not found in arena map")
    }
}

/// A write guard that keeps the Arena's internal RwLockWriteGuard alive and provides
/// mutable access to a Trie node at a given index via Deref/DerefMut.
pub struct Trie2WriteGuard<'a, EK: Ord, EV, T> {
    guard: RwLockWriteGuard<'a, BTreeMap<usize, Trie<EK, EV, T>>>,
    index: usize,
}

impl<'a, EK: Ord, EV, T> Deref for Trie2WriteGuard<'a, EK, EV, T> {
    type Target = Trie<EK, EV, T>;
    fn deref(&self) -> &Self::Target {
        self.guard
            .get(&self.index)
            .expect("Trie2WriteGuard: index not found in arena map")
    }
}

impl<'a, EK: Ord, EV, T> DerefMut for Trie2WriteGuard<'a, EK, EV, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard
            .get_mut(&self.index)
            .expect("Trie2WriteGuard: index not found in arena map")
    }
}

impl<EK, EV, T> JSONConvertible for Trie<EK, EV, T>
where
    EK: Ord + Clone + JSONConvertible + Debug,
    EV: Clone + JSONConvertible,
    T: Clone + JSONConvertible,
{
    /// Note: With the index-based design, we only serialize this node's local data:
    /// - value
    /// - max_depth
    /// - children: encoded as [EK_json, [[child_index, EV_json], ...]]
    ///
    /// We do NOT expand into a "nodes" array (no BFS). This format is concise and avoids
    /// requiring access to the Arena during serialization.
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("value".to_string(), self.value.to_json());
        obj.insert("max_depth".to_string(), self.max_depth.to_json());

        let children_json: Vec<JSONNode> = self.children.iter().map(|(ek, dest_map)| {
            let ek_json = ek.to_json();
            let dests_json: Vec<JSONNode> = dest_map.iter().map(|(idx, ev)| {
                JSONNode::Array(vec![
                    (idx.as_usize()).to_json(),
                    ev.to_json(),
                ])
            }).collect();
            JSONNode::Array(vec![ek_json, JSONNode::Array(dests_json)])
        }).collect();

        obj.insert("children".to_string(), JSONNode::Array(children_json));

        JSONNode::Object(obj)
    }

    /// Parses the local-node JSON format produced by to_json above.
    /// This reconstructs a Trie node that references children by indices only.
    /// The Arena is not created/returned here; the caller is expected to manage
    /// nodes in an Arena externally.
    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;

        let value = T::from_json(obj.remove("value").ok_or("Missing 'value' field")?)?;
        let max_depth = usize::from_json(obj.remove("max_depth").ok_or("Missing 'max_depth' field")?)?;

        let children_node = obj.remove("children").ok_or("Missing 'children' field")?;
        let children_arr = match children_node {
            JSONNode::Array(arr) => arr,
            _ => return Err("'children' field must be an array".to_string()),
        };

        let mut children = BTreeMap::new();
        for child_entry_node in children_arr {
            let mut child_entry_arr = match child_entry_node {
                JSONNode::Array(arr) if arr.len() == 2 => arr,
                _ => return Err("Child entry must be a 2-element array [ek, dests]".to_string()),
            };
            let dests_node = child_entry_arr.pop().unwrap();
            let ek_node = child_entry_arr.pop().unwrap();

            let ek = EK::from_json(ek_node)?;

            let dests_arr = match dests_node {
                JSONNode::Array(arr) => arr,
                _ => return Err("Destinations list must be an array".to_string()),
            };

            let mut dest_map = OrderedHashMap::new();
            for dest_pair_node in dests_arr {
                let mut dest_pair_arr = match dest_pair_node {
                    JSONNode::Array(arr) if arr.len() == 2 => arr,
                    _ => return Err("Destination pair must be a 2-element array [idx, ev]".to_string()),
                };
                let ev_node = dest_pair_arr.pop().unwrap();
                let idx_node = dest_pair_arr.pop().unwrap();

                let idx_usize = usize::from_json(idx_node)?;
                let idx = Trie2Index::from_usize(idx_usize);
                let ev = EV::from_json(ev_node)?;

                dest_map.insert(idx, ev);
            }
            children.insert(ek, dest_map);
        }

        Ok(Trie {
            value,
            children,
            max_depth,
        })
    }
}

impl JSONConvertible for Trie2Index {
    fn to_json(&self) -> JSONNode {
        self.as_usize().to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let u = usize::from_json(node)?;
        Ok(Trie2Index::from_usize(u))
    }
}

// Implementation block for core Trie functionality
// Added Clone bound for EK needed in insertion and others
impl<EK: Ord + Clone, EV, T> Trie<EK, EV, T> {
    /// Creates a new trie node with the given value and no children.
    /// The max_depth is initialized to usize::MAX and will be updated later
    /// when recompute_all_max_depths is called.
    pub fn new(value: T) -> Self {
        Trie {
            value,
            children: BTreeMap::new(),
            max_depth: usize::MAX,
        }
    }

    /// Create a new destination node in the arena and insert an edge to it.
    /// Returns the index of the newly created node.
    pub fn force_insert_to_new_node(
        &mut self,
        arena: &Arena<Trie<EK, EV, T>>,
        edge_key: EK,
        edge_value: EV,
        value: T,
    ) -> Trie2Index {
        let new_index = Trie2Index::new(arena.insert(Trie::new(value)));
        self.children
            .entry(edge_key)
            .or_default()
            .insert(new_index, edge_value);
        new_index
    }

    /// Insert an edge to an existing destination node index.
    pub fn force_insert_to_node(&mut self, edge_key: EK, edge_value: EV, dst: Trie2Index) {
        self.children.entry(edge_key).or_default().insert(dst, edge_value);
    }

    pub fn already_has_dst(&self, edge_key: EK, dst: Trie2Index) -> bool {
        self.children
            .get(&edge_key)
            .map_or(false, |dest_map| dest_map.contains_key(&dst))
    }

    pub fn already_has_dst_for_any_key(&self, dst: Trie2Index) -> bool {
        self.children.values().any(|dest_map| dest_map.contains_key(&dst))
    }

    pub fn get_edge_value(&self, edge_key: EK, dst: Trie2Index) -> Option<&EV> {
        self.children.get(&edge_key).and_then(|dest_map| dest_map.get(&dst))
    }

    pub fn get_edge_value_mut(&mut self, edge_key: EK, dst: &Trie2Index) -> Option<&mut EV> {
        self.children.get_mut(&edge_key).and_then(|dest_map| dest_map.get_mut(&dst))
    }

    /// Inserts an edge (no cycle checks).
    /// Depths can be recomputed later by calling `recompute_all_max_depths`.
    #[time_it]
    pub fn try_insert(
        &mut self,
        edge_key: EK,
        edge_value: &mut Option<EV>,
        child: Trie2Index,
    ) {
        self.try_insert_unchecked(edge_key, edge_value, child)
    }

    /// Inserts an edge without any checks or automatic depth updates.
    #[time_it]
    pub fn try_insert_unchecked(
        &mut self,
        edge_key: EK,
        edge_value: &mut Option<EV>,
        child: Trie2Index,
    ) {
        self.children
            .entry(edge_key)
            .or_default()
            .insert(child, edge_value.take().expect("edge_value must be Some when inserting"));
    }

    pub fn get(
        &self,
        edge_key: &EK,
    ) -> Option<&OrderedHashMap<Trie2Index, EV>>
    {
        self.children.get(edge_key)
    }

    pub fn get_mut(
        &mut self,
        edge_key: &EK,
    ) -> Option<&mut OrderedHashMap<Trie2Index, EV>>
    {
        self.children.get_mut(edge_key)
    }

    pub fn children(&self) -> &BTreeMap<EK, OrderedHashMap<Trie2Index, EV>> {
        &self.children
    }

    pub fn children_mut(&mut self) -> &mut BTreeMap<EK, OrderedHashMap<Trie2Index, EV>> {
        &mut self.children
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Collects all unique nodes (by index) reachable from the given roots (BFS).
    pub fn all_nodes(
        arena: &Arena<Trie<EK, EV, T>>,
        roots: &[Trie2Index],
    ) -> Vec<Trie2Index> {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        for &root in roots {
            if visited.insert(root.as_usize()) {
                queue.push_back(root);
            }
        }

        while let Some(node_idx) = queue.pop_front() {
            result.push(node_idx);

            if let Some(guard) = node_idx.read(arena) {
                for children_map in guard.children.values() {
                    for (child_idx, _) in children_map.iter() {
                        let c_u = child_idx.as_usize();
                        if visited.insert(c_u) {
                            queue.push_back(*child_idx);
                        }
                    }
                }
            } else {
                panic!("Trie::all_nodes: node index {} not found in arena", node_idx.as_usize());
            }
        }
        result
    }

    /// Performs garbage collection on the arena, keeping only nodes reachable from `roots`.
    pub fn gc(arena: &Arena<Self>, roots: &[Trie2Index]) {
        let live_nodes_vec = Self::all_nodes(arena, roots);
        let live_nodes_set: HashSet<usize> = live_nodes_vec.into_iter().map(|idx| idx.as_usize()).collect();
        let mut values_guard = arena.values.write().expect("Arena write lock poisoned during GC");
        values_guard.retain(|&k, _| live_nodes_set.contains(&k));
    }

    /// Recomputes `max_depth` for all nodes reachable from the given roots.
    /// Call this once after you finish building the trie graph. Before calling,
    /// nodes may have max_depth == usize::MAX which is suboptimal for scheduling
    /// but not incorrect for traversal.
    ///
    /// Uses a topological order (Kahn's algorithm). Assumes the graph is acyclic.
    pub fn recompute_all_max_depths(
        arena: &Arena<Trie<EK, EV, T>>,
        roots: &[Trie2Index],
    ) {
        let all_nodes = Self::all_nodes(arena, roots);
        if all_nodes.is_empty() {
            return;
        }

        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();

        for node_idx in &all_nodes {
            let u = node_idx.as_usize();
            in_degree.entry(u).or_insert(0);
            adj.entry(u).or_default();

            let guard = node_idx
                .read(arena)
                .expect("Arena read failed during recompute_all_max_depths");
            for child_idx in guard
                .children
                .values()
                .flat_map(|m| m.keys().cloned())
            {
                let v = child_idx.as_usize();
                adj.entry(u).or_default().push(v);
                *in_degree.entry(v).or_default() += 1;
            }
        }

        // Initialize depths to 0 for all nodes; sources will be processed first.
        let mut queue = VecDeque::new();
        for node_idx in &all_nodes {
            let u = node_idx.as_usize();
            if in_degree.get(&u).cloned().unwrap_or(0) == 0 {
                queue.push_back(u);
            }
            let mut w = node_idx
                .write(arena)
                .expect("Arena write failed when initializing depths");
            w.max_depth = 0;
        }

        while let Some(u) = queue.pop_front() {
            let u_depth = {
                let g = Trie2Index::from(u).read(arena).expect("read");
                g.max_depth
            };

            if let Some(children) = adj.get(&u) {
                for &v in children {
                    {
                        let mut vg = Trie2Index::from(v).write(arena).expect("write");
                        vg.max_depth = vg.max_depth.max(u_depth + 1);
                    }

                    let deg = in_degree.get_mut(&v).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(v);
                    }
                }
            }
        }
    }

    /// Recomputes the max_depth of this node based on its children's depths.
    /// Returns true if the depth changed.
    /// This does NOT propagate changes.
    pub fn recompute_max_depth(&mut self, arena: &Arena<Trie<EK, EV, T>>) -> bool {
        let new_max_depth = self
            .children
            .values()
            .flat_map(|dest_map| dest_map.keys().cloned())
            .map(|child_idx| {
                let g = child_idx.read(arena).expect("Arena read failed in recompute_max_depth");
                g.max_depth + 1
            })
            .max()
            .unwrap_or(0);

        if new_max_depth != self.max_depth {
            self.max_depth = new_max_depth;
            true
        } else {
            false
        }
    }
}

// Add this impl block for the recursive comparison helper (index-based)
impl<EK, EV, T> Trie<EK, EV, T>
where
    EK: Ord + Clone,
    EV: PartialEq + Clone,
    T: PartialEq,
{
    pub fn are_graphs_equal(
        arena_a: &Arena<Trie<EK, EV, T>>,
        a_idx: Trie2Index,
        arena_b: &Arena<Trie<EK, EV, T>>,
        b_idx: Trie2Index,
    ) -> bool {
        let mut cache = HashMap::new();
        Self::compare_indexes_recursive(arena_a, a_idx, arena_b, b_idx, &mut cache)
    }

    /// Recursively compares two Trie nodes referenced by indices for equality across an Arena.
    ///
    /// - `a_idx`, `b_idx`: The indices pointing to the Trie nodes to compare.
    /// - `comparison_cache`: Tracks pairs of (a_usize, b_usize) and their comparison result (bool).
    fn compare_indexes_recursive(
        arena_a: &Arena<Trie<EK, EV, T>>,
        a_idx: Trie2Index,
        arena_b: &Arena<Trie<EK, EV, T>>,
        b_idx: Trie2Index,
        comparison_cache: &mut HashMap<(usize, usize), bool>,
    ) -> bool {
        let a_u = a_idx.as_usize();
        let b_u = b_idx.as_usize();

        if Arc::ptr_eq(&arena_a.values, &arena_b.values) && a_u == b_u {
            return true;
        }

        let (k1, k2) = if a_u < b_u { (a_u, b_u) } else { (b_u, a_u) };

        if let Some(&cached) = comparison_cache.get(&(k1, k2)) {
            return cached;
        }

        // Optimistically assume true; update to false on mismatch.
        comparison_cache.insert((k1, k2), true);

        let a_guard = match a_idx.read(arena_a) {
            Some(g) => g,
            None => {
                comparison_cache.insert((k1, k2), false);
                return false;
            }
        };
        let b_guard = match b_idx.read(arena_b) {
            Some(g) => g,
            None => {
                comparison_cache.insert((k1, k2), false);
                return false;
            }
        };

        // 1. Compare non-recursive fields: value and max_depth.
        if a_guard.value != b_guard.value || a_guard.max_depth != b_guard.max_depth {
            comparison_cache.insert((k1, k2), false);
            return false;
        }

        // 2. Compare children structure (number of distinct edge keys).
        if a_guard.children.len() != b_guard.children.len() {
            comparison_cache.insert((k1, k2), false);
            return false;
        }

        // 3. Compare children for each edge key.
        for (a_ek, a_dest_map) in &a_guard.children {
            match b_guard.children.get(a_ek) {
                None => {
                    comparison_cache.insert((k1, k2), false);
                    return false;
                }
                Some(b_dest_map) => {
                    if a_dest_map.len() != b_dest_map.len() {
                        comparison_cache.insert((k1, k2), false);
                        return false;
                    }

                    let a_child_pairs: Vec<(Trie2Index, EV)> = a_dest_map
                        .iter()
                        .map(|(idx, ev)| (*idx, ev.clone()))
                        .collect();

                    let mut b_child_pairs: Vec<(Trie2Index, EV)> = b_dest_map
                        .iter()
                        .map(|(idx, ev)| (*idx, ev.clone()))
                        .collect();

                    'outer: for (a_child, a_ev) in &a_child_pairs {
                        let mut found = false;
                        for i in 0..b_child_pairs.len() {
                            if &b_child_pairs[i].1 == a_ev {
                                let b_child = b_child_pairs[i].0;
                                if Trie::compare_indexes_recursive(arena_a, *a_child, arena_b, b_child, comparison_cache) {
                                    b_child_pairs.remove(i);
                                    found = true;
                                    break;
                                }
                            }
                        }
                        if !found {
                            comparison_cache.insert((k1, k2), false);
                            return false;
                        }
                    }
                }
            }
        }

        true
    }
}

// Implementation block for special_map and related functionality
// Requires T: Clone, EK: Ord + Clone, EV: Clone
impl<T: Clone, EK: Ord + Clone, EV: Clone> Trie<EK, EV, T> {
    fn count_all_edges(
        arena: &Arena<Trie<EK, EV, T>>,
        root_nodes: &[Trie2Index],
    ) -> usize {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut queue: VecDeque<Trie2Index> = VecDeque::new();
        let mut total_edges = 0;

        for &root in root_nodes {
            if visited.insert(root.as_usize()) {
                queue.push_back(root);
            }
        }

        while let Some(node_idx) = queue.pop_front() {
            let guard = node_idx.read(arena).expect("RwLock poisoned during edge count");
            for children_map in guard.children.values() {
                for (child_idx, _) in children_map.iter() {
                    total_edges += 1;
                    let c_u = child_idx.as_usize();
                    if visited.insert(c_u) {
                        queue.push_back(*child_idx);
                    }
                }
            }
        }
        total_edges
    }

    /// Performs a specialized breadth-first traversal for propagation/scheduling.
    ///
    /// Traverses all edges and allows callers to:
    /// - process a node (process)
    /// - compute new values for children based on an edge (step)
    /// - merge multiple incoming values for the same child (merge)
    ///
    /// Scheduling uses node.max_depth; if not recomputed yet it may be usize::MAX,
    /// which is suboptimal but not incorrect.
    #[time_it]
    pub fn special_map<V: Clone>(
        arena: &Arena<Trie<EK, EV, T>>,
        initial_nodes_and_values: Vec<(Trie2Index, V)>,
        mut step: impl FnMut(&V, &EK, &EV, &Trie<EK, EV, T>) -> Option<V>,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&Trie<EK, EV, T>, &mut V) -> bool,
    ) {
        // ------------------------------------------------------------------
        //  Simple depth-driven scheduler.
        // ------------------------------------------------------------------
        let mut values: HashMap<usize, V> = HashMap::new();
        let mut stopped_nodes: HashSet<usize> = HashSet::new();
        let mut todo: BTreeMap<usize, OrderedHashSet<Trie2Index>> = BTreeMap::new();

        let initial_nodes: Vec<_> = initial_nodes_and_values.iter().map(|(n, _)| *n).collect();
        let total_edges = Self::count_all_edges(arena, &initial_nodes);
        if PROGRESS_BAR_ENABLED {
            println!("Progress bar enabled");
        } else {
            println!("Progress bar disabled")
        }
        let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED, leave=false);

        // Seed with the user-supplied starting set
        for (node_idx, v0) in initial_nodes_and_values {
            let ptr = node_idx.as_usize();
            values
                .entry(ptr)
                .and_modify(|old| merge(old, v0.clone()))
                .or_insert(v0);
            let depth = node_idx.read(arena).expect("poison").max_depth;
            todo.entry(depth).or_default().insert(node_idx);
        }

        // Main loop ---------------------------------------------------------
        while let Some((_depth, node_indices)) = todo.pop_first() {
            for node_idx in &node_indices {
                let ptr = node_idx.as_usize();
                if stopped_nodes.contains(&ptr) {
                    continue;
                }

                let mut agg_v = match values.remove(&ptr) {
                    Some(v) => v,
                    None => continue,
                };

                // ---------- user ‘process’ callback ----------
                let proceed = {
                    let guard = node_idx.read(arena).expect("poison");
                    process(&guard, &mut agg_v)
                };

                if !proceed {
                    stopped_nodes.insert(ptr);
                    continue;
                }

                // ---------- propagate to children -------------
                let edges: Vec<(EK, EV, Trie2Index)> = {
                    let guard = node_idx.read(arena).expect("poison");
                    guard
                        .children
                        .iter()
                        .flat_map(|(ek, dst_map)| {
                            let ekc = ek.clone();
                            dst_map
                                .iter()
                                .map(move |(child_idx, ev)| (ekc.clone(), ev.clone(), *child_idx))
                        })
                        .collect()
                };

                for (ek, ev, child_idx) in edges {
                    let _ = pb.update(1);
                    let child_ptr = child_idx.as_usize();

                    if stopped_nodes.contains(&child_ptr) {
                        continue;
                    }

                    let maybe_v = {
                        let child_guard = child_idx.read(arena).expect("poison");
                        step(&agg_v, &ek, &ev, &child_guard)
                    };
                    if let Some(new_v) = maybe_v {
                        values
                            .entry(child_ptr)
                            .and_modify(|old| merge(old, new_v.clone()))
                            .or_insert(new_v);

                        let child_depth = child_idx.read(arena).expect("poison").max_depth;
                        todo.entry(child_depth).or_default().insert(child_idx);
                    }
                }
            }
        }
    }

    /// Performs a specialized breadth-first traversal, grouping children by edge key.
    /// This is more efficient than `special_map` when many edges share the same key,
    /// as the `step` function is called once per key, not once per edge.
    ///
    /// Scheduling uses node.max_depth; if not recomputed yet it may be usize::MAX,
    /// which is suboptimal but not incorrect.
    #[time_it]
    pub fn special_map_grouped<V, S, I>(
        arena: &Arena<Trie<EK, EV, T>>,
        initial_nodes_and_values: Vec<(Trie2Index, V)>,
        mut step: S,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&Trie<EK, EV, T>, &mut V) -> bool,
    )
    where
        V: Clone,
        S: FnMut(
            &V, &EK, &OrderedHashMap<Trie2Index, EV>
        ) -> I,
        I: IntoIterator<Item = (Trie2Index, V)>,
    {
        // ------------------------------------------------------------------
        //  Simple depth-driven scheduler. (Same as special_map)
        // ------------------------------------------------------------------
        let mut values: HashMap<usize, V> = HashMap::new();
        let mut stopped_nodes: HashSet<usize> = HashSet::new();
        let mut todo: BTreeMap<usize, OrderedHashSet<Trie2Index>> = BTreeMap::new();

        let initial_nodes: Vec<_> = initial_nodes_and_values.iter().map(|(n, _)| *n).collect();
        let total_edges = Self::count_all_edges(arena, &initial_nodes);
        let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED, leave=false);

        // Seed with the user-supplied starting set
        for (node_idx, v0) in initial_nodes_and_values {
            let ptr = node_idx.as_usize();
            values
                .entry(ptr)
                .and_modify(|old| merge(old, v0.clone()))
                .or_insert(v0);
            let depth = node_idx.read(arena).expect("poison").max_depth;
            todo.entry(depth).or_default().insert(node_idx);
        }

        // Main loop ---------------------------------------------------------
        while let Some((_depth, node_indices)) = todo.pop_first() {
            for node_idx in &node_indices {
                let ptr = node_idx.as_usize();
                if stopped_nodes.contains(&ptr) { continue; }

                let mut agg_v = match values.remove(&ptr) {
                    Some(v) => v,
                    None => continue,
                };

                let proceed = {
                    let guard = node_idx.read(arena).expect("poison");
                    process(&guard, &mut agg_v)
                };

                if !proceed {
                    stopped_nodes.insert(ptr);
                    continue;
                }

                // ---------- propagate to children (grouped by edge key) -------------
                let children_by_ek: Vec<(EK, OrderedHashMap<Trie2Index, EV>)> = {
                    let guard = node_idx.read(arena).expect("poison");
                    guard.children.iter()
                        .map(|(ek, dst_map)| (ek.clone(), dst_map.clone()))
                        .collect()
                };

                for (ek, dest_map) in children_by_ek {
                    let valid_edges_count = dest_map.len();
                    if valid_edges_count > 0 {
                        let _ = pb.update(valid_edges_count);
                    }

                    let new_values_for_children = step(&agg_v, &ek, &dest_map);

                    for (child_idx, new_v) in new_values_for_children {
                        let child_ptr = child_idx.as_usize();

                        if stopped_nodes.contains(&child_ptr) {
                            continue;
                        }

                        values.entry(child_ptr)
                            .and_modify(|old| merge(old, new_v.clone()))
                            .or_insert(new_v);

                        let child_depth = child_idx.read(arena).expect("poison").max_depth;
                        todo.entry(child_depth).or_default().insert(child_idx);
                    }
                }
            }
        }
    }
}

// Implement PartialEq for Trie (shallow: compares value, max_depth, and immediate children lists)
impl<EK, EV, T> PartialEq for Trie<EK, EV, T>
where
    EK: Ord + PartialEq,
    EV: PartialEq,
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if self.value != other.value || self.max_depth != other.max_depth {
            return false;
        }
        if self.children.len() != other.children.len() {
            return false;
        }
        for (ek, self_map) in &self.children {
            match other.children.get(ek) {
                None => return false,
                Some(other_map) => {
                    if self_map.len() != other_map.len() {
                        return false;
                    }
                    // OrderedHashMap preserves insertion order; but we only need equality as sets of (idx, ev)
                    for (idx, ev) in self_map {
                        match other_map.get(idx) {
                            Some(o_ev) if o_ev == ev => {}
                            _ => return false,
                        }
                    }
                }
            }
        }
        true
    }
}

// Implement Eq for Trie
impl<EK, EV, T> Eq for Trie<EK, EV, T>
where
    EK: Ord + Eq,
    EV: Eq + Clone,
    T: Eq,
{
}

// Implement PartialOrd for Trie
impl<EK, EV, T> PartialOrd for Trie<EK, EV, T>
where
    EK: Ord,
    EV: Ord + Clone,
    T: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Implement Ord for Trie
impl<EK, EV, T> Ord for Trie<EK, EV, T>
where
    EK: Ord,
    EV: Ord + Clone,
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
            .then_with(|| self.max_depth.cmp(&other.max_depth))
            .then_with(|| self.children.len().cmp(&other.children.len()))
            .then_with(|| {
                for ((sk, sv), (ok, ov)) in self.children.iter().zip(other.children.iter()) {
                    match sk.cmp(ok) {
                        Ordering::Equal => (),
                        non_eq => return non_eq,
                    }
                    match sv.len().cmp(&ov.len()) {
                        Ordering::Equal => (),
                        non_eq => return non_eq,
                    }
                    for ((s_idx, s_ev), (o_idx, o_ev)) in sv.iter().zip(ov.iter()) {
                        match s_idx.cmp(o_idx) {
                            Ordering::Equal => (),
                            non_eq => return non_eq,
                        }
                        match s_ev.cmp(o_ev) {
                            Ordering::Equal => (),
                            non_eq => return non_eq,
                        }
                    }
                }
                Ordering::Equal
            })
    }
}

// Implement Hash for Trie (shallow: value, max_depth, and immediate children lists)
impl<EK, EV, T> Hash for Trie<EK, EV, T>
where
    EK: Ord + Hash,
    EV: PartialEq + Clone + Hash,
    T: PartialEq + Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
        self.max_depth.hash(state);

        self.children.len().hash(state);
        for (ek, dest_map) in &self.children {
            ek.hash(state);
            dest_map.len().hash(state);

            // To be order-independent with respect to child order, hash pairs and then sort.
            let mut pair_hashes = Vec::with_capacity(dest_map.len());
            for (child_idx, ev) in dest_map {
                let mut pair_hasher = DeterministicHasher::new(DefaultHasher::new());
                child_idx.as_usize().hash(&mut pair_hasher);
                ev.hash(&mut pair_hasher);
                pair_hashes.push(pair_hasher.finish());
            }
            pair_hashes.sort_unstable();
            for h in pair_hashes {
                h.hash(state);
            }
        }
    }
}

/// A helper struct to facilitate inserting an edge into a Trie,
/// trying multiple potential destinations and optionally creating a new node.
/// Provides a chainable interface.
///
/// This index-based version stores a reference to the Arena and a source Trie2Index.
pub struct EdgeInserter<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
where
    EK: Ord + Clone,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    arena: &'a Arena<Trie<EK, EV, T>>,
    source_idx: Trie2Index,                      // The source node for the edge
    edge_key: EK,                                // The key for the edge to be inserted
    edge_value: Option<EV>,                      // The value for the edge to be inserted
    merge_edge_value: FMergeEV,                  // The function to merge edge values
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
    result: Option<Trie2Index>,                  // Stores the successful destination node
}

impl<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
    EdgeInserter<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    /// Creates a new `EdgeInserter`.
    ///
    /// # Arguments
    ///
    /// * `arena`: The arena holding Trie nodes.
    /// * `source_idx`: The source node where the edge originates.
    /// * `edge_key`: The key for the new edge.
    /// * `edge_value`: The value for the new edge.
    /// * `merge_edge_value`: A closure that merges the existing edge value with the new edge value.
    pub fn new(
        arena: &'a Arena<Trie<EK, EV, T>>,
        source_idx: Trie2Index,
        edge_key: EK,
        edge_value: EV,
        merge_edge_value: FMergeEV,
        update_node_value: FUpdateT,
        mut merge_edge_value_and_source_node_value: FMergeEV_T,
    ) -> Self {
        // Avoid unused-parameter warnings
        let mut edge_value = edge_value;
        {
            let source_guard = source_idx
                .read(arena)
                .expect("Arena read poisoned while reading source node value for edge value merge");
            merge_edge_value_and_source_node_value(&mut edge_value, &source_guard.value);
        }

        EdgeInserter {
            arena,
            source_idx,
            edge_key,
            edge_value: Some(edge_value),
            merge_edge_value,
            update_node_value,
            merge_edge_value_and_source_node_value,
            result: None,
        }
    }

    /// Tries to establish an edge to the given `destination` index.
    ///
    /// If an edge with the same `edge_key` already exists pointing to `destination`,
    /// it merges the `edge_value` using the `merge_edge_value` closure.
    /// If no such edge exists, it inserts a new edge.
    ///
    /// Returns `self` to allow chaining.
    #[time_it]
    pub fn try_destination(mut self, destination: Trie2Index) -> Self {
        if self.result.is_some() {
            return self; // Already found a destination
        }

        let mut update_info: Option<(Trie2Index, EV)> = None;

        { // Scope for source_guard
            let mut source_guard = self.source_idx
                .write(self.arena)
                .expect("Arena write poisoned while locking source in try_destination");

            if let Some(existing_ev_mut) = source_guard
                .children
                .get_mut(&self.edge_key)
                .and_then(|dest_map| dest_map.get_mut(&destination))
            {
                let new_ev = self.edge_value.take().unwrap();
                crate::debug!(7, "Merging edge value {:?} into existing edge value {:?} for edge {:?} to node {}", new_ev, existing_ev_mut, self.edge_key, destination.as_usize());
                (self.merge_edge_value)(existing_ev_mut, new_ev);
                let updated_ev = existing_ev_mut.clone();
                self.result = Some(destination);
                update_info = Some((destination, updated_ev));
            } else {
                let edge_val_clone = self.edge_value.as_ref().unwrap().clone();
                crate::debug!(7, "Inserting edge {:?} with value {:?} to node {}", self.edge_key, edge_val_clone, destination.as_usize());
                source_guard.try_insert(self.edge_key.clone(), &mut self.edge_value, destination);
                self.result = Some(destination);
                update_info = Some((destination, edge_val_clone));
            }
        }

        if let Some((dest_idx, ev)) = update_info {
            crate::debug!(7, "Updating node value for destination {} with edge value {:?}. self.edge_value: {:?}", dest_idx.as_usize(), ev, self.edge_value);
            let mut dest_w = dest_idx.write(self.arena).expect("Arena write");
            (self.update_node_value)(&mut dest_w.value, &ev);
        }

        self
    }

    /// Tries to establish an edge to any destination in the provided slice.
    /// Iterates through `destinations` and calls `try_destination` for each until one succeeds.
    /// Returns `self` to allow chaining.
    #[time_it]
    pub fn try_destinations(mut self, destinations: &[Trie2Index]) -> Self {
        for &destination in destinations {
            if self.result.is_some() {
                break; // Stop trying once a destination is found
            }
            self = self.try_destination(destination);
        }
        self
    }

    #[time_it]
    pub fn try_destinations_iter(mut self, destinations: impl Iterator<Item = Trie2Index>) -> Self {
        for destination in destinations {
            if self.result.is_some() {
                break; // Stop trying once a destination is found
            }
            self = self.try_destination(destination);
        }
        self
    }

    #[time_it]
    pub fn try_destinations_iter_with<F, R>(mut self, destinations: F) -> Self
    where
        F: Fn() -> R,
        R: Iterator<Item = Trie2Index>,
    {
        for destination in destinations() {
            if self.result.is_some() {
                break;
            }
            self = self.try_destination(destination);
        }
        self
    }

    /// Merges the edge with existing children under `self.edge_key` (if any).
    /// Returns `self` to allow chaining.
    pub fn try_children(mut self) -> Self {
        if self.result.is_some() {
            return self;
        }

        let children_for_this_key: Vec<Trie2Index> = {
            let source_guard = self.source_idx
                .read(self.arena)
                .expect("Arena read poisoned while locking source in try_children");
            if let Some(dest_map) = source_guard.children.get(&self.edge_key) {
                dest_map.keys().cloned().collect()
            } else {
                Vec::new()
            }
        };

        if !children_for_this_key.is_empty() {
            self = self.try_destinations(&children_for_this_key);
        }
        self
    }

    /// If no destination has been found yet, creates a new node with the given `value`,
    /// inserts an edge to it from the source, and sets it as the result.
    ///
    /// Returns `self` to allow chaining.
    pub fn else_create_destination_with_value(mut self, value: T) -> Self {
        if self.result.is_some() {
            return self;
        }

        let new_node_idx = Trie2Index::new(self.arena.insert(Trie::new(value)));
        let edge_val_clone = self.edge_value.as_ref().unwrap().clone();

        { // Scope for source_guard
            let mut source_guard = self.source_idx
                .write(self.arena)
                .expect("Arena write poisoned while locking source in else_create_with_value");
            source_guard.try_insert(self.edge_key.clone(), &mut self.edge_value, new_node_idx);
            self.result = Some(new_node_idx);
        }

        if let Some(dest_idx) = self.result {
            let mut dest_w = dest_idx.write(self.arena).expect("Arena write");
            (self.update_node_value)(&mut dest_w.value, &edge_val_clone);
        }

        self
    }

    /// If no destination has been found yet, creates a new node by calling `value_fn`,
    /// inserts an edge to it from the source, and sets it as the result.
    ///
    /// Returns `self` to allow chaining.
    pub fn else_create_destination_with(self, value_fn: impl FnOnce() -> T) -> Self {
        if self.result.is_some() {
            return self;
        }
        self.else_create_destination_with_value(value_fn())
    }

    /// If no destination has been found yet, creates a new node with the default value (`T::default()`),
    /// inserts an edge to it from the source, and sets it as the result.
    ///
    /// Requires `T: Default`.
    /// Returns `self` to allow chaining.
    pub fn else_create_destination(self) -> Self
    where
        T: Default,
    {
        if self.result.is_some() {
            return self;
        }
        self.else_create_destination_with_value(T::default())
    }

    /// Returns the resulting destination node index, if one was found or created.
    pub fn into_option(self) -> Option<Trie2Index> {
        self.result
    }

    pub fn is_some(&self) -> bool {
        self.result.is_some()
    }

    pub fn clone_into_option(&self) -> Option<Trie2Index> {
        self.result
    }

    /// Returns the resulting destination node index, panicking if none was found or created.
    pub fn unwrap(self) -> Trie2Index {
        self.result.expect("EdgeInserter::unwrap() called but no destination was found or created")
    }

    /// Returns the resulting destination node index, panicking with the given message if none was found or created.
    pub fn expect(self, msg: &str) -> Trie2Index {
        self.result.expect(msg)
    }
}

// Optional: Add a convenience method to Trie2Index to create an EdgeInserter easily.
impl Trie2Index {
    /// Creates an `EdgeInserter` to help add an edge starting from this node index.
    ///
    /// # Example (after migration)
    ///
    /// ```ignore
    /// let root_idx: Trie2Index = ...;
    /// let arena: Arena<Trie<String, i32, NodeValue>> = ...;
    /// let god = /* obtain GodWrapper */ unimplemented!();
    /// let new_or_existing_node_idx = root_idx
    ///     .insert_edge(
    ///         &arena, &god, "key".to_string(), 1,
    ///         |ev_old, ev_new| *ev_old += ev_new,        // merge edge value
    ///         |_node_value, _ev| {},                     // update node value from edge value
    ///         |_edge_value, _source_node_value| {},      // merge edge value with source node value
    ///     )
    ///     .try_children()
    ///     .else_create_destination()
    ///     .unwrap();
    /// ```
    pub fn insert_edge<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>(
        self,
        arena: &'a Arena<Trie<EK, EV, T>>,
        edge_key: EK,
        edge_value: EV,
        merge_edge_value: FMergeEV,
        update_node_value: FUpdateT,
        merge_edge_value_and_source_node_value: FMergeEV_T,
    ) -> EdgeInserter<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
    where
        EK: Ord + Clone + Debug,
        EV: Clone + Debug,
        T: Clone,
        FMergeEV: FnMut(&mut EV, EV),
        FUpdateT: FnMut(&mut T, &EV),
        FMergeEV_T: FnMut(&mut EV, &T),
    {
        EdgeInserter::new(
            arena,
            self,
            edge_key,
            edge_value,
            merge_edge_value,
            update_node_value,
            merge_edge_value_and_source_node_value,
        )
    }
}

/// Attempts to establish an edge from `source` to a single `destination`,
/// optionally merging edge values if an edge already exists.
/// Returns `Some(Trie2Index)` if merge or insert succeeded.
pub fn try_destination<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>(
    arena: &'a Arena<Trie<EK, EV, T>>,
    source: Trie2Index,
    edge_key: EK,
    edge_value: EV,
    destination: Trie2Index,
    merge_edge_value: FMergeEV,
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
) -> Option<Trie2Index>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    EdgeInserter::new(
        arena,
        source,
        edge_key,
        edge_value,
        merge_edge_value,
        update_node_value,
        merge_edge_value_and_source_node_value,
    )
    .try_destination(destination)
    .into_option()
}

/// Attempts to establish an edge from `source` to any of the provided `destinations`,
/// returning the first successful one (merge or insert), or `None` if none matched.
pub fn try_destination_with<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>(
    arena: &'a Arena<Trie<EK, EV, T>>,
    source: Trie2Index,
    edge_key: EK,
    edge_value: EV,
    destinations: &[Trie2Index],
    merge_edge_value: FMergeEV,
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
) -> Option<Trie2Index>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    EdgeInserter::new(
        arena,
        source,
        edge_key,
        edge_value,
        merge_edge_value,
        update_node_value,
        merge_edge_value_and_source_node_value,
    )
    .try_destinations(destinations)
    .into_option()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Index {
    index: usize,
}

impl From<usize> for Index {
    fn from(index: usize) -> Self {
        Index { index }
    }
}

impl From<Index> for usize {
    fn from(idx: Index) -> usize {
        idx.index
    }
}

impl Index {
    pub fn as_usize(self) -> usize {
        self.index
    }
}

#[derive(Debug, Clone)]
pub struct Arena<T> {
    pub(crate) values: Arc<RwLock<BTreeMap<usize, T>>>,
}

impl<T> PartialEq for Arena<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.values, &other.values)
            || PartialEq::eq(
                &*self.values.read().unwrap(),
                &*other.values.read().unwrap(),
            )
    }
}
impl<T> Eq for Arena<T> where T: Eq {}

impl<T> PartialOrd for Arena<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if Arc::ptr_eq(&self.values, &other.values) {
            return Some(Ordering::Equal);
        }
        PartialOrd::partial_cmp(
            &*self.values.read().unwrap(),
            &*other.values.read().unwrap(),
        )
    }
}

impl<T> Ord for Arena<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        if Arc::ptr_eq(&self.values, &other.values) {
            return Ordering::Equal;
        }
        Ord::cmp(
            &*self.values.read().unwrap(),
            &*other.values.read().unwrap(),
        )
    }
}

// Note: hashing only the pointer means two Arenas with equal content but different Arcs
// will not hash equal even though `eq` may return true. If you need Eq/Hash consistency,
// hash the map contents instead.
impl<T> Hash for Arena<T>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&Arc::as_ptr(&self.values), state);
    }
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Arena {
            values: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    // Inserts `value` at the next free index (max_index + 1) and returns the Index.
    pub fn insert(&self, value: T) -> Index {
        let mut map = self.values.write().unwrap();
        let next = match map.keys().next_back().copied() {
            Some(k) => k.checked_add(1).expect("Arena index overflow"),
            None => 0,
        };
        let old = map.insert(next, value);
        debug_assert!(old.is_none());
        Index { index: next }
    }

    // Replace or set a value at a specific index. Returns the old value if any.
    pub fn insert_at(&self, index: Index, value: T) -> Option<T> {
        self.values.write().unwrap().insert(index.index, value)
    }

    // Remove a value at index, returning it if present.
    pub fn remove(&self, index: Index) -> Option<T> {
        self.values.write().unwrap().remove(&index.index)
    }

    // Returns true if the index exists in the arena.
    pub fn contains(&self, index: Index) -> bool {
        self.values.read().unwrap().contains_key(&index.index)
    }

    // Returns a clone of the value at index (requires T: Clone).
    pub fn get(&self, index: Index) -> Option<T>
    where
        T: Clone,
    {
        self.values.read().unwrap().get(&index.index).cloned()
    }

    // Read access via a closure (no Clone bound required).
    pub fn with<R>(&self, index: Index, f: impl FnOnce(&T) -> R) -> Option<R> {
        let guard = self.values.read().unwrap();
        guard.get(&index.index).map(f)
    }

    // Mutable access via a closure (no Clone bound required).
    pub fn with_mut<R>(&self, index: Index, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let mut guard = self.values.write().unwrap();
        guard.get_mut(&index.index).map(f)
    }

    pub fn len(&self) -> usize {
        self.values.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.read().unwrap().is_empty()
    }

    pub fn clear(&self) {
        self.values.write().unwrap().clear();
    }

    // Snapshot of all indices.
    pub fn indices(&self) -> Vec<Index> {
        self.values
            .read()
            .unwrap()
            .keys()
            .copied()
            .map(Index::from)
            .collect()
    }

    // Snapshot of all entries as (Index, T) pairs (requires T: Clone).
    pub fn to_vec(&self) -> Vec<(Index, T)>
    where
        T: Clone,
    {
        self.values
            .read()
            .unwrap()
            .iter()
            .map(|(&k, v)| (Index::from(k), v.clone()))
            .collect()
    }
}

impl<T> JSONConvertible for Arena<T>
where
    T: JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        let guard = self.values.read().unwrap();
        let items: Vec<JSONNode> = guard.iter().map(|(&k, v)| {
            JSONNode::Array(vec![
                k.to_json(),
                v.to_json(),
            ])
        }).collect();
        JSONNode::Array(items)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let arr = match node {
            JSONNode::Array(arr) => arr,
            _ => return Err("Expected JSONNode::Array for Arena".to_string()),
        };

        let mut map = BTreeMap::new();
        for item_node in arr {
            let mut pair = match item_node {
                JSONNode::Array(pair) if pair.len() == 2 => pair,
                _ => return Err("Expected 2-element array for Arena entry".to_string()),
            };
            let value_node = pair.pop().unwrap();
            let key_node = pair.pop().unwrap();

            let key = usize::from_json(key_node)?;
            let value = T::from_json(value_node)?;
            map.insert(key, value);
        }

        Ok(Arena {
            values: Arc::new(RwLock::new(map)),
        })
    }
}

pub type GodWrapper<EK, EV, T> = Arena<Trie<EK, EV, T>>;
pub type God<EK, EV, T> = Arena<Trie<EK, EV, T>>;