use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use rand::prelude::*;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Display};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::EntryApi;
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::profiler::PROGRESS_BAR_ENABLED;
use deterministic_hash::DeterministicHasher;
use kdam::{tqdm, BarExt};
use ordered_hash_map::{OrderedHashMap, OrderedHashSet};
use profiler_macro::time_it;
use range_set_blaze::RangeSetBlaze;

/// Represents statistics about a Trie graph reachable from a set of roots.
#[derive(Debug, Clone, PartialEq)]
pub struct TrieStats {
    pub num_reachable_nodes: usize,
    pub num_reachable_edges: usize,
    pub max_depth: usize,
    pub num_roots: usize,
    pub num_leaves: usize,
    pub max_in_degree: usize,
    pub avg_in_degree: f64,
    pub max_out_degree: usize,
    pub avg_out_degree: f64,
}

/// Precomputed data for efficient traversal using `special_map_grouped`.
#[derive(Debug, Clone)]
pub struct TrieTraversalData {
    /// The nodes reachable from the roots, in a fixed order.
    nodes: Vec<Trie2Index>,
    /// A map from a node's `usize` index to its position in the `nodes` vector.
    pos_of_u: HashMap<usize, usize>,
    /// A map from a node's position in `nodes` to its SCC ID.
    comp_id: Vec<usize>,
    /// The list of SCCs. Each inner vector contains node positions.
    sccs: Vec<Vec<usize>>,
    /// The topologically sorted list of SCC IDs.
    topo: Vec<usize>,
}

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
        let guard = arena.inner.read();
        if guard.values.get(self.index.as_usize())?.is_none() {
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
        let guard = arena.inner.write();
        if guard.values.get(self.index.as_usize())?.is_none() {
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
    guard: RwLockReadGuard<'a, ArenaInner<Trie<EK, EV, T>>>,
    index: usize,
}

impl<'a, EK: Ord, EV, T> Deref for Trie2ReadGuard<'a, EK, EV, T> {
    type Target = Trie<EK, EV, T>;
    fn deref(&self) -> &Self::Target {
        self.guard.values[self.index]
            .as_ref()
            .expect("Trie2ReadGuard: index not found in arena map")
    }
}

/// A write guard that keeps the Arena's internal RwLockWriteGuard alive and provides
/// mutable access to a Trie node at a given index via Deref/DerefMut.
pub struct Trie2WriteGuard<'a, EK: Ord, EV, T> {
    guard: RwLockWriteGuard<'a, ArenaInner<Trie<EK, EV, T>>>,
    index: usize,
}

impl<'a, EK: Ord, EV, T> Deref for Trie2WriteGuard<'a, EK, EV, T> {
    type Target = Trie<EK, EV, T>;
    fn deref(&self) -> &Self::Target {
        self.guard.values[self.index]
            .as_ref()
            .expect("Trie2WriteGuard: index not found in arena map")
    }
}

impl<'a, EK: Ord, EV, T> DerefMut for Trie2WriteGuard<'a, EK, EV, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.values[self.index]
            .as_mut()
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
    fn force_insert_to_new_node(
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
    fn force_insert_to_node(&mut self, edge_key: EK, edge_value: EV, dst: Trie2Index) {
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
        let mut inner_guard = arena.inner.write();
        for i in 0..inner_guard.values.len() {
            if !live_nodes_set.contains(&i) {
                inner_guard.values[i] = None;
            }
        }
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

    /// Detects if there are any cycles in the graph reachable from the given roots.
    /// This uses a depth-first search approach.
    pub fn has_cycle(
        arena: &Arena<Trie<EK, EV, T>>,
        roots: impl IntoIterator<Item = Trie2Index>,
    ) -> bool {
        let mut visiting = HashSet::new(); // Gray set: nodes currently in the recursion stack.
        let mut visited = HashSet::new();  // Black set: nodes that have been fully explored.

        for root in roots {
            if !visited.contains(&root.as_usize()) {
                if Self::detect_cycle_recursive(root, arena, &mut visiting, &mut visited) {
                    return true;
                }
            }
        }
        false
    }

    /// Recursive helper for cycle detection.
    fn detect_cycle_recursive(
        node_idx: Trie2Index,
        arena: &Arena<Trie<EK, EV, T>>,
        visiting: &mut HashSet<usize>,
        visited: &mut HashSet<usize>,
    ) -> bool {
        let u = node_idx.as_usize();
        visiting.insert(u);

        let children_indices: Vec<Trie2Index> = if let Some(guard) = node_idx.read(arena) {
            guard.children.values().flat_map(|m| m.keys().cloned()).collect()
        } else {
            Vec::new()
        };

        for child_idx in children_indices {
            let v = child_idx.as_usize();
            if visiting.contains(&v) {
                return true; // Cycle detected: found a back edge to a node in the current recursion stack.
            }
            if !visited.contains(&v) {
                if Self::detect_cycle_recursive(child_idx, arena, visiting, visited) {
                    return true;
                }
            }
        }

        visiting.remove(&u);
        visited.insert(u);
        false
    }

    /// Computes statistics for the graph reachable from the given roots.
    pub fn stats(arena: &Arena<Self>, roots: &[Trie2Index]) -> TrieStats {
        if roots.is_empty() {
            return TrieStats {
                num_reachable_nodes: 0,
                num_reachable_edges: 0,
                max_depth: 0,
                num_roots: 0,
                num_leaves: 0,
                max_in_degree: 0,
                avg_in_degree: 0.0,
                max_out_degree: 0,
                avg_out_degree: 0.0,
            };
        }

        let unique_roots: HashSet<_> = roots.iter().cloned().collect();
        let num_unique_roots = unique_roots.len();

        let reachable_nodes = Self::all_nodes(arena, roots);
        let num_reachable_nodes = reachable_nodes.len();

        if num_reachable_nodes == 0 {
            return TrieStats {
                num_reachable_nodes: 0,
                num_reachable_edges: 0,
                max_depth: 0,
                num_roots: num_unique_roots,
                num_leaves: 0,
                max_in_degree: 0,
                avg_in_degree: 0.0,
                max_out_degree: 0,
                avg_out_degree: 0.0,
            };
        }

        let mut num_reachable_edges = 0;
        let mut max_depth = 0;
        let mut num_leaves = 0;
        let mut total_out_degree = 0;
        let mut max_out_degree = 0;
        let mut in_degrees: HashMap<usize, usize> = HashMap::new();

        for node_idx in &reachable_nodes {
            let guard = node_idx
                .read(arena)
                .expect("Node not found during stats calculation");

            max_depth = max_depth.max(guard.max_depth);

            let mut out_degree = 0;
            for children_map in guard.children.values() {
                for (child_idx, _) in children_map.iter() {
                    out_degree += 1;
                    let v = child_idx.as_usize();
                    *in_degrees.entry(v).or_insert(0) += 1;
                }
            }

            if out_degree == 0 {
                num_leaves += 1;
            }
            num_reachable_edges += out_degree;
            total_out_degree += out_degree;
            max_out_degree = max_out_degree.max(out_degree);
        }

        let max_in_degree = in_degrees.values().cloned().max().unwrap_or(0);
        let total_in_degree: usize = in_degrees.values().sum();

        let avg_in_degree = total_in_degree as f64 / num_reachable_nodes as f64;
        let avg_out_degree = total_out_degree as f64 / num_reachable_nodes as f64;

        TrieStats {
            num_reachable_nodes,
            num_reachable_edges,
            max_depth,
            num_roots: num_unique_roots,
            num_leaves,
            max_in_degree,
            avg_in_degree,
            max_out_degree,
            avg_out_degree,
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

        if Arc::ptr_eq(&arena_a.inner, &arena_b.inner) && a_u == b_u {
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

    /// TEST UTILITY: Traverses the graph from the given roots and returns all
    /// possible paths that end in a node satisfying the `is_end` predicate.
    ///
    /// A "path" is a tuple containing the value of the root node and a vector of
    /// (edge key, edge value, destination node value) tuples. This function
    /// correctly handles DAGs and cycles.
    pub fn get_all_paths<F>(
        arena: &Arena<Self>,
        roots: &[Trie2Index],
        is_end: F,
    ) -> Vec<(T, Vec<(EK, EV, T)>)>
    where
        F: Fn(Trie2Index, &Trie<EK, EV, T>) -> bool,
        T: Clone,
        EV: Clone,
    {
        let mut all_paths = Vec::new();

        for &root in roots {
            let mut visiting = std::collections::HashSet::new();
            if let Some(root_guard) = root.read(arena) {
                let root_value = root_guard.value.clone();
                Self::get_all_paths_recursive(
                    arena,
                    root,
                    vec![],
                    &mut all_paths,
                    &mut visiting,
                    &is_end,
                    root_value,
                );
            }
        }
        all_paths
    }

    fn get_all_paths_recursive<F>(
        arena: &Arena<Self>,
        node_idx: Trie2Index,
        current_path: Vec<(EK, EV, T)>,
        all_paths: &mut Vec<(T, Vec<(EK, EV, T)>)>,
        visiting: &mut std::collections::HashSet<Trie2Index>,
        is_end: &F,
        root_value: T,
    ) where
        F: Fn(Trie2Index, &Trie<EK, EV, T>) -> bool,
        T: Clone,
        EV: Clone,
    {
        if !visiting.insert(node_idx) {
            // Cycle detected, stop this path.
            return;
        }

        if let Some(guard) = node_idx.read(arena) {
            if is_end(node_idx, &guard) {
                all_paths.push((root_value, current_path));
            } else {
                for (edge_key, dest_map) in guard.children() {
                    for (child_idx, edge_value) in dest_map.iter() {
                        if let Some(child_guard) = child_idx.read(arena) {
                            let mut new_path = current_path.clone();
                            let child_value = child_guard.value.clone();
                            new_path.push((edge_key.clone(), edge_value.clone(), child_value));
                            Self::get_all_paths_recursive(
                                arena,
                                *child_idx,
                                new_path,
                                all_paths,
                                visiting,
                                is_end,
                                root_value.clone(),
                            );
                        }
                    }
                }
            }
        }

        visiting.remove(&node_idx);
    }

    /// TEST UTILITY: Traverses the graph from the given roots and returns all
    /// possible paths that end in a node satisfying the `is_end` predicate.
    /// This version correctly handles cycles by allowing nodes to be revisited.
    ///
    /// A "path" is a tuple containing the value of the root node and a vector of
    /// (edge key, edge value, destination node value) tuples representing the edges.
    /// A path with zero edges (i.e., just the root node) is returned if the root
    /// itself satisfies the `is_end` predicate.
    ///
    /// # Arguments
    /// * `arena`: The arena containing the graph.
    /// * `roots`: The starting nodes for path traversal.
    /// * `is_end`: A predicate that returns true if a node is a valid end point for a path.
    ///             A path is only returned if its final node satisfies this predicate.
    /// * `is_path_edge`: A predicate that decides whether an edge should be included in a returned path.
    ///             Edges for which this returns `false` are still traversed to find further path segments,
    ///             but they are not included in the output path vector, nor do they count towards `max_path_length`.
    ///             This allows for modeling "transient" or "zero-cost" edges.
    ///             Cycle prevention uses an "active stamp" per node keyed by the path edge count.
    ///             This blocks infinite loops made solely of non-path edges while allowing revisits
    ///             after making progress on path edges.
    /// * `max_path_length`: The maximum length of a path (number of edges) to explore.
    pub fn get_all_paths_with_cycles<F, G>(
        arena: &Arena<Self>,
        roots: &[Trie2Index],
        is_end: F,
        is_path_edge: G,
        max_path_length: usize,
    ) -> Vec<(T, Vec<(EK, EV, T)>)>
    where
        F: Fn(Trie2Index, &Trie<EK, EV, T>) -> bool,
        G: Fn(&EK, &EV, Trie2Index) -> bool,
        T: Clone,
        EV: Clone,
    {
        let mut all_paths = Vec::new();

        for &root in roots {
            if let Some(root_guard) = root.read(arena) {
                let root_value = root_guard.value.clone();
                Self::get_all_paths_with_cycles_recursive(
                    arena,
                    root,
                    &mut vec![],
                    &mut all_paths,
                    &is_end,
                    &is_path_edge,
                    &root_value,
                    max_path_length,
                );
            }
        }
        all_paths
    }

    fn get_all_paths_with_cycles_recursive<F, G>(
        arena: &Arena<Self>,
        node_idx: Trie2Index,
        current_path: &mut Vec<(EK, EV, T)>,
        all_paths: &mut Vec<(T, Vec<(EK, EV, T)>)>,
        is_end: &F,
        is_path_edge: &G,
        root_value: &T,
        max_path_length: usize,
    ) where
        F: Fn(Trie2Index, &Trie<EK, EV, T>) -> bool,
        G: Fn(&EK, &EV, Trie2Index) -> bool,
        T: Clone,
        EV: Clone,
    {
        // Iterative DFS to avoid stack overflows on deep/large graphs.
        // Semantics:
        // - Record a path when is_end(current_node) is true (including the root with a zero-length path).
        // - Enforce max_path_length AFTER recording the current node as an endpoint, based on counted edges.
        // - Allow revisiting nodes (cycles are allowed) but stop expanding when the counted-edge path length reaches max_path_length.
        // - Prevent infinite loops formed solely by uncounted edges using an "active stamps" map:
        //   a node cannot be revisited at the same counted length (stamp), but can be revisited after progress.
        // Work on a local copy of the path, so callers' mutable reference remains unchanged.
        let mut path: Vec<(EK, EV, T)> = current_path.clone();
        // Stack frame for iterative DFS.
        struct Frame<EK2, EV2> {
            node: Trie2Index,
            // Snapshot of edges in deterministic order: (edge_key, edge_value, child_idx)
            edges: Vec<(EK2, EV2, Trie2Index)>,
            idx: usize,                 // Next edge index to process
            started: bool,              // Whether we've done the "node entry" work (is_end + length check)
            has_incoming_edge: bool,    // Whether entering this frame pushed an edge onto `path`
            // "Stamp" equals counted-edge length (path.len()) at entry; used to manage active stamps.
            stamp: usize,
        }
        // Helper to snapshot a node's outgoing edges in deterministic order (no locks held after).
        let edges_for = |n: Trie2Index| -> Vec<(EK, EV, Trie2Index)> {
            if let Some(g) = n.read(arena) {
                g.children().iter()
                    .flat_map(|(ek, dest_map)| {
                        let ekc = ek.clone();
                        dest_map.iter()
                            .map(move |(child_idx, ev)| (ekc.clone(), ev.clone(), *child_idx))
                    })
                    .collect()
            } else {
                Vec::new()
            }
        };
        let mut stack: Vec<Frame<EK, EV>> = Vec::new();
        // Active stamps: for each node, which counted-length stamps are currently on the path.
        let mut active: HashMap<Trie2Index, Vec<usize>> = HashMap::new();
        stack.push(Frame {
            node: node_idx,
            edges: edges_for(node_idx),
            idx: 0,
            started: false,
            has_incoming_edge: false, // root has no incoming edge
            stamp: 0,                  // root is entered at counted length 0
        });
        active.entry(node_idx).or_default().push(0);
        while let Some(frame) = stack.last_mut() {
            // On first entry to this node: evaluate `is_end` and enforce max_path_length.
            if !frame.started {
                frame.started = true;
                if let Some(g) = frame.node.read(arena) {
                    if is_end(frame.node, &g) {
                        all_paths.push((root_value.clone(), path.clone()));
                    }
                }
                // If we've reached the max number of edges on this path, do not expand further.
                if path.len() >= max_path_length {
                    let popped = stack.pop().unwrap();
                    if let Some(vec) = active.get_mut(&popped.node) {
                        if let Some(last) = vec.pop() {
                            debug_assert!(last == popped.stamp);
                        }
                        if vec.is_empty() {
                            active.remove(&popped.node);
                        }
                    }
                    if popped.has_incoming_edge {
                        path.pop(); // backtrack edge added when entering this frame
                    }
                    continue;
                }
            }
            // If we've exhausted edges for this node, backtrack.
            if frame.idx >= frame.edges.len() {
                let popped = stack.pop().unwrap();
                if let Some(vec) = active.get_mut(&popped.node) {
                    if let Some(last) = vec.pop() {
                        debug_assert!(last == popped.stamp);
                    }
                    if vec.is_empty() {
                        active.remove(&popped.node);
                    }
                }
                if popped.has_incoming_edge {
                    path.pop(); // backtrack the edge that led here
                }
                continue;
            }
            // Take the next outgoing edge and descend to the child.
            let (ek, ev, child_idx) = frame.edges[frame.idx].clone();
            frame.idx += 1;
            // Determine whether this edge increases the counted length.
            let include_edge = is_path_edge(&ek, &ev, child_idx);
            let next_stamp = path.len() + if include_edge { 1 } else { 0 };

            // Prevent infinite loops formed solely by uncounted edges:
            // do not re-enter the same node at the same counted length.
            if active.get(&child_idx).map_or(false, |v| v.contains(&next_stamp)) {
                continue;
            }
            if let Some(child_guard) = child_idx.read(arena) {
                let child_value = child_guard.value.clone();
                if include_edge {
                    // Add this edge to the current path before exploring the child.
                    path.push((ek, ev, child_value));
                }
                // Snapshot child's edges now (no locks held across iterations).
                let child_edges = edges_for(child_idx);
                stack.push(Frame {
                    node: child_idx,
                    edges: child_edges,
                    idx: 0,
                    started: false,
                    has_incoming_edge: include_edge,
                    stamp: next_stamp,
                });
                active.entry(child_idx).or_default().push(next_stamp);
            } else {
                // If child is missing, skip it and proceed.
                continue;
            }
        }
    }
}

/// Options for customizing the output of `pretty_print_with_options`.
pub struct PrettyPrintOptions<'a, EK, EV, T> {
    /// Whether to show the `max_depth` of each node.
    pub show_max_depth: bool,
    /// A closure to format the value `T` of a node.
    /// It receives the node's index and value, and can return a string to be appended.
    pub format_node: Box<dyn Fn(Trie2Index, &T) -> Option<String> + 'a>,
    /// A closure to format the edge information (key `EK` and value `EV`).
    /// It receives source and destination indices, and the edge key/value.
    pub format_edge: Box<dyn Fn(Trie2Index, Trie2Index, &EK, &EV) -> Option<String> + 'a>,
}

impl<'a, EK: Debug, EV: Debug, T> Default for PrettyPrintOptions<'a, EK, EV, T> {
    fn default() -> Self {
        PrettyPrintOptions {
            show_max_depth: true,
            format_node: Box::new(|_idx, _val| None),
            format_edge: Box::new(|_src, _dst, ek, ev| Some(format!("{:?}, {:?}", ek, ev))),
        }
    }
}

impl<'a, EK, EV, T> PrettyPrintOptions<'a, EK, EV, T> {
    /// Sets the node value formatter to use `T: Debug`.
    pub fn debug_nodes(mut self) -> Self
    where
        T: Debug,
    {
        self.format_node = Box::new(|_idx, val| Some(format!("{:?}", val)));
        self
    }

    /// Sets the node value formatter to use `T: Display`.
    pub fn display_nodes(mut self) -> Self
    where
        T: Display,
    {
        self.format_node = Box::new(|_idx, val| Some(format!("{}", val)));
        self
    }

    /// Sets the edge formatter to use `EK: Debug` and `EV: Debug`.
    pub fn debug_edges(mut self) -> Self
    where
        EK: Debug,
        EV: Debug,
    {
        self.format_edge = Box::new(|_src, _dst, ek, ev| Some(format!("{:?}, {:?}", ek, ev)));
        self
    }

    /// Sets the edge formatter to use `EK: Display` and `EV: Display`.
    pub fn display_edges(mut self) -> Self
    where
        EK: Display,
        EV: Display,
    {
        self.format_edge = Box::new(|_src, _dst, ek, ev| Some(format!("{}, {}", ek, ev)));
        self
    }

    /// Sets the edge formatter to use only `EK: Display`.
    pub fn display_edge_keys_only(mut self) -> Self
    where
        EK: Display,
    {
        self.format_edge = Box::new(|_src, _dst, ek, _ev| Some(format!("{}", ek)));
        self
    }

    /// Sets the edge formatter to use only `EK: Debug`.
    pub fn debug_edge_keys_only(mut self) -> Self
    where
        EK: Debug,
    {
        self.format_edge = Box::new(|_src, _dst, ek, _ev| Some(format!("{:?}", ek)));
        self
    }

    /// Sets the edge formatter to use only `EV: Display`.
    pub fn display_edge_values_only(mut self) -> Self
    where
        EV: Display,
    {
        self.format_edge = Box::new(|_src, _dst, _ek, ev| Some(format!("{}", ev)));
        self
    }

    /// Sets the edge formatter to use only `EV: Debug`.
    pub fn debug_edge_values_only(mut self) -> Self
    where
        EV: Debug,
    {
        self.format_edge = Box::new(|_src, _dst, _ek, ev| Some(format!("{:?}", ev)));
        self
    }

    /// Sets both node and edge formatters to use `Debug`.
    pub fn debug_all(mut self) -> Self
    where
        T: Debug,
        EK: Debug,
        EV: Debug,
    {
        self = self.debug_nodes();
        self = self.debug_edges();
        self
    }

    /// Sets both node and edge formatters to use `Display`.
    pub fn display_all(mut self) -> Self
    where
        T: Display,
        EK: Display,
        EV: Display,
    {
        self = self.display_nodes();
        self = self.display_edges();
        self
    }

    /// Sets the node value formatter to always return `None`, omitting node values from the output.
    pub fn omit_nodes(mut self) -> Self {
        self.format_node = Box::new(|_idx, _val| None);
        self
    }

    /// Sets the edge formatter to always return `None`, omitting edge information from the output.
    pub fn omit_edges(mut self) -> Self {
        self.format_edge = Box::new(|_src, _dst, _ek, _ev| None);
        self
    }

    /// Sets both node and edge formatters to always return `None`, omitting all custom formatting.
    pub fn omit_all(mut self) -> Self {
        self = self.omit_nodes();
        self = self.omit_edges();
        self
    }

    /// Sets `show_max_depth` to `false`, omitting the max depth from the output.
    pub fn omit_depth(mut self) -> Self {
        self.show_max_depth = false;
        self
    }
}

// Add this impl block for pretty-printing functionality
impl<EK, EV, T> Trie<EK, EV, T>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Debug,
{
    /// Pretty-prints the trie structure starting from the given roots into a String.
    /// Handles shared subtrees and cycles to avoid infinite loops and redundant output.
        pub fn pretty_print(arena: &Arena<Self>, roots: &[Trie2Index]) -> String {
        let mut output = String::new();
        let mut printed_nodes = HashSet::new();

        for (i, &root) in roots.iter().enumerate() {
            if i > 0 {
                output.push_str("\n");
            }
            output.push_str(&format!("[Root {}]\n", i));
            let mut visiting = HashSet::new();
            Self::pretty_print_recursive(root, arena, "", true, &mut visiting, &mut printed_nodes, &mut output);
        }
        output
    }

    /// Pretty-prints the entire trie structure in the arena.
    /// It automatically identifies root nodes (those with an in-degree of 0)
    /// and prints from there. If no roots are found (e.g., a graph of only cycles),
    /// it will print from all nodes.
    pub fn pretty_print_arena(arena: &Arena<Self>) -> String {
        let all_node_indices: Vec<Trie2Index> =
            arena.indices().into_iter().map(Trie2Index::from).collect();
        if all_node_indices.is_empty() {
            return "[Arena is empty]\n".to_string();
        }

        let mut in_degrees: std::collections::HashMap<Trie2Index, usize> =
            all_node_indices.iter().map(|&idx| (idx, 0)).collect();

        for &node_idx in &all_node_indices {
            if let Some(guard) = node_idx.read(arena) {
                for dest_map in guard.children.values() {
                    for &child_idx in dest_map.keys() {
                        if let Some(degree) = in_degrees.get_mut(&child_idx) {
                            *degree += 1;
                        }
                    }
                }
            }
        }

        let mut roots: Vec<Trie2Index> = in_degrees
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(&idx, _)| idx)
            .collect();

        roots.sort();

        if roots.is_empty() {
            // This case handles graphs with no nodes of in-degree 0 (e.g., all nodes are in cycles).
            let mut output = String::from("[Warning: No nodes with in-degree 0 found. Graph may contain only cycles.]\n[Printing from all nodes as roots.]\n\n");
            let mut sorted_nodes = all_node_indices;
            sorted_nodes.sort(); // Sort for deterministic output
            output.push_str(&Self::pretty_print(arena, &sorted_nodes));
            return output;
        }

        Self::pretty_print(arena, &roots)
    }

    fn pretty_print_recursive(
        node_idx: Trie2Index,
        arena: &Arena<Self>,
        prefix: &str,
        is_last: bool,
        visiting: &mut HashSet<usize>,
        printed_nodes: &mut HashSet<usize>,
        output: &mut String,
    ) {
        let node_guard = match node_idx.read(arena) {
            Some(guard) => guard,
            None => {
                let connector = if is_last { "└── " } else { "├── " };
                output.push_str(&format!("{}{}[Invalid Node Index: {}]\n", prefix, connector, node_idx.as_usize()));
                return;
            }
        };

        let connector = if is_last { "└── " } else { "├── " };
        output.push_str(&format!(
            "{}{}[Node {}] (max_depth: {}, value: {:?})",
            prefix,
            connector,
            node_idx.as_usize(),
            node_guard.max_depth,
            &node_guard.value
        ));

        if visiting.contains(&node_idx.as_usize()) {
            output.push_str(" (Cycle detected)\n");
            return;
        }

        if printed_nodes.contains(&node_idx.as_usize()) {
            output.push_str(" (Shared, already shown)\n");
            return;
        }
        output.push_str("\n");

        visiting.insert(node_idx.as_usize());
        printed_nodes.insert(node_idx.as_usize());

        let new_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

        let children_edges: Vec<_> = node_guard.children.iter().flat_map(|(ek, dest_map)| dest_map.iter().map(move |(child_idx, ev)| (ek, ev, *child_idx))).collect();

        let num_children = children_edges.len();
        for (i, (ek, ev, child_idx)) in children_edges.iter().enumerate() {
            let is_last_child = i == num_children - 1;
            let child_connector = if is_last_child { "└── " } else { "├── " };
            output.push_str(&format!("{}{}Edge: {:?} (value: {:?})\n", &new_prefix, child_connector, ek, ev));
            let recursive_prefix = format!("{}{}", &new_prefix, if is_last_child { "    " } else { "│   " });
            Self::pretty_print_recursive(*child_idx, arena, &recursive_prefix, true, visiting, printed_nodes, output);
        }

        visiting.remove(&node_idx.as_usize());
    }

    /// Pretty-prints the trie structure with advanced formatting options.
    pub fn pretty_print_with_options(
        arena: &Arena<Self>,
        roots: &[Trie2Index],
        options: &PrettyPrintOptions<EK, EV, T>,
    ) -> String {
        let mut output = String::new();
        let mut printed_nodes = HashSet::new();

        for (i, &root) in roots.iter().enumerate() {
            if i > 0 {
                output.push_str("\n");
            }
            output.push_str(&format!("--- Root State ID: {} ---\n", i));

            let root_guard = match root.read(arena) {
                Some(g) => g,
                None => {
                    output.push_str(&format!("Root Node {} [Invalid Index]\n", root.as_usize()));
                    continue;
                }
            };

            let mut root_line = format!("Root Node {}", root.as_usize());
            if options.show_max_depth {
                root_line.push_str(&format!(" (max_depth: {})", root_guard.max_depth));
            }
            if let Some(formatted) = (options.format_node)(root, &root_guard.value) {
                root_line.push_str(&format!(" {}", formatted));
            }
            output.push_str(&root_line);
            output.push_str("\n");

            printed_nodes.insert(root.as_usize());
            let mut visiting = HashSet::new();
            visiting.insert(root.as_usize());

            Self::pretty_print_children_recursive(
                root,
                arena,
                "",
                &mut visiting,
                &mut printed_nodes,
                &mut output,
                options,
            );
        }
        output
    }

    /// Pretty-prints the entire trie structure in the arena with advanced formatting options.
    pub fn pretty_print_arena_with_options(
        arena: &Arena<Self>,
        options: &PrettyPrintOptions<EK, EV, T>,
    ) -> String {
        let all_node_indices: Vec<Trie2Index> =
            arena.indices().into_iter().map(Trie2Index::from).collect();
        if all_node_indices.is_empty() {
            return "[Arena is empty]\n".to_string();
        }

        let mut in_degrees: std::collections::HashMap<Trie2Index, usize> =
            all_node_indices.iter().map(|&idx| (idx, 0)).collect();

        for &node_idx in &all_node_indices {
            if let Some(guard) = node_idx.read(arena) {
                for dest_map in guard.children.values() {
                    for &child_idx in dest_map.keys() {
                        if let Some(degree) = in_degrees.get_mut(&child_idx) {
                            *degree += 1;
                        }
                    }
                }
            }
        }

        let mut roots: Vec<Trie2Index> = in_degrees
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(&idx, _)| idx)
            .collect();

        roots.sort();

        if roots.is_empty() {
            // This case handles graphs with no nodes of in-degree 0 (e.g., all nodes are in cycles).
            let mut output = String::from("[Warning: No nodes with in-degree 0 found. Graph may contain only cycles.]\n[Printing from all nodes as roots.]\n\n");
            let mut sorted_nodes = all_node_indices;
            sorted_nodes.sort();
            output.push_str(&Self::pretty_print_with_options(arena, &sorted_nodes, options));
            return output;
        }

        Self::pretty_print_with_options(arena, &roots, options)
    }

    fn pretty_print_children_recursive(
        node_idx: Trie2Index,
        arena: &Arena<Self>,
        prefix: &str,
        visiting: &mut HashSet<usize>,
        printed_nodes: &mut HashSet<usize>,
        output: &mut String,
        options: &PrettyPrintOptions<EK, EV, T>,
    ) {
        let node_guard = match node_idx.read(arena) {
            Some(guard) => guard,
            None => return, // Should have been handled before call
        };

        let children_edges: Vec<_> = node_guard
            .children
            .iter()
            .flat_map(|(ek, dest_map)| {
                dest_map
                    .iter()
                    .map(move |(child_idx, ev)| (ek, ev, *child_idx))
            })
            .collect();

        let num_children = children_edges.len();
        for (i, (ek, ev, child_idx)) in children_edges.iter().enumerate() {
            let is_last_child = i == num_children - 1;
            let connector = if is_last_child { "└── " } else { "├── " };

            let mut line = format!("{}{}", prefix, connector);
            if let Some(edge_str) = (options.format_edge)(node_idx, *child_idx, ek, ev) {
                line.push_str(&format!("Edge ({}): -> ", edge_str));
            } else {
                line.push_str("Edge: -> ");
            }
            line.push_str(&format!("Node {}", child_idx.as_usize()));

            let child_guard = match child_idx.read(arena) {
                Some(g) => g,
                None => {
                    line.push_str(" [Invalid Index]");
                    output.push_str(&line);
                    output.push_str("\n");
                    continue;
                }
            };

            if options.show_max_depth {
                line.push_str(&format!(" (max_depth: {})", child_guard.max_depth));
            }
            if let Some(node_str) = (options.format_node)(*child_idx, &child_guard.value) {
                line.push_str(&format!(" {}", node_str));
            }

            if visiting.contains(&child_idx.as_usize()) {
                line.push_str(" (Cycle detected)");
                output.push_str(&line);
                output.push_str("\n");
                continue;
            }
            if printed_nodes.contains(&child_idx.as_usize()) {
                line.push_str(" (Shared, already shown)");
                output.push_str(&line);
                output.push_str("\n");
                continue;
            }

            output.push_str(&line);
            output.push_str("\n");

            visiting.insert(child_idx.as_usize());
            printed_nodes.insert(child_idx.as_usize());

            let new_prefix = format!("{}{}", prefix, if is_last_child { "    " } else { "│   " });
            Self::pretty_print_children_recursive(
                *child_idx,
                arena,
                &new_prefix,
                visiting,
                printed_nodes,
                output,
                options,
            );

            visiting.remove(&child_idx.as_usize());
        }
    }
}

// Implementation block for special_map and related functionality
// Requires T: Clone, EK: Ord + Clone, EV: Clone
impl<T: Clone, EK: Ord + Clone, EV: Clone> Trie<EK, EV, T> {
    /// Deep copies the subtrees rooted at `roots` from `source_arena` into a new `Arena`.
    ///
    /// This function performs a full traversal from the given roots and duplicates all
    /// reachable nodes and edges into a new `Arena`. It correctly handles shared subtrees
    /// (copying them only once) and cycles.
    ///
    /// # Arguments
    /// * `source_arena`: The arena containing the original graph.
    /// * `roots`: A slice of `Trie2Index` pointing to the root nodes of the subtrees to copy.
    ///
    /// # Returns
    /// A tuple containing:
    /// - A new `Arena<Trie<EK, EV, T>>` with the copied subtrees.
    /// - A `Vec<Trie2Index>` containing the indices of the new roots in the new arena, in the
    ///   same order as the input `roots`.
    /// - A `HashMap<Trie2Index, Trie2Index>` mapping old node indices to new node indices.
    pub fn deep_copy_subtrees(
        source_arena: &Arena<Self>,
        roots: &[Trie2Index],
    ) -> (Arena<Self>, Vec<Trie2Index>, HashMap<Trie2Index, Trie2Index>) {
        let new_arena = Arena::new();
        let mut old_to_new_map: HashMap<Trie2Index, Trie2Index> = HashMap::new();
        let mut new_roots = Vec::with_capacity(roots.len());

        for &root in roots {
            let new_root =
                Self::deep_copy_recursive(root, source_arena, &new_arena, &mut old_to_new_map);
            new_roots.push(new_root);
        }

        (new_arena, new_roots, old_to_new_map)
    }

    /// Deep copies the subtrees rooted at `roots` from `source_arena` into `dest_arena`.
    ///
    /// This function performs a full traversal from the given roots and duplicates all
    /// reachable nodes and edges from `source_arena` into `dest_arena`. It correctly
    /// handles shared subtrees (copying them only once) and cycles.
    ///
    /// # Arguments
    /// * `source_arena`: The arena containing the original graph.
    /// * `dest_arena`: The arena to copy the subtrees into.
    /// * `roots`: A slice of `Trie2Index` pointing to the root nodes of the subtrees to copy.
    ///
    /// # Returns
    /// A tuple containing:
    /// - A `Vec<Trie2Index>` containing the indices of the new roots in the `dest_arena`,
    ///   in the same order as the input `roots`.
    /// - A `HashMap<Trie2Index, Trie2Index>` mapping old node indices to new node indices.
    pub fn deep_copy_subtrees_into(
        source_arena: &Arena<Self>,
        dest_arena: &Arena<Self>,
        roots: &[Trie2Index],
    ) -> (Vec<Trie2Index>, HashMap<Trie2Index, Trie2Index>) {
        let mut old_to_new_map: HashMap<Trie2Index, Trie2Index> = HashMap::new();
        let mut new_roots = Vec::with_capacity(roots.len());

        for &root in roots {
            let new_root =
                Self::deep_copy_recursive(root, source_arena, dest_arena, &mut old_to_new_map);
            new_roots.push(new_root);
        }

        (new_roots, old_to_new_map)
    }

    /// Recursive helper for `deep_copy_subtrees`.
    fn deep_copy_recursive(
        old_idx: Trie2Index,
        source_arena: &Arena<Self>,
        new_arena: &Arena<Self>,
        old_to_new_map: &mut HashMap<Trie2Index, Trie2Index>,
    ) -> Trie2Index {
        if let Some(&new_idx) = old_to_new_map.get(&old_idx) {
            return new_idx;
        }

        // 1. Read data from old node and drop the lock immediately.
        let (value, max_depth, children_to_copy) = {
            let old_node_guard = old_idx
                .read(source_arena)
                .expect("Source node not found during deep copy");

            // Now, collect children info to avoid holding the read guard during recursive calls.
            let children_to_copy: Vec<(EK, OrderedHashMap<Trie2Index, EV>)> = old_node_guard
                .children
                .iter()
                .map(|(ek, dest_map)| (ek.clone(), dest_map.clone()))
                .collect();

            (
                old_node_guard.value.clone(),
                old_node_guard.max_depth,
                children_to_copy,
            )
        }; // old_node_guard is dropped here, releasing the read lock on source_arena.inner

        // 2. Create the new node and insert it into the new arena.
        let new_node_skeleton = Trie {
            value,
            children: BTreeMap::new(), // Children will be added after recursive calls.
            max_depth,
        };
        let new_idx = Trie2Index::from(new_arena.insert(new_node_skeleton));

        // 3. Insert the mapping *before* recursing to handle cycles correctly.
        old_to_new_map.insert(old_idx, new_idx);

        // 4. Recurse for all children and build up the new children map.
        let mut new_children = BTreeMap::new();
        for (ek, dest_map) in children_to_copy {
            let mut new_dest_map = OrderedHashMap::with_capacity(dest_map.len());
            for (old_child_idx, ev) in dest_map.iter() {
                let new_child_idx =
                    Self::deep_copy_recursive(*old_child_idx, source_arena, new_arena, old_to_new_map);
                new_dest_map.insert(new_child_idx, ev.clone());
            }
            new_children.insert(ek, new_dest_map);
        }

        // 5. Update the new node with the children map.
        let mut new_node_guard = new_idx
            .write(new_arena)
            .expect("Newly created node not found during deep copy");
        new_node_guard.children = new_children;

        new_idx
    }

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

    /// Precomputes traversal data (SCCs, topological sort) for `special_map_grouped`.
    /// This is useful to avoid recomputing this data if `special_map_grouped` is called
    /// multiple times on the same graph structure.
    #[time_it]
    pub fn compute_traversal_data(
        arena: &Arena<Trie<EK, EV, T>>,
        initial_nodes: &[Trie2Index],
    ) -> Option<TrieTraversalData> {
        use std::collections::VecDeque;
        // Build reachable set and adjacency for SCC computation.
        let nodes: Vec<Trie2Index> = Self::all_nodes(arena, initial_nodes);
        if nodes.is_empty() {
            return None;
        }
        let n = nodes.len();
        let mut pos_of_u: HashMap<usize, usize> = HashMap::with_capacity(n);
        for (i, idx) in nodes.iter().enumerate() {
            pos_of_u.insert(idx.as_usize(), i);
        }

        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut radj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, idx) in nodes.iter().enumerate() {
            if let Some(g) = idx.read(arena) {
                for dest_map in g.children.values() {
                    for (child_idx, _) in dest_map.iter() {
                        if let Some(&j) = pos_of_u.get(&child_idx.as_usize()) {
                            adj[i].push(j);
                            radj[j].push(i);
                        }
                    }
                }
            }
        }

        // Kosaraju (iterative) to compute SCCs.
        let mut visited = vec![false; n];
        let mut order: Vec<usize> = Vec::with_capacity(n);
        for u in 0..n {
            if !visited[u] {
                let mut stack: Vec<(usize, usize)> = vec![(u, 0)];
                visited[u] = true;
                while let Some((node, next_i)) = stack.last_mut() {
                    if *next_i < adj[*node].len() {
                        let v = adj[*node][*next_i];
                        *next_i += 1;
                        if !visited[v] {
                            visited[v] = true;
                            stack.push((v, 0));
                        }
                    } else {
                        order.push(*node);
                        stack.pop();
                    }
                }
            }
        }

        let mut comp_id = vec![usize::MAX; n];
        let mut cid = 0;
        for &u in order.iter().rev() {
            if comp_id[u] == usize::MAX {
                let mut stack: Vec<usize> = vec![u];
                comp_id[u] = cid;
                while let Some(x) = stack.pop() {
                    for &v in &radj[x] {
                        if comp_id[v] == usize::MAX {
                            comp_id[v] = cid;
                            stack.push(v);
                        }
                    }
                }
                cid += 1;
            }
        }

        let scc_count = cid;
        let mut sccs: Vec<Vec<usize>> = vec![Vec::new(); scc_count];
        for i in 0..n {
            sccs[comp_id[i]].push(i);
        }

        // Build condensation DAG of SCCs and topologically sort it.
        let mut scc_adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); scc_count];
        let mut indeg: Vec<usize> = vec![0; scc_count];
        for u in 0..n {
            let cu = comp_id[u];
            for &v in &adj[u] {
                let cv = comp_id[v];
                if cu != cv {
                    if scc_adj[cu].insert(cv) {
                        indeg[cv] += 1;
                    }
                }
            }
        }
        let mut topo: Vec<usize> = Vec::with_capacity(scc_count);
        let mut q_scc: VecDeque<usize> = VecDeque::new();
        for s in 0..scc_count {
            if indeg[s] == 0 {
                q_scc.push_back(s);
            }
        }
        while let Some(s) = q_scc.pop_front() {
            topo.push(s);
            for &t in &scc_adj[s] {
                indeg[t] -= 1;
                if indeg[t] == 0 {
                    q_scc.push_back(t);
                }
            }
        }

        Some(TrieTraversalData {
            nodes,
            pos_of_u,
            comp_id,
            sccs,
            topo,
        })
    }

    /// Performs a specialized breadth-first traversal, grouping children by edge key.
    /// This is more efficient than `special_map` when many edges share the same key,
    /// as the `step` function is called once per key, not once per edge.
    ///
    /// This version uses pre-computed traversal data for efficiency.
    #[time_it]
    pub fn special_map_grouped<V, S, I>(
        arena: &Arena<Trie<EK, EV, T>>,
        traversal_data: &TrieTraversalData,
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
        //  SCC-aware scheduler:
        //  - Process SCCs in topological order.
        //  - Inside each SCC, run a local worklist until stabilization.
        // ------------------------------------------------------------------
        use std::collections::VecDeque;

        let mut values: HashMap<usize, V> = HashMap::new();
        let mut stopped_nodes: HashSet<usize> = HashSet::new();

        let initial_nodes: Vec<_> = initial_nodes_and_values.iter().map(|(n, _)| *n).collect();
        let total_edges = Self::count_all_edges(arena, &initial_nodes);
        let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED, leave=false);
        for (node_idx, v0) in initial_nodes_and_values {
            let ptr = node_idx.as_usize();
            values
                .entry(ptr)
                .and_modify(|old| merge(old, v0.clone()))
                .or_insert(v0);
        }

        // Use pre-computed traversal data.
        let nodes = &traversal_data.nodes;
        let pos_of_u = &traversal_data.pos_of_u;
        let comp_id = &traversal_data.comp_id;
        let sccs = &traversal_data.sccs;
        let topo = &traversal_data.topo;

        // Worklist inside each SCC until stabilization; process SCCs in topological order.
        let mut in_queue: HashSet<usize> = HashSet::new(); // node.usize currently in the local SCC queue
        for &s in topo {
            // Seed local queue with nodes in this SCC that currently have pending values.
            let mut local_queue: VecDeque<usize> = VecDeque::new(); // holds positions (indices into `nodes`)
            for &pos in &sccs[s] {
                let u = nodes[pos].as_usize();
                if values.contains_key(&u) && !stopped_nodes.contains(&u) {
                    if in_queue.insert(u) {
                        local_queue.push_back(pos);
                    }
                }
            }
            if local_queue.is_empty() {
                continue; // nothing pending in this SCC yet
            }

            while let Some(pos) = local_queue.pop_front() {
                let node_idx = nodes[pos];
                let u = node_idx.as_usize();
                // We are about to process u; mark as not in queue until we decide to requeue.
                in_queue.remove(&u);

                if stopped_nodes.contains(&u) {
                    continue;
                }
                let mut agg_v = match values.remove(&u) {
                    Some(v) => v,
                    None => continue,
                };

                let proceed = {
                    let guard = node_idx.read(arena).expect("poison");
                    process(&guard, &mut agg_v)
                };
                if !proceed {
                    stopped_nodes.insert(u);
                    continue;
                }

                // Propagate to children grouped by edge key.
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
                        let child_u = child_idx.as_usize();
                        if stopped_nodes.contains(&child_u) {
                            continue;
                        }
                        values
                            .entry(child_u)
                            .and_modify(|old| merge(old, new_v.clone()))
                            .or_insert(new_v);

                        // If the child is in the same SCC, schedule immediately in local queue.
                        if let Some(&child_pos) = pos_of_u.get(&child_u) {
                            if comp_id[child_pos] == s {
                                if in_queue.insert(child_u) {
                                    local_queue.push_back(child_pos);
                                }
                            }
                            // If in a different SCC, it will be picked up when that SCC is reached in topo order.
                        }
                    }
                }

                // If new inputs accumulated for this node while it was processing, re-queue it to continue local fixpoint.
                if values.contains_key(&u) && !stopped_nodes.contains(&u) {
                    if in_queue.insert(u) {
                        local_queue.push_back(pos);
                    }
                }
            }
        }
    }

    /// Creates a deep copy of the trie and randomly removes edges.
    ///
    /// This is useful for fuzz testing to simplify a complex graph.
    ///
    /// # Arguments
    /// * `source_arena`: The original arena.
    /// * `roots`: The roots of the graph to trim.
    /// * `p`: The probability (between 0.0 and 1.0) of removing any given edge.
    /// * `rng`: A random number generator.
    ///
    /// # Returns
    /// A new, potentially smaller trie as `(Arena, Vec<Trie2Index>)`.
    pub fn trim_randomly<R: Rng + ?Sized>(
        source_arena: &Arena<Self>,
        roots: &[Trie2Index],
        p: f64,
        rng: &mut R,
    ) -> (Arena<Self>, Vec<Trie2Index>) {
        // 1. Deep copy the trie
        let (new_arena, new_roots, _) = Self::deep_copy_subtrees(source_arena, roots);
        if roots.is_empty() {
            return (new_arena, new_roots);
        }

        // 2. Collect all edges from the new trie
        let all_nodes = Self::all_nodes(&new_arena, &new_roots);
        let mut all_edges = Vec::new();
        for &node_idx in &all_nodes {
            if let Some(guard) = node_idx.read(&new_arena) {
                for (ek, dest_map) in guard.children() {
                    for (child_idx, _) in dest_map.iter() {
                        all_edges.push((node_idx, ek.clone(), *child_idx));
                    }
                }
            }
        }

        // 3. Randomly remove edges
        for (src, ek, dst) in all_edges {
            if rng.gen_bool(p) {
                new_arena.remove_edge(src, dst, &ek);
            }
        }

        // 4. Garbage collect unreachable nodes
        Self::gc(&new_arena, &new_roots);

        // 5. Return the trimmed trie
        (new_arena, new_roots)
    }

    /// Returns a set of all nodes that are part of any cycle in the graph reachable from `roots`.
    pub fn nodes_in_cycles(
        arena: &Arena<Self>,
        roots: &[Trie2Index],
    ) -> HashSet<Trie2Index> {
        let traversal_data = match Self::compute_traversal_data(arena, roots) {
            Some(data) => data,
            None => return HashSet::new(),
        };

        // Build adj list to check for self-loops in single-node SCCs.
        // This is duplicated from compute_traversal_data because adj is not returned from it.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); traversal_data.nodes.len()];
        for (i, idx) in traversal_data.nodes.iter().enumerate() {
            if let Some(g) = idx.read(arena) {
                for dest_map in g.children.values() {
                    for (child_idx, _) in dest_map.iter() {
                        if let Some(&j) = traversal_data.pos_of_u.get(&child_idx.as_usize()) {
                            adj[i].push(j);
                        }
                    }
                }
            }
        }

        let mut cyclic_nodes = HashSet::new();

        // An SCC is part of a cycle if it has more than one node, or if a single-node SCC has a self-loop.
        for scc_id in 0..traversal_data.sccs.len() {
            let scc_nodes_pos = &traversal_data.sccs[scc_id];
            if scc_nodes_pos.len() > 1 {
                for &pos in scc_nodes_pos {
                    cyclic_nodes.insert(traversal_data.nodes[pos]);
                }
            } else if scc_nodes_pos.len() == 1 {
                // Check for self-loop
                let pos = scc_nodes_pos[0];
                for &neighbor_pos in &adj[pos] {
                    if neighbor_pos == pos {
                        cyclic_nodes.insert(traversal_data.nodes[pos]);
                        break;
                    }
                }
            }
        }

        cyclic_nodes
    }
}

/// The result of comparing two paths in a Trie.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PathComparison {
    /// The first path is a prefix of the second path.
    Prefix,
    /// The two paths are equal.
    Equal,
    /// The paths are different and neither is a prefix of the other.
    Different,
}

/// Represents a path in a Trie, starting from a root.
/// It consists of the root's value and a sequence of edges and destination node values.
pub type TriePath<EK, EV, T> = (T, Vec<(EK, EV, T)>);

// Implementation block for stochastic equivalence testing
impl<EK, EV, T> Trie<EK, EV, T>
where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone + PartialEq,
{
    /// Stochastically checks if two tries are equivalent by sampling paths from each and
    /// verifying their existence in the other.
    ///
    /// # Arguments
    /// * `arena_a`, `roots_a`: The first trie.
    /// * `arena_b`, `roots_b`: The second trie.
    /// * `num_samples`: The number of paths to sample from each trie.
    /// * `max_path_len`: The maximum length of paths to sample.
    /// * `compare`: A closure that compares two paths and returns a `PathComparison`.
    /// * `rng`: A random number generator.
    ///
    /// # Returns
    /// `true` if all sampled paths from one trie are found in the other, `false` otherwise.
    pub fn are_tries_equivalent_stochastic<F, R>(
        arena_a: &Arena<Self>,
        roots_a: &[Trie2Index],
        arena_b: &Arena<Self>,
        roots_b: &[Trie2Index],
        num_samples: usize,
        max_path_len: usize,
        mut compare: F,
        rng: &mut R,
    ) -> bool
    where
        F: FnMut(&TriePath<EK, EV, T>, &TriePath<EK, EV, T>) -> PathComparison + Clone,
        R: Rng + ?Sized,
    {
        // Sample from A, check in B
        for _ in 0..num_samples {
            if let Some(path_a) = Self::sample_path(arena_a, roots_a, max_path_len, rng) {
                if !Self::path_exists(arena_b, roots_b, &path_a, compare.clone()) {
                    return false; // Found a path in A that's not in B
                }
            }
        }

        // Sample from B, check in A
        for _ in 0..num_samples {
            if let Some(path_b) = Self::sample_path(arena_b, roots_b, max_path_len, rng) {
                if !Self::path_exists(arena_a, roots_a, &path_b, compare.clone()) {
                    return false; // Found a path in B that's not in A
                }
            }
        }

        true
    }

    /// Samples a random path from the trie, starting from one of the roots.
    pub fn sample_path<R: Rng + ?Sized>(
        arena: &Arena<Self>,
        roots: &[Trie2Index],
        max_len: usize,
        rng: &mut R,
    ) -> Option<TriePath<EK, EV, T>> {
        if roots.is_empty() {
            return None;
        }

        let root_idx = *roots.choose(rng).unwrap();
        let root_guard = root_idx.read(arena)?;
        let root_value = root_guard.value.clone();

        let mut current_path = Vec::new();
        let mut current_node_idx = root_idx;
        let path_len = if max_len > 0 { rng.gen_range(0..=max_len) } else { 0 };

        for _ in 0..path_len {
            let current_node_guard = current_node_idx.read(arena)?;
            let all_edges: Vec<(EK, EV, Trie2Index)> = current_node_guard
                .children()
                .iter()
                .flat_map(|(ek, dest_map)| {
                    dest_map
                        .iter()
                        .map(move |(child_idx, ev)| (ek.clone(), ev.clone(), *child_idx))
                })
                .collect();

            if all_edges.is_empty() {
                break; // Reached a leaf
            }

            let (edge_key, edge_value, next_node_idx) = all_edges.choose(rng).unwrap().clone();

            let next_node_guard = next_node_idx.read(arena)?;
            let next_node_value = next_node_guard.value.clone();

            current_path.push((edge_key, edge_value, next_node_value));
            current_node_idx = next_node_idx;
        }

        Some((root_value, current_path))
    }

    /// Checks if a given path exists in the trie, starting from its roots.
    /// The `compare` closure is used to guide the search.
    pub fn path_exists<F>(
        arena: &Arena<Self>,
        roots: &[Trie2Index],
        path_to_find: &TriePath<EK, EV, T>,
        mut compare: F,
    ) -> bool
    where
        F: FnMut(&TriePath<EK, EV, T>, &TriePath<EK, EV, T>) -> PathComparison,
    {
        for &root in roots {
            if let Some(root_guard) = root.read(arena) {
                let candidate_path = (root_guard.value.clone(), Vec::new());
                match compare(&candidate_path, path_to_find) {
                    PathComparison::Equal => return true,
                    PathComparison::Prefix => {
                        if Self::path_exists_recursive(
                            arena,
                            root,
                            candidate_path,
                            path_to_find,
                            &mut compare,
                        ) {
                            return true;
                        }
                    }
                    PathComparison::Different => continue,
                }
            }
        }
        false
    }

    /// Recursive helper for `path_exists`.
    fn path_exists_recursive<F>(
        arena: &Arena<Self>,
        current_node_idx: Trie2Index,
        current_path: TriePath<EK, EV, T>,
        path_to_find: &TriePath<EK, EV, T>,
        compare: &mut F,
    ) -> bool
    where
        F: FnMut(&TriePath<EK, EV, T>, &TriePath<EK, EV, T>) -> PathComparison,
    {
        let current_node_guard = match current_node_idx.read(arena) {
            Some(g) => g,
            None => return false,
        };

        for (ek, dest_map) in current_node_guard.children() {
            for (child_idx, ev) in dest_map.iter() {
                if let Some(child_guard) = child_idx.read(arena) {
                    let mut new_path = current_path.clone();
                    new_path
                        .1
                        .push((ek.clone(), ev.clone(), child_guard.value.clone()));

                    match compare(&new_path, path_to_find) {
                        PathComparison::Equal => return true,
                        PathComparison::Prefix => {
                            if Self::path_exists_recursive(
                                arena,
                                *child_idx,
                                new_path,
                                path_to_find,
                                compare,
                            ) {
                                return true;
                            }
                        }
                        PathComparison::Different => continue,
                    }
                }
            }
        }
        false
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
    pub edge_value: Option<EV>,                      // The value for the edge to be inserted
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
pub struct ArenaInner<T> {
    values: Vec<Option<T>>,
    counter: usize,
}

impl<T: PartialEq> PartialEq for ArenaInner<T> {
    fn eq(&self, other: &Self) -> bool {
        self.counter == other.counter && self.values == other.values
    }
}
impl<T: Eq> Eq for ArenaInner<T> {}

impl<T: PartialOrd> PartialOrd for ArenaInner<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.counter.partial_cmp(&other.counter) {
            Some(Ordering::Equal) => self.values.partial_cmp(&other.values),
            other => other,
        }
    }
}

impl<T: Ord> Ord for ArenaInner<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.counter.cmp(&other.counter)
            .then_with(|| self.values.cmp(&other.values))
    }
}

impl<T> ArenaInner<T> {
    pub fn get(&self, index: usize) -> Option<&T> {
        if let Some(value) = self.values.get(index) {
            value.as_ref()
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if let Some(value) = self.values.get_mut(index) {
            value.as_mut()
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Arena<T> {
    pub(crate) inner: Arc<RwLock<ArenaInner<T>>>,
}
impl<T> PartialEq for Arena<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
            || PartialEq::eq(
                &*self.inner.read(),
                &*other.inner.read(),
            )
    }
}
impl<T> Eq for Arena<T> where T: Eq {}

impl<T> PartialOrd for Arena<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return Some(Ordering::Equal);
        }
        PartialOrd::partial_cmp(
            &*self.inner.read(),
            &*other.inner.read(),
        )
    }
}

impl<T> Ord for Arena<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return Ordering::Equal;
        }
        Ord::cmp(
            &*self.inner.read(),
            &*other.inner.read(),
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
        Hash::hash(&Arc::as_ptr(&self.inner), state);
    }
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Arena {
            inner: Arc::new(RwLock::new(ArenaInner {
                values: Vec::new(),
                counter: 0,
            })),
        }
    }

    // Inserts `value` at the next free index and returns the Index.
    pub fn insert(&self, value: T) -> Index {
        let mut inner = self.inner.write();
        let next = inner.counter;
        inner.counter = inner.counter.checked_add(1).expect("Arena index overflow");

        if next >= inner.values.len() {
            inner.values.resize_with(next + 1, || None);
        }

        let old = inner.values[next].replace(value);
        debug_assert!(old.is_none());
        Index { index: next }
    }

    // Replace or set a value at a specific index. Returns the old value if any.
    pub fn insert_at(&self, index: Index, value: T) -> Option<T> {
        let mut inner = self.inner.write();
        let idx = index.index;
        if idx >= inner.values.len() {
            inner.values.resize_with(idx + 1, || None);
        }
        if idx >= inner.counter {
            inner.counter = idx + 1;
        }
        inner.values[idx].replace(value)
    }

    // Remove a value at index, returning it if present.
    pub fn remove(&self, index: Index) -> Option<T> {
        let mut inner = self.inner.write();
        let idx = index.index;
        if idx < inner.values.len() {
            inner.values[idx].take()
        } else {
            None
        }
    }

    // Returns true if the index exists in the arena.
    pub fn contains(&self, index: Index) -> bool {
        let inner = self.inner.read();
        inner.values.get(index.index).map_or(false, |v| v.is_some())
    }

    // Returns a clone of the value at index (requires T: Clone).
    pub fn get(&self, index: Index) -> Option<T>
    where
        T: Clone,
    {
        self.inner.read().values.get(index.index).and_then(|v| v.as_ref()).cloned()
    }

    // Read access via a closure (no Clone bound required).
    pub fn with<R>(&self, index: Index, f: impl FnOnce(&T) -> R) -> Option<R> {
        let guard = self.inner.read();
        guard.values.get(index.index).and_then(|opt| opt.as_ref()).map(f)
    }

    // Mutable access via a closure (no Clone bound required).
    pub fn with_mut<R>(&self, index: Index, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let mut guard = self.inner.write();
        guard.values.get_mut(index.index).and_then(|opt| opt.as_mut()).map(f)
    }

    pub fn len(&self) -> usize {
        self.inner.read().values.iter().filter(|v| v.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().values.iter().all(|v| v.is_none())
    }

    pub fn clear(&self) {
        let mut inner = self.inner.write();
        inner.values.clear();
        inner.counter = 0;
    }

    // Snapshot of all indices.
    pub fn indices(&self) -> Vec<Index> {
        self.inner
            .read()
            .values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if v.is_some() { Some(Index::from(i)) } else { None })
            .collect()
    }

    // Snapshot of all entries as (Index, T) pairs (requires T: Clone).
    pub fn to_vec(&self) -> Vec<(Index, T)>
    where
        T: Clone,
    {
        self.inner
            .read()
            .values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_ref().map(|val| (Index::from(i), val.clone())))
            .collect()
    }

    pub fn deep_clone(&self) -> Self
    where
        T: Clone,
    {
        let inner_guard = self.inner.read();
        let cloned_inner = inner_guard.clone();
        Arena { inner: Arc::new(RwLock::new(cloned_inner)) }
    }

    pub fn replace_with(&self, other: Self)
    where
        T: Clone,
    {
        match Arc::try_unwrap(other.inner) {
            Ok(rwlock) => {
                let mut self_inner = self.inner.write();
                *self_inner = rwlock.into_inner();
            }
            Err(arc) => {
                let other_inner = arc.read();
                let mut self_inner = self.inner.write();
                *self_inner = other_inner.clone();
            }
        }
    }
}

impl<T> JSONConvertible for Arena<T>
where
    T: JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        let guard = self.inner.read();
        let mut obj = BTreeMap::new();
        obj.insert("counter".to_string(), guard.counter.to_json());
        let items: Vec<JSONNode> = guard.values.iter().enumerate().filter_map(|(k, v)| {
            v.as_ref().map(|val| {
                JSONNode::Array(vec![
                    k.to_json(),
                    val.to_json(),
                ])
            })
        }).collect();
        obj.insert("values".to_string(), JSONNode::Array(items));
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        let counter = usize::from_json(obj.remove("counter").ok_or("Missing 'counter' field")?)?;
        let values_node = obj.remove("values").ok_or("Missing 'values' field")?;
        let arr = match values_node {
            JSONNode::Array(arr) => arr,
            _ => return Err("Expected JSONNode::Array for Arena values".to_string()),
        };

        let mut values = Vec::new();
        for item_node in arr {
            let mut pair = match item_node {
                JSONNode::Array(pair) if pair.len() == 2 => pair,
                _ => return Err("Expected 2-element array for Arena entry".to_string()),
            };
            let value_node = pair.pop().unwrap();
            let key_node = pair.pop().unwrap();

            let key = usize::from_json(key_node)?;
            let value = T::from_json(value_node)?;

            if key >= values.len() {
                values.resize_with(key + 1, || None);
            }
            values[key] = Some(value);
        }

        Ok(Arena {
            inner: Arc::new(RwLock::new(ArenaInner {
                values,
                counter,
            })),
        })
    }
}

/// A trait for edge values that can be merged.
pub trait MergeableEdgeValue: Sized {
    /// Merges another value into this one.
    fn merge(&mut self, other: Self);
}

impl MergeableEdgeValue for () {
    fn merge(&mut self, _other: Self) {
        // Nothing to do for unit type.
    }
}

impl MergeableEdgeValue for HybridBitset {
    #[time_it]
    fn merge(&mut self, other: Self) {
        *self |= &other;
    }
}

impl MergeableEdgeValue for RangeSetBlaze<usize> {
    #[time_it]
    fn merge(&mut self, other: Self) {
        *self |= &other;
    }
}

impl<T: Ord> MergeableEdgeValue for BTreeSet<T> {
    #[time_it]
    fn merge(&mut self, mut other: Self) {
        self.append(&mut other);
    }
}

impl<EK, EV, T> Arena<Trie<EK, EV, T>>
where
    EK: Ord + Clone,
    EV: Clone,
{
    /// Removes the edge from `src` to `dst` with the given `edge_key`.
    ///
    /// Returns the removed edge value if the edge existed, otherwise `None`.
    pub fn remove_edge(
        &self,
        src: Trie2Index,
        dst: Trie2Index,
        edge_key: &EK,
    ) -> Option<EV> {
        let mut src_guard = match src.write(self) {
            Some(g) => g,
            None => return None, // Source node not found
        };

        let removed_ev = src_guard
            .children
            .get_mut(edge_key)
            .and_then(|dest_map| dest_map.remove(&dst));

        // Clean up the BTreeMap entry if the OrderedHashMap is now empty
        if removed_ev.is_some() && src_guard.children.get(edge_key).map_or(false, |m| m.is_empty()) {
            src_guard.children.remove(edge_key);
        }

        removed_ev
    }
}

impl<EK, EV, T> Arena<Trie<EK, EV, T>>
where
    EK: Ord + Clone,
    EV: Clone + MergeableEdgeValue,
{
    /// Inserts an edge from `src` to `dst` with the given `edge_key` and `edge_value`.
    /// If an edge with the same key to the same destination already exists, the new
    /// `edge_value` is merged into the existing one using the `MergeableEdgeValue` trait.
    pub fn insert_edge_simple(
        &self,
        src: Trie2Index,
        dst: Trie2Index,
        edge_key: EK,
        edge_value: EV,
    ) {
        if let Some(mut src_guard) = src.write(self) {
            src_guard.children.entry(edge_key).or_default()
                .entry(dst)
                .and_modify(|ev| ev.merge(edge_value.clone()))
                .or_insert(edge_value);
        }
    }
    /// Inserts multiple edges from a single `src` node.
    /// This is more efficient than calling `insert_edge_simple` multiple times as it
    /// only acquires one write lock on the source node.
    pub fn insert_edges_bulk_per_src<I, J>(
        &self,
        src: Trie2Index,
        per_key: I,
    )
    where
        I: IntoIterator<Item = (EK, J)>,
        J: IntoIterator<Item = (Trie2Index, EV)>,
    {
        if let Some(mut src_guard) = src.write(self) {
            for (key, dsts) in per_key {
                let dst_map = src_guard.children.entry(key).or_default();
                for (dst, val) in dsts {
                    dst_map.entry(dst)
                        .and_modify(|ev| ev.merge(val.clone()))
                        .or_insert(val);
                }
            }
        }
    }
}

pub type GodWrapper<EK, EV, T> = Arena<Trie<EK, EV, T>>;
pub type God<EK, EV, T> = Arena<Trie<EK, EV, T>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_all_paths_with_cycles_long_path() {
        type TestTrie = Trie<String, i32, String>;
        let arena = Arena::<TestTrie>::new();

        // Create a chain of nodes: root -> n1 -> ... -> n10
        let root = Trie2Index::from(arena.insert(Trie::new("root".to_string())));
        let mut nodes = vec![root];
        let mut prev_node_idx = root;

        for i in 1..=10 {
            let new_node_idx = Trie2Index::from(arena.insert(Trie::new(format!("n{}", i))));
            let mut prev_node_w = prev_node_idx.write(&arena).unwrap();
            prev_node_w.force_insert_to_node(format!("edge_{}", i), i as i32, new_node_idx);
            drop(prev_node_w);
            nodes.push(new_node_idx);
            prev_node_idx = new_node_idx;
        }

        // Create a cycle: n10 -> n5
        let n10_idx = nodes[10];
        let n5_idx = nodes[5];
        n10_idx
            .write(&arena)
            .unwrap()
            .force_insert_to_node("cycle_edge".to_string(), 100, n5_idx);

        // We expect to find paths ending at n8, with a max length of 20
        let n8_idx = nodes[8];
        let max_len = 20;

        let paths = TestTrie::get_all_paths_with_cycles(
            &arena,
            &[root],
            |idx, _| idx == n8_idx,
            |_, _, _| true, // is_path_edge
            max_len,
        );

        // Expected paths:
        // 1. root -> ... -> n8 (length 8)
        // 2. root -> ... -> n10 -> n5 -> n6 -> n7 -> n8 (length 10 + 1 + 3 = 14)
        // 3. root -> ... -> n10 -> n5 -> ... -> n10 -> n5 -> n6 -> n7 -> n8 (length 14 + 6 = 20)
        //    The cycle is n5 -> ... -> n10 -> n5, which has 6 edges.
        assert_eq!(paths.len(), 3, "Should find 3 paths ending at n8 within max length");

        let mut path_lengths: Vec<usize> = paths.iter().map(|(_, p)| p.len()).collect();
        path_lengths.sort_unstable();

        assert_eq!(path_lengths, vec![8, 14, 20]);

        // Verify path contents for the shortest one.
        let shortest_path = paths.iter().find(|(_, p)| p.len() == 8).unwrap();
        assert_eq!(shortest_path.0, "root");
        for i in 0..8 {
            let (ek, ev, t) = &shortest_path.1[i];
            assert_eq!(*ek, format!("edge_{}", i + 1));
            assert_eq!(*ev, (i + 1) as i32);
            assert_eq!(*t, format!("n{}", i + 1));
        }
    }

    #[test]
    fn test_get_all_paths_with_cycles_uncounted_cycle() {
        type TestTrie = Trie<String, (), String>;
        let arena = Arena::<TestTrie>::new();

        // Graph: root -> c1 -> target
        //          ^    |
        //          |    v
        //          -- c2 -- (uncounted cycle)
        let root = Trie2Index::from(arena.insert(Trie::new("root".to_string())));
        let c1 = Trie2Index::from(arena.insert(Trie::new("c1".to_string())));
        let c2 = Trie2Index::from(arena.insert(Trie::new("c2".to_string())));
        let target = Trie2Index::from(arena.insert(Trie::new("target".to_string())));

        // Edges
        // Counted edge: root -> c1
        root.write(&arena).unwrap().force_insert_to_node("counted".to_string(), (), c1);
        // Uncounted cycle: c1 -> c2 -> c1
        c1.write(&arena).unwrap().force_insert_to_node("uncounted".to_string(), (), c2);
        c2.write(&arena).unwrap().force_insert_to_node("uncounted".to_string(), (), c1);
        // Counted edge: c1 -> target
        c1.write(&arena).unwrap().force_insert_to_node("counted".to_string(), (), target);

        let paths = TestTrie::get_all_paths_with_cycles(
            &arena,
            &[root],
            |idx, _| idx == target,
            |ek, _, _| ek == "counted", // is_path_edge
            5, // max_path_length
        );

        // The traversal should be: root -> c1 -> (cycle c1-c2 is traversed but not infinitely) -> target
        // The `visiting` set should prevent infinite loops.
        // The returned path should only contain the "counted" edges.
        assert_eq!(paths.len(), 1, "Should find exactly one path to the target");

        let (root_val, path) = &paths[0];
        assert_eq!(*root_val, "root");
        assert_eq!(path.len(), 2, "Path should have 2 counted edges");

        // Verify path contents
        let (ek1, _ev1, t1) = &path[0];
        assert_eq!(*ek1, "counted");
        assert_eq!(*t1, "c1");

        let (ek2, _ev2, t2) = &path[1];
        assert_eq!(*ek2, "counted");
        assert_eq!(*t2, "target");
    }

    #[test]
    fn test_get_all_paths_with_cycles_mixed_cycle_produces_multiple_paths() {
        type TestTrie = Trie<String, (), String>;
        let arena = Arena::<TestTrie>::new();

        // Graph:
        // root -> A -> target
        //   ^     |
        //   |     v
        //   C <- B
        //
        // Cycle: A -> B -> C -> A
        // Edges:
        // root -> A: counted
        // A -> B: counted ("cycle_counted")
        // B -> C: uncounted ("cycle_uncounted")
        // C -> A: uncounted ("cycle_uncounted")
        // A -> target: counted
        let root = Trie2Index::from(arena.insert(Trie::new("root".to_string())));
        let node_a = Trie2Index::from(arena.insert(Trie::new("A".to_string())));
        let node_b = Trie2Index::from(arena.insert(Trie::new("B".to_string())));
        let node_c = Trie2Index::from(arena.insert(Trie::new("C".to_string())));
        let target = Trie2Index::from(arena.insert(Trie::new("target".to_string())));

        root.write(&arena).unwrap().force_insert_to_node("to_a".to_string(), (), node_a);
        node_a.write(&arena).unwrap().force_insert_to_node("cycle_counted".to_string(), (), node_b);
        node_b.write(&arena).unwrap().force_insert_to_node("cycle_uncounted".to_string(), (), node_c);
        node_c.write(&arena).unwrap().force_insert_to_node("cycle_uncounted".to_string(), (), node_a);
        node_a.write(&arena).unwrap().force_insert_to_node("to_target".to_string(), (), target);

        let paths = TestTrie::get_all_paths_with_cycles(
            &arena,
            &[root],
            |idx, _| idx == target,
            |ek, _, _| ek != "cycle_uncounted", // is_path_edge: only "to_a", "cycle_counted", "to_target" are path edges
            5, // max_path_length
        );

        // With the current implementation, the `visiting` set prevents re-entering `A` from `C`,
        // so the cycle is never traversed more than once. Only one path is found:
        // 1. root -> A -> target. (length 2)
        //
        // The desired behavior is to allow traversing the cycle as long as the path length limit
        // is not exceeded, because the cycle contains a counted edge.
        // Expected paths to `target`:
        // - path 1: root -> A -> target.
        //   Counted edges: (root,A), (A,target). Length 2.
        // - path 2: root -> A -> B -> C -> A -> target.
        //   Counted edges: (root,A), (A,B), (A,target). Length 3.
        // - path 3: root -> A -> B -> C -> A -> B -> C -> A -> target.
        //   Counted edges: (root,A), (A,B), (A,B), (A,target). Length 4.
        // - path 4: root -> A -> B -> C -> A -> B -> C -> A -> B -> C -> A -> target.
        //   Counted edges: (root,A), (A,B), (A,B), (A,B), (A,target). Length 5.
        //
        // The next path would have length 6, which is > max_path_length.
        // So we expect 4 paths. The current implementation will fail this test.
        // To make this test pass, the cycle detection in `get_all_paths_with_cycles_recursive`
        // needs to be relaxed.
        assert_eq!(paths.len(), 4, "Should find 4 paths to the target within max length by traversing the cycle");

        let mut path_lengths: Vec<usize> = paths.iter().map(|(_, p)| p.len()).collect();
        path_lengths.sort_unstable();
        assert_eq!(path_lengths, vec![2, 3, 4, 5]);

        // Verify the longest path to ensure the cycle was traversed.
        let longest_path = paths.iter().find(|(_, p)| p.len() == 5).unwrap();
        let counted_edges_in_longest_path: Vec<_> = longest_path.1.iter().map(|(ek, _, _)| ek.as_str()).collect();
        assert_eq!(counted_edges_in_longest_path, vec!["to_a", "cycle_counted", "cycle_counted", "cycle_counted", "to_target"]);
    }

    #[test]
    fn test_trim_randomly() {
        type TestTrie = Trie<String, (), String>;
        let arena = Arena::<TestTrie>::new();

        // root -> n1 -> n3
        //   |      |
        //   +----->n2
        let root = Trie2Index::from(arena.insert(Trie::new("root".to_string())));
        let n1 = Trie2Index::from(arena.insert(Trie::new("n1".to_string())));
        let n2 = Trie2Index::from(arena.insert(Trie::new("n2".to_string())));
        let n3 = Trie2Index::from(arena.insert(Trie::new("n3".to_string())));

        arena.insert_edge_simple(root, n1, "edge1".to_string(), ());
        arena.insert_edge_simple(root, n2, "edge2".to_string(), ());
        arena.insert_edge_simple(n1, n2, "edge3".to_string(), ());
        arena.insert_edge_simple(n1, n3, "edge4".to_string(), ());

        let roots = vec![root];
        let original_stats = TestTrie::stats(&arena, &roots);
        assert_eq!(original_stats.num_reachable_nodes, 4);
        assert_eq!(original_stats.num_reachable_edges, 4);

        let mut rng = rand::thread_rng();

        // Test with p=1.0 (remove all edges)
        let (trimmed_arena_all, trimmed_roots_all) =
            TestTrie::trim_randomly(&arena, &roots, 1.0, &mut rng);
        let trimmed_stats_all = TestTrie::stats(&trimmed_arena_all, &trimmed_roots_all);
        assert_eq!(trimmed_stats_all.num_reachable_nodes, 1); // Only root should remain after GC
        assert_eq!(trimmed_stats_all.num_reachable_edges, 0);
        assert_eq!(trimmed_roots_all.len(), 1);
        assert!(trimmed_roots_all[0].read(&trimmed_arena_all).unwrap().value == "root");

        // Test with p=0.0 (remove no edges)
        let (trimmed_arena_none, trimmed_roots_none) =
            TestTrie::trim_randomly(&arena, &roots, 0.0, &mut rng);
        let trimmed_stats_none = TestTrie::stats(&trimmed_arena_none, &trimmed_roots_none);
        assert_eq!(trimmed_stats_none.num_reachable_nodes, 4);
        assert_eq!(trimmed_stats_none.num_reachable_edges, 4);
        assert!(TestTrie::are_graphs_equal(
            &arena,
            roots[0],
            &trimmed_arena_none,
            trimmed_roots_none[0]
        ));

        // Test with p=0.5 (remove some edges)
        // Run a few times to reduce chance of all/none being removed.
        let mut edges_removed = false;
        for _ in 0..10 {
            let (trimmed_arena_some, trimmed_roots_some) =
                TestTrie::trim_randomly(&arena, &roots, 0.5, &mut rng);
            let trimmed_stats_some = TestTrie::stats(&trimmed_arena_some, &trimmed_roots_some);
            if trimmed_stats_some.num_reachable_edges < original_stats.num_reachable_edges {
                edges_removed = true;
                assert!(trimmed_stats_some.num_reachable_edges >= 0);
                assert!(
                    trimmed_stats_some.num_reachable_nodes <= original_stats.num_reachable_nodes
                );
            }
            if edges_removed {
                break;
            }
        }
        assert!(
            edges_removed,
            "After 10 trials with p=0.5, no edges were removed, which is highly unlikely."
        );
    }
}
