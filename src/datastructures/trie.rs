// A Trie-like graph structure using an arena allocator (NodeId indirection).
// This version eliminates Arc/Weak/RwLock and stores node IDs in edges.
// The arena owns all nodes; algorithms receive &Arena or &mut Arena to access/modify the graph.
//
// Key points:
// - No weak pointers, no dangling references.
// - Nodes are referenced by NodeId (usize wrapper).
// - Cycles are prevented by try_insert (unless you use force_insert_* which skip checks).
// - Depth propagation only follows existing edges (since there's only one kind now).
// - Many algorithms mirror the previous design but operate on NodeId + Arena.
//
// Notes about API changes vs. the previous Arc/Weak-based version:
// - Edges only have one kind now (no weak/strong distinction).
// - Functions that previously returned or accepted Arc<RwLock<...>> now use NodeId and &Arena/&mut Arena.
// - EdgeInserter now borrows &mut Arena for the lifetime of the chain.
// - Functions that used to create weak edges (e.g., to_destination_weakly, promote_weak_edges_to_strong) are removed.
// - JSON (de)serialization is provided via arena helper methods (to_json_from_root/from_json_graph).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use ordered_hash_map::{OrderedHashMap, OrderedHashSet};

use kdam::{tqdm, BarExt};
use profiler_macro::{time_it, timeit};

use crate::datastructures::hybrid_bitset::HybridBitset; // used in tests
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::profiler::PROGRESS_BAR_ENABLED;
use deterministic_hash::DeterministicHasher;

// ─────────────────────────────────────────────────────────────────────────────
// Basic arena and NodeId
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct NodeId(pub usize);

impl NodeId {
    pub fn index(self) -> usize {
        self.0
    }
}

impl JSONConvertible for NodeId {
    fn to_json(&self) -> JSONNode {
        self.0.to_json()
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        usize::from_json(node).map(NodeId)
    }
}

pub struct Arena<N> {
    nodes: Vec<N>,
}

impl<N> Arena<N> {
    pub fn new() -> Self {
        Arena { nodes: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Arena { nodes: Vec::with_capacity(cap) }
    }

    pub fn insert(&mut self, node: N) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn get(&self, id: NodeId) -> &N {
        &self.nodes[id.0]
    }

    pub fn get_mut(&mut self, id: NodeId) -> &mut N {
        &mut self.nodes[id.0]
    }

    pub fn iter_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        (0..self.nodes.len()).map(NodeId)
    }

    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &N)> + '_ {
        self.nodes.iter().enumerate().map(|(i, n)| (NodeId(i), n))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (NodeId, &mut N)> + '_ {
        self.nodes.iter_mut().enumerate().map(|(i, n)| (NodeId(i), n))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trie node definition
// ─────────────────────────────────────────────────────────────────────────────

/// Represents a node in a Trie-like DAG/graph.
/// Multiple children can exist for the same edge key. Each edge instance has a value.
///
/// EK: type of the edge key (must be Ord).
/// EV: type of the edge value.
/// T:  type of the value stored within the node.
#[derive(Debug, Clone)]
pub struct Trie<EK: Ord, EV, T> {
    pub value: T,
    /// Map from EdgeKey to map of destination node IDs and edge values.
    children: BTreeMap<EK, OrderedHashMap<NodeId, EV>>,
    /// "Longest distance" from some source node (as computed during insertion).
    /// If A -> B, then A.max_depth < B.max_depth.
    pub max_depth: usize,
}

impl<EK: Ord, EV, T> Trie<EK, EV, T> {
    pub fn new(value: T) -> Self {
        Trie {
            value,
            children: BTreeMap::new(),
            max_depth: 0,
        }
    }

    pub fn children(&self) -> &BTreeMap<EK, OrderedHashMap<NodeId, EV>> {
        &self.children
    }

    pub fn children_mut(&mut self) -> &mut BTreeMap<EK, OrderedHashMap<NodeId, EV>> {
        &mut self.children
    }

    pub fn is_leaf(&self) -> bool {
        self.children.values().all(|m| m.is_empty())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleDetectedError;

impl fmt::Display for CycleDetectedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cycle detected in Trie structure")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Arena-based graph algorithms and mutation API
// ─────────────────────────────────────────────────────────────────────────────

impl<EK, EV, T> Arena<Trie<EK, EV, T>>
where
    EK: Ord + Clone,
{
    /// Returns the ordered map for a given edge key on the source node, if present.
    pub fn get_map(&self, src: NodeId, edge_key: &EK) -> Option<&OrderedHashMap<NodeId, EV>> {
        self.get(src).children.get(edge_key)
    }

    pub fn get_map_mut(&mut self, src: NodeId, edge_key: &EK) -> Option<&mut OrderedHashMap<NodeId, EV>> {
        self.get_mut(src).children.get_mut(edge_key)
    }

    pub fn get_edge_value(&self, src: NodeId, edge_key: &EK, dst: NodeId) -> Option<&EV> {
        self.get(src).children.get(edge_key).and_then(|m| m.get(&dst))
    }

    pub fn get_edge_value_mut(&mut self, src: NodeId, edge_key: &EK, dst: NodeId) -> Option<&mut EV> {
        self.get_mut(src).children.get_mut(edge_key).and_then(|m| m.get_mut(&dst))
    }

    pub fn already_has_dst(&self, src: NodeId, edge_key: &EK, dst: NodeId) -> bool {
        self.get(src)
            .children
            .get(edge_key)
            .map_or(false, |m| m.contains_key(&dst))
    }

    pub fn already_has_dst_for_any_key(&self, src: NodeId, dst: NodeId) -> bool {
        self.get(src).children.values().any(|m| m.contains_key(&dst))
    }
}

impl<EK, EV, T> Arena<Trie<EK, EV, T>>
where
    EK: Ord + Clone,
{
    /// Insert an edge WITHOUT cycle checks and WITHOUT depth propagation.
    pub fn force_insert_to_node(&mut self, src: NodeId, edge_key: EK, edge_value: EV, dst: NodeId) {
        self.get_mut(src)
            .children
            .entry(edge_key)
            .or_default()
            .insert(dst, edge_value);
    }

    /// Convenience: create a new node and insert an edge to it.
    /// Returns the new child's NodeId.
    pub fn force_insert_to_new_node(&mut self, src: NodeId, edge_key: EK, edge_value: EV, value: T) -> NodeId {
        let new_node = Trie::new(value);
        let new_id = self.insert(new_node);
        self.force_insert_to_node(src, edge_key, edge_value, new_id);
        new_id
    }
}

impl<EK, EV, T> Arena<Trie<EK, EV, T>>
where
    EK: Ord + Clone,
{
    /// Returns true if `target` is reachable from `start`.
    #[time_it]
    pub fn detect_cycle(&self, target: NodeId, start: NodeId) -> bool {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut q: VecDeque<NodeId> = VecDeque::new();

        if visited.insert(start) {
            q.push_back(start);
        }

        while let Some(cur) = q.pop_front() {
            if cur == target {
                return true;
            }
            let node = self.get(cur);
            for dest_map in node.children.values() {
                for (&child_id, _) in dest_map.iter() {
                    if visited.insert(child_id) {
                        q.push_back(child_id);
                    }
                }
            }
        }
        false
    }

    /// Propagates a max_depth update to all descendants (DFS), detecting cycles.
    fn propagate_max_depth(&mut self, node: NodeId, current_depth: usize) -> Result<(), CycleDetectedError> {
        let mut rec_stack: HashSet<NodeId> = HashSet::new();
        self._propagate_max_depth(node, current_depth, &mut rec_stack)
    }

    fn _propagate_max_depth(
        &mut self,
        node: NodeId,
        current_depth: usize,
        rec_stack: &mut HashSet<NodeId>,
    ) -> Result<(), CycleDetectedError> {
        if rec_stack.contains(&node) {
            return Err(CycleDetectedError);
        }
        rec_stack.insert(node);

        let candidate = current_depth.saturating_add(1);
        let children: Vec<NodeId> = {
            let n = self.get(node);
            n.children
                .values()
                .flat_map(|m| m.keys().cloned())
                .collect()
        };

        for child in children {
            let should_propagate = {
                let child_ref = self.get(child);
                candidate > child_ref.max_depth
            };
            if should_propagate {
                self.get_mut(child).max_depth = candidate;
                self._propagate_max_depth(child, candidate, rec_stack)?;
            }
        }

        rec_stack.remove(&node);
        Ok(())
    }
}

impl<EK, EV, T> Arena<Trie<EK, EV, T>>
where
    EK: Ord + Clone,
{
    /// Tries to insert a cycle-free edge from src to child.
    /// - Updates child's max_depth and propagates to descendants when needed.
    /// - If adding the edge would create a cycle, returns Err(CycleDetectedError).
    #[time_it]
    pub fn try_insert(
        &mut self,
        src: NodeId,
        edge_key: EK,
        edge_value: &mut Option<EV>,
        child: NodeId,
    ) -> Result<(), CycleDetectedError> {
        if !self.already_has_dst_for_any_key(src, child) && self.detect_cycle(src, child) {
            return Err(CycleDetectedError);
        }
        self.try_insert_unchecked(src, edge_key, edge_value, child)
    }

    #[time_it]
    pub fn try_insert_unchecked(
        &mut self,
        src: NodeId,
        edge_key: EK,
        edge_value: &mut Option<EV>,
        child: NodeId,
    ) -> Result<(), CycleDetectedError> {
        let candidate_depth = self.get(src).max_depth.saturating_add(1);
        let prev_child_depth = self.get(child).max_depth;
        let needs_update = candidate_depth > prev_child_depth;

        if needs_update {
            self.get_mut(child).max_depth = candidate_depth;
            if let Err(e) = self.propagate_max_depth(child, candidate_depth) {
                if self.get(child).max_depth == candidate_depth {
                    self.get_mut(child).max_depth = prev_child_depth;
                }
                return Err(e);
            }
        }

        self.get_mut(src)
            .children
            .entry(edge_key)
            .or_default()
            .insert(child, edge_value.take().expect("edge_value must be Some(...)"));
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Traversals and utility methods
// ─────────────────────────────────────────────────────────────────────────────

impl<EK, EV, T> Arena<Trie<EK, EV, T>>
where
    EK: Ord + Clone,
{
    /// Collects all unique nodes (by NodeId) reachable from the given roots (BFS).
    pub fn all_nodes(&self, roots: &[NodeId]) -> Vec<NodeId> {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut res = Vec::new();
        let mut q = VecDeque::new();

        for &r in roots {
            if visited.insert(r) {
                q.push_back(r);
            }
        }

        while let Some(id) = q.pop_front() {
            res.push(id);
            let node = self.get(id);
            for dest_map in node.children.values() {
                for (&child, _) in dest_map.iter() {
                    if visited.insert(child) {
                        q.push_back(child);
                    }
                }
            }
        }

        res
    }

    /// Checks if there are any cycles reachable from the given `root`.
    pub fn has_any_cycle(&self, root: NodeId) -> bool {
        let mut global_visited: HashSet<NodeId> = HashSet::new();
        let mut rec_stack: HashSet<NodeId> = HashSet::new();
        self._has_any_cycle_recursive(root, &mut global_visited, &mut rec_stack)
    }

    fn _has_any_cycle_recursive(
        &self,
        node: NodeId,
        global_visited: &mut HashSet<NodeId>,
        rec_stack: &mut HashSet<NodeId>,
    ) -> bool {
        if rec_stack.contains(&node) {
            return true;
        }
        if global_visited.contains(&node) {
            return false;
        }

        rec_stack.insert(node);
        global_visited.insert(node);

        let children: Vec<NodeId> = {
            let n = self.get(node);
            n.children
                .values()
                .flat_map(|m| m.keys().cloned())
                .collect()
        };

        for child in children {
            if self._has_any_cycle_recursive(child, global_visited, rec_stack) {
                return true;
            }
        }

        rec_stack.remove(&node);
        false
    }

    /// Recomputes `max_depth` for all nodes reachable from the given roots (Kahn's algorithm).
    pub fn recompute_all_max_depths(&mut self, roots: &[NodeId]) {
        let all = self.all_nodes(roots);
        if all.is_empty() {
            return;
        }

        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for &id in &all {
            in_degree.entry(id).or_insert(0);
            adj.entry(id).or_default();

            let n = self.get(id);
            for (&child, _) in n.children.values().flat_map(|m| m.iter()) {
                adj.entry(id).or_default().push(child);
                *in_degree.entry(child).or_default() += 1;
            }
        }

        let mut q = VecDeque::new();
        for &id in &all {
            if in_degree.get(&id).cloned().unwrap_or(0) == 0 {
                q.push_back(id);
                self.get_mut(id).max_depth = 0;
            } else {
                self.get_mut(id).max_depth = 0;
            }
        }

        while let Some(u) = q.pop_front() {
            let u_depth = self.get(u).max_depth;

            if let Some(children) = adj.get(&u) {
                for &v in children {
                    {
                        let vref = self.get_mut(v);
                        vref.max_depth = vref.max_depth.max(u_depth + 1);
                    }
                    let entry = in_degree.get_mut(&v).unwrap();
                    *entry -= 1;
                    if *entry == 0 {
                        q.push_back(v);
                    }
                }
            }
        }
    }

    /// Recomputes the max_depth of one node based on its children's depths.
    /// Returns true if the depth changed.
    pub fn recompute_max_depth(&mut self, id: NodeId) -> bool {
        let new_depth = {
            let node = self.get(id);
            node.children
                .values()
                .flat_map(|m| m.keys().cloned())
                .map(|child| self.get(child).max_depth + 1)
                .max()
                .unwrap_or(0)
        };
        if new_depth != self.get(id).max_depth {
            self.get_mut(id).max_depth = new_depth;
            true
        } else {
            false
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Special traversal maps (scheduler by depth)
// ─────────────────────────────────────────────────────────────────────────────

impl<EK, EV, T> Arena<Trie<EK, EV, T>>
where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
{
    fn count_all_edges(&self, root_nodes: &[NodeId]) -> usize {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut q: VecDeque<NodeId> = VecDeque::new();
        let mut total = 0;

        for &root in root_nodes {
            if visited.insert(root) {
                q.push_back(root);
            }
        }

        while let Some(id) = q.pop_front() {
            let node = self.get(id);
            for dest_map in node.children.values() {
                for (&child, _) in dest_map.iter() {
                    total += 1;
                    if visited.insert(child) {
                        q.push_back(child);
                    }
                }
            }
        }
        total
    }

    /// Specialized breadth-first traversal with user-specified step/merge/process closures.
    #[time_it]
    pub fn special_map<V: Clone>(
        &self,
        initial_nodes_and_values: Vec<(NodeId, V)>,
        mut step: impl FnMut(&V, &EK, &EV, &Trie<EK, EV, T>) -> Option<V>,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&Trie<EK, EV, T>, &mut V) -> bool,
    ) {
        let mut values: HashMap<NodeId, V> = HashMap::new();
        let mut stopped: HashSet<NodeId> = HashSet::new();
        let mut todo: BTreeMap<usize, OrderedHashSet<NodeId>> = BTreeMap::new();

        let initial_nodes: Vec<NodeId> = initial_nodes_and_values.iter().map(|(n, _)| *n).collect();
        let total_edges = self.count_all_edges(&initial_nodes);
        if PROGRESS_BAR_ENABLED {
            println!("Progress bar enabled");
        } else {
            println!("Progress bar disabled");
        }
        let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED, leave=false);

        for (node, v0) in initial_nodes_and_values {
            values
                .entry(node)
                .and_modify(|old| merge(old, v0.clone()))
                .or_insert(v0);
            let depth = self.get(node).max_depth;
            todo.entry(depth).or_default().insert(node);
        }

        while let Some((_depth, ids)) = todo.pop_first() {
            for id in &ids {
                if stopped.contains(id) {
                    continue;
                }
                let mut agg_v = match values.remove(id) {
                    Some(v) => v,
                    None => continue,
                };
                let n = self.get(*id);
                let proceed = process(n, &mut agg_v);
                if !proceed {
                    stopped.insert(*id);
                    continue;
                }

                let edges: Vec<(EK, EV, NodeId)> = n
                    .children
                    .iter()
                    .flat_map(|(ek, dst)| dst.iter().map(move |(&child, ev)| (ek.clone(), ev.clone(), child)))
                    .collect();

                for (ek, ev, child) in edges {
                    let _ = pb.update(1);
                    if stopped.contains(&child) {
                        continue;
                    }
                    let maybe_v = {
                        let child_node = self.get(child);
                        step(&agg_v, &ek, &ev, child_node)
                    };
                    if let Some(new_v) = maybe_v {
                        values
                            .entry(child)
                            .and_modify(|old| merge(old, new_v.clone()))
                            .or_insert(new_v);
                        let child_depth = self.get(child).max_depth;
                        todo.entry(child_depth).or_default().insert(child);
                    }
                }
            }
        }
    }

    /// Grouped special_map: the step is called once per edge key with the full destination map.
    #[time_it]
    pub fn special_map_grouped<V, S, I>(
        &self,
        initial_nodes_and_values: Vec<(NodeId, V)>,
        mut step: S,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&Trie<EK, EV, T>, &mut V) -> bool,
    ) where
        V: Clone,
        S: FnMut(&V, &EK, &OrderedHashMap<NodeId, EV>) -> I,
        I: IntoIterator<Item = (NodeId, V)>,
    {
        let mut values: HashMap<NodeId, V> = HashMap::new();
        let mut stopped: HashSet<NodeId> = HashSet::new();
        let mut todo: BTreeMap<usize, OrderedHashSet<NodeId>> = BTreeMap::new();

        let initial_nodes: Vec<NodeId> = initial_nodes_and_values.iter().map(|(n, _)| *n).collect();
        let total_edges = self.count_all_edges(&initial_nodes);
        let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED, leave=false);

        for (node, v0) in initial_nodes_and_values {
            values
                .entry(node)
                .and_modify(|old| merge(old, v0.clone()))
                .or_insert(v0);
            let depth = self.get(node).max_depth;
            todo.entry(depth).or_default().insert(node);
        }

        while let Some((_depth, ids)) = todo.pop_first() {
            for id in &ids {
                if stopped.contains(id) {
                    continue;
                }

                let mut agg_v = match values.remove(id) {
                    Some(v) => v,
                    None => continue,
                };
                let node = self.get(*id);
                let proceed = process(node, &mut agg_v);
                if !proceed {
                    stopped.insert(*id);
                    continue;
                }

                let grouped: Vec<(EK, OrderedHashMap<NodeId, EV>)> = node
                    .children
                    .iter()
                    .map(|(ek, dst)| (ek.clone(), dst.clone()))
                    .collect();

                for (ek, dest_map) in grouped {
                    let valid_edges_count = dest_map.len();
                    if valid_edges_count > 0 {
                        let _ = pb.update(valid_edges_count);
                    }

                    let new_values = step(&agg_v, &ek, &dest_map);
                    for (child, v) in new_values.into_iter() {
                        if stopped.contains(&child) {
                            continue;
                        }
                        values
                            .entry(child)
                            .and_modify(|old| merge(old, v.clone()))
                            .or_insert(v);
                        let child_depth = self.get(child).max_depth;
                        todo.entry(child_depth).or_default().insert(child);
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Deep equality and hashing (structural), based on NodeId and arena
// ─────────────────────────────────────────────────────────────────────────────

impl<EK, EV, T> Arena<Trie<EK, EV, T>>
where
    EK: Ord,
    EV: PartialEq + Clone,
    T: PartialEq,
{
    fn compare_nodes_recursive(
        &self,
        a: NodeId,
        b: NodeId,
        cache: &mut HashMap<(NodeId, NodeId), bool>,
    ) -> bool {
        if a == b {
            return true;
        }

        let (k1, k2) = if a < b { (a, b) } else { (b, a) };
        if let Some(&res) = cache.get(&(k1, k2)) {
            return res;
        }
        // optimistic
        cache.insert((k1, k2), true);

        let na = self.get(a);
        let nb = self.get(b);
        if na.value != nb.value || na.max_depth != nb.max_depth {
            cache.insert((k1, k2), false);
            return false;
        }
        if na.children.len() != nb.children.len() {
            cache.insert((k1, k2), false);
            return false;
        }

        for (ek, map_a) in &na.children {
            let Some(map_b) = nb.children.get(ek) else {
                cache.insert((k1, k2), false);
                return false;
            };
            if map_a.len() != map_b.len() {
                cache.insert((k1, k2), false);
                return false;
            }

            let mut other_pairs: Vec<(NodeId, EV)> = map_b.iter().map(|(&id, ev)| (id, ev.clone())).collect();

            'outer: for (id_a, ev_a) in map_a.iter() {
                for i in 0..other_pairs.len() {
                    if ev_a == &other_pairs[i].1 {
                        let node_b = other_pairs[i].0;
                        if self.compare_nodes_recursive(*id_a, node_b, cache) {
                            other_pairs.remove(i);
                            continue 'outer;
                        }
                    }
                }
                cache.insert((k1, k2), false);
                return false;
            }
        }
        true
    }

    pub fn deep_eq(&self, a: NodeId, b: NodeId) -> bool {
        let mut cache: HashMap<(NodeId, NodeId), bool> = HashMap::new();
        self.compare_nodes_recursive(a, b, &mut cache)
    }
}

impl<EK, EV, T> Arena<Trie<EK, EV, T>>
where
    EK: Ord + Hash,
    EV: PartialEq + Clone + Hash,
    T: PartialEq + Hash,
{
    pub fn deep_hash<H: Hasher>(&self, root: NodeId, state: &mut H) {
        let mut recursion_marker: HashMap<NodeId, usize> = HashMap::new();
        self.hash_recursive(root, state, &mut recursion_marker, 0);
    }

    fn hash_recursive<S: Hasher>(
        &self,
        id: NodeId,
        state: &mut S,
        recursion_marker: &mut HashMap<NodeId, usize>,
        current_depth: usize,
    ) {
        if let Some(visited_depth) = recursion_marker.get(&id) {
            id.hash(state);
            visited_depth.hash(state);
            return;
        }
        recursion_marker.insert(id, current_depth);

        let n = self.get(id);
        n.value.hash(state);
        n.max_depth.hash(state);

        n.children.len().hash(state);
        for (ek, dest_map) in &n.children {
            ek.hash(state);

            let mut pair_hashes = Vec::with_capacity(dest_map.len());
            for (&child, ev) in dest_map.iter() {
                let mut pair_hasher = DeterministicHasher::new(std::collections::hash_map::DefaultHasher::new());
                ev.hash(&mut pair_hasher);
                self.hash_recursive(child, &mut pair_hasher, recursion_marker, current_depth + 1);
                pair_hashes.push(pair_hasher.finish());
            }

            pair_hashes.sort_unstable();
            for h in pair_hashes {
                h.hash(state);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON (de)serialization helpers for graphs rooted at a NodeId
// ─────────────────────────────────────────────────────────────────────────────

impl<EK, EV, T> Arena<Trie<EK, EV, T>>
where
    EK: Ord + Clone + JSONConvertible + Debug,
    EV: Clone + JSONConvertible,
    T: Clone + JSONConvertible,
{
    /// Serialize the subgraph reachable from `root` to JSON.
    /// Format:
    /// {
    ///   "nodes": [ { "value": ..., "max_depth": ..., "children": [ [EK, [ [child_idx, EV], ... ] ], ... ] } , ... ],
    ///   "root_idx": 0-based index into the nodes array
    /// }
    pub fn to_json_from_root(&self, root: NodeId) -> JSONNode {
        let mut nodes_json: Vec<JSONNode> = Vec::new();
        let mut id_to_idx: HashMap<NodeId, usize> = HashMap::new();
        let mut q: VecDeque<NodeId> = VecDeque::new();

        id_to_idx.insert(root, 0);
        q.push_back(root);
        nodes_json.push(JSONNode::Null);

        while let Some(cur) = q.pop_front() {
            let cur_idx = *id_to_idx.get(&cur).unwrap();
            let node = self.get(cur);

            let mut children_json_data = Vec::new();

            for (ek, dest_map) in &node.children {
                let mut dests = Vec::new();
                for (&child, ev) in dest_map.iter() {
                    let child_idx = if let Some(&idx) = id_to_idx.get(&child) {
                        idx
                    } else {
                        let new_idx = nodes_json.len();
                        id_to_idx.insert(child, new_idx);
                        q.push_back(child);
                        nodes_json.push(JSONNode::Null);
                        new_idx
                    };
                    dests.push(JSONNode::Array(vec![child_idx.to_json(), ev.to_json()]));
                }
                if !dests.is_empty() {
                    children_json_data.push(JSONNode::Array(vec![
                        ek.to_json(),
                        JSONNode::Array(dests),
                    ]));
                }
            }

            nodes_json[cur_idx] = JSONNode::Object(BTreeMap::from_iter(vec![
                ("value".to_string(), node.value.to_json()),
                ("max_depth".to_string(), node.max_depth.to_json()),
                ("children".to_string(), JSONNode::Array(children_json_data)),
            ]));
        }

        JSONNode::Object(BTreeMap::from_iter(vec![
            ("nodes".to_string(), JSONNode::Array(nodes_json)),
            ("root_idx".to_string(), id_to_idx.get(&root).unwrap().to_json()),
        ]))
    }

    /// Deserialize a graph produced by `to_json_from_root`.
    /// Returns (arena, root_id_in_new_arena).
    pub fn from_json_graph(node: JSONNode) -> Result<(Self, NodeId), String> {
        let (nodes_array, root_idx) = match node {
            JSONNode::Object(mut obj) => {
                let nodes_json = obj.remove("nodes").ok_or_else(|| "Missing 'nodes'".to_string())?;
                let root_idx_json = obj.remove("root_idx").ok_or_else(|| "Missing 'root_idx'".to_string())?;
                let arr = match nodes_json {
                    JSONNode::Array(arr) => arr,
                    _ => return Err("'nodes' must be an array".to_string()),
                };
                let root = usize::from_json(root_idx_json)?;
                (arr, root)
            }
            _ => return Err("Expected JSON object for graph".to_string()),
        };

        if root_idx >= nodes_array.len() {
            return Err(format!(
                "root_idx {} out of bounds for nodes len {}",
                root_idx,
                nodes_array.len()
            ));
        }

        // Pass 1: allocate nodes with values and max_depth, empty children
        let mut arena: Arena<Trie<EK, EV, T>> = Arena::with_capacity(nodes_array.len());
        for (i, node_json) in nodes_array.iter().enumerate() {
            match node_json {
                JSONNode::Object(obj) => {
                    let value_json = obj.get("value").ok_or_else(|| format!("Node {} missing 'value'", i))?;
                    let depth_json = obj.get("max_depth").ok_or_else(|| format!("Node {} missing 'max_depth'", i))?;
                    let value = T::from_json(value_json.clone())?;
                    let max_depth = usize::from_json(depth_json.clone())?;
                    let mut node = Trie::new(value);
                    node.max_depth = max_depth;
                    arena.insert(node);
                }
                _ => return Err(format!("Node {} is not an object", i)),
            }
        }

        // Pass 2: link edges
        for (i, node_json) in nodes_array.iter().enumerate() {
            let obj = match node_json {
                JSONNode::Object(obj) => obj,
                _ => unreachable!(),
            };
            let children_json = obj.get("children").ok_or_else(|| format!("Node {} missing 'children'", i))?;
            let src_id = NodeId(i);

            match children_json {
                JSONNode::Array(ek_entries) => {
                    for ek_entry in ek_entries {
                        match ek_entry {
                            JSONNode::Array(pair) if pair.len() == 2 => {
                                let ek_json = &pair[0];
                                let dests_json = &pair[1];

                                let ek = EK::from_json(ek_json.clone())?;
                                let mut map = OrderedHashMap::new();

                                match dests_json {
                                    JSONNode::Array(dest_pairs) => {
                                        for dest_pair in dest_pairs {
                                            match dest_pair {
                                                JSONNode::Array(inner) if inner.len() == 2 => {
                                                    let child_idx = usize::from_json(inner[0].clone())?;
                                                    let ev = EV::from_json(inner[1].clone())?;
                                                    map.insert(NodeId(child_idx), ev);
                                                }
                                                _ => return Err(format!("Invalid child-ev pair for node {}", i)),
                                            }
                                        }
                                    }
                                    _ => return Err(format!("'children' entry not array for node {}", i)),
                                }

                                arena.get_mut(src_id).children.insert(ek, map);
                            }
                            _ => return Err(format!("Invalid EK entry for node {}", i)),
                        }
                    }
                }
                _ => return Err(format!("'children' for node {} is not an array", i)),
            }
        }

        Ok((arena, NodeId(root_idx)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EdgeInserter (arena-aware)
// ─────────────────────────────────────────────────────────────────────────────

pub struct EdgeInserter<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
where
    EK: Ord + Clone,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    arena: &'a mut Arena<Trie<EK, EV, T>>,
    source: NodeId,
    edge_key: EK,
    edge_value: Option<EV>,
    merge_edge_value: FMergeEV,
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
    result: Option<NodeId>,
}

impl<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T> EdgeInserter<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    pub fn new(
        arena: &'a mut Arena<Trie<EK, EV, T>>,
        source: NodeId,
        edge_key: EK,
        mut edge_value: EV,
        mut merge_edge_value: FMergeEV,
        update_node_value: FUpdateT,
        mut merge_edge_value_and_source_node_value: FMergeEV_T,
    ) -> Self {
        let src_val = arena.get(source).value.clone();
        merge_edge_value_and_source_node_value(&mut edge_value, &src_val);

        EdgeInserter {
            arena,
            source,
            edge_key,
            edge_value: Some(edge_value),
            merge_edge_value,
            update_node_value,
            merge_edge_value_and_source_node_value,
            result: None,
        }
    }

    #[time_it]
    pub fn try_destination(mut self, destination: NodeId) -> Self {
        if self.result.is_some() {
            return self;
        }

        let mut update_info: Option<(NodeId, EV)> = None;
        let exists = self
            .arena
            .get(self.source)
            .children
            .get(&self.edge_key)
            .and_then(|m| m.get(&destination))
            .cloned();

        if let Some(mut existing_ev) = exists {
            let new_ev = self.edge_value.take().expect("edge_value must be Some");
            (self.merge_edge_value)(&mut existing_ev, new_ev);
            self.arena
                .get_mut(self.source)
                .children
                .get_mut(&self.edge_key)
                .unwrap()
                .insert(destination, existing_ev.clone());
            update_info = Some((destination, existing_ev));
            self.result = Some(destination);
        } else {
            let edge_val_clone = self.edge_value.as_ref().unwrap().clone();
            if self
                .arena
                .try_insert(self.source, self.edge_key.clone(), &mut self.edge_value, destination)
                .is_ok()
            {
                self.result = Some(destination);
                update_info = Some((destination, edge_val_clone));
            }
        }

        if let Some((dest, ev)) = update_info {
            (self.update_node_value)(&mut self.arena.get_mut(dest).value, &ev);
        }

        self
    }

    /// Auto variant: without weak edges, this behaves like try_destination.
    #[time_it]
    pub fn try_destination_auto(self, destination: NodeId) -> Self {
        self.try_destination(destination)
    }

    pub fn try_destinations(mut self, destinations: &[NodeId]) -> Self {
        for &d in destinations {
            if self.result.is_some() {
                break;
            }
            self = self.try_destination(d);
        }
        self
    }

    #[time_it]
    pub fn try_destinations_iter(mut self, destinations: impl Iterator<Item = NodeId>) -> Self {
        for d in destinations {
            if self.result.is_some() {
                break;
            }
            self = self.try_destination(d);
        }
        self
    }

    #[time_it]
    pub fn try_destinations_iter_with<F, R>(mut self, destinations: F) -> Self
    where
        F: Fn() -> R,
        R: Iterator<Item = NodeId>,
    {
        for d in destinations() {
            if self.result.is_some() {
                break;
            }
            self = self.try_destination(d);
        }
        self
    }

    pub fn try_children(mut self) -> Self {
        if self.result.is_some() {
            return self;
        }
        let children_for_key: Vec<NodeId> = {
            let src = self.arena.get(self.source);
            if let Some(dest_map) = src.children.get(&self.edge_key) {
                dest_map.keys().cloned().collect()
            } else {
                Vec::new()
            }
        };

        if !children_for_key.is_empty() {
            self = self.try_destinations(&children_for_key);
        }
        self
    }

    pub fn else_create_destination_with_value(mut self, value: T) -> Self {
        if self.result.is_some() {
            return self;
        }

        let new_id = self.arena.insert(Trie::new(value));
        let edge_val_clone = self.edge_value.as_ref().unwrap().clone();

        if self
            .arena
            .try_insert(self.source, self.edge_key.clone(), &mut self.edge_value, new_id)
            .is_ok()
        {
            (self.update_node_value)(&mut self.arena.get_mut(new_id).value, &edge_val_clone);
            self.result = Some(new_id);
        }

        self
    }

    pub fn else_create_destination(self) -> Self
    where
        T: Default,
    {
        self.else_create_destination_with_value(T::default())
    }

    pub fn into_option(self) -> Option<NodeId> {
        self.result
    }

    pub fn is_some(&self) -> bool {
        self.result.is_some()
    }

    pub fn clone_into_option(&self) -> Option<NodeId> {
        self.result
    }

    pub fn unwrap(self) -> NodeId {
        self.result
            .expect("EdgeInserter::unwrap() called but no destination was found or created")
    }

    pub fn expect(self, msg: &str) -> NodeId {
        self.result.expect(msg)
    }
}

impl<EK, EV, T> Arena<Trie<EK, EV, T>>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
{
    pub fn insert_edge<'a, FMergeEV, FUpdateT, FMergeEV_T>(
        &'a mut self,
        source: NodeId,
        edge_key: EK,
        edge_value: EV,
        merge_edge_value: FMergeEV,
        update_node_value: FUpdateT,
        merge_edge_value_and_source_node_value: FMergeEV_T,
    ) -> EdgeInserter<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
    where
        FMergeEV: FnMut(&mut EV, EV) + 'a,
        FUpdateT: FnMut(&mut T, &EV) + 'a,
        FMergeEV_T: FnMut(&mut EV, &T) + 'a,
    {
        EdgeInserter::new(
            self,
            source,
            edge_key,
            edge_value,
            merge_edge_value,
            update_node_value,
            merge_edge_value_and_source_node_value,
        )
    }
}

// Convenience free functions mirroring old helpers but using NodeId and arena.
pub fn try_destination<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>(
    arena: &mut Arena<Trie<EK, EV, T>>,
    source: NodeId,
    edge_key: EK,
    edge_value: EV,
    destination: NodeId,
    merge_edge_value: FMergeEV,
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
) -> Option<NodeId>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    arena
        .insert_edge(
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

pub fn try_destination_with<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>(
    arena: &mut Arena<Trie<EK, EV, T>>,
    source: NodeId,
    edge_key: EK,
    edge_value: EV,
    destinations: &[NodeId],
    merge_edge_value: FMergeEV,
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
) -> Option<NodeId>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    arena
        .insert_edge(
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

pub fn try_destination_auto<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>(
    arena: &mut Arena<Trie<EK, EV, T>>,
    source: NodeId,
    edge_key: EK,
    edge_value: EV,
    destination: NodeId,
    merge_edge_value: FMergeEV,
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
) -> Option<NodeId>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    // Same as try_destination in the arena-based implementation (no weak fallback).
    arena
        .insert_edge(
            source,
            edge_key,
            edge_value,
            merge_edge_value,
            update_node_value,
            merge_edge_value_and_source_node_value,
        )
        .try_destination_auto(destination)
        .into_option()
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Use concrete types for merge tests
    type TestTrieMerge = Trie<&'static str, Vec<i32>, String>;
    type TestArenaMerge = Arena<TestTrieMerge>;
    // Use simpler types for basic tests
    type TestTrieBasic = Trie<&'static str, &'static str, i32>;
    type TestArenaBasic = Arena<TestTrieBasic>;

    // Use concrete types for EdgeInserter tests
    type TestTrieEI = Trie<&'static str, HybridBitset, String>;
    type TestArenaEI = Arena<TestTrieEI>;

    // Helper merge functions used in tests
    fn merge_bitset_union(existing: &mut HybridBitset, new: HybridBitset) {
        *existing |= new
    }

    fn merge_nv_append_if_flag(existing_nv: &String, new_nv: String) -> Option<String> {
        if existing_nv.contains("mergeable") && !existing_nv.contains("not_mergeable") {
            Some(format!("{}|{}", existing_nv, new_nv))
        } else {
            None
        }
    }

    #[test]
    fn test_try_insertion_and_retrieval() {
        let mut arena: TestArenaBasic = Arena::new();
        let root = arena.insert(TestTrieBasic::new(0));
        let child1 = arena.insert(TestTrieBasic::new(1));
        let child2 = arena.insert(TestTrieBasic::new(2));
        let child3 = arena.insert(TestTrieBasic::new(3));

        arena.try_insert(root, "a", &mut Some("edge_a1"), child1).unwrap();
        arena.try_insert(root, "b", &mut Some("edge_b"), child2).unwrap();
        arena.try_insert(root, "a", &mut Some("edge_a3"), child3).unwrap();

        let root_node = arena.get(root);
        let retrieved_children_a = root_node.children().get(&"a").expect("Failed to get children for 'a'");
        assert_eq!(retrieved_children_a.len(), 2);
        let retrieved_set: HashSet<(&str, NodeId)> = retrieved_children_a.iter().map(|(&id, &ev)| (ev, id)).collect();
        assert!(retrieved_set.contains(&("edge_a1", child1)));
        assert!(retrieved_set.contains(&("edge_a3", child3)));

        let retrieved_children_b = root_node.children().get(&"b").expect("Failed to get child 'b'");
        assert_eq!(retrieved_children_b.len(), 1);
        let (&only_id, &only_ev) = retrieved_children_b.iter().next().unwrap();
        assert_eq!(only_ev, "edge_b");
        assert_eq!(only_id, child2);

        assert!(root_node.children().get(&"c").is_none());

        let mut keys: Vec<_> = root_node.children().keys().cloned().collect();
        assert_eq!(keys, vec!["a", "b"]);

        assert!(!root_node.is_leaf());
        assert!(arena.get(child1).is_leaf());
        assert!(arena.get(child2).is_leaf());
        assert!(arena.get(child3).is_leaf());
    }

    #[test]
    fn test_multiple_children_same_edge_key() {
        let mut arena: TestArenaBasic = Arena::new();
        let root = arena.insert(TestTrieBasic::new(0));
        let child1 = arena.insert(TestTrieBasic::new(1));
        let child2 = arena.insert(TestTrieBasic::new(2));

        arena.try_insert(root, "edge", &mut Some("val1"), child1).unwrap();
        arena.try_insert(root, "edge", &mut Some("val2"), child2).unwrap();

        let root_node = arena.get(root);
        let children_map = root_node.children().get(&"edge").unwrap();
        assert_eq!(children_map.len(), 2);
        let set: HashSet<(&str, NodeId)> = children_map.iter().map(|(&id, &ev)| (ev, id)).collect();
        assert!(set.contains(&("val1", child1)));
        assert!(set.contains(&("val2", child2)));

        let all = arena.all_nodes(&[root]);
        assert_eq!(all.len(), 3);
        let all_set: HashSet<_> = all.into_iter().collect();
        assert!(all_set.contains(&root));
        assert!(all_set.contains(&child1));
        assert!(all_set.contains(&child2));

        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();
        let mut edge_info_at_step = Vec::new();

        arena.special_map(
            vec![(root, 100)],
            |parent_val, ek, ev, _child| {
                edge_info_at_step.push((ek.clone(), ev.clone()));
                Some(parent_val + 1)
            },
            |current, new| *current = new,
            |node, computed_val| {
                processed_node_values.push(node.value);
                computed_values.push(*computed_val);
                true
            },
        );

        assert_eq!(processed_node_values.len(), 3);
        assert!(processed_node_values.contains(&0));
        assert!(processed_node_values.contains(&1));
        assert!(processed_node_values.contains(&2));

        assert_eq!(processed_node_values[0], 0);
        let s1: HashSet<_> = processed_node_values[1..].iter().cloned().collect();
        assert!(s1.contains(&1));
        assert!(s1.contains(&2));

        assert_eq!(computed_values.len(), 3);
        assert_eq!(computed_values[0], 100);
        let results_map: HashMap<i32, i32> = processed_node_values
            .iter()
            .cloned()
            .zip(computed_values.iter().cloned())
            .collect();
        assert_eq!(results_map.get(&0), Some(&100));
        assert_eq!(results_map.get(&1), Some(&101));
        assert_eq!(results_map.get(&2), Some(&101));

        assert_eq!(edge_info_at_step.len(), 2);
        assert!(edge_info_at_step.contains(&("edge", "val1")));
        assert!(edge_info_at_step.contains(&("edge", "val2")));
    }

    #[test]
    fn test_special_map_bfs_order_with_edges() {
        let mut arena: TestArenaBasic = Arena::new();
        let root = arena.insert(TestTrieBasic::new(0));
        let child1 = arena.insert(TestTrieBasic::new(1));
        let child2 = arena.insert(TestTrieBasic::new(2));
        let grandchild = arena.insert(TestTrieBasic::new(3));

        arena.try_insert(root, "r->c1", &mut Some("e1"), child1).unwrap();
        arena.try_insert(root, "r->c2", &mut Some("e2"), child2).unwrap();
        arena.try_insert(child1, "c1->gc", &mut Some("e3"), grandchild).unwrap();

        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();
        let mut edge_info_at_step = Vec::new();

        arena.special_map(
            vec![(root, 100)],
            |parent_val, ek, ev, _child| {
                edge_info_at_step.push((ek.clone(), ev.clone()));
                Some(parent_val + 1)
            },
            |current, new| *current = new,
            |node, computed_val| {
                processed_node_values.push(node.value);
                computed_values.push(*computed_val);
                true
            },
        );

        assert_eq!(processed_node_values.len(), 4);
        assert_eq!(processed_node_values[0], 0);
        let depth1: HashSet<_> = processed_node_values[1..3].iter().cloned().collect();
        assert!(depth1.contains(&1));
        assert!(depth1.contains(&2));
        assert_eq!(processed_node_values[3], 3);

        let results_map: HashMap<i32, i32> = processed_node_values
            .iter()
            .cloned()
            .zip(computed_values.iter().cloned())
            .collect();
        assert_eq!(results_map.get(&0), Some(&100));
        assert_eq!(results_map.get(&1), Some(&101));
        assert_eq!(results_map.get(&2), Some(&101));
        assert_eq!(results_map.get(&3), Some(&102));

        assert_eq!(edge_info_at_step.len(), 3);
        assert!(edge_info_at_step.contains(&("r->c1", "e1")));
        assert!(edge_info_at_step.contains(&("r->c2", "e2")));
        assert!(edge_info_at_step.contains(&("c1->gc", "e3")));
    }

    #[test]
    fn test_all_nodes_diamond() {
        // root -> child1, child2; child1 -> grandchild; child2 -> grandchild
        let mut arena: TestArenaBasic = Arena::new();
        let root = arena.insert(TestTrieBasic::new(0));
        let child1 = arena.insert(TestTrieBasic::new(1));
        let child2 = arena.insert(TestTrieBasic::new(2));
        let grandchild = arena.insert(TestTrieBasic::new(3));

        arena.try_insert(root, "r1", &mut Some("e1"), child1).unwrap();
        arena.try_insert(root, "r2", &mut Some("e2"), child2).unwrap();
        arena.try_insert(child1, "c1", &mut Some("e3"), grandchild).unwrap();
        arena.try_insert(child2, "c2", &mut Some("e4"), grandchild).unwrap();

        let all_nodes = arena.all_nodes(&[root]);
        assert_eq!(all_nodes.len(), 4);
        let s: HashSet<_> = all_nodes.into_iter().collect();
        assert!(s.contains(&root));
        assert!(s.contains(&child1));
        assert!(s.contains(&child2));
        assert!(s.contains(&grandchild));
    }

    #[test]
    fn test_special_map_diamond_merge_max() {
        let mut arena: TestArenaBasic = Arena::new();
        let root = arena.insert(TestTrieBasic::new(0));
        let child1 = arena.insert(TestTrieBasic::new(1));
        let child2 = arena.insert(TestTrieBasic::new(2));
        let grandchild = arena.insert(TestTrieBasic::new(3));

        arena.try_insert(root, "r->c1", &mut Some("edge1"), child1).unwrap();
        arena.try_insert(root, "r->c2", &mut Some("edge2"), child2).unwrap();
        arena.try_insert(child1, "c1->gc", &mut Some("edge3"), grandchild).unwrap();
        arena.try_insert(child2, "c2->gc", &mut Some("edge4"), grandchild).unwrap();

        assert_eq!(arena.get(root).max_depth, 0);
        assert_eq!(arena.get(child1).max_depth, 1);
        assert_eq!(arena.get(child2).max_depth, 1);
        assert_eq!(arena.get(grandchild).max_depth, 2);

        let processed_nodes = Arc::new(std::sync::RwLock::new(HashMap::<i32, i32>::new()));
        let process_count = Arc::new(AtomicUsize::new(0));

        arena.special_map(
            vec![(root, 100)],
            |p_val, _ek, _ev, _child| Some(p_val + 1),
            |current_v, new_v| *current_v = (*current_v).max(new_v),
            {
                let processed_nodes = processed_nodes.clone();
                let process_count = process_count.clone();
                move |node, final_v| {
                    let mut map = processed_nodes.write().unwrap();
                    map.insert(node.value, *final_v);
                    process_count.fetch_add(1, Ordering::SeqCst);
                    true
                }
            },
        );

        let final_results = processed_nodes.read().unwrap();
        assert_eq!(process_count.load(Ordering::SeqCst), 4);
        assert_eq!(final_results.get(&0), Some(&100));
        assert_eq!(final_results.get(&1), Some(&101));
        assert_eq!(final_results.get(&2), Some(&101));
        assert_eq!(final_results.get(&3), Some(&102));
    }

    #[test]
    fn test_empty_trie() {
        let mut arena: TestArenaBasic = Arena::new();
        let root = arena.insert(TestTrieBasic::new(42));
        let nodes = arena.all_nodes(&[root]);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0], root);
        assert!(arena.get(root).is_leaf());

        let mut processed = false;
        arena.special_map(
            vec![(root, 100)],
            |_p, _ek, _ev, _n| panic!("Step should not be called for leaf"),
            |_cur, _new| {},
            |node, v| {
                assert_eq!(node.value, 42);
                assert_eq!(*v, 100);
                processed = true;
                true
            },
        );
        assert!(processed);
    }

    #[test]
    fn test_cycle_detection_on_try_insert() {
        let mut arena: TestArenaBasic = Arena::new();
        let root = arena.insert(TestTrieBasic::new(0));
        let child = arena.insert(TestTrieBasic::new(1));

        let r = arena.try_insert(root, "r->c", &mut Some("e1"), child);
        assert!(r.is_ok());
        assert_eq!(arena.get(child).max_depth, 1);
        assert_eq!(arena.get(root).max_depth, 0);

        // Attempt child -> root should detect cycle
        let r2 = arena.try_insert(child, "c->r", &mut Some("e2"), root);
        assert!(r2.is_err());
        assert_eq!(r2.err(), Some(CycleDetectedError));

        let has_edge = arena
            .get(child)
            .children()
            .get("c->r")
            .map_or(false, |m| m.contains_key(&root));
        assert!(!has_edge);

        assert_eq!(arena.get(root).max_depth, 0);
        assert_eq!(arena.get(child).max_depth, 1);
    }

    #[test]
    fn test_cycle_all_nodes_no_panic() {
        let mut arena: TestArenaBasic = Arena::new();
        let root = arena.insert(TestTrieBasic::new(0));
        let child = arena.insert(TestTrieBasic::new(1));

        arena.force_insert_to_node(root, "r->c", "e1", child);
        arena.force_insert_to_node(child, "c->r", "e2", root);

        let all = arena.all_nodes(&[root]);
        assert_eq!(all.len(), 2);
        let s: HashSet<_> = all.into_iter().collect();
        assert!(s.contains(&root));
        assert!(s.contains(&child));
    }

    #[test]
    fn test_has_any_cycle() {
        let mut arena: TestArenaBasic = Arena::new();

        // No cycle
        let root1 = arena.insert(TestTrieBasic::new(0));
        let child1 = arena.insert(TestTrieBasic::new(1));
        let child2 = arena.insert(TestTrieBasic::new(2));
        let grandchild = arena.insert(TestTrieBasic::new(3));
        arena.force_insert_to_node(root1, "a", "e1", child1);
        arena.force_insert_to_node(root1, "b", "e2", child2);
        arena.force_insert_to_node(child1, "c", "e3", grandchild);
        arena.force_insert_to_node(child2, "d", "e4", grandchild);
        assert!(!arena.has_any_cycle(root1));

        // Simple cycle
        let root2 = arena.insert(TestTrieBasic::new(10));
        let child3 = arena.insert(TestTrieBasic::new(11));
        arena.force_insert_to_node(root2, "x", "e5", child3);
        arena.force_insert_to_node(child3, "y", "e6", root2);
        assert!(arena.has_any_cycle(root2));

        // Larger cycle: A -> B -> C -> A
        let root3 = arena.insert(TestTrieBasic::new(20));
        let node_a = arena.insert(TestTrieBasic::new(21));
        let node_b = arena.insert(TestTrieBasic::new(22));
        let node_c = arena.insert(TestTrieBasic::new(23));
        arena.force_insert_to_node(root3, "r->a", "e7", node_a);
        arena.force_insert_to_node(node_a, "a->b", "e8", node_b);
        arena.force_insert_to_node(node_b, "b->c", "e9", node_c);
        arena.force_insert_to_node(node_c, "c->a", "e10", node_a);
        assert!(arena.has_any_cycle(root3));

        // Cycle with unconnected node
        let root4 = arena.insert(TestTrieBasic::new(30));
        let node_a2 = arena.insert(TestTrieBasic::new(31));
        let node_b2 = arena.insert(TestTrieBasic::new(32));
        let node_c2 = arena.insert(TestTrieBasic::new(33));
        arena.force_insert_to_node(root4, "r->a", "e11", node_a2);
        arena.force_insert_to_node(node_a2, "a->b", "e12", node_b2);
        arena.force_insert_to_node(node_b2, "b->a", "e13", node_a2);
        assert!(arena.has_any_cycle(root4));

        // Disconnected graph: root5 chain, root6 cycle
        let root5 = arena.insert(TestTrieBasic::new(40));
        let node_d = arena.insert(TestTrieBasic::new(41));
        arena.force_insert_to_node(root5, "r->d", "e14", node_d);
        let root6_in_cycle = arena.insert(TestTrieBasic::new(50));
        let node_e = arena.insert(TestTrieBasic::new(51));
        arena.force_insert_to_node(root6_in_cycle, "c1->e", "e15", node_e);
        arena.force_insert_to_node(node_e, "e->c1", "e16", root6_in_cycle);

        assert!(!arena.has_any_cycle(root5));
        assert!(arena.has_any_cycle(root6_in_cycle));
    }

    #[test]
    fn test_cycle_special_map_no_panic_limited_processing() {
        // root -> child -> root (cycle), but scheduler should process root then child and stop naturally.
        let mut arena: TestArenaBasic = Arena::new();
        let root = arena.insert(TestTrieBasic::new(0));
        let child = arena.insert(TestTrieBasic::new(1));

        arena.force_insert_to_node(root, "r->c", "e1", child);
        arena.force_insert_to_node(child, "c->r", "e2", root);
        arena.get_mut(root).max_depth = 0;
        arena.get_mut(child).max_depth = 1;

        let mut processed_vals = Vec::new();
        let mut computed_vals = Vec::new();

        arena.special_map(
            vec![ (root, 100) ],
            |p, _ek, _ev, _n| Some(p + 1),
            |cur, new| *cur = (*cur).max(new),
            |node, v| {
                processed_vals.push(node.value);
                computed_vals.push(*v);
                true
            }
        );

        assert_eq!(processed_vals.len(), 2);
        let res: HashMap<i32, i32> = processed_vals.into_iter().zip(computed_vals.into_iter()).collect();
        assert_eq!(res.get(&0), Some(&100));
        assert_eq!(res.get(&1), Some(&101));
    }

    #[test]
    fn test_special_map_stop_processing() {
        let mut arena: TestArenaBasic = Arena::new();
        let root = arena.insert(TestTrieBasic::new(0));
        let child1 = arena.insert(TestTrieBasic::new(1));
        let child2 = arena.insert(TestTrieBasic::new(2));
        let grandchild1 = arena.insert(TestTrieBasic::new(3));
        let grandchild2 = arena.insert(TestTrieBasic::new(4));

        arena.try_insert(root, "r->c1", &mut Some("edge1"), child1).unwrap();
        arena.try_insert(root, "r->c2", &mut Some("edge2"), child2).unwrap();
        arena.try_insert(child1, "c1->gc", &mut Some("edge3"), grandchild1).unwrap();
        arena.try_insert(child2, "c2->gc", &mut Some("edge4"), grandchild2).unwrap();

        let processed_nodes = Arc::new(std::sync::RwLock::new(HashSet::<i32>::new()));
        let computed_values = Arc::new(std::sync::RwLock::new(HashMap::<i32, i32>::new()));

        arena.special_map(
            vec![(root, 100)],
            |p_val, _ek, _ev, _child_node| Some(p_val + 1),
            |current_v, new_v| *current_v = new_v,
            {
                let processed_nodes = processed_nodes.clone();
                let computed_values = computed_values.clone();
                move |node, final_v| {
                    processed_nodes.write().unwrap().insert(node.value);
                    computed_values.write().unwrap().insert(node.value, *final_v);
                    if node.value == 1 {
                        false
                    } else {
                        true
                    }
                }
            }
        );

        let final_processed = processed_nodes.read().unwrap();
        let final_values = computed_values.read().unwrap();

        assert_eq!(final_processed.len(), 4);
        assert!(final_processed.contains(&0));
        assert!(final_processed.contains(&1));
        assert!(final_processed.contains(&2));
        assert!(!final_processed.contains(&3));
        assert!(final_processed.contains(&4));

        assert_eq!(final_values.get(&0), Some(&100));
        assert_eq!(final_values.get(&1), Some(&101));
        assert_eq!(final_values.get(&2), Some(&101));
        assert_eq!(final_values.get(&3), None);
        assert_eq!(final_values.get(&4), Some(&102));
    }

    #[test]
    fn test_special_map_step_returns_none() {
        let mut arena: TestArenaBasic = Arena::new();
        let root = arena.insert(TestTrieBasic::new(0));
        let child1 = arena.insert(TestTrieBasic::new(1));
        let child2 = arena.insert(TestTrieBasic::new(2));
        let grandchild2 = arena.insert(TestTrieBasic::new(3));

        arena.try_insert(root, "keep", &mut Some("e1"), child1).unwrap();
        arena.try_insert(root, "skip", &mut Some("e2"), child2).unwrap();
        arena.try_insert(child2, "keep", &mut Some("e3"), grandchild2).unwrap();

        let processed_nodes = Arc::new(std::sync::RwLock::new(HashSet::<i32>::new()));
        let computed_values = Arc::new(std::sync::RwLock::new(HashMap::<i32, i32>::new()));

        arena.special_map(
            vec![(root, 100)],
            |p_val, ek, _ev, _child| {
                if *ek == "keep" {
                    Some(p_val + 1)
                } else {
                    None
                }
            },
            |cur, new| *cur = new,
            {
                let processed_nodes = processed_nodes.clone();
                let computed_values = computed_values.clone();
                move |node, final_v| {
                    processed_nodes.write().unwrap().insert(node.value);
                    computed_values.write().unwrap().insert(node.value, *final_v);
                    true
                }
            }
        );

        let final_processed = processed_nodes.read().unwrap();
        let final_values = computed_values.read().unwrap();

        assert_eq!(final_processed.len(), 2);
        assert!(final_processed.contains(&0));
        assert!(final_processed.contains(&1));

        assert_eq!(final_values.get(&0), Some(&100));
        assert_eq!(final_values.get(&1), Some(&101));
        assert_eq!(final_values.get(&2), None);
        assert_eq!(final_values.get(&3), None);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EdgeInserter tests (arena-based)
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_ei_try_destination_success_new_edge() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let dest = arena.insert(TestTrieEI::new("dest".to_string()));
        let edge_val: HybridBitset = vec![1].into_iter().collect();

        let result_node = arena
            .insert_edge(source, "key", edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_destination(dest)
            .unwrap();

        assert_eq!(result_node, dest);
        let children_map = arena.get(source).children().get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (&node_id, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, edge_val);
        assert_eq!(node_id, dest);
        assert_eq!(arena.get(dest).max_depth, 1);
    }

    #[test]
    fn test_ei_try_destination_success_merge_ev() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let dest = arena.insert(TestTrieEI::new("dest".to_string()));
        let initial_edge_val: HybridBitset = vec![10].into_iter().collect();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();
        let merged_edge_val: HybridBitset = vec![1, 10].into_iter().collect();

        arena.try_insert(source, "key", &mut Some(initial_edge_val), dest).unwrap();
        assert_eq!(arena.get(dest).max_depth, 1);

        let result_node = arena
            .insert_edge(source, "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_destination(dest)
            .unwrap();

        assert_eq!(result_node, dest);
        let children_map = arena.get(source).children().get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (&node_id, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, merged_edge_val);
        assert_eq!(node_id, dest);
        assert_eq!(arena.get(dest).max_depth, 1);
    }

    #[test]
    fn test_ei_try_destination_fail_merge_ev() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let dest = arena.insert(TestTrieEI::new("dest".to_string()));
        let initial_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        arena.try_insert(source, "key", &mut Some(initial_edge_val), dest).unwrap();

        let opt = arena
            .insert_edge(source, "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_destination(dest)
            .into_option();

        assert!(opt.is_some());
        let children_map = arena.get(source).children().get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (&node_id, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, new_edge_val);
        assert_eq!(node_id, dest);
    }

    #[test]
    fn test_ei_try_destination_fail_cycle() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let dest = arena.insert(TestTrieEI::new("dest".to_string()));
        let dummy_edge_val = HybridBitset::zeros();

        arena.force_insert_to_node(dest, "dest_to_src", dummy_edge_val.clone(), source);

        let new_edge_val: HybridBitset = vec![1].into_iter().collect();
        let opt = arena
            .insert_edge(source, "src_to_dest", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_destination(dest)
            .into_option();

        assert!(opt.is_none());
    }

    #[test]
    fn test_ei_try_slice_success() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let dest1 = arena.insert(TestTrieEI::new("dest1".to_string()));
        let dest2 = arena.insert(TestTrieEI::new("dest2".to_string()));
        let dest3 = arena.insert(TestTrieEI::new("dest3".to_string()));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        arena.force_insert_to_node(dest2, "d2->s", dummy_edge_val.clone(), source);

        let destinations = [dest1, dest2, dest3];

        let result_node = arena
            .insert_edge(source, "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_destinations(&destinations)
            .unwrap();

        assert_eq!(result_node, dest1);
        let children_map = arena.get(source).children().get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (&node_id, ev) = children_map.iter().next().unwrap();
        assert_eq!(node_id, dest1);
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_try_slice_success_later() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let dest1 = arena.insert(TestTrieEI::new("dest1".to_string()));
        let dest2 = arena.insert(TestTrieEI::new("dest2".to_string()));
        let dest3 = arena.insert(TestTrieEI::new("dest3".to_string()));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        arena.force_insert_to_node(dest1, "d1->s", dummy_edge_val.clone(), source);

        let destinations = [dest1, dest2, dest3];

        let result_node = arena
            .insert_edge(source, "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_destinations(&destinations)
            .unwrap();

        assert_eq!(result_node, dest2);
        let children_map = arena.get(source).children().get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (&node_id, ev) = children_map.iter().next().unwrap();
        assert_eq!(node_id, dest2);
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_try_slice_fail_all() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let dest1 = arena.insert(TestTrieEI::new("dest1".to_string()));
        let dest2 = arena.insert(TestTrieEI::new("dest2".to_string()));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        arena.force_insert_to_node(dest1, "d1->s", dummy_edge_val.clone(), source);
        arena.force_insert_to_node(dest2, "d2->s", dummy_edge_val.clone(), source);

        let destinations = [dest1, dest2];

        let result_opt = arena
            .insert_edge(source, "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_destinations(&destinations)
            .into_option();

        assert!(result_opt.is_none());
        assert!(arena.get(source).children().get(&"key").is_none());
    }

    #[test]
    fn test_ei_try_children_success_merge() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let child1 = arena.insert(TestTrieEI::new("child1".to_string()));
        let child2 = arena.insert(TestTrieEI::new("child2".to_string()));
        let child_other_key = arena.insert(TestTrieEI::new("child_other_key".to_string()));

        let edge_key = "target_key";
        let initial_ev_c1: HybridBitset = vec![10].into_iter().collect();
        let initial_ev_c2: HybridBitset = vec![20].into_iter().collect();
        let new_ev_for_inserter: HybridBitset = vec![1].into_iter().collect();
        let merged_ev_c1: HybridBitset = vec![1, 10].into_iter().collect();

        arena.try_insert(source, edge_key, &mut Some(initial_ev_c1), child1).unwrap();
        arena.try_insert(source, edge_key, &mut Some(initial_ev_c2.clone()), child2).unwrap();
        arena.try_insert(source, "other_key", &mut Some(HybridBitset::zeros()), child_other_key)
            .unwrap();

        let result_node_opt = arena
            .insert_edge(source, edge_key, new_ev_for_inserter.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_children()
            .into_option();

        assert!(result_node_opt.is_some());
        let result_node = result_node_opt.unwrap();
        assert_eq!(result_node, child1);

        let s_guard = arena.get(source);
        let children_map_target_key = s_guard.children().get(&edge_key).unwrap();
        let ev_c1 = children_map_target_key.get(&child1).unwrap();
        assert_eq!(*ev_c1, merged_ev_c1);
        let ev_c2 = children_map_target_key.get(&child2).unwrap();
        assert_eq!(*ev_c2, initial_ev_c2);

        let children_map_other_key = s_guard.children().get(&"other_key").unwrap();
        assert_eq!(children_map_other_key.len(), 1);

        // No-children-under-key case
        let source_empty = arena.insert(TestTrieEI::new("source_empty".to_string()));
        let edge_key_empty = "empty_key";
        let new_ev_inserter_empty: HybridBitset = vec![7].into_iter().collect();
        let result_node_empty_opt = arena
            .insert_edge(
                source_empty,
                edge_key_empty,
                new_ev_inserter_empty.clone(),
                merge_bitset_union,
                |_, _| {},
                |_, _| {},
            )
            .try_children()
            .into_option();
        assert!(result_node_empty_opt.is_none());

        // Chain try_children then else_create
        let source_chain = arena.insert(TestTrieEI::new("source_chain".to_string()));
        let edge_key_chain = "chain_key";
        let new_ev_chain: HybridBitset = vec![8].into_iter().collect();
        let created_val = "created_node_via_fallback".to_string();

        let result_node_chain = arena
            .insert_edge(source_chain, edge_key_chain, new_ev_chain.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_children()
            .else_create_destination_with_value(created_val.clone())
            .unwrap();

        assert_eq!(arena.get(result_node_chain).value, created_val);
        let children_map_chain = arena.get(source_chain).children().get(&edge_key_chain).unwrap();
        assert_eq!(children_map_chain.len(), 1);
        let (&nid, ev) = children_map_chain.iter().next().unwrap();
        assert_eq!(nid, result_node_chain);
        assert_eq!(*ev, new_ev_chain);
    }

    #[test]
    fn test_ei_else_create_with_value() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        let result_node = arena
            .insert_edge(source, "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .else_create_destination_with_value("created".to_string())
            .unwrap();

        assert_eq!(arena.get(result_node).value, "created");
        assert_eq!(arena.get(result_node).max_depth, 1);
        let children_map = arena.get(source).children().get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (&nid, ev) = children_map.iter().next().unwrap();
        assert_eq!(nid, result_node);
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_else_create_with() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let created_flag = Arc::new(AtomicUsize::new(0));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        let flag_clone = created_flag.clone();
        let result_node = arena
            .insert_edge(source, "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .else_create_destination_with(|| {
                flag_clone.fetch_add(1, Ordering::SeqCst);
                "created_via_fn".to_string()
            })
            .unwrap();

        assert_eq!(created_flag.load(Ordering::SeqCst), 1);
        assert_eq!(arena.get(result_node).value, "created_via_fn");
        assert_eq!(arena.get(result_node).max_depth, 1);
    }

    #[test]
    fn test_ei_else_create_default() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        let result_node = arena
            .insert_edge(source, "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .else_create_destination()
            .unwrap();

        assert_eq!(arena.get(result_node).value, "");
        assert_eq!(arena.get(result_node).max_depth, 1);
    }

    #[test]
    fn test_ei_chaining_try_then_else() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let dest1 = arena.insert(TestTrieEI::new("dest1".to_string()));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        arena.force_insert_to_node(dest1, "d1->s", dummy_edge_val.clone(), source);

        let result_node = arena
            .insert_edge(source, "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_destination(dest1)
            .else_create_destination_with_value("fallback".to_string())
            .unwrap();

        assert_eq!(arena.get(result_node).value, "fallback");
        assert_ne!(result_node, dest1);
        let children_map = arena.get(source).children().get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (&nid, ev) = children_map.iter().next().unwrap();
        assert_eq!(nid, result_node);
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_chaining_try_success_skips_else() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let dest1 = arena.insert(TestTrieEI::new("dest1".to_string()));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        let result_node = arena
            .insert_edge(source, "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_destination(dest1)
            .else_create_destination_with_value("fallback".to_string())
            .unwrap();

        assert_eq!(result_node, dest1);
        assert_eq!(arena.get(result_node).value, "dest1");
        let children_map = arena.get(source).children().get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (&nid, ev) = children_map.iter().next().unwrap();
        assert_eq!(nid, dest1);
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    #[should_panic(expected = "EdgeInserter::unwrap() called but no destination was found or created")]
    fn test_ei_unwrap_panic() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let dest1 = arena.insert(TestTrieEI::new("dest1".to_string()));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        arena.force_insert_to_node(dest1, "d1->s", dummy_edge_val.clone(), source);

        // Try fails, no else_create called
        arena
            .insert_edge(source, "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_destination(dest1)
            .unwrap(); // should panic
    }

    #[test]
    fn test_ei_get() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let dest1 = arena.insert(TestTrieEI::new("dest1".to_string()));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        arena.force_insert_to_node(dest1, "d1->s", dummy_edge_val.clone(), source);

        let inserter = arena.insert_edge(source, "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});

        let inserter_after_try = inserter.try_destination(dest1);
        assert!(inserter_after_try.clone_into_option().is_none());

        let inserter_after_else = inserter_after_try.else_create_destination_with_value("fallback".to_string());
        let result_opt = inserter_after_else.into_option();
        assert!(result_opt.is_some());
        assert_eq!(arena.get(result_opt.unwrap()).value, "fallback");
    }

    #[test]
    fn test_ei_chaining_stops_after_success() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let child1 = arena.insert(TestTrieEI::new("child1".to_string()));
        let child2 = arena.insert(TestTrieEI::new("child2".to_string()));
        let new_node_val_if_created = "new_node_val".to_string();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        let destinations_for_slice = vec![child2];

        let result_node = arena
            .insert_edge(source, "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_destination(child1) // success
            .try_destinations(&destinations_for_slice) // skipped
            .else_create_destination_with_value(new_node_val_if_created.clone()) // skipped
            .unwrap();

        assert_eq!(result_node, child1);
        let children_map = arena.get(source).children().get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (&nid, ev) = children_map.iter().next().unwrap();
        assert_eq!(nid, child1);
        assert_eq!(*ev, new_edge_val);
        assert_ne!(arena.get(result_node).value, new_node_val_if_created);
    }

    #[test]
    fn test_ei_try_children_new_logic() {
        let mut arena: TestArenaEI = Arena::new();
        let source = arena.insert(TestTrieEI::new("source".to_string()));
        let child1 = arena.insert(TestTrieEI::new("child1".to_string()));
        let child2 = arena.insert(TestTrieEI::new("child2".to_string()));
        let child_other_key = arena.insert(TestTrieEI::new("child_other_key".to_string()));

        let edge_key = "target_key";
        let initial_ev_c1: HybridBitset = vec![10].into_iter().collect();
        let initial_ev_c2: HybridBitset = vec![20].into_iter().collect();
        let new_ev_for_inserter: HybridBitset = vec![1].into_iter().collect();
        let merged_ev_c1: HybridBitset = vec![1, 10].into_iter().collect();

        arena.try_insert(source, edge_key, &mut Some(initial_ev_c1), child1).unwrap();
        arena.try_insert(source, edge_key, &mut Some(initial_ev_c2.clone()), child2).unwrap();
        arena.try_insert(source, "other_key", &mut Some(HybridBitset::zeros()), child_other_key)
            .unwrap();

        let result_node_opt = arena
            .insert_edge(source, edge_key, new_ev_for_inserter.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_children()
            .into_option();

        assert!(result_node_opt.is_some(), "Should find and merge with child1");
        let result_node = result_node_opt.unwrap();
        assert_eq!(result_node, child1, "Result should be child1");

        // Check edge values
        {
            let s_guard = arena.get(source);
            let children_map_target_key = s_guard.children().get(&edge_key).expect("Target key should exist");
            let ev_c1 = children_map_target_key.get(&child1).expect("Child1 should be there");
            assert_eq!(*ev_c1, merged_ev_c1);
            let ev_c2 = children_map_target_key.get(&child2).expect("Child2 should be there");
            assert_eq!(*ev_c2, initial_ev_c2);
            let children_map_other_key = s_guard.children().get(&"other_key").expect("Other key should exist");
            assert_eq!(children_map_other_key.len(), 1);
        }

        // No-children-under-key check
        let source_empty = arena.insert(TestTrieEI::new("source_empty".to_string()));
        let edge_key_empty = "empty_key";
        let new_ev_inserter_empty: HybridBitset = vec![7].into_iter().collect();

        let result_node_empty_opt = arena
            .insert_edge(
                source_empty,
                edge_key_empty,
                new_ev_inserter_empty.clone(),
                merge_bitset_union,
                |_, _| {},
                |_, _| {},
            )
            .try_children()
            .into_option();
        assert!(result_node_empty_opt.is_none());

        // Chain with else_create
        let source_chain = arena.insert(TestTrieEI::new("source_chain".to_string()));
        let edge_key_chain = "chain_key";
        let new_ev_chain: HybridBitset = vec![8].into_iter().collect();
        let created_val = "created_node_via_fallback".to_string();

        let result_node_chain = arena
            .insert_edge(source_chain, edge_key_chain, new_ev_chain.clone(), merge_bitset_union, |_, _| {}, |_, _| {})
            .try_children()
            .else_create_destination_with_value(created_val.clone())
            .unwrap();

        assert_eq!(arena.get(result_node_chain).value, created_val);
        let children_map_chain = arena.get(source_chain).children().get(&edge_key_chain).unwrap();
        assert_eq!(children_map_chain.len(), 1);
        let (&nid, ev) = children_map_chain.iter().next().unwrap();
        assert_eq!(nid, result_node_chain);
        assert_eq!(*ev, new_ev_chain);
    }
}
