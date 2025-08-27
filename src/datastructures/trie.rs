// A lightweight, arena-backed Trie with NodeId handles and no Arc/Mutex/Weak pointers.
// This replaces the previous Arc<RwLock>-based design and removes the weak-pointer semantics.
// All edges are "strong" edges now. Cycle creation is prevented at insertion time.
// You can keep the arena around and pass NodeId handles through your code.
// Traversal/algorithms accept &TrieArena and NodeId handles instead of Arc-wrapped nodes.

use std::collections::{BTreeMap, BTreeSet, VecDeque, HashMap, HashSet};
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use ordered_hash_map::{OrderedHashMap, OrderedHashSet};
use deterministic_hash::DeterministicHasher;

use crate::json_serialization::{JSONConvertible, JSONNode};

// A node handle. We use usize indices into an arena Vec.
pub type NodeId = usize;

/// Error type indicating that a cycle was detected during an edge insertion or propagation-related operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleDetectedError;

/// Result type indicating whether an inserted edge became Strong or (if we had weak, but we don't) a degraded insert.
/// With no weak edges now, this is mostly for API compatibility if you want to mimic "auto" insertion outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertedEdgeKind {
    Strong,
    // No Weak anymore, but keep variant if needed later
}

/// A single Trie node stored inside the arena.
#[derive(Debug, Clone)]
pub struct TrieNode<EK: Ord, EV, T> {
    pub value: T,
    pub children: BTreeMap<EK, OrderedHashMap<NodeId, EV>>,
    pub max_depth: usize,
}

impl<EK: Ord, EV, T> TrieNode<EK, EV, T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            children: BTreeMap::new(),
            max_depth: 0,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.values().all(|dst_map| dst_map.is_empty())
    }
}

/// An arena that contains all nodes for a given Trie graph (or multiple roots).
/// Edges are stored as NodeId -> NodeId inside each node's `children`.
#[derive(Debug, Clone)]
pub struct TrieArena<EK: Ord, EV, T> {
    nodes: Vec<TrieNode<EK, EV, T>>,
}

impl<EK: Ord, EV, T> Default for TrieArena<EK, EV, T> {
    fn default() -> Self {
        Self { nodes: Vec::new() }
    }
}

impl<EK: Ord, EV, T> TrieArena<EK, EV, T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, value: T) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(TrieNode::new(value));
        id
    }

    pub fn node(&self, id: NodeId) -> &TrieNode<EK, EV, T> {
        &self.nodes[id]
    }

    pub fn node_mut(&mut self, id: NodeId) -> &mut TrieNode<EK, EV, T> {
        &mut self.nodes[id]
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns true if `src` already has an outgoing edge with key `ek` to `dst`.
    pub fn already_has_dst(&self, src: NodeId, ek: &EK, dst: NodeId) -> bool {
        self.nodes[src]
            .children
            .get(ek)
            .map_or(false, |m| m.contains_key(&dst))
    }

    /// Returns true if `src` already has an outgoing edge (under any EK) to `dst`.
    pub fn already_has_dst_for_any_key(&self, src: NodeId, dst: NodeId) -> bool {
        self.nodes[src]
            .children
            .values()
            .any(|m| m.contains_key(&dst))
    }

    /// Return a reference to the edge value if it exists.
    pub fn get_edge_value(&self, src: NodeId, ek: &EK, dst: NodeId) -> Option<&EV> {
        self.nodes[src].children.get(ek).and_then(|m| m.get(&dst))
    }

    /// Return a mutable reference to the edge value if it exists.
    pub fn get_edge_value_mut(&mut self, src: NodeId, ek: &EK, dst: NodeId) -> Option<&mut EV> {
        self.nodes[src].children.get_mut(ek).and_then(|m| m.get_mut(&dst))
    }

    /// Strong cycle detection: would adding an edge src -> dst create a cycle?
    /// A cycle would exist iff `src` is reachable from `dst`.
    pub fn would_create_cycle(&self, src: NodeId, dst: NodeId) -> bool {
        self.is_reachable(dst, src)
    }

    /// Reachability using BFS. Returns true if target is reachable from start.
    pub fn is_reachable(&self, start: NodeId, target: NodeId) -> bool {
        if start == target {
            return true;
        }
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut q: VecDeque<NodeId> = VecDeque::new();
        visited.insert(start);
        q.push_back(start);
        while let Some(u) = q.pop_front() {
            let u_node = &self.nodes[u];
            for dst_map in u_node.children.values() {
                for (&v, _ev) in dst_map.iter() {
                    if v == target {
                        return true;
                    }
                    if visited.insert(v) {
                        q.push_back(v);
                    }
                }
            }
        }
        false
    }

    /// Insert a strong edge src -(ek, ev)-> dst. Prevents cycles.
    /// Updates and propagates max_depth when needed.
    pub fn try_insert(
        &mut self,
        src: NodeId,
        ek: EK,
        mut ev: EV,
        dst: NodeId,
    ) -> Result<(), CycleDetectedError>
    where
        EK: Clone,
        EV: Clone,
    {
        // Check cycle
        if !self.already_has_dst_for_any_key(src, dst) && self.would_create_cycle(src, dst) {
            return Err(CycleDetectedError);
        }

        // Merge or insert
        let dst_map = self.nodes[src].children.entry(ek).or_default();
        match dst_map.get_mut(&dst) {
            Some(existing) => {
                // If merging semantics are different, you can customize via an EdgeInserter.
                // Here we replace by default (or you can combine EV if it supports).
                *existing = ev;
            }
            None => {
                dst_map.insert(dst, ev.clone());
            }
        }

        // Depth propagation
        let parent_depth = self.nodes[src].max_depth;
        let candidate_child_depth = parent_depth.saturating_add(1);
        let prev_child_depth = self.nodes[dst].max_depth;
        if candidate_child_depth > prev_child_depth {
            self.nodes[dst].max_depth = candidate_child_depth;
            self.propagate_max_depth(dst, candidate_child_depth);
        }

        Ok(())
    }

    /// Internal helper: propagate updated max_depth from `from` through descendants.
    fn propagate_max_depth(&mut self, from: NodeId, current_depth: usize) {
        let mut rec_stack: HashSet<NodeId> = HashSet::new();
        self._propagate_max_depth(from, current_depth, &mut rec_stack);
    }

    fn _propagate_max_depth(&mut self, node: NodeId, current_depth: usize, rec_stack: &mut HashSet<NodeId>) {
        if rec_stack.contains(&node) {
            // Found a cycle during propagation; given try_insert shouldn't allow cycles,
            // this should practically never occur. We just stop here.
            return;
        }
        rec_stack.insert(node);

        let candidate_depth = current_depth.saturating_add(1);
        // Collect children in a separate list to avoid borrow conflicts
        let children: Vec<NodeId> = self.nodes[node]
            .children
            .values()
            .flat_map(|dest_map| dest_map.keys().copied())
            .collect();

        for child in children {
            if candidate_depth > self.nodes[child].max_depth {
                self.nodes[child].max_depth = candidate_depth;
                self._propagate_max_depth(child, candidate_depth, rec_stack);
            }
        }

        rec_stack.remove(&node);
    }

    /// List all nodes reachable from the given roots (by NodeId).
    pub fn all_nodes(&self, roots: &[NodeId]) -> Vec<NodeId> {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut q: VecDeque<NodeId> = VecDeque::new();
        for &r in roots {
            if visited.insert(r) {
                q.push_back(r);
            }
        }
        let mut order = Vec::new();
        while let Some(u) = q.pop_front() {
            order.push(u);
            let node = &self.nodes[u];
            for dst_map in node.children.values() {
                for (&v, _) in dst_map.iter() {
                    if visited.insert(v) {
                        q.push_back(v);
                    }
                }
            }
        }
        order
    }

    /// Recompute all max_depths for reachable nodes from roots using Kahn's algorithm (DAG).
    /// We forbid cycles on insertion, so this is safe.
    pub fn recompute_all_max_depths(&mut self, roots: &[NodeId])
    where
        EK: Clone,
    {
        let nodes = self.all_nodes(roots);
        if nodes.is_empty() {
            return;
        }
        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();

        // indegree and adjacency lists
        let mut indeg: HashMap<NodeId, usize> = HashMap::new();
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &u in &nodes {
            indeg.entry(u).or_insert(0);
            adj.entry(u).or_default();
        }
        for &u in &nodes {
            for dst_map in self.nodes[u].children.values() {
                for (&v, _) in dst_map.iter() {
                    if node_set.contains(&v) {
                        *indeg.entry(v).or_insert(0) += 1;
                        adj.entry(u).or_default().push(v);
                    }
                }
            }
        }

        // Initialize all depths to 0
        for &u in &nodes {
            self.nodes[u].max_depth = 0;
        }

        // Kahn queue
        let mut q: VecDeque<NodeId> = VecDeque::new();
        for &u in &nodes {
            if indeg.get(&u).cloned().unwrap_or(0) == 0 {
                q.push_back(u);
                self.nodes[u].max_depth = 0;
            }
        }

        while let Some(u) = q.pop_front() {
            let u_depth = self.nodes[u].max_depth;
            if let Some(children) = adj.get(&u) {
                for &v in children {
                    let nd = u_depth.saturating_add(1);
                    if self.nodes[v].max_depth < nd {
                        self.nodes[v].max_depth = nd;
                    }
                    let e = indeg.get_mut(&v).unwrap();
                    *e -= 1;
                    if *e == 0 {
                        q.push_back(v);
                    }
                }
            }
        }
    }

    /// A specialized traversal that schedules nodes by depth and calls user callbacks.
    /// - initial: list of (root NodeId, initial value V).
    /// - step: given a value at parent + EK/EV + child node, compute a value for child (or None to skip).
    /// - merge: merge multiple inputs into the child's accumulator V.
    /// - process: called once with the aggregated value for a node; return false to stop propagating from this node.
    pub fn special_map<V: Clone>(
        &self,
        initial: Vec<(NodeId, V)>,
        mut step: impl FnMut(&V, &EK, &EV, &TrieNode<EK, EV, T>) -> Option<V>,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&TrieNode<EK, EV, T>, &mut V) -> bool,
    ) where
        EK: Clone,
        T: Debug,
    {
        // Depth-based scheduler
        let roots: Vec<NodeId> = initial.iter().map(|(id, _)| *id).collect();
        let mut values: HashMap<NodeId, V> = HashMap::new();
        let mut stopped: HashSet<NodeId> = HashSet::new();
        let mut todo: BTreeMap<usize, OrderedHashSet<NodeId>> = BTreeMap::new();

        // Initialize
        for (id, v) in initial {
            values
                .entry(id)
                .and_modify(|old| merge(old, v.clone()))
                .or_insert(v.clone());
            let d = self.nodes[id].max_depth;
            todo.entry(d).or_default().insert(id);
        }

        while let Some((_depth, set)) = todo.pop_first() {
            for id in set.iter().copied() {
                if stopped.contains(&id) {
                    continue;
                }
                let mut agg_v = match values.remove(&id) {
                    Some(v) => v,
                    None => continue,
                };
                let proceed = process(&self.nodes[id], &mut agg_v);
                if !proceed {
                    stopped.insert(id);
                    continue;
                }

                let out_edges: Vec<(EK, EV, NodeId)> = self.nodes[id]
                    .children
                    .iter()
                    .flat_map(|(ek, dest_map)| {
                        dest_map.iter().map(move |(&dst, ev)| (ek.clone(), ev.clone(), dst))
                    })
                    .collect();

                for (ek, ev, child) in out_edges {
                    if stopped.contains(&child) {
                        continue;
                    }
                    if let Some(new_v) = step(&agg_v, &ek, &ev, &self.nodes[child]) {
                        values
                            .entry(child)
                            .and_modify(|old| merge(old, new_v.clone()))
                            .or_insert(new_v);
                        let dd = self.nodes[child].max_depth;
                        todo.entry(dd).or_default().insert(child);
                    }
                }
            }
        }

        // Silence unused variable warning if not used
        let _ = roots;
    }

    /// Variant that groups children by edge key and calls step once per key with a reference to the whole map.
    pub fn special_map_grouped<V, S, I>(
        &self,
        initial: Vec<(NodeId, V)>,
        mut step: S,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&TrieNode<EK, EV, T>, &mut V) -> bool,
    ) where
        V: Clone,
        S: FnMut(&V, &EK, &OrderedHashMap<NodeId, EV>) -> I,
        I: IntoIterator<Item = (NodeId, V)>,
    {
        let mut values: HashMap<NodeId, V> = HashMap::new();
        let mut stopped: HashSet<NodeId> = HashSet::new();
        let mut todo: BTreeMap<usize, OrderedHashSet<NodeId>> = BTreeMap::new();

        for (id, v) in initial {
            values
                .entry(id)
                .and_modify(|old| merge(old, v.clone()))
                .or_insert(v.clone());
            let d = self.nodes[id].max_depth;
            todo.entry(d).or_default().insert(id);
        }

        while let Some((_depth, set)) = todo.pop_first() {
            for id in set.iter().copied() {
                if stopped.contains(&id) {
                    continue;
                }
                let mut agg_v = match values.remove(&id) {
                    Some(v) => v,
                    None => continue,
                };
                let proceed = process(&self.nodes[id], &mut agg_v);
                if !proceed {
                    stopped.insert(id);
                    continue;
                }

                // Run step once per edge key
                let keys_and_maps: Vec<(&EK, &OrderedHashMap<NodeId, EV>)> = self.nodes[id]
                    .children
                    .iter()
                    .map(|(ek, m)| (ek, m))
                    .collect();

                for (ek, m) in keys_and_maps {
                    let out = step(&agg_v, ek, m);
                    for (child, v2) in out.into_iter() {
                        if stopped.contains(&child) {
                            continue;
                        }
                        values
                            .entry(child)
                            .and_modify(|old| merge(old, v2.clone()))
                            .or_insert(v2);
                        let dd = self.nodes[child].max_depth;
                        todo.entry(dd).or_default().insert(child);
                    }
                }
            }
        }
    }
}

/// A rooted Trie graph (arena + root handle).
#[derive(Debug, Clone)]
pub struct RootedTrie<EK: Ord, EV, T> {
    pub arena: TrieArena<EK, EV, T>,
    pub root: NodeId,
}

impl<EK, EV, T> RootedTrie<EK, EV, T>
where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
{
    pub fn new(mut arena: TrieArena<EK, EV, T>, root: NodeId) -> Self {
        // Invariant: root must be in arena
        assert!(root < arena.nodes.len(), "Root NodeId out of bounds");
        Self { arena, root }
    }

    pub fn root_node(&self) -> &TrieNode<EK, EV, T> {
        &self.arena.node(self.root)
    }

    pub fn root_node_mut(&mut self) -> &mut TrieNode<EK, EV, T> {
        self.arena.node_mut(self.root)
    }
}

// JSON serialization format:
// {
//   "nodes": [
//     { "value": T_json, "max_depth": usize, "children": [[EK_json, [[child_idx, EV_json], ...]], ...] }
//   ],
//   "root_idx": usize
// }
impl<EK, EV, T> JSONConvertible for RootedTrie<EK, EV, T>
where
    EK: Ord + Clone + JSONConvertible + Debug,
    EV: Clone + JSONConvertible,
    T: Clone + JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        // BFS order from root, assign indices
        let mut ptr_to_idx: HashMap<NodeId, usize> = HashMap::new();
        let mut idx_to_id: Vec<NodeId> = Vec::new();

        let mut q: VecDeque<NodeId> = VecDeque::new();
        q.push_back(self.root);
        ptr_to_idx.insert(self.root, 0);
        idx_to_id.push(self.root);

        while let Some(u) = q.pop_front() {
            let node = self.arena.node(u);
            for dst_map in node.children.values() {
                for (&v, _ev) in dst_map.iter() {
                    if !ptr_to_idx.contains_key(&v) {
                        let idx = idx_to_id.len();
                        ptr_to_idx.insert(v, idx);
                        idx_to_id.push(v);
                        q.push_back(v);
                    }
                }
            }
        }

        let mut nodes_json: Vec<JSONNode> = Vec::with_capacity(idx_to_id.len());
        nodes_json.resize(idx_to_id.len(), JSONNode::Null);

        for (i, &id) in idx_to_id.iter().enumerate() {
            let n = self.arena.node(id);
            // children: [[ek_json, [[child_idx, ev_json], ...]], ...]
            let mut children_json = Vec::new();
            for (ek, dst_map) in &n.children {
                let ek_json = ek.to_json();
                let mut arr = Vec::new();
                for (&dst, ev) in dst_map.iter() {
                    let dst_idx = ptr_to_idx[&dst];
                    arr.push(JSONNode::Array(vec![dst_idx.to_json(), ev.to_json()]));
                }
                if !arr.is_empty() {
                    children_json.push(JSONNode::Array(vec![ek_json, JSONNode::Array(arr)]));
                }
            }

            nodes_json[i] = JSONNode::Object(BTreeMap::from_iter(vec![
                ("value".into(), n.value.to_json()),
                ("max_depth".into(), n.max_depth.to_json()),
                ("children".into(), JSONNode::Array(children_json)),
            ]));
        }

        JSONNode::Object(BTreeMap::from_iter(vec![
            ("nodes".into(), JSONNode::Array(nodes_json)),
            ("root_idx".into(), 0usize.to_json()),
        ]))
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let nodes_json = obj
                    .remove("nodes")
                    .ok_or_else(|| "Missing 'nodes'".to_string())?;
                let root_idx_json = obj
                    .remove("root_idx")
                    .ok_or_else(|| "Missing 'root_idx'".to_string())?;

                let nodes_arr = match nodes_json {
                    JSONNode::Array(a) => a,
                    _ => return Err("'nodes' must be array".to_string()),
                };
                let root_idx = usize::from_json(root_idx_json)?;

                if root_idx >= nodes_arr.len() {
                    return Err(format!(
                        "root_idx {} out of bounds for nodes of len {}",
                        root_idx,
                        nodes_arr.len()
                    ));
                }

                let mut arena: TrieArena<EK, EV, T> = TrieArena::new();
                // First pass: create nodes with empty children
                for (i, node_json) in nodes_arr.iter().enumerate() {
                    let obj = match node_json {
                        JSONNode::Object(m) => m,
                        _ => return Err(format!("node[{}] not an object", i)),
                    };
                    let value_json = obj
                        .get("value")
                        .ok_or_else(|| format!("node[{}] missing 'value'", i))?
                        .clone();
                    let max_depth_json = obj
                        .get("max_depth")
                        .ok_or_else(|| format!("node[{}] missing 'max_depth'", i))?
                        .clone();

                    let val = T::from_json(value_json)?;
                    let node_id = arena.add_node(val);
                    debug_assert_eq!(node_id, i);
                    arena.nodes[node_id].max_depth = usize::from_json(max_depth_json)?;
                }

                // Second pass: link edges
                for (i, node_json) in nodes_arr.into_iter().enumerate() {
                    let obj = match node_json {
                        JSONNode::Object(m) => m,
                        _ => unreachable!(),
                    };
                    let children_json = obj
                        .get("children")
                        .cloned()
                        .unwrap_or(JSONNode::Array(vec![]));

                    let pairs = match children_json {
                        JSONNode::Array(a) => a,
                        _ => return Err(format!("node[{}].children not array", i)),
                    };

                    for pair in pairs {
                        let arr = match pair {
                            JSONNode::Array(a) => a,
                            _ => return Err("invalid children entry".to_string()),
                        };
                        if arr.len() != 2 {
                            return Err("expected [ek, arr]".to_string());
                        }
                        let ek = EK::from_json(arr[0].clone())?;
                        let dst_arr = match arr[1].clone() {
                            JSONNode::Array(a) => a,
                            _ => return Err("children entry second must be array".to_string()),
                        };

                        for entry in dst_arr {
                            let p = match entry {
                                JSONNode::Array(a) => a,
                                _ => return Err("child entry must be array".to_string()),
                            };
                            if p.len() != 2 {
                                return Err("child entry must be [idx, ev]".to_string());
                            }
                            let child_idx = usize::from_json(p[0].clone())?;
                            let ev = EV::from_json(p[1].clone())?;
                            arena.nodes[i].children.entry(ek.clone()).or_default().insert(child_idx, ev);
                        }
                    }
                }

                Ok(RootedTrie::new(arena, root_idx))
            }
            _ => Err("Expected JSON object".to_string()),
        }
    }
}

/// Structural equality and hashing for RootedTrie (cycle-safe).
impl<EK, EV, T> PartialEq for RootedTrie<EK, EV, T>
where
    EK: Ord + Hash + Clone + Debug,
    EV: PartialEq + Hash + Clone,
    T: PartialEq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        // Compare via structural recursion with cache
        let mut cache: HashMap<(NodeId, NodeId), bool> = HashMap::new();
        fn eq_nodes<EK, EV, T>(
            arena1: &TrieArena<EK, EV, T>,
            arena2: &TrieArena<EK, EV, T>,
            u: NodeId,
            v: NodeId,
            cache: &mut HashMap<(NodeId, NodeId), bool>,
        ) -> bool
        where
            EK: Ord + Hash + Clone + Debug,
            EV: PartialEq + Hash + Clone,
            T: PartialEq + Hash,
        {
            if u == v && std::ptr::eq(arena1, arena2) {
                return true;
            }
            if let Some(&res) = cache.get(&(u, v)) {
                return res;
            }
            cache.insert((u, v), true); // optimistic to break cycles

            let n1 = &arena1.nodes[u];
            let n2 = &arena2.nodes[v];

            if n1.value != n2.value || n1.max_depth != n2.max_depth || n1.children.len() != n2.children.len() {
                cache.insert((u, v), false);
                return false;
            }

            for (ek, m1) in &n1.children {
                let m2 = match n2.children.get(ek) {
                    Some(m) => m,
                    None => {
                        cache.insert((u, v), false);
                        return false;
                    }
                };
                if m1.len() != m2.len() {
                    cache.insert((u, v), false);
                    return false;
                }

                // multiset compare edges by (ev, child structure)
                let mut pairs2: Vec<(EV, NodeId)> = m2.iter().map(|(&id, ev)| (ev.clone(), id)).collect();
                for (&child_u, ev_u) in m1.iter() {
                    let mut matched = false;
                    for i in 0..pairs2.len() {
                        if *ev_u == pairs2[i].0 {
                            let child_v = pairs2[i].1;
                            if eq_nodes(arena1, arena2, child_u, child_v, cache) {
                                pairs2.remove(i);
                                matched = true;
                                break;
                            }
                        }
                    }
                    if !matched {
                        cache.insert((u, v), false);
                        return false;
                    }
                }
            }
            true
        }
        eq_nodes(&self.arena, &other.arena, self.root, other.root, &mut cache)
    }
}

impl<EK, EV, T> Eq for RootedTrie<EK, EV, T>
where
    EK: Ord + Hash + Clone + Debug,
    EV: Eq + Hash + Clone,
    T: Eq + Hash,
{
}

impl<EK, EV, T> Hash for RootedTrie<EK, EV, T>
where
    EK: Ord + Hash + Clone + Debug,
    EV: PartialEq + Hash + Clone,
    T: PartialEq + Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Cycle-safe recursive hash with memo
        let mut memo: HashMap<NodeId, u64> = HashMap::new();

        fn hash_node<EK, EV, T>(
            arena: &TrieArena<EK, EV, T>,
            u: NodeId,
            memo: &mut HashMap<NodeId, u64>,
        ) -> u64
        where
            EK: Ord + Hash + Clone + Debug,
            EV: PartialEq + Hash + Clone,
            T: PartialEq + Hash,
        {
            if let Some(&h) = memo.get(&u) {
                return h;
            }
            // insert placeholder to break cycles deterministically
            memo.insert(u, 0);
            let n = &arena.nodes[u];
            let mut hasher = DeterministicHasher::new(std::collections::hash_map::DefaultHasher::new());
            n.value.hash(&mut hasher);
            n.max_depth.hash(&mut hasher);

            let mut edge_hashes: Vec<u64> = Vec::new();
            for (ek, m) in &n.children {
                for (&v, ev) in m.iter() {
                    let mut pair = DeterministicHasher::new(std::collections::hash_map::DefaultHasher::new());
                    ek.hash(&mut pair);
                    ev.hash(&mut pair);
                    let hchild = hash_node(arena, v, memo);
                    hchild.hash(&mut pair);
                    edge_hashes.push(pair.finish());
                }
            }
            edge_hashes.sort_unstable();
            for h in edge_hashes {
                h.hash(&mut hasher);
            }
            let out = hasher.finish();
            memo.insert(u, out);
            out
        }

        let root_h = hash_node(&self.arena, self.root, &mut memo);
        root_h.hash(state);
    }
}

/// A chainable EdgeInserter that operates on the arena and NodeId handles.
/// It performs cycle checks and depth propagation by delegating to TrieArena::try_insert.
/// Merging semantics are delegated to user-provided closures.
pub struct EdgeInserter<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
where
    EK: Ord + Clone,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    arena: &'a mut TrieArena<EK, EV, T>,
    src: NodeId,
    ek: EK,
    ev_opt: Option<EV>,
    merge_ev: FMergeEV,
    update_node_val: FUpdateT,
    preload_ev_with_src_val: FMergeEV_T,
    result: Option<NodeId>,
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
    pub fn new(
        arena: &'a mut TrieArena<EK, EV, T>,
        src: NodeId,
        ek: EK,
        mut ev: EV,
        merge_ev: FMergeEV,
        update_node_val: FUpdateT,
        preload_ev_with_src_val: FMergeEV_T,
    ) -> Self {
        {
            let src_val = &arena.node(src).value;
            // allow EV to depend on the source node's current value
            preload_ev_with_src_val(&mut ev, src_val);
        }
        Self {
            arena,
            src,
            ek,
            ev_opt: Some(ev),
            merge_ev,
            update_node_val,
            preload_ev_with_src_val,
            result: None,
        }
    }

    pub fn try_destination(mut self, dst: NodeId) -> Self {
        if self.result.is_some() {
            return self;
        }

        // If edge already exists -> merge EV
        if let Some(ex) = self.arena.get_edge_value_mut(self.src, &self.ek, dst) {
            let new_ev = self.ev_opt.take().unwrap();
            (self.merge_ev)(ex, new_ev);
            let ev_for_update = ex.clone();
            self.result = Some(dst);
            // Update node value using merged EV
            (self.update_node_val)(&mut self.arena.node_mut(dst).value, &ev_for_update);
            return self;
        }

        // Else: try to insert (cycle-safe)
        if let Some(ev) = self.ev_opt.take() {
            if self.arena.try_insert(self.src, self.ek.clone(), ev.clone(), dst).is_ok() {
                self.result = Some(dst);
                (self.update_node_val)(&mut self.arena.node_mut(dst).value, &ev);
            } else {
                // cycle -> do nothing
            }
        }
        self
    }

    pub fn try_destinations(mut self, destinations: &[NodeId]) -> Self {
        for &dst in destinations {
            if self.result.is_some() {
                break;
            }
            self = self.try_destination(dst);
        }
        self
    }

    pub fn try_destinations_iter(mut self, mut it: impl Iterator<Item = NodeId>) -> Self {
        while let Some(dst) = it.next() {
            if self.result.is_some() {
                break;
            }
            self = self.try_destination(dst);
        }
        self
    }

    pub fn else_create_destination_with_value(mut self, value: T) -> Self {
        if self.result.is_some() {
            return self;
        }
        let new_id = self.arena.add_node(value);
        if let Some(ev) = self.ev_opt.take() {
            if self.arena.try_insert(self.src, self.ek.clone(), ev.clone(), new_id).is_ok() {
                (self.update_node_val)(&mut self.arena.node_mut(new_id).value, &ev);
                self.result = Some(new_id);
            }
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

    pub fn unwrap(self) -> NodeId {
        self.result
            .expect("EdgeInserter::unwrap() called but no destination was found or created")
    }

    pub fn clone_into_option(&self) -> Option<NodeId> {
        self.result
    }
}

// Convenience free functions mirroring the original style but using the arena + NodeId.
pub fn try_destination<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>(
    arena: &'a mut TrieArena<EK, EV, T>,
    src: NodeId,
    ek: EK,
    ev: EV,
    dst: NodeId,
    merge_ev: FMergeEV,
    update_node_val: FUpdateT,
    preload_ev_with_src_val: FMergeEV_T,
) -> Option<NodeId>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    EdgeInserter::new(arena, src, ek, ev, merge_ev, update_node_val, preload_ev_with_src_val)
        .try_destination(dst)
        .into_option()
}

pub fn try_destination_with<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>(
    arena: &'a mut TrieArena<EK, EV, T>,
    src: NodeId,
    ek: EK,
    ev: EV,
    destinations: &[NodeId],
    merge_ev: FMergeEV,
    update_node_val: FUpdateT,
    preload_ev_with_src_val: FMergeEV_T,
) -> Option<NodeId>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    EdgeInserter::new(arena, src, ek, ev, merge_ev, update_node_val, preload_ev_with_src_val)
        .try_destinations(destinations)
        .into_option()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    type EK = &'static str;
    type EV = &'static str;
    type T = i32;

    fn arc_ptr(id: NodeId) -> NodeId { id }

    #[test]
    fn test_basic_insert_and_retrieval() {
        let mut arena: TrieArena<EK, EV, T> = TrieArena::new();
        let root = arena.add_node(0);
        let c1 = arena.add_node(1);
        let c2 = arena.add_node(2);
        let c3 = arena.add_node(3);

        arena.try_insert(root, "a", "e1", c1).unwrap();
        arena.try_insert(root, "b", "e2", c2).unwrap();
        arena.try_insert(root, "a", "e3", c3).unwrap();

        assert_eq!(arena.node(root).children.len(), 2);
        let map_a = arena.node(root).children.get(&"a").unwrap();
        assert_eq!(map_a.len(), 2);
        let data: HashSet<(&str, NodeId)> =
            map_a.iter().map(|(&id, &ev)| (ev, arc_ptr(id))).collect();
        assert!(data.contains(&("e1", arc_ptr(c1))));
        assert!(data.contains(&("e3", arc_ptr(c3))));

        let map_b = arena.node(root).children.get(&"b").unwrap();
        assert_eq!(map_b.len(), 1);
        let (id, &ev) = map_b.iter().next().unwrap();
        assert_eq!(ev, "e2");
        assert_eq!(*id, c2);

        assert!(!arena.node(root).is_leaf());
        assert!(arena.node(c1).is_leaf());
        assert!(arena.node(c2).is_leaf());
        assert!(arena.node(c3).is_leaf());
    }

    #[test]
    fn test_cycle_detection() {
        let mut arena: TrieArena<EK, EV, T> = TrieArena::new();
        let a = arena.add_node(0);
        let b = arena.add_node(1);

        arena.try_insert(a, "ab", "e", b).unwrap();
        // Attempt b->a should cycle
        let r = arena.try_insert(b, "ba", "e2", a);
        assert!(r.is_err());

        // Ensure structure unchanged
        assert!(arena.node(b).children.get(&"ba").is_none());
    }

    #[test]
    fn test_special_map() {
        let mut arena: TrieArena<EK, EV, T> = TrieArena::new();
        let root = arena.add_node(0);
        let c1 = arena.add_node(1);
        let c2 = arena.add_node(2);
        let gc = arena.add_node(3);

        arena.try_insert(root, "r->c1", "e1", c1).unwrap();
        arena.try_insert(root, "r->c2", "e2", c2).unwrap();
        arena.try_insert(c1, "c1->gc", "e3", gc).unwrap();

        // recompute depths from root
        arena.recompute_all_max_depths(&[root]);

        let mut processed_vals = Vec::new();
        let mut computed_vals = Vec::new();
        let mut edges_seen = Vec::new();

        arena.special_map(
            vec![(root, 100)],
            |parent_val, ek, ev, _child| {
                edges_seen.push((ek.clone(), ev.clone()));
                Some(parent_val + 1)
            },
            |cur, new| *cur = new,
            |node, v| {
                processed_vals.push(node.value);
                computed_vals.push(*v);
                true
            },
        );

        assert!(processed_vals.contains(&0));
        assert!(processed_vals.contains(&1));
        assert!(processed_vals.contains(&2));
        assert!(processed_vals.contains(&3));
        // root depth 0
        // c1/c2 depth 1
        // gc depth 2
        assert_eq!(arena.node(root).max_depth, 0);
        assert_eq!(arena.node(c1).max_depth, 1);
        assert_eq!(arena.node(c2).max_depth, 1);
        assert_eq!(arena.node(gc).max_depth, 2);

        assert_eq!(edges_seen.len(), 3);
        assert!(edges_seen.contains(&("r->c1", "e1")));
        assert!(edges_seen.contains(&("r->c2", "e2")));
        assert!(edges_seen.contains(&("c1->gc", "e3")));
    }

    #[test]
    fn test_json_roundtrip() {
        let mut arena: TrieArena<EK, EV, T> = TrieArena::new();
        let root = arena.add_node(0);
        let a = arena.add_node(1);
        let b = arena.add_node(2);
        arena.try_insert(root, "x", "e1", a).unwrap();
        arena.try_insert(root, "y", "e2", b).unwrap();
        arena.recompute_all_max_depths(&[root]);

        let g1 = RootedTrie::new(arena.clone(), root);
        let json = g1.to_json();
        let g2 = RootedTrie::<EK, EV, T>::from_json(json).expect("from_json failed");
        assert_eq!(g1, g2);
    }

    #[test]
    fn test_edge_inserter_chain() {
        let mut arena: TrieArena<&str, Vec<i32>, String> = TrieArena::new();
        let root = arena.add_node("root".to_string());
        let d1 = arena.add_node("d1".to_string());
        let d2 = arena.add_node("d2".to_string());

        let merge_vec = |e: &mut Vec<i32>, n: Vec<i32>| e.extend(n);
        let upd = |t: &mut String, ev: &Vec<i32>| {
            if !ev.is_empty() {
                t.push_str("+");
            }
        };
        let preload = |ev: &mut Vec<i32>, _src: &String| {
            // no-op
            let _ = ev;
        };

        // Insert to d1
        let id = EdgeInserter::new(&mut arena, root, "k", vec![1], merge_vec, upd, preload)
            .try_destination(d1)
            .unwrap();
        assert_eq!(id, d1);

        // Merge into existing edge (append)
        let _ = EdgeInserter::new(&mut arena, root, "k", vec![2, 3], merge_vec, upd, preload)
            .try_destination(d1)
            .unwrap();

        // Try destination that creates cycle -> should not insert
        // Create a path d2 -> root first
        arena.try_insert(d2, "back", vec![0], root).unwrap();
        let res = EdgeInserter::new(&mut arena, root, "cyc", vec![9], merge_vec, upd, preload)
            .try_destination(d2)
            .into_option();
        assert!(res.is_none());
    }
}
