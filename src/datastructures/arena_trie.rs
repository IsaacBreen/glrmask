//! Arena-based Trie (graph-like) implementation without Arc/Mutex.
//! Nodes are identified by NodeId (usize) and stored in an arena.
//!
//! Key features compared to the pointer-based version:
//! - No Arc/RwLock/Weak pointers. Everything is single-threaded and managed by an arena.
//! - Edges can be Strong or Weak (logical notion only), without relying on weak pointers.
//! - Cycle detection and depth propagation operate over Strong edges only.
//! - A small set of algorithms is provided (insert with cycle checks, special_map traversal, etc.).
//!
//! This module is self-contained and includes unit tests demonstrating usage.

use std::collections::{BTreeMap, VecDeque, HashSet, HashMap};
use std::fmt::Debug;

use ordered_hash_map::OrderedHashMap;

/// Node identifier in the arena
pub type NodeId = usize;

/// Error indicating that a cycle would be created by a strong edge insertion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleDetectedError;

/// Classification of edges
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Strong,
    Weak,
}

/// Result indicating whether a requested insertion became a strong or weak edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertedEdgeKind {
    Strong,
    Weak,
}

/// Internal structure holding target destinations for a single edge key.
/// We separate strong vs weak edges explicitly.
#[derive(Debug, Clone)]
pub struct EdgeList<EV> {
    pub strong: OrderedHashMap<NodeId, EV>,
    pub weak: OrderedHashMap<NodeId, EV>,
}

impl<EV> Default for EdgeList<EV> {
    fn default() -> Self {
        Self {
            strong: OrderedHashMap::new(),
            weak: OrderedHashMap::new(),
        }
    }
}

impl<EV: Clone> EdgeList<EV> {
    fn total_len(&self) -> usize {
        self.strong.len() + self.weak.len()
    }
}

/// A node in the arena-based Trie
#[derive(Debug, Clone)]
pub struct TrieNode<EK: Ord, EV, T> {
    pub value: T,
    pub children: BTreeMap<EK, EdgeList<EV>>,
    pub max_depth: usize, // longest distance from some source via strong edges
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
        self.children.values().all(|el| el.strong.is_empty())
    }
}

/// The arena (registry) that stores nodes
#[derive(Debug, Clone)]
pub struct TrieArena<EK: Ord + Clone, EV: Clone, T: Clone> {
    nodes: Vec<TrieNode<EK, EV, T>>,
}

impl<EK: Ord + Clone, EV: Clone, T: Clone> Default for TrieArena<EK, EV, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<EK: Ord + Clone, EV: Clone, T: Clone> TrieArena<EK, EV, T> {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn create_node(&mut self, value: T) -> NodeId {
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

    /// Force insert: create a new node as child of `src` with a strong edge.
    pub fn force_insert_to_new_node(&mut self, src: NodeId, edge_key: EK, edge_value: EV, value: T) -> NodeId {
        let dst = self.create_node(value);
        self.force_insert_to_node(src, edge_key, edge_value, dst, EdgeKind::Strong);
        dst
    }

    /// Force insert: add an edge of given kind, no checks.
    pub fn force_insert_to_node(&mut self, src: NodeId, edge_key: EK, edge_value: EV, dst: NodeId, kind: EdgeKind) {
        let el = self.nodes[src].children.entry(edge_key).or_default();
        match kind {
            EdgeKind::Strong => {
                el.strong.insert(dst, edge_value);
            }
            EdgeKind::Weak => {
                el.weak.insert(dst, edge_value);
            }
        }
    }

    /// Returns whether src already has an edge (strong or weak) under any key to dst
    pub fn already_has_dst_for_any_key(&self, src: NodeId, dst: NodeId) -> bool {
        self.nodes[src].children.values().any(|el| el.strong.contains_key(&dst) || el.weak.contains_key(&dst))
    }

    /// Return edge value for strong edge under edge_key
    pub fn get_edge_value(&self, src: NodeId, edge_key: &EK, dst: NodeId) -> Option<&EV> {
        self.nodes[src].children.get(edge_key).and_then(|el| el.strong.get(&dst))
    }

    /// Return edge value for weak edge under edge_key
    pub fn get_weak_edge_value(&self, src: NodeId, edge_key: &EK, dst: NodeId) -> Option<&EV> {
        self.nodes[src].children.get(edge_key).and_then(|el| el.weak.get(&dst))
    }

    /// Try to insert a strong edge (src --ek--> dst) with cycle detection and depth propagation.
    pub fn try_insert(&mut self, src: NodeId, edge_key: EK, mut edge_value: EV, dst: NodeId) -> Result<(), CycleDetectedError> {
        // If adding this strong edge would create a cycle, reject.
        if !self.already_has_dst_for_any_key(src, dst) && self.detect_cycle(src, dst) {
            return Err(CycleDetectedError);
        }

        // Update child's depth if needed and propagate along strong edges
        let candidate_depth = self.nodes[src].max_depth.saturating_add(1);
        let prev_depth = self.nodes[dst].max_depth;
        let needs_update = candidate_depth > prev_depth;
        if needs_update {
            self.nodes[dst].max_depth = candidate_depth;
            if let Err(_) = self.propagate_max_depth(dst, candidate_depth) {
                // rollback
                if self.nodes[dst].max_depth == candidate_depth {
                    self.nodes[dst].max_depth = prev_depth;
                }
                return Err(CycleDetectedError);
            }
        }

        let el = self.nodes[src].children.entry(edge_key).or_default();
        el.strong.insert(dst, edge_value);
        Ok(())
    }

    /// Like try_insert, but degrade to Weak edge if a cycle would be created.
    pub fn try_insert_auto(&mut self, src: NodeId, edge_key: EK, edge_value: EV, dst: NodeId) -> InsertedEdgeKind {
        // If it already exists, make sure it's a strong insertion path:
        let would_cycle = if self.already_has_dst_for_any_key(src, dst) {
            false
        } else {
            self.detect_cycle(src, dst)
        };

        if would_cycle {
            let el = self.nodes[src].children.entry(edge_key).or_default();
            el.weak.insert(dst, edge_value);
            InsertedEdgeKind::Weak
        } else {
            self.try_insert(src, edge_key, edge_value, dst).expect("Unexpected cycle in try_insert_auto");
            InsertedEdgeKind::Strong
        }
    }

    /// Detect whether adding a strong edge src -> dst would create a cycle,
    /// i.e., detect if src is reachable from dst via existing strong edges.
    pub fn detect_cycle(&self, src: NodeId, dst: NodeId) -> bool {
        // BFS from dst along strong edges; if we reach src => cycle.
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut q = VecDeque::new();

        visited.insert(dst);
        q.push_back(dst);

        while let Some(u) = q.pop_front() {
            if u == src {
                return true;
            }
            for (_ek, el) in &self.nodes[u].children {
                for (&v, _) in el.strong.iter() {
                    if visited.insert(v) {
                        q.push_back(v);
                    }
                }
            }
        }

        false
    }

    /// Propagate max_depth updates forward along strong edges from `node`, ensuring no cycles occur.
    fn propagate_max_depth(&mut self, node: NodeId, current_depth: usize) -> Result<(), CycleDetectedError> {
        let mut rec_stack: HashSet<NodeId> = HashSet::new();
        self._propagate_max_depth(node, current_depth, &mut rec_stack)
    }

    fn _propagate_max_depth(&mut self, node: NodeId, current_depth: usize, rec_stack: &mut HashSet<NodeId>) -> Result<(), CycleDetectedError> {
        if rec_stack.contains(&node) {
            return Err(CycleDetectedError);
        }
        rec_stack.insert(node);

        let child_depth = current_depth.saturating_add(1);
        let children: Vec<NodeId> = self.nodes[node]
            .children
            .values()
            .flat_map(|el| el.strong.keys().cloned())
            .collect();

        for c in children {
            if child_depth > self.nodes[c].max_depth {
                self.nodes[c].max_depth = child_depth;
                self._propagate_max_depth(c, child_depth, rec_stack)?;
            }
        }

        rec_stack.remove(&node);
        Ok(())
    }

    /// Collect all nodes reachable from given roots (both strong and weak edges are used for reachability).
    pub fn all_nodes(&self, roots: &[NodeId]) -> Vec<NodeId> {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut order = Vec::new();
        let mut q = VecDeque::new();

        for &r in roots {
            if visited.insert(r) {
                q.push_back(r);
            }
        }

        while let Some(u) = q.pop_front() {
            order.push(u);
            for (_ek, el) in &self.nodes[u].children {
                for (&v, _) in el.strong.iter().chain(el.weak.iter()) {
                    if visited.insert(v) {
                        q.push_back(v);
                    }
                }
            }
        }

        order
    }

    /// Check if any strong cycle exists reachable from `root`.
    pub fn has_any_cycle(&self, root: NodeId) -> bool {
        fn dfs<EK: Ord + Clone, EV: Clone, T: Clone>(
            arena: &TrieArena<EK, EV, T>,
            u: NodeId,
            visited: &mut HashSet<NodeId>,
            rec: &mut HashSet<NodeId>,
        ) -> bool {
            if rec.contains(&u) {
                return true;
            }
            if !visited.insert(u) {
                return false;
            }
            rec.insert(u);
            for (_ek, el) in &arena.nodes[u].children {
                for (&v, _) in el.strong.iter() {
                    if dfs(arena, v, visited, rec) {
                        return true;
                    }
                }
            }
            rec.remove(&u);
            false
        }

        let mut visited = HashSet::new();
        let mut rec = HashSet::new();
        dfs(self, root, &mut visited, &mut rec)
    }

    /// Recompute max_depth for all nodes reachable from `roots`, using only strong edges (Kahn's algorithm).
    pub fn recompute_all_max_depths(&mut self, roots: &[NodeId]) {
        let nodes = self.all_nodes(roots);
        if nodes.is_empty() {
            return;
        }

        let mut in_deg: HashMap<NodeId, usize> = HashMap::new();
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for &u in &nodes {
            in_deg.entry(u).or_insert(0);
            adj.entry(u).or_default();
        }

        for &u in &nodes {
            for (_ek, el) in &self.nodes[u].children {
                for (&v, _) in el.strong.iter() {
                    adj.get_mut(&u).unwrap().push(v);
                    *in_deg.entry(v).or_insert(0) += 1;
                }
            }
        }

        let mut q = VecDeque::new();
        for &u in &nodes {
            self.nodes[u].max_depth = 0; // reset
            if *in_deg.get(&u).unwrap_or(&0) == 0 {
                q.push_back(u);
            }
        }

        while let Some(u) = q.pop_front() {
            let d = self.nodes[u].max_depth;
            if let Some(children) = adj.get(&u) {
                for &v in children {
                    if self.nodes[v].max_depth < d + 1 {
                        self.nodes[v].max_depth = d + 1;
                    }
                    let entry = in_deg.get_mut(&v).unwrap();
                    *entry -= 1;
                    if *entry == 0 {
                        q.push_back(v);
                    }
                }
            }
        }
    }

    /// Promote weak edges to strong when doing so does not create a strong cycle.
    /// Returns the count of promotions made.
    pub fn promote_weak_edges_to_strong(&mut self, roots: &[NodeId]) -> usize {
        let all = self.all_nodes(roots);
        let mut promotions = 0usize;

        for &src in &all {
            // Collect weak edges snapshot to avoid borrow issues
            let snapshot: Vec<(EK, Vec<(NodeId, EV)>)> = self.nodes[src]
                .children
                .iter()
                .map(|(ek, el)| {
                    let list = el.weak.iter().map(|(dst, ev)| (*dst, ev.clone())).collect::<Vec<_>>();
                    (ek.clone(), list)
                })
                .collect();

            for (ek, items) in snapshot {
                for (dst, ev) in items {
                    // Check if promoting would create cycle: is src reachable from dst via strong edges
                    if self.detect_cycle(src, dst) {
                        continue;
                    }
                    // Promote: remove from weak, insert into strong, adjust depths
                    // First compute candidate depth
                    let candidate_depth = self.nodes[src].max_depth.saturating_add(1);
                    let prev_depth = self.nodes[dst].max_depth;
                    let needs_update = candidate_depth > prev_depth;
                    if needs_update {
                        self.nodes[dst].max_depth = candidate_depth;
                        if let Err(_) = self.propagate_max_depth(dst, candidate_depth) {
                            // rollback on failure; skip promotion
                            if self.nodes[dst].max_depth == candidate_depth {
                                self.nodes[dst].max_depth = prev_depth;
                            }
                            continue;
                        }
                    }

                    // Rewire el: move from weak to strong (if it still exists)
                    if let Some(el) = self.nodes[src].children.get_mut(&ek) {
                        if let Some(val) = el.weak.remove(&dst) {
                            el.strong.insert(dst, ev); // we use original ev (not val) but both are identical clone type
                            promotions += 1;
                        }
                    }
                }
            }
        }

        promotions
    }

    /// A simple traversal akin to Dijkstra/BFS using depth ordering (by max_depth).
    /// Traverses both strong and weak edges. The scheduler visits nodes grouped by their max_depth.
    /// - initial: starting set of (node_id, V)
    /// - step: given (&V, &EK, &EV, &TrieNode) compute child V
    /// - merge: merge values for a node
    /// - process: on visiting node, can mutate &mut V and decide to continue or not
    pub fn special_map<V: Clone>(
        &self,
        initial: Vec<(NodeId, V)>,
        mut step: impl FnMut(&V, &EK, &EV, &TrieNode<EK, EV, T>) -> Option<V>,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&TrieNode<EK, EV, T>, &mut V) -> bool,
    ) {
        use ordered_hash_map::OrderedHashSet;
        let mut values: HashMap<NodeId, V> = HashMap::new();
        let mut stopped: HashSet<NodeId> = HashSet::new();
        let mut buckets: BTreeMap<usize, OrderedHashSet<NodeId>> = BTreeMap::new();

        for (n, v0) in initial {
            values.entry(n).and_modify(|old| merge(old, v0.clone())).or_insert(v0);
            let d = self.nodes[n].max_depth;
            buckets.entry(d).or_default().insert(n);
        }

        while let Some((_d, ids)) = buckets.pop_first() {
            for node_id in ids {
                if stopped.contains(&node_id) {
                    continue;
                }
                let mut v = match values.remove(&node_id) {
                    Some(v) => v,
                    None => continue,
                };

                let proceed = process(&self.nodes[node_id], &mut v);
                if !proceed {
                    stopped.insert(node_id);
                    continue;
                }

                // traverse both strong and weak
                // Snapshot edges
                let edges: Vec<(EK, EV, NodeId)> = self.nodes[node_id]
                    .children
                    .iter()
                    .flat_map(|(ek, el)| {
                        el.strong.iter().map(move |(dst, ev)| (ek.clone(), ev.clone(), *dst))
                            .chain(el.weak.iter().map(move |(dst, ev)| (ek.clone(), ev.clone(), *dst)))
                    })
                    .collect();

                for (ek, ev, child) in edges {
                    if stopped.contains(&child) {
                        continue;
                    }
                    if let Some(new_v) = step(&v, &ek, &ev, &self.nodes[child]) {
                        values.entry(child).and_modify(|old| merge(old, new_v.clone())).or_insert(new_v);
                        let d = self.nodes[child].max_depth;
                        buckets.entry(d).or_default().insert(child);
                    }
                }
            }
        }
    }

    /// A grouped variant of special_map where step receives the whole destinations map under a key.
    /// step returns an iterator of (NodeId, V) to enqueue.
    pub fn special_map_grouped<V, S, I>(
        &self,
        initial: Vec<(NodeId, V)>,
        mut step: S,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&TrieNode<EK, EV, T>, &mut V) -> bool,
    )
    where
        V: Clone,
        S: FnMut(&V, &EK, &EdgeList<EV>) -> I,
        I: IntoIterator<Item = (NodeId, V)>,
    {
        use ordered_hash_map::OrderedHashSet;
        let mut values: HashMap<NodeId, V> = HashMap::new();
        let mut stopped: HashSet<NodeId> = HashSet::new();
        let mut buckets: BTreeMap<usize, OrderedHashSet<NodeId>> = BTreeMap::new();

        for (n, v0) in initial {
            values.entry(n).and_modify(|old| merge(old, v0.clone())).or_insert(v0);
            let d = self.nodes[n].max_depth;
            buckets.entry(d).or_default().insert(n);
        }

        while let Some((_d, ids)) = buckets.pop_first() {
            for node_id in ids {
                if stopped.contains(&node_id) {
                    continue;
                }
                let mut v = match values.remove(&node_id) {
                    Some(v) => v,
                    None => continue,
                };

                let proceed = process(&self.nodes[node_id], &mut v);
                if !proceed {
                    stopped.insert(node_id);
                    continue;
                }

                // For each edge key, hand the entire EdgeList to step
                let children_snapshot: Vec<(EK, EdgeList<EV>)> = self.nodes[node_id]
                    .children
                    .iter()
                    .map(|(ek, el)| (ek.clone(), el.clone()))
                    .collect();

                for (ek, el) in children_snapshot {
                    let out = step(&v, &ek, &el);
                    for (dst, v2) in out {
                        if stopped.contains(&dst) {
                            continue;
                        }
                        values.entry(dst).and_modify(|old| merge(old, v2.clone())).or_insert(v2);
                        let d = self.nodes[dst].max_depth;
                        buckets.entry(d).or_default().insert(dst);
                    }
                }
            }
        }
    }
}

/// A convenience builder to insert edges with merging logic.
/// This is similar to the pointer-based EdgeInserter but references the arena and node ids.
pub struct EdgeInserter<'a, EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    arena: &'a mut TrieArena<EK, EV, T>,
    src: NodeId,
    edge_key: EK,
    edge_value: Option<EV>,
    merge_edge_value: FMergeEV,
    update_node_value: FUpdateT,
    merge_edge_value_and_src_value: FMergeEV_T,
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
        arena: &'a mut TrieArena<EK, EV, T>,
        src: NodeId,
        edge_key: EK,
        mut edge_value: EV,
        mut merge_edge_value: FMergeEV,
        update_node_value: FUpdateT,
        mut merge_edge_value_and_src_value: FMergeEV_T,
    ) -> Self {
        // Merge edge value with source node's value if needed (user logic)
        let src_val = arena.node(src).value.clone();
        merge_edge_value_and_src_value(&mut edge_value, &src_val);

        Self {
            arena,
            src,
            edge_key,
            edge_value: Some(edge_value),
            merge_edge_value,
            update_node_value,
            merge_edge_value_and_src_value,
            result: None,
        }
    }

    /// Try to insert or merge a strong edge to destination DST.
    pub fn try_destination(mut self, dst: NodeId) -> Self {
        if self.result.is_some() {
            return self;
        }
        let src = self.src;
        // If edge exists -> merge EV
        let mut updated_ev_opt: Option<EV> = None;

        if let Some(el) = self.arena.node(src).children.get(&self.edge_key) {
            if let Some(ev) = el.strong.get(&dst) {
                // merge
                let mut new_ev = self.edge_value.take().unwrap();
                let mut cur_ev = ev.clone();
                (self.merge_edge_value)(&mut cur_ev, new_ev);
                updated_ev_opt = Some(cur_ev);
            }
        }

        if let Some(new_ev) = updated_ev_opt {
            // write back updated ev
            if let Some(el) = self.arena.node_mut(src).children.get_mut(&self.edge_key) {
                el.strong.insert(dst, new_ev.clone());
            }
            self.result = Some(dst);
            // Update destination's node value using EV
            let ev_copy = new_ev.clone();
            (self.update_node_value)(&mut self.arena.node_mut(dst).value, &ev_copy);
            return self;
        }

        // No existing strong edge -> try insertion (cycle-aware)
        let ev_clone = self.edge_value.as_ref().unwrap().clone();
        let insert_res = self.arena.try_insert(src, self.edge_key.clone(), ev_clone.clone(), dst);
        if insert_res.is_ok() {
            self.result = Some(dst);
            (self.update_node_value)(&mut self.arena.node_mut(dst).value, &ev_clone);
        }
        self
    }

    /// Try to insert a weak edge if strong would create cycle.
    pub fn try_destination_auto(mut self, dst: NodeId) -> Self {
        if self.result.is_some() {
            return self;
        }
        let src = self.src;

        // Try merge
        let mut updated_ev_opt: Option<EV> = None;
        if let Some(el) = self.arena.node(src).children.get(&self.edge_key) {
            if let Some(ev) = el.strong.get(&dst) {
                let mut new_ev = self.edge_value.take().unwrap();
                let mut cur_ev = ev.clone();
                (self.merge_edge_value)(&mut cur_ev, new_ev);
                updated_ev_opt = Some(cur_ev);
            }
        }
        if let Some(new_ev) = updated_ev_opt {
            if let Some(el) = self.arena.node_mut(src).children.get_mut(&self.edge_key) {
                el.strong.insert(dst, new_ev.clone());
            }
            self.result = Some(dst);
            (self.update_node_value)(&mut self.arena.node_mut(dst).value, &new_ev);
            return self;
        }

        // Not present: try strong, else weak
        let ev_clone = self.edge_value.as_ref().unwrap().clone();
        let kind = self.arena.try_insert_auto(src, self.edge_key.clone(), ev_clone.clone(), dst);
        self.result = Some(dst);
        (self.update_node_value)(&mut self.arena.node_mut(dst).value, &ev_clone);
        let _ = kind;
        self
    }

    /// Insert explicitly as weak (always exists after).
    pub fn to_destination_weakly(mut self, dst: NodeId) -> Self {
        if self.result.is_some() {
            return self;
        }
        let src = self.src;

        // Attempt merge with strong if exists; strong remains strong
        let mut updated_ev_opt: Option<EV> = None;
        if let Some(el) = self.arena.node(src).children.get(&self.edge_key) {
            if let Some(ev) = el.strong.get(&dst) {
                let mut new_ev = self.edge_value.take().unwrap();
                let mut cur_ev = ev.clone();
                (self.merge_edge_value)(&mut cur_ev, new_ev);
                updated_ev_opt = Some(cur_ev);
            }
        }
        if let Some(new_ev) = updated_ev_opt {
            if let Some(el) = self.arena.node_mut(src).children.get_mut(&self.edge_key) {
                el.strong.insert(dst, new_ev.clone());
            }
            self.result = Some(dst);
            (self.update_node_value)(&mut self.arena.node_mut(dst).value, &new_ev);
            return self;
        }

        // Otherwise insert/update weak map
        let ev = self.edge_value.take().unwrap();
        let el = self.arena.node_mut(src).children.entry(self.edge_key.clone()).or_default();
        if let Some(existing) = el.weak.get_mut(&dst) {
            (self.merge_edge_value)(existing, ev.clone());
        } else {
            el.weak.insert(dst, ev.clone());
        }
        self.result = Some(dst);
        (self.update_node_value)(&mut self.arena.node_mut(dst).value, &ev);
        self
    }

    pub fn else_create_destination_with_value(mut self, value: T) -> Self {
        if self.result.is_some() {
            return self;
        }
        let dst = self.arena.create_node(value.clone());
        let ev = self.edge_value.take().unwrap();
        // Strong insertion with cycle-checks
        let _ = self.arena.try_insert(self.src, self.edge_key.clone(), ev.clone(), dst);
        self.result = Some(dst);
        (self.update_node_value)(&mut self.arena.node_mut(dst).value, &ev);
        self
    }

    pub fn into_option(self) -> Option<NodeId> {
        self.result
    }

    pub fn unwrap(self) -> NodeId {
        self.result.expect("EdgeInserter::unwrap() called but no destination was found or created")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashSet, HashMap};

    // Basic test types
    type EK = &'static str;
    type EV = &'static str;
    type T = i32;

    #[test]
    fn test_create_and_basic_insert() {
        let mut arena: TrieArena<EK, EV, T> = TrieArena::new();
        let root = arena.create_node(0);
        let c1 = arena.create_node(1);
        let c2 = arena.create_node(2);

        // Insert edges
        arena.try_insert(root, "a", "e1", c1).unwrap();
        arena.try_insert(root, "b", "e2", c2).unwrap();

        assert_eq!(arena.node(root).children.len(), 2);
        assert_eq!(arena.get_edge_value(root, &"a", c1), Some(&"e1"));
        assert_eq!(arena.get_edge_value(root, &"b", c2), Some(&"e2"));
        assert_eq!(arena.node(c1).max_depth, 1);
        assert_eq!(arena.node(c2).max_depth, 1);
        assert!(!arena.node(root).is_leaf());
        assert!(arena.node(c1).is_leaf());
        assert!(arena.node(c2).is_leaf());
    }

    #[test]
    fn test_multiple_children_same_key() {
        let mut arena: TrieArena<EK, EV, T> = TrieArena::new();
        let root = arena.create_node(0);
        let c1 = arena.create_node(1);
        let c2 = arena.create_node(2);

        arena.try_insert(root, "edge", "v1", c1).unwrap();
        arena.try_insert(root, "edge", "v2", c2).unwrap();

        let el = arena.node(root).children.get("edge").unwrap();
        assert_eq!(el.strong.len(), 2);
        assert_eq!(el.strong.get(&c1), Some(&"v1"));
        assert_eq!(el.strong.get(&c2), Some(&"v2"));
        assert_eq!(arena.node(c1).max_depth, 1);
        assert_eq!(arena.node(c2).max_depth, 1);

        // all_nodes
        let all = arena.all_nodes(&[root]);
        let set: HashSet<_> = all.into_iter().collect();
        assert!(set.contains(&root));
        assert!(set.contains(&c1));
        assert!(set.contains(&c2));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_special_map_simple() {
        let mut arena: TrieArena<EK, EV, T> = TrieArena::new();
        let root = arena.create_node(0);
        let c1 = arena.create_node(1);
        let c2 = arena.create_node(2);
        let gc = arena.create_node(3);

        arena.try_insert(root, "r->c1", "e1", c1).unwrap();
        arena.try_insert(root, "r->c2", "e2", c2).unwrap();
        arena.try_insert(c1, "c1->gc", "e3", gc).unwrap();

        let mut processed_vals = Vec::new();
        let mut computed = Vec::new();
        let mut edge_seen = Vec::new();

        arena.special_map(
            vec![(root, 100)],
            |p, ek, ev, _child| {
                edge_seen.push((ek.to_string(), ev.to_string()));
                Some(p + 1)
            },
            |cur, new| *cur = new,
            |node, v| {
                processed_vals.push(node.value);
                computed.push(*v);
                true
            }
        );

        // Depths: root 0, c1 1, c2 1, gc 2
        assert_eq!(processed_vals.len(), 4);
        assert_eq!(processed_vals[0], 0);
        let set: HashSet<_> = processed_vals[1..3].iter().cloned().collect();
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert_eq!(processed_vals[3], 3);

        let map: HashMap<_, _> = processed_vals.into_iter().zip(computed.into_iter()).collect();
        assert_eq!(map.get(&0), Some(&100));
        assert_eq!(map.get(&1), Some(&101));
        assert_eq!(map.get(&2), Some(&101));
        assert_eq!(map.get(&3), Some(&102));

        assert_eq!(edge_seen.len(), 3);
        assert!(edge_seen.contains(&("r->c1".to_string(), "e1".to_string())));
        assert!(edge_seen.contains(&("r->c2".to_string(), "e2".to_string())));
        assert!(edge_seen.contains(&("c1->gc".to_string(), "e3".to_string())));
    }

    #[test]
    fn test_cycle_detection_strong_insert_fails() {
        let mut arena: TrieArena<EK, EV, T> = TrieArena::new();
        let root = arena.create_node(0);
        let child = arena.create_node(1);

        arena.try_insert(root, "r->c", "e1", child).unwrap();
        // Next would make cycle (child->root) in strong graph
        let res = arena.try_insert(child, "c->r", "e2", root);
        assert_eq!(res, Err(CycleDetectedError));

        // Ensure no edge added
        assert!(arena.get_edge_value(child, &"c->r", root).is_none());
        // Depths unchanged
        assert_eq!(arena.node(root).max_depth, 0);
        assert_eq!(arena.node(child).max_depth, 1);
    }

    #[test]
    fn test_special_map_stop_processing() {
        let mut arena: TrieArena<EK, EV, T> = TrieArena::new();
        let root = arena.create_node(0);
        let c1 = arena.create_node(1);
        let c2 = arena.create_node(2);
        let gc1 = arena.create_node(3);
        let gc2 = arena.create_node(4);

        arena.try_insert(root, "r->c1", "e1", c1).unwrap();
        arena.try_insert(root, "r->c2", "e2", c2).unwrap();
        arena.try_insert(c1, "c1->gc", "e3", gc1).unwrap();
        arena.try_insert(c2, "c2->gc", "e4", gc2).unwrap();

        let mut processed = HashSet::new();
        let mut values = HashMap::new();

        arena.special_map(
            vec![(root, 100)],
            |p, _ek, _ev, _child| Some(p + 1),
            |cur, new| *cur = new,
            |node, v| {
                processed.insert(node.value);
                values.insert(node.value, *v);
                // Stop propagation at node 1
                node.value != 1
            }
        );

        assert!(processed.contains(&0));
        assert!(processed.contains(&1));
        assert!(processed.contains(&2));
        assert!(processed.contains(&4));
        assert!(!processed.contains(&3));

        assert_eq!(values.get(&0), Some(&100));
        assert_eq!(values.get(&1), Some(&101));
        assert_eq!(values.get(&2), Some(&101));
        assert_eq!(values.get(&4), Some(&102));
        assert!(!values.contains_key(&3));
    }

    #[test]
    fn test_try_insert_auto_weak_on_cycle() {
        let mut arena: TrieArena<EK, EV, T> = TrieArena::new();
        let a = arena.create_node(0);
        let b = arena.create_node(1);

        // Strong edge a->b
        assert_eq!(arena.try_insert_auto(a, "k", "v", b), InsertedEdgeKind::Strong);
        assert_eq!(arena.node(b).max_depth, 1);
        // Attempt b->a would cycle: becomes weak
        assert_eq!(arena.try_insert_auto(b, "k2", "v2", a), InsertedEdgeKind::Weak);
        assert!(arena.get_edge_value(b, &"k2", a).is_none());
        assert_eq!(arena.get_weak_edge_value(b, &"k2", a), Some(&"v2"));
    }

    #[test]
    fn test_promote_weak_edges() {
        let mut arena: TrieArena<EK, EV, T> = TrieArena::new();
        let r = arena.create_node(0);
        let a = arena.create_node(1);
        let b = arena.create_node(2);

        // r -> a (strong), a -> b (strong), b -> r (weak) forms weak cycle
        arena.try_insert(r, "ra", "e1", a).unwrap();
        arena.try_insert(a, "ab", "e2", b).unwrap();
        let k = "br";
        arena.force_insert_to_node(b, k, "e3", r, EdgeKind::Weak);
        // Initially weak due to potential cycle if strong
        assert!(arena.get_edge_value(b, &k, r).is_none());
        assert_eq!(arena.get_weak_edge_value(b, &k, r), Some(&"e3"));

        // Can't promote b->r because it would create strong cycle (r reachable from r via strong path).
        let promoted = arena.promote_weak_edges_to_strong(&[r]);
        assert_eq!(promoted, 0);

        // Add a new node c that breaks the cycle when attaching from r
        let c = arena.create_node(3);
        let k2 = "rc";
        arena.force_insert_to_node(r, k2, "e4", c, EdgeKind::Weak); // Currently weak
        assert!(arena.get_edge_value(r, &k2, c).is_none());
        let promoted2 = arena.promote_weak_edges_to_strong(&[r]);
        // Promoting r->c is always fine (no cycle). It should convert to strong.
        assert_eq!(promoted2, 1);
        assert_eq!(arena.get_edge_value(r, &k2, c), Some(&"e4"));
        assert!(arena.get_weak_edge_value(r, &k2, c).is_none());
    }

    #[test]
    fn test_edge_inserter_basic() {
        // Demonstrate EdgeInserter usage
        let mut arena: TrieArena<&str, Vec<i32>, String> = TrieArena::new();
        let root = arena.create_node("root".to_string());
        let d1 = arena.create_node("d1".to_string());

        // Merge function: append new list to existing
        fn merge_ev(existing: &mut Vec<i32>, new: Vec<i32>) {
            existing.extend(new.into_iter());
        }
        // Update node value: append numbers length to the string
        fn update_node_value(val: &mut String, ev: &Vec<i32>) {
            val.push_str(&format!("+{}", ev.len()));
        }
        // Merge EV and source node value: push length of src string into ev
        fn mix_ev_src(ev: &mut Vec<i32>, src_val: &String) {
            ev.push(src_val.len() as i32);
        }

        let inserter = EdgeInserter::new(&mut arena, root, "k", vec![1], merge_ev, update_node_value, mix_ev_src);
        let got = inserter.try_destination(d1).unwrap();

        assert_eq!(got, d1);
        // Edge exists
        let el = arena.node(root).children.get("k").unwrap();
        let ev = el.strong.get(&d1).unwrap();
        // ev contains 1 plus len("root")=4 from mix_ev_src: so ev == [1,4]
        assert_eq!(ev, &vec![1, 4]);
        // d1 value updated with "+2"
        assert_eq!(arena.node(d1).value, "d1+2");
    }
}
