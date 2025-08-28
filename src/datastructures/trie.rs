use std::collections::{BTreeMap, VecDeque, HashMap, HashSet, BTreeSet};
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

use deterministic_hash::DeterministicHasher;
use ordered_hash_map::{OrderedHashMap, OrderedHashSet};
use kdam::{tqdm, BarExt};

use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::profiler::PROGRESS_BAR_ENABLED;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleDetectedError;

impl fmt::Display for CycleDetectedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Cycle detected in Trie structure")
    }
}

pub type NodeId = usize;

/// A simple arena that owns nodes contiguously and exposes NodeId handles.
#[derive(Debug, Default)]
pub struct Arena<N> {
    nodes: Vec<N>,
}
impl<N> Arena<N> {
    pub fn new() -> Self { Self { nodes: Vec::new() } }
    pub fn clear(&mut self) { self.nodes.clear(); }
    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }
    pub fn alloc(&mut self, node: N) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }
    pub fn get(&self, id: NodeId) -> &N {
        &self.nodes[id]
    }
    pub fn get_mut(&mut self, id: NodeId) -> &mut N {
        &mut self.nodes[id]
    }
    pub fn iter_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        0..self.nodes.len()
    }
}

/// A "God" that owns the arena for a particular Trie instantiation.
/// It is wrapped in an Arc<RwLock<...>> via GodWrapper so you can pass it around cheaply,
/// while avoiding any Arc use inside the Trie structure itself.
#[derive(Debug)]
pub struct God<EK, EV, T> {
    arena: Arena<ArcFreeTrie<EK, EV, T>>,
}

impl<EK, EV, T> God<EK, EV, T> {
    pub fn new() -> Self { Self { arena: Arena::new() } }
    pub fn alloc_node(&mut self, node: ArcFreeTrie<EK, EV, T>) -> NodeId {
        self.arena.alloc(node)
    }
    pub fn create(&mut self, value: T) -> NodeId
    where
        EK: Ord,
    {
        self.arena.alloc(ArcFreeTrie::new(value))
    }
    pub fn node(&self, id: NodeId) -> &ArcFreeTrie<EK, EV, T> { self.arena.get(id) }
    pub fn node_mut(&mut self, id: NodeId) -> &mut ArcFreeTrie<EK, EV, T> { self.arena.get_mut(id) }
    pub fn len(&self) -> usize { self.arena.len() }
    pub fn is_empty(&self) -> bool { self.arena.is_empty() }
    pub fn all_ids(&self) -> impl Iterator<Item = NodeId> + '_ { self.arena.iter_ids() }
}

#[derive(Clone)]
pub struct GodWrapper<EK, EV, T>(pub Arc<RwLock<God<EK, EV, T>>>);

impl<EK, EV, T> GodWrapper<EK, EV, T> {
    pub fn new() -> Self { Self(Arc::new(RwLock::new(God::new()))) }

    pub fn create(&self, value: T) -> NodeId
    where
        EK: Ord,
    {
        self.0.write().expect("God poisoned").create(value)
    }
    pub fn with_node<R>(&self, id: NodeId, f: impl FnOnce(&ArcFreeTrie<EK, EV, T>) -> R) -> R {
        let g = self.0.read().expect("God poisoned");
        let n = g.node(id);
        f(n)
    }
    pub fn with_node_mut<R>(&self, id: NodeId, f: impl FnOnce(&mut ArcFreeTrie<EK, EV, T>) -> R) -> R {
        let mut g = self.0.write().expect("God poisoned");
        let n = g.node_mut(id);
        f(n)
    }
    pub fn try_insert(
        &self,
        parent: NodeId,
        edge_key: EK,
        edge_value: &mut Option<EV>,
        child: NodeId,
    ) -> Result<(), CycleDetectedError>
    where
        EK: Ord + Clone,
        EV: Clone,
        T: Clone,
    {
        let candidate_depth = {
            let g = self.0.read().expect("God poisoned");
            let p = g.node(parent);
            p.max_depth.saturating_add(1)
        };
        let (mut needs_propagate, previous_child_depth) = {
            let mut g = self.0.write().expect("God poisoned");
            let ch = g.node_mut(child);
            let prev = ch.max_depth;
            let needs = candidate_depth > prev;
            if needs {
                ch.max_depth = candidate_depth;
            }
            (needs, prev)
        };

        if !self.with_node(parent, |p| p.already_has_dst_for_any_key(child)) && self.detect_cycle(parent, child) {
            // rollback child depth if changed
            if needs_propagate {
                let mut g = self.0.write().expect("God poisoned");
                let ch = g.node_mut(child);
                if ch.max_depth == candidate_depth {
                    ch.max_depth = previous_child_depth;
                }
            }
            return Err(CycleDetectedError);
        }

        if needs_propagate {
            if let Err(e) = self.propagate_max_depth(child, candidate_depth) {
                // rollback child depth
                let mut g = self.0.write().expect("God poisoned");
                let ch = g.node_mut(child);
                if ch.max_depth == candidate_depth {
                    ch.max_depth = previous_child_depth;
                }
                return Err(e);
            }
        }

        // commit structural mutation
        self.with_node_mut(parent, |p| {
            p.children
                .entry(edge_key)
                .or_default()
                .insert(child, edge_value.take().expect("edge value should be Some"));
        });

        // parent depth unchanged; child updated earlier (and propagated).
        Ok(())
    }

    pub fn try_insert_unchecked(
        &self,
        parent: NodeId,
        edge_key: EK,
        edge_value: &mut Option<EV>,
        child: NodeId,
    ) -> Result<(), CycleDetectedError>
    where
        EK: Ord + Clone,
        EV: Clone,
        T: Clone,
    {
        self.with_node_mut(parent, |p| {
            p.children
                .entry(edge_key)
                .or_default()
                .insert(child, edge_value.take().expect("edge value should be Some"));
        });
        Ok(())
    }

    pub fn detect_cycle(&self, target: NodeId, start: NodeId) -> bool
    where
        EK: Ord + Clone,
        EV: Clone,
        T: Clone,
    {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut q: VecDeque<NodeId> = VecDeque::new();
        visited.insert(start);
        q.push_back(start);
        while let Some(nid) = q.pop_front() {
            if nid == target {
                return true;
            }
            self.with_node(nid, |node| {
                for m in node.children.values() {
                    for (&child_id, _) in m.iter() {
                        if visited.insert(child_id) {
                            q.push_back(child_id);
                        }
                    }
                }
            });
        }
        false
    }

    fn propagate_max_depth(&self, root: NodeId, current_depth: usize) -> Result<(), CycleDetectedError>
    where
        EK: Ord + Clone,
        EV: Clone,
        T: Clone,
    {
        let mut rec_stack: HashSet<NodeId> = HashSet::new();
        self._propagate_max_depth(root, current_depth, &mut rec_stack)
    }

    fn _propagate_max_depth(&self, node_id: NodeId, current_depth: usize, rec_stack: &mut HashSet<NodeId>) -> Result<(), CycleDetectedError>
    where
        EK: Ord + Clone,
        EV: Clone,
        T: Clone,
    {
        if rec_stack.contains(&node_id) {
            return Err(CycleDetectedError);
        }
        rec_stack.insert(node_id);
        let children_ids: Vec<NodeId> = self.with_node(node_id, |n| {
            n.children.values().flat_map(|m| m.keys().copied()).collect()
        });
        let cand = current_depth.saturating_add(1);
        for child in children_ids {
            let mut should_prop = false;
            {
                let mut g = self.0.write().expect("God poisoned");
                let c = g.node_mut(child);
                if cand > c.max_depth {
                    c.max_depth = cand;
                    should_prop = true;
                }
            }
            if should_prop {
                self._propagate_max_depth(child, cand, rec_stack)?;
            }
        }
        rec_stack.remove(&node_id);
        Ok(())
    }

    pub fn all_nodes(&self, roots: &[NodeId]) -> Vec<NodeId>
    where
        EK: Ord,
    {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut q: VecDeque<NodeId> = VecDeque::new();
        for &r in roots {
            if visited.insert(r) {
                q.push_back(r);
            }
        }
        let mut out = Vec::new();
        while let Some(id) = q.pop_front() {
            out.push(id);
            self.with_node(id, |n| {
                for m in n.children.values() {
                    for (&child, _) in m.iter() {
                        if visited.insert(child) {
                            q.push_back(child);
                        }
                    }
                }
            });
        }
        out
    }

    pub fn has_any_cycle(&self, root: NodeId) -> bool
    where
        EK: Ord,
    {
        fn dfs<EK, EV, T>(
            god: &GodWrapper<EK, EV, T>,
            node: NodeId,
            visited: &mut HashSet<NodeId>,
            stack: &mut HashSet<NodeId>,
        ) -> bool
        where
            EK: Ord,
        {
            if stack.contains(&node) {
                return true;
            }
            if visited.contains(&node) {
                return false;
            }
            visited.insert(node);
            stack.insert(node);
            let children: Vec<NodeId> = god.with_node(node, |n| n.children.values().flat_map(|m| m.keys().copied()).collect());
            for c in children {
                if dfs(god, c, visited, stack) {
                    return true;
                }
            }
            stack.remove(&node);
            false
        }
        let mut visited = HashSet::new();
        let mut stack = HashSet::new();
        dfs(self, root, &mut visited, &mut stack)
    }

    pub fn recompute_all_max_depths(&self, roots: &[NodeId])
    where
        EK: Ord + Clone,
        EV: Clone,
        T: Clone,
    {
        let all = self.all_nodes(roots);
        if all.is_empty() {
            return;
        }
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &id in &all {
            in_degree.entry(id).or_insert(0);
            adj.entry(id).or_default();
            self.with_node(id, |n| {
                for m in n.children.values() {
                    for (&child, _) in m.iter() {
                        adj.entry(id).or_default().push(child);
                        *in_degree.entry(child).or_insert(0) += 1;
                    }
                }
            });
        }
        {
            let mut god = self.0.write().expect("God poisoned");
            for &id in &all {
                let deg = in_degree.get(&id).copied().unwrap_or(0);
                let node = god.node_mut(id);
                node.max_depth = 0; // reset
                if deg == 0 {
                    // source nodes will be queued below
                }
            }
        }
        let mut q: VecDeque<NodeId> = VecDeque::new();
        for &id in &all {
            if in_degree.get(&id).copied().unwrap_or(0) == 0 {
                q.push_back(id);
            }
        }
        while let Some(u) = q.pop_front() {
            let u_depth = self.with_node(u, |n| n.max_depth);
            if let Some(children) = adj.get(&u) {
                for &v in children {
                    {
                        let mut god = self.0.write().expect("God poisoned");
                        let vn = god.node_mut(v);
                        vn.max_depth = vn.max_depth.max(u_depth + 1);
                    }
                    let e = in_degree.get_mut(&v).unwrap();
                    *e -= 1;
                    if *e == 0 {
                        q.push_back(v);
                    }
                }
            }
        }
    }
}

/// Arena-based, Arc-free Trie node.
/// Children reference other nodes by NodeId; no Arc or Mutex inside the structure.
#[derive(Debug, Clone)]
pub struct ArcFreeTrie<EK: Ord, EV, T> {
    pub value: T,
    pub children: BTreeMap<EK, OrderedHashMap<NodeId, EV>>,
    pub max_depth: usize,
}

impl<EK: Ord, EV, T> ArcFreeTrie<EK, EV, T> {
    pub fn new(value: T) -> Self {
        Self { value, children: BTreeMap::new(), max_depth: 0 }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn already_has_dst_for_any_key(&self, dst: NodeId) -> bool {
        self.children.values().any(|m| m.contains_key(&dst))
    }

    pub fn get_edge_value(&self, edge_key: &EK, dst: NodeId) -> Option<&EV> {
        self.children.get(edge_key).and_then(|m| m.get(&dst))
    }

    pub fn get_edge_value_mut(&mut self, edge_key: &EK, dst: NodeId) -> Option<&mut EV> {
        self.children.get_mut(edge_key).and_then(|m| m.get_mut(&dst))
    }

    pub fn children(&self) -> &BTreeMap<EK, OrderedHashMap<NodeId, EV>> {
        &self.children
    }

    pub fn children_mut(&mut self) -> &mut BTreeMap<EK, OrderedHashMap<NodeId, EV>> {
        &mut self.children
    }

    pub fn recompute_max_depth<F>(&mut self, child_depth: F) -> bool
    where
        F: Fn(NodeId) -> usize,
    {
        let new_max = self.children.values()
            .flat_map(|m| m.keys().copied())
            .map(|cid| child_depth(cid) + 1)
            .max()
            .unwrap_or(0);
        if new_max != self.max_depth {
            self.max_depth = new_max;
            true
        } else {
            false
        }
    }
}

// JSON for God/GodWrapper: stateless marker; arena is not serialized here
impl<EK, EV, T> JSONConvertible for GodWrapper<EK, EV, T>
where EK: JSONConvertible, EV: JSONConvertible, T: JSONConvertible
{
    fn to_json(&self) -> JSONNode { JSONNode::Null }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Null => Ok(GodWrapper::new()),
            _ => Err("Expected JSONNode::Null for GodWrapper".into()),
        }
    }
}
impl<EK, EV, T> JSONConvertible for God<EK, EV, T>
where EK: JSONConvertible, EV: JSONConvertible, T: JSONConvertible
{
    fn to_json(&self) -> JSONNode { JSONNode::Null }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Null => Ok(God::new()),
            _ => Err("Expected JSONNode::Null for God".into()),
        }
    }
}

/// A chainable edge insertion helper that operates over the arena via GodWrapper.
pub struct EdgeInserter<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
where
    EK: Ord + Clone,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    god: GodWrapper<EK, EV, T>,
    source: NodeId,
    edge_key: EK,
    edge_value: Option<EV>,
    merge_ev: FMergeEV,
    update_node_value: FUpdateT,
    merge_ev_and_source_val: FMergeEV_T,
    result: Option<NodeId>,
}

impl<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T> EdgeInserter<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    pub fn new(
        god: &GodWrapper<EK, EV, T>,
        source: NodeId,
        edge_key: EK,
        mut edge_value: EV,
        mut merge_ev_and_source_val: FMergeEV_T,
        mut merge_ev: FMergeEV,
        update_node_value: FUpdateT,
    ) -> Self {
        // incorporate source node value into edge value
        god.with_node(source, |s| {
            merge_ev_and_source_val(&mut edge_value, &s.value);
        });
        Self {
            god: GodWrapper(god.0.clone()),
            source,
            edge_key,
            edge_value: Some(edge_value),
            merge_ev,
            update_node_value,
            merge_ev_and_source_val,
            result: None,
        }
    }

    pub fn try_destination(mut self, destination: NodeId) -> Self {
        if self.result.is_some() { return self; }
        let mut update_info: Option<(NodeId, EV)> = None;
        // Try merge with existing edge or insert a new one.
        self.god.with_node_mut(self.source, |src| {
            if let Some(ev_mut) = src.children.get_mut(&self.edge_key).and_then(|m| m.get_mut(&destination)) {
                let new_ev = self.edge_value.take().unwrap();
                (self.merge_ev)(ev_mut, new_ev);
                update_info = Some((destination, ev_mut.clone()));
                self.result = Some(destination);
            }
        });

        if self.result.is_none() {
            let mut val_clone = self.edge_value.clone().unwrap();
            let res = self.god.try_insert(self.source, self.edge_key.clone(), &mut self.edge_value, destination);
            if res.is_ok() {
                update_info = Some((destination, val_clone));
                self.result = Some(destination);
            }
        }
        if let Some((dst, ev)) = update_info {
            self.god.with_node_mut(dst, |n| (self.update_node_value)(&mut n.value, &ev));
        }
        self
    }

    pub fn try_destinations(mut self, destinations: &[NodeId]) -> Self {
        for &d in destinations {
            if self.result.is_some() { break; }
            self = self.try_destination(d);
        }
        self
    }

    pub fn try_children(mut self) -> Self {
        if self.result.is_some() { return self; }
        let kids: Vec<NodeId> = self.god.with_node(self.source, |s| {
            s.children.get(&self.edge_key)
                .map(|m| m.keys().copied().collect())
                .unwrap_or_else(Vec::new)
        });
        if !kids.is_empty() {
            self = self.try_destinations(&kids);
        }
        self
    }

    pub fn else_create_destination_with_value(mut self, value: T) -> Self {
        if self.result.is_some() { return self; }
        let new_id = self.god.create(value);
        let val_clone = self.edge_value.clone().unwrap();
        if self.god.try_insert(self.source, self.edge_key.clone(), &mut self.edge_value, new_id).is_ok() {
            self.god.with_node_mut(new_id, |n| (self.update_node_value)(&mut n.value, &val_clone));
            self.result = Some(new_id);
        }
        self
    }

    pub fn else_create_destination(self) -> Self
    where T: Default
    {
        self.else_create_destination_with_value(T::default())
    }

    pub fn else_create_destination_with(self, value_fn: impl FnOnce() -> T) -> Self {
        self.else_create_destination_with_value(value_fn())
    }

    pub fn into_option(self) -> Option<NodeId> { self.result }
    pub fn is_some(&self) -> bool { self.result.is_some() }
    pub fn clone_into_option(&self) -> Option<NodeId> { self.result }
    pub fn unwrap(self) -> NodeId { self.result.expect("EdgeInserter::unwrap() called but no destination was found or created") }
    pub fn expect(self, msg: &str) -> NodeId { self.result.expect(msg) }
}

/// Traversal utilities (special_map and special_map_grouped) over the arena-based Trie.
impl<EK: Ord + Clone, EV: Clone, T: Clone> ArcFreeTrie<EK, EV, T> {
    fn count_all_edges(god: &GodWrapper<EK, EV, T>, roots: &[NodeId]) -> usize {
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut q: VecDeque<NodeId> = VecDeque::new();
        for &r in roots {
            if visited.insert(r) {
                q.push_back(r);
            }
        }
        let mut edges = 0usize;
        while let Some(id) = q.pop_front() {
            god.with_node(id, |n| {
                for m in n.children.values() {
                    for (&child, _) in m.iter() {
                        edges += 1;
                        if visited.insert(child) {
                            q.push_back(child);
                        }
                    }
                }
            });
        }
        edges
    }

    pub fn special_map<V: Clone>(
        god: &GodWrapper<EK, EV, T>,
        initial: Vec<(NodeId, V)>,
        mut step: impl FnMut(&V, &EK, &EV, &ArcFreeTrie<EK, EV, T>) -> Option<V>,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&ArcFreeTrie<EK, EV, T>, &mut V) -> bool,
    ) {
        let mut values: HashMap<NodeId, V> = HashMap::new();
        let mut stopped: HashSet<NodeId> = HashSet::new();
        let mut todo: BTreeMap<usize, OrderedHashSet<NodeId>> = BTreeMap::new();

        let roots: Vec<_> = initial.iter().map(|(n, _)| *n).collect();
        let total_edges = Self::count_all_edges(god, &roots);
        if PROGRESS_BAR_ENABLED { println!("Progress bar enabled"); } else { println!("Progress bar disabled"); }
        let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED, leave=false);

        for (n, v) in initial {
            values.entry(n).and_modify(|old| merge(old, v.clone())).or_insert(v);
            let depth = god.with_node(n, |node| node.max_depth);
            todo.entry(depth).or_default().insert(n);
        }

        while let Some((_depth, ids)) = todo.pop_first() {
            for id in &ids {
                if stopped.contains(id) { continue; }
                let id = *id;
                let mut agg_v = match values.remove(&id) {
                    Some(v) => v,
                    None => continue,
                };
                let proceed = god.with_node(id, |node| process(node, &mut agg_v));
                if !proceed {
                    stopped.insert(id);
                    continue;
                }
                let edges: Vec<(EK, EV, NodeId)> = god.with_node(id, |node| {
                    node.children.iter().flat_map(|(ek, dst)| {
                        dst.iter().map(move |(&child, ev)| (ek.clone(), ev.clone(), child))
                    }).collect()
                });
                for (ek, ev, child) in edges {
                    let _ = pb.update(1);
                    if stopped.contains(&child) { continue; }
                    let maybe_v = god.with_node(child, |child_node| step(&agg_v, &ek, &ev, child_node));
                    if let Some(nv) = maybe_v {
                        values.entry(child).and_modify(|old| merge(old, nv.clone())).or_insert(nv);
                        let d = god.with_node(child, |n| n.max_depth);
                        todo.entry(d).or_default().insert(child);
                    }
                }
            }
        }
    }

    pub fn special_map_grouped<V, S, I>(
        god: &GodWrapper<EK, EV, T>,
        initial: Vec<(NodeId, V)>,
        mut step: S,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&ArcFreeTrie<EK, EV, T>, &mut V) -> bool,
    )
    where
        V: Clone,
        S: FnMut(
            &V, &EK, &OrderedHashMap<NodeId, EV>
        ) -> I,
        I: IntoIterator<Item = (NodeId, V)>,
    {
        let mut values: HashMap<NodeId, V> = HashMap::new();
        let mut stopped: HashSet<NodeId> = HashSet::new();
        let mut todo: BTreeMap<usize, OrderedHashSet<NodeId>> = BTreeMap::new();

        let roots: Vec<_> = initial.iter().map(|(n, _)| *n).collect();
        let total_edges = Self::count_all_edges(god, &roots);
        let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED, leave=false);

        for (n, v) in initial {
            values.entry(n).and_modify(|old| merge(old, v.clone())).or_insert(v);
            let depth = god.with_node(n, |node| node.max_depth);
            todo.entry(depth).or_default().insert(n);
        }

        while let Some((_depth, ids)) = todo.pop_first() {
            for id in &ids {
                if stopped.contains(id) { continue; }
                let id = *id;
                let mut agg_v = match values.remove(&id) {
                    Some(v) => v,
                    None => continue,
                };
                let proceed = god.with_node(id, |node| process(node, &mut agg_v));
                if !proceed {
                    stopped.insert(id);
                    continue;
                }
                let grouped: Vec<(EK, OrderedHashMap<NodeId, EV>)> = god.with_node(id, |node| {
                    node.children.iter().map(|(ek, m)| (ek.clone(), m.clone())).collect()
                });
                for (ek, dest_map) in grouped {
                    let valid_edges_count = dest_map.len();
                    if valid_edges_count > 0 {
                        let _ = pb.update(valid_edges_count);
                    }
                    for (child, nv) in step(&agg_v, &ek, &dest_map) {
                        if stopped.contains(&child) { continue; }
                        values.entry(child).and_modify(|old| merge(old, nv.clone())).or_insert(nv);
                        let d = god.with_node(child, |n| n.max_depth);
                        todo.entry(d).or_default().insert(child);
                    }
                }
            }
        }
    }
}

/// Serialize a Trie subgraph rooted at `root` into a JSONNode.
/// The format mirrors the original Arc-based implementation:
/// {
///   "nodes": [
///      {"value": ..., "max_depth": ..., "children": [[EK, [[child_idx, EV], ...]] , ...]]},
///      ...
///   ],
///   "root_idx": 0
/// }
pub fn serialize_graph<EK, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    root: NodeId,
) -> JSONNode
where
    EK: Ord + Clone + JSONConvertible + Debug,
    EV: Clone + JSONConvertible,
    T: Clone + JSONConvertible,
{
    let mut nodes_json: Vec<JSONNode> = Vec::new();
    let mut id_to_idx: HashMap<NodeId, usize> = HashMap::new();
    let mut q: VecDeque<NodeId> = VecDeque::new();

    id_to_idx.insert(root, 0);
    nodes_json.push(JSONNode::Null);
    q.push_back(root);

    while let Some(id) = q.pop_front() {
        let idx = id_to_idx[&id];
        god.with_node(id, |n| {
            let mut children_array = Vec::new();
            for (ek, dest_map) in &n.children {
                let mut arr = Vec::new();
                for (&child, ev) in dest_map {
                    let cidx = match id_to_idx.get(&child) {
                        Some(&i) => i,
                        None => {
                            let i = nodes_json.len();
                            id_to_idx.insert(child, i);
                            nodes_json.push(JSONNode::Null);
                            q.push_back(child);
                            i
                        }
                    };
                    arr.push(JSONNode::Array(vec![cidx.to_json(), ev.to_json()]));
                }
                if !arr.is_empty() {
                    children_array.push(JSONNode::Array(vec![ek.to_json(), JSONNode::Array(arr)]));
                }
            }
            nodes_json[idx] = JSONNode::Object(BTreeMap::from_iter(vec![
                ("value".to_string(), n.value.to_json()),
                ("max_depth".to_string(), n.max_depth.to_json()),
                ("children".to_string(), JSONNode::Array(children_array)),
            ]));
        });
    }

    JSONNode::Object(BTreeMap::from_iter(vec![
        ("nodes".to_string(), JSONNode::Array(nodes_json)),
        ("root_idx".to_string(), 0usize.to_json()),
    ]))
}

/// Deserialize a graph previously produced by `serialize_graph` and allocate nodes in `god`.
/// Returns the NodeId of the root in the arena.
pub fn deserialize_graph<EK, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    node: JSONNode,
) -> Result<NodeId, String>
where
    EK: Ord + Clone + JSONConvertible + Debug,
    EV: Clone + JSONConvertible,
    T: Clone + JSONConvertible,
{
    let (nodes_array, root_idx) = match node {
        JSONNode::Object(mut obj) => {
            let nodes = obj.remove("nodes").ok_or("Missing 'nodes'")?;
            let root_idx = obj.remove("root_idx").ok_or("Missing 'root_idx'")?;
            let arr = match nodes {
                JSONNode::Array(a) => a,
                _ => return Err("'nodes' must be an array".into()),
            };
            let ri = usize::from_json(root_idx)?;
            (arr, ri)
        }
        _ => return Err("Expected JSON object".into()),
    };
    if nodes_array.is_empty() { return Err("Empty nodes".into()); }
    if root_idx >= nodes_array.len() { return Err("root_idx out of bounds".into()); }

    // First pass: allocate nodes with values and max_depth.
    let mut idx_to_id: Vec<NodeId> = Vec::with_capacity(nodes_array.len());
    for (i, n) in nodes_array.iter().enumerate() {
        let (value, max_depth) = match n {
            JSONNode::Object(m) => {
                let v = m.get("value").ok_or(format!("Node {} missing value", i))?;
                let md = m.get("max_depth").ok_or(format!("Node {} missing max_depth", i))?;
                (T::from_json(v.clone())?, usize::from_json(md.clone())?)
            }
            _ => return Err(format!("Node {} not an object", i)),
        };
        let id = god.create(value);
        // Set max_depth directly
        god.with_node_mut(id, |node| node.max_depth = max_depth);
        idx_to_id.push(id);
    }

    // Second pass: link children
    for (i, n) in nodes_array.iter().enumerate() {
        let id = idx_to_id[i];
        let children_json = match n {
            JSONNode::Object(m) => m.get("children").ok_or(format!("Node {} missing children", i))?,
            _ => unreachable!(),
        };
        match children_json {
            JSONNode::Array(arr) => {
                for ek_entry in arr {
                    match ek_entry {
                        JSONNode::Array(two) if two.len() == 2 => {
                            let ek = EK::from_json(two[0].clone())?;
                            let dests = match &two[1] {
                                JSONNode::Array(a) => a,
                                _ => return Err("children map must be array".into()),
                            };
                            for d in dests {
                                match d {
                                    JSONNode::Array(pair) if pair.len() == 2 => {
                                        let child_idx = usize::from_json(pair[0].clone())?;
                                        let ev = EV::from_json(pair[1].clone())?;
                                        let child_id = idx_to_id[child_idx];
                                        let mut opt = Some(ev);
                                        god.try_insert_unchecked(id, ek.clone(), &mut opt, child_id)?;
                                    }
                                    _ => return Err("child pair invalid".into()),
                                }
                            }
                        }
                        _ => return Err("edge key entry invalid".into()),
                    }
                }
            }
            _ => return Err("children must be array".into()),
        }
    }

    Ok(idx_to_id[root_idx])
}
