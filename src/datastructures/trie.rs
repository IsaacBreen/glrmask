// #![deny(clippy::iter_over_hash_type)]
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{self, Debug};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use ordered_hash_map::OrderedHashMap;


use crate::json_serialization::{JSONConvertible, JSONNode};
use ordered_hash_map::OrderedHashSet;
use kdam::{tqdm, BarExt};
use profiler_macro::{time_it};
use crate::profiler::PROGRESS_BAR_ENABLED;


/// Error type indicating that a cycle was detected during an operation
/// that updates graph structure or properties like max_depth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleDetectedError;

impl fmt::Display for CycleDetectedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cycle detected in Trie structure")
    }
}

impl Error for CycleDetectedError {}

pub type NodeId = usize;

#[derive(Debug, Clone)]
pub struct TrieNode<EK: Ord, EV, T> {
    pub value: T,
    pub children: BTreeMap<EK, OrderedHashMap<NodeRef<EK, EV, T>, EV>>,
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
        self.children.is_empty()
    }

    pub fn recompute_max_depth(&mut self, god: &GodWrapper<EK, EV, T>, self_ref: NodeRef<EK, EV, T>) -> bool {
        let god_guard = god.0.read().unwrap();
        let new_max_depth = self.children.values()
            .flat_map(|dest_map| dest_map.keys())
            .map(|&child_ref| god_guard.get(child_ref).max_depth + 1)
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

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct NodeRef<EK, EV, T> {
    id: NodeId,
    _marker: PhantomData<(EK, EV, T)>,
}

impl<EK, EV, T> NodeRef<EK, EV, T> {
    pub fn new(id: NodeId) -> Self {
        Self { id, _marker: PhantomData }
    }
    pub fn id(&self) -> NodeId {
        self.id
    }
}

impl<EK, EV, T> JSONConvertible for NodeRef<EK, EV, T> {
    fn to_json(&self) -> JSONNode {
        self.id.to_json()
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        usize::from_json(node).map(Self::new)
    }
}

pub struct God<EK: Ord, EV, T> {
    nodes: Vec<RwLock<TrieNode<EK, EV, T>>>,
}

impl<EK: Ord, EV, T> God<EK, EV, T> {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn alloc_node(&mut self, value: T) -> NodeRef<EK, EV, T> {
        let id = self.nodes.len();
        self.nodes.push(RwLock::new(TrieNode::new(value)));
        NodeRef::new(id)
    }

    pub fn get(&self, n: NodeRef<EK, EV, T>) -> RwLockReadGuard<'_, TrieNode<EK, EV, T>> {
        self.nodes[n.id()].read().expect("RwLock poisoned")
    }

    pub fn try_get(&self, n: NodeRef<EK, EV, T>) -> Option<RwLockReadGuard<'_, TrieNode<EK, EV, T>>> {
        self.nodes.get(n.id()).and_then(|lock| lock.try_read().ok())
    }

    pub fn get_mut(&self, n: NodeRef<EK, EV, T>) -> RwLockWriteGuard<'_, TrieNode<EK, EV, T>> {
        self.nodes[n.id()].write().expect("RwLock poisoned")
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl<EK: Ord, EV, T> Default for God<EK, EV, T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct GodWrapper<EK: Ord, EV, T>(pub Arc<RwLock<God<EK, EV, T>>>);

impl<EK: Ord, EV, T> GodWrapper<EK, EV, T> {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(God::new())))
    }
}

impl<EK: Ord, EV, T> Default for GodWrapper<EK, EV, T> {
    fn default() -> Self {
        Self::new()
    }
}

#[time_it]
pub fn detect_cycle<EK: Ord, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    target: NodeRef<EK, EV, T>,
    start: NodeRef<EK, EV, T>,
) -> bool {
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut queue: VecDeque<NodeRef<EK, EV, T>> = VecDeque::new();

    if visited.insert(start.id()) {
        queue.push_back(start);
    }

    let god_guard = god.0.read().unwrap();

    while let Some(current_node) = queue.pop_front() {
        if current_node == target {
            return true;
        }

        match god_guard.try_get(current_node) {
            Some(node_guard) => {
                for child_map in node_guard.children.values() {
                    for &child_ref in child_map.keys() {
                        if visited.insert(child_ref.id()) {
                            queue.push_back(child_ref);
                        }
                    }
                }
            }
            None => {
                // Failed to lock, assume it's held by the calling thread (the target).
                return true;
            }
        }
    }
    false
}

fn _propagate_max_depth<EK: Ord, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    node: NodeRef<EK, EV, T>,
    current_depth: usize,
    rec_stack: &mut HashSet<NodeId>,
) -> Result<(), CycleDetectedError> {
    if !rec_stack.insert(node.id()) {
        return Err(CycleDetectedError);
    }

    let god_guard = god.0.read().unwrap();
    let children_refs: Vec<NodeRef<EK, EV, T>> = {
        let node_guard = god_guard.get(node);
        node_guard.children.values().flat_map(|map| map.keys().copied()).collect()
    };

    let candidate_depth = current_depth.saturating_add(1);
    for child in children_refs {
        let should_propagate;
        {
            let mut child_guard = god_guard.get_mut(child);
            if candidate_depth > child_guard.max_depth {
                child_guard.max_depth = candidate_depth;
                should_propagate = true;
            } else {
                should_propagate = false;
            }
        }

        if should_propagate {
            _propagate_max_depth(god, child, candidate_depth, rec_stack)?;
        }
    }

    rec_stack.remove(&node.id());
    Ok(())
}

pub fn propagate_max_depth<EK: Ord, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    node: NodeRef<EK, EV, T>,
    current_depth: usize,
) -> Result<(), CycleDetectedError> {
    let mut rec_stack = HashSet::new();
    _propagate_max_depth(god, node, current_depth, &mut rec_stack)
}

pub fn already_has_dst<EK: Ord + Clone, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    source: NodeRef<EK, EV, T>,
    edge_key: &EK,
    dest: NodeRef<EK, EV, T>,
) -> bool {
    let god_guard = god.0.read().unwrap();
    let source_guard = god_guard.get(source);
    source_guard.children.get(edge_key).map_or(false, |dest_map| dest_map.contains_key(&dest))
}

pub fn already_has_dst_for_any_key<EK: Ord, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    source: NodeRef<EK, EV, T>,
    dest: NodeRef<EK, EV, T>,
) -> bool {
    let god_guard = god.0.read().unwrap();
    let source_guard = god_guard.get(source);
    source_guard.children.values().any(|dest_map| dest_map.contains_key(&dest))
}

#[time_it]
pub fn try_insert_edge<EK: Ord + Clone, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    source: NodeRef<EK, EV, T>,
    edge_key: EK,
    edge_value: &mut Option<EV>,
    dest: NodeRef<EK, EV, T>,
) -> Result<(), CycleDetectedError> {
    if !already_has_dst_for_any_key(god, source, dest) && detect_cycle(god, source, dest) {
        return Err(CycleDetectedError);
    }

    let source_depth = god.0.read().unwrap().get(source).max_depth;
    let candidate_depth = source_depth.saturating_add(1);
    let previous_child_depth;
    let needs_depth_update;

    {
        let god_guard = god.0.read().unwrap();
        let mut child_guard = god_guard.get_mut(dest);
        previous_child_depth = child_guard.max_depth;
        needs_depth_update = candidate_depth > previous_child_depth;
        if needs_depth_update {
            child_guard.max_depth = candidate_depth;
        }
    }

    if needs_depth_update {
        if let Err(e) = propagate_max_depth(god, dest, candidate_depth) {
            let god_guard = god.0.read().unwrap();
            let mut child_guard = god_guard.get_mut(dest);
            if child_guard.max_depth == candidate_depth {
                child_guard.max_depth = previous_child_depth;
            }
            return Err(e);
        }
    }

    let god_guard = god.0.write().unwrap();
    let mut source_guard = god_guard.get_mut(source);
    source_guard.children
        .entry(edge_key)
        .or_default()
        .insert(dest, edge_value.take().unwrap());

    Ok(())
}

pub fn force_insert_to_node<EK: Ord + Clone, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    source: NodeRef<EK, EV, T>,
    edge_key: EK,
    edge_value: EV,
    dest: NodeRef<EK, EV, T>,
) {
    let god_guard = god.0.write().unwrap();
    let mut source_guard = god_guard.get_mut(source);
    source_guard.children.entry(edge_key).or_default().insert(dest, edge_value);
}

pub fn all_nodes<EK: Ord, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[NodeRef<EK, EV, T>],
) -> Vec<NodeRef<EK, EV, T>> {
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut result = Vec::new();
    let mut queue = VecDeque::new();

    for &root in roots {
        if visited.insert(root.id()) {
            queue.push_back(root);
        }
    }

    let god_guard = god.0.read().unwrap();
    while let Some(node_ref) = queue.pop_front() {
        result.push(node_ref);
        let node_guard = god_guard.get(node_ref);
        for children_map in node_guard.children.values() {
            for &child_ref in children_map.keys() {
                if visited.insert(child_ref.id()) {
                    queue.push_back(child_ref);
                }
            }
        }
    }
    result
}

fn _has_any_cycle_recursive<EK: Ord, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    node: NodeRef<EK, EV, T>,
    global_visited: &mut HashSet<NodeId>,
    recursion_stack: &mut HashSet<NodeId>,
) -> bool {
    if !recursion_stack.insert(node.id()) {
        return true;
    }
    if !global_visited.insert(node.id()) {
        recursion_stack.remove(&node.id());
        return false;
    }

    let god_guard = god.0.read().unwrap();
    let children_refs: Vec<NodeRef<EK, EV, T>> = {
        let node_guard = god_guard.get(node);
        node_guard.children.values().flat_map(|map| map.keys().copied()).collect()
    };

    for child in children_refs {
        if _has_any_cycle_recursive(god, child, global_visited, recursion_stack) {
            return true;
        }
    }

    recursion_stack.remove(&node.id());
    false
}

pub fn has_any_cycle<EK: Ord, EV, T>(god: &GodWrapper<EK, EV, T>, root: NodeRef<EK, EV, T>) -> bool {
    let mut global_visited = HashSet::new();
    let mut recursion_stack = HashSet::new();
    _has_any_cycle_recursive(god, root, &mut global_visited, &mut recursion_stack)
}

pub fn recompute_all_max_depths<EK: Ord, EV, T>(god: &GodWrapper<EK, EV, T>, roots: &[NodeRef<EK, EV, T>]) {
    let all_node_refs = all_nodes(god, roots);
    if all_node_refs.is_empty() {
        return;
    }

    let god_guard = god.0.read().unwrap();
    let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
    let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

    for &node_ref in &all_node_refs {
        let node_id = node_ref.id();
        in_degree.entry(node_id).or_insert(0);
        adj.entry(node_id).or_default();

        let node_guard = god_guard.get(node_ref);
        for child_ref in node_guard.children.values().flat_map(|m| m.keys()) {
            let child_id = child_ref.id();
            adj.entry(node_id).or_default().push(child_id);
            *in_degree.entry(child_id).or_default() += 1;
        }
    }

    let mut queue = VecDeque::new();
    for &node_ref in &all_node_refs {
        let node_id = node_ref.id();
        if in_degree.get(&node_id).cloned().unwrap_or(0) == 0 {
            queue.push_back(node_ref);
            god_guard.get_mut(node_ref).max_depth = 0;
        } else {
            god_guard.get_mut(node_ref).max_depth = 0;
        }
    }

    while let Some(u_ref) = queue.pop_front() {
        let u_depth = god_guard.get(u_ref).max_depth;
        if let Some(children_ids) = adj.get(&u_ref.id()) {
            for &v_id in children_ids {
                let v_ref = NodeRef::new(v_id);
                {
                    let mut v_guard = god_guard.get_mut(v_ref);
                    v_guard.max_depth = v_guard.max_depth.max(u_depth + 1);
                }
                let v_in_degree = in_degree.get_mut(&v_id).unwrap();
                *v_in_degree -= 1;
                if *v_in_degree == 0 {
                    queue.push_back(v_ref);
                }
            }
        }
    }
}

pub struct EdgeInserter<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    god: GodWrapper<EK, EV, T>,
    source: NodeRef<EK, EV, T>,
    edge_key: EK,
    edge_value: Option<EV>,
    merge_edge_value: FMergeEV,
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
    result: Option<NodeRef<EK, EV, T>>,
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
        source: NodeRef<EK, EV, T>,
        edge_key: EK,
        edge_value: EV,
        merge_edge_value: FMergeEV,
        update_node_value: FUpdateT,
        merge_edge_value_and_source_node_value: FMergeEV_T,
    ) -> Self {
        let mut edge_value = edge_value;
        let mut merge_fn = merge_edge_value_and_source_node_value;
        {
            let god_guard = god.0.read().unwrap();
            let source_guard = god_guard.get(source);
            merge_fn(&mut edge_value, &source_guard.value);
        }

        EdgeInserter {
            god: god.clone(),
            source,
            edge_key,
            edge_value: Some(edge_value),
            merge_edge_value,
            update_node_value,
            merge_edge_value_and_source_node_value: merge_fn,
            result: None,
        }
    }

    #[time_it]
    pub fn try_destination(mut self, destination: NodeRef<EK, EV, T>) -> Self {
        if self.result.is_some() {
            return self;
        }

        let mut update_info: Option<(NodeRef<EK, EV, T>, EV)> = None;
        let mut edge_exists_and_merged = false;

        {
            let god_guard = self.god.0.write().unwrap();
            let mut source_guard = god_guard.get_mut(self.source);
            if let Some(existing_ev_mut) = source_guard.children.get_mut(&self.edge_key).and_then(|dest_map| dest_map.get_mut(&destination)) {
                let new_ev = self.edge_value.take().unwrap();
                (self.merge_edge_value)(existing_ev_mut, new_ev);
                let updated_ev = existing_ev_mut.clone();
                self.result = Some(destination);
                update_info = Some((destination, updated_ev));
                edge_exists_and_merged = true;
            }
        }

        if !edge_exists_and_merged {
            let edge_val_clone = self.edge_value.as_ref().unwrap().clone();
            if try_insert_edge(&self.god, self.source, self.edge_key.clone(), &mut self.edge_value, destination).is_ok() {
                self.result = Some(destination);
                update_info = Some((destination, edge_val_clone));
            }
        }

        if let Some((dest_ref, ev)) = update_info {
            let god_guard = self.god.0.read().unwrap();
            let mut dest_guard = god_guard.get_mut(dest_ref);
            (self.update_node_value)(&mut dest_guard.value, &ev);
        }

        self
    }

    #[time_it]
    pub fn try_destinations_iter(mut self, destinations: impl Iterator<Item = NodeRef<EK, EV, T>>) -> Self {
        for destination in destinations {
            if self.result.is_some() {
                break;
            }
            self = self.try_destination(destination);
        }
        self
    }

    pub fn try_children(mut self) -> Self {
        if self.result.is_some() {
            return self;
        }

        let children_for_this_key: Vec<NodeRef<EK, EV, T>> = {
            let god_guard = self.god.0.read().unwrap();
            let source_guard = god_guard.get(self.source);
            if let Some(dest_map) = source_guard.children.get(&self.edge_key) {
                dest_map.keys().copied().collect()
            } else {
                Vec::new()
            }
        };

        if !children_for_this_key.is_empty() {
            self = self.try_destinations_iter(children_for_this_key.into_iter());
        }
        self
    }

    pub fn else_create_destination_with_value(mut self, value: T) -> Self {
        if self.result.is_some() {
            return self;
        }

        let new_node_ref = self.god.0.write().unwrap().alloc_node(value);
        let edge_val_clone = self.edge_value.as_ref().unwrap().clone();

        if try_insert_edge(&self.god, self.source, self.edge_key.clone(), &mut self.edge_value, new_node_ref).is_ok() {
            self.result = Some(new_node_ref);
            let god_guard = self.god.0.read().unwrap();
            let mut new_node_guard = god_guard.get_mut(new_node_ref);
            (self.update_node_value)(&mut new_node_guard.value, &edge_val_clone);
        }

        self
    }

    pub fn else_create_destination_with(self, value_fn: impl FnOnce() -> T) -> Self {
        if self.result.is_some() {
            return self;
        }
        self.else_create_destination_with_value(value_fn())
    }

    pub fn else_create_destination(self) -> Self where T: Default {
        if self.result.is_some() {
            return self;
        }
        self.else_create_destination_with_value(T::default())
    }

    pub fn into_option(self) -> Option<NodeRef<EK, EV, T>> {
        self.result
    }

    pub fn unwrap(self) -> NodeRef<EK, EV, T> {
        self.result.expect("EdgeInserter::unwrap() called but no destination was found or created")
    }
}

#[time_it]
pub fn special_map<V: Clone, EK, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    initial: Vec<(NodeRef<EK, EV, T>, V)>,
    mut step: impl FnMut(&V, &EK, &EV, &TrieNode<EK, EV, T>) -> Option<V>,
    mut merge: impl FnMut(&mut V, V),
    mut process: impl FnMut(&TrieNode<EK, EV, T>, &mut V) -> bool,
) where EK: Ord + Clone, EV: Clone, T: Clone {
    let mut values: HashMap<NodeRef<EK, EV, T>, V> = HashMap::new();
    let mut stopped_nodes: HashSet<NodeRef<EK, EV, T>> = HashSet::new();
    let mut todo: BTreeMap<usize, OrderedHashSet<NodeRef<EK, EV, T>>> = BTreeMap::new();

    let initial_nodes: Vec<_> = initial.iter().map(|(n, _)| *n).collect();
    let god_guard = god.0.read().unwrap();
    let total_edges = all_nodes(god, &initial_nodes).into_iter().map(|n| {
        god_guard.get(n).children.values().map(|m| m.len()).sum::<usize>()
    }).sum();
    drop(god_guard);

    let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED, leave=false);

    for (node_ref, v0) in initial {
        values.entry(node_ref).and_modify(|old| merge(old, v0.clone())).or_insert(v0);
        let depth = god.0.read().unwrap().get(node_ref).max_depth;
        todo.entry(depth).or_default().insert(node_ref);
    }

    while let Some((_depth, node_refs)) = todo.pop_first() {
        for &node_ref in &node_refs {
            if stopped_nodes.contains(&node_ref) { continue; }

            let mut agg_v = match values.remove(&node_ref) {
                Some(v) => v,
                None => continue,
            };

            let god_guard = god.0.read().unwrap();
            let proceed = {
                let guard = god_guard.get(node_ref);
                process(&guard, &mut agg_v)
            };

            if !proceed {
                stopped_nodes.insert(node_ref);
                continue;
            }

            let edges: Vec<(EK, EV, NodeRef<EK, EV, T>)> = {
                let guard = god_guard.get(node_ref);
                guard.children.iter().flat_map(|(ek, dst_map)| {
                    dst_map.iter().map(move |(node_ref, ev)| (ek.clone(), ev.clone(), *node_ref))
                }).collect()
            };

            for (ek, ev, child_ref) in edges {
                let _ = pb.update(1);
                if stopped_nodes.contains(&child_ref) { continue; }

                let maybe_v = {
                    let child_guard = god_guard.get(child_ref);
                    step(&agg_v, &ek, &ev, &child_guard)
                };
                if let Some(new_v) = maybe_v {
                    values.entry(child_ref).and_modify(|old| merge(old, new_v.clone())).or_insert(new_v);
                    let child_depth = god_guard.get(child_ref).max_depth;
                    todo.entry(child_depth).or_default().insert(child_ref);
                }
            }
        }
    }
}

#[time_it]
pub fn special_map_grouped<V, S, I, EK, EV, T>(
    god: &GodWrapper<EK, EV, T>,
    initial: Vec<(NodeRef<EK, EV, T>, V)>,
    mut step: S,
    mut merge: impl FnMut(&mut V, V),
    mut process: impl FnMut(&TrieNode<EK, EV, T>, &mut V) -> bool,
)
where
    V: Clone,
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    S: FnMut(&V, &EK, &OrderedHashMap<NodeRef<EK, EV, T>, EV>) -> I,
    I: IntoIterator<Item = (NodeRef<EK, EV, T>, V)>,
{
    let mut values: HashMap<NodeRef<EK, EV, T>, V> = HashMap::new();
    let mut stopped_nodes: HashSet<NodeRef<EK, EV, T>> = HashSet::new();
    let mut todo: BTreeMap<usize, OrderedHashSet<NodeRef<EK, EV, T>>> = BTreeMap::new();

    let initial_nodes: Vec<_> = initial.iter().map(|(n, _)| *n).collect();
    let god_guard = god.0.read().unwrap();
    let total_edges = all_nodes(god, &initial_nodes).into_iter().map(|n| {
        god_guard.get(n).children.values().map(|m| m.len()).sum::<usize>()
    }).sum();
    drop(god_guard);

    let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED, leave=false);

    for (node_ref, v0) in initial {
        values.entry(node_ref).and_modify(|old| merge(old, v0.clone())).or_insert(v0);
        let depth = god.0.read().unwrap().get(node_ref).max_depth;
        todo.entry(depth).or_default().insert(node_ref);
    }

    while let Some((_depth, node_refs)) = todo.pop_first() {
        for &node_ref in &node_refs {
            if stopped_nodes.contains(&node_ref) { continue; }

            let mut agg_v = match values.remove(&node_ref) {
                Some(v) => v,
                None => continue,
            };

            let god_guard = god.0.read().unwrap();
            let proceed = {
                let guard = god_guard.get(node_ref);
                process(&guard, &mut agg_v)
            };

            if !proceed {
                stopped_nodes.insert(node_ref);
                continue;
            }

            let children_by_ek: Vec<(EK, OrderedHashMap<NodeRef<EK, EV, T>, EV>)> = {
                let guard = god_guard.get(node_ref);
                guard.children.iter().map(|(ek, dst_map)| (ek.clone(), dst_map.clone())).collect()
            };

            for (ek, dest_map) in children_by_ek {
                let _ = pb.update(dest_map.len());
                let new_values_for_children = step(&agg_v, &ek, &dest_map);

                for (child_ref, new_v) in new_values_for_children {
                    if stopped_nodes.contains(&child_ref) { continue; }
                    values.entry(child_ref).and_modify(|old| merge(old, new_v.clone())).or_insert(new_v);
                    let child_depth = god_guard.get(child_ref).max_depth;
                    todo.entry(child_depth).or_default().insert(child_ref);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(false)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use crate::datastructures::hybrid_bitset::HybridBitset;

    type TestTrieNodeBasic = TrieNode<&'static str, &'static str, i32>;
    type TestNodeRefBasic = NodeRef<&'static str, &'static str, i32>;

    type TestTrieNodeEI = TrieNode<&'static str, HybridBitset, String>;
    type TestNodeRefEI = NodeRef<&'static str, HybridBitset, String>;

    #[test]
    fn test_try_insertion_and_retrieval() {
        let god = GodWrapper::new();
        let root_ref = god.0.write().unwrap().alloc_node(0);
        let child1_ref = god.0.write().unwrap().alloc_node(1);
        let child2_ref = god.0.write().unwrap().alloc_node(2);
        let child3_ref = god.0.write().unwrap().alloc_node(3);

        try_insert_edge(&god, root_ref, "a", &mut Some("edge_a1"), child1_ref).expect("Insert failed");
        try_insert_edge(&god, root_ref, "b", &mut Some("edge_b"), child2_ref).expect("Insert failed");
        try_insert_edge(&god, root_ref, "a", &mut Some("edge_a3"), child3_ref).expect("Insert failed");

        let god_guard = god.0.read().unwrap();
        let root_guard = god_guard.get(root_ref);

        let retrieved_children_a = root_guard.children.get("a").expect("Failed to get children for 'a'");
        assert_eq!(retrieved_children_a.len(), 2);
        let retrieved_data_a: HashSet<(&str, NodeRef<_, _, _>)> = retrieved_children_a
            .iter()
            .map(|(node_ref, ev_ref)| (*ev_ref, *node_ref))
            .collect();
        assert!(retrieved_data_a.contains(&("edge_a1", child1_ref)));
        assert!(retrieved_data_a.contains(&("edge_a3", child3_ref)));

        let retrieved_children_b = root_guard.children.get("b").expect("Failed to get child 'b'");
        assert_eq!(retrieved_children_b.len(), 1);
        let (node_ref, ev_ref) = retrieved_children_b.iter().next().unwrap();
        assert_eq!(*ev_ref, "edge_b");
        assert_eq!(*node_ref, child2_ref);

        assert!(root_guard.children.get("c").is_none());

        let children_keys: Vec<_> = root_guard.children.keys().cloned().collect();
        assert_eq!(children_keys, vec!["a", "b"]);

        assert!(!root_guard.is_leaf());
        assert!(god_guard.get(child1_ref).is_leaf());
        assert!(god_guard.get(child2_ref).is_leaf());
        assert!(god_guard.get(child3_ref).is_leaf());
    }

    #[test]
    fn test_cycle_detection_on_try_insert() {
        let god = GodWrapper::new();
        let root_ref = god.0.write().unwrap().alloc_node(0);
        let child_ref = god.0.write().unwrap().alloc_node(1);

        let insert1_result = try_insert_edge(&god, root_ref, "r->c", &mut Some("e1"), child_ref);
        assert!(insert1_result.is_ok());

        let god_guard = god.0.read().unwrap();
        assert_eq!(god_guard.get(child_ref).max_depth, 1);
        assert_eq!(god_guard.get(root_ref).max_depth, 0);
        drop(god_guard);

        let insert2_result = try_insert_edge(&god, child_ref, "c->r", &mut Some("e2"), root_ref);
        assert_eq!(insert2_result, Err(CycleDetectedError));

        let god_guard = god.0.read().unwrap();
        let child_guard = god_guard.get(child_ref);
        assert!(!child_guard.children.contains_key("c->r"));
        assert_eq!(god_guard.get(root_ref).max_depth, 0);
        assert_eq!(child_guard.max_depth, 1);
    }

    #[test]
    fn test_ei_try_destination_success_new_edge() {
        let god = GodWrapper::new();
        let source_ref = god.0.write().unwrap().alloc_node("source".to_string());
        let dest_ref = god.0.write().unwrap().alloc_node("dest".to_string());
        let edge_val: HybridBitset = vec![1].into_iter().collect();

        let inserter = EdgeInserter::new(&god, source_ref, "key", edge_val.clone(), |e, n| *e |= n, |_, _| {}, |_, _| {});
        let result_node = inserter.try_destination(dest_ref).unwrap();

        assert_eq!(result_node, dest_ref);
        let god_guard = god.0.read().unwrap();
        let s_guard = god_guard.get(source_ref);
        let children_map = s_guard.children.get("key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (node_ref, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, edge_val);
        assert_eq!(*node_ref, dest_ref);
        assert_eq!(god_guard.get(dest_ref).max_depth, 1);
    }

    #[test]
    fn test_ei_try_destination_success_merge_ev() {
        let god = GodWrapper::new();
        let source_ref = god.0.write().unwrap().alloc_node("source".to_string());
        let dest_ref = god.0.write().unwrap().alloc_node("dest".to_string());
        let initial_edge_val: HybridBitset = vec![10].into_iter().collect();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();
        let merged_edge_val: HybridBitset = vec![1, 10].into_iter().collect();

        try_insert_edge(&god, source_ref, "key", &mut Some(initial_edge_val), dest_ref).unwrap();
        assert_eq!(god.0.read().unwrap().get(dest_ref).max_depth, 1);

        let inserter = EdgeInserter::new(&god, source_ref, "key", new_edge_val.clone(), |e, n| *e |= n, |_, _| {}, |_, _| {});
        let result_node = inserter.try_destination(dest_ref).unwrap();

        assert_eq!(result_node, dest_ref);
        let god_guard = god.0.read().unwrap();
        let s_guard = god_guard.get(source_ref);
        let children_map = s_guard.children.get("key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (node_ref, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, merged_edge_val);
        assert_eq!(*node_ref, dest_ref);
        assert_eq!(god_guard.get(dest_ref).max_depth, 1);
    }

    #[test]
    fn test_ei_else_create_with_value() {
        let god = GodWrapper::new();
        let source_ref = god.0.write().unwrap().alloc_node("source".to_string());
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        let inserter = EdgeInserter::new(&god, source_ref, "key", new_edge_val.clone(), |e, n| *e |= n, |_, _| {}, |_, _| {});
        let result_node = inserter.else_create_destination_with_value("created".to_string()).unwrap();

        let god_guard = god.0.read().unwrap();
        assert_eq!(god_guard.get(result_node).value, "created");
        assert_eq!(god_guard.get(result_node).max_depth, 1);
        let s_guard = god_guard.get(source_ref);
        let children_map = s_guard.children.get("key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (node_ref, ev) = children_map.iter().next().unwrap();
        assert_eq!(*node_ref, result_node);
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_special_map_diamond_merge_max() {
        let god = GodWrapper::new();
        let root_ref = god.0.write().unwrap().alloc_node(0);
        let child1_ref = god.0.write().unwrap().alloc_node(1);
        let child2_ref = god.0.write().unwrap().alloc_node(2);
        let grandchild_ref = god.0.write().unwrap().alloc_node(3);

        try_insert_edge(&god, root_ref, "r->c1", &mut Some("edge1"), child1_ref).unwrap();
        try_insert_edge(&god, root_ref, "r->c2", &mut Some("edge2"), child2_ref).unwrap();
        try_insert_edge(&god, child1_ref, "c1->gc", &mut Some("edge3"), grandchild_ref).unwrap();
        try_insert_edge(&god, child2_ref, "c2->gc", &mut Some("edge4"), grandchild_ref).unwrap();

        let god_guard = god.0.read().unwrap();
        assert_eq!(god_guard.get(root_ref).max_depth, 0);
        assert_eq!(god_guard.get(child1_ref).max_depth, 1);
        assert_eq!(god_guard.get(child2_ref).max_depth, 1);
        assert_eq!(god_guard.get(grandchild_ref).max_depth, 2);
        drop(god_guard);

        let processed_nodes = Arc::new(RwLock::new(HashMap::<i32, i32>::new()));
        let process_count = Arc::new(AtomicUsize::new(0));

        special_map(
            &god,
            vec![(root_ref, 100)],
            |p_val, _ek, _ev, _child_node| Some(p_val + 1),
            |current_v, new_v| *current_v = (*current_v).max(new_v),
            {
                let processed_nodes = processed_nodes.clone();
                let process_count = process_count.clone();
                move |node, final_v| {
                    processed_nodes.write().unwrap().insert(node.value, *final_v);
                    process_count.fetch_add(1, Ordering::SeqCst);
                    true
                }
            }
        );

        let final_results = processed_nodes.read().unwrap();
        assert_eq!(process_count.load(Ordering::SeqCst), 4);
        assert_eq!(final_results.get(&0), Some(&100));
        assert_eq!(final_results.get(&1), Some(&101));
        assert_eq!(final_results.get(&2), Some(&101));
        assert_eq!(final_results.get(&3), Some(&102));
    }
}
