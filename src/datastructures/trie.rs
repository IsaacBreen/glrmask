// #![deny(clippy::iter_over_hash_type)]
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{self, Debug};
use std::sync::{Arc, RwLock, TryLockError};
use std::sync::atomic::{AtomicUsize, Ordering}; // Added for tests
use std::cmp::Reverse;          // min-heap helper
use std::collections::BinaryHeap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::cell::RefCell;
use ordered_hash_map::OrderedHashMap;

use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::ArcPtrWrapper;
use crate::json_serialization::{JSONConvertible, JSONNode};
use deterministic_hash::DeterministicHasher;
use ordered_hash_map::OrderedHashSet;
use kdam::{tqdm, BarExt};
use profiler_macro::{time_it, timeit};
use crate::datastructures::arc_wrapper::WeakPtrWrapper;
use crate::profiler::PROGRESS_BAR_ENABLED;

/// Error type indicating that a cycle was detected during an operation
/// that updates graph structure or properties like max_depth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleDetectedError;

/// Result type indicating whether an inserted edge became Strong or Weak.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertedEdgeKind { Strong, Weak }

impl fmt::Display for CycleDetectedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cycle detected in Trie structure")
    }
}

impl Error for CycleDetectedError {}

/// Represents a node in a Trie–like structure (allowing shared subtrees and DAGs).
/// Multiple children can exist for the same edge key. Each edge instance has a value.
///
/// EK: type of the edge key (must be Ord).
/// EV: type of the edge value.
/// T: type of the value stored within the node.
#[derive(Debug, Clone)]
pub struct Trie<EK: Ord, EV, T> {
    pub value: T,
    /// Stores a map from EdgeKey to (a map from ChildArc (wrapped) to EdgeValue).
    children: BTreeMap<EK, OrderedHashMap<ArcPtrWrapper<RwLock<Trie<EK, EV, T>>>, EV>>,
    /// Weak edges: allow explicit cycles without keeping the target alive.
    /// Edges here do NOT affect max_depth and are not considered for cycle detection.
    weak_children: BTreeMap<EK, OrderedHashMap<WeakPtrWrapper<RwLock<Trie<EK, EV, T>>>, EV>>,
    /// The “longest distance” from some source node (as computed during insertion).
    /// This value is set (or updated) when an edge is inserted.
    /// If A -> B, then A.max_depth < B.max_depth.
    pub max_depth: usize,
}

impl<EK, EV, T> JSONConvertible for Trie<EK, EV, T>
where
    EK: Ord + Clone + JSONConvertible + Debug,
    EV: Clone + JSONConvertible,
    T: Clone + JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        let mut nodes_json_list: Vec<JSONNode> = Vec::new();
        // Maps the raw pointer of an Arc<RwLock<Trie>> to its index in nodes_json_list
        let mut arc_ptr_to_idx_map: HashMap<*const RwLock<Trie<EK, EV, T>>, usize> = HashMap::new();
        // Queue for BFS traversal, storing Arcs to keep them alive and allow locking
        let mut bfs_q: VecDeque<Arc<RwLock<Trie<EK, EV, T>>>> = VecDeque::new();

        // Root (self) at index 0
        let root_idx = 0;
        nodes_json_list.push(JSONNode::Null);

        let mut root_children_json_data = Vec::new();
        let mut root_weak_children_json_data = Vec::new();

        for (edge_key, destinations_map) in &self.children {
            let ek_json = edge_key.to_json();
            let mut dest_map_json_array = Vec::new();
            for (child_arc_ptr_wrapper, edge_val) in destinations_map {
                let child_arc = child_arc_ptr_wrapper.as_arc();
                let child_arc_ptr = Arc::as_ptr(child_arc);

                let child_idx = match arc_ptr_to_idx_map.get(&child_arc_ptr) {
                    Some(idx) => *idx,
                    None => {
                        let new_idx = nodes_json_list.len();
                        arc_ptr_to_idx_map.insert(child_arc_ptr, new_idx);
                        bfs_q.push_back(child_arc.clone());
                        nodes_json_list.push(JSONNode::Null);
                        new_idx
                    }
                };
                dest_map_json_array.push(JSONNode::Array(vec![
                    child_idx.to_json(),
                    edge_val.to_json(),
                ]));
            }
            root_children_json_data.push(JSONNode::Array(vec![ek_json, JSONNode::Array(dest_map_json_array)]));
        }

        // weak children (upgrade if possible)
        for (edge_key, destinations_map) in &self.weak_children {
            let ek_json = edge_key.to_json();
            let mut dest_map_json_array = Vec::new();
            for (weak_wrapper, edge_val) in destinations_map {
                if let Some(child_arc) = weak_wrapper.upgrade() {
                    let child_arc_ptr = Arc::as_ptr(&child_arc);
                    let child_idx = match arc_ptr_to_idx_map.get(&child_arc_ptr) {
                        Some(idx) => *idx,
                        None => {
                            let new_idx = nodes_json_list.len();
                            arc_ptr_to_idx_map.insert(child_arc_ptr, new_idx);
                            bfs_q.push_back(child_arc.clone());
                            nodes_json_list.push(JSONNode::Null);
                            new_idx
                        }
                    };
                    dest_map_json_array.push(JSONNode::Array(vec![
                        child_idx.to_json(),
                        edge_val.to_json(),
                    ]));
                }
            }
            root_weak_children_json_data.push(JSONNode::Array(vec![ek_json, JSONNode::Array(dest_map_json_array)]));
        }

        nodes_json_list[root_idx] = JSONNode::Object(BTreeMap::from_iter(vec![
            ("value".to_string(), self.value.to_json()),
            ("max_depth".to_string(), self.max_depth.to_json()),
            ("children".to_string(), JSONNode::Array(root_children_json_data)),
            ("weak_children".to_string(), JSONNode::Array(root_weak_children_json_data)),
        ]));

        // BFS for other nodes
        while let Some(current_arc) = bfs_q.pop_front() {
            let current_arc_ptr = Arc::as_ptr(&current_arc);
            let current_node_json_idx = *arc_ptr_to_idx_map.get(&current_arc_ptr)
                .expect("Node in BFS queue must have an assigned index");

            let node_guard = current_arc.read().expect("RwLock poisoned during Trie serialization (BFS part)");
            let mut current_node_children_json_bfs = Vec::new();
            let mut current_node_weak_children_json_bfs = Vec::new();

            for (edge_key, destinations_map) in &node_guard.children {
                let ek_json = edge_key.to_json();
                let mut dest_map_json_array_bfs = Vec::new();
                for (child_arc_ptr_wrapper, edge_val) in destinations_map {
                    let child_arc = child_arc_ptr_wrapper.as_arc();
                    let child_arc_ptr = Arc::as_ptr(child_arc);

                    let child_idx = match arc_ptr_to_idx_map.get(&child_arc_ptr) {
                        Some(idx) => *idx,
                        None => {
                            let new_idx = nodes_json_list.len();
                            arc_ptr_to_idx_map.insert(child_arc_ptr, new_idx);
                            bfs_q.push_back(child_arc.clone());
                            nodes_json_list.push(JSONNode::Null);
                            new_idx
                        }
                    };
                    dest_map_json_array_bfs.push(JSONNode::Array(vec![
                        child_idx.to_json(),
                        edge_val.to_json(),
                    ]));
                }
                current_node_children_json_bfs.push(JSONNode::Array(vec![ek_json, JSONNode::Array(dest_map_json_array_bfs)]));
            }

            for (edge_key, destinations_map) in &node_guard.weak_children {
                let ek_json = edge_key.to_json();
                let mut dest_map_json_array_bfs = Vec::new();
                for (weak_wrapper, edge_val) in destinations_map {
                    if let Some(child_arc) = weak_wrapper.upgrade() {
                        let child_arc_ptr = Arc::as_ptr(&child_arc);
                        let child_idx = match arc_ptr_to_idx_map.get(&child_arc_ptr) {
                            Some(idx) => *idx,
                            None => {
                                let new_idx = nodes_json_list.len();
                                arc_ptr_to_idx_map.insert(child_arc_ptr, new_idx);
                                bfs_q.push_back(child_arc.clone());
                                nodes_json_list.push(JSONNode::Null);
                                new_idx
                            }
                        };
                        dest_map_json_array_bfs.push(JSONNode::Array(vec![
                            child_idx.to_json(),
                            edge_val.to_json(),
                        ]));
                    }
                }
                current_node_weak_children_json_bfs.push(JSONNode::Array(vec![ek_json, JSONNode::Array(dest_map_json_array_bfs)]));
            }

            nodes_json_list[current_node_json_idx] = JSONNode::Object(BTreeMap::from_iter(vec![
                ("value".to_string(), node_guard.value.to_json()),
                ("max_depth".to_string(), node_guard.max_depth.to_json()),
                ("children".to_string(), JSONNode::Array(current_node_children_json_bfs)),
                ("weak_children".to_string(), JSONNode::Array(current_node_weak_children_json_bfs)),
            ]));
        }

        JSONNode::Object(BTreeMap::from_iter(vec![
            ("nodes".to_string(), JSONNode::Array(nodes_json_list)),
            ("root_idx".to_string(), root_idx.to_json()),
        ]))
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let nodes_json = obj.remove("nodes").ok_or_else(|| "Missing 'nodes' field for Trie deserialization".to_string())?;
                let root_idx_json = obj.remove("root_idx").ok_or_else(|| "Missing 'root_idx' field for Trie deserialization".to_string())?;

                let nodes_array = match nodes_json {
                    JSONNode::Array(arr) => arr,
                    _ => return Err("'nodes' field is not an array".to_string()),
                };
                let root_idx = usize::from_json(root_idx_json)?;

                if root_idx >= nodes_array.len() {
                    return Err(format!("Root index {} is out of bounds for nodes array of length {}", root_idx, nodes_array.len()));
                }

                let mut deserialized_arcs: HashMap<usize, Arc<RwLock<Trie<EK, EV, T>>>> = HashMap::new();

                // Pass 1: Create node shells (value, max_depth, empty children)
                for (i, node_data_json) in nodes_array.iter().enumerate() {
                    match node_data_json {
                        JSONNode::Object(n_obj) => {
                            let value_json = n_obj.get("value").ok_or_else(|| format!("Node at index {} missing 'value'", i))?;
                            let max_depth_json = n_obj.get("max_depth").ok_or_else(|| format!("Node at index {} missing 'max_depth'", i))?;

                            let value = T::from_json(value_json.clone())?;
                            let max_depth = usize::from_json(max_depth_json.clone())?;

                            let new_node_arc = Arc::new(RwLock::new(Trie {
                                value,
                                children: BTreeMap::new(),
                                weak_children: BTreeMap::new(),
                                max_depth,
                            }));
                            deserialized_arcs.insert(i, new_node_arc);
                        }
                        _ => return Err(format!("Node data at index {} is not an object", i)),
                    }
                }

                // Pass 2: Link children
                for (i, node_data_json) in nodes_array.iter().enumerate() {
                    match node_data_json {
                        JSONNode::Object(n_obj) => {
                            let current_node_arc = deserialized_arcs.get(&i)
                                .ok_or_else(|| format!("Failed to find node for index {} in Pass 2", i))?
                                .clone();
                            let mut current_node_guard = current_node_arc.write().unwrap();

                            let children_json_outer_array = n_obj.get("children")
                                .ok_or_else(|| format!("Node at index {} missing 'children' field in Pass 2", i))?;
                            let weak_children_json_outer_array_opt = n_obj.get("weak_children");

                            match children_json_outer_array {
                                JSONNode::Array(children_ek_map_array) => {
                                    for ek_entry_json in children_ek_map_array {
                                        match ek_entry_json {
                                            JSONNode::Array(ek_pair) if ek_pair.len() == 2 => {
                                                let ek_json = &ek_pair[0];
                                                let dest_map_json_array = &ek_pair[1];

                                                let edge_key = EK::from_json(ek_json.clone())?;
                                                let mut destinations_for_this_ek = OrderedHashMap::new();

                                                match dest_map_json_array {
                                                    JSONNode::Array(dest_array_inner) => {
                                                        for child_ev_pair_json in dest_array_inner {
                                                            match child_ev_pair_json {
                                                                JSONNode::Array(child_ev_pair_inner) if child_ev_pair_inner.len() == 2 => {
                                                                    let child_idx_json = &child_ev_pair_inner[0];
                                                                    let ev_json = &child_ev_pair_inner[1];

                                                                    let child_idx = usize::from_json(child_idx_json.clone())?;
                                                                    let child_arc = deserialized_arcs.get(&child_idx)
                                                                        .ok_or_else(|| format!("Child index {} not found for node {} in Pass 2", child_idx, i))?
                                                                        .clone();
                                                                    let edge_value = EV::from_json(ev_json.clone())?;
                                                                    destinations_for_this_ek.insert(ArcPtrWrapper::new(child_arc), edge_value);
                                                                }
                                                                _ => return Err(format!("Invalid child_idx-EV pair format for node {} under edge key {:?}", i, edge_key)),
                                                            }
                                                        }
                                                    }
                                                    _ => return Err(format!("Children destination map for node {} under edge key {:?} is not an array", i, edge_key)),
                                                }
                                                current_node_guard.children.insert(edge_key, destinations_for_this_ek);
                                            }
                                            _ => return Err(format!("Invalid EK-children_map_array pair format for node {}", i)),
                                        }
                                    }
                                }
                                _ => return Err(format!("'children' field for node {} is not an array of EK-entries", i)),
                            }

                            // weak_children if present
                            if let Some(weak_children_node) = weak_children_json_outer_array_opt {
                                match weak_children_node {
                                    JSONNode::Array(children_ek_map_array) => {
                                        for ek_entry_json in children_ek_map_array {
                                            match ek_entry_json {
                                                JSONNode::Array(ek_pair) if ek_pair.len() == 2 => {
                                                    let ek_json = &ek_pair[0];
                                                    let dest_map_json_array = &ek_pair[1];

                                                    let edge_key = EK::from_json(ek_json.clone())?;
                                                    let mut destinations_for_this_ek: OrderedHashMap<WeakPtrWrapper<RwLock<Trie<EK, EV, T>>>, EV> = OrderedHashMap::new();

                                                    match dest_map_json_array {
                                                        JSONNode::Array(dest_array_inner) => {
                                                            for child_ev_pair_json in dest_array_inner {
                                                                match child_ev_pair_json {
                                                                    JSONNode::Array(child_ev_pair_inner) if child_ev_pair_inner.len() == 2 => {
                                                                        let child_idx_json = &child_ev_pair_inner[0];
                                                                        let ev_json = &child_ev_pair_inner[1];

                                                                        let child_idx = usize::from_json(child_idx_json.clone())?;
                                                                        let child_arc = deserialized_arcs.get(&child_idx)
                                                                            .ok_or_else(|| format!("Child index {} not found for node {} in Pass 2 (weak)", child_idx, i))?
                                                                            .clone();
                                                                        let edge_value = EV::from_json(ev_json.clone())?;
                                                                        let weak_wrapper = WeakPtrWrapper::new(Arc::downgrade(&child_arc));
                                                                        destinations_for_this_ek.insert(weak_wrapper, edge_value);
                                                                    }
                                                                    _ => return Err(format!("Invalid weak child_idx-EV pair format for node {} under edge key {:?}", i, edge_key)),
                                                                }
                                                            }
                                                        }
                                                        _ => return Err(format!("Weak children destination map for node {} under edge key {:?} is not an array", i, edge_key)),
                                                    }
                                                    current_node_guard.weak_children.insert(edge_key, destinations_for_this_ek);
                                                }
                                                _ => return Err(format!("Invalid EK-weak_children_map_array pair format for node {}", i)),
                                            }
                                        }
                                    }
                                    _ => return Err(format!("'weak_children' field for node {} is not an array of EK-entries", i)),
                                }
                            }
                        }
                        _ => unreachable!("Node data should be an object, checked in Pass 1"),
                    }
                }

                let root_arc_final = deserialized_arcs.get(&root_idx)
                    .ok_or_else(|| format!("Root index {} not found in deserialized_arcs map after linking", root_idx))?
                    .clone();

                let root_trie_content = root_arc_final.read().unwrap().clone();
                Ok(root_trie_content)
            }
            _ => Err("Expected JSONNode::Object for Trie graph structure".to_string()),
        }
    }
}

impl<T> JSONConvertible for Arc<RwLock<T>>
where
    T: JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        self.read()
            .expect("RwLock poisoned during JSON serialization")
            .to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        T::from_json(node).map(|val| Arc::new(RwLock::new(val)))
    }
}

impl<EK: Ord + Clone, EV, T> Trie<EK, EV, T> {
    /// Creates a new trie node with the given value and no children.
    /// The max_depth is initialized to 0.
    pub fn new(value: T) -> Self {
        Trie {
            value,
            children: BTreeMap::new(),
            weak_children: BTreeMap::new(),
            max_depth: 0,
        }
    }

    pub fn force_insert_to_new_node(&mut self, edge_key: EK, edge_value: EV, value: T) -> Arc<RwLock<Trie<EK, EV, T>>> {
        let new_node = Arc::new(RwLock::new(Trie::new(value)));
        let new_node_comparable = ArcPtrWrapper::new(new_node.clone());
        self.children.entry(edge_key).or_default().insert(new_node_comparable, edge_value);
        // Note: force_insert does NOT update max_depth or check for cycles. Use with caution.
        new_node.clone()
    }

    pub fn force_insert_to_node(&mut self, edge_key: EK, edge_value: EV, dst: &Arc<RwLock<Trie<EK, EV, T>>>) {
        let dst_comparable = ArcPtrWrapper::new(dst.clone());
        self.children.entry(edge_key).or_default().insert(dst_comparable, edge_value);
    }

    /// Insert a weak edge explicitly. This allows cycles by design.
    /// Does NOT update max_depth and will not keep the destination alive.
    pub fn insert_weak_to_node(&mut self, edge_key: EK, edge_value: EV, dst: &Arc<RwLock<Trie<EK, EV, T>>>) {
        let weak = WeakPtrWrapper::new(Arc::downgrade(dst));
        self.weak_children.entry(edge_key).or_default().insert(weak, edge_value);
    }

    /// Convenience: create a new node and insert a weak edge to it.
    pub fn insert_weak_to_new_node(&mut self, edge_key: EK, edge_value: EV, value: T) -> Arc<RwLock<Trie<EK, EV, T>>> {
        let new_node = Arc::new(RwLock::new(Trie::new(value)));
        {
            let weak = WeakPtrWrapper::new(Arc::downgrade(&new_node));
            self.weak_children.entry(edge_key).or_default().insert(weak, edge_value);
        }
        new_node
    }

    pub fn already_has_dst(&self, edge_key: EK, dst: &Arc<RwLock<Trie<EK, EV, T>>>) -> bool {
        let lookup_key = ArcPtrWrapper::new(dst.clone());
        self.children.get(&edge_key).map_or(false, |dest_map| dest_map.contains_key(&lookup_key))
    }

    pub fn already_has_dst_for_any_key(&self, dst: &Arc<RwLock<Trie<EK, EV, T>>>) -> bool {
        let lookup_key = ArcPtrWrapper::new(dst.clone());
        self.children.values().any(|dest_map| dest_map.contains_key(&lookup_key))
    }

    pub fn get_edge_value(&self, edge_key: EK, dst: &Arc<RwLock<Trie<EK, EV, T>>>) -> Option<&EV> {
        let lookup_key = ArcPtrWrapper::new(dst.clone());
        self.children.get(&edge_key).and_then(|dest_map| dest_map.get(&lookup_key))
    }

    /// Weak variant: check if a weak edge already exists to dst.
    pub fn already_has_weak_dst(&self, edge_key: EK, dst: &Arc<RwLock<Trie<EK, EV, T>>>) -> bool {
        let weak = WeakPtrWrapper::new(Arc::downgrade(dst));
        self.weak_children.get(&edge_key).map_or(false, |dest_map| dest_map.contains_key(&weak))
    }

    /// Weak variant: get weak edge EV (if the weak pointer matches).
    pub fn get_weak_edge_value(&self, edge_key: EK, dst: &Arc<RwLock<Trie<EK, EV, T>>>) -> Option<&EV> {
        let weak = WeakPtrWrapper::new(Arc::downgrade(dst));
        self.weak_children.get(&edge_key).and_then(|dest_map| dest_map.get(&weak))
    }

    pub fn get_edge_value_mut(&mut self, edge_key: EK, dst: &Arc<RwLock<Trie<EK, EV, T>>>) -> Option<&mut EV> {
        let lookup_key = ArcPtrWrapper::new(dst.clone());
        self.children.get_mut(&edge_key).and_then(|dest_map| dest_map.get_mut(&lookup_key))
    }

    /// Budget to bound worst-case cycle detection work.
    const DETECT_CYCLE_VISIT_BUDGET: usize = 1_000_000;

    #[time_it]
    pub fn try_insert(
        &mut self,
        self_arc: &Arc<RwLock<Trie<EK, EV, T>>>,
        edge_key: EK,
        edge_value: &mut Option<EV>,
        child: Arc<RwLock<Trie<EK, EV, T>>>,
    ) -> Result<(), CycleDetectedError> {
        // 1) Fast cycle test: if adding self -> child would create a cycle,
        // then self must be reachable from child.
        let self_mutex_ptr = Arc::as_ptr(self_arc);
        let self_max_depth = self.max_depth;

        if !self.already_has_dst_for_any_key(&child) &&
            Self::detect_cycle(self_mutex_ptr, self_max_depth, &child) {
            return Err(CycleDetectedError);
        }

        self.try_insert_unchecked(self_arc, edge_key, edge_value, child)
    }

    #[time_it]
    pub fn try_insert_unchecked(
        &mut self,
        self_arc: &Arc<RwLock<Trie<EK, EV, T>>>,
        edge_key: EK,
        edge_value: &mut Option<EV>,
        child: Arc<RwLock<Trie<EK, EV, T>>>,
    ) -> Result<(), CycleDetectedError> {

        // 2) Update child's max_depth before linking edge (for clean rollback if needed)
        let candidate_depth = self.max_depth.saturating_add(1);
        let previous_child_depth;
        let needs_depth_update;

        {
            let mut child_guard = child.write().expect("RwLock poisoned while updating child's max_depth");
            previous_child_depth = child_guard.max_depth;
            needs_depth_update = candidate_depth > previous_child_depth;
            if needs_depth_update {
                child_guard.max_depth = candidate_depth;
            }
        }

        if needs_depth_update {
            if let Err(e) = Self::propagate_max_depth(child.clone(), candidate_depth) {
                let mut child_guard = child.write().expect("RwLock poisoned while rolling back max_depth");
                if child_guard.max_depth == candidate_depth {
                    child_guard.max_depth = previous_child_depth;
                }
                return Err(e);
            }
        }

        // 3) Perform the mutation
        let child_comparable = ArcPtrWrapper::new(child.clone());
        self.children
            .entry(edge_key)
            .or_default()
            .insert(child_comparable, edge_value.take().unwrap());

        Ok(())
    }

    /// Try to insert an edge; if it would create a strong cycle, insert it as a WEAK edge instead.
    #[time_it]
    pub fn try_insert_auto(
        &mut self,
        self_arc: &Arc<RwLock<Trie<EK, EV, T>>>,
        edge_key: EK,
        edge_value: &mut Option<EV>,
        child: Arc<RwLock<Trie<EK, EV, T>>>,
    ) -> InsertedEdgeKind {
        let self_mutex_ptr = Arc::as_ptr(self_arc);
        let self_max_depth = self.max_depth;

        let would_cycle = if self.already_has_dst_for_any_key(&child) {
            false
        } else {
            Self::detect_cycle(self_mutex_ptr, self_max_depth, &child)
        };
        if would_cycle {
            if let Some(ev) = edge_value.take() {
                let weak = WeakPtrWrapper::new(Arc::downgrade(&child));
                self.weak_children.entry(edge_key).or_default().insert(weak, ev);
            }
            return InsertedEdgeKind::Weak;
        }

        let mut ev_opt = edge_value.take();
        let _ = self.try_insert_unchecked(self_arc, edge_key, &mut ev_opt, child)
            .expect("Cycle re-appeared unexpectedly during try_insert_auto strong path");
        *edge_value = ev_opt;
        InsertedEdgeKind::Strong
    }

    /// Fast cycle detection that avoids locking the target and prunes by depth.
    /// Returns true if the node pointed by `target_mutex_ptr` is reachable from `start_arc`.
    #[time_it]
    pub fn detect_cycle(
        target_mutex_ptr: *const RwLock<Trie<EK, EV, T>>,
        target_max_depth: usize,
        start_arc: &Arc<RwLock<Trie<EK, EV, T>>>,
    ) -> bool {
        // Quick positive: identical arc
        if Arc::as_ptr(start_arc) == target_mutex_ptr {
            return true;
        }

        // Quick negative: depth gate
        let start_depth = match start_arc.try_read() {
            Ok(g) => g.max_depth,
            Err(TryLockError::WouldBlock) => {
                // If we cannot read the start node immediately, conservatively proceed with BFS.
                // In typical usage this won't happen for `child`.
                0
            }
            Err(TryLockError::Poisoned(p)) => {
                panic!("RwLock poisoned during cycle detection (start): {:?}", p);
            }
        };
        if start_depth >= target_max_depth {
            return false;
        }

        // BFS with depth gating and visit budget
        let mut visited_arcs: HashSet<*const RwLock<Trie<EK, EV, T>>> = HashSet::new();
        let mut queue: VecDeque<Arc<RwLock<Trie<EK, EV, T>>>> = VecDeque::new();

        let start_ptr = Arc::as_ptr(start_arc);
        visited_arcs.insert(start_ptr);
        queue.push_back(start_arc.clone());

        let mut visits = 0usize;

        while let Some(node_arc) = queue.pop_front() {
            visits += 1;
            if visits > Self::DETECT_CYCLE_VISIT_BUDGET {
                // In pathological cases, bail out to keep insertion responsive.
                // Returning false degrades to strong insert unless the caller uses try_insert_auto
                // that will choose Weak on actual cycle.
                return false;
            }

            // Pointer match is enough to detect target without locking it.
            let node_mutex_ptr = Arc::as_ptr(&node_arc);
            if node_mutex_ptr == target_mutex_ptr {
                return true;
            }

            // Read current node and collect children
            let (children_arcs, cur_depth) = {
                let guard = node_arc.read().expect("RwLock poisoned in detect_cycle");
                let cur_depth = guard.max_depth;
                // Small optimization: if cur_depth >= target_max_depth, no descendants can reach target
                if cur_depth >= target_max_depth {
                    (Vec::new(), cur_depth)
                } else {
                    let v: Vec<Arc<RwLock<Trie<EK, EV, T>>>> = guard.children
                        .values()
                        .flat_map(|dest_map| dest_map.keys().map(|wrap| wrap.as_arc().clone()))
                        .collect();
                    (v, cur_depth)
                }
            };

            for child_arc in children_arcs {
                let child_ptr = Arc::as_ptr(&child_arc);
                if visited_arcs.contains(&child_ptr) {
                    continue;
                }
                // Depth gate on child (read depth cheaply)
                let pass = {
                    let child_guard = child_arc.read().expect("RwLock poisoned reading child in detect_cycle");
                    child_guard.max_depth <= target_max_depth
                };
                if pass {
                    visited_arcs.insert(child_ptr);
                    queue.push_back(child_arc);
                }
            }
        }

        false
    }

    /// Propagates a max_depth update to all descendant nodes, detecting cycles.
    ///
    /// Returns `Ok(())` if propagation completes successfully.
    /// Returns `Err(CycleDetectedError)`.
    fn propagate_max_depth(node_arc: Arc<RwLock<Trie<EK, EV, T>>>, current_depth: usize) -> Result<(), CycleDetectedError> {
        let mut rec_stack: HashSet<*const RwLock<Trie<EK, EV, T>>> = HashSet::new();
        Self::_propagate_max_depth(node_arc, current_depth, &mut rec_stack)
    }

    fn _propagate_max_depth(
        node_arc: Arc<RwLock<Trie<EK, EV, T>>>,
        current_depth: usize,
        rec_stack: &mut HashSet<*const RwLock<Trie<EK, EV, T>>>,
    ) -> Result<(), CycleDetectedError> {
        let node_arc_ptr = Arc::as_ptr(&node_arc);

        if rec_stack.contains(&node_arc_ptr) {
            return Err(CycleDetectedError);
        }

        rec_stack.insert(node_arc_ptr);

        let children_arcs: Vec<Arc<RwLock<Trie<EK, EV, T>>>> = {
            let node_guard_val = node_arc.read().expect("RwLock poisoned in _propagate_max_depth (getting children)");
            node_guard_val.children
                .values()
                .flat_map(|dest_map| dest_map.keys().map(|wrapper_arc| wrapper_arc.as_arc().clone()))
                .collect()
        };

        let candidate_depth_val = current_depth.saturating_add(1);
        for child_arc in children_arcs {
            let should_propagate = {
                let mut child_guard = child_arc.write().expect("RwLock poisoned in _propagate_max_depth (checking child depth)");
                if candidate_depth_val > child_guard.max_depth {
                    child_guard.max_depth = candidate_depth_val;
                    true
                } else {
                    false
                }
            };

            if should_propagate {
                Self::_propagate_max_depth(child_arc, candidate_depth_val, rec_stack)?;
            }
        }

        rec_stack.remove(&node_arc_ptr);
        Ok(())
    }

    pub fn get(
        &self,
        edge_key: &EK,
    ) -> Option<&OrderedHashMap<ArcPtrWrapper<RwLock<Trie<EK, EV, T>>>, EV>>
    {
        self.children.get(edge_key)
    }

    pub fn get_mut(
        &mut self,
        edge_key: &EK,
    ) -> Option<&mut OrderedHashMap<ArcPtrWrapper<RwLock<Trie<EK, EV, T>>>, EV>>
    {
        self.children.get_mut(edge_key)
    }

    pub fn children(&self) -> &BTreeMap<EK, OrderedHashMap<ArcPtrWrapper<RwLock<Trie<EK, EV, T>>>, EV>> {
        &self.children
    }

    pub fn children_mut(&mut self) -> &mut BTreeMap<EK, OrderedHashMap<ArcPtrWrapper<RwLock<Trie<EK, EV, T>>>, EV>> {
        &mut self.children
    }

    pub fn weak_children(&self) -> &BTreeMap<EK, OrderedHashMap<WeakPtrWrapper<RwLock<Trie<EK, EV, T>>>, EV>> {
        &self.weak_children
    }

    pub fn weak_children_mut(&mut self) -> &mut BTreeMap<EK, OrderedHashMap<WeakPtrWrapper<RwLock<Trie<EK, EV, T>>>, EV>> {
        &mut self.weak_children
    }

    pub fn get_weak(
        &self,
        edge_key: &EK,
    ) -> Option<&OrderedHashMap<WeakPtrWrapper<RwLock<Trie<EK, EV, T>>>, EV>> {
        self.weak_children.get(edge_key)
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Collects all unique nodes (by pointer) reachable from the given root (BFS).
    pub fn all_nodes(root: Arc<RwLock<Trie<EK, EV, T>>>) -> Vec<Arc<RwLock<Trie<EK, EV, T>>>> {
        let mut visited_arcs: HashSet<*const RwLock<Trie<EK, EV, T>>> = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        let root_arc_ptr = Arc::as_ptr(&root);
        if visited_arcs.insert(root_arc_ptr) {
            queue.push_back(root);
        }

        while let Some(node_arc) = queue.pop_front() {
            result.push(node_arc.clone());

            let node_guard = node_arc.read().expect("RwLock poisoned during BFS");
            for children_map in node_guard.children.values() {
                for child_wrapper_arc in children_map.keys() {
                    let child_arc = child_wrapper_arc.as_arc();
                    let child_arc_ptr = Arc::as_ptr(child_arc);
                    if visited_arcs.insert(child_arc_ptr) {
                        queue.push_back(child_arc.clone());
                    }
                }
            }
            for children_map in node_guard.weak_children.values() {
                for weak_wrapper in children_map.keys() {
                    if let Some(child_arc) = weak_wrapper.upgrade() {
                        let child_ptr = Arc::as_ptr(&child_arc);
                        if visited_arcs.insert(child_ptr) {
                            queue.push_back(child_arc.clone());
                        }
                    }
                }
            }
        }
        result
    }

    pub fn has_any_cycle(root_arc: Arc<RwLock<Trie<EK, EV, T>>>) -> bool {
        let mut global_visited_arcs: HashSet<*const RwLock<Trie<EK, EV, T>>> = HashSet::new();
        let mut recursion_stack_arcs: HashSet<*const RwLock<Trie<EK, EV, T>>> = HashSet::new();
        Self::_has_any_cycle_recursive(root_arc, &mut global_visited_arcs, &mut recursion_stack_arcs)
    }

    fn _has_any_cycle_recursive(
        node_arc: Arc<RwLock<Trie<EK, EV, T>>>,
        global_visited_arcs: &mut HashSet<*const RwLock<Trie<EK, EV, T>>>,
        recursion_stack_arcs: &mut HashSet<*const RwLock<Trie<EK, EV, T>>>,
    ) -> bool {
        let node_arc_ptr = Arc::as_ptr(&node_arc);

        if recursion_stack_arcs.contains(&node_arc_ptr) {
            return true;
        }

        if global_visited_arcs.contains(&node_arc_ptr) {
            return false;
        }

        recursion_stack_arcs.insert(node_arc_ptr);
        global_visited_arcs.insert(node_arc_ptr);

        let children_arcs: Vec<Arc<RwLock<Trie<EK, EV, T>>>> = {
            let node_guard_val = node_arc.read().expect("RwLock poisoned during has_any_cycle traversal");
            node_guard_val.children
                .values()
                .flat_map(|dest_map| dest_map.keys().map(|wrapper_arc| wrapper_arc.as_arc().clone()))
                .collect()
        };

        for child_arc in children_arcs {
            if Self::_has_any_cycle_recursive(child_arc, global_visited_arcs, recursion_stack_arcs) {
                return true;
            }
        }

        recursion_stack_arcs.remove(&node_arc_ptr);
        false
    }
}

/// Panics if the lock is poisoned. Returns None if lock fails (WouldBlock).
#[allow(dead_code)]
pub(crate) fn try_get_node_data_ptr<EK: Ord, EV, T>(node_arc: &Arc<RwLock<Trie<EK, EV, T>>>) -> Option<*const Trie<EK, EV, T>> {
    match node_arc.try_read() {
        Ok(guard) => {
            let ptr = &*guard as *const Trie<EK, EV, T>;
            Some(ptr)
        }
        Err(TryLockError::Poisoned(p)) => {
            panic!("RwLock poisoned when trying to get node data pointer: {:?}", p);
        }
        Err(TryLockError::WouldBlock) => {
            None
        }
    }
}

#[allow(dead_code)]
pub(crate) fn node_ptr<EK: Ord, EV, T>(node_arc: &Arc<RwLock<Trie<EK, EV, T>>>) -> *const Trie<EK, EV, T> {
    let guard = node_arc.read().expect("RwLock poisoned or lock failed when getting node pointer");
    &*guard as *const _
}

impl<EK, EV, T> Trie<EK, EV, T>
where
    EK: Ord,
    EV: PartialEq + Clone,
    T: PartialEq,
{
    fn compare_arcs_recursive(
        self_arc: &Arc<RwLock<Trie<EK, EV, T>>>,
        other_arc: &Arc<RwLock<Trie<EK, EV, T>>>,
        comparison_cache: &mut HashMap<(*const RwLock<Self>, *const RwLock<Self>), bool>,
    ) -> bool {
        let self_ptr = Arc::as_ptr(self_arc);
        let other_ptr = Arc::as_ptr(other_arc);

        if self_ptr == other_ptr {
            return true;
        }

        let (cache_key_ptr1, cache_key_ptr2) = if self_ptr < other_ptr {
            (self_ptr, other_ptr)
        } else {
            (other_ptr, self_ptr)
        };

        if let Some(&cached_result) = comparison_cache.get(&(cache_key_ptr1, cache_key_ptr2)) {
            return cached_result;
        }

        comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), true);

        let self_node_guard = match self_arc.try_read() {
            Ok(g) => g,
            Err(_) => {
                comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false);
                return false;
            }
        };
        let other_node_guard = match other_arc.try_read() {
            Ok(g) => g,
            Err(_) => {
                comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false);
                return false;
            }
        };

        let self_node = &*self_node_guard;
        let other_node = &*other_node_guard;

        if self_node.value != other_node.value || self_node.max_depth != other_node.max_depth {
            comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false);
            return false;
        }

        if self_node.children.len() != other_node.children.len() {
            comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false);
            return false;
        }

        for (self_ek, self_dest_map) in &self_node.children {
            match other_node.children.get(self_ek) {
                None => {
                    comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false);
                    return false;
                }
                Some(other_dest_map) => {
                    if self_dest_map.len() != other_dest_map.len() {
                        comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false);
                        return false;
                    }

                    let self_child_pairs: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, EV)> = self_dest_map
                        .iter()
                        .map(|(apw, ev)| (apw.as_arc().clone(), ev.clone()))
                        .collect();

                    let mut other_child_pairs: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, EV)> = other_dest_map
                        .iter()
                        .map(|(apw, ev)| (apw.as_arc().clone(), ev.clone()))
                        .collect();

                    'self_pair_loop: for (s_arc, s_ev) in &self_child_pairs {
                        let mut found_match_for_current_self_pair = false;
                        for i in 0..other_child_pairs.len() {
                            if s_ev == &other_child_pairs[i].1 {
                                let o_arc_for_recursion = other_child_pairs[i].0.clone();
                                if Trie::compare_arcs_recursive(s_arc, &o_arc_for_recursion, comparison_cache) {
                                    other_child_pairs.remove(i);
                                    found_match_for_current_self_pair = true;
                                    break;
                                }
                            }
                        }
                        if !found_match_for_current_self_pair {
                            comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false);
                            return false;
                        }
                    }
                }
            }
        }

        for (self_ek, self_dest_map) in &self_node.weak_children {
            match other_node.weak_children.get(self_ek) {
                None => return false,
                Some(other_dest_map) => {
                    let self_child_pairs: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, EV)> = self_dest_map
                        .iter()
                        .filter_map(|(wpw, ev)| wpw.upgrade().map(|arc| (arc, ev.clone())))
                        .collect();
                    let mut other_child_pairs: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, EV)> = other_dest_map
                        .iter()
                        .filter_map(|(wpw, ev)| wpw.upgrade().map(|arc| (arc, ev.clone())))
                        .collect();

                    if self_child_pairs.len() != other_child_pairs.len() {
                        return false;
                    }

                    'self_weak_loop: for (s_arc, s_ev) in &self_child_pairs {
                        let mut found_match = false;
                        for i in 0..other_child_pairs.len() {
                            if s_ev == &other_child_pairs[i].1 {
                                let o_arc = other_child_pairs[i].0.clone();
                                if Trie::compare_arcs_recursive(s_arc, &o_arc, comparison_cache) {
                                    other_child_pairs.remove(i);
                                    found_match = true;
                                    break;
                                }
                            }
                        }
                        if !found_match {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }
}

impl<T: Clone, EK: Ord + Clone, EV: Clone> Trie<EK, EV, T> {
    fn count_all_edges(root_nodes: &[Arc<RwLock<Trie<EK, EV, T>>>]) -> usize {
        let mut visited_arcs: HashSet<*const RwLock<Trie<EK, EV, T>>> = HashSet::new();
        let mut queue: VecDeque<Arc<RwLock<Trie<EK, EV, T>>>> = VecDeque::new();
        let mut total_edges = 0;

        for root in root_nodes {
            let root_arc_ptr = Arc::as_ptr(root);
            if visited_arcs.insert(root_arc_ptr) {
                queue.push_back(root.clone());
            }
        }

        while let Some(node_arc) = queue.pop_front() {
            let node_guard = node_arc.read().expect("RwLock poisoned during edge count");
            for children_map in node_guard.children.values() {
                total_edges += children_map.len();
                for child_wrapper_arc in children_map.keys() {
                    let child_arc = child_wrapper_arc.as_arc();
                    if visited_arcs.insert(Arc::as_ptr(child_arc)) {
                        queue.push_back(child_arc.clone());
                    }
                }
            }
        }
        total_edges
    }

    #[time_it]
    pub fn special_map<V: Clone>(
        initial_nodes_and_values: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, V)>,
        mut step: impl FnMut(&V, &EK, &EV, &Trie<EK, EV, T>) -> Option<V>,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&mut Trie<EK, EV, T>, &mut V) -> bool,
    ) {
        let mut values   : HashMap<*const RwLock<Self>, V> = HashMap::new();
        let mut done     : HashSet <*const RwLock<Self>>   = HashSet ::new();
        let mut todo     : BTreeMap<usize, OrderedHashSet<ArcPtrWrapper<RwLock<Self>>>> = BTreeMap::new();

        let initial_nodes: Vec<_> = initial_nodes_and_values.iter().map(|(n, _)| n.clone()).collect();
        let total_edges = Self::count_all_edges(&initial_nodes);
        let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED);

        for (node_arc, v0) in initial_nodes_and_values {
            let ptr = Arc::as_ptr(&node_arc);
            values
                .entry(ptr)
                .and_modify(|old| merge(old, v0.clone()))
                .or_insert(v0);
            let depth = node_arc.read().expect("poison").max_depth;
            todo.entry(depth).or_default().insert(ArcPtrWrapper::new(node_arc.clone()));
        }

        while let Some((_depth, node_arc_ptr_wrappers)) = todo.pop_first() {
            for node_arc_ptr_wrapper in &node_arc_ptr_wrappers {
                let ptr = Arc::as_ptr(node_arc_ptr_wrapper.as_arc());
                if done.contains(&ptr) { continue; }

                let mut agg_v = match values.remove(&ptr) {
                    Some(v) => v,
                    None => continue,
                };

                let proceed = {
                    let mut guard = node_arc_ptr_wrapper.as_arc().write().expect("poison");
                    process(&mut guard, &mut agg_v)
                };
                done.insert(ptr);

                if !proceed { continue; }

                let edges: Vec<(EK, EV, Arc<RwLock<Self>>)> = {
                    let guard = node_arc_ptr_wrapper.as_arc().read().expect("poison");
                    guard.children
                        .iter()
                        .flat_map(|(ek, dst_map)| {
                            dst_map.iter().map(move |(wrap, ev)| (ek.clone(), ev.clone(), wrap.as_arc().clone()))
                        })
                        .collect()
                };

                for (ek, ev, child_arc) in edges {
                    let _ = pb.update(1);
                    let child_ptr = Arc::as_ptr(&child_arc);

                    let maybe_v = {
                        let child_guard = child_arc.read().expect("poison");
                        step(&agg_v, &ek, &ev, &child_guard)
                    };
                    if let Some(new_v) = maybe_v {
                        values
                            .entry(child_ptr)
                            .and_modify(|old| merge(old, new_v.clone()))
                            .or_insert(new_v);

                        let child_depth = child_arc.read().expect("poison").max_depth;
                        todo.entry(child_depth).or_default().insert(ArcPtrWrapper::new(child_arc));
                    }
                }
            }
        }
    }

    #[time_it]
    pub fn special_map_grouped<V, S, I>(
        initial_nodes_and_values: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, V)>,
        mut step: S,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&mut Trie<EK, EV, T>, &mut V) -> bool,
    )
    where
        V: Clone,
        S: FnMut(
            &V, &EK, &OrderedHashMap<ArcPtrWrapper<RwLock<Trie<EK, EV, T>>>, EV>
        ) -> I,
        I: IntoIterator<Item = (ArcPtrWrapper<RwLock<Trie<EK, EV, T>>>, V)>,
    {
        let mut values: HashMap<*const RwLock<Self>, V> = HashMap::new();
        let mut done: HashSet<*const RwLock<Self>> = HashSet::new();
        let mut todo: BTreeMap<usize, OrderedHashSet<ArcPtrWrapper<RwLock<Self>>>> = BTreeMap::new();

        let initial_nodes: Vec<_> = initial_nodes_and_values.iter().map(|(n, _)| n.clone()).collect();
        let total_edges = Self::count_all_edges(&initial_nodes);
        let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED);

        for (node_arc, v0) in initial_nodes_and_values {
            let ptr = Arc::as_ptr(&node_arc);
            values
                .entry(ptr)
                .and_modify(|old| merge(old, v0.clone()))
                .or_insert(v0);
            let depth = node_arc.read().expect("poison").max_depth;
            todo.entry(depth).or_default().insert(ArcPtrWrapper::new(node_arc.clone()));
        }

        while let Some((_depth, node_arc_ptr_wrappers)) = todo.pop_first() {
            for node_arc_ptr_wrapper in &node_arc_ptr_wrappers {
                let ptr = Arc::as_ptr(node_arc_ptr_wrapper.as_arc());
                if done.contains(&ptr) { continue; }

                let mut agg_v = match values.remove(&ptr) {
                    Some(v) => v,
                    None => continue,
                };

                let proceed = {
                    let mut guard = node_arc_ptr_wrapper.as_arc().write().expect("poison");
                    process(&mut guard, &mut agg_v)
                };
                done.insert(ptr);

                if !proceed { continue; }

                let children_by_ek: Vec<(EK, OrderedHashMap<ArcPtrWrapper<RwLock<Self>>, EV>)> = {
                    let guard = node_arc_ptr_wrapper.as_arc().read().expect("poison");
                    guard.children.iter().map(|(ek, dst_map)| (ek.clone(), dst_map.clone())).collect()
                };

                for (ek, dest_map) in children_by_ek {
                    let _ = pb.update(dest_map.len());
                    let new_values_for_children = step(&agg_v, &ek, &dest_map);
                    for (child_arc, new_v) in new_values_for_children {
                        let child_ptr = Arc::as_ptr(&child_arc);
                        values.entry(child_ptr).and_modify(|old| merge(old, new_v.clone())).or_insert(new_v);
                        let child_depth = child_arc.read().expect("poison").max_depth;
                        todo.entry(child_depth).or_default().insert(child_arc);
                    }
                }
            }
        }
    }
}

impl<EK, EV, T> PartialEq for Trie<EK, EV, T>
where
    EK: Ord,
    EV: PartialEq + Clone,
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if self.value != other.value || self.max_depth != other.max_depth {
            return false;
        }

        if self.children.len() != other.children.len() {
            return false;
        }
        if self.weak_children.len() != other.weak_children.len() { return false; }

        type NodeLockPtr<EKK, EVV, TT> = *const RwLock<Trie<EKK, EVV, TT>>;
        let mut comparison_cache: HashMap<(NodeLockPtr<EK, EV, T>, NodeLockPtr<EK, EV, T>), bool> = HashMap::new();

        for (self_ek, self_dest_map) in &self.children {
            match other.children.get(self_ek) {
                None => {
                    return false;
                }
                Some(other_dest_map) => {
                    if self_dest_map.len() != other_dest_map.len() {
                        return false;
                    }

                    let self_child_pairs: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, EV)> = self_dest_map
                        .iter()
                        .map(|(apw, ev)| (apw.as_arc().clone(), ev.clone()))
                        .collect();

                    let mut other_child_pairs: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, EV)> = other_dest_map
                        .iter()
                        .map(|(apw, ev)| (apw.as_arc().clone(), ev.clone()))
                        .collect();

                    'self_pair_loop: for (s_arc, s_ev) in &self_child_pairs {
                        let mut found_match_for_current_self_pair = false;
                        for i in 0..other_child_pairs.len() {
                            if s_ev == &other_child_pairs[i].1 {
                                let o_arc_for_recursion = other_child_pairs[i].0.clone();
                                if Trie::compare_arcs_recursive(s_arc, &o_arc_for_recursion, &mut comparison_cache) {
                                    other_child_pairs.remove(i);
                                    found_match_for_current_self_pair = true;
                                    break;
                                }
                            }
                        }
                        if !found_match_for_current_self_pair {
                            return false;
                        }
                    }
                }
            }
        }

        for (self_ek, self_dest_map) in &self.weak_children {
            match other.weak_children.get(self_ek) {
                None => return false,
                Some(other_dest_map) => {
                    let self_child_pairs: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, EV)> = self_dest_map
                        .iter()
                        .filter_map(|(wpw, ev)| wpw.upgrade().map(|arc| (arc, ev.clone())))
                        .collect();
                    let mut other_child_pairs: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, EV)> = other_dest_map
                        .iter()
                        .filter_map(|(wpw, ev)| wpw.upgrade().map(|arc| (arc, ev.clone())))
                        .collect();

                    if self_child_pairs.len() != other_child_pairs.len() {
                        return false;
                    }

                    'self_weak_loop: for (s_arc, s_ev) in &self_child_pairs {
                        let mut found_match = false;
                        for i in 0..other_child_pairs.len() {
                            if s_ev == &other_child_pairs[i].1 {
                                let o_arc = other_child_pairs[i].0.clone();
                                if Trie::compare_arcs_recursive(s_arc, &o_arc, &mut comparison_cache) {
                                    other_child_pairs.remove(i);
                                    found_match = true;
                                    break;
                                }
                            }
                        }
                        if !found_match {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }
}

impl<EK, EV, T> Eq for Trie<EK, EV, T>
where
    EK: Ord,
    EV: Eq + Clone,
    T: Eq,
{
}

impl<EK, EV, T> Hash for Trie<EK, EV, T>
where
    EK: Ord + Hash,
    EV: PartialEq + Clone + Hash,
    T: PartialEq + Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut recursion_marker: HashMap<*const Trie<EK, EV, T>, usize> = HashMap::new();
        Self::hash_trie_recursive(self, state, &mut recursion_marker, 0);
    }
}

impl<EK, EV, T> Trie<EK, EV, T>
where
    EK: Ord + Hash,
    EV: PartialEq + Clone + Hash,
    T: PartialEq + Hash,
{
    fn hash_trie_recursive<S: Hasher>(
        node: &Trie<EK, EV, T>,
        state: &mut S,
        recursion_marker: &mut HashMap<*const Trie<EK, EV, T>, usize>,
        current_depth: usize,
    ) {
        let node_ptr = node as *const _;
        if let Some(visited_depth) = recursion_marker.get(&node_ptr) {
            node_ptr.hash(state);
            visited_depth.hash(state);
            return;
        }
        recursion_marker.insert(node_ptr, current_depth);

        node.value.hash(state);
        node.max_depth.hash(state);

        node.children.len().hash(state);
        for (ek, dest_map) in &node.children {
            ek.hash(state);
            dest_map.len().hash(state);

            let mut pair_hashes = Vec::with_capacity(dest_map.len());
            for (apw, ev) in dest_map {
                let mut pair_hasher = DeterministicHasher::new(DefaultHasher::new());
                ev.hash(&mut pair_hasher);
                let child_guard = apw.as_arc().read().expect("RwLock poisoned during Hash");
                Self::hash_trie_recursive(&*child_guard, &mut pair_hasher, recursion_marker, current_depth + 1);
                pair_hashes.push(pair_hasher.finish());
            }

            pair_hashes.sort_unstable();
            for h in pair_hashes {
                h.hash(state);
            }
        }

        node.weak_children.len().hash(state);
        for (ek, dest_map) in &node.weak_children {
            ek.hash(state);
            let mut pair_hashes: Vec<u64> = Vec::new();
            for (wpw, ev) in dest_map {
                if let Some(child_arc) = wpw.upgrade() {
                    let mut pair_hasher = DeterministicHasher::new(DefaultHasher::new());
                    ev.hash(&mut pair_hasher);
                    let child_guard = child_arc.read().expect("RwLock poisoned during Hash (weak)");
                    Self::hash_trie_recursive(&*child_guard, &mut pair_hasher, recursion_marker, current_depth + 1);
                    pair_hashes.push(pair_hasher.finish());
                }
            }
            pair_hashes.sort_unstable();
            (pair_hashes.len()).hash(state);
            for h in pair_hashes {
                h.hash(state);
            }
        }
    }
}

/// A helper struct to facilitate inserting an edge into a Trie,
/// trying multiple potential destinations and optionally creating a new node.
/// Provides a chainable interface.
pub struct EdgeInserter<EK, EV, T, FMergeEV>
where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
{
    source_arc: Arc<RwLock<Trie<EK, EV, T>>>,
    edge_key: EK,
    edge_value: Option<EV>,
    merge_edge_value: FMergeEV,
    result: Option<Arc<RwLock<Trie<EK, EV, T>>>>,
}

impl<EK, EV, T, FMergeEV> EdgeInserter<EK, EV, T, FMergeEV>
where
    EK: Ord + Clone + Debug,
    EV: Clone,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
{
    pub fn new(
        source_arc: Arc<RwLock<Trie<EK, EV, T>>>,
        edge_key: EK,
        edge_value: EV,
        merge_edge_value: FMergeEV,
    ) -> Self {
        EdgeInserter {
            source_arc,
            edge_key,
            edge_value: Some(edge_value),
            merge_edge_value,
            result: None,
        }
    }

    pub fn try_destination(mut self, destination: Arc<RwLock<Trie<EK, EV, T>>>) -> Self {
        if self.result.is_some() {
            return self;
        }

        let mut source_guard = self.source_arc.write().expect("RwLock poisoned while locking source in try_destination");
        let destination_wrapper = ArcPtrWrapper::new(destination.clone());

        if let Some(existing_ev_mut) = source_guard.children.get_mut(&self.edge_key).and_then(|dest_map| dest_map.get_mut(&destination_wrapper)) {
            (self.merge_edge_value)(existing_ev_mut, self.edge_value.take().unwrap());
            self.result = Some(destination);
        } else {
            match source_guard.try_insert(&self.source_arc, self.edge_key.clone(), &mut self.edge_value, destination.clone()) {
                Ok(()) => {
                    self.result = Some(destination);
                }
                Err(CycleDetectedError) => {
                    crate::debug!(4, "Cycle detected trying to insert edge {:?} to node {:p}", self.edge_key, Arc::as_ptr(&destination));
                }
            }
        }
        drop(source_guard);
        self
    }

    #[time_it]
    pub fn try_destination_auto(mut self, destination: Arc<RwLock<Trie<EK, EV, T>>>) -> Self {
        if self.result.is_some() {
            return self;
        }
        let mut source_guard = self.source_arc.write().expect("RwLock poisoned while locking source in try_destination_auto");
        let destination_wrapper = ArcPtrWrapper::new(destination.clone());
        if let Some(existing_ev_mut) = source_guard.children.get_mut(&self.edge_key).and_then(|dest_map| dest_map.get_mut(&destination_wrapper)) {
            (self.merge_edge_value)(existing_ev_mut, self.edge_value.take().unwrap());
            self.result = Some(destination);
        } else {
            let kind = source_guard.try_insert_auto(&self.source_arc, self.edge_key.clone(), &mut self.edge_value, destination.clone());
            match kind {
                InsertedEdgeKind::Strong | InsertedEdgeKind::Weak => {
                    self.result = Some(destination);
                }
            }
        }
        drop(source_guard);
        self
    }

    pub fn try_destinations(mut self, destinations: &[Arc<RwLock<Trie<EK, EV, T>>>]) -> Self {
        for destination in destinations {
            if self.result.is_some() {
                break;
            }
            self = self.try_destination(destination.clone());
        }
        self
    }

    pub fn try_destinations_iter(mut self, destinations: impl Iterator<Item = Arc<RwLock<Trie<EK, EV, T>>>>) -> Self {
        for destination in destinations {
            if self.result.is_some() {
                break;
            }
            self = self.try_destination(destination.clone());
        }
        self
    }

    pub fn try_destinations_iter_with<F, R>(mut self, destinations: F) -> Self
    where
        F: Fn() -> R,
        R: Iterator<Item = Arc<RwLock<Trie<EK, EV, T>>>>,
    {
        for destination in destinations() {
            if self.result.is_some() {
                break;
            }
            self = self.try_destination(destination.clone());
        }
        self
    }

    pub fn try_children(mut self) -> Self {
        if self.result.is_some() {
            return self;
        }

        let children_for_this_key: Vec<Arc<RwLock<Trie<EK, EV, T>>>> = {
            let source_guard = self.source_arc.read().expect("RwLock poisoned while locking source in try_children");
            if let Some(dest_map) = source_guard.children.get(&self.edge_key) {
                dest_map.keys().map(|wrapper_arc| wrapper_arc.as_arc().clone()).collect()
            } else {
                Vec::new()
            }
        };

        if !children_for_this_key.is_empty() {
            self = self.try_destinations(&children_for_this_key);
        }
        self
    }

    pub fn else_create_destination_with_value(mut self, value: T) -> Self {
        if self.result.is_some() {
            return self;
        }

        let new_node_arc = Arc::new(RwLock::new(Trie::new(value)));
        let mut source_guard = self.source_arc.write().expect("RwLock poisoned while locking source in else_create_with_value");

        match source_guard.try_insert(&self.source_arc, self.edge_key.clone(), &mut self.edge_value, new_node_arc.clone()) {
            Ok(()) => {
                self.result = Some(new_node_arc);
            }
            Err(CycleDetectedError) => {
                crate::debug!(1, "Cycle detected trying to insert edge {:?} to NEW node {:p}. Creation failed.", self.edge_key, Arc::as_ptr(&new_node_arc));
            }
        }
        drop(source_guard);
        self
    }

    pub fn else_create_destination_with(self, value_fn: impl FnOnce() -> T) -> Self {
        if self.result.is_some() {
            return self;
        }
        self.else_create_destination_with_value(value_fn())
    }

    pub fn else_create_destination(self) -> Self
    where
        T: Default,
    {
        if self.result.is_some() {
            return self;
        }
        self.else_create_destination_with_value(T::default())
    }

    pub fn into_option(self) -> Option<Arc<RwLock<Trie<EK, EV, T>>>> {
        self.result
    }

    pub fn clone_into_option(&self) -> Option<Arc<RwLock<Trie<EK, EV, T>>>> {
        self.result.clone()
    }

    pub fn unwrap(self) -> Arc<RwLock<Trie<EK, EV, T>>> {
        self.result.expect("EdgeInserter::unwrap() called but no destination was found or created")
    }

    pub fn expect(self, msg: &str) -> Arc<RwLock<Trie<EK, EV, T>>> {
        self.result.expect(msg)
    }
}

impl<EK: Ord + Clone + Debug, EV: Clone, T: Clone> Trie<EK, EV, T> {
    pub fn insert_edge<FMergeEV>(
        &self,
        edge_key: EK,
        edge_value: EV,
        merge_edge_value: FMergeEV,
    ) -> EdgeInserter<EK, EV, T, FMergeEV>
    where
         FMergeEV: FnMut(&mut EV, EV),
    {
        EdgeInserter::new(Arc::new(RwLock::new(self.clone())), edge_key, edge_value, merge_edge_value)
    }
}

pub fn try_destination<EK, EV, T, FMergeEV>(
    source: Arc<RwLock<Trie<EK, EV, T>>>,
    edge_key: EK,
    edge_value: EV,
    destination: Arc<RwLock<Trie<EK, EV, T>>>,
    merge_edge_value: FMergeEV,
) -> Option<Arc<RwLock<Trie<EK, EV, T>>>>
where
    EK: Ord + Clone + Debug,
    EV: Clone,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
{
    EdgeInserter::new(source, edge_key, edge_value, merge_edge_value)
        .try_destination(destination)
        .into_option()
}

pub fn try_destination_with<EK, EV, T, FMergeEV>(
    source: Arc<RwLock<Trie<EK, EV, T>>>,
    edge_key: EK,
    edge_value: EV,
    destinations: &[Arc<RwLock<Trie<EK, EV, T>>>],
    merge_edge_value: FMergeEV,
) -> Option<Arc<RwLock<Trie<EK, EV, T>>>>
where
    EK: Ord + Clone + Debug,
    EV: Clone,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
{
    EdgeInserter::new(source, edge_key, edge_value, merge_edge_value)
        .try_destinations(destinations)
        .into_option()
}

pub fn try_destination_auto<EK, EV, T, FMergeEV>(
    source: Arc<RwLock<Trie<EK, EV, T>>>,
    edge_key: EK,
    edge_value: EV,
    destination: Arc<RwLock<Trie<EK, EV, T>>>,
    merge_edge_value: FMergeEV,
) -> Option<Arc<RwLock<Trie<EK, EV, T>>>>
where
    EK: Ord + Clone + Debug,
    EV: Clone,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
{
    EdgeInserter::new(source, edge_key, edge_value, merge_edge_value).try_destination_auto(destination).into_option()
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::collections::{HashSet, HashMap};
    use crate::datastructures::hybrid_bitset::HybridBitset;
    use std::iter::FromIterator;

    type TestTrieMerge = Trie<&'static str, Vec<i32>, String>;
    type TestNodeMerge = Arc<RwLock<TestTrieMerge>>;
    type TestTrieBasic = Trie<&'static str, &'static str, i32>;
    type TestNodeBasic = Arc<RwLock<TestTrieBasic>>;

    type TestTrieEI = Trie<&'static str, HybridBitset, String>;
    type TestNodeEI = Arc<RwLock<TestTrieEI>>;

    fn arc_ptr<N>(arc: &Arc<RwLock<N>>) -> *const RwLock<N> {
        Arc::as_ptr(arc)
    }

    #[test]
    fn test_try_insertion_and_retrieval() {
        let root_node: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let child3: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));

        {
            let mut root = root_node.write().unwrap();
            root.try_insert(&root_node, "a", &mut Some("edge_a1"), child1.clone()).expect("Insert failed");
            root.try_insert(&root_node, "b", &mut Some("edge_b"), child2.clone()).expect("Insert failed");
            root.try_insert(&root_node, "a", &mut Some("edge_a3"), child3.clone()).expect("Insert failed");
        }

        let root = root_node.read().unwrap();

        let retrieved_children_a = root.get(&"a").expect("Failed to get children for 'a'");
        assert_eq!(retrieved_children_a.len(), 2);
        let retrieved_data_a: HashSet<(&str, *const RwLock<TestTrieBasic>)> = retrieved_children_a
            .iter()
            .map(|(wrapper_arc, ev_ref)| (*ev_ref, arc_ptr(wrapper_arc.as_arc())))
            .collect();
        assert!(retrieved_data_a.contains(&("edge_a1", arc_ptr(&child1))));
        assert!(retrieved_data_a.contains(&("edge_a3", arc_ptr(&child3))));

        let retrieved_children_b = root.children().get(&"b").expect("Failed to get child 'b'");
        assert_eq!(retrieved_children_b.len(), 1);
        let (wrapper_arc, ev_ref) = retrieved_children_b.iter().next().unwrap();
        assert_eq!(*ev_ref, "edge_b");
        assert!(Arc::ptr_eq(wrapper_arc.as_arc(), &child2));

        assert!(root.get(&"c").is_none());

        let children_keys: Vec<_> = root.children().keys().cloned().collect();
        assert_eq!(children_keys, vec!["a", "b"]);
        assert_eq!(root.children().get("a").unwrap().len(), 2);
        assert_eq!(root.children().get("b").unwrap().len(), 1);

        assert!(!root.is_leaf());
        drop(root);
        assert!(child1.read().unwrap().is_leaf());
        assert!(child2.read().unwrap().is_leaf());
        assert!(child3.read().unwrap().is_leaf());
    }

    #[test]
    fn test_multiple_children_same_edge_key() {
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));

        {
            let mut r = root.write().unwrap();
            r.try_insert(&root, "edge", &mut Some("val1"), child1.clone()).unwrap();
            r.try_insert(&root, "edge", &mut Some("val2"), child2.clone()).unwrap();
        }

        {
            let binding = root.read().unwrap();
            let children_map = binding.get(&"edge").unwrap();
            assert_eq!(children_map.len(), 2);
            let child_data: HashSet<(&str, *const RwLock<TestTrieBasic>)> = children_map
                .iter()
                .map(|(wrapper_arc, ev_ref)| (*ev_ref, arc_ptr(wrapper_arc.as_arc())))
                .collect();
            assert!(child_data.contains(&("val1", arc_ptr(&child1))));
            assert!(child_data.contains(&("val2", arc_ptr(&child2))));
        }

        let all = Trie::all_nodes(root.clone());
        assert_eq!(all.len(), 3);
        let all_ptrs: HashSet<_> = all.iter().map(arc_ptr).collect();
        assert!(all_ptrs.contains(&arc_ptr(&root)));
        assert!(all_ptrs.contains(&arc_ptr(&child1)));
        assert!(all_ptrs.contains(&arc_ptr(&child2)));

        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();
        let mut edge_info_at_step = Vec::new();

        Trie::special_map(
            vec![(root.clone(), 100)],
            |parent_val, ek, ev, _child_node| {
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
        let depth1_nodes: HashSet<_> = processed_node_values[1..].iter().cloned().collect();
        assert!(depth1_nodes.contains(&1));
        assert!(depth1_nodes.contains(&2));

        assert_eq!(computed_values.len(), 3);
        assert_eq!(computed_values[0], 100);
        let results_map: HashMap<i32, i32> = processed_node_values.iter().cloned().zip(computed_values.iter().cloned()).collect();
        assert_eq!(results_map.get(&0), Some(&100));
        assert_eq!(results_map.get(&1), Some(&101));
        assert_eq!(results_map.get(&2), Some(&101));

        assert_eq!(edge_info_at_step.len(), 2);
        assert!(edge_info_at_step.contains(&("edge", "val1")));
        assert!(edge_info_at_step.contains(&("edge", "val2")));
    }

    #[test]
    fn test_special_map_bfs_order_with_edges() {
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));

        {
            let mut r = root.write().unwrap();
            r.try_insert(&root, "r->c1", &mut Some("e1"), child1.clone()).unwrap();
            r.try_insert(&root, "r->c2", &mut Some("e2"), child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.write().unwrap();
            c1.try_insert(&child1, "c1->gc", &mut Some("e3"), grandchild.clone()).unwrap();
        }

        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();
        let mut edge_info_at_step = Vec::new();

        Trie::special_map(
            vec![(root.clone(), 100)],
            |parent_val, ek, ev, _child_node| {
                edge_info_at_step.push((ek.clone(), ev.clone()));
                Some(parent_val + 1)
            },
            |current, new| { *current = new; },
            |node, computed_val| {
                processed_node_values.push(node.value);
                computed_values.push(*computed_val);
                true
            },
        );

        assert_eq!(processed_node_values.len(), 4);
        assert_eq!(processed_node_values[0], 0);
        let depth1_nodes: HashSet<_> = processed_node_values[1..3].iter().cloned().collect();
        assert!(depth1_nodes.contains(&1));
        assert!(depth1_nodes.contains(&2));
        assert_eq!(processed_node_values[3], 3);

        let results_map: HashMap<i32, i32> = processed_node_values.iter().cloned()
            .zip(computed_values.iter().cloned()).collect();
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
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));

        {
            let mut r = root.write().unwrap();
            r.try_insert(&root, "r1", &mut Some("e1"), child1.clone()).unwrap();
            r.try_insert(&root, "r2", &mut Some("e2"), child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.write().unwrap();
            c1.try_insert(&child1, "c1", &mut Some("e3"), grandchild.clone()).unwrap();
        }
        {
            let mut c2 = child2.write().unwrap();
            c2.try_insert(&child2, "c2", &mut Some("e4"), grandchild.clone()).unwrap();
        }

        let all_nodes = Trie::all_nodes(root.clone());

        assert_eq!(all_nodes.len(), 4);
        let node_ptrs: HashSet<_> = all_nodes.iter().map(arc_ptr).collect();
        assert_eq!(node_ptrs.len(), 4);
        assert!(node_ptrs.contains(&arc_ptr(&root)));
        assert!(node_ptrs.contains(&arc_ptr(&child1)));
        assert!(node_ptrs.contains(&arc_ptr(&child2)));
        assert!(node_ptrs.contains(&arc_ptr(&grandchild)));
    }

    #[test]
    fn test_special_map_diamond_merge_max() {
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));

        {
            let mut r = root.write().unwrap();
            r.try_insert(&root, "r->c1", &mut Some("edge1"), child1.clone()).unwrap();
            r.try_insert(&root, "r->c2", &mut Some("edge2"), child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.write().unwrap();
            c1.try_insert(&child1, "c1->gc", &mut Some("edge3"), grandchild.clone()).unwrap();
        }
        {
            let mut c2 = child2.write().unwrap();
            c2.try_insert(&child2, "c2->gc", &mut Some("edge4"), grandchild.clone()).unwrap();
        }

        assert_eq!(root.read().unwrap().max_depth, 0);
        assert_eq!(child1.read().unwrap().max_depth, 1);
        assert_eq!(child2.read().unwrap().max_depth, 1);
        assert_eq!(grandchild.read().unwrap().max_depth, 2);

        let processed_nodes = Arc::new(RwLock::new(HashMap::<i32, i32>::new()));
        let process_count = Arc::new(AtomicUsize::new(0));

        Trie::special_map(
            vec![(root.clone(), 100)],
            |p_val, _ek, _ev, _child_node| Some(p_val + 1),
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
            }
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
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(42)));
        let nodes = Trie::all_nodes(root.clone());
        assert_eq!(nodes.len(), 1);
        assert!(Arc::ptr_eq(&nodes[0], &root));
        assert!(root.read().unwrap().is_leaf());

        let mut processed = false;
        Trie::special_map(
            vec![(root.clone(), 100)],
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
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));

        let insert1_result = {
            let mut r = root.write().unwrap();
            r.try_insert(&root, "r->c", &mut Some("e1"), child.clone())
        };
        assert!(insert1_result.is_ok());
        assert_eq!(child.read().unwrap().max_depth, 1);
        assert_eq!(root.read().unwrap().max_depth, 0);

        let insert2_result = {
            let mut c = child.write().unwrap();
            c.try_insert(&child, "c->r", &mut Some("e2"), root.clone())
        };

        assert!(insert2_result.is_err());
        assert_eq!(insert2_result.err(), Some(CycleDetectedError));

        let child_locked = child.read().unwrap();
        let has_edge_to_root = if let Some(dest_map) = child_locked.children.get("c->r") {
             let lookup_key = ArcPtrWrapper::new(root.clone());
             dest_map.contains_key(&lookup_key)
         } else {
             false
         };
        assert!(!has_edge_to_root);

        assert_eq!(root.read().unwrap().max_depth, 0);
        assert_eq!(child_locked.max_depth, 1);

        println!("Done testing cycle detection on try_insert");
    }

    #[test]
    fn test_cycle_all_nodes_no_panic() {
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));

        root.write().unwrap().force_insert_to_node("r->c", "e1", &child);
        child.write().unwrap().force_insert_to_node("c->r", "e2", &root);
        root.write().unwrap().max_depth = 0;
        child.write().unwrap().max_depth = 1;

        let all_nodes = Trie::all_nodes(root.clone());

        assert_eq!(all_nodes.len(), 2);
        let node_ptrs: HashSet<_> = all_nodes.iter().map(arc_ptr).collect();
        assert_eq!(node_ptrs.len(), 2);
        assert!(node_ptrs.contains(&arc_ptr(&root)));
        assert!(node_ptrs.contains(&arc_ptr(&child)));
    }

    #[test]
    fn test_has_any_cycle() {
        let root1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));
        root1.write().unwrap().force_insert_to_node("a", "e1", &child1);
        root1.write().unwrap().force_insert_to_node("b", "e2", &child2);
        child1.write().unwrap().force_insert_to_node("c", "e3", &grandchild);
        child2.write().unwrap().force_insert_to_node("d", "e4", &grandchild);
        assert!(!Trie::has_any_cycle(root1.clone()));

        let root2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(10)));
        let child3: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(11)));
        root2.write().unwrap().force_insert_to_node("x", "e5", &child3);
        child3.write().unwrap().force_insert_to_node("y", "e6", &root2);
        assert!(Trie::has_any_cycle(root2.clone()));

        let root3: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(20)));
        let node_a: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(21)));
        let node_b: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(22)));
        let node_c: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(23)));
        root3.write().unwrap().force_insert_to_node("r->a", "e7", &node_a);
        node_a.write().unwrap().force_insert_to_node("a->b", "e8", &node_b);
        node_b.write().unwrap().force_insert_to_node("b->c", "e9", &node_c);
        node_c.write().unwrap().force_insert_to_node("c->a", "e10", &node_a);
        assert!(Trie::has_any_cycle(root3.clone()));

        let root4: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(30)));
        let node_a2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(31)));
        let node_b2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(32)));
        let node_c2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(33)));
        root4.write().unwrap().force_insert_to_node("r->a", "e11", &node_a2);
        node_a2.write().unwrap().force_insert_to_node("a->b", "e12", &node_b2);
        node_b2.write().unwrap().force_insert_to_node("b->a", "e13", &node_a2);
        assert!(Trie::has_any_cycle(root4.clone()));

        let root5: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(40)));
        let node_d: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(41)));
        root5.write().unwrap().force_insert_to_node("r->d", "e14", &node_d);
        let root6_in_cycle: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(50)));
        let node_e: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(51)));
        root6_in_cycle.write().unwrap().force_insert_to_node("c1->e", "e15", &node_e);
        node_e.write().unwrap().force_insert_to_node("e->c1", "e16", &root6_in_cycle);
        assert!(!Trie::has_any_cycle(root5.clone()));
        assert!(Trie::has_any_cycle(root6_in_cycle.clone()));
    }

    #[test]
    fn test_cycle_special_map_no_panic_limited_processing() {
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));

        root.write().unwrap().force_insert_to_node("r->c", "e1", &child);
        child.write().unwrap().force_insert_to_node("c->r", "e2", &root);
        root.write().unwrap().max_depth = 0;
        child.write().unwrap().max_depth = 1;

        let mut processed_vals = Vec::new();
        let mut computed_vals = Vec::new();

        Trie::special_map(
            vec![(root.clone(), 100)],
            |p, _ek, _ev, _n| Some(p + 1),
            |cur, new| *cur = (*cur).max(new),
            |node, v| {
                processed_vals.push(node.value);
                computed_vals.push(*v);
                true
            },
        );

        assert_eq!(processed_vals.len(), 2);
        assert!(processed_vals.contains(&0));
        assert!(processed_vals.contains(&1));

        let results_map: HashMap<i32, i32> = processed_vals.iter().cloned()
            .zip(computed_vals.iter().cloned()).collect();
        assert_eq!(results_map.get(&0), Some(&100));
        assert_eq!(results_map.get(&1), Some(&101));
    }

    #[test]
    fn test_special_map_stop_processing() {
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));
        let grandchild2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(4)));

        {
            let mut r = root.write().unwrap();
            r.try_insert(&root, "r->c1", &mut Some("edge1"), child1.clone()).unwrap();
            r.try_insert(&root, "r->c2", &mut Some("edge2"), child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.write().unwrap();
            c1.try_insert(&child1, "c1->gc", &mut Some("edge3"), grandchild1.clone()).unwrap();
        }
        {
            let mut c2 = child2.write().unwrap();
            c2.try_insert(&child2, "c2->gc", &mut Some("edge4"), grandchild2.clone()).unwrap();
        }

        let processed_nodes = Arc::new(RwLock::new(HashSet::<i32>::new()));
        let computed_values = Arc::new(RwLock::new(HashMap::<i32, i32>::new()));

        Trie::special_map(
            vec![(root.clone(), 100)],
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
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));

        {
            let mut r = root.write().unwrap();
            r.try_insert(&root, "keep", &mut Some("e1"), child1.clone()).unwrap();
            r.try_insert(&root, "skip", &mut Some("e2"), child2.clone()).unwrap();
        }
        {
            let mut c2 = child2.write().unwrap();
            c2.try_insert(&child2, "keep", &mut Some("e3"), grandchild2.clone()).unwrap();
        }

        let processed_nodes = Arc::new(RwLock::new(HashSet::<i32>::new()));
        let computed_values = Arc::new(RwLock::new(HashMap::<i32, i32>::new()));

        Trie::special_map(
            vec![(root.clone(), 100)],
            |p_val, ek, _ev, _child_node| {
                if *ek == "keep" {
                    Some(p_val + 1)
                } else {
                    None
                }
            },
            |current_v, new_v| *current_v = new_v,
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

        assert_eq!(final_processed.clone(), vec![0, 1].into_iter().collect());

        assert_eq!(final_values.get(&0), Some(&100));
        assert_eq!(final_values.get(&1), Some(&101));
        assert_eq!(final_values.get(&2), None);
        assert_eq!(final_values.get(&3), None);
    }

    // merge helpers
    fn merge_ev_append(existing_ev: &mut Vec<i32>, new_ev: Vec<i32>) {
        existing_ev.extend(new_ev.iter().copied());
    }

    fn merge_nv_append_if_flag(existing_nv: &String, new_nv: String) -> Option<String> {
        if existing_nv.contains("mergeable") && !existing_nv.contains("not_mergeable") {
            Some(format!("{}|{}", existing_nv, new_nv))
        } else {
            None
        }
    }

    fn merge_bitset_union(existing: &mut HybridBitset, new: HybridBitset) {
        *existing |= new
    }

    #[test]
    fn test_ei_try_destination_success_new_edge() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest".to_string())));
        let edge_val: HybridBitset = vec![1].into_iter().collect();

        let inserter = EdgeInserter::new(source.clone(), "key", edge_val.clone(), merge_bitset_union);
        let result_node = inserter.try_destination(dest.clone()).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest));
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (wrapper_arc, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, edge_val);
        assert!(Arc::ptr_eq(wrapper_arc.as_arc(), &dest));
        assert_eq!(dest.read().unwrap().max_depth, 1);
    }

    #[test]
    fn test_ei_try_destination_success_merge_ev() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest".to_string())));
        let initial_edge_val: HybridBitset = vec![10].into_iter().collect();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();
        let merged_edge_val: HybridBitset = vec![1, 10].into_iter().collect();

        source.write().unwrap().try_insert(&source, "key", &mut Some(initial_edge_val), dest.clone()).unwrap();
        assert_eq!(dest.read().unwrap().max_depth, 1);

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_node = inserter.try_destination(dest.clone()).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest));
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (wrapper_arc, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, merged_edge_val);
        assert!(Arc::ptr_eq(wrapper_arc.as_arc(), &dest));
        assert_eq!(dest.read().unwrap().max_depth, 1);
    }

    #[test]
    fn test_ei_try_destination_fail_merge_ev() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest".to_string())));
        let initial_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        source.write().unwrap().try_insert(&source, "key", &mut Some(initial_edge_val), dest.clone()).unwrap();

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_opt = inserter.try_destination(dest.clone()).into_option();

        assert!(result_opt.is_some());
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (wrapper_arc, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, new_edge_val);
        assert!(Arc::ptr_eq(wrapper_arc.as_arc(), &dest));
    }

    #[test]
    fn test_ei_try_destination_fail_cycle() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest".to_string())));
        let dummy_edge_val = HybridBitset::zeros();

        dest.write().unwrap().force_insert_to_node("dest_to_src", dummy_edge_val.clone(), &source);

        let new_edge_val: HybridBitset = vec![1].into_iter().collect();
        let inserter = EdgeInserter::new(source.clone(), "src_to_dest", new_edge_val.clone(), merge_bitset_union);
        let result_opt = inserter.try_destination(dest.clone()).into_option();

        assert!(result_opt.is_none());
    }

    #[test]
    fn test_ei_try_slice_success() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dest2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest2".to_string())));
        let dest3: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest3".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        dest2.write().unwrap().force_insert_to_node("d2->s", dummy_edge_val.clone(), &source);

        let destinations = [dest1.clone(), dest2.clone(), dest3.clone()];

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_node = inserter.try_destinations(&destinations).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest1));
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (wrapper_arc, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(wrapper_arc.as_arc(), &dest1));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_try_slice_success_later() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dest2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest2".to_string())));
        let dest3: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest3".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let destinations = [dest1.clone(), dest2.clone(), dest3.clone()];

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_node = inserter.try_destinations(&destinations).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest2));
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (wrapper_arc, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(wrapper_arc.as_arc(), &dest2));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_try_slice_fail_all() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dest2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest2".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);
        dest2.write().unwrap().force_insert_to_node("d2->s", dummy_edge_val.clone(), &source);

        let destinations = [dest1.clone(), dest2.clone()];

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_opt = inserter.try_destinations(&destinations).into_option();

        assert!(result_opt.is_none());
        assert!(source.read().unwrap().get(&"key").is_none());
    }

    #[test]
    fn test_ei_try_children_success_merge() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let child1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child1".to_string())));
        let child2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child2".to_string())));
        let child_other_key: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child_other_key".to_string())));

        let edge_key = "target_key";
        let initial_ev_c1: HybridBitset = vec![10].into_iter().collect();
        let initial_ev_c2: HybridBitset = vec![20].into_iter().collect();
        let new_ev_for_inserter: HybridBitset = vec![1].into_iter().collect();
        let merged_ev_c1: HybridBitset = vec![1, 10].into_iter().collect();

        {
            let mut s = source.write().unwrap();
            s.try_insert(&source, edge_key, &mut Some(initial_ev_c1), child1.clone()).unwrap();
            s.try_insert(&source, edge_key, &mut Some(initial_ev_c2.clone()), child2.clone()).unwrap();
            s.try_insert(&source, "other_key", &mut Some(HybridBitset::zeros()), child_other_key.clone()).unwrap();
        }

        let inserter = EdgeInserter::new(source.clone(), edge_key, new_ev_for_inserter.clone(), merge_bitset_union);
        let result_node_opt = inserter.try_children().into_option();

        assert!(result_node_opt.is_some(), "Should find and merge with child1");
        let result_node = result_node_opt.unwrap();
        assert!(Arc::ptr_eq(&result_node, &child1), "Result should be child1");

        {
            let s_guard = source.read().unwrap();
            let children_map_target_key = s_guard.get(&edge_key).expect("Target key should exist");

            let ev_c1 = children_map_target_key.get(&ArcPtrWrapper::new(child1.clone())).expect("Child1 should be under target_key");
            assert_eq!(*ev_c1, merged_ev_c1, "Edge value for child1 should be merged");

            let ev_c2 = children_map_target_key.get(&ArcPtrWrapper::new(child2.clone())).expect("Child2 should be under target_key");
            assert_eq!(*ev_c2, initial_ev_c2, "Edge value for child2 should be unchanged");

            let children_map_other_key = s_guard.get(&"other_key").expect("Other key should exist");
            assert_eq!(children_map_other_key.len(), 1, "Should be one child under other_key");
        }

        let source_nm: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_nm".to_string())));
        let child1_nm: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child1_nm".to_string())));
        let edge_key_nm = "nm_key";
        let initial_ev_nm: HybridBitset = vec![50].into_iter().collect();
        let new_ev_inserter_nm: HybridBitset = vec![5].into_iter().collect();

        source_nm.write().unwrap().try_insert(&source_nm, edge_key_nm, &mut Some(initial_ev_nm.clone()), child1_nm.clone()).unwrap();

        let source_empty: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_empty".to_string())));
        let edge_key_empty = "empty_key";
        let new_ev_inserter_empty: HybridBitset = vec![7].into_iter().collect();

        let inserter_empty = EdgeInserter::new(source_empty.clone(), edge_key_empty, new_ev_inserter_empty.clone(), merge_bitset_union);
        let result_node_empty_opt = inserter_empty.try_children().into_option();
        assert!(result_node_empty_opt.is_none(), "try_children should return None if no children under the key");

        let source_chain: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_chain".to_string())));
        let edge_key_chain = "chain_key";
        let new_ev_chain: HybridBitset = vec![8].into_iter().collect();
        let created_val = "created_node_via_fallback".to_string();

        let inserter_chain = EdgeInserter::new(source_chain.clone(), edge_key_chain, new_ev_chain.clone(), merge_bitset_union);
        let result_node_chain = inserter_chain
            .try_children()
            .else_create_destination_with_value(created_val.clone())
            .unwrap();

        assert_eq!(result_node_chain.read().unwrap().value, created_val, "Fallback node should be created with correct value");
        let s_chain_guard = source_chain.read().unwrap();
        let children_map_chain = s_chain_guard.get(&edge_key_chain).expect("Chain key should now exist in source_chain");
        assert_eq!(children_map_chain.len(), 1, "One edge should be created under chain_key");
        let (wrapper_chain, ev_chain) = children_map_chain.iter().next().unwrap();
        assert!(Arc::ptr_eq(wrapper_chain.as_arc(), &result_node_chain), "Edge should point to the newly created node");
        assert_eq!(*ev_chain, new_ev_chain, "Edge should have the new_ev_chain value");
    }

    #[test]
    fn test_ei_else_create_with_value() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_node = inserter.else_create_destination_with_value("created".to_string()).unwrap();

        assert_eq!(result_node.read().unwrap().value, "created");
        assert_eq!(result_node.read().unwrap().max_depth, 1);
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(ca.as_arc(), &result_node));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_else_create_with() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let created_flag = Arc::new(AtomicUsize::new(0));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let flag_clone = created_flag.clone();
        let result_node = inserter.else_create_destination_with(|| {
            flag_clone.fetch_add(1, Ordering::SeqCst);
            "created_via_fn".to_string()
        }).unwrap();

        assert_eq!(created_flag.load(Ordering::SeqCst), 1);
        assert_eq!(result_node.read().unwrap().value, "created_via_fn");
        assert_eq!(result_node.read().unwrap().max_depth, 1);
    }

    #[test]
    fn test_ei_else_create_default() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_node = inserter.else_create_destination().unwrap();

        assert_eq!(result_node.read().unwrap().value, "");
        assert_eq!(result_node.read().unwrap().max_depth, 1);
    }

    #[test]
    fn test_ei_chaining_try_then_else() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_node = inserter
            .try_destination(dest1.clone())
            .else_create_destination_with_value("fallback".to_string())
            .unwrap();

        assert_eq!(result_node.read().unwrap().value, "fallback");
        assert!(!Arc::ptr_eq(&result_node, &dest1));
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(ca.as_arc(), &result_node));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_chaining_try_success_skips_else() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_node = inserter
            .try_destination(dest1.clone())
            .else_create_destination_with_value("fallback".to_string())
            .unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest1));
        assert_eq!(result_node.read().unwrap().value, "dest1");
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(ca.as_arc(), &dest1));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    #[should_panic(expected = "EdgeInserter::unwrap() called but no destination was found or created")]
    fn test_ei_unwrap_panic() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        inserter.try_destination(dest1.clone()).unwrap();
    }

    #[test]
    fn test_ei_get() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);

        let inserter_after_try = inserter.try_destination(dest1.clone());
        assert!(inserter_after_try.clone_into_option().is_none());

        let inserter_after_else = inserter_after_try.else_create_destination_with_value("fallback".to_string());
        let result_opt = inserter_after_else.into_option();
        assert!(result_opt.is_some());
        assert_eq!(result_opt.unwrap().read().unwrap().value, "fallback");
    }

    #[test]
    fn test_ei_chaining_stops_after_success() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let child1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child1".to_string())));
        let child2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child2".to_string())));
        let new_node_val_if_created = "new_node_val".to_string();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        let destinations_for_slice = vec![child2.clone()];

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union);
        let result_node = inserter
            .try_destination(child1.clone())
            .try_destinations(&destinations_for_slice)
            .else_create_destination_with_value(new_node_val_if_created.clone())
            .unwrap();

        assert!(Arc::ptr_eq(&result_node, &child1), "Chain should stop after first success (try_insert)");

        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (ca, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(ca.as_arc(), &child1));
        assert_eq!(*ev, new_edge_val);

        assert_ne!(result_node.read().unwrap().value, new_node_val_if_created);
    }

    #[test]
    fn test_ei_try_children_new_logic() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let child1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child1".to_string())));
        let child2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child2".to_string())));
        let child_other_key: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child_other_key".to_string())));

        let edge_key = "target_key";
        let initial_ev_c1: HybridBitset = vec![10].into_iter().collect();
        let initial_ev_c2: HybridBitset = vec![20].into_iter().collect();
        let new_ev_for_inserter: HybridBitset = vec![1].into_iter().collect();
        let merged_ev_c1: HybridBitset = vec![1, 10].into_iter().collect();

        {
            let mut s = source.write().unwrap();
            s.try_insert(&source, edge_key, &mut Some(initial_ev_c1), child1.clone()).unwrap();
            s.try_insert(&source, edge_key, &mut Some(initial_ev_c2.clone()), child2.clone()).unwrap();
            s.try_insert(&source, "other_key", &mut Some(HybridBitset::zeros()), child_other_key.clone()).unwrap();
        }

        let inserter = EdgeInserter::new(source.clone(), edge_key, new_ev_for_inserter.clone(), merge_bitset_union);
        let result_node_opt = inserter.try_children().into_option();

        assert!(result_node_opt.is_some(), "Should find and merge with child1");
        let result_node = result_node_opt.unwrap();
        assert!(Arc::ptr_eq(&result_node, &child1), "Result should be child1, got {:?} and {:?}", result_node, child1);

        {
            let s_guard = source.read().unwrap();
            let children_map_target_key = s_guard.get(&edge_key).expect("Target key should exist");

            let ev_c1 = children_map_target_key.get(&ArcPtrWrapper::new(child1.clone())).expect("Child1 should be under target_key");
            assert_eq!(*ev_c1, merged_ev_c1, "Edge value for child1 should be merged");

            let ev_c2 = children_map_target_key.get(&ArcPtrWrapper::new(child2.clone())).expect("Child2 should be under target_key");
            assert_eq!(*ev_c2, initial_ev_c2, "Edge value for child2 should be unchanged");

            let children_map_other_key = s_guard.get(&"other_key").expect("Other key should exist");
            assert_eq!(children_map_other_key.len(), 1, "Should be one child under other_key");
        }

        let source_nm: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_nm".to_string())));
        let child1_nm: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child1_nm".to_string())));
        let edge_key_nm = "nm_key";
        let initial_ev_nm: HybridBitset = vec![50].into_iter().collect();
        let new_ev_inserter_nm: HybridBitset = vec![5].into_iter().collect();

        source_nm.write().unwrap().try_insert(&source_nm, edge_key_nm, &mut Some(initial_ev_nm.clone()), child1_nm.clone()).unwrap();

        let source_empty: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_empty".to_string())));
        let edge_key_empty = "empty_key";
        let new_ev_inserter_empty: HybridBitset = vec![7].into_iter().collect();

        let inserter_empty = EdgeInserter::new(source_empty.clone(), edge_key_empty, new_ev_inserter_empty.clone(), merge_bitset_union);
        let result_node_empty_opt = inserter_empty.try_children().into_option();
        assert!(result_node_empty_opt.is_none(), "try_children should return None if no children under the key");

        let source_chain: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_chain".to_string())));
        let edge_key_chain = "chain_key";
        let new_ev_chain: HybridBitset = vec![8].into_iter().collect();
        let created_val = "created_node_via_fallback".to_string();

        let inserter_chain = EdgeInserter::new(source_chain.clone(), edge_key_chain, new_ev_chain.clone(), merge_bitset_union);
        let result_node_chain = inserter_chain
            .try_children()
            .else_create_destination_with_value(created_val.clone())
            .unwrap();

        assert_eq!(result_node_chain.read().unwrap().value, created_val, "Fallback node should be created with correct value");
        let s_chain_guard = source_chain.read().unwrap();
        let children_map_chain = s_chain_guard.get(&edge_key_chain).expect("Chain key should now exist in source_chain");
        assert_eq!(children_map_chain.len(), 1, "One edge should be created under chain_key");
        let (wrapper_chain, ev_chain) = children_map_chain.iter().next().unwrap();
        assert!(Arc::ptr_eq(wrapper_chain.as_arc(), &result_node_chain), "Edge should point to the newly created node");
        assert_eq!(*ev_chain, new_ev_chain, "Edge should have the new_ev_chain value");
    }
}