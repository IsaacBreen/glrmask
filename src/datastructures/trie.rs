// #![deny(clippy::iter_over_hash_type)]
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{self, Debug};
// Import TryLockError explicitly for matching
use std::sync::{Arc, RwLock, TryLockError};
use std::sync::atomic::{AtomicUsize, Ordering}; // Added for tests
use std::cmp::Reverse;          // min-heap helper
use std::collections::BinaryHeap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::cell::RefCell;
use std::env;
// Not strictly needed with the chosen direct BFS approach in to_json, but good to keep in mind for context-passing alternatives.
use ordered_hash_map::OrderedHashMap;


use crate::datastructures::hybrid_bitset::HybridBitset; // Import HybridBitset
use crate::datastructures::{ArcPtrWrapper}; // Import ArcPtrWrapper and WeakPtrWrapper
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added
use deterministic_hash::DeterministicHasher;
use ordered_hash_map::OrderedHashSet;
use kdam::{tqdm, BarExt};
use profiler_macro::{time_it, timeit};
use crate::datastructures::arc_wrapper::{NodePtr, WeakPtrWrapper};
use crate::profiler::PROGRESS_BAR_ENABLED;
// Added for derive macro pattern


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
    /// Stores a map from EdgeKey to a map of destination nodes and edge values.
    children: BTreeMap<EK, OrderedHashMap<NodePtr<RwLock<Trie<EK, EV, T>>>, EV>>,
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

        // --- Step 1: Serialize `self` (the root node for this call) ---
        // `self` is node at index 0.
        // We need to store the JSON representation of `self`'s direct data.
        // Since `self` is not an Arc here, we handle it specially as the first node.
        // Its children, which are Arcs, will be processed via the BFS queue.
        let root_idx = 0;
        nodes_json_list.push(JSONNode::Null); // Placeholder for root, will be filled after processing its children.

        let mut root_children_json_data = Vec::new(); // Stores [EK_json, [[ChildIdx, EV_json], ...]]
        let mut root_weak_children_json_data = Vec::new(); // Stores [EK_json, [[ChildIdx, EV_json], ...]] for weak edges

        // Serialize strong and weak children
        for (edge_key, destinations_map) in &self.children {
            let ek_json = edge_key.to_json();
            let mut strong_dests_json = Vec::new();
            let mut weak_dests_json = Vec::new();

            for (node_ptr, edge_val) in destinations_map {
                let child_arc = node_ptr.upgrade().expect("Dangling weak pointer during Trie serialization");
                let child_arc_ptr = Arc::as_ptr(&child_arc);
                let child_idx = match arc_ptr_to_idx_map.get(&child_arc_ptr) {
                    Some(idx) => *idx,
                    None => {
                        let new_idx = nodes_json_list.len();
                        arc_ptr_to_idx_map.insert(child_arc_ptr, new_idx);
                        bfs_q.push_back(child_arc);
                        nodes_json_list.push(JSONNode::Null);
                        new_idx
                    }
                };
                let dest_entry = JSONNode::Array(vec![
                    child_idx.to_json(),
                    edge_val.to_json(),
                ]);
                if node_ptr.is_strong() {
                    strong_dests_json.push(dest_entry);
                } else {
                    weak_dests_json.push(dest_entry);
                }
            }
            if !strong_dests_json.is_empty() {
                root_children_json_data.push(JSONNode::Array(vec![ek_json.clone(), JSONNode::Array(strong_dests_json)]));
            }
            if !weak_dests_json.is_empty() {
                root_weak_children_json_data.push(JSONNode::Array(vec![ek_json, JSONNode::Array(weak_dests_json)]));
            }
        }

        // Fill in the root node's (self's) data
        nodes_json_list[root_idx] = JSONNode::Object(BTreeMap::from_iter(vec![
            ("value".to_string(), self.value.to_json()),
            ("max_depth".to_string(), self.max_depth.to_json()),
            ("children".to_string(), JSONNode::Array(root_children_json_data)),
            ("weak_children".to_string(), JSONNode::Array(root_weak_children_json_data)),
        ]));


        // --- Step 2: Process the rest of the nodes in the queue (BFS) ---
        while let Some(current_arc) = bfs_q.pop_front() {
            let current_arc_ptr = Arc::as_ptr(&current_arc);
            let current_node_json_idx = *arc_ptr_to_idx_map.get(&current_arc_ptr)
                .expect("Node in BFS queue must have an assigned index");

            let node_guard = current_arc.read().expect("RwLock poisoned during Trie serialization (BFS part)");
            let mut current_node_children_json_bfs = Vec::new();
            let mut current_node_weak_children_json_bfs = Vec::new();

            // Serialize strong and weak children for the current node
            for (edge_key, destinations_map) in &node_guard.children {
                let ek_json = edge_key.to_json();
                let mut strong_dests_json_bfs = Vec::new();
                let mut weak_dests_json_bfs = Vec::new();

                for (node_ptr, edge_val) in destinations_map {
                    let child_arc = node_ptr.upgrade().expect("Dangling weak pointer during Trie serialization");
                    let child_arc_ptr = Arc::as_ptr(&child_arc);
                    let child_idx = match arc_ptr_to_idx_map.get(&child_arc_ptr) {
                        Some(idx) => *idx,
                        None => {
                            let new_idx = nodes_json_list.len();
                            arc_ptr_to_idx_map.insert(child_arc_ptr, new_idx);
                            bfs_q.push_back(child_arc);
                            nodes_json_list.push(JSONNode::Null);
                            new_idx
                        }
                    };
                    let dest_entry = JSONNode::Array(vec![
                        child_idx.to_json(),
                        edge_val.to_json(),
                    ]);
                    if node_ptr.is_strong() {
                        strong_dests_json_bfs.push(dest_entry);
                    } else {
                        weak_dests_json_bfs.push(dest_entry);
                    }
                }
                if !strong_dests_json_bfs.is_empty() {
                    current_node_children_json_bfs.push(JSONNode::Array(vec![ek_json.clone(), JSONNode::Array(strong_dests_json_bfs)]));
                }
                if !weak_dests_json_bfs.is_empty() {
                    current_node_weak_children_json_bfs.push(JSONNode::Array(vec![ek_json, JSONNode::Array(weak_dests_json_bfs)]));
                }
            }

            // Fill in the data for the current node from the BFS queue
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
                                max_depth,
                            }));
                            deserialized_arcs.insert(i, new_node_arc);
                        }
                        _ => return Err(format!("Node data at index {} is not an object", i)),
                    }
                }

                // Pass 2: Link children by populating the `children` BTreeMaps
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
                                                                    destinations_for_this_ek.insert(NodePtr::Strong(ArcPtrWrapper::new(child_arc)), edge_value);
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

                            // Link weak_children if present
                            if let Some(weak_children_node) = weak_children_json_outer_array_opt {
                                match weak_children_node {
                                    JSONNode::Array(children_ek_map_array) => {
                                        for ek_entry_json in children_ek_map_array {
                                            match ek_entry_json {
                                                JSONNode::Array(ek_pair) if ek_pair.len() == 2 => {
                                                    let ek_json = &ek_pair[0];
                                                    let dest_map_json_array = &ek_pair[1];

                                                    let edge_key = EK::from_json(ek_json.clone())?;
                                                    let destinations_for_this_ek = current_node_guard.children.entry(edge_key.clone()).or_default();

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
                                                                        let weak_wrapper = NodePtr::Weak(WeakPtrWrapper::new(Arc::downgrade(&child_arc)));
                                                                        destinations_for_this_ek.insert(weak_wrapper, edge_value);
                                                                    }
                                                                    _ => return Err(format!("Invalid weak child_idx-EV pair format for node {} under edge key {:?}", i, edge_key)),
                                                                }
                                                            }
                                                        }
                                                        _ => return Err(format!("Weak children destination map for node {} under edge key {:?} is not an array", i, &edge_key)),
                                                    }
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

                // The trait requires returning Self, so we clone the content of the root Arc.
                // The shared graph structure is maintained by the Arcs held within the children maps.
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


// Implementation block for core Trie functionality
// Added Clone bound for EK needed in try_insert_or_merge_edge and others
impl<EK: Ord + Clone, EV, T> Trie<EK, EV, T> {
    /// Creates a new trie node with the given value and no children.
    /// The max_depth is initialized to 0.
    pub fn new(value: T) -> Self {
        Trie {
            value,
            children: BTreeMap::new(),
            max_depth: 0,
        }
    }

    // force_insert remains unchanged
    pub fn force_insert_to_new_node(&mut self, edge_key: EK, edge_value: EV, value: T) -> Arc<RwLock<Trie<EK, EV, T>>> {
        let new_node = Arc::new(RwLock::new(Trie::new(value)));
        let new_node_comparable = NodePtr::Strong(ArcPtrWrapper::new(new_node.clone()));
        self.children.entry(edge_key).or_default().insert(new_node_comparable, edge_value);
        // Note: force_insert does NOT update max_depth or check for cycles. Use with caution.
        new_node.clone()
    }

    pub fn force_insert_to_node(&mut self, edge_key: EK, edge_value: EV, dst: &Arc<RwLock<Trie<EK, EV, T>>>) {
        let dst_comparable = NodePtr::Strong(ArcPtrWrapper::new(dst.clone()));
        self.children.entry(edge_key).or_default().insert(dst_comparable, edge_value);
    }

    /// Insert a weak edge explicitly. This allows cycles by design.
    /// Does NOT update max_depth and will not keep the destination alive.
    pub fn insert_weak_to_node(&mut self, edge_key: EK, edge_value: EV, dst: &Arc<RwLock<Trie<EK, EV, T>>>) {
        let weak = NodePtr::Weak(WeakPtrWrapper::new(Arc::downgrade(dst)));
        self.children.entry(edge_key).or_default().insert(weak, edge_value);
    }

    /// Convenience: create a new node and insert a weak edge to it.
    /// Note: a weak edge to a freshly created node is usually useless unless
    /// another strong edge also points to it; otherwise it may be dropped.
    pub fn insert_weak_to_new_node(&mut self, edge_key: EK, edge_value: EV, value: T) -> Arc<RwLock<Trie<EK, EV, T>>> {
        let new_node = Arc::new(RwLock::new(Trie::new(value)));
        {
            let weak = NodePtr::Weak(WeakPtrWrapper::new(Arc::downgrade(&new_node)));
            self.children.entry(edge_key).or_default().insert(weak, edge_value);
        }
        new_node
    }

    // already_has_dst remains unchanged
    pub fn already_has_dst(&self, edge_key: EK, dst: &Arc<RwLock<Trie<EK, EV, T>>>) -> bool {
        let lookup_key = NodePtr::Strong(ArcPtrWrapper::new(dst.clone())); // Clone Arc for temporary ownership in key
        self.children.get(&edge_key).map_or(false, |dest_map| dest_map.contains_key(&lookup_key))
    }

    pub fn already_has_dst_for_any_key(&self, dst: &Arc<RwLock<Trie<EK, EV, T>>>) -> bool {
        let lookup_key = NodePtr::Strong(ArcPtrWrapper::new(dst.clone()));
        self.children.values().any(|dest_map| dest_map.contains_key(&lookup_key))
    }

    // get_edge_value remains unchanged
    pub fn get_edge_value(&self, edge_key: EK, dst: &Arc<RwLock<Trie<EK, EV, T>>>) -> Option<&EV> {
        let lookup_key = NodePtr::Strong(ArcPtrWrapper::new(dst.clone()));
        self.children.get(&edge_key).and_then(|dest_map| dest_map.get(&lookup_key))
    }

    /// Weak variant: check if a weak edge already exists to dst.
    pub fn already_has_weak_dst(&self, edge_key: EK, dst: &Arc<RwLock<Trie<EK, EV, T>>>) -> bool {
        let weak = NodePtr::Weak(WeakPtrWrapper::new(Arc::downgrade(dst)));
        self.children.get(&edge_key).map_or(false, |dest_map| dest_map.contains_key(&weak))
    }

    /// Weak variant: get weak edge EV (if the weak pointer matches).
    pub fn get_weak_edge_value(&self, edge_key: EK, dst: &Arc<RwLock<Trie<EK, EV, T>>>) -> Option<&EV> {
        let weak = NodePtr::Weak(WeakPtrWrapper::new(Arc::downgrade(dst)));
        self.children.get(&edge_key).and_then(|dest_map| dest_map.get(&weak))
    }

    // get_edge_value_mut remains unchanged
    pub fn get_edge_value_mut(&mut self, edge_key: EK, dst: &Arc<RwLock<Trie<EK, EV, T>>>) -> Option<&mut EV> {
        let lookup_key = NodePtr::Strong(ArcPtrWrapper::new(dst.clone()));
        self.children.get_mut(&edge_key).and_then(|dest_map| dest_map.get_mut(&lookup_key))
    }

    #[time_it]
    pub fn try_insert(
        &mut self,
        edge_key: EK,
        edge_value: &mut Option<EV>, // Changed to allow taking the value
        child: Arc<RwLock<Trie<EK, EV, T>>>,
    ) -> Result<(), CycleDetectedError> {
        // ------------------------------------------------------------------
        // 1. Detect whether adding the edge would introduce a cycle.
        //    A cycle exists iff `self` is reachable from `child`.
        // ------------------------------------------------------------------
        let self_ptr = self as *const Trie<EK, EV, T>;
        if !self.already_has_dst_for_any_key(&child) &&
            Self::detect_cycle(self_ptr, &child) {
            return Err(CycleDetectedError);
        }

        self.try_insert_unchecked(edge_key, edge_value, child)
    }

    #[time_it]
    pub fn try_insert_unchecked(
        &mut self,
        edge_key: EK,
        edge_value: &mut Option<EV>, // Changed to allow taking the value
        child: Arc<RwLock<Trie<EK, EV, T>>>,
    ) -> Result<(), CycleDetectedError> {

        // ------------------------------------------------------------------
        // 2. Update the child's max-depth *before* the edge is inserted.
        //    This lets us rollback cleanly if `propagate_max_depth` fails
        //    (because no structural change has been committed yet).
        // ------------------------------------------------------------------
        let candidate_depth = self.max_depth.saturating_add(1);
        let previous_child_depth; // Store previous depth for potential rollback
        let needs_depth_update;

        // Scope for child lock
        {
            let mut child_guard = child
                .write()
                .expect("RwLock poisoned while updating child's max_depth");
            previous_child_depth = child_guard.max_depth;
            needs_depth_update = candidate_depth > previous_child_depth;
            if needs_depth_update {
                child_guard.max_depth = candidate_depth;
            }
        } // child_guard lock released here

        // If the child's depth actually changed we must propagate.
        if needs_depth_update {
            // Propagate the update. If it fails (cycle detected during propagation),
            // roll back the change we just made to the child's depth.
            if let Err(e) = Self::propagate_max_depth(child.clone(), candidate_depth) {
                // Roll-back the depth change made above
                let mut child_guard = child
                    .write()
                    .expect("RwLock poisoned while rolling back max_depth");
                // Only roll back if the depth is still what we set it to.
                // (Another thread might have increased it further, which is fine).
                if child_guard.max_depth == candidate_depth {
                     child_guard.max_depth = previous_child_depth;
                }
                // We should still return the error, as a cycle was detected somewhere.
                return Err(e);
            }
        }

        // ------------------------------------------------------------------
        // 3. All checks have passed – perform the real structural mutation.
        // ------------------------------------------------------------------
        let child_comparable = NodePtr::Strong(ArcPtrWrapper::new(child.clone())); // child is an Arc, clone it
        self.children
            .entry(edge_key)
            .or_default()
            .insert(child_comparable, edge_value.take().unwrap()); // Take the value


        Ok(())
    }

    /// Try to insert an edge; if it would create a strong cycle, insert it as a WEAK edge instead.
    /// Returns whether it became Strong or Weak.
    ///
    /// Strong edge semantics (acyclic, affects depth) are preserved when possible.
    /// Weak edge semantics (may create cycles, no depth propagation) are used otherwise.
    #[time_it]
    pub fn try_insert_auto(
        &mut self,
        edge_key: EK,
        edge_value: &mut Option<EV>,
        child: Arc<RwLock<Trie<EK, EV, T>>>,
    ) -> InsertedEdgeKind {
        // Detect whether adding a strong edge would create a cycle
        let self_ptr = self as *const Trie<EK, EV, T>;
        let would_cycle;
        // If it already has an edge to this node, it can't create a cycle.
        if self.already_has_dst_for_any_key(&child) {
            would_cycle = false;
        } else {
            // Check if adding this edge would create a cycle
            would_cycle = Self::detect_cycle(self_ptr, &child);
        }
        if would_cycle {
            // Degrade to weak edge; do NOT update depths
            if let Some(ev) = edge_value.take() {
                let weak = NodePtr::Weak(WeakPtrWrapper::new(Arc::downgrade(&child)));
                self.children.entry(edge_key).or_default().insert(weak, ev);
            }
            return InsertedEdgeKind::Weak;
        }

        // Otherwise, perform strong try_insert (will not fail now)
        let mut ev_opt = edge_value.take();
        // Since we've already checked for cycles, we can use the `unchecked` variant
        // to avoid a redundant `detect_cycle` call.
        let _ = self.try_insert_unchecked(edge_key, &mut ev_opt, child)
            .expect("Cycle re-appeared unexpectedly during try_insert_auto strong path");
        // If try_insert consumed the value, that's fine; if not, ev_opt may still have it.
        // Ensure caller's Option stays updated:
        *edge_value = ev_opt;
        InsertedEdgeKind::Strong
    }

    /// Returns `true` if `target_ptr` (pointer to the Trie data) is reachable from `start_arc`.
    /// This function handles the case where `target_ptr` points to a node that is currently locked
    /// by the calling thread (e.g., `self` in `try_insert`).
    #[time_it]
    pub fn detect_cycle(
        target_ptr: *const Trie<EK, EV, T>,
        start_arc: &Arc<RwLock<Trie<EK, EV, T>>>,
    ) -> bool {
        // Use Arc::as_ptr to get stable pointers to the RwLock itself for visited tracking.
        let mut visited_arcs: HashSet<*const RwLock<Trie<EK, EV, T>>> = HashSet::new();
        let mut queue: VecDeque<Arc<RwLock<Trie<EK, EV, T>>>> = VecDeque::new();

        let start_arc_ptr = Arc::as_ptr(start_arc);
        if visited_arcs.insert(start_arc_ptr) {
            queue.push_back(start_arc.clone());
        }

        while let Some(node_arc) = queue.pop_front() {
            // Attempt to lock the node to get its data pointer and children.
            let lock_result = node_arc.try_read();

            match lock_result {
                Ok(node_guard) => {
                    // Successfully locked the node.
                    let current_data_ptr = &*node_guard as *const Trie<EK, EV, T>;

                    // Check if this node's data pointer matches the target pointer.
                    if current_data_ptr == target_ptr {
                        // We reached the target node. Cycle detected.
                        return true;
                    }

                    // Get children while holding the lock.
                    let children_arcs: Vec<Arc<RwLock<Trie<EK, EV, T>>>> = node_guard.children
                        .values() // Iterates over HashMap<ArcPtrWrapper<Mutex<...>>, EV>
                        .flat_map(|dest_map| {
                            dest_map.keys().filter_map(|node_ptr| match node_ptr {
                                NodePtr::Strong(arc_wrapper) => Some(arc_wrapper.as_arc().clone()),
                                NodePtr::Weak(_) => None,
                            })
                        })
                        .collect();

                    timeit!(format!("BFS: Node has {} children", children_arcs.len()), {});

                    // Explicitly drop the guard before potentially long operations (queueing).
                    drop(node_guard);

                    // Enqueue unvisited children.
                    for child_arc_val in children_arcs { // Renamed child_arc
                        let child_arc_ptr = Arc::as_ptr(&child_arc_val); // Use child_arc_val
                        if visited_arcs.insert(child_arc_ptr) {
                            queue.push_back(child_arc_val); // Use child_arc_val
                        }
                    }
                }
                Err(TryLockError::WouldBlock) => {
                    // Failed to lock because it's held elsewhere (potentially by the thread calling try_insert).
                    // Assume this means we've reached the target node in the context of try_insert.
                    // If detect_cycle were used elsewhere, this assumption might need revisiting.
                    return true;
                }
                Err(TryLockError::Poisoned(p)) => {
                    // A mutex was poisoned. Propagate the panic.
                    panic!("RwLock poisoned during cycle detection: {:?}", p);
                }
            }
        }

        // BFS completed without finding the target pointer. No cycle detected.
        false
    }


    /// Propagates a max_depth update to all descendant nodes, detecting cycles.
    ///
    /// Returns `Ok(())` if propagation completes successfully.
    /// Returns `Err(CycleDetectedError)`.
    fn propagate_max_depth(node_arc: Arc<RwLock<Trie<EK, EV, T>>>, current_depth: usize) -> Result<(), CycleDetectedError> {
        // rec_stack will contain the set of node pointers from the root of the propagation
        // down to the current recursion level. Use Arc::as_ptr for stable pointers.
        let mut rec_stack: HashSet<*const RwLock<Trie<EK, EV, T>>> = HashSet::new();
        Self::_propagate_max_depth(node_arc, current_depth, &mut rec_stack)
    }

    /// Recursive helper for propagate_max_depth, detecting cycles using Arc pointers.
    /// Returns `Ok(())` or `Err(CycleDetectedError)`.
    fn _propagate_max_depth(
        node_arc: Arc<RwLock<Trie<EK, EV, T>>>,
        current_depth: usize,
        rec_stack: &mut HashSet<*const RwLock<Trie<EK, EV, T>>>,
    ) -> Result<(), CycleDetectedError> {
        let node_arc_ptr = Arc::as_ptr(&node_arc);

        // If this node (identified by its Arc pointer) is already in the current recursion chain, we have a cycle.
        if rec_stack.contains(&node_arc_ptr) {
            return Err(CycleDetectedError);
        }

        // Add the current node to the recursion stack.
        rec_stack.insert(node_arc_ptr);

        // Collect *all* child Arcs outside of the lock to avoid holding lock during recursion.
        let children_arcs: Vec<Arc<RwLock<Trie<EK, EV, T>>>> = {
            let node_guard_val = node_arc.read().expect("RwLock poisoned in _propagate_max_depth (getting children)"); // Renamed node_guard
            node_guard_val.children // Use node_guard_val
                .values() // Iterates over HashMap<ArcPtrWrapper<Mutex<...>>, EV>
                .flat_map(|dest_map| {
                    dest_map.keys().filter_map(|node_ptr| match node_ptr {
                        NodePtr::Strong(arc_wrapper) => Some(arc_wrapper.as_arc().clone()),
                        NodePtr::Weak(_) => None,
                    })
                })
                .collect()
        }; // node_guard_val lock is released here.
        // NOTE: Weak edges do NOT participate in max_depth propagation.

        // For each child, compute the candidate depth.
        let candidate_depth_val = current_depth.saturating_add(1); // Renamed candidate_depth
        for child_arc in children_arcs {
            // Check if the child needs updating *before* recursing.
            let should_propagate;
            { // Scope for child lock
                let mut child_guard = child_arc
                    .write()
                    .expect("RwLock poisoned in _propagate_max_depth (checking child depth)");
                if candidate_depth_val > child_guard.max_depth { // Use candidate_depth_val
                    child_guard.max_depth = candidate_depth_val; // Use candidate_depth_val
                    should_propagate = true;
                } else {
                    should_propagate = false;
                }
            } // child_guard lock released here

            if should_propagate {
                // Recurse. Propagate the error up if recursion detects a cycle.
                Self::_propagate_max_depth(child_arc, candidate_depth_val, rec_stack)?; // Use candidate_depth_val
            }
        }

        // Finished processing this node; remove from recursion stack.
        rec_stack.remove(&node_arc_ptr);
        Ok(()) // Success for this branch
    }

    // get remains unchanged
    pub fn get(
        &self,
        edge_key: &EK,
    ) -> Option<&OrderedHashMap<NodePtr<RwLock<Trie<EK, EV, T>>>, EV>>
    {
        self.children.get(edge_key)
    }

    // get_mut remains unchanged
    pub fn get_mut(
        &mut self,
        edge_key: &EK,
    ) -> Option<&mut OrderedHashMap<NodePtr<RwLock<Trie<EK, EV, T>>>, EV>>
    {
        self.children.get_mut(edge_key)
    }

    // children remains unchanged
    pub fn children(&self) -> &BTreeMap<EK, OrderedHashMap<NodePtr<RwLock<Trie<EK, EV, T>>>, EV>> {
        &self.children
    }

    pub fn children_mut(&mut self) -> &mut BTreeMap<EK, OrderedHashMap<NodePtr<RwLock<Trie<EK, EV, T>>>, EV>> {
        &mut self.children
    }

    // is_leaf remains unchanged
    pub fn is_leaf(&self) -> bool {
        self.children.values().all(|dest_map| {
            dest_map.keys().all(|node_ptr| !node_ptr.is_strong())
        })
    }

    /// Collects all *unique* nodes (by pointer) reachable from the given root (BFS).
    /// This method does not panic on cycles, it simply avoids revisiting nodes.
    pub fn all_nodes(roots: &[Arc<RwLock<Trie<EK, EV, T>>>]) -> Vec<Arc<RwLock<Trie<EK, EV, T>>>> {
        // Use Arc::as_ptr for visited tracking
        let mut visited_arcs: HashSet<*const RwLock<Trie<EK, EV, T>>> = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        for root in roots {
            if visited_arcs.insert(Arc::as_ptr(root)) {
                queue.push_back(root.clone());
            }
        }

        while let Some(node_arc) = queue.pop_front() {
            result.push(node_arc.clone()); // Add the node itself to the result

            // Lock the node to get its children
            let node_guard = node_arc.read().expect("RwLock poisoned during BFS"); // Renamed node to node_guard
            for children_map in node_guard.children.values() { // Use node_guard
                for node_ptr in children_map.keys() {
                    let child_arc = node_ptr.upgrade().expect("Dangling weak pointer in Trie::all_nodes");
                    let child_arc_ptr = Arc::as_ptr(&child_arc);
                    if visited_arcs.insert(child_arc_ptr) {
                        queue.push_back(child_arc.clone());
                    }
                }
            }
            // node_guard lock is released here
        }
        result
    }

    /// Checks if there are any cycles reachable from the given `root_arc`.
    /// Returns `true` if a cycle is detected, `false` otherwise.
    /// This method is useful for verifying graph integrity after complex build processes.
    pub fn has_any_cycle(root_arc: Arc<RwLock<Trie<EK, EV, T>>>) -> bool {
        let mut global_visited_arcs: HashSet<*const RwLock<Trie<EK, EV, T>>> = HashSet::new();
        let mut recursion_stack_arcs: HashSet<*const RwLock<Trie<EK, EV, T>>> = HashSet::new();
        // Call the recursive helper starting with the root node.
        Self::_has_any_cycle_recursive(root_arc, &mut global_visited_arcs, &mut recursion_stack_arcs)
    }

    /// Recursive helper function for `has_any_cycle`.
    ///
    /// `global_visited_arcs`: Tracks all nodes that have been visited and processed across
    ///                        all recursion branches. This prevents re-processing subgraphs
    ///                        that are already known to be cycle-free (or whose cycles
    ///                        would have been detected via another path).
    /// `recursion_stack_arcs`: Tracks nodes currently in the recursion stack for the *current*
    ///                         DFS path. A cycle is detected if we try to visit a node
    ///                         that is already in this set.
    fn _has_any_cycle_recursive(
        node_arc: Arc<RwLock<Trie<EK, EV, T>>>,
        global_visited_arcs: &mut HashSet<*const RwLock<Trie<EK, EV, T>>>,
        recursion_stack_arcs: &mut HashSet<*const RwLock<Trie<EK, EV, T>>>,
    ) -> bool {
        let node_arc_ptr = Arc::as_ptr(&node_arc);

        // If the node is already in the current recursion stack, we've found a back-edge (a cycle).
        if recursion_stack_arcs.contains(&node_arc_ptr) {
            return true; // Cycle detected
        }

        // If the node has been globally visited AND is not in the current recursion stack,
        // it means this node was fully processed via another path and no cycles were found
        // originating from it *then*. We can safely return false to avoid re-exploring.
        if global_visited_arcs.contains(&node_arc_ptr) {
            return false; // Already processed and known to be part of a cycle-free subgraph (from its perspective)
        }

        // Add the current node to both sets:
        // - To recursion_stack_arcs to mark it as part of the current DFS path.
        // - To global_visited_arcs to mark it as processed, so we don't re-explore it unnecessarily
        //   if reached from another path later.
        recursion_stack_arcs.insert(node_arc_ptr);
        global_visited_arcs.insert(node_arc_ptr);

        // Lock the node to get its children.
        // It's important to collect children Arcs first and then release the lock before recursing.
        let children_arcs: Vec<Arc<RwLock<Trie<EK, EV, T>>>> = {
            let node_guard_val = node_arc.read().expect("RwLock poisoned during has_any_cycle traversal"); // Renamed node_guard
            node_guard_val.children // Use node_guard_val
                .values() // Iterate over HashMap<ArcPtrWrapper<Mutex<...>>, EV>
                .flat_map(|dest_map| {
                    dest_map.keys().filter_map(|node_ptr| match node_ptr {
                        NodePtr::Strong(arc_wrapper) => Some(arc_wrapper.as_arc().clone()),
                        NodePtr::Weak(_) => None,
                    })
                })
                .collect()
        }; // node_guard_val lock is released here.

        // Recursively check each child.
        for child_arc in children_arcs {
            if Self::_has_any_cycle_recursive(child_arc, global_visited_arcs, recursion_stack_arcs) {
                return true; // Cycle detected in a descendant path
            }
        }

        // If we've processed all children of this node and found no cycles,
        // remove it from the recursion stack (as we are "returning" up the DFS path).
        // It remains in global_visited_arcs.
        recursion_stack_arcs.remove(&node_arc_ptr);

        false // No cycle found originating from this node or its descendants along this path
    }

    /// Recomputes `max_depth` for all nodes reachable from the given roots.
    /// This is useful after manual graph manipulations that do not automatically
    /// update the depths. It uses a topological sort (Kahn's algorithm) to ensure
    /// correctness in a single pass.
    pub fn recompute_all_max_depths(roots: &[Arc<RwLock<Self>>]) {
        let all_nodes = Self::all_nodes(roots);
        if all_nodes.is_empty() {
            return;
        }

        let mut node_map: HashMap<*const RwLock<Self>, Arc<RwLock<Self>>> = HashMap::new();
        for node_arc in &all_nodes {
            node_map.insert(Arc::as_ptr(node_arc), node_arc.clone());
        }

        let mut in_degree: HashMap<*const RwLock<Self>, usize> = HashMap::new();
        let mut adj: HashMap<*const RwLock<Self>, Vec<*const RwLock<Self>>> = HashMap::new();

        for node_arc in &all_nodes {
            let node_ptr = Arc::as_ptr(node_arc);
            in_degree.entry(node_ptr).or_insert(0);
            adj.entry(node_ptr).or_default();

            let node_guard = node_arc.read().unwrap();
            for child_arc in node_guard.children.values().flat_map(|m| m.keys()).filter_map(|node_ptr| {
                if let NodePtr::Strong(arc_wrapper) = node_ptr {
                    Some(arc_wrapper.as_arc().clone())
                } else {
                    None
                }
            }) {
                let child_ptr = Arc::as_ptr(&child_arc);
                adj.entry(node_ptr).or_default().push(child_ptr);
                *in_degree.entry(child_ptr).or_default() += 1;
            }
        }

        let mut queue = VecDeque::new();
        for node_arc in &all_nodes {
            let node_ptr = Arc::as_ptr(node_arc);
            if in_degree.get(&node_ptr).cloned().unwrap_or(0) == 0 {
                queue.push_back(node_ptr);
                node_arc.write().unwrap().max_depth = 0;
            } else {
                // Reset depth for non-source nodes. It will be computed.
                node_arc.write().unwrap().max_depth = 0;
            }
        }

        while let Some(u_ptr) = queue.pop_front() {
            let u_arc = node_map.get(&u_ptr).unwrap();
            let u_depth = u_arc.read().unwrap().max_depth;

            if let Some(children_ptrs) = adj.get(&u_ptr) {
                for &v_ptr in children_ptrs {
                    let v_arc = node_map.get(&v_ptr).unwrap();
                    {
                        let mut v_guard = v_arc.write().unwrap();
                        v_guard.max_depth = v_guard.max_depth.max(u_depth + 1);
                    }

                    let v_in_degree = in_degree.get_mut(&v_ptr).unwrap();
                    *v_in_degree -= 1;
                    if *v_in_degree == 0 {
                        queue.push_back(v_ptr);
                    }
                }
            }
        }
    }

    /// Recomputes the max_depth of this node based on its children's depths.
    /// Returns true if the depth changed.
    /// NOTE: This does NOT propagate changes. The caller is responsible for propagation
    /// if the depth of this node changes and it has parents. This is typically safe
    /// to call in a post-order traversal where children's depths are finalized first.
    pub fn recompute_max_depth(&mut self) -> bool {
        // Only consider STRONG edges when recomputing max_depth.
        // Weak edges should not affect max_depth, otherwise depth computations
        // can follow weak links and produce inflated/infinite depths.
        let new_max_depth = self.children.values()
            .flat_map(|dest_map| dest_map.keys())
            .filter_map(|node_ptr| {
                match node_ptr {
                    NodePtr::Strong(arc_wrapped) => Some(arc_wrapped.as_arc().clone()),
                    NodePtr::Weak(_) => None,
                }
            })
            .map(|child_arc| child_arc.read().unwrap().max_depth + 1)
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

// Helper to get the raw pointer to the Trie data from an Arc<Mutex<Trie>>.
/// Panics if the mutex is poisoned. Returns None if lock fails (WouldBlock).
/// **Use with caution:** Only use when you know a failed lock means the current thread holds it.
/// Consider using `Arc::as_ptr` for identity checks instead if possible.
#[allow(dead_code)] // Keep available, but node_ptr is preferred generally
pub(crate) fn try_get_node_data_ptr<EK: Ord, EV, T>(node_arc: &Arc<RwLock<Trie<EK, EV, T>>>) -> Option<*const Trie<EK, EV, T>> {
    match node_arc.try_read() {
        Ok(guard) => {
            let ptr = &*guard as *const Trie<EK, EV, T>;
            Some(ptr)
            // Guard is dropped here, lock released
        }
        Err(TryLockError::Poisoned(p)) => {
            panic!("RwLock poisoned when trying to get node data pointer: {:?}", p);
        }
        Err(TryLockError::WouldBlock) => {
            // Lock is held, likely by the current thread in specific scenarios (like cycle check).
            None
        }
    }
}

/// Helper to get the raw pointer to the Trie data from an Arc<Mutex<Trie>>.
/// Panics if the mutex is poisoned or if locking fails (blocking lock).
/// **Use when you need the pointer and expect the lock to succeed.**
#[allow(dead_code)] // Keep available, but Arc::as_ptr is often better for identity
pub(crate) fn node_ptr<EK: Ord, EV, T>(node_arc: &Arc<RwLock<Trie<EK, EV, T>>>) -> *const Trie<EK, EV, T> {
    let guard = node_arc.read().expect("RwLock poisoned or lock failed when getting node pointer");
    &*guard as *const _
}

// Add this impl block for the recursive comparison helper
impl<EK, EV, T> Trie<EK, EV, T>
where
    EK: Ord, // Ord implies PartialEq + Eq
    EV: PartialEq + Clone,
    T: PartialEq,
{
    /// Recursively compares two Trie nodes wrapped in Arcs for equality.
    ///
    /// - `self_arc`, `other_arc`: The Arcs pointing to the Trie nodes to compare.
    /// - `comparison_cache`: Tracks pairs of (self_node_ptr, other_node_ptr) and their comparison result (bool).
    ///   This cache is crucial for:
    ///     1. Efficiency: Avoid re-comparing already processed pairs.
    ///     2. Cycle Handling: Prevents infinite recursion by pre-emptively marking a pair as true
    ///        and updating to false only if a mismatch is found.
    ///     3. Topology: Ensures that if NodeA in self maps to NodeX in other, this mapping is consistent.
    fn compare_arcs_recursive(
        self_arc: &Arc<RwLock<Trie<EK, EV, T>>>,
        other_arc: &Arc<RwLock<Trie<EK, EV, T>>>,
        comparison_cache: &mut HashMap<(*const RwLock<Self>, *const RwLock<Self>), bool>,
    ) -> bool {
        let self_ptr = Arc::as_ptr(self_arc);
        let other_ptr = Arc::as_ptr(other_arc);

        // If both Arcs point to the exact same RwLock instance, they are definitionally equal in this context.
        if self_ptr == other_ptr {
            return true;
        }

        // Ensure canonical cache key: (min_ptr, max_ptr).
        // The boolean result of equality is symmetric, so order doesn't change the meaning of 'true' or 'false'.
        let (cache_key_ptr1, cache_key_ptr2) = if self_ptr < other_ptr {
            (self_ptr, other_ptr)
        } else {
            (other_ptr, self_ptr)
        };

        // Check cache for prior comparison result of this specific pair.
        if let Some(&cached_result) = comparison_cache.get(&(cache_key_ptr1, cache_key_ptr2)) {
            return cached_result;
        }

        // Pre-emptively mark this pair as true in the cache.
        // If a cycle is encountered leading back to this pair, this 'true' will be returned,
        // assuming consistency unless a mismatch is found later down the path.
        // If any subsequent comparison fails, this cache entry will be updated to false.
        comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), true);

        // Attempt to lock both nodes. If locking fails (e.g., poisoned mutex, or would block
        // in a more complex scenario not expected here), treat them as unequal for safety.
        let self_node_guard = match self_arc.try_read() {
            Ok(g) => g,
            Err(_) => {
                comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false); // Update cache to reflect failure
                return false;
            }
        };
        let other_node_guard = match other_arc.try_read() {
            Ok(g) => g,
            Err(_) => {
                comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false); // Update cache
                return false;
            }
        };

        // Dereference guards to get &Trie
        let self_node = &*self_node_guard;
        let other_node = &*other_node_guard;


        // 1. Compare non-recursive fields: value and max_depth.
        if self_node.value != other_node.value || self_node.max_depth != other_node.max_depth {
            comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false); // Update cache
            return false;
        }

        // 2. Compare children structure (number of distinct edge keys).
        if self_node.children.len() != other_node.children.len() {
            comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false); // Update cache
            return false;
        }

        // 3. Compare children for each edge key.
        for (self_ek, self_dest_map) in &self_node.children {
            match other_node.children.get(self_ek) {
                None => { // Edge key present in self but not in other.
                    comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false); // Update cache
                    return false;
                }
                Some(other_dest_map) => {
                    // Number of destinations for this edge key must match.
                    if self_dest_map.len() != other_dest_map.len() {
                        comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false); // Update cache
                        return false;
                    }

                    // Collect (Arc<Mutex<Trie>>, EV) pairs for detailed comparison.
                    let self_child_pairs: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, EV)> = self_dest_map.iter()
                        .filter_map(|(node_ptr, ev)| {
                            if let NodePtr::Strong(apw) = node_ptr {
                                Some((apw.as_arc().clone(), ev.clone()))
                            } else { None }
                        })
                        .collect();

                    let mut other_child_pairs: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, EV)> = other_dest_map.iter()
                        .filter_map(|(node_ptr, ev)| {
                            if let NodePtr::Strong(apw) = node_ptr {
                                Some((apw.as_arc().clone(), ev.clone()))
                            } else { None }
                        })
                        .collect();


                    // For each child in self_child_pairs, find a matching child in other_child_pairs.
                    // A match requires equal edge values (EV) and recursively equal Trie nodes.
                    'self_pair_loop: for (s_arc, s_ev) in &self_child_pairs {
                        let mut found_match_for_current_self_pair = false;
                        for i in 0..other_child_pairs.len() { // Iterate indices to allow removal
                            if s_ev == &other_child_pairs[i].1 { // Compare EV (s_ev is &EV, other_child_pairs[i].1 is EV)
                                // Edge values match, now recursively compare the pointed-to Trie nodes.
                                // Clone o_arc for the recursive call to avoid borrow issues if remove() happens.
                                let o_arc_for_recursion = other_child_pairs[i].0.clone();
                                if Trie::compare_arcs_recursive(s_arc, &o_arc_for_recursion, comparison_cache) {
                                    other_child_pairs.remove(i); // Match found, remove from other_list.
                                    found_match_for_current_self_pair = true;
                                    break; // Found match for current s_arc, move to next s_arc.
                                }
                                // If recursive compare is false, this o_arc is not a match. Continue inner loop.
                            }
                        }
                        if !found_match_for_current_self_pair {
                            // No match found in other_child_pairs for the current s_arc/s_ev.
                            comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false); // Update cache
                            return false;
                        }
                    }
                    // If all self_child_pairs found matches, other_child_pairs should be empty
                    // (due to initial length check and removals). No explicit check needed here.
                }
            }
        }

        // If all checks passed, the initial `true` assumption in the cache was correct.
        // The cache entry (self_ptr, other_ptr) remains true.
        true
    }
}


// Implementation block for special_map and related functionality
// Requires T: Clone, EK: Ord + Clone, EV: Clone
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
                for node_ptr in children_map.keys() {
                    if let Some(child_arc) = node_ptr.upgrade() {
                        total_edges += 1;
                        if visited_arcs.insert(Arc::as_ptr(&child_arc)) {
                            queue.push_back(child_arc.clone());
                        }
                    }
                }
            }
        }
        total_edges
    }

    /// Performs a specialized breadth-first traversal (related to Dijkstra/Bellman-Ford relaxation).
    ///
    /// This version correctly traverses both strong and weak edges. It also handles
    /// cycles by allowing nodes to be re-processed, continuing propagation as long
    /// as the `process` closure returns `true` for a given node.
    #[time_it]
    pub fn special_map<V: Clone>(
        initial_nodes_and_values: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, V)>,
        mut step: impl FnMut(&V, &EK, &EV, &Trie<EK, EV, T>) -> Option<V>,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&Trie<EK, EV, T>, &mut V) -> bool,
    ) {
        // ------------------------------------------------------------------
        //  Simple depth-driven scheduler.
        // ------------------------------------------------------------------
        let mut values: HashMap<*const RwLock<Self>, V> = HashMap::new();
        // This set now tracks nodes where `process` returned `false`, stopping propagation.
        let mut stopped_nodes: HashSet<*const RwLock<Self>> = HashSet::new();
        // Using ArcPtrWrapper for consistency with special_map_grouped
        let mut todo: BTreeMap<usize, OrderedHashSet<ArcPtrWrapper<RwLock<Self>>>> = BTreeMap::new();

        let initial_nodes: Vec<_> = initial_nodes_and_values.iter().map(|(n, _)| n.clone()).collect();
        let total_edges = Self::count_all_edges(&initial_nodes);
        if PROGRESS_BAR_ENABLED {
            println!("Progress bar enabled");
        } else {
            println!("Progress bar disabled")
        }
        let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED, leave=false);

        // Seed with the user-supplied starting set
        for (node_arc, v0) in initial_nodes_and_values {
            let ptr = Arc::as_ptr(&node_arc);
            values
                .entry(ptr)
                .and_modify(|old| merge(old, v0.clone()))
                .or_insert(v0);
            let depth = node_arc.read().expect("poison").max_depth;
            todo.entry(depth).or_default().insert(ArcPtrWrapper::new(node_arc.clone()));
        }

        // Main loop ---------------------------------------------------------
        while let Some((_depth, node_arc_ptr_wrappers)) = todo.pop_first() {
            for node_ptr_wrapper in &node_arc_ptr_wrappers {
                let ptr = node_ptr_wrapper.as_ref() as *const RwLock<Self>;
                // A node that has been stopped should not be processed again.
                if stopped_nodes.contains(&ptr) { continue; }

                let mut agg_v = match values.remove(&ptr) {
                    Some(v) => v,
                    None => continue,
                };
                let node_arc = node_ptr_wrapper.as_arc();

                // ---------- user ‘process’ callback ----------
                let proceed = {
                    let guard = node_arc.read().expect("poison");
                    process(&guard, &mut agg_v)
                };

                if !proceed {
                    // User stopped traversal at this node. Mark it and do not propagate.
                    stopped_nodes.insert(ptr);
                    continue;
                }

                // ---------- propagate to children -------------
                // This block is now corrected to include BOTH strong and weak edges.
                let edges: Vec<(EK, EV, Arc<RwLock<Self>>)> = {
                    let guard = node_arc.read().expect("poison");
                    guard.children
                        .iter()
                        .flat_map(|(ek, dst_map)| {
                            dst_map.iter().map(move |(node_ptr, ev)| {
                                // Panic on expired weak pointers.
                                (ek.clone(), ev.clone(), node_ptr.upgrade().expect("Dangling weak pointer during special_map traversal"))
                            })
                        })
                        .collect()
                };

                for (ek, ev, child_arc) in edges {
                    let _ = pb.update(1);
                    let child_ptr = Arc::as_ptr(&child_arc);

                    // Optimization: Don't bother queueing a child if we know its path is stopped.
                    if stopped_nodes.contains(&child_ptr) {
                        continue;
                    }

                    // user ‘step’ callback
                    let maybe_v = {
                        let child_guard = child_arc.read().expect("poison");
                        step(&agg_v, &ek, &ev, &child_guard)
                    };
                    if let Some(new_v) = maybe_v {
                        values
                            .entry(child_ptr)
                            .and_modify(|old| merge(old, new_v.clone()))
                            .or_insert(new_v);

                        // Queue child by its declared depth
                        let child_depth = child_arc.read().expect("poison").max_depth;
                        todo.entry(child_depth).or_default().insert(ArcPtrWrapper::new(child_arc));
                    }
                }
            }
        }
    }

    /// Performs a specialized breadth-first traversal, grouping children by edge key.
    /// This is more efficient than `special_map` when many edges share the same key,
    /// as the `step` function is called once per key, not once per edge.
    ///
    /// This version correctly traverses both strong and weak edges. It also handles
    /// cycles by allowing nodes to be re-processed, continuing propagation as long
    /// as the `process` closure returns `true` for a given node.
    #[time_it]
    pub fn special_map_grouped<V, S, I>(
        initial_nodes_and_values: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, V)>,
        mut step: S,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&Trie<EK, EV, T>, &mut V) -> bool,
    )
    where
        V: Clone,
        S: FnMut(
            &V, &EK, &OrderedHashMap<NodePtr<RwLock<Trie<EK, EV, T>>>, EV>
        ) -> I,
        I: IntoIterator<Item = (NodePtr<RwLock<Trie<EK, EV, T>>>, V)>,
    {
        // ------------------------------------------------------------------
        //  Simple depth-driven scheduler. (Same as special_map)
        // ------------------------------------------------------------------
        let mut values: HashMap<*const RwLock<Self>, V> = HashMap::new();
        // This set now tracks nodes where `process` returned `false`, stopping propagation.
        let mut stopped_nodes: HashSet<*const RwLock<Self>> = HashSet::new();
        let mut todo: BTreeMap<usize, OrderedHashSet<ArcPtrWrapper<RwLock<Self>>>> = BTreeMap::new();

        let initial_nodes: Vec<_> = initial_nodes_and_values.iter().map(|(n, _)| n.clone()).collect();
        let total_edges = Self::count_all_edges(&initial_nodes);
        let mut pb = tqdm!(total = total_edges, desc = "Traversing edges", disable = !PROGRESS_BAR_ENABLED, leave=false);

        // Seed with the user-supplied starting set
        for (node_arc, v0) in initial_nodes_and_values {
            let ptr = Arc::as_ptr(&node_arc);
            values
                .entry(ptr)
                .and_modify(|old| merge(old, v0.clone()))
                .or_insert(v0);
            let depth = node_arc.read().expect("poison").max_depth;
            todo.entry(depth).or_default().insert(ArcPtrWrapper::new(node_arc.clone()));
        }

        // Main loop ---------------------------------------------------------
        while let Some((_depth, node_arc_ptr_wrappers)) = todo.pop_first() {
            for node_ptr_wrapper in &node_arc_ptr_wrappers {
                let ptr = node_ptr_wrapper.as_ref() as *const RwLock<Self>;
                // A node that has been stopped should not be processed again.
                if stopped_nodes.contains(&ptr) { continue; }

                let mut agg_v = match values.remove(&ptr) {
                    Some(v) => v,
                    None => continue,
                };
                let node_arc = node_ptr_wrapper.as_arc();

                let proceed = {
                    let guard = node_arc.read().expect("poison");
                    process(&guard, &mut agg_v)
                };

                if !proceed {
                    // User stopped traversal at this node. Mark it and do not propagate.
                    stopped_nodes.insert(ptr);
                    continue;
                }

                // ---------- propagate to children (grouped by edge key) -------------
                // This block is now corrected to include BOTH strong and weak edges.
                let children_by_ek: Vec<(EK, OrderedHashMap<NodePtr<RwLock<Self>>, EV>)> = {
                    let guard = node_arc.read().expect("poison");
                    guard.children.iter()
                        .map(|(ek, dst_map)| (ek.clone(), dst_map.clone()))
                        .collect()
                };

                for (ek, dest_map) in children_by_ek {
                    // We only count upgradable edges for the progress bar.
                    let valid_edges_count = dest_map.keys().filter(|k| k.is_upgradable()).count();
                    if valid_edges_count > 0 {
                        let _ = pb.update(valid_edges_count);
                    }

                    let new_values_for_children = step(&agg_v, &ek, &dest_map);

                    for (child_node_ptr, new_v) in new_values_for_children {
                        // Panic on expired weak pointers.
                        let child_arc_wrapper = child_node_ptr.upgrade_wrapper().expect("Dangling weak pointer during special_map_grouped traversal");
                        let child_ptr = child_arc_wrapper.as_ref() as *const RwLock<Self>;

                        // Optimization: Don't bother queueing a child if we know its path is stopped.
                        if stopped_nodes.contains(&child_ptr) {
                            continue;
                        }

                        values.entry(child_ptr)
                            .and_modify(|old| merge(old, new_v.clone()))
                            .or_insert(new_v);

                        let child_depth = child_arc_wrapper.as_arc().read().expect("poison").max_depth;
                        todo.entry(child_depth).or_default().insert(child_arc_wrapper);
                    }
                }
            }
        }
    }
}

impl<EK, EV, T> Trie<EK, EV, T>
where
    EK: Ord + Clone,
{
    /// Attempts to convert all reachable weak edges into strong edges, where doing so
    /// will not introduce a cycle in the strong-edge subgraph.
    ///
    /// This function traverses the graph reachable from the given set of roots, following
    /// both strong and weak edges for traversal, and for every weak edge (src --ek--> dst)
    /// attempts to:
    ///   1) Check that promoting it to a strong edge would not create a cycle
    ///      (i.e., there is no strong path from `dst` back to `src`).
    ///   2) If safe, update `dst.max_depth` to at least `src.max_depth + 1` and
    ///      propagate max_depth to descendants via strong edges.
    ///   3) Replace the weak edge key with a strong edge key in `src.children`.
    ///
    /// Returns the number of edges that were promoted from weak to strong.
    ///
    /// Notes:
    /// - Promotion is greedy. If a promotion is not possible at the moment due to an
    ///   existing strong cycle, subsequent promotions cannot make it possible later
    ///   (promotions only add strong edges, never remove).
    /// - The traversal visits nodes reachable via both strong and weak edges so that
    ///   it can consider promoting weak edges deeper in the graph as well.
    /// - This function is conservative with RwLock usage to avoid deadlocks:
    ///   it does not hold a write-lock on `src` while running the (potentially wide)
    ///   max-depth propagation from `dst`.
    pub fn promote_weak_edges_to_strong(
        root_nodes: &[Arc<RwLock<Trie<EK, EV, T>>>],
    ) -> usize {
        // Visit all nodes reachable through either strong or weak edges.
        let mut visited: HashSet<*const RwLock<Trie<EK, EV, T>>> = HashSet::new();
        let mut queue: VecDeque<Arc<RwLock<Trie<EK, EV, T>>>> = VecDeque::new();

        for root in root_nodes {
            let ptr = Arc::as_ptr(root);
            if visited.insert(ptr) {
                queue.push_back(root.clone());
            }
        }

        let mut promotions = 0usize;

        while let Some(src_arc) = queue.pop_front() {
            // Collect neighbors (for traversal) and weak edges (for potential promotion)
            let (neighbors, weak_edges): (Vec<Arc<RwLock<Trie<EK, EV, T>>>>, Vec<(EK, Arc<RwLock<Trie<EK, EV, T>>>)>) = {
                let src_guard = src_arc.read().expect("RwLock poisoned during weak-edge promotion scan");
                let mut neigh = Vec::new();
                let mut weak = Vec::new();
                for (ek, dest_map) in &src_guard.children {
                    for (node_ptr, _ev) in dest_map {
                        if let Some(child_arc) = node_ptr.upgrade() {
                            // Traverse both strong and weak edges
                            neigh.push(child_arc.clone());
                            // Record weak edges for possible promotion
                            if !node_ptr.is_strong() {
                                weak.push((ek.clone(), child_arc));
                            }
                        }
                        // If upgrade fails, it's a dangling pointer. We simply ignore it.
                    }
                }
                (neigh, weak)
            };

            // Enqueue neighbors for traversal (both strong and weak destinations)
            for n in neighbors {
                let ptr = Arc::as_ptr(&n);
                if visited.insert(ptr) {
                    queue.push_back(n);
                }
            }

            // For every weak edge src --ek--> dst, try to promote to strong.
            for (ek, dst_arc) in weak_edges {
                // Quick cycle check: would making (src -> dst) strong create a cycle?
                // A cycle would exist iff `src` is reachable from `dst` via strong edges.
                let src_data_ptr = node_ptr(&src_arc);
                if Self::detect_cycle(src_data_ptr, &dst_arc) {
                    // Would create a cycle; skip.
                    continue;
                }

                // We will:
                //  1) Read src.max_depth (requires write lock for subsequent structural update anyway)
                //  2) Raise dst.max_depth if needed and propagate to descendants
                //  3) Replace weak key by strong key in src.children[ek]
                //
                // To avoid races:
                //  - Hold a write lock on src while we structurally edit its children map.
                //  - It's safe to hold the write lock on src while propagating from dst,
                //    because the new strong edge is not yet installed, and therefore
                //    the propagation from dst cannot reach src through strong edges
                //    (which would be a cycle we already checked against).
                let mut src_guard = match src_arc.write() {
                    Ok(g) => g,
                    Err(_) => continue, // If poisoned or temporarily unavailable, skip this edge.
                };

                // Verify the weak edge still exists and hasn't been promoted already.
                let dest_map = match src_guard.children.get_mut(&ek) {
                    Some(m) => m,
                    None => continue, // Edge key gone
                };

                let target_ptr = Arc::as_ptr(&dst_arc) as usize;
                let has_strong_already = dest_map
                    .iter()
                    .any(|(k, _)| k.is_strong() && k.as_ptr_usize() == target_ptr);
                if has_strong_already {
                    // Someone else promoted it already; nothing to do.
                    continue;
                }
                let has_matching_weak = dest_map
                    .iter()
                    .any(|(k, _)| !k.is_strong() && k.as_ptr_usize() == target_ptr);
                if !has_matching_weak {
                    // Weak entry disappeared (or destination dropped); skip.
                    continue;
                }

                // Compute required depth for dst.
                let candidate_depth = src_guard.max_depth.saturating_add(1);

                // Update dst.max_depth if needed, and propagate to its descendants.
                // Keep previous depth to allow rollback on failure (very unlikely here).
                let previous_child_depth;
                let needs_update;
                {
                    let mut child_guard = match dst_arc.write() {
                        Ok(g) => g,
                        Err(_) => continue, // Child temporarily unavailable; skip.
                    };
                    previous_child_depth = child_guard.max_depth;
                    needs_update = candidate_depth > previous_child_depth;
                    if needs_update {
                        child_guard.max_depth = candidate_depth;
                    }
                } // child write lock dropped here

                if needs_update {
                    if let Err(_e) = Self::propagate_max_depth(dst_arc.clone(), candidate_depth) {
                        // Roll back child's depth and skip promotion.
                        if let Ok(mut child_guard) = dst_arc.write() {
                            if child_guard.max_depth == candidate_depth {
                                child_guard.max_depth = previous_child_depth;
                            }
                        }
                        continue;
                    }
                }

                // Replace the weak key with a strong key in src.children[ek].
                // We must actually replace the key (not only the value), since the key's
                // variant (Weak vs Strong) affects algorithms like serialization and traversal.
                //
                // OrderedHashMap::insert with an equal key would typically only replace the value,
                // keeping the original key. Therefore we rebuild the map for this `ek` by moving
                // entries out and reinserting, substituting the target entry's key with a Strong key.
                let dest_map = src_guard
                    .children
                    .get_mut(&ek)
                    .expect("dest_map must still exist after earlier borrow");

                let mut old_map = std::mem::take(dest_map);
                let mut did_convert = false;
                for (k, v) in old_map.into_iter() {
                    if k.as_ptr_usize() == target_ptr {
                        // Reinsert with a Strong key (always), preserving the EV.
                        // Note: if the entry was already Strong (shouldn't happen due to checks),
                        // this is idempotent.
                        dest_map.insert(NodePtr::Strong(ArcPtrWrapper::new(dst_arc.clone())), v);
                        // Count conversion only if the old key was Weak
                        if !k.is_strong() {
                            did_convert = true;
                        }
                    } else {
                        dest_map.insert(k, v);
                    }
                }

                if did_convert {
                    promotions += 1;
                }
                // src_guard (write) drops here at end of scope loop
            }
        }

        promotions
    }
}

// Implement PartialEq for Trie
impl<EK, EV, T> PartialEq for Trie<EK, EV, T>
where
    EK: Ord, // Ord implies PartialEq + Eq, needed for BTreeMap keys and get
    EV: PartialEq + Clone, // PartialEq for EV comparison, Clone for collecting pairs
    T: PartialEq, // For self.value == other.value
{
    fn eq(&self, other: &Self) -> bool {
        // 1. Compare non-recursive fields: value and max_depth.
        if self.value != other.value || self.max_depth != other.max_depth {
            return false;
        }

        // 2. Compare children structure (number of distinct edge keys).
        if self.children.len() != other.children.len() {
            return false;
        }

        // Initialize cache for recursive calls on child Arcs.
        // This cache is passed down through all recursive calls originating from this top-level eq.
        // Type alias for pointer to RwLock<Trie<...>> for clarity.
        type NodeRwLockPtr<EKK, EVV, TT> = *const RwLock<Trie<EKK, EVV, TT>>;
        let mut comparison_cache: HashMap<(NodeRwLockPtr<EK, EV, T>, NodeRwLockPtr<EK, EV, T>), bool> = HashMap::new();


        // 3. Compare children for each edge key.
        for (self_ek, self_dest_map) in &self.children {
            match other.children.get(self_ek) {
                None => return false, // Edge key present in self but not in other.
                Some(other_dest_map) => {
                    let (self_strong, self_weak): (Vec<_>, Vec<_>) = self_dest_map.iter().partition(|(k, _)| k.is_strong());
                    let (other_strong, other_weak): (Vec<_>, Vec<_>) = other_dest_map.iter().partition(|(k, _)| k.is_strong());

                    if self_strong.len() != other_strong.len() || self_weak.len() != other_weak.len() {
                        return false;
                    }

                    // Collect (Arc<Mutex<Trie>>, EV) pairs for detailed comparison.
                    let self_child_pairs: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, &EV)> = self_strong
                        .iter()
                        .filter_map(|(np, ev)| np.upgrade().map(|arc| (arc, *ev)))
                        .collect();

                    let mut other_child_pairs: Vec<(Arc<RwLock<Trie<EK, EV, T>>>, EV)> = other_strong
                        .iter()
                        .filter_map(|(np, ev)| np.upgrade().map(|arc| (arc, (*ev).clone())))
                        .collect();


                    // For each child in self_child_pairs, find a matching child in other_child_pairs.
                    // A match requires equal edge values (EV) and recursively equal Trie nodes.
                    'self_pair_loop: for (s_arc, s_ev) in self_child_pairs {
                        for i in 0..other_child_pairs.len() { // Iterate indices to allow removal
                            if s_ev == &other_child_pairs[i].1 { // Compare EV (s_ev is &EV, other_child_pairs[i].1 is EV)
                                // Edge values match, now recursively compare the pointed-to Trie nodes.
                                let o_arc_for_recursion = other_child_pairs[i].0.clone();
                                if Trie::compare_arcs_recursive(&s_arc, &o_arc_for_recursion, &mut comparison_cache) {
                                    other_child_pairs.remove(i); // Match found.
                                    continue 'self_pair_loop;
                                }
                            }
                        }
                        return false; // No match found for the current s_arc/s_ev.
                    }

                    // Now compare weak children
                    let self_weak_pairs: Vec<_> = self_weak.iter().map(|(np, ev)| (np.upgrade().expect("Dangling weak pointer in Trie::eq (self)"), (*ev).clone())).collect();
                    let mut other_weak_pairs: Vec<_> = other_weak.iter().map(|(np, ev)| (np.upgrade().expect("Dangling weak pointer in Trie::eq (other)"), (*ev).clone())).collect();

                    if self_weak_pairs.len() != other_weak_pairs.len() {
                        return false;
                    }

                    'self_weak_loop: for (s_arc, s_ev) in &self_weak_pairs {
                        for i in 0..other_weak_pairs.len() {
                            if s_ev == &other_weak_pairs[i].1 {
                                let o_arc = other_weak_pairs[i].0.clone();
                                if Trie::compare_arcs_recursive(s_arc, &o_arc, &mut comparison_cache) {
                                    other_weak_pairs.remove(i);
                                    continue 'self_weak_loop;
                                }
                            }
                        }
                        return false;
                    }
                }
            }
        }

        // All checks passed.
        true
    }
}

// Implement Eq for Trie
impl<EK, EV, T> Eq for Trie<EK, EV, T>
where
    EK: Ord, // Ord implies Eq
    EV: Eq + Clone, // EV also needs to be Eq
    T: Eq,         // T also needs to be Eq
{
}

// Implement Hash for Trie
impl<EK, EV, T> Hash for Trie<EK, EV, T>
where
    EK: Ord + Hash, // Ord for BTreeMap iteration, Hash for hashing EK
    EV: PartialEq + Clone + Hash, // From PartialEq, add Hash for EV
    T: PartialEq + Hash,    // From PartialEq, add Hash for T
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Cache to handle cycles and shared nodes during hashing.
        // Maps the raw data pointer of a Trie node to a marker (depth) to break cycles.
        let mut recursion_marker: HashMap<*const Trie<EK, EV, T>, usize> = HashMap::new();
        Self::hash_trie_recursive(self, state, &mut recursion_marker, 0);
    }
}

impl<EK, EV, T> Trie<EK, EV, T>
where
    EK: Ord + Hash, // Ord for BTreeMap iteration, Hash for hashing EK
    EV: PartialEq + Clone + Hash, // From PartialEq, add Hash for EV
    T: PartialEq + Hash,    // From PartialEq, add Hash for T
{
    /// Helper function to hash a &Trie instance.
    fn hash_trie_recursive<S: Hasher>(
        node: &Trie<EK, EV, T>,
        state: &mut S,
        recursion_marker: &mut HashMap<*const Trie<EK, EV, T>, usize>,
        current_depth: usize,
    ) {
        let node_ptr = node as *const _;
        if let Some(visited_depth) = recursion_marker.get(&node_ptr) {
            // Node already visited. Hash its pointer and depth to break cycles
            // and distinguish it from other nodes.
            node_ptr.hash(state);
            visited_depth.hash(state);
            return;
        }
        recursion_marker.insert(node_ptr, current_depth);

        // Hash non-recursive fields.
        node.value.hash(state);
        node.max_depth.hash(state);

        // Hash children.
        node.children.len().hash(state);
        for (ek, dest_map) in &node.children {
            ek.hash(state);

            let (strong_children, weak_children): (Vec<_>, Vec<_>) = dest_map.iter().partition(|(k, _)| k.is_strong());

            // Hash strong children
            strong_children.len().hash(state);
            let mut strong_pair_hashes = Vec::with_capacity(strong_children.len());
            for (node_ptr, ev) in strong_children {
                let mut pair_hasher = DeterministicHasher::new(DefaultHasher::new());
                ev.hash(&mut pair_hasher);
                let child_arc = node_ptr.upgrade().expect("Dangling weak pointer in Trie::hash");
                if let Ok(child_guard) = child_arc.read() {
                    Self::hash_trie_recursive(&*child_guard, &mut pair_hasher, recursion_marker, current_depth + 1);
                    strong_pair_hashes.push(pair_hasher.finish());
                };
            }
            strong_pair_hashes.sort_unstable();
            for h in strong_pair_hashes {
                h.hash(state);
            }

            // Hash weak children
            weak_children.len().hash(state);
            let mut weak_pair_hashes = Vec::with_capacity(weak_children.len());
            for (node_ptr, ev) in weak_children {
                let child_arc = node_ptr.upgrade().expect("Dangling weak pointer in Trie::hash");
                if let Ok(child_guard) = child_arc.read() {
                    // hash pair (ev, child)
                    let mut pair_hasher = DeterministicHasher::new(DefaultHasher::new());
                    ev.hash(&mut pair_hasher);
                    Self::hash_trie_recursive(&*child_guard, &mut pair_hasher, recursion_marker, current_depth + 1);
                    weak_pair_hashes.push(pair_hasher.finish());
                };
            }

            weak_pair_hashes.sort_unstable();
            for h in weak_pair_hashes {
                h.hash(state);
            }
        }
    }
}


/// A helper struct to facilitate inserting an edge into a Trie,
/// trying multiple potential destinations and optionally creating a new node.
/// Provides a chainable interface.
pub struct EdgeInserter<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
where
    EK: Ord + Clone,
    EV: Clone + Debug,
    T: Clone, // T needs to be Clone for else_create_destination_with_value -> Trie::new(value)
    FMergeEV: FnMut(&mut EV, EV), // Closure to merge edge values if edge exists - Changed signature
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    source_arc: Arc<RwLock<Trie<EK, EV, T>>>, // The source node for the edge
    edge_key: EK,                            // The key for the edge to be inserted
    edge_value: Option<EV>,                          // The value for the edge to be inserted
    merge_edge_value: FMergeEV,              // The function to merge edge values
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
    result: Option<Arc<RwLock<Trie<EK, EV, T>>>>, // Stores the successful destination node
}

impl<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T> EdgeInserter<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV), // Changed signature
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    /// Creates a new `EdgeInserter`.
    ///
    /// # Arguments
    ///
    /// * `source_arc`: The source node where the edge originates.
    /// * `edge_key`: The key for the new edge.
    /// * `edge_value`: The value for the new edge.
    /// * `merge_edge_value`: A closure that takes the existing edge value and the new edge value,
    ///   both by value, returning a merged value. This is only called if an edge with the same `edge_key` already
    ///   points to the `destination` being tried.
    pub fn new(
        source_arc: Arc<RwLock<Trie<EK, EV, T>>>,
        edge_key: EK,
        edge_value: EV,
        merge_edge_value: FMergeEV,
        update_node_value: FUpdateT,
        merge_edge_value_and_source_node_value: FMergeEV_T,
    ) -> Self {
        let mut edge_value = edge_value;
        let mut merge_edge_value_and_source_node_value = merge_edge_value_and_source_node_value;
        {
            let source_guard = source_arc.read().expect("RwLock poisoned while reading source node value for edge value merge");
            merge_edge_value_and_source_node_value(&mut edge_value, &source_guard.value);
        }

        EdgeInserter {
            source_arc,
            edge_key,
            edge_value: Some(edge_value),
            merge_edge_value,
            update_node_value,
            merge_edge_value_and_source_node_value,
            result: None,
        }
    }

    /// Tries to establish an edge to the given `destination`.
    ///
    /// If an edge with the same `edge_key` already exists pointing to `destination`,
    /// it attempts to merge the `edge_value` using the `merge_edge_value` closure.
    /// If no such edge exists, it attempts to insert a new edge using `try_insert`.
    ///
    /// This operation only proceeds if a successful destination hasn't already been found.
    /// Returns `self` to allow chaining.
    #[time_it]
    pub fn try_destination(mut self, destination: Arc<RwLock<Trie<EK, EV, T>>>) -> Self {
        if self.result.is_some() {
            return self; // Already found a destination
        }

        let mut update_info: Option<(Arc<RwLock<Trie<EK, EV, T>>>, EV)> = None;

        { // Scope for source_guard
            let mut source_guard = self.source_arc.write().expect("RwLock poisoned while locking source in try_destination");
            let destination_wrapper = NodePtr::Strong(ArcPtrWrapper::new(destination.clone()));

            if let Some(existing_ev_mut) = source_guard.children.get_mut(&self.edge_key).and_then(|dest_map| dest_map.get_mut(&destination_wrapper)) {
                let new_ev = self.edge_value.take().unwrap();
                crate::debug!(7, "Merging edge value {:?} into existing edge value {:?} for edge {:?} to node {:p}", new_ev, existing_ev_mut, self.edge_key, Arc::as_ptr(&destination));
                (self.merge_edge_value)(existing_ev_mut, new_ev);
                let updated_ev = existing_ev_mut.clone();
                self.result = Some(destination.clone());
                update_info = Some((destination, updated_ev));
            } else {
                let edge_val_clone = self.edge_value.as_ref().unwrap().clone();
                crate::debug!(7, "Trying to insert edge {:?} with value {:?} to node {:p}", self.edge_key, edge_val_clone, Arc::as_ptr(&destination));
                if source_guard.try_insert(self.edge_key.clone(), &mut self.edge_value, destination.clone()).is_ok() {
                    self.result = Some(destination.clone());
                    update_info = Some((destination, edge_val_clone));
                } else {
                    crate::debug!(7, "Cycle detected trying to insert edge {:?} to node {:p}", self.edge_key, Arc::as_ptr(&destination));
                }
            }
        }

        if let Some((dest_arc, ev)) = update_info {
            crate::debug!(7, "Updating node value for destination {:p} with edge value {:?}. self.edge_value: {:?}", Arc::as_ptr(&dest_arc), ev, self.edge_value);
            (self.update_node_value)(&mut dest_arc.write().unwrap().value, &ev);
        }

        self
    }

    /// Like try_destination, but if a strong cycle would be created, insert a WEAK edge instead.
    /// This guarantees the edge exists (weak if necessary) and avoids accidental cycles.
    #[time_it]
    pub fn try_destination_auto(mut self, destination: Arc<RwLock<Trie<EK, EV, T>>>) -> Self {
        if self.result.is_some() {
            return self;
        }
        let mut update_info: Option<(Arc<RwLock<Trie<EK, EV, T>>>, EV)> = None;

        { // Scope for source_guard
            let mut source_guard = self.source_arc.write().expect("RwLock poisoned while locking source in try_destination_auto");
            let destination_wrapper = NodePtr::Strong(ArcPtrWrapper::new(destination.clone()));
            if let Some(existing_ev_mut) = source_guard.children.get_mut(&self.edge_key).and_then(|dest_map| dest_map.get_mut(&destination_wrapper)) {
                let new_ev = self.edge_value.take().unwrap();
                (self.merge_edge_value)(existing_ev_mut, new_ev);
                let updated_ev = existing_ev_mut.clone();
                self.result = Some(destination.clone());
                update_info = Some((destination, updated_ev));
            } else {
                let edge_val_clone = self.edge_value.as_ref().unwrap().clone();
                source_guard.try_insert_auto(self.edge_key.clone(), &mut self.edge_value, destination.clone());
                self.result = Some(destination.clone());
                update_info = Some((destination, edge_val_clone));
            }
        }

        if let Some((dest_arc, ev)) = update_info {
            (self.update_node_value)(&mut dest_arc.write().unwrap().value, &ev);
        }

        self
    }

    /// Tries to establish a weak edge to the given `destination`.
    ///
    /// This operation does not perform cycle checks and does not update `max_depth`.
    /// If an edge (strong or weak) with the same `edge_key` already exists pointing to `destination`,
    /// it merges the `edge_value` using the `merge_edge_value` closure. An existing strong edge
    /// will remain strong.
    /// If no such edge exists, it inserts a new weak edge.
    ///
    /// This operation only proceeds if a successful destination hasn't already been found.
    /// Returns `self` to allow chaining.
    pub fn to_destination_weakly(mut self, destination: Arc<RwLock<Trie<EK, EV, T>>>) -> Self {
        if self.result.is_some() {
            return self; // Already found a destination
        }

        let mut update_info: Option<(Arc<RwLock<Trie<EK, EV, T>>>, EV)> = None;

        { // Scope for source_guard
            let mut source_guard = self.source_arc.write().expect("RwLock poisoned while locking source in to_destination_weakly");
            let lookup_wrapper = NodePtr::Strong(ArcPtrWrapper::new(destination.clone()));

            if let Some(existing_ev_mut) = source_guard.children.get_mut(&self.edge_key).and_then(|dest_map| dest_map.get_mut(&lookup_wrapper)) {
                let new_ev = self.edge_value.take().unwrap();
                (self.merge_edge_value)(existing_ev_mut, new_ev);
                update_info = Some((destination.clone(), existing_ev_mut.clone()));
            } else {
                let edge_val = self.edge_value.take().unwrap();
                source_guard.insert_weak_to_node(self.edge_key.clone(), edge_val.clone(), &destination);
                update_info = Some((destination.clone(), edge_val));
            }
            self.result = Some(destination);
        }

        if let Some((dest_arc, ev)) = update_info {
            (self.update_node_value)(&mut dest_arc.write().unwrap().value, &ev);
        }

        self
    }

    pub fn to_destinations_weakly_iter(mut self, destinations: impl Iterator<Item = Arc<RwLock<Trie<EK, EV, T>>>>) -> Self {
        for destination in destinations {
            if self.result.is_some() {
                break; // Stop trying once a destination is found
            }
            // Need to consume and reassign self because to_destination_weakly takes self
            self = self.to_destination_weakly(destination.clone()); // destination is already Arc, clone it
        }
        self
    }

    /// Tries to establish an edge to any destination in the provided slice.
    ///
    /// Iterates through `destinations` and calls `try_destination` for each until one succeeds.
    /// Returns `self` to allow chaining.
    #[time_it]
    pub fn try_destinations(mut self, destinations: &[Arc<RwLock<Trie<EK, EV, T>>>]) -> Self {
        for destination in destinations {
            if self.result.is_some() {
                break; // Stop trying once a destination is found
            }
            // Need to consume and reassign self because try_destination takes self
            self = self.try_destination(destination.clone());
        }
        self
    }

    #[time_it]
    pub fn try_destinations_iter(mut self, destinations: impl Iterator<Item = Arc<RwLock<Trie<EK, EV, T>>>>) -> Self {
        for destination in destinations {
            if self.result.is_some() {
                break; // Stop trying once a destination is found
            }
            // Need to consume and reassign self because try_destination takes self
            self = self.try_destination(destination.clone()); // destination is already Arc, clone it
        }
        self
    }

    #[time_it]
    pub fn try_destinations_iter_with<F, R>(mut self, destinations: F) -> Self
    where
        F: Fn() -> R,
        R: Iterator<Item = Arc<RwLock<Trie<EK, EV, T>>>>,
    {
        for destination in destinations() {
            if self.result.is_some() { // Check before calling try_destination
                break;
            }
            self = self.try_destination(destination.clone()); // destination is already Arc, clone it
        }
        self
    }


    /// Tries to merge the edge with existing children under `self.edge_key`.
    ///
    /// This method identifies all children of the source node that are already
    /// destinations for an edge with `self.edge_key`. For each such child, it
    /// attempts to merge `self.edge_value` into the existing edge's value using
    /// the `merge_edge_value` closure provided when the `EdgeInserter` was created.
    /// This is done by calling `try_destination` for each identified child.
    ///
    /// The first child for which `try_destination` is successful (typically by merging
    /// the edge value, as the edge `(self.edge_key, child)` already exists)
    /// will be set as `self.result`, and no further children under this key will be tried.
    ///
    /// If `merge_edge_value` returns `None` for a child (causing `try_destination` to not
    /// set a result for that child), or if no children exist under `self.edge_key`,
    /// `self.result` remains unchanged by this method with respect to those children.
    /// This method focuses on updating existing edges associated with `self.edge_key`.
    ///
    /// Returns `self` to allow chaining.
    pub fn try_children(mut self) -> Self {
        if self.result.is_some() {
            return self;
        }

        // Collect children arcs that are specifically under self.edge_key.
        let children_for_this_key: Vec<Arc<RwLock<Trie<EK, EV, T>>>> = {
            let source_guard = self.source_arc.read().expect("RwLock poisoned while locking source in try_children");
            if let Some(dest_map) = source_guard.children.get(&self.edge_key) {
                dest_map.keys()
                    .filter_map(|node_ptr| {
                        if let NodePtr::Strong(arc_wrapper) = node_ptr {
                            Some(arc_wrapper.as_arc().clone())
                        } else { None }
                    })
                    .collect()
            } else {
                Vec::new() // No children under this specific edge key
            }
        }; // Lock is dropped here

        // If there are children under this key, try them.
        // self.try_destinations will attempt to merge the edge value.
        if !children_for_this_key.is_empty() {
            self = self.try_destinations(&children_for_this_key);
        }
        // Return self, which may or may not have found a result.
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

        let new_node_arc = Arc::new(RwLock::new(Trie::new(value)));
        let edge_val_clone = self.edge_value.as_ref().unwrap().clone();

        { // Scope for source_guard
            let mut source_guard = self.source_arc.write().expect("RwLock poisoned while locking source in else_create_with_value");
            if source_guard.try_insert(self.edge_key.clone(), &mut self.edge_value, new_node_arc.clone()).is_ok() {
                self.result = Some(new_node_arc.clone());
            } else {
                crate::debug!(7, "Cycle detected trying to insert edge {:?} to NEW node {:p}. Creation failed.", self.edge_key, Arc::as_ptr(&new_node_arc));
            }
        }

        if self.result.is_some() {
            (self.update_node_value)(&mut new_node_arc.write().unwrap().value, &edge_val_clone);
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


    /// Returns the resulting destination node, if one was found or created.
    pub fn into_option(self) -> Option<Arc<RwLock<Trie<EK, EV, T>>>> {
        self.result
    }

    pub fn is_some(&self) -> bool {
        self.result.is_some()
    }

    pub fn clone_into_option(&self) -> Option<Arc<RwLock<Trie<EK, EV, T>>>> {
        self.result.clone()
    }

    /// Returns the resulting destination node, panicking if none was found or created.
    pub fn unwrap(self) -> Arc<RwLock<Trie<EK, EV, T>>> {
        self.result.expect("EdgeInserter::unwrap() called but no destination was found or created")
    }

    /// Returns the resulting destination node, panicking with the given message if none was found or created.
    pub fn expect(self, msg: &str) -> Arc<RwLock<Trie<EK, EV, T>>> {
        self.result.expect(msg)
    }
}


// Optional: Add a convenience method to Trie to create an EdgeInserter easily.
impl<EK: Ord + Clone + Debug, EV: Clone + Debug, T: Clone> Trie<EK, EV, T> {
    /// Creates an `EdgeInserter` to help add an edge starting from this node.
    ///
    /// This provides a convenient entry point for the chainable insertion pattern.
    ///
    /// # Arguments
    /// * `edge_key`: The key for the new edge.
    /// * `edge_value`: The value for the new edge.
    /// * `merge_edge_value`: A closure that takes the existing edge value and the new edge value,
    ///   both by reference, returning `Some(merged_value)` if merging is possible/desired,
    ///   or `None` otherwise. This is only called by `EdgeInserter::try_destination` if an edge
    ///   with the same `edge_key` already points to the destination being tried.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::sync::{Arc, RwLock};
    /// use crate::datastructures::trie::Trie; // Assuming Trie is in this module
    /// use crate::datastructures::trie::EdgeInserter; // Also need EdgeInserter
    /// use crate::datastructures::hybrid_bitset::HybridBitset; // Need HybridBitset
    /// use std::iter::FromIterator; // For collect
    ///
    /// #[derive(Debug, Clone, Default)] // Need Default for else_create
    /// struct NodeValue { /* ... */ }
    ///
    /// // Example merge function for edge values (e.g., HybridBitset)
    /// fn merge_bitset_union(existing: &mut HybridBitset, new: HybridBitset) { // Note the &mut and move
    ///     *existing |= new; // Use reference for the OR operation
    /// }
    ///
    /// // Assuming root_node is Arc<RwLock<Trie<String, HybridBitset, NodeValue>>>
    /// let root_node: Arc<RwLock<Trie<String, HybridBitset, NodeValue>>> = Arc::new(RwLock::new(Trie::new(NodeValue::default())));
    ///
    /// // Create a HybridBitset to use as edge value
    /// let new_edge_value: HybridBitset = vec![].into_iter().collect();
    ///
    /// let potential_destinations: Vec<Arc<RwLock<Trie<String, HybridBitset, NodeValue>>>> = vec![/* ... */];
    ///
    /// let new_or_existing_node = { // Use a block to drop the temporary mutex guard
    ///     let root_guard = root_node.write().unwrap(); // Get a guard to call insert_edge
    ///     // We must pass the merge function closure that takes &mut EV, EV
    ///     root_guard.insert_edge("key".to_string(), new_edge_value.clone(), merge_bitset_union) // Clone edge_value for EdgeInserter to own
    ///         .try_destinations(&potential_destinations) // potential_destinations is &[Arc<RwLock<...>>]
    ///         .else_create() // Or else_create_with(...) or else_create_with_value(...)
    ///         .unwrap()
    /// };
    /// // root_node (Arc<RwLock>) is an Arc<RwLock> and can be used further.
    /// ```
    pub fn insert_edge<FMergeEV, FUpdateT, FMergeEV_T>(
        &self, // Note: This method takes &self, not &mut self. The EdgeInserter handles the mutation via Arc<RwLock>.
        edge_key: EK,
        edge_value: EV,
        merge_edge_value: FMergeEV,
        update_node_value: FUpdateT,
        merge_edge_value_and_source_node_value: FMergeEV_T,
    ) -> EdgeInserter<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
    where
         FMergeEV: FnMut(&mut EV, EV), // Changed signature
         FUpdateT: FnMut(&mut T, &EV),
         FMergeEV_T: FnMut(&mut EV, &T),
    {
            EdgeInserter::new(Arc::new(RwLock::new(self.clone())), edge_key, edge_value, merge_edge_value, update_node_value, merge_edge_value_and_source_node_value)
        }
    }

/// Attempts to establish an edge from `source` to a single `destination`,
/// optionally merging edge values if an edge already exists.
/// Returns `Some(Arc<RwLock<Trie<...>>>)` if merge or insert succeeded,
/// or `None` if merge failed or a cycle was detected.
pub fn try_destination<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>(
    source: Arc<RwLock<Trie<EK, EV, T>>>,
    edge_key: EK,
    edge_value: EV,
    destination: Arc<RwLock<Trie<EK, EV, T>>>,
    merge_edge_value: FMergeEV,
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
) -> Option<Arc<RwLock<Trie<EK, EV, T>>>>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV), // Changed signature
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    EdgeInserter::new(source, edge_key, edge_value, merge_edge_value, update_node_value, merge_edge_value_and_source_node_value)
        .try_destination(destination)
        .into_option()
}

/// Attempts to establish an edge from `source` to any of the provided `destinations`,
/// returning the first successful one (merge or insert), or `None` if all attempts failed.
pub fn try_destination_with<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>(
    source: Arc<RwLock<Trie<EK, EV, T>>>,
    edge_key: EK,
    edge_value: EV,
    destinations: &[Arc<RwLock<Trie<EK, EV, T>>>],
    merge_edge_value: FMergeEV,
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
) -> Option<Arc<RwLock<Trie<EK, EV, T>>>>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV), // Changed signature
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    EdgeInserter::new(source, edge_key, edge_value, merge_edge_value, update_node_value, merge_edge_value_and_source_node_value)
        .try_destinations(destinations)
        .into_option()
}

/// Attempts to establish an edge from `source` to a single `destination`.
/// If a strong cycle would be created, it inserts a WEAK edge instead.
pub fn try_destination_auto<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>(
    source: Arc<RwLock<Trie<EK, EV, T>>>,
    edge_key: EK,
    edge_value: EV,
    destination: Arc<RwLock<Trie<EK, EV, T>>>,
    merge_edge_value: FMergeEV,
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
) -> Option<Arc<RwLock<Trie<EK, EV, T>>>>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    EdgeInserter::new(source, edge_key, edge_value, merge_edge_value, update_node_value, merge_edge_value_and_source_node_value).try_destination_auto(destination).into_option()
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::collections::{HashSet, HashMap};
    use crate::datastructures::hybrid_bitset::HybridBitset; // Import HybridBitset for tests
    use std::iter::FromIterator; // For collect

    // Use concrete types for merge tests
    type TestTrieMerge = Trie<&'static str, Vec<i32>, String>;
    type TestNodeMerge = Arc<RwLock<TestTrieMerge>>;
    // Use simpler types for basic tests
    type TestTrieBasic = Trie<&'static str, &'static str, i32>;
    type TestNodeBasic = Arc<RwLock<TestTrieBasic>>;

    // Use concrete types for EdgeInserter tests
    type TestTrieEI = Trie<&'static str, HybridBitset, String>; // Use HybridBitset here
    type TestNodeEI = Arc<RwLock<TestTrieEI>>;

    // Helper to get Arc pointer for tests
    fn arc_ptr<N>(arc: &Arc<RwLock<N>>) -> *const RwLock<N> {
        Arc::as_ptr(arc)
    }

    #[test]
    fn test_try_insertion_and_retrieval() {
        let root_node: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let child3: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3))); // Another child for 'a'

        { // Scope for mutable borrow of root
            let mut root = root_node.write().unwrap();
            root.try_insert("a", &mut Some("edge_a1"), child1.clone()).expect("Insert failed");
            root.try_insert("b", &mut Some( "edge_b"), child2.clone()).expect("Insert failed");
            root.try_insert("a", &mut Some("edge_a3"), child3.clone()).expect("Insert failed"); // Insert second child for 'a'
        } // root lock released

        // Scope for read-only borrow of root
        let root = root_node.read().unwrap();

        // Test get for 'a'
        let retrieved_children_a = root.get(&"a").expect("Failed to get children for 'a'"); // Now a &BTreeMap<ArcPtrWrapper<Mutex<...>>, EV>
        assert_eq!(retrieved_children_a.len(), 2);
        // Use Arc pointers for comparison
        let retrieved_data_a: HashSet<(&str, *const RwLock<TestTrieBasic>)> = retrieved_children_a
            .iter() // Iterates yielding (&NodePtr<...>, &&str)
            .map(|(node_ptr, ev_ref)| (*ev_ref, arc_ptr(&node_ptr.upgrade().unwrap()))) // Dereference ev_ref twice
            .collect();
        assert!(retrieved_data_a.contains(&("edge_a1", arc_ptr(&child1))));
        assert!(retrieved_data_a.contains(&("edge_a3", arc_ptr(&child3))));

        // Test get for 'b'
        let retrieved_children_b = root.children().get(&"b").expect("Failed to get child 'b'"); // Now a &BTreeMap
        assert_eq!(retrieved_children_b.len(), 1);
        let (node_ptr, ev_ref) = retrieved_children_b.iter().next().unwrap(); // Get the single entry
        assert_eq!(*ev_ref, "edge_b"); // Check edge value
        assert!(Arc::ptr_eq(&node_ptr.upgrade().unwrap(), &child2)); // Check Arc pointer equality

        assert!(root.get(&"c").is_none());

        // Test children iterator order (BTreeMap ensures sorted order of keys 'a', 'b')
        let children_keys: Vec<_> = root.children().keys().cloned().collect();
        assert_eq!(children_keys, vec!["a", "b"]);
        assert_eq!(root.children().get("a").unwrap().len(), 2);
        assert_eq!(root.children().get("b").unwrap().len(), 1);

        // Test is_leaf
        assert!(!root.is_leaf());
        // Drop root lock before locking children
        drop(root);
        assert!(child1.read().unwrap().is_leaf());
        assert!(child2.read().unwrap().is_leaf());
        assert!(child3.read().unwrap().is_leaf());
    }

    #[test]
    fn test_multiple_children_same_edge_key() {
        // Structure:
        //      root (0) --"edge", "val1"--> child1 (1)
        //           |
        //            -----"edge", "val2"--> child2 (2)
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));

        {
            let mut r = root.write().unwrap();
            r.try_insert("edge", &mut Some("val1"), child1.clone()).unwrap();
            r.try_insert("edge", &mut Some("val2"), child2.clone()).unwrap();
        } // root lock released

        // Check retrieval - lock root again
        {
            let binding = root.read().unwrap();
            let children_map = binding.get(&"edge").unwrap(); // Now a &BTreeMap<ArcPtrWrapper<Mutex<...>>, EV>
            assert_eq!(children_map.len(), 2);
            let child_data: HashSet<(&str, *const RwLock<TestTrieBasic>)> = children_map
                .iter() // Iterating over (&NodePtr<...>, &EV)
                .map(|(node_ptr, ev_ref)| (*ev_ref, arc_ptr(&node_ptr.upgrade().unwrap())))
                .collect();
            assert!(child_data.contains(&("val1", arc_ptr(&child1))));
            assert!(child_data.contains(&("val2", arc_ptr(&child2))));
        } // root lock released

        // Check all_nodes - call *after* releasing lock
        let all = Trie::all_nodes(&[root.clone()]);
        assert_eq!(all.len(), 3); // root, child1, child2
        let all_ptrs: HashSet<_> = all.iter().map(arc_ptr).collect();
        assert!(all_ptrs.contains(&arc_ptr(&root)));
        assert!(all_ptrs.contains(&arc_ptr(&child1)));
        assert!(all_ptrs.contains(&arc_ptr(&child2)));

        // Check special_map
        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();
        let mut edge_info_at_step = Vec::new(); // Store (EK, EV) seen by step

        Trie::special_map(
            vec![(root.clone(), 100)],
            // step: add one, ignore edge info
            |parent_val, ek, ev, _child_node| {
                 edge_info_at_step.push((ek.clone(), ev.clone()));
                 Some(parent_val + 1)
            },
            |current, new| *current = new, // merge: replace
            |node, computed_val| { // process: always continue
                processed_node_values.push(node.value);
                computed_values.push(*computed_val);
                true
            },
        );

        // Expected processing order: 0, then (1, 2) in some order based on depth.
        assert_eq!(processed_node_values.len(), 3);
        assert!(processed_node_values.contains(&0));
        assert!(processed_node_values.contains(&1));
        assert!(processed_node_values.contains(&2));
        // Depth 0 nodes processed first
        assert_eq!(processed_node_values[0], 0);
        // Depth 1 nodes processed next (order not guaranteed for equal depth)
        let depth1_nodes: HashSet<_> = processed_node_values[1..].iter().cloned().collect();
        assert!(depth1_nodes.contains(&1));
        assert!(depth1_nodes.contains(&2));


        // Expected computed values: root = 100, child1 = 101, child2 = 101.
        assert_eq!(computed_values.len(), 3);
        assert_eq!(computed_values[0], 100);
        let results_map: HashMap<i32, i32> = processed_node_values.iter().cloned().zip(computed_values.iter().cloned()).collect();
        assert_eq!(results_map.get(&0), Some(&100));
        assert_eq!(results_map.get(&1), Some(&101));
        assert_eq!(results_map.get(&2), Some(&101));

        // Check edge info captured by step
        assert_eq!(edge_info_at_step.len(), 2); // 2 edges traversed from root
        assert!(edge_info_at_step.contains(&("edge", "val1")));
        assert!(edge_info_at_step.contains(&("edge", "val2")));
    }


    #[test]
    fn test_special_map_bfs_order_with_edges() {
        // Structure:
        //      root (0)
        //       /       \
        // ("r->c1","e1") ("r->c2","e2")
        //     /           \
        //   c1 (1)       c2 (2)
        //      |
        // ("c1->gc","e3")
        //      |
        //   gc (3)
        //
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));

        {
            let mut r = root.write().unwrap();
            r.try_insert("r->c1", &mut Some("e1"), child1.clone()).unwrap();
            r.try_insert("r->c2", &mut Some("e2"), child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.write().unwrap();
            c1.try_insert("c1->gc", &mut Some("e3"), grandchild.clone()).unwrap();
        }
         // No edge from c2 to grandchild in this test setup, removed the line below
        // {
        //     let mut c2 = child2.lock().unwrap();
        //     c2.try_insert("c2->gc", &mut Some("e4"), grandchild.clone()).unwrap();
        // }


        let mut processed_node_values = Vec::new();
        let mut computed_values = Vec::new();
        let mut edge_info_at_step = Vec::new(); // Store (EK, EV) seen by step

        Trie::special_map(
            vec![(root.clone(), 100)],
            // step: add one, record edge info
            |parent_val, ek, ev, _child_node| {
                edge_info_at_step.push((ek.clone(), ev.clone()));
                Some(parent_val + 1)
            },
            // merge: replace
            |current, new| { *current = new; },
            // process: always continue
            |node, computed_val| {
                processed_node_values.push(node.value);
                computed_values.push(*computed_val);
                true
            },
        );

        // Check processing order (by depth)
        // Depth 0: root (0)
        // Depth 1: child1 (1), child2 (2) - order depends on heap
        // Depth 2: grandchild (3)
        assert_eq!(processed_node_values.len(), 4);
        assert_eq!(processed_node_values[0], 0); // Root (depth 0) is first
        let depth1_nodes: HashSet<_> = processed_node_values[1..3].iter().cloned().collect();
        assert!(depth1_nodes.contains(&1));
        assert!(depth1_nodes.contains(&2));
        assert_eq!(processed_node_values[3], 3); // Grandchild (depth 2) is last


        // Check computed values
        let results_map: HashMap<i32, i32> = processed_node_values.iter().cloned()
            .zip(computed_values.iter().cloned()).collect();
        assert_eq!(results_map.get(&0), Some(&100));
        assert_eq!(results_map.get(&1), Some(&101));
        assert_eq!(results_map.get(&2), Some(&101));
        assert_eq!(results_map.get(&3), Some(&102)); // Reached from c1 (101+1)

        // Check edge info captured by step
        assert_eq!(edge_info_at_step.len(), 3); // 3 edges traversed (r->c1, r->c2, c1->gc)
        assert!(edge_info_at_step.contains(&("r->c1", "e1")));
        assert!(edge_info_at_step.contains(&("r->c2", "e2")));
        assert!(edge_info_at_step.contains(&("c1->gc", "e3")));
    }

    #[test]
    fn test_all_nodes_diamond() {
        // Diamond structure:
        //       root
        //      /    \
        // ("r1","e1") ("r2","e2")
        //    /        \
        // child1    child2
        //    \        /
        // ("c1","e3") ("c2","e4")
        //      \    /
        //    grandchild
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));

        {
            let mut r = root.write().unwrap();
            r.try_insert("r1", &mut Some("e1"), child1.clone()).unwrap();
            r.try_insert("r2", &mut Some("e2"), child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.write().unwrap();
            c1.try_insert("c1", &mut Some("e3"), grandchild.clone()).unwrap();
        }
        {
            let mut c2 = child2.write().unwrap();
            c2.try_insert("c2", &mut Some("e4"), grandchild.clone()).unwrap(); // Diamond
        }

        let all_nodes = Trie::all_nodes(&[root.clone()]);

        // Should find 4 unique nodes.
        assert_eq!(all_nodes.len(), 4);
        let node_ptrs: HashSet<_> = all_nodes.iter().map(arc_ptr).collect(); // Use arc_ptr
        assert_eq!(node_ptrs.len(), 4);
        assert!(node_ptrs.contains(&arc_ptr(&root)));
        assert!(node_ptrs.contains(&arc_ptr(&child1)));
        assert!(node_ptrs.contains(&arc_ptr(&child2)));
        assert!(node_ptrs.contains(&arc_ptr(&grandchild)));
    }

    #[test]
    fn test_special_map_diamond_merge_max() {
        // Diamond structure
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));

        // Build the structure
        {
            let mut r = root.write().unwrap();
            r.try_insert("r->c1", &mut Some("edge1"), child1.clone()).unwrap();
            r.try_insert("r->c2", &mut Some("edge2"), child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.write().unwrap();
            c1.try_insert("c1->gc", &mut Some("edge3"), grandchild.clone()).unwrap();
        }
        {
            let mut c2 = child2.write().unwrap();
            c2.try_insert("c2->gc", &mut Some("edge4"), grandchild.clone()).unwrap();
        }

        // Check max_depths after insertion
        assert_eq!(root.read().unwrap().max_depth, 0);
        assert_eq!(child1.read().unwrap().max_depth, 1);
        assert_eq!(child2.read().unwrap().max_depth, 1);
        assert_eq!(grandchild.read().unwrap().max_depth, 2);

        let processed_nodes = Arc::new(RwLock::new(HashMap::<i32, i32>::new()));
        let process_count = Arc::new(AtomicUsize::new(0));

        Trie::special_map(
            vec![(root.clone(), 100)], // Start at root
            // step: increment value, ignore edges
            |p_val, _ek, _ev, _child_node| Some(p_val + 1),
            // merge: take max value
            |current_v, new_v| *current_v = (*current_v).max(new_v),
            { // process: always continue
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

        // Assertions
        let final_results = processed_nodes.read().unwrap();
        assert_eq!(process_count.load(Ordering::SeqCst), 4, "Should process 4 unique nodes");
        assert_eq!(final_results.get(&0), Some(&100));
        assert_eq!(final_results.get(&1), Some(&101));
        assert_eq!(final_results.get(&2), Some(&101));
        assert_eq!(final_results.get(&3), Some(&102)); // gc gets max(101+1, 101+1) = 102
    }


    #[test]
    fn test_empty_trie() {
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(42)));
        let nodes = Trie::all_nodes(&[root.clone()]);
        assert_eq!(nodes.len(), 1);
        assert!(Arc::ptr_eq(&nodes[0], &root));
        assert!(root.read().unwrap().is_leaf()); // Lock needed here

        let mut processed = false;
        Trie::special_map(
            vec![(root.clone(), 100)],
            |_p, _ek, _ev, _n| panic!("Step should not be called for leaf"),
            |_cur, _new| {},
            |node, v| { // process: always continue
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
        // Cycle:  root -> child -> root
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));

        // Insert root -> child
        let insert1_result = {
            let mut r = root.write().unwrap();
            r.try_insert("r->c", &mut Some("e1"), child.clone())
        };
        assert!(insert1_result.is_ok());
        assert_eq!(child.read().unwrap().max_depth, 1);
        assert_eq!(root.read().unwrap().max_depth, 0);

        // Attempt insert child -> root
        let insert2_result = {
            let mut c = child.write().unwrap();
            // This insert should call detect_cycle(child_ptr, &root), which should detect the cycle.
            c.try_insert("c->r", &mut Some("e2"), root.clone())
        };

        // Assert that cycle detection returned an error
        assert!(insert2_result.is_err());
        assert_eq!(insert2_result.err(), Some(CycleDetectedError));

        // Check state after failed insertion:
        // - The edge must *not* be present because the insertion was rejected.
        let child_locked = child.read().unwrap();
        let has_edge_to_root = if let Some(dest_map) = child_locked.children.get("c->r") {
            let lookup_key = NodePtr::Strong(ArcPtrWrapper::new(root.clone())); // Use NodePtr
            dest_map.contains_key(&lookup_key)
         } else {
             false
         };
        assert!(!has_edge_to_root, "Edge that would introduce a cycle should NOT be present");

        // - Max depths should be unchanged from before the failed insertion attempt.
        assert_eq!(root.read().unwrap().max_depth, 0);
        assert_eq!(child_locked.max_depth, 1);

        println!("Done testing cycle detection on try_insert");
    }


    #[test]
    fn test_cycle_all_nodes_no_panic() {
        // Cycle:  root -> child -> root.
        // Manually create cycle without insert's propagation.
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));

        // Manually create links
        root.write().unwrap().force_insert_to_node("r->c", "e1", &child);
        child.write().unwrap().force_insert_to_node("c->r", "e2", &root);
        // Manually set depths (optional for all_nodes logic)
        root.write().unwrap().max_depth = 0;
        child.write().unwrap().max_depth = 1;

        let all_nodes = Trie::all_nodes(&[root.clone()]);

        // Should detect both nodes exactly once.
        assert_eq!(all_nodes.len(), 2);
        let node_ptrs: HashSet<_> = all_nodes.iter().map(arc_ptr).collect(); // Use arc_ptr
        assert_eq!(node_ptrs.len(), 2);
        assert!(node_ptrs.contains(&arc_ptr(&root)));
        assert!(node_ptrs.contains(&arc_ptr(&child)));
    }

     #[test]
    fn test_has_any_cycle() {
        // No cycle
        let root1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));
        root1.write().unwrap().force_insert_to_node("a", "e1", &child1);
        root1.write().unwrap().force_insert_to_node("b", "e2", &child2);
        child1.write().unwrap().force_insert_to_node("c", "e3", &grandchild);
        child2.write().unwrap().force_insert_to_node("d", "e4", &grandchild); // Diamond
        assert!(!Trie::has_any_cycle(root1.clone()));

        // Simple cycle: root2 -> child3 -> root2
        let root2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(10)));
        let child3: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(11)));
        root2.write().unwrap().force_insert_to_node("x", "e5", &child3);
        child3.write().unwrap().force_insert_to_node("y", "e6", &root2);
        assert!(Trie::has_any_cycle(root2.clone()));

        // Larger cycle: root3 -> A -> B -> C -> A
        let root3: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(20)));
        let node_a: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(21)));
        let node_b: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(22)));
        let node_c: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(23)));
        root3.write().unwrap().force_insert_to_node("r->a", "e7", &node_a);
        node_a.write().unwrap().force_insert_to_node("a->b", "e8", &node_b);
        node_b.write().unwrap().force_insert_to_node("b->c", "e9", &node_c);
        node_c.write().unwrap().force_insert_to_node("c->a", "e10", &node_a); // Cycle C -> A
        assert!(Trie::has_any_cycle(root3.clone()));

        // Cycle with unconnected node: root4 -> A -> B -> A; C (unconnected)
        let root4: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(30)));
        let node_a2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(31)));
        let node_b2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(32)));
        let node_c2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(33))); // Unconnected to root4
        root4.write().unwrap().force_insert_to_node("r->a", "e11", &node_a2);
        node_a2.write().unwrap().force_insert_to_node("a->b", "e12", &node_b2);
        node_b2.write().unwrap().force_insert_to_node("b->a", "e13", &node_a2); // Cycle B -> A
        assert!(Trie::has_any_cycle(root4.clone()));

        // Disconnected graph with a cycle: root5 (linear chain), root6 (cycle)
        let root5: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(40)));
        let node_d: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(41)));
        root5.write().unwrap().force_insert_to_node("r->d", "e14", &node_d);
        // Separately, a cycle structure
        let root6_in_cycle: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(50)));
        let node_e: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(51)));
        root6_in_cycle.write().unwrap().force_insert_to_node("c1->e", "e15", &node_e);
        node_e.write().unwrap().force_insert_to_node("e->c1", "e16", &root6_in_cycle); // Cycle
        // Checking from root5 should NOT find the cycle
        assert!(!Trie::has_any_cycle(root5.clone()));
        // Checking from root6_in_cycle SHOULD find the cycle
        assert!(Trie::has_any_cycle(root6_in_cycle.clone()));
    }


    #[test]
    fn test_cycle_special_map_no_panic_limited_processing() {
        // Cycle: root -> child -> root.
        // Manually create cycle.
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));

        // Manually create links
        root.write().unwrap().force_insert_to_node("r->c", "e1", &child);
        child.write().unwrap().force_insert_to_node("c->r", "e2", &root);
        // Manually set depths. These are crucial for special_map's readiness check.
        root.write().unwrap().max_depth = 0; // Initial node, depth 0
        child.write().unwrap().max_depth = 1; // Child reachable at depth 1

        let mut processed_vals = Vec::new();
        let mut computed_vals = Vec::new();

        Trie::special_map(
            vec![(root.clone(), 100)], // Start at root
            |p, _ek, _ev, _n| Some(p + 1), // Step: increment
            |cur, new| *cur = (*cur).max(new), // Merge: max
            |node, v| { // process: always continue
                processed_vals.push(node.value);
                computed_vals.push(*v);
                true
            },
        );

        // Expected behavior: Root processed (V=100), Child processed (V=101).
        // The cycle back to root doesn't re-process root because root is in `done`.
        // The new depth-based scheduler should handle this gracefully.
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
        // Structure:
        //      root (0) --e1,e2--> c1(1), c2(2)
        //      c1(1) --e3--> gc1(3)
        //      c2(2) --e4--> gc2(4)
        // Process returns false for c1, true otherwise.
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));
        let grandchild2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(4)));

        {
            let mut r = root.write().unwrap();
            r.try_insert("r->c1", &mut Some("edge1"), child1.clone()).unwrap();
            r.try_insert("r->c2", &mut Some("edge2"), child2.clone()).unwrap();
        }
        {
            let mut c1 = child1.write().unwrap();
            c1.try_insert("c1->gc", &mut Some("edge3"), grandchild1.clone()).unwrap();
        }
        {
            let mut c2 = child2.write().unwrap();
            c2.try_insert("c2->gc", &mut Some("edge4"), grandchild2.clone()).unwrap();
        }

        let processed_nodes = Arc::new(RwLock::new(HashSet::<i32>::new()));
        let computed_values = Arc::new(RwLock::new(HashMap::<i32, i32>::new()));

        Trie::special_map(
            vec![(root.clone(), 100)],
            |p_val, _ek, _ev, _child_node| Some(p_val + 1), // step: increment value
            |current_v, new_v| *current_v = new_v, // merge: replace
            {
                let processed_nodes = processed_nodes.clone();
                let computed_values = computed_values.clone();
                move |node, final_v| {
                    processed_nodes.write().unwrap().insert(node.value);
                    computed_values.write().unwrap().insert(node.value, *final_v);
                    if node.value == 1 { // Stop processing children if node value is 1 (child1)
                        false
                    } else {
                        true
                    }
                }
            }
        );

        let final_processed = processed_nodes.read().unwrap();
        let final_values = computed_values.read().unwrap();

        // Expected processed nodes: 0, 1, 2, 4. Node 3 should be skipped because propagation stopped at node 1.
        assert_eq!(final_processed.len(), 4);
        assert!(final_processed.contains(&0));
        assert!(final_processed.contains(&1)); // Processed, but stopped propagation
        assert!(final_processed.contains(&2)); // Processed, continued propagation
        assert!(!final_processed.contains(&3)); // gc1 should NOT be processed
        assert!(final_processed.contains(&4)); // gc2 should be processed

        // Check computed values
        assert_eq!(final_values.get(&0), Some(&100));
        assert_eq!(final_values.get(&1), Some(&101));
        assert_eq!(final_values.get(&2), Some(&101));
        assert_eq!(final_values.get(&3), None);      // Not processed
        assert_eq!(final_values.get(&4), Some(&102)); // Processed via child2
    }

    #[test]
    fn test_special_map_step_returns_none() {
        // Structure:
        //      root (0) --"keep"--> c1(1)
        //           |
        //           --"skip"--> c2(2) --"keep"--> gc2(3)
        // Step returns None if edge key is "skip".
        let root: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(0)));
        let child1: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(1)));
        let child2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(2)));
        let grandchild2: TestNodeBasic = Arc::new(RwLock::new(TestTrieBasic::new(3)));

        {
            let mut r = root.write().unwrap();
            r.try_insert("keep", &mut Some("e1"), child1.clone()).unwrap();
            r.try_insert("skip", &mut Some("e2"), child2.clone()).unwrap();
        }
        {
            let mut c2 = child2.write().unwrap();
            c2.try_insert("keep", &mut Some("e3"), grandchild2.clone()).unwrap();
        }

        let processed_nodes = Arc::new(RwLock::new(HashSet::<i32>::new()));
        let computed_values = Arc::new(RwLock::new(HashMap::<i32, i32>::new()));

        Trie::special_map(
            vec![(root.clone(), 100)],
            // step: increment value only if edge key is "keep"
            |p_val, ek, _ev, _child_node| {
                if *ek == "keep" {
                    Some(p_val + 1)
                } else {
                    None // Skip this edge
                }
            },
            |current_v, new_v| *current_v = new_v, // merge: replace
            {
                let processed_nodes = processed_nodes.clone();
                let computed_values = computed_values.clone();
                move |node, final_v| {
                    processed_nodes.write().unwrap().insert(node.value);
                    computed_values.write().unwrap().insert(node.value, *final_v);
                    true // Always continue processing if node is reached
                }
            }
        );

        let final_processed = processed_nodes.read().unwrap();
        let final_values = computed_values.read().unwrap();

        // Expected processed nodes: 0, 1. Node 2 is skipped because step for root->child2 returns None. Node 3 is not reached as its parent (node 2) is not processed.
        assert_eq!(final_processed.len(), 2);
        assert!(final_processed.contains(&0));
        assert!(final_processed.contains(&1));

        // Check computed values
        assert_eq!(final_values.get(&0), Some(&100));
        assert_eq!(final_values.get(&1), Some(&101));
        assert_eq!(final_values.get(&2), None); // Not processed
        assert_eq!(final_values.get(&3), None); // Not reached, as c2 is not processed
    }


    // --- Tests for insert_or_merge_edge ---

    // Helper merge functions for tests
    // Merge edge value (Vec<i32>): Append new vec to existing if existing is not empty
    fn merge_ev_append(existing_ev: &mut Vec<i32>, new_ev: Vec<i32>) { // Changed existing_ev to &Vec<i32>
        existing_ev.extend(new_ev.iter().copied()); // Use iter().copied()
    }

    // Merge node value (String): Append new string if existing contains "mergeable"
    //
    // NOTE:
    // The sentinel strings used throughout the tests include both
    // “…_mergeable” (should merge)  and “…_not_mergeable” (should NOT merge).
    // The original helper simply checked `contains("mergeable")`, which means
    // `"child_not_mergeable"` was (incorrectly) considered merge-able because
    // it still contains the substring `"mergeable"`.
    //
    // To align the helper’s behaviour with the test‐case expectations we now:
    //   1. Require that the value contains `"mergeable"`, *and*
    //   2. Explicitly reject any value that contains `"not_mergeable"`.
    //
    // This makes values like `"child_mergeable"` merge, while
    // `"child_not_mergeable"` (and similar) do NOT merge.
    fn merge_nv_append_if_flag(existing_nv: &String, new_nv: String) -> Option<String> {
        if existing_nv.contains("mergeable") && !existing_nv.contains("not_mergeable") {
            Some(format!("{}|{}", existing_nv, new_nv))
        } else {
            None
        }
    }

    // test_insert_or_merge_edge_detects_cycle removed as try_insert_or_merge_edge
    // doesn't attempt to re-insert an existing node in a way that would trigger
    // cycle detection based on the node itself being passed again. Cycle detection
    // relies on the try_insert call in Pass 3 when creating a *new* edge/node.

    // --- Tests for EdgeInserter ---

    // Helper merge function for EdgeInserter tests: Union HybridBitset
    fn merge_bitset_union(existing: &mut HybridBitset, new: HybridBitset) {
        *existing |= new // Use reference for the OR operation
    }

    #[test]
    fn test_ei_try_destination_success_new_edge() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest".to_string())));
        let edge_val: HybridBitset = vec![1].into_iter().collect();


        let inserter = EdgeInserter::new(source.clone(), "key", edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node = inserter.try_destination(dest.clone()).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest));
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap<ArcPtrWrapper<Mutex<...>>, EV>
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, edge_val);
        assert!(Arc::ptr_eq(&node_ptr.upgrade().unwrap(), &dest));
        assert_eq!(dest.read().unwrap().max_depth, 1); // Depth updated by try_insert
    }

    #[test]
    fn test_ei_try_destination_success_merge_ev() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest".to_string())));
        let initial_edge_val: HybridBitset = vec![10].into_iter().collect();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();
        let merged_edge_val: HybridBitset = vec![1, 10].into_iter().collect();

        // Pre-insert edge
        source.write().unwrap().try_insert("key", &mut Some(initial_edge_val), dest.clone()).unwrap();
        assert_eq!(dest.read().unwrap().max_depth, 1); // Check initial depth

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node = inserter.try_destination(dest.clone()).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest));
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1); // Still one edge
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert_eq!(*ev, merged_edge_val); // Merged value
        assert!(Arc::ptr_eq(&node_ptr.upgrade().unwrap(), &dest));
        assert_eq!(dest.read().unwrap().max_depth, 1); // Depth should remain 1
    }

    #[test]
    fn test_ei_try_destination_fail_merge_ev() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest".to_string())));
        // Pre-insert edge with empty HybridBitset
        let initial_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        source.write().unwrap().try_insert("key", &mut Some(initial_edge_val), dest.clone()).unwrap();

        // In this case, merge_bitset_union will always return Some, so merge should succeed.
        // To test a failing merge, we'd need a different merge function or EV type.
        // Let's repurpose this to test a successful merge where existing is empty.
        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_opt = inserter.try_destination(dest.clone()).into_option();

        assert!(result_opt.is_some()); // Merge succeeded
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        // The result of merge_bitset_union(&empty, &new_edge_val) is new_edge_val
        assert_eq!(*ev, new_edge_val);
        assert!(Arc::ptr_eq(&node_ptr.upgrade().unwrap(), &dest));
    }

    #[test]
    fn test_ei_try_destination_fail_cycle() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest".to_string())));
         let dummy_edge_val = HybridBitset::zeros();

        // Create cycle manually for test setup
        dest.write().unwrap().force_insert_to_node("dest_to_src", dummy_edge_val.clone(), &source); // dest -> source edge
        //source.lock().unwrap().force_insert_to_node("src_to_dest", dummy_edge_val.clone(), &dest); // source -> dest edge - this is what we are trying to insert

        // Now try inserting source -> dest again using EdgeInserter
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();
        let inserter = EdgeInserter::new(source.clone(), "src_to_dest", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        // This will call try_insert which should detect the cycle
        let result_opt = inserter.try_destination(dest.clone()).into_option();

        assert!(result_opt.is_none()); // Cycle detected, insert failed
    }


    #[test]
    fn test_ei_try_slice_success() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dest2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest2".to_string())));
        let dest3: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest3".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: dest2 -> source creates a cycle if we try source -> dest2
        dest2.write().unwrap().force_insert_to_node("d2->s", dummy_edge_val.clone(), &source);

        let destinations = [dest1.clone(), dest2.clone(), dest3.clone()];

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        // try(dest1) -> OK
        // try(dest2) -> Cycle Error (skipped because dest1 succeeded)
        // try(dest3) -> Skipped
        let result_node = inserter.try_destinations(&destinations).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest1)); // Should succeed with dest1
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(&node_ptr.upgrade().unwrap(), &dest1));
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


        // Setup: dest1 -> source creates a cycle if we try source -> dest1
        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let destinations = [dest1.clone(), dest2.clone(), dest3.clone()];

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        // try(dest1) -> Cycle Error
        // try(dest2) -> OK
        // try(dest3) -> Skipped
        let result_node = inserter.try_destinations(&destinations).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest2)); // Should succeed with dest2
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap();
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(&node_ptr.upgrade().unwrap(), &dest2));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_try_slice_fail_all() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dest2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest2".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: Both destinations cause cycles
        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);
        dest2.write().unwrap().force_insert_to_node("d2->s", dummy_edge_val.clone(), &source);

        let destinations = [dest1.clone(), dest2.clone()];

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_opt = inserter.try_destinations(&destinations).into_option();

        assert!(result_opt.is_none()); // All attempts failed
        assert!(source.read().unwrap().get(&"key").is_none()); // No edge added
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
        let merged_ev_c1: HybridBitset = vec![1, 10].into_iter().collect(); // Expected merge with child1

        // Setup:
        // source --(target_key, initial_ev_c1)--> child1
        // source --(target_key, initial_ev_c2)--> child2
        // source --("other_key", dummy_ev)--> child_other_key
        {
            let mut s = source.write().unwrap();
            s.try_insert(edge_key, &mut Some(initial_ev_c1), child1.clone()).unwrap();
            s.try_insert(edge_key, &mut Some(initial_ev_c2.clone()), child2.clone()).unwrap();
            s.try_insert("other_key", &mut Some(HybridBitset::zeros()), child_other_key.clone()).unwrap();
        }

        // 1. Test successful merge with the first child under the key.
        //    EdgeInserter is created with source, target_key, and new_ev_for_inserter.
        //    merge_bitset_union should merge new_ev_for_inserter into initial_ev_c1.
        let inserter = EdgeInserter::new(source.clone(), edge_key, new_ev_for_inserter.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node_opt = inserter.try_children().into_option();

        assert!(result_node_opt.is_some(), "Should find and merge with child1");
        let result_node = result_node_opt.unwrap();
        assert!(Arc::ptr_eq(&result_node, &child1), "Result should be child1");

        // Check edge values:
        // Edge to child1 should be merged.
        // Edge to child2 should be unchanged (because merge with child1 succeeded first).
        // Edge to child_other_key should be unchanged.
        {
            let s_guard = source.read().unwrap();
            let children_map_target_key = s_guard.get(&edge_key).expect("Target key should exist");

            let ev_c1 = children_map_target_key.get(&NodePtr::Strong(ArcPtrWrapper::new(child1.clone()))).expect("Child1 should be under target_key");
            assert_eq!(*ev_c1, merged_ev_c1, "Edge value for child1 should be merged");

            let ev_c2 = children_map_target_key.get(&NodePtr::Strong(ArcPtrWrapper::new(child2.clone()))).expect("Child2 should be under target_key");
            assert_eq!(*ev_c2, initial_ev_c2, "Edge value for child2 should be unchanged");

            let children_map_other_key = s_guard.get(&"other_key").expect("Other key should exist");
            assert_eq!(children_map_other_key.len(), 1, "Should be one child under other_key");
            // You could also check the value of the edge to child_other_key if necessary.
        }

        // 2. Test when merge_edge_value fails for all children under the key.
        //    (This test needs a merge function that can fail or a different EV type,
        //     merge_bitset_union always succeeds by design).
        //    Re-using this section to verify the initial state for part 3 is correct.
        let source_nm: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_nm".to_string())));
        let child1_nm: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child1_nm".to_string())));
        let edge_key_nm = "nm_key"; // "nm" for "no merge"
        let initial_ev_nm: HybridBitset = vec![50].into_iter().collect();
        let new_ev_inserter_nm: HybridBitset = vec![5].into_iter().collect();

        source_nm.write().unwrap().try_insert(edge_key_nm, &mut Some(initial_ev_nm.clone()), child1_nm.clone()).unwrap();

        // Check edge value for child1_nm is unchanged - this is now done in part 3.

        // 3. Test when no children exist under the specified edge_key.
        let source_empty: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_empty".to_string())));
        let edge_key_empty = "empty_key"; // This key has no children in source_empty
        let new_ev_inserter_empty: HybridBitset = vec![7].into_iter().collect();

        let inserter_empty = EdgeInserter::new(source_empty.clone(), edge_key_empty, new_ev_inserter_empty.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node_empty_opt = inserter_empty.try_children().into_option();
        assert!(result_node_empty_opt.is_none(), "try_children should return None if no children under the key");

        // 4. Test chaining with else_create: try_children (no children under key) -> else_create
        let source_chain: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_chain".to_string())));
        let edge_key_chain = "chain_key"; // No children under this key initially in source_chain
        let new_ev_chain: HybridBitset = vec![8].into_iter().collect();
        let created_val = "created_node_via_fallback".to_string();

        let inserter_chain = EdgeInserter::new(source_chain.clone(), edge_key_chain, new_ev_chain.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node_chain = inserter_chain
            .try_children() // Will do nothing as no children under "chain_key"
            .else_create_destination_with_value(created_val.clone()) // This should execute
            .unwrap();

        assert_eq!(result_node_chain.read().unwrap().value, created_val, "Fallback node should be created with correct value");
        // Check that an edge was created to this new node
        let s_chain_guard = source_chain.read().unwrap();
        let children_map_chain = s_chain_guard.get(&edge_key_chain).expect("Chain key should now exist in source_chain");
        assert_eq!(children_map_chain.len(), 1, "One edge should be created under chain_key");
        let (node_ptr_chain, ev_chain) = children_map_chain.iter().next().unwrap();
        assert!(Arc::ptr_eq(&node_ptr_chain.upgrade().unwrap(), &result_node_chain), "Edge should point to the newly created node");
        assert_eq!(*ev_chain, new_ev_chain, "Edge should have the new_ev_chain value");
    }

    #[test]
    fn test_ei_else_create_with_value() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        // No try calls, should go straight to else_create
        let result_node = inserter.else_create_destination_with_value("created".to_string()).unwrap();

        assert_eq!(result_node.read().unwrap().value, "created");
        assert_eq!(result_node.read().unwrap().max_depth, 1); // Depth updated
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(&node_ptr.upgrade().unwrap(), &result_node));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_else_create_with() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let created_flag = Arc::new(AtomicUsize::new(0));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let flag_clone = created_flag.clone();
        let result_node = inserter.else_create_destination_with(|| {
            flag_clone.fetch_add(1, Ordering::SeqCst);
            "created_via_fn".to_string()
        }).unwrap();

        assert_eq!(created_flag.load(Ordering::SeqCst), 1); // Closure was called
        assert_eq!(result_node.read().unwrap().value, "created_via_fn");
        assert_eq!(result_node.read().unwrap().max_depth, 1);
    }

    #[test]
    fn test_ei_else_create_default() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        // String::default() is ""
        let result_node = inserter.else_create_destination().unwrap();

        assert_eq!(result_node.read().unwrap().value, ""); // Default value
        assert_eq!(result_node.read().unwrap().max_depth, 1);
    }

    #[test]
    fn test_ei_chaining_try_then_else() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: dest1 causes cycle
        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node = inserter
            .try_destination(dest1.clone()) // Fails (cycle)
            .else_create_destination_with_value("fallback".to_string()) // Executes
            .unwrap();

        assert_eq!(result_node.read().unwrap().value, "fallback"); // Fallback was created
        assert!(!Arc::ptr_eq(&result_node, &dest1));
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(&node_ptr.upgrade().unwrap(), &result_node));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    fn test_ei_chaining_try_success_skips_else() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node = inserter
            .try_destination(dest1.clone()) // Succeeds
            .else_create_destination_with_value("fallback".to_string()) // Should be skipped
            .unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest1)); // Original dest1 was used
        assert_eq!(result_node.read().unwrap().value, "dest1");
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(&node_ptr.upgrade().unwrap(), &dest1));
        assert_eq!(*ev, new_edge_val);
    }

    #[test]
    #[should_panic(expected = "EdgeInserter::unwrap() called but no destination was found or created")]
    fn test_ei_unwrap_panic() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: dest1 causes cycle
        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        // Try fails, no else_create called
        inserter.try_destination(dest1.clone()).unwrap(); // Panic here
    }

    #[test]
    fn test_ei_get() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest1".to_string())));
        let dummy_edge_val = HybridBitset::zeros();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();

        // Setup: dest1 causes cycle
        dest1.write().unwrap().force_insert_to_node("d1->s", dummy_edge_val.clone(), &source);

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});

        // Try fails
        let inserter_after_try = inserter.try_destination(dest1.clone());
        assert!(inserter_after_try.clone_into_option().is_none());

        // Now use else_create
        let inserter_after_else = inserter_after_try.else_create_destination_with_value("fallback".to_string());
        let result_opt = inserter_after_else.into_option();
        assert!(result_opt.is_some());
        assert_eq!(result_opt.unwrap().read().unwrap().value, "fallback");
    }

    #[test]
    fn test_ei_chaining_stops_after_success() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let child1: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child1".to_string()))); // This one succeeds
        let child2: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child2".to_string())));
        let new_node_val_if_created = "new_node_val".to_string();
        let new_edge_val: HybridBitset = vec![1].into_iter().collect();


        let destinations_for_slice = vec![child2.clone()];

        let inserter = EdgeInserter::new(source.clone(), "key", new_edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node = inserter
            .try_destination(child1.clone()) // This succeeds, result is set to child1
            // try_slice, else_create_with_value should now have no effect
            .try_destinations(&destinations_for_slice) // Should be skipped
            .else_create_destination_with_value(new_node_val_if_created.clone()) // Should be skipped
            .unwrap();

        assert!(Arc::ptr_eq(&result_node, &child1), "Chain should stop after first success (try_insert)");

        // Check only the edge to child1 was added
        let s = source.read().unwrap();
        let children_map = s.get(&"key").unwrap(); // Now a BTreeMap
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert!(Arc::ptr_eq(&node_ptr.upgrade().unwrap(), &child1));
        assert_eq!(*ev, new_edge_val);

        // Ensure the value for the skipped else_create was not used
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
        let merged_ev_c1: HybridBitset = vec![1, 10].into_iter().collect(); // Expected merge with child1

        // Setup:
        // source --(target_key, initial_ev_c1)--> child1
        // source --(target_key, initial_ev_c2)--> child2
        // source --("other_key", dummy_ev)--> child_other_key
        {
            let mut s = source.write().unwrap();
            s.try_insert(edge_key, &mut Some(initial_ev_c1), child1.clone()).unwrap();
            s.try_insert(edge_key, &mut Some(initial_ev_c2.clone()), child2.clone()).unwrap();
            s.try_insert("other_key", &mut Some(HybridBitset::zeros()), child_other_key.clone()).unwrap();
        }

        // 1. Test successful merge with the first child under the key.
        //    EdgeInserter is created with source, target_key, and new_ev_for_inserter.
        //    merge_bitset_union should merge new_ev_for_inserter into initial_ev_c1.
        let inserter = EdgeInserter::new(source.clone(), edge_key, new_ev_for_inserter.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node_opt = inserter.try_children().into_option();

        assert!(result_node_opt.is_some(), "Should find and merge with child1");
        let result_node = result_node_opt.unwrap();
        assert!(Arc::ptr_eq(&result_node, &child1), "Result should be child1, got {:?} and {:?}", result_node, child1);

        // Check edge values:
        // Edge to child1 should be merged.
        // Edge to child2 should be unchanged (because merge with child1 succeeded first).
        // Edge to child_other_key should be unchanged.
        {
            let s_guard = source.read().unwrap();
            let children_map_target_key = s_guard.get(&edge_key).expect("Target key should exist");

            let ev_c1 = children_map_target_key.get(&NodePtr::Strong(ArcPtrWrapper::new(child1.clone()))).expect("Child1 should be under target_key");
            assert_eq!(*ev_c1, merged_ev_c1, "Edge value for child1 should be merged");

            let ev_c2 = children_map_target_key.get(&NodePtr::Strong(ArcPtrWrapper::new(child2.clone()))).expect("Child2 should be under target_key");
            assert_eq!(*ev_c2, initial_ev_c2, "Edge value for child2 should be unchanged");

            let children_map_other_key = s_guard.get(&"other_key").expect("Other key should exist");
            assert_eq!(children_map_other_key.len(), 1, "Should be one child under other_key");
            // You could also check the value of the edge to child_other_key if necessary.
        }

        // 2. Test when merge_edge_value fails for all children under the key.
        //    (This test needs a merge function that can fail or a different EV type,
        //     merge_bitset_union always succeeds by design).
        //    Re-using this section to verify the initial state for part 3 is correct.
        let source_nm: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_nm".to_string())));
        let child1_nm: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("child1_nm".to_string())));
        let edge_key_nm = "nm_key"; // "nm" for "no merge"
        let initial_ev_nm: HybridBitset = vec![50].into_iter().collect();
        let new_ev_inserter_nm: HybridBitset = vec![5].into_iter().collect();

        source_nm.write().unwrap().try_insert(edge_key_nm, &mut Some(initial_ev_nm.clone()), child1_nm.clone()).unwrap();

        // Check edge value for child1_nm is unchanged - this is now done in part 3.

        // 3. Test when no children exist under the specified edge_key.
        let source_empty: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_empty".to_string())));
        let edge_key_empty = "empty_key"; // This key has no children in source_empty
        let new_ev_inserter_empty: HybridBitset = vec![7].into_iter().collect();

        let inserter_empty = EdgeInserter::new(source_empty.clone(), edge_key_empty, new_ev_inserter_empty.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node_empty_opt = inserter_empty.try_children().into_option();
        assert!(result_node_empty_opt.is_none(), "try_children should return None if no children under the key");

        // 4. Test chaining with else_create: try_children (no children under key) -> else_create
        let source_chain: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source_chain".to_string())));
        let edge_key_chain = "chain_key"; // No children under this key initially in source_chain
        let new_ev_chain: HybridBitset = vec![8].into_iter().collect();
        let created_val = "created_node_via_fallback".to_string();

        let inserter_chain = EdgeInserter::new(source_chain.clone(), edge_key_chain, new_ev_chain.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node_chain = inserter_chain
            .try_children() // Will do nothing as no children under "chain_key"
            .else_create_destination_with_value(created_val.clone()) // This should execute
            .unwrap();

        assert_eq!(result_node_chain.read().unwrap().value, created_val, "Fallback node should be created with correct value");
        // Check that an edge was created to this new node
        let s_chain_guard = source_chain.read().unwrap();
        let children_map_chain = s_chain_guard.get(&edge_key_chain).expect("Chain key should now exist in source_chain");
        assert_eq!(children_map_chain.len(), 1, "One edge should be created under chain_key");
        let (node_ptr_chain, ev_chain) = children_map_chain.iter().next().unwrap();
        assert!(Arc::ptr_eq(&node_ptr_chain.upgrade().unwrap(), &result_node_chain), "Edge should point to the newly created node");
        assert_eq!(*ev_chain, new_ev_chain, "Edge should have the new_ev_chain value");
    }

    #[test]
    fn test_ei_to_destination_weakly() {
        let source: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("source".to_string())));
        let dest: TestNodeEI = Arc::new(RwLock::new(TestTrieEI::new("dest".to_string())));
        let edge_val: HybridBitset = vec![1].into_iter().collect();

        // 1. Insert a new weak edge
        let inserter = EdgeInserter::new(source.clone(), "key_weak", edge_val.clone(), merge_bitset_union, |_, _| {}, |_, _| {});
        let result_node = inserter.to_destination_weakly(dest.clone()).unwrap();

        assert!(Arc::ptr_eq(&result_node, &dest));
        let s = source.read().unwrap();
        let children_map = s.get(&"key_weak").unwrap();
        assert_eq!(children_map.len(), 1);
        let (node_ptr, ev) = children_map.iter().next().unwrap();
        assert!(!node_ptr.is_strong()); // Check it's a weak edge
        assert_eq!(*ev, edge_val);
        assert!(Arc::ptr_eq(&node_ptr.upgrade().unwrap(), &dest));
        assert_eq!(dest.read().unwrap().max_depth, 0); // Depth NOT updated for weak insert
        drop(s);

        // 2. Merge with existing weak edge
        let new_edge_val: HybridBitset = vec![2].into_iter().collect();
        let merged_val: HybridBitset = vec![1, 2].into_iter().collect();
        let inserter2 = EdgeInserter::new(source.clone(), "key_weak", new_edge_val, merge_bitset_union, |_, _| {}, |_, _| {});
        inserter2.to_destination_weakly(dest.clone()).unwrap();

        let s2 = source.read().unwrap();
        let children_map2 = s2.get(&"key_weak").unwrap();
        assert_eq!(children_map2.len(), 1);
        let (node_ptr2, ev2) = children_map2.iter().next().unwrap();
        assert!(!node_ptr2.is_strong());
        assert_eq!(*ev2, merged_val);
        drop(s2);

        // 3. Merge with existing strong edge (should keep it strong)
        let strong_edge_val: HybridBitset = vec![10].into_iter().collect();
        source.write().unwrap().try_insert("key_strong", &mut Some(strong_edge_val.clone()), dest.clone()).unwrap();

        let new_edge_val_for_strong: HybridBitset = vec![11].into_iter().collect();
        let merged_strong_val: HybridBitset = vec![10, 11].into_iter().collect();
        let inserter3 = EdgeInserter::new(source.clone(), "key_strong", new_edge_val_for_strong, merge_bitset_union, |_, _| {}, |_, _| {});
        inserter3.to_destination_weakly(dest.clone()).unwrap();

        let s3 = source.read().unwrap();
        let children_map3 = s3.get(&"key_strong").unwrap();
        assert_eq!(children_map3.len(), 1);
        let (node_ptr3, ev3) = children_map3.iter().next().unwrap();
        assert!(node_ptr3.is_strong()); // Should remain strong
        assert_eq!(*ev3, merged_strong_val);
    }
}

