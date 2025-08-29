use std::cmp::Ordering;
// #![deny(clippy::iter_over_hash_type)]
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, LockResult, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::datastructures::ArcPtrWrapper;
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::constraint::GodWrapper;
use crate::profiler::PROGRESS_BAR_ENABLED;
use deterministic_hash::DeterministicHasher;
use kdam::{tqdm, BarExt};
use ordered_hash_map::{OrderedHashMap, OrderedHashSet};
use profiler_macro::time_it;

/// Represents a node in a Trie2–like structure (allowing shared subtrees and DAGs).
/// Multiple children can exist for the same edge key. Each edge instance has a value.
///
/// EK: type of the edge key (must be Ord).
/// EV: type of the edge value.
/// T: type of the value stored within the node.
#[derive(Debug, Clone)]
pub struct Trie2<EK: Ord, EV, T> {
    pub value: T,
    /// Stores a map from EdgeKey to a map of destination nodes and edge values.
    children: BTreeMap<EK, OrderedHashMap<ArcPtrWrapper<RwLock<Trie2<EK, EV, T>>>, EV>>,
    /// The “longest distance” from some source node (as computed by recompute_all_max_depths).
    /// Defaults to usize::MAX and is only updated when the user calls recompute_all_max_depths.
    pub max_depth: usize,
}

impl<EK, EV, T> JSONConvertible for Trie2<EK, EV, T>
where
    EK: Ord + Clone + JSONConvertible + Debug,
    EV: Clone + JSONConvertible,
    T: Clone + JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        let mut nodes_json_list: Vec<JSONNode> = Vec::new();
        // Maps the raw pointer of an Arc<RwLock<Trie2>> to its index in nodes_json_list
        let mut arc_ptr_to_idx_map: HashMap<*const RwLock<Trie2<EK, EV, T>>, usize> = HashMap::new();
        // Queue for BFS traversal, storing Arcs to keep them alive and allow locking
        let mut bfs_q: VecDeque<Arc<RwLock<Trie2<EK, EV, T>>>> = VecDeque::new();

        // --- Step 1: Serialize `self` (the root node for this call) ---
        // `self` is node at index 0.
        // We need to store the JSON representation of `self`'s direct data.
        // Since `self` is not an Arc here, we handle it specially as the first node.
        // Its children, which are Arcs, will be processed via the BFS queue.
        let root_idx = 0;
        nodes_json_list.push(JSONNode::Null); // Placeholder for root, will be filled after processing its children.

        let mut root_children_json_data = Vec::new(); // Stores [EK_json, [[ChildIdx, EV_json], ...]]

        // Serialize children
        for (edge_key, destinations_map) in &self.children {
            let ek_json = edge_key.to_json();
            let mut dests_json = Vec::new();

            for (node_ptr, edge_val) in destinations_map {
                let child_arc = node_ptr.as_arc().clone();
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
                dests_json.push(dest_entry);
            }
            if !dests_json.is_empty() {
                root_children_json_data.push(JSONNode::Array(vec![ek_json.clone(), JSONNode::Array(dests_json)]));
            }
        }

        // Fill in the root node's (self's) data
        nodes_json_list[root_idx] = JSONNode::Object(BTreeMap::from_iter(vec![
            ("value".to_string(), self.value.to_json()),
            ("max_depth".to_string(), self.max_depth.to_json()),
            ("children".to_string(), JSONNode::Array(root_children_json_data))
        ]));

        // --- Step 2: Process the rest of the nodes in the queue (BFS) ---
        while let Some(current_arc) = bfs_q.pop_front() {
            let current_arc_ptr = Arc::as_ptr(&current_arc);
            let current_node_json_idx = *arc_ptr_to_idx_map.get(&current_arc_ptr)
                .expect("Node in BFS queue must have an assigned index");

            let node_guard = current_arc.read().expect("RwLock poisoned during Trie2 serialization (BFS part)");
            let mut current_node_children_json_bfs = Vec::new();

            // Serialize children for the current node
            for (edge_key, destinations_map) in &node_guard.children {
                let ek_json = edge_key.to_json();
                let mut dests_json_bfs = Vec::new();

                for (node_ptr, edge_val) in destinations_map {
                    let child_arc = node_ptr.as_arc().clone();
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
                    dests_json_bfs.push(dest_entry);
                }
                if !dests_json_bfs.is_empty() {
                    current_node_children_json_bfs.push(JSONNode::Array(vec![ek_json.clone(), JSONNode::Array(dests_json_bfs)]));
                }
            }

            // Fill in the data for the current node from the BFS queue
            nodes_json_list[current_node_json_idx] = JSONNode::Object(BTreeMap::from_iter(vec![
                ("value".to_string(), node_guard.value.to_json()),
                ("max_depth".to_string(), node_guard.max_depth.to_json()),
                ("children".to_string(), JSONNode::Array(current_node_children_json_bfs))
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
                let nodes_json = obj.remove("nodes").ok_or_else(|| "Missing 'nodes' field for Trie2 deserialization".to_string())?;
                let root_idx_json = obj.remove("root_idx").ok_or_else(|| "Missing 'root_idx' field for Trie2 deserialization".to_string())?;

                let nodes_array = match nodes_json {
                    JSONNode::Array(arr) => arr,
                    _ => return Err("'nodes' field is not an array".to_string()),
                };
                let root_idx = usize::from_json(root_idx_json)?;

                if root_idx >= nodes_array.len() {
                    return Err(format!("Root index {} is out of bounds for nodes array of length {}", root_idx, nodes_array.len()));
                }

                let mut deserialized_arcs: HashMap<usize, Arc<RwLock<Trie2<EK, EV, T>>>> = HashMap::new();

                let mut pb_pass1 = tqdm!(total = nodes_array.len(), desc = "Deserializing nodes (pass 1/2)", disable = !PROGRESS_BAR_ENABLED, leave=false);

                // Pass 1: Create node shells (value, max_depth, empty children)
                for (i, node_data_json) in nodes_array.iter().enumerate() {
                    match node_data_json {
                        JSONNode::Object(n_obj) => {
                            let value_json = n_obj.get("value").ok_or_else(|| format!("Node at index {} missing 'value'", i))?;
                            let max_depth_json = n_obj.get("max_depth").ok_or_else(|| format!("Node at index {} missing 'max_depth'", i))?;

                            let value = T::from_json(value_json.clone())?;
                            let max_depth = usize::from_json(max_depth_json.clone())?;

                            let new_node_arc = Arc::new(RwLock::new(Trie2 {
                                value,
                                children: BTreeMap::new(),
                                max_depth,
                            }));
                            deserialized_arcs.insert(i, new_node_arc);
                        }
                        _ => return Err(format!("Node data at index {} is not an object", i)),
                    }
                    let _ = pb_pass1.update(1);
                }

                let mut pb_pass2 = tqdm!(total = nodes_array.len(), desc = "Linking nodes (pass 2/2)", disable = !PROGRESS_BAR_ENABLED, leave=false);

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
                        }
                        _ => unreachable!("Node data should be an object, checked in Pass 1"),
                    }
                    let _ = pb_pass2.update(1);
                }

                let root_arc_final = deserialized_arcs.get(&root_idx)
                    .ok_or_else(|| format!("Root index {} not found in deserialized_arcs map after linking", root_idx))?
                    .clone();

                // The trait requires returning Self, so we clone the content of the root Arc.
                // The shared graph structure is maintained by the Arcs held within the children maps.
                let root_trie_content = root_arc_final.read().unwrap().clone();
                Ok(root_trie_content)
            }
            _ => Err("Expected JSONNode::Object for Trie2 graph structure".to_string()),
        }
    }
}

// Implementation block for core Trie2 functionality
// Added Clone bound for EK needed in insertion and others
impl<EK: Ord + Clone, EV, T> Trie2<EK, EV, T> {
    /// Creates a new trie node with the given value and no children.
    /// The max_depth is initialized to usize::MAX and will be updated later
    /// when recompute_all_max_depths is called.
    pub fn new(value: T) -> Self {
        Trie2 {
            value,
            children: BTreeMap::new(),
            max_depth: usize::MAX,
        }
    }

    pub fn force_insert_to_new_node(&mut self, edge_key: EK, edge_value: EV, value: T) -> Arc<RwLock<Trie2<EK, EV, T>>> {
        let new_node = Arc::new(RwLock::new(Trie2::new(value)));
        let new_node_comparable = ArcPtrWrapper::new(new_node.clone());
        self.children.entry(edge_key).or_default().insert(new_node_comparable, edge_value);
        new_node.clone()
    }

    pub fn force_insert_to_node(&mut self, edge_key: EK, edge_value: EV, dst: &Arc<RwLock<Trie2<EK, EV, T>>>) {
        let dst_comparable = ArcPtrWrapper::new(dst.clone());
        self.children.entry(edge_key).or_default().insert(dst_comparable, edge_value);
    }

    pub fn already_has_dst(&self, edge_key: EK, dst: &Arc<RwLock<Trie2<EK, EV, T>>>) -> bool {
        let lookup_key = ArcPtrWrapper::new(dst.clone()); // Clone Arc for temporary ownership in key
        self.children.get(&edge_key).map_or(false, |dest_map| dest_map.contains_key(&lookup_key))
    }

    pub fn already_has_dst_for_any_key(&self, dst: &Arc<RwLock<Trie2<EK, EV, T>>>) -> bool {
        let lookup_key = ArcPtrWrapper::new(dst.clone());
        self.children.values().any(|dest_map| dest_map.contains_key(&lookup_key))
    }

    pub fn get_edge_value(&self, edge_key: EK, dst: &Arc<RwLock<Trie2<EK, EV, T>>>) -> Option<&EV> {
        let lookup_key = ArcPtrWrapper::new(dst.clone());
        self.children.get(&edge_key).and_then(|dest_map| dest_map.get(&lookup_key))
    }

    pub fn get_edge_value_mut(&mut self, edge_key: EK, dst: &Arc<RwLock<Trie2<EK, EV, T>>>) -> Option<&mut EV> {
        let lookup_key = ArcPtrWrapper::new(dst.clone());
        self.children.get_mut(&edge_key).and_then(|dest_map| dest_map.get_mut(&lookup_key))
    }

    /// Inserts an edge without any cycle checks or automatic depth updates.
    /// Depths can be recomputed later by calling `recompute_all_max_depths`.
    #[time_it]
    pub fn try_insert(
        &mut self,
        edge_key: EK,
        edge_value: &mut Option<EV>,
        child: Arc<RwLock<Trie2<EK, EV, T>>>,
    ) {
        self.try_insert_unchecked(edge_key, edge_value, child)
    }

    /// Inserts an edge without any checks or automatic depth updates.
    #[time_it]
    pub fn try_insert_unchecked(
        &mut self,
        edge_key: EK,
        edge_value: &mut Option<EV>,
        child: Arc<RwLock<Trie2<EK, EV, T>>>,
    ) {
        let child_comparable = ArcPtrWrapper::new(child.clone());
        self.children
            .entry(edge_key)
            .or_default()
            .insert(child_comparable, edge_value.take().expect("edge_value must be Some when inserting"));
    }

    pub fn get(
        &self,
        edge_key: &EK,
    ) -> Option<&OrderedHashMap<ArcPtrWrapper<RwLock<Trie2<EK, EV, T>>>, EV>>
    {
        self.children.get(edge_key)
    }

    pub fn get_mut(
        &mut self,
        edge_key: &EK,
    ) -> Option<&mut OrderedHashMap<ArcPtrWrapper<RwLock<Trie2<EK, EV, T>>>, EV>>
    {
        self.children.get_mut(edge_key)
    }

    pub fn children(&self) -> &BTreeMap<EK, OrderedHashMap<ArcPtrWrapper<RwLock<Trie2<EK, EV, T>>>, EV>> {
        &self.children
    }

    pub fn children_mut(&mut self) -> &mut BTreeMap<EK, OrderedHashMap<ArcPtrWrapper<RwLock<Trie2<EK, EV, T>>>, EV>> {
        &mut self.children
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Collects all unique nodes (by pointer) reachable from the given roots (BFS).
    pub fn all_nodes(roots: &[Arc<RwLock<Trie2<EK, EV, T>>>]) -> Vec<Arc<RwLock<Trie2<EK, EV, T>>>> {
        let mut visited_arcs: HashSet<*const RwLock<Trie2<EK, EV, T>>> = HashSet::new();
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
            let node_guard = node_arc.read().expect("RwLock poisoned during BFS");
            for children_map in node_guard.children.values() {
                for node_ptr in children_map.keys() {
                    let child_arc = node_ptr.as_arc().clone();
                    let child_arc_ptr = Arc::as_ptr(&child_arc);
                    if visited_arcs.insert(child_arc_ptr) {
                        queue.push_back(child_arc.clone());
                    }
                }
            }
        }
        result
    }

    /// Recomputes `max_depth` for all nodes reachable from the given roots.
    /// Call this once after you finish building the trie graph. Before calling,
    /// nodes may have max_depth == usize::MAX which is suboptimal for scheduling
    /// but not incorrect for traversal.
    ///
    /// Uses a topological order (Kahn's algorithm). Assumes the graph is acyclic.
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
            for child_arc in node_guard.children.values().flat_map(|m| m.keys()).map(|arc_wrapper| arc_wrapper.as_arc().clone()) {
                let child_ptr = Arc::as_ptr(&child_arc);
                adj.entry(node_ptr).or_default().push(child_ptr);
                *in_degree.entry(child_ptr).or_default() += 1;
            }
        }

        // Initialize depths to 0 for sources and 0 for others (we'll compute actual values).
        let mut queue = VecDeque::new();
        for node_arc in &all_nodes {
            let node_ptr = Arc::as_ptr(node_arc);
            if in_degree.get(&node_ptr).cloned().unwrap_or(0) == 0 {
                queue.push_back(node_ptr);
                node_arc.write().unwrap().max_depth = 0;
            } else {
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
    /// This does NOT propagate changes. The caller is responsible for propagation
    /// if needed. This is typically safe to call in a post-order traversal where
    /// children's depths are finalized first.
    pub fn recompute_max_depth(&mut self) -> bool {
        let new_max_depth = self.children.values()
            .flat_map(|dest_map| dest_map.keys().map(|arc_wrapper| arc_wrapper.as_arc().clone()))
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

// Helper to get the raw pointer to the Trie2 data from an Arc<RwLock<Trie2>>.
/// Panics if the mutex is poisoned. Returns None if lock fails (WouldBlock).
/// Use with caution: Only use when you know a failed lock means the current thread holds it.
/// Consider using `Arc::as_ptr` for identity checks instead if possible.
#[allow(dead_code)]
pub(crate) fn try_get_node_data_ptr<EK: Ord, EV, T>(node_arc: &Arc<RwLock<Trie2<EK, EV, T>>>) -> Option<*const Trie2<EK, EV, T>> {
    match node_arc.try_read() {
        Ok(guard) => {
            let ptr = &*guard as *const Trie2<EK, EV, T>;
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

/// Helper to get the raw pointer to the Trie2 data from an Arc<RwLock<Trie2>>.
/// Panics if the mutex is poisoned or if locking fails (blocking lock).
#[allow(dead_code)]
pub(crate) fn node_ptr<EK: Ord, EV, T>(node_arc: &Arc<RwLock<Trie2<EK, EV, T>>>) -> *const Trie2<EK, EV, T> {
    let guard = node_arc.read().expect("RwLock poisoned or lock failed when getting node pointer");
    &*guard as *const _
}

// Add this impl block for the recursive comparison helper
impl<EK, EV, T> Trie2<EK, EV, T>
where
    EK: Ord,
    EV: PartialEq + Clone,
    T: PartialEq,
{
    /// Recursively compares two Trie2 nodes wrapped in Arcs for equality.
    ///
    /// - `self_arc`, `other_arc`: The Arcs pointing to the Trie2 nodes to compare.
    /// - `comparison_cache`: Tracks pairs of (self_node_ptr, other_node_ptr) and their comparison result (bool).
    ///   This cache is important for efficiency on DAGs with shared subgraphs: it avoids re-comparing pairs
    ///   already processed and ensures consistent topology checks.
    fn compare_arcs_recursive(
        self_arc: &Arc<RwLock<Trie2<EK, EV, T>>>,
        other_arc: &Arc<RwLock<Trie2<EK, EV, T>>>,
        comparison_cache: &mut HashMap<(*const RwLock<Self>, *const RwLock<Self>), bool>,
    ) -> bool {
        let self_ptr = Arc::as_ptr(self_arc);
        let other_ptr = Arc::as_ptr(other_arc);

        if self_ptr == other_ptr {
            return true;
        }

        // Canonical cache key: (min_ptr, max_ptr).
        let (cache_key_ptr1, cache_key_ptr2) = if self_ptr < other_ptr {
            (self_ptr, other_ptr)
        } else {
            (other_ptr, self_ptr)
        };

        if let Some(&cached_result) = comparison_cache.get(&(cache_key_ptr1, cache_key_ptr2)) {
            return cached_result;
        }

        // Optimistically mark this pair as true; will be updated to false on mismatch.
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

        // 1. Compare non-recursive fields: value and max_depth.
        if self_node.value != other_node.value || self_node.max_depth != other_node.max_depth {
            comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false);
            return false;
        }

        // 2. Compare children structure (number of distinct edge keys).
        if self_node.children.len() != other_node.children.len() {
            comparison_cache.insert((cache_key_ptr1, cache_key_ptr2), false);
            return false;
        }

        // 3. Compare children for each edge key.
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

                    let self_child_pairs: Vec<(Arc<RwLock<Trie2<EK, EV, T>>>, EV)> = self_dest_map.iter()
                        .map(|(apw, ev)| (apw.as_arc().clone(), ev.clone()))
                        .collect();

                    let mut other_child_pairs: Vec<(Arc<RwLock<Trie2<EK, EV, T>>>, EV)> = other_dest_map.iter()
                        .map(|(apw, ev)| (apw.as_arc().clone(), ev.clone()))
                        .collect();

                    'self_pair_loop: for (s_arc, s_ev) in &self_child_pairs {
                        let mut found_match_for_current_self_pair = false;
                        for i in 0..other_child_pairs.len() {
                            if s_ev == &other_child_pairs[i].1 {
                                let o_arc_for_recursion = other_child_pairs[i].0.clone();
                                if Trie2::compare_arcs_recursive(s_arc, &o_arc_for_recursion, comparison_cache) {
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

        true
    }
}

// Implementation block for special_map and related functionality
// Requires T: Clone, EK: Ord + Clone, EV: Clone
impl<T: Clone, EK: Ord + Clone, EV: Clone> Trie2<EK, EV, T> {
    fn count_all_edges(root_nodes: &[Arc<RwLock<Trie2<EK, EV, T>>>]) -> usize {
        let mut visited_arcs: HashSet<*const RwLock<Trie2<EK, EV, T>>> = HashSet::new();
        let mut queue: VecDeque<Arc<RwLock<Trie2<EK, EV, T>>>> = VecDeque::new();
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
                    let child_arc = node_ptr.as_arc(); total_edges += 1; if visited_arcs.insert(Arc::as_ptr(&child_arc)) { queue.push_back(child_arc.clone()); }
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
        initial_nodes_and_values: Vec<(Arc<RwLock<Trie2<EK, EV, T>>>, V)>,
        mut step: impl FnMut(&V, &EK, &EV, &Trie2<EK, EV, T>) -> Option<V>,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&Trie2<EK, EV, T>, &mut V) -> bool,
    ) {
        // ------------------------------------------------------------------
        //  Simple depth-driven scheduler.
        // ------------------------------------------------------------------
        let mut values: HashMap<*const RwLock<Self>, V> = HashMap::new();
        let mut stopped_nodes: HashSet<*const RwLock<Self>> = HashSet::new();
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
                    stopped_nodes.insert(ptr);
                    continue;
                }

                // ---------- propagate to children -------------
                let edges: Vec<(EK, EV, Arc<RwLock<Self>>)> = {
                    let guard = node_arc.read().expect("poison");
                    guard.children
                        .iter()
                        .flat_map(|(ek, dst_map)| {
                            dst_map.iter().map(move |(node_ptr, ev)| (ek.clone(), ev.clone(), node_ptr.as_arc().clone()))
                        })
                        .collect()
                };

                for (ek, ev, child_arc) in edges {
                    let _ = pb.update(1);
                    let child_ptr = Arc::as_ptr(&child_arc);

                    if stopped_nodes.contains(&child_ptr) {
                        continue;
                    }

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

    /// Performs a specialized breadth-first traversal, grouping children by edge key.
    /// This is more efficient than `special_map` when many edges share the same key,
    /// as the `step` function is called once per key, not once per edge.
    ///
    /// Scheduling uses node.max_depth; if not recomputed yet it may be usize::MAX,
    /// which is suboptimal but not incorrect.
    #[time_it]
    pub fn special_map_grouped<V, S, I>(
        initial_nodes_and_values: Vec<(Arc<RwLock<Trie2<EK, EV, T>>>, V)>,
        mut step: S,
        mut merge: impl FnMut(&mut V, V),
        mut process: impl FnMut(&Trie2<EK, EV, T>, &mut V) -> bool,
    )
    where
        V: Clone,
        S: FnMut(
            &V, &EK, &OrderedHashMap<ArcPtrWrapper<RwLock<Trie2<EK, EV, T>>>, EV>
        ) -> I,
        I: IntoIterator<Item = (ArcPtrWrapper<RwLock<Trie2<EK, EV, T>>>, V)>,
    {
        // ------------------------------------------------------------------
        //  Simple depth-driven scheduler. (Same as special_map)
        // ------------------------------------------------------------------
        let mut values: HashMap<*const RwLock<Self>, V> = HashMap::new();
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
                    stopped_nodes.insert(ptr);
                    continue;
                }

                // ---------- propagate to children (grouped by edge key) -------------
                let children_by_ek: Vec<(EK, OrderedHashMap<ArcPtrWrapper<RwLock<Self>>, EV>)> = {
                    let guard = node_arc.read().expect("poison");
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

                    for (child_node_ptr, new_v) in new_values_for_children {
                        let child_arc_wrapper = child_node_ptr.clone();
                        let child_ptr = child_arc_wrapper.as_ref() as *const RwLock<Self>;

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

// Implement PartialEq for Trie2
impl<EK, EV, T> PartialEq for Trie2<EK, EV, T>
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

        type NodeRwLockPtr<EKK, EVV, TT> = *const RwLock<Trie2<EKK, EVV, TT>>;
        let mut comparison_cache: HashMap<(NodeRwLockPtr<EK, EV, T>, NodeRwLockPtr<EK, EV, T>), bool> = HashMap::new();

        for (self_ek, self_dest_map) in &self.children {
            match other.children.get(self_ek) {
                None => return false,
                Some(other_dest_map) => {
                    if self_dest_map.len() != other_dest_map.len() {
                        return false;
                    }

                    let self_child_pairs: Vec<(Arc<RwLock<Trie2<EK, EV, T>>>, &EV)> = self_dest_map
                        .iter()
                        .map(|(np, ev)| (np.as_arc().clone(), ev))
                        .collect();

                    let mut other_child_pairs: Vec<(Arc<RwLock<Trie2<EK, EV, T>>>, EV)> = other_dest_map
                        .iter()
                        .map(|(np, ev)| (np.as_arc().clone(), ev.clone()))
                        .collect();

                    'self_pair_loop: for (s_arc, s_ev) in self_child_pairs {
                        for i in 0..other_child_pairs.len() {
                            if s_ev == &other_child_pairs[i].1 {
                                let o_arc_for_recursion = other_child_pairs[i].0.clone();
                                if Trie2::compare_arcs_recursive(&s_arc, &o_arc_for_recursion, &mut comparison_cache) {
                                    other_child_pairs.remove(i);
                                    continue 'self_pair_loop;
                                }
                            }
                        }
                        return false;
                    }
                }
            }
        }

        true
    }
}

// Implement Eq for Trie2
impl<EK, EV, T> Eq for Trie2<EK, EV, T>
where
    EK: Ord,
    EV: Eq + Clone,
    T: Eq,
{
}

// Implement Hash for Trie2
impl<EK, EV, T> Hash for Trie2<EK, EV, T>
where
    EK: Ord + Hash,
    EV: PartialEq + Clone + Hash,
    T: PartialEq + Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Cache to handle shared nodes during hashing.
        // Maps the raw data pointer of a Trie2 node to a marker (depth) to break revisits.
        let mut recursion_marker: HashMap<*const Trie2<EK, EV, T>, usize> = HashMap::new();
        Self::hash_trie_recursive(self, state, &mut recursion_marker, 0);
    }
}

impl<EK, EV, T> Trie2<EK, EV, T>
where
    EK: Ord + Hash,
    EV: PartialEq + Clone + Hash,
    T: PartialEq + Hash,
{
    /// Helper function to hash a &Trie2 instance.
    fn hash_trie_recursive<S: Hasher>(
        node: &Trie2<EK, EV, T>,
        state: &mut S,
        recursion_marker: &mut HashMap<*const Trie2<EK, EV, T>, usize>,
        current_depth: usize,
    ) {
        let node_ptr = node as *const _;
        if let Some(visited_depth) = recursion_marker.get(&node_ptr) {
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

            dest_map.len().hash(state);
            let mut pair_hashes = Vec::with_capacity(dest_map.len());
            for (node_ptr, ev) in dest_map {
                let mut pair_hasher = DeterministicHasher::new(DefaultHasher::new());
                ev.hash(&mut pair_hasher);
                let child_arc = node_ptr.as_arc().clone();
                if let Ok(child_guard) = child_arc.read() {
                    Self::hash_trie_recursive(&*child_guard, &mut pair_hasher, recursion_marker, current_depth + 1);
                    pair_hashes.push(pair_hasher.finish());
                };
            }
            pair_hashes.sort_unstable();
            for h in pair_hashes {
                h.hash(state);
            }
        }
    }
}

/// A helper struct to facilitate inserting an edge into a Trie2,
/// trying multiple potential destinations and optionally creating a new node.
/// Provides a chainable interface.
pub struct EdgeInserter<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
where
    EK: Ord + Clone,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    source_arc: Arc<RwLock<Trie2<EK, EV, T>>>, // The source node for the edge
    edge_key: EK,                            // The key for the edge to be inserted
    edge_value: Option<EV>,                  // The value for the edge to be inserted
    merge_edge_value: FMergeEV,              // The function to merge edge values
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
    result: Option<Arc<RwLock<Trie2<EK, EV, T>>>>, // Stores the successful destination node
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
    /// Creates a new `EdgeInserter`.
    ///
    /// # Arguments
    ///
    /// * `source_arc`: The source node where the edge originates.
    /// * `edge_key`: The key for the new edge.
    /// * `edge_value`: The value for the new edge.
    /// * `merge_edge_value`: A closure that merges the existing edge value with the new edge value.
    pub fn new(
        god: &GodWrapper<EK, EV, T>,
        source_arc: Arc<RwLock<Trie2<EK, EV, T>>>,
        edge_key: EK,
        edge_value: EV,
        merge_edge_value: FMergeEV,
        update_node_value: FUpdateT,
        mut merge_edge_value_and_source_node_value: FMergeEV_T,
    ) -> Self {
        // Avoid unused-parameter warnings
        let _ = god;

        let mut edge_value = edge_value;
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
    /// it merges the `edge_value` using the `merge_edge_value` closure.
    /// If no such edge exists, it inserts a new edge.
    ///
    /// Returns `self` to allow chaining.
    #[time_it]
    pub fn try_destination(mut self, destination: Arc<RwLock<Trie2<EK, EV, T>>>) -> Self {
        if self.result.is_some() {
            return self; // Already found a destination
        }

        let mut update_info: Option<(Arc<RwLock<Trie2<EK, EV, T>>>, EV)> = None;

        { // Scope for source_guard
            let mut source_guard = self.source_arc.write().expect("RwLock poisoned while locking source in try_destination");
            let destination_wrapper = ArcPtrWrapper::new(destination.clone());

            if let Some(existing_ev_mut) = source_guard.children.get_mut(&self.edge_key).and_then(|dest_map| dest_map.get_mut(&destination_wrapper)) {
                let new_ev = self.edge_value.take().unwrap();
                crate::debug!(7, "Merging edge value {:?} into existing edge value {:?} for edge {:?} to node {:p}", new_ev, existing_ev_mut, self.edge_key, Arc::as_ptr(&destination));
                (self.merge_edge_value)(existing_ev_mut, new_ev);
                let updated_ev = existing_ev_mut.clone();
                self.result = Some(destination.clone());
                update_info = Some((destination, updated_ev));
            } else {
                let edge_val_clone = self.edge_value.as_ref().unwrap().clone();
                crate::debug!(7, "Inserting edge {:?} with value {:?} to node {:p}", self.edge_key, edge_val_clone, Arc::as_ptr(&destination));
                source_guard.try_insert(self.edge_key.clone(), &mut self.edge_value, destination.clone());
                self.result = Some(destination.clone());
                update_info = Some((destination, edge_val_clone));
            }
        }

        if let Some((dest_arc, ev)) = update_info {
            crate::debug!(7, "Updating node value for destination {:p} with edge value {:?}. self.edge_value: {:?}", Arc::as_ptr(&dest_arc), ev, self.edge_value);
            (self.update_node_value)(&mut dest_arc.write().unwrap().value, &ev);
        }

        self
    }

    /// Tries to establish an edge to any destination in the provided slice.
    /// Iterates through `destinations` and calls `try_destination` for each until one succeeds.
    /// Returns `self` to allow chaining.
    #[time_it]
    pub fn try_destinations(mut self, destinations: &[Arc<RwLock<Trie2<EK, EV, T>>>]) -> Self {
        for destination in destinations {
            if self.result.is_some() {
                break; // Stop trying once a destination is found
            }
            self = self.try_destination(destination.clone());
        }
        self
    }

    #[time_it]
    pub fn try_destinations_iter(mut self, destinations: impl Iterator<Item = Arc<RwLock<Trie2<EK, EV, T>>>>) -> Self {
        for destination in destinations {
            if self.result.is_some() {
                break; // Stop trying once a destination is found
            }
            self = self.try_destination(destination.clone());
        }
        self
    }

    #[time_it]
    pub fn try_destinations_iter_with<F, R>(mut self, destinations: F) -> Self
    where
        F: Fn() -> R,
        R: Iterator<Item = Arc<RwLock<Trie2<EK, EV, T>>>>,
    {
        for destination in destinations() {
            if self.result.is_some() {
                break;
            }
            self = self.try_destination(destination.clone());
        }
        self
    }

    /// Merges the edge with existing children under `self.edge_key` (if any).
    /// Returns `self` to allow chaining.
    pub fn try_children(mut self) -> Self {
        if self.result.is_some() {
            return self;
        }

        let children_for_this_key: Vec<Arc<RwLock<Trie2<EK, EV, T>>>> = {
            let source_guard = self.source_arc.read().expect("RwLock poisoned while locking source in try_children");
            if let Some(dest_map) = source_guard.children.get(&self.edge_key) {
                dest_map.keys()
                    .map(|arc_wrapper| arc_wrapper.as_arc().clone())
                    .collect()
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

        let new_node_arc = Arc::new(RwLock::new(Trie2::new(value)));
        let edge_val_clone = self.edge_value.as_ref().unwrap().clone();

        { // Scope for source_guard
            let mut source_guard = self.source_arc.write().expect("RwLock poisoned while locking source in else_create_with_value");
            source_guard.try_insert(self.edge_key.clone(), &mut self.edge_value, new_node_arc.clone());
            self.result = Some(new_node_arc.clone());
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
    pub fn into_option(self) -> Option<Arc<RwLock<Trie2<EK, EV, T>>>> {
        self.result
    }

    pub fn is_some(&self) -> bool {
        self.result.is_some()
    }

    pub fn clone_into_option(&self) -> Option<Arc<RwLock<Trie2<EK, EV, T>>>> {
        self.result.clone()
    }

    /// Returns the resulting destination node, panicking if none was found or created.
    pub fn unwrap(self) -> Arc<RwLock<Trie2<EK, EV, T>>> {
        self.result.expect("EdgeInserter::unwrap() called but no destination was found or created")
    }

    /// Returns the resulting destination node, panicking with the given message if none was found or created.
    pub fn expect(self, msg: &str) -> Arc<RwLock<Trie2<EK, EV, T>>> {
        self.result.expect(msg)
    }
}

// Optional: Add a convenience method to Trie2 to create an EdgeInserter easily.
impl<EK: Ord + Clone + Debug, EV: Clone + Debug, T: Clone> Trie2<EK, EV, T> {
    /// Creates an `EdgeInserter` to help add an edge starting from this node.
    ///
    /// This provides a convenient entry point for the chainable insertion pattern.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::sync::{Arc, RwLock};
    /// use crate::datastructures::trie::Trie2;
    /// use crate::datastructures::trie::EdgeInserter;
    ///
    /// #[derive(Debug, Clone, Default)]
    /// struct NodeValue { /* ... */ }
    ///
    /// // Example merge function for edge values
    /// fn merge_ev(existing: &mut i32, new: i32) {
    ///     *existing += new;
    /// }
    ///
    /// let root_node: Arc<RwLock<Trie2<String, i32, NodeValue>>> = Arc::new(RwLock::new(Trie2::new(NodeValue::default())));
    ///
    /// let potential_destinations: Vec<Arc<RwLock<Trie2<String, i32, NodeValue>>>> = vec![/* ... */];
    ///
    /// let new_or_existing_node = {
    ///     let root_guard = root_node.write().unwrap();
    ///     let god = /* obtain GodWrapper */ unimplemented!();
    ///     root_guard.insert_edge(&god, "key".to_string(), 1, merge_ev, |_t, _ev| {}, |_ev, _t| {})
    ///         .try_destinations(&potential_destinations)
    ///         .else_create_destination()
    ///         .unwrap()
    /// };
    /// ```
    pub fn insert_edge<FMergeEV, FUpdateT, FMergeEV_T>(
        &self,
        god: &GodWrapper<EK, EV, T>,
        edge_key: EK,
        edge_value: EV,
        merge_edge_value: FMergeEV,
        update_node_value: FUpdateT,
        merge_edge_value_and_source_node_value: FMergeEV_T,
    ) -> EdgeInserter<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>
    where
         FMergeEV: FnMut(&mut EV, EV),
         FUpdateT: FnMut(&mut T, &EV),
         FMergeEV_T: FnMut(&mut EV, &T),
    {
        EdgeInserter::new(
            god,
            Arc::new(RwLock::new(self.clone())),
            edge_key,
            edge_value,
            merge_edge_value,
            update_node_value,
            merge_edge_value_and_source_node_value
        )
    }
}

/// Attempts to establish an edge from `source` to a single `destination`,
/// optionally merging edge values if an edge already exists.
/// Returns `Some(Arc<RwLock<Trie2<...>>>)` if merge or insert succeeded.
pub fn try_destination<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>(
    god: &GodWrapper<EK, EV, T>,
    source: Arc<RwLock<Trie2<EK, EV, T>>>,
    edge_key: EK,
    edge_value: EV,
    destination: Arc<RwLock<Trie2<EK, EV, T>>>,
    merge_edge_value: FMergeEV,
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
) -> Option<Arc<RwLock<Trie2<EK, EV, T>>>>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    EdgeInserter::new(
        god,
        source,
        edge_key,
        edge_value,
        merge_edge_value,
        update_node_value,
        merge_edge_value_and_source_node_value
    )
        .try_destination(destination)
        .into_option()
}

/// Attempts to establish an edge from `source` to any of the provided `destinations`,
/// returning the first successful one (merge or insert), or `None` if none matched.
pub fn try_destination_with<EK, EV, T, FMergeEV, FUpdateT, FMergeEV_T>(
    god: &GodWrapper<EK, EV, T>,
    source: Arc<RwLock<Trie2<EK, EV, T>>>,
    edge_key: EK,
    edge_value: EV,
    destinations: &[Arc<RwLock<Trie2<EK, EV, T>>>],
    merge_edge_value: FMergeEV,
    update_node_value: FUpdateT,
    merge_edge_value_and_source_node_value: FMergeEV_T,
) -> Option<Arc<RwLock<Trie2<EK, EV, T>>>>
where
    EK: Ord + Clone + Debug,
    EV: Clone + Debug,
    T: Clone,
    FMergeEV: FnMut(&mut EV, EV),
    FUpdateT: FnMut(&mut T, &EV),
    FMergeEV_T: FnMut(&mut EV, &T),
{
    EdgeInserter::new(
        god,
        source,
        edge_key,
        edge_value,
        merge_edge_value,
        update_node_value,
        merge_edge_value_and_source_node_value
    )
        .try_destinations(destinations)
        .into_option()
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Index {
    index: usize,
}

#[derive(Debug, Clone)]
pub struct Arena<T> {
    values: Arc<RwLock<BTreeMap<usize, T>>>,
}
impl<T> PartialEq for Arena<T> where T: PartialEq {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.values, &other.values) || PartialEq::eq(&*self.values.read().unwrap(), &*other.values.read().unwrap())
    }
}
impl<T> Eq for Arena<T> where T: Eq {}
impl<T> PartialOrd for Arena<T> where T: PartialOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if Arc::ptr_eq(&self.values, &other.values) {
            return Some(Ordering::Equal);
        }
        PartialOrd::partial_cmp(&*self.values.read().unwrap(), &*other.values.read().unwrap())
    }
}
impl<T> Ord for Arena<T> where T: Ord {
    fn cmp(&self, other: &Self) -> Ordering {
        if Arc::ptr_eq(&self.values, &other.values) {
            return Ordering::Equal;
        }
        Ord::cmp(&*self.values.read().unwrap(), &*other.values.read().unwrap())
    }
}
impl<T> Hash for Arena<T> where T: Hash {
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

    pub fn insert(&self, value: T) -> Index {
        let mut guard = self.values.write().unwrap();
        let new_index = guard.len();
        guard.insert(new_index, value);
        Index { index: new_index }
    }

    pub fn get(&self, index: Index) -> Option<T>
    where
        T: Clone,
    {
        let guard = self.values.read().unwrap();
        guard.get(&index.index).cloned()
    }

    pub fn get_mut(&self, index: Index) -> Option<std::sync::RwLockWriteGuard<'_, T>> {
 
    }

    pub fn len(&self) -> usize {
        self.values.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}