use serde::{Serialize, Deserialize, Serializer, Deserializer};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use crate::constraint::{PrecomputeNode, PrecomputedNodeContents, Precomputed, LLMTokenBV};
use crate::types::{TerminalID as GrammarTokenID};
use crate::tokenizer::TokenizerStateID;
use crate::datastructures::ArcPtrWrapper; // Ensure this is the correct path

// Helper type for node ID during serialization/deserialization
type NodeId = usize;

#[derive(Serialize, Deserialize)]
struct SerializablePrecomputeNode {
    id: NodeId,
    value: PrecomputedNodeContents,
    max_depth: usize,
    children: BTreeMap<Option<GrammarTokenID>, BTreeMap<NodeId, LLMTokenBV>>,
}

#[derive(Serialize, Deserialize)]
struct SerializablePrecomputedData {
    nodes: Vec<SerializablePrecomputeNode>,
    roots: BTreeMap<TokenizerStateID, NodeId>,
}

pub fn serialize<S>(precomputed_map: &Precomputed, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut node_to_id: HashMap<ArcPtrWrapper<Mutex<PrecomputeNode>>, NodeId> = HashMap::new();
    let mut id_counter: NodeId = 0;
    let mut serializable_nodes_map: BTreeMap<NodeId, SerializablePrecomputeNode> = BTreeMap::new(); // Use BTreeMap for sorted output by ID

    let mut queue: VecDeque<Arc<Mutex<PrecomputeNode>>> = VecDeque::new();

    // First pass: Discover all nodes and assign IDs
    for root_arc in precomputed_map.values() {
        if !node_to_id.contains_key(&ArcPtrWrapper::new(root_arc.clone())) {
            queue.push_back(root_arc.clone());
            // Assign ID immediately to handle self-loops if they were possible at root level
            node_to_id.insert(ArcPtrWrapper::new(root_arc.clone()), id_counter);
            id_counter += 1;
        }
    }

    let mut bfs_q_idx = 0;
    while bfs_q_idx < queue.len() {
        let node_arc = queue[bfs_q_idx].clone();
        bfs_q_idx += 1;

        let node_guard = node_arc.lock().expect("Mutex poisoned during serialization (ID assignment)");
        for dest_map in node_guard.children().values() {
            for child_wrapper in dest_map.keys() {
                let child_arc = child_wrapper.as_arc();
                if !node_to_id.contains_key(&ArcPtrWrapper::new(child_arc.clone())) {
                     // Check if already in queue to avoid redundant processing, though node_to_id check is primary
                    if !queue.iter().any(|qn_arc| Arc::ptr_eq(qn_arc, child_arc)) {
                        queue.push_back(child_arc.clone());
                    }
                    node_to_id.insert(ArcPtrWrapper::new(child_arc.clone()), id_counter);
                    id_counter += 1;
                }
            }
        }
    }

    // Second pass: Build SerializablePrecomputeNode for each unique node
    for (node_wrapper, &current_node_id) in &node_to_id {
        let node_arc = node_wrapper.as_arc();
        let node_guard = node_arc.lock().expect("Mutex poisoned during serialization (node construction)");

        let mut s_children_map_for_node = BTreeMap::new();
        for (edge_key, dest_map) in node_guard.children().iter() {
            let mut s_dest_map_for_key = BTreeMap::new();
            for (child_arc_ptr_wrapper, edge_val) in dest_map.iter() {
                let child_id = *node_to_id.get(child_arc_ptr_wrapper)
                    .expect("Child node not found in ID map during serialization (pass 2)");
                s_dest_map_for_key.insert(child_id, edge_val.clone());
            }
            if !s_dest_map_for_key.is_empty() {
                s_children_map_for_node.insert(edge_key.clone(), s_dest_map_for_key);
            }
        }

        serializable_nodes_map.insert(current_node_id, SerializablePrecomputeNode {
            id: current_node_id,
            value: node_guard.value.clone(),
            max_depth: node_guard.max_depth,
            children: s_children_map_for_node,
        });
    }

    let serializable_nodes_vec: Vec<SerializablePrecomputeNode> = serializable_nodes_map.into_values().collect();

    let serializable_roots = precomputed_map.iter().map(|(tokenizer_state_id, root_arc)| {
        let root_id = *node_to_id.get(&ArcPtrWrapper::new(root_arc.clone()))
            .expect("Root node not found in ID map during serialization (roots)");
        (*tokenizer_state_id, root_id)
    }).collect();

    let data = SerializablePrecomputedData {
        nodes: serializable_nodes_vec,
        roots: serializable_roots,
    };
    data.serialize(serializer)
}

pub fn deserialize<'de, D>(deserializer: D) -> Result<Precomputed, D::Error>
where
    D: Deserializer<'de>,
{
    let data = SerializablePrecomputedData::deserialize(deserializer)?;
    let mut id_to_node: HashMap<NodeId, Arc<Mutex<PrecomputeNode>>> = HashMap::new();

    // First pass: Create all nodes with their values and max_depth, but without children yet.
    for s_node in &data.nodes {
        let mut precompute_node = PrecomputeNode::new(s_node.value.clone());
        precompute_node.max_depth = s_node.max_depth;
        let arc_node = Arc::new(Mutex::new(precompute_node));
        id_to_node.insert(s_node.id, arc_node);
    }

    // Second pass: Populate children for each node.
    for s_node in &data.nodes {
        let current_node_arc = id_to_node.get(&s_node.id)
            .expect("Node not found in map during deserialization (pass 2)").clone();

        // Children need to be inserted while holding the lock.
        let mut current_node_guard = current_node_arc.lock().expect("Mutex poisoned during deserialization (pass 2)");

        for (edge_key, s_dest_map) in &s_node.children {
            let mut dest_map_for_trie = BTreeMap::new();
            for (&child_id, edge_val) in s_dest_map {
                let child_arc = id_to_node.get(&child_id)
                    .expect("Child node not found in map during deserialization (pass 2 child lookup)").clone();
                dest_map_for_trie.insert(ArcPtrWrapper::new(child_arc), edge_val.clone());
            }
            if !dest_map_for_trie.is_empty() {
                // This directly modifies the children of the PrecomputeNode (Trie)
                current_node_guard.children.insert(edge_key.clone(), dest_map_for_trie);
            }
        }
    }

    let mut precomputed_result = Precomputed::new();
    for (tokenizer_state_id, root_id) in data.roots {
        let root_arc = id_to_node.get(&root_id)
            .expect("Root node not found in map during deserialization (reconstructing Precomputed)").clone();
        precomputed_result.insert(tokenizer_state_id, root_arc);
    }

    Ok(precomputed_result)
}
