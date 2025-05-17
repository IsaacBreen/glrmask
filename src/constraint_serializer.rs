use serde::{Serialize, Deserialize, Serializer, Deserializer};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use crate::constraint::{PrecomputeNode, PrecomputedNodeContents, Precomputed, LLMTokenBV};
use crate::types::TerminalID as GrammarTokenID;
use crate::tokenizer::TokenizerStateID;
use crate::datastructures::ArcPtrWrapper; // Ensure this is the correct path
use crate::datastructures::trie::{Trie, CycleDetectedError}; // Import Trie and CycleDetectedError

// Helper type for node ID during serialization/deserialization
type NodeId = usize;
type TrieType = Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;

const OPTION_NONE_KEY_STR: &str = "__OPTION_NONE_KEY__";

#[derive(Serialize, Deserialize)]
struct SerializablePrecomputeNode {
    id: NodeId,
    value: PrecomputedNodeContents,
    max_depth: usize,
    children: BTreeMap<String, BTreeMap<NodeId, LLMTokenBV>>,
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
    let mut node_to_id: HashMap<*const Mutex<PrecomputeNode>, NodeId> = HashMap::new();
    let mut id_counter: NodeId = 0;
    let mut serializable_nodes_map: BTreeMap<NodeId, SerializablePrecomputeNode> = BTreeMap::new(); // Use BTreeMap for sorted output by ID

    let mut queue: VecDeque<Arc<Mutex<PrecomputeNode>>> = VecDeque::new();

    // First pass: Discover all nodes and assign IDs using BFS starting from roots.
    // Roots in precomputed_map are BTreeMap<TokenizerStateID, PrecomputeNode>.
    // We need to wrap them in Arc<Mutex> for processing.
    let root_arcs: BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>> = precomputed_map.iter()
        .map(|(sid, node)| (*sid, Arc::new(Mutex::new(node.clone()))))
        .collect();


    for root_arc in root_arcs.values() {
        let root_ptr = Arc::as_ptr(root_arc);
        if node_to_id.insert(root_ptr, id_counter).is_none() {
             queue.push_back(root_arc.clone());
             id_counter += 1;
        }
    }

    let mut bfs_q_idx = 0;
    while bfs_q_idx < queue.len() {
        let node_arc = queue[bfs_q_idx].clone();
        bfs_q_idx += 1;

        let node_guard = node_arc.lock().expect("Mutex poisoned during serialization (ID assignment)");
        for dest_map in node_guard.children().values() { // Access children via method
            for child_wrapper in dest_map.keys() {
                let child_arc = child_wrapper.as_arc();
                let child_ptr = Arc::as_ptr(child_arc);
                if node_to_id.insert(child_ptr, id_counter).is_none() {
                     queue.push_back(child_arc.clone());
                    id_counter += 1;
                }
            }
        }
    }

    // Second pass: Build SerializablePrecomputeNode for each unique node
    // Iterate over nodes in order of assigned IDs for consistent serialization.
    let mut nodes_by_id: BTreeMap<NodeId, Arc<Mutex<PrecomputeNode>>> = BTreeMap::new();
    for (ptr, id) in node_to_id.iter() {
        // Find the corresponding Arc in the queue (or perhaps rebuild a map from ptr to Arc).
        // A simpler approach is to iterate through the original queue/visited set.
    }

    // Rebuild a ptr -> Arc map from the queue which contains all visited Arcs
    let ptr_to_arc: HashMap<*const Mutex<PrecomputeNode>, Arc<Mutex<PrecomputeNode>>> = queue.into_iter()
        .map(|arc| (Arc::as_ptr(&arc), arc))
        .collect();

    // Iterate through node_to_id map (which contains all visited nodes) in ID order
    let mut sorted_nodes_to_process: Vec<_> = node_to_id.iter().collect();
    sorted_nodes_to_process.sort_by_key(|(_, id)| *id);

    for (node_ptr, &current_node_id) in sorted_nodes_to_process {
        let node_arc = ptr_to_arc.get(node_ptr)
            .expect("Node pointer not found in collected arcs during serialization (pass 2)").clone();

        let node_guard = node_arc.lock().expect("Mutex poisoned during serialization (node construction)");

        let mut s_children_map_for_node = BTreeMap::new();
        for (edge_key_opt, dest_map) in node_guard.children().iter() { // edge_key_opt is Option<GrammarTokenID>
            let mut s_dest_map_for_key = BTreeMap::new();
            for (child_arc_ptr_wrapper, edge_val) in dest_map.iter() {
                let child_arc = child_arc_ptr_wrapper.as_arc();
                let child_ptr = Arc::as_ptr(child_arc);
                let child_id = *node_to_id.get(&child_ptr)
                    .expect("Child node not found in ID map during serialization (pass 2 child lookup)");
                s_dest_map_for_key.insert(child_id, edge_val.clone());
            }
            if !s_dest_map_for_key.is_empty() {
                let string_key = match edge_key_opt {
                    Some(gtid) => gtid.0.to_string(), // GrammarTokenID(usize) -> usize.to_string()
                    None => OPTION_NONE_KEY_STR.to_string(),
                };
                s_children_map_for_node.insert(string_key, s_dest_map_for_key);
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

    let serializable_roots = root_arcs.iter().map(|(tokenizer_state_id, root_arc)| {
        let root_ptr = Arc::as_ptr(root_arc);
        let root_id = *node_to_id.get(&root_ptr)
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

        for (string_key, s_dest_map) in &s_node.children { // string_key is &String
            let mut dest_map_for_trie = BTreeMap::new();
            for (&child_id, edge_val) in s_dest_map {
                let child_arc = id_to_node.get(&child_id)
                    .expect("Child node not found in map during deserialization (pass 2 child lookup)").clone();
                dest_map_for_trie.insert(ArcPtrWrapper::new(child_arc), edge_val.clone());
            }
            if !dest_map_for_trie.is_empty() {
                let original_edge_key_opt = if string_key == OPTION_NONE_KEY_STR {
                    None
                } else {
                    // Parse the string key back to usize, then wrap in GrammarTokenID
                    // Handle potential parse error for robustness.
                    let val = string_key.parse::<usize>().map_err(|e| {
                        serde::de::Error::custom(format!(
                            "Failed to parse GrammarTokenID from string key '{}': {}",
                            string_key, e
                        ))
                    })?;
                    Some(GrammarTokenID(val))
                };
                current_node_guard.children_mut().insert(original_edge_key_opt, dest_map_for_trie);
            }
        }
    }

    // Finally, reconstruct the Precomputed map using the deserialized roots.
    let mut precomputed_result = Precomputed::new();
    for (tokenizer_state_id, root_id) in data.roots {
        let root_arc = id_to_node.get(&root_id)
            .expect("Root node not found in map during deserialization (reconstructing Precomputed)").clone();

        // We need to unwrap the Arc<Mutex<PrecomputeNode>> back into a plain PrecomputeNode
        // for the final Precomputed map. Since the deserialized graph structure should
        // be a DAG, each root node should ideally have only one strong reference (the one
        // we just retrieved from id_to_node).
        match Arc::try_unwrap(root_arc) {
            Ok(mutex) => {
                precomputed_result.insert(tokenizer_state_id, mutex.into_inner().expect("Mutex poisoned during root unwrap"));
            }
            Err(arc) => {
                 // If unwrap failed, it means this root node is unexpectedly shared.
                 // This could indicate an issue in the serialization/deserialization logic
                 // or the expected graph structure. For now, clone and warn.
                 eprintln!("Warning: Deserialized root node for tokenizer state {} is unexpectedly shared. Cloning.", tokenizer_state_id.0);
                 precomputed_result.insert(tokenizer_state_id, arc.lock().unwrap().clone());
            }
        }
    }

    Ok(precomputed_result)
}
