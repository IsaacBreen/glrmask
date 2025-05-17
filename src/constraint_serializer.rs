// src/constraint_serializer.rs
use crate::constraint::{
    GrammarConstraint, Precomputed, PrecomputeNode, PrecomputedNodeContents,
    PrecomputedFinalizer, LLMTokenBV,
};
use crate::tokenizer::{TokenizerStateID, LLMTokenID, LLMTokenMap};
use crate::types::TerminalID as GrammarTokenID;
use crate::finite_automata::Regex;
use crate::glr::parser::{GLRParser, ParseStateNodeContent, ParseState, MergeAndIntersect};
use crate::glr::table::{
    Stage7Table, StateID, ProductionID, NonTerminalID, TerminalID as GLRTerminalID, Stage7Row, Stage7ShiftsAndReduces,
};
use crate::glr::grammar::{Production, Symbol, Terminal, NonTerminal};
use crate::glr::items::Item;
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::u8set::U8Set;
use crate::datastructures::ArcPtrWrapper; // For reconstructing children map

use serde::{Serialize, Deserialize, Serializer, Deserializer};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::iter::FromIterator;
use bimap::BiBTreeMap;
use base64::{Engine as _, engine::general_purpose::STANDARD as Base64Engine};

// --- Serde Helper Modules ---

mod hybrid_bitset_serde {
    use super::*;
    pub fn serialize<S>(bitset: &HybridBitset, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        let v: Vec<usize> = bitset.iter().collect();
        v.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<HybridBitset, D::Error>
    where D: Deserializer<'de> {
        let v: Vec<usize> = Vec::deserialize(deserializer)?;
        Ok(HybridBitset::from_iter(v))
    }
}

mod optional_hybrid_bitset_serde {
    use super::*;
    pub fn serialize<S>(opt_bitset: &Option<HybridBitset>, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        #[derive(Serialize)]
        struct Helper<'a>(#[serde(with = "super::hybrid_bitset_serde")] &'a HybridBitset);
        opt_bitset.as_ref().map(Helper).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<HybridBitset>, D::Error>
    where D: Deserializer<'de> {
        #[derive(Deserialize)]
        struct Helper(#[serde(with = "super::hybrid_bitset_serde")] HybridBitset);
        Option::<Helper>::deserialize(deserializer).map(|opt_helper| opt_helper.map(|h| h.0))
    }
}

mod btreemap_tokenizerstate_llmtokenbv_serde {
    use super::*;
    pub fn serialize<S>(map: &BTreeMap<TokenizerStateID, LLMTokenBV>, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        let mut new_map = BTreeMap::new();
        for (k, bv) in map {
            #[derive(Serialize)]
            struct BVHelper<'a>(#[serde(with = "super::hybrid_bitset_serde")] &'a LLMTokenBV);
            new_map.insert(k, BVHelper(bv));
        }
        new_map.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<BTreeMap<TokenizerStateID, LLMTokenBV>, D::Error>
    where D: Deserializer<'de> {
        #[derive(Deserialize)]
        struct BVHelper(#[serde(with = "super::hybrid_bitset_serde")] HybridBitset);
        let map_helpers: BTreeMap<TokenizerStateID, BVHelper> = BTreeMap::deserialize(deserializer)?;
        Ok(map_helpers.into_iter().map(|(k, h)| (k, h.0)).collect())
    }
}

pub(crate) mod u8set_serde { // Made pub(crate) for Regex
    use super::*;
    pub fn serialize<S>(u8set: &U8Set, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        let v: Vec<u8> = u8set.iter().collect();
        v.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<U8Set, D::Error>
    where D: Deserializer<'de> {
        let v: Vec<u8> = Vec::deserialize(deserializer)?;
        Ok(U8Set::from_iter(v))
    }
}

mod vec_u8_key_base64_serde {
    use super::*;
    pub fn serialize<S>(vec: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        Base64Engine.encode(vec).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where D: Deserializer<'de> {
        let s = String::deserialize(deserializer)?;
        Base64Engine.decode(s).map_err(serde::de::Error::custom)
    }
}

// --- Key Wrappers for BTreeMap/BiBTreeMap if complex keys are problematic for JSON ---

#[derive(Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
struct SerGrammarTokenIDKey(String);

impl From<Option<GrammarTokenID>> for SerGrammarTokenIDKey {
    fn from(opt_gtid: Option<GrammarTokenID>) -> Self {
        match opt_gtid {
            None => SerGrammarTokenIDKey("N".to_string()),
            Some(gtid) => SerGrammarTokenIDKey(format!("S{}", gtid.0)),
        }
    }
}

impl From<SerGrammarTokenIDKey> for Option<GrammarTokenID> {
    fn from(s: SerGrammarTokenIDKey) -> Self {
        if s.0 == "N" {
            None
        } else if s.0.starts_with('S') {
            s.0[1..].parse().ok().map(GrammarTokenID)
        } else {
            panic!("Invalid SerGrammarTokenIDKey format: {}", s.0);
        }
    }
}

// --- Serializable Graph Structures ---

#[derive(Serialize, Deserialize, Debug)]
struct SerializablePrecomputedNodeData {
    id: usize,
    value: SerializablePrecomputedNodeContents,
    children: BTreeMap<SerGrammarTokenIDKey, BTreeMap<usize /*NodeID*/, SerdeLLMTokenBV>>,
    max_depth: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct SerdeLLMTokenBV(#[serde(with = "hybrid_bitset_serde")] LLMTokenBV);


#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct SerializablePrecomputedFinalizer {
    #[serde(with = "btreemap_tokenizerstate_llmtokenbv_serde")]
    pub content: BTreeMap<TokenizerStateID, LLMTokenBV>,
}

impl From<&PrecomputedFinalizer> for SerializablePrecomputedFinalizer {
    fn from(pf: &PrecomputedFinalizer) -> Self {
        Self { content: pf.content.clone() }
    }
}
impl From<SerializablePrecomputedFinalizer> for PrecomputedFinalizer {
    fn from(spf: SerializablePrecomputedFinalizer) -> Self {
        Self { content: spf.content }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct SerializablePrecomputedNodeContents {
    finalizers: BTreeMap<GrammarTokenID, SerializablePrecomputedFinalizer>,
    #[serde(with = "optional_hybrid_bitset_serde")]
    pub clean_end: Option<LLMTokenBV>,
    #[serde(with = "hybrid_bitset_serde")]
    pub active: LLMTokenBV,
}

impl From<&PrecomputedNodeContents> for SerializablePrecomputedNodeContents {
    fn from(pnc: &PrecomputedNodeContents) -> Self {
        Self {
            finalizers: pnc.finalizers().iter().map(|(k,v)| (*k, v.into())).collect(),
            clean_end: pnc.clean_end.clone(),
            active: pnc.active.clone(),
            // private field finalizers is reconstructed
        }
    }
}
impl From<SerializablePrecomputedNodeContents> for PrecomputedNodeContents {
    fn from(spnc: SerializablePrecomputedNodeContents) -> Self {
        Self {
            finalizers: spnc.finalizers.into_iter().map(|(k,v)| (k, v.into())).collect(),
            clean_end: spnc.clean_end,
            active: spnc.active,
            // private field finalizers is reconstructed
        }
    }
}


#[derive(Serialize, Deserialize, Debug)]
struct SerializablePrecomputedGraph {
    nodes: Vec<SerializablePrecomputedNodeData>,
    roots: BTreeMap<TokenizerStateID, usize /*NodeID*/>,
}

// --- Serializable Top-Level Structs ---

#[derive(Serialize, Deserialize, Debug)]
struct SerializableGLRParser {
    stage_7_table: Stage7Table,
    productions: Vec<Production>,
    terminal_map: BiBTreeMap<Terminal, GLRTerminalID>,
    non_terminal_map: BiBTreeMap<NonTerminal, NonTerminalID>,
    item_set_map: Vec<(BTreeSet<Item>, StateID)>, // Serialized as Vec of pairs
    start_state_id: StateID,
}

#[derive(Serialize, Deserialize, Debug)]
struct SerializableGrammarConstraint {
    tokenizer: Regex, // Regex needs to derive Serialize/Deserialize
    parser: SerializableGLRParser,
    precomputed_graph: SerializablePrecomputedGraph,
    llm_token_map: Vec<(String, LLMTokenID)>, // Vec<u8> key as base64 String
    token_name_map: BiBTreeMap<String, usize>, // Assuming bimap/serde handles this
    max_original_llm_token_id: usize,
    original_to_internal_id_bimap: BiBTreeMap<usize, usize>, // Assuming bimap/serde handles this
    internal_max_llm_token: usize,
}

// --- Conversion Implementations ---

impl From<&GrammarConstraint> for SerializableGrammarConstraint {
    fn from(gc: &GrammarConstraint) -> Self {
        // Serialize Precomputed Graph
        let mut node_map: HashMap<*const Mutex<PrecomputeNode>, usize> = HashMap::new();
        let mut serializable_nodes_vec = Vec::new();
        let mut q: VecDeque<Arc<Mutex<PrecomputeNode>>> = VecDeque::new();
        
        let mut roots_data = BTreeMap::new();

        for (tokenizer_state_id, root_node_trie) in &gc.precomputed {
            // The precomputed field stores PrecomputeNode directly, not Arc<Mutex<PrecomputeNode>>
            // For serialization, we need to treat these roots as the starting points of Arcs.
            // This part is tricky because PrecomputeNode::all_nodes expects an Arc.
            // We'll manually traverse from roots.
            let root_arc = Arc::new(Mutex::new(root_node_trie.clone())); // Temporarily wrap for traversal logic
            q.push_back(root_arc.clone());
            // The ID assigned here will be used for roots_data
        }
        
        let mut visited_for_id_assignment: HashSet<*const Mutex<PrecomputeNode>> = HashSet::new();
        let mut temp_all_nodes_ordered: Vec<Arc<Mutex<PrecomputeNode>>> = Vec::new();

        while let Some(node_arc) = q.pop_front() {
            let ptr = Arc::as_ptr(&node_arc);
            if visited_for_id_assignment.insert(ptr) {
                node_map.insert(ptr, temp_all_nodes_ordered.len());
                temp_all_nodes_ordered.push(node_arc.clone());
                
                let node_guard = node_arc.lock().expect("Mutex poisoned during serialization");
                for children_map_for_key in node_guard.children().values() {
                    for child_arc_ptr_wrapper in children_map_for_key.keys() {
                        q.push_back(child_arc_ptr_wrapper.as_arc().clone());
                    }
                }
            }
        }
        
        for node_arc in &temp_all_nodes_ordered {
            let node_guard = node_arc.lock().expect("Mutex poisoned during serialization");
            let id = *node_map.get(&Arc::as_ptr(node_arc)).unwrap();

            let mut s_children = BTreeMap::new();
            for (edge_key_opt, children_map_for_key) in node_guard.children() {
                let ser_edge_key = SerGrammarTokenIDKey::from(edge_key_opt.clone());
                let mut s_children_for_key = BTreeMap::new();
                for (child_arc_ptr_wrapper, edge_val_bv) in children_map_for_key {
                    let child_ptr = Arc::as_ptr(child_arc_ptr_wrapper.as_arc());
                    let child_id = *node_map.get(&child_ptr)
                        .expect("Child node not found in ID map during serialization");
                    s_children_for_key.insert(child_id, SerdeLLMTokenBV(edge_val_bv.clone()));
                }
                s_children.insert(ser_edge_key, s_children_for_key);
            }

            serializable_nodes_vec.push(SerializablePrecomputedNodeData {
                id,
                value: SerializablePrecomputedNodeContents::from(&node_guard.value),
                children: s_children,
                max_depth: node_guard.max_depth,
            });
        }

        for (tokenizer_state_id, root_node_trie) in &gc.precomputed {
            // Find the ID of this root_node_trie in our map.
            // This requires finding an Arc in temp_all_nodes_ordered that points to the same data.
            // This is a bit indirect. A better way would be to map root_node_trie directly if possible.
            // For now, we assume the initial roots pushed to queue will be processed and get IDs.
            // We need a stable way to link original root_node_trie to its ID.
            // Let's re-iterate roots and find their IDs from node_map by comparing content or pointer if stable.
            // The initial Arc::new(Mutex::new(root_node_trie.clone())) was temporary.
            // The `temp_all_nodes_ordered` contains the Arcs whose pointers are in `node_map`.
            // We need to find which of these corresponds to `root_node_trie`.
            let mut found_root_id = None;
            for arc_node in &temp_all_nodes_ordered {
                // This comparison is tricky. If PrecomputeNode is not Arc<Mutex<...>> in the map,
                // we need a way to identify it.
                // Assuming node_map keys are stable pointers to the *Mutex data* within the Arcs
                // that were part of the graph traversal.
                // The original `gc.precomputed` has `PrecomputeNode`, not `Arc<Mutex<PrecomputeNode>>`.
                // The traversal built `node_map` based on `Arc`s.
                // This part needs careful handling of identity.
                // Simplification: if `gc.precomputed` stored `Arc<Mutex<PrecomputeNode>>`, it would be direct.
                // Given it stores `PrecomputeNode`, we rely on the fact that the cloned Arcs during traversal
                // will lead to the correct IDs.
                // The `root_arc` created temporarily for queueing needs its ID.
                // The easiest is to iterate `temp_all_nodes_ordered` and if its content matches `root_node_trie`, use its ID.
                // This assumes `PrecomputeNode: PartialEq`. It derives `Clone`, `Debug`. It should be `PartialEq`.
                let locked_arc_node = arc_node.lock().unwrap();
                if *locked_arc_node == *root_node_trie { // Requires PrecomputeNode to be PartialEq
                    found_root_id = Some(*node_map.get(&Arc::as_ptr(arc_node)).unwrap());
                    break;
                }
            }
            roots_data.insert(*tokenizer_state_id, found_root_id.expect("Root ID not found for serialization"));
        }


        let precomputed_graph = SerializablePrecomputedGraph {
            nodes: serializable_nodes_vec,
            roots: roots_data,
        };

        // Serialize GLRParser
        let serializable_parser = SerializableGLRParser {
            stage_7_table: gc.parser.stage_7_table.clone(),
            productions: gc.parser.productions.clone(),
            terminal_map: gc.parser.terminal_map.clone(),
            non_terminal_map: gc.parser.non_terminal_map.clone(),
            item_set_map: gc.parser.item_set_map.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            start_state_id: gc.parser.start_state_id,
        };

        // Serialize LLMTokenMap (BiBTreeMap<Vec<u8>, LLMTokenID>)
        let llm_token_map_serializable = gc.llm_token_map.iter()
            .map(|(k_bytes, v_id)| (Base64Engine.encode(k_bytes), *v_id))
            .collect();

        Self {
            tokenizer: gc.tokenizer.clone(), // Assumes Regex is Clone
            parser: serializable_parser,
            precomputed_graph,
            llm_token_map: llm_token_map_serializable,
            token_name_map: gc.token_name_map.clone(),
            max_original_llm_token_id: gc.max_original_llm_token_id,
            original_to_internal_id_bimap: gc.original_to_internal_id_bimap.clone(),
            internal_max_llm_token: gc.internal_max_llm_token,
        }
    }
}

impl TryFrom<SerializableGrammarConstraint> for GrammarConstraint {
    type Error = String; // Or a custom error type

    fn try_from(sgc: SerializableGrammarConstraint) -> Result<Self, Self::Error> {
        // Deserialize Precomputed Graph
        let mut deserialized_nodes_map: HashMap<usize, Arc<Mutex<PrecomputeNode>>> = HashMap::new();
        
        // Pass 1: Create all nodes with their values but without children
        for s_node_data in &sgc.precomputed_graph.nodes {
            let node_value = PrecomputedNodeContents::from(s_node_data.value.clone());
            let mut new_node = PrecomputeNode::new(node_value);
            new_node.max_depth = s_node_data.max_depth;
            deserialized_nodes_map.insert(s_node_data.id, Arc::new(Mutex::new(new_node)));
        }

        // Pass 2: Populate children for each node
        for s_node_data in &sgc.precomputed_graph.nodes {
            let source_node_arc = deserialized_nodes_map.get(&s_node_data.id)
                .ok_or_else(|| format!("Node ID {} not found in deserialized map", s_node_data.id))?.clone();
            let mut source_node_guard = source_node_arc.lock().expect("Mutex poisoned during deserialization");

            for (ser_edge_key, s_children_for_key) in &s_node_data.children {
                let edge_key_opt: Option<GrammarTokenID> = ser_edge_key.clone().into();
                let mut children_map_for_key = BTreeMap::new();
                for (child_id, ser_edge_val_bv) in s_children_for_key {
                    let child_node_arc = deserialized_nodes_map.get(child_id)
                        .ok_or_else(|| format!("Child Node ID {} not found for parent {}", child_id, s_node_data.id))?.clone();
                    children_map_for_key.insert(ArcPtrWrapper::new(child_node_arc), ser_edge_val_bv.0.clone());
                }
                source_node_guard.children_mut().insert(edge_key_opt, children_map_for_key);
            }
        }
        
        let mut precomputed_map = BTreeMap::new();
        for (tokenizer_state_id, root_id) in sgc.precomputed_graph.roots {
            let root_node_arc = deserialized_nodes_map.get(&root_id)
                .ok_or_else(|| format!("Root Node ID {} not found for TokenizerStateID {:?}", root_id, tokenizer_state_id))?.clone();
            // We need to unwrap Arc<Mutex<PrecomputeNode>> to PrecomputeNode for the map
            let precompute_node = Arc::try_unwrap(root_node_arc)
                .map_err(|_e| "Failed to unwrap Arc for root node, still has multiple owners".to_string())?
                .into_inner()
                .map_err(|_p| "Mutex poisoned for root node".to_string())?;
            precomputed_map.insert(tokenizer_state_id, precompute_node);
        }

        // Deserialize GLRParser
        let parser = GLRParser {
            stage_7_table: sgc.parser.stage_7_table,
            productions: sgc.parser.productions,
            terminal_map: sgc.parser.terminal_map,
            non_terminal_map: sgc.parser.non_terminal_map,
            item_set_map: sgc.parser.item_set_map.into_iter().collect(),
            start_state_id: sgc.parser.start_state_id,
        };

        // Deserialize LLMTokenMap
        let mut llm_token_map = BiBTreeMap::new();
        for (k_base64, v_id) in sgc.llm_token_map {
            let k_bytes = Base64Engine.decode(k_base64)
                .map_err(|e| format!("Base64 decode error for llm_token_map key: {}", e))?;
            llm_token_map.insert(k_bytes, v_id);
        }

        Ok(GrammarConstraint {
            tokenizer: sgc.tokenizer,
            parser,
            precomputed: precomputed_map,
            llm_token_map,
            token_name_map: sgc.token_name_map,
            max_original_llm_token_id: sgc.max_original_llm_token_id,
            original_to_internal_id_bimap: sgc.original_to_internal_id_bimap,
            internal_max_llm_token: sgc.internal_max_llm_token,
        })
    }
}


/// Serializes the GrammarConstraint to a JSON string.
pub fn serialize_grammar_constraint(gc: &GrammarConstraint) -> Result<String, String> {
    let serializable_gc = SerializableGrammarConstraint::from(gc);
    serde_json::to_string_pretty(&serializable_gc).map_err(|e| e.to_string())
}

/// Deserializes a GrammarConstraint from a JSON string.
pub fn deserialize_grammar_constraint(json_str: &str) -> Result<GrammarConstraint, String> {
    let serializable_gc: SerializableGrammarConstraint = serde_json::from_str(json_str).map_err(|e| e.to_string())?;
    GrammarConstraint::try_from(serializable_gc)
}
