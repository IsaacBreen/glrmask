use std::collections::HashMap;
use crate::constraint::{GrammarConstraint, Precomputed, PrecomputeNode};
use crate::datastructures::gss::PrecomputeNode2;
use crate::types::{TerminalID as GrammarTokenID};
use crate::datastructures::trie::Trie;
use crate::tokenizer::{TokenizerStateID, LLMTokenID};
use std::collections::{HashSet, VecDeque, BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};
use bitvec::prelude::BitVec;
use crate::datastructures::hybrid_bitset::HybridBitset;
use bimap::BiBTreeMap;
use crate::datastructures::ArcPtrWrapper;
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use crate::datastructures::gss::LLMTokenBV;
use crate::glr::grammar::Terminal;
use crate::glr::table::StateID;

/// Creates a neat string representation of a HybridBitset, showing values as ranges.
fn format_hybrid_bitset_neatly(bv: &HybridBitset) -> String {
    if bv.is_empty() {
        return "[]".to_string();
    }

    const MAX_RANGES_TO_SHOW: usize = 5;
    let total_ranges = bv.inner().ranges_len();

    let ranges_to_show_str = bv.inner().ranges().take(MAX_RANGES_TO_SHOW).map(|range| {
        if range.start() == range.end() {
            format!("{}", range.start())
        } else {
            let range_end_str = if range.end() == &usize::MAX {
                "usize::MAX".to_string()
            } else {
                range.end().to_string()
            };
            format!("{}..={}", range.start(), range_end_str)
        }
    }).collect::<Vec<_>>().join(", ");

    let ellipsis = if total_ranges > MAX_RANGES_TO_SHOW { ", ..." } else { "" };
    format!("[{}{}]", ranges_to_show_str, ellipsis)
}

/// Helper function to format a HybridBitset for display, showing its debug representation
/// and a sample of the corresponding LLM tokens.
fn format_bv_with_tokens(
    bv: &LLMTokenBV,
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    limit: usize,
) -> String {
    let bv_neat_str = format_hybrid_bitset_neatly(bv);

    let (bimap, token_map) = match (original_internal_bimap, llm_token_map) {
        (Some(b), Some(t)) => (b, t),
        _ => return bv_neat_str, // If we don't have maps, just return the neat string.
    };

    let mut token_samples = Vec::new();
    for internal_id in bv.iter().take(limit) {
        if let Some(original_id) = bimap.get_by_right(&internal_id) {
            if let Some(token_bytes) = token_map.get_by_right(&LLMTokenID(*original_id)) {
                token_samples.push(format!("{:?}", String::from_utf8_lossy(token_bytes)));
            }
        }
    }

    if token_samples.is_empty() {
        return bv_neat_str;
    }

    let samples_str = token_samples.join(", ");
    let ellipsis = if bv.len() > limit { ", ..." } else { "" };

    format!("{} (e.g., [{}]{})", bv_neat_str, samples_str, ellipsis)
}

/// Helper function to recursively dump the structure of a PrecomputeNode Trie.
pub fn dump_precompute_trie_recursive(
    node_arc: &Arc<RwLock<PrecomputeNode>>,
    prefix: String,
    visited: &mut HashSet<*const PrecomputeNode>,
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
    token_name_map: Option<&BiBTreeMap<Terminal, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
) {
    let children_to_visit;

    {
        let node = node_arc.read().expect("RwLock poisoned during dump");
        // Collect children information while holding the lock
        children_to_visit = node.children().iter().flat_map(|(edge_key, dest_map)| {
            dest_map.iter().map(move |(child_wrapper, edge_val)| {
                (
                    edge_key.clone(),
                    edge_val.clone(),
                    child_wrapper.upgrade().unwrap(),
                )
            })
        }).collect::<Vec<_>>();
    }

    let num_children = children_to_visit.len();
    for (i, (edge_key, edge_val_bv, child_arc)) in children_to_visit.iter().enumerate() {
        let is_last = i == num_children - 1;
        let connector = if is_last { "└──" } else { "├──" };

        let edge_key_display = match edge_key {
            Some(gtid) => {
                if let Some(name_map) = token_name_map {
                    name_map.get_by_right(&gtid.0)
                        .map(|name| format!("'{}'", name))
                        .unwrap_or_else(|| format!("ID:{}", gtid.0))
                } else {
                    format!("ID:{}", gtid.0)
                }
            },
            None => "ε".to_string(),
        };

        let tokens_display = format_bv_with_tokens(&edge_val_bv, original_internal_bimap, llm_token_map, 5);

        let child_ptr;
        let child_info;
        let is_visited;
        let is_end_node;
        {
            let child_node = child_arc.read().unwrap();
            child_ptr = &*child_node as *const PrecomputeNode;
            is_visited = visited.contains(&child_ptr);
            is_end_node = child_node.value.end;
            child_info = format!("Node {:p} (MaxDepth: {}){}", child_ptr, child_node.max_depth, if is_end_node { " [END]" } else { "" });
        }

        // Don't shortcut the display for end nodes, even if they are visited.
        if is_visited && !is_end_node {
            println!("{}{} Edge {}: {} -> Ref to {}", prefix, connector, edge_key_display, tokens_display, child_info);
        } else {
            // Print full info for unvisited nodes or for any end node.
            println!("{}{} Edge {}: {} -> {}", prefix, connector, edge_key_display, tokens_display, child_info);

            // Only recurse if the node has not been visited before.
            // This prevents re-printing the children of a shared node and avoids cycles.
            // End nodes are leaves, so they won't recurse anyway.
            if !is_visited {
                visited.insert(child_ptr);
                let child_prefix = if is_last {
                    format!("{}   ", prefix)
                } else {
                    format!("{}│  ", prefix)
                };
                dump_precompute_trie_recursive(child_arc, child_prefix, visited, original_internal_bimap, token_name_map, llm_token_map);
            }
        }
    }
}

impl GrammarConstraint { // This is in constraint_extra.rs
    /// Dumps the structure of the precomputed Trie map for visualization.
    pub fn dump_precomputed(&self) {
        GrammarConstraint::_dump_precomputed(
            &self.precomputed,
            &self.llm_vocab.original_to_internal_id_bimap,
            &self.token_name_map,
            &self.llm_vocab.llm_token_map,
        );
    }

    pub fn _dump_precomputed(
        precomputed: &BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode>>>,
        original_to_internal_id_bimap: &BiBTreeMap<usize, usize>,
        token_name_map: &BiBTreeMap<Terminal, usize>,
        llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
    ) {
        println!("Dumping Precomputed Trie 1 Structure (showing original LLM Token IDs):");
        println!("===================================");

        let mut visited: HashSet<*const PrecomputeNode> = HashSet::new();
        for (tokenizer_state_id, root_node_trie) in precomputed {
            println!("\n--- Tokenizer State ID: {} ---", tokenizer_state_id.0);

            let root_ptr;
            let root_info;
            {
                let root_node = root_node_trie.read().unwrap();
                root_ptr = &*root_node as *const PrecomputeNode;
                root_info = format!("Root Node {:p} (MaxDepth: {}){}", root_ptr, root_node.max_depth, if root_node.value.end { " [END]" } else { "" });
            }
            println!("{}", root_info);

            if visited.contains(&root_ptr) {
                println!("  (Root already visited)");
            } else {
                visited.insert(root_ptr);
                dump_precompute_trie_recursive(root_node_trie, "".to_string(), &mut visited, Some(original_to_internal_id_bimap), Some(token_name_map), Some(llm_token_map));
            }
        }
        println!("\n===================================");
        println!("Dump Complete.");
    }

    /// Dumps the structure of the precomputed Trie 2 map for visualization.
    pub fn dump_precomputed2(&self) {
        GrammarConstraint::_dump_precomputed2(
            &self.precomputed2,
            &self.llm_vocab.original_to_internal_id_bimap,
            &self.llm_vocab.llm_token_map,
        );
    }

    pub fn _dump_precomputed2(precomputed2: &BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>, original_to_internal_id_bimap: &BiBTreeMap<usize, usize>, llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>) {
        println!("Dumping Precomputed Trie 2 Structure (showing original LLM Token IDs):");
        println!("===================================");

        let mut visited: HashSet<*const PrecomputeNode2> = HashSet::new();
        for (tokenizer_state_id, root_node_trie) in precomputed2 {
            println!("\n--- Tokenizer State ID: {} ---", tokenizer_state_id.0);

            let root_ptr;
            let root_info;
            {
                let root_node = root_node_trie.read().unwrap();
                root_ptr = &*root_node as *const PrecomputeNode2;
                root_info = format!("Root Node {:p} (MaxDepth: {}){}", root_ptr, root_node.max_depth, if root_node.value.end { " [END]" } else { "" });
            }
            println!("{}", root_info);

            if visited.contains(&root_ptr) {
                println!("  (Root already visited)");
            } else {
                visited.insert(root_ptr);
                dump_precompute_trie2_recursive(
                    root_node_trie,
                    "".to_string(),
                    &mut visited,
                    Some(&original_to_internal_id_bimap),
                    Some(&llm_token_map),
                );
            }
        }
        println!("\n===================================");
        println!("Dump Complete.");
    }
}

pub fn dump_precompute_trie2_recursive(
    node_arc: &Arc<RwLock<PrecomputeNode2>>,
    prefix: String,
    visited: &mut HashSet<*const PrecomputeNode2>,
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
) {
    let children_to_visit = {
        let node = node_arc.read().expect("RwLock poisoned during dump");
        node.children().iter().flat_map(|(edge_key, dest_map)| {
            dest_map.iter().map(move |(child_wrapper, edge_val)| {
                (
                    edge_key.clone(),
                    edge_val.clone(),
                    child_wrapper.upgrade().unwrap(),
                )
            })
        }).collect::<Vec<_>>()
    };

    for (i, (edge_key, edge_val_bv, child_arc)) in children_to_visit.iter().enumerate() {
        let is_last = i == children_to_visit.len() - 1;
        let connector = if is_last { "└──" } else { "├──" };

        let (pop_len, state_id_opt) = edge_key;
        let edge_key_display = format!("(pop: {}, state: {})", pop_len, state_id_opt.map_or("None".to_string(), |sid| sid.0.to_string()));
        let tokens_display = format_bv_with_tokens(edge_val_bv, original_internal_bimap, llm_token_map, 5);

        let (child_ptr, child_info, is_visited, is_end_node) = {
            let child_node = child_arc.read().unwrap();
            let ptr = Arc::as_ptr(child_arc) as *const PrecomputeNode2;
            (ptr, format!("Node {:p} (MaxDepth: {}){}", ptr, child_node.max_depth, if child_node.value.end { " [END]" } else { "" }), visited.contains(&ptr), child_node.value.end)
        };

        if is_visited && !is_end_node {
            println!("{}{} Edge {}: {} -> Ref to {}", prefix, connector, edge_key_display, tokens_display, child_info);
        } else {
            println!("{}{} Edge {}: {} -> {}", prefix, connector, edge_key_display, tokens_display, child_info);
            if !is_visited {
                visited.insert(child_ptr);
                let child_prefix = if is_last { format!("{}   ", prefix) } else { format!("{}│  ", prefix) };
                dump_precompute_trie2_recursive(child_arc, child_prefix, visited, original_internal_bimap, llm_token_map);
            }
        }
    }
}


// Add this struct definition before impl GrammarConstraint
#[derive(Default, Debug)]
pub struct PrecomputeStats {
    // Gross counts (before sharing/merging reduces them in the final structure)
    pub initial_root_nodes_created: usize,

    // Final structure stats (net counts, after all processing and sharing)
    pub final_unique_nodes_count: usize,
    pub final_root_nodes_count: usize,
    pub final_non_root_internal_nodes_count: usize, // Renamed from final_internal_nodes_count
    pub final_leaf_nodes_count: usize,             // New field
    pub final_edges_count: usize,
    pub final_edges_with_none_key: usize,
    pub final_edges_with_some_key: usize,
    pub final_nodes_with_clean_end: usize,

    // For average edge occupancy per key type
    pub final_total_occupancy_sum_for_some_keys: usize,
    pub final_num_occupied_some_edge_keys: usize,
    pub final_total_occupancy_sum_for_none_keys: usize,
    pub final_num_occupied_none_edge_keys: usize,

    // New fields for grammar token edge key statistics
    pub final_grammar_token_edge_key_counts: BTreeMap<GrammarTokenID, usize>,
    pub final_grammar_token_edge_fanouts_dist: BTreeMap<GrammarTokenID, Vec<usize>>,
    pub final_grammar_token_edge_token_set_sizes_dist: BTreeMap<GrammarTokenID, Vec<usize>>,

    // New fields for edge pruning statistics
    pub final_edges_pruned_total: usize,
    pub final_edges_pruned_by_token: BTreeMap<GrammarTokenID, usize>,

    pub edges_pruned_by_terminal_sequence: usize,
    pub final_total_ranges_in_bvs: usize,
}

// Manual impl for PrecomputeStats
impl JSONConvertible for PrecomputeStats {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("initial_root_nodes_created".to_string(), self.initial_root_nodes_created.to_json());
        obj.insert("final_unique_nodes_count".to_string(), self.final_unique_nodes_count.to_json());
        obj.insert("final_root_nodes_count".to_string(), self.final_root_nodes_count.to_json());
        obj.insert("final_non_root_internal_nodes_count".to_string(), self.final_non_root_internal_nodes_count.to_json());
        obj.insert("final_leaf_nodes_count".to_string(), self.final_leaf_nodes_count.to_json());
        obj.insert("final_edges_count".to_string(), self.final_edges_count.to_json());
        obj.insert("final_edges_with_none_key".to_string(), self.final_edges_with_none_key.to_json());
        obj.insert("final_edges_with_some_key".to_string(), self.final_edges_with_some_key.to_json());
        obj.insert("final_nodes_with_clean_end".to_string(), self.final_nodes_with_clean_end.to_json());
        obj.insert("final_total_occupancy_sum_for_some_keys".to_string(), self.final_total_occupancy_sum_for_some_keys.to_json());
        obj.insert("final_num_occupied_some_edge_keys".to_string(), self.final_num_occupied_some_edge_keys.to_json());
        obj.insert("final_total_occupancy_sum_for_none_keys".to_string(), self.final_num_occupied_none_edge_keys.to_json());
        obj.insert("final_num_occupied_none_edge_keys".to_string(), self.final_num_occupied_none_edge_keys.to_json());
        obj.insert("final_grammar_token_edge_key_counts".to_string(), self.final_grammar_token_edge_key_counts.to_json());
        obj.insert("final_grammar_token_edge_fanouts_dist".to_string(), self.final_grammar_token_edge_fanouts_dist.to_json());
        obj.insert("final_grammar_token_edge_token_set_sizes_dist".to_string(), self.final_grammar_token_edge_token_set_sizes_dist.to_json());
        obj.insert("final_edges_pruned_total".to_string(), self.final_edges_pruned_total.to_json());
        obj.insert("final_edges_pruned_by_token".to_string(), self.final_edges_pruned_by_token.to_json());
        obj.insert("edges_pruned_by_terminal_sequence".to_string(), self.edges_pruned_by_terminal_sequence.to_json());
        obj.insert("final_total_ranges_in_bvs".to_string(), self.final_total_ranges_in_bvs.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let initial_root_nodes_created = obj.remove("initial_root_nodes_created").ok_or_else(|| "Missing field initial_root_nodes_created for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_unique_nodes_count = obj.remove("final_unique_nodes_count").ok_or_else(|| "Missing field final_unique_nodes_count for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_root_nodes_count = obj.remove("final_root_nodes_count").ok_or_else(|| "Missing field final_root_nodes_count for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_non_root_internal_nodes_count = obj.remove("final_non_root_internal_nodes_count").ok_or_else(|| "Missing field final_non_root_internal_nodes_count for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_leaf_nodes_count = obj.remove("final_leaf_nodes_count").ok_or_else(|| "Missing field final_leaf_nodes_count for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_edges_count = obj.remove("final_edges_count").ok_or_else(|| "Missing field final_edges_count for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_edges_with_none_key = obj.remove("final_edges_with_none_key").ok_or_else(|| "Missing field final_edges_with_none_key for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_edges_with_some_key = obj.remove("final_edges_with_some_key").ok_or_else(|| "Missing field final_edges_with_some_key for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_nodes_with_clean_end = obj.remove("final_nodes_with_clean_end").ok_or_else(|| "Missing field final_nodes_with_clean_end for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_total_occupancy_sum_for_some_keys = obj.remove("final_total_occupancy_sum_for_some_keys").ok_or_else(|| "Missing field final_total_occupancy_sum_for_some_keys for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_num_occupied_some_edge_keys = obj.remove("final_num_occupied_some_edge_keys").ok_or_else(|| "Missing field final_num_occupied_some_edge_keys for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_total_occupancy_sum_for_none_keys = obj.remove("final_total_occupancy_sum_for_none_keys").ok_or_else(|| "Missing field final_total_occupancy_sum_for_none_keys for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_num_occupied_none_edge_keys = obj.remove("final_num_occupied_none_edge_keys").ok_or_else(|| "Missing field final_num_occupied_none_edge_keys for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_grammar_token_edge_key_counts = obj.remove("final_grammar_token_edge_key_counts").ok_or_else(|| "Missing field final_grammar_token_edge_key_counts for PrecomputeStats".to_string()).and_then(|n| BTreeMap::<GrammarTokenID, usize>::from_json(n))?;
                let final_grammar_token_edge_fanouts_dist = obj.remove("final_grammar_token_edge_fanouts_dist").ok_or_else(|| "Missing field final_grammar_token_edge_fanouts_dist for PrecomputeStats".to_string()).and_then(|n| BTreeMap::<GrammarTokenID, Vec<usize>>::from_json(n))?;
                let final_grammar_token_edge_token_set_sizes_dist = obj.remove("final_grammar_token_edge_token_set_sizes_dist").ok_or_else(|| "Missing field final_grammar_token_edge_token_set_sizes_dist for PrecomputeStats".to_string()).and_then(|n| BTreeMap::<GrammarTokenID, Vec<usize>>::from_json(n))?;
                let final_edges_pruned_total = obj.remove("final_edges_pruned_total").ok_or_else(|| "Missing field final_edges_pruned_total for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_edges_pruned_by_token = obj.remove("final_edges_pruned_by_token").ok_or_else(|| "Missing field final_edges_pruned_by_token for PrecomputeStats".to_string()).and_then(|n| BTreeMap::<GrammarTokenID, usize>::from_json(n))?;
                let edges_pruned_by_terminal_sequence = obj.remove("edges_pruned_by_terminal_sequence").ok_or_else(|| "Missing field edges_pruned_by_terminal_sequence for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                let final_total_ranges_in_bvs = obj.remove("final_total_ranges_in_bvs").ok_or_else(|| "Missing field final_total_ranges_in_bvs for PrecomputeStats".to_string()).and_then(usize::from_json)?;
                Ok(PrecomputeStats {
                    initial_root_nodes_created,
                    final_unique_nodes_count,
                    final_root_nodes_count,
                    final_non_root_internal_nodes_count,
                    final_leaf_nodes_count,
                    final_edges_count,
                    final_edges_with_none_key,
                    final_edges_with_some_key,
                    final_nodes_with_clean_end,
                    final_total_occupancy_sum_for_some_keys,
                    final_num_occupied_some_edge_keys,
                    final_total_occupancy_sum_for_none_keys,
                    final_num_occupied_none_edge_keys,
                    final_grammar_token_edge_key_counts,
                    final_grammar_token_edge_fanouts_dist,
                    final_grammar_token_edge_token_set_sizes_dist,
                    final_edges_pruned_total,
                    final_edges_pruned_by_token,
                    edges_pruned_by_terminal_sequence,
                    final_total_ranges_in_bvs,
                })
            }
            _ => Err("Expected JSONNode::Object for PrecomputeStats".to_string()),
        }
    }
}

// Add this struct definition before impl GrammarConstraint
#[derive(Default, Debug)]
pub struct PrecomputeStats2 {
    // Final structure stats
    pub final_unique_nodes_count: usize,
    pub final_root_nodes_count: usize,
    pub final_non_root_internal_nodes_count: usize,
    pub final_leaf_nodes_count: usize,
    pub final_edges_count: usize,
    pub final_nodes_with_clean_end: usize,
    pub final_total_ranges_in_bvs: usize,

    // Stats about edge keys (k, Option<StateID>)
    pub final_k_dist: BTreeMap<usize, usize>, // count of edges for each k
    pub final_state_id_dist: BTreeMap<Option<StateID>, usize>, // count of edges for each state_id
    pub final_edge_fanouts_dist: Vec<usize>,
    pub final_edge_token_set_sizes_dist: Vec<usize>,
}

impl JSONConvertible for PrecomputeStats2 {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("final_unique_nodes_count".to_string(), self.final_unique_nodes_count.to_json());
        obj.insert("final_root_nodes_count".to_string(), self.final_root_nodes_count.to_json());
        obj.insert("final_non_root_internal_nodes_count".to_string(), self.final_non_root_internal_nodes_count.to_json());
        obj.insert("final_leaf_nodes_count".to_string(), self.final_leaf_nodes_count.to_json());
        obj.insert("final_edges_count".to_string(), self.final_edges_count.to_json());
        obj.insert("final_nodes_with_clean_end".to_string(), self.final_nodes_with_clean_end.to_json());
        obj.insert("final_total_ranges_in_bvs".to_string(), self.final_total_ranges_in_bvs.to_json());
        obj.insert("final_k_dist".to_string(), self.final_k_dist.to_json());
        obj.insert("final_state_id_dist".to_string(), self.final_state_id_dist.to_json());
        obj.insert("final_edge_fanouts_dist".to_string(), self.final_edge_fanouts_dist.to_json());
        obj.insert("final_edge_token_set_sizes_dist".to_string(), self.final_edge_token_set_sizes_dist.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => Ok(PrecomputeStats2 {
                final_unique_nodes_count: obj.remove("final_unique_nodes_count").ok_or_else(|| "Missing field final_unique_nodes_count".to_string())?.from_json()?,
                final_root_nodes_count: obj.remove("final_root_nodes_count").ok_or_else(|| "Missing field final_root_nodes_count".to_string())?.from_json()?,
                final_non_root_internal_nodes_count: obj.remove("final_non_root_internal_nodes_count").ok_or_else(|| "Missing field final_non_root_internal_nodes_count".to_string())?.from_json()?,
                final_leaf_nodes_count: obj.remove("final_leaf_nodes_count").ok_or_else(|| "Missing field final_leaf_nodes_count".to_string())?.from_json()?,
                final_edges_count: obj.remove("final_edges_count").ok_or_else(|| "Missing field final_edges_count".to_string())?.from_json()?,
                final_nodes_with_clean_end: obj.remove("final_nodes_with_clean_end").ok_or_else(|| "Missing field final_nodes_with_clean_end".to_string())?.from_json()?,
                final_total_ranges_in_bvs: obj.remove("final_total_ranges_in_bvs").ok_or_else(|| "Missing field final_total_ranges_in_bvs".to_string())?.from_json()?,
                final_k_dist: obj.remove("final_k_dist").ok_or_else(|| "Missing field final_k_dist".to_string())?.from_json()?,
                final_state_id_dist: obj.remove("final_state_id_dist").ok_or_else(|| "Missing field final_state_id_dist".to_string())?.from_json()?,
                final_edge_fanouts_dist: obj.remove("final_edge_fanouts_dist").ok_or_else(|| "Missing field final_edge_fanouts_dist".to_string())?.from_json()?,
                final_edge_token_set_sizes_dist: obj.remove("final_edge_token_set_sizes_dist").ok_or_else(|| "Missing field final_edge_token_set_sizes_dist".to_string())?.from_json()?,
            }),
            _ => Err("Expected JSONNode::Object for PrecomputeStats2".to_string()),
        }
    }
}

pub fn calculate_final_stats2(
    precomputed_roots: &BTreeMap<TokenizerStateID, Arc<RwLock<PrecomputeNode2>>>,
    stats: &mut PrecomputeStats2,
) {
    crate::debug!(2, "Calculating final precompute 2 statistics...");

    let mut all_reachable_nodes: BTreeMap<*const PrecomputeNode2, Arc<RwLock<PrecomputeNode2>>> = BTreeMap::new();
    let mut queue: VecDeque<Arc<RwLock<PrecomputeNode2>>> = precomputed_roots.values().cloned().collect();
    let mut visited_data_ptrs: HashSet<*const PrecomputeNode2> = HashSet::new();

    while let Some(node_arc) = queue.pop_front() {
        let (children_to_queue, node_ptr) = {
            let node_guard = node_arc.read().unwrap();
            let ptr = &*node_guard as *const PrecomputeNode2;
            let children = node_guard.children()
                .values()
                .flat_map(|dest_map| {
                    dest_map
                        .keys()
                        .filter_map(|wrapper| wrapper.upgrade())
                })
                .collect::<Vec<_>>();
            (children, ptr)
        };

        if visited_data_ptrs.insert(node_ptr) {
            all_reachable_nodes.insert(node_ptr, node_arc.clone());
            for child_arc in children_to_queue {
                queue.push_back(child_arc);
            }
        }
    }

    stats.final_unique_nodes_count = all_reachable_nodes.len();

    let root_node_pointers: HashSet<*const PrecomputeNode2> = precomputed_roots
        .values()
        .map(|arc| {
            let guard = arc.read().unwrap();
            &*guard as *const PrecomputeNode2
        })
        .collect();
    stats.final_root_nodes_count = root_node_pointers.len();

    // Initialize stats fields
    stats.final_edges_count = 0;
    stats.final_nodes_with_clean_end = 0;
    stats.final_non_root_internal_nodes_count = 0;
    stats.final_leaf_nodes_count = 0;
    stats.final_total_ranges_in_bvs = 0;
    stats.final_k_dist.clear();
    stats.final_state_id_dist.clear();
    stats.final_edge_fanouts_dist.clear();
    stats.final_edge_token_set_sizes_dist.clear();

    for (node_ptr, node_arc) in &all_reachable_nodes {
        let node_guard = node_arc.read().expect("RwLock poisoned during final stats calculation");

        if !root_node_pointers.contains(node_ptr) {
            if node_guard.children().is_empty() {
                stats.final_leaf_nodes_count += 1;
            } else {
                stats.final_non_root_internal_nodes_count += 1;
            }
        }

        for (edge_key, dest_map) in node_guard.children() {
            let (k, state_id_opt) = edge_key;
            let num_edges = dest_map.len();
            stats.final_edges_count += num_edges;

            *stats.final_k_dist.entry(*k).or_default() += num_edges;
            *stats.final_state_id_dist.entry(*state_id_opt).or_default() += num_edges;
            
            stats.final_edge_fanouts_dist.push(num_edges);
            for bv in dest_map.values() {
                stats.final_edge_token_set_sizes_dist.push(bv.len());
                stats.final_total_ranges_in_bvs += bv.inner().ranges_len();
            }
        }

        if node_guard.value.end {
            stats.final_nodes_with_clean_end += 1;
        }
    }
    crate::debug!(2, "Finished calculating final precompute 2 statistics.");
}

pub fn print_precompute_stats2(
    stats: &PrecomputeStats2,
) {
    println!("--- Precomputation 2 Statistics ---");

    println!("\nNode Counts Breakdown:");
    println!("  There are:");
    println!("  - {} unique nodes, of which", stats.final_unique_nodes_count);
    println!("    - {} are roots", stats.final_root_nodes_count);
    let non_root_count = stats.final_unique_nodes_count.saturating_sub(stats.final_root_nodes_count);
    println!("    - {} are non-roots, of which", non_root_count);
    println!("        - {} are internal (non-root, non-leaf)", stats.final_non_root_internal_nodes_count);
    println!("        - {} are leaves (non-root)", stats.final_leaf_nodes_count);

    println!("\nFinal Graph Structure (Trie 2):");
    println!("  Unique Nodes: {}", stats.final_unique_nodes_count);
    println!("  Total Edges: {}", stats.final_edges_count);
    println!("  Nodes with Clean End: {}", stats.final_nodes_with_clean_end);
    println!("  Total ranges in all HybridBitsets: {}", stats.final_total_ranges_in_bvs);

    println!("\nEdge Key Statistics (k = pop length):");
    let mut k_dist: Vec<_> = stats.final_k_dist.iter().collect();
    k_dist.sort_by_key(|&(_, count)| std::cmp::Reverse(*count));
    for (k, count) in k_dist.iter().take(10) {
        println!("  - k={:<4}: {:>6} edges", k, count);
    }
    if k_dist.len() > 10 {
        println!("  ... ({} more)", k_dist.len() - 10);
    }

    println!("\nEdge Key Statistics (State ID):");
    let mut state_dist: Vec<_> = stats.final_state_id_dist.iter().collect();
    state_dist.sort_by_key(|&(_, count)| std::cmp::Reverse(*count));
    for (state_id_opt, count) in state_dist.iter().take(10) {
        let state_str = state_id_opt.map_or("None".to_string(), |s| s.0.to_string());
        println!("  - State {:<4}: {:>6} edges", state_str, count);
    }
    if state_dist.len() > 10 {
        println!("  ... ({} more)", state_dist.len() - 10);
    }

    let (fanout_sum, fanout_avg, fanout_med) = calculate_stats_from_vec_usize(&stats.final_edge_fanouts_dist);
    let (tokens_sum, tokens_avg, tokens_med) = calculate_stats_from_vec_usize(&stats.final_edge_token_set_sizes_dist);
    
    println!("\nEdge Distribution Stats:");
    println!("  Fanout (destinations per edge key):");
    println!("    Sum: {}, Avg: {:.2}, Median: {:.2}", fanout_sum, fanout_avg.unwrap_or(0.0), fanout_med.unwrap_or(0.0));
    println!("  Token Set Size (tokens per edge):");
    println!("    Sum: {}, Avg: {:.2}, Median: {:.2}", tokens_sum, tokens_avg.unwrap_or(0.0), tokens_med.unwrap_or(0.0));

    println!("---------------------------------");
}

/// Helper function to calculate sum, mean, and median from Vec<usize>
fn calculate_stats_from_vec_usize(numbers: &Vec<usize>) -> (usize, Option<f64>, Option<f64>) {
    if numbers.is_empty() {
        return (0, None, None);
    }
    let sum: usize = numbers.iter().sum();
    let mean: Option<f64> = Some(sum as f64 / numbers.len() as f64);

    let mut sorted_numbers = numbers.clone();
    sorted_numbers.sort_unstable();
    let len = sorted_numbers.len();
    let mid = len / 2;
    let median: Option<f64> = if len == 0 {
        None
    } else if len % 2 == 0 {
        Some((sorted_numbers[mid - 1] as f64 + sorted_numbers[mid] as f64) / 2.0)
    } else {
        Some(sorted_numbers[mid] as f64)
    };
    (sum, mean, median)
}

