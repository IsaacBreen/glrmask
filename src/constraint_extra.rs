use std::collections::HashMap;
use crate::constraint::{GrammarConstraint, Precomputed, PrecomputeNode, PrecomputeNodeIndex, PrecomputeNode2Index, Trie1GodWrapper, Trie2GodWrapper, PrecomputeNode2, PrecomputeNode3Index, Trie3GodWrapper};
use crate::types::{TerminalID as GrammarTokenID};
use crate::datastructures::trie::{Trie, Trie2Index};
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
    node_arc: &PrecomputeNodeIndex,
    prefix: String,
    visited: &mut HashSet<PrecomputeNodeIndex>,
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
    token_name_map: Option<&BiBTreeMap<Terminal, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    trie1_god: &Trie1GodWrapper,
) {
    let children_to_visit;

    {
        let node = node_arc.read(trie1_god).expect("RwLock poisoned during dump");
        // Collect children information while holding the lock
        children_to_visit = node.children().iter().flat_map(|(edge_key, dest_map)| {
            dest_map.iter().map(move |(child_wrapper, edge_val)| {
                (
                    edge_key.clone(),
                    edge_val.clone(),
                    child_wrapper.as_arc().clone(),
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
            let child_node = child_arc.read(trie1_god).unwrap();
            child_ptr = child_arc;
            is_visited = visited.contains(&child_ptr);
            is_end_node = child_node.value.end;
            let live_tokens_str = format_bv_with_tokens(&child_node.value.live_tokens, original_internal_bimap, llm_token_map, 5);
            child_info = format!("Node {} (MaxDepth: {}){} [Live: {}]", child_ptr, child_node.max_depth, if is_end_node { " [END]" } else { "" }, live_tokens_str);
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
                visited.insert(*child_ptr);
                let child_prefix = if is_last {
                    format!("{}   ", prefix)
                } else {
                    format!("{}│  ", prefix)
                };
                dump_precompute_trie_recursive(child_arc, child_prefix, visited, original_internal_bimap, token_name_map, llm_token_map, trie1_god);
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
            &self.trie1_god,
        );
    }

    pub fn _dump_precomputed(
        precomputed: &BTreeMap<TokenizerStateID, PrecomputeNodeIndex>,
        original_to_internal_id_bimap: &BiBTreeMap<usize, usize>,
        token_name_map: &BiBTreeMap<Terminal, usize>,
        llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        trie1_god: &Trie1GodWrapper,
    ) {
        println!("Dumping Precomputed Trie 1 Structure (showing original LLM Token IDs):");
        println!("===================================");

        let mut visited: HashSet<PrecomputeNodeIndex> = HashSet::new();
        for (tokenizer_state_id, root_node_trie) in precomputed {
            println!("\n--- Tokenizer State ID: {} ---", tokenizer_state_id.0);

            let root_ptr;
            let root_info;
            {
                let root_node = root_node_trie.read(trie1_god).unwrap();
                root_ptr = root_node_trie;
                let live_tokens_str = format_bv_with_tokens(&root_node.value.live_tokens, Some(original_to_internal_id_bimap), Some(llm_token_map), 5);
                root_info = format!("Root Node {} (MaxDepth: {}){} [Live: {}]", root_ptr, root_node.max_depth, if root_node.value.end { " [END]" } else { "" }, live_tokens_str);
            }
            println!("{}", root_info);

            if visited.contains(&root_ptr) {
                println!("  (Root already visited)");
            } else {
                visited.insert(*root_ptr);
                dump_precompute_trie_recursive(root_node_trie, "".to_string(), &mut visited, Some(original_to_internal_id_bimap), Some(token_name_map), Some(llm_token_map), trie1_god);
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
            &self.trie2_god,
        );
    }

    pub fn _dump_precomputed2(precomputed2: &BTreeMap<TokenizerStateID, PrecomputeNode2Index>, original_to_internal_id_bimap: &BiBTreeMap<usize, usize>, llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>, trie2_god: &Trie2GodWrapper) {
        println!("Dumping Precomputed Trie 2 Structure (showing original LLM Token IDs):");
        println!("===================================");

        let mut visited: HashSet<PrecomputeNode2Index> = HashSet::new();
        for (tokenizer_state_id, root_node_trie) in precomputed2 {
            println!("\n--- Tokenizer State ID: {} ---", tokenizer_state_id.0);

            let root_ptr;
            let root_info;
            {
                let root_node = root_node_trie.read(trie2_god).unwrap();
                root_ptr = root_node_trie;
                let live_tokens_str = format_bv_with_tokens(&root_node.value.live_tokens, Some(original_to_internal_id_bimap), Some(llm_token_map), 5);
                root_info = format!("Root Node {} (MaxDepth: {}){} [Live: {}]", root_ptr, root_node.max_depth, if root_node.value.end { " [END]" } else { "" }, live_tokens_str);
            }
            println!("{}", root_info);

            if visited.contains(&root_ptr) {
                println!("  (Root already visited)");
            } else {
                visited.insert(*root_ptr);
                dump_precompute_trie2_recursive(
                    root_node_trie,
                    "".to_string(),
                    &mut visited,
                    Some(&original_to_internal_id_bimap),
                    Some(&llm_token_map),
                    trie2_god,
                );
            }
        }
        println!("\n===================================");
        println!("Dump Complete.");
    }

    /// Dumps the structure of the precomputed Trie 3 map for visualization.
    pub fn dump_precomputed3(&self) {
        GrammarConstraint::_dump_precomputed3(
            &self.precomputed3,
            &self.llm_vocab.original_to_internal_id_bimap,
            &self.llm_vocab.llm_token_map,
            &self.trie3_god,
        );
    }

    pub fn _dump_precomputed3(precomputed3: &BTreeMap<TokenizerStateID, PrecomputeNode3Index>, original_to_internal_id_bimap: &BiBTreeMap<usize, usize>, llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>, trie3_god: &Trie3GodWrapper) {
        println!("Dumping Precomputed Trie 3 Structure (showing original LLM Token IDs):");
        println!("===================================");

        let mut visited: HashSet<PrecomputeNode3Index> = HashSet::new();
        for (tokenizer_state_id, root_node_trie) in precomputed3 {
            println!("\n--- Tokenizer State ID: {} ---", tokenizer_state_id.0);

            let root_ptr;
            let root_info;
            {
                let root_node = root_node_trie.read(trie3_god).unwrap();
                root_ptr = root_node_trie;
                root_info = format!("Root Node {} (MaxDepth: {}){}", root_ptr, root_node.max_depth, if root_node.value.end() { " [END]" } else { "" });
            }
            println!("{}", root_info);

            if visited.contains(&root_ptr) {
                println!("  (Root already visited)");
            } else {
                visited.insert(*root_ptr);
                dump_precompute_trie3_recursive(
                    root_node_trie,
                    "".to_string(),
                    &mut visited,
                    Some(&original_to_internal_id_bimap),
                    Some(&llm_token_map),
                    trie3_god,
                );
            }
        }
        println!("\n===================================");
        println!("Dump Complete.");
    }
}

pub fn dump_precompute_trie2_recursive(
    node_arc: &Trie2Index,
    prefix: String,
    visited: &mut HashSet<PrecomputeNode2Index>,
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    trie2_god: &Trie2GodWrapper,
) {
    let children_to_visit = {
        let node = node_arc.read(trie2_god).expect("RwLock poisoned during dump");
        node.children().iter().flat_map(|(edge_key, dest_map)| {
            dest_map.iter().map(move |(child_wrapper, edge_val)| {
                (
                    edge_key.clone(),
                    edge_val.clone(),
                    child_wrapper.as_arc().clone(),
                )
            })
        }).collect::<Vec<_>>()
    };

    for (i, (edge_key, edge_val_bv, child_arc)) in children_to_visit.iter().enumerate() {
        let is_last = i == children_to_visit.len() - 1;
        let connector = if is_last { "└──" } else { "├──" };
        let (pop_len, state_id_opt) = edge_key; let edge_key_display = format!("(pop: {}, state: {})", pop_len, state_id_opt.map_or("None".to_string(), |sid| sid.0.to_string())); let tokens_display = format_bv_with_tokens(edge_val_bv, original_internal_bimap, llm_token_map, 5);
        let (child_ptr, child_info, is_visited, is_end_node) = {
            let child_node = child_arc.read(trie2_god).unwrap();
            let ptr = child_arc;
            let live_tokens_str = format_bv_with_tokens(&child_node.value.live_tokens, original_internal_bimap, llm_token_map, 5);
            (ptr, format!("Node {} (MaxDepth: {}){} [Live: {}]", ptr, child_node.max_depth, if child_node.value.end { " [END]" } else { "" }, live_tokens_str), visited.contains(&ptr), child_node.value.end)
        };

        if is_visited && !is_end_node {
            println!("{}{} Edge {}: {} -> Ref to {}", prefix, connector, edge_key_display, tokens_display, child_info);
        } else {
            println!("{}{} Edge {}: {} -> {}", prefix, connector, edge_key_display, tokens_display, child_info);
            if !is_visited {
                visited.insert(*child_ptr);
                let child_prefix = if is_last { format!("{}   ", prefix) } else { format!("{}│  ", prefix) };
                dump_precompute_trie2_recursive(child_arc, child_prefix, visited, original_internal_bimap, llm_token_map, trie2_god);
            }
        }
    }
}

pub fn dump_precompute_trie3_recursive(
    node_arc: &Trie2Index,
    prefix: String,
    visited: &mut HashSet<Trie2Index>,
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    trie3_god: &Trie3GodWrapper,
) {
    let children_to_visit = {
        let node = node_arc.read(trie3_god).expect("RwLock poisoned during dump");
        node.children().iter().flat_map(|(edge_key, dest_map)| {
            dest_map.iter().map(move |(child_wrapper, edge_val)| {
                (
                    edge_key.clone(),
                    edge_val.clone(),
                    child_wrapper.as_arc().clone(),
                )
            })
        }).collect::<Vec<_>>()
    };

    for (i, (edge_key, edge_val_bv, child_arc)) in children_to_visit.iter().enumerate() {
        let is_last = i == children_to_visit.len() - 1;
        let connector = if is_last { "└──" } else { "├──" };
        let (pop_len, llm_bv) = edge_key;
        let llm_tokens_display = format_bv_with_tokens(llm_bv, original_internal_bimap, llm_token_map, 5);
        let edge_key_display = format!("(pop: {}, tokens: {})", pop_len, llm_tokens_display);
        let state_ids_display = format_hybrid_bitset_neatly(edge_val_bv);

        let (child_ptr, child_info, is_visited, is_end_node) = {
            let child_node = child_arc.read(trie3_god).unwrap();
            let ptr = child_arc;
            (ptr, format!("Node {} (MaxDepth: {}){}", ptr, child_node.max_depth, if child_node.value.end() { " [END]" } else { "" }), visited.contains(&ptr), child_node.value.end())
        };

        if is_visited && !is_end_node {
            println!("{}{} Edge {}: states {} -> Ref to {}", prefix, connector, edge_key_display, state_ids_display, child_info);
        } else {
            println!("{}{} Edge {}: states {} -> {}", prefix, connector, edge_key_display, state_ids_display, child_info);
            if !is_visited {
                visited.insert(*child_ptr);
                let child_prefix = if is_last { format!("{}   ", prefix) } else { format!("{}│  ", prefix) };
                dump_precompute_trie3_recursive(child_arc, child_prefix, visited, original_internal_bimap, llm_token_map, trie3_god);
            }
        }
    }
}

pub fn calculate_final_stats2(
    precomputed_roots: &BTreeMap<TokenizerStateID, PrecomputeNode2Index>,
    stats: &mut PrecomputeStats,
    trie2_god: &Trie2GodWrapper,
) {
    crate::debug!(2, "Calculating final precompute2 statistics...");

    let mut all_reachable_nodes: BTreeMap<PrecomputeNode2Index, PrecomputeNode2Index> = BTreeMap::new();
    let mut queue: VecDeque<PrecomputeNode2Index> = precomputed_roots.values().cloned().collect();
    let mut visited_data_ptrs: HashSet<PrecomputeNode2Index> = HashSet::new();

    while let Some(node_arc) = queue.pop_front() {
        let (children_to_queue, node_ptr) = {
            let node_guard = node_arc.read(trie2_god).unwrap();
            let ptr = node_arc;
            let children = node_guard.children()
                .values()
                .flat_map(|dest_map| {
                    dest_map.keys().map(|wrapper| {
                        wrapper.as_arc().clone()
                    })
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

    *stats = PrecomputeStats::default();
    stats.final_unique_nodes_count = all_reachable_nodes.len();

    let root_node_pointers: HashSet<PrecomputeNode2Index> = precomputed_roots
        .values()
        .map(|arc| {
            let guard = arc.read(trie2_god).unwrap();
            arc.clone()
        })
        .collect();
    stats.final_root_nodes_count = root_node_pointers.len();

    for (node_ptr, node_arc) in &all_reachable_nodes {
        let node_guard = node_arc.read(trie2_god).expect("RwLock poisoned during final stats calculation");

        if !root_node_pointers.contains(node_ptr) {
            if node_guard.children().is_empty() {
                stats.final_leaf_nodes_count += 1;
            } else {
                stats.final_non_root_internal_nodes_count += 1;
            }
        }

        for (_edge_key, dest_map) in node_guard.children() {
            let num_edges_for_this_key = dest_map.len();
            stats.final_edges_count += num_edges_for_this_key;
            
            // For PrecomputeNode2, all keys are "some" keys in a sense.
            stats.final_edges_with_some_key += num_edges_for_this_key;
            if num_edges_for_this_key > 0 {
                stats.final_total_occupancy_sum_for_some_keys += num_edges_for_this_key;
                stats.final_num_occupied_some_edge_keys += 1;
            }

            for llm_token_bv_on_edge in dest_map.values() {
                stats.final_total_ranges_in_bvs += llm_token_bv_on_edge.inner().ranges_len();
            }
        }

        if node_guard.value.end {
            stats.final_nodes_with_clean_end += 1;
        }
    }
    crate::debug!(2, "Finished calculating final precompute2 statistics.");
}

pub fn calculate_final_stats3(
    precomputed_roots: &BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    stats: &mut PrecomputeStats,
    trie3_god: &Trie3GodWrapper,
) {
    crate::debug!(2, "Calculating final precompute3 statistics...");

    let mut all_reachable_nodes: BTreeMap<PrecomputeNode3Index, PrecomputeNode3Index> = BTreeMap::new();
    let mut queue: VecDeque<PrecomputeNode3Index> = precomputed_roots.values().cloned().collect();
    let mut visited_data_ptrs: HashSet<PrecomputeNode3Index> = HashSet::new();

    while let Some(node_arc) = queue.pop_front() {
        let (children_to_queue, node_ptr) = {
            let node_guard = node_arc.read(trie3_god).unwrap();
            let ptr = node_arc;
            let children = node_guard.children()
                .values()
                .flat_map(|dest_map| {
                    dest_map.keys().map(|wrapper| {
                        wrapper.as_arc().clone()
                    })
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

    *stats = PrecomputeStats::default();
    stats.final_unique_nodes_count = all_reachable_nodes.len();

    let root_node_pointers: HashSet<PrecomputeNode3Index> = precomputed_roots
        .values()
        .map(|arc| {
            let guard = arc.read(trie3_god).unwrap();
            arc.clone()
        })
        .collect();
    stats.final_root_nodes_count = root_node_pointers.len();

    for (node_ptr, node_arc) in &all_reachable_nodes {
        let node_guard = node_arc.read(trie3_god).expect("RwLock poisoned during final stats calculation");

        if !root_node_pointers.contains(node_ptr) {
            if node_guard.children().is_empty() {
                stats.final_leaf_nodes_count += 1;
            } else {
                stats.final_non_root_internal_nodes_count += 1;
            }
        }

        for (_edge_key, dest_map) in node_guard.children() {
            let num_edges_for_this_key = dest_map.len();
            stats.final_edges_count += num_edges_for_this_key;
            stats.final_edges_with_some_key += num_edges_for_this_key;
            if num_edges_for_this_key > 0 {
                stats.final_total_occupancy_sum_for_some_keys += num_edges_for_this_key;
                stats.final_num_occupied_some_edge_keys += 1;
            }
            for state_id_bv_on_edge in dest_map.values() {
                stats.final_total_ranges_in_bvs += state_id_bv_on_edge.inner().ranges_len();
            }
        }
        if node_guard.value.end() { stats.final_nodes_with_clean_end += 1; }
    }
    crate::debug!(2, "Finished calculating final precompute3 statistics.");
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
        obj.insert("final_total_occupancy_sum_for_none_keys".to_string(), self.final_total_occupancy_sum_for_none_keys.to_json());
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

pub fn calculate_final_stats(
    precomputed_roots: &BTreeMap<TokenizerStateID, PrecomputeNodeIndex>,
    stats: &mut PrecomputeStats,
    trie1_god: &Trie1GodWrapper,
) {
    crate::debug!(2, "Calculating final precompute statistics (within constraint_extra)...");

    // Custom implementation of all_nodes using PrecomputeNodeIndex for visited set
    let mut all_reachable_nodes: BTreeMap<PrecomputeNodeIndex, PrecomputeNodeIndex> = BTreeMap::new();
    let mut queue: VecDeque<PrecomputeNodeIndex> = precomputed_roots.values().cloned().collect();
    let mut visited_data_ptrs: HashSet<PrecomputeNodeIndex> = HashSet::new();

    while let Some(node_arc) = queue.pop_front() {
        let (children_to_queue, node_ptr) = {
            let node_guard = node_arc.read(trie1_god).unwrap();
            let ptr = node_arc;
            let children = node_guard.children()
                .values()
                .flat_map(|dest_map| {
                    dest_map.keys().map(|wrapper| wrapper.as_arc().clone())
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

    let root_node_pointers: HashSet<PrecomputeNodeIndex> = precomputed_roots
        .values()
        .map(|arc| {
            let guard = arc.read(trie1_god).unwrap();
            *arc
        })
        .collect();
    stats.final_root_nodes_count = root_node_pointers.len();

    // Initialize stats fields
    stats.final_total_occupancy_sum_for_some_keys = 0;
    stats.final_num_occupied_some_edge_keys = 0;
    stats.final_total_occupancy_sum_for_none_keys = 0;
    stats.final_num_occupied_none_edge_keys = 0;
    stats.final_edges_count = 0;
    stats.final_edges_with_none_key = 0;
    stats.final_edges_with_some_key = 0;
    stats.final_nodes_with_clean_end = 0;
    stats.final_grammar_token_edge_key_counts.clear();
    stats.final_grammar_token_edge_fanouts_dist.clear();
    stats.final_grammar_token_edge_token_set_sizes_dist.clear();
    // stats.final_root_nodes_count = 0; // Already set above
    stats.final_non_root_internal_nodes_count = 0;
    stats.final_leaf_nodes_count = 0;
    stats.edges_pruned_by_terminal_sequence = 0;
    stats.final_total_ranges_in_bvs = 0;

    for (node_ptr, node_arc) in &all_reachable_nodes {
        let node_guard = node_arc.read(trie1_god).expect("RwLock poisoned during final stats calculation");

        // New logic for non-root internal and leaf nodes
        if !root_node_pointers.contains(node_ptr) {
            if node_guard.children().is_empty() {
                stats.final_leaf_nodes_count += 1;
            } else {
                stats.final_non_root_internal_nodes_count += 1;
            }
        }

        // Existing logic for edges
        for (edge_key_opt, dest_map) in node_guard.children() {
            let num_edges_for_this_key_to_distinct_children = dest_map.len();
            stats.final_edges_count += num_edges_for_this_key_to_distinct_children;

            if let Some(gtid) = edge_key_opt {
                stats.final_edges_with_some_key += num_edges_for_this_key_to_distinct_children;
                *stats.final_grammar_token_edge_key_counts.entry(*gtid).or_insert(0) += num_edges_for_this_key_to_distinct_children; // Corrected: sum edges not occurrences of key

                stats.final_grammar_token_edge_fanouts_dist
                    .entry(*gtid)
                    .or_default()
                    .push(num_edges_for_this_key_to_distinct_children);
                for llm_token_bv_on_edge in dest_map.values() {
                    stats.final_grammar_token_edge_token_set_sizes_dist
                        .entry(*gtid)
                        .or_default()
                        .push(llm_token_bv_on_edge.len());
                    stats.final_total_ranges_in_bvs += llm_token_bv_on_edge.inner().ranges_len();
                }
                if num_edges_for_this_key_to_distinct_children > 0 {
                    stats.final_total_occupancy_sum_for_some_keys += num_edges_for_this_key_to_distinct_children;
                    stats.final_num_occupied_some_edge_keys += 1;
                }
            } else {
                stats.final_edges_with_none_key += num_edges_for_this_key_to_distinct_children;
                if num_edges_for_this_key_to_distinct_children > 0 {
                    stats.final_total_occupancy_sum_for_none_keys += num_edges_for_this_key_to_distinct_children;
                    stats.final_num_occupied_none_edge_keys += 1;
                }
            }
        }

        // Existing logic for clean_end
        // if let Some(clean_end_bv) = &node_guard.value.clean_end {
        //     stats.final_nodes_with_clean_end += 1;
        //     stats.final_total_ranges_in_bvs += clean_end_bv.inner().ranges_len();
        // }
        if node_guard.value.end {
            stats.final_nodes_with_clean_end += 1;
        }
    }
    crate::debug!(2, "Finished calculating final precompute statistics (within constraint_extra).");
}


pub fn print_precompute_stats(
    stats: &PrecomputeStats,
    token_name_map: &BiBTreeMap<Terminal, usize>, // Used to get token names from GrammarTokenID
    trie_god: &Trie1GodWrapper,
) {
    let avg_some = if stats.final_num_occupied_some_edge_keys > 0 {
        stats.final_total_occupancy_sum_for_some_keys as f64 / stats.final_num_occupied_some_edge_keys as f64
    } else { 0.0 };
    let avg_none = if stats.final_num_occupied_none_edge_keys > 0 {
        stats.final_total_occupancy_sum_for_none_keys as f64 / stats.final_num_occupied_none_edge_keys as f64
    } else { 0.0 };

    println!("--- Precomputation Statistics ---");
    println!("  Initial Root Nodes Created: {}", stats.initial_root_nodes_created);

    println!("\nNode Counts Breakdown:");
    println!("  There are:");
    println!("  - {} unique nodes, of which", stats.final_unique_nodes_count);
    println!("    - {} are roots", stats.final_root_nodes_count);
    let non_root_count = stats.final_unique_nodes_count.saturating_sub(stats.final_root_nodes_count); // Use saturating_sub
    println!("    - {} are non-roots, of which", non_root_count);
    println!("        - {} are internal (non-root, non-leaf)", stats.final_non_root_internal_nodes_count);
    println!("        - {} are leaves (non-root)", stats.final_leaf_nodes_count);

    println!("\nFinal Graph Structure (after sharing and deduplication):");
    println!("  Unique Nodes: {}", stats.final_unique_nodes_count);
    println!("  Total Edges: {}", stats.final_edges_count);
    println!("    Edges with None Key: {}", stats.final_edges_with_none_key);
    println!("    Edges with Some Key: {}", stats.final_edges_with_some_key);
    println!("  Nodes with Clean End: {}", stats.final_nodes_with_clean_end);
    println!("  Average edge occupancy for Some-key edges:    {:.2}", avg_some);
    println!("  Average edge occupancy for None-key edges:    {:.2}", avg_none);
    println!("  Total ranges in all HybridBitsets: {}", stats.final_total_ranges_in_bvs);

    let mut grammar_token_stats_new: Vec<(
        GrammarTokenID,
        usize, // key_usages (KeyUse)
        (usize, Option<f64>, Option<f64>), // fanout_stats (SumChild, AvgChild, MedChild)
        (usize, Option<f64>, Option<f64>)  // token_set_size_stats (SumToks, AvgToks, MedToks)
    )> = Vec::new();

    for (gtid, key_usages_count) in &stats.final_grammar_token_edge_key_counts {
        let fanouts_for_gtid = stats.final_grammar_token_edge_fanouts_dist
                                    .get(gtid)
                                    .cloned()
                                    .unwrap_or_else(Vec::new);
        let child_stats = calculate_stats_from_vec_usize(&fanouts_for_gtid); // Uses the helper

        let token_set_sizes_for_gtid = stats.final_grammar_token_edge_token_set_sizes_dist
                                            .get(gtid)
                                            .cloned()
                                            .unwrap_or_else(Vec::new);
        let toks_stats = calculate_stats_from_vec_usize(&token_set_sizes_for_gtid); // Uses the helper

        grammar_token_stats_new.push((*gtid, *key_usages_count, child_stats, toks_stats));
    }

    grammar_token_stats_new.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by KeyUse (key_usages_count)

    println!("\nGrammar Token Edge Key Frequencies (Most Common First):");
    println!(
        "  {:<25} {:<5} {:<8} {:<8} {:<10} {:<10} {:<10} {:<12} {:<10} {:<10}",
        "Token Name", "ID", "KeyUse", "SumChild", "AvgChild", "MedChild",
        "AvgKeyToks", "SumToks", "AvgToks", "MedToks"
    );
    println!(
        "  {:-<25} {:-<5} {:-<8} {:-<8} {:-<10} {:-<10} {:-<10} {:-<12} {:-<10} {:-<10}",
        "", "", "", "", "", "", "", "", "", ""
    );

    for (gtid, key_usages, child_stats, toks_stats) in grammar_token_stats_new {
        let name = token_name_map
            .get_by_right(&gtid.0) // gtid is GrammarTokenID
            .cloned()
            .map_or(
                format!("ID:{}", gtid.0),
                |t| t.to_string()
            );

        let (sum_child, avg_child, med_child) = child_stats;
        let (sum_toks, avg_toks, med_toks) = toks_stats;
        let avg_key_toks = if key_usages > 0 {
            Some(sum_toks as f64 / key_usages as f64)
        } else {
            None
        };


        let format_opt_f64 = |val: Option<f64>| val.map_or_else(|| "N/A".to_string(), |v| format!("{:.2}", v));

        println!(
            "  {:<25} {:>5} {:>8} {:>8} {:>10} {:>10} {:>10} {:>12} {:>10} {:>10}",
            name,
            gtid.0,
            key_usages,
            sum_child,
            format_opt_f64(avg_child),
            format_opt_f64(med_child),
            format_opt_f64(avg_key_toks),
            sum_toks,
            format_opt_f64(avg_toks),
            format_opt_f64(med_toks)
        );
    }
    println!("---------------------------------");
}

pub fn print_precompute_stats2(
    stats: &PrecomputeStats,
    trie2_god: &Trie2GodWrapper,
) {
    let avg_fanout = if stats.final_num_occupied_some_edge_keys > 0 {
        stats.final_total_occupancy_sum_for_some_keys as f64 / stats.final_num_occupied_some_edge_keys as f64
    } else { 0.0 };

    println!("--- Precomputation 2 Statistics ---");
    
    println!("\nNode Counts Breakdown:");
    println!("  There are:");
    println!("  - {} unique nodes, of which", stats.final_unique_nodes_count);
    println!("    - {} are roots", stats.final_root_nodes_count);
    let non_root_count = stats.final_unique_nodes_count.saturating_sub(stats.final_root_nodes_count);
    println!("    - {} are non-roots, of which", non_root_count);
    println!("        - {} are internal (non-root, non-leaf)", stats.final_non_root_internal_nodes_count);
    println!("        - {} are leaves (non-root)", stats.final_leaf_nodes_count);

    println!("\nFinal Graph Structure (after sharing and deduplication):");
    println!("  Unique Nodes: {}", stats.final_unique_nodes_count);
    println!("  Total Edges: {}", stats.final_edges_count);
    println!("  Nodes with End Marker: {}", stats.final_nodes_with_clean_end);
    println!("  Average edge fanout:    {:.2}", avg_fanout);
    println!("  Total ranges in all HybridBitsets: {}", stats.final_total_ranges_in_bvs);
    println!("---------------------------------");
}

pub fn print_precompute_stats3(
    stats: &PrecomputeStats,
    trie3_god: &Trie3GodWrapper,
) {
    let avg_fanout = if stats.final_num_occupied_some_edge_keys > 0 {
        stats.final_total_occupancy_sum_for_some_keys as f64 / stats.final_num_occupied_some_edge_keys as f64
    } else { 0.0 };

    println!("--- Precomputation 3 Statistics ---");
    
    println!("\nNode Counts Breakdown:");
    println!("  There are:");
    println!("  - {} unique nodes, of which", stats.final_unique_nodes_count);
    println!("    - {} are roots", stats.final_root_nodes_count);
    let non_root_count = stats.final_unique_nodes_count.saturating_sub(stats.final_root_nodes_count);
    println!("    - {} are non-roots, of which", non_root_count);
    println!("        - {} are internal (non-root, non-leaf)", stats.final_non_root_internal_nodes_count);
    println!("        - {} are leaves (non-root)", stats.final_leaf_nodes_count);

    println!("\nFinal Graph Structure (after sharing and deduplication):");
    println!("  Unique Nodes: {}", stats.final_unique_nodes_count);
    println!("  Total Edges: {}", stats.final_edges_count);
    println!("  Nodes with End Marker: {}", stats.final_nodes_with_clean_end);
    println!("  Average edge fanout:    {:.2}", avg_fanout);
    println!("  Total ranges in all HybridBitsets: {}", stats.final_total_ranges_in_bvs);
    println!("---------------------------------");
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use crate::finite_automata::{eat_u8, Regex};
    use crate::glr::grammar::{nt, prod, t, regex_name, Terminal};
    use crate::glr::parser::GLRParser;
    use crate::glr::table::generate_glr_parser_with_terminal_map;
    use crate::tokenizer::{LLMTokenID, LLMTokenMap};
    use crate::types::TerminalID;
    use bimap::BiBTreeMap;
    use super::*;
    use bitvec::prelude::*;
    use crate::seq;
    use crate::datastructures::hybrid_bitset::HybridBitset;

    #[test]
    fn test_format_bv_with_tokens_no_maps() {
        let bv = HybridBitset::from_iter(vec![1, 2]);
        assert_eq!(format_bv_with_tokens(&bv, None, None, 5), "[1..=2]");
    }

    #[test]
    fn test_format_bv_with_tokens_with_maps() {
        let bv = HybridBitset::from_iter(vec![0, 1]); // internal IDs
        let mut bimap = BiBTreeMap::new();
        bimap.insert(10, 0); // original 10 -> internal 0
        bimap.insert(20, 1); // original 20 -> internal 1
        let mut llm_map = BiBTreeMap::new();
        llm_map.insert(b"ten".to_vec(), LLMTokenID(10));
        llm_map.insert(b"twenty".to_vec(), LLMTokenID(20));

        let expected = "[0..=1] (e.g., [\"ten\", \"twenty\"])";
        assert_eq!(format_bv_with_tokens(&bv, Some(&bimap), Some(&llm_map), 5), expected);
    }

    #[test]
    fn test_format_bv_with_tokens_limit_and_ellipsis() {
        let bv = HybridBitset::from_iter(0..10); // internal IDs 0-9
        let mut bimap = BiBTreeMap::new();
        let mut llm_map = BiBTreeMap::new();
        for i in 0..10 {
            bimap.insert(100 + i, i);
            llm_map.insert(format!("{}", 100 + i).into_bytes(), LLMTokenID(100 + i));
        }

        let expected = "[0..=9] (e.g., [\"100\", \"101\", \"102\"], ...)";
        assert_eq!(format_bv_with_tokens(&bv, Some(&bimap), Some(&llm_map), 3), expected);
    }

    // Helper function to create a minimal constraint for testing dump
    fn create_minimal_constraint() -> GrammarConstraint {
        // Tokenizer: Matches "a" (token 0) or "$" (token 1)
        let expr = crate::groups![
            eat_u8(b'a'), // Grammar Token 0
            seq![eat_u8(b'a'), eat_u8(b'a')], // Grammar Token 1
            eat_u8(b'$')  // Grammar Token 2 (EOF)
        ];
        let tokenizer = expr.build();

        // LLM Token Map: "aaaa" -> 0, "$" -> 1
        let mut llm_token_map = LLMTokenMap::new();
        llm_token_map.insert(b"aaaa".to_vec(), LLMTokenID(0));
        llm_token_map.insert(b"$".to_vec(), LLMTokenID(1));
        let max_llm_token_id = 1;

        // Grammar: S -> A $
        let productions = vec![
            prod("S", vec![nt("A"), nt("AA"), t("EOF")]), // S' -> S EOF is implicit
        ];

        // Map grammar terminals to the tokenizer's token IDs
        let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
        grammar_token_map.insert(regex_name("A"), TerminalID(0)); // "a" from tokenizer
        grammar_token_map.insert(regex_name("AA"), TerminalID(1)); // "aa" from tokenizer
        grammar_token_map.insert(regex_name("EOF"), TerminalID(2)); // "$" from tokenizer

        // Generate parser
        let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map, None);

        let mut terminal_name_map = BiBTreeMap::new();
        terminal_name_map.insert(regex_name("A"), 0);
        terminal_name_map.insert(regex_name("AA"), 1);
        terminal_name_map.insert(regex_name("EOF"), 2);

        // Create constraint (this runs precomputation)
        GrammarConstraint::new(tokenizer, parser, llm_token_map, terminal_name_map, max_llm_token_id)
    }

    #[test]
    fn test_dump_precomputed_runs() {
        let constraint = create_minimal_constraint();
        println!("--- Starting dump_precomputed test output ---");
        constraint.dump_precomputed(); // Just ensure it runs without panic
        println!("--- Finished dump_precomputed test output ---");
    }

    #[test]
    fn test_dump_precomputed2_runs() {
        let constraint = create_minimal_constraint();
        println!("--- Starting dump_precomputed2 test output ---");
        constraint.dump_precomputed2(); // Just ensure it runs without panic
        println!("--- Finished dump_precomputed2 test output ---");
    }
}
