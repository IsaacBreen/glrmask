use crate::constraint::LLMTokenBV;
use crate::constraint::{
    GrammarConstraint, PrecomputeNode0Index, PrecomputeNode1, PrecomputeNode1Index, Precomputed,
    Trie0GodWrapper, Trie1GodWrapper,
};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::{Trie, Trie2Index};
use crate::glr::grammar::Terminal;
use crate::glr::table::StateID;
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID as GrammarTokenID;
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use std::collections::BTreeMap as StdMap;
use std::collections::HashMap;
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::ops::BitOrAssign;
use std::sync::{Arc, RwLock};

/// Creates a neat string representation of a HybridBitset, showing values as ranges.
fn format_hybrid_bitset_neatly(bv: &HybridBitset) -> String {
    if bv.is_empty() {
        return "[]".to_string();
    }

    const MAX_RANGES_TO_SHOW: usize = 5;
    let total_ranges = bv.inner().ranges_len();

    let ranges_to_show_str = bv
        .inner()
        .ranges()
        .take(MAX_RANGES_TO_SHOW)
        .map(|range| {
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
        })
        .collect::<Vec<_>>()
        .join(", ");

    let ellipsis = if total_ranges > MAX_RANGES_TO_SHOW {
        ", ..."
    } else {
        ""
    };
    format!("[{}{}]", ranges_to_show_str, ellipsis)
}

/// Helper function to format a HybridBitset for display, showing its debug representation
/// and a sample of the corresponding LLM tokens.
fn format_bv_with_tokens(
    bv: &LLMTokenBV,
    internal_to_original_map: Option<&BTreeMap<usize, LLMTokenBV>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    limit: usize,
    internal_max_llm_token_id: usize,
) -> String {
    let (i2o_map, token_map) = match (internal_to_original_map, llm_token_map) {
        (Some(i), Some(t)) => (i, t),
        _ => return format_hybrid_bitset_neatly(bv),
    };

    // Convert internal IDs to original LLM token IDs
    let mut original_tokens_bv = LLMTokenBV::zeros();
    for internal_id in bv.iter_up_to(internal_max_llm_token_id) {
        if let Some(original_ids_bv) = i2o_map.get(&internal_id) {
            original_tokens_bv.bitor_assign(original_ids_bv);
        }
    }
    let bv_neat_str = format_hybrid_bitset_neatly(&original_tokens_bv);

    if original_tokens_bv.is_empty() {
        return bv_neat_str;
    }

    let mut token_samples = Vec::new();
    let total_original_tokens = original_tokens_bv.len();

    for original_id in original_tokens_bv.iter_up_to(limit) {
        if let Some(token_bytes) = token_map.get_by_right(&LLMTokenID(original_id)) {
            token_samples.push(format!("{:?}", String::from_utf8_lossy(token_bytes)));
        }
    }

    if token_samples.is_empty() {
        return bv_neat_str;
    }

    let samples_str = token_samples.join(", ");
    let ellipsis = if total_original_tokens > token_samples.len() {
        ", ..."
    } else {
        ""
    };

    format!("{} (e.g., [{}]{})", bv_neat_str, samples_str, ellipsis)
}

/// Helper function to recursively dump the structure of a PrecomputeNode0 Trie.
pub fn dump_precompute_trie0_recursive(
    node_arc: &PrecomputeNode0Index,
    prefix: String,
    visited: &mut HashSet<PrecomputeNode0Index>,
    internal_to_original_map: Option<&BTreeMap<usize, LLMTokenBV>>,
    token_name_map: Option<&BiBTreeMap<Terminal, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    trie0_god: &Trie0GodWrapper,
) {
    let children_to_visit;

    {
        let node = node_arc
            .read(trie0_god)
            .expect("RwLock poisoned during dump");
        // Collect children information while holding the lock
        children_to_visit = node
            .children()
            .iter()
            .flat_map(|(edge_key, dest_map)| {
                dest_map.iter().map(move |(child_wrapper, edge_val)| {
                    (
                        edge_key.clone(),
                        edge_val.clone(),
                        child_wrapper.as_arc().clone(),
                    )
                })
            })
            .collect::<Vec<_>>();
    }

    let num_children = children_to_visit.len();
    for (i, (edge_key, edge_val_bv, child_arc)) in children_to_visit.iter().enumerate() {
        let is_last = i == num_children - 1;
        let connector = if is_last { "└──" } else { "├──" };

        let edge_key_display = match edge_key {
            Some((gtid, disallow_opt)) => {
                if let Some(name_map) = token_name_map {
                    let gtid_str = name_map
                        .get_by_right(&gtid.0)
                        .map(|name| format!("'{}'", name))
                        .unwrap_or_else(|| format!("ID:{}", gtid.0));
                    let disallow_str = if let Some(sid) = disallow_opt {
                        format!(", disallow=(S{})", sid.0)
                    } else {
                        "".to_string()
                    };
                    format!("{}{}", gtid_str, disallow_str)
                } else {
                    format!("ID:{}", gtid.0)
                }
            }
            None => "ε".to_string(),
        };

        let internal_max_llm_token_id =
            *internal_to_original_map.unwrap().keys().max().unwrap_or(&0);
        let tokens_display = format_bv_with_tokens(
            &edge_val_bv,
            internal_to_original_map,
            llm_token_map,
            5,
            internal_max_llm_token_id,
        );

        let child_ptr;
        let child_info;
        let is_visited;
        let is_end_node;
        {
            let child_node = child_arc.read(trie0_god).unwrap();
            child_ptr = child_arc;
            is_visited = visited.contains(&child_ptr);
            is_end_node = child_node.value.final_tokenizer_state.is_some();
            let live_tokens_str = format_bv_with_tokens(
                &child_node.value.live_tokens,
                internal_to_original_map,
                llm_token_map,
                5,
                internal_max_llm_token_id,
            );
            let end_str = if is_end_node {
                if let Some(sid) = child_node.value.final_tokenizer_state {
                    format!(" [END -> S{}]", sid.0)
                } else {
                    " [END]".to_string()
                }
            } else {
                "".to_string()
            };
            child_info = format!(
                "Node {} (MaxDepth: {}){} [Live: {}]",
                child_ptr, child_node.max_depth, end_str, live_tokens_str
            );
        }

        if is_visited && !is_end_node {
            println!(
                "{}{} Edge {}: {} -> Ref to {}",
                prefix, connector, edge_key_display, tokens_display, child_info
            );
        } else {
            println!(
                "{}{} Edge {}: {} -> {}",
                prefix, connector, edge_key_display, tokens_display, child_info
            );

            if !is_visited {
                visited.insert(*child_ptr);
                let child_prefix = if is_last {
                    format!("{}   ", prefix)
                } else {
                    format!("{}│  ", prefix)
                };
                dump_precompute_trie0_recursive(
                    child_arc,
                    child_prefix,
                    visited,
                    internal_to_original_map,
                    token_name_map,
                    llm_token_map,
                    trie0_god,
                );
            }
        }
    }
}

/// Helper function to recursively dump the structure of a PrecomputeNode Trie.
pub fn dump_precompute_trie_recursive(
    node_arc: &PrecomputeNode1Index,
    prefix: String,
    visited: &mut HashSet<PrecomputeNode1Index>,
    internal_to_original_map: Option<&BTreeMap<usize, LLMTokenBV>>,
    token_name_map: Option<&BiBTreeMap<Terminal, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    trie1_god: &Trie1GodWrapper,
) {
    let children_to_visit;

    {
        let node = node_arc
            .read(trie1_god)
            .expect("RwLock poisoned during dump");
        // Collect children information while holding the lock
        children_to_visit = node
            .children()
            .iter()
            .flat_map(|(edge_key, dest_map)| {
                dest_map.iter().map(move |(child_wrapper, edge_val)| {
                    (
                        edge_key.clone(),
                        edge_val.clone(),
                        child_wrapper.as_arc().clone(),
                    )
                })
            })
            .collect::<Vec<_>>();
    }

    let num_children = children_to_visit.len();
    for (i, (edge_key, edge_val_bv, child_arc)) in children_to_visit.iter().enumerate() {
        let is_last = i == num_children - 1;
        let connector = if is_last { "└──" } else { "├──" };

        let edge_key_display = match edge_key {
            Some(gtid) => {
                if let Some(name_map) = token_name_map {
                    name_map
                        .get_by_right(&gtid.0)
                        .map(|name| format!("'{}'", name))
                        .unwrap_or_else(|| format!("ID:{}", gtid.0))
                } else {
                    format!("ID:{}", gtid.0)
                }
            }
            None => "ε".to_string(),
        };

        let internal_max_llm_token_id =
            *internal_to_original_map.unwrap().keys().max().unwrap_or(&0);
        let tokens_display = format_bv_with_tokens(
            &edge_val_bv,
            internal_to_original_map,
            llm_token_map,
            5,
            internal_max_llm_token_id,
        );

        let child_ptr;
        let child_info;
        let is_visited;
        let is_end_node;
        {
            let child_node = child_arc.read(trie1_god).unwrap();
            child_ptr = child_arc;
            is_visited = visited.contains(&child_ptr);
            is_end_node = child_node.value.end;
            let live_tokens_str = format_bv_with_tokens(
                &child_node.value.live_tokens,
                internal_to_original_map,
                llm_token_map,
                5,
                internal_max_llm_token_id,
            );
            child_info = format!(
                "Node {} (MaxDepth: {}){} [Live: {}]",
                child_ptr,
                child_node.max_depth,
                if is_end_node { " [END]" } else { "" },
                live_tokens_str
            );
        }

        // Don't shortcut the display for end nodes, even if they are visited.
        if is_visited && !is_end_node {
            println!(
                "{}{} Edge {}: {} -> Ref to {}",
                prefix, connector, edge_key_display, tokens_display, child_info
            );
        } else {
            // Print full info for unvisited nodes or for any end node.
            println!(
                "{}{} Edge {}: {} -> {}",
                prefix, connector, edge_key_display, tokens_display, child_info
            );

            // Only recurse if the node has not been visited before.
            if !is_visited {
                visited.insert(*child_ptr);
                let child_prefix = if is_last {
                    format!("{}   ", prefix)
                } else {
                    format!("{}│  ", prefix)
                };
                dump_precompute_trie_recursive(
                    child_arc,
                    child_prefix,
                    visited,
                    internal_to_original_map,
                    token_name_map,
                    llm_token_map,
                    trie1_god,
                );
            }
        }
    }
}

impl GrammarConstraint {
    /// Dumps the structure of the precomputed Trie map for visualization.
    pub fn dump_precomputed1(&self) {
        GrammarConstraint::_dump_precomputed(
            &self.precomputed1,
            &self.vocab.internal_to_original,
            &self.token_name_map,
            &self.llm_vocab.llm_token_map,
            &self.trie1_god,
        );
    }

    pub fn _dump_precomputed(
        precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
        internal_to_original_map: &BTreeMap<usize, LLMTokenBV>,
        token_name_map: &BiBTreeMap<Terminal, usize>,
        llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        trie1_god: &Trie1GodWrapper,
    ) {
        println!("Dumping Precomputed Trie 1 Structure (showing original LLM Token IDs):");
        println!("===================================");

        let mut visited: HashSet<PrecomputeNode1Index> = HashSet::new();
        for (tokenizer_state_id, root_node_trie) in precomputed1 {
            println!("\n--- Tokenizer State ID: {} ---", tokenizer_state_id.0);

            let root_ptr;
            let root_info;
            {
                let root_node = root_node_trie.read(trie1_god).unwrap();
                root_ptr = root_node_trie;
                let internal_max_llm_token_id =
                    *internal_to_original_map.keys().max().unwrap_or(&0);
                let live_tokens_str = format_bv_with_tokens(
                    &root_node.value.live_tokens,
                    Some(internal_to_original_map),
                    Some(llm_token_map),
                    5,
                    internal_max_llm_token_id,
                );
                root_info = format!(
                    "Root Node {} (MaxDepth: {}){} [Live: {}]",
                    root_ptr,
                    root_node.max_depth,
                    if root_node.value.end { " [END]" } else { "" },
                    live_tokens_str
                );
            }
            println!("{}", root_info);

            if visited.contains(&root_ptr) {
                println!("  (Root already visited)");
            } else {
                visited.insert(*root_ptr);
                dump_precompute_trie_recursive(
                    root_node_trie,
                    "".to_string(),
                    &mut visited,
                    Some(internal_to_original_map),
                    Some(token_name_map),
                    Some(llm_token_map),
                    trie1_god,
                );
            }
        }
        println!("\n===================================");
        println!("Dump Complete.");
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
    pub final_non_root_internal_nodes_count: usize,
    pub final_leaf_nodes_count: usize,
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
        obj.insert(
            "initial_root_nodes_created".to_string(),
            self.initial_root_nodes_created.to_json(),
        );
        obj.insert(
            "final_unique_nodes_count".to_string(),
            self.final_unique_nodes_count.to_json(),
        );
        obj.insert(
            "final_root_nodes_count".to_string(),
            self.final_root_nodes_count.to_json(),
        );
        obj.insert(
            "final_non_root_internal_nodes_count".to_string(),
            self.final_non_root_internal_nodes_count.to_json(),
        );
        obj.insert(
            "final_leaf_nodes_count".to_string(),
            self.final_leaf_nodes_count.to_json(),
        );
        obj.insert(
            "final_edges_count".to_string(),
            self.final_edges_count.to_json(),
        );
        obj.insert(
            "final_edges_with_none_key".to_string(),
            self.final_edges_with_none_key.to_json(),
        );
        obj.insert(
            "final_edges_with_some_key".to_string(),
            self.final_edges_with_some_key.to_json(),
        );
        obj.insert(
            "final_nodes_with_clean_end".to_string(),
            self.final_nodes_with_clean_end.to_json(),
        );
        obj.insert(
            "final_total_occupancy_sum_for_some_keys".to_string(),
            self.final_total_occupancy_sum_for_some_keys.to_json(),
        );
        obj.insert(
            "final_num_occupied_some_edge_keys".to_string(),
            self.final_num_occupied_some_edge_keys.to_json(),
        );
        obj.insert(
            "final_total_occupancy_sum_for_none_keys".to_string(),
            self.final_total_occupancy_sum_for_none_keys.to_json(),
        );
        obj.insert(
            "final_num_occupied_none_edge_keys".to_string(),
            self.final_num_occupied_none_edge_keys.to_json(),
        );
        obj.insert(
            "final_grammar_token_edge_key_counts".to_string(),
            self.final_grammar_token_edge_key_counts.to_json(),
        );
        obj.insert(
            "final_grammar_token_edge_fanouts_dist".to_string(),
            self.final_grammar_token_edge_fanouts_dist.to_json(),
        );
        obj.insert(
            "final_grammar_token_edge_token_set_sizes_dist".to_string(),
            self.final_grammar_token_edge_token_set_sizes_dist.to_json(),
        );
        obj.insert(
            "final_edges_pruned_total".to_string(),
            self.final_edges_pruned_total.to_json(),
        );
        obj.insert(
            "final_edges_pruned_by_token".to_string(),
            self.final_edges_pruned_by_token.to_json(),
        );
        obj.insert(
            "edges_pruned_by_terminal_sequence".to_string(),
            self.edges_pruned_by_terminal_sequence.to_json(),
        );
        obj.insert(
            "final_total_ranges_in_bvs".to_string(),
            self.final_total_ranges_in_bvs.to_json(),
        );
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let initial_root_nodes_created = obj
                    .remove("initial_root_nodes_created")
                    .ok_or_else(|| {
                        "Missing field initial_root_nodes_created for PrecomputeStats".to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_unique_nodes_count = obj
                    .remove("final_unique_nodes_count")
                    .ok_or_else(|| {
                        "Missing field final_unique_nodes_count for PrecomputeStats".to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_root_nodes_count = obj
                    .remove("final_root_nodes_count")
                    .ok_or_else(|| {
                        "Missing field final_root_nodes_count for PrecomputeStats".to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_non_root_internal_nodes_count = obj
                    .remove("final_non_root_internal_nodes_count")
                    .ok_or_else(|| {
                        "Missing field final_non_root_internal_nodes_count for PrecomputeStats"
                            .to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_leaf_nodes_count = obj
                    .remove("final_leaf_nodes_count")
                    .ok_or_else(|| {
                        "Missing field final_leaf_nodes_count for PrecomputeStats".to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_edges_count = obj
                    .remove("final_edges_count")
                    .ok_or_else(|| {
                        "Missing field final_edges_count for PrecomputeStats".to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_edges_with_none_key = obj
                    .remove("final_edges_with_none_key")
                    .ok_or_else(|| {
                        "Missing field final_edges_with_none_key for PrecomputeStats".to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_edges_with_some_key = obj
                    .remove("final_edges_with_some_key")
                    .ok_or_else(|| {
                        "Missing field final_edges_with_some_key for PrecomputeStats".to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_nodes_with_clean_end = obj
                    .remove("final_nodes_with_clean_end")
                    .ok_or_else(|| {
                        "Missing field final_nodes_with_clean_end for PrecomputeStats".to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_total_occupancy_sum_for_some_keys = obj
                    .remove("final_total_occupancy_sum_for_some_keys")
                    .ok_or_else(|| {
                        "Missing field final_total_occupancy_sum_for_some_keys for PrecomputeStats"
                            .to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_num_occupied_some_edge_keys = obj
                    .remove("final_num_occupied_some_edge_keys")
                    .ok_or_else(|| {
                        "Missing field final_num_occupied_some_edge_keys for PrecomputeStats"
                            .to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_total_occupancy_sum_for_none_keys = obj
                    .remove("final_total_occupancy_sum_for_none_keys")
                    .ok_or_else(|| {
                        "Missing field final_total_occupancy_sum_for_none_keys for PrecomputeStats"
                            .to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_num_occupied_none_edge_keys = obj
                    .remove("final_num_occupied_none_edge_keys")
                    .ok_or_else(|| {
                        "Missing field final_num_occupied_none_edge_keys for PrecomputeStats"
                            .to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_grammar_token_edge_key_counts = obj
                    .remove("final_grammar_token_edge_key_counts")
                    .ok_or_else(|| {
                        "Missing field final_grammar_token_edge_key_counts for PrecomputeStats"
                            .to_string()
                    })
                    .and_then(|n| BTreeMap::<GrammarTokenID, usize>::from_json(n))?;
                let final_grammar_token_edge_fanouts_dist = obj
                    .remove("final_grammar_token_edge_fanouts_dist")
                    .ok_or_else(|| {
                        "Missing field final_grammar_token_edge_fanouts_dist for PrecomputeStats"
                            .to_string()
                    })
                    .and_then(|n| BTreeMap::<GrammarTokenID, Vec<usize>>::from_json(n))?;
                let final_grammar_token_edge_token_set_sizes_dist = obj
                    .remove("final_grammar_token_edge_token_set_sizes_dist")
                    .ok_or_else(|| {
                        "Missing field final_grammar_token_edge_token_set_sizes_dist for PrecomputeStats"
                            .to_string()
                    })
                    .and_then(|n| BTreeMap::<GrammarTokenID, Vec<usize>>::from_json(n))?;
                let final_edges_pruned_total = obj
                    .remove("final_edges_pruned_total")
                    .ok_or_else(|| {
                        "Missing field final_edges_pruned_total for PrecomputeStats".to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_edges_pruned_by_token = obj
                    .remove("final_edges_pruned_by_token")
                    .ok_or_else(|| {
                        "Missing field final_edges_pruned_by_token for PrecomputeStats".to_string()
                    })
                    .and_then(|n| BTreeMap::<GrammarTokenID, usize>::from_json(n))?;
                let edges_pruned_by_terminal_sequence = obj
                    .remove("edges_pruned_by_terminal_sequence")
                    .ok_or_else(|| {
                        "Missing field edges_pruned_by_terminal_sequence for PrecomputeStats"
                            .to_string()
                    })
                    .and_then(usize::from_json)?;
                let final_total_ranges_in_bvs = obj
                    .remove("final_total_ranges_in_bvs")
                    .ok_or_else(|| {
                        "Missing field final_total_ranges_in_bvs for PrecomputeStats".to_string()
                    })
                    .and_then(usize::from_json)?;
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
    let mut sum: usize = 0;
    for &n in numbers {
        sum = sum.saturating_add(n);
    }
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

pub fn calculate_final_stats0(
    precomputed_roots: &BTreeMap<TokenizerStateID, PrecomputeNode0Index>,
    stats: &mut PrecomputeStats,
    trie0_god: &Trie0GodWrapper,
) {
    crate::debug!(4, "Calculating final precompute0 statistics...");

    let mut all_reachable_nodes: BTreeMap<PrecomputeNode0Index, PrecomputeNode0Index> =
        BTreeMap::new();
    let mut queue: VecDeque<PrecomputeNode0Index> =
        precomputed_roots.values().cloned().collect();
    let mut visited_data_ptrs: HashSet<PrecomputeNode0Index> = HashSet::new();

    while let Some(node_arc) = queue.pop_front() {
        let (children_to_queue, node_ptr) = {
            let node_guard = node_arc.read(trie0_god).unwrap();
            let ptr = node_arc;
            let children = node_guard
                .children()
                .values()
                .flat_map(|dest_map| dest_map.keys().map(|wrapper| wrapper.as_arc().clone()))
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

    let root_node_pointers: HashSet<PrecomputeNode0Index> = precomputed_roots
        .values()
        .map(|arc| {
            let guard = arc.read(trie0_god).unwrap();
            *arc
        })
        .collect();
    stats.final_root_nodes_count = root_node_pointers.len();

    for (node_ptr, node_arc) in &all_reachable_nodes {
        let node_guard = node_arc
            .read(trie0_god)
            .expect("RwLock poisoned during final stats calculation");

        if !root_node_pointers.contains(node_ptr) {
            if node_guard.children().is_empty() {
                stats.final_leaf_nodes_count += 1;
            } else {
                stats.final_non_root_internal_nodes_count += 1;
            }
        }

        for (edge_key_opt, dest_map) in node_guard.children() {
            let num_edges_for_this_key_to_distinct_children = dest_map.len();
            stats.final_edges_count += num_edges_for_this_key_to_distinct_children;

            if let Some((gtid, _)) = edge_key_opt {
                stats.final_edges_with_some_key += num_edges_for_this_key_to_distinct_children;
                *stats
                    .final_grammar_token_edge_key_counts
                    .entry(*gtid)
                    .or_insert(0) += num_edges_for_this_key_to_distinct_children;

                stats
                    .final_grammar_token_edge_fanouts_dist
                    .entry(*gtid)
                    .or_default()
                    .push(num_edges_for_this_key_to_distinct_children);
                for llm_token_bv_on_edge in dest_map.values() {
                    stats
                        .final_grammar_token_edge_token_set_sizes_dist
                        .entry(*gtid)
                        .or_default()
                        .push(llm_token_bv_on_edge.len());
                    stats.final_total_ranges_in_bvs += llm_token_bv_on_edge.inner().ranges_len();
                }
                if num_edges_for_this_key_to_distinct_children > 0 {
                    stats.final_total_occupancy_sum_for_some_keys +=
                        num_edges_for_this_key_to_distinct_children;
                    stats.final_num_occupied_some_edge_keys += 1;
                }
            } else {
                stats.final_edges_with_none_key += num_edges_for_this_key_to_distinct_children;
                if num_edges_for_this_key_to_distinct_children > 0 {
                    stats.final_total_occupancy_sum_for_none_keys +=
                        num_edges_for_this_key_to_distinct_children;
                    stats.final_num_occupied_none_edge_keys += 1;
                }
            }
        }

        if node_guard.value.final_tokenizer_state.is_some() {
            stats.final_nodes_with_clean_end += 1;
        }
    }
    crate::debug!(4, "Finished calculating final precompute0 statistics.");
}

pub fn calculate_final_stats1(
    precomputed_roots: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    stats: &mut PrecomputeStats,
    trie1_god: &Trie1GodWrapper,
) {
    crate::debug!(4, "Calculating final precompute1 statistics...");

    // Custom implementation of all_nodes using PrecomputeNodeIndex for visited set
    let mut all_reachable_nodes: BTreeMap<PrecomputeNode1Index, PrecomputeNode1Index> =
        BTreeMap::new();
    let mut queue: VecDeque<PrecomputeNode1Index> =
        precomputed_roots.values().cloned().collect();
    let mut visited_data_ptrs: HashSet<PrecomputeNode1Index> = HashSet::new();

    while let Some(node_arc) = queue.pop_front() {
        let (children_to_queue, node_ptr) = {
            let node_guard = node_arc.read(trie1_god).unwrap();
            let ptr = node_arc;
            let children = node_guard
                .children()
                .values()
                .flat_map(|dest_map| dest_map.keys().map(|wrapper| wrapper.as_arc().clone()))
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

    let root_node_pointers: HashSet<PrecomputeNode1Index> = precomputed_roots
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
    stats
        .final_grammar_token_edge_token_set_sizes_dist
        .clear();
    stats.final_non_root_internal_nodes_count = 0;
    stats.final_leaf_nodes_count = 0;
    stats.edges_pruned_by_terminal_sequence = 0;
    stats.final_total_ranges_in_bvs = 0;

    for (node_ptr, node_arc) in &all_reachable_nodes {
        let node_guard = node_arc
            .read(trie1_god)
            .expect("RwLock poisoned during final stats calculation");

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
                *stats
                    .final_grammar_token_edge_key_counts
                    .entry(*gtid)
                    .or_insert(0) += num_edges_for_this_key_to_distinct_children;

                stats
                    .final_grammar_token_edge_fanouts_dist
                    .entry(*gtid)
                    .or_default()
                    .push(num_edges_for_this_key_to_distinct_children);
                for llm_token_bv_on_edge in dest_map.values() {
                    stats
                        .final_grammar_token_edge_token_set_sizes_dist
                        .entry(*gtid)
                        .or_default()
                        .push(llm_token_bv_on_edge.len());
                    stats.final_total_ranges_in_bvs += llm_token_bv_on_edge.inner().ranges_len();
                }
                if num_edges_for_this_key_to_distinct_children > 0 {
                    stats.final_total_occupancy_sum_for_some_keys +=
                        num_edges_for_this_key_to_distinct_children;
                    stats.final_num_occupied_some_edge_keys += 1;
                }
            } else {
                stats.final_edges_with_none_key += num_edges_for_this_key_to_distinct_children;
                if num_edges_for_this_key_to_distinct_children > 0 {
                    stats.final_total_occupancy_sum_for_none_keys +=
                        num_edges_for_this_key_to_distinct_children;
                    stats.final_num_occupied_none_edge_keys += 1;
                }
            }
        }

        if node_guard.value.end {
            stats.final_nodes_with_clean_end += 1;
        }
    }
    crate::debug!(4, "Finished calculating final precompute1 statistics.");
}

pub fn print_precompute_stats0(
    stats: &PrecomputeStats,
    _token_name_map: &BiBTreeMap<Terminal, usize>,
    _trie_god: &Trie0GodWrapper,
) {
    let avg_some = if stats.final_num_occupied_some_edge_keys > 0 {
        stats.final_total_occupancy_sum_for_some_keys as f64
            / stats.final_num_occupied_some_edge_keys as f64
    } else {
        0.0
    };
    let avg_none = if stats.final_num_occupied_none_edge_keys > 0 {
        stats.final_total_occupancy_sum_for_none_keys as f64
            / stats.final_num_occupied_none_edge_keys as f64
    } else {
        0.0
    };

    println!("--- Precomputation 0 Statistics ---");
    println!(
        "  Initial Root Nodes Created: {}",
        stats.initial_root_nodes_created
    );

    println!("\nNode Counts Breakdown:");
    println!("  There are:");
    println!(
        "  - {} unique nodes, of which",
        stats.final_unique_nodes_count
    );
    println!("    - {} are roots", stats.final_root_nodes_count);
    let non_root_count = stats
        .final_unique_nodes_count
        .saturating_sub(stats.final_root_nodes_count);
    println!("    - {} are non-roots, of which", non_root_count);
    println!(
        "        - {} are internal (non-root, non-leaf)",
        stats.final_non_root_internal_nodes_count
    );
    println!(
        "        - {} are leaves (non-root)",
        stats.final_leaf_nodes_count
    );

    println!("\nFinal Graph Structure (after sharing and deduplication):");
    println!("  Unique Nodes: {}", stats.final_unique_nodes_count);
    println!("  Total Edges: {}", stats.final_edges_count);
    println!(
        "    Edges with None Key: {}",
        stats.final_edges_with_none_key
    );
    println!(
        "    Edges with Some Key: {}",
        stats.final_edges_with_some_key
    );
    println!(
        "  Nodes with Clean End: {}",
        stats.final_nodes_with_clean_end
    );
    println!(
        "  Average edge occupancy for Some-key edges:    {:.2}",
        avg_some
    );
    println!(
        "  Average edge occupancy for None-key edges:    {:.2}",
        avg_none
    );
    println!(
        "  Total ranges in all HybridBitsets: {}",
        stats.final_total_ranges_in_bvs
    );
}

pub fn print_precompute_stats1(
    stats: &PrecomputeStats,
    _token_name_map: &BiBTreeMap<Terminal, usize>,
    _trie_god: &Trie1GodWrapper,
) {
    let avg_some = if stats.final_num_occupied_some_edge_keys > 0 {
        stats.final_total_occupancy_sum_for_some_keys as f64
            / stats.final_num_occupied_some_edge_keys as f64
    } else {
        0.0
    };
    let avg_none = if stats.final_num_occupied_none_edge_keys > 0 {
        stats.final_total_occupancy_sum_for_none_keys as f64
            / stats.final_num_occupied_none_edge_keys as f64
    } else {
        0.0
    };

    println!("--- Precomputation 1 Statistics ---");
    println!(
        "  Initial Root Nodes Created: {}",
        stats.initial_root_nodes_created
    );

    println!("\nNode Counts Breakdown:");
    println!("  There are:");
    println!(
        "  - {} unique nodes, of which",
        stats.final_unique_nodes_count
    );
    println!("    - {} are roots", stats.final_root_nodes_count);
    let non_root_count = stats
        .final_unique_nodes_count
        .saturating_sub(stats.final_root_nodes_count);
    println!("    - {} are non-roots, of which", non_root_count);
    println!(
        "        - {} are internal (non-root, non-leaf)",
        stats.final_non_root_internal_nodes_count
    );
    println!(
        "        - {} are leaves (non-root)",
        stats.final_leaf_nodes_count
    );

    println!("\nFinal Graph Structure (after sharing and deduplication):");
    println!("  Unique Nodes: {}", stats.final_unique_nodes_count);
    println!("  Total Edges: {}", stats.final_edges_count);
    println!(
        "    Edges with None Key: {}",
        stats.final_edges_with_none_key
    );
    println!(
        "    Edges with Some Key: {}",
        stats.final_edges_with_some_key
    );
    println!(
        "  Nodes with Clean End: {}",
        stats.final_nodes_with_clean_end
    );
    println!(
        "  Average edge occupancy for Some-key edges:    {:.2}",
        avg_some
    );
    println!(
        "  Average edge occupancy for None-key edges:    {:.2}",
        avg_none
    );
    println!(
        "  Total ranges in all HybridBitsets: {}",
        stats.final_total_ranges_in_bvs
    );
}
