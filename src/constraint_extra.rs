use std::collections::HashMap;
use crate::constraint::{GrammarConstraint, Precomputed, PrecomputeNode};
use crate::datastructures::gss::PrecomputeNode2;
use crate::types::{TerminalID as GrammarTokenID};
use crate::datastructures::trie::{ArcFreeTrie as Trie, NodeId, GodWrapper};
use crate::tokenizer::{TokenizerStateID, LLMTokenID};
use std::collections::{HashSet, VecDeque, BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};
use bitvec::prelude::BitVec;
use crate::datastructures::hybrid_bitset::HybridBitset;
use bimap::BiBTreeMap;
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
        _ => return bv_neat_str,
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
    _node_arc: &NodeId,
    _prefix: String,
    _visited: &mut HashSet<*const PrecomputeNode>,
    _original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
    _token_name_map: Option<&BiBTreeMap<Terminal, usize>>,
    _llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
) {
    // Arena-based dump can be added if necessary for textual visualization.
    // Intentionally left minimal to avoid clutter.
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
        precomputed: &BTreeMap<TokenizerStateID, NodeId>,
        original_to_internal_id_bimap: &BiBTreeMap<usize, usize>,
        token_name_map: &BiBTreeMap<Terminal, usize>,
        llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        god: &GodWrapper<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>,
    ) {
        println!("Dumping Precomputed Trie 1 Structure (showing original LLM Token IDs):");
        println!("===================================");

        for (tokenizer_state_id, &root_id) in precomputed {
            println!("\n--- Tokenizer State ID: {} ---", tokenizer_state_id.0);

            god.with_node(root_id, |root_node| {
                let live_tokens_str = format_bv_with_tokens(&root_node.value.live_tokens, Some(original_to_internal_id_bimap), Some(llm_token_map), 5);
                println!("Root Node (MaxDepth: {}){} [Live: {}]", root_node.max_depth, if root_node.value.end { " [END]" } else { "" }, live_tokens_str);
            });
            // Additional recursive printing could be added here.
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

    pub fn _dump_precomputed2(
        precomputed2: &BTreeMap<TokenizerStateID, NodeId>,
        original_to_internal_id_bimap: &BiBTreeMap<usize, usize>,
        llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        god: &GodWrapper<(usize, Option<StateID>), LLMTokenBV, PrecomputedNodeContents>,
    ) {
        println!("Dumping Precomputed Trie 2 Structure (showing original LLM Token IDs):");
        println!("===================================");
        for (tokenizer_state_id, &root_id) in precomputed2 {
            println!("\n--- Tokenizer State ID: {} ---", tokenizer_state_id.0);
            god.with_node(root_id, |root_node| {
                let live_tokens_str = format_bv_with_tokens(&root_node.value.live_tokens, Some(original_to_internal_id_bimap), Some(llm_token_map), 5);
                println!("Root Node (MaxDepth: {}){} [Live: {}]", root_node.max_depth, if root_node.value.end { " [END]" } else { "" }, live_tokens_str);
            });
        }
        println!("\n===================================");
        println!("Dump Complete.");
    }
}

pub fn dump_precompute_trie2_recursive(
    _node_arc: &Arc<RwLock<PrecomputeNode2>>,
    _prefix: String,
    _visited: &mut HashSet<*const PrecomputeNode2>,
    _original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
    _llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
) {
    // Kept for compatibility; not used in arena refactor.
}

pub fn calculate_final_stats2(
    precomputed_roots: &BTreeMap<TokenizerStateID, NodeId>,
    stats: &mut PrecomputeStats,
    god: &GodWrapper<(usize, Option<StateID>), LLMTokenBV, PrecomputedNodeContents>,
) {
    crate::debug!(2, "Calculating final precompute2 statistics...");

    let mut all_reachable_nodes: BTreeMap<NodeId, NodeId> = BTreeMap::new();
    let mut queue: VecDeque<NodeId> = precomputed_roots.values().copied().collect();
    let mut visited: HashSet<NodeId> = HashSet::new();

    while let Some(id) = queue.pop_front() {
        if visited.insert(id) {
            all_reachable_nodes.insert(id, id);
            let children: Vec<NodeId> = god.with_node(id, |n| n.children.values().flat_map(|m| m.keys().copied()).collect());
            for c in children { queue.push_back(c); }
        }
    }

    *stats = PrecomputeStats::default();
    stats.final_unique_nodes_count = all_reachable_nodes.len();
    stats.final_root_nodes_count = precomputed_roots.len();
    for (_ptr, id) in &all_reachable_nodes {
        god.with_node(*id, |n| {
            if n.children().is_empty() {
                stats.final_leaf_nodes_count += 1;
            } else {
                stats.final_non_root_internal_nodes_count += 1;
            }
            for (_ek, dest_map) in n.children() {
                let num_edges_for_this_key = dest_map.len();
                stats.final_edges_count += num_edges_for_this_key;
                stats.final_edges_with_some_key += num_edges_for_this_key;
                if num_edges_for_this_key > 0 {
                    stats.final_total_occupancy_sum_for_some_keys += num_edges_for_this_key;
                    stats.final_num_occupied_some_edge_keys += 1;
                }
                for llm_token_bv_on_edge in dest_map.values() {
                    stats.final_total_ranges_in_bvs += llm_token_bv_on_edge.inner().ranges_len();
                }
            }
            if n.value.end {
                stats.final_nodes_with_clean_end += 1;
            }
        });
    }
    crate::debug!(2, "Finished calculating final precompute2 statistics.");
}


#[derive(Default, Debug)]
pub struct PrecomputeStats {
    pub initial_root_nodes_created: usize,

    pub final_unique_nodes_count: usize,
    pub final_root_nodes_count: usize,
    pub final_non_root_internal_nodes_count: usize,
    pub final_leaf_nodes_count: usize,
    pub final_edges_count: usize,
    pub final_edges_with_none_key: usize,
    pub final_edges_with_some_key: usize,
    pub final_nodes_with_clean_end: usize,

    pub final_total_occupancy_sum_for_some_keys: usize,
    pub final_num_occupied_some_edge_keys: usize,
    pub final_total_occupancy_sum_for_none_keys: usize,
    pub final_num_occupied_none_edge_keys: usize,

    pub final_grammar_token_edge_key_counts: BTreeMap<GrammarTokenID, usize>,
    pub final_grammar_token_edge_fanouts_dist: BTreeMap<GrammarTokenID, Vec<usize>>,
    pub final_grammar_token_edge_token_set_sizes_dist: BTreeMap<GrammarTokenID, Vec<usize>>,

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
    precomputed_roots: &BTreeMap<TokenizerStateID, NodeId>,
    stats: &mut PrecomputeStats,
    god: &GodWrapper<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>,
) {
    crate::debug!(2, "Calculating final precompute statistics (within constraint_extra)...");

    let mut all_reachable_nodes: BTreeMap<NodeId, NodeId> = BTreeMap::new();
    let mut queue: VecDeque<NodeId> = precomputed_roots.values().copied().collect();
    let mut visited: HashSet<NodeId> = HashSet::new();

    while let Some(id) = queue.pop_front() {
        if visited.insert(id) {
            all_reachable_nodes.insert(id, id);
            let children = god.with_node(id, |n| n.children.values().flat_map(|m| m.keys().copied()).collect::<Vec<_>>());
            for c in children {
                queue.push_back(c);
            }
        }
    }

    stats.final_unique_nodes_count = all_reachable_nodes.len();

    let root_node_pointers: HashSet<NodeId> = precomputed_roots
        .values()
        .copied()
        .collect();
    stats.final_root_nodes_count = root_node_pointers.len();

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
    stats.final_non_root_internal_nodes_count = 0;
    stats.final_leaf_nodes_count = 0;
    stats.edges_pruned_by_terminal_sequence = 0;
    stats.final_total_ranges_in_bvs = 0;

    for (&id, _) in &all_reachable_nodes {
        god.with_node(id, |node| {
            if !root_node_pointers.contains(&id) {
                if node.children().is_empty() {
                    stats.final_leaf_nodes_count += 1;
                } else {
                    stats.final_non_root_internal_nodes_count += 1;
                }
            }

            for (edge_key_opt, dest_map) in node.children() {
                let num_edges_for_this_key_to_distinct_children = dest_map.len();
                stats.final_edges_count += num_edges_for_this_key_to_distinct_children;

                if let Some(gtid) = edge_key_opt {
                    stats.final_edges_with_some_key += num_edges_for_this_key_to_distinct_children;
                    *stats.final_grammar_token_edge_key_counts.entry(*gtid).or_insert(0) += num_edges_for_this_key_to_distinct_children;

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

            if node.value.end {
                stats.final_nodes_with_clean_end += 1;
            }
        });
    }
    crate::debug!(2, "Finished calculating final precompute statistics (within constraint_extra).");
}


pub fn print_precompute_stats(
    stats: &PrecomputeStats,
    token_name_map: &BiBTreeMap<Terminal, usize>, // Used to get token names from GrammarTokenID
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
    let non_root_count = stats.final_unique_nodes_count.saturating_sub(stats.final_root_nodes_count);
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
        usize,
        (usize, Option<f64>, Option<f64>),
        (usize, Option<f64>, Option<f64>)
    )> = Vec::new();

    for (gtid, key_usages_count) in &stats.final_grammar_token_edge_key_counts {
        let fanouts_for_gtid = stats.final_grammar_token_edge_fanouts_dist
                                    .get(gtid)
                                    .cloned()
                                    .unwrap_or_else(Vec::new);
        let child_stats = calculate_stats_from_vec_usize(&fanouts_for_gtid);

        let token_set_sizes_for_gtid = stats.final_grammar_token_edge_token_set_sizes_dist
                                            .get(gtid)
                                            .cloned()
                                            .unwrap_or_else(Vec::new);
        let toks_stats = calculate_stats_from_vec_usize(&token_set_sizes_for_gtid);

        grammar_token_stats_new.push((*gtid, *key_usages_count, child_stats, toks_stats));
    }

    grammar_token_stats_new.sort_by(|a, b| b.1.cmp(&a.1));

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
            .get_by_right(&gtid.0)
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
}
