use crate::constraint::{GrammarConstraint, Precomputed, PrecomputeNode, PrecomputedNodeContents, PrecomputedFinalizer, LLMTokenBV};
use crate::types::{TerminalID as GrammarTokenID};
use crate::datastructures::trie::{Trie, node_ptr};
use crate::tokenizer::{TokenizerStateID, LLMTokenID};
use std::collections::{HashSet, VecDeque, BTreeMap};
use std::sync::{Arc, Mutex};
use bitvec::prelude::BitVec;
use crate::datastructures::hybrid_bitset::HybridBitset;
use bimap::BiBTreeMap;
use crate::datastructures::ArcPtrWrapper;

/// Helper function to print the indices of set bits in a HybridBitset, optionally mapping them.
fn format_bv_indices(
    bv: &LLMTokenBV,
    // This bimap maps: OriginalTokenID.0 (left) <-> InternalTokenID.0 (right)
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>
) -> String {
    let indices: Vec<String> = bv.iter().map(|internal_id_val| {
        if let Some(bimap) = original_internal_bimap {
            // We have an internal_id_val (which is a right-side value in the bimap),
            // and we want to find its corresponding original_id_val (left-side value).
            bimap.get_by_right(&internal_id_val).map_or_else(
                || format!("{} (unmapped internal)", internal_id_val),
                |original_id_val| original_id_val.to_string()
            )
        } else {
            internal_id_val.to_string() // No map provided, print the internal ID
        }
    }).collect();
    if indices.len() > 10 {
        format!("[{} indices starting with {}...]", indices.len(), indices[0..5].join(", "))
    } else if indices.is_empty() {
        "[]".to_string()
    } else {
        format!("[{}]", indices.join(", "))
    }
}

/// Helper function to print PrecomputedFinalizer details.
pub(crate) fn print_finalizer(
    grammar_token_id: GrammarTokenID,
    finalizer: &PrecomputedFinalizer,
    indent: &str,
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>
) {
    println!("{}  - Finalizer for GrammarTokenID({}):", indent, grammar_token_id.0);
    for (tokenizer_state_id, llm_tokens) in &finalizer.content { // llm_tokens are internal
        println!("{}    Tokenizer State {}:", indent, tokenizer_state_id.0);
        // Pass original_internal_bimap to format_bv_indices
        println!("{}      LLM Tokens: {}", indent, format_bv_indices(llm_tokens, original_internal_bimap));
    }
}

/// Helper function to recursively dump the structure of a PrecomputeNode Trie.
fn dump_precompute_trie_recursive(
    node_arc: &Arc<Mutex<PrecomputeNode>>,
    indent: String,
    visited: &mut HashSet<*const PrecomputeNode>,
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>
) {
    let node_ptr_val = node_ptr(node_arc);
    if !visited.insert(node_ptr_val) {
        println!("{}-> Ref {:p} (already printed)", indent, node_ptr_val);
        return;
    }

    let node = node_arc.lock().expect("Mutex poisoned during dump");

    println!("{}-> Node {:p} (MaxDepth: {})", indent, &node, node.max_depth); // Node struct doesn't have max_depth field based on the original Trie definition provided. Keeping this as is based on previous code state.

    // Print Node Value (Finalizers)
    if !node.value.finalizers().is_empty() {
        println!("{}  Finalizers:", indent);
        for (grammar_token_id, finalizer) in node.value.finalizers() {
            print_finalizer(*grammar_token_id, finalizer, &indent, original_internal_bimap); // Pass original_internal_bimap
        }
    }
    if let Some(clean_end) = &node.value.clean_end { // clean_end stores internal IDs
        println!("{}  Clean End LLM Tokens: {}", indent, format_bv_indices(clean_end, original_internal_bimap)); // Pass original_internal_bimap
    }

    // Print Children (Edges)
    if node.children().is_empty() {
        println!("{}  (Leaf Node)", indent);
    } else {
        println!("{}  Children:", indent);
        let new_indent = format!("{}    ", indent);
        for (edge_key, children_vec) in node.children() {
            for (child_wrapper_arc, edge_val_bv) in children_vec {
                 println!(
                    "{}Edge GrammarTokenID({:?}): LLM Tokens: {} -> Child Ptr: {:p}",
                    indent,
                    edge_key.map(|grammar_token_id| grammar_token_id.0),
                    format_bv_indices(edge_val_bv, original_internal_bimap), // Pass original_internal_bimap
                    node_ptr(child_wrapper_arc.as_arc()) // Use as_arc() to get the Arc
                );
                // Recurse
                dump_precompute_trie_recursive(child_wrapper_arc.as_arc(), new_indent.clone(), visited, original_internal_bimap); // Pass original_internal_bimap
            }
        }
    }
}

impl GrammarConstraint { // This is in constraint_extra.rs
    /// Dumps the structure of the precomputed Trie map for visualization.
    pub fn dump_precomputed(&self) {
        println!("Dumping Precomputed Trie Structure (showing original LLM Token IDs):");
        println!("===================================");

        for (tokenizer_state_id, root_node_trie) in &self.precomputed {
            println!("\n--- Tokenizer State ID: {} ---", tokenizer_state_id.0);

            // Need to wrap the root_node_trie (which is a Trie, not an Arc<Mutex<Trie>>)
            // in an Arc<Mutex<>> to match the recursive function's expectation.
            // This is slightly awkward but necessary for the shared recursive logic.
            let root_node_arc = Arc::new(Mutex::new(root_node_trie.clone()));

            let mut visited: HashSet<*const PrecomputeNode> = HashSet::new();
            // Pass the bimap
            dump_precompute_trie_recursive(&root_node_arc, "".to_string(), &mut visited, Some(&self.original_to_internal_id_bimap));
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
    pub final_edges_count: usize,
    pub final_edges_with_none_key: usize,
    pub final_edges_with_some_key: usize,
    pub final_nodes_with_clean_end: usize,
    pub final_total_finalizer_entries_in_graph: usize, // Sum of node.value.finalizers.values().map(|pf| pf.content.len()).sum() across unique nodes

    // For average edge occupancy per key type
    pub final_total_occupancy_sum_for_some_keys: usize,
    pub final_num_occupied_some_edge_keys: usize,
    pub final_total_occupancy_sum_for_none_keys: usize,
    pub final_num_occupied_none_edge_keys: usize,

    // New fields for grammar token edge key statistics
    pub final_grammar_token_edge_key_counts: BTreeMap<GrammarTokenID, usize>,
    pub final_grammar_token_edge_fanouts_dist: BTreeMap<GrammarTokenID, Vec<usize>>,
    pub final_grammar_token_edge_token_set_sizes_dist: BTreeMap<GrammarTokenID, Vec<usize>>,
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
    precomputed_roots: &BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>>,
    stats: &mut PrecomputeStats,
) {
    crate::debug!(2, "Calculating final precompute statistics (within constraint_extra)...");
    let mut all_reachable_nodes_for_final_stats: HashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>> = HashSet::new();
    for root_arc_mutex_node in precomputed_roots.values() {
        // Assuming PrecomputeNode::all_nodes is accessible and returns HashSet<Arc<Mutex<PrecomputeNode>>>
        // If PrecomputeNode is crate::constraint::PrecomputeNode, it should be.
        let nodes_from_this_root = crate::constraint::PrecomputeNode::all_nodes(root_arc_mutex_node.clone());
        for node_arc in nodes_from_this_root {
            all_reachable_nodes_for_final_stats.insert(ArcPtrWrapper::new(node_arc));
        }
    }
    stats.final_unique_nodes_count = all_reachable_nodes_for_final_stats.len();

    stats.final_total_occupancy_sum_for_some_keys = 0;
    stats.final_num_occupied_some_edge_keys = 0;
    stats.final_total_occupancy_sum_for_none_keys = 0;
    stats.final_num_occupied_none_edge_keys = 0;
    stats.final_edges_count = 0; // Initialize explicitly
    stats.final_edges_with_none_key = 0; // Initialize explicitly
    stats.final_edges_with_some_key = 0; // Initialize explicitly
    stats.final_nodes_with_clean_end = 0; // Initialize explicitly
    stats.final_total_finalizer_entries_in_graph = 0; // Initialize explicitly
    stats.final_grammar_token_edge_key_counts.clear();
    stats.final_grammar_token_edge_fanouts_dist.clear();
    stats.final_grammar_token_edge_token_set_sizes_dist.clear();


    for comp_arc_node in &all_reachable_nodes_for_final_stats {
        let node_arc = comp_arc_node.as_arc();
        let node_guard = node_arc.lock().expect("Mutex poisoned during final stats calculation");

        for (edge_key_opt, dest_map) in node_guard.children() {
            let num_edges_for_this_key_to_distinct_children = dest_map.len();
            stats.final_edges_count += num_edges_for_this_key_to_distinct_children;

            if let Some(gtid) = edge_key_opt {
                stats.final_edges_with_some_key += num_edges_for_this_key_to_distinct_children;
                *stats.final_grammar_token_edge_key_counts.entry(*gtid).or_insert(0) += 1; // Note: original was +=1, this might be per node using the key. Let's assume it's counting how many nodes use this gtid as an edge key.
                                                                                            // If it's sum of fanouts, it should be `+= num_edges_for_this_key_to_distinct_children`.
                                                                                            // The original code `*stats.final_grammar_token_edge_key_counts.entry(*gtid).or_insert(0) += 1;` means "number of source nodes that have an outgoing edge with this gtid".
                                                                                            // Let's stick to the original logic.

                stats.final_grammar_token_edge_fanouts_dist
                    .entry(*gtid)
                    .or_default()
                    .push(num_edges_for_this_key_to_distinct_children);
                for llm_token_bv_on_edge in dest_map.values() {
                    stats.final_grammar_token_edge_token_set_sizes_dist
                        .entry(*gtid)
                        .or_default()
                        .push(llm_token_bv_on_edge.len());
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

        if node_guard.value.clean_end.is_some() {
            stats.final_nodes_with_clean_end += 1;
        }
        for finalizer_for_gtid in node_guard.value.finalizers().values() { // Use .finalizers() method
            stats.final_total_finalizer_entries_in_graph += finalizer_for_gtid.content.len();
        }
    }
    crate::debug!(2, "Finished calculating final precompute statistics (within constraint_extra).");
}


pub fn print_precompute_stats(
    stats: &PrecomputeStats,
    token_name_map: &BiBTreeMap<String, usize>, // Used to get token names from GrammarTokenID
) {
    let avg_some = if stats.final_num_occupied_some_edge_keys > 0 {
        stats.final_total_occupancy_sum_for_some_keys as f64 / stats.final_num_occupied_some_edge_keys as f64
    } else { 0.0 };
    let avg_none = if stats.final_num_occupied_none_edge_keys > 0 {
        stats.final_total_occupancy_sum_for_none_keys as f64 / stats.final_num_occupied_none_edge_keys as f64
    } else { 0.0 };

    println!("--- Precomputation Statistics ---");
    println!("  Initial Root Nodes Created: {}", stats.initial_root_nodes_created);
    println!("\nFinal Graph Structure (after sharing and deduplication):");
    println!("  Unique Nodes: {}", stats.final_unique_nodes_count);
    println!("  Total Edges: {}", stats.final_edges_count);
    println!("    Edges with None Key: {}", stats.final_edges_with_none_key);
    println!("    Edges with Some Key: {}", stats.final_edges_with_some_key);
    println!("  Nodes with Clean End: {}", stats.final_nodes_with_clean_end);
    println!("  Total Finalizer Entries (sum of map sizes in all unique nodes): {}", stats.final_total_finalizer_entries_in_graph);
    println!("  Average edge occupancy for Some-key edges:    {:.2}", avg_some);
    println!("  Average edge occupancy for None-key edges:    {:.2}", avg_none);

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
            .unwrap_or_else(|| gtid.0.to_string());

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


#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use crate::finite_automata::{eat_u8, Regex};
    use crate::glr::grammar::{prod, t, Terminal};
    use crate::glr::parser::GLRParser;
    use crate::glr::table::generate_glr_parser_with_terminal_map;
    use crate::datastructures::hybrid_bitset::HybridBitset; // Explicitly import HybridBitset
    use std::hash::{Hash, Hasher};
    use crate::interface::{eat_u8_fast, eat_u8_negation_fast, eat_u8_range_fast, repeat0_fast, eat_any_fast}; // Added eat_any_fast

    use std::fs::{self, File};
    use std::io::{BufReader, Read, Write};
    use std::path::Path;
    use std::sync::{Arc, Mutex};
    use bimap::BiBTreeMap;
    use reqwest::blocking;
    use serde_json;
    use crate::constraint::GrammarConstraint;
    use crate::datastructures::trie::Trie;
    // Already a main dependency, but good to be explicit if used directly
    // reqwest will be used if the file isn't cached, ensure it's in dev-dependencies
    use crate::tokenizer::{LLMTokenID, LLMTokenMap};
    use crate::types::TerminalID;

    // Use concrete types for merge tests
    type TestTrieMerge = Trie<&'static str, Vec<i32>, String>;
    type TestNodeMerge = Arc<Mutex<TestTrieMerge>>;
    // Use simpler types for basic tests
    type TestTrieBasic = Trie<&'static str, &'static str, i32>;
    type TestNodeBasic = Arc<Mutex<TestTrieBasic>>;

    // Use concrete types for EdgeInserter tests
    type TestTrieEI = Trie<&'static str, HybridBitset, String>; // Use HybridBitset here
    type TestNodeEI = Arc<Mutex<TestTrieEI>>;

    // Helper to get Arc pointer for tests
    fn arc_ptr<N>(arc: &Arc<Mutex<N>>) -> *const Mutex<N> {
        Arc::as_ptr(arc)
    }

    // Helper function to load or download GPT-2 vocab
    fn load_or_download_gpt2_vocab(
        cache_dir: &Path,
        file_name: &str,
        url: &str,
    ) -> Result<BTreeMap<String, u32>, Box<dyn std::error::Error>> {
        fs::create_dir_all(cache_dir)?;
        let cache_path = cache_dir.join(file_name);

        if cache_path.exists() {
            println!("Loading GPT-2 vocab from cache: {:?}", cache_path);
            let file = File::open(cache_path)?;
            let reader = BufReader::new(file);
            let vocab: BTreeMap<String, u32> = serde_json::from_reader(reader)?;
            Ok(vocab)
        } else {
            println!("Downloading GPT-2 vocab from: {}", url);
            let response = blocking::get(url)?.error_for_status()?;
            let content = response.text()?;

            let mut file = File::create(&cache_path)?;
            file.write_all(content.as_bytes())?;
            println!("Saved GPT-2 vocab to cache: {:?}", cache_path);

            let vocab: BTreeMap<String, u32> = serde_json::from_str(&content)?;
            Ok(vocab)
        }
    }


    #[test]
    fn test_constraint_simple() {
        // LLM tokens: "ab", "ac", "$"
        // Grammar tokens: "a", "ab", "b|c", "$" (EOF)
        // Grammar: S -> X $ ; X -> "a" ("b|c") | "ab"
        let expr = groups![
            eat_u8(b'a'),
            seq![eat_u8(b'a'), eat_u8(b'b')],
            choice![eat_u8(b'b'), eat_u8(b'c')], // ID 2
            eat_u8(b'$'),
        ];
        let tokenizer = expr.build();

        let mut llm_token_map = LLMTokenMap::new();
        llm_token_map.insert(b"ab".to_vec(), LLMTokenID(0));
        llm_token_map.insert(b"ac".to_vec(), LLMTokenID(1));
        llm_token_map.insert(b"$".to_vec(), LLMTokenID(2));

        // Grammar Terminals mapped to Tokenizer IDs
        let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
        grammar_token_map.insert(Terminal("A".to_string()), TerminalID(0)); // Corresponds to eat_u8(b'a')
        grammar_token_map.insert(Terminal("AB".to_string()), TerminalID(1)); // Corresponds to seq![eat_u8(b'a'), eat_u8(b'b')]
        grammar_token_map.insert(Terminal("B_OR_C".to_string()), TerminalID(2)); // Corresponds to choice![eat_u8(b'b'), eat_u8(b'c')]
        grammar_token_map.insert(Terminal("EOF".to_string()), TerminalID(3)); // Corresponds to eat_u8(b'$')

        let productions = vec![
            prod("S", vec![nt("X"), t("EOF")]), // S -> X $
            prod("X", vec![t("A"), t("B_OR_C")]), // X -> a (b|c)
            prod("X", vec![t("AB")]),             // X -> ab
        ];

        let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map.clone());
        dbg!(&parser);

        let mut token_name_map = BiBTreeMap::new();
         for (term, id) in &grammar_token_map {
            token_name_map.insert(term.0.clone(), id.0);
        }

        let constraint = GrammarConstraint::new(
            tokenizer,
            parser,
            llm_token_map,
            token_name_map,
            3, // max_llm_token_id should be 3 for 0, 1, 2
        );
        // constraint.dump_precomputed(); // Commented out dump for cleaner test output

        let mut constraint_state = constraint.init();

        constraint_state.step_with_all_llm_tokens();

        // Initially, we can match "a" (part of "ab" or "ac") or "ab".
        // "a" leads to expecting "b" or "c".
        // "ab" leads to expecting "$".
        let mask = constraint_state.get_mask();
        assert_eq!(mask, HybridBitset::from_iter(vec![0, 1])); // Expect "ab" or "ac"

        // Commit "ab" (LLMTokenID 0)
        constraint_state.commit(LLMTokenID(0));
        constraint_state.step_with_all_llm_tokens();
        let mask = constraint_state.get_mask();
        assert_eq!(mask, HybridBitset::from_iter(vec![2])); // Expect "$" (EOF)
    }

    #[test]
    fn test_constraint_expression() {
        // Example grammar: E -> E '+' T | T; T -> T '*' F | F; F -> '(' E ')' | 'i'
        // LLM token vocabulary: i, +, *, (, ), (i, +i
        let mut llm_token_map = LLMTokenMap::new();
        llm_token_map.insert(b"i".to_vec(), LLMTokenID(0));
        llm_token_map.insert(b"+".to_vec(), LLMTokenID(1));
        llm_token_map.insert(b"*".to_vec(), LLMTokenID(2));
        llm_token_map.insert(b"(".to_vec(), LLMTokenID(3));
        llm_token_map.insert(b")".to_vec(), LLMTokenID(4));
        llm_token_map.insert(b"(i".to_vec(), LLMTokenID(5));
        llm_token_map.insert(b"+i".to_vec(), LLMTokenID(6));

        // Tokenizer regex for grammar tokens '+' '*' '(' ')' 'i'
        let expr = groups![
            eat_u8(b'+'),
            eat_u8(b'*'),
            eat_u8(b'('),
            eat_u8(b')'),
            eat_u8(b'i'),
        ];
        let tokenizer = expr.build();

        // Grammar productions
        let productions = vec![
            prod("S", vec![nt("E"), t("EOF")]), // Start production
            prod("E", vec![nt("E"), t("PLUS"), nt("T")]),
            prod("E", vec![nt("T")]),
            prod("T", vec![nt("T"), t("TIMES"), nt("F")]),
            prod("T", vec![nt("F")]),
            prod("F", vec![t("LPAREN"), nt("E"), t("RPAREN")]),
            prod("F", vec![t("I")]),
        ];
        // Map grammar terminals to IDs matching regex order
        let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
        grammar_token_map.insert(Terminal("PLUS".to_string()), TerminalID(0));
        grammar_token_map.insert(Terminal("TIMES".to_string()), TerminalID(1));
        grammar_token_map.insert(Terminal("LPAREN".to_string()), TerminalID(2));
        grammar_token_map.insert(Terminal("RPAREN".to_string()), TerminalID(3));
        grammar_token_map.insert(Terminal("I".to_string()), TerminalID(4));
        grammar_token_map.insert(Terminal("EOF".to_string()), TerminalID(5));

        let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map.clone()); // Start production is index 6
        dbg!(&parser);

        let mut token_name_map = BiBTreeMap::new();
         for (term, id) in &grammar_token_map {
            token_name_map.insert(term.0.clone(), id.0);
        }

        let constraint = GrammarConstraint::new(
            tokenizer,
            parser,
            llm_token_map,
            token_name_map,
            7, // max_llm_token_id should be 7 for IDs 0-6
        );
        // constraint.dump_precomputed(); // Commented out dump for cleaner test output

        // Initial state and step
        let mut state = constraint.init();
        state.step_with_all_llm_tokens();
        let mask = state.get_mask();
        // Expect LLM tokens that can start an expression: i (0), '(' (3), "(i" (5)
        assert_eq!(mask, HybridBitset::from_iter(vec![0, 3, 5]));

        // Commit "(i"
        state.commit(LLMTokenID(5));
        state.step_with_all_llm_tokens();
        let mask = state.get_mask();
        // Now expect '+', '*', ')', '+i' => IDs 1,2,4,6
        assert_eq!(mask, HybridBitset::from_iter(vec![1, 2, 4, 6]));

        // // Commit "(i"
        // state.commit(LLMTokenID(5));
        // state.step_with_all_llm_tokens();
        // state.commit(LLMTokenID(4)); // Assuming ")"
        // state.step_with_all_llm_tokens();
        // let mask = state.get_mask();
        // assert_eq!(mask, HybridBitset::from_iter(vec![1, 2, 5, 6, 3])); // Expect '+', '*', '(', '(i', '+i'

    }

    #[test]
    fn test_precompute_for_python_name_token() {
        // ignore = rep(choice([
        //     eat_u8(ord(" ")),
        //     seq([eat_u8(ord("#")), rep(eat_u8_negation(ord("\n"))), eat_u8(ord("\n"))]),
        // ]))
        // digit = choice([eat_u8(c) for c in range(ord("0"), ord("9") + 1)])
        // alph_lower = choice([eat_u8(c) for c in range(ord("a"), ord("z") + 1)])
        // alph_upper = choice([eat_u8(c) for c in range(ord("A"), ord("Z") + 1)])
        //
        // name_start = choice([
        //     alph_lower,
        //     alph_upper,
        //     eat_u8(ord("_"))
        // ])
        // name_middle = choice([
        //     name_start,
        //     digit,
        // ])
        let ignore = repeat0_fast(choice_fast!(eat_u8_fast(b' '), seq_fast!(eat_u8_fast(b'#'), repeat0_fast(eat_u8_negation_fast(b'\n')), eat_u8_fast(b'\n'))));

        let digit = eat_u8_range_fast(b'0', b'9');
        let alph_lower = eat_u8_range_fast(b'a', b'z');
        let alph_upper = eat_u8_range_fast(b'A', b'Z');

        let name_start = choice_fast!(alph_lower, alph_upper, eat_u8_fast(b'_'));
        let name_middle = choice_fast!(name_start.clone(), digit);
        let name = seq_fast!(ignore, name_start, repeat0_fast(seq_fast!(name_middle)));

        let tokenizer = name.build();
        dbg!(&tokenizer);

        let llm_tokens: Vec<Vec<u8>> = (0..2).map(|i| format!("abcdefghijk{}", i).as_bytes().to_vec()).collect();
        let llm_tokens_slices: Vec<&[u8]> = llm_tokens.iter().map(|token| &token[..]).collect();
        let llm_token_map: LLMTokenMap = llm_tokens.iter().enumerate().map(|(i, token)| (token.clone(), LLMTokenID(i))).collect();
        let _eof_llm_token_id = llm_tokens.len();
        let internal_num_llm_tokens = llm_tokens.len(); // This corresponds to the number of tokens for precompute

        // For the purpose of this test calling precompute directly, the IDs in llm_token_map are sequential 0..N-1,
        // which serves as the internal mapping. We don't need a separate internal_llm_token_map here.
        let mut internal_llm_token_map_for_precompute = BiBTreeMap::new();
        for (i, token) in llm_tokens.iter().enumerate() {
             internal_llm_token_map_for_precompute.insert(token.clone(), LLMTokenID(i));
        }


        let _precomputed = GrammarConstraint::precompute(
            &tokenizer,
            &internal_llm_token_map_for_precompute, // Use the manually created internal map
            &BiBTreeMap::new(), // empty name‐map
            internal_num_llm_tokens, // Pass the number of tokens
        );
        // print_precomputed(&precomputed);
        println!("Done precomputing");
    }

    #[test]
    fn test_precompute_explosion() {
        let tokenizer = groups![
            eat_u8(b'a'),
            eat_u8(b'a'),
        ].build();

        let llm_tokens: Vec<Vec<u8>> = vec![b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_vec()];
         let llm_token_map: LLMTokenMap = llm_tokens.iter().enumerate().map(|(i, token)| (token.clone(), LLMTokenID(i))).collect();
        let _eof_llm_token_id = llm_tokens.len();
        let internal_num_llm_tokens = llm_tokens.len(); // This corresponds to the number of tokens for precompute

        // For the purpose of this test calling precompute directly, the IDs in llm_token_map are sequential 0..N-1,
        // which serves as the internal mapping. We don't need a separate internal_llm_token_map here.
        let mut internal_llm_token_map_for_precompute = BiBTreeMap::new();
        for (i, token) in llm_tokens.iter().enumerate() {
             internal_llm_token_map_for_precompute.insert(token.clone(), LLMTokenID(i));
        }

        let _precomputed = GrammarConstraint::precompute(
            &tokenizer,
            &internal_llm_token_map_for_precompute, // Use the manually created internal map
            &BiBTreeMap::new(), // empty name‐map
            internal_num_llm_tokens, // Pass the number of tokens
        );
        // print_precomputed(&precomputed);
        println!("Done precomputing");
    }

    #[test]
    fn test_precompute_with_gpt2_vocab() -> Result<(), Box<dyn std::error::Error>> {
        // 1. Define tokenizer: matches anything
        // The tokenizer will have one group (ID 0)
        let tokenizer_expr = groups![repeat0_fast(eat_any_fast())];
        let tokenizer = tokenizer_expr.build();

        // 2. Load LLM tokens from GPT-2 vocab.json
        let vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
        let cache_dir = Path::new(".cache/test_vocabs");
        let vocab_file_name = "gpt2_vocab.json";

        let gpt2_raw_vocab = load_or_download_gpt2_vocab(cache_dir, vocab_file_name, vocab_url)?;

        let mut llm_token_map = LLMTokenMap::new();
        let mut max_llm_token_id_val: u32 = 0;

        // Sample GPT-2 tokens to speed up this test
        // let prop = 1.0;
        let prop = 0.05;
        let total_tokens = gpt2_raw_vocab.len();
        let sample_size = (total_tokens as f64 * prop) as usize;
        println!("Sampling {} out of {} GPT-2 tokens for precompute", sample_size, total_tokens);
        for (token_str, id_val) in gpt2_raw_vocab.into_iter().take(sample_size) {
            llm_token_map.insert(token_str.into_bytes(), LLMTokenID(id_val as usize));
            if id_val > max_llm_token_id_val {
                max_llm_token_id_val = id_val;
            }
        }

        // Manually perform mapping for the test, similar to setup_llm_token_mappings
        // We need a map from bytes to internal IDs (0..N-1 sequence based on sorted bytes)
        let mut sorted_tokens_for_test: Vec<(Vec<u8>, LLMTokenID)> = llm_token_map
            .iter()
            .map(|(bytes, original_id)| (bytes.clone(), *original_id))
            .collect();
        sorted_tokens_for_test.sort_by(|(bytes_a, _), (bytes_b, _)| bytes_a.cmp(bytes_b));

        let mut test_internal_llm_token_map = BiBTreeMap::new(); // bytes -> internal LLMTokenID
        let mut internal_id_counter_for_test = 0;

        for (bytes, _original_llm_id) in sorted_tokens_for_test { // original_llm_id not directly used to make internal map
            let internal_llm_id = LLMTokenID(internal_id_counter_for_test);
            test_internal_llm_token_map.insert(bytes.clone(), internal_llm_id);
            internal_id_counter_for_test += 1;
        }
        let test_internal_num_llm_tokens = internal_id_counter_for_test;


        // 3. Create token_name_map for grammar tokens
        // Our tokenizer has one grammar token (GroupID 0)
        let mut token_name_map = BiBTreeMap::new();
        token_name_map.insert("ANYTHING_GRAMMAR_TOKEN".to_string(), 0 as usize); // GrammarTokenID 0


        // 4. Call precompute
        println!(
            "Starting precompute with GPT-2 vocab ({} tokens, max_original_id_val: {}, internal_num_tokens: {})...",
            llm_token_map.len(),
            max_llm_token_id_val, // Max original ID value encountered
            test_internal_num_llm_tokens // Number of unique internal tokens for precompute
        );

        // This is the main part of the test: ensure it runs without error.
        let _precomputed = GrammarConstraint::precompute(
            &tokenizer,
            &test_internal_llm_token_map,
            &token_name_map,
            test_internal_num_llm_tokens,
        );

        println!("Successfully precomputed with GPT-2 vocab.");
        Ok(())
    }
}
