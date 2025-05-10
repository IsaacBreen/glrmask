use std::cmp::Ordering;
use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
use crate::glr::parser::ParseStateNodeContent;
use crate::glr::parser::{MergeAndIntersect, GLRParser, GLRParserState, ParseState}; // Removed ParseStateKey
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::*;
// Removed duplicate: use bitvec::prelude::*; // Keep for macros or other uses if needed
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque, HashSet};
use std::ops::BitOr;
use std::sync::{Arc, Mutex}; // Removed MutexGuard as it's not used in type signatures
// Removed: use bitvec::macros::internal::funty::Fundamental;
use crate::constraint_extra::{PrecomputeStats, print_precompute_stats}; // Removed print_finalizer
use crate::datastructures::charmap::TrieMap;
use crate::datastructures::gss::prune_and_transform_recursive;
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::types::TerminalID as GrammarTokenID; // Simplified import
use crate::datastructures::trie::EdgeInserter;
use indicatif::{ProgressBar, ProgressStyle};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::ArcPtrWrapper; // Use the new wrapper struct
use std::hash::{Hash, Hasher};


pub type LLMTokenBV = HybridBitset;
pub type GrammarTokenBV = BitVec; // Assuming GrammarTokenBV remains BitVec

#[derive(Debug, Clone)]
pub struct GrammarConstraintState<'a> {
    pub(crate) parent: &'a GrammarConstraint,
    pub(crate) state: BTreeMap<TokenizerStateID, GLRParserState<'a, LLMTokenInfo>>,
}

#[derive(Default, Debug, Clone)]
pub struct PrecomputedFinalizer {
    pub(crate) content: BTreeMap<TokenizerStateID, LLMTokenBV>,
}

impl PrecomputedFinalizer {
    pub(crate) fn new(compatible_llm_tokens: LLMTokenBV, tokenizer_state_id: TokenizerStateID) -> Self {
        let content = BTreeMap::from([(tokenizer_state_id, compatible_llm_tokens)]);
        Self { content }
    }
}

#[derive(Default, Debug, Clone)]
pub(crate) struct PrecomputedNodeContents {
    finalizers: BTreeMap<GrammarTokenID, PrecomputedFinalizer>,
    pub(crate) clean_end: Option<LLMTokenBV>,
}

pub(crate) type PrecomputeNode = Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;
pub(crate) type Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>;

/// Holds the set of active LLM tokens and the intersection of tokens
/// guaranteed to be possible in all paths *below* this GSS node.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LLMTokenInfo {
    /// Union of possible LLM tokens allowed by paths reaching this node.
    pub active: LLMTokenBV,
    /// Intersection of LLM tokens guaranteed by *all* paths descending from this node.
    /// Used for optimization during commit.
    pub intersection: LLMTokenBV,
}

impl Default for LLMTokenInfo {
    fn default() -> Self {
        Self { active: Default::default(), intersection: Default::default() }
    }
}

impl std::fmt::Debug for LLMTokenInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const MAX_ITEMS: usize = 10;

        let format_bv = |bv: &LLMTokenBV| -> String {
            let ids: Vec<_> = bv.iter().collect();
            if ids.len() > MAX_ITEMS {
                format!("[{:?}... ({} total)]", &ids[..MAX_ITEMS], ids.len())
            } else {
                format!("{:?}", ids)
            }
        };

        f.debug_struct("LLMTokenInfo")
            .field("active", &format_args!("{}", format_bv(&self.active)))
            .field(
                "intersection",
                &format_args!("{}", format_bv(&self.intersection)),
            )
            .finish()
    }
}


#[derive(Debug, Clone)] // Removed pub(crate) as it's likely used externally
pub struct GrammarConstraint {
    pub(crate) tokenizer: Regex,
    pub(crate) parser: GLRParser,
    pub(crate) precomputed: Precomputed,
    pub(crate) llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID>,
    /// Bidirectional map: token name ↔ grammar-ID
    pub(crate) token_name_map: BiBTreeMap<String, usize>,
    pub(crate) max_llm_token_id: usize,
}


impl MergeAndIntersect for LLMTokenInfo {
    fn merge(&self, other: &Self) -> Self {
        // Merge: Active tokens are unioned, Intersection tokens are intersected.
        Self {
            active: &self.active | &other.active, // Use reference for ops
            intersection: &self.intersection & &other.intersection, // Use reference for ops
        }
    }
    fn intersect(&self, other: &Self) -> Self {
        // Intersect: Active tokens are intersected, Intersection is also intersected.
        Self {
            active: &self.active & &other.active, // Use reference for ops
            intersection: &self.intersection & &other.intersection, // Use reference for ops
        }
    }
}

impl PrecomputedNodeContents {
    pub(crate) fn finalizers(&self) -> &BTreeMap<GrammarTokenID, PrecomputedFinalizer> { &self.finalizers }

    /// Adds information about a final state reachable via a specific grammar token.
    /// If an entry for the grammar token already exists, it merges the information.
    pub fn push_finalizer_info(&mut self, possible_final_grammar_token: GrammarTokenID, llm_token_id: LLMTokenID, tokenizer_state_id: TokenizerStateID) {
        // max_llm_token_id is no longer needed here, HybridBitset handles size dynamically
        let mut current_compatible_llm_tokens = HybridBitset::new();
        current_compatible_llm_tokens.insert(llm_token_id.0);

        self.finalizers.entry(possible_final_grammar_token)
            .and_modify(|existing_finalizer| {
                existing_finalizer.content.entry(tokenizer_state_id).and_modify(|existing_llm_tokens| {
                    // Use BitOrAssign<&HybridBitset>
                    *existing_llm_tokens |= &current_compatible_llm_tokens;
                }).or_insert(current_compatible_llm_tokens.clone());
            })
            .or_insert_with(|| PrecomputedFinalizer::new(current_compatible_llm_tokens, tokenizer_state_id));
    }
}

impl GrammarConstraint {
    pub fn new(
        tokenizer: Regex,
        parser: GLRParser,
        llm_token_map: LLMTokenMap,
        token_name_map: BiBTreeMap<String, usize>,
        max_llm_token_id: usize
    ) -> Self {
        let precomputed =
            GrammarConstraint::precompute(&tokenizer, &llm_token_map, &token_name_map, max_llm_token_id);
        Self {
            tokenizer,
            parser,
            precomputed,
            llm_token_map,
            token_name_map,
            max_llm_token_id,
        }
    }

    pub fn precompute<'a>(
        tokenizer: &Regex,
        llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        token_name_map: &BiBTreeMap<String, usize>,
        max_llm_token_id: usize,
    ) -> Precomputed {
        let mut stats = PrecomputeStats::default();

        // ---- Helper function to count nodes in VocabPrefixTree ----
        fn count_vocab_nodes(node: &VocabPrefixTreeNode) -> u64 {
            let mut count = 1u64; // Count this node
            for child_node in node.children().values() {
                count += count_vocab_nodes(child_node);
            }
            count
        }
        // -----------------------------------------------------------

        const MERGE_THRESHOLD: usize = 100;

        // ----  ArcPtrWrapper for `Arc<Mutex<PrecomputeNode>>` --------------------------
        // We use ArcPtrWrapper to enable Ord and Hash based on pointer for BTreeSet/HashSet.
        // (Removed NodeHandle struct and its impls as ArcPtrWrapper is used directly)
        // ------------------------------------------------------------------------------------

        // Create the vocab prefix tree.
        let mut tokens_for_vocab_prefix_tree_builder: Vec<(usize, Vec<u8>)> = vec![];
        for (content, id) in llm_token_map {
            tokens_for_vocab_prefix_tree_builder.push((id.0, content.clone()));
        }
        crate::debug!(2, "Building vocab prefix tree");
        let vocab_prefix_tree = VocabPrefixTree::build(&tokens_for_vocab_prefix_tree_builder);
        crate::debug!(2, "Done building vocab prefix tree");

        // --- Count nodes for progress bar ---
        let total_vocab_nodes = count_vocab_nodes(&vocab_prefix_tree.root);
        crate::debug!(2, "Total vocab nodes to process: {}", total_vocab_nodes);

        // --- Initialize Progress Bar ---
        let pb = ProgressBar::new(total_vocab_nodes);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta}) {msg}")
            .expect("Failed to set progress bar template")
            .progress_chars("##-"));
        pb.set_message("Precomputing constraints...");


        // Create the roots.
        let mut precomputed_roots: BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>> = BTreeMap::new();
        for tokenizer_state_id in 0..tokenizer.max_state() {
            let precompute_node = Arc::new(Mutex::new(PrecomputeNode::new(PrecomputedNodeContents::default())));
            precomputed_roots.insert(TokenizerStateID(tokenizer_state_id), precompute_node);
            stats.initial_root_nodes_created += 1;
        }

        // merge_map for deduplicating PrecomputeNode structures resulting from merged paths
        let all_llm_tokens_for_merge_edge = HybridBitset::ones(max_llm_token_id);

        // ---- Helper function to merge ArcPtrWrapper<Mutex<PrecomputeNode>> ----
        fn merge_node_handles_internal(
            set_of_handles: &BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>, // Use ArcPtrWrapper
            all_llm_tokens_for_merge_edge: &HybridBitset,
            threshold: usize,
        ) -> BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>> { // Changed return type to ArcPtrWrapper
            if set_of_handles.is_empty() {
                return BTreeSet::new();
            }
            // If set size is within threshold, don't merge, return the set as is.
            if set_of_handles.len() <= threshold {
                return set_of_handles.clone();
            }

            // Create a new node to represent the merged set.
            let new_merged_pc_node_arc = Arc::new(Mutex::new(PrecomputeNode::new(PrecomputedNodeContents::default())));

            let mut result_set = BTreeSet::new();

            for handle_child in set_of_handles {
                // handle_child is &ArcPtrWrapper<Mutex<PrecomputeNode>>
                // Use handle_child.as_arc() to get &Arc<Mutex<PrecomputeNode>>
                let mut insert_result = EdgeInserter::new(
                    handle_child.as_arc().clone(), // Source Arc<Mutex<PrecomputeNode>>
                    None,                           // Edge key: Option<GrammarTokenID> = None
                    all_llm_tokens_for_merge_edge.clone(), // Edge value: LLMTokenBV
                    // Merge function for edge values (LLMTokenBV)
                    |ev_exist: &LLMTokenBV, ev_new: HybridBitset| Some(ev_exist | &ev_new),
                );

                insert_result = insert_result.try_children(); 

                if insert_result.clone_into_option().is_none() { // Check if destination was found by try_children
                    insert_result = insert_result.try_destination(new_merged_pc_node_arc.clone()); 
                }
            }

            result_set.insert(ArcPtrWrapper::new(new_merged_pc_node_arc)); // The result is the single new merge parent wrapped in ArcPtrWrapper
            result_set // Return a set containing only the new merged handle.
        }
        // --------------------------------------------

        // ---- Define the recursive helper function ----
        fn precompute_recursive_helper(
            vocab_node: &VocabPrefixTreeNode,
            associated_pc_nodes_by_state_param: BTreeMap<TokenizerStateID, BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>>, // Use ArcPtrWrapper
            // Captured or passed-in context:
            tokenizer: &Regex,
            all_llm_tokens_for_merge_edge: &HybridBitset,
            pb: &ProgressBar,
            // max_llm_token_id: usize, // No longer needed here
            merge_threshold: usize, // Renamed from MERGE_THRESHOLD_val
            // Note: merge_node_handles_internal is accessible as it's defined in the outer scope
        ) {
            // --- Increment progress bar ---
            pb.inc(1);

            crate::debug!(3, "Processing VocabPrefixTreeNode ({} children), prefix: '{}'",
                vocab_node.iter_children().count(),
                String::from_utf8_lossy(vocab_node.prefix())
            );

            // Step 1: For each TokenizerStateID, apply merge policy to the set of incoming PrecomputeNode handles.
            let mut effective_source_pc_nodes_map: BTreeMap<TokenizerStateID, BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>> = BTreeMap::new(); // Use ArcPtrWrapper
            for (tokenizer_state_id, set_of_handles) in associated_pc_nodes_by_state_param {
                let effective_handles_set = merge_node_handles_internal(
                    &set_of_handles,
                    all_llm_tokens_for_merge_edge,
                    merge_threshold, // Use renamed parameter
                );
                if !effective_handles_set.is_empty() {
                    effective_source_pc_nodes_map.insert(tokenizer_state_id, effective_handles_set);
                }
            }

            // Step 2: Iterate over children of current_vocab_node in the VocabPrefixTree
            for (bytes_segment, child_vocab_node) in vocab_node.iter_children() {
                crate::debug!(3, "  Transitioning via segment: '{}' to child_vocab_node (prefix: '{}')",
                    String::from_utf8_lossy(bytes_segment),
                    String::from_utf8_lossy(child_vocab_node.prefix())
                );

                // Initialize yellow_nodes set for the current segment
                let mut yellow_nodes: HashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>> = HashSet::new(); // Use ArcPtrWrapper


                // This map will collect PrecomputeNodes formed at the *end* of processing bytes_segment,
                // to be associated with child_vocab_node for the next DFS level.
                // Key: TokenizerStateID (at that offset within the segment processing).
                // Value: Map from TokenizerStateID (at the end of the segment) to a SET of PrecomputeNode handles.
                let mut next_level_associations_for_child: BTreeMap<TokenizerStateID, BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>> = BTreeMap::new(); // Use ArcPtrWrapper

                // Inner queue for processing bytes_segment (offset-based)
                // Key: offset within bytes_segment.
                // Value: Map from TokenizerStateID (at that offset) to a SET of PrecomputeNode handles that are sources for the *remaining* part of the segment.
                let mut segment_processing_q: BTreeMap<usize, BTreeMap<TokenizerStateID, BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>>> = BTreeMap::new(); // Use ArcPtrWrapper

                // Initialize segment_processing_q at offset 0 using effective_source_pc_nodes_map (nodes at current_vocab_node after merge policy)
                for (initial_tokenizer_state_id, effective_source_handles_set) in &effective_source_pc_nodes_map {
                    segment_processing_q.entry(0).or_default()
                        .entry(*initial_tokenizer_state_id).or_default()
                        .extend(effective_source_handles_set.iter().cloned()); // cloning ArcPtrWrapper clones the inner Arc
                }

                // Process the current bytes_segment
                while let Some((current_offset, states_at_offset_map)) = segment_processing_q.pop_first() {
                    for (tokenizer_state_before_suffix, current_path_source_handles_set) in states_at_offset_map {
                        if current_path_source_handles_set.is_empty() {
                            continue;
                        }

                        // Apply the merge policy to the set of handles for this specific path in segment processing.
                        let effective_source_handles_for_suffix: BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>> = merge_node_handles_internal( // Use ArcPtrWrapper
                                                                                                                                               &current_path_source_handles_set,
                                                                                                                                               all_llm_tokens_for_merge_edge,
                                                                                                                                               merge_threshold, // Use renamed parameter
                        );

                        if effective_source_handles_for_suffix.is_empty() {
                            continue;
                        }

                        // Paint the current source nodes yellow while processing edges originating from them in this segment step.
                        for handle in &effective_source_handles_for_suffix {
                            yellow_nodes.insert(handle.clone()); // clone ArcPtrWrapper
                        }


                        let segment_suffix_to_process = &bytes_segment[current_offset..];
                        let results = tokenizer.execute_from_state(segment_suffix_to_process, tokenizer_state_before_suffix);

                        // --- Existing logic for results.matches ---
                        for match_info in &results.matches {
                            let grammar_token_id = GrammarTokenID(match_info.id);
                            let match_end_offset = current_offset + match_info.width;

                            let edge_llm_tokens = child_vocab_node.reachable_token_ids().clone();


                            // Iterate over each effective source handle
                            for segment_source_pc_handle in &effective_source_handles_for_suffix {
                                // segment_source_pc_handle is &ArcPtrWrapper<Mutex<PrecomputeNode>>
                                // Use as_arc() to get &Arc<Mutex<PrecomputeNode>>
                                let source_for_inserter_arc = segment_source_pc_handle.as_arc().clone(); // Clone Arc for EdgeInserter
                                let edge_key_for_inserter = Some(grammar_token_id);

                                let mut inserter = EdgeInserter::new(
                                    source_for_inserter_arc.clone(), // Clone Arc for EdgeInserter
                                    edge_key_for_inserter,
                                    edge_llm_tokens.clone(), // Use the already cloned edge_llm_tokens
                                    |ev_exist: &HybridBitset, ev_new: HybridBitset| Some(ev_exist | &ev_new),
                                );

                                let mut potential_targets: Vec<Arc<Mutex<PrecomputeNode>>> = Vec::new();
                                if match_end_offset < bytes_segment.len() {
                                    if let Some(map_at_offset) = segment_processing_q.get(&match_end_offset) {
                                        if let Some(set_at_state0) = map_at_offset.get(&TokenizerStateID(0)) {
                                            // Only add targets if they are NOT yellow
                                            potential_targets.extend(set_at_state0.iter()
                                                .filter(|h| !yellow_nodes.contains(h)) // h is &ArcPtrWrapper, yellow_nodes contains ArcPtrWrapper
                                                .map(|h| h.as_arc().clone())); // h is &ArcPtrWrapper, use as_arc()
                                        }
                                    }
                                }
                                if match_end_offset == bytes_segment.len() {
                                    if let Some(set_at_state0) = next_level_associations_for_child.get(&TokenizerStateID(0)) {
                                        // Only add targets if they are NOT yellow
                                        potential_targets.extend(set_at_state0.iter()
                                            .filter(|h| !yellow_nodes.contains(h)) // h is &ArcPtrWrapper
                                            .map(|h| h.as_arc().clone())); // h is &ArcPtrWrapper, use as_arc()
                                    }
                                }


                                inserter = inserter.try_destinations(&potential_targets);

                                // ---- START: Replacement for try_children ----
                                if inserter.clone_into_option().is_none() {
                                    let mut additional_potential_targets_from_children: Vec<Arc<Mutex<PrecomputeNode>>> = Vec::new();
                                    let source_guard = source_for_inserter_arc.lock().unwrap(); // Lock the source node
                                    // edge_key_for_inserter is Option<GrammarTokenID> (Some(grammar_token_id))
                                    if let Some(dest_map_for_current_key) = source_guard.children().get(&edge_key_for_inserter) {
                                        // dest_map_for_current_key is &BTreeMap<ArcPtrWrapper<Mutex<...>>, EV>
                                        for child_wrapper_arc in dest_map_for_current_key.keys() { // Iterate over ArcPtrWrapper keys
                                            let child_arc = child_wrapper_arc.as_arc();
                                            let child_handle = ArcPtrWrapper::new(child_arc.clone()); // Create ArcPtrWrapper for the child
                                            if !yellow_nodes.contains(&child_handle) { // Check if child is NOT yellow
                                                additional_potential_targets_from_children.push(child_arc.clone());
                                            }
                                        }
                                    }
                                    drop(source_guard); // Release lock

                                    if !additional_potential_targets_from_children.is_empty() {
                                        inserter = inserter.try_destinations(&additional_potential_targets_from_children);
                                    }
                                }
                                // ---- END: Replacement for try_children ----


                                let target_pc_node_arc;
                                if inserter.clone_into_option().is_some() {
                                    target_pc_node_arc = inserter.unwrap();
                                    // Edge was made to an existing node (or existing edge value merged)
                                } else {
                                    target_pc_node_arc = inserter.else_create_destination_with_value(PrecomputedNodeContents::default()).unwrap();
                                }

                                if match_end_offset == bytes_segment.len() {
                                    next_level_associations_for_child
                                        .entry(TokenizerStateID(0))
                                        .or_default()
                                        .insert(ArcPtrWrapper::new(target_pc_node_arc.clone())); // Wrap in ArcPtrWrapper

                                    let mut target_guard = target_pc_node_arc.lock().unwrap();
                                    // Update stats for clean_end
                                    let _ = target_guard.value.clean_end.get_or_insert_with(HybridBitset::new); // Ensure it exists
                                    let _inserted = target_guard.value.clean_end.as_mut().unwrap().insert(child_vocab_node.token_id());
                                } else {
                                    segment_processing_q.entry(match_end_offset)
                                        .or_default()
                                        .entry(TokenizerStateID(0))
                                        .or_default()
                                        .insert(ArcPtrWrapper::new(target_pc_node_arc)); // Wrap in ArcPtrWrapper
                                }
                            } // End for segment_source_pc_handle in effective_source_handles_for_suffix
                        } // End for match_info in results.matches

                        // --- Existing logic for results.end_state ---
                        if let Some(final_tokenizer_state_val) = results.end_state {
                            let final_tokenizer_state_id = TokenizerStateID(final_tokenizer_state_val);

                            for segment_source_pc_handle in &effective_source_handles_for_suffix {
                                next_level_associations_for_child
                                    .entry(final_tokenizer_state_id)
                                    .or_default()
                                    .insert(segment_source_pc_handle.clone()); // segment_source_pc_handle is &ArcPtrWrapper, clone it

                                // segment_source_pc_handle is &ArcPtrWrapper<Mutex<PrecomputeNode>>
                                // Use as_arc() to get &Arc<Mutex<PrecomputeNode>>
                                let mut segment_source_guard = segment_source_pc_handle.as_arc().lock().unwrap();
                                let possible_final_grammar_tokens = tokenizer.tokens_accessible_from_state(final_tokenizer_state_id);
                                for gtid in possible_final_grammar_tokens {
                                    segment_source_guard.value.push_finalizer_info(
                                        gtid,
                                        LLMTokenID(child_vocab_node.token_id()),
                                        final_tokenizer_state_id,
                                        // max_llm_token_id, // Removed
                                    );
                                }
                            }
                        }
                        for handle in &effective_source_handles_for_suffix {
                            yellow_nodes.remove(handle); // handle is ArcPtrWrapper, yellow_nodes contains ArcPtrWrapper
                        }
                    }
                }
                // RECURSIVE CALL INSTEAD OF PUSHING TO STACK
                precompute_recursive_helper(
                    child_vocab_node,
                    next_level_associations_for_child,
                    tokenizer,
                    all_llm_tokens_for_merge_edge,
                    pb,
                    // max_llm_token_id, // Removed
                    merge_threshold, // Use renamed parameter
                );
            }
        }
        // ---- End of recursive helper function definition ----


        // Initialize the DFS stack with the root of the VocabPrefixTree.
        let mut initial_associations_at_vocab_root = BTreeMap::new();
        for (tokenizer_id, pc_root_arc) in &precomputed_roots {
            let mut set_for_state = BTreeSet::new();
            set_for_state.insert(ArcPtrWrapper::new(pc_root_arc.clone())); // Each precomputed_root is a starting point, wrap in ArcPtrWrapper
            initial_associations_at_vocab_root.insert(*tokenizer_id, set_for_state);
        }

        crate::debug!(2, "Starting precompute DFS recursion over VocabPrefixTree"); 
        // Initial call to the recursive helper
        precompute_recursive_helper(
            &vocab_prefix_tree.root,
            initial_associations_at_vocab_root,
            tokenizer, 
            &all_llm_tokens_for_merge_edge, 
            &pb, 
            // max_llm_token_id, // Removed from helper's signature
            MERGE_THRESHOLD, 
        );
        crate::debug!(2, "Done with precompute DFS recursion."); 

        pb.finish_with_message("Precomputation complete");


        // ---- Perform final cycle check on the fully built graph ----
        crate::debug!(2, "Performing final cycle check on precomputed graph structure...");
        for (tokenizer_state_id, root_arc) in &precomputed_roots {
            if PrecomputeNode::has_any_cycle(root_arc.clone()) {
                panic!(
                    "Cycle detected in precomputed graph for tokenizer_state_id {:?} after all precomputation steps. This indicates an issue in the graph construction logic.",
                    tokenizer_state_id
                );
            }
        }
        crate::debug!(2, "Final cycle check passed: No cycles found in precomputed graph structure.");
        // -------------------------------------------------------------

        // --- Calculate Final Statistics from precomputed_roots before unwrapping ---
        crate::debug!(2, "Calculating final precompute statistics...");
        let mut all_reachable_nodes_for_final_stats: HashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>> = HashSet::new();
        for root_arc_mutex_node in precomputed_roots.values() {
            let nodes_from_this_root = PrecomputeNode::all_nodes(root_arc_mutex_node.clone());
            for node_arc in nodes_from_this_root {
                all_reachable_nodes_for_final_stats.insert(ArcPtrWrapper::new(node_arc));
            }
        }
        stats.final_unique_nodes_count = all_reachable_nodes_for_final_stats.len();

        stats.final_total_occupancy_sum_for_some_keys = 0;
        stats.final_num_occupied_some_edge_keys = 0;
        stats.final_total_occupancy_sum_for_none_keys = 0;
        stats.final_num_occupied_none_edge_keys = 0;


        for comp_arc_node in &all_reachable_nodes_for_final_stats {
            let node_arc = comp_arc_node.as_arc(); 
            let node_guard = node_arc.lock().expect("Mutex poisoned during final stats calculation");

            for (edge_key_opt, dest_map) in node_guard.children() { 
                let num_edges_for_this_key_to_distinct_children = dest_map.len();
                stats.final_edges_count += num_edges_for_this_key_to_distinct_children;

                if let Some(gtid) = edge_key_opt { 
                    stats.final_edges_with_some_key += num_edges_for_this_key_to_distinct_children;
                    *stats.final_grammar_token_edge_key_counts.entry(*gtid).or_insert(0) += 1;
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
            for finalizer_for_gtid in node_guard.value.finalizers.values() {
                stats.final_total_finalizer_entries_in_graph += finalizer_for_gtid.content.len();
            }
        }

        print_precompute_stats(&stats, token_name_map);

        let mut final_precomputed: Precomputed = BTreeMap::new();
        let mut clone_count = 0;
        for (id, arc_mutex_node) in precomputed_roots {
            match Arc::try_unwrap(arc_mutex_node) {
                Ok(mutex_node) => {
                    final_precomputed.insert(
                        id,
                        mutex_node.into_inner().expect("Mutex poisoned at end of precompute"),
                    );
                }
                Err(arc_still_owned) => {
                    clone_count += 1;
                    final_precomputed.insert(id, arc_still_owned.lock().unwrap().clone());
                }
            }
        }
        if clone_count > 0 {
            crate::debug!(
                4,
                "Warning: {} precomputed root(s) had multiple owners; cloned inner Trie for them.",
                clone_count
            );
        }
        final_precomputed
    }

    pub fn init(&self) -> GrammarConstraintState<'_> {
        let initial_token_info = LLMTokenInfo {
            active: HybridBitset::ones(self.max_llm_token_id),
            intersection: HybridBitset::ones(self.max_llm_token_id),
        };
        let mut state = BTreeMap::new();
        let initial_glr_parser_state: GLRParserState<'_, LLMTokenInfo> = self.parser.init_glr_parser_with_t(initial_token_info);
        state.insert(self.tokenizer.initial_state_id(), initial_glr_parser_state);

        GrammarConstraintState {
            parent: self,
            state,
        }
    }
}

impl<'a> GrammarConstraintState<'a> {
    pub fn get_mask(&mut self) -> LLMTokenBV {
        let mut mask = HybridBitset::new();
        for (_tokenizer_state_id, glr_parser_state) in &self.state {
            for active_state in glr_parser_state.active_states.values() {
                mask |= &active_state.stack.peek().t.active;
            }
        }
        mask
    }

    pub fn step_with_all_llm_tokens(&mut self) {
        let all_llm_tokens = HybridBitset::ones(self.parent.max_llm_token_id);
        self.step(&all_llm_tokens);
    }

    pub fn step_with_llm_token(&mut self, llm_token_id: LLMTokenID) {
        let mut llm_tokens = HybridBitset::new();
        llm_tokens.insert(llm_token_id.0);
        self.step(&llm_tokens); 
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) {
        let all_true_token_info = LLMTokenInfo {
            active: HybridBitset::ones(self.parent.max_llm_token_id),
            intersection: HybridBitset::ones(self.parent.max_llm_token_id),
        };
        let all_true_intersection = all_true_token_info.intersection.clone();

        let closure = |content: &ParseStateNodeContent<LLMTokenInfo>| -> Option<(ParseStateNodeContent<LLMTokenInfo>, bool)> {
            if content.t.active.contains(llm_token_id.0) {
                if content.t.intersection == all_true_intersection { 
                     Some((ParseStateNodeContent { state_id: content.state_id, t: all_true_token_info.clone() }, false)) 
                } else {
                     Some((ParseStateNodeContent { state_id: content.state_id, t: all_true_token_info.clone() }, true)) 
                }
            } else {
                None 
            }
        };

        let mut memo = HashMap::new();
        self.state.retain(|_tokenizer_state_id, glr_state| {
            glr_state.active_states.retain(|_key, parse_state| { 
                let maybe_new_node = prune_and_transform_recursive(&parse_state.stack, &closure, &mut memo);
                if let Some(new_node) = maybe_new_node {
                    parse_state.stack = new_node; 
                    true
                } else {
                    false
                }
            });
            !glr_state.active_states.is_empty()
        });
    }

    pub fn step_with_llm_token_sequence(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.step_with_llm_token(llm_token_id);
        }
    }

    fn prepare_initial_nodes_and_values_for_special_map(&mut self, llm_tokens: &LLMTokenBV) -> Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'a, LLMTokenInfo>)> {
        let mut initial_nodes_and_values: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'_, LLMTokenInfo>)> = Vec::new();
        let mut tokenizer_state_id_to_parse_states: BTreeMap<TokenizerStateID, GLRParserState<'_, LLMTokenInfo>> = BTreeMap::new();

        for (tokenizer_state_id, state) in &self.state {
            let mut cloned_state = state.clone(); 
            for parse_state in cloned_state.active_states.values_mut() { 
                Arc::make_mut(&mut parse_state.stack).value.t.active &= llm_tokens;
            }
            tokenizer_state_id_to_parse_states.insert(*tokenizer_state_id, cloned_state);
        }

        for (tokenizer_state_id, state) in tokenizer_state_id_to_parse_states {
            let token_trie_node = self.parent.precomputed[&tokenizer_state_id].clone();
            let token_trie_arc_mutex = Arc::new(Mutex::new(token_trie_node));
            initial_nodes_and_values.push((token_trie_arc_mutex, state));
        }
        initial_nodes_and_values
    }

    pub fn step(&mut self, llm_tokens: &LLMTokenBV) {
        crate::debug!(2, "Stepping grammar constraint state with tokenizer states {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens);

        self.state = BTreeMap::new();

        Trie::special_map(
            initial_nodes_and_values,
            |glr_parse_state, grammar_token_id, edge_llm_tokens, child_node| {
                let node_ptr = std::ptr::addr_of!(*child_node);
                crate::debug!(3, "Processing grammar node {:p} token {:?} with {} active states", node_ptr, grammar_token_id.map(|gtid| gtid.0), glr_parse_state.active_states.len());
                let mut cloned_glr_parse_state = glr_parse_state.clone();
                cloned_glr_parse_state.active_states.retain(|_key, parse_state| { 
                    let current_active_tokens = parse_state.stack.value.t.active.clone();
                    Arc::make_mut(&mut parse_state.stack).value.t.intersection &= &current_active_tokens;
                    Arc::make_mut(&mut parse_state.stack).value.t.active &= edge_llm_tokens;
                    !parse_state.stack.value.t.active.is_empty() 
                });
                grammar_token_id.map(|gtid| cloned_glr_parse_state.step(gtid));
                if cloned_glr_parse_state.active_states.is_empty() {
                    crate::debug!(3, "No active states after processing grammar token {:?}", grammar_token_id.map(|gtid| gtid.0));
                    return None;
                } else {
                    crate::debug!(3, "Processed grammar token {:?}, {} active states.", grammar_token_id.map(|gtid| gtid.0), cloned_glr_parse_state.active_states.len());
                    Some(cloned_glr_parse_state)
                }
            },
            |managed_parse_state1, managed_parse_state2| {
                managed_parse_state1.merge_with(managed_parse_state2);
            },
            |node, current_glr_parse_state| { 
                let mut active_llm_tokens = HybridBitset::new();
                for parse_state in current_glr_parse_state.active_states.values() {
                    active_llm_tokens |= &parse_state.stack.value.t.active;
                }
                let node_ptr = std::ptr::addr_of!(*node);
                crate::debug!(3, "Processing node {:p} with {} active states, {} LLM tokens, {} finalizers", node_ptr, current_glr_parse_state.active_states.len(), active_llm_tokens.len(), node.value.finalizers.len());
                if let Some(clean_end) = &node.value.clean_end {
                    let mut final_glr_parse_state = current_glr_parse_state.clone();
                    final_glr_parse_state.active_states.retain(|_key, parse_state| {
                        let current_active_tokens = parse_state.stack.value.t.active.clone();
                        Arc::make_mut(&mut parse_state.stack).value.t.intersection &= &current_active_tokens;
                        Arc::make_mut(&mut parse_state.stack).value.t.active &= clean_end;
                        !parse_state.stack.value.t.active.is_empty()
                    });
                    crate::debug!(3, "At clean end state");
                    if final_glr_parse_state.is_ok() {
                        crate::debug!(3, "GLR parse state at clean end is OK");
                        if let Some(existing) = self.state.get_mut(&TokenizerStateID(0)) {
                            existing.merge_with(final_glr_parse_state.clone());
                        } else {
                            self.state.insert(TokenizerStateID(0), final_glr_parse_state.clone());
                        }
                    }
                }

                for (possible_final_grammar_token, precomputed_finalizer) in &node.value.finalizers {
                    let mut possible_next_glr_parse_state = current_glr_parse_state.clone();
                    crate::debug!(3, "Stepping semi-final GLR parse state");
                    possible_next_glr_parse_state.step(*possible_final_grammar_token);
                    if possible_next_glr_parse_state.is_ok() {
                        crate::debug!(3, "Semi-final GLR parse state is OK");
                        for (tokenizer_state_id, llm_tokens_from_finalizer) in &precomputed_finalizer.content {
                            let mut glr_parse_state_filtered = current_glr_parse_state.clone(); // Start from current_glr_parse_state for filtering
                            glr_parse_state_filtered.active_states.retain(|_key, parse_state| {
                                let current_active_tokens = parse_state.stack.value.t.active.clone();
                                Arc::make_mut(&mut parse_state.stack).value.t.intersection &= &current_active_tokens;
                                Arc::make_mut(&mut parse_state.stack).value.t.active &= llm_tokens_from_finalizer;
                                !parse_state.stack.value.t.active.is_empty()
                            });
                            // Now, take the successfully filtered state and step it with the grammar token
                            // This ensures that the GLR state is valid *before* stepping and also compatible with finalizer tokens.
                            // However, the original logic applies the finalizer tokens to the *original* current_glr_parse_state,
                            // and then merges *that* into self.state if it's compatible.
                            // The `possible_next_glr_parse_state` is only used to check if the grammar token itself is valid.
                            // Let's stick to the original logic structure for applying finalizer tokens:
                            // Filter the `current_glr_parse_state` by `llm_tokens_from_finalizer`
                            // If this filtered state is OK, then this is a valid terminal state for the *current* GLR configuration.
                            // The `possible_next_glr_parse_state.step` is a check that the grammar *could* accept this token.
                            // The actual state to be stored is `glr_parse_state_filtered` (which is `current_glr_parse_state` filtered by `llm_tokens_from_finalizer`)
                            // but associated with the *next* tokenizer_state_id.

                            // Re-evaluating: The `glr_parse_state_filtered` should be based on `possible_next_glr_parse_state`
                            // if the finalizer applies *after* the grammar token.
                            // The current code filters `current_glr_parse_state` and if that's ok, stores it.
                            // This implies the finalizer's `llm_tokens` are alternatives for the *current* position,
                            // leading to a specific `tokenizer_state_id`.

                            // The original logic:
                            // 1. Clone `current_glr_parse_state` -> `possible_next_glr_parse_state`
                            // 2. Step `possible_next_glr_parse_state` with `possible_final_grammar_token`.
                            // 3. If `possible_next_glr_parse_state` is OK:
                            //    For each (tokenizer_state_id, llm_tokens) in finalizer:
                            //        Clone `current_glr_parse_state` -> `glr_parse_state_filtered` (this seems to be the point of confusion)
                            //        Filter `glr_parse_state_filtered.t.active` by `llm_tokens`.
                            //        If `glr_parse_state_filtered` is OK, add to `self.state[tokenizer_state_id]`.
                            // This means the `llm_tokens` from the finalizer are applied to the GLR state *before* it's stepped by `possible_final_grammar_token`.
                            // This seems correct if the finalizer represents LLM tokens that *complete* a grammar token.
                            // The `possible_next_glr_parse_state.is_ok()` check ensures the grammar token is valid from the *original* state.
                            // The state stored is the *original* state, but filtered by the finalizer's tokens, and associated with the *next* tokenizer state.

                            crate::debug!(3, "Processing finalizer for token_state_id {:?}", tokenizer_state_id);
                            if glr_parse_state_filtered.is_ok() { // This is current_glr_parse_state filtered by finalizer's llm_tokens
                                crate::debug!(3, "Finalizer is compatible with current GLR state (pre-step by final_grammar_token)");
                                if let Some(existing) = self.state.get_mut(tokenizer_state_id) {
                                    existing.merge_with(glr_parse_state_filtered.clone());
                                } else {
                                    self.state.insert(*tokenizer_state_id, glr_parse_state_filtered.clone());
                                }
                            }
                        }
                    }
                }
                !current_glr_parse_state.active_states.is_empty()
            },
        );
    }
}

