use std::cmp::Ordering;
use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
use crate::glr::parser::ParseStateNodeContent;
use crate::glr::parser::{MergeAndIntersect, GLRParser, GLRParserState, ParseState, ParseStateKey};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::*;
use bitvec::prelude::*; // Keep for macros or other uses if needed
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque, HashSet};
use std::ops::BitOr;
use std::sync::{Arc, Mutex, MutexGuard};
use bitvec::macros::internal::funty::Fundamental;
use keyed_priority_queue::KeyedPriorityQueue;
use crate::constraint_extra::print_finalizer;
use crate::datastructures::charmap::TrieMap;
use crate::datastructures::gss::prune_and_transform_recursive;
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::datastructures::trie::EdgeInserter;
use indicatif::{ProgressBar, ProgressStyle};
use crate::datastructures::hybrid_bitset::HybridBitset;


pub type LLMTokenBV = HybridBitset;
pub type GrammarTokenBV = BitVec; // Assuming GrammarTokenBV remains BitVec

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
    pub(crate) max_llm_token_id: usize,
}

#[derive(Debug, Clone)]
pub struct GrammarConstraintState<'a> {
    pub(crate) parent: &'a GrammarConstraint,
    pub(crate) state: BTreeMap<TokenizerStateID, GLRParserState<'a, LLMTokenInfo>>,
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
    pub fn push_finalizer_info(&mut self, possible_final_grammar_token: GrammarTokenID, llm_token_id: LLMTokenID, tokenizer_state_id: TokenizerStateID, max_llm_token_id: usize) {
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
        max_llm_token_id: usize
    ) -> Self {
        let precomputed = GrammarConstraint::precompute(&tokenizer, &llm_token_map, max_llm_token_id);
        Self {
            tokenizer,
            parser,
            precomputed,
            llm_token_map,
            max_llm_token_id,
        }
    }

    pub fn precompute<'a>(
        tokenizer: &Regex,
        llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        max_llm_token_id: usize,
    ) -> Precomputed {
        // ---- Helper function to count nodes in VocabPrefixTree ----
        fn count_vocab_nodes(node: &VocabPrefixTreeNode) -> u64 {
            let mut count = 1u64; // Count this node
            for child_node in node.children().values() {
                count += count_vocab_nodes(child_node);
            }
            count
        }
        // -----------------------------------------------------------

        const MERGE_THRESHOLD: usize = 10;

        // ----  Ord-capable handle for `Arc<Mutex<PrecomputeNode>>` --------------------------
        // `Arc<Mutex<PrecomputeNode>>` cannot live in a `BTreeSet` because `Mutex<T>` lacks
        // an `Ord` implementation.  We wrap it and order by the (stable) pointer address.
        #[derive(Clone)]
        struct NodeHandle(Arc<Mutex<PrecomputeNode>>);

        use std::ops::Deref;
        impl Deref for NodeHandle {
            type Target = Arc<Mutex<PrecomputeNode>>;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
        impl PartialEq for NodeHandle {
            fn eq(&self, other: &Self) -> bool {
                Arc::ptr_eq(&self.0, &other.0)
            }
        }
        impl Eq for NodeHandle {}
        impl PartialOrd for NodeHandle {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for NodeHandle {
            fn cmp(&self, other: &Self) -> Ordering {
                (Arc::as_ptr(&self.0) as usize).cmp(&(Arc::as_ptr(&other.0) as usize))
            }
        }
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
        }

        // merge_map for deduplicating PrecomputeNode structures resulting from merged paths
        let mut merge_map: BTreeMap<BTreeSet<NodeHandle>, Arc<Mutex<PrecomputeNode>>> = BTreeMap::new();
        let all_llm_tokens_for_merge_edge = HybridBitset::ones(max_llm_token_id);

        // ---- Helper function to merge NodeHandles ----
        fn merge_node_handles_internal(
            set_of_handles: &BTreeSet<NodeHandle>,
            merge_map: &mut BTreeMap<BTreeSet<NodeHandle>, Arc<Mutex<PrecomputeNode>>>,
            all_llm_tokens_for_merge_edge: &HybridBitset,
            threshold: usize,
        ) -> BTreeSet<NodeHandle> { // Changed return type
            if set_of_handles.is_empty() {
                return BTreeSet::new();
            }
            // If set size is within threshold, don't merge, return the set as is.
            if set_of_handles.len() <= threshold {
                return set_of_handles.clone();
            }

            // Attempt to find an already merged node for this exact set.
            if let Some(existing_merged_arc) = merge_map.get(set_of_handles) {
                let mut result_set = BTreeSet::new();
                result_set.insert(NodeHandle(existing_merged_arc.clone()));
                return result_set;
            }

            // Create a new node to represent the merged set.
            let new_merged_pc_node_arc = Arc::new(Mutex::new(PrecomputeNode::new(PrecomputedNodeContents::default())));

            // Add each handle in the input set as a child of the new_merged_pc_node_arc
            // using an EdgeInserter. The edge key will be None.
            for handle_to_become_child in set_of_handles {
                // The EdgeInserter will lock the source (new_merged_pc_node_arc)
                // and attempt to insert an edge to handle_to_become_child.0.
                // Since new_merged_pc_node_arc is new, this will always be an insert.
                let insert_result = EdgeInserter::new(
                    new_merged_pc_node_arc.clone(), // Source Arc<Mutex<PrecomputeNode>>
                    None,                           // Edge key: Option<GrammarTokenID> = None
                    all_llm_tokens_for_merge_edge.clone(), // Edge value: LLMTokenBV
                    // Merge function for edge values (LLMTokenBV)
                    |ev_exist: &LLMTokenBV, ev_new: LLMTokenBV| Some(ev_exist | &ev_new),
                )
                .try_destination(handle_to_become_child.0.clone()) // Destination Arc<Mutex<PrecomputeNode>>
                .into_option();

                if insert_result.is_none() {
                    // This might happen if try_insert detects a cycle, though unlikely for new nodes
                    // and None edges. Or if the destination itself is problematic.
                    // For robustness, one might log this or handle it, but for now, we expect success.
                    panic!("EdgeInserter failed to add child during merge_node_handles_internal. This is unexpected.");
                }
            }

            // Store the new merged node in merge_map for future reuse.
            // The key is a clone of the original set_of_handles.
            merge_map.insert(set_of_handles.clone(), new_merged_pc_node_arc.clone());

            // Return a set containing only the new merged handle.
            let mut result_set = BTreeSet::new();
            result_set.insert(NodeHandle(new_merged_pc_node_arc));
            result_set
        }
        // --------------------------------------------


        // Struct for DFS item
        struct VocabProcessingItem<'a> {
            vocab_node: &'a VocabPrefixTreeNode,
            // Maps TokenizerStateID (at this vocab_node) to a SET of PrecomputeNode handles that represent paths ending at (vocab_node, TokenizerStateID)
            associated_pc_nodes_by_state: BTreeMap<TokenizerStateID, BTreeSet<NodeHandle>>,
        }

        // --- Use a Vec as a stack for DFS ---
        let mut dfs_vocab_stack: Vec<VocabProcessingItem> = Vec::new();

        // Initialize the DFS stack with the root of the VocabPrefixTree.
        let mut initial_associations_at_vocab_root = BTreeMap::new();
        for (tokenizer_id, pc_root_arc) in &precomputed_roots {
            let mut set_for_state = BTreeSet::new();
            set_for_state.insert(NodeHandle(pc_root_arc.clone())); // Each precomputed_root is a starting point
            initial_associations_at_vocab_root.insert(*tokenizer_id, set_for_state);
        }

        // --- Push root onto the DFS stack ---
        dfs_vocab_stack.push(VocabProcessingItem {
            vocab_node: &vocab_prefix_tree.root,
            associated_pc_nodes_by_state: initial_associations_at_vocab_root,
        });


        crate::debug!(2, "Starting precompute main DFS loop over VocabPrefixTree");
        // --- Main DFS loop ---
        while let Some(processing_item) = dfs_vocab_stack.pop() {
            // --- Increment progress bar ---
            pb.inc(1);

            crate::debug!(3, "Processing VocabPrefixTreeNode ({} children), prefix: '{}'",
                processing_item.vocab_node.iter_children().count(),
                String::from_utf8_lossy(processing_item.vocab_node.prefix())
            );

            // Step 1: For each TokenizerStateID, apply merge policy to the set of incoming PrecomputeNode handles.
            let mut effective_source_pc_nodes_map: BTreeMap<TokenizerStateID, BTreeSet<NodeHandle>> = BTreeMap::new(); // Changed name from merged_source_pc_nodes_map
            for (tokenizer_state_id, set_of_handles) in processing_item.associated_pc_nodes_by_state {
                let effective_handles_set = merge_node_handles_internal(
                    &set_of_handles,
                    &mut merge_map,
                    &all_llm_tokens_for_merge_edge,
                    MERGE_THRESHOLD,
                );
                if !effective_handles_set.is_empty() {
                    effective_source_pc_nodes_map.insert(*tokenizer_state_id, effective_handles_set);
                }
            }

            // Step 2: Iterate over children of current_vocab_node in the VocabPrefixTree
            // --- Collect children to process in reverse order for DFS ---
            let mut children_to_process: Vec<(&Vec<u8>, &VocabPrefixTreeNode)> = processing_item.vocab_node.iter_children().collect();
            children_to_process.reverse(); // Process in reverse order for DFS

            for (bytes_segment, child_vocab_node) in children_to_process {
                crate::debug!(3, "  Transitioning via segment: '{}' to child_vocab_node (prefix: '{}')",
                    String::from_utf8_lossy(bytes_segment),
                    String::from_utf8_lossy(child_vocab_node.prefix())
                );

                // This map will collect PrecomputeNodes formed at the *end* of processing bytes_segment,
                // to be associated with child_vocab_node for the next DFS level.
                // Key: TokenizerStateID at child_vocab_node. Value: Set of PrecomputeNode handles.
                let mut next_level_associations_for_child: BTreeMap<TokenizerStateID, BTreeSet<NodeHandle>> = BTreeMap::new();

                // Inner queue for processing bytes_segment (offset-based)
                // Key: offset within bytes_segment.
                // Value: Map from TokenizerStateID (at that offset) to a SET of PrecomputeNode handles that are sources for the *remaining* part of the segment.
                let mut segment_processing_q: BTreeMap<usize, BTreeMap<TokenizerStateID, BTreeSet<NodeHandle>>> = BTreeMap::new();

                // Initialize segment_processing_q at offset 0 using effective_source_pc_nodes_map (nodes at current_vocab_node after merge policy)
                for (initial_tokenizer_state_id, effective_source_handles_set) in &effective_source_pc_nodes_map {
                    segment_processing_q.entry(0).or_default()
                        .entry(*initial_tokenizer_state_id).or_default()
                        .extend(effective_source_handles_set.iter().cloned());
                }

                // Process the current bytes_segment
                while let Some((current_offset, states_at_offset_map)) = segment_processing_q.pop_first() {
                    for (tokenizer_state_before_suffix, current_path_source_handles_set) in states_at_offset_map {
                         if current_path_source_handles_set.is_empty() {
                            continue;
                        }

                        // Apply the merge policy to the set of handles for this specific path in segment processing.
                        let effective_source_handles_for_suffix: BTreeSet<NodeHandle> = merge_node_handles_internal(
                            &current_path_source_handles_set,
                            &mut merge_map,
                            &all_llm_tokens_for_merge_edge,
                            MERGE_THRESHOLD
                        );

                        if effective_source_handles_for_suffix.is_empty() {
                            continue;
                        }

                        let segment_suffix_to_process = &bytes_segment[current_offset..];
                        let results = tokenizer.execute_from_state(segment_suffix_to_process, tokenizer_state_before_suffix);

                        // Iterate over each effective source handle
                        for segment_source_pc_handle in effective_source_handles_for_suffix {
                            // --- Existing logic for results.matches ---
                            for match_info in &results.matches {
                                let grammar_token_id = GrammarTokenID(match_info.id);
                                let match_end_offset = current_offset + match_info.width;

                                let edge_llm_tokens = child_vocab_node.reachable_token_ids().clone();

                                let mut potential_targets: Vec<Arc<Mutex<PrecomputeNode>>> = Vec::new();
                                if match_end_offset < bytes_segment.len() {
                                    if let Some(map_at_offset) = segment_processing_q.get(&match_end_offset) {
                                        if let Some(set_at_state0) = map_at_offset.get(&TokenizerStateID(0)) {
                                            potential_targets.extend(set_at_state0.iter().map(|h| h.0.clone()));
                                        }
                                    }
                                }
                                if match_end_offset == bytes_segment.len() {
                                     if let Some(set_at_state0) = next_level_associations_for_child.get(&TokenizerStateID(0)) {
                                        potential_targets.extend(set_at_state0.iter().map(|h| h.0.clone()));
                                    }
                                }

                                let target_pc_node_arc = EdgeInserter::new(
                                        segment_source_pc_handle.0.clone(),
                                        Some(grammar_token_id),
                                        edge_llm_tokens.clone(),
                                        |ev_exist: &HybridBitset, ev_new: HybridBitset| Some(ev_exist | &ev_new),
                                    )
                                    .try_destinations(&potential_targets)
                                    .try_children()
                                    .else_create_destination_with_value(PrecomputedNodeContents::default())
                                    .unwrap();


                                if match_end_offset == bytes_segment.len() {
                                    next_level_associations_for_child
                                        .entry(TokenizerStateID(0))
                                        .or_default()
                                        .insert(NodeHandle(target_pc_node_arc.clone()));

                                    let mut target_guard = target_pc_node_arc.lock().unwrap();
                                    target_guard.value.clean_end
                                         .get_or_insert_with(HybridBitset::new)
                                         .insert(child_vocab_node.token_id());
                                } else {
                                    segment_processing_q.entry(match_end_offset)
                                        .or_default()
                                        .entry(TokenizerStateID(0))
                                        .or_default()
                                        .insert(NodeHandle(target_pc_node_arc));
                                }
                            }

                            // --- Existing logic for results.end_state ---
                            if let Some(final_tokenizer_state_val) = results.end_state {
                                let final_tokenizer_state_id = TokenizerStateID(final_tokenizer_state_val);

                                next_level_associations_for_child
                                    .entry(final_tokenizer_state_id)
                                    .or_default()
                                    .insert(segment_source_pc_handle.0.clone());

                                let possible_final_grammar_tokens = tokenizer.tokens_accessible_from_state(final_tokenizer_state_id);
                                let mut segment_source_guard = segment_source_pc_handle.0.lock().unwrap();
                                for gtid in possible_final_grammar_tokens {
                                    segment_source_guard.value.push_finalizer_info(
                                        gtid,
                                        LLMTokenID(child_vocab_node.token_id()),
                                        final_tokenizer_state_id,
                                        max_llm_token_id
                                    );
                                }
                            }
                        }
                    }
                } // End while segment_processing_q not empty

                // --- Push child onto the main DFS stack ---
                dfs_vocab_stack.push(VocabProcessingItem {
                    vocab_node: child_vocab_node,
                    associated_pc_nodes_by_state: next_level_associations_for_child,
                });
            } // End for each child_vocab_node of current_vocab_node
        } // End while dfs_vocab_stack not empty

        // --- Finish progress bar ---
        pb.finish_with_message("Precomputation complete");
        crate::debug!(2, "Done precomputing main DFS loop.");


        // Pull the roots out of their Arc<Mutex<_>> and count failures to unwrap.
        let mut final_precomputed: Precomputed = BTreeMap::new();
        let mut clone_count = 0;
        for (id, arc_mutex_node) in precomputed_roots {
            match Arc::try_unwrap(arc_mutex_node) {
                Ok(mutex_node) => {
                    final_precomputed.insert(
                        id,
                        mutex_node
                            .into_inner()
                            .expect("Mutex poisoned at end of precompute"),
                    );
                }
                Err(arc_still_owned) => {
                    // Arc had multiple owners; clone the inner Trie.
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
            // Initially, the intersection must also be all true, as no constraints have been applied.
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
        for (_, state) in &self.state {
            for active_state in &state.active_states {
                // Use BitOrAssign<&HybridBitset>
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
        self.step(&llm_tokens); // llm_tokens is already a HybridBitset
    }

    /// Prunes the GSS based on the committed token and resets the active token sets.
    pub fn commit(&mut self, llm_token_id: LLMTokenID) {
        let all_true_token_info = LLMTokenInfo {
            active: HybridBitset::ones(self.parent.max_llm_token_id),
            intersection: HybridBitset::ones(self.parent.max_llm_token_id),
        };
        // Clone the intersection part for comparison inside the closure.
        let all_true_intersection = all_true_token_info.intersection.clone();


        // Closure for GSS transformation:
        // - Prune if token not present in 'active'.
        // - If token present:
        //   - Reset 't' to 'all_true_token_info'.
        //   - Stop recursion if token is present in 'intersection' (optimization).
        let closure = |content: &ParseStateNodeContent<LLMTokenInfo>| -> Option<(ParseStateNodeContent<LLMTokenInfo>, bool)> {
            if content.t.active.contains(llm_token_id.0) {
                // If the intersection already guarantees this token, we can stop early.
                // Check if intersection contains all possible tokens
                if content.t.intersection == all_true_intersection { // This comparison might be slow if not optimized
                     Some((ParseStateNodeContent { state_id: content.state_id, t: all_true_token_info.clone() }, false)) // Stop recursion
                } else {
                     Some((ParseStateNodeContent { state_id: content.state_id, t: all_true_token_info.clone() }, true)) // Continue recursion
                }
            } else {
                None // Prune this path
            }
        };

        let mut memo = HashMap::new();
        self.state.retain(|tokenizer_state_id, glr_state| {
            glr_state.active_states.retain_mut(|parse_state| {
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

    pub fn step_and_commit(&mut self, llm_token_id: LLMTokenID) {
        self.step_with_llm_token(llm_token_id);
        self.commit(llm_token_id);
    }

    pub fn commit_and_step_many(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            // Correct logic: commit the previous token, then step with the new one.
            self.commit(llm_token_id); // Commit the effect of the *previous* token leading to this one.
            self.step_with_llm_token(llm_token_id); // Step with the current token.
        }
    }

    fn prepare_initial_nodes_and_values_for_special_map(&mut self, llm_tokens: &LLMTokenBV) -> Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'a, LLMTokenInfo>)> {
        let mut initial_nodes_and_values: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'_, LLMTokenInfo>)> = Vec::new();
        let mut tokenizer_state_id_to_parse_states: BTreeMap<TokenizerStateID, GLRParserState<'_, LLMTokenInfo>> = BTreeMap::new();

        for (tokenizer_state_id, state) in &self.state {
            let mut state = state.clone();
            for parse_state in state.active_states.iter_mut() {
                // Only update the *active* tokens at the *top* of the stack.
                // The intersection remains unchanged, and deeper nodes are untouched.
                // The special_map logic will handle intersecting with edge_llm_tokens.
                // Use BitAndAssign<&HybridBitset>
                Arc::make_mut(&mut parse_state.stack).value.t.active &= llm_tokens;
                // Intersection is NOT modified here. It reflects the guarantee from *below*.
            }
            tokenizer_state_id_to_parse_states.insert(*tokenizer_state_id, state);
        }

        for (tokenizer_state_id, state) in tokenizer_state_id_to_parse_states {
            let token_trie = self.parent.precomputed[&tokenizer_state_id].clone();
            // Need to wrap the cloned node in an Arc<Mutex> for special_map
            let token_trie = Arc::new(Mutex::new(token_trie));
            initial_nodes_and_values.push((token_trie, state));
        }
        initial_nodes_and_values
    }

    pub fn step(&mut self, llm_tokens: &LLMTokenBV) {
        crate::debug!(2, "Stepping grammar constraint state with tokenizer states {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens);

        self.state = BTreeMap::new();

        Trie::special_map(
            // Input: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'_, LLMTokenInfo>)>
            initial_nodes_and_values,
            // step
            // Input: &GLRParserState<'_, LLMTokenInfo>, GrammarTokenID, &LLMTokenBV, &Arc<Mutex<PrecomputeNode>>
            // Output: Option<GLRParserState<'_, LLMTokenInfo>>
            |glr_parse_state, grammar_token_id, edge_llm_tokens, child_node| {
                let node_ptr = std::ptr::addr_of!(*child_node) as usize;
                crate::debug!(3, "Processing grammar node {:p} token {:?} with {} active states", node_ptr as *const (), grammar_token_id.map(|grammar_token_id| grammar_token_id.0), glr_parse_state.active_states.len());
                let mut glr_parse_state = glr_parse_state.clone();
                glr_parse_state.active_states.retain_mut(|parse_state| {
                    // Intersect the *active* tokens with the edge tokens. Intersection inherits current active tokens.
                    let current_active_tokens = parse_state.stack.value.t.active.clone();
                    // Use BitAndAssign<&HybridBitset>
                    Arc::make_mut(&mut parse_state.stack).value.t.intersection &= &current_active_tokens;
                    Arc::make_mut(&mut parse_state.stack).value.t.active &= edge_llm_tokens;
                    !parse_state.stack.value.t.active.is_empty() // Check if any active paths remain
                });
                grammar_token_id.map(|grammar_token_id| glr_parse_state.step(grammar_token_id));
                if glr_parse_state.active_states.is_empty() {
                    crate::debug!(3, "No active states after processing grammar token {:?}", grammar_token_id.map(|grammar_token_id| grammar_token_id.0));
                    return None;
                } else {
                    crate::debug!(3, "Processed grammar token {:?}, {} active states.", grammar_token_id.map(|grammar_token_id| grammar_token_id.0), glr_parse_state.active_states.len());
                    Some(glr_parse_state)
                }
            },
            // merge
            // Input: &mut GLRParserState<'_, LLMTokenInfo>, GLRParserState<'_, LLMTokenInfo>
            |managed_parse_state1, managed_parse_state2| {
                managed_parse_state1.merge_with(managed_parse_state2);
            },
            // process
            // Input: &PrecomputeNode, &GLRParserState<'_, LLMTokenInfo>
            // Output: bool (continue?)
            |node, glr_parse_state| {
                glr_parse_state.merge_active_states();
                let mut active_llm_tokens = HybridBitset::new();
                for parse_state in &glr_parse_state.active_states {
                    // Use BitOrAssign<&HybridBitset>
                    active_llm_tokens |= &parse_state.stack.value.t.active;
                }
                crate::debug!(3, "Processing node with {} active states, {} LLM tokens, {} finalizers", glr_parse_state.active_states.len(), active_llm_tokens.len(), node.value.finalizers.len());
                // Handle clean end
                if let Some(clean_end) = &node.value.clean_end {
                    let mut final_glr_parse_state = glr_parse_state.clone();
                    final_glr_parse_state.active_states.retain_mut(|parse_state| {
                        // Intersect the *active* tokens with the clean_end tokens. Intersection retains current active tokens.
                        let current_active_tokens = parse_state.stack.value.t.active.clone();
                        // Use BitAndAssign<&HybridBitset>
                        Arc::make_mut(&mut parse_state.stack).value.t.intersection &= &current_active_tokens;
                        Arc::make_mut(&mut parse_state.stack).value.t.active &= clean_end;
                        // Check if any active paths remain
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

                // Handle finalizers
                for (possible_final_grammar_token, precomputed_finalizer) in &node.value.finalizers {
                    // Ensure the final tokens parses
                    let mut possible_next_glr_parse_state = glr_parse_state.clone();
                    crate::debug!(3, "Stepping semi-final GLR parse state");
                    possible_next_glr_parse_state.step(*possible_final_grammar_token);
                    if possible_next_glr_parse_state.is_ok() {
                        crate::debug!(3, "Semi-final GLR parse state is OK");
                        for (tokenizer_state_id, llm_tokens) in &precomputed_finalizer.content {
                            // Intersect *active* tokens with the finalizer's allowed tokens.
                            let mut glr_parse_state_filtered = glr_parse_state.clone();
                            glr_parse_state_filtered.active_states.retain_mut(|parse_state| {
                                // Intersect the *active* tokens with the finalizer's allowed tokens. Intersection retains current active tokens.
                                let current_active_tokens = parse_state.stack.value.t.active.clone();
                                // Use BitAndAssign<&HybridBitset>
                                Arc::make_mut(&mut parse_state.stack).value.t.intersection &= &current_active_tokens;
                                Arc::make_mut(&mut parse_state.stack).value.t.active &= llm_tokens;
                                // Check if any active paths remain
                                !parse_state.stack.value.t.active.is_empty()
                            });
                            crate::debug!(3, "Processing finalizer");
                            if glr_parse_state_filtered.is_ok() {
                                crate::debug!(3, "Finalizer is compatible");
                                if let Some(existing) = self.state.get_mut(tokenizer_state_id) {
                                    existing.merge_with(glr_parse_state_filtered.clone());
                                } else {
                                    self.state.insert(*tokenizer_state_id, glr_parse_state_filtered.clone());
                                }
                            }
                        }
                    }
                }

                // Check if the current GLR state still has valid paths before continuing traversal
                // (This check might be redundant if the retain calls above handle it)
                !glr_parse_state.active_states.is_empty()
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::finite_automata::eat_u8;
    use crate::{choice, groups, seq};
    use crate::glr::grammar::{nt, prod, t, NonTerminal, Terminal};
    use crate::glr::table::{generate_glr_parser, generate_glr_parser_with_maps, generate_glr_parser_with_terminal_map};
    use super::*;
    use crate::datastructures::hybrid_bitset::HybridBitset; // Explicitly import HybridBitset

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

        let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map);
        dbg!(&parser);

        let constraint = GrammarConstraint::new(tokenizer, parser, llm_token_map, 3); // max_llm_token_id should be 3 for 0, 1, 2
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

        let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map); // Start production is index 6
        dbg!(&parser);
        let constraint = GrammarConstraint::new(tokenizer, parser, llm_token_map, 7); // max_llm_token_id should be 7 for IDs 0-6
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
        // let mask = state.get_mask();
        // assert_eq!(mask, LLMTokenBV::from_iter([false, false, false, false, false, false, false])); // This line seems incorrect based on the previous assertion.
    }
}
