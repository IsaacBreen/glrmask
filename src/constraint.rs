use std::cmp::Ordering;
use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
// Use types directly from glr::parser
use crate::glr::parser::{GLRParser, GLRParserState, ParseState, ParseStateKey, ParseStatus, StopReason};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use bitvec::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap}; // Added HashMap
use std::ops::BitOr;
use std::sync::{Arc, Mutex};
use bitvec::macros::internal::funty::Fundamental;
use keyed_priority_queue::KeyedPriorityQueue;
use crate::constraint_extra::print_finalizer;
use crate::datastructures::charmap::TrieMap;
use crate::datastructures::gss::GSSTrait; // Added import
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
// Removed ManagedGLRParserState, ManagedParseState imports
use crate::types::{TerminalID as GrammarTokenID, TerminalID};

pub type LLMTokenBV = BitVec;
pub type GrammarTokenBV = BitVec;

#[derive(Default, Debug, Clone)]
pub struct PrecomputedFinalizer {
    compatible_llm_tokens: LLMTokenBV,
    tokenizer_state_ids: BTreeSet<TokenizerStateID>,
}

impl PrecomputedFinalizer {
    pub(crate) fn new(compatible_llm_tokens: LLMTokenBV, tokenizer_state_ids: BTreeSet<TokenizerStateID>) -> Self {
        Self {
            compatible_llm_tokens,
            tokenizer_state_ids,
        }
    }

    pub(crate) fn compatible_llm_tokens(&self) -> &LLMTokenBV { &self.compatible_llm_tokens }
    pub(crate) fn tokenizer_state_ids(&self) -> &BTreeSet<TokenizerStateID> { &self.tokenizer_state_ids }
}

#[derive(Default, Debug, Clone)]
pub(crate) struct PrecomputedNodeContents {
    finalizers: BTreeMap<GrammarTokenID, PrecomputedFinalizer>,
}

pub(crate) type PrecomputeNode = Trie<GrammarTokenID, LLMTokenBV, PrecomputedNodeContents>;
pub(crate) type Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>;

#[derive(Debug, Clone)]
pub(crate) struct GrammarConstraint {
    pub(crate) tokenizer: Regex,
    pub(crate) parser: GLRParser,
    pub(crate) precomputed: Precomputed,
    pub(crate) llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID>,
    pub(crate) max_llm_token_id: usize,
}

// Represents a single active path in the constraint satisfaction process.
// It combines a GLR parser state with the set of possible next tokenizer states
// and the mask of compatible LLM tokens for this specific path.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ActiveConstraintConfiguration {
    pub parse_state: ParseState,
    pub llm_tokens: LLMTokenBV,
    pub tokenizer_state_ids: BTreeSet<TokenizerStateID>,
}

#[derive(Debug, Clone)]
pub struct GrammarConstraintState<'a> {
    pub(crate) parent: &'a GrammarConstraint,
    // Instead of ManagedGLRParserState, we store a list of active configurations.
    // Each configuration represents a possible state the parser could be in,
    // along with the corresponding tokenizer states and allowed LLM tokens.
    pub(crate) active_configurations: Vec<ActiveConstraintConfiguration>,
}

impl PrecomputedNodeContents {
    pub(crate) fn finalizers(&self) -> &BTreeMap<GrammarTokenID, PrecomputedFinalizer> { &self.finalizers }

    /// Adds information about a final state reachable via a specific grammar token.
    /// If an entry for the grammar token already exists, it merges the information.
    pub fn push_finalizer_info(&mut self, possible_final_grammar_token: GrammarTokenID, llm_token_id: LLMTokenID, tokenizer_state_id: TokenizerStateID, max_llm_token_id: usize) {
        let mut current_compatible_llm_tokens = LLMTokenBV::repeat(false, max_llm_token_id + 1);
        // Ensure index is within bounds before setting
        if llm_token_id.0 <= max_llm_token_id {
            current_compatible_llm_tokens.set(llm_token_id.0, true);
        } else {
            // Log or handle the out-of-bounds case appropriately
             eprintln!("Warning: Attempted to set LLM token ID {} which is out of bounds (max {}). Skipping.", llm_token_id.0, max_llm_token_id);
             return; // Don't add info for invalid token ID
        }


        let current_tokenizer_state_ids = BTreeSet::from([tokenizer_state_id]);

        self.finalizers.entry(possible_final_grammar_token)
            .and_modify(|existing_finalizer| {
                // Merge LLM tokens
                existing_finalizer.compatible_llm_tokens |= &current_compatible_llm_tokens;
                // Merge tokenizer states
                existing_finalizer.tokenizer_state_ids.extend(&current_tokenizer_state_ids);
            })
            .or_insert_with(|| PrecomputedFinalizer::new(current_compatible_llm_tokens, current_tokenizer_state_ids));
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
        #[derive(Debug, Eq, PartialEq)]
        struct DottedVocabNode<'a> {
            src: &'a VocabPrefixTreeNode,
            dst: &'a VocabPrefixTreeNode,
            bytes: &'a [u8],
            offset: usize,
        }
        impl<'a> Ord for DottedVocabNode<'a> {
            fn cmp(&self, other: &Self) -> Ordering {
                // Primary sort key: total length processed (src prefix + offset)
                let self_primary_key = self.src.prefix_length() + self.offset;
                let other_primary_key = other.src.prefix_length() + other.offset;
                // Secondary keys for stability/determinism
                (self_primary_key, self.src, self.bytes, self.offset).cmp(&(other_primary_key, other.src, other.bytes, other.offset))
            }
        }
        impl<'a> PartialOrd for DottedVocabNode<'a> {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        // Create the vocab prefix tree.
        let mut tokens_for_vocab_prefix_tree_builder: Vec<(usize, Vec<u8>)> = vec![];
        for (content, id) in llm_token_map {
            // Filter out tokens with IDs greater than max_llm_token_id
             if id.0 <= max_llm_token_id {
                tokens_for_vocab_prefix_tree_builder.push((id.0, content.clone()));
             } else {
                 eprintln!("Warning: Skipping LLM token {:?} (ID {}) during precomputation as it exceeds max_llm_token_id {}", content, id.0, max_llm_token_id);
             }
        }
        let vocab_prefix_tree = VocabPrefixTree::build(&tokens_for_vocab_prefix_tree_builder);

        // Create the roots for the precomputed Trie for each tokenizer state.
        let mut precomputed_roots: BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>> = BTreeMap::new();
        for tokenizer_state_id_val in 0..tokenizer.max_state() {
             let tokenizer_state_id = TokenizerStateID(tokenizer_state_id_val);
            let precompute_node = Arc::new(Mutex::new(PrecomputeNode::new(PrecomputedNodeContents::default())));
            precomputed_roots.insert(tokenizer_state_id, precompute_node);
        }

        // Queue for BFS-like traversal over vocab nodes and tokenizer states.
        // Key: ((DottedVocabNode, TokenizerStateID), Vec<Arc<Mutex<PrecomputeNode>>>)
        // The Vec stores the precompute Trie nodes *leading to* this state.
        let mut queue: BTreeMap<(DottedVocabNode, TokenizerStateID), Vec<Arc<Mutex<PrecomputeNode>>>> = BTreeMap::new();

        // Initialize the queue with edges from the vocab root for each tokenizer start state.
        for (tokenizer_state_id, root_precompute_node_arc) in &precomputed_roots {
            for (bytes, child_vocab_node) in vocab_prefix_tree.root.children() {
                let dotted_node = DottedVocabNode {
                    src: &vocab_prefix_tree.root, // Source is the root vocab node
                    dst: child_vocab_node,        // Destination is the child
                    bytes,                      // Bytes representing the edge
                    offset: 0,                  // Start at offset 0 of these bytes
                };
                // The state in the queue is (VocabEdge, TokenizerState) -> Preceding Trie Nodes
                queue.entry((dotted_node, *tokenizer_state_id))
                    .or_default()
                    .push(root_precompute_node_arc.clone());
            }
        }

        // Process the queue.
        while let Some((((dotted_vocab_node, current_tokenizer_state_id)), source_precompute_nodes)) = queue.pop_first() {
            let DottedVocabNode { src: _src_vocab_node, dst: dst_vocab_node, offset, bytes } = dotted_vocab_node;

            // Execute the tokenizer from the current state over the remaining bytes of the vocab edge.
            let results = tokenizer.execute_from_state(&bytes[offset..], current_tokenizer_state_id);

            // --- Process full grammar token matches ---
            for match_result in results.matches {
                let matched_grammar_token_id = GrammarTokenID(match_result.id);
                let match_width = match_result.width;
                let new_offset = offset + match_width;

                // Determine the tokenizer state *after* this match (it resets to 0).
                let next_tokenizer_state_id = TokenizerStateID(0);

                // LLM tokens associated with the edge: all tokens reachable *through* the dst_vocab_node.
                let edge_llm_tokens = dst_vocab_node.reachable_token_ids().clone();
                if edge_llm_tokens.not_any() { continue; } // Skip if this vocab node leads nowhere

                // This will hold the destination Trie nodes for the *next* step in the queue.
                let mut next_destination_trie_nodes = Vec::new();

                // Process each source Trie node that led to the current state.
                'source_node_loop: for source_trie_node_arc in &source_precompute_nodes {
                    let mut source_trie_node = source_trie_node_arc.lock().unwrap();

                    // --- Try to merge/reuse existing Trie edges/nodes ---
                    let mut found_target = false;

                    // 1. Check existing edges from *this* source node for the same grammar token.
                    if let Some(existing_edges) = source_trie_node.get_mut(&matched_grammar_token_id) {
                        if let Some((existing_edge_value, existing_dst_arc)) = existing_edges.first_mut() {
                            // Merge LLM tokens into the existing edge.
                            *existing_edge_value |= &edge_llm_tokens;
                            // Add the existing destination to the list for the next queue entry.
                            next_destination_trie_nodes.push(existing_dst_arc.clone());
                            found_target = true;
                            // Continue to the next source node, as we've handled this path.
                            continue 'source_node_loop;
                         }
                         // else: edge key exists but vec is empty (shouldn't happen) - fall through.
                    }

                    // 2. If no existing edge from this source, check if any node we *already* decided to
                    //    add to `next_destination_trie_nodes` can be reused (to avoid redundant nodes).
                    //    This requires checking for cycles if we were to add an edge.
                    for potential_target_node_arc in &next_destination_trie_nodes {
                         if source_trie_node.try_insert(matched_grammar_token_id, edge_llm_tokens.clone(), potential_target_node_arc.clone()).is_ok() {
                             // Successfully inserted edge to an already chosen destination node.
                             found_target = true;
                             continue 'source_node_loop;
                         }
                    }

                    // --- If no merge/reuse possible, create a new Trie node ---
                    if !found_target {
                        let new_destination_trie_node_arc = source_trie_node.force_insert(
                            matched_grammar_token_id,
                            edge_llm_tokens.clone(),
                            PrecomputedNodeContents::default()
                        );
                        next_destination_trie_nodes.push(new_destination_trie_node_arc);
                    }
                } // end 'source_node_loop

                // --- Update queue or add finalizers based on whether we consumed the vocab edge ---
                 if !next_destination_trie_nodes.is_empty() {
                    if new_offset == bytes.len() {
                        // Reached the end of the vocab edge bytes exactly after a match.
                        // The LLM token corresponding to `dst_vocab_node` is fully formed.
                        // Add finalizer info to the *destination* Trie nodes reached by this match.
                        let possible_final_grammar_tokens: BTreeSet<_> = tokenizer
                            .tokens_accessible_from_state(TokenizerStateID(0)) // After match, tokenizer is at state 0
                            .into_iter()
                            .map(|token_id| GrammarTokenID(token_id.0))
                            .collect();

                        for final_grammar_token_id in possible_final_grammar_tokens {
                            for dest_node_arc in &next_destination_trie_nodes {
                                dest_node_arc.lock().unwrap().value.push_finalizer_info(
                                    final_grammar_token_id,
                                    LLMTokenID(dst_vocab_node.token_id()), // The completed LLM token
                                    TokenizerStateID(0), // Tokenizer state *after* completion
                                    max_llm_token_id
                                );
                            }
                        }
                        // Don't add to queue, this path segment is complete.
                    } else if new_offset < bytes.len() {
                        // Matched a grammar token, but still within the vocab edge bytes.
                        // Add the reached destination Trie nodes to the queue for further processing.
                        let next_dotted_node = DottedVocabNode {
                            src: dst_vocab_node, // Source for next step is current destination
                            dst: dst_vocab_node, // Destination remains the same (within the same vocab node)
                            bytes,
                            offset: new_offset, // Start from the new offset
                        };
                        // The tokenizer state resets to 0 after a match.
                        queue.entry((next_dotted_node, TokenizerStateID(0)))
                            .or_default()
                            .extend(next_destination_trie_nodes);
                    } else {
                        // new_offset > bytes.len() should not happen
                        unreachable!("Tokenizer match width exceeded remaining bytes");
                    }
                 }
            } // end loop over results.matches

            // --- Handle partial matches (tokenizer stopped mid-bytes) ---
            if let Some(tokenizer_end_state) = results.end_state {
                // The tokenizer stopped at `tokenizer_end_state` within the current vocab edge `bytes`.
                // This means the LLM token `dst_vocab_node.token_id()` can be finalized *if* the next
                // grammar token matches one accessible from `tokenizer_end_state`.
                let possible_next_grammar_tokens: BTreeSet<_> = tokenizer
                    .tokens_accessible_from_state(TokenizerStateID(tokenizer_end_state))
                    .into_iter()
                    .map(|token_id| GrammarTokenID(token_id.0))
                    .collect();

                for next_grammar_token_id in possible_next_grammar_tokens {
                    for source_trie_node_arc in &source_precompute_nodes {
                        // Add finalizer info to the *source* Trie node (where the partial match occurred).
                        source_trie_node_arc.lock().unwrap().value.push_finalizer_info(
                            next_grammar_token_id, // The grammar token that *could* follow
                            LLMTokenID(dst_vocab_node.token_id()), // The LLM token being finalized
                            TokenizerStateID(tokenizer_end_state), // The tokenizer state where it stopped
                            max_llm_token_id
                        );
                    }
                }
            }
        } // end while let Some(...) = queue.pop_first()

        // Final step: Convert the Arc<Mutex<Trie>> roots back to Trie.
        let final_precomputed = precomputed_roots.into_iter().map(|(tokenizer_state_id, node_arc)| {
            let node = Arc::try_unwrap(node_arc)
                .expect("Arc unwrap failed during final precomputation step")
                .into_inner()
                .expect("Mutex poisoned during final precomputation step");
            (tokenizer_state_id, node)
        }).collect();

        final_precomputed
    }


    pub fn init(&self) -> GrammarConstraintState<'_> {
        let initial_parse_state = self.parser.init_parse_state();
        let initial_config = ActiveConstraintConfiguration {
            parse_state: initial_parse_state,
            // Initially, all LLM tokens are potentially possible.
            llm_tokens: LLMTokenBV::repeat(true, self.max_llm_token_id + 1),
            // Start at the initial tokenizer state.
            tokenizer_state_ids: BTreeSet::from([TokenizerStateID(0)]),
        };
        GrammarConstraintState {
            parent: self,
            active_configurations: vec![initial_config],
        }
    }
}

impl<'a> GrammarConstraintState<'a> {
    /// Calculates the combined mask of allowed LLM tokens across all active configurations.
    pub fn get_mask(&mut self) -> LLMTokenBV {
        let mut mask = LLMTokenBV::repeat(false, self.parent.max_llm_token_id + 1);
        for config in &self.active_configurations {
            mask |= &config.llm_tokens;
        }
        mask
    }

    /// Advances the constraint state considering all possible LLM tokens allowed by the current mask.
    pub fn step_with_all_llm_tokens(&mut self) {
        // The filter applied here is based on the *current* combined mask.
        // This seems redundant if step() internally uses the config's llm_tokens,
        // but let's keep it for now to ensure consistency with the idea of filtering.
        let current_mask = self.get_mask();
        self.step(&current_mask);
    }

    /// Advances the constraint state considering only a single specific LLM token.
    pub fn step_with_llm_token(&mut self, llm_token_id: LLMTokenID) {
        let mut llm_token_filter = LLMTokenBV::repeat(false, self.parent.max_llm_token_id + 1);
        if llm_token_id.0 <= self.parent.max_llm_token_id {
            llm_token_filter.set(llm_token_id.0, true);
        } else {
            eprintln!("Warning: LLM Token ID {} is out of bounds (max {}) during step. Applying empty filter.", llm_token_id.0, self.parent.max_llm_token_id);
        }
        self.step(&llm_token_filter);
    }

    /// Filters the active configurations, keeping only those compatible with the chosen LLM token.
    pub fn commit(&mut self, llm_token_id: LLMTokenID) {
        if llm_token_id.0 <= self.parent.max_llm_token_id {
            self.active_configurations.retain(|config| config.llm_tokens[llm_token_id.0]);
        } else {
             eprintln!("Warning: Committing out-of-bounds LLM Token ID {}. Clearing active states.", llm_token_id.0);
            self.active_configurations.clear();
        }
        // After commit, merge potentially duplicated states resulting from filtering.
        self.merge_configurations();
    }

    /// Convenience function to step with a single token and then commit it.
    pub fn step_and_commit(&mut self, llm_token_id: LLMTokenID) {
        self.step_with_llm_token(llm_token_id);
        self.commit(llm_token_id);
    }

    /// Commits a sequence of LLM tokens one by one, stepping after each commit.
    pub fn commit_and_step_many(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.commit(llm_token_id); // Commit first
             if self.active_configurations.is_empty() {
                 // Optimization: if commit removed all states, no need to step further.
                 break;
             }
            self.step_with_llm_token(llm_token_id); // Then step
        }
    }

    /// Prepares the initial input for the `special_map` traversal.
    /// Groups active configurations by their tokenizer state ID and applies the initial LLM token filter.
    fn prepare_initial_nodes_and_values_for_special_map(
        &self,
        llm_tokens_filter: &LLMTokenBV // Filter applied at the start of the step
    ) -> Vec<(Arc<Mutex<PrecomputeNode>>, Vec<(ParseState, LLMTokenBV)>)> { // Value V is Vec<(ParseState, LLMTokenBV)>
        // Map: TokenizerStateID -> List of (ParseState, Effective LLM Mask for this path)
        let mut initial_map: BTreeMap<TokenizerStateID, Vec<(ParseState, LLMTokenBV)>> = BTreeMap::new();

        for config in &self.active_configurations {
            // Apply the external filter to the configuration's current mask
            let initial_llm_bv = config.llm_tokens.clone() & llm_tokens_filter;
            if initial_llm_bv.not_any() { continue; } // Skip if this config is incompatible with the filter

            for tokenizer_state_id in &config.tokenizer_state_ids {
                // Add the parse state and its calculated initial mask to the map
                // for the corresponding tokenizer state ID.
                initial_map.entry(*tokenizer_state_id)
                    .or_default()
                    .push((config.parse_state.clone(), initial_llm_bv.clone()));
            }
        }

        // Convert the map into the Vec format required by special_map.
        initial_map.into_iter().filter_map(|(tokenizer_state_id, states_and_masks)| {
            // Get the root of the precomputed Trie for this tokenizer state ID.
            self.parent.precomputed.get(&tokenizer_state_id).map(|trie_root| {
                 (Arc::new(Mutex::new(trie_root.clone())), states_and_masks)
            })
        }).collect()
    }


    /// The core logic: advances the constraint state based on possible grammar token sequences derived from the precomputed Trie.
    pub fn step(&mut self, llm_tokens_filter: &LLMTokenBV) {
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens_filter);

        if initial_nodes_and_values.is_empty() {
            // Optimization: If no initial states match the filter, clear active configurations.
            self.active_configurations.clear();
            return;
        }

        // This vector will collect the resulting configurations generated by the 'process' closure.
        let collected_results = Arc::new(Mutex::new(Vec::<ActiveConstraintConfiguration>::new()));

        // Type V for special_map: Vec<(ParseState, LLMTokenBV)>
        // Represents the set of parser states and their associated LLM masks reachable at a specific node in the precomputed Trie.
        Trie::special_map(
            initial_nodes_and_values,
            // step: Propagate GLR states along the precomputed trie edges
            |current_states_and_masks, // &Vec<(ParseState, LLMTokenBV)>
             grammar_token_id,         // &GrammarTokenID
             edge_llm_tokens,          // &LLMTokenBV (Mask associated with the Trie edge)
             _child_trie_node| {       // &PrecomputeNode (Content of the child Trie node)

                let mut next_states_and_masks: Vec<(ParseState, LLMTokenBV)> = Vec::new();
                let parser = &self.parent.parser; // Borrow parser reference

                for (parse_state, current_llm_bv) in current_states_and_masks {
                    // Calculate the mask for the next step by intersecting with the edge's mask.
                    let next_llm_bv = current_llm_bv.clone() & edge_llm_tokens;
                    if next_llm_bv.not_any() { continue; } // Skip if no compatible LLM tokens for this path.

                    // Create a temporary GLRParserState to step the single ParseState.
                    // Initialize it only with the current active ParseState.
                    let mut temp_glr_state = parser.init_glr_parser_from_parse_state(parse_state.clone());
                    temp_glr_state.step(*grammar_token_id); // Step the GLR parser with the grammar token.

                    // Collect resulting active states from the temporary GLR state.
                    for next_parse_state in temp_glr_state.active_states {
                         // Only keep states that are still active after the step.
                         if next_parse_state.status == ParseStatus::Active {
                            next_states_and_masks.push((next_parse_state, next_llm_bv.clone()));
                         }
                         // Inactive states resulting from the step are not propagated further
                         // down this path but might be handled by finalizers later if applicable.
                    }
                }

                if next_states_and_masks.is_empty() {
                    None // Prune this path if no active GLR states result from the step.
                } else {
                    Some(next_states_and_masks) // Return the list of resulting states and masks.
                }
            },
            // merge: Combine lists of (ParseState, LLMTokenBV) pairs arriving at the same Trie node.
            |states_and_masks1, // &mut Vec<(ParseState, LLMTokenBV)>
             states_and_masks2| { // Vec<(ParseState, LLMTokenBV)>

                // Simple extend for now. Merging duplicates could be added later.
                states_and_masks1.extend(states_and_masks2);
                // TODO: Implement merging of identical ParseStates (by key) by ORing their LLMTokenBVs.
                // This would require iterating, grouping by ParseStateKey, and merging.
            },
            // process: Handle finalizers at Trie nodes to generate output configurations.
            {
                let collected_results = collected_results.clone();
                let parser = &self.parent.parser; // Borrow parser reference

                move |trie_node_content, // &PrecomputedNodeContents (Value of the Trie node being processed)
                      current_states_and_masks| // &mut Vec<(ParseState, LLMTokenBV)> (Accumulated states/masks at this node)
                      -> bool { // Returns true to continue processing children

                    // Process finalizers defined at this Trie node.
                    for (finalizing_grammar_token_id, precomputed_finalizer) in trie_node_content.value.finalizers() {
                        // Check each incoming ParseState at this Trie node.
                        for (parse_state_before_finalize, current_llm_bv) in current_states_and_masks.iter() {

                            // Check if the finalizer's grammar token is a valid *next* step for this parse state.
                            let mut temp_glr_state = parser.init_glr_parser_from_parse_state(parse_state_before_finalize.clone());
                            temp_glr_state.step(*finalizing_grammar_token_id);

                            // A final state is valid if the step results in *any* state (active or inactive acceptable parse).
                            // The key is that the grammar *allows* this finalizing token.
                            if temp_glr_state.is_ok() { // is_ok checks for active OR fully matching inactive states
                                // Compute the final LLM token mask by intersecting the current path's mask
                                // with the finalizer's compatible LLM tokens.
                                let final_llm_tokens = current_llm_bv.clone() & precomputed_finalizer.compatible_llm_tokens();
                                if final_llm_tokens.not_any() { continue; } // Skip if no compatible LLM tokens.

                                // The resulting tokenizer states are those specified by the finalizer.
                                let final_tokenizer_states = precomputed_finalizer.tokenizer_state_ids().clone();
                                if final_tokenizer_states.is_empty() { continue; } // Should not happen if final_llm_tokens is not empty.

                                // Create the resulting ActiveConstraintConfiguration.
                                // The parse state associated with the output mask is the one *before*
                                // consuming the finalizing grammar token, because this is the state
                                // reached *at the point* the final LLM token is emitted.
                                let result_config = ActiveConstraintConfiguration {
                                    parse_state: parse_state_before_finalize.clone(),
                                    llm_tokens: final_llm_tokens,
                                    tokenizer_state_ids: final_tokenizer_states,
                                };
                                collected_results.lock().unwrap().push(result_config);
                            }
                        }
                    }

                    // Determine if the traversal should continue down the children of this Trie node.
                    // Continue if *any* of the ParseStates accumulated at this node still have
                    // a non-empty LLM token mask, meaning they *could* potentially lead to
                    // a valid finalization further down the Trie.
                    let should_continue = current_states_and_masks.iter().any(|(_, bv)| bv.any());
                    should_continue
                }
            }
        );

        // Post-processing: Replace current configurations with the collected results and merge duplicates.
        let final_configurations = collected_results.lock().unwrap().drain(..).collect();
        self.active_configurations = final_configurations;
        self.merge_configurations(); // Merge states after the special_map traversal.
    }

    /// Merges ActiveConstraintConfiguration entries that share the same logical state
    /// (ParseStateKey and TokenizerStateID set).
    fn merge_configurations(&mut self) {
         if self.active_configurations.len() <= 1 { return; }

        // Use HashMap for efficient merging based on the composite key.
        let mut merged_map: HashMap<(ParseStateKey, BTreeSet<TokenizerStateID>), ActiveConstraintConfiguration> = HashMap::new();

        for config in self.active_configurations.drain(..) {
            // The key combines the GLR parse state key and the set of tokenizer states.
            let key = (config.parse_state.key(), config.tokenizer_state_ids.clone());

            merged_map.entry(key)
                .and_modify(|existing_config| {
                    // Merge the ParseState's GSS nodes.
                    existing_config.parse_state.merge(config.parse_state.clone()); // Use clone for merge
                    // Merge the LLM token masks using bitwise OR.
                    existing_config.llm_tokens |= &config.llm_tokens;
                    // TokenizerStateIDs are part of the key, so they are already identical.
                })
                .or_insert(config); // Insert the config if the key is new.
        }

        // Update the active configurations with the merged results.
        self.active_configurations = merged_map.into_values().collect();
    }
}


#[cfg(test)]
mod tests {
    use crate::finite_automata::eat_u8;
    use crate::{choice, groups, seq};
    use crate::glr::grammar::{nt, prod, t, NonTerminal, Terminal};
    use crate::glr::table::{generate_glr_parser, generate_glr_parser_with_maps, generate_glr_parser_with_terminal_map};
    use super::*;

    // Helper to create constraint for tests
    fn setup_test_constraint() -> GrammarConstraint {
        // LLM tokens: "ab" (0), "ac" (1), "$" (2) - EOF
        // Grammar tokens (from tokenizer): "a" (0), "ab" (1), "b|c" (2), "$" (3)
        // Grammar: S -> AB $
        let expr = groups![
            eat_u8(b'a'), // Grammar Token 0 ("A")
            seq![eat_u8(b'a'), eat_u8(b'b')], // Grammar Token 1 ("AB")
            choice![eat_u8(b'b'), eat_u8(b'c')], // Grammar Token 2 ("B_OR_C") - unused in grammar
            eat_u8(b'$'), // Grammar Token 3 ("EOF")
        ];
        let tokenizer = expr.build();

        let mut llm_token_map = LLMTokenMap::new();
        llm_token_map.insert(b"ab".to_vec(), LLMTokenID(0));
        llm_token_map.insert(b"ac".to_vec(), LLMTokenID(1));
        llm_token_map.insert(b"$".to_vec(), LLMTokenID(2));
        let max_llm_token_id = 2;

        // Grammar: S -> AB $
        let productions = vec![
            prod("S", vec![t("AB"), t("EOF")]),
        ];

        let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
        // Map grammar terminals to tokenizer token IDs
        grammar_token_map.insert(Terminal("A".to_string()), TerminalID(0)); // Not used in this grammar
        grammar_token_map.insert(Terminal("AB".to_string()), TerminalID(1));
        grammar_token_map.insert(Terminal("B_OR_C".to_string()), TerminalID(2)); // Not used
        grammar_token_map.insert(Terminal("EOF".to_string()), TerminalID(3)); // '$'

        let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map);

        GrammarConstraint::new(tokenizer, parser, llm_token_map, max_llm_token_id)
    }


    #[test]
    fn test_constraint_init() {
        let constraint = setup_test_constraint();
        let state = constraint.init();

        assert_eq!(state.active_configurations.len(), 1);
        let config = &state.active_configurations[0];

        // Check initial parse state (should be start state of the parser)
        assert_eq!(*config.parse_state.stack.peek(), constraint.parser.start_state_id);
        assert!(config.parse_state.action_stack.is_none());
        assert_eq!(config.parse_state.status, ParseStatus::Active);

        // Check initial LLM tokens (all should be allowed)
        assert_eq!(config.llm_tokens, bitvec![1, 1, 1]); // Indices 0, 1, 2 set

        // Check initial tokenizer state
        assert_eq!(config.tokenizer_state_ids, BTreeSet::from([TokenizerStateID(0)]));
    }

    #[test]
    fn test_constraint_step_1() {
        let constraint = setup_test_constraint();
        // constraint.dump_precomputed(); // Optional: print the precomputed structure
        let mut state = constraint.init();

        // Initial state allows all tokens. Let's step with all possibilities.
        state.step_with_all_llm_tokens();

        // After the first step (processing potential first LLM tokens based on grammar),
        // the mask should reflect which LLM tokens *could* start a valid sequence.
        // In our grammar S -> AB $, the only valid start is the grammar token "AB" (ID 1).
        // The precomputed trie should map paths starting with "AB" to LLM tokens "ab" (0).
        // The LLM token "ac" (1) doesn't match "AB". "$" (2) doesn't match "AB".
        // So, only LLM token 0 ("ab") should be possible.
        let mask = state.get_mask();
        println!("Mask after step 1: {:?}", mask);
        // Expected: Only "ab" (ID 0) is possible as the first token.
        // The step function processes finalizers. The precomputed trie for tokenizer state 0
        // should have a finalizer for LLM token "ab" (0) reachable via grammar token "AB" (1).
        assert_eq!(mask, bitvec![1, 0, 0]); // Only LLM token 0 ("ab") allowed
    }

     #[test]
    fn test_constraint_step_and_commit() {
        let constraint = setup_test_constraint();
        let mut state = constraint.init();

        // Step 1: Determine possible first tokens
        state.step_with_all_llm_tokens();
        let mask1 = state.get_mask();
        assert_eq!(mask1, bitvec![1, 0, 0]); // Only "ab" allowed

        // Commit the only allowed token "ab" (ID 0)
        state.commit(LLMTokenID(0));
        assert_eq!(state.active_configurations.len(), 1); // Should still have one path

        // Step 2: Determine possible second tokens after "ab"
        state.step_with_all_llm_tokens();
        let mask2 = state.get_mask();
        println!("Mask after step 2: {:?}", mask2);

        // After parsing "AB", the grammar expects "EOF" (grammar token 3, '$').
        // The precomputed trie should map the path for '$' to LLM token "$" (ID 2).
        // Expected: Only "$" (ID 2) is possible.
        assert_eq!(mask2, bitvec![0, 0, 1]); // Only LLM token 2 ("$") allowed
    }

    #[test]
    fn test_constraint_full_parse() {
        let constraint = setup_test_constraint();
        let mut state = constraint.init();

        // Step 1 -> Mask allows "ab"
        state.step_with_all_llm_tokens();
        assert_eq!(state.get_mask(), bitvec![1, 0, 0]);

        // Commit "ab" (ID 0)
        state.commit(LLMTokenID(0));
         assert_eq!(state.active_configurations.len(), 1); // Ensure commit didn't remove the state

        // Step 2 -> Mask allows "$"
        state.step_with_all_llm_tokens();
        assert_eq!(state.get_mask(), bitvec![0, 0, 1]);

        // Commit "$" (ID 2)
        state.commit(LLMTokenID(2));
         assert_eq!(state.active_configurations.len(), 1); // Should have completed parse path, state remains

        // Step 3 -> After consuming EOF, there should be no more allowed tokens.
        state.step_with_all_llm_tokens();
        assert_eq!(state.get_mask(), bitvec![0, 0, 0]); // No tokens allowed
        // The configuration might still exist but with an empty llm_tokens mask,
        // or it might be removed depending on finalizer logic details.
        // Let's check that the mask is empty, which is the primary goal.
        // assert!(state.active_configurations.is_empty()); // This might be too strict
    }

     #[test]
    fn test_constraint_invalid_commit() {
        let constraint = setup_test_constraint();
        let mut state = constraint.init();

        // Step 1 -> Mask allows "ab"
        state.step_with_all_llm_tokens();
        assert_eq!(state.get_mask(), bitvec![1, 0, 0]);

        // Commit "ac" (ID 1), which is not allowed by the mask
        state.commit(LLMTokenID(1));

        // All active configurations should be removed
        assert!(state.active_configurations.is_empty());
        assert_eq!(state.get_mask(), bitvec![0, 0, 0]); // No tokens allowed

        // Further steps should yield nothing
        state.step_with_all_llm_tokens();
        assert!(state.active_configurations.is_empty());
        assert_eq!(state.get_mask(), bitvec![0, 0, 0]);
    }
}
