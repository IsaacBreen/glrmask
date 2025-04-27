use std::cmp::Ordering;
use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
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
        current_compatible_llm_tokens.set(llm_token_id.0, true);

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
                let self_primary_key = self.src.prefix_length() + self.offset;
                let other_primary_key = other.src.prefix_length() + other.offset;
                (self_primary_key, self.src, self.bytes).cmp(&(other_primary_key, other.src, other.bytes))
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
            tokens_for_vocab_prefix_tree_builder.push((id.0, content.clone()));
        }
        let vocab_prefix_tree = VocabPrefixTree::build(&tokens_for_vocab_prefix_tree_builder);

        // Create the roots.
        let mut precomputed_roots: BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>> = BTreeMap::new();
        for tokenizer_state_id in 0..tokenizer.max_state() {
            let precompute_node = Arc::new(Mutex::new(PrecomputeNode::new(PrecomputedNodeContents::default())));
            precomputed_roots.insert(TokenizerStateID(tokenizer_state_id), precompute_node);
        }

        // Queue keyed by vocab node and tokenizer state ID.
        let mut queue: BTreeMap<(DottedVocabNode, TokenizerStateID), Vec<Arc<Mutex<PrecomputeNode>>>> = BTreeMap::new();

        // Initialize the queue with the roots.
        for (tokenizer_state_id, precompute_node) in &precomputed_roots {
            for (bytes, new_vocab_node) in vocab_prefix_tree.root.children() {
                let dotted_new_vocab_node = DottedVocabNode { src: &vocab_prefix_tree.root, dst: new_vocab_node, bytes, offset: 0 };
                queue.insert((dotted_new_vocab_node, *tokenizer_state_id), vec![precompute_node.clone()]);
            }
        }

        while let Some((((dotted_vocab_node, initial_tokenizer_state_id)), precomputed_nodes)) = queue.pop_first() {
            let DottedVocabNode { src, dst, offset, bytes } = dotted_vocab_node;

            let results = tokenizer.execute_from_state(&bytes[offset..], initial_tokenizer_state_id);

            for result in results.matches {
                let matched_token_id = GrammarTokenID(result.id);
                let new_offset = offset + result.width;
                // There's still more input to process. Insert trie edge(s) and update the queue.
                let new_dotted_node = DottedVocabNode { src, dst, offset: new_offset, bytes };
                // After a match, the tokenizer always resets to state 0 for the next token.
                let next_tokenizer_state_id = TokenizerStateID(0);
                let new_queue_key = (new_dotted_node, next_tokenizer_state_id);
                let mut next_precomputed_nodes = Vec::new();
                'outer: for precompute_node_arc in &precomputed_nodes {
                    let mut precompute_node = precompute_node_arc.lock().unwrap();
                    // The LLM tokens associated with this edge are *all* tokens reachable from the destination vocab node.
                    let edge_llm_tokens = dst.reachable_token_ids().clone();

                    // --- Try to merge with or reuse existing edges/nodes ---

                    // Check if an edge to an existing node (already in the queue for the *same* next state) exists.
                    if let Some(existing_queued_nodes) = queue.get(&new_queue_key) {
                        for existing_queued_node_arc in existing_queued_nodes {
                            // 1. Try merging into an existing edge value
                            if let Some(existing_edge_value) = precompute_node.get_edge_value_mut(matched_token_id, existing_queued_node_arc) {
                                *existing_edge_value |= &edge_llm_tokens; // Merge LLM tokens
                                // We merged, no need to add this path to next_precomputed_nodes, continue outer loop.
                                continue 'outer;
                            }
                            // 2. Try inserting a new edge to this existing node (checks for cycles)
                            if precompute_node.try_insert(matched_token_id, edge_llm_tokens.clone(), existing_queued_node_arc.clone()).is_ok() {
                                // Successfully inserted edge to existing node, continue outer loop.
                                continue 'outer;
                            }
                        }
                    }

                    // 3. Check if *any* edge for this grammar token already exists from the current node.
                    if let Some(existing_edges) = precompute_node.get_mut(&matched_token_id) {
                        if let Some((existing_edge_value, existing_dst_arc)) = existing_edges.first_mut() {
                            // Merge into the first existing edge's value and reuse its destination.
                            *existing_edge_value |= &edge_llm_tokens;
                            next_precomputed_nodes.push(existing_dst_arc.clone());
                            continue 'outer;
                        }
                        // If get_mut returned Some but it was empty (shouldn't happen with BTreeMap::entry), fall through.
                    }

                    // --- If no merge/reuse possible, create a new node ---
                    let new_precomputed_node_arc = precompute_node.force_insert(matched_token_id, edge_llm_tokens.clone(), PrecomputedNodeContents::default());
                    next_precomputed_nodes.push(new_precomputed_node_arc);
                } // end loop over precomputed_nodes

                // --- Handle queue update and finalizers based on offset ---
                if new_offset == bytes.len() {
                    // Reached the end of the vocab node bytes exactly after a match.
                    // This means the LLM token corresponding to `dst` is fully formed.
                    // The tokenizer is ready for the *next* token (state 0).
                    // We need to add finalizer info to the *destination* nodes reached by this match.
                    let possible_final_grammar_tokens: BTreeSet<_> = tokenizer.tokens_accessible_from_state(TokenizerStateID(0)).into_iter().map(|token_id| GrammarTokenID(token_id.0)).collect();
                    for possible_final_grammar_token in possible_final_grammar_tokens {
                        for next_node_arc in &next_precomputed_nodes {
                            // Add finalizer info for the completed LLM token `dst.token_id()`
                            // The tokenizer state *after* this token is 0.
                            next_node_arc.lock().unwrap().value.push_finalizer_info(
                                possible_final_grammar_token,
                                LLMTokenID(dst.token_id()),
                                TokenizerStateID(0), // State after completing the token
                                max_llm_token_id
                            );
                        }
                    }
                    // Since we reached the end, don't add to queue based on this path.
                } else if new_offset < bytes.len() {
                    // Matched a grammar token, but still within the vocab node bytes.
                    // Add the reached nodes to the queue for further processing from the new offset and state 0.
                    if !next_precomputed_nodes.is_empty() {
                         queue.entry(new_queue_key).or_default().extend(next_precomputed_nodes);
                    }
                } else {
                    // new_offset > bytes.len() should not happen
                    unreachable!("Tokenizer match width exceeded remaining bytes");
                }
            } // end loop over results.matches

            // Handle partial matches (end state reached before end of vocab node bytes)
            if let Some(end_state) = results.end_state {
                // The tokenizer stopped at `end_state` within the current vocab node `dst`.
                // This means the LLM token `dst.token_id()` can be finalized if the *next*
                // grammar token matches one accessible from `end_state`.
                let possible_final_grammar_tokens: BTreeSet<_> = tokenizer.tokens_accessible_from_state(TokenizerStateID(end_state)).into_iter().map(|token_id| GrammarTokenID(token_id.0)).collect();
                for possible_final_grammar_token in possible_final_grammar_tokens {
                    for precompute_node_arc in &precomputed_nodes {
                        // Add finalizer info to the *current* node (where the partial match occurred).
                        precompute_node_arc.lock().unwrap().value.push_finalizer_info(
                            possible_final_grammar_token,
                            LLMTokenID(dst.token_id()),
                            TokenizerStateID(end_state), // The state where the tokenizer stopped
                            max_llm_token_id
                        );
                    }
                }
            }
        } // end while let Some(...) = queue.pop_first()

        // Pull the roots out of their Arc<Mutex<_>>
        let precomputed_roots = precomputed_roots.into_iter().map(|(tokenizer_state_id, node)| (tokenizer_state_id, Arc::try_unwrap(node).expect("Arc unwrap failed").into_inner().expect("Mutex poison"))).collect();
        precomputed_roots
    }

    pub fn init(&self) -> GrammarConstraintState<'_> {
        let initial_parse_state = self.parser.init_parse_state();
        let initial_config = ActiveConstraintConfiguration {
            parse_state: initial_parse_state,
            llm_tokens: LLMTokenBV::repeat(true, self.max_llm_token_id + 1),
            tokenizer_state_ids: BTreeSet::from([TokenizerStateID(0)]),
        };
        GrammarConstraintState {
            parent: self,
            active_configurations: vec![initial_config],
        }
    }
}

impl GrammarConstraintState<'_> {
    pub fn get_mask(&mut self) -> LLMTokenBV {
        let mut mask = LLMTokenBV::repeat(false, self.parent.max_llm_token_id + 1);
        for config in &self.active_configurations {
            mask |= &config.llm_tokens;
        }
        mask
    }

    pub fn step_with_all_llm_tokens(&mut self) {
        let all_llm_tokens = LLMTokenBV::repeat(true, self.parent.max_llm_token_id + 1);
        self.step(&all_llm_tokens);
    }

    pub fn step_with_llm_token(&mut self, llm_token_id: LLMTokenID) {
        let mut llm_tokens = LLMTokenBV::repeat(false, self.parent.max_llm_token_id + 1);
        if llm_token_id.0 <= self.parent.max_llm_token_id {
            llm_tokens.set(llm_token_id.0, true);
        } else {
            // Handle error or warning: LLM token ID out of bounds
            eprintln!("Warning: LLM Token ID {} is out of bounds (max {})", llm_token_id.0, self.parent.max_llm_token_id);
        }
        self.step(&llm_tokens);
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) {
        // Keep only the active configurations for which this LLM token is set
        if llm_token_id.0 <= self.parent.max_llm_token_id {
            self.active_configurations.retain(|config| config.llm_tokens[llm_token_id.0]);
        } else {
            // If token is out of bounds, commit likely fails for all, clear active states.
             eprintln!("Warning: Committing out-of-bounds LLM Token ID {}. Clearing active states.", llm_token_id.0);
            self.active_configurations.clear();
        }
    }

    pub fn step_and_commit(&mut self, llm_token_id: LLMTokenID) {
        self.step_with_llm_token(llm_token_id);
        self.commit(llm_token_id);
    }

    pub fn commit_and_step_many(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.commit(llm_token_id); // Commit first
            self.step_with_llm_token(llm_token_id); // Then step
        }
    }

    fn prepare_initial_nodes_and_values_for_special_map(
        &self,
        llm_tokens_filter: &LLMTokenBV // Filter applied at the start
    ) -> Vec<(Arc<Mutex<PrecomputeNode>>, Vec<(ParseState, LLMTokenBV)>)> {
        let mut initial_map: BTreeMap<TokenizerStateID, Vec<(ParseState, LLMTokenBV)>> = BTreeMap::new();

        for config in &self.active_configurations {
            let initial_llm_bv = config.llm_tokens.clone() & llm_tokens_filter;
            if initial_llm_bv.not_any() { continue; } // Skip if no overlap with filter

            for tokenizer_state_id in &config.tokenizer_state_ids {
                initial_map.entry(*tokenizer_state_id)
                    .or_default()
                    .push((config.parse_state.clone(), initial_llm_bv.clone()));
            }
        }

        // Convert map to the required Vec format for special_map
        initial_map.into_iter().map(|(tokenizer_state_id, states_and_masks)| {
            // Clone the precomputed node for this tokenizer state ID
            let precompute_node = self.parent.precomputed[&tokenizer_state_id].clone();
            (Arc::new(Mutex::new(precompute_node)), states_and_masks)
        }).collect()
    }

    pub fn step(&mut self, llm_tokens_filter: &LLMTokenBV) {
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens_filter);

        // This vector will collect the results from the 'process' closure.
        // Each element represents a potential next state configuration.
        let collected_results = Arc::new(Mutex::new(Vec::<ActiveConstraintConfiguration>::new()));

        Trie::special_map(
            initial_nodes_and_values,
            // step: Propagate GLR states along the precomputed trie edges
            |current_states_and_masks, grammar_token_id, edge_llm_tokens, _child_node| {
                let mut next_states_and_masks = Vec::new();
                let parser = self.parent.parser; // Borrow parser reference

                for (parse_state, current_llm_bv) in current_states_and_masks {
                    // Calculate the mask for the next step by intersecting with edge mask
                    let next_llm_bv = current_llm_bv.clone() & edge_llm_tokens;
                    if next_llm_bv.not_any() { continue; } // Skip if no compatible LLM tokens

                    // Create a temporary GLRParserState to step the single ParseState
                    let mut temp_glr_state = parser.init_glr_parser_from_parse_state(parse_state.clone());
                    temp_glr_state.step(*grammar_token_id);

                    // Collect resulting active states
                    for next_parse_state in temp_glr_state.active_states {
                        next_states_and_masks.push((next_parse_state, next_llm_bv.clone()));
                    }
                    // We might also need to consider inactive states if they represent successful parses (GotoNotFound)
                    // For now, focusing on active states propagation. Handling inactive states happens in 'process'.
                }

                if next_states_and_masks.is_empty() {
                    None // Prune this path if no active states result
                } else {
                    Some(next_states_and_masks)
                }
            },
            // merge: Combine lists of (ParseState, LLMTokenBV) pairs
            |states_and_masks1, states_and_masks2| {
                // Simple extend for now. Merging duplicates could be added for optimization.
                states_and_masks1.extend(states_and_masks2);
                // TODO: Consider merging identical ParseStates by ORing their LLMTokenBVs
            },
            // process: Handle finalizers at Trie nodes
            {
                let collected_results = collected_results.clone();
                let parser = self.parent.parser; // Borrow parser reference

                move |trie_node_content, current_states_and_masks| {
                    let mut has_valid_continuation = false; // Track if any state can continue

                    for (possible_final_grammar_token, precomputed_finalizer) in trie_node_content.finalizers() {
                        for (parse_state, current_llm_bv) in current_states_and_masks.iter() {
                            // Check if the finalizer's grammar token leads to a valid state
                            let mut temp_glr_state = parser.init_glr_parser_from_parse_state(parse_state.clone());
                            temp_glr_state.step(*possible_final_grammar_token);

                            // A final state is valid if the step results in *any* state (active or inactive)
                            // OR if the original state was already accepting (though step handles this implicitly).
                            // We are interested if the *finalizer's* tokenizer states are reachable.
                            if !temp_glr_state.active_states.is_empty() || !temp_glr_state.inactive_states.is_empty() {
                                // Compute final LLM token mask
                                let final_llm_tokens = current_llm_bv.clone() & precomputed_finalizer.compatible_llm_tokens();
                                if final_llm_tokens.not_any() { continue; } // Skip if no compatible LLM tokens

                                // The tokenizer states are those specified by the finalizer
                                let final_tokenizer_states = precomputed_finalizer.tokenizer_state_ids().clone();
                                if final_tokenizer_states.is_empty() { continue; } // Should not happen if final_llm_tokens is not empty

                                // Create the resulting configuration
                                // The parse state used here is the one *before* applying the final grammar token,
                                // as this represents the state *at* the point where the final LLM token is emitted.
                                let result_config = ActiveConstraintConfiguration {
                                    parse_state: parse_state.clone(),
                                    llm_tokens: final_llm_tokens,
                                    tokenizer_state_ids: final_tokenizer_states,
                                };
                                collected_results.lock().unwrap().push(result_config);
                                has_valid_continuation = true; // Mark that we found at least one valid final state
                            }
                        }
                    }

                    // Determine if the path should continue based *only* on whether any active states remain
                    // in the current_states_and_masks *before* considering finalizers.
                    // The finalizers generate *new* configurations for the *next* step, they don't prune the current path.
                    let can_continue_normally = current_states_and_masks.iter().any(|(_, bv)| bv.any());

                    // Return true to continue processing children if there are still active states
                    // OR if a finalizer produced a valid result (meaning this path could lead to a valid next state).
                    // We return true if *any* state in current_states_and_masks has a non-empty LLM mask.
                    can_continue_normally
                }
            }
        );

        // Post-processing: Replace current configurations and merge duplicates
        let mut final_configurations = collected_results.lock().unwrap();

        // Merge configurations with the same (ParseStateKey, TokenizerStateID set)
        let mut merged_map: HashMap<(ParseStateKey, BTreeSet<TokenizerStateID>), ActiveConstraintConfiguration> = HashMap::new();

        for config in final_configurations.drain(..) {
            let key = (config.parse_state.key(), config.tokenizer_state_ids.clone());
            merged_map.entry(key)
                .and_modify(|existing_config| {
                    // Merge ParseState (GSS nodes)
                    existing_config.parse_state.merge(config.parse_state.clone()); // Use clone here
                    // Merge LLM tokens
                    existing_config.llm_tokens |= &config.llm_tokens;
                })
                .or_insert(config);
        }

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
        constraint.dump_precomputed(); // Optional: print the precomputed structure
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

        // Step 2 -> Mask allows "$"
        state.step_with_all_llm_tokens();
        assert_eq!(state.get_mask(), bitvec![0, 0, 1]);

        // Commit "$" (ID 2)
        state.commit(LLMTokenID(2));
         assert_eq!(state.active_configurations.len(), 1); // Should have completed parse path

        // Step 3 -> After consuming EOF, there should be no more allowed tokens.
        state.step_with_all_llm_tokens();
        assert_eq!(state.get_mask(), bitvec![0, 0, 0]); // No tokens allowed
        assert!(state.active_configurations.is_empty()); // No active configurations remain
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
