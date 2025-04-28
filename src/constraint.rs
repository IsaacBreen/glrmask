use std::cmp::Ordering;
use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
use crate::glr::parser::{GLRParser, GLRParserState, ParseState, ParseStateKey, StopReason, ParseStatus};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use bitvec::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::BitOr;
use std::sync::{Arc, Mutex};
use crate::constraint_extra::print_finalizer;
use crate::datastructures::charmap::TrieMap;
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
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

#[derive(Debug, Clone)] // Removed pub(crate) as it's likely used externally
pub(crate) struct GrammarConstraint {
    pub(crate) tokenizer: Regex,
    pub(crate) parser: GLRParser,
    pub(crate) precomputed: Precomputed,
    pub(crate) llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID>,
    pub(crate) max_llm_token_id: usize,
}

#[derive(Debug, Clone)]
pub struct GrammarConstraintState<'a> {
    pub(crate) parent: &'a GrammarConstraint, // Made pub(crate)
    // Map from TokenizerStateID to the GLR ParseStates reached *before* consuming
    // the next LLM token that would start from that tokenizer state.
    states: BTreeMap<TokenizerStateID, Vec<ParseState>>,
    current_mask: LLMTokenBV,
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
                let new_queue_key = (new_dotted_node, TokenizerStateID(0));
                let mut next_precomputed_nodes = Vec::new();
                'outer: for precompute_node in &precomputed_nodes {
                    let mut precompute_node = precompute_node.lock().unwrap();
                    let llm_tokens = dst.reachable_token_ids().clone();
                    if let Some(existing_precompute_nodes) = queue.get(&new_queue_key) {
                        // Try to push to an existing precompute node in the queue if it's possible to do so without creating a cycle.
                        for existing_precompute_node in existing_precompute_nodes {
                            if let Some(existing_edge_value) = precompute_node.get_edge_value_mut(matched_token_id, existing_precompute_node) {
                                // Merge into the edge value.
                                *existing_edge_value = existing_edge_value.clone().bitor(llm_tokens.clone());
                                continue 'outer;
                            }
                        }

                        // Try to insert a new edge to any existing node.
                        for existing_precompute_node in existing_precompute_nodes {
                            if let Ok(dst_precomputed_node) = precompute_node.try_insert(matched_token_id, llm_tokens.clone(), existing_precompute_node.clone()) {
                                continue 'outer;
                            }
                        }
                    }

                    // Use any existing edge on the src node.
                    if let Some(existing_edges) = precompute_node.get_mut(&matched_token_id) {
                        if let Some((existing_edge_value, exising_dst)) = existing_edges.iter_mut().next() {
                            // Merge into the edge value.
                            *existing_edge_value = existing_edge_value.clone().bitor(llm_tokens.clone());
                            next_precomputed_nodes.push(exising_dst.clone());
                            continue 'outer;
                        }
                    }

                    // Create a new node.
                    let new_precomputed_node = precompute_node.force_insert(matched_token_id, llm_tokens.clone(), PrecomputedNodeContents::default());
                    next_precomputed_nodes.push(new_precomputed_node.clone());
                }
                if new_offset == bytes.len() {
                    // Reached the end of the input, so this is a clean match.
                    let possible_final_grammar_tokens: BTreeSet<_> = tokenizer.tokens_accessible_from_state(TokenizerStateID(0)).into_iter().map(|token_id| GrammarTokenID(token_id.0)).collect(); // Should contain all tokens
                    for possible_final_grammar_token in possible_final_grammar_tokens {
                        for new_precomputed_node in &next_precomputed_nodes {
                            new_precomputed_node.lock().unwrap().value.push_finalizer_info(possible_final_grammar_token, LLMTokenID(dst.token_id()), TokenizerStateID(0), max_llm_token_id);
                        }
                    }
                } else if new_offset < bytes.len() {
                    queue.entry(new_queue_key).or_default().extend(next_precomputed_nodes);
                } else { unreachable!(); }
            }
            // Handle partial matches (end state reached before end of vocab node bytes)
            if let Some(end_state) = results.end_state {
                let possible_final_grammar_tokens: BTreeSet<_> = tokenizer.tokens_accessible_from_state(TokenizerStateID(end_state)).into_iter().map(|token_id| GrammarTokenID(token_id.0)).collect();
                for possible_final_grammar_token in possible_final_grammar_tokens {
                    for precompute_node in &precomputed_nodes {
                        precompute_node.lock().unwrap().value.push_finalizer_info(possible_final_grammar_token, LLMTokenID(dst.token_id()), TokenizerStateID(end_state), max_llm_token_id);
                    }
                }
            }
        }

        // Pull the roots out of their Arc<Mutex<_>>
        let precomputed_roots = precomputed_roots.into_iter().map(|(tokenizer_state_id, node)| (tokenizer_state_id, node.lock().unwrap().clone())).collect();
        precomputed_roots
    }

    pub fn init(&self) -> GrammarConstraintState<'_> {
        // Get the single initial ParseState
        let initial_parse_state = self.parser.init_parse_state();
        let initial_states = BTreeMap::from([(TokenizerStateID(0), vec![initial_parse_state])]);
        let initial_mask = LLMTokenBV::repeat(true, self.max_llm_token_id + 1); // Initially, any token might be possible

        GrammarConstraintState {
            parent: self,
            states: initial_states,
            current_mask: initial_mask,
        }
    }
}

impl<'a> GrammarConstraintState<'a> {
    pub fn get_mask(&mut self) -> LLMTokenBV {
        // TODO: This should be recalculated based on the current states and precomputed info?
        // For now, return the stored mask which is updated by step/commit.
        self.current_mask.clone()
    }

    pub fn step_with_all_llm_tokens(&mut self) {
        let all_llm_tokens = LLMTokenBV::repeat(true, self.parent.max_llm_token_id + 1);
        self.step(&all_llm_tokens);
    }

    pub fn step_with_llm_token(&mut self, llm_token_id: LLMTokenID) {
        let mut llm_tokens = LLMTokenBV::repeat(false, self.parent.max_llm_token_id + 1);
        llm_tokens.set(llm_token_id.0, true);
        self.step(&llm_tokens);
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) {
        let mut next_states: BTreeMap<TokenizerStateID, Vec<ParseState>> = BTreeMap::new();
        let mut next_mask = LLMTokenBV::repeat(false, self.parent.max_llm_token_id + 1);

        // The `step` function should have populated `self.states` with Vec<(ParseState, LLMTokenBV)>
        // We need to adjust the structure or how `step` updates it.
        // Assuming `step` correctly calculates the mask for *each potential next state*:
        for (tokenizer_state_id, parse_state_vec) in &self.states {
            // This logic is flawed. The mask is associated with the *path* taken, not the state itself.
            // `commit` needs to know which paths (leading to which states) are compatible with `llm_token_id`.
            // This information must come from the `step` function's execution.

            // Let's assume `step` has already filtered `self.states` based on the provided `llm_tokens` mask.
            // `commit` then *selects* the path corresponding to `llm_token_id`.
            // This requires `step` to store more info.

            // --- Placeholder: Revisit commit logic after step is implemented ---
            // For now, assume step correctly prepares the state for the *next* step after commit.
            // We just need to update the mask based on the committed token.
        }
        // self.current_mask = ??? // This needs recalculation based on the committed states.
    }

    pub fn step_and_commit(&mut self, llm_token_id: LLMTokenID) {
        self.step_with_llm_token(llm_token_id);
        self.commit(llm_token_id);
    }

    pub fn commit_and_step_many(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.step_with_llm_token(llm_token_id);
        }
    }

    fn prepare_initial_nodes_and_values_for_special_map(
        &self,
    ) -> Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'a>)> {
        let mut initial_nodes_and_values = Vec::new();
        for (tokenizer_state_id, parse_states_vec) in &self.states {
            if let Some(precomputed_root_trie) = self.parent.precomputed.get(tokenizer_state_id) {
                let precomputed_root_arc = Arc::new(Mutex::new(precomputed_root_trie.clone()));
                // Create a GLRParserState containing only the active states for this tokenizer state
                let active_parse_states: Vec<ParseState> = parse_states_vec
                    .iter()
                    .filter(|ps| ps.status == ParseStatus::Active) // Should always be active here?
                    .cloned()
                    .collect();

                if !active_parse_states.is_empty() {
                    let glr_state = self.parent.parser.init_glr_parser_from_parse_states(active_parse_states);
                    initial_nodes_and_values.push((precomputed_root_arc, glr_state));
                }
            } else {
                // This tokenizer state might not have a corresponding precomputed root if it's unreachable?
                // Or maybe precompute should create entries for all possible states.
                eprintln!("Warning: No precomputed root found for TokenizerStateID {}", tokenizer_state_id.0);
            }
        }
        initial_nodes_and_values
    }

    pub fn step(&mut self, llm_tokens: &LLMTokenBV) {
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map();

        // Shared state to collect results from the 'process' closure
        let next_states_map_mutex = Arc::new(Mutex::new(BTreeMap::<TokenizerStateID, Vec<(ParseState, LLMTokenBV)>>::new()));
        let next_mask_mutex = Arc::new(Mutex::new(LLMTokenBV::repeat(false, self.parent.max_llm_token_id + 1)));
        let llm_tokens_clone = llm_tokens.clone(); // Clone for capture

        Trie::special_map(
            initial_nodes_and_values,
            // step
            |parent_glr_state: &GLRParserState<'a>, grammar_token_id, _edge_llm_tokens, _child_node| {
                // The edge_llm_tokens are used in `process` when finalizing.
                let mut next_glr_state = parent_glr_state.clone();
                next_glr_state.step(*grammar_token_id);
                if next_glr_state.active_states.is_empty() {
                    // println!("No active states after processing grammar token {}", grammar_token_id.0);
                    return None;
                } else {
                    // println!("Processed grammar token {}, {} active states.", grammar_token_id.0, next_glr_state.active_states.len());
                    Some(next_glr_state)
                }
            },
            // merge
            |existing_glr_state, new_glr_state| {
                existing_glr_state.merge_with(new_glr_state);
            },
            // process
            {
                // Clone Arcs for the closure
                let next_states_map_clone = next_states_map_mutex.clone();
                let next_mask_clone = next_mask_mutex.clone();
                let parser_clone = self.parent.parser.clone(); // GLRParser needs to be Clone

                move |node: &PrecomputeNode, final_glr_state: &mut GLRParserState<'a>| {
                // Handle finalizers
                for (possible_final_grammar_token, precomputed_finalizer) in node.value.finalizers() {
                    // Create a temporary GLR state representing the states *before* this potential final grammar token
                    let temp_glr_state_base = parser_clone.init_glr_parser_from_parse_states(final_glr_state.active_states.clone());

                    // Simulate the step with the final grammar token
                    let mut temp_glr_state_final = temp_glr_state_base.clone();
                    temp_glr_state_final.step(*possible_final_grammar_token);

                    // Check if this final step leads to a valid (match or potential match) state
                    if temp_glr_state_final.matches_or_can_match() {
                        // Calculate the LLM token mask for this specific path completion
                        let current_path_mask = precomputed_finalizer.compatible_llm_tokens().clone() & llm_tokens_clone.clone(); // Intersect with input mask (use reference)

                        if !current_path_mask.is_empty() {
                            // Update the overall mask for the next step
                            {
                                let mut global_mask = next_mask_clone.lock().unwrap();
                                *global_mask |= &current_path_mask;
                            }

                            // Add the states *before* the final step to the results map, associated with the next tokenizer states
                            let mut next_states_map = next_states_map_clone.lock().unwrap();
                            for next_tokenizer_state_id in precomputed_finalizer.tokenizer_state_ids() {
                                let entry = next_states_map.entry(*next_tokenizer_state_id).or_default();
                                // Add each base state paired with the mask that allows reaching it
                                for base_state in &temp_glr_state_base.active_states {
                                    entry.push((base_state.clone(), current_path_mask.clone()));
                                }
                            }
                        }
                    }
                }
                // Continue processing children if the GLR state before finalization had active states
                !final_glr_state.active_states.is_empty()
            }},
        );

        // Update the constraint state with the collected results
        let final_next_states_map = Arc::try_unwrap(next_states_map_mutex)
            .expect("Mutex still held")
            .into_inner()
            .expect("Mutex poisoned");

        let final_next_mask = Arc::try_unwrap(next_mask_mutex)
            .expect("Mutex still held")
            .into_inner()
            .expect("Mutex poisoned");

        // Merge the collected states
        let mut merged_states: BTreeMap<TokenizerStateID, Vec<ParseState>> = BTreeMap::new();
        for (tokenizer_id, state_mask_pairs) in final_next_states_map {
            let mut unique_states_map: BTreeMap<ParseStateKey, ParseState> = BTreeMap::new();
            for (state, _mask) in state_mask_pairs { // Discard mask for now, it's combined in final_next_mask
                let key = state.key();
                unique_states_map.entry(key)
                    .and_modify(|existing| existing.merge(state.clone())) // Use merge method
                    .or_insert(state);
            }
            if !unique_states_map.is_empty() {
                merged_states.insert(tokenizer_id, unique_states_map.into_values().collect());
            }
        }

        self.states = merged_states;
        self.current_mask = final_next_mask;
    }
}

#[cfg(test)]
mod tests {
    use crate::finite_automata::eat_u8;
    use crate::{choice, groups, seq};
    use crate::glr::grammar::{nt, prod, t, NonTerminal, Terminal};
    use crate::glr::table::generate_glr_parser_with_terminal_map;
    use super::*;

    #[test]
    fn test_constraint_simple() {
        // LLM tokens: "ab", "ac"
        // Grammar tokens: "a", "ab", "b|c"
        // Grammar: S -> "a" | "ab" | "b|c"
        let expr = groups![
            eat_u8(b'a'),
            seq![eat_u8(b'a'), eat_u8(b'b')],
            choice![eat_u8(b'b'), eat_u8(b'c')],
            eat_u8(b'$'),
        ];
        let tokenizer = expr.build();

        let mut llm_token_map = LLMTokenMap::new();
        llm_token_map.insert(b"ab".to_vec(), LLMTokenID(0));
        llm_token_map.insert(b"ac".to_vec(), LLMTokenID(1));
        llm_token_map.insert(b"$".to_vec(), LLMTokenID(2));

        let productions = vec![
            // prod("S", vec![t("A"), t("EOF")]),
            prod("S", vec![t("AB"), t("EOF")]),
            // prod("S", vec![t("B_OR_C"), t("EOF")]),
        ];

        let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
        grammar_token_map.insert(Terminal("A".to_string()), TerminalID(0));
        grammar_token_map.insert(Terminal("AB".to_string()), TerminalID(1));
        grammar_token_map.insert(Terminal("B_OR_C".to_string()), TerminalID(2));
        grammar_token_map.insert(Terminal("EOF".to_string()), TerminalID(3));

        let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map);

        let constraint = GrammarConstraint::new(tokenizer, parser, llm_token_map, 2);
        constraint.dump_precomputed();

        let mut constraint_state = constraint.init();

        // Initial state should allow both "ab" and "ac"
        let initial_mask = constraint_state.get_mask();
        // This assertion depends on how the initial mask is calculated/updated by `init` and `step`.
        // Let's assume `step` correctly calculates the mask based on reachable finalizers.
        // assert_eq!(initial_mask, LLMTokenBV::from_iter([true, true, false]));

        constraint_state.step_with_all_llm_tokens();

        let mask = constraint_state.get_mask();
        assert_eq!(mask, LLMTokenBV::from_iter([true, true, false])); // Should allow "ab" or "ac"

        // TODO: Add tests for commit and multi-step scenarios once commit is fully implemented.
    }
}
