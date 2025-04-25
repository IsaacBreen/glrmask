use std::cmp::Ordering;
use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
use crate::glr::parser::{GLRParser, GLRParserState, ParseState, ParseStateKey};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use bitvec::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::BitOr;
use std::sync::{Arc, Mutex};
use bitvec::macros::internal::funty::Fundamental;
use keyed_priority_queue::KeyedPriorityQueue;
use crate::datastructures::charmap::TrieMap;
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::managed_glr_parser::{ManagedGLRParserState, ManagedParseState};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};

pub type LLMTokenBV = BitVec;
pub type GrammarTokenBV = BitVec;

#[derive(Default, Debug, Clone)]
pub struct PrecomputedFinalizer {
    pub(crate) possible_final_grammar_tokens: BTreeSet<GrammarTokenID>,
    pub(crate) compatible_llm_tokens: LLMTokenBV,
    pub(crate) tokenizer_state_ids: BTreeSet<TokenizerStateID>,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)] // Added PartialEq, Eq for potential future use/testing
pub(crate) struct PrecomputedNodeContents {
    pub(crate) finalizers: Vec<PrecomputedFinalizer>,
}

type PrecomputeNode = Trie<GrammarTokenID, LLMTokenBV, PrecomputedNodeContents>;
type Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>;

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
    pub(crate) parent: &'a GrammarConstraint,
    pub(crate) state: ManagedGLRParserState<'a>,
}

impl PrecomputedNodeContents {
    pub fn push_finalizer_info(&mut self, possible_final_grammar_tokens: &BTreeSet<GrammarTokenID>, token_id: usize, tokenizer_state_id: TokenizerStateID) {
        let mut finalizer = PrecomputedFinalizer::default();
        finalizer.possible_final_grammar_tokens = possible_final_grammar_tokens.clone();
        if finalizer.compatible_llm_tokens.len() < token_id + 1 {
            finalizer.compatible_llm_tokens.resize(token_id + 1, false);
        }
        finalizer.compatible_llm_tokens.set(token_id, true);
        finalizer.tokenizer_state_ids.insert(tokenizer_state_id);
        self.finalizers.push(finalizer);
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
                if new_offset == bytes.len() {
                    // Reached the end of the input, so this is a clean match.
                    let possible_final_grammar_tokens: BTreeSet<_> = tokenizer.tokens_accessible_from_state(TokenizerStateID(0)).into_iter().map(|token_id| GrammarTokenID(token_id.0)).collect(); // Should contain all tokens
                    for precompute_node in &precomputed_nodes {
                        precompute_node.lock().unwrap().value.push_finalizer_info(&possible_final_grammar_tokens, dst.token_id(), TokenizerStateID(0));
                    }
                } else if new_offset < bytes.len() {
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
                    queue.insert(new_queue_key, next_precomputed_nodes);
                } else { unreachable!(); }
            }

            if let Some(end_state) = results.end_state {
                let possible_final_grammar_tokens: BTreeSet<_> = tokenizer.tokens_accessible_from_state(TokenizerStateID(end_state)).into_iter().map(|token_id| GrammarTokenID(token_id.0)).collect();
                for precompute_node in &precomputed_nodes {
                    precompute_node.lock().unwrap().value.push_finalizer_info(&possible_final_grammar_tokens, dst.token_id(), TokenizerStateID(end_state));
                }
            }
        }

        // Pull the roots out of their Arc<Mutex<_>>
        let precomputed_roots = precomputed_roots.into_iter().map(|(tokenizer_state_id, node)| (tokenizer_state_id, node.lock().unwrap().clone())).collect();
        precomputed_roots
    }

    pub fn init(&self) -> GrammarConstraintState<'_> {
        let glr_parser_initial_state = self.parser.init_managed_glr_parser();

        GrammarConstraintState {
            parent: self,
            state: glr_parser_initial_state,
        }
    }
}

impl GrammarConstraintState<'_> {
    pub fn get_mask(&mut self) -> LLMTokenBV {
        let mut mask = LLMTokenBV::repeat(false, self.parent.max_llm_token_id + 1);
        for managed_parse_state in &self.state.active_states {
            mask |= managed_parse_state.llm_tokens.clone();
        }
        mask
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
        // Keep only the active states for which this LLM token is set
        self.state.active_states.retain(|managed_parse_state| managed_parse_state.llm_tokens[llm_token_id.0]);
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

    fn prepare_initial_nodes_and_values_for_special_map(&mut self, llm_tokens: &LLMTokenBV) -> Vec<(Arc<Mutex<Trie<TerminalID, LLMTokenBV, PrecomputedNodeContents>>>, ManagedGLRParserState)> {
        // The BTreeSet<TokenizerStateID> in each Trie node here is the set of terminal states at this node.
        // Each terminal state indicates that the path through the trie can terminate here.
        // (todo: explain this better)
        let mut initial_nodes_and_values: Vec<(Arc<Mutex<PrecomputeNode>>, ManagedGLRParserState)> = Vec::new();

        let mut tokenizer_state_id_to_parse_states: BTreeMap<TokenizerStateID, (BTreeSet<ManagedParseState>, LLMTokenBV)> = BTreeMap::new();
        for managed_parse_state in self.state.active_states.iter() {
            for tokenizer_state_id in managed_parse_state.tokenizer_state_ids.iter() {
                tokenizer_state_id_to_parse_states.entry(*tokenizer_state_id).or_default().0.insert(managed_parse_state.clone());
                tokenizer_state_id_to_parse_states.entry(*tokenizer_state_id).or_default().1 = llm_tokens.clone();
            }
        }

        for (tokenizer_state_id, (parse_states, llm_tokens)) in tokenizer_state_id_to_parse_states {
            let token_trie = self.parent.precomputed[&tokenizer_state_id].clone();
            let token_trie = Arc::new(Mutex::new(token_trie));
            let managed_glr_parser_state = GLRParser::init_managed_glr_parser_from_managed_parse_states(self.state.parser, parse_states.into_iter().collect());
            initial_nodes_and_values.push((token_trie, managed_glr_parser_state));
        }
        initial_nodes_and_values
    }

    pub fn step(&mut self, llm_tokens: &LLMTokenBV) {
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens);

        let mut final_active_parse_states: Vec<ManagedParseState> = Vec::new();
        let mut final_inactive_parse_states: Vec<ManagedParseState> = Vec::new();

        Trie::special_map(
            initial_nodes_and_values,
            // step
            |managed_parse_state, grammar_token_id, edge_llm_tokens, child_node| {
                let mut managed_parse_state = managed_parse_state.clone();
                managed_parse_state.active_states.retain_mut(|managed_parse_state| {
                    managed_parse_state.llm_tokens &= edge_llm_tokens.clone();
                    !managed_parse_state.llm_tokens.is_empty()
                });
                if managed_parse_state.active_states.is_empty() { return None; } else { Some(managed_parse_state) }
            },
            // merge
            |managed_parse_state1, managed_parse_state2| {
                managed_parse_state1.merge_with(managed_parse_state2);
            },
            // process
            |node, managed_glr_parse_state| {
                // Handle finalizers
                for precomputed_finalizer in &node.value.finalizers {
                    for managed_parse_state in &managed_glr_parse_state.active_states {
                        // Ensure at least one of the final tokens parses
                        let mut valid_final_tokenizer_state_ids = BTreeSet::new();
                        for possible_final_grammar_token in &precomputed_finalizer.possible_final_grammar_tokens {
                            let mut parse_state = managed_glr_parse_state.parser.init_glr_parser_from_parse_state(ParseState::from(managed_parse_state.clone()));
                            parse_state.step(*possible_final_grammar_token);
                            if parse_state.matches_or_can_match() {
                                valid_final_tokenizer_state_ids = managed_parse_state.tokenizer_state_ids.clone();
                                break;
                            }
                        }
                        if valid_final_tokenizer_state_ids.is_empty() {
                            // If we've reached the initial state, we've matched the final token cleanly, and we can proceed without any additional tokens.
                            if precomputed_finalizer.tokenizer_state_ids.contains(&TokenizerStateID(0)) {
                                valid_final_tokenizer_state_ids.insert(TokenizerStateID(0));
                            } else {
                                continue;
                            }
                        }
                        // Compute final LLM token mask
                        let final_llm_tokens = managed_parse_state.llm_tokens.clone() & precomputed_finalizer.compatible_llm_tokens.clone();
                        if final_llm_tokens.is_empty() { continue; }
                        // Create a new managed parse state
                        let mut managed_parse_state = managed_parse_state.clone();
                        managed_parse_state.tokenizer_state_ids = valid_final_tokenizer_state_ids;
                        managed_parse_state.llm_tokens = final_llm_tokens;
                        final_active_parse_states.push(managed_parse_state);
                    }
                }
                managed_glr_parse_state.active_states.retain(|managed_parse_state| !managed_parse_state.llm_tokens.is_empty());
                !managed_glr_parse_state.active_states.is_empty()
            },
        );

        self.state.active_states = final_active_parse_states;
        self.state.inactive_states.extend(final_inactive_parse_states);
    }
}

#[cfg(test)]
mod tests {
    use crate::finite_automata::eat_u8;
    use crate::{choice, groups, seq};
    use crate::glr::grammar::{nt, prod, t, NonTerminal, Terminal};
    use crate::glr::table::{generate_glr_parser, generate_glr_parser_with_maps, generate_glr_parser_with_terminal_map};
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
            prod("S", vec![t("A"), t("EOF")]),
            prod("S", vec![t("AB"), t("EOF")]),
            prod("S", vec![t("B_OR_C"), t("EOF")]),
        ];

        let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
        grammar_token_map.insert(Terminal("A".to_string()), TerminalID(0));
        grammar_token_map.insert(Terminal("AB".to_string()), TerminalID(1));
        grammar_token_map.insert(Terminal("B_OR_C".to_string()), TerminalID(2));
        grammar_token_map.insert(Terminal("EOF".to_string()), TerminalID(3));

        let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map);

        let constraint = GrammarConstraint::new(tokenizer, parser, llm_token_map, 2);

        let mut constraint_state = constraint.init();

        constraint_state.step_with_all_llm_tokens();

        let mask = constraint_state.get_mask();
        // assert_eq!(mask, LLMTokenBV::from_iter([true, true, false]));

        constraint_state.commit(LLMTokenID(1));
        constraint_state.step_with_all_llm_tokens();

        let mask = constraint_state.get_mask();
        assert_eq!(mask, LLMTokenBV::from_iter([false, false, true]));
    }
}