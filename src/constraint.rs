use std::cmp::Ordering;
use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
use crate::glr::parser::{AndAndOr, GLRParser, GLRParserState, ParseState, ParseStateKey};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use bitvec::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::BitOr;
use std::sync::{Arc, Mutex};
use bitvec::macros::internal::funty::Fundamental;
use keyed_priority_queue::KeyedPriorityQueue;
use crate::constraint_extra::print_finalizer;
use crate::datastructures::charmap::TrieMap;
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};

pub type LLMTokenBV = BitVec;
pub type GrammarTokenBV = BitVec;

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

pub(crate) type PrecomputeNode = Trie<GrammarTokenID, LLMTokenBV, PrecomputedNodeContents>;
pub(crate) type Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>;

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
    pub(crate) state: BTreeMap<TokenizerStateID, GLRParserState<'a, LLMTokenBV>>,
}

impl AndAndOr for LLMTokenBV {
    fn and(&self, other: &Self) -> Self {
        self.clone() & other.clone()
    }

    fn or(&self, other: &Self) -> Self {
        self.clone() | other.clone()
    }
}

impl PrecomputedNodeContents {
    pub(crate) fn finalizers(&self) -> &BTreeMap<GrammarTokenID, PrecomputedFinalizer> { &self.finalizers }

    /// Adds information about a final state reachable via a specific grammar token.
    /// If an entry for the grammar token already exists, it merges the information.
    pub fn push_finalizer_info(&mut self, possible_final_grammar_token: GrammarTokenID, llm_token_id: LLMTokenID, tokenizer_state_id: TokenizerStateID, max_llm_token_id: usize) {
        let mut current_compatible_llm_tokens = LLMTokenBV::repeat(false, max_llm_token_id + 1);
        current_compatible_llm_tokens.set(llm_token_id.0, true);

        self.finalizers.entry(possible_final_grammar_token)
            .and_modify(|existing_finalizer| {
                existing_finalizer.content.entry(tokenizer_state_id).and_modify(|existing_llm_tokens| {
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
                    for new_precomputed_node in &next_precomputed_nodes {
                        new_precomputed_node.lock().unwrap().value.clean_end.get_or_insert_with(|| LLMTokenBV::repeat(false, max_llm_token_id + 1)).set(dst.token_id(), true);
                    }
                    let next_src = dst;
                    for (next_bytes, next_dst) in next_src.children() {
                        let new_dotted_node = DottedVocabNode { src: next_src, dst: next_dst, bytes: next_bytes, offset: 0 };
                        let new_queue_key = (new_dotted_node, TokenizerStateID(0));
                        queue.entry(new_queue_key).or_default().extend(next_precomputed_nodes.iter().cloned());
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
                let next_src = dst;
                for (next_bytes, next_dst) in next_src.children() {
                    let new_dotted_node = DottedVocabNode { src: next_src, dst: next_dst, bytes: next_bytes, offset: 0 };
                    let new_queue_key = (new_dotted_node, TokenizerStateID(0));
                    queue.entry(new_queue_key).or_default().extend(precomputed_nodes.iter().cloned());
                }
            }
        }

        // Pull the roots out of their Arc<Mutex<_>>
        let precomputed_roots = precomputed_roots.into_iter().map(|(tokenizer_state_id, node)| (tokenizer_state_id, node.lock().unwrap().clone())).collect();
        precomputed_roots
    }

    pub fn init(&self) -> GrammarConstraintState<'_> {
        let llm_tokens = LLMTokenBV::repeat(true, self.max_llm_token_id + 1);
        let initial_glr_parser_state: GLRParserState<'_, LLMTokenBV> = self.parser.init_glr_parser_with_t(llm_tokens);
        let mut state = BTreeMap::new();
        state.insert(self.tokenizer.initial_state_id(), initial_glr_parser_state);

        GrammarConstraintState {
            parent: self,
            state,
        }
    }
}

impl<'a> GrammarConstraintState<'a> {
    pub fn get_mask(&mut self) -> LLMTokenBV {
        let mut mask = LLMTokenBV::repeat(false, self.parent.max_llm_token_id + 1);
        for (_, state) in &self.state {
            for active_state in &state.active_states {
                mask |= active_state.stack.peek().t.clone();
            }
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
        for (_, state) in &mut self.state {
            state.active_states.retain(|active_state| active_state.stack.peek().t[llm_token_id.0]);
        }
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

    fn prepare_initial_nodes_and_values_for_special_map(&mut self, llm_tokens: &LLMTokenBV) -> Vec<(Arc<Mutex<Trie<TerminalID, LLMTokenBV, PrecomputedNodeContents>>>, GLRParserState<'a, LLMTokenBV>)> {
        // The BTreeSet<TokenizerStateID> in each Trie node here is the set of terminal states at this node.
        // Each terminal state indicates that the path through the trie can terminate here.
        // (todo: explain this better)
        let mut initial_nodes_and_values: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'_, LLMTokenBV>)> = Vec::new();

        let mut tokenizer_state_id_to_parse_states: BTreeMap<TokenizerStateID, GLRParserState<'_, LLMTokenBV>> = BTreeMap::new();
        // for managed_parse_state in self.state.active_states.iter() {
        for (tokenizer_state_id, state) in &self.state {
            let mut state = state.clone();
            for parse_state in state.active_states.iter_mut() {
                Arc::make_mut(&mut parse_state.stack).value.t = llm_tokens.clone();
            }
            tokenizer_state_id_to_parse_states.insert(*tokenizer_state_id, state);
        }

        for (tokenizer_state_id, state) in tokenizer_state_id_to_parse_states {
            let token_trie = self.parent.precomputed[&tokenizer_state_id].clone();
            let token_trie = Arc::new(Mutex::new(token_trie));
            initial_nodes_and_values.push((token_trie, state));
        }
        initial_nodes_and_values
    }

    pub fn step(&mut self, llm_tokens: &LLMTokenBV) {
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens);

        self.state = BTreeMap::new();

        Trie::special_map(
            initial_nodes_and_values,
            // step
            |glr_parse_state, grammar_token_id, edge_llm_tokens, child_node| {
                let mut glr_parse_state = glr_parse_state.clone();
                glr_parse_state.active_states.retain_mut(|parse_state| {
                    Arc::make_mut(&mut parse_state.stack).value.t &= edge_llm_tokens.clone();
                    !parse_state.stack.value.t.is_empty()
                });
                glr_parse_state.step(*grammar_token_id);
                if glr_parse_state.active_states.is_empty() {
                    println!("No active states after processing grammar token {}", grammar_token_id.0);
                    return None;
                } else {
                    println!("Processed grammar token {}, {} active states.", grammar_token_id.0, glr_parse_state.active_states.len());
                    Some(glr_parse_state)
                }
            },
            // merge
            |managed_parse_state1, managed_parse_state2| {
                managed_parse_state1.merge_with(managed_parse_state2);
            },
            // process
            |node, glr_parse_state| {
                // Handle clean end
                if let Some(clean_end) = &node.value.clean_end {
                    let mut final_glr_parse_state = glr_parse_state.clone();
                    final_glr_parse_state.active_states.retain_mut(|parse_state| {
                        Arc::make_mut(&mut parse_state.stack).value.t |= clean_end.clone();
                        !parse_state.stack.value.t.is_empty()
                    });
                    if final_glr_parse_state.is_ok() {
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
                    let mut semi_final_glr_parse_state = glr_parse_state.clone();
                    semi_final_glr_parse_state.step(*possible_final_grammar_token);
                    if semi_final_glr_parse_state.is_ok() {
                        for (tokenizer_state_id, llm_tokens) in &precomputed_finalizer.content {
                            // Merge LLM tokens
                            let mut semi_final_glr_parse_state = semi_final_glr_parse_state.clone();
                            semi_final_glr_parse_state.active_states.retain_mut(|parse_state| {
                                Arc::make_mut(&mut parse_state.stack).value.t |= llm_tokens.clone();
                                !parse_state.stack.value.t.is_empty()
                            });
                            if semi_final_glr_parse_state.is_ok() {
                                if let Some(existing) = self.state.get_mut(tokenizer_state_id) {
                                    existing.merge_with(semi_final_glr_parse_state.clone());
                                } else {
                                    self.state.insert(*tokenizer_state_id, semi_final_glr_parse_state.clone());
                                }
                            }
                        }
                    }
                }
                glr_parse_state.active_states.retain(|parse_state| !parse_state.stack.value.t.is_empty());
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

    #[test]
    fn test_constraint_simple() {
        // LLM tokens: "ab", "ac"
        // Grammar tokens: "a", "ab", "b|c", "$" (EOF)
        // Grammar: S -> X $ ; X -> "a" ("b|c") ("b|c") | "ab"
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

        let productions = vec![
            prod("S", vec![nt("X")]),
            prod("X", vec![t("A"), t("B_OR_C"), t("B_OR_C"), t("EOF")]),
            prod("X", vec![t("AB"), t("EOF")]),
        ];

        let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
        grammar_token_map.insert(Terminal("A".to_string()), TerminalID(0));
        grammar_token_map.insert(Terminal("AB".to_string()), TerminalID(1));
        grammar_token_map.insert(Terminal("B_OR_C".to_string()), TerminalID(2));
        grammar_token_map.insert(Terminal("EOF".to_string()), TerminalID(3));

        let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map);
        dbg!(&parser);

        let constraint = GrammarConstraint::new(tokenizer, parser, llm_token_map, 2);
        constraint.dump_precomputed();

        let mut constraint_state = constraint.init();

        constraint_state.step_with_all_llm_tokens();

        let mask = constraint_state.get_mask();
        assert_eq!(mask, LLMTokenBV::from_iter([true, true, false]));

        constraint_state.commit(LLMTokenID(0));
        constraint_state.step_with_all_llm_tokens();

        let mask = constraint_state.get_mask();
        assert_eq!(mask, LLMTokenBV::from_iter([false, false, true]));
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
        let constraint = GrammarConstraint::new(tokenizer, parser, llm_token_map, 6);
        constraint.dump_precomputed();

        // Initial state and step
        let mut state = constraint.init();
        state.step_with_all_llm_tokens();
        let mask = state.get_mask();
        // Expect LLM tokens that can start an expression: i (0), '(' (3), "(i" (5)
        assert_eq!(mask, LLMTokenBV::from_iter([true, false, false, true, false, true, false]));

        // Commit "i(" twice
        state.commit(LLMTokenID(5));
        state.step_with_all_llm_tokens();
        state.commit(LLMTokenID(5));
        state.step_with_all_llm_tokens();
        let mask = state.get_mask();
        assert_eq!(mask, LLMTokenBV::from_iter([false, false, false, false, false, false, false]));
    }
}
