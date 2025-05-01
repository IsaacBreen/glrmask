use std::cmp::Ordering;
use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
use crate::glr::parser::{MergeAndIntersect, GLRParser, GLRParserState, ParseState, ParseStateKey};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use bitvec::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::BitOr;
use std::sync::{Arc, Mutex};
use bitvec::macros::internal::funty::Fundamental;
use keyed_priority_queue::KeyedPriorityQueue;
use crate::constraint_extra::print_finalizer;
use crate::datastructures::charmap::TrieMap;
use crate::datastructures::gss::prune_and_transform_recursive;
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::glr::parser::ParseStateNodeContent;

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

/// Holds the set of active LLM tokens and the intersection of tokens
/// guaranteed to be possible in all future paths from this GSS node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
            active: self.active.clone() | other.active.clone(),
            intersection: self.intersection.clone() & other.intersection.clone(),
        }
    }
    fn intersect(&self, other: &Self) -> Self {
        // Intersect: Active tokens are intersected, Intersection is also intersected.
        Self {
            active: self.active.clone() & other.active.clone(),
            intersection: self.intersection.clone() & other.intersection.clone(),
        }
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
        #[derive(Debug, Copy, Clone, Eq, PartialEq)]
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

        // Create the roots.
        let mut precomputed_roots: BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>> = BTreeMap::new();
        for tokenizer_state_id in 0..tokenizer.max_state() {
            let precompute_node = Arc::new(Mutex::new(PrecomputeNode::new(PrecomputedNodeContents::default())));
            precomputed_roots.insert(TokenizerStateID(tokenizer_state_id), precompute_node);
        }

        // Queue keyed by vocab node and tokenizer state ID.
        let mut queue: BTreeMap<(DottedVocabNode, TokenizerStateID), BTreeSet<NodeHandle>> = BTreeMap::new();

        // Initialize the queue with the roots.
        for (tokenizer_state_id, precompute_node) in &precomputed_roots {
            for (bytes, new_vocab_node) in vocab_prefix_tree.root.children() {
                let dotted_new_vocab_node = DottedVocabNode { src: &vocab_prefix_tree.root, dst: new_vocab_node, bytes, offset: 0 };
                queue.insert(
                    (dotted_new_vocab_node, *tokenizer_state_id),
                    BTreeSet::from([NodeHandle(precompute_node.clone())]),
                );
            }
        }

        crate::debug!(2, "precompute main loop");
        while let Some((((dotted_vocab_node, initial_tokenizer_state_id)), precomputed_nodes)) = queue.pop_first() {
            crate::debug!(3, "Popped from queue. Queue size: {}, Precomputed nodes: {}, Prefix length: {}, Offet: {}, Total length (prefix + offset): {}, Prefix: {}, Bytes: {}",
                queue.len(),
                precomputed_nodes.len(),
                dotted_vocab_node.src.prefix_length(),
                dotted_vocab_node.offset,
                dotted_vocab_node.src.prefix_length() + dotted_vocab_node.offset,
                String::from_utf8_lossy(dotted_vocab_node.src.prefix()),
                String::from_utf8_lossy(dotted_vocab_node.bytes),
            );
            let DottedVocabNode { src, dst, offset, bytes } = dotted_vocab_node;

            let results = tokenizer.execute_from_state(&bytes[offset..], initial_tokenizer_state_id);

            for result in results.matches {
                let matched_token_id = GrammarTokenID(result.id);
                let new_offset = offset + result.width;
                // There's still more input to process. Insert trie edge(s) and update the queue.
                let new_dotted_node = DottedVocabNode { src, dst, offset: new_offset, bytes };
                let new_queue_key = (new_dotted_node, TokenizerStateID(0));
                'outer: for precompute_node in &precomputed_nodes {
                    let mut precompute_node = precompute_node.lock().unwrap();
                    let llm_tokens = dst.reachable_token_ids().clone();
                    if let Some(existing_precompute_nodes) = queue.get(&new_queue_key) {
                        // Check whether there's already an edge from this node to a node in the queue.
                        crate::debug!(3, "Trying to push to existing precompute node");
                        for existing_precompute_node in existing_precompute_nodes {
                            if let Some(existing_edge_value) = precompute_node.get_edge_value_mut(matched_token_id, existing_precompute_node) {
                                // Merge into the edge value.
                                crate::debug!(3, "Success! Merging into existing edge value");
                                continue 'outer;
                            }
                        }

                        // Try to push to an existing precompute node in the queue if it's possible to do so without creating a cycle.
                        crate::debug!(3, "Trying to insert to existing precompute node");
                        for existing_precompute_node in existing_precompute_nodes {
                            if let Ok(dst_precomputed_node) = precompute_node.try_insert(
                                matched_token_id,
                                llm_tokens.clone(),
                                Arc::clone(&*existing_precompute_node),
                            ) {
                                crate::debug!(3, "Success! Inserting into existing precompute node");
                                continue 'outer;
                            }
                        }
                    }

                    // Use any existing edge on the src node.
                    crate::debug!(3, "Trying to find existing edge value");
                    if let Some(existing_edges) = precompute_node.get_mut(&matched_token_id) {
                        if let Some((existing_edge_value, existing_dst)) = existing_edges.iter_mut().next() {
                            // Merge into the edge value.
                            crate::debug!(3, "Success! Found existing edge value");
                            *existing_edge_value = existing_edge_value.clone() | llm_tokens.clone();
                            if new_offset == bytes.len() {
                                // Reached the end of the input, so this is a clean match.
                                crate::debug!(3, "Reached the end of the input, so this is a clean match.");
                                existing_dst.lock().unwrap().value.clean_end.get_or_insert_with(|| LLMTokenBV::repeat(false, max_llm_token_id + 1)).set(dst.token_id(), true);
                                let next_src = dst;
                                for (next_bytes, next_dst) in next_src.children() {
                                    let new_dotted_node = DottedVocabNode { src: next_src, dst: next_dst, bytes: next_bytes, offset: 0 };
                                    let new_queue_key = (new_dotted_node, TokenizerStateID(0));
                                    queue.entry(new_queue_key).or_default().insert(NodeHandle(existing_dst.clone()));
                                }
                            } else if new_offset < bytes.len() {
                                crate::debug!(3, "Didn't reach end of input, so this is not a clean match");
                                queue.entry(new_queue_key).or_default().insert(NodeHandle(existing_dst.clone()));
                            } else { unreachable!(); }
                            continue 'outer;
                        }
                    }

                    // Create a new node.
                    crate::debug!(3, "Creating new precompute node");
                    let new_precomputed_node = precompute_node.force_insert(matched_token_id, llm_tokens.clone(), PrecomputedNodeContents::default());
                    if new_offset == bytes.len() {
                        // Reached the end of the input, so this is a clean match.
                        crate::debug!(3, "Reached the end of the input, so this is a clean match.");
                        new_precomputed_node.lock().unwrap().value.clean_end.get_or_insert_with(|| LLMTokenBV::repeat(false, max_llm_token_id + 1)).set(dst.token_id(), true);
                        let next_src = dst;
                        for (next_bytes, next_dst) in next_src.children() {
                            let new_dotted_node = DottedVocabNode { src: next_src, dst: next_dst, bytes: next_bytes, offset: 0 };
                            let new_queue_key = (new_dotted_node, TokenizerStateID(0));
                            queue.entry(new_queue_key).or_default().insert(NodeHandle(new_precomputed_node.clone()));
                        }
                    } else if new_offset < bytes.len() {
                        crate::debug!(3, "Didn't reach end of input, so this is not a clean match");
                        queue.entry(new_queue_key).or_default().insert(NodeHandle(new_precomputed_node.clone()));
                    } else { unreachable!(); }
                }
            }
            // Handle partial matches (end state reached before end of vocab node bytes)
            if let Some(end_state) = results.end_state {
                crate::debug!(3, "Reached end state");
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
                    queue.entry(new_queue_key).or_default().extend(precomputed_nodes.clone());
                }
            }
        }

        // Pull the roots out of their Arc<Mutex<_>>
        let precomputed_roots = precomputed_roots.into_iter().map(|(tokenizer_state_id, node)| (tokenizer_state_id, node.lock().unwrap().clone())).collect();
        precomputed_roots
    }

    pub fn init(&self) -> GrammarConstraintState<'_> {
        let initial_token_info = LLMTokenInfo {
            active: LLMTokenBV::repeat(true, self.max_llm_token_id + 1),
            // Initially, the intersection must also be all true, as no constraints have been applied.
            intersection: LLMTokenBV::repeat(true, self.max_llm_token_id + 1),
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
        let mut mask = LLMTokenBV::repeat(false, self.parent.max_llm_token_id + 1);
        for (_, state) in &self.state {
            for active_state in &state.active_states {
                mask |= active_state.stack.peek().t.active.clone();
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

    /// Prunes the GSS based on the committed token and resets the active token sets.
    pub fn commit(&mut self, llm_token_id: LLMTokenID) {
        let all_true_token_info = LLMTokenInfo {
            active: LLMTokenBV::repeat(true, self.parent.max_llm_token_id + 1),
            intersection: LLMTokenBV::repeat(true, self.parent.max_llm_token_id + 1),
        };

        // Closure for GSS transformation:
        // - Prune if token not present in 'active'.
        // - If token present:
        //   - Reset 't' to 'all_true_token_info'.
        //   - Stop recursion if token is present in 'intersection' (optimization).
        let closure = |content: &ParseStateNodeContent<LLMTokenInfo>| -> Option<(ParseStateNodeContent<LLMTokenInfo>, bool)> {
            if content.t.active[llm_token_id.0] {
                // If the intersection already guarantees this token, we can stop early.
                if content.t.intersection.all() {
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
            self.step_with_llm_token(llm_token_id);
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
                Arc::make_mut(&mut parse_state.stack).value.t.active &= llm_tokens;
                // Intersection is NOT modified here. It reflects the guarantee from *below*.
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
            // Input: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'_, LLMTokenInfo>)>
            initial_nodes_and_values,
            // step
            // Input: &GLRParserState<'_, LLMTokenInfo>, GrammarTokenID, &LLMTokenBV, &Arc<Mutex<PrecomputeNode>>
            // Output: Option<GLRParserState<'_, LLMTokenInfo>>
            |glr_parse_state, grammar_token_id, edge_llm_tokens, child_node| {
                let mut glr_parse_state = glr_parse_state.clone();
                glr_parse_state.active_states.retain_mut(|parse_state| {
                    // Intersect the *active* tokens with the edge tokens. Intersection inherits current active tokens.
                    let current_active_tokens = parse_state.stack.value.t.active.clone();
                    Arc::make_mut(&mut parse_state.stack).value.t.intersection &= current_active_tokens;
                    Arc::make_mut(&mut parse_state.stack).value.t.active &= edge_llm_tokens;
                    !parse_state.stack.value.t.active.is_empty() // Check if any active paths remain
                });
                glr_parse_state.step(*grammar_token_id);
                if glr_parse_state.active_states.is_empty() {
                    crate::debug!(3, "No active states after processing grammar token {}", grammar_token_id.0);
                    return None;
                } else {
                    crate::debug!(3, "Processed grammar token {}, {} active states.", grammar_token_id.0, glr_parse_state.active_states.len());
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
                // Handle clean end
                if let Some(clean_end) = &node.value.clean_end {
                    let mut final_glr_parse_state = glr_parse_state.clone();
                    final_glr_parse_state.active_states.retain_mut(|parse_state| {
                        // Intersect the *active* tokens with the clean_end tokens. Intersection retains current active tokens.
                        let current_active_tokens = parse_state.stack.value.t.active.clone();
                        Arc::make_mut(&mut parse_state.stack).value.t.intersection &= current_active_tokens;
                        Arc::make_mut(&mut parse_state.stack).value.t.active &= clean_end;
                        // Check if any active paths remain
                        !parse_state.stack.value.t.active.is_empty()
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
                            // Intersect *active* tokens with the finalizer's allowed tokens.
                            let mut semi_final_glr_parse_state = semi_final_glr_parse_state.clone();
                            semi_final_glr_parse_state.active_states.retain_mut(|parse_state| {
                                // Intersect the *active* tokens with the finalizer's allowed tokens. Intersection retains current active tokens.
                                let current_active_tokens = parse_state.stack.value.t.active.clone();
                                Arc::make_mut(&mut parse_state.stack).value.t.intersection &= current_active_tokens;
                                Arc::make_mut(&mut parse_state.stack).value.t.active &= llm_tokens;
                                // Check if any active paths remain
                                !parse_state.stack.value.t.active.is_empty()
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
                glr_parse_state.active_states.retain(|parse_state| !parse_state.stack.value.t.active.is_empty());
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

        // Commit "(i"
        state.commit(LLMTokenID(5));
        state.step_with_all_llm_tokens();
        let mask = state.get_mask();
        // Now expect '+', '*', ')', '+i' => IDs 1,2,4,6
        assert_eq!(mask, LLMTokenBV::from_iter([false, true, true, false, true, false, true]));

        // // Commit "(i"
        // state.commit(LLMTokenID(5));
        // state.step_with_all_llm_tokens();
        // let mask = state.get_mask();
        // assert_eq!(mask, LLMTokenBV::from_iter([false, false, false, false, false, false, false]));
    }
}
