use std::cmp::Ordering;
use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
use crate::glr::parser::ParseStateNodeContent;
use crate::glr::parser::{MergeAndIntersect, GLRParser, GLRParserState, ParseState, ParseStateKey};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use bitvec::prelude::*;
use std::collections::{BTreeMap, VecDeque}; // Add VecDeque
use std::ops::BitOr;
use std::sync::{Arc, Mutex}; // Removed Mutex, MutexGuard
use rustc_hash::{FxHashMap}; // Added FxHashMap, FxHashSet is not needed now
use bitvec::macros::internal::funty::Fundamental;
use keyed_priority_queue::KeyedPriorityQueue;
use crate::constraint_extra::print_finalizer;
use crate::datastructures::charmap::TrieMap;
use crate::datastructures::gss::prune_and_transform_recursive;
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

pub(crate) type PrecomputeNode = Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;
pub(crate) type Precomputed = BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>>;

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
            let ids: Vec<_> = bv.iter_ones().collect();
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


// Define a cheap key for the unique (tokenizer-state, vocab-location) pair
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct NodeKey {
    tok_state: TokenizerStateID,
    vocab_ptr: *const VocabPrefixTreeNode,
    offset: usize,
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

        // Use FxHashMap for cache and VecDeque for queue
        let mut node_cache: FxHashMap<NodeKey, Arc<Mutex<PrecomputeNode>>> = FxHashMap::default();
        let mut queue: VecDeque<(NodeKey, Arc<Mutex<PrecomputeNode>>)> = VecDeque::new();

        // Seed the queue with the roots.
        for tok_state in 0..tokenizer.max_state() {
            let root_pc_node = Arc::new(Mutex::new(PrecomputeNode::new(PrecomputedNodeContents::default())));

            // one NodeKey per *child* of the root prefix-tree node
            for (bytes, child) in vocab_prefix_tree.root.iter_children() {
                let key = NodeKey {
                    tok_state: TokenizerStateID(tok_state),
                    vocab_ptr: child as *const _,
                    offset: 0,
                };
                node_cache.insert(key, root_pc_node.clone());
                queue.push_back((key, root_pc_node.clone()));
            }

            precomputed_roots.insert(TokenizerStateID(tok_state), root_pc_node);
        }

        // Helper: fetch or create the next PrecomputeNode
        let mut get_or_create = |key: NodeKey,
                                 cache: &mut FxHashMap<NodeKey, Arc<Mutex<PrecomputeNode>>>,
                                 q: &mut VecDeque<(NodeKey, Arc<Mutex<PrecomputeNode>>)>|
        -> Arc<Mutex<PrecomputeNode>> {
            if let Some(node) = cache.get(&key) {
                node.clone()
            } else {
                let new_node = Arc::new(Mutex::new(PrecomputeNode::new(PrecomputedNodeContents::default())));
                cache.insert(key, new_node.clone());
                q.push_back((key, new_node.clone()));
                new_node
            }
        };


        crate::debug!(2, "precompute main loop");
        while let Some((key, src_pc_node)) = queue.pop_front() {
            let vocab_node   = unsafe { &*key.vocab_ptr };
            let bytes        = vocab_node.prefix();                // whole byte slice
            let slice        = &bytes[key.offset ..];              // where we are now

            // run the tokenizer once – cache the result locally
            let tk_results = tokenizer.execute_from_state(slice, key.tok_state);

            // concrete matches (complete grammar tokens)
            for m in &tk_results.matches {
                let g_token   = GrammarTokenID(m.id);
                let new_off   = key.offset + m.width;

                let edge_mask = vocab_node.reachable_token_ids().clone();

                let dst_key   = if new_off == bytes.len() {
                    // end of this prefix-node: jump to the *child* node
                    NodeKey { tok_state: TokenizerStateID(0),
                              vocab_ptr: vocab_node as *const _,
                              offset: new_off }
                } else {
                    // stay inside the current node
                    NodeKey { tok_state: TokenizerStateID(0),
                              vocab_ptr: vocab_node as *const _,
                              offset: new_off }
                };

                let dst_pc_node = get_or_create(dst_key, &mut node_cache, &mut queue);

                // Lock the source node to modify its children
                let mut src_pc_node_guard = src_pc_node.lock().expect("Mutex poisoned during precompute edge insertion");
                src_pc_node_guard.force_insert_to_node(
                    Some(g_token),
                    edge_mask,
                    &dst_pc_node, // dst_pc_node is now Arc<Mutex<Trie>>
                );
                // src_pc_node_guard lock released here
            }

            // partial match (tokenizer still wants more input)
            if let Some(end_state) = tk_results.end_state {
                // record every grammar token still reachable from that FA state
                // Lock the source node to modify its value (finalizers)
                let mut src_pc_node_guard = src_pc_node.lock().expect("Mutex poisoned during precompute finalizer update");
                for grammar_token in tokenizer
                    .tokens_accessible_from_state(TokenizerStateID(end_state))
                    .into_iter()
                {
                    src_pc_node.lock().unwrap().value.push_finalizer_info(
                        GrammarTokenID(grammar_token.0),
                        LLMTokenID(vocab_node.token_id()),
                        TokenizerStateID(end_state),
                        max_llm_token_id,
                    );
                }
                // src_pc_node_guard lock released here

                // enqueue all children of the current vocab-node
                for (bytes2, child2) in vocab_node.iter_children() {
                    let child_key = NodeKey {
                        tok_state: TokenizerStateID(end_state),
                        vocab_ptr: child2 as *const _,
                        offset: 0,
                    };
                    let _ = get_or_create(child_key, &mut node_cache, &mut queue);
                    // no edge added here – the *next* iteration will add it
                }
            }

            // clean-end mark
            if key.offset == bytes.len() {
                 // Lock the source node to modify its value (clean_end)
                 let mut src_pc_node_guard = src_pc_node.lock().expect("Mutex poisoned during precompute clean_end update");
                 src_pc_node_guard
                    .value
                    .clean_end
                    .get_or_insert_with(|| LLMTokenBV::repeat(false, max_llm_token_id + 1))
                    .set(vocab_node.token_id(), true);
                 // src_pc_node_guard lock released here
            }
        }

        let precomputed_roots: Precomputed = precomputed_roots
            .into_iter()
            .map(|(id, arc_mutex)| (id, arc_mutex)) // Keep Arc<Mutex<Trie>>
            .collect();
        crate::debug!(2, "Done precomputing");
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

        let mut memo = FxHashMap::default(); // Use FxHashMap
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
            initial_nodes_and_values.push((token_trie.clone(), state));
        }
        initial_nodes_and_values
    }

    pub fn step(&mut self, llm_tokens: &LLMTokenBV) {
        crate::debug!(2, "Stepping grammar constraint state with tokenizer states {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens);

        self.state = BTreeMap::new();

        Trie::special_map(
            // Input: Vec<(Arc<PrecomputeNode>, GLRParserState<'_, LLMTokenInfo>)>
            initial_nodes_and_values, // Now Vec<(Arc<Mutex<PrecomputeNode>>, ...)>
            // step
            // Input: &GLRParserState<'_, LLMTokenInfo>, GrammarTokenID, &LLMTokenBV, &Arc<PrecomputeNode>
            // Output: Option<GLRParserState<'_, LLMTokenInfo>>
            |glr_parse_state, grammar_token_id, edge_llm_tokens, child_node| {
                let node_ptr = std::ptr::addr_of!(*child_node) as usize; // Use Arc::as_ptr
                crate::debug!(3, "Processing grammar node {} token {:?} with {} active states", node_ptr, grammar_token_id.map(|grammar_token_id| grammar_token_id.0), glr_parse_state.active_states.len());
                let mut glr_parse_state = glr_parse_state.clone();
                glr_parse_state.active_states.retain_mut(|parse_state| {
                    // Intersect the *active* tokens with the edge tokens. Intersection inherits current active tokens.
                    let current_active_tokens = parse_state.stack.value.t.active.clone();
                    Arc::make_mut(&mut parse_state.stack).value.t.intersection &= current_active_tokens;
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
                let mut active_llm_tokens = LLMTokenBV::repeat(false, self.parent.max_llm_token_id + 1);
                for parse_state in &glr_parse_state.active_states {
                    active_llm_tokens |= parse_state.stack.value.t.active.clone();
                }
                crate::debug!(3, "Processing node with {} active states, {} LLM tokens, {} finalizers", glr_parse_state.active_states.len(), active_llm_tokens.count_ones(), node.value.finalizers.len());
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
                                Arc::make_mut(&mut parse_state.stack).value.t.intersection &= current_active_tokens;
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

    #[test]
    fn test_constraint_simple() {
        // LLM tokens: "ab", "ac", "$"
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

        let constraint = GrammarConstraint::new(tokenizer, parser, llm_token_map, 2);
        // Removed constraint.dump_precomputed(); as it's not part of the core logic

        let mut constraint_state = constraint.init();

        constraint_state.step_with_all_llm_tokens();

        // Initially, we can match "a" (part of "ab" or "ac") or "ab".
        // "a" leads to expecting "b" or "c".
        // "ab" leads to expecting "$".
        let mask = constraint_state.get_mask();
        assert_eq!(mask, LLMTokenBV::from_iter([true, true, false])); // Expect "ab" or "ac"

        // Commit "ab" (LLMTokenID 0)
        constraint_state.commit(LLMTokenID(0));
        constraint_state.step_with_all_llm_tokens();
        let mask = constraint_state.get_mask();
        assert_eq!(mask, LLMTokenBV::from_iter([false, false, true])); // Expect "$" (EOF)
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
        // Removed constraint.dump_precomputed();

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
