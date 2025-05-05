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
use std::sync::{Arc, Mutex}; // Added Mutex
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
pub(crate) type PrecomputeNodeArc = Arc<Mutex<PrecomputeNode>>; // New type alias
pub(crate) type Precomputed = BTreeMap<TokenizerStateID, PrecomputeNodeArc>; // Use PrecomputeNodeArc

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
        tokenizer: &Regex,
        parser: &GLRParser,
        llm_token_map: &LLMTokenMap,
        max_llm_token_id: usize
    ) -> Self {
        let precomputed = GrammarConstraint::precompute(tokenizer, llm_token_map, max_llm_token_id);
        Self {
            tokenizer: tokenizer.clone(),
            parser: parser.clone(),
            precomputed,
            llm_token_map: llm_token_map.clone(),
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
        let mut precomputed_roots: BTreeMap<TokenizerStateID, PrecomputeNodeArc> = BTreeMap::new();

        // Use FxHashMap for cache and VecDeque for queue
        let mut node_cache: FxHashMap<NodeKey, PrecomputeNodeArc> = FxHashMap::default();
        let mut queue: VecDeque<(NodeKey, PrecomputeNodeArc)> = VecDeque::new();


        // Seed the queue with the roots.
        for tok_state in 0..tokenizer.max_state() {
            let root_pc_node_arc = Arc::new(Mutex::new(PrecomputeNode::new(PrecomputedNodeContents::default())));

            // one NodeKey per *child* of the root prefix-tree node
            for (bytes, child) in vocab_prefix_tree.root.iter_children() {
                let key = NodeKey {
                    tok_state: TokenizerStateID(tok_state),
                    vocab_ptr: child as *const _,
                    offset: 0,
                };
                node_cache.insert(key, root_pc_node_arc.clone());
                queue.push_back((key, root_pc_node_arc.clone()));
            }

            precomputed_roots.insert(TokenizerStateID(tok_state), root_pc_node_arc.clone());
        }

        // Helper: fetch or create the next PrecomputeNode
        let mut get_or_create = |key: NodeKey,
                                 cache: &mut FxHashMap<NodeKey, PrecomputeNodeArc>,
                                 q: &mut VecDeque<(NodeKey, PrecomputeNodeArc)>|
        -> PrecomputeNodeArc {
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
        while let Some((key, src_pc_node_arc)) = queue.pop_front() {
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
                    // Note: TokState 0 is the initial state of the regex tokenizer
                    NodeKey { tok_state: TokenizerStateID(0),
                              vocab_ptr: vocab_node as *const _,
                              offset: new_off }
                } else {
                    // stay inside the current node
                    // Note: TokState remains 0 here because the regex tokenizer
                    // transitions to state 0 upon consuming a full match.
                    NodeKey { tok_state: TokenizerStateID(0),
                              vocab_ptr: vocab_node as *const _,
                              offset: new_off }
                };

                let dst_pc_node_arc = get_or_create(dst_key, &mut node_cache, &mut queue);

                // insert or merge the edge
                // Lock the source node to insert the edge
                { // Scope for the lock guard
                    let mut src_pc_node_guard = src_pc_node_arc.lock().expect("Mutex poisoned during precompute edge insert");
                    src_pc_node_guard.force_insert_to_node(
                        Some(g_token),
                        edge_mask.clone(),
                        &dst_pc_node_arc,
                    );
                } // Lock released here
            }

            // partial match (tokenizer still wants more input)
            if let Some(end_state) = tk_results.end_state {
                // record every grammar token still reachable from that FA state
                // Lock the source node to update finalizers
                { // Scope for the lock guard
                    let mut src_pc_node_guard = src_pc_node_arc.lock().expect("Mutex poisoned during precompute finalizer update");
                    for grammar_token in tokenizer
                        .tokens_accessible_from_state(TokenizerStateID(end_state))
                        .into_iter()
                    {
                        src_pc_node_guard.value.push_finalizer_info(
                            GrammarTokenID(grammar_token.0),
                            LLMTokenID(vocab_node.token_id()),
                            TokenizerStateID(end_state),
                            max_llm_token_id,
                        );
                    }
                } // Lock released here

                // enqueue all children of the current vocab-node
                for (_bytes2, child2) in vocab_node.iter_children() {
                    let child_key = NodeKey {
                        tok_state: TokenizerStateID(end_state),
                        vocab_ptr: child2 as *const _,
                        offset: 0,
                    };
                    // We don't add an edge from the current src_pc_node_arc here;
                    // the edge will be implicitly created when the child_key
                    // is processed from the queue in a future iteration,
                    // connecting a node representing (end_state, child2)
                    // to nodes it can reach.
                    let _ = get_or_create(child_key, &mut node_cache, &mut queue);
                }
            }

            // clean-end mark
            if key.offset == bytes.len() {
                 // Lock the source node to update clean_end
                 { // Scope for the lock guard
                     let mut src_pc_node_guard = src_pc_node_arc.lock().expect("Mutex poisoned during precompute clean_end update");
                     src_pc_node_guard
                        .value
                        .clean_end
                        .get_or_insert_with(|| LLMTokenBV::repeat(false, max_llm_token_id + 1))
                        .set(vocab_node.token_id(), true);
                 } // Lock released here
            }
        }

        // The precomputed_roots map already contains the Arc<Mutex<Trie>> roots.
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
        // Note: This only considers the 'active' tokens at the current level of the GSS stack.
        // It does not look ahead using the precomputed graph.
        for (_, state) in &self.state {
            for active_state in &state.active_states {
                // Access the value inside the Arc<Mutex> via lock
                let parse_state_content = active_state.stack.lock().expect("Mutex poisoned in get_mask").value;
                mask |= parse_state_content.t.active.clone();
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
                // Note: The original logic `content.t.intersection.all()` might be too aggressive.
                // A more precise check might be `content.t.intersection[llm_token_id.0]`.
                // Let's stick to the original for now based on the prompt's request
                // to retain existing parts where possible.
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
        self.state.retain(|_tokenizer_state_id, glr_state| { // Use _tokenizer_state_id as it's not used
            glr_state.active_states.retain_mut(|parse_state| {
                // prune_and_transform_recursive requires a mutable reference to the Arc<Mutex<Trie>>
                let mut stack_arc = parse_state.stack.clone(); // Clone the Arc
                let maybe_new_node = prune_and_transform_recursive(&mut stack_arc, &closure, &mut memo);
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

    fn prepare_initial_nodes_and_values_for_special_map(&mut self, llm_tokens: &LLMTokenBV) -> Vec<(PrecomputeNodeArc, GLRParserState<'a, LLMTokenInfo>)> {
        let mut initial_nodes_and_values: Vec<(PrecomputeNodeArc, GLRParserState<'_, LLMTokenInfo>)> = Vec::new();
        let mut tokenizer_state_id_to_parse_states: BTreeMap<TokenizerStateID, GLRParserState<'_, LLMTokenInfo>> = BTreeMap::new();

        for (tokenizer_state_id, state) in self.state.iter_mut() { // Iterate mutably
            let mut state = state.clone(); // Clone the GLRParserState for this tokenizer state
            state.active_states.retain_mut(|parse_state| {
                 // Get mutable access to the top of the stack's value (ParseStateNodeContent)
                 let stack_top_content = Arc::make_mut(&mut parse_state.stack).value;
                 // Only update the *active* tokens at the *top* of the stack.
                 // The intersection remains unchanged, and deeper nodes are untouched.
                 // The special_map logic will handle intersecting with edge_llm_tokens.
                 stack_top_content.t.active &= llm_tokens;
                 // Intersection is NOT modified here. It reflects the guarantee from *below*.
                 !stack_top_content.t.active.is_empty() // Keep the state if any active paths remain
            });
            if !state.active_states.is_empty() {
                tokenizer_state_id_to_parse_states.insert(*tokenizer_state_id, state);
            }
        }

        for (tokenizer_state_id, state) in tokenizer_state_id_to_parse_states {
            // Use the correct type PrecomputeNodeArc from the precomputed map
            let token_trie_arc = self.parent.precomputed[&tokenizer_state_id].clone();
            initial_nodes_and_values.push((token_trie_arc, state));
        }
        initial_nodes_and_values
    }

    pub fn step(&mut self, llm_tokens: &LLMTokenBV) {
        crate::debug!(2, "Stepping grammar constraint state with tokenizer states {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens);

        // Clear the current state as special_map will rebuild it
        self.state = BTreeMap::new();

        Trie::special_map(
            // Input: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'_, LLMTokenInfo>)>
            initial_nodes_and_values,
            // step
            // Input: &GLRParserState<'_, LLMTokenInfo>, Option<GrammarTokenID>, &LLMTokenBV, &PrecomputeNode
            // Output: Option<GLRParserState<'_, LLMTokenInfo>>
            |glr_parse_state, grammar_token_id_opt, edge_llm_tokens, child_trie_data| {
                let grammar_token_id = grammar_token_id_opt.clone(); // Clone Option<GrammarTokenID>
                crate::debug!(3, "Step closure: Processing edge token {:?} (filter mask: {} ones) for node (data @ {:p})", grammar_token_id.map(|id| id.0), edge_llm_tokens.count_ones(), child_trie_data);

                // 1. Clone the incoming GLR state to modify it.
                let mut next_glr_parse_state = glr_parse_state.clone();

                // 2. Perform the GLR step for the grammar token associated with this edge.
                if let Some(grammar_token_id_val) = grammar_token_id {
                    crate::debug!(4, "Step closure: Stepping GLR with grammar token {:?}", grammar_token_id_val.0);
                    next_glr_parse_state.step(grammar_token_id_val);
                    // If step resulted in no active states, prune early.
                    if next_glr_parse_state.active_states.is_empty() {
                         crate::debug!(4, "Step closure: Pruned by GLR step.");
                         return None;
                    }
                } else {
                    crate::debug!(4, "Step closure: No grammar token for this edge.");
                }


                // 3. Filter the resulting active states based on the edge_llm_tokens.
                crate::debug!(4, "Step closure: Filtering {} active states with edge mask.", next_glr_parse_state.active_states.len());
                next_glr_parse_state.active_states.retain_mut(|parse_state| {
                    // Get mutable access to the top of the stack's value (ParseStateNodeContent)
                    let stack_top_content = Arc::make_mut(&mut parse_state.stack).value;
                    // Intersect the *active* tokens with the edge tokens.
                    // The intersection field should accumulate the guarantee *before* this step.
                    stack_top_content.t.intersection &= stack_top_content.t.active.clone();
                    stack_top_content.t.active &= edge_llm_tokens;
                    // Keep the state only if there are still possible LLM tokens.
                    !stack_top_content.t.active.is_empty()
                });

                // 4. Return the modified state if it's still valid.
                if next_glr_parse_state.active_states.is_empty() {
                    crate::debug!(3, "Step closure: Pruned by edge mask filter.");
                    None
                } else {
                    crate::debug!(3, "Step closure: Resulting state has {} active states.", next_glr_parse_state.active_states.len());
                    Some(next_glr_parse_state)
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
            |trie_data, glr_parse_state| {
                crate::debug!(3, "Process closure: Entering for node (data @ {:p}) with {} active states, {} finalizers", trie_data, glr_parse_state.active_states.len(), trie_data.value.finalizers.len());

                // Merge active states first for cleaner processing below
                let mut current_glr_parse_state = glr_parse_state.clone();
                current_glr_parse_state.merge_active_states();

                let mut current_active_llm_tokens = LLMTokenBV::repeat(false, self.parent.max_llm_token_id + 1);
                for parse_state in &current_glr_parse_state.active_states {
                    // Lock to access the active tokens at the top of the stack
                    let stack_top_content = parse_state.stack.lock().expect("Mutex poisoned in process closure (getting active tokens)").value;
                    current_active_llm_tokens |= stack_top_content.t.active.clone();
                }
                 crate::debug!(3, "Process closure: {} LLM tokens active after merges.", current_active_llm_tokens.count_ones());


                // Handle clean end
                if let Some(clean_end_mask) = &trie_data.value.clean_end {
                     crate::debug!(4, "Process closure: Found clean_end mask ({} ones).", clean_end_mask.count_ones());
                    // Filter the current GLR state based on the clean_end mask
                    let mut final_glr_parse_state = current_glr_parse_state.clone();
                    final_glr_parse_state.active_states.retain_mut(|parse_state| {
                        // Lock to access and modify the active tokens at the top of the stack
                        let stack_top_content = Arc::make_mut(&mut parse_state.stack).value;
                        // Intersect the *active* tokens with the clean_end tokens. Intersection retains current active tokens.
                        let current_active_tokens = stack_top_content.t.active.clone();
                        stack_top_content.t.intersection &= current_active_tokens; // Accumulate guarantee *before* filtering
                        stack_top_content.t.active &= clean_end_mask;
                        // Keep the state if any active paths remain
                        !stack_top_content.t.active.is_empty()
                    });

                    if final_glr_parse_state.is_ok() {
                        crate::debug!(4, "Process closure: Clean end state is OK ({} active states). Merging into tokenizer state 0.", final_glr_parse_state.active_states.len());
                        if let Some(existing) = self.state.get_mut(&TokenizerStateID(0)) {
                            existing.merge_with(final_glr_parse_state.clone());
                        } else {
                            self.state.insert(TokenizerStateID(0), final_glr_parse_state.clone());
                        }
                    } else {
                        crate::debug!(4, "Process closure: Clean end state is NOT OK.");
                    }
                }

                // Handle finalizers
                for (possible_final_grammar_token, precomputed_finalizer) in &trie_data.value.finalizers {
                     crate::debug!(4, "Process closure: Found finalizer for grammar token {:?}.", possible_final_grammar_token.0);
                    // Step the current GLR state with the possible final grammar token
                    let mut possible_next_glr_parse_state = current_glr_parse_state.clone();
                    possible_next_glr_parse_state.step(*possible_final_grammar_token);

                    if possible_next_glr_parse_state.is_ok() {
                        crate::debug!(5, "Process closure: Stepping with grammar token {:?} is OK ({} active states).", possible_final_grammar_token.0, possible_next_glr_parse_state.active_states.len());
                        for (tokenizer_state_id, llm_tokens_mask) in &precomputed_finalizer.content {
                             crate::debug!(5, "Process closure: Considering finalizer to tokenizer state {:?} with LLM mask ({} ones).", tokenizer_state_id.0, llm_tokens_mask.count_ones());
                            // Filter the current GLR state based on the finalizer's allowed LLM tokens
                            let mut glr_parse_state_filtered = current_glr_parse_state.clone();
                            glr_parse_state_filtered.active_states.retain_mut(|parse_state| {
                                // Lock to access and modify the active tokens at the top of the stack
                                let stack_top_content = Arc::make_mut(&mut parse_state.stack).value;
                                // Intersect the *active* tokens with the finalizer's allowed tokens. Intersection retains current active tokens.
                                let current_active_tokens = stack_top_content.t.active.clone();
                                stack_top_content.t.intersection &= current_active_tokens; // Accumulate guarantee *before* filtering
                                stack_top_content.t.active &= llm_tokens_mask;
                                // Keep the state if any active paths remain
                                !stack_top_content.t.active.is_empty()
                            });

                            if glr_parse_state_filtered.is_ok() {
                                crate::debug!(5, "Process closure: Finalizer is compatible. Merging into tokenizer state {:?}.", tokenizer_state_id.0);
                                if let Some(existing) = self.state.get_mut(tokenizer_state_id) {
                                    existing.merge_with(glr_parse_state_filtered.clone());
                                } else {
                                    self.state.insert(*tokenizer_state_id, glr_parse_state_filtered.clone());
                                }
                            } else {
                                crate::debug!(5, "Process closure: Finalizer is NOT compatible.");
                            }
                        }
                    } else {
                        crate::debug!(5, "Process closure: Stepping with grammar token {:?} is NOT OK.", possible_final_grammar_token.0);
                    }
                }

                // Check if the current GLR state still has valid paths before continuing traversal
                // (This check might be redundant if the retain calls above handle it)
                 !current_glr_parse_state.active_states.is_empty()
            },
        );
        crate::debug!(2, "Done stepping, new tokenizer states: {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());
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
        // Grammar: S -> X $ ; X -> "a" ("b|c") | "ab"
        let expr = groups![
            eat_u8(b'a'), // ID 0
            seq![eat_u8(b'a'), eat_u8(b'b')], // ID 1
            choice![eat_u8(b'b'), eat_u8(b'c')], // ID 2
            eat_u8(b'$'), // ID 3
        ];
        let tokenizer = expr.build();

        let mut llm_token_map = LLMTokenMap::new();
        llm_token_map.insert(b"ab".to_vec(), LLMTokenID(0)); // Corresponds to tokenizer match seq![a,b] (ID 1)
        llm_token_map.insert(b"ac".to_vec(), LLMTokenID(1)); // Corresponds to tokenizer matches 'a' (ID 0) and 'c' (part of ID 2)
        llm_token_map.insert(b"$".to_vec(), LLMTokenID(2));  // Corresponds to tokenizer match '$' (ID 3)

        // Grammar Terminals mapped to Tokenizer IDs
        let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
        grammar_token_map.insert(Terminal("A".to_string()), TerminalID(0)); // Corresponds to eat_u8(b'a')
        grammar_token_map.insert(Terminal("AB_TERM".to_string()), TerminalID(1)); // Corresponds to seq![eat_u8(b'a'), eat_u8(b'b')]
        grammar_token_map.insert(Terminal("B_OR_C_TERM".to_string()), TerminalID(2)); // Corresponds to choice![eat_u8(b'b'), eat_u8(b'c')]
        grammar_token_map.insert(Terminal("EOF".to_string()), TerminalID(3)); // Corresponds to eat_u8(b'$')

        let productions = vec![
            prod("S", vec![nt("X"), t("EOF")]), // S -> X $ (Prod ID 0)
            prod("X", vec![t("A"), t("B_OR_C_TERM")]), // X -> a (b|c) (Prod ID 1)
            prod("X", vec![t("AB_TERM")]),             // X -> ab (Prod ID 2)
        ];

        let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map); // Start production is ID 0 (S)
        dbg!(&parser);

        let constraint = GrammarConstraint::new(&tokenizer, &parser, &llm_token_map, 2);
        // Removed constraint.dump_precomputed(); as it's not part of the core logic

        let mut constraint_state = constraint.init();

        // Initial state: Expecting tokens that can start an S rule (which is X).
        // X can start with 'A' (tokenizer ID 0) or 'AB_TERM' (tokenizer ID 1).
        // LLM tokens:
        // "ab" (LLM ID 0) -> matches 'AB_TERM' (tokenizer ID 1) - Possible
        // "ac" (LLM ID 1) -> matches 'A' (tokenizer ID 0) partially - Possible
        // "$"  (LLM ID 2) -> matches '$' (tokenizer ID 3) - Not possible at the start
        constraint_state.step_with_all_llm_tokens();

        let mask = constraint_state.get_mask();
        assert_eq!(mask, LLMTokenBV::from_iter([true, true, false]), "After initial step, should allow 'ab' (0) and 'ac' (1)"); // Expect "ab" or "ac"

        // Commit "ab" (LLMTokenID 0)
        // "ab" maps to tokenizer ID 1 ("AB_TERM").
        // The GLR parser should consume "AB_TERM" and reach a state expecting "EOF".
        // "EOF" maps to tokenizer ID 3 ("$").
        constraint_state.commit(LLMTokenID(0));
        constraint_state.step_with_all_llm_tokens();
        let mask = constraint_state.get_mask();
        // Expect "$" (LLM ID 2) which matches tokenizer ID 3 ("$")
        assert_eq!(mask, LLMTokenBV::from_iter([false, false, true]), "After committing 'ab', should allow only '$' (2)"); // Expect "$" (EOF)

        // Commit "$" (LLMTokenID 2)
        // "$" maps to tokenizer ID 3 ("$").
        // The GLR parser should consume "EOF" and complete the "S" rule.
        constraint_state.commit(LLMTokenID(2));
        constraint_state.step_with_all_llm_tokens();
        let mask = constraint_state.get_mask();
        // Expect no further tokens allowed, as the grammar is complete.
         assert_eq!(mask, LLMTokenBV::repeat(false, constraint.max_llm_token_id + 1), "After committing '$', should allow no tokens");
    }

     #[test]
    fn test_constraint_simple_ac() {
        // Same setup as test_constraint_simple but commit "ac"
        let expr = groups![
            eat_u8(b'a'), // ID 0
            seq![eat_u8(b'a'), eat_u8(b'b')], // ID 1
            choice![eat_u8(b'b'), eat_u8(b'c')], // ID 2
            eat_u8(b'$'), // ID 3
        ];
        let tokenizer = expr.build();

        let mut llm_token_map = LLMTokenMap::new();
        llm_token_map.insert(b"ab".to_vec(), LLMTokenID(0)); // Corresponds to tokenizer match seq![a,b] (ID 1)
        llm_token_map.insert(b"ac".to_vec(), LLMTokenID(1)); // Corresponds to tokenizer matches 'a' (ID 0) and 'c' (part of ID 2)
        llm_token_map.insert(b"$".to_vec(), LLMTokenID(2));  // Corresponds to tokenizer match '$' (ID 3)

        // Grammar Terminals mapped to Tokenizer IDs
        let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
        grammar_token_map.insert(Terminal("A".to_string()), TerminalID(0)); // Corresponds to eat_u8(b'a')
        grammar_token_map.insert(Terminal("AB_TERM".to_string()), TerminalID(1)); // Corresponds to seq![eat_u8(b'a'), eat_u8(b'b')]
        grammar_token_map.insert(Terminal("B_OR_C_TERM".to_string()), TerminalID(2)); // Corresponds to choice![eat_u8(b'b'), eat_u8(b'c')]
        grammar_token_map.insert(Terminal("EOF".to_string()), TerminalID(3)); // Corresponds to eat_u8(b'$')

        let productions = vec![
            prod("S", vec![nt("X"), t("EOF")]), // S -> X $ (Prod ID 0)
            prod("X", vec![t("A"), t("B_OR_C_TERM")]), // X -> a (b|c) (Prod ID 1)
            prod("X", vec![t("AB_TERM")]),             // X -> ab (Prod ID 2)
        ];

        let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map); // Start production is ID 0 (S)
        let constraint = GrammarConstraint::new(&tokenizer, &parser, &llm_token_map, 2);

        let mut constraint_state = constraint.init();

        constraint_state.step_with_all_llm_tokens();
        let mask = constraint_state.get_mask();
        assert_eq!(mask, LLMTokenBV::from_iter([true, true, false]), "After initial step, should allow 'ab' (0) and 'ac' (1)"); // Expect "ab" or "ac"

        // Commit "ac" (LLMTokenID 1)
        // "ac" does NOT match a single tokenizer token.
        // It should match 'a' (tokenizer ID 0) first,
        // then the partial match 'c' should match part of 'b|c' (tokenizer ID 2).
        // The GLR parser should consume 'A', then expect 'B_OR_C_TERM'.
        // After committing "ac", the constraint state should be in a tokenizer state
        // that has consumed 'a' and is looking for 'c' to complete 'b|c',
        // and simultaneously the GLR state should be expecting 'B_OR_C_TERM'.
        constraint_state.commit(LLMTokenID(1)); // This should trigger the tokenizer to process "ac"

        // After committing "ac" (which matches 'a' and then 'c' via partial),
        // the tokenizer state should be 0 (initial state, after consuming full token(s)).
        // The GLR state should be expecting 'B_OR_C_TERM'.
        // The next possible LLM token should be "$" which matches tokenizer ID 3 "EOF"
        // allowing the S -> X $ rule to complete after X -> a B_OR_C_TERM.
        constraint_state.step_with_all_llm_tokens();
        let mask = constraint_state.get_mask();
        // Expect "$" (LLM ID 2)
         assert_eq!(mask, LLMTokenBV::from_iter([false, false, true]), "After committing 'ac', should allow only '$' (2)");

         // Commit "$" (LLMTokenID 2)
        // "$" maps to tokenizer ID 3 ("$").
        // The GLR parser should consume "EOF" and complete the "S" rule.
        constraint_state.commit(LLMTokenID(2));
        constraint_state.step_with_all_llm_tokens();
        let mask = constraint_state.get_mask();
        // Expect no further tokens allowed, as the grammar is complete.
         assert_eq!(mask, LLMTokenBV::repeat(false, constraint.max_llm_token_id + 1), "After committing '$', should allow no tokens");
    }


    #[test]
    fn test_constraint_expression() {
        // Example grammar: E -> E '+' T | T; T -> T '*' F | F; F -> '(' E ')' | 'i'
        // LLM token vocabulary: i, +, *, (, ), (i, +i
        let mut llm_token_map = LLMTokenMap::new();
        llm_token_map.insert(b"i".to_vec(), LLMTokenID(0)); // Tokenizer ID 4
        llm_token_map.insert(b"+".to_vec(), LLMTokenID(1)); // Tokenizer ID 0
        llm_token_map.insert(b"*".to_vec(), LLMTokenID(2)); // Tokenizer ID 1
        llm_token_map.insert(b"(".to_vec(), LLMTokenID(3)); // Tokenizer ID 2
        llm_token_map.insert(b")".to_vec(), LLMTokenID(4)); // Tokenizer ID 3
        llm_token_map.insert(b"(i".to_vec(), LLMTokenID(5)); // Matches '(' (ID 2) then 'i' (ID 4)
        llm_token_map.insert(b"+i".to_vec(), LLMTokenID(6)); // Matches '+' (ID 0) then 'i' (ID 4)

        // Tokenizer regex for grammar tokens '+' '*' '(' ')' 'i'
        let expr = groups![
            eat_u8(b'+'), // ID 0
            eat_u8(b'*'), // ID 1
            eat_u8(b'('), // ID 2
            eat_u8(b')'), // ID 3
            eat_u8(b'i'), // ID 4
        ];
        let tokenizer = expr.build();

        // Grammar productions
        let productions = vec![
            prod("S", vec![nt("E"), t("EOF")]), // Start production (Prod ID 0)
            prod("E", vec![nt("E"), t("PLUS"), nt("T")]), // Prod ID 1
            prod("E", vec![nt("T")]), // Prod ID 2
            prod("T", vec![nt("T"), t("TIMES"), nt("F")]), // Prod ID 3
            prod("T", vec![nt("F")]), // Prod ID 4
            prod("F", vec![t("LPAREN"), nt("E"), t("RPAREN")]), // Prod ID 5
            prod("F", vec![t("I")]), // Prod ID 6
        ];
        // Map grammar terminals to IDs matching regex order
        let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
        grammar_token_map.insert(Terminal("PLUS".to_string()), TerminalID(0));
        grammar_token_map.insert(Terminal("TIMES".to_string()), TerminalID(1));
        grammar_token_map.insert(Terminal("LPAREN".to_string()), TerminalID(2));
        grammar_token_map.insert(Terminal("RPAREN".to_string()), TerminalID(3));
        grammar_token_map.insert(Terminal("I".to_string()), TerminalID(4));
        grammar_token_map.insert(Terminal("EOF".to_string()), TerminalID(5)); // Assuming EOF is grammar token ID 5

        let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map); // Start production is ID 0 (S)
        dbg!(&parser);
        let constraint = GrammarConstraint::new(&tokenizer, &parser, &llm_token_map, 6);
        // Removed constraint.dump_precomputed();

        // Initial state and step
        let mut state = constraint.init();
        state.step_with_all_llm_tokens();
        let mask = state.get_mask();
        // Expect LLM tokens that can start an expression (an E rule):
        // E -> T -> F -> '(' E ')' (requires LLM token '(' (ID 3), or "(i" (ID 5) which starts with '(')
        // E -> T -> F -> 'i' (requires LLM token 'i' (ID 0))
        assert_eq!(mask, LLMTokenBV::from_iter([true, false, false, true, false, true, false]), "Initial mask: should allow 'i' (0), '(' (3), '(i' (5)");

        // Commit "(i" (LLMTokenID 5)
        // "(i" matches tokenizer tokens '(' (ID 2) and then 'i' (ID 4).
        // The GLR parser should consume 'LPAREN', then 'I'.
        // After '(' the parser expects E. After 'i' (as an F, then a T), the parser expects ')' or an operator.
        state.commit(LLMTokenID(5));
        state.step_with_all_llm_tokens();
        let mask = state.get_mask();
        // Now inside F -> ( E )
        // Expect tokens that can follow 'i' inside parentheses:
        // '+', '*', or ')'
        // '+' (LLM ID 1) matches tokenizer ID 0 ('+').
        // '*' (LLM ID 2) matches tokenizer ID 1 ('*').
        // ')' (LLM ID 4) matches tokenizer ID 3 (')').
        // '+i' (LLM ID 6) starts with '+', matches tokenizer ID 0 ('+').
        assert_eq!(mask, LLMTokenBV::from_iter([false, true, true, false, true, false, true]), "After committing '(i', should allow '+', '*', ')', '+i'");

        // Commit "+" (LLMTokenID 1)
        // "+" matches tokenizer ID 0.
        // Parser consumes PLUS, expects T.
        state.commit(LLMTokenID(1));
        state.step_with_all_llm_tokens();
        let mask = state.get_mask();
        // Now after 'i +', inside E -> E + T, expecting T.
        // T can start with F. F can start with '(' or 'i'.
        // Expect tokens that can start an F: 'i' (LLM ID 0), '(' (LLM ID 3), '(i' (LLM ID 5)
         assert_eq!(mask, LLMTokenBV::from_iter([true, false, false, true, false, true, false]), "After committing '+', should allow 'i', '(', '(i'");

         // Commit "i" (LLMTokenID 0)
         // "i" matches tokenizer ID 4.
         // Parser consumes 'I', reduces to F, reduces to T. Now has E + T. Expects operator or ')'.
         state.commit(LLMTokenID(0));
         state.step_with_all_llm_tokens();
         let mask = state.get_mask();
         // Now after 'i + i', inside E -> E + T, expecting operator or ')'.
         // This is inside F -> ( E )
         // Expect '+', '*', or ')'
         // '+' (LLM ID 1), '*' (LLM ID 2), ')' (LLM ID 4), '+i' (LLM ID 6)
         assert_eq!(mask, LLMTokenBV::from_iter([false, true, true, false, true, false, true]), "After committing 'i' after '+', should allow '+', '*', ')', '+i'");

        // Commit ")" (LLMTokenID 4)
        // ")" matches tokenizer ID 3.
        // Parser consumes 'RPAREN', reduces F. Inside E -> E + T, has E + T F. Reduces T. Has E + T T. Reduces E. Has E E. Wait, grammar...
        // After '( i + i )', the parser has reduced 'i' to F, then T. Has ( E + T. Consumes ')', reduces F. Has ( E F ). Reduces E. Has ( E ). Reduces F. Reduces T. Reduces E. Reduces S if $ is next.
        // The state is now ready to accept EOF.
        state.commit(LLMTokenID(4));
        state.step_with_all_llm_tokens();
        let mask = state.get_mask();
        // Expect EOF ($), LLM ID 2.
        assert_eq!(mask, LLMTokenBV::from_iter([false, false, true, false, false, false, false]), "After committing ')', should allow only '$'");

        // Commit "$" (LLMTokenID 2)
        // "$" matches tokenizer ID 5 (EOF).
        // Parser consumes EOF, reduces S. Grammar complete.
        state.commit(LLMTokenID(2));
        state.step_with_all_llm_tokens();
        let mask = state.get_mask();
        // Expect no tokens.
        assert_eq!(mask, LLMTokenBV::repeat(false, constraint.max_llm_token_id + 1), "After committing '$', should allow no tokens");
    }
}
