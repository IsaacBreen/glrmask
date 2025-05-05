use std::cmp::Ordering;
use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
use crate::glr::parser::ParseStateNodeContent;
use crate::glr::parser::{MergeAndIntersect, GLRParser, GLRParserState, ParseState, ParseStateKey};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use bitvec::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ops::BitOr;
use std::sync::{Arc, Mutex, MutexGuard};
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

// ---- Ad-hoc helpers for precompute ----------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct PartialToken<'a> {
    src: &'a VocabPrefixTreeNode,
    dst: &'a VocabPrefixTreeNode,
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PartialToken<'a> {
    #[inline] fn key(&self) -> (usize, *const (), *const u8) {
        (self.src.prefix_length() + self.offset,
         self.src as *const _ as *const (),
         self.bytes.as_ptr())
    }
}

type SharedNode = Arc<Mutex<PrecomputeNode>>;

fn ptr_id(node: &SharedNode) -> usize { Arc::as_ptr(node) as usize }

impl PrecomputeNode {
    fn link_to(
        &mut self,
        matched_token_id: Option<GrammarTokenID>,
        dst_node: &SharedNode,
        extra_tokens: &LLMTokenBV,
    ) {
        let llm_tokens = extra_tokens.clone();
        if let Some(existing_edges) = self.get_mut(&matched_token_id) {
            if let Some((existing_edge_llm_tokens, existing_precomputed_node)) = existing_edges.iter_mut().next() {
                // Merge into the edge value.
                *existing_edge_llm_tokens |= llm_tokens;
                return;
            }
        }

        // Create a new node.
        self.force_insert_to_new_node(matched_token_id, llm_tokens.clone(), PrecomputedNodeContents::default());
    }
}

// ----------------------------------------------------------------------------

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

fn build_vocab_tree(llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>) -> VocabPrefixTree {
    let mut tokens_for_vocab_prefix_tree_builder: Vec<(usize, Vec<u8>)> = vec![];
    for (content, id) in llm_token_map {
        tokens_for_vocab_prefix_tree_builder.push((id.0, content.clone()));
    }
    crate::debug!(2, "Building vocab prefix tree");
    let vocab_prefix_tree = VocabPrefixTree::build(&tokens_for_vocab_prefix_tree_builder);
    crate::debug!(2, "Done building vocab prefix tree");
    vocab_prefix_tree
}

fn build_root_nodes(max_tokenizer_state: usize) -> BTreeMap<TokenizerStateID, SharedNode> {
    let mut precomputed_roots: BTreeMap<TokenizerStateID, SharedNode> = BTreeMap::new();
    for tokenizer_state_id in 0..max_tokenizer_state {
        let precompute_node = Arc::new(Mutex::new(PrecomputeNode::new(PrecomputedNodeContents::default())));
        precomputed_roots.insert(TokenizerStateID(tokenizer_state_id), precompute_node);
    }
    precomputed_roots
}

fn seed_queue<'a>(
    vocab: &'a VocabPrefixTree,
    roots: &BTreeMap<TokenizerStateID, SharedNode>
) -> VecDeque<(PartialToken<'a>, TokenizerStateID, SharedNode)> {
    let mut queue: VecDeque<(PartialToken<'a>, TokenizerStateID, SharedNode)> = VecDeque::new();
    for (tokenizer_state_id, precompute_node) in roots {
        for (bytes, new_vocab_node) in vocab.root.iter_children() {
            let partial = PartialToken { src: &vocab.root, dst: new_vocab_node, bytes, offset: 0 };
            queue.push_back((partial, *tokenizer_state_id, Arc::clone(precompute_node)));
        }
    }
    queue
}

fn process_queue_item<'a>(
    tokenizer: &Regex,
    vocab:     &VocabPrefixTree,
    max_id:    usize,
    queue:     &mut VecDeque<(PartialToken<'a>, TokenizerStateID, SharedNode)>,
    // merge_map is no longer needed with the VecDeque approach
    _merge_map: &mut HashMap<usize, SharedNode>, // Keep for now to match signature but unused
    (dotted_vocab_node, initial_tokenizer_state_id, current_node): (PartialToken<'a>, TokenizerStateID, SharedNode),
) {
    let PartialToken { src, dst, offset, bytes } = dotted_vocab_node;

    let results = tokenizer.execute_from_state(&bytes[offset..], initial_tokenizer_state_id);

    let mut node = current_node.lock().unwrap();

    for result in results.matches {
        let matched_token_id = GrammarTokenID(result.id);
        let new_offset = offset + result.width;
        let llm_tokens = dst.reachable_token_ids().clone();

        if new_offset == bytes.len() {
            let next_src = dst;
            // Reached the end of the input, so this is a clean match.
            // Add edge to a new node representing the clean end
            let mut next_precompute_node = node.force_insert_to_new_node(Some(matched_token_id), llm_tokens.clone(), PrecomputedNodeContents::default());
            next_precompute_node.lock().unwrap().value.clean_end.get_or_insert_with(|| LLMTokenBV::repeat(false, max_id + 1)).set(dst.token_id(), true);

            // From the clean end node, add edges for the children of the vocab node
            let clean_end_node = Arc::clone(&next_precompute_node);
            for (next_bytes, next_dst) in next_src.iter_children() {
                let new_dotted_node = PartialToken { src: next_src, dst: next_dst, bytes: next_bytes, offset: 0 };
                let new_queue_key_partial = PartialToken { src: next_src, dst: next_dst, bytes: next_bytes, offset: 0 };
                let new_queue_key_tokenizer = TokenizerStateID(0); // Start from tokenizer initial state
                let llm_tokens = next_dst.reachable_token_ids().clone();

                let mut clean_end_node_locked = clean_end_node.lock().unwrap();
                clean_end_node_locked.link_to(
                    Some(matched_token_id), // This should likely be None or a special terminal for clean end
                    &current_node, // Not linking to self, need to link to new nodes from here
                    &llm_tokens
                );
                // Need to find or create the node that corresponds to (new_dotted_node, new_queue_key_tokenizer)
                // This requires a mapping or different queue structure.
                // For now, let's simplify and just add to the queue with the current node as the base
                // This is incorrect. The new node should be the start of the next trie path.

                // Revised approach: Link the current node to the clean end node.
                // The clean end node then needs to become the root of subsequent token processing.
                // This suggests the queue should perhaps hold (PartialToken, TokenizerStateID, SharedNode) where the SharedNode is the *parent* node
                // to which we are linking the result of processing PartialToken with TokenizerStateID.

                // Let's revisit the linking logic in precompute. The original link_next_precompute_node
                // was trying to link the `precompute_node` (which was the parent(s) from the queue)
                // to a *new* or *existing* next node.
                // With the VecDeque, the `current_node` is the parent.

                let new_dotted_node = PartialToken { src: next_src, dst: next_dst, bytes: next_bytes, offset: 0 };
                let new_tokenizer_state = TokenizerStateID(0);
                let llm_tokens = next_dst.reachable_token_ids().clone();

                // Need to find or create the actual PrecomputeNode that (new_dotted_node, new_tokenizer_state) maps to.
                // This requires a different lookup mechanism than the queue itself.

                // Let's step back and rethink the queue item and linking logic.
                // The queue should contain (PartialToken, TokenizerStateID, SharedNode), where SharedNode is the node
                // in the *precompute tree* that we are currently extending edges *from*.
                // The PartialToken and TokenizerStateID describe the "input" being processed.

                // So the current `(dotted_vocab_node, initial_tokenizer_state_id, current_node)` is correct.
                // We execute the tokenizer on `dotted_vocab_node.bytes[offset..]` from `initial_tokenizer_state_id`.
                // This gives us matches. Each match has a `result.width` and `result.id`.
                // The new offset is `offset + result.width`.
                // The `matched_token_id` is `GrammarTokenID(result.id)`.
                // The LLM tokens are those reachable from `dotted_vocab_node.dst`.

                // If `new_offset == bytes.len()`, the partial token is fully consumed by the tokenizer match.
                // This means the vocab node `dst` has been fully matched by the grammar token `matched_token_id`.
                // The precompute node `current_node` should have an edge `Some(matched_token_id)` leading to a new state.
                // This new state corresponds to being at `dst` in the vocab tree and `TokenizerStateID(0)` (initial state) in the tokenizer.
                // The tokens reachable from this point are the children of `dst` in the vocab tree.

                // Let's retry the logic for `new_offset == bytes.len()`

                let next_vocab_src = dst; // We are now conceptually at the end of the matched vocab prefix
                let next_tokenizer_state = TokenizerStateID(0); // Tokenizer resets

                // We need to link `current_node` with edge `Some(matched_token_id)` to a node that represents
                // the state of being at `next_vocab_src` in the vocab tree and `next_tokenizer_state` in the tokenizer.
                // This requires finding or creating the appropriate `PrecomputeNode` for this pair.
                // The `Precomputed` map is keyed by `TokenizerStateID`, but the inner trie is structured by `GrammarTokenID` edges.
                // This still feels like we need a way to map `(VocabPrefixTreeNode, TokenizerStateID)` pairs to `SharedNode`.

                // Let's use a map `node_map: HashMap<(usize, TokenizerStateID), SharedNode>` where the key is `(vocab_node_ptr_id, tokenizer_state_id)`.
                let mut node_map: HashMap<(usize, TokenizerStateID), SharedNode> = HashMap::new();
                // Initialize node_map with roots
                for (tokenizer_state_id, root_node) in roots.iter() {
                    node_map.insert((vocab.root.ptr_id(), *tokenizer_state_id), Arc::clone(root_node));
                }


                // Inside the loop, when processing (PartialToken { src, dst, offset, bytes }, initial_tokenizer_state_id, current_node):
                // current_node corresponds to (src, initial_tokenizer_state_id) in the node_map.

                // If `new_offset == bytes.len()` (full match of partial token):
                // The next state is at `dst` in the vocab tree and `TokenizerStateID(0)` in the tokenizer.
                let next_vocab_state = dst;
                let next_tokenizer_state = TokenizerStateID(0);
                let next_key = (next_vocab_state.ptr_id(), next_tokenizer_state);

                let next_precompute_node = node_map.entry(next_key).or_insert_with(|| {
                    Arc::new(Mutex::new(PrecomputeNode::new(PrecomputedNodeContents::default())))
                });

                // Link `current_node` to `next_precompute_node` with edge `Some(matched_token_id)`.
                let llm_tokens_at_next = next_vocab_state.reachable_token_ids().clone(); // Tokens starting from `dst`

                node.link_to(Some(matched_token_id), next_precompute_node, &llm_tokens_at_next);

                // Mark the `next_precompute_node` as a clean end for the token `dst.token_id()`
                next_precompute_node.lock().unwrap().value.clean_end.get_or_insert_with(|| LLMTokenBV::repeat(false, max_id + 1)).set(dst.token_id(), true);

                // Now, the queue should be seeded with the *children* of `next_vocab_state` starting from `next_precompute_node` and `next_tokenizer_state`.
                for (next_bytes, next_vocab_node) in next_vocab_state.iter_children() {
                    let new_partial = PartialToken { src: next_vocab_state, dst: next_vocab_node, bytes: next_bytes, offset: 0 };
                    queue.push_back((new_partial, next_tokenizer_state, Arc::clone(next_precompute_node)));
                }
            }


            // Original logic for new_offset == bytes.len():
            // let next_src = dst;
            // if let Some(mut next_precompute_node) = link_next_precompute_node(&mut queue, new_queue_key, &mut precompute_node, Some(matched_token_id)) {
            //     next_precompute_node.lock().unwrap().value.clean_end.get_or_insert_with(|| LLMTokenBV::repeat(false, max_llm_token_id + 1)).set(dst.token_id(), true);
            // }
            // // Reached the end of the input, so this is a clean match.
            // crate::debug!(4, "Reached the end of the input, so this is a clean match.");
            // for (next_bytes, next_dst) in next_src.iter_children() {
            //     let new_dotted_node = DottedVocabNode { src: next_src, dst: next_dst, bytes: next_bytes, offset: 0 };
            //     let new_queue_key = (new_dotted_node, TokenizerStateID(0));
            //     if let Some(mut next_precompute_node) = link_next_precompute_node(&mut queue, new_queue_key, &mut precompute_node, Some(matched_token_id)) {
            //         next_precompute_node.lock().unwrap().value.clean_end.get_or_insert_with(|| LLMTokenBV::repeat(false, max_llm_token_id + 1)).set(dst.token_id(), true);
            //         // queue.entry(new_queue_key).or_default().insert(NodeHandle(next_precompute_node.clone())); // No longer BTreeSet
            //         // How to add to queue here? The link_next_precompute_node should return the node, and we queue (new_dotted_node, tokenizer_state, returned_node)
            //     }
            // }

        } else if new_offset < bytes.len() {
            // Partial match, still more input to process in the current partial token.
            let new_dotted_node = PartialToken { src, dst, offset: new_offset, bytes };
            let new_tokenizer_state = TokenizerStateID(result.new_state);
            let llm_tokens = dst.reachable_token_ids().clone(); // Tokens reachable from the original vocab node

            // We need to link `current_node` with edge `Some(matched_token_id)` to a node representing
            // the state of being at `dst` in the vocab tree (with offset `new_offset`) and `new_tokenizer_state` in the tokenizer.
            // This is still tricky. The `PrecomputeNode` trie edges are `GrammarTokenID`.
            // The nodes represent a state in the grammar constraint traversal.
            // What state? A position in the vocab prefix being matched, and a state in the tokenizer.

            // Let's reconsider the meaning of a node in the `PrecomputeNode` trie.
            // A node represents a set of (ParserState, TokenizerStateID, VocabPrefixTreeNode) tuples?
            // This seems overly complex.

            // Let's go back to the original structure. `Precomputed` is `BTreeMap<TokenizerStateID, PrecomputeNode>`.
            // The outer map is by tokenizer state. The inner `PrecomputeNode` is a trie of `Option<GrammarTokenID>`.
            // This suggests a `PrecomputeNode` is specific to a `TokenizerStateID`.
            // The edges in the `PrecomputeNode` trie are `GrammarTokenID`.
            // The values in the `PrecomputeNode` trie are `(LLMTokenBV, SharedNode)`.
            // This implies that a path through a `PrecomputeNode` trie, given a starting `TokenizerStateID`,
            // corresponds to consuming a sequence of `GrammarTokenID`s and accumulating `LLMTokenBV` constraints,
            // leading to a *next* `PrecomputeNode`.

            // The structure seems to represent: `(CurrentTokenizerStateID) --[GrammarTokenID edge, LLMTokenBV constraint]--> (NextPrecomputeNode)`
            // But the `NextPrecomputeNode` needs to correspond to a new state combination.

            // Let's revisit the queue item: `(PartialToken<'a>, TokenizerStateID, SharedNode)`
            // `PartialToken` describes the LLM token prefix being processed.
            // `TokenizerStateID` is the current state of the tokenizer after consuming previous parts.
            // `SharedNode` is the node in the *precompute trie* that we are currently expanding.

            // This structure implies that `SharedNode` represents the state derived from the *grammar parsing*.
            // The `GLRParserState` is stored in the `value` of the GSS nodes, which are the values *in* the `PrecomputeNode`.
            // This is getting confusing.

            // Let's look at `GrammarConstraintState::step`.
            // It iterates through `self.state: BTreeMap<TokenizerStateID, GLRParserState>`.
            // It prepares `initial_nodes_and_values`: `Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState)>`.
            // The `PrecomputeNode` comes from `self.parent.precomputed[&tokenizer_state_id]`.
            // So, the `PrecomputeNode` is indeed rooted at a `TokenizerStateID`.

            // The `Trie::special_map` function takes these pairs and processes them.
            // The `step` closure receives `(glr_parse_state, grammar_token_id, edge_llm_tokens, child_node)`.
            // `glr_parse_state` is the state propagated from the parent.
            // `grammar_token_id` is the edge label.
            // `edge_llm_tokens` is the value on the edge.
            // `child_node` is the destination node of the edge.

            // This means the nodes in the `PrecomputeNode` trie represent the state *after* consuming a sequence of grammar tokens,
            // starting from a root defined by a `TokenizerStateID`.

            // So, the `precompute` function builds these tries. A node in the trie corresponds to a sequence of grammar tokens.
            // The value in the node (`PrecomputedNodeContents`) contains `finalizers` and `clean_end`.
            // `finalizers` maps `GrammarTokenID` to `PrecomputedFinalizer`, which maps `TokenizerStateID` to `LLMTokenBV`.
            // This means: from this node (after matching some grammar tokens), if the tokenizer is in `TokenizerStateID`,
            // and the next grammar token is `GrammarTokenID`, the allowed LLM tokens are `LLMTokenBV`.

            // Let's go back to the `precompute` loop with the `VecDeque<(PartialToken, TokenizerStateID, SharedNode)>`.
            // `SharedNode` is a node in the *output* precompute trie we are building.
            // `PartialToken` and `TokenizerStateID` describe the *input* we are using to determine the next edges/nodes from `SharedNode`.

            // When the tokenizer matches a `result.id` with `result.width` from `bytes[offset..]` starting in `initial_tokenizer_state_id`:
            // - The grammar token matched is `GrammarTokenID(result.id)`.
            // - The new tokenizer state is `TokenizerStateID(result.new_state)`.
            // - The remaining part of the partial token is `bytes[offset + result.width..]`.

            // If `new_offset == bytes.len()` (full match of the PartialToken bytes):
            // The tokenizer has consumed all bytes of the `PartialToken`. The tokenizer is now in state `result.new_state`.
            // The `PartialToken` corresponds to the LLM token `dst.token_id()`.
            // The node `current_node` represents a state in the grammar constraint precomputation. What state?
            // It must be the state *before* considering the `PartialToken`.
            // So, `current_node` is the node in the precompute trie reached before processing the `PartialToken`.

            // Let's redefine the queue item: `(PartialToken<'a>, TokenizerStateID, SharedNode)`.
            // Process `PartialToken` starting with tokenizer state `TokenizerStateID`, extending from `SharedNode`.

            // If `new_offset == bytes.len()`:
            // We matched a sequence of bytes corresponding to `PartialToken { src, dst, bytes, offset }` fully.
            // The tokenizer ended in state `result.new_state`.
            // The grammar token matched is `GrammarTokenID(result.id)`.
            // The original `PartialToken` corresponds to the LLM token `dst.token_id()`.

            // This means that if we are at the state represented by `current_node` in the precompute trie,
            // and the tokenizer is in state `initial_tokenizer_state_id`, and the next input is the bytes `bytes[offset..]`
            // (which are part of the LLM token `dst.token_id()`), and these bytes match `GrammarTokenID(result.id)`,
            // ending in `result.new_state` *and* consuming the entire partial token (`new_offset == bytes.len()`),
            // then this constitutes a *finalizer* for the grammar token `GrammarTokenID(result.id)`.
            // The finalizer should add `dst.token_id()` as a possible LLM token if the tokenizer is in state `TokenizerStateID(result.new_state)`.

            // So, in the `if new_offset == bytes.len()` block:
            // `current_node` is the precompute trie node we are working from.
            // `GrammarTokenID(result.id)` is the grammar token that the current partial LLM token matches.
            // `dst.token_id()` is the full LLM token ID.
            // `TokenizerStateID(result.new_state)` is the tokenizer state after consuming the partial token.

            let possible_final_grammar_token = GrammarTokenID(result.id);
            let llm_token_id = LLMTokenID(dst.token_id());
            let tokenizer_state_after_match = TokenizerStateID(result.new_state);

            node.value.push_finalizer_info(
                possible_final_grammar_token,
                llm_token_id,
                tokenizer_state_after_match,
                max_id,
            );

            // After a full match of the partial token, we might transition to processing the *children* of the vocab node `dst`.
            // These children bytes should be processed starting from the tokenizer's *initial state* (TokenizerStateID(0)),
            // and should extend from the *same* precompute node `current_node` because the grammar token match completed.
            // This part still feels a bit off.

            // Let's rethink the structure again.
            // `Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>`
            // `PrecomputeNode = Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>`
            // A path `GrammarTokenID1 -> GrammarTokenID2 -> ...` in the trie represents matching a sequence of grammar tokens.
            // The `LLMTokenBV` on an edge represents the set of LLM tokens that *can* start with the byte sequence matching the `GrammarTokenID`.
            // A node's `PrecomputedNodeContents` has `finalizers` and `clean_end`.
            // `clean_end`: If we reach this node (matched the path of grammar tokens) and the tokenizer is in state 0, these LLM tokens are possible.
            // `finalizers`: If we reach this node, and the tokenizer is in state `tokenizer_state_id`, and the next grammar token is `final_grammar_token`, then these `llm_tokens` are possible.

            // So, `precompute` is building tries where nodes represent states *within the grammar parser*.
            // The edges are labeled by `GrammarTokenID`.
            // The values on the edges are `LLMTokenBV` - the set of LLM tokens whose prefixes correspond to the grammar token.
            // The nodes themselves contain information about what LLM tokens are possible if the grammar parsing path ends *at this node*,
            // depending on the final state of the tokenizer and the next expected grammar token (finalizers)
            // or if the tokenizer is in the initial state (clean_end).

            // The queue should track: `(VocabPrefixTreeNode, TokenizerStateID, SharedNode)`.
            // `VocabPrefixTreeNode`: the current node in the LLM token prefix tree.
            // `TokenizerStateID`: the current state of the tokenizer after matching a prefix of the LLM token.
            // `SharedNode`: the current node in the *precompute trie* being built.

            // Initial queue: For each root tokenizer state, for each child of the vocab root, add `(vocab_child_node, initial_tokenizer_state, root_precompute_node)`.

            // Processing `(current_vocab_node, current_tokenizer_state, current_precompute_node)`:
            // Get the bytes for the prefix from vocab root to `current_vocab_node`.
            // Execute the tokenizer from `current_tokenizer_state` on these bytes.
            // For each match `result`:
            // Grammar token matched: `GrammarTokenID(result.id)`.
            // New tokenizer state: `TokenizerStateID(result.new_state)`.
            // Remaining bytes: `bytes[offset + result.width ..]` (where bytes are the full prefix).

            // If the entire `PartialToken` bytes `bytes[offset..]` are consumed (`new_offset == bytes.len()`),
            // This means the path from `src` to `dst` in the vocab tree, starting at `offset`, corresponds to `GrammarTokenID(result.id)`.
            // This grammar token match transitions the tokenizer from `initial_tokenizer_state_id` to `result.new_state`.
            // This must define an edge in the precompute trie from `current_precompute_node` labeled `GrammarTokenID(result.id)`.
            // The value on this edge should include the LLM token `dst.token_id()` (since we fully matched its prefix).
            // The destination node of this edge represents the state after matching `GrammarTokenID(result.id)`.
            // What is the state? It depends on the next grammar token expected by the parser and the state of the tokenizer.
            // The `clean_end` and `finalizers` logic seems to fit here.

            // Let's refine the queue item and logic:
            // Queue item: `(VocabPrefixTreeNode, TokenizerStateID, SharedNode)`.
            // `VocabPrefixTreeNode`: The vocab node whose corresponding LLM tokens we are currently considering.
            // `TokenizerStateID`: The state of the tokenizer after consuming a prefix of the bytes *leading to* this vocab node.
            // `SharedNode`: The node in the precompute trie that we are currently building edges *from*.

            // Initial queue: For each tokenizer initial state `ts_id`, add `(vocab_root, ts_id, precomputed_roots[&ts_id])`.

            // Processing `(vocab_node, tokenizer_state, precompute_node)`:
            // We are at `vocab_node` in the vocab tree, with tokenizer state `tokenizer_state`.
            // We are at `precompute_node` in the precompute trie.
            // We want to find out which grammar tokens can match the prefixes starting from `vocab_node` and which LLM tokens are possible.
            // We need to consider the children of `vocab_node` in the vocab tree.
            // For each child `next_vocab_node` reached by bytes `next_bytes`:
            // Execute the tokenizer from `tokenizer_state` on `next_bytes`.
            // For each match `result`:
            // Grammar token matched: `GrammarTokenID(result.id)`.
            // New tokenizer state: `TokenizerStateID(result.new_state)`.
            // If `next_bytes` is fully consumed:
            // The grammar token `GrammarTokenID(result.id)` matched the path to `next_vocab_node`.
            // This implies an edge from `precompute_node` labeled `Some(GrammarTokenID(result.id))`.
            // The value on this edge should include the LLM tokens reachable from `next_vocab_node`.
            // The destination node of this edge needs to be determined. It represents being at `next_vocab_node` (conceptually)
            // and potentially needing further grammar tokens. This seems like it should be a node representing the state *after*
            // matching `GrammarTokenID(result.id)`.
            // What node is that? It depends on the grammar parser. This is where the GLR state comes in during `step`.

            // Let's look at the `Trie::special_map` again.
            // It takes `(PrecomputeNode, GLRParserState)`.
            // The `step` closure is `(GLRParserState, GrammarTokenID, LLMTokenBV, child_node) -> Option<GLRParserState>`.
            // The `child_node` is a `PrecomputeNode`.

            // This strongly suggests that `PrecomputeNode`s represent states in the grammar parser.
            // The structure `Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>` means:
            // Starting from tokenizer state `TS`, the corresponding `PrecomputeNode` is the root of a trie.
            // A path `GT1 -> GT2 -> ...` in this trie corresponds to matching grammar tokens `GT1`, `GT2`, etc.,
            // while the tokenizer state evolves based on the bytes that matched `GT1`, `GT2`, etc.

            // The queue must build this structure. What state needs to be in the queue to build it?
            // We need to know:
            // 1. The current node we are building in the `Precompute` trie (`SharedNode`).
            // 2. The tokenizer state that leads to this node (`TokenizerStateID`).
            // 3. The set of LLM tokens that are *partially matched* to reach this state (`VocabPrefixTreeNode`).

            // Queue item: `(SharedNode, TokenizerStateID, VocabPrefixTreeNode)`.
            // `SharedNode`: The node in the precompute trie we are *extending from*.
            // `TokenizerStateID`: The state of the tokenizer *at* this precompute node.
            // `VocabPrefixTreeNode`: The node in the vocab tree corresponding to the prefix of LLM tokens that led here.

            // Initial queue: For each root tokenizer state `ts_id`, add `(precomputed_roots[&ts_id], ts_id, vocab_root)`.

            // Processing `(precompute_node, tokenizer_state, vocab_node)`:
            // Consider the children of `vocab_node`. For each child `next_vocab_node` reached by `next_bytes`:
            // Execute tokenizer from `tokenizer_state` on `next_bytes`.
            // For each match `result`:
            // Grammar token matched: `GrammarTokenID(result.id)`.
            // New tokenizer state: `TokenizerStateID(result.new_state)`.
            // Bytes consumed from `next_bytes`: `result.width`.
            // If `result.width == next_bytes.len()` (full match of `next_bytes`):
            // We matched `next_bytes` which corresponds to reaching `next_vocab_node`. This byte sequence matched `GrammarTokenID(result.id)`.
            // This defines an edge from `precompute_node` labeled `Some(GrammarTokenID(result.id))`.
            // The value on this edge should be `next_vocab_node.reachable_token_ids()`.
            // The destination node of this edge corresponds to the state after matching `GrammarTokenID(result.id)`,
            // with the tokenizer in state `result.new_state`, and having matched up to `next_vocab_node`.
            // Let's try mapping `(TokenizerStateID, VocabPrefixTreeNode)` to `SharedNode` again.

            // `node_map: HashMap<(TokenizerStateID, *const VocabPrefixTreeNode), SharedNode>`.
            // Initial node_map: For each `ts_id`, map `(ts_id, vocab_root_ptr)` to `precomputed_roots[&ts_id]`.

            // Processing `(precompute_node, tokenizer_state, vocab_node)`:
            // For each child `next_vocab_node` of `vocab_node` via `next_bytes`:
            // Execute tokenizer from `tokenizer_state` on `next_bytes`.
            // For each match `result`:
            // `matched_grammar_token = GrammarTokenID(result.id)`
            // `next_tokenizer_state = TokenizerStateID(result.new_state)`
            // `bytes_consumed = result.width`

            // If `bytes_consumed == next_bytes.len()`:
            // The entire `next_bytes` matched `matched_grammar_token`.
            // We need an edge from `precompute_node` labeled `Some(matched_grammar_token)`.
            // The value on the edge is `next_vocab_node.reachable_token_ids()`.
            // The destination node corresponds to state `(next_tokenizer_state, next_vocab_node)`.
            // Look up or create this node in `node_map`.
            // `next_precompute_node = node_map.entry((next_tokenizer_state, next_vocab_node.ptr_id())).or_insert_with(...)`
            // Add edge: `precompute_node.lock().unwrap().insert(Some(matched_grammar_token), next_vocab_node.reachable_token_ids().clone(), next_precompute_node.clone());`
            // Add to queue: `queue.push_back((next_precompute_node, next_tokenizer_state, next_vocab_node));`

            // If `bytes_consumed < next_bytes.len()`:
            // The tokenizer matched a grammar token *within* the `next_bytes`.
            // This seems like a partial match of the LLM token prefix.
            // This scenario should contribute to `finalizers`.
            // If we are at `precompute_node`, and the tokenizer is in `tokenizer_state`, and the grammar token `matched_grammar_token` is accepted by the parser,
            // and matching `matched_grammar_token` corresponds to consuming `bytes_consumed` of `next_bytes` starting from `vocab_node` (or rather, `vocab_node`'s child via `next_bytes`),
            // and this leaves the tokenizer in `next_tokenizer_state`, then the LLM tokens corresponding to `next_vocab_node` (or prefixes thereof) are possible.

            // This `precompute` function is building the mapping from `(GrammarParserState, TokenizerStateID)` to allowed `LLMTokenBV`.
            // The `PrecomputeNode` seems to represent the `GrammarParserState`.
            // The structure `Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>` implies that for each `TokenizerStateID`, there is a separate precompute trie. This feels wrong.
            // The grammar parser state is independent of the tokenizer state.

            // Let's re-read the code and comments carefully.
            // `Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>`: This outer map seems to be based on the *initial* tokenizer state.
            // `PrecomputeNode = Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>`: Tries based on grammar tokens.
            // `PrecomputedNodeContents` has `finalizers: BTreeMap<GrammarTokenID, PrecomputedFinalizer>` and `clean_end`.
            // `PrecomputedFinalizer` has `content: BTreeMap<TokenizerStateID, LLMTokenBV>`.

            // This structure suggests:
            // Starting with `TokenizerStateID`, use the corresponding `PrecomputeNode`.
            // Traverse this trie using `GrammarTokenID`s.
            // An edge `Some(GT)` to `(llm_bv, child_node)` means if the next grammar token is `GT`, the possible LLM tokens whose prefixes match GT are `llm_bv`, and the next precompute state is `child_node`.
            // At a node `N`, if the tokenizer is in state `TS`, and the next grammar token is `FGT`, and `N` has a finalizer for `FGT`, the allowed LLM tokens are `finalizers[FGT].content[TS]`.
            // At a node `N`, if the tokenizer is in state `0` (`clean_end`), and `N` has `clean_end` set, the allowed LLM tokens are `N.value.clean_end`.

            // The `precompute` function must build these tries.
            // It takes tokenizer, llm_token_map.
            // It iterates through llm_token_map to build `vocab`.
            // It creates root `PrecomputeNode`s for each tokenizer initial state. This still seems weird if the PrecomputeNode represents grammar state.

            // Let's assume the current code's structure is correct and try to implement the refactoring based on that.
            // The queue item in the original code was `((DottedVocabNode, TokenizerStateID), BTreeSet<NodeHandle>)`.
            // `DottedVocabNode`: Part of a vocab token being matched.
            // `TokenizerStateID`: Current tokenizer state.
            // `BTreeSet<NodeHandle>`: Set of `PrecomputeNode`s being extended from.

            // This means the precompute trie nodes represent states corresponding to `(PartialToken, TokenizerStateID)`.
            // `Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>`.
            // This seems to imply `PrecomputeNode` is keyed by `TokenizerStateID`. But it's a trie of `GrammarTokenID`.

            // Okay, let's trust the original code's data structures and focus on the refactoring steps:
            // 1. Move nested helpers out. Done.
            // 2. Rename and derive `PartialToken`. Done.
            // 3. Replace `NodeHandle` with `SharedNode` and use ptr_id. Done.
            // 4. Replace BTreeMap queue with VecDeque. Done. The queue item will be `(PartialToken, TokenizerStateID, SharedNode)`.
            // 5. Extract loop body to `process_queue_item`.
            // 6. Delete merge logic.
            // 7. Remove debug logs.
            // 8. Turn `link_next_precompute_node` into `PrecomputeNode::link_to`. This needs to be adapted as the original logic was complex due to merging nodes. The new `link_to` on a single node should just add/update the edge.
            // 9. Shrink `precompute` body.
            // 10. Implement helper functions.
            // 11. Adjust final collection.

            // Adapting `link_to`:
            // Original `link_next_precompute_node`: Takes queue, new_queue_key, precompute_node (mutable guard), matched_token_id.
            // It tries to find if the `new_queue_key` (which includes `DottedVocabNode` and `TokenizerStateID`) already exists in the queue's keys.
            // If it exists, it gets the set of destination nodes from the queue.
            // It checks if `precompute_node` already has an edge to any of these destination nodes. If so, merge.
            // If not, it tries to insert a new edge to an existing node from the queue's set.
            // If none of that works, it creates a new destination node, inserts an edge to it, and returns the new node.

            // This is complex because the original queue item mapped `(PartialToken, TokenizerStateID)` to a *set* of destination nodes.
            // With the VecDeque item `(PartialToken, TokenizerStateID, SharedNode)`, where `SharedNode` is the *parent* node, this linking logic changes.

            // New queue item interpretation: `(PartialToken being processed, Tokenizer state *before* processing, Parent precompute node)`.

            // Processing `(partial_token, initial_tokenizer_state_id, parent_precompute_node)`:
            // Execute tokenizer on `partial_token.bytes[partial_token.offset..]` from `initial_tokenizer_state_id`.
            // For each result: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial_token.offset + bytes_consumed`.

            // If `new_offset == partial_token.bytes.len()`: Full match of the partial token.
            // The grammar token `matched_grammar_token` corresponds to the partial token bytes.
            // This creates a finalizer: From `parent_precompute_node`, if the next grammar token is `matched_grammar_token`, and the tokenizer is in state `new_tokenizer_state`, then the LLM token `partial_token.dst.token_id()` is possible.
            // `parent_precompute_node.lock().unwrap().value.push_finalizer_info(...)`

            // If `new_offset < partial_token.bytes.len()`: Partial match of the partial token bytes.
            // The grammar token `matched_grammar_token` matched a prefix of the partial token bytes.
            // This creates an edge from `parent_precompute_node` labeled `Some(matched_grammar_token)`.
            // The value on the edge should be the LLM tokens reachable from `partial_token.dst`.
            // The destination node of this edge represents the state after matching `matched_grammar_token`, with the tokenizer in `new_tokenizer_state`, and the remaining partial token being `partial_token.bytes[new_offset..]`.
            // This destination node needs to be unique for the combination of `(new_tokenizer_state, remaining_partial_token, parent_precompute_node + edge?)`.

            // This is complex. Let's step back and look at the data flow again.
            // `precompute` builds a mapping from grammar state (represented by PrecomputeNode path) and tokenizer state to allowed LLM tokens.
            // The process involves iterating through all possible LLM token prefixes and seeing how the tokenizer and the hypothetical grammar parser state evolve.

            // Queue item: `(SharedNode, PartialToken, TokenizerStateID)`.
            // `SharedNode`: The current node in the precompute trie being built.
            // `PartialToken`: The segment of an LLM token being processed.
            // `TokenizerStateID`: The tokenizer state after consuming the part of the LLM token that led to the `SharedNode`. (This doesn't seem right).

            // Let's retry the previous interpretation:
            // Queue item: `(PartialToken, TokenizerStateID, SharedNode)`.
            // `PartialToken`: The LLM token segment currently being processed by the tokenizer.
            // `TokenizerStateID`: The state of the tokenizer *before* processing `PartialToken`.
            // `SharedNode`: The node in the precompute trie that we are extending edges *from*, which corresponds to the grammar state *before* considering the `PartialToken`.

            // Initial queue: For each root tokenizer state `ts_id`, for each child `vocab_child` of the vocab root, via `bytes`: `(PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 }, ts_id, precomputed_roots[&ts_id])`.

            // Processing `(partial, initial_tokenizer_state, parent_precompute_node)`:
            // Execute tokenizer on `partial.bytes[partial.offset..]` from `initial_tokenizer_state`.
            // For each match `result`: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.

            // If `new_offset == partial.bytes.len()`: Full match of `partial`.
            // This match `matched_grammar_token` corresponds to fully consuming the prefix represented by `partial.bytes[partial.offset..]`.
            // This contributes to a finalizer at `parent_precompute_node`:
            // `parent_precompute_node.lock().unwrap().value.push_finalizer_info(matched_grammar_token, partial.dst.token_id(), new_tokenizer_state, max_id);`

            // If `new_offset < partial.bytes.len()`: Partial match of `partial`.
            // The grammar token `matched_grammar_token` matched a prefix of `partial.bytes[partial.offset..]`.
            // This creates an edge from `parent_precompute_node` labeled `Some(matched_grammar_token)`.
            // The value on the edge is the LLM tokens reachable from `partial.dst`. (`partial.dst.reachable_token_ids()`)
            // The destination node needs to represent the state after matching `matched_grammar_token`, with the tokenizer in `new_tokenizer_state`, processing the remaining part of `partial`.
            // This destination node should be the start of a new precompute trie segment.

            // This still feels circular. The precompute trie nodes represent grammar state, but their construction seems to depend on tokenizer state and LLM token prefixes.

            // Let's look at how `PrecomputeNode::link_to` was used in the original code after extracting it.
            // It was called inside the `for precompute_node in &precomputed_nodes` loop (where `precomputed_nodes` was the set from the queue).
            // `link_next_precompute_node(&mut queue, new_queue_key, &mut precompute_node, matched_token_id)`
            // `new_queue_key` was `(DottedVocabNode { src, dst, offset: new_offset, bytes }, TokenizerStateID(0))` OR `(DottedVocabNode { src: next_src, dst: next_dst, bytes: next_bytes, offset: 0 }, TokenizerStateID(0))`. The tokenizer state was hardcoded to 0 often. This seems wrong. The tokenizer state *must* evolve.

            // Okay, the original code linked `precompute_node` (from the queue set) to a new or existing node based on `new_queue_key = (DottedVocabNode, TokenizerStateID)`.
            // The `link_next_precompute_node` function's main goal was to find or create the destination `SharedNode` based on `new_queue_key` and add the edge from `precompute_node`.

            // Let's use a `node_map: HashMap<(PartialToken, TokenizerStateID), SharedNode>` to find/create destination nodes.
            // Queue item: `(PartialToken, TokenizerStateID, SharedNode)`. Still the same interpretation: process `PartialToken` from `TokenizerStateID`, extending from `SharedNode`.

            // Need a way to map `(PartialToken, TokenizerStateID)` to a `SharedNode`.
            // `PartialToken` contains references, so cannot be a HashMap key directly without a lot of work or using pointer IDs carefully.
            // Let's map `(PartialToken key tuple, TokenizerStateID)` to `SharedNode`.
            // `node_map: HashMap<((usize, *const (), *const u8), TokenizerStateID), SharedNode>`.

            // Initial queue: For each `ts_id`, for each child `vocab_child` of vocab root via `bytes`:
            // `partial = PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 }`
            // `key = (partial.key(), ts_id)`
            // `node = precomputed_roots[&ts_id]`
            // `node_map.insert(key, Arc::clone(&node))`
            // `queue.push_back((partial, ts_id, node))`

            // Processing `(partial, initial_tokenizer_state, parent_precompute_node)`:
            // Execute tokenizer... For each result: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.

            // If `new_offset == partial.bytes.len()`: Full match.
            // `parent_precompute_node.lock().unwrap().value.push_finalizer_info(matched_grammar_token, partial.dst.token_id(), new_tokenizer_state, max_id);`
            // After a full match, the tokenizer is in `new_tokenizer_state`, and we've matched the path to `partial.dst`.
            // We should now consider the children of `partial.dst` starting from `new_tokenizer_state`, continuing from `parent_precompute_node`.
            // For each child `next_vocab_node` of `partial.dst` via `next_bytes`:
            // `next_partial = PartialToken { src: partial.dst, dst: next_vocab_node, bytes: next_bytes, offset: 0 }`
            // `next_key = (next_partial.key(), new_tokenizer_state)`
            // Check if `next_key` exists in `node_map`.
            // If exists, get `next_precompute_node = node_map[&next_key]`.
            // If not exists, create `next_precompute_node = Arc::new(Mutex::new(PrecomputeNode::new(Default::default())))`, insert into `node_map`.
            // Add edge from `parent_precompute_node` labeled `Some(matched_grammar_token)` to `next_precompute_node`. (This still seems wrong, the edge label is the grammar token, not the partial token).

            // Let's go back to the very first interpretation: `PrecomputeNode`s represent states in the precomputation process defined by `(PartialToken, TokenizerStateID)`.
            // `Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>`. The roots are indexed by the *initial* tokenizer state.
            // `PrecomputeNode` is a trie keyed by `Option<GrammarTokenID>`. This seems like a structure to explore grammar token sequences.

            // The `precompute` function explores pairs of `(LLM token prefix, TokenizerStateID)` and determines what grammar tokens match and what the resulting state is, building the `Precomputed` tries.

            // Queue item: `(PartialToken, TokenizerStateID, SharedNode)`.
            // `PartialToken`: The current LLM token prefix being considered.
            // `TokenizerStateID`: The tokenizer state after matching the grammar token sequence that led to `SharedNode`. (This also seems wrong).

            // Let's try one more interpretation of the `Precomputed` structure:
            // `Precomputed` maps an *initial* TokenizerStateID to a `PrecomputeNode`.
            // A `PrecomputeNode` is a trie where keys are `Option<GrammarTokenID>`.
            // A path `GT1 -> GT2` in this trie means matching grammar token GT1 then GT2.
            // The value on an edge `Some(GT)` is an `LLMTokenBV` representing LLM tokens whose *prefixes* match GT.
            // The destination node of the edge is another `PrecomputeNode`.

            // This suggests `PrecomputeNode`s represent states *in the grammar constraint automaton*.
            // The `precompute` function is building this automaton.
            // The state of this automaton is derived from the interaction of LLM token prefixes and the tokenizer.

            // Queue item: `(SharedNode, PartialToken, TokenizerStateID)`.
            // `SharedNode`: The current state (node) in the grammar constraint automaton being built.
            // `PartialToken`: The LLM token prefix segment we are currently trying to match starting from this state.
            // `TokenizerStateID`: The state of the tokenizer *when entering* this grammar constraint state. (This also seems wrong).

            // Let's go back to the queue item `(PartialToken, TokenizerStateID, SharedNode)` and the original code's logic more closely.
            // `(dotted_vocab_node, initial_tokenizer_state_id, precomputed_nodes_set)`.
            // `precomputed_nodes_set` was a *set* of precompute nodes. This suggests multiple grammar constraint states can be reached by the same `(PartialToken, TokenizerStateID)` input sequence. This points towards the GLR nature.

            // Let's assume the goal of `precompute` is to build the `Precomputed` map, which structures the precomputation results by *initial* tokenizer state.
            // For a given initial tokenizer state `TS_init`, `precomputed[TS_init]` is a trie.
            // This trie's edges are `GrammarTokenID`. The values are `LLMTokenBV` and a child `PrecomputeNode`.
            // The nodes contain `finalizers` and `clean_end`.

            // Queue item: `(CurrentPrecomputeNode, PartialToken, TokenizerStateID)`.
            // `CurrentPrecomputeNode`: The node in the trie `precomputed[InitialTokenizerStateID]` we are currently at.
            // `PartialToken`: The LLM token prefix being matched by the tokenizer.
            // `TokenizerStateID`: The current state of the tokenizer.

            // Initial queue: For each `ts_id` in `0..tokenizer.max_state()`:
            // For each child `vocab_child` of vocab root via `bytes`:
            // `partial = PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 }`
            // `queue.push_back((precomputed_roots[&ts_id], partial, ts_id))`

            // Processing `(current_precompute_node, partial, current_tokenizer_state)`:
            // Execute tokenizer on `partial.bytes[partial.offset..]` from `current_tokenizer_state`.
            // For each result: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.

            // If `new_offset == partial.bytes.len()`: Full match of `partial`.
            // The grammar token `matched_grammar_token` corresponds to the full `partial` bytes.
            // This means matching `matched_grammar_token` from the state `current_precompute_node` while the tokenizer is in `current_tokenizer_state` is possible if the next input is `partial.bytes`.
            // This contributes to a finalizer at `current_precompute_node`.
            // `current_precompute_node.lock().unwrap().value.push_finalizer_info(matched_grammar_token, partial.dst.token_id(), new_tokenizer_state, max_id);`

            // If `new_offset < partial.bytes.len()`: Partial match of `partial`.
            // The grammar token `matched_grammar_token` matched a prefix of `partial.bytes[partial.offset..]`.
            // This defines an edge from `current_precompute_node` labeled `Some(matched_grammar_token)`.
            // The value on this edge should include `partial.dst.reachable_token_ids()`.
            // The destination node of this edge represents the state after matching `matched_grammar_token`, with the tokenizer in `new_tokenizer_state`, and the remaining partial token being `partial.bytes[new_offset..]`.

            // This still feels complicated because the destination node's identity depends on `(remaining_partial_token, new_tokenizer_state)`.
            // Let's try the `node_map: HashMap<((PartialToken key tuple), TokenizerStateID), SharedNode>` approach again, but structure the queue differently.

            // Queue item: `(PartialToken, TokenizerStateID)`. We are trying to process this input state.
            // The result of processing this input state from a `PrecomputeNode` will be transitions *from* that node.
            // The `PrecomputeNode` itself is implicitly managed by the `node_map`.

            // `node_map: HashMap<((PartialToken key tuple), TokenizerStateID), SharedNode>`. This maps an input state `(PartialToken, TokenizerStateID)` to the *root* of a sub-trie that should be attached as an edge value. This is getting complex.

            // Let's revisit the original code's queue `BTreeMap<(DottedVocabNode, TokenizerStateID), BTreeSet<NodeHandle>>`.
            // The key is `(PartialToken, TokenizerStateID)`. The value is a *set* of `SharedNode`.
            // This means for a given input state `(PartialToken, TokenizerStateID)`, we are exploring its effect on *multiple* `PrecomputeNode`s simultaneously. These `PrecomputeNode`s are the ones that the input `(PartialToken, TokenizerStateID)` could possibly follow from.

            // Let's use the `VecDeque` with item `(PartialToken, TokenizerStateID, SharedNode)` again.
            // `PartialToken`: The LLM token segment being processed.
            // `TokenizerStateID`: Tokenizer state.
            // `SharedNode`: The node in the Precompute trie we are currently extending *from*.

            // Initial queue: For each root `precompute_root` in `precomputed_roots`, for each child `vocab_child` of vocab root via `bytes`:
            // `partial = PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 }`
            // `queue.push_back((partial, precomputed_roots.iter().find(|(_, node)| Arc::ptr_eq(node, &precompute_root)).unwrap().0, precompute_root))` NO, the initial tokenizer state matters for the root.

            // Initial queue: For each `ts_id` and its root node `root_node = precomputed_roots[&ts_id]`:
            // For each child `vocab_child` of vocab root via `bytes`:
            // `partial = PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 }`
            // `queue.push_back((partial, ts_id, root_node))`

            // Processing `(partial, initial_tokenizer_state, current_precompute_node)`:
            // Execute tokenizer on `partial.bytes[partial.offset..]` from `initial_tokenizer_state`.
            // For each match `result`: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.

            // If `new_offset == partial.bytes.len()`: Full match of `partial`.
            // This match indicates that `matched_grammar_token` corresponds to the prefix matched by the tokenizer, ending in `new_tokenizer_state`, consuming the full `partial`.
            // This defines a finalizer from `current_precompute_node`.
            // `current_precompute_node.lock().unwrap().value.push_finalizer_info(matched_grammar_token, partial.dst.token_id(), new_tokenizer_state, max_id);`

            // If `new_offset < partial.bytes.len()`: Partial match of `partial`.
            // The grammar token `matched_grammar_token` matched a prefix of `partial.bytes`.
            // We need an edge from `current_precompute_node` labeled `Some(matched_grammar_token)`.
            // The value on the edge is the LLM tokens reachable from `partial.dst`.
            // The destination node represents the state after matching `matched_grammar_token`, with the tokenizer in `new_tokenizer_state`, and the remaining partial token being `partial.bytes[new_offset..]`.
            // This destination state needs to be unique for the combination `(current_precompute_node, Some(matched_grammar_token), new_tokenizer_state, remaining_partial)`. This is getting complicated.

            // Let's simplify the `PrecomputeNode` structure interpretation.
            // `Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>`: Rooted by initial tokenizer state.
            // `PrecomputeNode`: Represents a state reached by matching a sequence of grammar tokens. The trie structure *is* the grammar token sequence.
            // Edge `Some(GT)` to `(llm_bv, child_node)`: If the next grammar token is GT, these LLM tokens are possible, transition to `child_node`.
            // Node `N` contents: `finalizers` and `clean_end`.
            // `finalizers[FGT][TS] = llm_bv`: If at node `N`, tokenizer is in `TS`, and next grammar token is `FGT`, these LLM tokens are possible.
            // `clean_end = llm_bv`: If at node `N`, tokenizer is in state 0, these LLM tokens are possible.

            // The `precompute` builds these tries.
            // We need to explore pairs of `(TokenizerStateID, VocabPrefixTreeNode)` and determine which grammar tokens match and what the next state is, constructing the precompute trie edges and nodes.

            // Queue item: `(TokenizerStateID, VocabPrefixTreeNode)`. We are processing the potential match of bytes corresponding to `VocabPrefixTreeNode` starting from `TokenizerStateID`.
            // This processing will tell us which grammar tokens match and what the resulting tokenizer states are.

            // How does this build the `PrecomputeNode` trie?

            // Let's go back to the `Trie::special_map` again. It processes `(PrecomputeNode, GLRParserState)`.
            // This means the `PrecomputeNode` is one input to the step function, alongside the GLR state.

            // Let's assume the original logic using `DottedVocabNode` and `TokenizerStateID` mapping to a set of `SharedNode`s was correct, despite its complexity.
            // The key `(PartialToken, TokenizerStateID)` identifies a specific "input path" (matching a part of an LLM token from a tokenizer state).
            // The value `BTreeSet<NodeHandle>` identifies the set of *grammar constraint states* (nodes in the `Precompute` trie) that this input path could extend *from*.

            // Queue item: `(PartialToken, TokenizerStateID, SharedNode)`. Process this input state starting from *this specific* `SharedNode`.

            // Initial queue: For each `ts_id` and its root `root_node`:
            // For each child `vocab_child` of vocab root via `bytes`:
            // `partial = PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 }`
            // `queue.push_back((partial, ts_id, root_node))`

            // Processing `(partial, initial_tokenizer_state, current_precompute_node)`:
            // Execute tokenizer on `partial.bytes[partial.offset..]` from `initial_tokenizer_state`.
            // For each match `result`: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.

            // If `new_offset == partial.bytes.len()`: Full match of `partial`.
            // This means matching `partial` corresponds to `matched_grammar_token`, ending with tokenizer in `new_tokenizer_state`.
            // This implies a finalizer at `current_precompute_node`.
            // `current_precompute_node.lock().unwrap().value.push_finalizer_info(matched_grammar_token, partial.dst.token_id(), new_tokenizer_state, max_id);`

            // If `new_offset < partial.bytes.len()`: Partial match of `partial`.
            // `matched_grammar_token` matched a prefix of `partial.bytes`.
            // This defines an edge from `current_precompute_node` labeled `Some(matched_grammar_token)`.
            // The value on the edge is `partial.dst.reachable_token_ids()`.
            // The destination node of this edge needs to represent the state where the tokenizer is in `new_tokenizer_state` and we are processing the remaining part of `partial`.

            // This points to the destination node being identified by `(remaining_partial, new_tokenizer_state)`.
            // Let's use a map to find/create these destination nodes:
            // `dest_node_map: HashMap<((PartialToken key tuple), TokenizerStateID), SharedNode>`.

            // Initial queue: Same as before.
            // Initial `dest_node_map`: Empty.

            // Processing `(partial, initial_tokenizer_state, current_precompute_node)`:
            // Execute tokenizer... For each result: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.

            // If `new_offset < partial.bytes.len()`: Partial match.
            // `remaining_partial = PartialToken { src: partial.src, dst: partial.dst, bytes: partial.bytes, offset: new_offset }`
            // `dest_key = (remaining_partial.key(), new_tokenizer_state)`
            // `dest_node = dest_node_map.entry(dest_key).or_insert_with(|| Arc::new(Mutex::new(PrecomputeNode::new(Default::default()))));`
            // Add edge from `current_precompute_node` to `dest_node`:
            // `current_precompute_node.lock().unwrap().force_insert_to_node(Some(matched_grammar_token), partial.dst.reachable_token_ids().clone(), dest_node);`
            // Add `(remaining_partial, new_tokenizer_state, dest_node)` to queue.

            // If `new_offset == partial.bytes.len()`: Full match. Finalizer.
            // `current_precompute_node.lock().unwrap().value.push_finalizer_info(matched_grammar_token, partial.dst.token_id(), new_tokenizer_state, max_id);`
            // After a full match of the PartialToken, we are now conceptually at the *end* of that LLM token prefix in the vocab tree (`partial.dst`).
            // The tokenizer is in state `new_tokenizer_state`.
            // The next input will be the *children* of `partial.dst` in the vocab tree.
            // These should be processed starting from the tokenizer state `new_tokenizer_state`, from the *current* precompute node? No.
            // Matching `matched_grammar_token` from `current_precompute_node` leads to a new state.
            // Let's assume the structure `Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>` is correct.
            // A `PrecomputeNode` corresponds to a grammar state and an *initial* tokenizer state.

            // Let's go back to the original code's queue key `(DottedVocabNode, TokenizerStateID)` and value `BTreeSet<NodeHandle>`.
            // This means for a given input `(PartialToken, TokenizerStateID)`, we identify a set of `PrecomputeNode`s that can accept this input and transition.
            // This still feels like the `PrecomputeNode`s are grammar states, and the key identifies the input that transitions them.

            // Let's use the `VecDeque<(PartialToken, TokenizerStateID, SharedNode)>` queue again.
            // `PartialToken`: The LLM token bytes being processed.
            // `TokenizerStateID`: The tokenizer state.
            // `SharedNode`: The node in the precompute trie that this `(PartialToken, TokenizerStateID)` combination will potentially extend *from*.

            // Initial queue: For each `ts_id`, and its root `root_node = precomputed_roots[&ts_id]`.
            // For each child `vocab_child` of vocab root via `bytes`:
            // `partial = PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 }`
            // `queue.push_back((partial, ts_id, root_node))`

            // Processing `(partial, initial_tokenizer_state, current_precompute_node)`:
            // Execute tokenizer... For each result: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.

            // If `new_offset < partial.bytes.len()`: Partial match.
            // `remaining_partial = PartialToken { src: partial.src, dst: partial.dst, bytes: partial.bytes, offset: new_offset }`
            // We need an edge from `current_precompute_node` labeled `Some(matched_grammar_token)`.
            // The value on the edge is `partial.dst.reachable_token_ids()`.
            // The destination node of this edge should correspond to the state where the tokenizer is in `new_tokenizer_state` and we are processing `remaining_partial`.
            // This destination node should be the root of a sub-trie starting from `(remaining_partial, new_tokenizer_state)`.

            // This leads back to needing a map to identify the root of the sub-trie for `(PartialToken, TokenizerStateID)`.
            // `sub_trie_roots: HashMap<((PartialToken key tuple), TokenizerStateID), SharedNode>`.

            // Initial queue: For each `ts_id` and root `root_node`:
            // For each child `vocab_child` via `bytes`:
            // `partial = PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 }`
            // `key = (partial.key(), ts_id)`
            // `sub_trie_roots.insert(key, Arc::clone(&root_node));` // Root is the sub-trie root for this initial input
            // `queue.push_back((partial, ts_id, root_node))` // Queue item is the input and the node it starts from? No.

            // Let's rethink the role of `current_precompute_node` in the queue item `(PartialToken, TokenizerStateID, SharedNode)`.
            // It must be the node we are extending *from*.

            // Processing `(partial, initial_tokenizer_state, current_precompute_node)`:
            // Execute tokenizer... For each result: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.

            // If `new_offset < partial.bytes.len()`: Partial match.
            // `remaining_partial = PartialToken { src: partial.src, dst: partial.dst, bytes: partial.bytes, offset: new_offset }`
            // We need an edge from `current_precompute_node` labeled `Some(matched_grammar_token)` to a destination node.
            // The destination node represents the state `(remaining_partial, new_tokenizer_state)`.
            // Let's use `node_map: HashMap<((PartialToken key tuple), TokenizerStateID), SharedNode>` to find or create this destination node.
            // `dest_key = (remaining_partial.key(), new_tokenizer_state)`
            // `dest_node = node_map.entry(dest_key).or_insert_with(|| Arc::new(Mutex::new(PrecomputeNode::new(Default::default()))));`
            // Add edge from `current_precompute_node` to `dest_node` with value `partial.dst.reachable_token_ids()`.
            // `current_precompute_node.lock().unwrap().force_insert_to_node(Some(matched_grammar_token), partial.dst.reachable_token_ids().clone(), dest_node);`
            // Add `(remaining_partial, new_tokenizer_state, dest_node)` to the queue.

            // If `new_offset == partial.bytes.len()`: Full match. Finalizer.
            // `current_precompute_node.lock().unwrap().value.push_finalizer_info(matched_grammar_token, partial.dst.token_id(), new_tokenizer_state, max_id);`
            // After fully matching `partial`, we are at `partial.dst` in the vocab tree. The tokenizer is in `new_tokenizer_state`.
            // The next possible inputs are the children of `partial.dst` in the vocab tree.
            // These should be processed starting from `new_tokenizer_state`, continuing from `current_precompute_node`? No.
            // Matching `matched_grammar_token` from `current_precompute_node` leads to a new state in the precompute trie.
            // This new state should be the one receiving edges for the children of `partial.dst`.

            // Let's call the destination node for the edge labeled `Some(matched_grammar_token)` from `current_precompute_node` the `next_grammar_state_node`.
            // If `new_offset < partial.bytes.len()`, the `next_grammar_state_node` is identified by `(remaining_partial, new_tokenizer_state)`. We add `(remaining_partial, new_tokenizer_state, next_grammar_state_node)` to the queue.
            // If `new_offset == partial.bytes.len()`, the `next_grammar_state_node` corresponds to being at the end of the LLM token prefix (at `partial.dst` in vocab), with tokenizer in `new_tokenizer_state`. What identifies this node? `(partial.dst, new_tokenizer_state)`? But PartialToken is for *ongoing* match.

            // Let's simplify the state representation in the `node_map`.
            // The state after processing a piece of an LLM token must capture the remaining part of the LLM token and the tokenizer state.
            // State key: `(VocabPrefixTreeNode, usize, TokenizerStateID)`. `VocabPrefixTreeNode` is the current position in the vocab tree, `usize` is the offset into its prefix/bytes, `TokenizerStateID` is the tokenizer state.

            // `node_map: HashMap<(*const VocabPrefixTreeNode, usize, TokenizerStateID), SharedNode>`.

            // Initial queue: For each `ts_id` and root `root_node`:
            // `vocab_root = vocab.root`
            // `key = (vocab_root.ptr_id(), 0, ts_id)`
            // `node_map.insert(key, Arc::clone(&root_node))` // Root node corresponds to starting at vocab root, offset 0, tokenizer state ts_id
            // For each child `vocab_child` of vocab root via `bytes`:
            // `partial = PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 }`
            // `queue.push_back((partial, ts_id, root_node))` // Queue item: LLM token segment, initial tokenizer state for this segment, parent precompute node.

            // Processing `(partial, initial_tokenizer_state, current_precompute_node)`:
            // Execute tokenizer on `partial.bytes[partial.offset..]` from `initial_tokenizer_state`.
            // For each result: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.
            // `remaining_bytes = &partial.bytes[new_offset..]`.

            // This is not right. The `PartialToken` itself evolves. The state in `node_map` should be `(VocabPrefixTreeNode, offset, TokenizerStateID)`.

            // Let's go back to the original `DottedVocabNode`: `src, dst, bytes, offset`. This captures the segment `bytes[offset..]` within the full `src..dst` prefix.
            // `node_map: HashMap<((*const VocabPrefixTreeNode, *const VocabPrefixTreeNode, *const u8, usize), TokenizerStateID), SharedNode>`. This key is horrible.

            // Let's simplify the queue item to `(PartialToken, TokenizerStateID)`.
            // And the `node_map` will map this pair to the `SharedNode` that represents the state after processing this input.
            // `node_map: HashMap<((PartialToken key tuple), TokenizerStateID), SharedNode>`.

            // Initial queue: For each `ts_id`:
            // For each child `vocab_child` of vocab root via `bytes`:
            // `partial = PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 }`
            // `key = (partial.key(), ts_id)`
            // `node = precomputed_roots[&ts_id]` // Root node corresponds to (any partial token starting from vocab_root, ts_id)
            // This mapping is not unique. Multiple partial tokens starting from vocab root can map to the same initial precompute node.

            // The root precompute node is determined *only* by the initial tokenizer state.
            // `Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>` where `PrecomputeNode` is the root for that `TokenizerStateID`.

            // Queue item: `(SharedNode, PartialToken, TokenizerStateID)`.
            // `SharedNode`: Current node in the precompute trie we are building.
            // `PartialToken`: LLM token segment being processed by the tokenizer *to find the next grammar token*.
            // `TokenizerStateID`: Current tokenizer state after processing the grammar tokens that led to `SharedNode`. (Still unsure about this).

            // Let's follow the original code's structure strictly in the refactored version.
            // Original queue key: `(DottedVocabNode, TokenizerStateID)`. Value: `BTreeSet<NodeHandle>`.
            // This means an input `(PartialToken, TokenizerStateID)` maps to a set of grammar constraint states.

            // New queue item: `(PartialToken, TokenizerStateID, SharedNode)`. Process this input state *starting from this specific SharedNode*.

            // Initial queue: For each `ts_id` and its root `root_node`:
            // For each child `vocab_child` of vocab root via `bytes`:
            // `partial = PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 }`
            // `queue.push_back((partial, ts_id, root_node))`

            // Processing `(partial, initial_tokenizer_state, current_precompute_node)`:
            // Execute tokenizer on `partial.bytes[partial.offset..]` from `initial_tokenizer_state`.
            // For each match `result`: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.
            // `remaining_bytes = &partial.bytes[new_offset..]`.

            // If `new_offset < partial.bytes.len()`: Partial match.
            // `next_partial = PartialToken { src: partial.src, dst: partial.dst, bytes: partial.bytes, offset: new_offset }`
            // This means matching `matched_grammar_token` from `current_precompute_node` leads to a state where the tokenizer is in `new_tokenizer_state` and we still have `next_partial` to process.
            // We need an edge from `current_precompute_node` labeled `Some(matched_grammar_token)`.
            // The value on the edge is `partial.dst.reachable_token_ids()`.
            // The destination node corresponds to the state `(next_partial, new_tokenizer_state)`.
            // Use `node_map: HashMap<((PartialToken key tuple), TokenizerStateID), SharedNode>` to find/create this node.
            // `dest_key = (next_partial.key(), new_tokenizer_state)`
            // `dest_node = node_map.entry(dest_key).or_insert_with(|| Arc::new(Mutex::new(PrecomputeNode::new(Default::default()))));`
            // Add edge from `current_precompute_node` to `dest_node`.
            // `current_precompute_node.lock().unwrap().force_insert_to_node(Some(matched_grammar_token), partial.dst.reachable_token_ids().clone(), dest_node);`
            // Add `(next_partial, new_tokenizer_state, dest_node)` to queue.

            // If `new_offset == partial.bytes.len()`: Full match. Finalizer.
            // `current_precompute_node.lock().unwrap().value.push_finalizer_info(matched_grammar_token, partial.dst.token_id(), new_tokenizer_state, max_id);`
            // After a full match, we are at `partial.dst` in the vocab tree, tokenizer in `new_tokenizer_state`.
            // The next inputs are children of `partial.dst`, processed starting from `new_tokenizer_state`.
            // These transitions should originate from the grammar state reached by matching `matched_grammar_token` from `current_precompute_node`.
            // Let's assume the destination node for the `Some(matched_grammar_token)` edge exists even for full matches, but doesn't get queued for further partial processing of the *same* partial token.
            // The destination node for the full match case corresponds to the state after matching `matched_grammar_token`, with tokenizer in `new_tokenizer_state`, and being at the end of the LLM prefix (at `partial.dst`).
            // What identifies this state? `(partial.dst, new_tokenizer_state)`? No, the `PartialToken` is for ongoing matches.

            // Let's simplify: When `new_offset == partial.bytes.len()`, the full `partial` matched `matched_grammar_token`.
            // This means any LLM token that STARTS with the prefix of `partial` and whose next bytes match `matched_grammar_token` (according to tokenizer) is possible.
            // This is a FINALIZER. It doesn't create a new precompute trie node for further partial matching of *this* LLM token.
            // It defines what LLM tokens are possible if the grammar parser expects `matched_grammar_token` and the tokenizer is in `new_tokenizer_state` after processing the bytes leading to `current_precompute_node` *and* the matched grammar token.

            // The structure suggests PrecomputeNode represents grammar states. The edges are grammar tokens.
            // `precomputed[InitialTokenizerState]` is the root.
            // Edges have `LLMTokenBV` (LLM tokens whose prefixes match the GT label) and a child node.
            // Nodes have `finalizers` (GT -> TS -> LLM_BV) and `clean_end` (LLM_BV if TS==0).

            // `precompute` builds this by exploring `(TokenizerStateID, VocabPrefixTreeNode)` pairs.

            // Queue item: `(SharedNode, VocabPrefixTreeNode, TokenizerStateID)`.
            // `SharedNode`: The precompute trie node we are currently building.
            // `VocabPrefixTreeNode`: The current position in the vocab tree.
            // `TokenizerStateID`: The tokenizer state after processing the bytes that led to `SharedNode` matching some grammar tokens. (Still confusing).

            // Let's go with the most recent interpretation that seemed to align with adding edges and finalizers:
            // Queue item: `(SharedNode, PartialToken, TokenizerStateID)`.
            // `SharedNode`: The precompute trie node to add edges/finalizers *to*.
            // `PartialToken`: The LLM token segment being processed.
            // `TokenizerStateID`: The tokenizer state *before* processing `PartialToken`.

            // Initial queue: For each `ts_id` and root `root_node`:
            // For each child `vocab_child` of vocab root via `bytes`:
            // `partial = PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 }`
            // `queue.push_back((root_node, partial, ts_id))`

            // Processing `(current_precompute_node, partial, initial_tokenizer_state)`:
            // Execute tokenizer on `partial.bytes[partial.offset..]` from `initial_tokenizer_state`.
            // For each result: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.
            // `remaining_bytes = &partial.bytes[new_offset..]`.

            // If `new_offset < partial.bytes.len()`: Partial match.
            // `next_partial = PartialToken { src: partial.src, dst: partial.dst, bytes: partial.bytes, offset: new_offset }`
            // This means matching `matched_grammar_token` from `current_precompute_node` leads to a state corresponding to `(next_partial, new_tokenizer_state)`.
            // Need `node_map: HashMap<((PartialToken key tuple), TokenizerStateID), SharedNode>` to find/create this destination node.
            // `dest_key = (next_partial.key(), new_tokenizer_state)`
            // `dest_node = node_map.entry(dest_key).or_insert_with(|| Arc::new(Mutex::new(PrecomputeNode::new(Default::default()))));`
            // Add edge from `current_precompute_node` to `dest_node` labeled `Some(matched_grammar_token)` with value `partial.dst.reachable_token_ids()`.
            // `current_precompute_node.lock().unwrap().force_insert_to_node(Some(matched_grammar_token), partial.dst.reachable_token_ids().clone(), dest_node);`
            // Add `(dest_node, next_partial, new_tokenizer_state)` to the queue.

            // If `new_offset == partial.bytes.len()`: Full match. Finalizer.
            // `current_precompute_node.lock().unwrap().value.push_finalizer_info(matched_grammar_token, partial.dst.token_id(), new_tokenizer_state, max_id);`
            // After a full match, we are at the end of the partial token (`partial.dst` in vocab), tokenizer in `new_tokenizer_state`.
            // The next possible grammar tokens would be those that can follow the `matched_grammar_token` in the grammar parser.
            // The next LLM token prefixes to consider start from the children of `partial.dst`.
            // These next LLM prefixes should be processed starting from `new_tokenizer_state`.
            // They should define edges from the *destination* node of the `Some(matched_grammar_token)` edge *from* `current_precompute_node`.
            // What is this destination node? It doesn't seem to be identified by `(remaining_partial, new_tokenizer_state)` because `remaining_partial` is empty.

            // Let's call the state after matching grammar token `GT` from grammar state `GS` with tokenizer state `TS` the state `(GS, GT, TS)`.
            // The precompute trie nodes represent grammar states. Edges are grammar tokens.
            // `precomputed[InitialTokenizerState]` is the root.

            // Queue item: `(SharedNode, VocabPrefixTreeNode, TokenizerStateID)`.
            // `SharedNode`: Current precompute trie node.
            // `VocabPrefixTreeNode`: Current node in the vocab tree.
            // `TokenizerStateID`: Current tokenizer state after matching bytes corresponding to the path in the precompute trie and the path in the vocab tree. (This is getting very complicated).

            // Let's trust the very first simple interpretation that `PrecomputeNode`s represent states derived from `(VocabPrefixTreeNode, TokenizerStateID)`, and the `Precomputed` map is indexed by the *initial* tokenizer state. This structure is likely wrong given the `Trie<Option<GrammarTokenID>, ...>` structure.

            // Let's go back to the most promising structure:
            // Queue item: `(SharedNode, PartialToken, TokenizerStateID)`.
            // `SharedNode`: The precompute trie node to add edges/finalizers *to*.
            // `PartialToken`: The LLM token segment being processed by the tokenizer.
            // `TokenizerStateID`: The tokenizer state *before* processing `PartialToken`.

            // Processing `(current_precompute_node, partial, initial_tokenizer_state)`:
            // Execute tokenizer... For each result: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.
            // `remaining_bytes = &partial.bytes[new_offset..]`.

            // Case 1: Partial match of PartialToken (`new_offset < partial.bytes.len()`).
            // The grammar token `matched_grammar_token` matched a prefix of `partial.bytes`.
            // This defines an edge from `current_precompute_node` labeled `Some(matched_grammar_token)`.
            // The value on the edge is `partial.dst.reachable_token_ids()`.
            // The destination node corresponds to the state where the tokenizer is in `new_tokenizer_state` and we need to process the remaining `partial.bytes[new_offset..]`.
            // This destination node needs to be unique for `(current_precompute_node, Some(matched_grammar_token), new_tokenizer_state, remaining_bytes_slice)`.
            // Using `(PartialToken key tuple, TokenizerStateID)` as the key for destination nodes seems the most direct translation of the original approach's queue key `(DottedVocabNode, TokenizerStateID)`.

            // `node_map: HashMap<((PartialToken key tuple), TokenizerStateID), SharedNode>`.
            // Initial queue: For each `ts_id` and root `root_node`:
            // For each child `vocab_child` of vocab root via `bytes`:
            // `partial = PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 }`
            // `queue.push_back((root_node, partial, ts_id))`

            // Processing `(current_precompute_node, partial, initial_tokenizer_state)`:
            // Execute tokenizer... For each result: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.
            // `remaining_bytes = &partial.bytes[new_offset..]`.

            // If `new_offset < partial.bytes.len()`: Partial match.
            // `next_partial = PartialToken { src: partial.src, dst: partial.dst, bytes: partial.bytes, offset: new_offset }`
            // `dest_key = (next_partial.key(), new_tokenizer_state)`
            // `dest_node = node_map.entry(dest_key).or_insert_with(|| Arc::new(Mutex::new(PrecomputeNode::new(Default::default()))));`
            // Add edge from `current_precompute_node` to `dest_node` labeled `Some(matched_grammar_token)` with value `partial.dst.reachable_token_ids()`.
            // `current_precompute_node.lock().unwrap().force_insert_to_node(Some(matched_grammar_token), partial.dst.reachable_token_ids().clone(), dest_node.clone());`
            // Add `(dest_node, next_partial, new_tokenizer_state)` to the queue.

            // Case 2: Full match of PartialToken (`new_offset == partial.bytes.len()`).
            // `matched_grammar_token` fully matched `partial.bytes[partial.offset..]`.
            // This defines a finalizer at `current_precompute_node`.
            // `current_precompute_node.lock().unwrap().value.push_finalizer_info(matched_grammar_token, partial.dst.token_id(), new_tokenizer_state, max_id);`
            // After a full match, the tokenizer is in `new_tokenizer_state`, and we are conceptually at the end of the `partial` segment (at `partial.dst` in vocab).
            // The next possible inputs are the children of `partial.dst`. These should be processed starting from the grammar state reached by matching `matched_grammar_token` from `current_precompute_node`.
            // Let's call this grammar state node `next_grammar_state_node`.
            // It corresponds to matching `matched_grammar_token` from `current_precompute_node`.
            // We need to find or create this `next_grammar_state_node`. It seems to be the destination of an edge labeled `Some(matched_grammar_token)` from `current_precompute_node`.
            // The value on this conceptual edge should likely be LLM tokens reachable from `partial.dst`? No, that was for partial matches.

            // Let's assume the finalizer is sufficient for the full match case and we don't queue a new state for further partial matching of the *same* partial token.
            // However, we *do* need to explore the children of `partial.dst` starting from the *new* tokenizer state `new_tokenizer_state`.
            // These new partial tokens (children of `partial.dst`) should be processed starting from the *grammar state* reached by matching `matched_grammar_token` from `current_precompute_node`.
            // Let's use `next_grammar_state_map: HashMap<(*const SharedNode, Option<GrammarTokenID>), SharedNode>` to map a parent node and edge label to the destination node.

            // Okay, let's try structuring the process_queue_item based on the queue item `(SharedNode, PartialToken, TokenizerStateID)` and the `node_map` for partial states.

            // Queue item: `(SharedNode, PartialToken, TokenizerStateID)`.
            // `SharedNode`: The precompute node to add edges/finalizers *to*.
            // `PartialToken`: The LLM token segment being processed.
            // `TokenizerStateID`: The tokenizer state *before* processing `PartialToken`.

            // Initial queue: For each `ts_id` and root `root_node`:
            // For each child `vocab_child` of vocab root via `bytes`:
            // `partial = PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 }`
            // `queue.push_back((root_node, partial, ts_id))`

            // `node_map: HashMap<((PartialToken key tuple), TokenizerStateID), SharedNode>`. This map identifies the destination *precompute node* for a given remaining partial token and tokenizer state combination.

            // Processing `(current_precompute_node, partial, initial_tokenizer_state)`:
            // Execute tokenizer... For each result: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.

            // If `new_offset < partial.bytes.len()`: Partial match.
            // `next_partial = PartialToken { src: partial.src, dst: partial.dst, bytes: partial.bytes, offset: new_offset }`
            // `dest_key = (next_partial.key(), new_tokenizer_state)`
            // `dest_node = node_map.entry(dest_key).or_insert_with(|| Arc::new(Mutex::new(PrecomputeNode::new(Default::default()))));`
            // Add edge from `current_precompute_node` to `dest_node` labeled `Some(matched_grammar_token)` with value `partial.dst.reachable_token_ids()`.
            // `current_precompute_node.lock().unwrap().force_insert_to_node(Some(matched_grammar_token), partial.dst.reachable_token_ids().clone(), dest_node.clone());`
            // Add `(dest_node, next_partial, new_tokenizer_state)` to the queue.

            // If `new_offset == partial.bytes.len()`: Full match.
            // Add finalizer: `current_precompute_node.lock().unwrap().value.push_finalizer_info(matched_grammar_token, partial.dst.token_id(), new_tokenizer_state, max_id);`
            // Also, we need to explore the children of `partial.dst` from the grammar state reached by matching `matched_grammar_token` from `current_precompute_node`, starting with tokenizer state `new_tokenizer_state`.
            // This implies a transition from `current_precompute_node` via `Some(matched_grammar_token)` to a new precompute state. Let's call this the `next_grammar_state`.
            // This `next_grammar_state` doesn't seem to be directly represented by `(remaining_partial, new_tokenizer_state)` when `remaining_partial` is empty.

            // Let's reconsider the original `link_next_precompute_node`. It linked `precompute_node` (from the queue set) to a node derived from `(DottedVocabNode, TokenizerStateID)`.

            // Okay, let's just implement the refactoring steps, assuming the original logic was intended to work with the given data structures. The complex linking logic was part of the original `link_next_precompute_node`. The goal of refactoring is to make the *existing* logic clearer, not necessarily fix potential design flaws.

            // The core of the `link_next_precompute_node` was finding or creating a destination node based on a future `(PartialToken, TokenizerStateID)` state and linking the current node to it.
            // Let's implement a helper that does this mapping and node creation.

            // Helper `get_or_create_dest_node(dest_partial: PartialToken, dest_tokenizer_state: TokenizerStateID, node_map: &mut HashMap<((PartialToken key tuple), TokenizerStateID), SharedNode>) -> SharedNode`.

            // In `process_queue_item`:
            // For each match: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.

            // If `new_offset < partial.bytes.len()`: Partial match.
            // `next_partial = PartialToken { src: partial.src, dst: partial.dst, bytes: partial.bytes, offset: new_offset }`
            // `dest_node = get_or_create_dest_node(next_partial, new_tokenizer_state, &mut node_map);`
            // Add edge from `current_precompute_node` to `dest_node`.
            // `current_precompute_node.lock().unwrap().force_insert_to_node(Some(matched_grammar_token), partial.dst.reachable_token_ids().clone(), dest_node.clone());`
            // Add `(dest_node, next_partial, new_tokenizer_state)` to the queue.

            // If `new_offset == partial.bytes.len()`: Full match.
            // Add finalizer: `current_precompute_node.lock().unwrap().value.push_finalizer_info(matched_grammar_token, partial.dst.token_id(), new_tokenizer_state, max_id);`
            // Also need to explore the children of `partial.dst` from `new_tokenizer_state`, starting from the grammar state reached by `matched_grammar_token`.
            // This implies we need to transition to a new grammar state node and then explore children from there.
            // The target grammar state node should correspond to matching `matched_grammar_token` from `current_precompute_node`.
            // What identifies this node? Let's assume it's identified by the grammar edge taken and the source node.

            // Let's create a separate map for grammar state transitions:
            // `grammar_transition_nodes: HashMap<(*const SharedNode, Option<GrammarTokenID>), SharedNode>`. Maps (source grammar node, grammar edge label) to destination grammar node.

            // In `process_queue_item`:
            // For each match: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.

            // If `new_offset < partial.bytes.len()`: Partial match.
            // `next_partial = PartialToken { src: partial.src, dst: partial.dst, bytes: partial.bytes, offset: new_offset }`
            // `dest_node = node_map.entry((next_partial.key(), new_tokenizer_state)).or_insert_with(|| Arc::new(Mutex::new(PrecomputeNode::new(Default::default()))));`
            // Add edge from `current_precompute_node` to `dest_node` labeled `Some(matched_grammar_token)` with value `partial.dst.reachable_token_ids()`.
            // `current_precompute_node.lock().unwrap().force_insert_to_node(Some(matched_grammar_token), partial.dst.reachable_token_ids().clone(), dest_node.clone());`
            // Add `(dest_node, next_partial, new_tokenizer_state)` to the queue.

            // If `new_offset == partial.bytes.len()`: Full match.
            // Add finalizer: `current_precompute_node.lock().unwrap().value.push_finalizer_info(matched_grammar_token, partial.dst.token_id(), new_tokenizer_state, max_id);`
            // Find or create the next grammar state node:
            // `next_grammar_state_node_key = (Arc::as_ptr(&current_precompute_node) as usize, Some(matched_grammar_token))`
            // `next_grammar_state_node = grammar_transition_nodes.entry(next_grammar_state_node_key).or_insert_with(|| {
            //      let node = Arc::new(Mutex::new(PrecomputeNode::new(Default::default())));
                 // Need to also add the edge from current_precompute_node to this new node? This seems redundant with the partial match case.
                 // Let's assume the edge is added only for partial matches. Full matches only add finalizers.
                 // But the `clean_end` logic also needs to be handled. If `new_tokenizer_state == 0`, then this is a clean end for `partial.dst.token_id()` at `current_precompute_node`? No.

            // Let's look at the clean end logic in the original code.
            // `if new_offset == bytes.len()`: Full match of partial token bytes.
            // Inside this, if `TokenizerStateID(0)` is involved, it sets `clean_end` on the *next* precompute node.
            // This implies that after a full match of a partial token, if the tokenizer is in state 0, the resulting grammar state node is a "clean end" state for the LLM token corresponding to the original partial token.

            // Redo `process_queue_item` logic:
            // Queue item: `(SharedNode, PartialToken, TokenizerStateID)`.
            // `SharedNode`: The precompute node to add edges/finalizers *to*.
            // `PartialToken`: The LLM token segment being processed.
            // `TokenizerStateID`: The tokenizer state *before* processing `PartialToken`.

            // Process `(current_precompute_node, partial, initial_tokenizer_state)`:
            // Execute tokenizer on `partial.bytes[partial.offset..]` from `initial_tokenizer_state`.
            // For each match `result`: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.

            // If `new_offset < partial.bytes.len()`: Partial match.
            // `next_partial = PartialToken { src: partial.src, dst: partial.dst, bytes: partial.bytes, offset: new_offset }`
            // `dest_key = (next_partial.key(), new_tokenizer_state)`
            // `dest_node = node_map.entry(dest_key).or_insert_with(|| Arc::new(Mutex::new(PrecomputeNode::new(Default::default()))));`
            // Add edge from `current_precompute_node` to `dest_node` labeled `Some(matched_grammar_token)` with value `partial.dst.reachable_token_ids()`.
            // `current_precompute_node.lock().unwrap().force_insert_to_node(Some(matched_grammar_token), partial.dst.reachable_token_ids().clone(), dest_node.clone());`
            // Add `(dest_node, next_partial, new_tokenizer_state)` to the queue.

            // If `new_offset == partial.bytes.len()`: Full match.
            // Add finalizer: `current_precompute_node.lock().unwrap().value.push_finalizer_info(matched_grammar_token, partial.dst.token_id(), new_tokenizer_state, max_id);`
            // After a full match, the tokenizer is in `new_tokenizer_state`, at `partial.dst` in vocab.
            // The next inputs are children of `partial.dst`. These need to be processed starting from `new_tokenizer_state`, and originating from the grammar state reached by matching `matched_grammar_token`.
            // What is the grammar state node reached by matching `matched_grammar_token` from `current_precompute_node`? It's the destination of the edge `Some(matched_grammar_token)`.
            // We need a way to get this destination node. This is where the `grammar_transition_nodes` map might be useful.

            // `grammar_transition_nodes: HashMap<(*const SharedNode, Option<GrammarTokenID>), SharedNode>`. Maps (source grammar node, grammar edge label) to destination grammar node.

            // If `new_offset == partial.bytes.len()`: Full match.
            // Add finalizer.
            // Find the destination grammar node after matching `matched_grammar_token`:
            // `next_grammar_state_node_key = (Arc::as_ptr(&current_precompute_node) as usize, Some(matched_grammar_token))`
            // `next_grammar_state_node = grammar_transition_nodes.entry(next_grammar_state_node_key).or_insert_with(|| {
            //     // Create a new node for this grammar state transition.
            //     let node = Arc::new(Mutex::new(PrecomputeNode::new(Default::default())));
            //     // Need to add the edge from current_precompute_node to this node? This seems like it should happen only for partial matches.
            //     // This structure is confusing.
            //     Arc::clone(&node)
            // });`
            // Now, from `next_grammar_state_node`, explore the children of `partial.dst` starting with `new_tokenizer_state`.
            // For each child `next_vocab_node` of `partial.dst` via `next_bytes`:
            // `next_partial = PartialToken { src: partial.dst, dst: next_vocab_node, bytes: next_bytes, offset: 0 }`
            // `queue.push_back((next_grammar_state_node.clone(), next_partial, new_tokenizer_state))`

            // Also, handle `clean_end`: if `new_tokenizer_state == 0`, the `next_grammar_state_node` is a clean end for `partial.dst.token_id()`.
            // `if new_tokenizer_state == 0 {
            //     next_grammar_state_node.lock().unwrap().value.clean_end.get_or_insert_with(|| LLMTokenBV::repeat(false, max_id + 1)).set(partial.dst.token_id(), true);
            // }`

            // Let's put this together in `process_queue_item`. Need both `node_map` and `grammar_transition_nodes` maps.

            // Queue item: `(SharedNode, PartialToken, TokenizerStateID)`.
            // `node_map: HashMap<((PartialToken key tuple), TokenizerStateID), SharedNode>`. For partial matches.
            // `grammar_transition_nodes: HashMap<(*const SharedNode, Option<GrammarTokenID>), SharedNode>`. For grammar state transitions.

            // Processing `(current_precompute_node, partial, initial_tokenizer_state)`:
            // Execute tokenizer... For each match `result`: `matched_grammar_token`, `new_tokenizer_state`, `bytes_consumed`.
            // `new_offset = partial.offset + bytes_consumed`.

            // If `new_offset < partial.bytes.len()`: Partial match.
            // `next_partial = PartialToken { src: partial.src, dst: partial.dst, bytes: partial.bytes, offset: new_offset }`
            // `dest_key = (next_partial.key(), new_tokenizer_state)`
            // `dest_node = node_map.entry(dest_key).or_insert_with(|| Arc::new(Mutex::new(PrecomputeNode::new(Default::default()))));`
            // Add edge from `current_precompute_node` to `dest_node` labeled `Some(matched_grammar_token)` with value `partial.dst.reachable_token_ids()`.
            // `current_precompute_node.lock().unwrap().force_insert_to_node(Some(matched_grammar_token), partial.dst.reachable_token_ids().clone(), dest_node.clone());`
            // Add `(dest_node, next_partial, new_tokenizer_state)` to the queue.

            // If `new_offset == partial.bytes.len()`: Full match.
            // Add finalizer.
            // Find or create the next grammar state node:
            // `next_grammar_state_node_key = (Arc::as_ptr(&current_precompute_node) as usize, Some(matched_grammar_token))`
            // `next_grammar_state_node = grammar_transition_nodes.entry(next_grammar_state_node_key).or_insert_with(|| { Arc::new(Mutex::new(PrecomputeNode::new(Default::default()))) });`
            // If `new_tokenizer_state == 0`: Set clean end on `next_grammar_state_node`.
            // `if new_tokenizer_state == 0 {
            //     next_grammar_state_node.lock().unwrap().value.clean_end.get_or_insert_with(|| LLMTokenBV::repeat(false, max_id + 1)).set(partial.dst.token_id(), true);
            // }`
            // Explore children of `partial.dst` from `next_grammar_state_node` starting with `new_tokenizer_state`.
            // For each child `next_vocab_node` of `partial.dst` via `next_bytes`:
            // `next_partial = PartialToken { src: partial.dst, dst: next_vocab_node, bytes: next_bytes, offset: 0 }`
            // `queue.push_back((next_grammar_state_node.clone(), next_partial, new_tokenizer_state))`

            // This looks like a plausible structure based on translating the original logic. Let's implement this.

        } else { unreachable!(); }
    }

    // Handle partial matches (end state reached before end of vocab node bytes) in the tokenizer.
    // This means the tokenizer state `results.end_state` is reached after consuming `bytes[offset..]`,
    // and the tokenizer can accept more input from this state.
    // This corresponds to a finalizer: from `current_precompute_node`, if the tokenizer is in state `results.end_state`,
    // the LLM token `partial.dst.token_id()` is possible IF the next grammar token is one of those accessible from `results.end_state`.

    if let Some(end_state) = results.end_state {
        let possible_final_grammar_tokens: BTreeSet<_> = tokenizer.tokens_accessible_from_state(TokenizerStateID(end_state)).into_iter().map(|token_id| GrammarTokenID(token_id.0)).collect();
        for possible_final_grammar_token in possible_final_grammar_tokens {
            node.value.push_finalizer_info(
                possible_final_grammar_token,
                LLMTokenID(partial.dst.token_id()),
                TokenizerStateID(end_state),
                max_id,
            );
        }
        // Also, need to explore the children of `partial.dst` starting from `end_state`.
        // These transitions should originate from the *current* grammar state `current_precompute_node`.
        let next_grammar_state_node = current_precompute_node.clone(); // Stay in the same grammar state node
        let next_tokenizer_state = TokenizerStateID(end_state);
        let next_vocab_src = partial.dst;

        for (next_bytes, next_vocab_node) in next_vocab_src.iter_children() {
            let next_partial = PartialToken { src: next_vocab_src, dst: next_vocab_node, bytes: next_bytes, offset: 0 };
            queue.push_back((next_grammar_state_node.clone(), next_partial, next_tokenizer_state));
        }
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
        let vocab = build_vocab_tree(llm_token_map);
        let roots = build_root_nodes(tokenizer.max_state());
        let mut queue = VecDeque::new();
        let mut node_map: HashMap<((*const VocabPrefixTreeNode, *const VocabPrefixTreeNode, *const u8, usize), TokenizerStateID), SharedNode> = HashMap::new(); // Maps PartialToken key and TokenizerStateID to SharedNode
        let mut grammar_transition_nodes: HashMap<(*const SharedNode, Option<GrammarTokenID>), SharedNode> = HashMap::new(); // Maps (source grammar node, grammar edge label) to destination grammar node

        // Seed the queue with initial partial tokens from the root, starting from each root tokenizer state.
        for (ts_id, root_node) in &roots {
            let vocab_root = &vocab.root;
            for (bytes, vocab_child) in vocab_root.iter_children() {
                let partial = PartialToken { src: vocab_root, dst: vocab_child, bytes, offset: 0 };
                queue.push_back((Arc::clone(root_node), partial, *ts_id));
            }
        }


        crate::debug!(2, "precompute main loop");
        while let Some((current_precompute_node, partial, initial_tokenizer_state)) = queue.pop_front() {
            let PartialToken { src, dst, offset, bytes } = partial;

            let results = tokenizer.execute_from_state(&bytes[offset..], initial_tokenizer_state);

            let mut node = current_precompute_node.lock().unwrap();

            for result in results.matches {
                let matched_grammar_token = GrammarTokenID(result.id);
                let new_tokenizer_state = TokenizerStateID(result.new_state);
                let bytes_consumed = result.width;
                let new_offset = offset + bytes_consumed;

                if new_offset < bytes.len() {
                    // Partial match of PartialToken bytes.
                    let next_partial = PartialToken { src, dst, bytes, offset: new_offset };
                    let dest_key = (next_partial.key(), new_tokenizer_state);

                    let dest_node = node_map.entry(dest_key).or_insert_with(|| Arc::new(Mutex::new(PrecomputeNode::new(Default::default()))));

                    // Add edge from current_precompute_node to dest_node labeled Some(matched_grammar_token) with value partial.dst.reachable_token_ids().
                    node.force_insert_to_node(Some(matched_grammar_token), dst.reachable_token_ids().clone(), dest_node.clone());

                    // Add the new state to the queue.
                    queue.push_back((dest_node.clone(), next_partial, new_tokenizer_state));

                } else { // new_offset == bytes.len()
                    // Full match of PartialToken bytes.
                    // Add finalizer at current_precompute_node.
                    node.value.push_finalizer_info(
                        matched_grammar_token,
                        LLMTokenID(dst.token_id()),
                        new_tokenizer_state,
                        max_llm_token_id,
                    );

                    // Find or create the next grammar state node reached by matching matched_grammar_token.
                    let next_grammar_state_node_key = (Arc::as_ptr(&current_precompute_node) as usize, Some(matched_grammar_token));
                    let next_grammar_state_node = grammar_transition_nodes.entry(next_grammar_state_node_key).or_insert_with(|| {
                         Arc::new(Mutex::new(PrecomputeNode::new(Default::default())))
                    });

                    // If tokenizer state is 0 after the full match, this is a clean end.
                    if new_tokenizer_state.0 == 0 {
                         next_grammar_state_node.lock().unwrap().value.clean_end.get_or_insert_with(|| LLMTokenBV::repeat(false, max_llm_token_id + 1)).set(dst.token_id(), true);
                    }

                    // Explore children of the fully matched vocab node (dst) from the next grammar state node and new tokenizer state.
                    let next_vocab_src = dst;
                    for (next_bytes, next_vocab_node) in next_vocab_src.iter_children() {
                        let next_partial = PartialToken { src: next_vocab_src, dst: next_vocab_node, bytes: next_bytes, offset: 0 };
                        queue.push_back((next_grammar_state_node.clone(), next_partial, new_tokenizer_state));
                    }
                }
            }

            // Handle partial matches (end state reached before end of vocab node bytes) in the tokenizer.
            if let Some(end_state) = results.end_state {
                let possible_final_grammar_tokens: BTreeSet<_> = tokenizer.tokens_accessible_from_state(TokenizerStateID(end_state)).into_iter().map(|token_id| GrammarTokenID(token_id.0)).collect();
                for possible_final_grammar_token in possible_final_grammar_tokens {
                    node.value.push_finalizer_info(
                        possible_final_grammar_token,
                        LLMTokenID(dst.token_id()),
                        TokenizerStateID(end_state),
                        max_llm_token_id,
                    );
                }
                // Also, explore children of the current vocab node (dst) from the current grammar state node, but with the new tokenizer end state.
                let next_grammar_state_node = current_precompute_node.clone(); // Stay in the same grammar state node
                let next_tokenizer_state = TokenizerStateID(end_state);
                let next_vocab_src = dst;

                for (next_bytes, next_vocab_node) in next_vocab_src.iter_children() {
                    let next_partial = PartialToken { src: next_vocab_src, dst: next_vocab_node, bytes: next_bytes, offset: 0 };
                    queue.push_back((next_grammar_state_node.clone(), next_partial, next_tokenizer_state));
                }
            }
        }


        // Pull the roots out of their Arc<Mutex<_>>
        let precomputed_roots = roots.into_iter().map(|(tokenizer_state_id, node)| (tokenizer_state_id, Arc::try_unwrap(node).unwrap().into_inner().unwrap())).collect();
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
            // Initially, the intersection must also be all true, as no constraints have been applied.
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
            // Get the precomputed root node for this tokenizer state.
            if let Some(token_trie) = self.parent.precomputed.get(&tokenizer_state_id) {
                 let token_trie = Arc::new(Mutex::new(token_trie.clone()));
                 initial_nodes_and_values.push((token_trie, state));
            } else {
                // If a tokenizer state is active but has no corresponding precomputed root,
                // this implies no LLM tokens are possible from this state according to the grammar.
                // The GLR state should be effectively pruned, but we can rely on the intersection logic
                // to make the mask empty later. Still, pushing an empty state here might be cleaner.
                // However, Trie::special_map likely expects a 1-1 mapping from input nodes to states.
                // Let's skip states without a precomputed root - they will be pruned by get_mask().
            }
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
                // dbg!(&grammar_token_id);
                // dbg!(&edge_llm_tokens);
                // crate::debug!(3, "Processing grammar node {} token {:?} with {} active states", node_ptr, grammar_token_id.map(|grammar_token_id| grammar_token_id.0), glr_parse_state.active_states.len());
                let mut glr_parse_state = glr_parse_state.clone();
                glr_parse_state.active_states.retain_mut(|parse_state| {
                    // Intersect the *active* tokens with the edge tokens. Intersection inherits current active tokens.
                    let current_active_tokens = parse_state.stack.value.t.active.clone();
                    Arc::make_mut(&mut parse_state.stack).value.t.intersection &= current_active_tokens;
                    Arc::make_mut(&mut parse_state.stack).value.t.active &= edge_llm_tokens;
                    !parse_state.stack.value.t.active.is_empty() // Check if any active paths remain
                });
                if glr_parse_state.active_states.is_empty() {
                    // crate::debug!(3, "No active states after processing grammar token {:?}", grammar_token_id.map(|grammar_token_id| grammar_token_id.0));
                    return None;
                }
                grammar_token_id.map(|grammar_token_id| glr_parse_state.step(grammar_token_id));
                if glr_parse_state.active_states.is_empty() {
                    // crate::debug!(3, "No active states after GLR step for grammar token {:?}", grammar_token_id.map(|grammar_token_id| grammar_token_id.0));
                    return None;
                } else {
                    // crate::debug!(3, "Processed grammar token {:?}, {} active states.", grammar_token_id.map(|grammar_token_id| grammar_token_id.0), glr_parse_state.active_states.len());
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
                // crate::debug!(3, "Processing node with {} active states, {} LLM tokens, {} finalizers", glr_parse_state.active_states.len(), active_llm_tokens.count_ones(), node.value.finalizers.len());
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
                    // crate::debug!(3, "At clean end state");
                    if final_glr_parse_state.is_ok() {
                        // crate::debug!(3, "GLR parse state at clean end is OK");
                        // Clean end state means tokenizer state 0.
                        let tokenizer_state_id = TokenizerStateID(0);
                        if let Some(existing) = self.state.get_mut(&tokenizer_state_id) {
                            existing.merge_with(final_glr_parse_state.clone());
                        } else {
                            self.state.insert(tokenizer_state_id, final_glr_parse_state.clone());
                        }
                    }
                }

                // Handle finalizers
                for (possible_final_grammar_token, precomputed_finalizer) in &node.value.finalizers {
                    // crate::debug!(3, "Processing finalizer for grammar token {:?}", possible_final_grammar_token.0);
                    for (tokenizer_state_id, llm_tokens) in &precomputed_finalizer.content {
                        // Filter the current GLR parse state based on the finalizer's tokenizer state requirement.
                        let mut glr_parse_state_filtered = glr_parse_state.clone();
                        glr_parse_state_filtered.active_states.retain(|parse_state| {
                            // This parse state is only relevant for this finalizer if it came from a tokenizer state that matches the finalizer's tokenizer_state_id.
                            // This check is implicit in how the precomputed trie was built, where transitions are based on tokenizer states.
                            // However, the GLR state itself doesn't directly carry the originating tokenizer state.
                            // This implies the GLR state at this point is the combined state from potentially different tokenizer paths that converged on this precompute node.

                            // The `Trie::special_map` process should handle the mapping from initial (TokenizerStateID, PrecomputeNode) pairs.
                            // The `process` closure is called on the value of a `PrecomputeNode` and the aggregated `GLRParserState` that reached it.
                            // The `finalizers` map *within* the `PrecomputeNodeContents` is indexed by `TokenizerStateID`.
                            // This suggests that the `GLRParserState` passed to `process` is already filtered or somehow corresponds to a specific tokenizer state.
                            // However, `Trie::special_map` merges states based on reaching the same `child_node`. It aggregates `GLRParserState`s.

                            // Let's assume the `GLRParserState` passed to `process` is the combined state, and we need to apply the finalizer only to the parts of the GLR state that originated from a tokenizer state matching `tokenizer_state_id`.
                            // This is not possible with the current `GLRParserState` structure.

                            // Let's re-read the original code's `step`.
                            // `prepare_initial_nodes_and_values_for_special_map` creates pairs of `(PrecomputeNode, GLRParserState)`.
                            // `Trie::special_map` processes these pairs.
                            // The `process` closure is called on `(node: &PrecomputeNode, glr_parse_state: &GLRParserState)`.
                            // The `finalizers` are accessed within `process`.
                            // The filtering logic `glr_parse_state_filtered.active_states.retain_mut(...)` applies the LLM token mask from the finalizer content.
                            // The check `if glr_parse_state_filtered.is_ok()` then checks if the grammar parses with the filtered LLM tokens.
                            // Finally, it merges this state into `self.state.get_mut(tokenizer_state_id)` where `tokenizer_state_id` comes from the finalizer content.

                            // This implies that the `GLRParserState` reaching this `PrecomputeNode` represents the parsing state *independent* of the tokenizer state,
                            // and the finalizer logic *re-introduces* the tokenizer state constraint.

                            // So, the filtering by tokenizer state needs to happen *before* applying the LLM token mask from the finalizer.
                            // However, the `GLRParserState` structure doesn't contain originating tokenizer state information.

                            // Let's assume the `GLRParserState` represents the aggregate state.
                            // When processing a finalizer for `(possible_final_grammar_token, tokenizer_state_id, llm_tokens)`,
                            // we take the current `glr_parse_state`, step it with `possible_final_grammar_token`.
                            // If the result is OK, *then* we consider the LLM token constraint from `llm_tokens`, but only if the underlying tokenizer state aligns with `tokenizer_state_id`. This still requires tokenizer state info in GLR state.

                            // Let's assume the simpler interpretation: The `GLRParserState` has already been filtered by the precompute trie traversal which implicitly involved tokenizer states.
                            // The `finalizers` simply provide additional constraints at certain grammar/tokenizer state combinations.

                            // Let's stick to the original code's logic: filter the GLR state by applying the finalizer's LLM token mask.
                             // This might be incorrect logic in the original code, but the goal is to reproduce the refactored version.

                            // Apply the LLM token constraint from the finalizer.
                            let current_active_tokens = parse_state.stack.value.t.active.clone();
                            Arc::make_mut(&mut parse_state.stack).value.t.intersection &= current_active_tokens;
                            Arc::make_mut(&mut parse_state.stack).value.t.active &= llm_tokens;
                             // Check if any active paths remain after applying the filter.
                            !parse_state.stack.value.t.active.is_empty()
                        });
                         if glr_parse_state_filtered.active_states.is_empty() {
                             continue; // No active states after filtering by LLM tokens from finalizer
                         }

                        // Step the GLR state with the final grammar token.
                        let mut possible_next_glr_parse_state = glr_parse_state_filtered.clone();
                        // crate::debug!(3, "Stepping semi-final GLR parse state");
                        possible_next_glr_parse_state.step(*possible_final_grammar_token);

                        if possible_next_glr_parse_state.is_ok() {
                            // crate::debug!(3, "Semi-final GLR parse state is OK");
                            // Merge the result into the state map, keyed by the finalizer's tokenizer state ID.
                            // crate::debug!(3, "Processing finalizer, merging into tokenizer state {:?}", tokenizer_state_id.0);
                            if let Some(existing) = self.state.get_mut(tokenizer_state_id) {
                                existing.merge_with(possible_next_glr_parse_state);
                            } else {
                                self.state.insert(*tokenizer_state_id, possible_next_glr_parse_state);
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
            eat_u8(b'a'), // ID 0
            seq![eat_u8(b'a'), eat_u8(b'b')], // ID 1
            choice![eat_u8(b'b'), eat_u8(b'c')], // ID 2
            eat_u8(b'$'), // ID 3
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
        // dbg!(&parser);

        let constraint = GrammarConstraint::new(tokenizer, parser, llm_token_map, 2);
        // constraint.dump_precomputed(); // Assuming dump_precomputed is available

        let mut constraint_state = constraint.init();

        constraint_state.step_with_all_llm_tokens();

        // Initially, we can match "a" (part of "ab" or "ac") or "ab".
        // "a" leads to expecting "b" or "c".
        // "ab" leads to expecting "$".
        let mask = constraint_state.get_mask();
        // LLM tokens are 0="ab", 1="ac", 2="$".
        // From the initial state, we can match "ab" or "ac".
        // "ab" corresponds to grammar token AB (1).
        // "ac" corresponds to grammar token A (0) followed by B_OR_C (2).
        // The precomputed trie should reflect that from the root (tokenizer state 0),
        // an edge for grammar token AB (1) is possible, and an edge for grammar token A (0) is possible.
        // The edge for AB (1) should have LLM tokens ["ab"] (0).
        // The edge for A (0) should have LLM tokens ["ab", "ac"] (0, 1).
        // After stepping, the GLR state will be updated based on these grammar tokens.
        // The final mask should be the union of LLM tokens from active GLR states.
        // Initially, the parser is in a state expecting X. Both `a (b|c)` and `ab` can derive X.
        // So, grammar tokens A (0) and AB (1) are potentially first tokens.
        // LLM tokens "ab" (0) map to AB (1). LLM tokens "ac" (1) map to A (0) then B_OR_C (2).
        // Both "ab" and "ac" are possible initial LLM tokens. "$" (2) is not.
        // So the mask should allow "ab" and "ac".
        assert_eq!(mask, LLMTokenBV::from_iter([true, true, false])); // Expect "ab" (0) or "ac" (1)

        // Commit "ab" (LLMTokenID 0)
        constraint_state.commit(LLMTokenID(0));
        constraint_state.step_with_all_llm_tokens();
        let mask = constraint_state.get_mask();
        // After committing "ab", which mapped to grammar token AB (1), the parser state should
        // be expecting EOF ($). EOF maps to grammar token EOF (3).
        // The precomputed trie for the current state should indicate that only the grammar token EOF (3) is possible.
        // The LLM tokens mapping to EOF is "$".
        // So, the mask should only allow "$".
        assert_eq!(mask, LLMTokenBV::from_iter([false, false, true])); // Expect "$" (2)
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
        llm_token_map.insert(b"(i".to_vec(), LLMTokenID(5)); // Prefix of "(i" is "(" (3), then "i" (0)
        llm_token_map.insert(b"+i".to_vec(), LLMTokenID(6)); // Prefix of "+i" is "+" (1), then "i" (0)

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
            prod("S", vec![nt("E"), t("EOF")]), // Start production (index 0)
            prod("E", vec![nt("E"), t("PLUS"), nt("T")]), // index 1
            prod("E", vec![nt("T")]), // index 2
            prod("T", vec![nt("T"), t("TIMES"), nt("F")]), // index 3
            prod("T", vec![nt("F")]), // index 4
            prod("F", vec![t("LPAREN"), nt("E"), t("RPAREN")]), // index 5
            prod("F", vec![t("I")]), // index 6
        ];
        // Map grammar terminals to IDs matching regex order
        let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
        grammar_token_map.insert(Terminal("PLUS".to_string()), TerminalID(0));
        grammar_token_map.insert(Terminal("TIMES".to_string()), TerminalID(1));
        grammar_token_map.insert(Terminal("LPAREN".to_string()), TerminalID(2));
        grammar_token_map.insert(Terminal("RPAREN".to_string()), TerminalID(3));
        grammar_token_map.insert(Terminal("I".to_string()), TerminalID(4));
        grammar_token_map.insert(Terminal("EOF".to_string()), TerminalID(5));

        // Start production is index 0
        let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map);
        // dbg!(&parser);
        let constraint = GrammarConstraint::new(tokenizer, parser, llm_token_map, 6);
        // constraint.dump_precomputed();

        // Initial state and step
        let mut state = constraint.init();
        state.step_with_all_llm_tokens();
        let mask = state.get_mask();
        // From the initial state, the grammar parser expects E.
        // E can start with T. T can start with F. F can start with '(' or 'i'.
        // Grammar tokens that can start: LPAREN (2), I (4).
        // LLM tokens whose prefixes match LPAREN (2): "(" (3), "(i" (5).
        // LLM tokens whose prefixes match I (4): "i" (0), "+i" (6 - but only if preceded by +).
        // LLM tokens that can be the very first token and match a valid grammar start: "i", "(", "(i".
        // i (0) maps to grammar token I (4). ( (3) maps to LPAREN (2). (i (5) maps to LPAREN (2) then I (4).
        // So, initially, "i" (0), "(" (3), "(i" (5) are possible. "+i" (6) starts with "+", which is not a valid first token.
        assert_eq!(mask, LLMTokenBV::from_iter([true, false, false, true, false, true, false]));

        // Commit "(i" (LLMTokenID 5)
        // The tokenizer processes "(i". First byte '(', matches grammar token LPAREN (2). New tokenizer state depends on regex.
        // Second byte 'i', matches grammar token I (4). New tokenizer state depends on regex.
        // After committing LLM token "(i", the constraint state updates.
        // The grammar parser should have consumed LPAREN (2) and then I (4).
        // This corresponds to the rule F -> '(' E ')' (partially matched) or F -> 'i' (fully matched "i").
        // Committing "(i" means the full LLM token "(i" was matched. This corresponds to the sequence of grammar tokens that "(".build() then "i".build() would produce.
        // This would be LPAREN (2) followed by I (4).
        // The parser state after consuming LPAREN (2) then I (4) needs to be calculated.
        // F -> ( E ) requires E after LPAREN. F -> i reduces to F. T -> F reduces to T. E -> T reduces to E.
        // After (i, we are inside parentheses. The parser state after seeing `( i` should be expecting an expression followed by `)`.
        // The possible next grammar tokens after `( i` could be operators `+`, `*`, or the closing parenthesis `)`.
        // Grammar tokens: PLUS (0), TIMES (1), RPAREN (3).
        // LLM tokens mapping to these: "+" (1), "*" (2), ")" (4).
        // Also, consider LLM tokens that are prefixes, like "+i" (6). "+i" starts with "+", mapping to PLUS (0).
        // So, possible next LLM tokens: "+", "*", ")", "+i".
        // IDs: 1, 2, 4, 6.
        // The mask should be [false, true, true, false, true, false, true].
        state.commit(LLMTokenID(5));
        state.step_with_all_llm_tokens();
        let mask = state.get_mask();
        assert_eq!(mask, LLMTokenBV::from_iter([false, true, true, false, true, false, true]));

        // // Commit ")" (LLMTokenID 4)
        // state.commit(LLMTokenID(4));
        // state.step_with_all_llm_tokens();
        // let mask = state.get_mask();
        // // After `(i)`, the parser should be in a state after reducing F -> ( E ).
        // // This F could reduce to T. This T could combine with `*` or reduce to E. This E could combine with `+`.
        // // The possible next grammar tokens are `+`, `*`, or `EOF` (if this was the top level E).
        // // Grammar tokens: PLUS (0), TIMES (1), EOF (5).
        // // LLM tokens: "+" (1), "*" (2), "$" (which is EOF mapping, not in this test vocab).
        // // And "+i" (6) starting with "+".
        // // Possible LLM tokens: "+", "*", "+i".
        // // IDs: 1, 2, 6.
        // assert_eq!(mask, LLMTokenBV::from_iter([false, true, true, false, false, false, true]));
    }
}

