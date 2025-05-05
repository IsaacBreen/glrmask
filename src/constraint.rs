use std::cmp::Ordering;
use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
use crate::glr::parser::ParseStateNodeContent;
use crate::glr::parser::{MergeAndIntersect, GLRParser, GLRParserState, ParseState, ParseStateKey};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use bitvec::prelude::*;
use std::collections::{BTreeMap, VecDeque};
use std::ops::BitOr;
use std::cell::{RefCell, RefMut};
use std::rc::Rc;
use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;
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

pub type NodeRc    = Rc<RefCell<PrecomputeNode>>;
pub type NodeVec   = SmallVec<[NodeRc; 4]>;

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

        // Create the vocab prefix tree.
        let mut tokens_for_vocab_prefix_tree_builder: Vec<(usize, Vec<u8>)> = vec![];
        for (content, id) in llm_token_map {
            tokens_for_vocab_prefix_tree_builder.push((id.0, content.clone()));
        }
        crate::debug!(2, "Building vocab prefix tree");
        let vocab_prefix_tree = VocabPrefixTree::build(&tokens_for_vocab_prefix_tree_builder);
        crate::debug!(2, "Done building vocab prefix tree");

        // Create the roots.
        let mut precomputed_roots: BTreeMap<TokenizerStateID, NodeRc> = BTreeMap::new();
        for tokenizer_state_id in 0..tokenizer.max_state() {
            let precompute_node = Rc::new(RefCell::new(PrecomputeNode::new(PrecomputedNodeContents::default())));
            precomputed_roots.insert(TokenizerStateID(tokenizer_state_id), precompute_node);
        }

        let mut seen : HashMap<(usize, usize, TokenizerStateID), NodeVec> = HashMap::new();
        let mut work : VecDeque<(usize, usize, &'a VocabPrefixTreeNode, &'a [u8], TokenizerStateID, NodeVec)>
                      = VecDeque::new();

        // Initialize the queue with the roots.
        for (tokenizer_state_id, precompute_node) in &precomputed_roots {
            for (bytes, new_vocab_node) in vocab_prefix_tree.root.iter_children() {
                 // Use the root of the vocab tree as the src for initial dotted nodes
                let src = &vocab_prefix_tree.root;
                let offset = 0;
                let key = (src as *const _ as usize, offset, *tokenizer_state_id);
                let nodes = smallvec![precompute_node.clone()];

                if let Some(existing) = seen.get_mut(&key) {
                    existing.extend(nodes.iter().cloned());
                } else {
                    seen.insert(key, nodes.clone());
                    work.push_back((src.prefix_length(), offset, src, bytes, *tokenizer_state_id, nodes));
                }
            }
        }

        let mut merge_map: HashMap<SmallVec<[usize;4]>, NodeRc> = HashMap::new();

        macro_rules! enqueue {
            (src = $src:expr, off = $offset:expr, nodes = $nodes:expr, tok_state = $tok_state:expr) => {
                let key = ($src as *const _ as usize, $offset, $tok_state);
                if let Some(existing) = seen.get_mut(&key) {
                    existing.extend($nodes.iter().cloned());
                } else {
                    seen.insert(key, $nodes.clone());
                    // Note: DottedVocabNode is not used here, replaced by direct components
                    // Need dst and bytes to put into the work queue.
                    // We use the dst and bytes from the outer loop, which is maybe not right?
                    // No, the dst and bytes should come from the current position in the vocab tree.
                    // This macro needs rethinking based on the new loop structure.
                    // Let's define enqueue inline where it's used instead of a macro.
                    panic!("enqueue macro should not be used in the new structure");
                }
            };
        }


        let all_true = LLMTokenBV::repeat(true, max_llm_token_id+1);

        crate::debug!(2, "precompute main loop");
        // The tuple carried in work is (prefix_len, offset, src, bytes, tok_state, nodes)
        while let Some((_pref_len, offset, src, bytes, tok_state, mut nodes)) = work.pop_front() {
            crate::debug!(3, "Popped from queue. Tokenizer state: {}, Queue size: {}, Precomputed nodes: {}, Prefix length: {}, Offset: {}, Total length (prefix + offset): {}, Prefix: {}, Bytes: {}",
                tok_state.0,
                work.len(),
                nodes.len(),
                src.prefix_length(),
                offset,
                src.prefix_length() + offset,
                String::from_utf8_lossy(src.prefix()),
                String::from_utf8_lossy(bytes),
            );
            let dst = src; // In the new model, we process bytes of the current vocab node 'src'. There's no 'dst' from the dotted node concept.

            let mut node_addrs: SmallVec<[usize; 4]> = nodes.iter().map(|n| Rc::as_ptr(n) as usize).collect();
            node_addrs.sort_unstable(); // Sort for consistent key in merge_map

            if nodes.len() > 3 {
                // Merge the nodes
                if let Some(existing) = merge_map.get(&node_addrs) {
                    crate::debug!(3, "Merging {} nodes. Found existing merge", nodes.len());
                    nodes = smallvec![existing.clone()];
                } else {
                    crate::debug!(3, "Merging {} nodes. No existing merge", nodes.len());
                    let new_precomputed_node = Rc::new(RefCell::new(PrecomputeNode::new(PrecomputedNodeContents::default())));
                    for precomputed_node_rc in nodes.iter() {
                        new_precomputed_node.borrow_mut().force_insert_to_node(None, all_true.clone(), precomputed_node_rc);
                    }
                    merge_map.insert(node_addrs.clone(), new_precomputed_node.clone());
                    nodes = smallvec![new_precomputed_node.clone()];
                }
            }

            let reachable_tokens = src.reachable_token_ids().clone();   // BitVec
            let mut cur_tokenizer_state = tok_state.0;

            // Iterate once over every byte of the current vocabulary entry, starting from offset
            for idx in offset..bytes.len() {
                let b = bytes[idx];
                let byte_slice = &bytes[idx..];

                let exec = tokenizer.execute_from_state(byte_slice, TokenizerStateID(cur_tokenizer_state));

                if exec.consumed == 0 {
                    // No progress made by the tokenizer with this byte/state combination.
                    // This path is dead for this tokenizer state from this point on.
                    // The loop will naturally break if exec.end_state is None or cur_tokenizer_state doesn't change.
                    if exec.end_state.is_none() || exec.end_state == Some(cur_tokenizer_state) {
                         break;
                    }
                }

                let new_offset = idx + exec.consumed; // How far we've consumed *from the beginning of bytes* in this tokenizer step

                for m in exec.matches {
                    let grammar_id = GrammarTokenID(m.id);
                    let llm_tokens = reachable_tokens.clone();

                    let mut next_nodes: SmallVec<[NodeRc; 4]> = SmallVec::new();

                    for node_rc in &nodes {
                        let mut node = node_rc.borrow_mut();
                        // Check if an edge with this grammar_id already exists and try to merge
                        let mut existing_edge_found = false;
                        if let Some(edges) = node.get_mut(&Some(grammar_id)) {
                            for (edge_llm_tokens, child_rc) in edges.iter_mut() {
                                // For simplicity in the byte-by-byte scan, we create a new node for each match
                                // rather than trying to merge into existing child nodes within the loop.
                                // Merging into existing child nodes is handled by the `seen` map and the `work` queue.
                            }
                        }

                        let child_rc = node.force_insert_to_new_node(
                            Some(grammar_id),
                            llm_tokens.clone(),
                            PrecomputedNodeContents::default()
                        );
                        next_nodes.push(child_rc);

                    }

                    // If we have reached the end of the LLM token bytes, mark clean_end
                    if new_offset == bytes.len() {
                         for child_rc in &next_nodes {
                             child_rc.borrow_mut().value.clean_end
                                     .get_or_insert_with(|| LLMTokenBV::repeat(false, max_llm_token_id+1))
                                     .set(src.token_id(), true);
                         }
                    }

                    // Enqueue the rest of the bytes (if any) starting from the new_offset
                    if new_offset < bytes.len() {
                        let next_src = src; // Stay on the same vocab node
                        let next_tokenizer_state = TokenizerStateID(exec.end_state.unwrap_or(0)); // Use the tokenizer end state

                        let key = (next_src as *const _ as usize, new_offset, next_tokenizer_state);
                        if let Some(existing) = seen.get_mut(&key) {
                             existing.extend(next_nodes.iter().cloned());
                        } else {
                             seen.insert(key, next_nodes.clone());
                             work.push_back((next_src.prefix_length(), new_offset, next_src, bytes, next_tokenizer_state, next_nodes.clone()));
                        }
                    }
                }

                // advance the tokenizer state machine for the next byte
                if let Some(s) = exec.end_state {
                    cur_tokenizer_state = s;
                } else {
                    // Tokenizer is dead from this state with the current byte. Stop processing this byte sequence for this branch.
                    break;
                }

            }

            // Handle finalizers after processing all bytes for this vocab node path
            if let Some(end_state) = tokenizer.execute_from_state(bytes, tok_state).end_state {
                 // Ensure the tokenizer can reach a final state *after* consuming the current bytes
                 // This check is more about what can *follow* these bytes to form a grammar token.
                 // We need to check what grammar tokens are possible *immediately* after the current bytes
                 // if the tokenizer is in end_state.
                 let final_tokens: BTreeSet<_> = tokenizer.tokens_accessible_from_state(TokenizerStateID(end_state)).into_iter().map(|token_id| GrammarTokenID(token_id.0)).collect();

                 for possible_final_grammar_token in final_tokens {
                     for node_rc in &nodes {
                         node_rc.borrow_mut().value.push_finalizer_info(
                             possible_final_grammar_token,
                             LLMTokenID(src.token_id()), // The LLM token is the one associated with the src node
                             TokenizerStateID(end_state),
                             max_llm_token_id
                         );
                     }
                 }

                 // Enqueue the children of the current vocab node starting from tokenizer state 0
                 let next_src = src; // Stay on the same vocab node concept, moving to its children in the vocab tree
                 for (next_bytes, next_dst_vocab_node) in next_src.iter_children() {
                     let next_offset = 0;
                     let next_tokenizer_state = TokenizerStateID(0); // Start tokenizer from initial state for the new token

                     let key = (next_dst_vocab_node as *const _ as usize, next_offset, next_tokenizer_state);
                     if let Some(existing) = seen.get_mut(&key) {
                         existing.extend(nodes.iter().cloned());
                     } else {
                         seen.insert(key, nodes.clone());
                         work.push_back((next_dst_vocab_node.prefix_length(), next_offset, next_dst_vocab_node, next_bytes, next_tokenizer_state, nodes.clone()));
                     }
                 }

            }
        }


        // Pull the roots out of their Rc<RefCell<_>>
        let precomputed_roots = precomputed_roots.into_iter()
                                                 .map(|(tokenizer_state_id, node)| (tokenizer_state_id, Rc::try_unwrap(node).unwrap().into_inner()))
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

    fn prepare_initial_nodes_and_values_for_special_map(&mut self, llm_tokens: &LLMTokenBV) -> Vec<(NodeRc, GLRParserState<'a, LLMTokenInfo>)> {
        let mut initial_nodes_and_values: Vec<(NodeRc, GLRParserState<'_, LLMTokenInfo>)> = Vec::new();
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
            // Need to convert the cloned Trie back to Rc<RefCell<_>> for special_map
            // This conversion might be tricky. The precomputed structure is owned Trie.
            // special_map expects Arc<Mutex<_>> or Rc<RefCell<_>>.
            // Let's assume special_map is updated to work with owned Trie or provides a way to wrap.
            // For now, let's wrap the clone. This is inefficient but matches the required type.
            // A better approach would be for special_map to work with references or slices of Tries.
            let token_trie_rc = Rc::new(RefCell::new(token_trie));
            initial_nodes_and_values.push((token_trie_rc, state));
        }
        initial_nodes_and_values
    }

    pub fn step(&mut self, llm_tokens: &LLMTokenBV) {
        crate::debug!(2, "Stepping grammar constraint state with tokenizer states {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens);

        self.state = BTreeMap::new();

        Trie::special_map(
            // Input: Vec<(Rc<RefCell<PrecomputeNode>>, GLRParserState<'_, LLMTokenInfo>)>
            initial_nodes_and_values,
            // step
            // Input: &GLRParserState<'_, LLMTokenInfo>, GrammarTokenID, &LLMTokenBV, &Rc<RefCell<PrecomputeNode>>
            // Output: Option<GLRParserState<'_, LLMTokenInfo>>
            |glr_parse_state, grammar_token_id, edge_llm_tokens, child_node| {
                let node_ptr = std::ptr::addr_of!(*child_node) as usize;
                crate::debug!(3, "Processing grammar node {} token {:?} with {} active states", node_ptr, grammar_token_id.map(|grammar_token_id| grammar_token_id.0), glr_parse_state.active_states.len());
                let mut glr_parse_state = glr_parse_state.clone();
                glr_parse_state.active_states.retain_mut(|parse_state| {
                    // Intersect the *active* tokens with the edge tokens. Intersection inherits current active tokens.
                    let current_active_tokens = parse_state.stack.value.t.active.clone();
                    // The intersection should reflect the guarantee *from below*.
                    // When stepping across an edge, the intersection for the states *after* the edge
                    // should be the intersection *before* the edge intersected with the edge_llm_tokens.
                    // However, special_map's current logic passes the edge_llm_tokens only to modify the active set.
                    // The commit function relies on the intersection being a guarantee from *below* the current point in the GSS.
                    // Modifying intersection here based on edge_llm_tokens seems incorrect for the commit optimization.
                    // Let's revert the intersection modification here. Intersection is only updated during GSS merging.
                    // Arc::make_mut(&mut parse_state.stack).value.t.intersection &= current_active_tokens; // This line seems wrong based on intersection definition
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
                        // Similar to the step function, only modify active here. Intersection is handled by GSS merge.
                        // Arc::make_mut(&mut parse_state.stack).value.t.intersection &= current_active_tokens; // This seems wrong
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
                                // Similar to the step function, only modify active here.
                                // Arc::make_mut(&mut parse_state.stack).value.t.intersection &= current_active_tokens; // This seems wrong
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

        let constraint = GrammarConstraint::new(tokenizer, parser, llm_token_map, 2);
        // constraint.dump_precomputed(); // dump_precomputed might need updates for Rc/RefCell

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
        // constraint.dump_precomputed(); // dump_precomputed might need updates for Rc/RefCell

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

