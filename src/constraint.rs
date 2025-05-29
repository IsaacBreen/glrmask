// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::BitOr;
use std::sync::{Arc, Mutex};
use std::cell::{OnceCell, RefCell};

use bimap::BiBTreeMap;
use bitvec::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

use crate::constraint_extra::{calculate_final_stats, print_precompute_stats, PrecomputeStats};
use crate::datastructures::charmap::TrieMap;
use crate::datastructures::gss::{print_gss_forest, GSSNode, PathAccumulator, intersect_tokens_and_prune_arc, gather_gss_stats, reset_tokens};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::{EdgeInserter, Trie};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::datastructures::ArcPtrWrapper;
use crate::finite_automata::Regex;
use crate::glr::parser::{
    GLRParser, GLRParserState, ParseState, ParseStateEdgeContent,
};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;

pub type LLMTokenBV = HybridBitset;
pub type GrammarTokenBV = BitVec;

// -----------------------------------------------------------------------------
// Small data-types used by the constraint
// -----------------------------------------------------------------------------
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LLMTokenInfo {
    pub active:       LLMTokenBV,
    pub intersection: LLMTokenBV,
    // Removed: pub terminals:    Arc<GSSNode<TerminalID, ()>>,
}

impl Default for LLMTokenInfo {
    fn default() -> Self {
        Self {
            active:       HybridBitset::new(),
            intersection: HybridBitset::max_ones(),
            // Removed terminals initialization
        }
    }
}

impl PathAccumulator for LLMTokenInfo {
    fn union_assign(&mut self, other: Self) {
        self.active |= &other.active;
        self.intersection &= &other.intersection;
        // Removed terminals merge
    }

    fn intersect_assign(&mut self, right: Self) {
        // The 'pop' or 'intersect' operation for LLMTokenInfo means:
        // - The active set of the resulting path is the intersection of the left path's active set and the right path's active set.
        // - The intersection set also becomes stricter (AND).
        // - Terminals from the right path are kept (this logic is now removed).
        self.active &= &right.active;
        self.intersection &= &right.intersection;
        // Removed terminals assignment
    }
}


impl std::fmt::Debug for LLMTokenInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fmt_bv = |bv: &LLMTokenBV| -> String {
            format!("{:?}", bv)
        };

        f.debug_struct("LLMTokenInfo")
            .field("active", &fmt_bv(&self.active))
            .field("intersection", &fmt_bv(&self.intersection))
            // Removed terminals field
            .finish()
    }
}

// -----------------------------------------------------------------------------
// Pre-computation node values
// -----------------------------------------------------------------------------
#[derive(Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedFinalizer {
    pub content: BTreeMap<TokenizerStateID, LLMTokenBV>,
}

impl JSONConvertible for PrecomputedFinalizer {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("content".to_string(), self.content.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let content = obj.remove("content").ok_or_else(|| "Missing field content for PrecomputedFinalizer".to_string())
                                   .and_then(|n| BTreeMap::<TokenizerStateID, LLMTokenBV>::from_json(n))?;
                Ok(PrecomputedFinalizer { content })
            }
            _ => Err("Expected JSONNode::Object for PrecomputedFinalizer".to_string()),
        }
    }
}


impl PrecomputedFinalizer {
    fn new(tokens: LLMTokenBV, tokenizer_state: TokenizerStateID) -> Self {
        Self {
            content: BTreeMap::from([(tokenizer_state, tokens)]),
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedNodeContents {
    finalizers: BTreeMap<GrammarTokenID, PrecomputedFinalizer>,
    pub clean_end: Option<LLMTokenBV>,
}

impl JSONConvertible for PrecomputedNodeContents {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("finalizers".to_string(), self.finalizers.to_json());
        obj.insert("clean_end".to_string(), self.clean_end.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let finalizers = obj.remove("finalizers").ok_or_else(|| "Missing field finalizers for PrecomputedNodeContents".to_string())
                                    .and_then(|n| BTreeMap::<GrammarTokenID, PrecomputedFinalizer>::from_json(n))?;
                let clean_end = obj.remove("clean_end").ok_or_else(|| "Missing field clean_end for PrecomputedNodeContents".to_string())
                                   .and_then(Option::<LLMTokenBV>::from_json)?;
                Ok(PrecomputedNodeContents { finalizers, clean_end })
            }
            _ => Err("Expected JSONNode::Object for PrecomputedNodeContents".to_string()),
        }
    }
}


impl PrecomputedNodeContents {
    pub(crate) fn finalizers(&self) -> &BTreeMap<GrammarTokenID, PrecomputedFinalizer> {
        &self.finalizers
    }

    fn push_finalizer_info(
        &mut self,
        grammar_token: GrammarTokenID,
        llm_token: LLMTokenID, 
        tokenizer_state: TokenizerStateID,
    ) {
        let mut bv = HybridBitset::new();
        bv.insert(llm_token.0); 

        self.finalizers
            .entry(grammar_token)
            .and_modify(|f| {
                f.content
                    .entry(tokenizer_state)
                    .and_modify(|existing| *existing |= &bv)
                    .or_insert(bv.clone());
            })
            .or_insert_with(|| PrecomputedFinalizer::new(bv, tokenizer_state));
    }
}

pub type PrecomputeNode =
    Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;

pub type Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarConstraint {
    pub(crate) tokenizer:        Regex,
    pub(crate) parser:           GLRParser,
    pub(crate) precomputed:      Precomputed,
    pub(crate) llm_token_map:    BiBTreeMap<Vec<u8>, LLMTokenID>, 
    pub(crate) token_name_map:   BiBTreeMap<String, usize>,
    pub(crate) max_original_llm_token_id: usize, 
    pub(crate) original_to_internal_id_bimap: BiBTreeMap<usize, usize>, 
    pub(crate) internal_max_llm_token: usize, 
}

impl GrammarConstraint {
    pub fn assert_eq(&self, other: &Self) {
        assert_eq!(self.tokenizer, other.tokenizer);
        assert_eq!(self.parser, other.parser);
        assert_eq!(self.precomputed, other.precomputed);
        assert_eq!(self.llm_token_map, other.llm_token_map);
        assert_eq!(self.token_name_map, other.token_name_map);
        assert_eq!(self.max_original_llm_token_id, other.max_original_llm_token_id);
        assert_eq!(self.original_to_internal_id_bimap, other.original_to_internal_id_bimap);
        assert_eq!(self.internal_max_llm_token, other.internal_max_llm_token);
    }
}

impl JSONConvertible for GrammarConstraint {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("tokenizer".to_string(), self.tokenizer.to_json());
        obj.insert("parser".to_string(), self.parser.to_json());
        obj.insert("precomputed".to_string(), self.precomputed.to_json());
        obj.insert("llm_token_map".to_string(), self.llm_token_map.to_json());
        obj.insert("token_name_map".to_string(), self.token_name_map.to_json());
        obj.insert("max_original_llm_token_id".to_string(), self.max_original_llm_token_id.to_json());
        obj.insert("original_to_internal_id_bimap".to_string(), self.original_to_internal_id_bimap.to_json());
        obj.insert("internal_max_llm_token".to_string(), self.internal_max_llm_token.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let tokenizer = obj.remove("tokenizer").ok_or_else(|| "Missing field tokenizer".to_string())
                                   .and_then(Regex::from_json)?;
                let parser = obj.remove("parser").ok_or_else(|| "Missing field parser".to_string())
                                .and_then(GLRParser::from_json)?;
                let precomputed = obj.remove("precomputed").ok_or_else(|| "Missing field precomputed".to_string())
                                     .and_then(|n| Precomputed::from_json(n))?;

                let llm_token_map = obj.remove("llm_token_map").ok_or_else(|| "Missing field llm_token_map".to_string())
                                       .and_then(|n| BiBTreeMap::<Vec<u8>, LLMTokenID>::from_json(n))?;
                let token_name_map = obj.remove("token_name_map").ok_or_else(|| "Missing field token_name_map".to_string())
                                        .and_then(|n| BiBTreeMap::<String, usize>::from_json(n))?;
                let max_original_llm_token_id = obj.remove("max_original_llm_token_id").ok_or_else(|| "Missing field max_original_llm_token_id".to_string())
                                                   .and_then(usize::from_json)?;
                let original_to_internal_id_bimap = obj.remove("original_to_internal_id_bimap").ok_or_else(|| "Missing field original_to_internal_id_bimap".to_string())
                                                       .and_then(|n| BiBTreeMap::<usize, usize>::from_json(n))?;
                let internal_max_llm_token = obj.remove("internal_max_llm_token").ok_or_else(|| "Missing field internal_max_llm_token".to_string())
                                                  .and_then(usize::from_json)?;
                Ok(GrammarConstraint {
                    tokenizer,
                    parser,
                    precomputed,
                    llm_token_map,
                    token_name_map,
                    max_original_llm_token_id,
                    original_to_internal_id_bimap,
                    internal_max_llm_token,
                })
            }
            _ => Err("Expected JSONNode::Object for GrammarConstraint".to_string()),
        }
    }
}


impl GrammarConstraint {
    pub(crate) fn setup_llm_token_mappings(
        original_llm_token_map: &LLMTokenMap,
    ) -> BiBTreeMap<usize, usize>
    {
        // // TODO: delete this
        // let mut original_to_internal_id_bimap = BiBTreeMap::new();
        // for (_, id) in original_llm_token_map.iter() {
        //     original_to_internal_id_bimap.insert(id.0, id.0);
        // }
        // return original_to_internal_id_bimap;


        let mut sorted_tokens_with_original_ids: Vec<(Vec<u8>, LLMTokenID)> = original_llm_token_map
            .iter()
            .map(|(bytes, original_id)| (bytes.clone(), *original_id))
            .collect();
        sorted_tokens_with_original_ids.sort_by(|(bytes_a, _), (bytes_b, _)| bytes_a.cmp(bytes_b));

        let mut original_to_internal_id_bimap = BiBTreeMap::new();
        let mut internal_id_counter = 0;

        for (_bytes, original_llm_id) in sorted_tokens_with_original_ids {
            let internal_llm_id_val = internal_id_counter;
            original_to_internal_id_bimap.insert(original_llm_id.0, internal_llm_id_val);
            internal_id_counter += 1;
        }

        original_to_internal_id_bimap
    }

    pub fn new(
        tokenizer:        Regex,
        parser:           GLRParser,
        llm_token_map:    LLMTokenMap, 
        token_name_map:   BiBTreeMap<String, usize>,
        max_original_llm_token_id: usize, 
    ) -> Self {
        let original_to_internal_id_bimap = Self::setup_llm_token_mappings(&llm_token_map);

        let internal_max_llm_token = original_to_internal_id_bimap.iter().map(|(_, id)| *id).max().unwrap_or(0);

        let mut internal_llm_token_map_for_precompute = BiBTreeMap::new();
        for (bytes, original_id) in llm_token_map.iter() {
            if let Some(internal_id_val) = original_to_internal_id_bimap.get_by_left(&original_id.0) {
                internal_llm_token_map_for_precompute.insert(bytes.clone(), LLMTokenID(*internal_id_val));
            }
        }

        let precomputed = Self::precompute(
            &tokenizer,
            &internal_llm_token_map_for_precompute, 
            &token_name_map,
            internal_max_llm_token, 
        );

        Self {
            tokenizer,
            parser,
            precomputed,
            llm_token_map, 
            token_name_map,
            max_original_llm_token_id,
            original_to_internal_id_bimap,
            internal_max_llm_token,
        }
    }

    pub fn precompute(
        tokenizer:        &Regex,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>, 
        token_name_map:   &BiBTreeMap<String, usize>,
        internal_max_llm_token: usize,                       
    ) -> Precomputed {
        let mut helper = Precomputer::new(
            tokenizer,
            internal_llm_token_map,    
            internal_max_llm_token, 
            100, 
        );

        helper.run_dfs();
        helper.prune_precomputed_graph();
        helper.merge_nodes();
        helper.finish(token_name_map)
    }

    pub fn init(&self) -> GrammarConstraintState<'_> {
        let base_set_for_info = HybridBitset::ones(self.internal_max_llm_token + 1);
        let initial_llm_token_acc = LLMTokenInfo { 
            active:       base_set_for_info.clone(),
            intersection: base_set_for_info,
        };
        let mut state = BTreeMap::new();
        state.insert(
            self.tokenizer.initial_state_id(),
            self.parser.init_glr_parser_with_acc(initial_llm_token_acc),
        );

        GrammarConstraintState { parent: self, state }
    }

    #[inline]
    fn original_id_to_internal(&self, original_id: LLMTokenID) -> Option<LLMTokenID> {
        self.original_to_internal_id_bimap.get_by_left(&original_id.0).map(|internal_val| LLMTokenID(*internal_val))
    }

    #[inline]
    fn internal_id_to_original(&self, internal_id: LLMTokenID) -> Option<LLMTokenID> {
        self.original_to_internal_id_bimap.get_by_right(&internal_id.0).map(|original_val| LLMTokenID(*original_val))
    }

    #[allow(dead_code)] 
    fn original_bv_to_internal(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        let mut internal_bv = HybridBitset::new();
        for original_id_val in original_bv.iter() {
            let internal_id_val = self.original_to_internal_id_bimap.get_by_left(&(original_id_val as usize)).expect(format!("Original ID {} not found in original_to_internal_id_bimap", original_id_val).as_str());
            internal_bv.insert(*internal_id_val as usize);
        }
        internal_bv
    }

    fn internal_bv_to_original(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        let mut original_bv = HybridBitset::new();
        for internal_id_val in internal_bv.iter() {
            let original_id_val = self.original_to_internal_id_bimap.get_by_right(&(internal_id_val as usize)).expect(format!("Internal ID {} not found in original_to_internal_id_bimap", internal_id_val).as_str());
            original_bv.insert(*original_id_val as usize);
        }
        original_bv
    }
}

struct Precomputer<'r> {
    tokenizer:        &'r Regex,
    vocab:            VocabPrefixTree,
    roots:            BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>>,
    possible_matches: RefCell<BTreeMap<*const VocabPrefixTreeNode, BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>>>,
    possible_matches_from_root_vocab_node: RefCell<BTreeMap<TokenizerStateID, BTreeSet<GrammarTokenID>>>,
    all_llm_tokens:   HybridBitset,
    merge_threshold:  usize,
    pb:               ProgressBar,
    stats:            PrecomputeStats,
}

impl<'r> Precomputer<'r> {
    fn new(
        tokenizer:        &'r Regex,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>, 
        internal_max_llm_token: usize,                       
        merge_threshold:  usize,
    ) -> Self {
        let tokens: Vec<(usize, Vec<u8>)> = internal_llm_token_map 
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone())) 
            .collect();

        crate::debug!(2, "Building vocab prefix tree");
        let vocab = VocabPrefixTree::build(&tokens);
        crate::debug!(2, "Done building vocab prefix tree");

        let mut roots = BTreeMap::new();
        for sid in 0..tokenizer.max_state() {
            roots.insert(
                TokenizerStateID(sid),
                Arc::new(Mutex::new(PrecomputeNode::new(
                    PrecomputedNodeContents::default(),
                ))),
            );
        }

        let total_nodes = count_vocab_nodes(&vocab.root);
        let pb = ProgressBar::new(total_nodes);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] \
                           [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta})")
                .expect("progress-bar"),
        );

        Self {
            tokenizer,
            vocab,
            roots,
            possible_matches: RefCell::new(BTreeMap::new()),
            possible_matches_from_root_vocab_node: RefCell::new(BTreeMap::new()),
            all_llm_tokens: HybridBitset::ones(internal_max_llm_token + 1),
            merge_threshold,
            pb,
            stats: PrecomputeStats::default(),
        }
    }

    fn possible_matches(&self, vocab_node: &VocabPrefixTreeNode, tokenizer_state_id: TokenizerStateID) -> BTreeMap<GrammarTokenID, LLMTokenBV> {
        let cache_key_ptr = vocab_node as *const VocabPrefixTreeNode;

        if let Some(cached_for_vocab_node) = self.possible_matches.borrow().get(&cache_key_ptr) {
            if let Some(cached_result) = cached_for_vocab_node.get(&tokenizer_state_id) {
                return cached_result.clone();
            }
        }

        self.possible_matches.borrow_mut().entry(cache_key_ptr).or_default().insert(tokenizer_state_id, BTreeMap::new());

        let mut result_map: BTreeMap<GrammarTokenID, LLMTokenBV> = BTreeMap::new();

        for (segment_bytes, child_vocab_arc) in vocab_node.iter_children() {
            let exec_result = self.tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);
            for token in &exec_result.matches {
                let grammar_token_id = GrammarTokenID(token.id);
                let applicable_tokens = child_vocab_arc.reachable_token_ids();
                *result_map.entry(grammar_token_id).or_insert_with(LLMTokenBV::new) |= applicable_tokens;
            }
            if let Some(final_state_val) = exec_result.end_state {
                let matches_possible_from_tokenizer_state: BTreeSet<_> = self.tokenizer.tokens_accessible_from_state(TokenizerStateID(final_state_val)).into_iter().collect();
                let matches_here: BTreeSet<_> = exec_result.matches.iter().map(|m| GrammarTokenID(m.id)).collect();
                let possible_new_matches = &matches_possible_from_tokenizer_state - &matches_here;
                if !possible_new_matches.is_empty() {
                    // Possible matches from the child vocab node (considering longer LLM tokens)
                    let longer_token_results = self.possible_matches(child_vocab_arc, TokenizerStateID(final_state_val));
                    for (token, bv) in longer_token_results {
                        *result_map.entry(token).or_insert_with(LLMTokenBV::new) |= bv;
                    }
                    // Possible matches from the root vocab node (considering the LLM token that this vocab node represents)
                    // If a given grammar token matches for any future LLM tokens, then the grammar token can match from the LLM token represented by *this* vocab node as well.
                    let new_token_results = self.possible_matches_from_root_vocab_node(TokenizerStateID(final_state_val));
                    for token in new_token_results {
                        result_map.entry(token).or_insert_with(LLMTokenBV::new).set(child_vocab_arc.token_id(), true);
                    }
                }
            }
        }

        self.possible_matches.borrow_mut().entry(cache_key_ptr).or_default().insert(tokenizer_state_id, result_map.clone());

        result_map
    }

    fn possible_matches_from_vocab_node(&self, tokenizer_state_id: TokenizerStateID, vocab_node: &VocabPrefixTreeNode) -> BTreeSet<GrammarTokenID> {
        if let Some(cached_result) = self.possible_matches_from_root_vocab_node.borrow().get(&tokenizer_state_id) {
            return cached_result.clone();
        }

        let mut result = BTreeSet::new();
        for (segment_bytes, child_vocab_arc) in vocab_node.iter_children() {
            let exec_result = self.tokenizer.execute_from_state(segment_bytes, tokenizer_state_id);
            for token in &exec_result.matches {
                let grammar_token_id = GrammarTokenID(token.id);
                result.insert(grammar_token_id);
            }
            if let Some(final_state_val) = exec_result.end_state {
                result.extend(self.possible_matches_from_vocab_node(TokenizerStateID(final_state_val), child_vocab_arc));
                result.extend(self.possible_matches_from_vocab_node(TokenizerStateID(final_state_val), &self.vocab.root));
            }
        }

        result
    }

    fn possible_matches_from_root_vocab_node(&self, tokenizer_state_id: TokenizerStateID) -> BTreeSet<GrammarTokenID> {
        self.possible_matches_from_vocab_node(tokenizer_state_id, &self.vocab.root)
    }

    fn run_dfs(&mut self) {
        let mut assoc: BTreeMap<
            TokenizerStateID,
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        > = BTreeMap::new();

        for (sid, arc) in &self.roots {
            assoc
                .entry(*sid)
                .or_default()
                .insert(ArcPtrWrapper::new(arc.clone()));
        }

        crate::debug!(2, "Starting precompute DFS");
        crate::debug!(3, "Roots for each tokenizer state:");
        for (sid, root) in &self.roots {
            crate::debug!(3, "  {}: {:p}", sid.0, Arc::as_ptr(root));
        }
        self.dfs(&self.vocab.root, assoc);
        crate::debug!(2, "Finished precompute DFS");
        self.pb.finish_with_message("Precomputation complete");
        crate::debug!(2, "Precomputation complete");
    }

    fn merge_nodes(&mut self) {
        crate::debug!(2, "Merging nodes: first collecting unique roots");
        let mut unique = BTreeMap::new();
        for (tokenizer_state_id, root) in &self.roots {
            crate::debug!(4, "Processing root {:?}", tokenizer_state_id);
            let new_root = unique
                .entry(root.lock().unwrap().clone())
                .or_insert_with(|| root.clone());
            *new_root = root.clone();
        }

        crate::debug!(2, "Merging nodes: second pass rewriting roots");
        for (_tokenizer_state_id, root) in &mut self.roots {
            let new_root = unique
                .get(&root.lock().unwrap().clone())
                .unwrap()
                .clone();
            *root = new_root;
        }
    }

    fn prune_precomputed_graph(&mut self) {
        crate::debug!(2, "Starting precomputed graph pruning.");

        let mut completable_cache: HashMap<*const Mutex<PrecomputeNode>, LLMTokenBV> = HashMap::new();

        for root_arc in self.roots.values() {
            let mut recursion_stack_for_pass1 = HashSet::new();
            self.compute_completable_tokens_recursive(root_arc, &mut recursion_stack_for_pass1, &mut completable_cache);
        }
        crate::debug!(3, "Completed pass 1 (compute completable tokens). Cache size: {}", completable_cache.len());

        let mut visited_nodes_for_pruning_pass = HashSet::new();
        for root_arc in self.roots.clone().values() {
            self.filter_and_prune_edges_recursive(root_arc, &mut visited_nodes_for_pruning_pass, &completable_cache);
        }
        crate::debug!(3, "Completed pass 2 (filter and prune edges).");

        crate::debug!(2, "Finished precomputed graph pruning.");
    }

    fn compute_completable_tokens_recursive(
        &self,
        node_arc: &Arc<Mutex<PrecomputeNode>>,
        recursion_stack: &mut HashSet<*const Mutex<PrecomputeNode>>,
        completable_cache: &mut HashMap<*const Mutex<PrecomputeNode>, LLMTokenBV>
    ) -> LLMTokenBV {
        let node_ptr = Arc::as_ptr(node_arc);

        if let Some(cached_completable_tokens) = completable_cache.get(&node_ptr) {
            return cached_completable_tokens.clone();
        }

        if recursion_stack.contains(&node_ptr) {
            return LLMTokenBV::new(); 
        }

        recursion_stack.insert(node_ptr);

        let node_guard = node_arc.lock().expect("Mutex poisoned during compute_completable_tokens_recursive lock");
        
        let mut current_node_completable = node_guard.value.clean_end.as_ref().cloned().unwrap_or_else(LLMTokenBV::new);

        for finalizer in node_guard.value.finalizers.values() {
            for llm_token_bv_in_finalizer in finalizer.content.values() {
                current_node_completable |= llm_token_bv_in_finalizer;
            }
        }

        let children_arcs_to_visit: Vec<Arc<Mutex<PrecomputeNode>>> = node_guard.children().values()
            .flat_map(|destinations_map| destinations_map.keys().map(|arc_ptr_wrapper| arc_ptr_wrapper.as_arc().clone()))
            .collect();
        
        drop(node_guard); 

        for child_arc in children_arcs_to_visit {
            let child_completable_tokens = self.compute_completable_tokens_recursive(&child_arc, recursion_stack, completable_cache);
            current_node_completable |= &child_completable_tokens; 
        }

        recursion_stack.remove(&node_ptr); 
        
        completable_cache.insert(node_ptr, current_node_completable.clone());
        current_node_completable
    }

    fn filter_and_prune_edges_recursive(
        &mut self, 
        node_arc: &Arc<Mutex<PrecomputeNode>>,
        visited_for_pruning: &mut HashSet<*const Mutex<PrecomputeNode>>,
        completable_cache: &HashMap<*const Mutex<PrecomputeNode>, LLMTokenBV>
    ) {
        let node_ptr = Arc::as_ptr(node_arc);
        if !visited_for_pruning.insert(node_ptr) {
            return;
        }

        let children_to_visit_recursively: Vec<Arc<Mutex<PrecomputeNode>>> = {
            let node_guard = node_arc.lock().expect("Mutex poisoned: filter_and_prune_edges_recursive lock (read children)");
            node_guard.children().values()
                .flat_map(|destinations_map| destinations_map.keys().map(|arc_ptr_wrapper| arc_ptr_wrapper.as_arc().clone()))
                .collect()
        }; 

        {
            let mut node_guard = node_arc.lock().expect("Mutex poisoned: filter_and_prune_edges_recursive lock (modify children)");
            
            let original_children_map = std::mem::take(node_guard.children_mut());
            let mut new_children_map_for_current_node = BTreeMap::new();

            for (edge_key, destinations_map) in original_children_map {
                let mut new_destinations_for_this_edge_key = BTreeMap::new();
                for (child_arc_ptr_wrapper, current_edge_value) in destinations_map {
                    let child_arc = child_arc_ptr_wrapper.as_arc();
                    let child_ptr = Arc::as_ptr(child_arc);

                    let completable_tokens_for_child = completable_cache.get(&child_ptr)
                                                      .cloned()
                                                      .unwrap_or_else(LLMTokenBV::new);
                    
                    let mut filtered_edge_value = current_edge_value.clone(); 
                    filtered_edge_value &= &completable_tokens_for_child;

                    if !filtered_edge_value.is_empty() {
                        new_destinations_for_this_edge_key.insert(child_arc_ptr_wrapper.clone(), filtered_edge_value);
                    } else {
                        self.stats.final_edges_pruned_total += 1;
                    }
                }
                if !new_destinations_for_this_edge_key.is_empty() {
                    new_children_map_for_current_node.insert(edge_key.clone(), new_destinations_for_this_edge_key);
                }
            }
            *node_guard.children_mut() = new_children_map_for_current_node;
        } 

        for child_arc_to_recurse_on in children_to_visit_recursively {
            self.filter_and_prune_edges_recursive(&child_arc_to_recurse_on, visited_for_pruning, completable_cache);
        }
    }
    
    fn finish(mut self, token_name_map: &BiBTreeMap<String, usize>) -> Precomputed {
        crate::debug!(2, "Checking for cycles in precomputed graph…");
        for (sid, root) in &self.roots {
            if PrecomputeNode::has_any_cycle(root.clone()) {
                panic!(
                    "Cycle detected in precomputed graph for tokenizer_state_id {:?}",
                    sid
                );
            }
        }

        calculate_final_stats(&self.roots, &mut self.stats);
        print_precompute_stats(&self.stats, token_name_map);

        let mut out = Precomputed::new();
        let mut clones = 0;

        for (sid, arc_val) in self.roots { // Renamed arc to arc_val
            match Arc::try_unwrap(arc_val.clone()) { // Clone arc_val for try_unwrap
                Ok(mutex) => out.insert(
                    sid,
                    mutex.into_inner().expect("Mutex poisoned during unwrap"),
                ),
                Err(_) => { // Original arc_val is used if try_unwrap fails
                    clones += 1;
                    out.insert(sid, arc_val.lock().unwrap().clone()) 
                }
            };
        }

        if clones > 0 {
            crate::debug!(
                4,
                "Warning: {} precomputed root(s) had multiple owners; cloned.",
                clones
            );
        }
        out
    }

    fn dfs(
        &self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<
            TokenizerStateID,
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        >,
    ) {
        self.pb.inc(1);

        let mut effective: BTreeMap<
            TokenizerStateID,
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        > = BTreeMap::new();

        for (sid, set) in assoc_by_state {
            let merged = self.merge_handles(&set);
            if !merged.is_empty() {
                effective.insert(sid, merged);
            }
        }

        for (segment_bytes, child_vocab_arc) in vocab_node.iter_children() {
            let child_vocab_ref = &*child_vocab_arc;
            crate::debug!(
                3,
                "Segment '{}' -> prefix '{}'",
                String::from_utf8_lossy(segment_bytes),
                String::from_utf8_lossy(child_vocab_ref.prefix())
            );

            self.process_segment(segment_bytes, child_vocab_ref, &effective);
        }
    }

    fn process_segment(
        &self,
        segment_bytes: &[u8],
        child_vocab_of_segment: &VocabPrefixTreeNode,
        sources_per_state: &BTreeMap<
            TokenizerStateID,
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        >,
    ) {
        let mut next_level: BTreeMap<
            TokenizerStateID,
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        > = BTreeMap::new();

        let mut queue: BTreeMap<
            usize,
            BTreeMap<TokenizerStateID, BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>>,
        > = BTreeMap::new();

        for (sid, set) in sources_per_state {
            queue
                .entry(0)
                .or_default()
                .entry(*sid)
                .or_default()
                .extend(set.clone());
        }

        while let Some((offset, map_at_offset)) = queue.pop_first() {
            for (state_before, src_set_val) in map_at_offset { // Renamed src_set
                if src_set_val.is_empty() { // Use src_set_val
                    continue;
                }

                let merged_src_set = self.merge_handles(&src_set_val); // Use src_set_val
                if merged_src_set.is_empty() { 
                    continue;
                }

                let suffix      = &segment_bytes[offset..];
                let exec_result = self
                    .tokenizer
                    .execute_from_state(suffix, state_before);

                let possible_future_matches: BTreeMap<GrammarTokenID, LLMTokenBV> = exec_result.end_state.map_or_else(BTreeMap::new, |end_state_id| {
                    self.possible_matches(&child_vocab_of_segment, TokenizerStateID(end_state_id))
                });

                for m in &exec_result.matches {
                    let grammar_tok = GrammarTokenID(m.id);
                    let match_end_offset = offset + m.width;
                    let active_tokens = child_vocab_of_segment.reachable_token_ids().clone();
                    let tokens_with_future_match = possible_future_matches.get(&grammar_tok).cloned().unwrap_or_default();
                    let edge_tokens = active_tokens - tokens_with_future_match;

                    if !edge_tokens.is_empty() {
                        for src in &merged_src_set {
                            self.insert_edge(
                                src.as_arc().clone(),
                                grammar_tok,
                                edge_tokens.clone(),
                                child_vocab_of_segment.token_id(),
                                match_end_offset,
                                segment_bytes.len(),
                                &mut queue,
                                &mut next_level
                            );
                        }
                    }
                }

                if let Some(final_state_val) = exec_result.end_state {
                    let final_sid = TokenizerStateID(final_state_val);
                    for src in &merged_src_set { 
                        next_level
                            .entry(final_sid)
                            .or_default()
                            .insert(src.clone());

                        let mut guard = src.as_arc().lock().unwrap();
                        for gtid in self
                            .tokenizer
                            .tokens_accessible_from_state(final_sid)
                        {
                            crate::debug!(5, "Pushing finalizer info for token {:?} in state {:?}", gtid.0, final_sid.0);
                            guard.value.push_finalizer_info(
                                gtid,
                                LLMTokenID(child_vocab_of_segment.token_id()), 
                                final_sid,
                            );
                        }
                    }
                }
            }
        }

        self.dfs(child_vocab_of_segment, next_level);
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_edge(
        &self,
        source_arc: Arc<Mutex<PrecomputeNode>>,
        grammar_tok: GrammarTokenID,
        edge_tokens: LLMTokenBV,
        final_llm_token_id_at_child_vocab: usize, 
        match_end_offset_in_segment: usize,
        segment_len: usize,
        queue: &mut BTreeMap<
            usize,
            BTreeMap<TokenizerStateID, BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>>,
        >,
        next_level: &mut BTreeMap<
            TokenizerStateID,
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        >,
    ) {
        let mut inserter = EdgeInserter::new(
            source_arc.clone(),
            Some(grammar_tok),
            edge_tokens.clone(), 
            |existing: &mut HybridBitset, new_bv_ref: HybridBitset| *existing |= new_bv_ref,
        );

        inserter = inserter.try_children();

        let mut pot: Vec<Arc<Mutex<PrecomputeNode>>> = Vec::new();

        let gather_set = |set: &BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
                          pot_val: &mut Vec<Arc<Mutex<PrecomputeNode>>>| {
            pot_val.extend(
                set.iter()
                    .map(|h| h.as_arc().clone()),
            );
        };

        if match_end_offset_in_segment < segment_len {
            if let Some(map_at_offset) = queue.get(&match_end_offset_in_segment) {
                if let Some(set_of_nodes_at_offset_for_new_state) = map_at_offset.get(&TokenizerStateID(0)) {
                    gather_set(set_of_nodes_at_offset_for_new_state, &mut pot);
                }
            }
        } else {
            if let Some(set_of_nodes_at_next_level_for_new_state) = next_level.get(&TokenizerStateID(0)) {
                gather_set(set_of_nodes_at_next_level_for_new_state, &mut pot);
            }
        }

        inserter = inserter.try_destinations(&pot);

        if inserter.clone_into_option().is_none() {
            let mut extra = Vec::new();
            {
                let guard = source_arc.lock().unwrap();
                if let Some(dest_map) =
                    guard.children().get(&Some(grammar_tok))
                {
                    for child_wrap in dest_map.keys() {
                         extra.push(child_wrap.as_arc().clone());
                    }
                }
            }
            inserter = inserter.try_destinations(&extra);
        }

        let target = inserter
            .else_create_destination_with_value(PrecomputedNodeContents::default())
            .unwrap();

        // This block was empty, removed.
        // {
        //     let mut guard = target.lock().unwrap();
        // }

        let handle = ArcPtrWrapper::new(target.clone());

        if match_end_offset_in_segment == segment_len {
            crate::debug!(5, "Marking clean end for child vocab node {:p} representing LLM token {:?}", handle.as_ref(), final_llm_token_id_at_child_vocab);
            next_level
                .entry(TokenizerStateID(0)) 
                .or_default()
                .insert(handle);

            let mut g = target.lock().unwrap();
            g.value
                .clean_end
                .get_or_insert_with(HybridBitset::new)
                .insert(final_llm_token_id_at_child_vocab); 
        } else {
            queue
                .entry(match_end_offset_in_segment)
                .or_default()
                .entry(TokenizerStateID(0)) 
                .or_default()
                .insert(handle);
        }
    }

    fn merge_handles(
        &self,
        set: &BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
    ) -> BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>> {
        if set.len() <= self.merge_threshold {
            return set.clone();
        }

        let merged_node_arc = Arc::new(Mutex::new(PrecomputeNode::new( 
            PrecomputedNodeContents::default(),
        )));

        for child_wrapper in set { 
            let edge_tokens_for_merge = self.all_llm_tokens.clone();
            let mut inserter = EdgeInserter::new(
                child_wrapper.as_arc().clone(), 
                None::<GrammarTokenID>,   
                edge_tokens_for_merge.clone(), 
                |existing_edge_data: &mut HybridBitset, new_edge_data: HybridBitset| *existing_edge_data |= new_edge_data,
            );

            inserter = inserter.try_children();

            if inserter.clone_into_option().is_none() {
                inserter = inserter.try_destination(merged_node_arc.clone()); 
            }
        }

        let mut out = BTreeSet::new();
        out.insert(ArcPtrWrapper::new(merged_node_arc)); 
        out
    }
}

fn count_vocab_nodes(node: &VocabPrefixTreeNode) -> u64 {
    1 + node
        .children()
        .values()
        .map(|c| count_vocab_nodes(c))
        .sum::<u64>()
}

#[derive(Debug, Clone)]
pub struct GrammarConstraintState<'a> {
    pub(crate) parent: &'a GrammarConstraint,
    state:  BTreeMap<TokenizerStateID, GLRParserState<'a>>,
}

impl<'a> GrammarConstraintState<'a> {
    pub fn get_mask(&mut self) -> LLMTokenBV {
        let mut internal_mask = HybridBitset::new(); 
        for (_tokenizer_state_id, glr_parser_state) in &self.state {
            internal_mask |= &glr_parser_state.active_state.stack.acc().active; 
        }
        self.parent.internal_bv_to_original(&internal_mask) 
    }

    pub fn step_with_all_llm_tokens(&mut self) {
        let all_internal_llm_tokens = HybridBitset::ones(self.parent.internal_max_llm_token + 1);
        self.step(&all_internal_llm_tokens);
    }

    pub fn step_with_llm_token(&mut self, llm_token_id: LLMTokenID) { 
        if let Some(internal_llm_id) = self.parent.original_id_to_internal(llm_token_id) {
            let mut internal_llm_tokens_bv = HybridBitset::new();
            internal_llm_tokens_bv.insert(internal_llm_id.0 as usize);
            self.step(&internal_llm_tokens_bv); 
        } else {
            let empty_set = HybridBitset::new();
            self.step(&empty_set);
        }
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) {
        let internal_llm_id_val = self.parent.original_id_to_internal(llm_token_id).unwrap().0;
        crate::debug!(4, "Committing token {} (internal ID {})", llm_token_id.0, internal_llm_id_val);

        let full_bitset = HybridBitset::ones(self.parent.internal_max_llm_token + 1);

        let mut singular_bitset = HybridBitset::new();
        singular_bitset.insert(internal_llm_id_val);

        for glr_state in self.state.values_mut() {
            intersect_tokens_and_prune_arc(&mut glr_state.active_state.stack, &singular_bitset);
            reset_tokens(&mut glr_state.active_state.stack, &full_bitset);
        }

        crate::debug!(4, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        crate::debug!(4, "Committed llm_token_id {:?} to grammar constraint state", llm_token_id);
        self.state.retain(|_tokenizer_state_id, glr_state| glr_state.is_ok());
        for (tokenizer_state_id, glr_state) in self.state.iter() {
            glr_state.log_gss(format!("After committing llm_token_id {:?}, from tokenizer_state_id {:?}", llm_token_id, tokenizer_state_id).as_str(), GrammarTokenID(0));
        }
    }

    pub fn step_with_llm_token_sequence(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.step_with_llm_token(llm_token_id);
        }
    }

    fn prepare_initial_nodes_and_values_for_special_map(&mut self, llm_tokens: &LLMTokenBV) -> Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'a>)> {
        let mut initial_nodes_and_values: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'_>)> = Vec::new();
        let mut tokenizer_state_id_to_parse_states: BTreeMap<TokenizerStateID, GLRParserState<'_>> = BTreeMap::new();

        for (tokenizer_state_id, state_val) in &self.state { 
            let mut cloned_state = state_val.clone(); 
            intersect_tokens_and_prune_arc(&mut cloned_state.active_state.stack, llm_tokens);
            if !cloned_state.active_state.stack.is_empty() { // Only keep if not pruned
                tokenizer_state_id_to_parse_states.insert(*tokenizer_state_id, cloned_state);
            }
        }

        crate::debug!(4, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        crate::debug!(4, "Printing initial nodes and values for tokenizer states");
        for tokenizer_state_id_val in tokenizer_state_id_to_parse_states.keys() { 
            let glr_state_after = &tokenizer_state_id_to_parse_states[&tokenizer_state_id_val]; 
            if self.state.contains_key(tokenizer_state_id_val) {
                 let glr_state_before = &self.state[&tokenizer_state_id_val];
                 glr_state_before.log_gss(format!("Existing initial nodes and values for tokenizer state {}", tokenizer_state_id_val.0).as_str(), GrammarTokenID(0)); 
            }
            glr_state_after.log_gss(format!("Prepared (stage 1) initial nodes and values for tokenizer state {}", tokenizer_state_id_val.0).as_str(), GrammarTokenID(0)); 
        }
        crate::debug!(4, "----------------------------------------------------------------");

        for (tokenizer_state_id, state_val) in tokenizer_state_id_to_parse_states { 
            let token_trie_node = self.parent.precomputed[&tokenizer_state_id].clone();
            let token_trie_arc_mutex = Arc::new(Mutex::new(token_trie_node));
            initial_nodes_and_values.push((token_trie_arc_mutex, state_val)); 
        }

        initial_nodes_and_values
    }

    pub fn step(&mut self, llm_tokens: &LLMTokenBV) { 
        crate::debug!(2, "Stepping grammar constraint state with tokenizer states {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens);

        let step_counts = RefCell::new(BTreeMap::<GrammarTokenID, usize>::new());
        self.state = BTreeMap::new();

        Trie::special_map(
            initial_nodes_and_values,
            |glr_parse_state, grammar_token_id_opt, edge_llm_tokens, child_node| { 
                let node_ptr = std::ptr::addr_of!(*child_node);
                crate::debug!(3, "Processing grammar node {:p} token {:?}. Active LLM tokens: {:?}. LLM tokens allowed on edge: {:?}", node_ptr, grammar_token_id_opt.map(|gtid| gtid.0), glr_parse_state.active_state.stack.acc().active, edge_llm_tokens);
                
                let mut cloned_glr_parse_state = glr_parse_state.clone();
                intersect_tokens_and_prune_arc(&mut cloned_glr_parse_state.active_state.stack, edge_llm_tokens);

                if cloned_glr_parse_state.active_state.stack.is_empty() {
                    crate::debug!(3, "GSS became empty after intersecting with edge_llm_tokens for grammar node {:p}, token {:?}", node_ptr, grammar_token_id_opt.map(|gtid| gtid.0));
                    return None; 
                }
                
                crate::debug!(3, "GLR parse state stack: {}", print_gss_forest(&[cloned_glr_parse_state.active_state.stack.clone()], usize::MAX));
                if let Some(gtid) = grammar_token_id_opt { 
                    *step_counts.borrow_mut().entry(*gtid).or_insert(0) += 1;
                }
                grammar_token_id_opt.map(|gtid| {
                    cloned_glr_parse_state.step(gtid);
                }); 
                if cloned_glr_parse_state.active_state.stack.is_empty() {
                    crate::debug!(3, "No active states after processing grammar token {:?}", grammar_token_id_opt.map(|gtid| gtid.0)); 
                    return None;
                } else {
                    crate::debug!(3, "Processed grammar token {:?}. Active LLM tokens: {:?}", grammar_token_id_opt.map(|gtid| gtid.0), cloned_glr_parse_state.active_state.stack.acc().active); 
                    Some(cloned_glr_parse_state)
                }
            },
            |managed_parse_state1, managed_parse_state2| {
                managed_parse_state1.merge_with(managed_parse_state2);
            },
            |node, current_glr_parse_state| {
                // Intersecting with edge tokens and pruning empty active sets is done in the map_fn.
                // Here, we just simplify.
                let mut current_glr_parse_state_mut = current_glr_parse_state.clone(); // To modify
                Arc::make_mut(&mut current_glr_parse_state_mut.active_state.stack).simplify();

                if !current_glr_parse_state_mut.is_ok() {
                    return false;
                }

                let active_llm_tokens = current_glr_parse_state_mut.active_state.stack.acc().active.clone();
                let node_ptr = std::ptr::addr_of!(*node);
                crate::debug!(3, "Processing node {:p}, {} LLM tokens, {} finalizers", node_ptr, active_llm_tokens.len(), node.value.finalizers().len()); 
                
                if let Some(clean_end) = &node.value.clean_end { 
                    let mut final_glr_parse_state = current_glr_parse_state_mut.clone();
                    intersect_tokens_and_prune_arc(&mut final_glr_parse_state.active_state.stack, clean_end);

                    crate::debug!(3, "At clean end state");
                    if final_glr_parse_state.is_ok() && !final_glr_parse_state.active_state.stack.is_empty() {
                        crate::debug!(3, "GLR parse state at clean end is OK");
                        if let Some(existing) = self.state.get_mut(&TokenizerStateID(0)) {
                            crate::debug!(3, "Existing GLR parse state at clean end");
                            existing.merge_with(final_glr_parse_state.clone());
                            crate::debug!(3, "Merged GLR parse state at clean end");
                        } else {
                            self.state.insert(TokenizerStateID(0), final_glr_parse_state.clone());
                            crate::debug!(3, "Inserted GLR parse state at clean end");
                        }
                    }
                }

                for (possible_final_grammar_token, precomputed_finalizer) in node.value.finalizers().iter() { 
                    // let mut possible_next_glr_parse_state = current_glr_parse_state_mut.clone(); // This was used for stepping, but filtering should be on current
                    // crate::debug!(3, "Stepping semi-final GLR parse state");
                    // *step_counts.borrow_mut().entry(*possible_final_grammar_token).or_insert(0) += 1;
                    // possible_next_glr_parse_state.step(*possible_final_grammar_token); // Step happens after filtering

                    // if possible_next_glr_parse_state.is_ok() && !possible_next_glr_parse_state.active_state.stack.is_empty() {
                    //     crate::debug!(3, "Semi-final GLR parse state is OK after step by {:?}", possible_final_grammar_token);
                        for (tokenizer_state_id, llm_tokens_from_finalizer) in &precomputed_finalizer.content { 
                            let mut glr_parse_state_filtered = current_glr_parse_state_mut.clone(); 
                            intersect_tokens_and_prune_arc(&mut glr_parse_state_filtered.active_state.stack, llm_tokens_from_finalizer);
                            
                            crate::debug!(3, "Processing finalizer for token_state_id {:?}", tokenizer_state_id);
                            if glr_parse_state_filtered.is_ok() && !glr_parse_state_filtered.active_state.stack.is_empty() { 
                                crate::debug!(3, "Finalizer is compatible with current GLR state (pre-step by final_grammar_token)");
                                // Now step the filtered state
                                *step_counts.borrow_mut().entry(*possible_final_grammar_token).or_insert(0) += 1;
                                let mut glr_parse_state_to_insert = glr_parse_state_filtered.clone();
                                glr_parse_state_to_insert.step(*possible_final_grammar_token);

                                if glr_parse_state_to_insert.is_ok() && !glr_parse_state_to_insert.active_state.stack.is_empty() {
                                    glr_parse_state_to_insert.log_gss(format!("After filtering by finalizer and stepping by {:?}", possible_final_grammar_token).as_str(), TerminalID(tokenizer_state_id.0));
                                    if let Some(existing) = self.state.get_mut(tokenizer_state_id) {
                                        existing.merge_with(glr_parse_state_to_insert.clone());
                                    } else {
                                        self.state.insert(*tokenizer_state_id, glr_parse_state_to_insert.clone());
                                    }
                                } else {
                                     crate::debug!(3, "GLR state became invalid/empty after stepping with finalizer token {:?}", possible_final_grammar_token);
                                }
                            } else {
                                crate::debug!(3, "GLR state became invalid/empty after filtering with finalizer tokens for state {:?}", tokenizer_state_id);
                            }
                        }
                    // } else {
                    //      crate::debug!(3, "Semi-final GLR parse state became invalid/empty after stepping by {:?}", possible_final_grammar_token);
                    // }
                }
                !current_glr_parse_state_mut.active_state.stack.is_empty()
            },
        );

        let mut roots_to_simplify: Vec<&mut Arc<GSSNode>> = Vec::new();
        for glr_state in self.state.values_mut() {
            if !glr_state.active_state.stack.is_empty() { // Only simplify non-empty stacks
                glr_state.log_gss("Before simplifying GSS forest", TerminalID(0));
                roots_to_simplify.push(&mut glr_state.active_state.stack);
            }
        }

        crate::debug!(2, "Before simplifying GSS forest: {:?}", gather_gss_stats(&roots_to_simplify.iter().map(|arc| arc.as_ref()).collect::<Vec<_>>()));
        GSSNode::simplify_together(&mut roots_to_simplify);
        crate::debug!(2, "After simplifying GSS forest (1st pass): {:?}", gather_gss_stats(&roots_to_simplify.iter().map(|arc| arc.as_ref()).collect::<Vec<_>>()));
        GSSNode::simplify_together(&mut roots_to_simplify); // Potentially simplify again if structure changed significantly
        crate::debug!(2, "After simplifying GSS forest (2nd pass): {:?}", gather_gss_stats(&roots_to_simplify.iter().map(|arc| arc.as_ref()).collect::<Vec<_>>()));
        self.state.retain(|_tokenizer_state_id, glr_state| glr_state.is_ok());
        for glr_state in self.state.values_mut() {
            glr_state.log_gss("After simplifying GSS forest", TerminalID(0));
        }

        let mut sorted_counts: Vec<(GrammarTokenID, usize)> = step_counts.into_inner().into_iter().collect();
        if !sorted_counts.is_empty() {
            println!("--- GLRParserState::step call counts (this GrammarConstraintState::step) ---");
            sorted_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))); 

            for (gtid, count) in sorted_counts {
                let token_name = self
                    .parent
                    .token_name_map
                    .get_by_right(&gtid.0)
                    .map(|s| s.as_str())
                    .unwrap_or("<Unknown Name>");
                println!("  Token {} (ID: {}): {} calls", token_name, gtid.0, count);
            }
            println!("--------------------------------------------------------------------------");
        } else {
            println!("--- GLRParserState::step was not called in this GrammarConstraintState::step ---");
        }
    }

    pub fn is_active(&self) -> bool {
        !self.state.is_empty() && self.state.values().any(|s| !s.active_state.stack.is_empty())
    }

    pub fn state(&self) -> &BTreeMap<TokenizerStateID, GLRParserState<'a>> {
        &self.state
    }
}
