// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::BitOr;
use std::sync::{Arc, Mutex};
use std::cell::RefCell;

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
    // Removed: pub intersection: LLMTokenBV,
    // Removed: pub terminals:    Arc<GSSNode<TerminalID, ()>>,
}

impl Default for LLMTokenInfo {
    fn default() -> Self {
        Self {
            active:       HybridBitset::new(),
            // Removed intersection initialization
            // Removed terminals initialization
        }
    }
}

impl PathAccumulator for LLMTokenInfo {
    fn union_assign(&mut self, other: Self) {
        self.active |= &other.active;
        // self.intersection &= &other.intersection; // Removed
        // Removed terminals merge
    }

    fn intersect_assign(&mut self, right: Self) {
        // The 'pop' or 'intersect' operation for LLMTokenInfo means:
        // - The active set of the resulting path is the intersection of the left path's active set and the right path's active set.
        // - The intersection set also becomes stricter (AND). // Removed
        // - Terminals from the right path are kept (this logic is now removed).
        self.active &= &right.active;
        // self.intersection &= &right.intersection; // Removed
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
            // .field("intersection", &fmt_bv(&self.intersection)) // Removed
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
                    let next_results = self.possible_matches(child_vocab_arc, TokenizerStateID(final_state_val));
                    for (token, bv) in next_results {
                        *result_map.entry(token).or_insert_with(LLMTokenBV::new) |= bv;
                    }
                }
            }
        }

        self.possible_matches.borrow_mut().entry(cache_key_ptr).or_default().insert(tokenizer_state_id, result_map.clone());

        result_map
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
    pub fn get_mask(&self) -> LLMTokenBV {
        crate::debug!(2, "Calculating mask for constraint state with {} entries.", self.state.len());
        let mut overall_mask = HybridBitset::new();
        let step_counts = RefCell::new(BTreeMap::<GrammarTokenID, usize>::new());

        for (tokenizer_id, glr_state) in &self.state {
            if glr_state.active_state.stack.is_empty() {
                crate::debug!(4, "  Skipping tokenizer_id {}: GSS stack is empty.", tokenizer_id.0);
                continue;
            }
            crate::debug!(4, "  get_mask: Processing tokenizer_id {}.", tokenizer_id.0);

            let initial_gss_stack_arc = glr_state.active_state.stack.clone();
            
            let precompute_root_node_arc = match self.parent.precomputed.get(tokenizer_id) {
                Some(node) => Arc::new(Mutex::new(node.clone())), // Clone the PrecomputeNode for the Mutex
                None => {
                    crate::debug!(3, "    No precomputed root for tokenizer_id {}. Skipping.", tokenizer_id.0);
                    continue;
                }
            };

            let initial_nodes_and_values_for_special_map = vec![(
                precompute_root_node_arc,
                initial_gss_stack_arc
            )];

            Trie::special_map(
                initial_nodes_and_values_for_special_map,
                // step_fn (map_fn in special_map's terms)
                |current_gss_arc: &Arc<GSSNode>, // This is V from previous step/initial
                 grammar_token_opt: &Option<GrammarTokenID>, // This is EK (Edge Key)
                 edge_llm_tokens: &LLMTokenBV, // This is EV (Edge Value)
                 _child_precompute_node_contents: &PrecomputedNodeContents| // This is Trie::value of child Trie node
                 -> Option<Arc<GSSNode>> { // Returns new V for child
                    crate::debug!(5, "    step_fn: grammar_token {:?}, edge_llm_tokens non-empty: {}", grammar_token_opt.map(|g|g.0), !edge_llm_tokens.is_empty());
                    let mut next_gss_arc = current_gss_arc.clone();
                    
                    let mut current_active_set = Arc::make_mut(&mut next_gss_arc).acc_mut().active.clone();
                    current_active_set &= edge_llm_tokens;
                    
                    if current_active_set.is_empty() {
                        crate::debug!(5, "      step_fn: active set empty after edge_llm_tokens intersection. Pruning path.");
                        return None;
                    }
                    Arc::make_mut(&mut next_gss_arc).acc_mut().active = current_active_set;

                    if let Some(gtid) = grammar_token_opt {
                        crate::debug!(5, "      step_fn: stepping with grammar_token_id {}", gtid.0);
                        let mut temp_glr_state = self.parent.parser.init_glr_parser_from_parse_state(ParseState { stack: next_gss_arc });
                        temp_glr_state.step(*gtid);
                        next_gss_arc = temp_glr_state.active_state.stack;
                        
                        *step_counts.borrow_mut().entry(*gtid).or_insert(0) += 1;
                    }

                    if next_gss_arc.acc().active.is_empty() {
                        crate::debug!(5, "      step_fn: active set empty after GLR step. Pruning path.");
                        None
                    } else {
                        crate::debug!(5, "      step_fn: path valid. Resulting active set non-empty: {}", !next_gss_arc.acc().active.is_empty());
                        Some(next_gss_arc)
                    }
                },
                // merge_fn
                |gss_arc1: &mut Arc<GSSNode>, gss_arc2: Arc<GSSNode>| {
                    Arc::make_mut(gss_arc1).merge(&gss_arc2);
                },
                // process_fn
                |precompute_node_val: &PrecomputedNodeContents, final_gss_arc: &mut Arc<GSSNode>| -> bool {
                    Arc::make_mut(final_gss_arc).simplify();
                    if final_gss_arc.acc().active.is_empty() {
                        crate::debug!(5, "    process_fn: final_gss_arc active set is empty. Stopping propagation.");
                        return false; 
                    }
                    
                    let mut mask_contribution_from_this_path = LLMTokenBV::new();

                    if let Some(clean_tokens) = &precompute_node_val.clean_end {
                        crate::debug!(5, "    process_fn: found clean_end. Intersecting with active set.");
                        mask_contribution_from_this_path |= &(final_gss_arc.acc().active & clean_tokens);
                    }

                    for (gtid, finalizer) in precompute_node_val.finalizers() {
                        for (_finalizer_tokenizer_id, llm_tokens_from_finalizer) in &finalizer.content {
                            let mut temp_gss_arc_for_finalizer = final_gss_arc.clone();
                            
                            let mut temp_active_set_finalizer = Arc::make_mut(&mut temp_gss_arc_for_finalizer).acc_mut().active.clone();
                            temp_active_set_finalizer &= llm_tokens_from_finalizer;

                            if temp_active_set_finalizer.is_empty() {
                                continue;
                            }
                            Arc::make_mut(&mut temp_gss_arc_for_finalizer).acc_mut().active = temp_active_set_finalizer;
                            
                            let mut temp_glr_state_finalizer = self.parent.parser.init_glr_parser_from_parse_state(ParseState { stack: temp_gss_arc_for_finalizer });
                            temp_glr_state_finalizer.step(*gtid);
                            
                            *step_counts.borrow_mut().entry(*gtid).or_insert(0) += 1;

                            if temp_glr_state_finalizer.is_ok() && !temp_glr_state_finalizer.active_state.stack.is_empty() {
                                mask_contribution_from_this_path |= &temp_glr_state_finalizer.active_state.stack.acc().active;
                            }
                        }
                    }
                    
                    if !mask_contribution_from_this_path.is_empty() {
                        crate::debug!(5, "    process_fn: contributing to overall_mask. Contribution non-empty: {}", !mask_contribution_from_this_path.is_empty());
                        overall_mask |= &mask_contribution_from_this_path;
                    }
                    true 
                },
            );
        }
        
        let counts = step_counts.into_inner();
        if !counts.is_empty() {
            let mut sorted_counts: Vec<(GrammarTokenID, usize)> = counts.into_iter().collect();
            sorted_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            crate::debug!(3, "--- GLRParserState::step call counts (during get_mask) ---");
            for (gtid, count) in sorted_counts {
                let token_name = self.parent.token_name_map.get_by_right(&gtid.0)
                    .map(|s| s.as_str()).unwrap_or("<Unknown Name>");
                crate::debug!(3, "  Token {} (ID: {}): {} calls", token_name, gtid.0, count);
            }
            crate::debug!(3, "-------------------------------------------------------");
        }

        crate::debug!(2, "Calculated mask. Internal mask non-empty: {}. Converting to original IDs.", !overall_mask.is_empty());
        self.parent.internal_bv_to_original(&overall_mask)
    }

    pub fn commit(&mut self, llm_token_id_original: LLMTokenID) {
        let internal_llm_id = match self.parent.original_id_to_internal(llm_token_id_original) {
            Some(id) => id,
            None => {
                crate::debug!(2, "Committed LLMTokenID {} not found in internal mapping. Clearing constraint state.", llm_token_id_original.0);
                self.state.clear();
                return;
            }
        };

        let llm_token_bytes = match self.parent.llm_token_map.get_by_right(&llm_token_id_original) {
            Some(bytes) => bytes.clone(), // Clone here as it's used in a loop
            None => {
                crate::debug!(0, "Error: LLMTokenID {} (internal {}) has no byte mapping. Clearing constraint state.", llm_token_id_original.0, internal_llm_id.0);
                self.state.clear();
                return;
            }
        };

        crate::debug!(2, "Committing LLMTokenID {} (internal {}), bytes: {:?}", llm_token_id_original.0, internal_llm_id.0, String::from_utf8_lossy(&llm_token_bytes));

        let internal_llm_id_bitset = {
            let mut bv = LLMTokenBV::new();
            bv.insert(internal_llm_id.0);
            bv
        };

        let mut pending_glr_states: std::collections::VecDeque<(usize, TokenizerStateID, GLRParserState<'a>)> = std::collections::VecDeque::new();
        let step_counts = RefCell::new(BTreeMap::<GrammarTokenID, usize>::new()); // For logging GLR steps

        for (tokenizer_id, glr_state) in std::mem::take(&mut self.state) {
            let mut initial_glr_state_for_commit = glr_state.clone();
            // Filter current GSS states by the LLM token being committed.
            intersect_tokens_and_prune_arc(&mut initial_glr_state_for_commit.active_state.stack, &internal_llm_id_bitset);
            
            if initial_glr_state_for_commit.is_ok() && !initial_glr_state_for_commit.active_state.stack.is_empty() {
                crate::debug!(4, "Queueing initial GLR state for tokenizer_id {} after filtering by committed token.", tokenizer_id.0);
                pending_glr_states.push_back((0, *tokenizer_id, initial_glr_state_for_commit));
            } else {
                crate::debug!(4, "GLR state for tokenizer_id {} became invalid/empty after filtering by committed token. Discarding.", tokenizer_id.0);
            }
        }

        let mut next_constraint_states_map: BTreeMap<TokenizerStateID, GLRParserState<'a>> = BTreeMap::new();

        while let Some((offset, current_tokenizer_id, current_glr_state)) = pending_glr_states.pop_front() {
            crate::debug!(4, "Processing commit: offset {}, tokenizer_id {}", offset, current_tokenizer_id.0);
            let exec_result = self.parent.tokenizer.execute_from_state(&llm_token_bytes[offset..], current_tokenizer_id);

            if let Some(end_tokenizer_id_val) = exec_result.end_state {
                // This means the tokenizer consumed the rest of llm_token_bytes[offset..]
                // without emitting a grammar token, ending in end_tokenizer_id_val.
                // The current_glr_state is carried over to this new tokenizer state.
                crate::debug!(4, "  Tokenizer consumed rest of bytes, ended in state {}. Merging GLR state.", end_tokenizer_id_val);
                let target_tokenizer_id = TokenizerStateID(end_tokenizer_id_val);
                if let Some(existing_state) = next_constraint_states_map.get_mut(&target_tokenizer_id) {
                    existing_state.merge_with(current_glr_state.clone());
                } else {
                    next_constraint_states_map.insert(target_tokenizer_id, current_glr_state.clone());
                }
            }

            for match_info in exec_result.matches {
                let grammar_token_id = TerminalID(match_info.id);
                let match_width = match_info.width;
                let next_offset = offset + match_width;
                
                crate::debug!(4, "  Matched grammar_token_id {} (width {}), next_offset {}.", grammar_token_id.0, match_width, next_offset);

                let mut next_glr_state_after_match = current_glr_state.clone();
                next_glr_state_after_match.step(grammar_token_id);
                *step_counts.borrow_mut().entry(grammar_token_id).or_insert(0) += 1;


                if next_glr_state_after_match.is_ok() && !next_glr_state_after_match.active_state.stack.is_empty() {
                    // After a grammar token is matched, the tokenizer resets to its initial state for the next segment.
                    let final_tokenizer_state_for_match = self.parent.tokenizer.initial_state_id();
                    if next_offset == llm_token_bytes.len() {
                        // End of the committed LLM token's bytes.
                        crate::debug!(4, "    End of LLM token bytes. Merging GLR state for tokenizer_id {}.", final_tokenizer_state_for_match.0);
                        if let Some(existing_state) = next_constraint_states_map.get_mut(&final_tokenizer_state_for_match) {
                            existing_state.merge_with(next_glr_state_after_match.clone());
                        } else {
                            next_constraint_states_map.insert(final_tokenizer_state_for_match, next_glr_state_after_match.clone());
                        }
                    } else {
                        // More bytes from the committed LLM token to process.
                        crate::debug!(4, "    Queueing further processing: next_offset {}, tokenizer_id {}.", next_offset, final_tokenizer_state_for_match.0);
                        // Need to handle merging if multiple paths lead to the same (next_offset, final_tokenizer_state_for_match)
                        // For now, VecDeque just pushes. If merging here is critical, queue type might need to change or merge before push.
                        pending_glr_states.push_back((next_offset, final_tokenizer_state_for_match, next_glr_state_after_match));
                    }
                } else {
                     crate::debug!(4, "    GLR state became invalid/empty after stepping with grammar_token_id {}. Discarding path.", grammar_token_id.0);
                }
            }
        }
        
        self.state = next_constraint_states_map;
        self.state.retain(|_tid, s| s.is_ok() && !s.active_state.stack.is_empty());

        crate::debug!(3, "Commit processed. Resulting self.state has {} entries before final reset/simplify.", self.state.len());

        // Reset active sets in GSS nodes to all possible tokens, as we are now ready for the *next* LLM token.
        // The get_mask() function will then determine the actual allowed subset based on this.
        let all_internal_tokens = HybridBitset::ones(self.parent.internal_max_llm_token + 1);
        for (final_tokenizer_id, final_glr_state) in self.state.iter_mut() {
            crate::debug!(4, "Resetting tokens and simplifying for final tokenizer_id {}.", final_tokenizer_id.0);
            reset_tokens(&mut final_glr_state.active_state.stack, &all_internal_tokens);
            Arc::make_mut(&mut final_glr_state.active_state.stack).simplify();
            if final_glr_state.active_state.stack.is_empty() {
                 crate::debug!(4, "  Stack became empty after reset/simplify for tokenizer_id {}.", final_tokenizer_id.0);
            }
        }
        self.state.retain(|_tid, s| !s.active_state.stack.is_empty()); // Clean up states that became empty after reset/simplify

        // Logging step_counts for commit
        let counts = step_counts.into_inner();
        if !counts.is_empty() {
            let mut sorted_counts: Vec<(GrammarTokenID, usize)> = counts.into_iter().collect();
            sorted_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            crate::debug!(3, "--- GLRParserState::step call counts (during commit of LLM Token {}) ---", llm_token_id_original.0);
            for (gtid, count) in sorted_counts {
                let token_name = self.parent.token_name_map.get_by_right(&gtid.0)
                    .map(|s| s.as_str()).unwrap_or("<Unknown Name>");
                crate::debug!(3, "  Token {} (ID: {}): {} calls", token_name, gtid.0, count);
            }
            crate::debug!(3, "----------------------------------------------------------------------");
        }

        crate::debug!(2, "Commit finished. Final self.state has {} entries.", self.state.len());
    }

    pub fn is_active(&self) -> bool {
        !self.state.is_empty() && self.state.values().any(|s| !s.active_state.stack.is_empty())
    }

    pub fn state(&self) -> &BTreeMap<TokenizerStateID, GLRParserState<'a>> {
        &self.state
    }
}
