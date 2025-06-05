// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use ordered_hash_map::OrderedHashMap;
use ordered_hash_map::OrderedHashSet;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::ops::BitOr;
use std::sync::{Arc, Mutex};
use std::cell::RefCell;

use bimap::BiBTreeMap;
use bitvec::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

use crate::constraint_extra::{calculate_final_stats, print_precompute_stats, PrecomputeStats};
use crate::datastructures::charmap::TrieMap;
use crate::datastructures::gss::{print_gss_forest, GSSNode, PathAccumulator, intersect_llm_tokens_and_prune_arc, gather_gss_stats, reset_llm_tokens};
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
use crate::datastructures::gss::acc_mod::Acc;
use crate::glr::analyze::{compute_nullable_nonterminals, compute_terminal_follow_sets};

pub type LLMTokenBV = HybridBitset;
pub type TerminalBV = HybridBitset;


const MERGE_THRESHOLD: usize = 10;

// -----------------------------------------------------------------------------
// Pre-computation node values
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedFinalizer {
    pub content: LLMTokenBV,
}

impl Default for PrecomputedFinalizer {
    fn default() -> Self {
        Self { content: LLMTokenBV::zeros() }
    }
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
                                   .and_then(LLMTokenBV::from_json)?;
                Ok(PrecomputedFinalizer { content })
            }
            _ => Err("Expected JSONNode::Object for PrecomputedFinalizer".to_string()),
        }
    }
}


impl PrecomputedFinalizer {
    fn new(tokens: LLMTokenBV) -> Self {
        Self {
            content: tokens,
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
        _tokenizer_state: TokenizerStateID,
    ) {
        self.finalizers
            .entry(grammar_token)
            .or_default()
            .content
            .insert(llm_token.0);
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
    pub(crate) possible_matches: BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
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
        assert_eq!(self.possible_matches, other.possible_matches);
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
        obj.insert("possible_matches".to_string(), self.possible_matches.to_json());
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
                let possible_matches = obj.remove("possible_matches").ok_or_else(|| "Missing field possible_matches".to_string())
                                          .and_then(|n| BTreeMap::<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>::from_json(n))?;
                Ok(GrammarConstraint {
                    tokenizer,
                    parser,
                    precomputed,
                    llm_token_map,
                    token_name_map,
                    max_original_llm_token_id,
                    original_to_internal_id_bimap,
                    internal_max_llm_token,
                    possible_matches,
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
        let epsilon_terminal_group_ids: BTreeSet<_> = tokenizer.execute_from_state(&[], tokenizer.initial_state_id()).matches.iter().map(|token| token.id).collect();
        let epsilon_terminals: BTreeSet<&String> = epsilon_terminal_group_ids.iter().map(|id| token_name_map.get_by_right(id).unwrap()).collect();
        assert!(epsilon_terminals.is_empty(), "Epsilon tokens (tokens that can match an empty string) are not supported by the grammar constraint. Got: {:?}", epsilon_terminals);
        let original_to_internal_id_bimap = Self::setup_llm_token_mappings(&llm_token_map);

        let internal_max_llm_token = original_to_internal_id_bimap.iter().map(|(_, id)| *id).max().unwrap_or(0);

        let mut internal_llm_token_map_for_precompute = BiBTreeMap::new();
        for (bytes, original_id) in llm_token_map.iter() {
            if let Some(internal_id_val) = original_to_internal_id_bimap.get_by_left(&original_id.0) {
                internal_llm_token_map_for_precompute.insert(bytes.clone(), LLMTokenID(*internal_id_val));
            }
        }

        // Build VocabPrefixTree for internal LLM tokens (needed for possible_matches computation)
        let internal_tokens_for_vocab: Vec<(usize, Vec<u8>)> = internal_llm_token_map_for_precompute
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone()))
            .collect();
        // Note: The tokenizer parameter to `new` is shadowed here by the struct field.
        // We need to use the parameter `tokenizer` for the computation.
        // Let's rename the parameter to avoid confusion, or be careful.
        // Assuming `tokenizer` in `Self { tokenizer, ... }` refers to the parameter, it's fine.
        
        crate::debug!(2, "Building vocab prefix tree for possible_matches computation");
        let vocab_for_possible_matches = VocabPrefixTree::build(&internal_tokens_for_vocab);
        crate::debug!(2, "Done building vocab prefix tree for possible_matches computation");

        let mut computed_possible_matches = BTreeMap::new();
        // Cache for the possible_matches computation
        let mut pm_cache: HashMap<(*const VocabPrefixTreeNode, TokenizerStateID), BTreeMap<GrammarTokenID, LLMTokenBV>> = HashMap::new();

        crate::debug!(2, "Computing possible_matches for all {} tokenizer states", tokenizer.iter_states().count());
        for sid in tokenizer.iter_states() { // Use the `tokenizer` parameter passed to `new`
            let matches_for_sid = Self::compute_possible_matches_for_vocab_node(
                &tokenizer, // Pass the tokenizer parameter from `new`
                &vocab_for_possible_matches.root,
                sid,
                &mut pm_cache,
            );
            computed_possible_matches.insert(sid, matches_for_sid);
        }
        crate::debug!(2, "Finished computing possible_matches");
        // pm_cache is dropped here as it's no longer needed.

        let grammar_productions = &parser.productions; // Assuming parser is the GLRParser instance
        let grammar_term_map = &parser.terminal_map;

        // These might be computed elsewhere or need to be computed here.
        // Assuming compute_first_sets is available from grammar module.
        let first_sets = crate::glr::grammar::compute_first_sets(grammar_productions);
        let nullable_nonterminals = compute_nullable_nonterminals(grammar_productions);

        let terminal_follow_sets_named = compute_terminal_follow_sets(
            grammar_productions,
            &nullable_nonterminals,
            &first_sets,
        );

        let mut terminal_follow_map_ids: BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>> = BTreeMap::new();
        for (terminal1, following_terminals) in terminal_follow_sets_named {
            if let Some(t1_id) = grammar_term_map.get_by_left(&terminal1) {
                let mut following_ids = BTreeSet::new();
                for t2 in following_terminals {
                    if let Some(t2_id) = grammar_term_map.get_by_left(&t2) {
                        following_ids.insert(*t2_id);
                    }
                }
                if !following_ids.is_empty() {
                    terminal_follow_map_ids.insert(*t1_id, following_ids);
                }
            }
        }
        crate::debug!(2, "Computed terminal_follow_map_ids with {} entries.", terminal_follow_map_ids.len());

        let precomputed = Self::precompute(
            &tokenizer, // This is the tokenizer parameter being moved into the struct
            &internal_llm_token_map_for_precompute, 
            &token_name_map,
            internal_max_llm_token, 
            &terminal_follow_map_ids, // Pass the new map
        );

        Self {
            tokenizer, // This is the tokenizer parameter being moved into the struct
            parser,
            precomputed,
            llm_token_map, 
            token_name_map,
            max_original_llm_token_id,
            original_to_internal_id_bimap,
            internal_max_llm_token,
            possible_matches: computed_possible_matches, // Add this line
        }
    }

    pub fn precompute(
        tokenizer:        &Regex,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>, 
        token_name_map:   &BiBTreeMap<String, usize>,
        internal_max_llm_token: usize,                       
        terminal_follow_map_ids: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    ) -> Precomputed {
        let mut helper = Precomputer::new(
            tokenizer,
            internal_llm_token_map,    
            internal_max_llm_token, 
            MERGE_THRESHOLD,
            terminal_follow_map_ids, // Pass to Precomputer::new
        );

        helper.run_dfs();
        helper.prune_precomputed_graph();
        helper.prune_terminal_sequences(); // New pruning pass << ADD THIS LINE HERE
        helper.merge_nodes();
        helper.finish(token_name_map)
    }

    pub fn init(&self) -> GrammarConstraintState<'_> {
        let base_set_for_info = HybridBitset::ones(self.internal_max_llm_token + 1);
        let initial_llm_token_acc: Acc = Acc::default();
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
        let mut internal_bv = HybridBitset::zeros();
        for original_id_val in original_bv.iter() {
            let internal_id_val = self.original_to_internal_id_bimap.get_by_left(&(original_id_val as usize)).expect(format!("Original ID {} not found in original_to_internal_id_bimap", original_id_val).as_str());
            internal_bv.insert(*internal_id_val as usize);
        }
        internal_bv
    }

    fn internal_bv_to_original(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        let mut original_bv = HybridBitset::zeros();
        for internal_id_val in internal_bv.iter() {
            let original_id_val = self.original_to_internal_id_bimap.get_by_right(&(internal_id_val as usize)).expect(format!("Internal ID {} not found in original_to_internal_id_bimap", internal_id_val).as_str());
            original_bv.insert(*original_id_val as usize);
        }
        original_bv
    }

    pub(crate) fn all_internal_llm_tokens_bitset(&self) -> LLMTokenBV {
        HybridBitset::ones(self.internal_max_llm_token + 1)
    }

    fn compute_possible_matches_for_vocab_node(
        tokenizer: &Regex,
        vocab_node: &VocabPrefixTreeNode,
        tokenizer_state_id: TokenizerStateID,
        cache: &mut HashMap<(*const VocabPrefixTreeNode, TokenizerStateID), BTreeMap<GrammarTokenID, LLMTokenBV>>,
    ) -> BTreeMap<GrammarTokenID, LLMTokenBV> {
        let cache_key = (vocab_node as *const VocabPrefixTreeNode, tokenizer_state_id);
        if let Some(cached_result) = cache.get(&cache_key) {
            return cached_result.clone();
        }

        let mut result_map: BTreeMap<GrammarTokenID, LLMTokenBV> = BTreeMap::new();

        for (segment_bytes, child_vocab_arc) in vocab_node.iter_children() {
            let child_vocab_node_ref = child_vocab_arc; // Get &VocabPrefixTreeNode
            let exec_result = tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);

            for token_match in &exec_result.matches {
                let grammar_token_id = GrammarTokenID(token_match.id);
                // LLM tokens reachable under child_vocab_node_ref are those that start with segment_bytes
                let applicable_tokens = child_vocab_node_ref.reachable_token_ids();
                *result_map.entry(grammar_token_id).or_insert_with(LLMTokenBV::zeros) |= applicable_tokens;
            }

            if let Some(final_state_val) = exec_result.end_state {
                let final_tokenizer_state_id = TokenizerStateID(final_state_val);
                
                let matches_possible_from_new_tokenizer_state: BTreeSet<_> = tokenizer
                    .tokens_accessible_from_state(final_tokenizer_state_id)
                    .into_iter()
                    .collect();

                let matches_from_current_segment: BTreeSet<_> = exec_result
                    .matches
                    .iter()
                    .map(|m| GrammarTokenID(m.id))
                    .collect();
                
                let new_grammar_tokens_to_look_for = &matches_possible_from_new_tokenizer_state - &matches_from_current_segment;

                if !new_grammar_tokens_to_look_for.is_empty() {
                    let next_results = Self::compute_possible_matches_for_vocab_node(
                        tokenizer,
                        child_vocab_node_ref, // Recurse with the child node
                        final_tokenizer_state_id,
                        cache,
                    );
                    for (token, bv) in next_results {
                        *result_map.entry(token).or_insert_with(LLMTokenBV::zeros) |= bv;
                    }
                }
            }
        }
        cache.insert(cache_key, result_map.clone());
        result_map
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
    terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>, // New field
}

impl<'r> Precomputer<'r> {
    fn new(
        tokenizer:        &'r Regex,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>, 
        internal_max_llm_token: usize,                       
        merge_threshold:  usize,
        terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>, // New parameter
    ) -> Self {
        let tokens: Vec<(usize, Vec<u8>)> = internal_llm_token_map 
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone())) 
            .collect();

        crate::debug!(2, "Building vocab prefix tree");
        let vocab = VocabPrefixTree::build(&tokens);
        crate::debug!(2, "Done building vocab prefix tree");

        let mut roots = BTreeMap::new();
        for sid in tokenizer.iter_states() {
            roots.insert(
                sid,
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
            terminal_follow_map, // Store the map
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
                *result_map.entry(grammar_token_id).or_insert_with(LLMTokenBV::zeros) |= applicable_tokens;
            }
            if let Some(final_state_val) = exec_result.end_state {
                let matches_possible_from_tokenizer_state: BTreeSet<_> = self.tokenizer.tokens_accessible_from_state(TokenizerStateID(final_state_val)).into_iter().collect();
                let matches_here: BTreeSet<_> = exec_result.matches.iter().map(|m| GrammarTokenID(m.id)).collect();
                let possible_new_matches = &matches_possible_from_tokenizer_state - &matches_here;
                if !possible_new_matches.is_empty() {
                    let next_results = self.possible_matches(child_vocab_arc, TokenizerStateID(final_state_val));
                    for (token, bv) in next_results {
                        *result_map.entry(token).or_insert_with(LLMTokenBV::zeros) |= bv;
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
            OrderedHashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
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
        crate::debug!(2, "Merging nodes: first collecting unique roots and their canonical Arcs");
        let mut content_to_canonical_arc_map: HashMap<PrecomputeNode, Arc<Mutex<PrecomputeNode>>> = HashMap::new();
        
        for (_tokenizer_state_id, root_arc_ref) in &self.roots {
            let node_content = root_arc_ref.lock().unwrap().clone();
            // This will associate node_content with root_arc_ref.clone().
            // If node_content was already in the map, its associated Arc gets updated to root_arc_ref.clone().
            // This implements a "last one wins" policy for which Arc becomes canonical for a given content.
            content_to_canonical_arc_map.insert(node_content, root_arc_ref.clone());
        }

        crate::debug!(2, "Merging nodes: second pass, rewriting roots in self.roots to point to canonical Arcs");
        for (_tokenizer_state_id, root_arc_in_self_roots_mut) in &mut self.roots {
            let current_content = root_arc_in_self_roots_mut.lock().unwrap().clone();
            if let Some(canonical_arc) = content_to_canonical_arc_map.get(&current_content) {
                *root_arc_in_self_roots_mut = canonical_arc.clone();
            } else {
                // This should not happen if content_to_canonical_arc_map was built correctly from all roots
                // and PrecomputeNode's Ord/Eq implementations are consistent.
                panic!(
                    "Error in merge_nodes: content of a root from self.roots (tokenizer_state_id: {:?}) \
                    was not found in the canonical map. This indicates a potential issue with \
                    PrecomputeNode's Ord/Eq implementation or the merge_nodes logic itself.",
                    _tokenizer_state_id);
            };
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
            return LLMTokenBV::zeros();
        }

        recursion_stack.insert(node_ptr);

        let node_guard = node_arc.lock().expect("Mutex poisoned during compute_completable_tokens_recursive lock");
        
        let mut current_node_completable = node_guard.value.clean_end.as_ref().cloned().unwrap_or_else(LLMTokenBV::zeros);

        for finalizer in node_guard.value.finalizers.values() {
            current_node_completable |= &finalizer.content;
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
            let mut new_children_map_for_current_node: BTreeMap<_, OrderedHashMap<_, _>> = BTreeMap::new();

            for (edge_key, destinations_map) in original_children_map {
                let mut new_destinations_for_this_edge_key: OrderedHashMap<_, _> = OrderedHashMap::new();
                for (child_arc_ptr_wrapper, current_edge_value) in destinations_map {
                    let child_arc = child_arc_ptr_wrapper.as_arc();
                    let child_ptr = Arc::as_ptr(child_arc);

                    let completable_tokens_for_child = completable_cache.get(&child_ptr)
                                                      .cloned()
                                                      .unwrap_or_else(LLMTokenBV::zeros);
                    
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

    fn prune_terminal_sequences(&mut self) {
        crate::debug!(2, "Starting terminal sequence pruning.");
        let mut visited_contexts = HashSet::new(); // To avoid redundant work on same node from same context
        for root_arc in self.roots.clone().values() {
            // For roots, there's no "previous terminal", so pass None.
            self.prune_terminal_sequences_recursive(&root_arc, None, &mut visited_contexts);
        }
        crate::debug!(2, "Finished terminal sequence pruning. Edges pruned: {}", self.stats.edges_pruned_by_terminal_sequence);
    }

    fn prune_terminal_sequences_recursive(
        &mut self,
        node_arc: &Arc<Mutex<PrecomputeNode>>,
        prev_edge_terminal_opt: Option<GrammarTokenID>,
        visited_contexts: &mut HashSet<(*const Mutex<PrecomputeNode>, Option<GrammarTokenID>)>,
    ) {
        let node_ptr = Arc::as_ptr(node_arc);
        if !visited_contexts.insert((node_ptr, prev_edge_terminal_opt)) {
            return; // Already processed this node from this incoming terminal context
        }

        let mut children_to_recurse: Vec<(Arc<Mutex<PrecomputeNode>>, Option<GrammarTokenID>)> = Vec::new();
        let mut edges_before_pruning = 0;
        let mut edges_after_pruning = 0;

        { // Scoped lock for node_arc
            let mut node_guard = node_arc.lock().expect("Mutex poisoned: prune_terminal_sequences_recursive");

            let allowed_next_terminals: Option<&BTreeSet<GrammarTokenID>> =
                prev_edge_terminal_opt.and_then(|prev_term| self.terminal_follow_map.get(&prev_term));

            // Collect children before modifying map to avoid issues with iterator invalidation if retaining in place
            let original_children_keys: Vec<Option<GrammarTokenID>> = node_guard.children().keys().cloned().collect();
            
            for key_k_opt_t2 in original_children_keys { // key_k_opt_t2 is the Option<GrammarTokenID> for the outgoing edge
                if let Some(destinations_map) = node_guard.children().get(&key_k_opt_t2) {
                     edges_before_pruning += destinations_map.len();
                }

                let mut keep_edge = true;
                if let Some(allowed_set) = allowed_next_terminals {
                    // Pruning applies if prev_edge_terminal_opt was Some, and thus allowed_set is Some.
                    if let Some(t2_terminal) = key_k_opt_t2 {
                        if !allowed_set.contains(&t2_terminal) {
                            keep_edge = false;
                        }
                    }
                    // If key_k_opt_t2 is None, it's not a terminal-terminal sequence, so keep_edge remains true.
                }
                // If allowed_next_terminals is None (e.g. prev_edge_terminal_opt was None, or prev_term had no followers),
                // then keep_edge remains true (no restrictions from previous terminal).

                if keep_edge {
                    if let Some(destinations_map) = node_guard.children().get(&key_k_opt_t2) {
                        edges_after_pruning += destinations_map.len();
                        for child_arc_wrapper in destinations_map.keys() {
                            children_to_recurse.push((child_arc_wrapper.as_arc().clone(), key_k_opt_t2.clone()));
                        }
                    }
                } else {
                    // Edge is pruned
                    node_guard.children_mut().remove(&key_k_opt_t2);
                    // self.stats.edges_pruned_by_terminal_sequence is incremented outside based on total counts
                }
            }
        } // node_guard lock released

        self.stats.edges_pruned_by_terminal_sequence += edges_before_pruning.saturating_sub(edges_after_pruning);

        for (child_arc, current_edge_terminal_opt) in children_to_recurse {
            self.prune_terminal_sequences_recursive(&child_arc, current_edge_terminal_opt, visited_contexts);
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
            OrderedHashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        >,
    ) {
        self.pb.inc(1);

        let mut effective: BTreeMap<
            TokenizerStateID,
            OrderedHashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
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
            OrderedHashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        >,
    ) {
        let mut next_level: BTreeMap<
            TokenizerStateID,
            OrderedHashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        > = BTreeMap::new();

        let mut queue: BTreeMap<
            usize,
            BTreeMap<TokenizerStateID, OrderedHashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>>,
        > = BTreeMap::from([(0, sources_per_state.clone())]);

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
                    let active_tokens = child_vocab_of_segment.reachable_token_ids();
                    let tokens_with_future_match = possible_future_matches.get(&grammar_tok).cloned().unwrap_or(LLMTokenBV::zeros());
                    let edge_tokens = active_tokens.clone() - tokens_with_future_match;

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
            BTreeMap<TokenizerStateID, OrderedHashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>>,
        >,
        next_level: &mut BTreeMap<
            TokenizerStateID,
            OrderedHashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
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

        let gather_set = |set: &OrderedHashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
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
                .get_or_insert_with(HybridBitset::zeros)
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
        set: &OrderedHashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
    ) -> OrderedHashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>> {
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

        let mut out = OrderedHashSet::new();
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
        crate::debug!(2, "Computing mask");
        let mut final_mask_internal = HybridBitset::zeros();

        if self.state.is_empty() {
            return self.parent.internal_bv_to_original(&final_mask_internal);
        }

        let mut initial_values_for_map: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'a>)> = Vec::new();
        for (tokenizer_state_id, glr_state) in &self.state {
            // Ensure the GLR state's GSS stack is not empty before proceeding
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            if let Some(precomputed_trie_root_data) = self.parent.precomputed.get(tokenizer_state_id) {
                let precomputed_trie_arc = Arc::new(Mutex::new(precomputed_trie_root_data.clone()));
                initial_values_for_map.push((precomputed_trie_arc, glr_state.clone()));
            } else {
                crate::debug!(1, "Warning: No precomputed trie found for tokenizer state {:?}. This state will not contribute to the mask.", tokenizer_state_id);
            }
        }

        if initial_values_for_map.is_empty() {
             // This can happen if all GLR states had empty GSS stacks or no corresponding precomputed tries.
             crate::debug!(2, "No valid initial states for get_mask's special_map traversal.");
             return self.parent.internal_bv_to_original(&final_mask_internal);
        }

        Trie::special_map(
            initial_values_for_map,
            // step_fn: (current_glr_state, edge_grammar_token_opt, edge_llm_tokens_bv, child_precomputed_node_data)
            |glr_s, grammar_token_opt, edge_llm_tokens_bv, _child_node_trie_data| {
                let mut glr_s = glr_s.clone();
                
                intersect_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &edge_llm_tokens_bv);

                if let Some(gtid) = grammar_token_opt {
                    glr_s.step(*gtid);
                }

                if glr_s.is_ok() {
                    Some(glr_s)
                } else {
                    None
                }
            },
            // merge_fn
            |glr_s1, glr_s2| {
                glr_s1.merge_with(glr_s2);
            },
            // process_fn: (precomputed_node_data, final_glr_s_for_this_path)
            |precomputed_node_data, final_glr_s| {
                if final_glr_s.active_state.stack.is_empty() {
                    return false;
                }


                if let Some(clean_end_bv) = &precomputed_node_data.value.clean_end {
                    let glr_active_tokens = final_glr_s.active_state.stack.acc_acc().clone().unwrap_or_else(LLMTokenBV::max_ones);
                    let mask_contribution = &glr_active_tokens & clean_end_bv;
                    final_mask_internal |= mask_contribution;
                }

                for (grammar_token, finalizer) in precomputed_node_data.value.finalizers() {
                    let mut temp_glr_s_for_finalizer_step = final_glr_s.clone();
                    temp_glr_s_for_finalizer_step.step(*grammar_token);

                    if temp_glr_s_for_finalizer_step.is_ok() {
                        let glr_active_after_step = temp_glr_s_for_finalizer_step.active_state.stack.acc_acc().clone().unwrap_or_else(LLMTokenBV::max_ones);
                        let mask_contribution = &glr_active_after_step & &finalizer.content;
                        final_mask_internal |= &mask_contribution;
                    }
                }
                true 
            },
        );
        crate::debug!(2, "Done computing mask");
        self.parent.internal_bv_to_original(&final_mask_internal)
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) { // llm_token_id is original
        let llm_token_bytes = self.parent.llm_token_map.get_by_right(&llm_token_id).unwrap();
        self.commit_bytes(llm_token_bytes);
    }

    pub fn commit_bytes(&mut self, llm_token_bytes: &[u8]) { // llm_token_id is original
        crate::debug!(2, "Committing bytes: {:?}", String::from_utf8_lossy(llm_token_bytes));

        for state in self.state.values_mut() {
            Arc::make_mut(&mut state.active_state.stack).reset_llm_tokens();
        }

        // Handle allowed terminals
        let mut state_map: BTreeMap<TokenizerStateID, TokenizerStateID> = BTreeMap::new();
        
        for tokenizer_state_id in self.state.keys() {
            let exec_result = self.parent.tokenizer.execute_from_state(
                &llm_token_bytes,
                *tokenizer_state_id,
            );
            if let Some(new_state) = exec_result.end_state {
                state_map.insert(*tokenizer_state_id, TokenizerStateID(new_state));
            }
        }

        for state in self.state.values_mut() {
            Arc::make_mut(&mut state.active_state.stack).map_allowed_terminals_tokenizer_states(&state_map);
        }

        let mut new_overall_state: BTreeMap<TokenizerStateID, GLRParserState<'a>> = BTreeMap::new();

        let mut processing_queue: BTreeMap<usize, BTreeMap<TokenizerStateID, GLRParserState<'a>>> = BTreeMap::new();
        processing_queue.insert(0, std::mem::take(&mut self.state));

        while let Some((offset, states_to_process)) = processing_queue.pop_first() {
            crate::debug!(3, "Processing offset {} with states {:?}.", offset, states_to_process.keys().map(|k| k.0).collect::<Vec<_>>());
            for (tokenizer_s_id_at_offset, glr_s_at_offset) in states_to_process {
                assert!(offset <= llm_token_bytes.len());
                if offset == llm_token_bytes.len() {
                    // This path fully consumed the llm_token_bytes.
                    if let Some(existing_glr_s) = new_overall_state.get_mut(&tokenizer_s_id_at_offset) {
                        existing_glr_s.merge_with(glr_s_at_offset);
                    } else {
                        new_overall_state.insert(tokenizer_s_id_at_offset, glr_s_at_offset);
                    }
                    continue;
                }

                let exec_result = self.parent.tokenizer.execute_from_state(
                    &llm_token_bytes[offset..],
                    tokenizer_s_id_at_offset,
                );

                let mut possible_matches = BTreeMap::new();
                if let Some(final_tokenizer_s_id_for_llm_token_segment) = exec_result.end_state {
                    possible_matches = self.parent.possible_matches[&TokenizerStateID(final_tokenizer_s_id_for_llm_token_segment)].clone();
                }

                for match_info in &exec_result.matches {
                    let mut cloned_glr_s = glr_s_at_offset.clone();
                    if let Some(bv) = possible_matches.get(&TerminalID(match_info.id)) {
                        Arc::make_mut(&mut cloned_glr_s.active_state.stack).subtract_llm_tokens_and_prune_arc(&bv);
                    }

                    cloned_glr_s.step(TerminalID(match_info.id));

                    if cloned_glr_s.is_ok() {
                        let new_offset = offset + match_info.width;
                        // After a grammar token is consumed, the tokenizer resets for the next segment of the LLM token.
                        let next_tokenizer_id_for_segment = self.parent.tokenizer.initial_state_id();
                        processing_queue.entry(new_offset).or_default().entry(next_tokenizer_id_for_segment).and_modify(|existing| existing.merge_with(cloned_glr_s.clone())).or_insert_with(|| cloned_glr_s);
                    }
                }

                if let Some(final_tokenizer_s_id_for_llm_token_segment) = exec_result.end_state {
                    // The rest of llm_token_bytes (from offset) was consumed, tokenizer ended in this state.
                    // The glr_s_at_offset is carried over. This is a state *after* the current LLM token.
                    let final_tokenizer_state = TokenizerStateID(final_tokenizer_s_id_for_llm_token_segment);
                    if let Some(existing_glr_s) = new_overall_state.get_mut(&final_tokenizer_state) {
                        existing_glr_s.merge_with(glr_s_at_offset.clone());
                    } else {
                        new_overall_state.insert(final_tokenizer_state, glr_s_at_offset.clone());
                    }
                }
            }

        }

        self.state = new_overall_state.clone();

        // let mut roots_to_simplify_arcs = Vec::new();
        // for glr_parser_state in self.state.values_mut() {
        //     if !glr_parser_state.active_state.stack.is_empty() {
        //         roots_to_simplify_arcs.push(&mut glr_parser_state.active_state.stack);
        //     }
        // }
        //
        // if !roots_to_simplify_arcs.is_empty() {
        //     GSSNode::simplify_together(&mut roots_to_simplify_arcs);
        // }

        crate::debug!(2, "State after committing text (bytes {:?}): {} active tokenizer states.", llm_token_bytes, self.state.len());
        for (tokenizer_id, glr_state) in &self.state {
            if !glr_state.active_state.stack.is_empty() { // Log only for non-empty GSS
                glr_state.log_gss(
                    &format!("GSS for tokenizer state {} after commit of text", tokenizer_id.0),
                    TerminalID(0)
                );
            }
        }
    }

    pub fn is_active(&self) -> bool {
        !self.state.is_empty() && self.state.values().any(|s| !s.active_state.stack.is_empty())
    }

    pub fn state(&self) -> &BTreeMap<TokenizerStateID, GLRParserState<'a>> {
        &self.state
    }
}

