// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use crate::datastructures::ordered_hash_map::Retain;
use crate::datastructures::gss::{disallow_llm_tokens_and_prune_arc, fuse_predecessors_recursive};
use crate::datastructures::gss::{map_allowed_terminals_tokenizer_states, prune_disallowed_terminals};
use ordered_hash_map::OrderedHashMap;
use ordered_hash_map::OrderedHashSet;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::ops::BitOr;
use std::fmt::{self, Display, Formatter};
use std::sync::{Arc, Mutex};
use std::cell::RefCell;

use bimap::BiBTreeMap;
use bitvec::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

use crate::constraint_extra::{calculate_final_stats, dump_precompute_trie_recursive, print_precompute_stats, PrecomputeStats};
use crate::datastructures::gss::{print_gss_forest, GSSNode, allow_only_llm_tokens_and_prune_arc, gather_gss_stats, reset_llm_tokens, disallow_terminals_and_prune_arc, GSSPrintConfig};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::{EdgeInserter, Trie};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::datastructures::ArcPtrWrapper;
use crate::finite_automata::Regex;
use crate::glr::parser::{
    GLRParser, GLRParserState, ParseState, ParseStateEdgeContent,
};
use crate::tokenizer::{LLMToken, LLMTokenID, LLMTokenMap, Token, TokenizerStateID};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use profiler_macro::{time_it, timeit};
use crate::datastructures::gss::Acc;
use crate::glr::analyze::compute_terminal_follow_sets;
use crate::glr::grammar::Terminal;
use crate::glr::items::{LRMode, LR_MODE};
use crate::interface::CompiledGrammar;

pub type LLMTokenBV = HybridBitset;
pub type TerminalBV = HybridBitset;

const MERGE_THRESHOLD: usize = 20;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedNodeContents {
    pub end: bool,
}

impl PrecomputedNodeContents {
    pub fn no_end() -> Self {
        Self { end: false }
    }

    pub fn end() -> Self {
        Self { end: true }
    }
}

impl JSONConvertible for PrecomputedNodeContents {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("clean_end".to_string(), self.end.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let end = obj.remove("clean_end").ok_or_else(|| "Missing field clean_end for PrecomputedNodeContents".to_string())
                                   .and_then(bool::from_json)?;
                Ok(PrecomputedNodeContents { end })
            }
            _ => Err("Expected JSONNode::Object for PrecomputedNodeContents".to_string()),
        }
    }
}


impl PrecomputedNodeContents {
}

pub type PrecomputeNode =
    Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;

pub type Precomputed = BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LLMVocab {
    pub(crate) llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID>,
    pub(crate) max_original_llm_token_id: usize,
    pub(crate) original_to_internal_id_bimap: BiBTreeMap<usize, usize>,
    pub(crate) internal_max_llm_token: usize
}

#[derive(Debug, Clone)]
pub struct GrammarConstraint {
    pub(crate) tokenizer:        Regex,
    pub(crate) parser:           GLRParser,
    pub(crate) precomputed:      Precomputed,
    pub(crate) llm_vocab:        Arc<LLMVocab>,
    pub(crate) token_name_map:   BiBTreeMap<Terminal, usize>,
    pub(crate) possible_matches: BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
}

impl GrammarConstraint {
    pub fn assert_eq(&self, other: &Self) {
        assert_eq!(self.tokenizer, other.tokenizer);
        assert_eq!(self.parser, other.parser);
        assert_eq!(self.precomputed.len(), other.precomputed.len());
        for ((sid1, arc1), (sid2, arc2)) in self.precomputed.iter().zip(other.precomputed.iter()) {
            assert_eq!(sid1, sid2);
            let node1 = arc1.lock().unwrap();
            let node2 = arc2.lock().unwrap();
            assert_eq!(*node1, *node2);
        }
        assert_eq!(self.llm_vocab.llm_token_map, other.llm_vocab.llm_token_map);
        assert_eq!(self.token_name_map, other.token_name_map);
        assert_eq!(self.llm_vocab.max_original_llm_token_id, other.llm_vocab.max_original_llm_token_id);
        assert_eq!(self.llm_vocab.original_to_internal_id_bimap, other.llm_vocab.original_to_internal_id_bimap);
        assert_eq!(self.llm_vocab.internal_max_llm_token, other.llm_vocab.internal_max_llm_token);
        assert_eq!(self.possible_matches, other.possible_matches);
    }
}

impl JSONConvertible for GrammarConstraint {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("tokenizer".to_string(), self.tokenizer.to_json());
        obj.insert("parser".to_string(), self.parser.to_json());
        obj.insert("precomputed".to_string(), self.precomputed.to_json());
        obj.insert("llm_token_map".to_string(), self.llm_vocab.llm_token_map.to_json());
        obj.insert("token_name_map".to_string(), self.token_name_map.to_json());
        obj.insert("max_original_llm_token_id".to_string(), self.llm_vocab.max_original_llm_token_id.to_json());
        obj.insert("original_to_internal_id_bimap".to_string(), self.llm_vocab.original_to_internal_id_bimap.to_json());
        obj.insert("internal_max_llm_token".to_string(), self.llm_vocab.internal_max_llm_token.to_json());
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
                                        .and_then(|n| BiBTreeMap::<Terminal, usize>::from_json(n))?;
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
                    llm_vocab: Arc::new(LLMVocab { llm_token_map, max_original_llm_token_id, original_to_internal_id_bimap, internal_max_llm_token }),
                    token_name_map,
                    possible_matches,
                })
            }
            _ => Err("Expected JSONNode::Object for GrammarConstraint".to_string()),
        }
    }
}


impl GrammarConstraint {
    pub fn from_compiled_grammar(
        compiled_grammar: CompiledGrammar,
        llm_token_map: LLMTokenMap,
        _eof_token_id: LLMTokenID,
        max_original_llm_token_id: usize,
    ) -> Self {
        let token_name_map = compiled_grammar.definition.terminal_to_group_id().clone();

        Self::new(
            compiled_grammar.tokenizer,
            compiled_grammar.glr_parser,
            llm_token_map,
            token_name_map,
            max_original_llm_token_id,
        )
    }

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
        token_name_map:   BiBTreeMap<Terminal, usize>,
        max_original_llm_token_id: usize, 
    ) -> Self {
        let epsilon_terminal_group_ids: BTreeSet<_> = tokenizer.execute_from_state(&[], tokenizer.initial_state_id()).matches.iter().map(|token| token.id).collect();
        let epsilon_terminals: BTreeSet<&Terminal> = epsilon_terminal_group_ids.iter().map(|id| token_name_map.get_by_right(id).unwrap()).collect();
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

        let terminal_follow_sets_named = compute_terminal_follow_sets(grammar_productions);
        crate::debug!(3, "terminal_follow_sets_named:");
        for (terminal, following_terminals) in &terminal_follow_sets_named {
            crate::debug!(3, "{} -> {}", terminal, following_terminals.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "));
        }

        let mut terminal_follow_map: BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>> = BTreeMap::new();
        for (terminal1, following_terminals) in terminal_follow_sets_named {
            let t1_id = *grammar_term_map.get_by_left(&terminal1).unwrap();
            let mut following_ids = BTreeSet::new();
            for t2 in following_terminals {
                let t2_id = *grammar_term_map.get_by_left(&t2).unwrap();
                following_ids.insert(t2_id);
            }
            if !following_ids.is_empty() {
                terminal_follow_map.insert(t1_id, following_ids);
            }
        }

        crate::debug!(2, "Computed terminal_follow_map_ids with {} entries.", terminal_follow_map.len());

        let precomputed = Self::precompute(
            &tokenizer, // This is the tokenizer parameter being moved into the struct
            &internal_llm_token_map_for_precompute,
            &token_name_map,
            internal_max_llm_token,
            &terminal_follow_map, // Pass the new map
            parser.ignore_terminal_id,
            &mut computed_possible_matches,
        );

        Self {
            tokenizer, // This is the tokenizer parameter being moved into the struct
            parser,
            precomputed,
            llm_vocab: Arc::new(LLMVocab {
                llm_token_map,
                max_original_llm_token_id,
                original_to_internal_id_bimap,
                internal_max_llm_token,
            }),
            token_name_map,
            possible_matches: computed_possible_matches, // Add this line
        }
    }

    pub fn precompute(
        tokenizer:        &Regex,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        token_name_map:   &BiBTreeMap<Terminal, usize>,
        internal_max_llm_token: usize,
        terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        ignore_terminal_id: Option<TerminalID>,
        possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    ) -> BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>> {
        let mut helper = Precomputer::new(
            tokenizer,
            internal_llm_token_map,
            internal_max_llm_token,
            MERGE_THRESHOLD,
            terminal_follow_map, // Pass to Precomputer::new
            ignore_terminal_id,
        );

        helper.run_dfs();
        helper.prune_dead_paths();
        helper.prune_on_no_terminal_follow();
        helper.prune_dead_paths();
        helper.prune_on_no_terminal_follow();
        helper.merge_nodes();
        helper.finish(token_name_map, possible_matches, internal_max_llm_token)
    }

    pub fn init(&self) -> GrammarConstraintState<'_> {
        let mut state = BTreeMap::new();
        state.insert(
            self.tokenizer.initial_state_id(),
            self.parser.init_glr_parser(Some(self.llm_vocab.clone())),
        );

        GrammarConstraintState { parent: self, state }
    }

    #[inline]
    pub(crate) fn original_id_to_internal(&self, original_id: LLMTokenID) -> Option<LLMTokenID> {
        self.llm_vocab.original_to_internal_id_bimap.get_by_left(&original_id.0).map(|internal_val| LLMTokenID(*internal_val))
    }

    #[inline]
    pub(crate) fn internal_id_to_original(&self, internal_id: LLMTokenID) -> Option<LLMTokenID> {
        self.llm_vocab.original_to_internal_id_bimap.get_by_right(&internal_id.0).map(|original_val| LLMTokenID(*original_val))
    }

    #[allow(dead_code)] 
    pub(crate) fn original_bv_to_internal(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        let mut internal_bv = HybridBitset::zeros();
        for original_id_val in original_bv.iter() {
            let internal_id_val = self.llm_vocab.original_to_internal_id_bimap.get_by_left(&(original_id_val as usize)).expect(format!("Original ID {} not found in original_to_internal_id_bimap", original_id_val).as_str());
            internal_bv.insert(*internal_id_val as usize);
        }
        internal_bv
    }

    #[time_it]
    fn internal_bv_to_original(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        let internal_bv = internal_bv & &LLMTokenBV::max_ones();
        let mut original_bv = HybridBitset::zeros();
        // for internal_id_val in internal_bv.iter() {
        //     let original_id_val = self.llm_vocab.original_to_internal_id_bimap.get_by_right(&(internal_id_val as usize)).expect(format!("Internal ID {} not found in original_to_internal_id_bimap while converting to original BV from internal BV: {:?}", internal_id_val, internal_bv).as_str());
        //     original_bv.insert(*original_id_val as usize);
        // }
        for i in 0..=self.llm_vocab.internal_max_llm_token {
            if internal_bv.contains(i) {
                if let Some(original_id_val) = self.llm_vocab.original_to_internal_id_bimap.get_by_right(&i) {
                    original_bv.insert(*original_id_val);
                }
            }
        }
        original_bv
    }

    pub(crate) fn all_internal_llm_tokens_bitset(&self) -> LLMTokenBV {
        HybridBitset::max_ones()
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
    terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    ignore_terminal_id: Option<TerminalID>,
    // Map each precompute node to its contents and the token node/position/state used to compute its
    tags:             RefCell<HashMap<ArcPtrWrapper<Mutex<PrecomputeNode>>, LLMTokenBV>>,
    end_node:       ArcPtrWrapper<Mutex<PrecomputeNode>>,
}

impl<'r> Precomputer<'r> {
    fn new(
        tokenizer:        &'r Regex,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>, 
        internal_max_llm_token: usize,                       
        merge_threshold:  usize,
        terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>, // New parameter
        ignore_terminal_id: Option<TerminalID>,
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
                    PrecomputedNodeContents::no_end(),
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
            all_llm_tokens: HybridBitset::max_ones(),
            merge_threshold,
            pb,
            stats: PrecomputeStats::default(),
            terminal_follow_map, // Store the map
            ignore_terminal_id,
            tags: RefCell::new(Default::default()),
            end_node: ArcPtrWrapper::new(Arc::new(Mutex::new(PrecomputeNode::new(PrecomputedNodeContents::end())))),
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
        self.dfs(&self.vocab.root, assoc, HashMap::new());
        crate::debug!(2, "Finished precompute DFS");
        self.pb.finish_with_message("Precomputation complete");
        crate::debug!(2, "Precomputation complete");
    }
    
    fn prune_on_no_terminal_follow(&mut self) {
        crate::debug!(2, "Pruning based on terminal follow sets.");

        let terminal_follow_map = &self.terminal_follow_map;
        let ignore_terminal_id = self.ignore_terminal_id;

        // Collect all terminals from the tokenizer.
        let all_terminals: BTreeSet<GrammarTokenID> = terminal_follow_map.keys().cloned().collect();

        let initial_nodes_and_values: Vec<_> = self.roots.values()
            .map(|root_arc| (root_arc.clone(), all_terminals.clone()))
            .collect();

        Trie::special_map(
            initial_nodes_and_values,
            // step: Propagate predecessor terminals.
            |predecessors, edge_terminal_opt, _edge_bv, _child_node| {
                match edge_terminal_opt {
                    Some(t) if Some(*t) == ignore_terminal_id => Some(predecessors.clone()),
                    Some(t) => Some(BTreeSet::from([*t])),
                    None => Some(predecessors.clone()),
                }
            },
            // merge: Union of predecessor sets from different paths.
            |existing_set, new_set| {
                existing_set.extend(new_set);
            },
            // process: Prune outgoing edges based on allowed follows.
            move |node, all_immediate_predecessors| {
                // If there are no preceding terminals (e.g., root or only None-edges path from root),
                // all outgoing terminals are considered valid.
                if all_immediate_predecessors.is_empty() {
                    return true; // Continue traversal
                }

                // Compute the set of all allowed terminals that can follow any of the immediate predecessors.
                let mut allowed_follow_terminals = BTreeSet::new();
                for preceding_terminal in &*all_immediate_predecessors {
                    if let Some(follow_set) = terminal_follow_map.get(preceding_terminal) {
                        allowed_follow_terminals.extend(follow_set.iter().cloned());
                    }
                }

                // Prune children of the current node.
                node.children_mut().retain(|edge_terminal_opt, _dest_map| {
                    match edge_terminal_opt {
                        // Keep edges with terminals that are in the allowed follow set (or ignore edges).
                        Some(edge_terminal) => allowed_follow_terminals.contains(edge_terminal) || Some(*edge_terminal) == ignore_terminal_id,
                        // Always keep `None` edges, as they don't represent grammar terminals.
                        None => true,
                    }
                });
    
                true // Continue traversal
            },
        );
        crate::debug!(2, "Finished pruning based on terminal follow sets.");
    }

    fn prune_dead_paths(&mut self) {
        crate::debug!(2, "Pruning dead paths from precomputed trie.");

        let mut live_nodes_cache: HashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>> = HashSet::new();
        let mut visited_for_dfs: HashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>> = HashSet::new();

        // A node is "live" if it can reach a node with `value.end == true`. We do a post-order
        // traversal (DFS) from each root. `is_live_and_prune` recursively determines if a node
        // is live and prunes its dead children.
        //
        // We can't use `BTreeMap::retain` directly because its closure would borrow `self`
        // immutably (to call `is_live_and_prune`) while `retain` itself holds a mutable borrow
        // on `self.roots`. Instead, we collect the keys of roots to remove and then remove them.
        let sids_to_remove: Vec<_> = self.roots.iter().filter_map(|(sid, root_arc)| {
            let root_wrapper = ArcPtrWrapper::new(root_arc.clone());
            if self.is_live_and_prune(root_wrapper, &mut live_nodes_cache, &mut visited_for_dfs) {
                None // This root is live, keep it.
            } else {
                Some(*sid) // This root is not live, mark for removal.
            }
        }).collect();

        for sid in sids_to_remove {
            self.roots.remove(&sid);
        }

        crate::debug!(2, "Finished pruning dead paths.");
    }

    /// Recursively determines if a node is "live" (can reach an end node)
    /// and prunes its children that are not live. This is a post-order traversal.
    ///
    /// - `node_wrapper`: The node to check.
    /// - `live_nodes_cache`: A cache of nodes already determined to be live.
    /// - `visited_for_dfs`: Tracks nodes visited in the current DFS traversal to handle shared nodes and cycles.
    ///
    /// Returns `true` if `node_wrapper` is live, `false` otherwise.
    fn is_live_and_prune(
        &self,
        node_wrapper: ArcPtrWrapper<Mutex<PrecomputeNode>>,
        live_nodes_cache: &mut HashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        visited_for_dfs: &mut HashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
    ) -> bool {
        // If we've already determined this node is live, we're done.
        if live_nodes_cache.contains(&node_wrapper) {
            return true;
        }
        // If we've visited this node in the DFS but it's not in the live cache,
        // it means it was processed and found to be dead.
        if visited_for_dfs.contains(&node_wrapper) {
            return false;
        }
        visited_for_dfs.insert(node_wrapper.clone());

        let node_arc = node_wrapper.as_arc();

        // A node is live if it's an end node itself.
        let is_this_node_an_end_node = node_arc.lock().unwrap().value.end;

        // Or if it has at least one live child. We find out by recursing.
        // We must collect children before recursing to avoid holding the lock.
        let children_to_check: Vec<ArcPtrWrapper<Mutex<PrecomputeNode>>> = {
            let node_guard = node_arc.lock().unwrap();
            node_guard.children().values().flat_map(|dest_map| dest_map.keys().cloned()).collect()
        };

        let mut live_children_for_this_node = HashSet::new();
        let mut has_live_child = false;

        for child_wrapper in children_to_check {
            if self.is_live_and_prune(child_wrapper.clone(), live_nodes_cache, visited_for_dfs) {
                has_live_child = true;
                live_children_for_this_node.insert(child_wrapper);
            }
        }

        // Now that we know which children are live, prune the dead ones.
        {
            let mut node_guard = node_arc.lock().unwrap();
            node_guard.children_mut().retain(|_edge_key, dest_map| {
                dest_map.retain(|child_wrapper, _edge_value| {
                    live_children_for_this_node.contains(child_wrapper)
                });
                // Keep the edge key only if it still has destinations.
                !dest_map.is_empty()
            });
        }

        let is_node_live = is_this_node_an_end_node || has_live_child;

        if is_node_live {
            live_nodes_cache.insert(node_wrapper);
        }

        is_node_live
    }

    fn merge_nodes(&mut self) {
        crate::debug!(2, "Merging nodes: first collecting unique roots and their canonical Arcs");
        let mut content_to_canonical_arc_map: HashMap<PrecomputeNode, Arc<Mutex<PrecomputeNode>>> = HashMap::new();
        
        for (_tokenizer_state_id, root_arc_ref) in &self.roots {
            crate::debug!(3, "Merging nodes: first collecting unique roots and their canonical Arcs: Root {:p}", root_arc_ref);
            let node_content = root_arc_ref.lock().unwrap().clone();
            crate::debug!(3, "Merging nodes: first collecting unique roots and their canonical Arcs: Root {:p} lock acquired, content: {:?}", root_arc_ref, node_content);
            // This will associate node_content with root_arc_ref.clone().
            // If node_content was already in the map, its associated Arc gets updated to root_arc_ref.clone().
            // This implements a "last one wins" policy for which Arc becomes canonical for a given content.
            content_to_canonical_arc_map.insert(node_content, root_arc_ref.clone());
        }

        crate::debug!(2, "Merging nodes: second pass, rewriting roots in self.roots to point to canonical Arcs");
        for (_tokenizer_state_id, root_arc_in_self_roots_mut) in &mut self.roots {
            crate::debug!(3, "Merging nodes: second pass, rewriting roots in self.roots to point to canonical Arcs: Root {:p}", root_arc_in_self_roots_mut);
            let current_content = root_arc_in_self_roots_mut.lock().unwrap().clone();
            if let Some(canonical_arc) = content_to_canonical_arc_map.get(&current_content) {
                crate::debug!(3, "Merging nodes: canonical Arc found for {:?}, updating root to {:p}", current_content, canonical_arc);
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

    fn finish(
        mut self,
        token_name_map: &BiBTreeMap<Terminal, usize>,
        possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
        internal_max_llm_token: usize,
    ) -> BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>> {

        calculate_final_stats(&self.roots, &mut self.stats);
        print_precompute_stats(&self.stats, token_name_map);

        self.roots
    }

    fn dfs(
        &self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<
            TokenizerStateID,
            OrderedHashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        >,
        no_go: HashMap<ArcPtrWrapper<Mutex<PrecomputeNode>>, LLMTokenBV>,

    ) {
        self.pb.inc(1);

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let mut work_queue: BTreeMap<usize, BTreeMap<TokenizerStateID, OrderedHashSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>>> = BTreeMap::new();
            work_queue.insert(0, assoc_by_state.clone());

            let mut next_level_assoc: BTreeMap<_, OrderedHashSet<_>> = BTreeMap::new();

            while let Some((pos, states_at_pos)) = work_queue.pop_first() {
                if pos == segment_bytes.len() {
                    for (tokenizer_state_id, nodes) in states_at_pos {
                        next_level_assoc.entry(tokenizer_state_id).or_default().extend(nodes);
                    }
                    continue;
                }

                for (tokenizer_state_id, precompute_nodes) in states_at_pos {
                    let exec_result = self.tokenizer.execute_from_state(&segment_bytes[pos..], tokenizer_state_id);

                    let possible_matches_at_end = if let Some(end_state_val) = exec_result.end_state {
                        self.possible_matches(child_vocab_node, TokenizerStateID(end_state_val))
                    } else {
                        BTreeMap::new()
                    };

                    for match_info in &exec_result.matches {
                        let terminal_id = GrammarTokenID(match_info.id);
                        let next_pos = pos + match_info.width;

                        // TODO: could make this so much faster by moving loop down...
                        for src_node_wrapper in &precompute_nodes {
                            if next_pos == segment_bytes.len() {
                                // TODO: should be some way of avoiding ignored terminal here.
                                let llm_token_id = child_vocab_node.token_id();
                                let mut edge_bv = HybridBitset::zeros();
                                edge_bv.insert(llm_token_id);
                                let mut inserter = EdgeInserter::new(
                                    src_node_wrapper.as_arc().clone(),
                                    Some(terminal_id),
                                    edge_bv,
                                    |e, n| *e |= n,
                                );
                                // Print the source node.
                                // dump_precompute_trie_recursive(src_node_wrapper, String::new(), &mut HashSet::new(), None);
                                inserter.try_destination(self.end_node.as_arc().clone()).expect("Failed to insert end node for terminal at end of segment");
                            }

                            let mut edge_bv = child_vocab_node.reachable_token_ids().clone();
                            if next_pos == segment_bytes.len() {
                                edge_bv.set(child_vocab_node.token_id(), false);
                            }
                            if let Some(matches_for_terminal) = possible_matches_at_end.get(&terminal_id) {
                                edge_bv -= matches_for_terminal;
                            }

                            if edge_bv.is_empty() { continue; }

                            let mut inserter = EdgeInserter::new(
                                src_node_wrapper.as_arc().clone(),
                                Some(terminal_id),
                                edge_bv.clone(),
                                |e, n| *e |= n,
                            );

                            let next_tokenizer_state = self.tokenizer.initial_state_id();
                            let dest_nodes_in_queue = work_queue.entry(next_pos).or_default().entry(next_tokenizer_state).or_default();

                            inserter = inserter.try_destinations_iter(dest_nodes_in_queue.iter().map(|w| w.as_arc().clone()).filter(|w| !w.lock().unwrap().value.end));

                            if true {
                                let children_of_src: Vec<_> = if true { src_node_wrapper.lock().unwrap().children().values().flat_map(|m| m.keys().cloned()).collect() }
                                else { src_node_wrapper.lock().unwrap().children().get(&Some(terminal_id)).cloned().unwrap_or_default().keys().cloned().collect() };
                                let tags = self.tags.borrow();
                                let eligible_children = children_of_src.iter().filter(|child_wrapper| {
                                    tags.get(child_wrapper).map_or(true, |tag| (tag & &edge_bv).is_empty()) && !child_wrapper.lock().unwrap().value.end
                                }).map(|w| w.as_arc().clone());
                                inserter = inserter.try_destinations_iter(eligible_children);
                                drop(tags);
                            }

                            let result_node = inserter.else_create_destination_with_value(PrecomputedNodeContents::no_end()).unwrap();
                            dest_nodes_in_queue.insert(ArcPtrWrapper::new(result_node.clone()));
                            *self.tags.borrow_mut().entry(ArcPtrWrapper::new(result_node)).or_insert_with(HybridBitset::zeros) |= &edge_bv;
                        }
                    }

                    if let Some(end_state_val) = exec_result.end_state {
                        let possible_final_tokens = self.tokenizer.tokens_accessible_from_state(TokenizerStateID(end_state_val));
                        for terminal_id in possible_final_tokens {
                            for src_node_wrapper in &precompute_nodes {
                                let llm_token_id = child_vocab_node.token_id();
                                let mut edge_bv = HybridBitset::zeros();
                                edge_bv.insert(llm_token_id);
                                let mut inserter = EdgeInserter::new(
                                    src_node_wrapper.as_arc().clone(),
                                    Some(terminal_id),
                                    edge_bv,
                                    |e, n| *e |= n,
                                );
                                // Print the source node.
                                // dump_precompute_trie_recursive(src_node_wrapper, String::new(), &mut HashSet::new(), None);
                                inserter.try_destination(self.end_node.as_arc().clone()).expect("Failed to insert end node for terminal at end of segment");
                            }
                        }
                        next_level_assoc.entry(TokenizerStateID(end_state_val)).or_default().extend(precompute_nodes.iter().cloned());
                    }
                }
            }



            if !next_level_assoc.is_empty() {
                self.dfs(child_vocab_node, next_level_assoc, no_go.clone());
            }
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
            PrecomputedNodeContents::no_end(),
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
            inserter = inserter.try_destination(merged_node_arc.clone());
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
    pub(crate) state:  BTreeMap<TokenizerStateID, GLRParserState<'a>>,
}

impl<'a> PartialEq for GrammarConstraintState<'a> {
    fn eq(&self, other: &Self) -> bool {
        // Compare parent by pointer to ensure they originate from the same constraint object.
        std::ptr::eq(self.parent, other.parent) && self.state == other.state
    }
}

impl<'a> Eq for GrammarConstraintState<'a> {}

impl<'a> Display for GrammarConstraintState<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "GrammarConstraintState ({} active tokenizer states):", self.state.len())?;
        if self.state.is_empty() {
            return Ok(());
        }

        let mut gss_roots = Vec::new();
        let mut tokenizer_state_info = Vec::new();

        for (tokenizer_state_id, glr_state) in &self.state {
            if !glr_state.active_state.stack.is_empty() {
                gss_roots.push(glr_state.active_state.stack.clone());
                tokenizer_state_info.push(format!(
                    "  - Tokenizer State {:>3}: GSS Root ({} predecessors)",
                    tokenizer_state_id.0,
                    glr_state.active_state.stack.num_predecessors()
                ));
            } else {
                tokenizer_state_info.push(format!(
                    "  - Tokenizer State {:>3}: (Empty GSS)",
                    tokenizer_state_id.0
                ));
            }
        }

        for info in tokenizer_state_info {
            writeln!(f, "{}", info)?;
        }

        if !gss_roots.is_empty() {
            writeln!(f, "\nCombined GSS Forest (showing up to 50 nodes):")?;
            let config = GSSPrintConfig {
                labels: None,
                max_nodes: 50,
                original_internal_bimap: Some(&self.parent.llm_vocab.original_to_internal_id_bimap),
                llm_token_map: Some(&self.parent.llm_vocab.llm_token_map),
                verbose: false,
            };
            let (gss_str, _) =
                crate::datastructures::gss::print_gss_forest(&gss_roots, &self.parent.parser.terminal_map, &config);
            write!(f, "{}", gss_str)?;
        }

        Ok(())
    }
}

impl<'a> GrammarConstraintState<'a> {
    #[time_it]
    pub fn get_mask(&self) -> LLMTokenBV {
        let t0 = std::time::Instant::now();
        crate::debug!(2, "Computing mask with {} states: {:?}", self.state.len(), self.state.keys().map(|k|k.0).collect::<Vec<_>>());
        let stats = gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        );
        crate::debug!(3, "GSS stats: {:#?}", stats);
        let roots = self.state.values().map(|s| s.active_state.stack.clone()).collect::<Vec<_>>();
        let (s, state_ids) = print_gss_forest(&roots, &self.parent.parser.terminal_map, &GSSPrintConfig::default());
        println!("{}", s);
        println!("\n\n--- GSS State Explanations ---\n");
        for state_id in state_ids {
            let mut explanation = String::new();
            println!("\n--- State {} ---", state_id.0);
            self.parent.parser.format_state_details(&mut explanation, state_id, "  ").unwrap();
            println!("{}", explanation);
        }

        println!("\n\n--- Begin GSS Graphviz ---");
        let labels: Vec<String> = self.state.keys().map(|k| format!("State {}", k.0)).collect();
        let roots_with_labels: Vec<(&str, &GSSNode)> = labels.iter()
            .map(|s| s.as_str())
            .zip(self.state.values().map(|s| s.active_state.stack.as_ref()))
            .collect();
        println!("{}", self.parent.parser.gss_forest_to_dot(
            &roots_with_labels,
            Some(&self.parent.llm_vocab.original_to_internal_id_bimap),
            Some(&self.parent.llm_vocab.llm_token_map),
        ));
        println!("\n\n--- End GSS Graphviz ---");

        for (state_id, state) in self.state.iter() {
            crate::debug!(3, "State {}:", state_id.0);
        }

        let final_mask_internal = RefCell::new(HybridBitset::zeros());

        if self.state.is_empty() {
            return self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        }

        #[derive(Default, Clone, Copy, Debug)]
        struct StepCount {
            total: usize,
            successful: usize,
        }

        let step_counts = Arc::new(Mutex::new(BTreeMap::<TerminalID, StepCount>::new()));

        let mut initial_values_for_map: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'a>)> = Vec::new();
        for (tokenizer_state_id, glr_state) in &self.state {
            // crate::debug!(4, "Initializing GSS for state {}", tokenizer_state_id.0);
            // Ensure the GLR state's GSS stack is not empty before proceeding
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            if let Some(precomputed_trie_root_arc) = self.parent.precomputed.get(tokenizer_state_id) {
                let mut forbidden_llm_tokens = LLMTokenBV::zeros();
                let disallowed_terminals_l2 = glr_state.active_state.stack.disallowed_terminals();

                // Iterate over ranges of tokenizer states that have the same set of disallowed terminals.
                for (tokenizer_state_range, disallowed_terminals_for_range) in disallowed_terminals_l2.range_values() {
                    if disallowed_terminals_for_range.is_empty() {
                        continue;
                    }

                    // Get a sub-view of possible_matches that covers this range of tokenizer states.
                    let relevant_possible_matches = self.parent.possible_matches.range(TokenizerStateID(*tokenizer_state_range.start())..=TokenizerStateID(*tokenizer_state_range.end()));

                    for (_tokenizer_state_id, possible_matches_for_state) in relevant_possible_matches {
                        for (terminal_id, llm_tokens_that_match_this_terminal) in possible_matches_for_state {
                            if disallowed_terminals_for_range.contains(terminal_id.0) {
                                forbidden_llm_tokens |= llm_tokens_that_match_this_terminal;
                            }
                        }
                    }
                }
                let mut glr_state = glr_state.clone();
                if !forbidden_llm_tokens.is_empty() {
                    // glr_state.log_gss(format!("Subtracting forbidden LLM tokens: {:?}", forbidden_llm_tokens).as_str(), TerminalID(0));
                    disallow_llm_tokens_and_prune_arc(&mut glr_state.active_state.stack, &forbidden_llm_tokens, &mut HashMap::new());
                    // glr_state.log_gss("Done subtracting forbidden LLM tokens.", TerminalID(0));
                }
                initial_values_for_map.push((precomputed_trie_root_arc.clone(), glr_state));
            } else {
                panic!("No precomputed trie found for tokenizer state {:?}.", tokenizer_state_id);
            }
        }

        if initial_values_for_map.is_empty() {
             // This can happen if all GLR states had empty GSS stacks or no corresponding precomputed tries.
             crate::debug!(2, "No valid initial states for get_mask's special_map traversal.");
             return self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        }

        let t1 = std::time::Instant::now();
        println!("after initial_values_for_map: {:>15?}", t1.duration_since(t0));

        let step_counts_clone1 = Arc::clone(&step_counts);
        let step_counts_clone2 = Arc::clone(&step_counts);

        crate::profiler::reset();

        Trie::special_map_grouped(
            initial_values_for_map,
            // step_fn: (current_glr_state, edge_grammar_token_opt, destinations_map)
            |glr_s, grammar_token_opt, dest_map| {
                if true {
                    timeit!("get_mask try to avoid step for no additional llm tokens", {
                    let mut all_edge_llm_tokens = HybridBitset::zeros();
                    for edge_llm_tokens_bv in dest_map.values() {
                        all_edge_llm_tokens |= edge_llm_tokens_bv;
                    }
                    let glr_s_llm_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                    let potential_additional_llm_tokens = &glr_s_llm_tokens & &all_edge_llm_tokens;
                    if potential_additional_llm_tokens.is_subset(&final_mask_internal.borrow()) {
                        // If the potential additional tokens are already in the final mask, skip stepping.
                        crate::debug!(4, "Skipping step for grammar token {:?} as all edge LLM tokens are already in final mask.", grammar_token_opt);
                        return Vec::new();
                    }
                    });
                }

                // let mut glr_s = glr_s.clone();
                // disallow_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &final_mask_internal.borrow(), &mut HashMap::new());

                // Count num end nodes vs non end nodes
                let mut num_end = 0;
                let mut num_non_end = 0;
                for (child_node_trie_data, _edge_llm_tokens_bv) in dest_map.iter() {
                    if child_node_trie_data.as_arc().lock().unwrap().value.end {
                        num_end += 1;
                    } else {
                        num_non_end += 1;
                    }
                }
                timeit!(format!("get_mask step_fn - end only? {}", num_end > 0 && num_non_end == 0), {
                    if num_non_end == 0 {
                        if let Some(gtid) = grammar_token_opt {
                            // let stats = gather_gss_stats(&[glr_s.active_state.stack.as_ref()]);
                            // crate::debug!(3, "Step for grammar token {:?} with only end nodes, GSS stats: {:#?}", gtid, stats);
                            // Perhaps we can avoid stepping by calling `has_action_for`
                            match glr_s.has_action_for(*gtid) {
                                Some(glr_s_llm_tokens) => {
                                    timeit!(format!("get_mask step_fn - has_action_for"), {
                                        // This token will succeed
                                        crate::debug!(4, "Step with grammar token {:?} has action, but all children are end nodes, so we can skip stepping and update final mask directly.", gtid);
                                        let mut edge_llm_tokens = HybridBitset::zeros();
                                        for edge_llm_tokens_bv in dest_map.values() {
                                            edge_llm_tokens |= edge_llm_tokens_bv;
                                        }
                                        let llm_tokens = &glr_s_llm_tokens & &edge_llm_tokens;
                                        crate::debug!(4, "Adding tokens {:?} to final mask", llm_tokens);
                                        *final_mask_internal.borrow_mut() |= llm_tokens;
                                        return Vec::new();
                                    });
                                },
                                None => {
                                    timeit!(format!("get_mask step_fn - has_action_for - inconclusive"), {
                                        // Inconclusive
                                        crate::debug!(4, "Inconclusive step for grammar token {:?}, no action found.", gtid);
                                    });
                                },
                            }
                        }
                    }

                    let mut results = Vec::new();
                    let mut glr_s = glr_s.clone();

                    if let Some(gtid) = grammar_token_opt {
                        let mut counts_guard = step_counts_clone1.lock().unwrap();
                        let entry = counts_guard.entry(*gtid).or_default();
                        entry.total += 1;

                        let terminal_name = self.parent.parser.terminal_map.get_by_right(gtid)
                            .map(|s| s.to_string())
                            .unwrap_or("UNKNOWN_TERMINAL".to_string());
                        // timeit!(format!("get_mask step for terminal '{}'", terminal_name), {
                        glr_s.process_token(*gtid);
                        // });

                        crate::debug!(4, "glr_s.is_ok(): {}", glr_s.is_ok());

                        if glr_s.is_ok() {
                            entry.successful += 1;
                        } else {
                            return results;

                        }
                    }

                    // glr_s.log_gss("After stepping", grammar_token_opt.unwrap_or(TerminalID(0)));
                    // disallow_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &final_mask_internal.borrow(), &mut HashMap::new());

                    crate::debug!(4, "Processing edge: {:?}", grammar_token_opt);
                    for (child_node_trie_data, edge_llm_tokens_bv) in dest_map.iter() {
                        let mut glr_s = glr_s.clone();
                        allow_only_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &edge_llm_tokens_bv, &mut HashMap::new());
                        crate::debug!(4, "Stepping with grammar_token_opt: {:?}", grammar_token_opt);
                        glr_s.log_gss("Stepping with grammar_token_opt", grammar_token_opt.unwrap_or(TerminalID(0)), false, false);
                        crate::debug!(4, "Active LLM tokens: {:?}", glr_s.active_state.stack.allowed_llm_tokens());
                        crate::debug!(4, "Edge LLM tokens: {:?}", edge_llm_tokens_bv);
                        // crate::debug!(4, "Intersecting with edge_llm_tokens_bv: {:?}", edge_llm_tokens_bv);
                        // subtract_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &final_mask_internal.borrow(), &mut HashMap::new());
                        // glr_s.log_gss("After intersecting", grammar_token_opt.unwrap_or(TerminalID(0)));

                        if !glr_s.is_ok() {
                            crate::debug!(4, "GLR state is not alive after step, skipping.");
                            continue;
                        }

                        if child_node_trie_data.as_arc().lock().unwrap().value.end {
                            let glr_active_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                            crate::debug!(4, "Adding active tokens {:?} to final mask", glr_active_tokens);
                            // timeit!("get_mask final_mask update", {
                            *final_mask_internal.borrow_mut() |= glr_active_tokens;
                            // });
                        }

                        results.push((child_node_trie_data.clone(), glr_s));
                    }
                    crate::debug!(4, "Step function results len: {}", results.len());
                    results
                })
            },
            // merge_fn
            |glr_s1, glr_s2| {
                timeit!("get_mask merge_fn", {
                    crate::debug!(4, "Active LLM tokens in glr_s1 before merge: {:?}", glr_s1.active_state.stack.allowed_llm_tokens());
                    crate::debug!(4, "Active LLM tokens in glr_s2 before merge: {:?}", glr_s2.active_state.stack.allowed_llm_tokens());
                    glr_s1.merge_with(glr_s2);
                    crate::debug!(4, "Active LLM tokens in glr_s1 after merge: {:?}", glr_s1.active_state.stack.allowed_llm_tokens());
                })
            },
            // process_fn: (precomputed_node_data, final_glr_s_for_this_path)
            |precomputed_node_data, glr_s| {
                timeit!("get_mask process_fn", {
                    crate::debug!(4, "Processing precomputed node data: {:?}", precomputed_node_data);
                    if precomputed_node_data.value.end {
                        let glr_active_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                        *final_mask_internal.borrow_mut() |= glr_active_tokens;
                        false
                    } else {
                        let mut num_outgoing_edges_that_lead_to_non_end_nodes = 0;
                        for (edge_terminal_opt, dest_map) in precomputed_node_data.children().iter() {
                            if edge_terminal_opt.is_none() {
                                num_outgoing_edges_that_lead_to_non_end_nodes += 1
                            } else {
                                for (child_node_trie_data, _edge_llm_tokens_bv) in dest_map.iter() {
                                    if !child_node_trie_data.as_arc().lock().unwrap().value.end {
                                        num_outgoing_edges_that_lead_to_non_end_nodes += 1;
                                        break;
                                    }
                                }
                            }
                            if num_outgoing_edges_that_lead_to_non_end_nodes >= 2 {
                                break; // No need to check further, we have at least two non-end nodes.
                            }
                        }
                        // Print GSS stats
                        disallow_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &final_mask_internal.borrow(), &mut HashMap::new());
                        Arc::make_mut(&mut glr_s.active_state.stack).fuse_predecessors(1);
                        // let stats = gather_gss_stats(&[glr_s.active_state.stack.as_ref()]);
                        // crate::debug!(3, "GSS stats for precomputed node data: {:#?}", stats);
                        let mut do_phase3 = false;
                        do_phase3 |= num_outgoing_edges_that_lead_to_non_end_nodes >= 2;
                        do_phase3 |= !matches!(LR_MODE, LRMode::LR1);
                        // do_phase3 |= true;
                        if do_phase3 {
                            // There will be a split.
                            // Let's do some work ahead of time to avoid redundant computations due to the upcoming split.
                            crate::debug!(4, "Processing non-end precomputed node data");
                            crate::debug!(4, "Active LLM tokens before phase 3: {:?}", glr_s.active_state.stack.allowed_llm_tokens());
                            glr_s.process_default_reductions();
                            crate::debug!(4, "After phase 3, active stack.stack.is_empty(): {}", glr_s.active_state.stack.is_empty());
                            // Arc::make_mut(&mut glr_s.active_state.stack).fuse_predecessors(1);
                            crate::debug!(4, "Active LLM tokens after phase 3: {:?}", glr_s.active_state.stack.allowed_llm_tokens());
                            crate::debug!(4, "Disallowing LLM tokens and pruning arc for precomputed node data: {:?}", final_mask_internal.borrow());
                            Arc::make_mut(&mut glr_s.active_state.stack).fuse_predecessors(1);
                        }
                        crate::debug!(4, "After processing precomputed node data, active stack.stack.is_empty(): {}", glr_s.active_state.stack.is_empty());
                        crate::debug!(4, "Final active LLM tokens: {:?}", glr_s.active_state.stack.allowed_llm_tokens());
                        !glr_s.active_state.stack.is_empty()
                    }
                })
            },
        );

        crate::debug!(2, "Done main part of get_mask");
        let t1 = std::time::Instant::now();
        println!("after special_map: {:>15?}", t1.duration_since(t0));

        crate::profiler::print_summary_flat();
        
        let counts = step_counts.lock().unwrap();
        if !counts.is_empty() {
            let mut sorted_counts: Vec<_> = counts.iter().collect();
            sorted_counts.sort_by_key(|&(_, count)| std::cmp::Reverse(count.total));

            let mut log_msg = String::from("get_mask step() counts:");
            for (terminal_id, count) in sorted_counts {
                let terminal_name = self.parent.parser.terminal_map.get_by_right(terminal_id)
                    .map(|s| s.to_string())
                    .unwrap_or("UNKNOWN_TERMINAL".to_string());
                log_msg.push_str(&format!("\n  - '{}': {}/{} successful", terminal_name, count.successful, count.total));
            }
            crate::debug!(3, "{}", log_msg);
        }

        crate::profiler::print_summary();
        crate::profiler::reset();

        // Log the GSSs
        crate::debug!(3, "Final GSS states after get_mask:");
        let roots: Vec<_> = self.state.values().map(|s| s.active_state.stack.clone()).collect();
        let labels: Vec<_> = self.state.keys().map(|k| format!("Tokenizer State {}", k.0)).collect();
        let config = GSSPrintConfig {
            labels: Some(&labels),
            max_nodes: 300,
            original_internal_bimap: Some(&self.parent.llm_vocab.original_to_internal_id_bimap),
            llm_token_map: Some(&self.parent.llm_vocab.llm_token_map),
            verbose: false,
        };
        print!("{}", print_gss_forest(&roots, &self.parent.parser.terminal_map, &config).0);

        let final_mask_mapped = self.parent.internal_bv_to_original(&final_mask_internal.into_inner());

        crate::debug!(2, "Done computing mask");
        let t1 = std::time::Instant::now();
        println!("get_mask took: {:>15?}", t1.duration_since(t0));

        final_mask_mapped
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) { // llm_token_id is original
        let llm_token_bytes = self.parent.llm_vocab.llm_token_map.get_by_right(&llm_token_id).unwrap();
        self.commit_bytes(llm_token_bytes);
    }

    pub fn commit_bytes(&mut self, llm_token_bytes: &[u8]) { // llm_token_id is original
        if llm_token_bytes.is_empty() {
            return;
        }

        crate::debug!(2, "Committing bytes: {:?}", String::from_utf8_lossy(llm_token_bytes));

        let mut gss_transformation_memo = HashMap::new();

        for state in self.state.values_mut() {
            reset_llm_tokens(&mut state.active_state.stack, &mut gss_transformation_memo);
        }
        gss_transformation_memo.clear();

        // Handle allowed terminals
        let mut state_map: BTreeMap<TokenizerStateID, TokenizerStateID> = BTreeMap::new();
        let mut terminals_map: BTreeMap<TokenizerStateID, TerminalBV> = BTreeMap::new();
        for (tokenizer_state_id, _state) in self.state.iter() {
            let exec_result = self.parent.tokenizer.execute_from_state(
                &llm_token_bytes,
                *tokenizer_state_id,
            );
            if let Some(new_state) = exec_result.end_state {
                state_map.insert(*tokenizer_state_id, TokenizerStateID(new_state));
            }
            let mut terminals = TerminalBV::zeros();
            for token in exec_result.matches {
                terminals.insert(token.id);
            }
            terminals_map.insert(*tokenizer_state_id, terminals);
        }

        let gss_stats_before_pruning = gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        );
        crate::debug!(3, "GSS stats before pruning disallowed terminals: {:#?}", gss_stats_before_pruning);
        for state in self.state.values_mut() {
            prune_disallowed_terminals(&mut state.active_state.stack, &terminals_map, &mut gss_transformation_memo);
        }
        gss_transformation_memo.clear();
        let gss_stats_after_pruning = gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        );
        crate::debug!(3, "GSS stats after pruning disallowed terminals: {:#?}", gss_stats_after_pruning);
        if gss_stats_after_pruning != gss_stats_before_pruning {
            crate::debug!(3, "GSS stats changed after pruning disallowed terminals.");
        } else {
            crate::debug!(3, "GSS stats did not change after pruning disallowed terminals.");
        }

        for state in self.state.values_mut() {
            map_allowed_terminals_tokenizer_states(&mut state.active_state.stack, &state_map, &mut gss_transformation_memo);
        }
        gss_transformation_memo.clear();
        // println!("State after preparation: {}", self);

        let mut new_overall_state: BTreeMap<TokenizerStateID, GLRParserState<'a>> = BTreeMap::new();

        let mut processing_queue: BTreeMap<usize, BTreeMap<TokenizerStateID, GLRParserState<'a>>> = BTreeMap::new();
        processing_queue.insert(0, std::mem::take(&mut self.state));

        while let Some((offset, states_to_process)) = processing_queue.pop_first() {
            crate::debug!(3, "Processing offset {} with states {:?}.", offset, states_to_process.keys().map(|k| k.0).collect::<Vec<_>>());
            for (tokenizer_s_id_at_offset, glr_s_at_offset) in states_to_process {
                assert!(offset < llm_token_bytes.len());

                let exec_result = self.parent.tokenizer.execute_from_state(
                    &llm_token_bytes[offset..],
                    tokenizer_s_id_at_offset,
                );

                for match_info in &exec_result.matches {
                    let mut cloned_glr_s = glr_s_at_offset.clone();

                    cloned_glr_s.step(TerminalID(match_info.id));
                    // cloned_glr_s.do_phase3();

                    if cloned_glr_s.is_ok() {
                        let new_offset = offset + match_info.width;
                        // After a grammar token is consumed, the tokenizer resets for the next segment of the LLM token.
                        let next_tokenizer_id_for_segment = self.parent.tokenizer.initial_state_id();

                        let mut disallowed_terminals = crate::datastructures::hybrid_l2_bitset::HybridL2Bitset::new();
                        if let Some(end_state_id) = exec_result.end_state {
                            let mut disallowed_terminals_for_end_state = TerminalBV::zeros();
                            // Disallow this token from being matched again immediately.
                            disallowed_terminals_for_end_state.insert(match_info.id);
                            disallowed_terminals.insert_l2_bitset(end_state_id, disallowed_terminals_for_end_state);
                        }
                        disallow_terminals_and_prune_arc(&mut cloned_glr_s.active_state.stack, &disallowed_terminals, &mut HashMap::new());

                        if new_offset == llm_token_bytes.len() {
                            // reset_allowed_terminals(&mut cloned_glr_s.active_state.stack);
                            new_overall_state.entry(next_tokenizer_id_for_segment).and_modify(|existing| existing.merge_with(cloned_glr_s.clone())).or_insert(cloned_glr_s);
                        } else {
                            processing_queue.entry(new_offset).or_default().entry(next_tokenizer_id_for_segment).and_modify(|existing| existing.merge_with(cloned_glr_s.clone())).or_insert(cloned_glr_s);
                        }
                    }
                }

                if let Some(final_tokenizer_s_id_for_llm_token_segment) = exec_result.end_state {
                    // The rest of llm_token_bytes (from offset) was consumed, tokenizer ended in this state.
                    // The glr_s_at_offset is carried over. This is a state *after* the current LLM token.
                    let final_tokenizer_state = TokenizerStateID(final_tokenizer_s_id_for_llm_token_segment);
                    new_overall_state.entry(final_tokenizer_state).and_modify(|existing| existing.merge_with(glr_s_at_offset.clone())).or_insert(glr_s_at_offset.clone());
                }
            }
        }

        self.state = new_overall_state.clone();

        for glr_parser_state in self.state.values_mut() {
            glr_parser_state.process_default_reductions();
        }

        // TODO: this shouldn't be necessary, but due to some order-dependent LLM token BV weirdness in GSS, it is necessary to ensure commit order invariance.
        for state in self.state.values_mut() {
            reset_llm_tokens(&mut state.active_state.stack, &mut gss_transformation_memo);
        }
        gss_transformation_memo.clear();

        self.state.retain(|_, glr_parser_state| glr_parser_state.is_ok());

        let mut fuse_memo = HashMap::new();
        for state in self.state.values_mut() {
            state.active_state.stack = fuse_predecessors_recursive(&mut state.active_state.stack, 3, &mut fuse_memo);
        }
        fuse_memo.clear();

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

        crate::debug!(2, "Active tokenizer states after committing text (bytes {:?}): {:?}", llm_token_bytes, self.state.keys().map(|k|k.0).collect::<Vec<_>>());
        for (tokenizer_id, glr_state) in &self.state {
            if !glr_state.active_state.stack.is_empty() { // Log only for non-empty GSS
                // glr_state.log_gss(
                //     &format!("GSS for tokenizer state {} after commit of text", tokenizer_id.0),
                //     TerminalID(0)
                // );
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
