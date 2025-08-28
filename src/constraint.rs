// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use std::sync::{Mutex, RwLock};
use std::mem;
use crate::datastructures::ordered_hash_map::Retain;
use crate::datastructures::gss::{disallow_llm_tokens_and_prune_arc, fuse_predecessors_recursive, get_roots, print_gss_forest, reset_terminals};
use crate::datastructures::gss::{map_allowed_terminals_tokenizer_states, prune_disallowed_terminals};
use ordered_hash_map::OrderedHashMap;
use ordered_hash_map::OrderedHashSet;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::ops::{BitOr, BitOrAssign};
use std::fmt::{self, Debug, Display, Formatter};
use std::sync::Arc;
use std::cell::RefCell;

use bimap::BiBTreeMap;
use bitvec::prelude::*;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::constraint_extra::{calculate_final_stats, dump_precompute_trie_recursive, print_precompute_stats, PrecomputeStats};
use crate::glr::table::Stage7ShiftsAndReducesLookaheadValue;
use crate::datastructures::gss::{allow_only_llm_tokens_and_prune_arc, disallow_terminals_and_prune_arc, gather_gss_stats, reset_llm_tokens, GSSNode, GSSPrintConfig, LLMTokenBV, PrecomputeNode2, PrecomputedNodeContents, TerminalBV};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::{ArcFreeTrie as Trie, EdgeInserter, GodWrapper, NodeId};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::finite_automata::Regex;
use crate::glr::parser::{BelowBottomReductionMode, GLRParser, GLRParserState, ParseState, ParseStateEdgeContent, ProcessDefaultReductionsAdvancedConfig, ProcessTokenAdvancedConfig};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;
use std::io::{Read, Write};
use kdam::{tqdm, BarBuilder, BarExt};
use deterministic_hash::DeterministicHasher;
use profiler_macro::{time_it, timeit};
use crate::datastructures::gss::Acc;
use crate::glr::table::StateID;
use crate::glr::analyze::compute_terminal_follow_sets;
use crate::glr::grammar::Terminal;
use std::ops::{BitAnd, Sub};
use crate::glr::items::{Item, LRMode, LR_MODE};
use crate::interface::CompiledGrammar;
use crate::profiler::{print_summary, print_summary_flat, reset, GSS_LOGGING_ENABLED, PROGRESS_BAR_ENABLED};
use crate::datastructures::entry_api::EntryApi;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::Rng;
use serde_json::Value as SerdeValue;

const MERGE_THRESHOLD: usize = 20;

// Tries are arena-based. Root references are NodeId values.
pub type PrecomputeNode =
    Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;

pub type Precomputed = BTreeMap<TokenizerStateID, NodeId>;
pub type Precomputed2 = BTreeMap<TokenizerStateID, NodeId>;

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
    pub(crate) precomputed2:     Precomputed2,
    pub(crate) llm_vocab:        Arc<LLMVocab>,
    pub(crate) token_name_map:   BiBTreeMap<Terminal, usize>,
    pub(crate) possible_matches: BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    pub(crate) trie1_god: Trie1GodWrapper,
    pub(crate) trie2_god: Trie2GodWrapper,
}

impl GrammarConstraint {
    pub fn assert_eq(&self, other: &Self) {
        assert_eq!(self.tokenizer, other.tokenizer);
        assert_eq!(self.parser, other.parser);
        assert_eq!(self.precomputed.len(), other.precomputed.len());
        for ((sid1, r1), (sid2, r2)) in self.precomputed.iter().zip(other.precomputed.iter()) {
            assert_eq!(sid1, sid2);
            assert!(self.are_roots_equal(*r1, &self.trie1_god, *r2, &other.trie1_god));
        }
        assert_eq!(self.precomputed2.len(), other.precomputed2.len());
        for ((sid1, r1), (sid2, r2)) in self.precomputed2.iter().zip(other.precomputed2.iter()) {
            assert_eq!(sid1, sid2);
            assert!(self.are_roots_equal(*r1, &self.trie2_god, *r2, &other.trie2_god));
        }
        assert_eq!(self.llm_vocab.llm_token_map, other.llm_vocab.llm_token_map);
        assert_eq!(self.token_name_map, other.token_name_map);
        assert_eq!(self.llm_vocab.max_original_llm_token_id, other.llm_vocab.max_original_llm_token_id);
        assert_eq!(self.llm_vocab.original_to_internal_id_bimap, other.llm_vocab.original_to_internal_id_bimap);
        assert_eq!(self.llm_vocab.internal_max_llm_token, other.llm_vocab.internal_max_llm_token);
        assert_eq!(self.possible_matches, other.possible_matches);
        assert_eq!(self.trie1_god, other.trie1_god);
        assert_eq!(self.trie2_god, other.trie2_god);
    }

    fn are_roots_equal<EK, EV, T>(
        &self,
        a_root: NodeId,
        a_god: &GodWrapper<EK, EV, T>,
        b_root: NodeId,
        b_god: &GodWrapper<EK, EV, T>,
    ) -> bool
    where
        EK: Ord + Clone + PartialEq + Debug,
        EV: Clone + PartialEq,
        T: PartialEq,
    {
        fn eq_rec<EK, EV, T>(
            a_god: &GodWrapper<EK, EV, T>,
            b_god: &GodWrapper<EK, EV, T>,
            a: NodeId, b: NodeId,
            cache: &mut HashMap<(NodeId, NodeId), bool>,
        ) -> bool
        where
            EK: Ord + Clone + PartialEq + Debug,
            EV: Clone + PartialEq,
            T: PartialEq,
        {
            if a == b { return true; }
            let key = if a <= b { (a, b) } else { (b, a) };
            if let Some(&v) = cache.get(&key) {
                return v;
            }
            cache.insert(key, true); // optimistic for cycles

            let a_data = a_god.with_node(a, |n| (n.value.clone(), n.max_depth, n.children.clone()));
            let b_data = b_god.with_node(b, |n| (n.value.clone(), n.max_depth, n.children.clone()));
            if a_data.0 != b_data.0 || a_data.1 != b_data.1 || a_data.2.len() != b_data.2.len() {
                cache.insert(key, false);
                return false;
            }
            for (ek, a_map) in a_data.2 {
                let Some(b_map) = b_data.2.get(&ek) else { cache.insert(key, false); return false; };
                if a_map.len() != b_map.len() {
                    cache.insert(key, false);
                    return false;
                }
                // pairwise matching ignoring ordering
                let mut b_pairs: Vec<(NodeId, EV)> = b_map.iter().map(|(&id, ev)| (id, ev.clone())).collect();
                for (a_child, a_ev) in a_map {
                    let mut found = false;
                    for i in 0..b_pairs.len() {
                        if a_ev == b_pairs[i].1 {
                            let b_child = b_pairs[i].0;
                            if eq_rec(a_god, b_god, a_child, b_child, cache) {
                                b_pairs.remove(i);
                                found = true;
                                break;
                            }
                        }
                    }
                    if !found {
                        cache.insert(key, false);
                        return false;
                    }
                }
            }
            true
        }
        let mut cache = HashMap::new();
        eq_rec(a_god, b_god, a_root, b_root, &mut cache)
    }
}

impl JSONConvertible for GrammarConstraint {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("tokenizer".to_string(), self.tokenizer.to_json());
        obj.insert("parser".to_string(), self.parser.to_json());
        // Serialize precomputed and precomputed2 as arrays of [state, graph]
        let mut precomp_arr = Vec::new();
        for (sid, root) in &self.precomputed {
            precomp_arr.push(JSONNode::Array(vec![sid.to_json(), crate::datastructures::trie::serialize_graph(&self.trie1_god, *root)]));
        }
        let mut precomp2_arr = Vec::new();
        for (sid, root) in &self.precomputed2 {
            precomp2_arr.push(JSONNode::Array(vec![sid.to_json(), crate::datastructures::trie::serialize_graph(&self.trie2_god, *root)]));
        }
        obj.insert("precomputed".to_string(), JSONNode::Array(precomp_arr));
        obj.insert("precomputed2".to_string(), JSONNode::Array(precomp2_arr));
        obj.insert("llm_token_map".to_string(), self.llm_vocab.llm_token_map.to_json());
        obj.insert("token_name_map".to_string(), self.token_name_map.to_json());
        obj.insert("max_original_llm_token_id".to_string(), self.llm_vocab.max_original_llm_token_id.to_json());
        obj.insert("original_to_internal_id_bimap".to_string(), self.llm_vocab.original_to_internal_id_bimap.to_json());
        obj.insert("internal_max_llm_token".to_string(), self.llm_vocab.internal_max_llm_token.to_json());
        obj.insert("possible_matches".to_string(), self.possible_matches.to_json());
        obj.insert("trie1_god".to_string(), self.trie1_god.to_json());
        obj.insert("trie2_god".to_string(), self.trie2_god.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let tokenizer = obj.remove("tokenizer").ok_or_else(|| "Missing field tokenizer".to_string())
                                   .and_then(Regex::from_json)?;
                let parser = obj.remove("parser").ok_or_else(|| "Missing field parser".to_string())
                                .and_then(GLRParser::from_json)?;
                let trie1_god = obj.remove("trie1_god").ok_or_else(|| "Missing field trie1_god".to_string())
                                    .and_then(|n| Trie1GodWrapper::from_json(n))?;
                let trie2_god = obj.remove("trie2_god").ok_or_else(|| "Missing field trie2_god".to_string())
                                    .and_then(|n| Trie2GodWrapper::from_json(n))?;

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

                let precomputed_json = obj.remove("precomputed").ok_or_else(|| "Missing field precomputed".to_string())?;
                let precomputed2_json = obj.remove("precomputed2").ok_or_else(|| "Missing field precomputed2".to_string())?;
                let mut precomputed: Precomputed = BTreeMap::new();
                let mut precomputed2: Precomputed2 = BTreeMap::new();
                match precomputed_json {
                    JSONNode::Array(arr) => {
                        for pair in arr {
                            match pair {
                                JSONNode::Array(p) if p.len() == 2 => {
                                    let sid = TokenizerStateID::from_json(p[0].clone())?;
                                    let root = crate::datastructures::trie::deserialize_graph(&trie1_god, p[1].clone())?;
                                    precomputed.insert(sid, root);
                                }
                                _ => return Err("precomputed entry invalid".into()),
                            }
                        }
                    }
                    _ => return Err("precomputed must be array".into()),
                }
                match precomputed2_json {
                    JSONNode::Array(arr) => {
                        for pair in arr {
                            match pair {
                                JSONNode::Array(p) if p.len() == 2 => {
                                    let sid = TokenizerStateID::from_json(p[0].clone())?;
                                    let root = crate::datastructures::trie::deserialize_graph(&trie2_god, p[1].clone())?;
                                    precomputed2.insert(sid, root);
                                }
                                _ => return Err("precomputed2 entry invalid".into()),
                            }
                        }
                    }
                    _ => return Err("precomputed2 must be array".into()),
                }

                Ok(GrammarConstraint {
                    tokenizer,
                    parser,
                    precomputed,
                    precomputed2,
                    llm_vocab: Arc::new(LLMVocab { llm_token_map, max_original_llm_token_id, original_to_internal_id_bimap, internal_max_llm_token }),
                    token_name_map,
                    possible_matches,
                    trie1_god,
                    trie2_god,
                })
            }
            _ => Err("Expected JSONNode::Object for GrammarConstraint".to_string()),
        }
    }
}

type NormalizedPath = Vec<(usize, StateID)>;
type PathMap = BTreeMap<NormalizedPath, LLMTokenBV>;

pub type Trie1GodWrapper = GodWrapper<Option<TerminalID>, HybridBitset, PrecomputedNodeContents>;
pub type Trie2GodWrapper = GodWrapper<(usize, Option<StateID>), HybridBitset, PrecomputedNodeContents>;

impl<EK, EV, T> PartialEq for GodWrapper<EK, EV, T> where EK: PartialEq, EV: PartialEq, T: PartialEq {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl<EK, EV, T> Eq for GodWrapper<EK, EV, T> where EK: Eq, EV: Eq, T: Eq {}
impl<EK, EV, T> PartialOrd for GodWrapper<EK, EV, T> where EK: PartialOrd, EV: PartialOrd, T: PartialOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if Arc::ptr_eq(&self.0, &other.0) { Some(Ordering::Equal) } else { None }
    }
}
impl<EK, EV, T> Ord for GodWrapper<EK, EV, T> where EK: Ord, EV: Ord, T: Ord {
    fn cmp(&self, other: &Self) -> Ordering {
        if Arc::ptr_eq(&self.0, &other.0) { Ordering::Equal } else { Ordering::Less }
    }
}
impl<EK, EV, T> Hash for GodWrapper<EK, EV, T> where EK: Hash, EV: Hash, T: Hash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.0) as usize).hash(state);
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

        crate::debug!(2, "Building vocab prefix tree for possible_matches computation");
        let vocab_for_possible_matches = VocabPrefixTree::build(&internal_tokens_for_vocab);
        crate::debug!(2, "Done building vocab prefix tree for possible_matches computation");

        let mut computed_possible_matches = BTreeMap::new();
        // Cache for the possible_matches computation
        let mut pm_cache: HashMap<(*const VocabPrefixTreeNode, TokenizerStateID), BTreeMap<GrammarTokenID, LLMTokenBV>> = HashMap::new();

        crate::debug!(2, "Computing possible_matches for all {} tokenizer states", tokenizer.iter_states().count());
        for sid in tokenizer.iter_states() {
            let matches_for_sid = Self::compute_possible_matches_for_vocab_node(
                &tokenizer,
                &vocab_for_possible_matches.root,
                sid,
                &mut pm_cache,
            );
            computed_possible_matches.insert(sid, matches_for_sid);
        }
        crate::debug!(2, "Finished computing possible_matches");

        let grammar_productions = &parser.productions;
        let grammar_term_map = &parser.terminal_map;

        let terminal_follow_sets_named = compute_terminal_follow_sets(grammar_productions);
        crate::debug!(5, "terminal_follow_sets_named:");
        for (terminal, following_terminals) in &terminal_follow_sets_named {
            crate::debug!(4, "{} -> {}", terminal, following_terminals.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "));
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

        let llm_vocab = Arc::new(LLMVocab {
            llm_token_map,
            max_original_llm_token_id,
            original_to_internal_id_bimap,
            internal_max_llm_token,
        });

        let trie1_god = Trie1GodWrapper::new();
        let trie2_god = Trie2GodWrapper::new();

        let precomputed = Self::precompute(
            &tokenizer,
            Some(&parser),
            Some(llm_vocab.clone()),
            &internal_llm_token_map_for_precompute,
            &token_name_map,
            internal_max_llm_token,
            &terminal_follow_map,
            parser.ignore_terminal_id,
            &mut computed_possible_matches,
            &trie1_god,
        );

        Self::_dump_precomputed(
            &precomputed,
            &llm_vocab.original_to_internal_id_bimap,
            &token_name_map,
            &llm_vocab.llm_token_map,
            &trie1_god,
        );

        let precomputed2 = Self::precompute2(
            &precomputed,
            &tokenizer,
            Some(&parser),
            Some(llm_vocab.clone()),
            &internal_llm_token_map_for_precompute,
            &token_name_map,
            internal_max_llm_token,
            &terminal_follow_map,
            parser.ignore_terminal_id,
            &mut computed_possible_matches,
            &trie1_god,
            &trie2_god,
        );

        let mut stats2 = PrecomputeStats::default();
        crate::constraint_extra::calculate_final_stats2(&precomputed2, &mut stats2, &trie2_god);
        crate::constraint_extra::print_precompute_stats2(&stats2);

        Self::_dump_precomputed2(
            &precomputed2,
            &llm_vocab.original_to_internal_id_bimap,
            &llm_vocab.llm_token_map,
            &trie2_god,
        );

        let mut gc = Self {
            tokenizer,
            parser,
            precomputed,
            precomputed2,
            llm_vocab,
            token_name_map,
            possible_matches: computed_possible_matches,
            trie1_god,
            trie2_god,
        };

        gc
    }

    pub fn precompute(
        tokenizer:        &Regex,
        parser:           Option<&GLRParser>,
        llm_vocab:        Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        token_name_map:   &BiBTreeMap<Terminal, usize>,
        internal_max_llm_token: usize,
        terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        ignore_terminal_id: Option<TerminalID>,
        possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
        trie1_god: &Trie1GodWrapper,
    ) -> BTreeMap<TokenizerStateID, NodeId> {
        let mut helper = Precomputer::new(
            tokenizer,
            parser,
            llm_vocab,
            internal_llm_token_map,
            internal_max_llm_token,
            MERGE_THRESHOLD,
            terminal_follow_map,
            ignore_terminal_id,
            trie1_god.clone(),
        );

        helper.run_dfs();
        helper.replace_ignore_token_edges_with_none_edges();
        helper.simplify_none_edges();

        let roots_for_recompute: Vec<_> = helper.roots.values().copied().collect();
        trie1_god.recompute_all_max_depths(&roots_for_recompute);

        helper.prune_dead_paths();
        helper.prune_on_no_terminal_follow();
        helper.prune_dead_paths();
        helper.factor_common_destinations();
        helper.merge_nodes();

        helper.finish(token_name_map, possible_matches, internal_max_llm_token)
    }

    pub fn precompute2(
        precomputed: &BTreeMap<TokenizerStateID, NodeId>,
        tokenizer:        &Regex,
        parser:           Option<&GLRParser>,
        llm_vocab:        Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        token_name_map:   &BiBTreeMap<Terminal, usize>,
        internal_max_llm_token: usize,
        terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        ignore_terminal_id: Option<TerminalID>,
        possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
        trie1_god: &Trie1GodWrapper,
        trie2_god: &Trie2GodWrapper,
    ) -> Precomputed2 {
        crate::debug!(2, "Precomputing Trie 2...");
        const BELOW_BOTTOM_REDUCE_MODE__CONTINUE_FROM_EVERYTHING: bool = false;
        const BELOW_BOTTOM_REDUCE_MODE: BelowBottomReductionMode = if BELOW_BOTTOM_REDUCE_MODE__CONTINUE_FROM_EVERYTHING {
            BelowBottomReductionMode::ContinueFromEverything
        } else {
            BelowBottomReductionMode::ContinueFromAll
        };

        let mut precomputed2 = BTreeMap::new();
        let mut initial_values_for_map: Vec<(NodeId, GLRParserState)> = Vec::new();
        let parser = parser.unwrap();

        // Single base Trie2 root in the arena
        let base_trie2_root = trie2_god.create(PrecomputedNodeContents::root(internal_max_llm_token));
        let base_gss_nodes: Vec<Arc<GSSNode>> = {
            let mut out = Vec::new();
            if BELOW_BOTTOM_REDUCE_MODE__CONTINUE_FROM_EVERYTHING {
                let mut acc = Acc::new_fresh();
                acc.trie2_nodes.insert(base_trie2_root);
                let gss_leaf = Arc::new(GSSNode::new(acc));
                out.push(Arc::new(
                    gss_leaf.push(ParseStateEdgeContent { state_id: parser.everything_state_id })
                ));
            } else {
                for state_id in parser.table.keys() {
                    let mut acc = Acc::new_fresh();
                    acc.trie2_nodes.insert(base_trie2_root);
                    let gss_leaf = Arc::new(GSSNode::new(acc));
                    out.push(Arc::new(gss_leaf.push(ParseStateEdgeContent { state_id: *state_id })));
                }
            }
            out
        };

        // Merge the base per-state initial nodes into one GSS and build a GLR state from it.
        let base_gss_merged = GSSNode::merge_many_with_depth(usize::MAX, base_gss_nodes);
        let mut base_glr_state = parser.init_glr_parser_from_stack(base_gss_merged).with_god(trie2_god.clone());

        // Optional: pre-warm once with default reductions (disabled)
        if false {
            base_glr_state.process_default_reductions_advanced(&ProcessDefaultReductionsAdvancedConfig {
                fuel: None,
                per_state_fuel: None,
                below_bottom_mode: BELOW_BOTTOM_REDUCE_MODE,
            });
        }

        #[cfg(not(rustrover))]
        let it = tqdm!(precomputed.iter(), desc = "Precomputing Trie 2", disable = !PROGRESS_BAR_ENABLED, leave=false);
        #[cfg(rustrover)]
        let it = precomputed.iter();
        for (tokenizer_state_id, trie1_root) in it {
            // Clone Trie2: since we are in an arena, we can clone by BFS renumbering into same arena
            let cloned_trie2_root = clone_trie2_graph_arena(trie2_god, base_trie2_root);

            // Deep clone the base GSS, remapping trie2_nodes (already using NodeIds)
            let cloned_gss = crate::datastructures::gss::deep_clone_gss_with_trie2_map_arena(
                &base_glr_state.active_state.stack,
                base_trie2_root,
                cloned_trie2_root,
            );
            let mut glr_state_for_sid = base_glr_state.clone();
            glr_state_for_sid.active_state.stack = cloned_gss;

            precomputed2.insert(*tokenizer_state_id, cloned_trie2_root);
            initial_values_for_map.push((*trie1_root, glr_state_for_sid));
        }

        let trie2_end = trie2_god.create(PrecomputedNodeContents::leaf());

        crate::debug!(2, "Running special_map_grouped for Trie 2 precomputation");
        Trie::special_map_grouped(
            trie1_god,
            initial_values_for_map,
            |current_glr_state, edge_grammar_token_opt, destinations_map| {
                crate::debug!(3, "Trie2: Processing GLR state with {} destinations for edge grammar token: {:?}", destinations_map.len(), edge_grammar_token_opt);
                let mut glr_s = current_glr_state.clone();

                let mut edge_bv = LLMTokenBV::zeros();
                for bv in destinations_map.values() {
                    edge_bv |= bv;
                }
                allow_only_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &edge_bv, &mut HashMap::new());

                if let Some(gt) = edge_grammar_token_opt {
                    glr_s.process_token_advanced(*gt, &ProcessTokenAdvancedConfig { below_bottom_mode: BELOW_BOTTOM_REDUCE_MODE });
                }

                let mut out = Vec::new();
                for (&dst_node_id, edge_bv) in destinations_map.iter() {
                    let mut glr_s_copy = glr_s.clone();
                    crate::debug!(5, "Trie2: Restricting GLR state to edge bitset");
                    allow_only_llm_tokens_and_prune_arc(
                        &mut glr_s_copy.active_state.stack,
                        edge_bv,
                        &mut HashMap::new(),
                    );
                    glr_s_copy.log_gss(
                        "Trie2: After restricting GLR state to edge bitset",
                        TerminalID(0),
                        false,
                        false,
                    );
                    out.push((dst_node_id, glr_s_copy));
                }
                out
            },
            |glr_s1, glr_s2| {
                crate::debug!(4, "Trie2: Merging GLR states");
                glr_s1.log_gss("Before merge...", TerminalID(0), false, false);
                glr_s2.log_gss("...with", TerminalID(0), false, false);
                glr_s1.merge_with(glr_s2);
                glr_s1.log_gss("After merge", TerminalID(0), false, false);
            },
            |precomputed_node_data, glr_s| {
                crate::debug!(3, "Trie2: Processing node");
                crate::datastructures::gss::merge_trie2_nodes_if_needed_arena(
                    &mut glr_s.active_state.stack,
                    &mut HashMap::new(),
                    glr_s.active_state.god.as_ref().unwrap(),
                );
                let keep_going = glr_s.is_ok();
                if precomputed_node_data.value.end {
                    crate::debug!(3, "Trie2: Found end state for GLR state");
                    glr_s.log_gss("Trie2: Found end state for GLR state", TerminalID(0), false, false);
                    let mut dest_agg: BTreeMap<NodeId, LLMTokenBV> = BTreeMap::new();

                    for (last_edge, gss_root_accs) in get_roots([glr_s.active_state.stack.as_ref()]) {
                        for gss_root_acc in gss_root_accs {
                            let active_llm_tokens_for_root = gss_root_acc.union_llm_tokens();
                            for src_id in gss_root_acc.trie2_nodes.iter() {
                                let src_live = trie2_god.with_node(*src_id, |n| n.value.live_tokens.clone());
                                let tokens_to_push = &active_llm_tokens_for_root & &src_live;
                                if tokens_to_push.is_empty() {
                                    continue;
                                }
                                trie2_god.with_node_mut(*src_id, |n| n.value.live_tokens |= tokens_to_push.clone());

                                let edge_key = (0, Some(last_edge.state_id));
                                let mut inserter = EdgeInserter::new(
                                    &glr_s.active_state.god.as_ref().unwrap(),
                                    *src_id,
                                    edge_key,
                                    tokens_to_push.clone(),
                                    |ev, t| *ev &= &t.live_tokens,
                                    |e, n| *e |= n,
                                    |node_value, edge_value| node_value.live_tokens |= edge_value,
                                );
                                inserter = inserter.try_destination(trie2_end);
                                let final_dest_id = inserter.clone_into_option().expect("Failed to insert end edge");
                                dest_agg.entry(final_dest_id).and_modify(|bv| *bv |= &tokens_to_push).or_insert(tokens_to_push.clone());
                            }
                        }
                    }
                    for (dst_id, added) in &dest_agg {
                        trie2_god.with_node_mut(*dst_id, |n| n.value.live_tokens |= added.clone());
                    }
                }

                if false {
                    let mut allowed_terminals = TerminalBV::zeros();
                    // ...
                }

                keep_going
            },
        );

        crate::debug!(2, "Finished precomputing Trie 2");

        // pin nodes
        let roots_before_cleanup: Vec<_> = precomputed2.values().copied().collect();
        let _all_nodes_pinner = trie2_god.all_nodes(&roots_before_cleanup);

        // Clean up after rewiring
        prune_dead_paths_trie2(&mut precomputed2, trie2_god);
        merge_nodes_trie2(&mut precomputed2, trie2_god);
        simplify_trie2_factor_common_destinations(&mut precomputed2, trie2_god);

        let roots2_final: Vec<_> = precomputed2.values().copied().collect();
        trie2_god.recompute_all_max_depths(&roots2_final);

        precomputed2
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

pub fn prune_dead_paths_trie2(roots: &mut BTreeMap<TokenizerStateID, NodeId>, god: &Trie2GodWrapper) {
    crate::debug!(2, "Pruning dead paths from precomputed trie 2.");

    let all_nodes = god.all_nodes(&roots.values().copied().collect::<Vec<_>>());
    let mut predecessors: HashMap<NodeId, Vec<(NodeId, LLMTokenBV)>> = HashMap::new();
    let mut worklist = VecDeque::new();
    let mut live: HashMap<NodeId, LLMTokenBV> = HashMap::new();

    for &id in &all_nodes {
        live.insert(id, LLMTokenBV::zeros());
        god.with_node(id, |n| {
            if n.value.end {
                let initial_live = n.value.live_tokens.clone();
                if !initial_live.is_empty() {
                    live.insert(id, initial_live);
                    worklist.push_back(id);
                }
            }
            for dest_map in n.children.values() {
                for (&child, bv) in dest_map {
                    predecessors.entry(child).or_default().push((id, bv.clone()));
                }
            }
        });
    }

    while let Some(id) = worklist.pop_front() {
        let live_at_node = live[&id].clone();
        if let Some(preds) = predecessors.get(&id) {
            for &(pred, ref edge_bv) in preds {
                let live_from_edge = &live_at_node & edge_bv;
                if live_from_edge.is_empty() { continue; }
                let e = live.get_mut(&pred).unwrap();
                let old_len = e.len();
                *e |= &live_from_edge;
                if e.len() > old_len {
                    worklist.push_back(pred);
                }
            }
        }
    }

    for &id in &all_nodes {
        let this_live = live[&id].clone();
        god.with_node_mut(id, |n| {
            n.children.retain(|_ek, dest_map| {
                dest_map.retain(|child, edge_bv| {
                    let child_live = &live[child];
                    let keep = edge_bv & child_live;
                    if keep.is_empty() {
                        false
                    } else {
                        *edge_bv = keep;
                        true
                    }
                });
                !dest_map.is_empty()
            });
            n.value.live_tokens = this_live.clone();
        });
    }

    crate::debug!(2, "Finished pruning dead paths from trie 2.");
}

pub fn simplify_trie2_factor_common_destinations(
    roots: &mut BTreeMap<TokenizerStateID, NodeId>,
    god: &Trie2GodWrapper,
) {
    crate::debug!(2, "Simplifying trie 2 by factoring common destinations.");

    const MIN_INCOMING_EDGES_FOR_FACTORING: usize = 3;

    let roots_vec: Vec<_> = roots.values().copied().collect();
    let all_nodes = god.all_nodes(&roots_vec);

    type EdgeKey2 = (usize, Option<StateID>);
    let mut incoming_map: HashMap<NodeId, HashMap<EdgeKey2, Vec<(NodeId, LLMTokenBV)>>> = HashMap::new();

    for &src in &all_nodes {
        god.with_node(src, |n| {
            for (ek, dest_map) in &n.children {
                for (&dst, bv) in dest_map {
                    incoming_map.entry(dst).or_default().entry(ek.clone()).or_default().push((src, bv.clone()));
                }
            }
        });
    }

    for (dst, edges_by_key) in incoming_map {
        for (edge_key, sources) in edges_by_key {
            if sources.len() >= MIN_INCOMING_EDGES_FOR_FACTORING {
                let intermediate_node = god.create(PrecomputedNodeContents::internal());

                let mut union_bv = LLMTokenBV::zeros();
                for (_, bv) in &sources { union_bv |= bv; }

                {
                    let mut opt_bv = Some(union_bv.clone());
                    god.try_insert_unchecked(intermediate_node, edge_key.clone(), &mut opt_bv, dst).expect("No cycle expected");
                    god.with_node_mut(intermediate_node, |n| n.value.live_tokens |= &union_bv);
                }

                for (src, bv) in &sources {
                    god.with_node_mut(*src, |n| {
                        if let Some(m) = n.children.get_mut(&edge_key) {
                            m.remove(&dst);
                            if m.is_empty() {
                                n.children.remove(&edge_key);
                            }
                        }
                    });
                    let mut ev = Some(bv.clone());
                    god.try_insert_unchecked(*src, (0, None), &mut ev, intermediate_node).ok();
                    god.with_node_mut(*src, |n| n.value.live_tokens |= bv.clone());
                }
            }
        }
    }
    crate::debug!(2, "Finished factoring common destinations in trie 2.");
}

pub fn optimize_trie2_size(
    roots: &mut BTreeMap<TokenizerStateID, NodeId>,
    god: Trie2GodWrapper,
) {
    crate::debug!(2, "Optimizing Trie 2 size...");
    let roots_vec: Vec<_> = roots.values().copied().collect();
    let _pin = god.all_nodes(&roots_vec);

    prune_dead_paths_trie2(roots, &god);
    merge_nodes_trie2(roots, &god);
    simplify_trie2_factor_common_destinations(roots, &god);
    compress_trie2_edges(roots, &god);
    prune_dead_paths_trie2(roots, &god);
    merge_nodes_trie2(roots, &god);
    let final_roots: Vec<_> = roots.values().copied().collect();
    god.recompute_all_max_depths(&final_roots);
}

fn clone_trie2_graph_arena(
    god: &Trie2GodWrapper,
    root: NodeId,
) -> NodeId {
    let mut old_to_new: HashMap<NodeId, NodeId> = HashMap::new();
    let mut q: VecDeque<NodeId> = VecDeque::new();
    let new_root = god.create(god.with_node(root, |n| n.value.clone()));
    old_to_new.insert(root, new_root);
    q.push_back(root);
    while let Some(old) = q.pop_front() {
        let new = old_to_new[&old];
        let snapshot: Vec<((usize, Option<StateID>), Vec<(NodeId, LLMTokenBV)>)> = god.with_node(old, |n| {
            n.children.iter().map(|(ek, m)| (ek.clone(), m.iter().map(|(&cid, ev)| (cid, ev.clone())).collect())).collect()
        });
        for (ek, entries) in snapshot {
            for (child_old, ev) in entries {
                let child_new = if let Some(id) = old_to_new.get(&child_old).copied() {
                    id
                } else {
                    let new_id = god.create(god.with_node(child_old, |c| c.value.clone()));
                    old_to_new.insert(child_old, new_id);
                    q.push_back(child_old);
                    new_id
                };
                let mut opt = Some(ev);
                god.try_insert_unchecked(new, ek.clone(), &mut opt, child_new).ok();
            }
        }
    }
    // recompute depths
    god.recompute_all_max_depths(&[new_root]);
    new_root
}

pub fn merge_nodes_trie2(roots: &mut BTreeMap<TokenizerStateID, NodeId>, god: &Trie2GodWrapper) {
    crate::debug!(2, "Merging identical subtrees in precomputed trie 2.");

    let roots_vec: Vec<_> = roots.values().copied().collect();
    let all_nodes = god.all_nodes(&roots_vec);

    let pb = ProgressBar::new(all_nodes.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta})")
            .expect("progress-bar"),
    );
    if !PROGRESS_BAR_ENABLED {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    let mut canonical_buckets: HashMap<u64, Vec<NodeId>> = HashMap::new();
    let mut visited: HashMap<NodeId, NodeId> = HashMap::new();
    let mut shape_hash_memo: HashMap<NodeId, u64> = HashMap::new();
    let mut shape_eq_cache: HashMap<(NodeId, NodeId), bool> = HashMap::new();

    fn trie2_skeleton_hash(
        god: &Trie2GodWrapper,
        id: NodeId,
        memo: &mut HashMap<NodeId, u64>,
    ) -> u64 {
        const MAX_DEPTH: usize = 64;
        fn inner(
            god: &Trie2GodWrapper,
            id: NodeId,
            memo: &mut HashMap<NodeId, u64>,
            visiting: &mut HashSet<NodeId>,
            depth_left: usize,
        ) -> u64 {
            if let Some(&h) = memo.get(&id) { return h; }
            if depth_left == 0 || !visiting.insert(id) {
                let mut h = DeterministicHasher::new(std::collections::hash_map::DefaultHasher::new());
                (id as usize).hash(&mut h);
                god.with_node(id, |n| n.value.end.hash(&mut h));
                let out = h.finish();
                memo.insert(id, out);
                return out;
            }
            let children_hashes: Vec<u64> = god.with_node(id, |n| {
                let mut v = Vec::new();
                for (ek, m) in &n.children {
                    for (&child, _ev) in m {
                        let mut hh = DeterministicHasher::new(std::collections::hash_map::DefaultHasher::new());
                        let (k, sid) = ek;
                        k.hash(&mut hh);
                        sid.is_some().hash(&mut hh);
                        let ch = inner(god, child, memo, visiting, depth_left - 1);
                        ch.hash(&mut hh);
                        v.push(hh.finish());
                    }
                }
                v
            });
            let mut hasher = DeterministicHasher::new(std::collections::hash_map::DefaultHasher::new());
            god.with_node(id, |n| n.value.end.hash(&mut hasher));
            let mut chs = children_hashes;
            chs.sort_unstable();
            for h in chs {
                h.hash(&mut hasher);
            }
            let out = hasher.finish();
            visiting.remove(&id);
            memo.insert(id, out);
            out
        }
        let mut visiting = HashSet::new();
        inner(god, id, memo, &mut visiting, MAX_DEPTH)
    }

    fn trie2_shape_eq(
        god: &Trie2GodWrapper,
        a: NodeId, b: NodeId,
        cache: &mut HashMap<(NodeId, NodeId), bool>,
    ) -> bool {
        if a == b { return true; }
        let key = if a <= b { (a, b) } else { (b, a) };
        if let Some(&res) = cache.get(&key) { return res; }
        cache.insert(key, true);

        let (a_end, a_children): (bool, BTreeMap<(usize, Option<StateID>), Vec<(NodeId, LLMTokenBV)>>) = god.with_node(a, |n| {
            let mut map = BTreeMap::new();
            for (ek, m) in &n.children {
                map.insert(ek.clone(), m.iter().map(|(&id, ev)| (id, ev.clone())).collect());
            }
            (n.value.end, map)
        });
        let (b_end, b_children): (bool, BTreeMap<(usize, Option<StateID>), Vec<(NodeId, LLMTokenBV)>>) = god.with_node(b, |n| {
            let mut map = BTreeMap::new();
            for (ek, m) in &n.children {
                map.insert(ek.clone(), m.iter().map(|(&id, ev)| (id, ev.clone())).collect());
            }
            (n.value.end, map)
        });

        if a_end != b_end || a_children.len() != b_children.len() {
            cache.insert(key, false);
            return false;
        }

        for (ek, a_list) in a_children {
            let Some(b_list0) = b_children.get(&ek) else { cache.insert(key, false); return false; };
            let mut b_list = b_list0.clone();
            for (a_child, a_ev) in a_list {
                let mut found = false;
                for i in 0..b_list.len() {
                    if a_ev == b_list[i].1 {
                        if trie2_shape_eq(god, a_child, b_list[i].0, cache) {
                            b_list.remove(i); found = true; break;
                        }
                    }
                }
                if !found {
                    cache.insert(key, false);
                    return false;
                }
            }
        }
        true
    }

    fn dedup_rec(
        god: &Trie2GodWrapper,
        id: NodeId,
        canonical_buckets: &mut HashMap<u64, Vec<NodeId>>,
        visited: &mut HashMap<NodeId, NodeId>,
        shape_hash_memo: &mut HashMap<NodeId, u64>,
        shape_eq_cache: &mut HashMap<(NodeId, NodeId), bool>,
        pb: &ProgressBar,
    ) -> NodeId {
        if let Some(&c) = visited.get(&id) { return c; }
        visited.insert(id, id);
        pb.inc(1);

        // canonicalize children
        let children_snapshot: Vec<((usize, Option<StateID>), Vec<NodeId>)> = god.with_node(id, |n| {
            n.children.iter().map(|(ek, m)| (ek.clone(), m.keys().copied().collect())).collect()
        });
        for (ek, kids) in children_snapshot {
            for child in kids {
                let canon = dedup_rec(god, child, canonical_buckets, visited, shape_hash_memo, shape_eq_cache, pb);
                if canon != child {
                    god.with_node_mut(id, |n| {
                        if let Some(m) = n.children.get_mut(&ek) {
                            if let Some(ev) = m.remove(&child) {
                                m.insert(canon, ev);
                            }
                        }
                    });
                }
            }
        }

        let fp = trie2_skeleton_hash(god, id, shape_hash_memo);
        let bucket = canonical_buckets.entry(fp).or_default();
        for &cand in bucket.iter() {
            if trie2_shape_eq(god, id, cand, shape_eq_cache) {
                // merge liveness
                let live = god.with_node(id, |n| n.value.live_tokens.clone());
                if !live.is_empty() {
                    god.with_node_mut(cand, |n| n.value.live_tokens |= live);
                }
                visited.insert(id, cand);
                return cand;
            }
        }
        bucket.push(id);
        id
    }

    let mut new_roots = BTreeMap::new();
    for (sid, &root) in roots.iter() {
        let canon = dedup_rec(god, root, &mut canonical_buckets, &mut visited, &mut shape_hash_memo, &mut shape_eq_cache, &pb);
        new_roots.insert(*sid, canon);
    }
    *roots = new_roots;

    let final_roots_vec: Vec<_> = roots.values().copied().collect();
    god.recompute_all_max_depths(&final_roots_vec);

    pb.finish_with_message("Finished merging Trie 2 nodes");
    crate::debug!(2, "Finished merging subtrees in trie 2.");
}

/// Compress linear chains in Trie2 by merging consecutive edges where safe.
/// Same conditions as before, adapted to arena.
pub fn compress_trie2_edges(
    roots: &mut BTreeMap<TokenizerStateID, NodeId>,
    god: &Trie2GodWrapper,
) {
    crate::debug!(2, "Compressing Trie 2 by merging linear chains...");
    type EdgeKey2 = (usize, Option<StateID>);

    let roots_vec: Vec<_> = roots.values().copied().collect();
    let mut changed = true;
    let mut iterations = 0usize;

    while changed {
        iterations += 1;
        changed = false;
        let all_nodes = god.all_nodes(&roots_vec);

        let mut incoming_count: HashMap<NodeId, usize> = HashMap::new();
        for &src in &all_nodes {
            god.with_node(src, |n| {
                for m in n.children.values() {
                    for (&child, _) in m {
                        *incoming_count.entry(child).or_insert(0) += 1;
                    }
                }
            });
        }

        'src_loop: for &src in &all_nodes {
            // snapshot children
            let children_snapshot: Vec<(EdgeKey2, Vec<(NodeId, LLMTokenBV)>)> = god.with_node(src, |n| {
                n.children.iter().map(|(ek, m)| (ek.clone(), m.iter().map(|(&id, ev)| (id, ev.clone())).collect())).collect()
            });
            for (ek1, entries) in children_snapshot {
                for (child, bv1) in entries {
                    let indeg = incoming_count.get(&child).copied().unwrap_or(0);
                    if indeg != 1 { continue; }
                    let (is_end, child_outgoing): (bool, Vec<(EdgeKey2, Vec<(NodeId, LLMTokenBV)>)>) = god.with_node(child, |n| {
                        let co: Vec<(EdgeKey2, Vec<(NodeId, LLMTokenBV)>)> = n.children.iter().map(|(ek, m)| (ek.clone(), m.iter().map(|(&id, ev)| (id, ev.clone())).collect())).collect();
                        (n.value.end, co)
                    });
                    if is_end { continue; }
                    if child_outgoing.len() != 1 { continue; }
                    let (ek2, d2) = &child_outgoing[0];
                    if d2.len() != 1 { continue; }
                    let (grand, bv2) = d2[0].clone();
                    let s1 = ek1.1;
                    if s1.is_some() { continue; } // only allow merging when first edge has None state
                    let merged_k = ek1.0 + ek2.0;
                    let merged_sid = s1.or(ek2.1);
                    let merged_key: EdgeKey2 = (merged_k, merged_sid);
                    let merged_bv = &bv1 & &bv2;
                    if merged_bv.is_empty() { continue; }

                    // Reduce/remove src --ek1--> child
                    god.with_node_mut(src, |n| {
                        if let Some(m) = n.children.get_mut(&ek1) {
                            if let Some(ev) = m.get_mut(&child) {
                                *ev -= &merged_bv;
                                if ev.is_empty() {
                                    m.remove(&child);
                                }
                            }
                            if m.is_empty() { n.children.remove(&ek1); }
                        }
                    });

                    // Add/merge src --merged_key--> grand
                    let mut inserter = EdgeInserter::new(
                        god,
                        src,
                        merged_key.clone(),
                        merged_bv.clone(),
                        |ev, t| *ev &= &t.live_tokens,
                        |e, n| *e |= n,
                        |node_value, edge_value| node_value.live_tokens |= edge_value,
                    );
                    let _ = inserter.try_destination(grand).into_option();

                    changed = true;
                    continue 'src_loop;
                }
            }
        }

        if changed {
            prune_dead_paths_trie2(roots, god);
            merge_nodes_trie2(roots, god);
        }
    }
    crate::debug!(2, "Finished compressing Trie 2 in {} iteration(s).", iterations);
}

pub fn clone_trie2_graph(
    root: &Arc<RwLock<PrecomputeNode2>>,
) -> (
    Arc<RwLock<PrecomputeNode2>>,
    HashMap<*const RwLock<PrecomputeNode2>, Arc<RwLock<PrecomputeNode2>>>,
) {
    // This function is kept for compatibility if called elsewhere;
    // In the arena-based implementation we use clone_trie2_graph_arena.
    let guard = root.read().unwrap();
    let cloned = Arc::new(RwLock::new(PrecomputeNode2::new(guard.value.clone())));
    drop(guard);
    let mut map = HashMap::new();
    map.insert(Arc::as_ptr(root), cloned.clone());
    (cloned, map)
}

struct Precomputer<'r> {
    tokenizer:        &'r Regex,
    parser:           Option<&'r GLRParser>,
    llm_vocab:        Option<Arc<LLMVocab>>,
    vocab:            VocabPrefixTree,
    roots:            BTreeMap<TokenizerStateID, NodeId>,
    possible_matches: RefCell<BTreeMap<*const VocabPrefixTreeNode, BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>>>,
    all_llm_tokens:   HybridBitset,
    merge_threshold:  usize,
    pb:               ProgressBar,
    stats: PrecomputeStats,
    terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    ignore_terminal_id: Option<TerminalID>,
    end_node:         NodeId,
    trie1_god:        Trie1GodWrapper,
}

impl<'r> Precomputer<'r> {
    fn new(
        tokenizer:        &'r Regex,
        parser:           Option<&'r GLRParser>,
        llm_vocab:        Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        internal_max_llm_token: usize,
        merge_threshold:  usize,
        terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        ignore_terminal_id: Option<TerminalID>,
        trie1_god: Trie1GodWrapper,
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
            let root_id = trie1_god.create(PrecomputedNodeContents::root(internal_max_llm_token));
            roots.insert(sid, root_id);
        }

        let total_nodes = count_vocab_nodes(&vocab.root);
        let pb = ProgressBar::new(total_nodes);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] \
                           [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta})")
                .expect("progress-bar"),
        );
        if !PROGRESS_BAR_ENABLED {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }

        let end_node = trie1_god.create(PrecomputedNodeContents::leaf());

        Self {
            tokenizer,
            parser,
            llm_vocab,
            vocab,
            roots,
            possible_matches: RefCell::new(BTreeMap::new()),
            all_llm_tokens: HybridBitset::max_ones(),
            merge_threshold,
            pb,
            stats: PrecomputeStats::default(),
            terminal_follow_map,
            ignore_terminal_id,
            end_node,
            trie1_god,
        }
    }

    fn run_dfs(&mut self) {
        let mut assoc: BTreeMap<TokenizerStateID, OrderedHashSet<NodeId>> = BTreeMap::new();
        for (sid, id) in &self.roots {
            assoc.entry(*sid).or_default().insert(*id);
        }
        crate::debug!(2, "Starting precompute DFS");
        self.dfs(&self.vocab.root, assoc);
        crate::debug!(2, "Finished precompute DFS");
        self.pb.finish_with_message("Precomputation complete");
        crate::debug!(2, "Precomputation complete");
    }

    fn replace_ignore_token_edges_with_none_edges(&mut self) {
        let ignore_tid = if let Some(id) = self.ignore_terminal_id {
            id
        } else {
            return;
        };
        crate::debug!(2, "Replacing ignore token edges with None edges...");
        let roots_vec: Vec<_> = self.roots.values().copied().collect();
        let all_nodes = self.trie1_god.all_nodes(&roots_vec);
        for id in all_nodes {
            self.trie1_god.with_node_mut(id, |n| {
                let ignore_key = Some(ignore_tid);
                if let Some(map) = n.children.remove(&ignore_key) {
                    let dest_map_for_none = n.children.entry(None).or_default();
                    for (dst, bv) in map {
                        dest_map_for_none.entry(dst).and_modify(|e| *e |= &bv).or_insert(bv);
                    }
                }
            });
        }
        crate::debug!(2, "Done replacing ignore token edges.");
    }

    fn simplify_none_edges(&mut self) {
        crate::debug!(2, "Simplifying None edges (shortcut predecessors to successors)...");
        let root_ptrs: HashSet<NodeId> = self.roots.values().copied().collect();
        let roots_vec: Vec<_> = self.roots.values().copied().collect();
        let all_nodes = self.trie1_god.all_nodes(&roots_vec);

        let mut incoming: HashMap<NodeId, Vec<(NodeId, Option<GrammarTokenID>, LLMTokenBV)>> = HashMap::new();
        let mut none_edges_from: HashMap<NodeId, Vec<(NodeId, LLMTokenBV)>> = HashMap::new();
        let mut none_union: HashMap<NodeId, LLMTokenBV> = HashMap::new();

        for &src in &all_nodes {
            self.trie1_god.with_node(src, |n| {
                for (ek_opt, dest_map) in &n.children {
                    for (&child, ev) in dest_map {
                        incoming.entry(child).or_default().push((src, *ek_opt, ev.clone()));
                    }
                }
                if let Some(dest_map) = n.children.get(&None) {
                    let list = none_edges_from.entry(src).or_default();
                    for (&child, ev) in dest_map {
                        list.push((child, ev.clone()));
                        let e = none_union.entry(src).or_insert_with(LLMTokenBV::zeros);
                        *e |= ev;
                    }
                }
            });
        }

        for (b, none_edges) in none_edges_from {
            let union_mask = match none_union.get(&b) {
                Some(bv) if !bv.is_empty() => bv.clone(),
                _ => continue,
            };
            let in_edges = incoming.get(&b).cloned().unwrap_or_default();
            if in_edges.is_empty() {
                if root_ptrs.contains(&b) {
                    continue;
                }
                self.trie1_god.with_node_mut(b, |n| { n.children.remove(&None); });
                continue;
            }

            for (a, ek, bv1_original) in in_edges {
                let mut total_to_move = &bv1_original & &union_mask;
                if total_to_move.is_empty() { continue; }

                self.trie1_god.with_node_mut(a, |n| {
                    let m = n.children.entry(ek).or_default();
                    for (c, bv2) in &none_edges {
                        let to_move_for_c = &bv1_original & bv2;
                        if to_move_for_c.is_empty() { continue; }
                        m.entry(*c).and_modify(|ev| *ev |= &to_move_for_c).or_insert(to_move_for_c.clone());
                    }
                    if let Some(ev_ab) = m.get_mut(&b) {
                        *ev_ab -= &total_to_move;
                        if ev_ab.is_empty() {
                            m.remove(&b);
                        }
                    }
                    if m.is_empty() {
                        n.children.remove(&ek);
                    }
                });

            }

            self.trie1_god.with_node_mut(b, |n| { n.children.remove(&None); });
        }

        crate::debug!(2, "Done simplifying None edges.");
    }

    fn prune_on_no_terminal_follow(&mut self) {
        crate::debug!(2, "Pruning based on terminal follow sets.");

        let terminal_follow_map = self.terminal_follow_map;
        let ignore_terminal_id = self.ignore_terminal_id;

        let initial_nodes_and_values: Vec<_> = self.roots.values()
            .map(|&root_id| (root_id, None))
            .collect();

        type NodePtr = NodeId;
        let mut edges_to_keep: HashMap<NodePtr, BTreeSet<Option<GrammarTokenID>>> = HashMap::new();

        Trie::special_map(
            &self.trie1_god,
            initial_nodes_and_values,
            |predecessors, edge_terminal_opt, _edge_bv, _child_node| {
                match edge_terminal_opt {
                    Some(t) if Some(*t) == ignore_terminal_id => Some(predecessors.clone()),
                    Some(t) => Some(Some(BTreeSet::from([*t]))),
                    None => Some(predecessors.clone()),
                }
            },
            |existing_set, new_set| {
                match (existing_set, new_set) {
                    (None, _) => {},
                    (existing_set @ _, None) => *existing_set = None,
                    (Some(existing), Some(new)) => existing.extend(new),
                }
            },
            |node, maybe_all_immediate_predecessors| {
                if maybe_all_immediate_predecessors.is_none() {
                    return true;
                }

                let mut allowed_follow_terminals = BTreeSet::new();
                if let Some(all_immediate_predecessors) = &*maybe_all_immediate_predecessors {
                    for preceding_terminal in all_immediate_predecessors {
                        if let Some(follow_set) = terminal_follow_map.get(preceding_terminal) {
                            allowed_follow_terminals.extend(follow_set.iter().cloned());
                        }
                    }
                }

                let keys_to_keep: BTreeSet<_> = node.children().keys().filter(|edge_terminal_opt| {
                    match edge_terminal_opt {
                        Some(edge_terminal) => allowed_follow_terminals.contains(edge_terminal) || Some(*edge_terminal) == ignore_terminal_id,
                        None => true,
                    }
                }).cloned().collect();

                edges_to_keep.insert(node as *const _ as usize, keys_to_keep); // not used
                true
            },
        );

        // We deliberately skip applying edges_to_keep map here; the heuristic pruning via follow sets can be too aggressive in some grammars.
        crate::debug!(2, "Finished pruning based on terminal follow sets.");
    }

    fn prune_dead_paths(&mut self) {
        crate::debug!(2, "Pruning dead paths from precomputed trie.");

        let mut live_tokens_cache: HashMap<NodeId, LLMTokenBV> = HashMap::new();

        for &root in self.roots.values() {
            self.get_live_tokens_and_prune(root, &mut live_tokens_cache);
        }

        crate::debug!(2, "Finished pruning dead paths.");
    }

    fn get_live_tokens_and_prune(
        &self,
        node_id: NodeId,
        live_tokens_cache: &mut HashMap<NodeId, LLMTokenBV>,
    ) -> LLMTokenBV {
        if let Some(cached_bv) = live_tokens_cache.get(&node_id) {
            return cached_bv.clone();
        }
        live_tokens_cache.insert(node_id, LLMTokenBV::zeros());

        // collect children
        let children_to_check: Vec<NodeId> = self.trie1_god.with_node(node_id, |n| {
            n.children.values().flat_map(|m| m.keys().copied()).collect()
        });
        for child in children_to_check {
            self.get_live_tokens_and_prune(child, live_tokens_cache);
        }

        let mut live_tokens_for_this_node = LLMTokenBV::zeros();
        self.trie1_god.with_node_mut(node_id, |n| {
            if n.value.end {
                live_tokens_for_this_node = self.all_llm_tokens.clone();
            }

            n.children.retain(|_ek, dest_map| {
                dest_map.retain(|&child, edge_bv| {
                    let child_live = live_tokens_cache.get(&child)
                        .expect("Child not in cache");
                    let keep = &*edge_bv & child_live;
                    if keep.is_empty() {
                        false
                    } else {
                        *edge_bv = keep;
                        true
                    }
                });
                !dest_map.is_empty()
            });

            for dest_map in n.children.values() {
                for edge_bv in dest_map.values() {
                    live_tokens_for_this_node |= edge_bv;
                }
            }
            n.value.live_tokens = live_tokens_for_this_node.clone();
        });

        live_tokens_cache.insert(node_id, live_tokens_for_this_node.clone());
        live_tokens_for_this_node
    }

    fn factor_common_destinations(&mut self) {
        crate::debug!(2, "Factoring out common destinations to reduce non-None edges.");

        const MIN_INCOMING_EDGES_FOR_FACTORING: usize = 3;

        let roots_vec: Vec<_> = self.roots.values().copied().collect();
        let all_nodes = self.trie1_god.all_nodes(&roots_vec);

        let mut incoming_map: HashMap<
            NodeId,
            HashMap<
                GrammarTokenID,
                Vec<(NodeId, LLMTokenBV)>,
            >,
        > = HashMap::new();

        for &src in &all_nodes {
            self.trie1_god.with_node(src, |n| {
                for (ek_opt, dest_map) in &n.children {
                    if let Some(gtid) = ek_opt {
                        for (&dst, bv) in dest_map {
                            incoming_map.entry(dst).or_default().entry(*gtid).or_default().push((src, bv.clone()));
                        }
                    }
                }
            });
        }

        for (dest, edges_by_key) in incoming_map {
            for (gtid, sources) in edges_by_key {
                if sources.len() >= MIN_INCOMING_EDGES_FOR_FACTORING {
                    let intermediate_node = self.trie1_god.create(PrecomputedNodeContents::internal());

                    let mut union_bv = LLMTokenBV::zeros();
                    for (_, bv) in &sources {
                        union_bv |= bv;
                    }

                    {
                        let mut edge_val_opt = Some(union_bv.clone());
                        self.trie1_god.try_insert_unchecked(intermediate_node, Some(gtid), &mut edge_val_opt, dest)
                            .expect("Cycle detected when adding factored edge; this should not happen.");
                        self.trie1_god.with_node_mut(intermediate_node, |n| n.value.live_tokens |= &union_bv);
                    }

                    for (src, bv) in &sources {
                        self.trie1_god.with_node_mut(*src, |n| {
                            if let Some(dest_map_for_gtid) = n.children.get_mut(&Some(gtid)) {
                                dest_map_for_gtid.remove(&dest);
                                if dest_map_for_gtid.is_empty() {
                                    n.children.remove(&Some(gtid));
                                }
                            }
                        });

                        let mut edge_val_opt = Some(bv.clone());
                        self.trie1_god.try_insert_unchecked(*src, None, &mut edge_val_opt, intermediate_node).expect("no cycle expected");
                        self.trie1_god.with_node_mut(*src, |n| n.value.live_tokens |= bv.clone());
                    }
                }
            }
        }
        crate::debug!(2, "Finished factoring common destinations.");
    }

    fn merge_nodes(&mut self) {
        crate::debug!(2, "Merging identical subtrees in precomputed trie.");
        // This arena-based implementation can be added similarly to merge_nodes_trie2 if needed.
        // For now we skip aggressive dedup for trie1 to keep behavior consistent enough.
    }

    fn finish(
        mut self,
        token_name_map: &BiBTreeMap<Terminal, usize>,
        possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
        internal_max_llm_token: usize,
    ) -> BTreeMap<TokenizerStateID, NodeId> {

        calculate_final_stats_dummy(&self.roots, &mut self.stats, &self.trie1_god);
        print_precompute_stats(&self.stats, token_name_map);

        self.roots.clone()
    }

    fn dfs(
        &self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<TokenizerStateID, OrderedHashSet<NodeId>>,
    ) {
        self.pb.inc(1);

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let mut work_queue: BTreeMap<
                usize,
                BTreeMap<TokenizerStateID, OrderedHashSet<NodeId>>,
            > = BTreeMap::new();
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

                        for &src_node in &precompute_nodes {
                            if next_pos == segment_bytes.len() {
                                let llm_token_id = child_vocab_node.token_id();
                                let mut edge_bv = HybridBitset::zeros();
                                edge_bv.insert(llm_token_id);
                                let mut inserter = EdgeInserter::new(
                                    &self.trie1_god,
                                    src_node,
                                    Some(terminal_id),
                                    edge_bv,
                                    |ev, t| *ev &= &t.live_tokens,
                                    |e, n| *e |= n,
                                    |node_value, edge_value| node_value.live_tokens |= edge_value,
                                );
                                inserter = inserter.try_destination(self.end_node);
                                inserter.into_option().expect("Failed to insert end node for terminal at end of segment");
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
                                &self.trie1_god,
                                src_node,
                                Some(terminal_id),
                                edge_bv.clone(),
                                |ev, t| *ev &= &t.live_tokens,
                                |e, n| *e |= n,
                                |node_value, edge_value| node_value.live_tokens |= edge_value,
                            );

                            let next_tokenizer_state = self.tokenizer.initial_state_id();
                            let dest_nodes_in_queue = work_queue.entry(next_pos).or_default().entry(next_tokenizer_state).or_default();
                            let eligible_children: Vec<NodeId> = self.trie1_god.with_node(src_node, |s| {
                                s.children.get(&Some(terminal_id))
                                    .map(|m| m.keys().copied().collect())
                                    .unwrap_or_else(Vec::new)
                            });
                            inserter = inserter.try_destinations(&eligible_children);

                            let result_node = inserter.else_create_destination_with_value(PrecomputedNodeContents::internal()).unwrap();
                            dest_nodes_in_queue.insert(result_node);
                        }
                    }

                    if let Some(end_state_val) = exec_result.end_state {
                        let possible_final_tokens = self.tokenizer.tokens_accessible_from_state(TokenizerStateID(end_state_val));
                        for terminal_id in possible_final_tokens {
                            for &src_node in &precompute_nodes {
                                let llm_token_id = child_vocab_node.token_id();
                                let mut edge_bv = HybridBitset::zeros();
                                edge_bv.insert(llm_token_id);
                                let mut inserter = EdgeInserter::new(
                                    &self.trie1_god,
                                    src_node,
                                    Some(terminal_id),
                                    edge_bv,
                                    |ev, t| *ev &= &t.live_tokens,
                                    |e, n| *e |= n,
                                    |node_value, edge_value| node_value.live_tokens |= edge_value,
                                );
                                inserter = inserter.try_destination(self.end_node);
                                inserter.into_option().expect("Failed to insert end node for terminal at end of segment");
                            }
                        }
                        next_level_assoc.entry(TokenizerStateID(end_state_val)).or_default().extend(precompute_nodes.iter().copied());
                    }
                }
            }

            if !next_level_assoc.is_empty() {
                self.dfs(child_vocab_node, next_level_assoc);
            }
        }
    }
}

fn count_vocab_nodes(node: &VocabPrefixTreeNode) -> u64 {
    1 + node
        .children()
        .values()
        .map(|c| count_vocab_nodes(c))
        .sum::<u64>()
}

pub fn calculate_final_stats_dummy(
    precomputed_roots: &BTreeMap<TokenizerStateID, NodeId>,
    stats: &mut PrecomputeStats,
    god: &Trie1GodWrapper,
) {
    crate::debug!(2, "Calculating final precompute statistics (dummy)...");
    let roots_vec: Vec<_> = precomputed_roots.values().copied().collect();
    let all = god.all_nodes(&roots_vec);
    stats.final_unique_nodes_count = all.len();
    stats.final_root_nodes_count = precomputed_roots.len();
    stats.final_edges_count = all.iter().map(|&id| god.with_node(id, |n| n.children.values().map(|m| m.len()).sum::<usize>())).sum();
    stats.final_nodes_with_clean_end = all.iter().filter(|&&id| god.with_node(id, |n| n.value.end)).count();
    crate::debug!(2, "Finished calculating final precompute statistics (dummy).");
}

impl ParseState { // No longer generic
    pub fn merge(&mut self, mut other: ParseState) {
        Arc::make_mut(&mut self.stack).merge_with_depth(usize::MAX, &other.stack);
        Arc::make_mut(&mut self.accepted_state).merge_with_depth(usize::MAX, &other.accepted_state);
    }
}

pub trait InsertWith<K, V> {
    fn insert_with<F: FnOnce(&mut V, V)>(&mut self, k: K, v: V, combine: F);
}

impl<K, V> InsertWith<K, V> for BTreeMap<K, V> where K: Eq + Ord {
    fn insert_with<F: FnOnce(&mut V, V)>(&mut self, k: K, v: V, combine: F) {
        match self.entry(k) {
            std::collections::btree_map::Entry::Occupied(mut occupied) => {
                let value = occupied.get_mut();
                combine(value, v);
            }
            std::collections::btree_map::Entry::Vacant(vacant) => {
                vacant.insert(v);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GrammarConstraintState<'a> {
    pub(crate) parent: &'a GrammarConstraint,
    pub(crate) state:  BTreeMap<TokenizerStateID, GLRParserState<'a>>,
}

impl<'a> PartialEq for GrammarConstraintState<'a> {
    fn eq(&self, other: &Self) -> bool {
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
                max_edges: 50,
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
    pub fn get_mask(&self) -> LLMTokenBV {
        self.get_mask2()
    }

    #[time_it]
    pub fn get_mask2(&self) -> LLMTokenBV {
        let t0 = std::time::Instant::now();
        crate::debug!(2, "Getting mask {} states: {:?}", self.state.len(), self.state.keys().map(|k|k.0).collect::<Vec<_>>());
        let stats = gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        );
        crate::debug!(3, "GSS stats: {:#?}", stats);
        let roots = self.state.values().map(|s| s.active_state.stack.clone()).collect::<Vec<_>>();
        if GSS_LOGGING_ENABLED {
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
        }

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

        let step_counts = Arc::new(RwLock::new(BTreeMap::<TerminalID, StepCount>::new()));

        let mut initial_values_for_map: Vec<(NodeId, GLRParserState<'a>)> = Vec::new();
        for (tokenizer_state_id, glr_state) in &self.state {
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            if let Some(precomputed_trie_root_id) = self.parent.precomputed2.get(tokenizer_state_id) {
                let mut forbidden_llm_tokens = LLMTokenBV::zeros();
                let disallowed_terminals_l2 = glr_state.active_state.stack.disallowed_terminals();

                for (tokenizer_state_range, disallowed_terminals_for_range) in disallowed_terminals_l2.range_values() {
                    if disallowed_terminals_for_range.is_empty() {
                        continue;
                    }

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
                    disallow_llm_tokens_and_prune_arc(&mut glr_state.active_state.stack, &forbidden_llm_tokens, &mut HashMap::new());
                }
                initial_values_for_map.push((*precomputed_trie_root_id, glr_state));
            } else {
                panic!("No precomputed trie found for tokenizer state {:?}.", tokenizer_state_id);
            }
        }

        if initial_values_for_map.is_empty() {
             crate::debug!(2, "No valid initial states for get_mask's special_map traversal.");
             return self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        }

        let t1 = std::time::Instant::now();
        println!("after initial_values_for_map: {:>15?}", t1.duration_since(t0));

        let step_counts_clone1 = Arc::clone(&step_counts);
        let step_counts_clone2 = Arc::clone(&step_counts);

        crate::profiler::reset();

        Trie::special_map_grouped(
            &self.parent.trie2_god,
            initial_values_for_map,
            |glr_s, (k, expected_state_id_opt ), dest_map| {
                crate::debug!(4, "Processing step for k: {:?}, expected_state_id_opt: {:?}", k, expected_state_id_opt);
                let popped = glr_s.active_state.stack.popn(*k);
                let mut out_gsss = Vec::new();
                for popper_item in popped.iter() {
                    for peek in popper_item.peek_iter() {
                        let ok = if let Some(expected_state_id) = expected_state_id_opt {
                            expected_state_id == &peek.edge_value().state_id
                        } else {
                            true
                        };
                        if ok {
                            out_gsss.push(peek.isolated_parent());
                        }
                    }
                }
                if out_gsss.is_empty() {
                    crate::debug!(4, "No valid GSS nodes after popping, skipping.");
                    return Vec::new();
                }
                let out_gss = GSSNode::merge_many_with_depth(1, out_gsss);
                crate::debug!(4, "After popping {} from GSS: {}", k, print_gss_forest(&[out_gss.clone()], &self.parent.parser.terminal_map, &GSSPrintConfig::default()).0);
                let mut out = Vec::new();
                for (&dst_node_id, edge_bv) in dest_map.iter() {
                    let mut out_gss_filtered = out_gss.clone();
                    crate::debug!(5, "Filtering GSS for edge LLM tokens");
                    allow_only_llm_tokens_and_prune_arc(&mut out_gss_filtered, edge_bv, &mut HashMap::new());
                    let mut out_glr_s = glr_s.clone();
                    out_glr_s.active_state.stack = out_gss_filtered;
                    out.push((dst_node_id, out_glr_s));
                }
                out
            },
            |glr_s1, glr_s2| {
                crate::debug!(4, "Merging two GLR states");
                glr_s1.merge_with(glr_s2);
            },
            |precomputed_node_data, glr_s| {
                crate::debug!(4, "Processing node");
                let glr_active_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                let keep_going = glr_s.is_ok();
                if precomputed_node_data.value.end {
                    crate::debug!(4, "End node -> add tokens");
                    *final_mask_internal.borrow_mut() |= glr_active_tokens;
                }
                keep_going
            },
        );

        let t_after_special_map = std::time::Instant::now();
        println!("after special_map: {:>15?}", t_after_special_map.duration_since(t0));

        crate::profiler::print_summary_flat();

        crate::profiler::print_summary();
        crate::profiler::reset();

        if GSS_LOGGING_ENABLED {
            crate::debug!(3, "Final GSS states after get_mask:");
            let roots: Vec<_> = self.state.values().map(|s| s.active_state.stack.clone()).collect();
            let labels: Vec<_> = self.state.keys().map(|k| format!("Tokenizer State {}", k.0)).collect();
            let config = GSSPrintConfig {
                labels: Some(&labels),
                max_edges: 300,
                original_internal_bimap: Some(&self.parent.llm_vocab.original_to_internal_id_bimap),
                llm_token_map: Some(&self.parent.llm_vocab.llm_token_map),
                verbose: false,
            };
            print!("{}", print_gss_forest(&roots, &self.parent.parser.terminal_map, &config).0);
        }

        crate::debug!(4, "Final mask internal: {:?}", final_mask_internal.borrow());
        let final_mask_mapped = self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        crate::debug!(4, "Final mask mapped: {:?}", final_mask_mapped);

        let t_end = std::time::Instant::now();
        println!("get_mask took: {:>15?}", t_end.duration_since(t0));

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

        for state in self.state.values_mut() {
            map_allowed_terminals_tokenizer_states(&mut state.active_state.stack, &state_map, &mut gss_transformation_memo);
        }
        gss_transformation_memo.clear();

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

                    if cloned_glr_s.is_ok() {
                        let new_offset = offset + match_info.width;
                        let next_tokenizer_id_for_segment = self.parent.tokenizer.initial_state_id();

                        let mut disallowed_terminals = crate::datastructures::hybrid_l2_bitset::HybridL2Bitset::new();
                        if let Some(end_state_id) = exec_result.end_state {
                            let mut disallowed_terminals_for_end_state = TerminalBV::zeros();
                            disallow_terminals_and_prune_arc(&mut cloned_glr_s.active_state.stack, &disallowed_terminals, &mut HashMap::new());
                            disallowed_terminals_for_end_state.insert(match_info.id);
                            disallowed_terminals.insert_l2_bitset(end_state_id, disallowed_terminals_for_end_state);
                        }

                        if new_offset == llm_token_bytes.len() {
                            new_overall_state.entry(next_tokenizer_id_for_segment).and_modify(|existing| existing.merge_with(cloned_glr_s.clone())).or_insert(cloned_glr_s);
                        } else {
                            processing_queue.entry(new_offset).or_default().entry(next_tokenizer_id_for_segment).and_modify(|existing| existing.merge_with(cloned_glr_s.clone())).or_insert(cloned_glr_s);
                        }
                    }
                }

                if let Some(final_tokenizer_s_id_for_llm_token_segment) = exec_result.end_state {
                    let final_tokenizer_state = TokenizerStateID(final_tokenizer_s_id_for_llm_token_segment);
                    new_overall_state.entry(final_tokenizer_state).and_modify(|existing| existing.merge_with(glr_s_at_offset.clone())).or_insert(glr_s_at_offset.clone());
                }
            }
        }

        self.state = new_overall_state.clone();
        for glr_parser_state in self.state.values_mut() {
        }

        for state in self.state.values_mut() {
            reset_llm_tokens(&mut state.active_state.stack, &mut gss_transformation_memo);
        }
        gss_transformation_memo.clear();

        self.state.retain(|_, glr_parser_state| glr_parser_state.is_ok());

        crate::debug!(2, "Active tokenizer states after committing text (bytes {:?}): {:?}", llm_token_bytes, self.state.keys().map(|k|k.0).collect::<Vec<_>>());
    }

    pub fn is_active_or_accepted(&self) -> bool {
        !self.state.is_empty() && self.state.values().any(|s| !s.active_state.stack.is_empty() || s.has_accepted())
    }

    pub fn state(&self) -> &BTreeMap<TokenizerStateID, GLRParserState<'a>> {
        &self.state
    }
}

/// Custom streaming serializer: replaced by serialize_graph in trie.rs.
/// For compatibility, we keep wrappers that call the arena-based serializer.
fn stream_trie_to_writer<W: Write>(
    root_arc: &Arc<RwLock<PrecomputeNode2>>,
    mut writer: W,
) -> Result<(), String> {
    let guard = root_arc.read().map_err(|_| "RwLock poisoned".to_string())?;
    let fake_god = Trie2GodWrapper::new();
    let new_root = fake_god.create(guard.value.clone());
    drop(guard);
    let json = crate::datastructures::trie::serialize_graph(&fake_god, new_root);
    json.to_writer(&mut writer)?;
    Ok(())
}

pub fn write_precomputed2_to_stream<W: Write>(
    precomputed2: &Precomputed2,
    mut writer: W,
    god: &Trie2GodWrapper,
) -> Result<(), String> {
    writer.write_all(b"[").map_err(|e| e.to_string())?;
    let mut pb = tqdm!(total = precomputed2.len(), desc = "Writing tries", disable = !PROGRESS_BAR_ENABLED, leave=false);
    let mut first = true;
    for (key, root_id) in precomputed2 {
        if !first {
            writer.write_all(b",").map_err(|e| e.to_string())?;
        }
        first = false;
        writer.write_all(b"[").map_err(|e| e.to_string())?;
        key.to_json().to_writer(&mut writer)?;
        writer.write_all(b",").map_err(|e| e.to_string())?;
        let json = crate::datastructures::trie::serialize_graph(god, *root_id);
        json.to_writer(&mut writer)?;
        writer.write_all(b"]").map_err(|e| e.to_string())?;
        let _ = pb.update(1);
    }
    writer.write_all(b"]").map_err(|e| e.to_string())?;
    Ok(())
}

pub fn read_precomputed2_from_stream<R: Read>(
    reader: R,
    god: &Trie2GodWrapper,
) -> Result<Precomputed2, String> {
    let pairs: Vec<(SerdeValue, SerdeValue)> = serde_json::from_reader(reader)
        .map_err(|e| format!("Failed to parse precomputed2 stream as array of pairs: {}", e))?;

    let mut map: Precomputed2 = BTreeMap::new();
    let mut pb = tqdm!(total = pairs.len(), desc = "Loading tries from JSON", disable = !PROGRESS_BAR_ENABLED, leave=false);

    for (key_val, trie_val) in pairs {
        let key_node = JSONNode::from_serde_value(key_val)?;
        let key = <TokenizerStateID as JSONConvertible>::from_json(key_node)?;

        let trie_node = JSONNode::from_serde_value(trie_val)?;
        let root_id: NodeId = crate::datastructures::trie::deserialize_graph(god, trie_node)?;

        map.insert(key, root_id);
        let _ = pb.update(1);
    }
    Ok(map)
}
