// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use crate::datastructures::gss::{disallow_llm_tokens_and_prune_arc, fuse_predecessors_recursive, get_roots, print_gss_forest, reset_terminals, sample_path};
use crate::datastructures::gss::{map_allowed_terminals_tokenizer_states, prune_disallowed_terminals};
use crate::datastructures::ordered_hash_map::Retain;
use ordered_hash_map::OrderedHashMap;
use ordered_hash_map::OrderedHashSet;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::{self, Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::{BitOr, BitOrAssign};
use std::sync::Arc;
use std::sync::{Mutex, RwLock};

use bimap::BiBTreeMap;
use bitvec::prelude::*;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::constraint_extra::{calculate_final_stats, dump_precompute_trie_recursive, print_precompute_stats, PrecomputeStats};
use crate::constraint_precompute2_utils;
use crate::datastructures::arc_wrapper::ArcPtrWrapper;
use crate::datastructures::entry_api::EntryApi;
use crate::datastructures::gss::Acc;
use crate::datastructures::gss::{allow_only_llm_tokens_and_prune_arc, disallow_terminals_and_prune_arc, gather_gss_stats, reset_llm_tokens, GSSNode, GSSPrintConfig, LLMTokenBV, PrecomputedNodeContents, TerminalBV};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::{EdgeInserter, Trie, Trie2Index};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::finite_automata::Regex;
use crate::glr::analyze::compute_terminal_follow_sets;
use crate::glr::grammar::Terminal;
use crate::glr::items::{Item, LRMode, LR_MODE};
use crate::glr::parser::{BelowBottomReductionMode, GLRParser, GLRParserState, ParseState, ParseStateEdgeContent, ProcessDefaultReductionsAdvancedConfig, ProcessTokenAdvancedConfig};
use crate::glr::table::Stage7ShiftsAndReducesLookaheadValue;
use crate::glr::table::StateID;
use crate::interface::CompiledGrammar;
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::profiler::{print_summary, print_summary_flat, reset, GSS_LOGGING_ENABLED, PROGRESS_BAR_ENABLED};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use deterministic_hash::DeterministicHasher;
use kdam::{tqdm, BarBuilder, BarExt};
use profiler_macro::{time_it, timeit};
use rand::seq::{IndexedRandom, SliceRandom};
use rand::Rng;
use serde_json::Value as SerdeValue;
use std::collections::BTreeMap as StdMap;
use std::io::{Read, Write};
use std::ops::{BitAnd, Sub};
use crate::constraint_precompute2_utils::optimize_trie2_size;
use crate::datastructures::trie::{God, GodWrapper};

const MERGE_THRESHOLD: usize = 20;

pub type StateIDBV = HybridBitset;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedNode3Contents {
    pub end: bool,
}
impl JSONConvertible for PrecomputedNode3Contents {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("end".to_string(), self.end.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let end = obj.remove("end").ok_or_else(|| "Missing field end".to_string())
                             .and_then(bool::from_json)?;
                Ok(PrecomputedNode3Contents { end })
            }
            _ => Err("Expected JSONNode::Object for PrecomputedNode3Contents".to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalAllowanceCheckMode {
    None,
    ImmediateSets,
    ImmediateProbe,
    StepProbe,
}

impl Default for TerminalAllowanceCheckMode {
    fn default() -> Self { TerminalAllowanceCheckMode::None }
}

impl JSONConvertible for TerminalAllowanceCheckMode {
    fn to_json(&self) -> JSONNode {
        let s = match self {
            TerminalAllowanceCheckMode::None => "none",
            TerminalAllowanceCheckMode::ImmediateSets => "immediate_sets",
            TerminalAllowanceCheckMode::ImmediateProbe => "immediate_probe",
            TerminalAllowanceCheckMode::StepProbe => "step_probe",
        };
        JSONNode::String(s.to_string())
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::String(s) => match s.as_str() {
                "none" => Ok(TerminalAllowanceCheckMode::None),
                "immediate_sets" => Ok(TerminalAllowanceCheckMode::ImmediateSets),
                "immediate_probe" => Ok(TerminalAllowanceCheckMode::ImmediateProbe),
                "step_probe" => Ok(TerminalAllowanceCheckMode::StepProbe),
                other => Err(format!("Unknown TerminalAllowanceCheckMode '{}'", other)),
            },
            other => Err(format!("Expected JSON string for TerminalAllowanceCheckMode, got {:?}", other)),
        }
    }
}

pub type PrecomputeNode = Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;
pub type PrecomputeNode2 = Trie<(usize, Option<StateID>), LLMTokenBV, PrecomputedNodeContents>;
pub type PrecomputeNode3 = Trie<(usize, LLMTokenBV), StateIDBV, PrecomputedNode3Contents>;

pub type PrecomputeNodeIndex = Trie2Index;
pub type PrecomputeNode2Index = Trie2Index;
pub type PrecomputeNode3Index = Trie2Index;

pub type Precomputed = BTreeMap<TokenizerStateID, PrecomputeNodeIndex>;
pub type Precomputed2 = BTreeMap<TokenizerStateID, PrecomputeNode2Index>;
pub type Precomputed3 = BTreeMap<TokenizerStateID, PrecomputeNode3Index>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LLMVocab {
    pub(crate) llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID>,
    pub(crate) max_original_llm_token_id: usize,
    pub original_to_internal_id_bimap: BiBTreeMap<usize, usize>,
    pub(crate) internal_max_llm_token: usize
}

#[derive(Debug, Clone)]
pub struct GrammarConstraintConfig {
    pub optimize_trie2_prune_dead_paths: bool,
    pub optimize_trie2_merge_nodes: bool,
    pub optimize_trie2_factor_common_destinations: bool,
    pub optimize_trie2_compress_edges: bool,
    pub optimize_trie2_gc: bool,
}

impl Default for GrammarConstraintConfig {
    fn default() -> Self {
        // Self {
        //     optimize_trie2_prune_dead_paths: true,
        //     optimize_trie2_merge_nodes: true,
        //     optimize_trie2_factor_common_destinations: false,
        //     optimize_trie2_compress_edges: true,
        //     optimize_trie2_gc: true,
        // }
        Self {
            optimize_trie2_prune_dead_paths: true,
            optimize_trie2_merge_nodes: true,
            optimize_trie2_factor_common_destinations: false,
            optimize_trie2_compress_edges: false,
            optimize_trie2_gc: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GrammarConstraint {
    pub(crate) tokenizer:        Regex,
    pub(crate) parser:           GLRParser,
    pub(crate) precomputed:      Precomputed,
    pub precomputed2:     Precomputed2,
    pub precomputed3:     Precomputed3,
    pub llm_vocab:        Arc<LLMVocab>,
    pub(crate) token_name_map:   BiBTreeMap<Terminal, usize>,
    pub possible_matches: BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    pub(crate) trie1_god: Trie1GodWrapper,
    pub trie2_god: Trie2GodWrapper,
    pub trie3_god: Trie3GodWrapper,
    pub post_commit_allow_check_mode: TerminalAllowanceCheckMode,
}

impl GrammarConstraint {
    pub fn assert_eq(&self, other: &Self) {
        assert_eq!(self.tokenizer, other.tokenizer);
        assert_eq!(self.parser, other.parser);
        assert_eq!(self.precomputed.len(), other.precomputed.len());
        for ((sid1, arc1), (sid2, arc2)) in self.precomputed.iter().zip(other.precomputed.iter()) {
            assert_eq!(sid1, sid2);
            assert!(PrecomputeNode::are_graphs_equal(&self.trie1_god, *arc1, &other.trie1_god, *arc2));
        }
        assert_eq!(self.precomputed2.len(), other.precomputed2.len());
        for ((sid1, arc1), (sid2, arc2)) in self.precomputed2.iter().zip(other.precomputed2.iter()) {
            assert_eq!(sid1, sid2);
            assert!(PrecomputeNode2::are_graphs_equal(&self.trie2_god, *arc1, &other.trie2_god, *arc2));
        }
        assert_eq!(self.precomputed3.len(), other.precomputed3.len());
        for ((sid1, arc1), (sid2, arc2)) in self.precomputed3.iter().zip(other.precomputed3.iter()) {
            assert_eq!(sid1, sid2);
            assert!(PrecomputeNode3::are_graphs_equal(&self.trie3_god, *arc1, &other.trie3_god, *arc2));
        }
        assert_eq!(self.llm_vocab.llm_token_map, other.llm_vocab.llm_token_map);
        assert_eq!(self.token_name_map, other.token_name_map);
        assert_eq!(self.llm_vocab.max_original_llm_token_id, other.llm_vocab.max_original_llm_token_id);
        assert_eq!(self.llm_vocab.original_to_internal_id_bimap, other.llm_vocab.original_to_internal_id_bimap);
        assert_eq!(self.llm_vocab.internal_max_llm_token, other.llm_vocab.internal_max_llm_token);
        assert_eq!(self.possible_matches, other.possible_matches);
        assert_eq!(self.post_commit_allow_check_mode, other.post_commit_allow_check_mode);
    }
}

impl JSONConvertible for GrammarConstraint {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("tokenizer".to_string(), self.tokenizer.to_json());
        obj.insert("parser".to_string(), self.parser.to_json());
        obj.insert("precomputed".to_string(), self.precomputed.to_json());
        obj.insert("precomputed2".to_string(), self.precomputed2.to_json());
        obj.insert("precomputed3".to_string(), self.precomputed3.to_json());
        obj.insert("llm_token_map".to_string(), self.llm_vocab.llm_token_map.to_json());
        obj.insert("token_name_map".to_string(), self.token_name_map.to_json());
        obj.insert("max_original_llm_token_id".to_string(), self.llm_vocab.max_original_llm_token_id.to_json());
        obj.insert("original_to_internal_id_bimap".to_string(), self.llm_vocab.original_to_internal_id_bimap.to_json());
        obj.insert("internal_max_llm_token".to_string(), self.llm_vocab.internal_max_llm_token.to_json());
        obj.insert("possible_matches".to_string(), self.possible_matches.to_json());
        obj.insert("trie1_god".to_string(), self.trie1_god.to_json());
        obj.insert("trie2_god".to_string(), self.trie2_god.to_json());
        obj.insert("trie3_god".to_string(), self.trie3_god.to_json());
        obj.insert("post_commit_allow_check_mode".to_string(), self.post_commit_allow_check_mode.to_json());
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
                let precomputed2 = obj.remove("precomputed2").ok_or_else(|| "Missing field precomputed2".to_string())
                                     .and_then(|n| Precomputed2::from_json(n))?;
                let precomputed3 = obj.remove("precomputed3").ok_or_else(|| "Missing field precomputed3".to_string())
                                     .and_then(|n| Precomputed3::from_json(n))?;

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
                let trie1_god = obj.remove("trie1_god").ok_or_else(|| "Missing field trie1_god".to_string())
                                    .and_then(|n| Trie1GodWrapper::from_json(n))?;
                let trie2_god = obj.remove("trie2_god").ok_or_else(|| "Missing field trie2_god".to_string())
                                    .and_then(|n| Trie2GodWrapper::from_json(n))?;
                let trie3_god = obj.remove("trie3_god").ok_or_else(|| "Missing field trie3_god".to_string())
                                    .and_then(|n| Trie3GodWrapper::from_json(n))?;
                let post_commit_allow_check_mode = match obj.remove("post_commit_allow_check_mode") {
                    Some(n) => TerminalAllowanceCheckMode::from_json(n)?,
                    None => TerminalAllowanceCheckMode::None,
                };

                Ok(GrammarConstraint {
                    tokenizer,
                    parser,
                    precomputed,
                    precomputed2,
                    precomputed3,
                    llm_vocab: Arc::new(LLMVocab { llm_token_map, max_original_llm_token_id, original_to_internal_id_bimap, internal_max_llm_token }),
                    token_name_map,
                    possible_matches,
                    trie1_god,
                    trie2_god,
                    trie3_god,
                    post_commit_allow_check_mode,
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
        Self::from_compiled_grammar_with_config(
            compiled_grammar,
            llm_token_map,
            _eof_token_id,
            max_original_llm_token_id,
            &GrammarConstraintConfig::default(),
        )
    }

    pub fn from_compiled_grammar_with_config(
        compiled_grammar: CompiledGrammar,
        llm_token_map: LLMTokenMap,
        _eof_token_id: LLMTokenID,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
    ) -> Self {
        let token_name_map = compiled_grammar.definition.terminal_to_group_id().clone();

        Self::new_with_config(
            compiled_grammar.tokenizer, compiled_grammar.glr_parser, llm_token_map, token_name_map,
            max_original_llm_token_id, config,
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
        Self::new_with_config(
            tokenizer,
            parser,
            llm_token_map,
            token_name_map,
            max_original_llm_token_id,
            &GrammarConstraintConfig::default(),
        )
    }

    pub fn new_with_config(
        tokenizer:        Regex,
        parser:           GLRParser,
        llm_token_map:    LLMTokenMap,
        token_name_map:   BiBTreeMap<Terminal, usize>,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
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

        let (precomputed, trie1_god) = Self::precompute(
            &tokenizer,
            Some(&parser),
            Some(llm_vocab.clone()),
            &internal_llm_token_map_for_precompute,
            &token_name_map,
            internal_max_llm_token,
            &terminal_follow_map,
            parser.ignore_terminal_id,
            &mut computed_possible_matches,
        );
        // Self::_dump_precomputed(
        //     &precomputed,
        //     &llm_vocab.original_to_internal_id_bimap,
        //     &token_name_map,
        //     &llm_vocab.llm_token_map,
        //     &trie1_god,
        // );

        let (precomputed2, trie2_god) = Self::precompute2(
            &precomputed,
            &trie1_god,
            &tokenizer,
            Some(&parser),
            Some(llm_vocab.clone()),
            &internal_llm_token_map_for_precompute,
            &token_name_map,
            internal_max_llm_token,
            &terminal_follow_map,
            parser.ignore_terminal_id,
            &mut computed_possible_matches,
            config,
        );

        let mut stats2 = PrecomputeStats::default();
        crate::constraint_extra::calculate_final_stats2(&precomputed2, &mut stats2, &trie2_god);
        crate::constraint_extra::print_precompute_stats2(&stats2, &trie2_god);

        // Self::_dump_precomputed2(
        //     &precomputed2,
        //     &llm_vocab.original_to_internal_id_bimap,
        //     &llm_vocab.llm_token_map,
        //     &trie2_god,
        // );

        let (precomputed3, trie3_god) = Self::precompute3(
            &precomputed2,
            &trie2_god,
            config,
        );

        let mut stats3 = PrecomputeStats::default();
        crate::constraint_extra::calculate_final_stats3(&precomputed3, &mut stats3, &trie3_god);
        crate::constraint_extra::print_precompute_stats3(&stats3, &trie3_god);

        // Self::_dump_precomputed3(
        //     &precomputed3,
        //     &llm_vocab.original_to_internal_id_bimap,
        //     &llm_vocab.llm_token_map,
        //     &trie3_god,
        // );

        let mut gc = Self {
            tokenizer,
            parser,
            precomputed,
            precomputed2,
            precomputed3,
            llm_vocab,
            token_name_map,
            possible_matches: computed_possible_matches,
            trie1_god,
            trie2_god,
            trie3_god,
            post_commit_allow_check_mode: TerminalAllowanceCheckMode::None,
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
    ) -> (BTreeMap<TokenizerStateID, PrecomputeNodeIndex>, Trie1GodWrapper) {
        return (BTreeMap::new(), Trie1GodWrapper::new()); // TEMP

        let mut helper = Precomputer::new(
            tokenizer,
            parser,
            llm_vocab,
            internal_llm_token_map,
            internal_max_llm_token,
            MERGE_THRESHOLD,
            terminal_follow_map,
            ignore_terminal_id,
        );

        helper.run_dfs();
        // helper.optimize_precomputed_via_substring_parser();
        helper.replace_ignore_token_edges_with_none_edges();
        helper.simplify_none_edges(); // This can invalidate max_depth.

        // Recompute all max_depth values after major graph surgery.
        let roots_for_recompute: Vec<_> = helper.roots.values().cloned().collect();
        Trie::recompute_all_max_depths(&helper.trie1_god, &roots_for_recompute);

        helper.prune_dead_paths();
        helper.prune_on_no_terminal_follow();
        helper.prune_dead_paths();
        // New: prune using substring parser in "everything state" mode
        // helper.prune_with_substring_everything_state();
        helper.prune_dead_paths(); // Clean up after GLR-based pruning
        helper.factor_common_destinations();
        helper.merge_nodes();
        // helper.merge_nodes_basic();
        helper.gc();
        helper.finish(token_name_map, possible_matches, internal_max_llm_token)
    }

    /// Build the "Trie 2" precomputation.
    pub fn precompute2(
        precomputed: &BTreeMap<TokenizerStateID, PrecomputeNodeIndex>,
        trie1_god: &Trie1GodWrapper,
        tokenizer:        &Regex,
        parser:           Option<&GLRParser>,
        llm_vocab:        Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        token_name_map:   &BiBTreeMap<Terminal, usize>,
        internal_max_llm_token: usize,
        terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        ignore_terminal_id: Option<TerminalID>,
        possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
        config: &GrammarConstraintConfig,
    ) -> (Precomputed2, Trie2GodWrapper) {
        crate::debug!(2, "Precomputing Trie 2...");
        const BELOW_BOTTOM_REDUCE_MODE__CONTINUE_FROM_EVERYTHING: bool = false;
        const BELOW_BOTTOM_REDUCE_MODE: BelowBottomReductionMode = if BELOW_BOTTOM_REDUCE_MODE__CONTINUE_FROM_EVERYTHING {
            BelowBottomReductionMode::ContinueFromEverything
        } else {
            BelowBottomReductionMode::ContinueFromAll
        };

        let mut precomputed2 = BTreeMap::new();
        let mut trie2_god = Trie2GodWrapper::new();

        // let mut memo: HashMap<PrecomputeNode2Index, Arc<RwLock<_>>> = HashMap::new(); // Old memo, removed

        let mut initial_values_for_map: Vec<(PrecomputeNodeIndex, GLRParserState)> =
            Vec::new();
        let parser = parser.unwrap();

        // 1) Build a single base Trie root.
        // let base_trie2_root = Arc::new(RwLock::new(PrecomputeNode2::new(
        //     PrecomputedNodeContents::root(internal_max_llm_token),
        // )));
        let base_trie2_root = PrecomputeNode2Index::new(trie2_god.insert(PrecomputeNode2::new(PrecomputedNodeContents::root(internal_max_llm_token))));
        let base_trie2_root_wr = base_trie2_root.clone();

        let mut base_gss_nodes: Vec<Arc<GSSNode>> = Vec::new();

        if BELOW_BOTTOM_REDUCE_MODE__CONTINUE_FROM_EVERYTHING {
            let mut acc = Acc::new_fresh();
            acc.trie2_nodes.insert(base_trie2_root_wr.clone());
            let gss_leaf = Arc::new(GSSNode::new(acc));
            base_gss_nodes.push(Arc::new(
                gss_leaf.push(ParseStateEdgeContent { state_id: parser.everything_state_id })
            ));
        } else {
            for state_id in parser.table.keys() {
                let mut acc = Acc::new_fresh();
                acc.trie2_nodes.insert(base_trie2_root_wr.clone());
                let gss_leaf = Arc::new(GSSNode::new(acc));
                base_gss_nodes.push(Arc::new(gss_leaf.push(ParseStateEdgeContent { state_id: *state_id })));
            }
        }

        // Merge the base per-state initial nodes into one GSS and build a GLR state from it.
        let base_gss_merged = GSSNode::merge_many_with_depth(usize::MAX, base_gss_nodes);
        let mut base_glr_state = parser.init_glr_parser_from_stack(base_gss_merged).with_god(trie2_god.clone());

        // Optional: pre-warm once with default reductions (your idea)
        const PROCESS_DEFAULT_REDUCTIONS: bool = false;
        if PROCESS_DEFAULT_REDUCTIONS {
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
            // Deep clone Trie
            let (cloned_trie2_root, trie2_map) = constraint_precompute2_utils::clone_trie2_graph(&base_trie2_root, &trie2_god);

            // Deep clone the base GSS, remapping trie2_nodes
            let cloned_gss = crate::datastructures::gss::deep_clone_gss_with_trie2_map(
                &base_glr_state.active_state.stack,
                &trie2_map,
            );
            let mut glr_state_for_sid = base_glr_state.clone();
            glr_state_for_sid.active_state.stack = cloned_gss;

            // Record per tokenizer state
            precomputed2.insert(*tokenizer_state_id, cloned_trie2_root);
            initial_values_for_map.push((trie1_root.clone(), glr_state_for_sid));
        }

        let trie2_end = PrecomputeNode2Index::new(trie2_god.insert(PrecomputeNode2::new(PrecomputedNodeContents::leaf())));

        crate::debug!(2, "Running special_map_grouped for Trie 2 precomputation");
        Trie::special_map_grouped(
            &trie1_god,
            initial_values_for_map,
            // step_fn: (current_glr_state, edge_grammar_token_opt, destinations_map)
            |current_glr_state, edge_grammar_token_opt, destinations_map| {
                crate::debug!(3, "Trie: Processing GLR state with {} destinations for edge grammar token: {:?}", destinations_map.len(), edge_grammar_token_opt);
                let mut glr_s = current_glr_state.clone();

                let mut edge_bv = LLMTokenBV::zeros();
                for bv in destinations_map.values() {
                    edge_bv |= bv;
                }
                // Restrict the GLR state to the LLM tokens allowed on this edge.
                allow_only_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &edge_bv, &mut HashMap::new());

                if let Some(gt) = edge_grammar_token_opt {
                    glr_s.process_token_advanced(*gt, &ProcessTokenAdvancedConfig { below_bottom_mode: BELOW_BOTTOM_REDUCE_MODE });
                        // print_summary_flat();
                        // print_summary();
                        // reset();
                }

                let mut out = Vec::new();
                for (dst_node_wrapper, edge_bv) in destinations_map.iter() {
                    let mut glr_s_copy = glr_s.clone();
                    // Restrict the GLR state to the LLM tokens allowed on this edge.
                    crate::debug!(5, "Trie: Restricting GLR state to edge bitset: {:?}", edge_bv);
                    allow_only_llm_tokens_and_prune_arc(
                        &mut glr_s_copy.active_state.stack,
                        edge_bv,
                        &mut HashMap::new(),
                    );
                    glr_s_copy.log_gss(
                        "Trie: After restricting GLR state to edge bitset",
                        TerminalID(0),
                        false,
                        false,
                    );
                    out.push((
                        dst_node_wrapper.clone(),
                        glr_s_copy,
                    ));
                }
                out
            },
            |glr_s1, glr_s2| {
                crate::debug!(4, "Trie: Merging GLR states");
                glr_s1.log_gss("Before merge...", TerminalID(0), false, false);
                glr_s2.log_gss("...with", TerminalID(0), false, false);
                glr_s1.merge_with(glr_s2);
                glr_s1.log_gss("After merge", TerminalID(0), false, false);
            },
            // process_fn
            |precomputed_node_data, glr_s| {
                crate::debug!(3, "Trie: At precomputed node {:p}, processing GLR state", precomputed_node_data);
                // Dump precomputed2
                // pub fn _dump_precomputed2(precomputed2: &BTreeMap<TokenizerStateID, PrecomputeNode2Index>, original_to_internal_id_bimap: &BiBTreeMap<usize, usize>, llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>) {
                // GrammarConstraint::_dump_precomputed2(&precomputed2, &llm_vocab.as_ref().unwrap().original_to_internal_id_bimap, &llm_vocab.as_ref().unwrap().llm_token_map);

                crate::datastructures::gss::merge_trie2_nodes_if_needed(
                    &mut glr_s.active_state.stack,
                    &mut HashMap::new(),
                    glr_s.active_state.trie2_god.as_ref().unwrap(),
                );
                let keep_going = glr_s.is_ok();
                if precomputed_node_data.value.end {
                    crate::debug!(3, "Trie: Found end state for GLR state");
                    glr_s.log_gss(
                        "Trie: Found end state for GLR state",
                        TerminalID(0),
                        false,
                        false,
                    );
                    let mut end_dest_agg: BTreeMap<PrecomputeNode2Index, LLMTokenBV> = BTreeMap::new();
                    let end_wr = trie2_end.clone();

                    let mut dest_agg: BTreeMap<PrecomputeNode2Index, LLMTokenBV> = BTreeMap::new();

                    // for (last_edge, gss_root_accs) in get_roots([glr_s.active_state.stack.as_ref(), glr_s.active_state.accepted_state.as_ref()]) {
                    for (last_edge, gss_root_accs) in get_roots([glr_s.active_state.stack.as_ref()]) {
                        for gss_root_acc in gss_root_accs {
                            let active_llm_tokens_for_root = gss_root_acc.union_llm_tokens();
                            crate::debug!(4, "Trie: For GSS root with edge {:?}, active LLM tokens: {:?}", last_edge, active_llm_tokens_for_root);

                            for src_wr in gss_root_acc.trie2_nodes.iter() {
                                let src_arc = src_wr.as_arc().clone();
                                let src_live = { src_arc.read(&trie2_god).expect("poison").value.live_tokens.clone() };
                                let tokens_to_push = &active_llm_tokens_for_root & &src_live;
                                if tokens_to_push.is_empty() {
                                    crate::debug!(4, "Trie: No tokens to push from this source node");
                                    continue;
                                }
                                {
                                    // Mark the source node as live for these tokens so the backward pass can see them.
                                    let mut src_w = src_arc.write(&trie2_god).expect("poison");
                                    src_w.value.live_tokens |= tokens_to_push.clone();
                                }
                                crate::debug!(4, "Trie: Pushing tokens {:?} from source node {:?}", tokens_to_push, src_arc);

                                let edge_key = (0, Some(last_edge.state_id));

                                let mut inserter = EdgeInserter::new(
                                    glr_s.active_state.trie2_god.as_ref().unwrap(),
                                    src_arc.clone(),
                                    edge_key,
                                    tokens_to_push.clone(),
                                    |e, n| *e |= n,
                                    |node_value, edge_value| node_value.live_tokens |= edge_value,
                                    |ev, t| *ev &= &t.live_tokens,
                                );

                                inserter = inserter.try_destination(trie2_end.clone());

                                let final_dest_arc = inserter.clone_into_option().expect("Failed to insert end edge into Trie node");
                                let final_dest_wr = final_dest_arc.clone();
                                dest_agg.entry(final_dest_wr.clone()).and_modify(|bv| *bv |= &tokens_to_push).or_insert(tokens_to_push.clone());
                            }
                        }
                    }
                    for (dst_wr, added) in &dest_agg {
                        let mut g = dst_wr.as_arc().write(&trie2_god).expect("poison");
                        g.value.live_tokens |= added.clone();
                    }
                }

                if PROCESS_DEFAULT_REDUCTIONS {
                    let mut allowed_terminals = TerminalBV::zeros();
                    for gtid_opt in precomputed_node_data.children().keys() {
                        if let Some(gtid) = gtid_opt {
                            allowed_terminals.insert(gtid.0);
                        }
                    }
                    let disallowed_terminals_bv = allowed_terminals.inverted();
                    if !disallowed_terminals_bv.is_empty() {
                        let disallowed_l2 = crate::datastructures::hybrid_l2_bitset::HybridL2Bitset::from_iter(
                            std::iter::once((0..=usize::MAX, disallowed_terminals_bv))
                        );
                        disallow_terminals_and_prune_arc(&mut glr_s.active_state.stack, &disallowed_l2, &mut HashMap::new());
                    }
                    // glr_s.process_default_reductions_advanced(&ProcessDefaultReductionsAdvancedConfig { fuel: None, per_state_fuel: Some(2), below_bottom_mode: BELOW_BOTTOM_REDUCE_MODE });
                    glr_s.process_default_reductions_advanced(&ProcessDefaultReductionsAdvancedConfig { fuel: None, per_state_fuel: None, below_bottom_mode: BELOW_BOTTOM_REDUCE_MODE });
                    reset_terminals(&mut glr_s.active_state.stack, &mut HashMap::new());
                }

                keep_going
            },
        );

        crate::debug!(2, "Finished precomputing Trie 2");

        // To prevent dangling weak pointers, we collect strong references to all nodes
        // before performing modifications. These strong references are held until
        // weak edges have been promoted back to strong ones where possible.
        // let roots_before_cleanup: Vec<_> = precomputed2.values().cloned().collect();
        // let all_nodes_pinner = Trie::all_nodes(&trie2_god, &roots_before_cleanup);

        // Clean up after rewiring
        optimize_trie2_size(&mut precomputed2, &trie2_god, config);

        // Trie::all_nodes(&trie2_god, &roots_before_cleanup); // Drop pinner, allow nodes to be freed if unreachable

        // Recompute depths again after promotions, as they can change the graph structure.
        let roots2_final: Vec<_> = precomputed2.values().cloned().collect();
        Trie::gc(&trie2_god, &roots2_final);
        Trie::recompute_all_max_depths(&trie2_god, &roots2_final);

        (precomputed2, trie2_god)
    }

    pub fn precompute3(
        precomputed2: &Precomputed2,
        trie2_god: &Trie2GodWrapper,
        _config: &GrammarConstraintConfig,
    ) -> (Precomputed3, Trie3GodWrapper) {
        crate::debug!(2, "Precomputing Trie 3...");
        let mut precomputed3 = BTreeMap::new();
        let trie3_god = Trie3GodWrapper::new();
        let mut memo: HashMap<Trie2Index, Trie2Index> = HashMap::new();

        for (sid, root2_idx) in precomputed2 {
            let root3_idx = Self::transform_trie2_to_trie3_recursive(
                *root2_idx,
                trie2_god,
                &trie3_god,
                &mut memo,
            );
            precomputed3.insert(*sid, root3_idx);
        }

        let roots3: Vec<_> = precomputed3.values().cloned().collect();
        Trie::recompute_all_max_depths(&trie3_god, &roots3);

        crate::debug!(2, "Finished precomputing Trie 3.");
        (precomputed3, trie3_god)
    }

    fn transform_trie2_to_trie3_recursive(
        node2_idx: Trie2Index,
        trie2_god: &Trie2GodWrapper,
        trie3_god: &Trie3GodWrapper,
        memo: &mut HashMap<Trie2Index, Trie2Index>,
    ) -> Trie2Index {
        if let Some(node3_idx) = memo.get(&node2_idx) {
            return *node3_idx;
        }

        let node2 = node2_idx.read(trie2_god).unwrap();
        let node3 = PrecomputeNode3::new(PrecomputedNode3Contents {
            end: node2.value.end,
        });
        let node3_idx = Trie2Index::new(trie3_god.insert(node3));
        memo.insert(node2_idx, node3_idx);

        // Intermediate map: (pop, llm_bv) -> (dest2_idx -> set of state_id_opt)
        let mut grouped_children: BTreeMap<(usize, LLMTokenBV), BTreeMap<Trie2Index, BTreeSet<Option<StateID>>>> = BTreeMap::new();

        for ((pop, state_id_opt), dest_map) in node2.children() {
            for (dest2_idx, llm_bv) in dest_map {
                grouped_children
                    .entry((*pop, llm_bv.clone()))
                    .or_default()
                    .entry(*dest2_idx)
                    .or_default()
                    .insert(*state_id_opt);
            }
        }

        for ((pop, llm_bv), dests_with_states) in grouped_children {
            for (dest2_idx, state_id_opts) in dests_with_states {
                let dest3_idx = Self::transform_trie2_to_trie3_recursive(dest2_idx, trie2_god, trie3_god, memo);

                let mut state_id_bv = StateIDBV::zeros();
                if state_id_opts.contains(&None) {
                    state_id_bv = StateIDBV::max_ones();
                } else {
                    for state_id_opt in state_id_opts {
                        if let Some(state_id) = state_id_opt {
                            state_id_bv.insert(state_id.0);
                        }
                    }
                }

                let mut node3_w = node3_idx.write(trie3_god).unwrap();
                node3_w.children_mut().entry((pop, llm_bv.clone())).or_default().insert(dest3_idx, state_id_bv);
            }
        }
        node3_idx
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
    pub fn original_bv_to_internal(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        let mut internal_bv = HybridBitset::zeros();
        for original_id_val in original_bv.iter() {
            let internal_id_val = self.llm_vocab.original_to_internal_id_bimap.get_by_left(&(original_id_val as usize)).expect(format!("Original ID {} not found in original_to_internal_id_bimap", original_id_val).as_str());
            internal_bv.insert(*internal_id_val as usize);
        }
        internal_bv
    }

    #[time_it]
    pub fn internal_bv_to_original(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        let internal_bv = internal_bv & &LLMTokenBV::max_ones();
        let mut original_bv = HybridBitset::zeros();
        // for internal_id_val in internal_bv.iter() {
        //     let original_id_val = self.llm_vocab.original_to_internal_id_bimap.get_by_right(&(internal_id_val as usize)).expect(format!("Internal ID {} not found in original_to_internal_id_bimap while converting to original BV: {:?}", internal_id_val, internal_bv).as_str());
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
    parser:           Option<&'r GLRParser>,
    llm_vocab:        Option<Arc<LLMVocab>>,
    vocab:            VocabPrefixTree,
    roots:            BTreeMap<TokenizerStateID, PrecomputeNodeIndex>,
    possible_matches: RefCell<BTreeMap<*const VocabPrefixTreeNode, BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>>>,
    all_llm_tokens:   HybridBitset,
    merge_threshold:  usize,
    pb:               ProgressBar,
    stats:            PrecomputeStats,
    terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    ignore_terminal_id: Option<TerminalID>,
    // Map each precompute node to the set of LLM tokens that can pass through it.
    // tags:             RefCell<HashMap<PrecomputeNodeIndex, LLMTokenBV>>, // Removed
    end_node:         PrecomputeNodeIndex,
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
    ) -> Self {
        let tokens: Vec<(usize, Vec<u8>)> = internal_llm_token_map
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone()))
            .collect();

        crate::debug!(2, "Building vocab prefix tree");
        let vocab = VocabPrefixTree::build(&tokens);
        crate::debug!(2, "Done building vocab prefix tree");

        let mut roots = BTreeMap::new();
        let trie1_god = Trie1GodWrapper::new();
        for sid in tokenizer.iter_states() {
            roots.insert(
                sid,
                PrecomputeNodeIndex::new(trie1_god.insert(PrecomputeNode::new(PrecomputedNodeContents::root(internal_max_llm_token)))),
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
        if !PROGRESS_BAR_ENABLED {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }

        let end_node = PrecomputeNode2Index::new(trie1_god.insert(PrecomputeNode::new(PrecomputedNodeContents::leaf())));

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
            // tags: RefCell::new(HashMap::new()), // Removed
            end_node,
            trie1_god,
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

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let exec_result = self.tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);
            for token in &exec_result.matches {
                let grammar_token_id = GrammarTokenID(token.id);
                let applicable_tokens = child_vocab_node.reachable_token_ids();
                *result_map.entry(grammar_token_id).or_insert_with(LLMTokenBV::zeros) |= applicable_tokens;
            }
            if let Some(final_state_val) = exec_result.end_state {
                let matches_possible_from_tokenizer_state: BTreeSet<_> = self.tokenizer.tokens_accessible_from_state(TokenizerStateID(final_state_val)).into_iter().collect();
                let matches_here: BTreeSet<_> = exec_result.matches.iter().map(|m| GrammarTokenID(m.id)).collect();
                let possible_new_matches = &matches_possible_from_tokenizer_state - &matches_here;
                if !possible_new_matches.is_empty() {
                    let next_results = self.possible_matches(child_vocab_node, TokenizerStateID(final_state_val));
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
            OrderedHashSet<PrecomputeNodeIndex>,
        > = BTreeMap::new();

        for (sid, arc) in &self.roots {
            assoc
                .entry(*sid)
                .or_default()
                .insert(arc.clone());
        }

        crate::debug!(2, "Starting precompute DFS");
        crate::debug!(3, "Roots for each tokenizer state:");
        for (sid, root) in &self.roots {
            crate::debug!(6, "  {}: {}", sid.0, root);
        }
        self.dfs(&self.vocab.root, assoc);
        crate::debug!(2, "Finished precompute DFS");
        self.pb.finish_with_message("Precomputation complete");
        crate::debug!(2, "Precomputation complete");
    }

    fn replace_ignore_token_edges_with_none_edges(&mut self) {
        let ignore_tid = if let Some(id) = self.ignore_terminal_id {
            id
        } else {
            return; // No ignore token, nothing to do.
        };

        crate::debug!(2, "Replacing ignore token edges with None edges...");

        // 1. Collect all unique nodes.
        let roots_vec: Vec<_> = self.roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(&self.trie1_god, &roots_vec);
        // 2. Iterate over each node and modify its children map.
        for node_arc in all_nodes {
            let mut node_guard = node_arc.write(&self.trie1_god).expect("poison");

            // Check if there are any edges with the ignore token key.
            let ignore_key = Some(ignore_tid);
            if let Some(dest_map_for_ignore_token) = node_guard.children_mut().remove(&ignore_key) {
                // Get or create the destination map for None edges.
                let dest_map_for_none = node_guard.children_mut().entry(None).or_default();

                // Move each destination from the ignore token map to the None map.
                for (dest_wrapper, edge_bv) in dest_map_for_ignore_token {
                    // If an edge to this destination already exists under None, merge the bitvectors.
                    if let Some(existing_bv) = dest_map_for_none.get_mut(&dest_wrapper) {
                        *existing_bv |= &edge_bv;
                    } else {
                        dest_map_for_none.insert(dest_wrapper, edge_bv);
                    }
                }
            }
        }

        crate::debug!(2, "Done replacing ignore token edges.");
    }

    /// Simplify out `None` edges by shortcutting predecessors to successors.
    ///
    /// For every `B -(None; bv2)-> C`, and for every incoming edge `A -(x; bv1)-> B`,
    /// we:
    ///   - add/merge an edge `A -(x; bv1 ∩ bv2)-> C`
    ///   - remove the moved tokens `bv1 ∩ bv2` from `A -(x; ...)-> B`
    /// After processing all incoming edges to B, we remove all `None` edges from B.
    ///
    /// This transformation preserves behavior while eliminating `None` edges and
    /// allows subsequent pruning and merging passes to operate on a simpler graph.
    fn simplify_none_edges(&mut self) {
        crate::debug!(2, "Simplifying None edges (shortcut predecessors to successors)...");

        let root_node_ptrs: HashSet<PrecomputeNodeIndex> = self.roots.values().cloned().collect();

        // 1) Collect all unique nodes reachable from any root
        let roots_vec: Vec<_> = self.roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(&self.trie1_god, &roots_vec);
        // Map pointer -> Arc for quick retrieval
        let mut arc_by_ptr: HashMap<PrecomputeNodeIndex, PrecomputeNodeIndex> = HashMap::new();
        for n in &all_nodes {
            arc_by_ptr.insert(*n, n.clone());
        }

        // 2) Build:
        //    - incoming[B] = vec of (A, key_x, bv1) for edges A -(x; bv1)-> B
        //    - none_edges_from[B] = vec of (C, bv2) for edges B -(None; bv2)-> C
        //    - none_union[B] = union of all bv2 for None edges from B
        let mut incoming: HashMap<
            PrecomputeNodeIndex,
            Vec<(PrecomputeNodeIndex, Option<GrammarTokenID>, LLMTokenBV)>
        > = HashMap::new();
        let mut none_edges_from: HashMap<
            PrecomputeNodeIndex,
            Vec<(PrecomputeNodeIndex, LLMTokenBV)>
        > = HashMap::new();
        let mut none_union: HashMap<PrecomputeNodeIndex, LLMTokenBV> = HashMap::new();

        for src_arc in &all_nodes {
            let src_ptr = src_arc;
            let guard = src_arc.read(&self.trie1_god).expect("poison");
            // Record all outgoing edges for incoming map
            for (ek, dest_map) in guard.children().iter() {
                for (child_wrap, ev_bv) in dest_map.iter() {
                    let child_arc = child_wrap.as_arc().clone();
                    let child_ptr = child_arc;
                    incoming.entry(child_ptr)
                        .or_default()
                        .push((src_arc.clone(), ek.clone(), ev_bv.clone()));
                }
            }
            // Record None edges out of src_arc (B -> C)
            if let Some(dest_map_none) = guard.children().get(&None) {
                let list = none_edges_from.entry(*src_ptr).or_default();
                for (child_wrap, ev_bv) in dest_map_none.iter() {
                    list.push((child_wrap.as_arc().clone(), ev_bv.clone()));
                    let entry = none_union.entry(*src_ptr).or_insert_with(LLMTokenBV::zeros);
                    *entry |= ev_bv;
                }
            }
        }

        // 3) For every node B that has None edges to children, rewrite predecessors.
        for (b_ptr, none_edges) in none_edges_from.into_iter() {
            let union_mask = match none_union.get(&b_ptr) {
                Some(bv) if !bv.is_empty() => bv.clone(),
                _ => continue,
            };
            // If no predecessors, still remove None edges later (could help pruning)
            let in_edges = match incoming.get(&b_ptr) {
                Some(v) if !v.is_empty() => v.clone(),
                _ => {
                    // No predecessors.
                    // If B is a root node, we must not remove its None edges, as there are no
                    // predecessors to shortcut from.
                    if root_node_ptrs.contains(&b_ptr) {
                        continue; // It's a root, leave its None edges.
                    }

                    // Not a root and no predecessors means it's an unreachable internal node.
                    // It's safe to remove its outgoing None edges.
                    if let Some(b_arc) = arc_by_ptr.get(&b_ptr).cloned() {
                        let mut b_guard = b_arc.write(&self.trie1_god).expect("poison");
                        b_guard.children_mut().remove(&None);
                    }
                    continue;
                }
            };

            let b_arc = match arc_by_ptr.get(&b_ptr) {
                Some(a) => a.clone(),
                None => continue,
            };
            let b_key = b_arc.clone();

            // For each incoming edge A -(x; bv1)-> B, split tokens:
            //   move:    to C with mask (bv1 ∩ bv2)
            //   leftover on A->B: bv1 - union_over_C(bv1 ∩ bv2) = bv1 ∩ (!union_mask)
            for (a_arc, edge_key, bv1_original) in in_edges.into_iter() {
                let mut total_to_move = bv1_original.clone();
                total_to_move &= &union_mask; // total tokens to redirect to all C via None edges
                if total_to_move.is_empty() {
                    continue;
                }

                let mut a_guard = a_arc.write(&self.trie1_god).expect("poison");
                let dest_map = a_guard.children_mut().entry(edge_key.clone()).or_default();

                // Add/merge edges to each C with per-child mask
                for (c_arc, bv2) in &none_edges {
                    let mut to_move_for_c = bv1_original.clone();
                    to_move_for_c &= bv2;
                    if to_move_for_c.is_empty() {
                        continue;
                    }
                    let c_key = c_arc.clone();
                    if let Some(existing_ev) = dest_map.get_mut(&c_key) {
                        *existing_ev |= &to_move_for_c;
                    } else {
                        dest_map.insert(c_key, to_move_for_c);
                    }
                }

                // Reduce/remove the A -> B edge for the moved tokens
                let mut remove_b_edge = false;
                if let Some(ev_ab) = dest_map.get_mut(&b_key) {
                    *ev_ab -= &total_to_move;
                    remove_b_edge = ev_ab.is_empty();
                }
                if remove_b_edge {
                    dest_map.remove(&b_key);
                }
            }

            // Finally, remove all None edges out of B
            {
                let mut b_guard = b_arc.write(&self.trie1_god).expect("poison");
                b_guard.children_mut().remove(&None);
            }
        }

        crate::debug!(2, "Done simplifying None edges.");
    }

    fn prune_on_no_terminal_follow(&mut self) {
        crate::debug!(2, "Pruning based on terminal follow sets.");

        let terminal_follow_map = self.terminal_follow_map;
        let ignore_terminal_id = self.ignore_terminal_id;

        let initial_nodes_and_values: Vec<_> = self.roots.values()
            .map(|root_arc| (root_arc.clone(), None))
            .collect();

        type NodePtr = *const PrecomputeNode;
        let mut edges_to_keep: HashMap<NodePtr, BTreeSet<Option<GrammarTokenID>>> = HashMap::new();

        Trie::special_map(
            &self.trie1_god,
            initial_nodes_and_values,
            |predecessors: &Option<BTreeSet<GrammarTokenID>>, edge_terminal_opt: &Option<GrammarTokenID>, _edge_bv, _child_node| {
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
                // If there are no preceding terminals (e.g., root or only None-edges path from root),
                // all outgoing terminals are considered valid.
                if maybe_all_immediate_predecessors.is_none() {
                    return true; // Continue traversal, no pruning needed for this node.
                }

                // Compute the set of all allowed terminals that can follow any of the immediate predecessors.
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
                        // Keep edges with terminals that are in the allowed follow set (or ignore edges).
                        Some(edge_terminal) => allowed_follow_terminals.contains(edge_terminal) || Some(*edge_terminal) == ignore_terminal_id,
                        // Always keep `None` edges, as they don't represent grammar terminals.
                        None => true,
                    }
                }).cloned().collect();

                let node_ptr: NodePtr = node;
                edges_to_keep.insert(node_ptr, keys_to_keep);

                true // Continue traversal
            },
        );

        // Now, apply the pruning.
        let roots_vec: Vec<_> = self.roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(&self.trie1_god, &roots_vec);
        for node_arc in all_nodes {
            let node_ptr: NodePtr = {
                let guard = node_arc.read(&self.trie1_god).expect("poison");
                &*guard as *const _
            };
            if let Some(keys_to_keep) = edges_to_keep.get(&node_ptr) {
                let mut node_guard = node_arc.write(&self.trie1_god).unwrap();
                node_guard.children_mut().retain(|k, _| keys_to_keep.contains(k));
            }
        }

        crate::debug!(2, "Finished pruning based on terminal follow sets.");
    }

    fn prune_dead_paths(&mut self) {
        crate::debug!(2, "Pruning dead paths from precomputed trie.");

        // A cache of nodes to the set of "live" LLM tokens reachable from them.
        let mut live_tokens_cache: HashMap<PrecomputeNodeIndex, LLMTokenBV> = HashMap::new();

        // For each root, run the pruning process. This will modify the trie in-place.
        // We do not remove the root from the map even if it becomes "dead" (has no live paths).
        // This ensures that every tokenizer state ID that started with a trie root still has one,
        // preventing panics in later stages that expect a complete map.
        for root_arc in self.roots.values() {
            let root_wrapper = root_arc.clone();
            self.get_live_tokens_and_prune(root_wrapper, &mut live_tokens_cache);
        }

        crate::debug!(2, "Finished pruning dead paths.");
    }

    /// Recursively computes the set of "live" LLM tokens reachable from a node
    /// and prunes its children that are not live or have dead token paths.
    /// This is a post-order traversal.
    ///
    /// - `node_wrapper`: The node to check.
    /// - `live_tokens_cache`: A cache of nodes to their live token bitvectors.
    ///
    /// Returns a `LLMTokenBV` of all live tokens reachable from `node_wrapper`.
    fn get_live_tokens_and_prune(
        &self,
        node_wrapper: PrecomputeNodeIndex,
        live_tokens_cache: &mut HashMap<PrecomputeNodeIndex, LLMTokenBV>,
    ) -> LLMTokenBV {
        // If we've already computed the live tokens for this node, return the cached result.
        if let Some(cached_bv) = live_tokens_cache.get(&node_wrapper) {
            return cached_bv.clone();
        }
        // Insert a temporary empty BV to break cycles. If we revisit this node during this
        // recursion, it will return an empty set, which is correct as no new live paths
        // have been found through it yet.
        live_tokens_cache.insert(node_wrapper.clone(), LLMTokenBV::zeros());

        let node_arc = node_wrapper.as_arc().clone();

        // We must collect children before recursing to avoid holding the lock.
        let children_to_check: Vec<PrecomputeNodeIndex> = {
            let node_guard = node_arc.read(&self.trie1_god).unwrap();
            node_guard.children().values().flat_map(|dest_map| dest_map.keys().cloned()).collect()
        };

        // Recursively call on all unique children to populate the cache for them.
        for child_wrapper in children_to_check {
            self.get_live_tokens_and_prune(child_wrapper, live_tokens_cache);
        }

        // Now that the cache is populated for all children, we can prune the current node.
        let mut live_tokens_for_this_node = LLMTokenBV::zeros();
        {
            let mut node_guard = node_arc.write(&self.trie1_god).unwrap();

            // A node is live if it's an end node itself. The tokens that end here are
            // on the edges pointing to this node.
            if node_guard.value.end {
                // This is the special "end node". It doesn't represent tokens itself,
                // but it is the source of "liveness". The tokens are on the edges leading *to* it.
                // When we calculate the live tokens for a parent, the edge BV leading to this
                // end node will be considered fully live. For the end node itself, we can
                // consider it to represent "all possible tokens" for the purpose of intersection,
                // so that any edge leading to it is kept.
                live_tokens_for_this_node = self.all_llm_tokens.clone();
            }

            node_guard.children_mut().retain(|_edge_key, dest_map| {
                dest_map.retain(|child_wrapper, edge_value_bv| {
                    // Get the live tokens reachable from the child node. This must be in the cache.
                    let live_tokens_from_child = live_tokens_cache.get(child_wrapper)
                        .expect("Child not found in live_tokens_cache. Logic error in post-order traversal.");

                    // The tokens on this edge that are actually live are the intersection
                    // of the edge's original tokens and the live tokens from the child.
                    let live_tokens_for_this_edge = &*edge_value_bv & live_tokens_from_child;

                    if live_tokens_for_this_edge.is_empty() {
                        false // Prune this destination, as no live paths go through it.
                    } else {
                        *edge_value_bv = live_tokens_for_this_edge; // Narrow the edge's BV.
                        true // Keep this destination.
                    }
                });
                // Keep the edge key only if it still has destinations.
                !dest_map.is_empty()
            });

            // The total live tokens for the current node are the union of all its (now narrowed) outgoing edge BVs.
            for dest_map in node_guard.children().values() {
                for edge_bv in dest_map.values() {
                    live_tokens_for_this_node |= edge_bv;
                }
            }
            // Update the node's own live_tokens field
            node_guard.value.live_tokens = live_tokens_for_this_node.clone();
        }

        // Update the cache with the final computed live tokens for this node.
        live_tokens_cache.insert(node_wrapper, live_tokens_for_this_node.clone());

        live_tokens_for_this_node
    }

    fn factor_common_destinations(&mut self) {
        crate::debug!(2, "Factoring out common destinations to reduce non-None edges.");

        const MIN_INCOMING_EDGES_FOR_FACTORING: usize = 3; // Configurable threshold

        // 1. Collect all nodes in the graph.
        let roots_vec: Vec<_> = self.roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(&self.trie1_god, &roots_vec);
        let arc_map: HashMap<_, _> = all_nodes.iter().map(|n| (n, n.clone())).collect();

        // 2. Build an incoming edge map for every node.
        // incoming_map: D_ptr -> (gtid -> Vec<(S_ptr, bv)>)
        let mut incoming_map: HashMap<
            PrecomputeNodeIndex, // Dst node ptr
            HashMap<
                GrammarTokenID, // Edge key 'gtid'
                Vec<(PrecomputeNodeIndex, LLMTokenBV)>, // List of (Src node ptr, edge bv)
            >,
        > = HashMap::new();

        for src_arc in &all_nodes {
            let src_ptr = src_arc;
            let guard = src_arc.read(&self.trie1_god).expect("poison");
            for (ek_opt, dest_map) in guard.children() {
                if let Some(gtid) = ek_opt { // Only consider non-None edges
                    for (dest_wrapper, bv) in dest_map {
                        let dest_arc = dest_wrapper.as_arc();
                        let dest_ptr = dest_arc;
                        incoming_map.entry(*dest_ptr).or_default().entry(*gtid).or_default().push((*src_ptr, bv.clone()));
                    }
                }
            }
        }

        // 3. Iterate through the map and find factoring opportunities.
        for (dest_ptr, edges_by_key) in incoming_map {
            for (gtid, sources) in edges_by_key {
                if sources.len() >= MIN_INCOMING_EDGES_FOR_FACTORING {
                    // Opportunity found!
                    let dest_arc = arc_map.get(&dest_ptr).unwrap().clone();

                    // a. Create a new intermediate node `I`.
                    let intermediate_node = PrecomputeNodeIndex::new(self.trie1_god.insert(PrecomputeNode::new(PrecomputedNodeContents::internal())));

                    // b. Add edge I --(gtid)--> D
                    let mut union_bv = LLMTokenBV::zeros();
                    for (_, bv) in &sources {
                        union_bv |= bv;
                    }

                    {
                        let mut intermediate_guard = intermediate_node.write(&self.trie1_god).expect("poison");
                        let mut edge_val_opt = Some(union_bv.clone());
                        // No cycle possible since I is new. Use unchecked for speed.
                        // Depth will be propagated to D.
                        intermediate_guard.try_insert_unchecked(Some(gtid), &mut edge_val_opt, dest_arc.clone());
                        intermediate_guard.value.live_tokens |= &union_bv; // Update live_tokens for intermediate node
                    }

                    // c. For each source, remove old edge and add new `None` edge to `I`.
                    for (src_ptr, bv) in &sources {
                        let src_arc = arc_map.get(src_ptr).unwrap();
                        let mut src_guard = src_arc.write(&self.trie1_god).expect("poison");

                        // Remove S --(gtid)--> D
                        if let Some(dest_map_for_gtid) = src_guard.children_mut().get_mut(&Some(gtid)) {
                            dest_map_for_gtid.remove(&dest_arc.clone());
                            if dest_map_for_gtid.is_empty() {
                                src_guard.children_mut().remove(&Some(gtid));
                            }
                        }

                        // Add S --(None)--> I
                        let mut edge_val_opt = Some(bv.clone());
                        src_guard.try_insert_unchecked(None, &mut edge_val_opt, intermediate_node.clone());
                        src_guard.value.live_tokens |= bv; // Update live_tokens for source node
                    }
                }
            }
        }
        crate::debug!(2, "Finished factoring common destinations.");
    }

    fn merge_nodes(&mut self) {
        crate::debug!(2, "Merging identical subtrees in precomputed trie.");
        // A map from a node's content to its canonical Arc.
        let mut canonical_nodes: HashMap<PrecomputeNode, PrecomputeNodeIndex> = HashMap::new();
        // A map from a node's pointer to its canonicalized Arc, to avoid re-processing.
        let mut visited: HashMap<PrecomputeNodeIndex, PrecomputeNodeIndex> = HashMap::new();

        // We need to process all roots.
        let mut new_roots = BTreeMap::new();
        for (sid, root_arc) in self.roots.iter() {
            let canonical_root = self.deduplicate_recursive(root_arc.clone(), &mut canonical_nodes, &mut visited);
            new_roots.insert(*sid, canonical_root);
        }
        self.roots = new_roots;
        crate::debug!(2, "Finished merging subtrees. Canonical nodes: {}", canonical_nodes.len());
    }

    fn deduplicate_recursive(
        &self,
        node_arc: PrecomputeNodeIndex,
        canonical_nodes: &mut HashMap<PrecomputeNode, PrecomputeNodeIndex>,
        visited: &mut HashMap<PrecomputeNodeIndex, PrecomputeNodeIndex>,
    ) -> PrecomputeNodeIndex {
        let node_ptr = node_arc;
        if let Some(canonical_arc) = visited.get(&node_ptr) {
            return canonical_arc.clone();
        }

        // Pre-emptively insert to break cycles.
        visited.insert(node_ptr, node_arc.clone());

        // Post-order traversal: first, canonicalize all children.
        let mut new_children_map = BTreeMap::new();
        let mut children_changed = false;

        {
            let node_guard = node_arc.read(&self.trie1_god).unwrap();
        for (edge_key, dest_map) in node_guard.children() {
            let mut new_dest_map = OrderedHashMap::new();
            for (node_ptr_wrapper, edge_val) in dest_map.iter() {
                let child_arc = node_ptr_wrapper.as_arc().clone();
                let canonical_child_arc = self.deduplicate_recursive(child_arc.clone(), canonical_nodes, visited);
                if &child_arc != &canonical_child_arc {
                    children_changed = true;
                }
                let new_node_ptr_wrapper = canonical_child_arc;
                new_dest_map.insert(new_node_ptr_wrapper, edge_val.clone());
            }
            if !new_dest_map.is_empty() {
                new_children_map.insert(edge_key.clone(), new_dest_map);
                }
            }
        }

    if children_changed {
        let mut node_guard = node_arc.write(&self.trie1_god).unwrap();
        *node_guard.children_mut() = new_children_map;
        node_guard.recompute_max_depth(&self.trie1_god);
        // The live_tokens field will be recomputed by prune_dead_paths after merging.
    }

    let canonical_arc = {
            let node_guard = node_arc.read(&self.trie1_god).unwrap();
            let node_content = (*node_guard).clone();
            canonical_nodes.entry(node_content).or_insert_with(|| node_arc.clone()).clone()
        };

        // Update with the final canonical arc.
        visited.insert(node_ptr, canonical_arc.clone());
        canonical_arc
    }

    pub fn gc(&mut self) {
        crate::debug!(2, "Running garbage collection on precomputed trie.");
        let roots: Vec<_> = self.roots.values().cloned().collect();
        Trie::gc(&self.trie1_god, &roots);
    }

    fn finish(
        mut self,
        token_name_map: &BiBTreeMap<Terminal, usize>,
        possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
        internal_max_llm_token: usize,
    ) -> (BTreeMap<TokenizerStateID, PrecomputeNodeIndex>, Trie1GodWrapper) {

        calculate_final_stats(&self.roots, &mut self.stats, &self.trie1_god);
        print_precompute_stats(&self.stats, token_name_map, &self.trie1_god);

        (self.roots, self.trie1_god)
    }

    fn dfs(
        &self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<TokenizerStateID, OrderedHashSet<PrecomputeNodeIndex>>,
    ) {
        self.pb.inc(1);

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let mut work_queue: BTreeMap<
                usize,
                BTreeMap<TokenizerStateID, OrderedHashSet<PrecomputeNodeIndex>>,
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

                        // TODO: could make this so much faster by moving loop down...
                        for src_node_wrapper in &precompute_nodes {
                            if next_pos == segment_bytes.len() {
                                // TODO: should be some way of avoiding ignored terminal here.
                                let llm_token_id = child_vocab_node.token_id();
                                let mut edge_bv = HybridBitset::zeros();
                                edge_bv.insert(llm_token_id);
                                let mut inserter = EdgeInserter::new(
                                    &self.trie1_god,
                                    src_node_wrapper.as_arc().clone(),
                                    Some(terminal_id),
                                    edge_bv,
                                    |e, n| *e |= n,
                                    |node_value, edge_value| {
                                        crate::debug!(7, "Before updating live tokens {:?} |= {:?}", node_value.live_tokens, edge_value);
                                        node_value.live_tokens |= edge_value;
                                        crate::debug!(7, "After updating live tokens: {:?}", node_value.live_tokens);
                                    },
                                    |ev, t| *ev &= &t.live_tokens,
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
                                &self.trie1_god,
                                src_node_wrapper.as_arc().clone(),
                                Some(terminal_id),
                                edge_bv.clone(),
                                |e, n| *e |= n,
                                |node_value, edge_value| node_value.live_tokens |= edge_value,
                                |ev, t| *ev &= &t.live_tokens,
                            );

                            let next_tokenizer_state = self.tokenizer.initial_state_id();
                            let dest_nodes_in_queue = work_queue.entry(next_pos).or_default().entry(next_tokenizer_state).or_default();

                            inserter = inserter.try_destinations_iter(dest_nodes_in_queue.iter().map(|w| w.as_arc().clone()).filter(|w| !w.read(&self.trie1_god).unwrap().value.end));

                            if true {
                                let children_of_src: Vec<_> = src_node_wrapper.as_arc().read(&self.trie1_god).unwrap().children().values().flat_map(|m| m.keys().cloned()).collect();
                                // let tags = self.tags.borrow(); // Removed
                                let eligible_children = children_of_src.iter().map(|child_node_ptr| {
                                    child_node_ptr.as_arc().clone()
                                }).filter(|child_arc| {
                                    (child_arc.read(&self.trie1_god).unwrap().value.live_tokens.clone() & &edge_bv).is_empty() && !child_arc.read(&self.trie1_god).unwrap().value.end
                                });
                                inserter = inserter.try_destinations_iter(eligible_children);
                                // drop(tags); // Removed
                            }

                            let result_node = inserter.else_create_destination_with_value(PrecomputedNodeContents::internal()).unwrap();
                            let result_node_ptr = result_node.clone();
                            dest_nodes_in_queue.insert(result_node_ptr.clone());
                            // *self.tags.borrow_mut().entry(result_node_ptr).or_insert_with(HybridBitset::zeros) |= &edge_bv; // Removed
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
                                    &self.trie1_god,
                                    src_node_wrapper.as_arc().clone(),
                                    Some(terminal_id),
                                    edge_bv,
                                    |e, n| *e |= n,
                                    |node_value, edge_value| node_value.live_tokens |= edge_value,
                                    |ev, t| *ev &= &t.live_tokens,
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParseStateKey {
    stack_state_id: StateID,
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
    pub parent: &'a GrammarConstraint,
    pub state:  BTreeMap<TokenizerStateID, GLRParserState<'a>>,
}

pub type Trie1GodWrapper = GodWrapper<Option<TerminalID>, HybridBitset, PrecomputedNodeContents>;
pub type Trie1God = God<Option<TerminalID>, HybridBitset, PrecomputedNodeContents>;
pub type Trie2GodWrapper = GodWrapper<(usize, Option<StateID>), HybridBitset, PrecomputedNodeContents>;
pub type Trie2God = God<(usize, Option<StateID>), HybridBitset, PrecomputedNodeContents>;
pub type Trie3GodWrapper = GodWrapper<(usize, LLMTokenBV), StateIDBV, PrecomputedNode3Contents>;
pub type Trie3God = God<(usize, LLMTokenBV), StateIDBV, PrecomputedNode3Contents>;

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
        return HybridBitset::ones(self.parent.llm_vocab.max_original_llm_token_id + 1); // TEMP
        // self.get_mask1()
        // self.get_mask2()
        self.get_mask3()
    }

    #[time_it]
    pub fn get_mask1(&self) -> LLMTokenBV {
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

        let mut initial_values_for_map: Vec<(PrecomputeNodeIndex, GLRParserState<'a>)> = Vec::new();
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
            &self.parent.trie1_god,
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
                for child_node_trie_data in dest_map.keys() {
                    if child_node_trie_data.as_arc().read(&self.parent.trie1_god).unwrap().value.end {
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
                                        crate::debug!(4, "Step with grammar token {:?} ({}) has action, but all children are end nodes, so we can skip stepping and update final mask directly.", gtid, self.parent.parser.terminal_map.get_by_right(gtid).map_or("UNKNOWN_TERMINAL".to_string(), |s| s.to_string()));
                                        let mut edge_llm_tokens = HybridBitset::zeros();
                                        for edge_llm_tokens_bv in dest_map.values() {
                                            edge_llm_tokens |= edge_llm_tokens_bv;
                                        }
                                        let llm_tokens = &glr_s_llm_tokens & &edge_llm_tokens;
                                        crate::debug!(4, "Adding active tokens {:?} to final mask", llm_tokens);
                                        *final_mask_internal.borrow_mut() |= llm_tokens;
                                        crate::debug!(4, "Final mask after adding tokens: {:?}", final_mask_internal.borrow());
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

                    let mut glr_s = glr_s.clone();

                    if let Some(gtid) = grammar_token_opt {
                        let mut counts_guard = step_counts_clone1.write().unwrap();
                        let entry = counts_guard.entry(*gtid).or_default();
                        entry.total += 1;

                        let terminal_name = self.parent.parser.terminal_map.get_by_right(gtid)
                            .map(|s| s.to_string())
                            .unwrap_or("UNKNOWN_TERMINAL".to_string());
                        // timeit!(format!("get_mask step for terminal '{}'", terminal_name), {
                        glr_s.process_token(*gtid);
                        // });

                        crate::debug!(4, "glr_s.is_ok()_after_process_token: {}", glr_s.is_ok());

                        if glr_s.is_ok() {
                            entry.successful += 1;
                        } else {
                            return Vec::new();

                        }
                    }

                    // glr_s.log_gss("After stepping", grammar_token_opt.unwrap_or(TerminalID(0)));
                    // disallow_llm_tokens_and_prune_arc(&mut glr_s.active_state.stack, &final_mask_internal.borrow(), &mut HashMap::new());

                    let mut results = Vec::new();

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

                        if child_node_trie_data.as_arc().read(&self.parent.trie1_god).unwrap().value.end {
                            let glr_active_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                            crate::debug!(4, "Adding active tokens {:?} to final mask", glr_active_tokens);
                            // timeit!("get_mask final_mask update", {
                            *final_mask_internal.borrow_mut() |= glr_active_tokens;
                            // });
                            crate::debug!(4, "Final mask after adding end node tokens: {:?}", final_mask_internal.borrow());
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
                        crate::debug!(4, "Precomputed node data is an end node, adding active tokens {:?} to final mask", glr_active_tokens);
                        *final_mask_internal.borrow_mut() |= glr_active_tokens;
                        crate::debug!(4, "Final mask after adding end node tokens: {:?}", final_mask_internal.borrow());
                        false
                    } else {
                        let mut num_outgoing_edges_that_lead_to_non_end_nodes = 0;
                        for (edge_terminal_opt, dest_map) in precomputed_node_data.children().iter() {
                            if edge_terminal_opt.is_none() {
                                num_outgoing_edges_that_lead_to_non_end_nodes += 1
                            } else {
                                for (child_node_trie_data, _edge_llm_tokens_bv) in dest_map.iter() {
                                    if !child_node_trie_data.as_arc().read(&self.parent.trie1_god).unwrap().value.end {
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
                        let stats = gather_gss_stats(&[glr_s.active_state.stack.as_ref()]);
                        // crate::debug!(3, "GSS stats for precomputed node data: {:#?}", stats);
                        let mut do_phase3 = false;
                        do_phase3 |= num_outgoing_edges_that_lead_to_non_end_nodes >= 2;
                        do_phase3 |= match LR_MODE {
                            LRMode::LR1 | LRMode::LALR_EX_SHIFT_STATES => false,
                            LRMode::LALR => true,
                        };
                        // do_phase3 |= true;
                        if do_phase3 {
                            // There will be a split.
                            // Let's do some work ahead of time to avoid redundant computations due to the upcoming split.
                            crate::debug!(4, "Processing non-end precomputed node data");
                            crate::debug!(4, "Active LLM tokens before phase 3: {:?}", glr_s.active_state.stack.allowed_llm_tokens());

                            let mut allowed_terminals = TerminalBV::zeros();
                            for gtid_opt in precomputed_node_data.children().keys() {
                                if let Some(gtid) = gtid_opt {
                                    allowed_terminals.insert(gtid.0);
                                }
                            }
                            let disallowed_terminals_bv = allowed_terminals.inverted();
                            if !disallowed_terminals_bv.is_empty() {
                                let disallowed_l2 = crate::datastructures::hybrid_l2_bitset::HybridL2Bitset::from_iter(
                                    std::iter::once((0..=usize::MAX, disallowed_terminals_bv))
                                );
                                disallow_terminals_and_prune_arc(&mut glr_s.active_state.stack, &disallowed_l2, &mut HashMap::new());
                            }

                            glr_s.process_default_reductions();
                            crate::debug!(4, "After phase 3, active stack.stack.is_empty(): {}", glr_s.active_state.stack.is_empty());
                            Arc::make_mut(&mut glr_s.active_state.stack).fuse_predecessors(1);
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

        let t_after_special_map = std::time::Instant::now();
        println!("after special_map: {:>15?}", t_after_special_map.duration_since(t0));

        crate::profiler::print_summary_flat();

        let counts = step_counts.read().unwrap();
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

        let final_mask_mapped = self.parent.internal_bv_to_original(&final_mask_internal.into_inner());

        let t_end = std::time::Instant::now();
        println!("get_mask took: {:>15?}", t_end.duration_since(t0));

        final_mask_mapped
    }

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

        let mut initial_values_for_map: Vec<(Trie2Index, GLRParserState<'a>)> = Vec::new();
        for (tokenizer_state_id, glr_state) in &self.state {
            // crate::debug!(4, "Initializing GSS for state {}", tokenizer_state_id.0);
            // Ensure the GLR state's GSS stack is not empty before proceeding
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            if let Some(precomputed_trie_root_arc) = self.parent.precomputed2.get(tokenizer_state_id) {
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
            &self.parent.trie2_god,
            initial_values_for_map,
            // step_fn: (current_glr_state, (k, option state ID), destinations_map)
            |glr_s, (k, expected_state_id_opt ), dest_map| {
                // if !glr_s.is_ok() {
                //     crate::debug!(4, "GLR state is not alive before popping, skipping.");
                //     return Vec::new();
                // }
                crate::debug!(4, "Processing step for k: {:?}, expected_state_id_opt: {:?}", k, expected_state_id_opt);
                // glr_s.log_gss("Before popping", TerminalID(0), false, false);
                let mut out_gsss = Vec::new();
                let popped = glr_s.active_state.stack.popn(*k);
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
                // if !out_gss.is_alive() {
                //     crate::debug!(4, "GLR state is not alive after popping, skipping.");
                //     return Vec::new();
                // }
                let mut out = Vec::new();
                for (dst_node_wrapper, edge_bv) in dest_map.iter() {
                    let mut out_gss_filtered = out_gss.clone();
                    crate::debug!(5, "Filtering GSS for edge LLM tokens: {:?}", edge_bv);
                    allow_only_llm_tokens_and_prune_arc(&mut out_gss_filtered, edge_bv, &mut HashMap::new());
                    let mut out_glr_s = glr_s.clone();
                    out_glr_s.active_state.stack = out_gss_filtered;
                    crate::debug!(4, "Allowed LLM tokens in out_gss_filtered: {:?}", out_glr_s.active_state.stack.allowed_llm_tokens());
                    // out_glr_s.log_gss("After filtering for edge LLM tokens", TerminalID(0), false, false);
                    // if out_glr_s.is_ok() {
                        out.push((dst_node_wrapper.clone(), out_glr_s));
                    }
                // }
                out
            },
            // merge_fn
            |glr_s1, glr_s2| {
                crate::debug!(4, "Merging two GLR states");
                glr_s1.merge_with(glr_s2);
            },
            // process_fn: (precomputed_node_data, final_glr_s_for_this_path)
            |precomputed_node_data, glr_s| {
                crate::debug!(4, "Processing node {:p}", precomputed_node_data);
                // glr_s.log_gss("At process_fn", TerminalID(0), false, false);
                let glr_active_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                // let keep_going = !glr_active_tokens.is_empty();
                let keep_going = glr_s.is_ok();
                if precomputed_node_data.value.end {
                    crate::debug!(4, "Precomputed node data is an end node, adding active tokens {:?} to final mask", glr_active_tokens);
                    *final_mask_internal.borrow_mut() |= glr_active_tokens;
                } else {
                    crate::debug!(4, "Precomputed node data is not an end node, active tokens: {:?}", glr_active_tokens);
                }
                keep_going
            },
        );

        let t_after_special_map = std::time::Instant::now();
        println!("after special_map: {:>15?}", t_after_special_map.duration_since(t0));

        crate::profiler::print_summary_flat();

        let counts = step_counts.read().unwrap();
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

    pub fn print_gss_stats(&self) {
        let stats = gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        );
        crate::debug!(2, "GSS stats: {:#?}", stats);
    }

    pub fn print_gss(&self) {
        let roots: Vec<_> = self.state.values().map(|s| s.active_state.stack.clone()).collect();
        if roots.is_empty() {
            println!("GSS is empty.");
            return;
        }

        let labels: Vec<_> = self.state.keys().map(|k| format!("Tokenizer State {}", k.0)).collect();

        let config = GSSPrintConfig {
            labels: Some(&labels),
            max_edges: 500,
            original_internal_bimap: Some(&self.parent.llm_vocab.original_to_internal_id_bimap),
            llm_token_map: Some(&self.parent.llm_vocab.llm_token_map),
            verbose: false,
        };

        let (gss_str, state_ids) = print_gss_forest(&roots, &self.parent.parser.terminal_map, &config);
        println!("{}", gss_str);

        if !state_ids.is_empty() {
            println!("\n--- GSS State Explanations ---");
            for state_id in state_ids {
                let mut explanation = String::new();
                println!("\n--- State {} ---", state_id.0);
                self.parent.parser.format_state_details(&mut explanation, state_id, "  ").unwrap();
                println!("{}", explanation);
            }
        }
    }

    pub fn explain_stack(&self) {
        for (state_id, state) in &self.state {
            println!("\n--- State {} ---", state_id.0);
            // Sample and print a bunch of stacks
            let mut seen = BTreeSet::new();
            let num_to_sample = 10;
            for i in 0..1000 {
                if let Some(sampled_path_edges) = sample_path(&[&state.active_state.stack], i) {
                    let mut sampled_stack: Vec<StateID> = sampled_path_edges.iter()
                        .map(|edge| edge.state_id)
                        .collect();
                    sampled_stack.reverse();
                    if seen.contains(&sampled_stack) {
                        continue;
                    }
                    println!("  Sampled stack {}: {:?}", i + 1, sampled_stack);
                    seen.insert(sampled_stack);
                    if seen.len() >= num_to_sample {
                        break;
                    }
                };
            }
            // Sample a stack
            if let Some(sampled_path_edges) = sample_path(&[&state.active_state.stack], 1) {
                let mut sampled_stack: Vec<StateID> = sampled_path_edges.iter()
                    .map(|edge| edge.state_id)
                    .collect();
                sampled_stack.reverse();
                let explanation = self.parent.parser.explain_stack(&sampled_stack);
                // Indent the explanation for readability
                for line in explanation.lines() {
                    println!("      {}", line);
                }
            };
        }
    }

    pub fn get_mask3(&self) -> LLMTokenBV {
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

        let mut initial_values_for_map: Vec<(PrecomputeNode3Index, GLRParserState<'a>)> = Vec::new();
        for (tokenizer_state_id, glr_state) in &self.state {
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            if let Some(precomputed_trie_root_arc) = self.parent.precomputed3.get(tokenizer_state_id) {
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
                initial_values_for_map.push((precomputed_trie_root_arc.clone(), glr_state));
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

        crate::profiler::reset();

        Trie::special_map_grouped(
            &self.parent.trie3_god,
            initial_values_for_map,
            // step_fn: (current_glr_state, (pop, llm_token_bv), destinations_map)
            |glr_s, (pop, llm_token_bv), dest_map| {
                let popped = glr_s.active_state.stack.popn(*pop);
                let mut results = Vec::new();

                for (dest_idx, state_id_bv) in dest_map.iter() {
                    let mut valid_gss_nodes = Vec::new();
                    for popper_item in popped.iter() {
                        for peek in popper_item.peek_iter() {
                            if state_id_bv.contains(peek.edge_value().state_id.0) {
                                valid_gss_nodes.push(peek.isolated_parent());
                            }
                        }
                    }

                    if valid_gss_nodes.is_empty() {
                        continue;
                    }

                    let merged_gss = GSSNode::merge_many_with_depth(1, valid_gss_nodes);
                    let mut new_glr_s = glr_s.clone();
                    new_glr_s.active_state.stack = merged_gss;

                    allow_only_llm_tokens_and_prune_arc(&mut new_glr_s.active_state.stack, llm_token_bv, &mut HashMap::new());

                    if new_glr_s.is_ok() {
                        results.push((dest_idx.clone(), new_glr_s));
                    }
                }
                results
            },
            // merge_fn
            |glr_s1, glr_s2| {
                glr_s1.merge_with(glr_s2);
            },
            // process_fn: (precomputed_node_data, final_glr_s_for_this_path)
            |precomputed_node_data, glr_s| {
                let glr_active_tokens = glr_s.active_state.stack.allowed_llm_tokens();
                let keep_going = glr_s.is_ok();
                if precomputed_node_data.value.end {
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

        crate::debug!(3, "Committing bytes: {:?}", String::from_utf8_lossy(llm_token_bytes));

        // for (state_id, state) in &self.state {
        //     crate::debug!(3, "State {} before commit:", state_id.0);
        //     state.log_gss("Before commit", TerminalID(0), false, false);
        // }

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
        crate::debug!(5, "Terminals map: {:?}", terminals_map);
        for state in self.state.values_mut() {
            prune_disallowed_terminals(&mut state.active_state.stack, &terminals_map, &mut gss_transformation_memo);
        }
        gss_transformation_memo.clear();
        let gss_stats_after_pruning = gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        );
        crate::debug!(3, "GSS stats before pruning disallowed terminals: {:#?}", gss_stats_before_pruning);
        if gss_stats_after_pruning != gss_stats_before_pruning {
            crate::debug!(3, "GSS stats after pruning disallowed terminals: {:#?}", gss_stats_after_pruning);
            crate::debug!(3, "GSS stats changed after pruning disallowed terminals.");
            self.print_gss();
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
                            let terminals_accessible_from_end_state = self.parent.tokenizer.tokens_accessible_from_state(TokenizerStateID(end_state_id));
                            if terminals_accessible_from_end_state.contains(&TerminalID(match_info.id)) {
                                let mut disallowed_terminals_for_end_state = TerminalBV::zeros();
                                // Disallow this token from being matched again immediately.
                                disallowed_terminals_for_end_state.insert(match_info.id);
                                disallowed_terminals.insert_l2_bitset(end_state_id, disallowed_terminals_for_end_state);
                            }
                        }
                        // cloned_glr_s.log_gss(format!("Before disallowing terminals {:?} after committing bytes {:?}", &disallowed_terminals, &llm_token_bytes[offset..new_offset]).as_str(), TerminalID(match_info.id), false, false);
                        disallow_terminals_and_prune_arc(&mut cloned_glr_s.active_state.stack, &disallowed_terminals, &mut HashMap::new());
                        // cloned_glr_s.log_gss(format!("After disallowing terminals {:?} after committing bytes {:?}", &disallowed_terminals, &llm_token_bytes[offset..new_offset]).as_str(), TerminalID(match_info.id), false, false);

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
            // glr_parser_state.process_default_reductions();
        }

        // TODO: this shouldn't be necessary, but due to some order-dependent LLM token BV weirdness in GSS, it is necessary to ensure commit order invariance.
        for state in self.state.values_mut() {
            reset_llm_tokens(&mut state.active_state.stack, &mut gss_transformation_memo);
        }
        gss_transformation_memo.clear();

        self.state.retain(|_, glr_parser_state| glr_parser_state.is_ok());

        let mut fuse_memo = HashMap::new();
        for state in self.state.values_mut() {
            state.active_state.stack = fuse_predecessors_recursive(&mut state.active_state.stack, 8, &mut fuse_memo);
        }
        fuse_memo.clear();

        // Post-commit allowance check: ensure each surviving state allows at least one
        // token the tokenizer can produce from its current tokenizer state.
        // Mode is controlled by self.parent.post_commit_allow_check_mode.
        match self.parent.post_commit_allow_check_mode {
            TerminalAllowanceCheckMode::None => {
                // no-op
            }
            TerminalAllowanceCheckMode::ImmediateSets => {
                self.state.retain(|tokenizer_state_id, glr_state| {
                    // Fast auto-pass if tokenizer can produce all grammar terminals.
                    let accessible = self.parent.tokenizer.tokens_accessible_from_state(*tokenizer_state_id);
                    if accessible.len() >= self.parent.parser.terminal_map.len() {
                        return true;
                    }

                    let mut union = glr_state.immediate_shift_terminals();
                    union.extend(glr_state.immediate_reduce_terminals());
                    !union.is_disjoint(&accessible)
                });
            }
            TerminalAllowanceCheckMode::ImmediateProbe => {
                self.state.retain(|tokenizer_state_id, glr_state| {
                    let accessible = self.parent.tokenizer.tokens_accessible_from_state(*tokenizer_state_id);
                    if accessible.len() >= self.parent.parser.terminal_map.len() {
                        return true;
                    }
                    for tid in &accessible {
                        if glr_state.has_immediate_action_for_terminal(*tid).unwrap_or(false) {
                            return true;
                        }
                    }
                    false
                });
            }
            TerminalAllowanceCheckMode::StepProbe => {
                self.state.retain(|tokenizer_state_id, glr_state| {
                    let accessible = self.parent.tokenizer.tokens_accessible_from_state(*tokenizer_state_id);
                    if accessible.len() >= self.parent.parser.terminal_map.len() {
                        return true;
                    }
                    for tid in &accessible {
                        if glr_state.allows_terminal(*tid) {
                            return true;
                        }
                    }
                    false
                });
            }
        }

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

        crate::debug!(4, "Active tokenizer states after committing text (bytes {:?}): {:?}", llm_token_bytes, self.state.keys().map(|k|k.0).collect::<Vec<_>>());
        for (tokenizer_id, glr_state) in &self.state {
            if !glr_state.active_state.stack.is_empty() { // Log only for non-empty GSS
                // glr_state.log_gss("After commit", TerminalID(0), false, false);
            }
        }
    }

    pub fn is_active(&self) -> bool {
        !self.state.is_empty()
    }

    pub fn state(&self) -> &BTreeMap<TokenizerStateID, GLRParserState<'a>> {
        &self.state
    }
}
