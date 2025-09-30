// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use crate::datastructures::hybrid_l2_bitset;
use std::collections::btree_map::Entry as BTreeEntry;
use crate::datastructures::gss::{disallow_llm_tokens_and_prune_arc, fuse_predecessors_recursive, get_roots, print_gss_forest, prune_llm_tokens_by_disallowed_terminals, reset_terminals, sample_path, simplify, simplify_roots_in_place};
use crate::datastructures::gss::{map_allowed_terminals_tokenizer_states, prune_disallowed_terminals};
use crate::datastructures::ordered_hash_map::Retain;
use ordered_hash_map::OrderedHashMap;
use ordered_hash_map::OrderedHashSet;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::env;
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
use crate::constraint_precompute1_utils;
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
use crate::glr::parser::{BelowBottomReductionMode, ExpectElse, GLRParser, GLRParserState, ParseState, ParseStateEdgeContent, ProcessDefaultReductionsAdvancedConfig, ProcessTokenAdvancedConfig};
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
pub(crate) use crate::constraint::constraint_precompute3_utils::clone_trie3_graph;
use crate::constraint_precompute3_utils::optimize_trie3_size;
use crate::datastructures::trie::{God, GodWrapper};

const MERGE_THRESHOLD: usize = 20;

pub type StateIDBV = HybridBitset;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminalAllowanceCheckMode {
    None,
    ImmediateSets,
    ImmediateProbe,
    #[default]
    StepProbe,
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

pub type PrecomputeNode1 = Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;
pub type PrecomputeNode2 = Trie<(usize, Option<StateID>), LLMTokenBV, PrecomputedNodeContents>;
pub type PrecomputeNode3 = Trie<(usize, LLMTokenBV), StateIDBV, PrecomputedNodeContents>;
// New: PrecomputeNode0 with (grammar token, optional (tokenizer end state, disallowed terminal))
pub type PrecomputeNode0 = Trie<(Option<GrammarTokenID>, Option<(TokenizerStateID, TerminalID)>), LLMTokenBV, PrecomputedNodeContents>;

pub type PrecomputeNode0Index = Trie2Index;
pub type PrecomputeNode1Index = Trie2Index;
pub type PrecomputeNode2Index = Trie2Index;
pub type PrecomputeNode3Index = Trie2Index;

pub type Precomputed0 = BTreeMap<TokenizerStateID, PrecomputeNode0Index>;
pub type Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode1Index>;
pub type Precomputed2 = BTreeMap<TokenizerStateID, PrecomputeNode2Index>;
pub type Precomputed3 = BTreeMap<TokenizerStateID, PrecomputeNode3Index>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LLMVocab {
    pub llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID>,
    pub max_original_llm_token_id: usize,
    // One-to-one original->internal index mapping for the baseline/global view.
	// Note: this is a simple mapping (not a BiBTreeMap) so we can evolve stage-local
	// vocabularies independently. Use internal_to_original_ for reverse lookups.
	pub original_to_internal_id_bimap: BTreeMap<usize, usize>,
	// Reverse mapping (many-to-one support): internal index -> set of original ids
	pub internal_to_original_: BTreeMap<usize, LLMTokenBV>,
	pub internal_max_llm_token: usize
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageVocab {
	pub original_to_internal: BTreeMap<usize, usize>,
	pub internal_to_original: BTreeMap<usize, LLMTokenBV>,
	pub internal_max_llm_token: usize,
}


impl JSONConvertible for StageVocab {
    fn to_json(&self) -> JSONNode {
        let mut m = StdMap::new();
		m.insert("original_to_internal".to_string(), self.original_to_internal.to_json());
		// Serialize internal_to_original as Vec<(usize, Vec<usize>)> to keep it compact
		let mut ito: Vec<(usize, Vec<usize>)> = Vec::new();
		for (k, bv) in &self.internal_to_original {
			ito.push((*k, bv.iter().collect::<Vec<_>>()));
		}
		m.insert("internal_to_original".to_string(), ito.to_json());
		m.insert("internal_max_llm_token".to_string(), self.internal_max_llm_token.to_json());
		JSONNode::Object(m)
	}
	fn from_json(node: JSONNode) -> Result<Self, String> {
		match node {
			JSONNode::Object(mut obj) => {
				let original_to_internal = obj.remove("original_to_internal").ok_or("StageVocab: missing original_to_internal".to_string()).and_then(|n| BTreeMap::<usize, usize>::from_json(n))?;
				let internal_max_llm_token = obj.remove("internal_max_llm_token").ok_or("StageVocab: missing internal_max_llm_token".to_string()).and_then(usize::from_json)?;
				let ito_vec: Vec<(usize, Vec<usize>)> = obj.remove("internal_to_original").ok_or("StageVocab: missing internal_to_original".to_string()).and_then(|n| Vec::from_json(n))?;
				let internal_to_original: BTreeMap<usize, LLMTokenBV> = ito_vec
					.into_iter()
					.map(|(k, v)| (k, v.into_iter().collect()))
					.collect();
				Ok(StageVocab { original_to_internal, internal_to_original, internal_max_llm_token })
			}
			_ => Err("StageVocab: expected object".to_string())
        }
    }
}

#[derive(Debug, Clone)]
pub struct GrammarConstraintConfig {
    pub optimize_trie2_prune_dead_paths: bool,
    pub optimize_trie2_merge_nodes: bool,
    pub optimize_trie2_factor_common_destinations: bool,
    pub optimize_trie2_compress_edges: bool,
    pub optimize_trie2_gc: bool,
    pub skip_precomputation: bool,
    pub optimize_trie3_constrain_bitvecs: bool,
    // Stage-level token optimizations (disabled by default to avoid changing
    // global token-ID semantics until explicitly enabled).
    pub optimize_trie1_merge_equivalent_llm_tokens: bool,
    pub optimize_trie1_reorder_llm_tokens: bool,
    pub optimize_trie3_merge_equivalent_llm_tokens: bool,
    pub optimize_trie3_reorder_llm_tokens: bool,
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
            optimize_trie2_compress_edges: true,
            optimize_trie2_gc: true,
            skip_precomputation: false,
            optimize_trie3_constrain_bitvecs: true,
            optimize_trie1_merge_equivalent_llm_tokens: true,
            optimize_trie1_reorder_llm_tokens: true,
            optimize_trie3_merge_equivalent_llm_tokens: true,
            optimize_trie3_reorder_llm_tokens: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GrammarConstraint {
    pub tokenizer:        Regex,
    pub parser:           GLRParser,
    pub(crate) precomputed:      Precomputed,
    pub precomputed2:     Precomputed2,
    pub precomputed3:     Precomputed3,
    // New: store precompute-0 too
    pub precomputed0:     Precomputed0,
    pub llm_vocab:        Arc<LLMVocab>,
    pub(crate) token_name_map:   BiBTreeMap<Terminal, usize>,
    pub possible_matches: BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    pub(crate) trie1_god: Trie1GodWrapper,
    pub trie2_god: Trie2GodWrapper,
    pub trie3_god: Trie3GodWrapper,
    pub post_commit_allow_check_mode: TerminalAllowanceCheckMode,
    // Stage-local vocabularies for internal<->original mappings
    pub trie0_god: Trie0GodWrapper,
    pub precompute_vocab: StageVocab,
    pub precompute2_vocab: StageVocab,
    pub precompute3_vocab: StageVocab,
}

impl GrammarConstraint {
    pub fn assert_eq(&self, other: &Self) {
        assert_eq!(self.tokenizer, other.tokenizer);
        assert_eq!(self.parser, other.parser);
        assert_eq!(self.precomputed.len(), other.precomputed.len());
        for ((sid1, arc1), (sid2, arc2)) in self.precomputed.iter().zip(other.precomputed.iter()) {
            assert_eq!(sid1, sid2);
            assert!(PrecomputeNode1::are_graphs_equal(&self.trie1_god, *arc1, &other.trie1_god, *arc2));
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
        assert_eq!(self.precomputed0.len(), other.precomputed0.len());
        for ((sid1, arc1), (sid2, arc2)) in self.precomputed0.iter().zip(other.precomputed0.iter()) {
            assert_eq!(sid1, sid2);
            assert!(PrecomputeNode0::are_graphs_equal(&self.trie0_god, *arc1, &other.trie0_god, *arc2));
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
        obj.insert("precomputed0".to_string(), self.precomputed0.to_json());
        obj.insert("llm_token_map".to_string(), self.llm_vocab.llm_token_map.to_json());
        obj.insert("token_name_map".to_string(), self.token_name_map.to_json());
        obj.insert("max_original_llm_token_id".to_string(), self.llm_vocab.max_original_llm_token_id.to_json());
        obj.insert("original_to_internal_id_bimap".to_string(), self.llm_vocab.original_to_internal_id_bimap.to_json());
        obj.insert("internal_max_llm_token".to_string(), self.llm_vocab.internal_max_llm_token.to_json());
        obj.insert("possible_matches".to_string(), self.possible_matches.to_json());
        obj.insert("trie1_god".to_string(), self.trie1_god.to_json());
        obj.insert("trie2_god".to_string(), self.trie2_god.to_json());
        obj.insert("trie3_god".to_string(), self.trie3_god.to_json());
        obj.insert("trie0_god".to_string(), self.trie0_god.to_json());
        obj.insert("post_commit_allow_check_mode".to_string(), self.post_commit_allow_check_mode.to_json());
        // Stage vocabs
        obj.insert("precompute_vocab".to_string(), self.precompute_vocab.to_json());
        obj.insert("precompute2_vocab".to_string(), self.precompute2_vocab.to_json());
        obj.insert("precompute3_vocab".to_string(), self.precompute3_vocab.to_json());
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
                let precomputed0_opt = obj.remove("precomputed0").map(|n| Precomputed0::from_json(n));

                let llm_token_map = obj.remove("llm_token_map").ok_or_else(|| "Missing field llm_token_map".to_string())
                                       .and_then(|n| BiBTreeMap::<Vec<u8>, LLMTokenID>::from_json(n))?;
                let token_name_map = obj.remove("token_name_map").ok_or_else(|| "Missing field token_name_map".to_string())
                                        .and_then(|n| BiBTreeMap::<Terminal, usize>::from_json(n))?;
                let max_original_llm_token_id = obj.remove("max_original_llm_token_id").ok_or_else(|| "Missing field max_original_llm_token_id".to_string())
                                                   .and_then(usize::from_json)?;
                let original_to_internal_id_bimap = obj.remove("original_to_internal_id_bimap").ok_or_else(|| "Missing field original_to_internal_id_bimap".to_string())
                                                       .and_then(|n| BTreeMap::<usize, usize>::from_json(n))?;
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
                let trie0_god = obj.remove("trie0_god").map(|n| Trie0GodWrapper::from_json(n))
                    .transpose()?.unwrap_or_else(|| Trie0GodWrapper::new());
                let post_commit_allow_check_mode = match obj.remove("post_commit_allow_check_mode") {
                    Some(n) => TerminalAllowanceCheckMode::from_json(n)?,
                    None => TerminalAllowanceCheckMode::default(),
                };
                // Stage vocabs (optional)
                let precompute_vocab = match obj.remove("precompute_vocab") {
					Some(n) => StageVocab::from_json(n)?,
					None => {
						// Synthesize from global mapping
						let mut ito: BTreeMap<usize, LLMTokenBV> = BTreeMap::new();
						for (orig, int_id) in &original_to_internal_id_bimap {
							ito.entry(*int_id).or_default().insert(*orig);
						}
                        StageVocab {
                            original_to_internal: original_to_internal_id_bimap.clone(),
                            internal_to_original: ito.clone(),
                            internal_max_llm_token,
                        }
                    }
                };
                let precompute2_vocab = match obj.remove("precompute2_vocab") {
                    Some(n) => StageVocab::from_json(n)?,
                    None => precompute_vocab.clone(),
                };
                let precompute3_vocab = match obj.remove("precompute3_vocab") {
                    Some(n) => StageVocab::from_json(n)?,
                    None => precompute_vocab.clone(),
				};

				// Build llm_vocab reverse map too
				let mut global_ito: BTreeMap<usize, LLMTokenBV> = BTreeMap::new();
				for (o, i) in &original_to_internal_id_bimap {
					global_ito.entry(*i).or_default().insert(*o);
				}
                Ok(GrammarConstraint {
                    tokenizer,
                    parser,
                    precomputed,
                    precomputed2,
                    precomputed3,
                    llm_vocab: Arc::new(LLMVocab { llm_token_map, max_original_llm_token_id, original_to_internal_id_bimap, internal_to_original_: global_ito, internal_max_llm_token }),
                    token_name_map,
                    possible_matches,
                    trie1_god,
                    trie2_god,
                    trie3_god,
                    precomputed0: {
                        // If precomputed0 was present, use it; otherwise, derive from precomputed1
                        match precomputed0_opt {
                            Some(Ok(p0)) => p0,
                            _ => {
                                let (p0, _t0) = GrammarConstraint::derive_precompute0_from_precompute1(&precomputed, &trie1_god);
                                p0
                            }
                        }
                    },
                    trie0_god: {
                        if trie0_god.is_empty() {
                            let (_p0, t0) = GrammarConstraint::derive_precompute0_from_precompute1(&precomputed, &trie1_god);
                            t0
                        } else {
                            trie0_god
                        }
                    },
                    post_commit_allow_check_mode,
                    precompute_vocab,
                    precompute2_vocab,
                    precompute3_vocab,
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
    ) -> BTreeMap<usize, usize>
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

        let mut original_to_internal_id_bimap = BTreeMap::new();
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
		// Build reverse mapping for global vocab
		let mut internal_to_original_: BTreeMap<usize, LLMTokenBV> = BTreeMap::new();
		for (orig, int_id) in &original_to_internal_id_bimap {
			internal_to_original_.entry(*int_id).or_default().insert(*orig);
		}

        let mut internal_llm_token_map_for_precompute = BiBTreeMap::new();
        for (bytes, original_id) in llm_token_map.iter() {
            if let Some(internal_id_val) = original_to_internal_id_bimap.get(&original_id.0) {
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
            let t1_id = *grammar_term_map.get_by_left(&terminal1).expect_else(|| format!("Terminal {:?} from follow sets not found in grammar_term_map {:?}", terminal1, grammar_term_map));
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
            internal_to_original_: internal_to_original_.clone(),
            internal_max_llm_token,
        });

        // Initialize per-stage vocabularies (start identical to global)
        let mut precompute_vocab = StageVocab {
            original_to_internal: llm_vocab.original_to_internal_id_bimap.clone(),
            internal_to_original: internal_to_original_.clone(),
            internal_max_llm_token: internal_max_llm_token,
        };
        let mut precompute2_vocab = precompute_vocab.clone();
        let mut precompute3_vocab = precompute_vocab.clone();

        if config.skip_precomputation {
            return Self {
                tokenizer,
                parser,
                precomputed: BTreeMap::new(),
                precomputed2: BTreeMap::new(),
                precomputed3: BTreeMap::new(),
                precomputed0: BTreeMap::new(),
                trie0_god: Trie0GodWrapper::new(),
                llm_vocab,
                token_name_map,
                possible_matches: computed_possible_matches,
                trie1_god: Trie1GodWrapper::new(),
                trie2_god: Trie2GodWrapper::new(),
                trie3_god: Trie3GodWrapper::new(),
                post_commit_allow_check_mode: TerminalAllowanceCheckMode::default(),
                precompute_vocab,
                precompute2_vocab,
                precompute3_vocab,
            };
        }

        let (precomputed, trie1_god) = Self::precompute1(
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
        // Derive Precompute 0 from Trie 1 (trivial lift), and then reduce back to Trie 1
        let (precomputed0, trie0_god) = Self::derive_precompute0_from_precompute1(&precomputed, &trie1_god);
        let (precomputed1_reduced, trie1_god_reduced) = Self::reduce_precompute0_to_precompute1(&precomputed0, &trie0_god);
        // Sanity: graphs should match
        for (sid, arc1) in &precomputed {
            let arc2 = precomputed1_reduced.get(sid).unwrap();
            assert!(PrecomputeNode1::are_graphs_equal(&trie1_god, *arc1, &trie1_god_reduced, *arc2));
        }
        // We will continue using the originally computed Trie1 (precomputed/trie1_god) for downstream
        // but we keep precomputed0/trie0_god in the constraint for commit.

        if config.optimize_trie1_merge_equivalent_llm_tokens {
            constraint_precompute1_utils::merge_equivalent_llm_tokens_trie1(&precomputed, &trie1_god, &mut precompute_vocab);
        }
        if config.optimize_trie1_reorder_llm_tokens {
            constraint_precompute1_utils::reorder_llm_tokens_for_range_minimization_trie1(&precomputed, &trie1_god, &mut precompute_vocab);
        }
        // Always run normalization pass after potential token changes.
        constraint_precompute1_utils::optimize_state_masks_and_edges_trie1(&precomputed, &trie1_god);

        // Rerun token optimizations at the end.
        if config.optimize_trie1_merge_equivalent_llm_tokens {
            constraint_precompute1_utils::merge_equivalent_llm_tokens_trie1(&precomputed, &trie1_god, &mut precompute_vocab);
        }
        if config.optimize_trie1_reorder_llm_tokens {
            constraint_precompute1_utils::reorder_llm_tokens_for_range_minimization_trie1(&precomputed, &trie1_god, &mut precompute_vocab);
        }

        // After Trie1 optimizations, the subsequent vocabs should be based on the (potentially modified) precompute_vocab.
        precompute2_vocab = precompute_vocab.clone();
        precompute3_vocab = precompute_vocab.clone();

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
            &precomputed,
            &trie1_god,
            &tokenizer, Some(&parser), Some(llm_vocab.clone()), &internal_llm_token_map_for_precompute, &token_name_map, internal_max_llm_token, &terminal_follow_map, parser.ignore_terminal_id, &mut computed_possible_matches,
            config,
            &mut precompute3_vocab,
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
            precomputed0,
            trie0_god,
            llm_vocab,
            token_name_map,
            possible_matches: computed_possible_matches,
            trie1_god,
            trie2_god,
            trie3_god,
            post_commit_allow_check_mode: TerminalAllowanceCheckMode::default(),
            precompute_vocab,
            precompute2_vocab,
            precompute3_vocab,
        };

        gc
    }

    pub fn precompute1(
        tokenizer:        &Regex,
        parser:           Option<&GLRParser>,
        llm_vocab:        Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        token_name_map:   &BiBTreeMap<Terminal, usize>,
        internal_max_llm_token: usize,
        terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        ignore_terminal_id: Option<TerminalID>,
        possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    ) -> (BTreeMap<TokenizerStateID, PrecomputeNode1Index>, Trie1GodWrapper) {
        // return (BTreeMap::new(), Trie1GodWrapper::new()); // TEMP

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
        Trie::recompute_all_max_depths(&helper.trie1_god, &helper.roots.values().cloned().collect::<Vec<_>>());

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
        Trie::recompute_all_max_depths(&helper.trie1_god, &helper.roots.values().cloned().collect::<Vec<_>>());
        helper.finish(token_name_map, possible_matches, internal_max_llm_token)
    }

    /// Build the "Trie 2" precomputation.
    pub fn precompute2(
        precomputed: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
        _trie1_god: &Trie1GodWrapper,
        _tokenizer: &Regex,
        _parser: Option<&GLRParser>,
        _llm_vocab: Option<Arc<LLMVocab>>,
        _internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        _token_name_map: &BiBTreeMap<Terminal, usize>,
        _internal_max_llm_token: usize,
        _terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        _ignore_terminal_id: Option<TerminalID>,
        _possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
        config: &GrammarConstraintConfig,
    ) -> (Precomputed2, Trie2GodWrapper) {
        (BTreeMap::new(), Trie2GodWrapper::new())
    }

    pub fn precompute3(
        precomputed: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
        trie1_god: &Trie1GodWrapper,
        tokenizer: &Regex,
        parser: Option<&GLRParser>,
        llm_vocab: Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        token_name_map: &BiBTreeMap<Terminal, usize>,
        internal_max_llm_token: usize,
        terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        ignore_terminal_id: Option<TerminalID>,
        possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
        config: &GrammarConstraintConfig,
        stage_vocab: &mut StageVocab,
    ) -> (Precomputed3, Trie3GodWrapper) {
        crate::debug!(2, "Precomputing Trie 3...");
        const BELOW_BOTTOM_REDUCE_MODE__CONTINUE_FROM_EVERYTHING: bool = false;
        const BELOW_BOTTOM_REDUCE_MODE: BelowBottomReductionMode = if BELOW_BOTTOM_REDUCE_MODE__CONTINUE_FROM_EVERYTHING {
            BelowBottomReductionMode::ContinueFromEverything
        } else {
            BelowBottomReductionMode::ContinueFromAll
        };

        let mut precomputed3 = BTreeMap::new();
        let trie3_god = Trie3GodWrapper::new();

        let parser = parser.unwrap();
        let mut initial_values_for_map: Vec<(PrecomputeNode1Index, GLRParserState)> = Vec::new();

        #[cfg(not(rustrover))]
        let it = tqdm!(precomputed.iter(), desc = "Precomputing Trie 3", disable = !PROGRESS_BAR_ENABLED, leave=false);
        #[cfg(rustrover)]
        let it = precomputed.iter();
        for (tokenizer_state_id, trie1_root) in it {
            let trie3_root = PrecomputeNode3Index::new(trie3_god.insert(PrecomputeNode3::new(PrecomputedNodeContents::root(internal_max_llm_token))));
            precomputed3.insert(*tokenizer_state_id, trie3_root.clone());

            let mut acc = Acc::new_fresh();
            acc.stored_trie_nodes_mut().insert(trie3_root);
            let gss_leaf = Arc::new(GSSNode::new(acc));

            let gss_stack = Arc::new(gss_leaf.push(ParseStateEdgeContent { state_id: parser.hallucinated_state_id }));

            let glr_state = parser.init_glr_parser_from_stack(gss_stack).with_god(trie3_god.clone());

            initial_values_for_map.push((trie1_root.clone(), glr_state));
        }

        let trie3_end = PrecomputeNode3Index::new(trie3_god.insert(PrecomputeNode3::new(PrecomputedNodeContents::leaf())));

        crate::debug!(2, "Running special_map_grouped for Trie 3 precomputation");
        Trie::special_map_grouped(
            &trie1_god,
            initial_values_for_map,
            |current_glr_state, edge_grammar_token_opt, destinations_map| {
                reset();
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
                for (dst_node_wrapper, edge_bv) in destinations_map.iter() {
                    let mut glr_s_copy = glr_s.clone();
                    allow_only_llm_tokens_and_prune_arc(&mut glr_s_copy.active_state.stack, edge_bv, &mut HashMap::new());
                    out.push((dst_node_wrapper.clone(), glr_s_copy));
                }
                print_summary();
                reset();
                out
            },
            |glr_s1, glr_s2| {
                reset();
                glr_s1.merge_with(glr_s2);
                // print_summary();
                reset();
            },
            |precomputed_node_data, glr_s| {
                reset();

                crate::datastructures::gss::merge_stored_trie_nodes(
                    &mut glr_s.active_state.stack,
                    &mut HashMap::new(),
                    glr_s.active_state.trie2_god.as_ref().unwrap(),
                );
                let keep_going = glr_s.is_ok();
                if precomputed_node_data.value.end {
                    for (_last_edge, gss_root_accs) in get_roots([glr_s.active_state.stack.as_ref()]) {
                        for gss_root_acc in gss_root_accs {
                            let active_llm_tokens_for_root = gss_root_acc.union_llm_tokens();
                            for src_wr in gss_root_acc.stored_trie_nodes().iter() {
                                let src_arc = src_wr.as_arc().clone();
                                // let src_live = { src_arc.read(&trie3_god).expect("poison").value.live_tokens.clone() };
                                // let tokens_to_push = &active_llm_tokens_for_root & &src_live;
                                let tokens_to_push = active_llm_tokens_for_root.clone();
                                if tokens_to_push.is_empty() { continue; }

                                {
                                    let mut src_w = src_arc.write(&trie3_god).expect("poison");
                                    src_w.value.live_tokens |= &tokens_to_push;
                                }

                                let edge_key = (0, tokens_to_push.clone());
                                let edge_value = StateIDBV::max_ones();

                                let inserter = EdgeInserter::new(
                                    glr_s.active_state.trie2_god.as_ref().unwrap(),
                                    src_arc.clone(),
                                    edge_key,
                                    edge_value,
                                    |e, n| *e |= n,
                                    |node_value, _edge_value| node_value.live_tokens |= &tokens_to_push,
                                    |_, _| {},
                                );
                                inserter.try_destination(trie3_end.clone()).expect("Failed to insert end edge");
                            }
                        }
                    }
                }

                const PROCESS_DEFAULT_REDUCTIONS: bool = false;
                if PROCESS_DEFAULT_REDUCTIONS {
                    // ... logic from precompute2 ...
                }

                let mut stack = vec![glr_s.active_state.stack.clone()];
                // simplify_roots_in_place(&mut stack);
                glr_s.active_state.stack = stack.into_iter().next().unwrap();

                // print_summary();
                reset();

                keep_going
            },
        );

        crate::debug!(2, "Finished precomputing Trie 3.");
        let max_state_id = parser.table.keys().map(|s| s.0).max().unwrap_or(0);
        optimize_trie3_size(&mut precomputed3, &trie3_god, config, max_state_id, internal_max_llm_token, stage_vocab);

        (precomputed3, trie3_god)
    }

    // Lift Trie1 -> Trie0 (trivial: attach disallow=None to every edge key)
    pub fn derive_precompute0_from_precompute1(
        precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
        trie1_god: &Trie1GodWrapper,
    ) -> (Precomputed0, Trie0GodWrapper) {
        let trie0_god = Trie0GodWrapper::new();
        let mut precomputed0: Precomputed0 = BTreeMap::new();

        // Map from Trie1Index -> Trie0Index
        let mut map_1_to_0: HashMap<PrecomputeNode1Index, PrecomputeNode0Index> = HashMap::new();
        let mut queue: VecDeque<PrecomputeNode1Index> = VecDeque::new();

        // Create roots
        for (sid, root1) in precomputed1.iter() {
            let root_value = { root1.read(trie1_god).expect("poison").value.clone() };
            let root0 = PrecomputeNode0Index::new(trie0_god.insert(PrecomputeNode0::new(root_value)));
            precomputed0.insert(*sid, root0);
            map_1_to_0.insert(*root1, root0);
            queue.push_back(*root1);
        }

        // BFS clone: edges (gtid_opt) -> ((gtid_opt, None))
        while let Some(n1) = queue.pop_front() {
            let n0 = map_1_to_0.get(&n1).cloned().expect("must be present");
            let children_snapshot: Vec<(Option<GrammarTokenID>, Vec<(PrecomputeNode1Index, LLMTokenBV)>)> = {
                let g = n1.read(trie1_god).expect("poison");
                g.children()
                    .iter()
                    .map(|(ek, dest_map)| {
                        let entries = dest_map.iter().map(|(idx, ev)| (*idx, ev.clone())).collect::<Vec<_>>();
                        (ek.clone(), entries)
                    })
                    .collect()
            };
            for (ek1, entries) in children_snapshot {
                let ek0 = (ek1, None);
                for (child1, ev) in entries {
                    // ensure child0 exists
                    let child0 = if let Some(&c0) = map_1_to_0.get(&child1) {
                        c0
                    } else {
                        let child_value = { child1.read(trie1_god).expect("poison").value.clone() };
                        let c0 = PrecomputeNode0Index::new(trie0_god.insert(PrecomputeNode0::new(child_value)));
                        map_1_to_0.insert(child1, c0);
                        queue.push_back(child1);
                        c0
                    };
                    // insert edge
                    let mut edge_val = Some(ev.clone());
                    let mut n0w = n0.write(&trie0_god).expect("poison");
                    n0w.try_insert_unchecked(ek0.clone(), &mut edge_val, child0);
                    // propagate live tokens
                    n0w.value.live_tokens |= &ev;
                }
            }
        }
        // Set max depths
        let roots: Vec<_> = precomputed0.values().cloned().collect();
        Trie::recompute_all_max_depths(&trie0_god, &roots);
        (precomputed0, trie0_god)
    }

    // Reduce Trie0 -> Trie1 (drop disallow info)
    pub fn reduce_precompute0_to_precompute1(
        precomputed0: &Precomputed0,
        trie0_god: &Trie0GodWrapper,
    ) -> (Precomputed, Trie1GodWrapper) {
        let trie1_god = Trie1GodWrapper::new();
        let mut precomputed1: Precomputed = BTreeMap::new();

        // Map 0->1
        let mut map_0_to_1: HashMap<PrecomputeNode0Index, PrecomputeNode1Index> = HashMap::new();
        let mut queue: VecDeque<PrecomputeNode0Index> = VecDeque::new();

        for (sid, root0) in precomputed0.iter() {
            let root_value = { root0.read(trie0_god).expect("poison").value.clone() };
            let root1 = PrecomputeNode1Index::new(trie1_god.insert(PrecomputeNode1::new(root_value)));
            precomputed1.insert(*sid, root1);
            map_0_to_1.insert(*root0, root1);
            queue.push_back(*root0);
        }

        while let Some(n0) = queue.pop_front() {
            let n1 = map_0_to_1.get(&n0).cloned().expect("must exist");
            let children_snapshot: Vec<(((Option<GrammarTokenID>, Option<(TokenizerStateID, TerminalID)>)), Vec<(PrecomputeNode0Index, LLMTokenBV)>)> = {
                let g = n0.read(trie0_god).expect("poison");
                g.children()
                    .iter()
                    .map(|(ek, dest_map)| {
                        let entries = dest_map.iter().map(|(idx, ev)| (*idx, ev.clone())).collect::<Vec<_>>();
                        (ek.clone(), entries)
                    })
                    .collect()
            };
            for ((gt_opt, _disallow_opt), entries) in children_snapshot {
                let ek1 = gt_opt;
                for (child0, ev) in entries {
                    let child1 = if let Some(&c1) = map_0_to_1.get(&child0) {
                        c1
                    } else {
                        let child_value = { child0.read(trie0_god).expect("poison").value.clone() };
                        let c1 = PrecomputeNode1Index::new(trie1_god.insert(PrecomputeNode1::new(child_value)));
                        map_0_to_1.insert(child0, c1);
                        queue.push_back(child0);
                        c1
                    };
                    let mut n1w = n1.write(&trie1_god).expect("poison");
                    let mut edge_val = Some(ev.clone());
                    n1w.try_insert_unchecked(ek1.clone(), &mut edge_val, child1);
                    n1w.value.live_tokens |= &ev;
                }
            }
        }
        let roots: Vec<_> = precomputed1.values().cloned().collect();
        Trie::recompute_all_max_depths(&trie1_god, &roots);
        (precomputed1, trie1_god)
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
        self.llm_vocab.original_to_internal_id_bimap.get(&original_id.0).map(|internal_val| LLMTokenID(*internal_val))
    }

    #[inline]
    pub(crate) fn internal_id_to_original(&self, internal_id: LLMTokenID) -> Option<LLMTokenID> {
        self.llm_vocab.original_to_internal_id_bimap.get(&internal_id.0).map(|original_val| LLMTokenID(*original_val))
    }

    #[allow(dead_code)]
    pub fn original_bv_to_internal(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        let mut internal_bv = HybridBitset::zeros();
        for original_id_val in original_bv.iter() {
            if let Some(internal_id_val) = self.llm_vocab.original_to_internal_id_bimap.get(&(original_id_val as usize)) {
                internal_bv.insert(*internal_id_val as usize);
            }
        }
        internal_bv
    }

    #[time_it]
    pub fn internal_bv_to_original(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        self.internal_bv_to_original_with_map(internal_bv, &self.llm_vocab.internal_to_original_, self.llm_vocab.internal_max_llm_token)
    }

    // Stage-aware conversion (for Trie1)
    pub fn internal_bv_to_original_precompute(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        self.internal_bv_to_original_with_map(internal_bv, &self.precompute_vocab.internal_to_original, self.precompute_vocab.internal_max_llm_token)
    }
    // Stage-aware conversion (for Trie2)
    pub fn internal_bv_to_original_precompute2(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        self.internal_bv_to_original_with_map(internal_bv, &self.precompute2_vocab.internal_to_original, self.precompute2_vocab.internal_max_llm_token)
    }
    // Stage-aware conversion (for Trie3)
    pub fn internal_bv_to_original_precompute3(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
		self.internal_bv_to_original_with_map(internal_bv, &self.precompute3_vocab.internal_to_original, self.precompute3_vocab.internal_max_llm_token)
	}

	fn internal_bv_to_original_with_map(
		&self,
		internal_bv: &LLMTokenBV,
		internal_to_original: &BTreeMap<usize, LLMTokenBV>,
		_internal_max_llm_token: usize,
	) -> LLMTokenBV {
		let mut original_bv = HybridBitset::zeros();
		if internal_bv.is_all() {
			// Fast path for "all tokens"
			for bv in internal_to_original.values() {
				original_bv |= bv;
			}
		} else {
			for i in internal_bv.iter() {
				if let Some(bv) = internal_to_original.get(&i) {
					original_bv |= bv;
				}
			}
		}
		original_bv
	}

	pub fn all_internal_llm_tokens_bitset(&self) -> LLMTokenBV {
        HybridBitset::ones(self.llm_vocab.internal_max_llm_token + 1)
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

    pub fn print_gss_nodes(&self, roots: &Vec<Arc<GSSNode>>, labels: Option<&[String]>) {
        let config = GSSPrintConfig {
            labels,
            max_edges: 500,
            original_internal_bimap: None,
            llm_token_map: Some(&self.llm_vocab.llm_token_map),
            verbose: false,
        };

        let (gss_str, state_ids) = print_gss_forest(roots, &self.parser.terminal_map, &config);
        println!("{}", gss_str);
    }

    pub fn state_with_nodes(&self, nodes: Vec<(usize, Arc<GSSNode>)>) -> GrammarConstraintState<'_> {
        let mut state = BTreeMap::new();
        for (tokenizer_state_id_val, gss_node) in nodes {
            let tokenizer_state_id = TokenizerStateID(tokenizer_state_id_val);
            let glr_state = self.parser.init_glr_parser_from_stack(gss_node).with_god(self.trie3_god.clone());
            state.insert(tokenizer_state_id, glr_state);
        }
        GrammarConstraintState { parent: self, state }
    }
}

struct Precomputer<'r> {
    tokenizer:        &'r Regex,
    parser:           Option<&'r GLRParser>,
    llm_vocab:        Option<Arc<LLMVocab>>,
    vocab:            VocabPrefixTree,
    roots:            BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    possible_matches: RefCell<BTreeMap<*const VocabPrefixTreeNode, BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>>>,
    all_llm_tokens:   HybridBitset,
    merge_threshold:  usize,
    pb:               ProgressBar,
    stats:            PrecomputeStats,
    terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    ignore_terminal_id: Option<TerminalID>,
    // Map each precompute node to the set of LLM tokens that can pass through it.
    // tags:             RefCell<HashMap<PrecomputeNodeIndex, LLMTokenBV>>, // Removed
    end_node: PrecomputeNode1Index,
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
                PrecomputeNode1Index::new(trie1_god.insert(PrecomputeNode1::new(PrecomputedNodeContents::root(internal_max_llm_token)))),
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

        let end_node = PrecomputeNode2Index::new(trie1_god.insert(PrecomputeNode1::new(PrecomputedNodeContents::leaf())));

        Self {
            tokenizer,
            parser,
            llm_vocab,
            vocab,
            roots,
            possible_matches: RefCell::new(BTreeMap::new()),
            all_llm_tokens: HybridBitset::ones(internal_max_llm_token + 1),
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
            OrderedHashSet<PrecomputeNode1Index>,
        > = BTreeMap::new();

        for (sid, arc) in &self.roots {
            assoc
                .entry(*sid)
                .or_default()
                .insert(arc.clone());
        }

        crate::debug!(2, "Starting precompute DFS");
        crate::debug!(6, "Roots for each tokenizer state:");
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

        let root_node_ptrs: HashSet<PrecomputeNode1Index> = self.roots.values().cloned().collect();

        // 1) Collect all unique nodes reachable from any root
        let roots_vec: Vec<_> = self.roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(&self.trie1_god, &roots_vec);
        // Map pointer -> Arc for quick retrieval
        let mut arc_by_ptr: HashMap<PrecomputeNode1Index, PrecomputeNode1Index> = HashMap::new();
        for n in &all_nodes {
            arc_by_ptr.insert(*n, n.clone());
        }

        // 2) Build:
        //    - incoming[B] = vec of (A, key_x, bv1) for edges A -(x; bv1)-> B
        //    - none_edges_from[B] = vec of (C, bv2) for edges B -(None; bv2)-> C
        //    - none_union[B] = union of all bv2 for None edges from B
        let mut incoming: HashMap<
            PrecomputeNode1Index,
            Vec<(PrecomputeNode1Index, Option<GrammarTokenID>, LLMTokenBV)>
        > = HashMap::new();
        let mut none_edges_from: HashMap<
            PrecomputeNode1Index,
            Vec<(PrecomputeNode1Index, LLMTokenBV)>
        > = HashMap::new();
        let mut none_union: HashMap<PrecomputeNode1Index, LLMTokenBV> = HashMap::new();

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

        type NodePtr = *const PrecomputeNode1;
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
        let mut live_tokens_cache: HashMap<PrecomputeNode1Index, LLMTokenBV> = HashMap::new();

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
        node_wrapper: PrecomputeNode1Index,
        live_tokens_cache: &mut HashMap<PrecomputeNode1Index, LLMTokenBV>,
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
        let children_to_check: Vec<PrecomputeNode1Index> = {
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
            PrecomputeNode1Index, // Dst node ptr
            HashMap<
                GrammarTokenID, // Edge key 'gtid'
                Vec<(PrecomputeNode1Index, LLMTokenBV)>, // List of (Src node ptr, edge bv)
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
                    let intermediate_node = PrecomputeNode1Index::new(self.trie1_god.insert(PrecomputeNode1::new(PrecomputedNodeContents::internal())));

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
        let mut canonical_nodes: HashMap<PrecomputeNode1, PrecomputeNode1Index> = HashMap::new();
        // A map from a node's pointer to its canonicalized Arc, to avoid re-processing.
        let mut visited: HashMap<PrecomputeNode1Index, PrecomputeNode1Index> = HashMap::new();

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
        node_arc: PrecomputeNode1Index,
        canonical_nodes: &mut HashMap<PrecomputeNode1, PrecomputeNode1Index>,
        visited: &mut HashMap<PrecomputeNode1Index, PrecomputeNode1Index>,
    ) -> PrecomputeNode1Index {
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
    ) -> (BTreeMap<TokenizerStateID, PrecomputeNode1Index>, Trie1GodWrapper) {

        calculate_final_stats(&self.roots, &mut self.stats, &self.trie1_god);
        print_precompute_stats(&self.stats, token_name_map, &self.trie1_god);

        (self.roots, self.trie1_god)
    }

    fn dfs(
        &self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<TokenizerStateID, OrderedHashSet<PrecomputeNode1Index>>,
    ) {
        self.pb.inc(1);

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let mut work_queue: BTreeMap<
                usize,
                BTreeMap<TokenizerStateID, OrderedHashSet<PrecomputeNode1Index>>,
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

fn format_bv(bv: &LLMTokenBV) -> String {
    if bv.is_empty() {
        "[]".to_string()
    } else if *bv == HybridBitset::max_ones() {
        "[ALL]".to_string()
    } else {
        format!("[len={}]", bv.len())
    }
}

#[derive(Debug, Clone)]
pub struct GrammarConstraintState<'a> {
    pub parent: &'a GrammarConstraint,
    pub state:  BTreeMap<TokenizerStateID, GLRParserState<'a>>,
}

pub(crate) mod constraint_precompute3_utils {
    use super::{PrecomputeNode3, PrecomputeNode3Index, Trie3GodWrapper};
    use crate::datastructures::gss::LLMTokenBV;
    use crate::datastructures::trie::{Trie, Trie2Index};
    use std::collections::{HashMap, VecDeque};

    pub fn clone_trie3_graph(
        root: &Trie2Index,
        trie3_god: &Trie3GodWrapper,
    ) -> (
        Trie2Index,
        HashMap<PrecomputeNode3Index, PrecomputeNode3Index>,
    ) {
        let mut map: HashMap<PrecomputeNode3Index, PrecomputeNode3Index> = HashMap::new();
        let mut q: VecDeque<PrecomputeNode3Index> = VecDeque::new();

        let root_ptr = *root;
        let root_value = { root.read(trie3_god).expect("poison").value.clone() };
        let new_root = PrecomputeNode3Index::new(trie3_god.insert(PrecomputeNode3::new(root_value)));
        map.insert(root_ptr, new_root.clone());
        q.push_back(root.clone());

        while let Some(old_arc) = q.pop_front() {
            let old_ptr = old_arc;
            let new_arc = map.get(&old_ptr).expect("parent must be created").clone();

            let children_snapshot: Vec<( (usize, LLMTokenBV), Vec<(PrecomputeNode3Index, crate::constraint::StateIDBV)> )> = {
                let g = old_arc.read(trie3_god).expect("poison");
                g.children()
                    .iter()
                    .map(|(ek, dest_map)| {
                        let entries = dest_map
                            .iter()
                            .map(|(node_ptr, ev)| {
                                (node_ptr.clone(), ev.clone())
                            })
                            .collect::<Vec<_>>();
                        (ek.clone(), entries)
                    })
                    .collect()
            };

            for (_ek, entries) in &children_snapshot {
                for (node_ptr, _ev) in entries {
                    let child_arc_old = node_ptr.as_arc().clone();
                    let child_ptr_old = child_arc_old;
                    if !map.contains_key(&child_ptr_old) {
                        let child_value = { child_arc_old.read(trie3_god).expect("poison").value.clone() };
                        let child_arc_new = PrecomputeNode3Index::new(trie3_god.insert(PrecomputeNode3::new(child_value)));
                        map.insert(child_ptr_old, child_arc_new);
                        q.push_back(child_arc_old);
                    }
                }
            }

            {
                let mut new_g = new_arc.write(trie3_god).expect("poison");
                for (ek, entries) in children_snapshot {
                    let dest_map = new_g.children_mut().entry(ek).or_default();
                    for (old_node_ptr, ev) in entries {
                        let child_arc_old = old_node_ptr.as_arc().clone();
                        let child_ptr_old = child_arc_old;
                        let child_arc_new = map.get(&child_ptr_old).expect("must exist").clone();
                        let new_key = child_arc_new;
                        dest_map.insert(new_key, ev);
                    }
                }
            }
        }

        Trie::recompute_all_max_depths(trie3_god, &[new_root.clone()]);
        (new_root, map)
    }
}

pub type Trie1GodWrapper = GodWrapper<Option<TerminalID>, HybridBitset, PrecomputedNodeContents>;
pub type Trie1God = God<Option<TerminalID>, HybridBitset, PrecomputedNodeContents>;
pub type Trie2GodWrapper = GodWrapper<(usize, Option<StateID>), HybridBitset, PrecomputedNodeContents>;
pub type Trie2God = God<(usize, Option<StateID>), HybridBitset, PrecomputedNodeContents>;
pub type Trie3GodWrapper = GodWrapper<(usize, LLMTokenBV), StateIDBV, PrecomputedNodeContents>;
pub type Trie3God = God<(usize, LLMTokenBV), StateIDBV, PrecomputedNodeContents>;
pub type Trie0GodWrapper = GodWrapper<(Option<GrammarTokenID>, Option<(TokenizerStateID, TerminalID)>), HybridBitset, PrecomputedNodeContents>;
pub type Trie0God = God<(Option<GrammarTokenID>, Option<(TokenizerStateID, TerminalID)>), HybridBitset, PrecomputedNodeContents>;

impl<'a> PartialEq for GrammarConstraintState<'a> {
    fn eq(&self, other: &Self) -> bool {
        // Compare parent by pointer to ensure they originate from the same constraint object.
        std::ptr::eq(self.parent, other.parent) && self.state == other.state
    }
}
