// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use std::borrow::Borrow;
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
use crate::datastructures::gss::{allow_only_llm_tokens_and_prune_arc, disallow_terminals_and_prune_arc, gather_gss_stats, reset_llm_tokens, GSSNode, GSSPrintConfig, LLMTokenBV, TerminalBV};
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
const DEDUP_START_ID: usize = 0;

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

pub type PrecomputeNode0 = Trie<Option<(GrammarTokenID, Option<TokenizerStateID>)>, LLMTokenBV, PrecomputedNodeContents0>;
pub type PrecomputeNode1 = Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;
pub type PrecomputeNode2 = Trie<(usize, Option<StateID>), LLMTokenBV, PrecomputedNodeContents>;
pub type PrecomputeNode3 = Trie<(usize, LLMTokenBV), StateIDBV, PrecomputedNodeContents>;

// Indices
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

/// A memory-sharing map: many keys can reference the same (large) value via an internal ID.
/// Externally, this behaves like a BTreeMap<K, V> for common operations: insert, get, iter, len.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupValueMap<K, V>
where
    K: Ord + Clone + Eq,
    V: Clone + Eq + std::hash::Hash,
{
    key_to_id: BTreeMap<K, usize>,
    id_to_value: BTreeMap<usize, V>,
    value_to_id: std::collections::HashMap<V, usize>,
    next_id: usize,
}

impl<K, V> Default for DedupValueMap<K, V>
where
    K: Ord + Clone + Eq,
    V: Clone + Eq + std::hash::Hash,
{
    fn default() -> Self {
        Self {
            key_to_id: BTreeMap::new(),
            id_to_value: BTreeMap::new(),
            value_to_id: std::collections::HashMap::new(),
            next_id: DEDUP_START_ID,
        }
    }
}

impl<K, V> DedupValueMap<K, V>
where
    K: Ord + Clone + Eq,
    V: Clone + Eq + std::hash::Hash,
{
    pub fn new() -> Self {
        Self::default()
    }

    fn intern_value(&mut self, v: V) -> usize {
        if let Some(&id) = self.value_to_id.get(&v) {
            return id;
        }
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).expect("DedupValueMap ID overflow");
        self.id_to_value.insert(id, v.clone());
        self.value_to_id.insert(v, id);
        id
    }

    pub fn len(&self) -> usize {
        self.key_to_id.len()
    }
    pub fn is_empty(&self) -> bool {
        self.key_to_id.is_empty()
    }
    pub fn contains_key<Q>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.key_to_id.contains_key(k)
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let id = self.intern_value(value);
        let old = self.key_to_id.insert(key, id);
        old.and_then(|old_id| self.id_to_value.get(&old_id).cloned())
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let id = self.key_to_id.get(key)?;
        self.id_to_value.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.key_to_id.iter().map(|(k, id)| (k, self.id_to_value.get(id).expect("dangling id")))
    }
}

impl<K, V> JSONConvertible for DedupValueMap<K, V>
where
    K: Ord + Clone + Eq + JSONConvertible,
    V: Clone + Eq + std::hash::Hash + JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("next_id".to_string(), self.next_id.to_json());
        // Serialize values as array of [id, value]
        let mut values_arr = Vec::new();
        for (id, v) in &self.id_to_value {
            values_arr.push(JSONNode::Array(vec![id.to_json(), v.to_json()]));
        }
        obj.insert("values".to_string(), JSONNode::Array(values_arr));
        // Serialize keys as array of [key, id]
        let mut keys_arr = Vec::new();
        for (k, id) in &self.key_to_id {
            keys_arr.push(JSONNode::Array(vec![k.to_json(), id.to_json()]));
        }
        obj.insert("keys".to_string(), JSONNode::Array(keys_arr));
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        let next_id = usize::from_json(obj.remove("next_id").ok_or("DedupValueMap: missing 'next_id'")?)?;
        let values_arr = obj.remove("values").ok_or("DedupValueMap: missing 'values'")?;
        let keys_arr = obj.remove("keys").ok_or("DedupValueMap: missing 'keys'")?;

        let mut id_to_value = BTreeMap::new();
        let mut value_to_id = std::collections::HashMap::new();
        match values_arr {
            JSONNode::Array(a) => {
                for n in a {
                    let mut pair = match n {
                        JSONNode::Array(p) if p.len() == 2 => p,
                        _ => return Err("DedupValueMap: values entry must be [id, value]".to_string()),
                    };
                    let v_node = pair.pop().unwrap();
                    let id_node = pair.pop().unwrap();
                    let id = usize::from_json(id_node)?;
                    let v = V::from_json(v_node)?;
                    id_to_value.insert(id, v.clone());
                    value_to_id.insert(v, id);
                }
            }
            _ => return Err("DedupValueMap: 'values' must be an array".to_string()),
        }

        let mut key_to_id = BTreeMap::new();
        match keys_arr {
            JSONNode::Array(a) => {
                for n in a {
                    let mut pair = match n {
                        JSONNode::Array(p) if p.len() == 2 => p,
                        _ => return Err("DedupValueMap: keys entry must be [key, id]".to_string()),
                    };
                    let id_node = pair.pop().unwrap();
                    let key_node = pair.pop().unwrap();
                    let id = usize::from_json(id_node)?;
                    let k = K::from_json(key_node)?;
                    key_to_id.insert(k, id);
                }
            }
            _ => return Err("DedupValueMap: 'keys' must be an array".to_string()),
        }

        Ok(Self { key_to_id, id_to_value, value_to_id, next_id })
    }
}



#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedNodeContents {
    pub(crate) end: bool,
    pub(crate) live_tokens: LLMTokenBV,
}

impl PrecomputedNodeContents {
    pub(crate) fn root(internal_max_llm_token_id: usize) -> Self {
        Self { end: false, live_tokens: LLMTokenBV::ones(internal_max_llm_token_id + 1) }
    }

    pub(crate) fn internal() -> Self {
        Self { end: false, live_tokens: LLMTokenBV::zeros() }
    }

    pub(crate) fn leaf() -> Self {
        Self { end: true, live_tokens: LLMTokenBV::zeros() }
    }
}

impl JSONConvertible for PrecomputedNodeContents {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("clean_end".to_string(), self.end.to_json());
        obj.insert("live_tokens".to_string(), self.live_tokens.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let end = obj.remove("clean_end").ok_or_else(|| "Missing field clean_end for PrecomputedNodeContents".to_string())
                                   .and_then(bool::from_json)?;
                let live_tokens = obj.remove("live_tokens").ok_or_else(|| "Missing field live_tokens for PrecomputedNodeContents".to_string())
                                       .and_then(LLMTokenBV::from_json)?;
                Ok(PrecomputedNodeContents { end, live_tokens })
            }
            _ => Err("Expected JSONNode::Object for PrecomputedNodeContents".to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedNodeContents0 {
    pub(crate) live_tokens: LLMTokenBV,
    // If this is an end node in Trie0, which tokenizer state should the GLR state be placed under?
    // Always Some(_) for leaf nodes; None for non-leaf nodes.
    pub(crate) final_tokenizer_state: Option<TokenizerStateID>,
}

impl PrecomputedNodeContents0 {
    pub(crate) fn root(internal_max_llm_token_id: usize) -> Self {
        Self {
            live_tokens: LLMTokenBV::ones(internal_max_llm_token_id + 1),
            final_tokenizer_state: None,
        }
    }

    pub(crate) fn internal() -> Self {
        Self {
            live_tokens: LLMTokenBV::zeros(),
            final_tokenizer_state: None,
        }
    }

    pub(crate) fn leaf(final_sid: TokenizerStateID) -> Self {
        Self { live_tokens: LLMTokenBV::zeros(), final_tokenizer_state: Some(final_sid) }
    }
}

impl JSONConvertible for PrecomputedNodeContents0 {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("clean_end".to_string(), self.final_tokenizer_state.is_some().to_json());
        obj.insert("live_tokens".to_string(), self.live_tokens.to_json());
        obj.insert("final_tokenizer_state".to_string(), self.final_tokenizer_state.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let live_tokens = obj.remove("live_tokens").ok_or_else(|| "Missing field live_tokens for PrecomputedNodeContents0".to_string())
                                       .and_then(LLMTokenBV::from_json)?;
                let final_tokenizer_state = obj.remove("final_tokenizer_state").ok_or_else(|| "Missing field final_tokenizer_state for PrecomputedNodeContents0".to_string())
                                               .and_then(|n| Option::<TokenizerStateID>::from_json(n))?;
                Ok(PrecomputedNodeContents0 { live_tokens, final_tokenizer_state })
            }
            _ => Err("Expected JSONNode::Object for PrecomputedNodeContents0".to_string()),
        }
    }
}

impl Into<PrecomputedNodeContents> for PrecomputedNodeContents0 {
    fn into(self) -> PrecomputedNodeContents {
        PrecomputedNodeContents { end: self.final_tokenizer_state.is_some(), live_tokens: self.live_tokens }
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
    pub(crate) precomputed0:     Precomputed0,
    pub(crate) precomputed1:      Precomputed,
    pub precomputed2:     Precomputed2,
    pub precomputed3:     Precomputed3,
    pub llm_vocab:        Arc<LLMVocab>,
    pub(crate) token_name_map:   BiBTreeMap<Terminal, usize>,
    pub possible_matches: BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    pub state_map_by_llm: DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TokenizerStateID>>,
    pub terminal_map_by_llm: DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TerminalBV>>,
    pub(crate) trie0_god: Trie0GodWrapper,
    pub(crate) trie1_god: Trie1GodWrapper,
    pub trie2_god: Trie2GodWrapper,
    pub trie3_god: Trie3GodWrapper,
    pub post_commit_allow_check_mode: TerminalAllowanceCheckMode,
    // Stage-local vocabularies for internal<->original mappings
    pub precompute0_vocab: StageVocab,
    pub precompute_vocab1: StageVocab,
    pub precompute2_vocab: StageVocab,
    pub precompute3_vocab: StageVocab,
}

impl GrammarConstraint {
    pub fn assert_eq(&self, other: &Self) {
        assert_eq!(self.tokenizer, other.tokenizer);
        assert_eq!(self.parser, other.parser);
        assert_eq!(self.precomputed0.len(), other.precomputed0.len());
        for ((sid1, arc1), (sid2, arc2)) in self.precomputed0.iter().zip(other.precomputed0.iter()) {
            assert_eq!(sid1, sid2);
            assert!(PrecomputeNode0::are_graphs_equal(&self.trie0_god, *arc1, &other.trie0_god, *arc2));
        }
        assert_eq!(self.precomputed1.len(), other.precomputed1.len());
        for ((sid1, arc1), (sid2, arc2)) in self.precomputed1.iter().zip(other.precomputed1.iter()) {
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
        assert_eq!(self.llm_vocab.llm_token_map, other.llm_vocab.llm_token_map);
        assert_eq!(self.token_name_map, other.token_name_map);
        assert_eq!(self.possible_matches, other.possible_matches);
        assert_eq!(self.post_commit_allow_check_mode, other.post_commit_allow_check_mode);
        assert_eq!(self.state_map_by_llm, other.state_map_by_llm);
        assert_eq!(self.terminal_map_by_llm, other.terminal_map_by_llm);
    }
}

impl JSONConvertible for GrammarConstraint {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("tokenizer".to_string(), self.tokenizer.to_json());
        obj.insert("parser".to_string(), self.parser.to_json());
        obj.insert("precomputed0".to_string(), self.precomputed0.to_json());
        obj.insert("precomputed1".to_string(), self.precomputed1.to_json());
        obj.insert("precomputed2".to_string(), self.precomputed2.to_json());
        obj.insert("precomputed3".to_string(), self.precomputed3.to_json());
        obj.insert("llm_token_map".to_string(), self.llm_vocab.llm_token_map.to_json());
        obj.insert("token_name_map".to_string(), self.token_name_map.to_json());
        obj.insert("possible_matches".to_string(), self.possible_matches.to_json());
        obj.insert("trie0_god".to_string(), self.trie0_god.to_json());
        obj.insert("trie1_god".to_string(), self.trie1_god.to_json());
        obj.insert("trie2_god".to_string(), self.trie2_god.to_json());
        obj.insert("trie3_god".to_string(), self.trie3_god.to_json());
        obj.insert("post_commit_allow_check_mode".to_string(), self.post_commit_allow_check_mode.to_json());
        // Stage vocabs
        obj.insert("state_map_by_llm".to_string(), self.state_map_by_llm.to_json());
        obj.insert("terminal_map_by_llm".to_string(), self.terminal_map_by_llm.to_json());
        obj.insert("precompute0_vocab".to_string(), self.precompute0_vocab.to_json());
        obj.insert("precompute_vocab".to_string(), self.precompute_vocab1.to_json());
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
                let precomputed0 = obj.remove("precomputed0").ok_or_else(|| "Missing field precomputed0".to_string())
                                     .and_then(|n| Precomputed0::from_json(n))?;
                let precomputed1 = obj.remove("precomputed1").ok_or_else(|| "Missing field precomputed1".to_string())
                                     .and_then(|n| Precomputed::from_json(n))?;
                let precomputed2 = obj.remove("precomputed2").ok_or_else(|| "Missing field precomputed2".to_string())
                                     .and_then(|n| Precomputed2::from_json(n))?;
                let precomputed3 = obj.remove("precomputed3").ok_or_else(|| "Missing field precomputed3".to_string())
                                     .and_then(|n| Precomputed3::from_json(n))?;

                let llm_token_map = obj.remove("llm_token_map").ok_or_else(|| "Missing field llm_token_map".to_string())
                                       .and_then(|n| BiBTreeMap::<Vec<u8>, LLMTokenID>::from_json(n))?;
                let token_name_map = obj.remove("token_name_map").ok_or_else(|| "Missing field token_name_map".to_string())
                                        .and_then(|n| BiBTreeMap::<Terminal, usize>::from_json(n))?;
                let possible_matches = obj.remove("possible_matches").ok_or_else(|| "Missing field possible_matches".to_string())
                                          .and_then(|n| BTreeMap::<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>::from_json(n))?;
                let trie0_god = obj.remove("trie0_god").ok_or_else(|| "Missing field trie0_god".to_string())
                                    .and_then(|n| Trie0GodWrapper::from_json(n))?;
                let trie1_god = obj.remove("trie1_god").ok_or_else(|| "Missing field trie1_god".to_string())
                                    .and_then(|n| Trie1GodWrapper::from_json(n))?;
                let trie2_god = obj.remove("trie2_god").ok_or_else(|| "Missing field trie2_god".to_string())
                                    .and_then(|n| Trie2GodWrapper::from_json(n))?;
                let trie3_god = obj.remove("trie3_god").ok_or_else(|| "Missing field trie3_god".to_string())
                                    .and_then(|n| Trie3GodWrapper::from_json(n))?;
                let post_commit_allow_check_mode = match obj.remove("post_commit_allow_check_mode") {
                    Some(n) => TerminalAllowanceCheckMode::from_json(n)?,
                    None => TerminalAllowanceCheckMode::default(),
                };
                let state_map_by_llm = match obj.remove("state_map_by_llm") {
                    Some(n) => DedupValueMap::<LLMTokenID, BTreeMap<TokenizerStateID, TokenizerStateID>>::from_json(n)?,
                    None => DedupValueMap::new(),
                };
                let terminal_map_by_llm = match obj.remove("terminal_map_by_llm") {
                    Some(n) => DedupValueMap::<LLMTokenID, BTreeMap<TokenizerStateID, TerminalBV>>::from_json(n)?,
                    None => DedupValueMap::new(),
                };
                // Stage vocabs (optional)
                let precompute0_vocab = obj.remove("precompute0_vocab")
                    .ok_or_else(|| "Missing required field 'precompute0_vocab'".to_string())
                    .and_then(StageVocab::from_json)?;
                let precompute_vocab = match obj.remove("precompute_vocab") {
					Some(n) => StageVocab::from_json(n)?,
					None => precompute0_vocab.clone(),
                };
                let precompute2_vocab = match obj.remove("precompute2_vocab") {
                    Some(n) => StageVocab::from_json(n)?,
                    None => precompute_vocab.clone(),
                };
                let precompute3_vocab = match obj.remove("precompute3_vocab") {
                    Some(n) => StageVocab::from_json(n)?,
                    None => precompute_vocab.clone(),
				};

                Ok(GrammarConstraint {
                    tokenizer,
                    parser,
                    precomputed0,
                    precomputed1,
                    precomputed2,
                    precomputed3,
                    llm_vocab: Arc::new(LLMVocab { llm_token_map, max_original_llm_token_id: 0 }), // TODO: fix this
                    token_name_map,
                    possible_matches,
                    trie0_god,
                    trie1_god,
                    trie2_god,
                    trie3_god,
                    post_commit_allow_check_mode,
                    state_map_by_llm,
                    terminal_map_by_llm,
                    precompute0_vocab,
                    precompute_vocab1: precompute_vocab,
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
        // let mut original_to_internal_id_bimap = BTreeMap::new();
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
        let original_to_internal_map = Self::setup_llm_token_mappings(&llm_token_map);


		let internal_max_llm_token = original_to_internal_map.values().copied().max().unwrap_or(0);
		// Build reverse mapping for global vocab
		let mut internal_to_original_map: BTreeMap<usize, LLMTokenBV> = BTreeMap::new();
		for (orig, int_id) in &original_to_internal_map {
			internal_to_original_map.entry(*int_id).or_default().insert(*orig);
		}

        let mut internal_llm_token_map_for_precompute = BiBTreeMap::new();
        for (bytes, original_id) in llm_token_map.iter() {
            if let Some(internal_id_val) = original_to_internal_map.get(&original_id.0) {
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

        // Build precomputed per-token (internal) maps.
        let state_map_by_llm = Self::build_state_map_by_llm(&tokenizer, &vocab_for_possible_matches.root);
        let terminal_map_by_llm = Self::rearrange_possible_matches(&computed_possible_matches);

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
        });

        // Initialize per-stage vocabularies (start identical to global)
        let mut precompute0_vocab = StageVocab {
            original_to_internal: original_to_internal_map.clone(),
            internal_to_original: internal_to_original_map.clone(),
            internal_max_llm_token,
        };
        let mut precompute_vocab = precompute0_vocab.clone();
        let mut precompute2_vocab = precompute_vocab.clone();
        let mut precompute3_vocab = precompute_vocab.clone();

        if config.skip_precomputation {
            return Self {
                tokenizer,
                parser,
                precomputed0: BTreeMap::new(),
                precomputed1: BTreeMap::new(),
                precomputed2: BTreeMap::new(),
                precomputed3: BTreeMap::new(),
                llm_vocab,
                token_name_map,
                possible_matches: computed_possible_matches,
                trie0_god: Trie0GodWrapper::new(),
                trie1_god: Trie1GodWrapper::new(),
                trie2_god: Trie2GodWrapper::new(),
                trie3_god: Trie3GodWrapper::new(),
                post_commit_allow_check_mode: TerminalAllowanceCheckMode::default(),
                state_map_by_llm,
                terminal_map_by_llm,
                precompute0_vocab,
                precompute_vocab1: precompute_vocab,
                precompute2_vocab,
                precompute3_vocab,
            };
        }

        let (precomputed0, trie0_god) = Self::precompute0(
            &tokenizer,
            Some(&parser),
            Some(llm_vocab.clone()),
            &internal_llm_token_map_for_precompute,
            &token_name_map,
            precompute0_vocab.internal_max_llm_token,
            &terminal_follow_map,
            parser.ignore_terminal_id,
            &mut computed_possible_matches,
        );

        let (precomputed1, trie1_god) = Self::precompute1(
            &precomputed0,
            &trie0_god,
            &tokenizer,
            Some(&parser),
            &terminal_follow_map,
            precompute_vocab.internal_max_llm_token,
        );

        if config.optimize_trie1_merge_equivalent_llm_tokens {
            constraint_precompute1_utils::merge_equivalent_llm_tokens_trie1(&precomputed1, &trie1_god, &mut precompute_vocab);
        }
        if config.optimize_trie1_reorder_llm_tokens {
            constraint_precompute1_utils::reorder_llm_tokens_for_range_minimization_trie1(&precomputed1, &trie1_god, &mut precompute_vocab);
        }

        // Rerun token optimizations at the end.
        if config.optimize_trie1_merge_equivalent_llm_tokens {
            constraint_precompute1_utils::merge_equivalent_llm_tokens_trie1(&precomputed1, &trie1_god, &mut precompute_vocab);
        }
        // Always run normalization pass after potential token changes.
        constraint_precompute1_utils::optimize_state_masks_and_edges_trie1(&precomputed1, &trie1_god);
        if config.optimize_trie1_reorder_llm_tokens {
            constraint_precompute1_utils::reorder_llm_tokens_for_range_minimization_trie1(&precomputed1, &trie1_god, &mut precompute_vocab);
        }

        // After Trie1 optimizations, the subsequent vocabs should be based on the (potentially modified) precompute_vocab.
        precompute2_vocab = precompute_vocab.clone();
        precompute3_vocab = precompute_vocab.clone();

        let (precomputed2, trie2_god) = Self::precompute2(
            &precomputed1,
            &trie1_god,
            &tokenizer,
            Some(&parser),
            Some(llm_vocab.clone()),
            &internal_llm_token_map_for_precompute,
            &token_name_map,
            precompute2_vocab.internal_max_llm_token,
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
        //     &precompute2_vocab.original_to_internal,
        //     &llm_vocab.llm_token_map,
        //     &trie2_god,
        // );

        let (precomputed3, trie3_god) = Self::precompute3(
            &precomputed1,
            &trie1_god,
            &tokenizer, Some(&parser), Some(llm_vocab.clone()), &internal_llm_token_map_for_precompute, &token_name_map, internal_max_llm_token, &terminal_follow_map, parser.ignore_terminal_id, &mut computed_possible_matches,
            config, // TODO: fix this
            &mut precompute3_vocab,
        );

        let mut stats3 = PrecomputeStats::default();
        crate::constraint_extra::calculate_final_stats3(&precomputed3, &mut stats3, &trie3_god);
        crate::constraint_extra::print_precompute_stats3(&stats3, &trie3_god);

        // Self::_dump_precomputed3(
        //     &precomputed3,
        //     &precompute3_vocab.original_to_internal,
        //     &llm_vocab.llm_token_map,
        //     &trie3_god,
        // );

        let mut gc = Self {
            tokenizer,
            parser,
            precomputed0,
            precomputed1,
            precomputed2,
            precomputed3,
            llm_vocab,
            token_name_map,
            possible_matches: computed_possible_matches,
            trie0_god,
            trie1_god,
            trie2_god,
            trie3_god,
            post_commit_allow_check_mode: TerminalAllowanceCheckMode::default(),
            state_map_by_llm,
            terminal_map_by_llm,
            precompute0_vocab,
            precompute_vocab1: precompute_vocab,
            precompute2_vocab,
            precompute3_vocab,
        };

        gc
    }

    pub fn precompute0(
        tokenizer:        &Regex,
        parser:           Option<&GLRParser>,
        llm_vocab:        Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        token_name_map:   &BiBTreeMap<Terminal, usize>,
        internal_max_llm_token: usize,
        terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        ignore_terminal_id: Option<TerminalID>,
        possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    ) -> (BTreeMap<TokenizerStateID, PrecomputeNode0Index>, Trie0GodWrapper) {
        // return (BTreeMap::new(), Trie1GodWrapper::new()); // TEMP

        let mut helper = Precomputer0::new(
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
        Trie::recompute_all_max_depths(&helper.trie0_god, &helper.roots.values().cloned().collect::<Vec<_>>());

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
        Trie::recompute_all_max_depths(&helper.trie0_god, &helper.roots.values().cloned().collect::<Vec<_>>());
        helper.finish(token_name_map, possible_matches, internal_max_llm_token)
    }

    pub fn precompute1(
        precomputed0: &BTreeMap<TokenizerStateID, PrecomputeNode0Index>,
        trie0_god: &Trie0GodWrapper,
        tokenizer: &Regex,
        parser: Option<&GLRParser>,
        terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        internal_max_llm_token: usize,
    ) -> (BTreeMap<TokenizerStateID, PrecomputeNode1Index>, Trie1GodWrapper) {
        let trie1_god = Trie1GodWrapper::new();
        let mut precomputed1: BTreeMap<TokenizerStateID, PrecomputeNode1Index> = BTreeMap::new();
        let mut node0_to_node1_map: HashMap<PrecomputeNode0Index, PrecomputeNode1Index> = HashMap::new();

        let mut q: VecDeque<PrecomputeNode0Index> = VecDeque::new();
        let mut visited: HashSet<PrecomputeNode0Index> = HashSet::new();

        // Create roots for trie1
        for (sid, root0_idx) in precomputed0 {
            let root0_val = root0_idx.read(trie0_god).unwrap().value.clone();
            let root1_idx = PrecomputeNode1Index::new(trie1_god.insert(PrecomputeNode1::new(root0_val.into())));
            precomputed1.insert(*sid, root1_idx.clone());
            node0_to_node1_map.insert(*root0_idx, root1_idx);
            if visited.insert(*root0_idx) {
                q.push_back(*root0_idx);
            }
        }

        while let Some(node0_idx) = q.pop_front() {
            let node1_idx = node0_to_node1_map.get(&node0_idx).unwrap().clone();

            let children0 = {
                let node0_guard = node0_idx.read(trie0_god).unwrap();
                node0_guard.children().clone()
            };

            for (edge_key, dest_map0) in children0 {
                let gtid_opt = edge_key.map(|(gtid, _)| gtid);
                for (child0_idx, edge_val) in dest_map0 {
                    let child0_guard = child0_idx.read(trie0_god).unwrap();
                    let child1_idx = match node0_to_node1_map.entry(child0_idx) {
                        std::collections::hash_map::Entry::Occupied(entry) => entry.get().clone(),
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            let child0_val = child0_guard.value.clone();
                            let new_node1 = PrecomputeNode1Index::new(trie1_god.insert(PrecomputeNode1::new(child0_val.into())));
                            entry.insert(new_node1.clone());
                            if visited.insert(child0_idx) {
                                q.push_back(child0_idx);
                            }
                            new_node1
                        }
                    };

                    let mut node1_guard = node1_idx.write(&trie1_god).unwrap();
                    let dest_map1 = node1_guard.children_mut().entry(gtid_opt).or_default();
                    dest_map1.entry(child1_idx).or_insert_with(LLMTokenBV::zeros).bitor_assign(&edge_val);
                }
            }
        }

        // Create a single leaf node for Trie1. All former end nodes will now point to this.
        let trie1_leaf = PrecomputeNode1Index::new(trie1_god.insert(PrecomputeNode1::new(PrecomputedNodeContents::leaf())));
        let all_llm_tokens = LLMTokenBV::ones(internal_max_llm_token + 1);

        // Find all nodes in precomputed0 that were end nodes and adapt them for precomputed1.
        for (node0_idx, node1_idx) in &node0_to_node1_map {
            let node0_guard = node0_idx.read(trie0_god).unwrap();
            if let Some(final_tokenizer_state) = node0_guard.value.final_tokenizer_state {
                if final_tokenizer_state == tokenizer.initial_state_id() {
                    continue;
                }
                // This was an end node in Trie0. In Trie1, it's no longer an end node itself.
                // Instead, it will have outgoing edges for all possible subsequent terminals.
                let mut node1_guard = node1_idx.write(&trie1_god).unwrap();

                // It should have been marked as an end node during conversion.
                assert!(node1_guard.value.end);
                node1_guard.value.end = false;

                // Get all terminals that the tokenizer can produce from this state.
                let accessible_terminals = tokenizer.tokens_accessible_from_state(final_tokenizer_state);

                for terminal_id in accessible_terminals {
                    // Add an edge for this terminal to the common leaf node.
                    // The edge value represents all possible LLM tokens, as we don't know which one will follow.
                    let dest_map = node1_guard.children_mut().entry(Some(terminal_id)).or_default();
                    dest_map.insert(trie1_leaf.clone(), all_llm_tokens.clone());
                }
            }
        }

        // Optimizations, similar to precompute0
        let ignore_terminal_id = parser.and_then(|p| p.ignore_terminal_id);

        Self::replace_ignore_token_edges_with_none_edges_trie1(&precomputed1, &trie1_god, ignore_terminal_id);
        Self::simplify_none_edges_trie1(&precomputed1, &trie1_god);
        Trie::recompute_all_max_depths(&trie1_god, &precomputed1.values().cloned().collect::<Vec<_>>());

        Self::prune_dead_paths_trie1(&precomputed1, &trie1_god, internal_max_llm_token);
        Self::prune_on_no_terminal_follow_trie1(&precomputed1, &trie1_god, terminal_follow_map, ignore_terminal_id);
        Self::prune_dead_paths_trie1(&precomputed1, &trie1_god, internal_max_llm_token);

        Self::factor_common_destinations_trie1(&precomputed1, &trie1_god);
        Self::merge_nodes_trie1(&mut precomputed1, &trie1_god);
        Trie::gc(&trie1_god, &precomputed1.values().cloned().collect::<Vec<_>>());

        Trie::recompute_all_max_depths(&trie1_god, &precomputed1.values().cloned().collect::<Vec<_>>());
        (precomputed1, trie1_god)
    }

    fn replace_ignore_token_edges_with_none_edges_trie1(
        roots: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
        trie1_god: &Trie1GodWrapper,
        ignore_terminal_id: Option<TerminalID>,
    ) {
        let ignore_tid = if let Some(id) = ignore_terminal_id {
            id
        } else {
            return;
        };

        crate::debug!(2, "Replacing ignore token edges with None edges in Trie1...");

        let roots_vec: Vec<_> = roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);

        for node_arc in all_nodes {
            let mut node_guard = node_arc.write(trie1_god).expect("poison");
            if let Some(dest_map_to_move) = node_guard.children_mut().remove(&Some(ignore_tid)) {
                let dest_map_for_new_key = node_guard.children_mut().entry(None).or_default();
                for (dest_wrapper, edge_bv) in dest_map_to_move {
                    if let Some(existing_bv) = dest_map_for_new_key.get_mut(&dest_wrapper) {
                        *existing_bv |= &edge_bv;
                    } else {
                        dest_map_for_new_key.insert(dest_wrapper, edge_bv);
                    }
                }
            }
        }
        crate::debug!(2, "Done replacing ignore token edges in Trie1.");
    }

    fn simplify_none_edges_trie1(
        roots: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
        trie1_god: &Trie1GodWrapper,
    ) {
        crate::debug!(2, "Simplifying None edges in Trie1...");
        let root_node_ptrs: HashSet<PrecomputeNode1Index> = roots.values().cloned().collect();
        let roots_vec: Vec<_> = roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
        let mut arc_by_ptr: HashMap<PrecomputeNode1Index, PrecomputeNode1Index> = HashMap::new();
        for n in &all_nodes {
            arc_by_ptr.insert(*n, n.clone());
        }

        let mut incoming: HashMap<
            PrecomputeNode1Index,
            Vec<(PrecomputeNode1Index, Option<GrammarTokenID>, LLMTokenBV)>,
        > = HashMap::new();
        let mut none_edges_from: HashMap<
            PrecomputeNode1Index,
            Vec<(PrecomputeNode1Index, LLMTokenBV)>,
        > = HashMap::new();
        let mut none_union: HashMap<PrecomputeNode1Index, LLMTokenBV> = HashMap::new();

        for src_arc in &all_nodes {
            let src_ptr = src_arc;
            let guard = src_arc.read(trie1_god).expect("poison");
            for (ek, dest_map) in guard.children().iter() {
                for (child_wrap, ev_bv) in dest_map.iter() {
                    let child_arc = child_wrap.as_arc().clone();
                    let child_ptr = child_arc;
                    incoming.entry(child_ptr)
                        .or_default()
                        .push((src_arc.clone(), ek.clone(), ev_bv.clone()));
                }
            }
            if let Some(dest_map) = guard.children().get(&None) {
                let list = none_edges_from.entry(*src_ptr).or_default();
                for (child_wrap, ev_bv) in dest_map.iter() {
                    list.push((child_wrap.as_arc().clone(), ev_bv.clone()));
                    let entry = none_union.entry(*src_ptr).or_insert_with(LLMTokenBV::zeros);
                    *entry |= ev_bv;
                }
            }
        }

        for (b_ptr, none_edges) in none_edges_from.into_iter() {
            let union_mask = match none_union.get(&b_ptr) {
                Some(bv) if !bv.is_empty() => bv.clone(),
                _ => continue,
            };
            let in_edges = match incoming.get(&b_ptr) {
                Some(v) if !v.is_empty() => v.clone(),
                _ => {
                    if root_node_ptrs.contains(&b_ptr) {
                        continue;
                    }
                    if let Some(b_arc) = arc_by_ptr.get(&b_ptr).cloned() {
                        let mut b_guard = b_arc.write(trie1_god).expect("poison");
                        b_guard.children_mut().remove(&None);
                    }
                    continue;
                }
            };

            let b_arc = arc_by_ptr.get(&b_ptr).unwrap().clone();
            let b_key = b_arc.clone();

            for (a_arc, edge_key, bv1_original) in in_edges.into_iter() {
                if edge_key.is_none() { continue; }

                let mut total_to_move = bv1_original.clone();
                total_to_move &= &union_mask;
                if total_to_move.is_empty() {
                    continue;
                }

                let mut a_guard = a_arc.write(trie1_god).expect("poison");
                let dest_map = a_guard.children_mut().entry(edge_key.clone()).or_default();

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

                let mut remove_b_edge = false;
                if let Some(ev_ab) = dest_map.get_mut(&b_key) {
                    *ev_ab -= &total_to_move;
                    remove_b_edge = ev_ab.is_empty();
                }
                if remove_b_edge {
                    dest_map.remove(&b_key);
                }
            }

            {
                let mut b_guard = b_arc.write(trie1_god).expect("poison");
                b_guard.children_mut().remove(&None);
            }
        }
        crate::debug!(2, "Done simplifying None edges in Trie1.");
    }

    fn prune_dead_paths_trie1(
        roots: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
        trie1_god: &Trie1GodWrapper,
        internal_max_llm_token: usize,
    ) {
        crate::debug!(2, "Pruning dead paths from Trie1.");
        let mut live_tokens_cache: HashMap<PrecomputeNode1Index, LLMTokenBV> = HashMap::new();
        let all_llm_tokens = HybridBitset::ones(internal_max_llm_token + 1);
        for root_arc in roots.values() {
            Self::get_live_tokens_and_prune_trie1(root_arc.clone(), &mut live_tokens_cache, trie1_god, &all_llm_tokens);
        }
        crate::debug!(2, "Finished pruning dead paths from Trie1.");
    }

    fn get_live_tokens_and_prune_trie1(
        node_wrapper: PrecomputeNode1Index,
        live_tokens_cache: &mut HashMap<PrecomputeNode1Index, LLMTokenBV>,
        trie1_god: &Trie1GodWrapper,
        all_llm_tokens: &LLMTokenBV,
    ) -> LLMTokenBV {
        if let Some(cached_bv) = live_tokens_cache.get(&node_wrapper) {
            return cached_bv.clone();
        }
        live_tokens_cache.insert(node_wrapper.clone(), LLMTokenBV::zeros());

        let node_arc = node_wrapper.as_arc().clone();

        let children_to_check: Vec<PrecomputeNode1Index> = {
            let node_guard = node_arc.read(trie1_god).unwrap();
            node_guard.children().values().flat_map(|dest_map| dest_map.keys().cloned()).collect()
        };

        for child_wrapper in children_to_check {
            Self::get_live_tokens_and_prune_trie1(child_wrapper, live_tokens_cache, trie1_god, all_llm_tokens);
        }

        let mut node_guard = node_arc.write(trie1_god).unwrap();

        node_guard.children_mut().retain(|_edge_key, dest_map| {
            dest_map.retain(|child_wrapper, edge_value_bv| {
                let live_tokens_from_child = live_tokens_cache.get(child_wrapper)
                    .expect("Child not found in live_tokens_cache. Logic error.");
                let live_tokens_for_this_edge = &*edge_value_bv & live_tokens_from_child;
                if live_tokens_for_this_edge.is_empty() {
                    false
                } else {
                    *edge_value_bv = live_tokens_for_this_edge;
                    true
                }
            });
            !dest_map.is_empty()
        });

        let mut current_node_live_tokens = LLMTokenBV::zeros();
        for dest_map in node_guard.children().values() {
            for edge_bv in dest_map.values() {
                current_node_live_tokens |= edge_bv;
            }
        }
        node_guard.value.live_tokens = current_node_live_tokens.clone();

        let is_end_node = node_guard.value.end;
        drop(node_guard);

        let returned_live_tokens = if is_end_node {
            all_llm_tokens.clone()
        } else {
            current_node_live_tokens
        };

        live_tokens_cache.insert(node_wrapper, returned_live_tokens.clone());
        returned_live_tokens
    }

    fn prune_on_no_terminal_follow_trie1(
        roots: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
        trie1_god: &Trie1GodWrapper,
        terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        ignore_terminal_id: Option<TerminalID>,
    ) {
        crate::debug!(2, "Pruning Trie1 based on terminal follow sets.");

        let initial_nodes_and_values: Vec<_> = roots.values()
            .map(|root_arc| (root_arc.clone(), None))
            .collect();

        type NodePtr = *const PrecomputeNode1;
        let mut edges_to_keep: HashMap<NodePtr, BTreeSet<Option<GrammarTokenID>>> = HashMap::new();

        Trie::special_map(
            trie1_god,
            initial_nodes_and_values,
            |predecessors: &Option<BTreeSet<GrammarTokenID>>, edge_key: &Option<GrammarTokenID>, _edge_bv, _child_node| {
                match edge_key {
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

                let keys_to_keep: BTreeSet<_> = node.children().keys().filter(|edge_key| {
                    match edge_key {
                        Some(edge_terminal) => allowed_follow_terminals.contains(edge_terminal) || Some(*edge_terminal) == ignore_terminal_id,
                        None => true,
                    }
                }).cloned().collect();

                let node_ptr: NodePtr = node;
                edges_to_keep.insert(node_ptr, keys_to_keep);
                true
            },
        );

        let roots_vec: Vec<_> = roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
        for node_arc in all_nodes {
            let node_ptr: NodePtr = {
                let guard = node_arc.read(trie1_god).expect("poison");
                &*guard as *const _
            };
            if let Some(keys_to_keep) = edges_to_keep.get(&node_ptr) {
                let mut node_guard = node_arc.write(trie1_god).unwrap();
                node_guard.children_mut().retain(|k, _| keys_to_keep.contains(k));
            }
        }

        crate::debug!(2, "Finished pruning Trie1 based on terminal follow sets.");
    }

    fn factor_common_destinations_trie1(
        roots: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
        trie1_god: &Trie1GodWrapper,
    ) {
        crate::debug!(2, "Factoring out common destinations in Trie1.");
        const MIN_INCOMING_EDGES_FOR_FACTORING: usize = 3;

        let roots_vec: Vec<_> = roots.values().cloned().collect();
        let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
        let arc_map: HashMap<_, _> = all_nodes.iter().map(|n| (n, n.clone())).collect();

        let mut incoming_map: HashMap<
            PrecomputeNode1Index,
            HashMap<
                Option<GrammarTokenID>,
                Vec<(PrecomputeNode1Index, LLMTokenBV)>,
            >,
        > = HashMap::new();

        for src_arc in &all_nodes {
            let src_ptr = src_arc;
            let guard = src_arc.read(trie1_god).expect("poison");
            for (edge_key, dest_map) in guard.children() {
                if edge_key.is_some() {
                    for (dest_wrapper, bv) in dest_map {
                        let dest_arc = dest_wrapper.as_arc();
                        let dest_ptr = dest_arc;
                        incoming_map.entry(*dest_ptr).or_default().entry(edge_key.clone()).or_default().push((*src_ptr, bv.clone()));
                    }
                }
            }
        }

        for (dest_ptr, edges_by_key) in incoming_map {
            for (edge_key, sources) in edges_by_key {
                if sources.len() >= MIN_INCOMING_EDGES_FOR_FACTORING {
                    let dest_arc = arc_map.get(&dest_ptr).unwrap().clone();
                    let intermediate_node = PrecomputeNode1Index::new(trie1_god.insert(PrecomputeNode1::new(PrecomputedNodeContents::internal())));

                    let mut union_bv = LLMTokenBV::zeros();
                    for (_, bv) in &sources {
                        union_bv |= bv;
                    }

                    {
                        let mut intermediate_guard = intermediate_node.write(trie1_god).expect("poison");
                        let mut edge_val_opt = Some(union_bv.clone());
                        intermediate_guard.try_insert_unchecked(edge_key.clone(), &mut edge_val_opt, dest_arc.clone());
                        intermediate_guard.value.live_tokens |= &union_bv;
                    }

                    for (src_ptr, bv) in &sources {
                        let src_arc = arc_map.get(src_ptr).unwrap();
                        let mut src_guard = src_arc.write(trie1_god).expect("poison");

                        if let Some(dest_map_for_key) = src_guard.children_mut().get_mut(&edge_key) {
                            dest_map_for_key.remove(&dest_arc.clone());
                            if dest_map_for_key.is_empty() {
                                src_guard.children_mut().remove(&edge_key);
                            }
                        }

                        let mut edge_val_opt = Some(bv.clone());
                        src_guard.try_insert_unchecked(None, &mut edge_val_opt, intermediate_node.clone());
                        src_guard.value.live_tokens |= bv;
                    }
                }
            }
        }
        crate::debug!(2, "Finished factoring common destinations in Trie1.");
    }

    fn merge_nodes_trie1(
        roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
        trie1_god: &Trie1GodWrapper,
    ) {
        crate::debug!(2, "Merging identical subtrees in Trie1.");
        let mut canonical_nodes: HashMap<PrecomputeNode1, PrecomputeNode1Index> = HashMap::new();
        let mut visited: HashMap<PrecomputeNode1Index, PrecomputeNode1Index> = HashMap::new();

        let mut new_roots = BTreeMap::new();
        for (sid, root_arc) in roots.iter() {
            let canonical_root = Self::deduplicate_recursive_trie1(root_arc.clone(), &mut canonical_nodes, &mut visited, trie1_god);
            new_roots.insert(*sid, canonical_root);
        }
        *roots = new_roots;
        crate::debug!(2, "Finished merging subtrees in Trie1. Canonical nodes: {}", canonical_nodes.len());
    }

    fn deduplicate_recursive_trie1(
        node_arc: PrecomputeNode1Index,
        canonical_nodes: &mut HashMap<PrecomputeNode1, PrecomputeNode1Index>,
        visited: &mut HashMap<PrecomputeNode1Index, PrecomputeNode1Index>,
        trie1_god: &Trie1GodWrapper,
    ) -> PrecomputeNode1Index {
        let node_ptr = node_arc;
        if let Some(canonical_arc) = visited.get(&node_ptr) {
            return canonical_arc.clone();
        }

        // Mark as visited early to break potential cycles.
        visited.insert(node_ptr, node_arc.clone());

        // Snapshot children under a short-lived read lock, then drop it before recursing.
        let children_snapshot: Vec<(
            Option<GrammarTokenID>,
            Vec<(PrecomputeNode1Index, LLMTokenBV)>,
        )> = {
            let g = node_arc.read(trie1_god).unwrap();
            g.children()
                .iter()
                .map(|(ek, dest_map)| {
                    let entries = dest_map
                        .iter()
                        .map(|(node_ptr, ev)| (node_ptr.clone(), ev.clone()))
                        .collect::<Vec<_>>();
                    (ek.clone(), entries)
                })
                .collect()
        };

        // Rebuild children map with canonicalized children (no locks held on the current node).
        let mut new_children_map = BTreeMap::new();
        let mut children_changed = false;
        for (edge_key, entries) in children_snapshot {
            let mut new_dest_map = OrderedHashMap::new();
            for (child_arc, edge_val) in entries {
                let canonical_child_arc = Self::deduplicate_recursive_trie1(
                    child_arc.clone(),
                    canonical_nodes,
                    visited,
                    trie1_god,
                );
                if child_arc != canonical_child_arc {
                    children_changed = true;
                }
                new_dest_map.insert(canonical_child_arc, edge_val);
            }
            if !new_dest_map.is_empty() {
                new_children_map.insert(edge_key, new_dest_map);
            }
        }

        // Write back updated children; avoid recompute_max_depth here to prevent lock re-entrancy.
        if children_changed {
            let mut g = node_arc.write(trie1_god).unwrap();
            *g.children_mut() = new_children_map;
            // Depths are recomputed globally after merging:
            // Trie::recompute_all_max_depths(...) is invoked by the caller.
        }

        // Canonicalize the current node by content after potential child rewrites.
        let canonical_arc = {
            let g = node_arc.read(trie1_god).unwrap();
            let node_content = (*g).clone();
            canonical_nodes
                .entry(node_content)
                .or_insert_with(|| node_arc.clone())
                .clone()
        };

        visited.insert(node_ptr, canonical_arc.clone());
        canonical_arc
    }

    /// Build the "Trie 2" precomputation.
    pub fn precompute2(
        precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
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
        precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
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
        let it = tqdm!(precomputed1.iter(), desc = "Precomputing Trie 3", disable = !PROGRESS_BAR_ENABLED, leave=false);
        #[cfg(rustrover)]
        let it = precomputed1.iter();
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

    /// Build per-token (internal id) mapping from initial tokenizer state to final tokenizer state
    /// after consuming the entire token. Computed by traversing the vocab prefix tree.
    pub fn build_state_map_by_llm(
        tokenizer: &Regex,
        vocab_root: &VocabPrefixTreeNode,
    ) -> DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TokenizerStateID>> {
        // Build initial mapping: start_state -> start_state (no consumption yet).
        let mut initial_map: BTreeMap<TokenizerStateID, TokenizerStateID> = BTreeMap::new();
        for sid in tokenizer.iter_states() {
            initial_map.insert(sid, sid);
        }
        let mut out = DedupValueMap::new();

        fn dfs(
            tokenizer: &Regex,
            node: &VocabPrefixTreeNode,
            current_map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
            out: &mut DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TokenizerStateID>>,
        ) {
            for (segment_bytes, child) in node.iter_children() {
                // Advance mapping through this segment
                let mut next_map: BTreeMap<TokenizerStateID, TokenizerStateID> = BTreeMap::new();
                for (start, cur) in current_map {
                    let exec = tokenizer.execute_from_state(&segment_bytes, *cur);
                    if let Some(end_state) = exec.end_state {
                        next_map.insert(*start, TokenizerStateID(end_state));
                    }
                }

                // Record mapping for the token at this node (if applicable).
                // Assumption: token_id() corresponds to the token whose bytes equal the path to 'child'.
                let tok_id = child.token_id();
                out.insert(LLMTokenID(tok_id), next_map.clone());

                // Recurse to longer tokens sharing this prefix.
                dfs(tokenizer, child, &next_map, out);
            }
        }

        dfs(tokenizer, vocab_root, &initial_map, &mut out);
        out
    }

    /// Rearrange possible_matches: state -> terminal -> BV(tokens)
    /// into token -> state -> set(terminals).
    pub fn rearrange_possible_matches(
        pm: &BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    ) -> DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TerminalBV>> {
        let mut tmp: BTreeMap<LLMTokenID, BTreeMap<TokenizerStateID, TerminalBV>> = BTreeMap::new();
        for (sid, tmap) in pm {
            for (term, bv) in tmap {
                if bv.is_all() {
                    // "All tokens" - we cannot enumerate efficiently; skip to avoid ballooning memory.
                    // Commit falls back for missing entries.
                    continue;
                }
                for tok in bv.iter() {
                    let tok_id = LLMTokenID(tok);
                    let per_state = tmp.entry(tok_id).or_default();
                    per_state.entry(*sid).or_default().insert(term.0);
                }
            }
        }
        let mut out = DedupValueMap::new();
        for (tok, m) in tmp {
            out.insert(tok, m);
        }
        out
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
    pub(crate) fn original_id_to_internal_stage0(&self, original_id: LLMTokenID) -> Option<LLMTokenID> {
        self.precompute0_vocab.original_to_internal.get(&original_id.0).map(|internal_val| LLMTokenID(*internal_val))
    }

    #[time_it]
    pub fn internal_bv_to_original(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        self.internal_bv_to_original_with_map(
            internal_bv,
            &self.precompute0_vocab.internal_to_original,
            self.precompute0_vocab.internal_max_llm_token,
        )
    }

    // Stage-aware conversion (for Trie1)
    pub fn internal_bv_to_original_precompute(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        self.internal_bv_to_original_with_map(internal_bv, &self.precompute_vocab1.internal_to_original, self.precompute_vocab1.internal_max_llm_token)
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

struct Precomputer0<'r> {
    tokenizer:        &'r Regex,
    parser:           Option<&'r GLRParser>,
    llm_vocab:        Option<Arc<LLMVocab>>,
    vocab:            VocabPrefixTree,
    roots:            BTreeMap<TokenizerStateID, PrecomputeNode0Index>,
    possible_matches: RefCell<BTreeMap<*const VocabPrefixTreeNode, BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>>>,
    all_llm_tokens:   HybridBitset,
    merge_threshold:  usize,
    pb:               ProgressBar,
    stats:            PrecomputeStats,
    terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    ignore_terminal_id: Option<TerminalID>,
    // Map each precompute node to the set of LLM tokens that can pass through it.
    // tags:             RefCell<HashMap<PrecomputeNodeIndex, LLMTokenBV>>, // Removed
    // One end node per final tokenizer state.
    end_nodes:        BTreeMap<TokenizerStateID, PrecomputeNode0Index>,
    trie0_god:        Trie0GodWrapper,
}

impl<'r> Precomputer0<'r> {
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
        let trie0_god = Trie0GodWrapper::new();
        for sid in tokenizer.iter_states() {
            roots.insert(
                sid,
                PrecomputeNode0Index::new(trie0_god.insert(PrecomputeNode0::new(PrecomputedNodeContents0::root(internal_max_llm_token)))),
            );
        }
        crate::debug!(2, "Created trie0 roots for {} tokenizer states", tokenizer.iter_states().count());

        crate::debug!(2, "Counting vocab nodes for progress bar...");
        let total_nodes = count_vocab_nodes(&vocab.root);
        crate::debug!(2, "Counted {} vocab nodes", total_nodes);
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

        let end_nodes = tokenizer.iter_states()
            .map(|tsid| (tsid, PrecomputeNode0Index::new(trie0_god.insert(PrecomputeNode0::new(PrecomputedNodeContents0::leaf(tsid))))))
            .collect();
        crate::debug!(2, "Created trie0 end nodes for {} tokenizer states", tokenizer.iter_states().count());

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
            end_nodes,
            trie0_god,
        }
    }

    fn get_end_node(&self, final_sid: TokenizerStateID) -> PrecomputeNode0Index {
        self.end_nodes[&final_sid].clone()
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
            OrderedHashSet<PrecomputeNode0Index>,
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
        let all_nodes = Trie::all_nodes(&self.trie0_god, &roots_vec);
        // 2. Iterate over each node and modify its children map.
        for node_arc in all_nodes {
            let mut node_guard = node_arc.write(&self.trie0_god).expect("poison");
            let mut edges_to_move = Vec::new();

            for (key, dest_map) in node_guard.children() {
                if let Some((gtid, tokenizer_state_id_opt)) = key {
                    if *gtid == ignore_tid && tokenizer_state_id_opt.is_none() {
                        edges_to_move.push((key.clone(), dest_map.clone()));
                    }
                }
            }

            for (old_key, dest_map_to_move) in edges_to_move {
                node_guard.children_mut().remove(&old_key);
                let dest_map_for_new_key = node_guard.children_mut().entry(None).or_default();
                for (dest_wrapper, edge_bv) in dest_map_to_move {
                    // If an edge to this destination already exists under None, merge the bitvectors.
                    if let Some(existing_bv) = dest_map_for_new_key.get_mut(&dest_wrapper) {
                        *existing_bv |= &edge_bv;
                    } else {
                        dest_map_for_new_key.insert(dest_wrapper, edge_bv);
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
        let all_nodes = Trie::all_nodes(&self.trie0_god, &roots_vec);
        // Map pointer -> Arc for quick retrieval
        let mut arc_by_ptr: HashMap<PrecomputeNode0Index, PrecomputeNode0Index> = HashMap::new();
        for n in &all_nodes {
            arc_by_ptr.insert(*n, n.clone());
        }

        // 2) Build:
        //    - incoming[B] = vec of (A, key_x, bv1) for edges A -(x; bv1)-> B
        //    - none_edges_from[B] = vec of (C, bv2) for edges B -(None; bv2)-> C
        //    - none_union[B] = union of all bv2 for None edges from B
        let mut incoming: HashMap<
            PrecomputeNode0Index,
            Vec<(PrecomputeNode0Index, Option<(GrammarTokenID, Option<TokenizerStateID>)>, LLMTokenBV)>
        > = HashMap::new();
        let mut none_edges_from: HashMap<
            PrecomputeNode0Index,
            Vec<(PrecomputeNode0Index, LLMTokenBV)>
        > = HashMap::new();
        let mut none_union: HashMap<PrecomputeNode0Index, LLMTokenBV> = HashMap::new();

        for src_arc in &all_nodes {
            let src_ptr = src_arc;
            let guard = src_arc.read(&self.trie0_god).expect("poison");
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
            for (ek, dest_map) in guard.children().iter() {
                if ek.is_none() {
                    let list = none_edges_from.entry(*src_ptr).or_default();
                    for (child_wrap, ev_bv) in dest_map.iter() {
                        list.push((child_wrap.as_arc().clone(), ev_bv.clone()));
                        let entry = none_union.entry(*src_ptr).or_insert_with(LLMTokenBV::zeros);
                        *entry |= ev_bv;
                    }
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
                        let mut b_guard = b_arc.write(&self.trie0_god).expect("poison");
                        b_guard.children_mut().retain(|k, _| k.is_some());
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

                let mut a_guard = a_arc.write(&self.trie0_god).expect("poison");
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
                let mut b_guard = b_arc.write(&self.trie0_god).expect("poison");
                b_guard.children_mut().retain(|k, _| k.is_some());
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

        type NodePtr = *const PrecomputeNode0; let mut edges_to_keep: HashMap<NodePtr, BTreeSet<Option<(GrammarTokenID, Option<TokenizerStateID>)>>> = HashMap::new();

        Trie::special_map(
            &self.trie0_god,
            initial_nodes_and_values,
            |predecessors: &Option<BTreeSet<GrammarTokenID>>, edge_key: &Option<(GrammarTokenID, Option<TokenizerStateID>)>, _edge_bv, _child_node| {
                match edge_key {
                    Some((t, _)) if Some(*t) == ignore_terminal_id => Some(predecessors.clone()),
                    Some((t, _)) => Some(Some(BTreeSet::from([*t]))),
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

                let keys_to_keep: BTreeSet<_> = node.children().keys().filter(|edge_key| {
                    match edge_key {
                        // Keep edges with terminals that are in the allowed follow set (or ignore edges).
                        Some((edge_terminal, _)) => allowed_follow_terminals.contains(edge_terminal) || Some(*edge_terminal) == ignore_terminal_id,
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
        let all_nodes = Trie::all_nodes(&self.trie0_god, &roots_vec);
        for node_arc in all_nodes {
            let node_ptr: NodePtr = {
                let guard = node_arc.read(&self.trie0_god).expect("poison");
                &*guard as *const _
            };
            if let Some(keys_to_keep) = edges_to_keep.get(&node_ptr) {
                let mut node_guard = node_arc.write(&self.trie0_god).unwrap();
                node_guard.children_mut().retain(|k, _| keys_to_keep.contains(k));
            }
        }

        crate::debug!(2, "Finished pruning based on terminal follow sets.");
    }

    fn prune_dead_paths(&mut self) {
        crate::debug!(2, "Pruning dead paths from precomputed trie.");

        // A cache of nodes to the set of "live" LLM tokens reachable from them.
        let mut live_tokens_cache: HashMap<PrecomputeNode0Index, LLMTokenBV> = HashMap::new();

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
        node_wrapper: PrecomputeNode0Index,
        live_tokens_cache: &mut HashMap<PrecomputeNode0Index, LLMTokenBV>,
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
        let children_to_check: Vec<PrecomputeNode0Index> = {
            let node_guard = node_arc.read(&self.trie0_god).unwrap();
            node_guard.children().values().flat_map(|dest_map| dest_map.keys().cloned()).collect()
        };

        // Recursively call on all unique children to populate the cache for them.
        for child_wrapper in children_to_check {
            self.get_live_tokens_and_prune(child_wrapper, live_tokens_cache);
        }

        // Now that the cache is populated for all children, we can prune the current node.
        let mut live_tokens_for_this_node = LLMTokenBV::zeros();
        {
            let mut node_guard = node_arc.write(&self.trie0_god).unwrap();

            // A node is live if it's an end node itself. The tokens that end here are
            // on the edges pointing to this node.
            if node_guard.value.final_tokenizer_state.is_some() {
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
        let all_nodes = Trie::all_nodes(&self.trie0_god, &roots_vec);
        let arc_map: HashMap<_, _> = all_nodes.iter().map(|n| (n, n.clone())).collect();

        // 2. Build an incoming edge map for every node.
        // incoming_map: D_ptr -> (gtid -> Vec<(S_ptr, bv)>)
        let mut incoming_map: HashMap<
            PrecomputeNode0Index, // Dst node ptr
            HashMap<
                Option<(GrammarTokenID, Option<TokenizerStateID>)>, // Full edge key
                Vec<(PrecomputeNode0Index, LLMTokenBV)>, // List of (Src node ptr, edge bv)
            >,
        > = HashMap::new();

        for src_arc in &all_nodes {
            let src_ptr = src_arc;
            let guard = src_arc.read(&self.trie0_god).expect("poison");
            for (edge_key, dest_map) in guard.children() {
                if edge_key.is_some() { // Only consider non-None edges
                    for (dest_wrapper, bv) in dest_map {
                        let dest_arc = dest_wrapper.as_arc();
                        let dest_ptr = dest_arc;
                        incoming_map.entry(*dest_ptr).or_default().entry(edge_key.clone()).or_default().push((*src_ptr, bv.clone()));
                    }
                }
            }
        }

        // 3. Iterate through the map and find factoring opportunities.
        for (dest_ptr, edges_by_key) in incoming_map {
            for (edge_key, sources) in edges_by_key {
                if sources.len() >= MIN_INCOMING_EDGES_FOR_FACTORING {
                    // Opportunity found!
                    let dest_arc = arc_map.get(&dest_ptr).unwrap().clone();

                    // a. Create a new intermediate node `I`.
                    let intermediate_node = PrecomputeNode0Index::new(self.trie0_god.insert(PrecomputeNode0::new(PrecomputedNodeContents0::internal())));

                    // b. Add edge I --(edge_key)--> D
                    let mut union_bv = LLMTokenBV::zeros();
                    for (_, bv) in &sources {
                        union_bv |= bv;
                    }

                    {
                        let mut intermediate_guard = intermediate_node.write(&self.trie0_god).expect("poison");
                        let mut edge_val_opt = Some(union_bv.clone());
                        // No cycle possible since I is new. Use unchecked for speed.
                        // Depth will be propagated to D.
                        intermediate_guard.try_insert_unchecked(edge_key.clone(), &mut edge_val_opt, dest_arc.clone());
                        intermediate_guard.value.live_tokens |= &union_bv; // Update live_tokens for intermediate node
                    }

                    // c. For each source, remove old edge and add new `None` edge to `I`.
                    for (src_ptr, bv) in &sources {
                        let src_arc = arc_map.get(src_ptr).unwrap();
                        let mut src_guard = src_arc.write(&self.trie0_god).expect("poison");

                        // Remove S --(edge_key)--> D
                        if let Some(dest_map_for_key) = src_guard.children_mut().get_mut(&edge_key) {
                            dest_map_for_key.remove(&dest_arc.clone());
                            if dest_map_for_key.is_empty() {
                                src_guard.children_mut().remove(&edge_key);
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
        let mut canonical_nodes: HashMap<PrecomputeNode0, PrecomputeNode0Index> = HashMap::new();
        // A map from a node's pointer to its canonicalized Arc, to avoid re-processing.
        let mut visited: HashMap<PrecomputeNode0Index, PrecomputeNode0Index> = HashMap::new();

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
        node_arc: PrecomputeNode0Index,
        canonical_nodes: &mut HashMap<PrecomputeNode0, PrecomputeNode0Index>,
        visited: &mut HashMap<PrecomputeNode0Index, PrecomputeNode0Index>,
    ) -> PrecomputeNode0Index {
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
            let node_guard = node_arc.read(&self.trie0_god).unwrap();
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
        let mut node_guard = node_arc.write(&self.trie0_god).unwrap();
        *node_guard.children_mut() = new_children_map;
        node_guard.recompute_max_depth(&self.trie0_god);
        // The live_tokens field will be recomputed by prune_dead_paths after merging.
    }

    let canonical_arc = {
            let node_guard = node_arc.read(&self.trie0_god).unwrap();
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
        Trie::gc(&self.trie0_god, &roots);
    }

    fn finish(
        mut self,
        token_name_map: &BiBTreeMap<Terminal, usize>,
        possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
        internal_max_llm_token: usize,
    ) -> (BTreeMap<TokenizerStateID, PrecomputeNode0Index>, Trie0GodWrapper) {

        // calculate_final_stats(&self.roots, &mut self.stats, &self.trie0_god);
        // print_precompute_stats(&self.stats, token_name_map, &self.trie0_god);

        (self.roots, self.trie0_god)
    }

    fn dfs(
        &self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<TokenizerStateID, OrderedHashSet<PrecomputeNode0Index>>,
    ) {
        self.pb.inc(1);

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let mut work_queue: BTreeMap<
                usize,
                BTreeMap<TokenizerStateID, OrderedHashSet<PrecomputeNode0Index>>,
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

                        let mut disallowed_tokenizer_state_info = None;
                        if let Some(end_state_val) = exec_result.end_state {
                            let end_tokenizer_state_id = TokenizerStateID(end_state_val);
                            let terminals_accessible = self.tokenizer.tokens_accessible_from_state(end_tokenizer_state_id);
                            if terminals_accessible.contains(&terminal_id) {
                                disallowed_tokenizer_state_info = Some(end_tokenizer_state_id);
                            }
                        }

                        for src_node_wrapper in &precompute_nodes {
                            if next_pos == segment_bytes.len() {
                                // Exact end-of-segment terminal match: finishing LLM token here goes to tokenizer initial state.
                                let llm_token_id = child_vocab_node.token_id();
                                let mut edge_bv = HybridBitset::zeros();
                                edge_bv.insert(llm_token_id);
                                let edge_key = Some((terminal_id, disallowed_tokenizer_state_info));
                                let mut inserter = EdgeInserter::new(
                                    &self.trie0_god,
                                    src_node_wrapper.as_arc().clone(),
                                    edge_key,
                                    edge_bv,
                                    |e, n| *e |= n,
                                    |node_value, edge_value| {
                                        node_value.live_tokens |= edge_value;
                                    },
                                    |ev, t| *ev &= &t.live_tokens,
                                );
                                let end_idx = {
                                    let s0 = self.tokenizer.initial_state_id();
                                    self.get_end_node(s0)
                                };
                                inserter.try_destination(end_idx.as_arc().clone()).expect("Failed to insert end node for terminal at end of segment");
                            }

                            let mut edge_bv = child_vocab_node.reachable_token_ids().clone();
                            if next_pos == segment_bytes.len() {
                                edge_bv.set(child_vocab_node.token_id(), false);
                            }
                            if let Some(matches_for_terminal) = possible_matches_at_end.get(&terminal_id) {
                                edge_bv -= matches_for_terminal;
                            }

                            if edge_bv.is_empty() { continue; }

                            let edge_key = Some((terminal_id, disallowed_tokenizer_state_info));
                            let mut inserter = EdgeInserter::new(
                                &self.trie0_god,
                                src_node_wrapper.as_arc().clone(),
                                edge_key,
                                edge_bv.clone(),
                                |e, n| *e |= n,
                                |node_value, edge_value| node_value.live_tokens |= edge_value,
                                |ev, t| *ev &= &t.live_tokens,
                            );

                            let next_tokenizer_state = self.tokenizer.initial_state_id();
                            let dest_nodes_in_queue = work_queue.entry(next_pos).or_default().entry(next_tokenizer_state).or_default();

                            inserter = inserter.try_destinations_iter(dest_nodes_in_queue.iter().map(|w| w.as_arc().clone()).filter(|w| w.read(&self.trie0_god).unwrap().value.final_tokenizer_state.is_none()));

                            let children_of_src: Vec<_> = src_node_wrapper.as_arc().read(&self.trie0_god).unwrap().children().values().flat_map(|m| m.keys().cloned()).collect();
                            let eligible_children = children_of_src.iter().map(|child_node_ptr| {
                                child_node_ptr.as_arc().clone()
                            }).filter(|child_arc| {
                                (child_arc.read(&self.trie0_god).unwrap().value.live_tokens.clone() & &edge_bv).is_empty() && child_arc.read(&self.trie0_god).unwrap().value.final_tokenizer_state.is_none()
                            });
                            inserter = inserter.try_destinations_iter(eligible_children);

                            let result_node = inserter.else_create_destination_with_value(PrecomputedNodeContents0::internal()).unwrap();
                            let result_node_ptr = result_node.clone();
                            dest_nodes_in_queue.insert(result_node_ptr.clone());
                        }
                    }

                    if let Some(end_state_val) = exec_result.end_state {
                        for src_node_wrapper in &precompute_nodes {
                            let llm_token_id = child_vocab_node.token_id();
                            let mut edge_bv = HybridBitset::zeros();
                            edge_bv.insert(llm_token_id);
                            let edge_key = None;
                            let mut inserter = EdgeInserter::new(
                                &self.trie0_god,
                                src_node_wrapper.as_arc().clone(),
                                edge_key,
                                edge_bv,
                                |e, n| *e |= n,
                                |node_value, edge_value| node_value.live_tokens |= edge_value,
                                |ev, t| *ev &= &t.live_tokens,
                            );
                            let end_idx = self.get_end_node(TokenizerStateID(end_state_val));
                            inserter.try_destination(end_idx.as_arc().clone()).expect("Failed to insert end node for terminal at end of segment");
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

pub type Trie0GodWrapper = GodWrapper<Option<(TerminalID, Option<TokenizerStateID>)>, HybridBitset, PrecomputedNodeContents0>;
pub type Trie0God = God<Option<(TerminalID, Option<TokenizerStateID>)>, HybridBitset, PrecomputedNodeContents>;
pub type Trie1GodWrapper = GodWrapper<Option<TerminalID>, HybridBitset, PrecomputedNodeContents>;
pub type Trie1God = God<Option<TerminalID>, HybridBitset, PrecomputedNodeContents>;
pub type Trie2GodWrapper = GodWrapper<(usize, Option<StateID>), HybridBitset, PrecomputedNodeContents>;
pub type Trie2God = God<(usize, Option<StateID>), HybridBitset, PrecomputedNodeContents>;
pub type Trie3GodWrapper = GodWrapper<(usize, LLMTokenBV), StateIDBV, PrecomputedNodeContents>;
pub type Trie3God = God<(usize, LLMTokenBV), StateIDBV, PrecomputedNodeContents>;

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
            original_internal_bimap: None,
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
    fn transform_gss_stacks<M, F>(&mut self, mut f: F)
    where
        M: Default,
        F: FnMut(&mut Arc<GSSNode>, &mut M),
    {
        let mut memo = M::default();
        for s in self.state.values_mut() {
            f(&mut s.active_state.stack, &mut memo);
        }
    }

    fn map_gss_stacks<M, F>(&mut self, mut f: F)
    where
        M: Default,
        F: FnMut(&mut Arc<GSSNode>, &mut M) -> Arc<GSSNode>,
    {
        let mut memo = M::default();
        for s in self.state.values_mut() {
            s.active_state.stack = f(&mut s.active_state.stack, &mut memo);
        }
    }

    pub fn compute_commit_maps(&self, llm_token_bytes: &[u8]) -> (BTreeMap<TokenizerStateID, TokenizerStateID>, BTreeMap<TokenizerStateID, TerminalBV>) {
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
        (state_map, terminals_map)
    }

    pub fn get_mask(&self) -> LLMTokenBV {
        // return HybridBitset::ones(self.parent.llm_vocab.max_original_llm_token_id + 1); // TEMP
        // self.get_mask1()
        // self.get_mask2()
        self.get_mask3()
    }

    #[time_it]
    pub fn get_mask1(&self) -> LLMTokenBV {
        let t0 = std::time::Instant::now();
        crate::debug!(3, "Getting mask {} states: {:?}", self.state.len(), self.state.keys().map(|k|k.0).collect::<Vec<_>>());
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
            println!("{}", self.parent.parser.gss_forest_to_dot( // TODO: fix this
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

        let mut initial_values_for_map: Vec<(PrecomputeNode1Index, GLRParserState<'a>)> = Vec::new();
        for (tokenizer_state_id, glr_state) in &self.state {
            // crate::debug!(4, "Initializing GSS for state {}", tokenizer_state_id.0);
            // Ensure the GLR state's GSS stack is not empty before proceeding
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            if let Some(precomputed_trie_root_arc) = self.parent.precomputed1.get(tokenizer_state_id) {
                let mut glr_state = glr_state.clone();
                prune_llm_tokens_by_disallowed_terminals(
                    &mut glr_state.active_state.stack,
                    &self.parent.possible_matches,
                    &mut HashMap::new(),
                );
                initial_values_for_map.push((precomputed_trie_root_arc.clone(), glr_state));
            } else {
                panic!("No precomputed trie found for tokenizer state {:?}.", tokenizer_state_id);
            }
        }

        if initial_values_for_map.is_empty() {
             // This can happen if all GLR states had empty GSS stacks or no corresponding precomputed tries.
             crate::debug!(2, "No valid initial states for get_mask's special_map traversal.");
             return self.parent.internal_bv_to_original_precompute(&final_mask_internal.into_inner());
        }

        let t1 = std::time::Instant::now();
        if env::var("RUST_LOG_MASK_TIMING").is_ok() {
            println!("after initial_values_for_map: {:>15?}", t1.duration_since(t0));
        }

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
        if env::var("RUST_LOG_MASK_TIMING").is_ok() {
            println!("after special_map: {:>15?}", t_after_special_map.duration_since(t0));
        }

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
                original_internal_bimap: None,
                llm_token_map: Some(&self.parent.llm_vocab.llm_token_map),
                verbose: false,
            };
            print!("{}", print_gss_forest(&roots, &self.parent.parser.terminal_map, &config).0);
        }

        let final_mask_mapped = self.parent.internal_bv_to_original_precompute(&final_mask_internal.into_inner());

        let t_end = std::time::Instant::now();
        if env::var("RUST_LOG_MASK_TIMING").is_ok() {
            println!("get_mask took: {:>15?}", t_end.duration_since(t0));
        }

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
            println!("{}", self.parent.parser.gss_forest_to_dot( // TODO: fix this
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
                let mut glr_state = glr_state.clone();
                prune_llm_tokens_by_disallowed_terminals(
                    &mut glr_state.active_state.stack,
                    &self.parent.possible_matches,
                    &mut HashMap::new(),
                );
                initial_values_for_map.push((precomputed_trie_root_arc.clone(), glr_state));
            } else {
                panic!("No precomputed trie found for tokenizer state {:?}.", tokenizer_state_id);
            }
        }

        if initial_values_for_map.is_empty() {
             // This can happen if all GLR states had empty GSS stacks or no corresponding precomputed tries.
             crate::debug!(2, "No valid initial states for get_mask's special_map traversal.");
             return self.parent.internal_bv_to_original_precompute2(&final_mask_internal.into_inner());
        }

        let t1 = std::time::Instant::now();
        if env::var("RUST_LOG_MASK_TIMING").is_ok() {
            println!("after initial_values_for_map: {:>15?}", t1.duration_since(t0));
        }

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
        if env::var("RUST_LOG_MASK_TIMING").is_ok() {
            println!("after special_map: {:>15?}", t_after_special_map.duration_since(t0));
        }

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
                original_internal_bimap: None,
                llm_token_map: Some(&self.parent.llm_vocab.llm_token_map),
                verbose: false,
            };
            print!("{}", print_gss_forest(&roots, &self.parent.parser.terminal_map, &config).0);
        }

        crate::debug!(4, "Final mask internal: {:?}", final_mask_internal.borrow());
        let final_mask_mapped = self.parent.internal_bv_to_original_precompute2(&final_mask_internal.into_inner());
        crate::debug!(4, "Final mask mapped: {:?}", final_mask_mapped);

        let t_end = std::time::Instant::now();
        println!("get_mask took: {:>15?}", t_end.duration_since(t0));

        final_mask_mapped
    }

    pub fn print_gss_stats(&self) {
        println!("GrammarConstraintState Stats:");
        println!("  - Active tokenizer states: {}", self.state.len());
        if self.state.is_empty() {
            println!("  - GSS is empty.");
            return;
        }
        let stats = gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        );
        println!("  - GSS Stats: {:#?}", stats);
    }

    pub fn print_gss(&self) {
        let roots: Vec<_> = self.state.values().map(|s| s.active_state.stack.clone()).collect();
        if roots.is_empty() {
            println!("GSS is empty.");
            return;
        }
        let labels: Vec<_> = self.state.keys().map(|k| format!("Tokenizer State {}", k.0)).collect();
        self.parent.print_gss_nodes(&roots, Some(&labels));
    }

    pub fn explain_stack(&self) {
        for (state_id, state) in &self.state {
            println!("\n--- State {} ---", state_id.0);
            // Sample and print a bunch of stacks
            let mut seen = BTreeSet::new();
            let num_to_sample = 10;
            for i in 0..1000 {
                if let Some(sampled_path_edges) = sample_path(&[&state.active_state.stack], i) {
                    let mut sampled_stack: Vec<usize> = sampled_path_edges.iter()
                        .map(|edge| edge.state_id.0)
                        .collect();
                    sampled_stack.reverse();
                    if seen.contains(&sampled_stack) {
                        continue;
                    }
                    seen.insert(sampled_stack);
                    if seen.len() >= num_to_sample {
                        break;
                    }
                };
            }
            for sampled_stack in seen {
                println!("  Sampled stack: {:?}", sampled_stack);
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
        crate::debug!(10, "\n--- get_mask3 START ---");
        crate::debug!(10, "GSS at start of get_mask3:");
        crate::debug!(3, "Getting mask {} states: {:?}", self.state.len(), self.state.keys().map(|k|k.0).collect::<Vec<_>>());
        let stats = gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        );
        crate::debug!(10, "Initial GSS stats: {:#?}", stats);
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
            println!("{}", self.parent.parser.gss_forest_to_dot( // TODO: fix this
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
            return self.parent.internal_bv_to_original_precompute3(&final_mask_internal.into_inner());
        }
        let mut initial_values_by_trie_node: BTreeMap<PrecomputeNode3Index, GLRParserState<'a>> = BTreeMap::new();
        crate::debug!(10, "\n--- Seeding work queue ---");
        for (&tokenizer_state_id, glr_state) in &self.state {
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            let mut glr_state = glr_state.clone();
            prune_llm_tokens_by_disallowed_terminals(
                &mut glr_state.active_state.stack,
                &self.parent.possible_matches,
                &mut HashMap::new(),
            );

            if let Some(precomputed_trie_root_arc) = self.parent.precomputed3.get(&tokenizer_state_id) {
                crate::debug!(10, "  SEED: sid={}, root_idx={}, gss_ptr={:p}", tokenizer_state_id.0, precomputed_trie_root_arc, glr_state.active_state.stack);
                
                initial_values_by_trie_node.entry(precomputed_trie_root_arc.clone())
                    .and_modify(|existing_glr| {
                        existing_glr.merge_with(glr_state.clone());
                    })
                    .or_insert(glr_state.clone());
            } else {
                panic!("No precomputed trie found for tokenizer state {:?}.", tokenizer_state_id);
            }
        }

        let initial_values_for_map: Vec<_> = initial_values_by_trie_node.into_iter().collect();

        if initial_values_for_map.is_empty() {
             crate::debug!(2, "No valid initial states for get_mask's special_map traversal.");
             return self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        }

        let t1 = std::time::Instant::now();
        if env::var("RUST_LOG_MASK_TIMING").is_ok() {
            println!("after initial_values_for_map: {:>15?}", t1.duration_since(t0));
        }

        crate::profiler::reset();

        Trie::special_map_grouped(
            &self.parent.trie3_god,
            initial_values_for_map,
            // step_fn: (current_state, (pop, llm_token_bv), destinations_map)
            |glr_s, (pop, llm_token_bv_from_edge), dest_map| {
                crate::debug!(10, "  - STEP: gss_ptr={:p}, edge=(pop={}, llm_bv={})", glr_s.active_state.stack, pop, format_bv(llm_token_bv_from_edge));
                let popped = glr_s.active_state.stack.popn(*pop);
                let num_peeks: usize = popped.iter().map(|p| p.peek_iter().count()).sum();
                if num_peeks > 0 {
                    crate::debug!(10, "      - Popped GSS has {} peeks", num_peeks);
                }
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

                    let merged_gss = GSSNode::merge_many_with_depth(1, valid_gss_nodes.clone());
                    let mut new_glr_s = glr_s.clone();
                    new_glr_s.active_state.stack = merged_gss;

                    allow_only_llm_tokens_and_prune_arc(&mut new_glr_s.active_state.stack, llm_token_bv_from_edge, &mut HashMap::new());

                    if new_glr_s.is_ok() {
                        crate::debug!(10, "      - Dest: idx={}, state_bv={}, matched={}, new_gss_ptr={:p}", dest_idx, format_bv(state_id_bv), valid_gss_nodes.len(), new_glr_s.active_state.stack);
                        results.push((dest_idx.clone(), new_glr_s));
                    }
                }
                results
            },
            // merge_fn
            |glr_s1, glr_s2| {
                crate::debug!(10, "    - MERGE: gss1(ptr={:p}) WITH gss2(ptr={:p})", glr_s1.active_state.stack, glr_s2.active_state.stack);
                glr_s1.merge_with(glr_s2);
            },
            // process_fn: (precomputed_node_data, final_state_for_this_path)
            |precomputed_node_data, glr_s| {
                crate::debug!(10, "  - PROCESS: node_ptr={:p}, gss_ptr={:p}", precomputed_node_data as *const _, glr_s.active_state.stack);
                let mut glr_s_copy = glr_s.clone();
                let glr_active_tokens = glr_s_copy.active_state.stack.allowed_llm_tokens();
                let keep_going = glr_s_copy.is_ok();
                if precomputed_node_data.value.end {
                    if !glr_active_tokens.is_empty() {
                        let before = final_mask_internal.borrow().len();
                        *final_mask_internal.borrow_mut() |= &glr_active_tokens;
                        let after = final_mask_internal.borrow().len();
                        if after > before {
                            crate::debug!(10, "    - END NODE. final_mask len: {} -> {} (+{}) with tokens {}", before, after, after - before, format_bv(&glr_active_tokens));
                        }
                    }
                }
                keep_going
            },
        );

        let t_after_special_map = std::time::Instant::now();
        if env::var("RUST_LOG_MASK_TIMING").is_ok() {
            println!("after special_map: {:>15?}", t_after_special_map.duration_since(t0));
        }

        crate::profiler::print_summary_flat();
        crate::profiler::print_summary();
        crate::profiler::reset();

        crate::debug!(10, "\n--- get_mask3 END ---");
        crate::debug!(10, "Final mask internal: {}", format_bv(&final_mask_internal.borrow()));
        crate::debug!(4, "Final mask internal: {}", format_bv(&final_mask_internal.borrow()));
        let final_mask_mapped = self.parent.internal_bv_to_original_precompute3(&final_mask_internal.into_inner());
        crate::debug!(10, "Final mask mapped: {}", format_bv(&final_mask_mapped));
        crate::debug!(4, "Final mask mapped: {}", format_bv(&final_mask_mapped));

        let t_end = std::time::Instant::now();
        if env::var("RUST_LOG_MASK_TIMING").is_ok() {
            println!("get_mask took: {:>15?}", t_end.duration_since(t0));
        }

        final_mask_mapped
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) { // original ID
        let mut self_clone = self.clone();
        self_clone.commit_bytes(&self_clone.parent.llm_vocab.llm_token_map.get_by_right(&llm_token_id)
            .unwrap_or_else(|| panic!("LLM token ID {:?} not found in LLM token map during commit.", llm_token_id))
            .clone());

        // Convert to internal id; if not present or no precomputed entry, fall back.
        let internal_id = self.parent.original_id_to_internal_stage0(llm_token_id)
            .unwrap_or_else(|| panic!("LLM token ID {:?} not found in internal mapping during commit.", llm_token_id));

        // let terminals_map_by_state = self.parent.terminal_map_by_llm.get(&internal_id)
        //     .unwrap_or_else(|| panic!("No terminal map found for internal LLM token ID {:?} during commit.", internal_id));
        // let state_map = self.parent.state_map_by_llm.get(&internal_id)
        //     .unwrap_or_else(|| panic!("No tokenizer state map found for internal LLM token ID {:?} during commit.", internal_id));
        let llm_token_bytes = self.parent.llm_vocab.llm_token_map.get_by_right(&llm_token_id).unwrap().clone();
        let (_state_map, terminals_map_by_state) = self.compute_commit_maps(&llm_token_bytes);
        let state_map = &_state_map;

        if self.state.is_empty() {
            return;
        }

        // 1) Reset LLM tokens on current stacks.
        self.transform_gss_stacks(|stack, memo| reset_llm_tokens(stack, memo));

        // 2) Prune disallowed terminals using the per-token precomputed terminal sets.
        self.transform_gss_stacks(|stack, memo| prune_disallowed_terminals(stack, &terminals_map_by_state, memo));

        // 3) Map tokenizer states
        self.transform_gss_stacks(|stack, memo| map_allowed_terminals_tokenizer_states(stack, state_map, memo));

        // 3) Traverse the precomputed Trie 0 specialized to this token, stepping the GLR state.
        //    We only follow edges whose LLMTokenBV contains this token's internal ID.
        //    We also apply any per-edge "disallowed" terminal constraints as encoded in the edge key.
        if self.state.is_empty() {
            return;
        }

        // Seed the traversal with one entry per active tokenizer state.
        // Carry the GLRParserState directly; we’ll assign final tokenizer state at end nodes.
        let mut initial_values_for_map: Vec<(PrecomputeNode0Index, GLRParserState<'a>)> = Vec::new();
        for (tokenizer_state_id, glr_state) in &self.state {
            let root_idx = self.parent.precomputed0.get(tokenizer_state_id)
                .unwrap_or_else(|| panic!("No precomputed trie root for tokenizer state {:?} during commit.", tokenizer_state_id));
            initial_values_for_map.push((*root_idx, glr_state.clone()));
        }

        let internal_id_val = internal_id.0;
        let mut new_overall_state: BTreeMap<TokenizerStateID, GLRParserState<'a>> = BTreeMap::new();

        Trie::special_map_grouped(
            &self.parent.trie0_god,
            initial_values_for_map,
            // step: for a given edge key, propagate only along children whose edge BV contains the token.
            |glr_s0: &GLRParserState<'a>,
             edge_key: &Option<(GrammarTokenID, Option<TokenizerStateID>)>,
             dest_map: &ordered_hash_map::OrderedHashMap<Trie2Index, LLMTokenBV>| {
                let mut out = Vec::new();

                for (child_idx, edge_bv) in dest_map.iter() {
                    // Only propagate to children compatible with the chosen token.
                    if !edge_bv.contains(internal_id_val) {
                        continue;
                    }

                    let mut glr_s = glr_s0.clone();

                    if let Some((gtid, disallowed_opt)) = edge_key {
                        // Step the GLR state on this grammar token (if any).
                        glr_s.process_token(*gtid);
                        if !glr_s.is_ok() {
                            continue;
                        }

                        // Apply "disallow" rule for immediate repetition at segment boundary if needed.
                        if let Some(end_state) = disallowed_opt {
                            let mut disallowed = crate::datastructures::hybrid_l2_bitset::HybridL2Bitset::new();
                            let mut tbv = TerminalBV::zeros();
                            tbv.insert(gtid.0);
                            disallowed.insert_l2_bitset(end_state.0, tbv);
                            disallow_terminals_and_prune_arc(
                                &mut glr_s.active_state.stack,
                                &disallowed,
                                &mut HashMap::new(),
                            );
                            if !glr_s.is_ok() {
                                continue;
                            }
                        }
                    }

                    out.push((*child_idx, glr_s));
                }

                out
            },
            // merge: merge per-final-tokenizer-state maps, merging GLR states per key.
            |dst: &mut GLRParserState<'a>, src: GLRParserState<'a>| {
                dst.merge_with(src);
            },
            // process: if at end node, accumulate results to new_overall_state and stop; otherwise continue if any GLR state is alive.
            |node, glr_s: &mut GLRParserState<'a>| {
                if !glr_s.is_ok() { return false; }
                if node.value.final_tokenizer_state.is_some() {
                    // Use the final tokenizer state encoded by the end node.
                    let final_tid = node.value.final_tokenizer_state
                        .expect("Trie0 end node must carry a final_tokenizer_state");
                    new_overall_state
                        .entry(final_tid)
                        .and_modify(|g| g.merge_with(glr_s.clone()))
                        .or_insert(glr_s.clone());
                    false
                } else {
                    true
                }
            },
        );

        // Replace with the newly computed per-final-tokenizer-state GLR states.
        self.state = new_overall_state;

        // 5) Cleanup: reset llm tokens to ensure order invariance; fuse; filter dead states.
        self.transform_gss_stacks(|stack, memo| reset_llm_tokens(stack, memo));
        self.map_gss_stacks(|stack, memo| fuse_predecessors_recursive(stack, 1, memo));
        self.state.retain(|_, glr| glr.is_ok());

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

        assert_eq!(*self, self_clone);
    }

    #[time_it]
    pub fn commit_bytes(&mut self, llm_token_bytes: &[u8]) { // llm_token_id is original
        if llm_token_bytes.is_empty() {
            return;
        }

        crate::debug!(3, "Committing bytes: {:?}", String::from_utf8_lossy(llm_token_bytes));

        // for (state_id, state) in &self.state {
        //     crate::debug!(3, "State {} before commit:", state_id.0);
        //     state.log_gss("Before commit", TerminalID(0), false, false);
        // }

        self.transform_gss_stacks(|stack, memo| reset_llm_tokens(stack, memo));

        // Handle allowed terminals
        let (state_map, terminals_map) = self.compute_commit_maps(llm_token_bytes);

        let gss_stats_before_pruning = gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        );
        crate::debug!(5, "Terminals map: {:?}", terminals_map);
        self.transform_gss_stacks(|stack, memo| prune_disallowed_terminals(stack, &terminals_map, memo));
        let gss_stats_after_pruning = gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        );
        crate::debug!(4, "GSS stats before pruning disallowed terminals: {:#?}", gss_stats_before_pruning);
        if gss_stats_after_pruning != gss_stats_before_pruning {
            crate::debug!(4, "GSS stats after pruning disallowed terminals: {:#?}", gss_stats_after_pruning);
            crate::debug!(4, "GSS stats changed after pruning disallowed terminals.");
        } else {
            crate::debug!(4, "GSS stats did not change after pruning disallowed terminals.");
        }

        self.transform_gss_stacks(|stack, memo| map_allowed_terminals_tokenizer_states(stack, &state_map, memo));
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

                    cloned_glr_s.process_token(TerminalID(match_info.id));
                    // cloned_glr_s.do_phase3();

                    if cloned_glr_s.is_ok() {
                        let new_offset = offset + match_info.width;
                        // After a grammar token is consumed, the tokenizer resets for the next segment of the LLM token.
                        let next_tokenizer_id_for_segment = self.parent.tokenizer.initial_state_id();

                        if let Some(end_state_id) = exec_result.end_state {
                            let terminals_accessible_from_end_state = self.parent.tokenizer.tokens_accessible_from_state(TokenizerStateID(end_state_id));
                            if terminals_accessible_from_end_state.contains(&TerminalID(match_info.id)) {
                                let mut disallowed_terminals = crate::datastructures::hybrid_l2_bitset::HybridL2Bitset::new();
                                let mut disallowed_terminals_for_end_state = TerminalBV::zeros();
                                // Disallow this token from being matched again immediately.
                                disallowed_terminals_for_end_state.insert(match_info.id);
                                disallowed_terminals.insert_l2_bitset(end_state_id, disallowed_terminals_for_end_state);
                                    disallow_terminals_and_prune_arc(&mut cloned_glr_s.active_state.stack, &disallowed_terminals, &mut HashMap::new());
                            }
                        }
                        // cloned_glr_s.log_gss(format!("Before disallowing terminals {:?} after committing bytes {:?}", &disallowed_terminals, &llm_token_bytes[offset..new_offset]).as_str(), TerminalID(match_info.id), false, false);
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
        self.transform_gss_stacks(|stack, memo| reset_llm_tokens(stack, memo));
        self.map_gss_stacks(|stack, memo| fuse_predecessors_recursive(stack, 1, memo));
        self.state.retain(|_, glr_parser_state| glr_parser_state.is_ok());


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

        // let mut roots: BTreeMap<TokenizerStateID, Arc<GSSNode>> = BTreeMap::new();
        // for (tokenizer_state_id, glr_state) in &self.state {
        //     roots.insert(*tokenizer_state_id, glr_state.active_state.stack.clone());
        // }
        // simplify(&mut roots);
        // for (tokenizer_state_id, glr_state) in &mut self.state {
        //     glr_state.active_state.stack = roots.get(tokenizer_state_id).unwrap().clone();
        // }

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
