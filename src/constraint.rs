// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use crate::datastructures::ordered_hash_map::Retain;
use crate::r#macro::is_debug_level_enabled;
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
use range_set_blaze::RangeSetBlaze;

use bimap::BiBTreeMap;
use bitvec::prelude::*;

use crate::constraint_extra::{calculate_final_stats2, dump_precompute_trie_recursive, print_precompute_stats2, PrecomputeStats};
use crate::constraint_precompute0_utils;
pub use crate::constraint_precompute0_utils::Trie0Config;
use crate::constraint_precompute1_utils;
pub use crate::constraint_precompute1_utils::Trie1Config;
use crate::constraint_precompute2_utils;
use crate::constraint_precompute2_utils::optimize_trie2_size;
pub use crate::constraint_precompute2_utils::Trie2Config;
use crate::constraint_precompute3_challenge_elimination::eliminate_pushes_and_pops;
use crate::constraint_precompute3_intermediate_utils::{optimize_intermediate_trie3, IntermediateTrie3Config};
use crate::constraint_special_precompute::SpecialPrecomputation;
use crate::datastructures::arc_wrapper::ArcPtrWrapper;
use crate::datastructures::gss_leveled_adapter::{allow_only_llm_tokens_and_prune_arc, disallow_terminals_and_prune_arc, gather_gss_stats, reset_llm_tokens, GSSNode, GSSPrintConfig};
use crate::datastructures::gss_leveled_adapter::{allow_only_llm_tokens_on_stored_trie_nodes_and_prune_arc, Acc};
use crate::datastructures::gss_leveled_adapter::{disallow_llm_tokens_and_prune_arc, fuse_predecessors_recursive, get_roots, map_allowed_terminals_tokenizer_states, print_gss_forest, prune_disallowed_terminals, prune_llm_tokens_by_disallowed_terminals, reset_terminals, sample_path, simplify, simplify_roots_in_place};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use crate::datastructures::trie::{EdgeInserter, PrettyPrintOptions, Trie, Trie2Index, TrieTraversalData};
use crate::datastructures::trie::{God, GodWrapper};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::datastructures::EntryApi;
use crate::equivalence_analysis_finite_automata;
use crate::finite_automata::Regex;
use crate::glr::analyze::compute_terminal_follow_sets;
use crate::glr::grammar::{Symbol, Terminal};
use crate::glr::items::{Item, LRMode, LR_MODE};
use crate::glr::parser::{BelowBottomCacheKey, BelowBottomReductionMode, ExpectElse, GLRParser, GLRParserState, ParseState, ParseStateEdgeContent, ProcessDefaultReductionsAdvancedConfig, ProcessTokenAdvancedConfig};
use crate::glr::table::StateID;
use crate::glr::table::{NonTerminalID, Stage7ShiftsAndReducesLookaheadValue};
use crate::interface::{CompiledGrammar, GrammarDefinition};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::profiler::{print_summary, print_summary_flat, reset, GSS_LOGGING_ENABLED, PROGRESS_BAR_ENABLED};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use deterministic_hash::DeterministicHasher;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use kdam::{tqdm, BarBuilder, BarExt};
use ordered_hash_map::{OrderedHashMap, OrderedHashSet};
use profiler_macro::{time_it, timeit};
use rand::seq::{IndexedRandom, SliceRandom};
use rand::Rng;
use serde_json::Value as SerdeValue;
use std::borrow::Borrow;
use std::collections::btree_map::Entry as BTreeEntry;
use std::collections::BTreeMap as StdMap;
use std::io::{Read, Write};
use std::iter::FromIterator;
use std::ops::{BitAnd, Sub};
use rustc_hash::FxHashMap;
use crate::trie3_opt::{optimize_trie3_size, Trie3Config};

#[derive(Default, Debug)]
struct DfsStats {
    num_pending_edges: usize,
    src_counts: HashMap<Trie2Index, usize>,
    dst_counts: HashMap<Trie2Index, usize>,
    key_counts: HashMap<Option<GrammarTokenID>, usize>,
    bitset_len_dist: BTreeMap<usize, usize>,
    dsts_per_src_key_dist: BTreeMap<usize, usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct EdgeKey {
    src: Trie2Index,
    key: Option<GrammarTokenID>,
    dst: Trie2Index,
}

impl DfsStats {
    fn analyze_pending_edges(&mut self, pending_edges: &FxHashMap<EdgeKey, TokenAcc>) {
        self.num_pending_edges += pending_edges.len();

        let mut dsts_per_src_key: HashMap<(PrecomputeNode1Index, Option<GrammarTokenID>), usize> = HashMap::new();

        for (edge_key, bitset) in pending_edges {
            *self.src_counts.entry(edge_key.src).or_default() += 1;
            *self.dst_counts.entry(edge_key.dst).or_default() += 1;
            *self.key_counts.entry(edge_key.key).or_default() += 1;

            let len = bitset.len();
            *self.bitset_len_dist.entry(len).or_default() += 1;

            *dsts_per_src_key.entry((edge_key.src, edge_key.key)).or_default() += 1;
        }

        for count in dsts_per_src_key.values() {
            *self.dsts_per_src_key_dist.entry(*count).or_default() += 1;
        }
    }

    fn print(&self) {
        println!("\n--- Precomputer1 DFS Stats ---");
        println!("Total pending edges processed: {}", self.num_pending_edges);
        println!("Unique src nodes: {}", self.src_counts.len());
        println!("Unique dst nodes: {}", self.dst_counts.len());
        println!("Unique keys: {}", self.key_counts.len());

        fn print_dist<T: std::fmt::Display + Ord>(name: &str, dist: &BTreeMap<T, usize>) {
            if dist.is_empty() { return; }
            println!("\nDistribution of {}:", name);
            let mut sorted_dist: Vec<_> = dist.iter().collect();
            sorted_dist.sort_by_key(|&(_, count)| std::cmp::Reverse(*count));
            for (val, count) in sorted_dist.iter().take(10) {
                println!("  - {}: {} times", val, count);
            }
            if sorted_dist.len() > 10 { println!("  - ..."); }
        }

        print_dist("Bitset Cardinality", &self.bitset_len_dist);
        print_dist("Destinations per (src, key)", &self.dsts_per_src_key_dist);
        println!("--- End Precomputer1 DFS Stats ---\n");
    }
}

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

// New temporary struct for precompute1
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TempPrecomputedNodeContents {
    pub(crate) end: bool,
    pub(crate) live_tokens: RangeSetBlaze<usize>,
}

impl TempPrecomputedNodeContents {
    pub(crate) fn root(internal_max_llm_token_id: usize) -> Self {
        Self { end: false, live_tokens: RangeSetBlaze::from_iter(0..=internal_max_llm_token_id) }
    }

    pub(crate) fn internal() -> Self {
        Self { end: false, live_tokens: RangeSetBlaze::new() }
    }

    pub(crate) fn leaf() -> Self {
        Self { end: true, live_tokens: RangeSetBlaze::new() }
    }
}

// New temporary type aliases
type TempPrecomputeNode1 = Trie<Option<GrammarTokenID>, RangeSetBlaze<usize>, TempPrecomputedNodeContents>;
type TempPrecomputeNode1Index = Trie2Index;
type TempTrie1GodWrapper = GodWrapper<Option<GrammarTokenID>, RangeSetBlaze<usize>, TempPrecomputedNodeContents>;
type TempCycleReport1 = (Vec<(TempPrecomputeNode1Index, Option<Option<GrammarTokenID>>)>, LLMTokenID);

pub type PrecomputeNode0 = Trie<Option<(GrammarTokenID, Option<TokenizerStateID>)>, LLMTokenBV, PrecomputedNodeContents0>;
pub type PrecomputeNode1 = Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;
pub type PrecomputeNode2 = Trie<(usize, Option<StateID>), LLMTokenBV, PrecomputedNodeContents>;

// New types for intermediate trie 3
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IntermediateTrie3EdgeKey {
    Pop(usize, StateIDBV),
    Push(StateIDBV),
    CheckLLM(LLMTokenBV),
    NoOp,
}

impl Display for IntermediateTrie3EdgeKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Helper to format HybridBitset into ranges
        fn format_bv(bv: &HybridBitset) -> String {
            if bv.is_empty() {
                return "[]".to_string();
            }
            if bv.is_all() {
                return "[ALL]".to_string();
            }

            const MAX_RANGES_TO_SHOW: usize = 10;
            let total_ranges = bv.inner().ranges_len();

            let mut parts: Vec<String> = bv.iter_ranges().take(MAX_RANGES_TO_SHOW).map(|(start, end)| {
                if start == end {
                    format!("{}", start)
                } else if end == usize::MAX {
                    format!("{}..", start)
                } else {
                    format!("{}..={}", start, end)
                }
            }).collect();

            if total_ranges > MAX_RANGES_TO_SHOW {
                parts.push(format!("... ({} more ranges)", total_ranges - MAX_RANGES_TO_SHOW));
            }

            if total_ranges > 1 {
                format!("[{}]", parts.join(", "))
            } else {
                parts.join(", ")
            }
        }

        match self {
            IntermediateTrie3EdgeKey::Pop(n, bv) => write!(f, "Pop({}, {})", n, format_bv(bv)),
            IntermediateTrie3EdgeKey::Push(bv) => write!(f, "Push({})", format_bv(bv)),
            IntermediateTrie3EdgeKey::CheckLLM(bv) => write!(f, "CheckLLM({})", format_bv(bv)),
            IntermediateTrie3EdgeKey::NoOp => write!(f, "NoOp"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IntermediatePrecomputedNodeContents3 {
    pub end: bool,
}

impl Display for IntermediatePrecomputedNodeContents3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.end {
            f.write_str("[END]")?
        }
        Ok(())
    }
}

impl IntermediatePrecomputedNodeContents3 {
    pub fn leaf() -> Self {
        Self { end: true }
    }
    pub fn internal() -> Self {
        Self { end: false }
    }
    pub fn root() -> Self {
        Self { end: false }
    }
}

pub type IntermediatePrecomputeNode3 = Trie<IntermediateTrie3EdgeKey, (), IntermediatePrecomputedNodeContents3>;
pub type IntermediatePrecomputeNode3Index = Trie2Index;
pub type IntermediateTrie3GodWrapper = GodWrapper<IntermediateTrie3EdgeKey, (), IntermediatePrecomputedNodeContents3>;

// Original Trie3 types remain for the final structure
pub type PrecomputeNode3 = Trie<(isize, LLMTokenBV), StateIDBV, PrecomputedNodeContents>;
pub type PrecomputeNode3Index = Trie2Index;

pub type Precomputed0 = BTreeMap<TokenizerStateID, PrecomputeNode0Index>;
pub type Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode1Index>;
pub type Precomputed2 = BTreeMap<TokenizerStateID, PrecomputeNode2Index>;
pub type Precomputed3 = BTreeMap<TokenizerStateID, PrecomputeNode3Index>;

// Indices
pub type PrecomputeNode0Index = Trie2Index;
pub type PrecomputeNode1Index = Trie2Index;
pub type PrecomputeNode2Index = Trie2Index;

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
#[derive(Debug, Clone)]
pub struct Precompute0Cache {
    pub tokenizer: Regex,
    pub llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID>,
    pub max_original_llm_token_id: usize,
    pub precompute0_vocab: StageVocab,
    pub precomputed0: Precomputed0,
    pub trie0_god: Trie0GodWrapper,
}

impl Precompute0Cache {
    pub fn is_compatible(
        &self,
        tokenizer: &Regex,
        llm_token_map: &LLMTokenMap,
        max_original_llm_token_id: usize,
        expected_original_to_internal: &BTreeMap<usize, usize>,
    ) -> bool {
        if &self.tokenizer != tokenizer {
            return false;
        }
        if &self.llm_token_map != llm_token_map {
            return false;
        }
        if self.max_original_llm_token_id != max_original_llm_token_id {
            return false;
        }
        if &self.precompute0_vocab.original_to_internal != expected_original_to_internal {
            return false;
        }
        true
    }
}

impl JSONConvertible for Precompute0Cache {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("tokenizer".to_string(), self.tokenizer.to_json());
        obj.insert("llm_token_map".to_string(), self.llm_token_map.to_json());
        obj.insert("max_original_llm_token_id".to_string(), self.max_original_llm_token_id.to_json());
        obj.insert("precompute0_vocab".to_string(), self.precompute0_vocab.to_json());
        obj.insert("precomputed0".to_string(), self.precomputed0.to_json());
        obj.insert("trie0_god".to_string(), self.trie0_god.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let tokenizer = obj.remove("tokenizer").ok_or("Precompute0Cache: missing tokenizer".to_string()).and_then(Regex::from_json)?;
                let llm_token_map = obj.remove("llm_token_map").ok_or("Precompute0Cache: missing llm_token_map".to_string()).and_then(|n| BiBTreeMap::<Vec<u8>, LLMTokenID>::from_json(n))?;
                let max_original_llm_token_id = obj.remove("max_original_llm_token_id").ok_or("Precompute0Cache: missing max_original_llm_token_id".to_string()).and_then(usize::from_json)?;
                let precompute0_vocab = obj.remove("precompute0_vocab").ok_or("Precompute0Cache: missing precompute0_vocab".to_string()).and_then(StageVocab::from_json)?;
                let precomputed0 = obj.remove("precomputed0").ok_or("Precompute0Cache: missing precomputed0".to_string()).and_then(Precomputed0::from_json)?;
                let trie0_god = obj.remove("trie0_god").ok_or("Precompute0Cache: missing trie0_god".to_string()).and_then(Trie0GodWrapper::from_json)?;
                Ok(Precompute0Cache { tokenizer, llm_token_map, max_original_llm_token_id, precompute0_vocab, precomputed0, trie0_god })
            }
            _ => Err("Precompute0Cache: expected object".to_string()),
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
        timeit!("PrecomputedNodeContents::internal", {});
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
    pub skip_precomputation: bool,
    pub precompute0_only: bool,
    pub trie0: Trie0Config,
    pub trie1: Trie1Config,
    pub trie2: Trie2Config,
    pub trie3: Trie3Config,
    pub intermediate_trie3_templates: IntermediateTrie3Config,
    pub intermediate_trie3_main: IntermediateTrie3Config,
    pub dummy_terminal_map: BTreeMap<String, BTreeSet<Terminal>>,
    pub dummy_terminal_penalties: BTreeMap<String, usize>,
}

impl Default for GrammarConstraintConfig {
    fn default() -> Self {
        Self {
            skip_precomputation: false,
            precompute0_only: false,
            trie0: Trie0Config::off(),
            trie1: Trie1Config::off(),
            trie2: Trie2Config::off(),
            trie3: Trie3Config::default(),
            intermediate_trie3_templates: IntermediateTrie3Config::off(),
            intermediate_trie3_main: IntermediateTrie3Config::off(),
            dummy_terminal_map: BTreeMap::new(),
            dummy_terminal_penalties: BTreeMap::new(),
        }
    }
}

impl GrammarConstraintConfig {
    pub fn off() -> Self {
        Self {
            skip_precomputation: false,
            precompute0_only: false,
            trie0: Trie0Config::off(),
            trie1: Trie1Config::off(),
            trie2: Trie2Config::off(),
            trie3: Trie3Config::off(),
            intermediate_trie3_templates: IntermediateTrie3Config::off(),
            intermediate_trie3_main: IntermediateTrie3Config::off(),
            dummy_terminal_map: BTreeMap::new(),
            dummy_terminal_penalties: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct TrieFeatures {
    num_nodes: usize,
    num_edges: usize,
    pop_count: usize,
    push_count: usize,
    check_llm_count: usize,
    no_op_count: usize,
    end_node_count: usize,
}

impl TrieFeatures {
    /// Calculates the distance to another feature vector.
    /// Returns a value between 0.0 (identical) and some upper bound. Lower is more similar.
    fn distance(&self, other: &Self) -> f64 {
        let mut dist_sq = 0.0;

        dist_sq += Self::feat_dist_sq(self.num_nodes, other.num_nodes);
        dist_sq += Self::feat_dist_sq(self.num_edges, other.num_edges);
        dist_sq += Self::feat_dist_sq(self.pop_count, other.pop_count);
        dist_sq += Self::feat_dist_sq(self.push_count, other.push_count);
        dist_sq += Self::feat_dist_sq(self.check_llm_count, other.check_llm_count);
        dist_sq += Self::feat_dist_sq(self.no_op_count, other.no_op_count);
        dist_sq += Self::feat_dist_sq(self.end_node_count, other.end_node_count);

        dist_sq.sqrt()
    }

    /// Helper for normalized squared distance for a single feature.
    fn feat_dist_sq(v1: usize, v2: usize) -> f64 {
        if v1 == v2 {
            return 0.0;
        }
        let diff = (v1 as f64) - (v2 as f64);
        // Normalize by the max value to get a relative difference.
        let norm = (v1.max(v2) as f64).max(1.0);
        (diff / norm).powi(2)
    }
}

/// Computes a feature vector for a given trie template.
fn compute_trie_features(
    arena: &IntermediateTrie3GodWrapper,
    root: IntermediatePrecomputeNode3Index,
) -> TrieFeatures {
    let nodes = IntermediatePrecomputeNode3::all_nodes(arena, &[root]);
    let num_nodes = nodes.len();
    let mut num_edges = 0;
    let mut pop_count = 0;
    let mut push_count = 0;
    let mut check_llm_count = 0;
    let mut no_op_count = 0;
    let mut end_node_count = 0;

    for node_idx in &nodes {
        if let Some(guard) = node_idx.read(arena) {
            if guard.value.end {
                end_node_count += 1;
            }
            for (ek, dest_map) in guard.children() {
                let edge_count = dest_map.len();
                num_edges += edge_count;
                match ek {
                    IntermediateTrie3EdgeKey::Pop(_, _) => pop_count += edge_count,
                    IntermediateTrie3EdgeKey::Push(_) => push_count += edge_count,
                    IntermediateTrie3EdgeKey::CheckLLM(_) => check_llm_count += edge_count,
                    IntermediateTrie3EdgeKey::NoOp => no_op_count += edge_count,
                }
            }
        }
    }

    TrieFeatures { num_nodes, num_edges, pop_count, push_count, check_llm_count, no_op_count, end_node_count }
}

#[derive(Debug, Clone)]
pub struct GrammarConstraint {
    pub tokenizer:        Regex,
    pub parser:           GLRParser,
    // This is the "raw" trie that maps byte sequences to (grammar token, next tokenizer state) pairs.
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
    pub trie3_traversal_data: Option<TrieTraversalData>,
    pub post_commit_allow_check_mode: TerminalAllowanceCheckMode,
    // Stage-local vocabularies for internal<->original mappings
    pub precompute0_vocab: StageVocab,
    pub precompute_vocab1: StageVocab,
    pub precompute2_vocab: StageVocab,
    pub precompute3_vocab: StageVocab,
    pub special_precomputation: SpecialPrecomputation,
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
        assert_eq!(self.special_precomputation, other.special_precomputation);
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
        obj.insert("max_original_llm_token_id".to_string(), self.llm_vocab.max_original_llm_token_id.to_json());
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
                let max_original_llm_token_id = obj.remove("max_original_llm_token_id")
                    .ok_or_else(|| "Missing field max_original_llm_token_id".to_string())
                    .and_then(usize::from_json)?;
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

                let trie3_roots: Vec<_> = precomputed3.values().cloned().collect();
                let trie3_traversal_data = Trie::compute_traversal_data(&trie3_god, &trie3_roots);

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

                let mut gc = GrammarConstraint {
                    tokenizer,
                    parser,
                    precomputed0,
                    precomputed1,
                    precomputed2,
                    precomputed3,
                    llm_vocab: Arc::new(LLMVocab { llm_token_map, max_original_llm_token_id }),
                    token_name_map,
                    possible_matches,
                    trie0_god,
                    trie1_god,
                    trie2_god,
                    trie3_god,
                    trie3_traversal_data,
                    post_commit_allow_check_mode,
                    state_map_by_llm,
                    terminal_map_by_llm,
                    precompute0_vocab,
                    precompute_vocab1: precompute_vocab,
                    precompute2_vocab,
                    precompute3_vocab,
                    special_precomputation: SpecialPrecomputation::default(),
                };
                gc.special_precomputation = gc.precompute_special();
                Ok(gc)
            }
            _ => Err("Expected JSONNode::Object for GrammarConstraint".to_string()),
        }
    }
}

type CycleReport0 = (Vec<(PrecomputeNode0Index, Option<Option<(GrammarTokenID, Option<TokenizerStateID>)>>)>, LLMTokenID);
type CycleReport1 = (Vec<(PrecomputeNode1Index, Option<Option<GrammarTokenID>>)>, LLMTokenID);

impl GrammarConstraint {
    pub fn from_compiled_grammar(
        compiled_grammar: CompiledGrammar,
        llm_token_map: LLMTokenMap,
        max_original_llm_token_id: usize,
    ) -> Self {
        Self::from_compiled_grammar_with_config(
            compiled_grammar,
            llm_token_map,
            max_original_llm_token_id,
            &GrammarConstraintConfig::default(),
        )
    }

    pub fn from_compiled_grammar_with_config(
        compiled_grammar: CompiledGrammar,
        llm_token_map: LLMTokenMap,
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
        tokenizer: &Regex,
    ) -> BTreeMap<usize, usize>
    {
        // return original_llm_token_map.iter().map(|(bytes, id)| (id.0, id.0)).collect(); // TEMP
        // 1. Prepare inputs for equivalence analysis.
        // We sort the tokens by their byte representation to ensure determinism.
        let mut sorted_tokens: Vec<_> = original_llm_token_map.iter().collect();
        sorted_tokens.sort_by_key(|(bytes, _id)| *bytes);

        let mut llm_token_strings: Vec<Vec<u8>> = Vec::with_capacity(sorted_tokens.len());
        let mut original_ids: Vec<LLMTokenID> = Vec::with_capacity(sorted_tokens.len());

        for (bytes, id) in sorted_tokens {
            llm_token_strings.push(bytes.clone());
            original_ids.push(*id);
        }

        // return original_ids.iter().map(|id| (id.0, id.0)).collect();

        let initial_states: Vec<usize> = tokenizer.iter_states().map(|s| s.0).collect();

        // 2. Find equivalence classes.
        // The result maps a signature vector (representing an equivalence class) to a list
        // of indices into the `llm_token_strings` vector.
        let equivalence_classes = equivalence_analysis_finite_automata::find_equivalence_classes(
            tokenizer,
            &llm_token_strings,
            &initial_states,
        );

        if is_debug_level_enabled(2) {
            let num_original_tokens = llm_token_strings.len();
            let num_classes = equivalence_classes.len();
            crate::debug!(2, "LLM Token Equivalence Analysis:");
            crate::debug!(2, "  - Original LLM tokens: {}", num_original_tokens);
            crate::debug!(2, "  - Equivalence classes: {}", num_classes);
            if num_classes > 0 {
                crate::debug!(2, "  - Reduction factor: {:.2}x", num_original_tokens as f64 / num_classes as f64);
            }

            let mut class_size_dist: BTreeMap<usize, usize> = BTreeMap::new();
            for string_indices in equivalence_classes.values() {
                *class_size_dist.entry(string_indices.len()).or_insert(0) += 1;
            }

            let mut dist_str = String::from("  - Class size distribution (top 10 largest):");
            let mut sorted_dist: Vec<_> = class_size_dist.into_iter().collect();
            sorted_dist.sort_by_key(|&(size, _count)| std::cmp::Reverse(size));
            for (size, count) in sorted_dist.iter().take(10) {
                dist_str.push_str(&format!("\n    - size {}: {} classes", size, count));
            }
            if sorted_dist.len() > 10 {
                dist_str.push_str("\n    - ...");
            }
            crate::debug!(2, "{}", dist_str);
        }

        // 3. Build the mapping from original to internal IDs based on the computed classes.
        // All tokens within the same class will be mapped to the same internal ID.
        let mut original_to_internal_map = BTreeMap::new();
        let mut internal_id_counter = 0;
        // The BTreeMap gives us a deterministic order for assigning internal IDs.
        for (_signature, string_indices) in equivalence_classes {
            let internal_id = internal_id_counter;
            internal_id_counter += 1;

            for string_index in string_indices {
                let original_llm_id = original_ids[string_index];
                original_to_internal_map.insert(original_llm_id.0, internal_id);
            }
        }

        original_to_internal_map
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

    pub fn new_from_grammar_definition(
        grammar_definition: Arc<GrammarDefinition>,
        llm_token_map: LLMTokenMap,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
        precompute0_cache: Option<Precompute0Cache>,
    ) -> Self {
        // This function assumes dummy terminals (e.g., `__DUMMY_TERMINAL_.*__`) are defined in the grammar.
        // It will group real terminals under these dummies based on the structural similarity of their
        // corresponding Trie3 templates.
        let initial_compiled_grammar = CompiledGrammar::from_definition(grammar_definition.clone());
        let parser = &initial_compiled_grammar.glr_parser;

        // We need internal_max_llm_token to build templates, so we compute it early.
        let original_to_internal_map = Self::setup_llm_token_mappings(&llm_token_map, &initial_compiled_grammar.tokenizer);
        let internal_max_llm_token = original_to_internal_map.values().copied().max().unwrap_or(0);

        // Build templates for all terminals to assess similarity.
        let intermediate_trie3_god = IntermediateTrie3GodWrapper::new();
        let templates = Self::build_terminal_trie3_templates(parser, &intermediate_trie3_god, internal_max_llm_token, &config.intermediate_trie3_templates);

        // Group terminals by the similarity of their template trie's features.
        const SIMILARITY_THRESHOLD: f64 = 0.5;

        let mut template_features = BTreeMap::new();
        for (tid, (start, _end)) in &templates {
            let features = compute_trie_features(&intermediate_trie3_god, *start);
            template_features.insert(*tid, features);
        }

        let mut groups: Vec<(TrieFeatures, Vec<TerminalID>, usize)> = Vec::new();

        // Sort templates by terminal ID for deterministic group formation.
        let mut sorted_tids: Vec<_> = templates.keys().copied().collect();
        sorted_tids.sort();

        for tid in sorted_tids {
            let features = template_features.get(&tid).unwrap();
            let complexity = features.num_nodes; // Use num_nodes as complexity metric
            let terminal_name = parser.terminal_map.get_by_right(&tid).unwrap();

            let mut best_group: Option<(usize, f64)> = None;

            for (i, (group_features, _, _)) in groups.iter().enumerate() {
                let dist = features.distance(group_features);
                if best_group.is_none() || dist < best_group.as_ref().unwrap().1 {
                    best_group = Some((i, dist));
                }
            }

            if let Some((best_idx, best_dist)) = best_group {
                if best_dist <= SIMILARITY_THRESHOLD {
                    // Add to existing similar group.
                    let best_group_tid = groups[best_idx].1[0];
                    let best_group_terminal_name = parser.terminal_map.get_by_right(&best_group_tid).unwrap();
                    crate::debug!(2, "Grouping terminal '{}' with group represented by '{}' (distance: {:.2})", terminal_name, best_group_terminal_name, best_dist);
                    groups[best_idx].1.push(tid);
                } else {
                    // No group is similar enough, create a new one.
                    let best_group_tid = groups[best_idx].1[0];
                    let best_group_terminal_name = parser.terminal_map.get_by_right(&best_group_tid).unwrap();
                    crate::debug!(
                        2,
                        "Creating new group for terminal '{}'. Closest group (rep by '{}') has distance {:.2} > threshold {}.",
                        terminal_name,
                        best_group_terminal_name,
                        best_dist,
                        SIMILARITY_THRESHOLD
                    );
                    crate::debug!(3, "  - Features for '{}': {:?}", terminal_name, features);
                    crate::debug!(3, "  - Features for group '{}': {:?}", best_group_terminal_name, &groups[best_idx].0);
                    groups.push((features.clone(), vec![tid], complexity));
                }
            } else {
                // This is the first group.
                crate::debug!(2, "Creating first group for terminal '{}'", terminal_name);
                groups.push((features.clone(), vec![tid], complexity));
            }
        }

        let mut new_config = config.clone();
        new_config.dummy_terminal_map.clear();
        new_config.dummy_terminal_penalties.clear();

        let mut sorted_groups: Vec<_> = groups.into_iter().map(|(_, tids, complexity)| (tids, complexity)).collect();
        // Assign larger groups first to the available dummy terminals.
        sorted_groups.sort_by_key(|(tids, _complexity)| std::cmp::Reverse(tids.len()));

        let mut dummy_idx = 0;
        for (tids, complexity) in sorted_groups {
            if tids.len() <= 1 { // Don't group single-terminal groups
                continue;
            }

            let dummy_name = format!("__DUMMY_TERMINAL_{}__", dummy_idx);
            let original_terminals: BTreeSet<Terminal> = tids.iter()
                .map(|tid| parser.terminal_map.get_by_right(tid).unwrap().clone())
                .collect();
            new_config.dummy_terminal_map.insert(dummy_name.clone(), original_terminals);
            new_config.dummy_terminal_penalties.insert(dummy_name.clone(), complexity);
            dummy_idx += 1;
        }

        println!("\n--- Dummy Terminal Groups ---");
        if new_config.dummy_terminal_map.is_empty() {
            println!("No dummy groups were created. This may be because no terminals were similar enough to be grouped.");
        } else {
            let mut sorted_dummies: Vec<_> = new_config.dummy_terminal_map.iter().collect();
            sorted_dummies.sort_by_key(|(k, _)| *k);

            for (dummy_name, original_terminals) in sorted_dummies {
                let penalty = new_config.dummy_terminal_penalties.get(dummy_name).unwrap_or(&0);
                println!("- {}: (penalty: {})", dummy_name, penalty);
                let mut sorted_originals: Vec<_> = original_terminals.iter().collect();
                sorted_originals.sort();
                for terminal in sorted_originals {
                    println!("  - {}", terminal);
                }
            }
        }
        println!("---------------------------\n");

        // --- RECOMPILATION STEP ---
        let (final_productions, new_dummy_terminals) = crate::glr::analyze::rewrite_productions_with_dummies(
            &grammar_definition.productions,
            &new_config.dummy_terminal_map,
        );

        let final_compiled_grammar = if !new_dummy_terminals.is_empty() {
            crate::debug!(1, "Recompiling grammar with {} new dummy terminals.", new_dummy_terminals.len());
            let mut final_grammar_def = (*grammar_definition).clone();
            final_grammar_def.productions = final_productions;
            for dummy_terminal in new_dummy_terminals {
                if let crate::glr::grammar::Terminal::RegexName(name) = dummy_terminal {
                    final_grammar_def.add_external_terminal(&name);
                }
            }
            println!("{}", &final_grammar_def);
            CompiledGrammar::from_definition(Arc::new(final_grammar_def))
        } else {
            initial_compiled_grammar
        };

        // println!("{}", &final_compiled_grammar.glr_parser);

        Self::new_with_config_and_precompute0_cache(
            final_compiled_grammar.tokenizer,
            final_compiled_grammar.glr_parser,
            llm_token_map,
            final_compiled_grammar.definition.terminal_to_group_id().clone(),
            max_original_llm_token_id,
            &new_config,
            precompute0_cache,
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
        let original_to_internal_map = Self::setup_llm_token_mappings(&llm_token_map, &tokenizer);


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

        crate::debug!(2, "Computing possible_matches for all {} tokenizer states", tokenizer.iter_states().count()); // slow-ish
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
        crate::debug!(2, "Building state_map_by_llm and terminal_map_by_llm");
        let state_map_by_llm = Self::build_state_map_by_llm(&tokenizer, &vocab_for_possible_matches.root); // slow
        crate::debug!(2, "Built state_map_by_llm with {} entries", state_map_by_llm.len());
        let terminal_map_by_llm = Self::rearrange_possible_matches(&computed_possible_matches);

        let grammar_productions = &parser.productions; // Assuming parser is the GLRParser instance
        let grammar_term_map = &parser.terminal_map;

        // These might be computed elsewhere or need to be computed here.
        // Assuming compute_first_sets is available from grammar module.

        crate::debug!(2, "Computing terminal follow sets");
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
                trie3_traversal_data: None,
                post_commit_allow_check_mode: TerminalAllowanceCheckMode::default(),
                state_map_by_llm,
                terminal_map_by_llm,
                precompute0_vocab,
                precompute_vocab1: precompute_vocab,
                precompute2_vocab,
                precompute3_vocab,
                special_precomputation: SpecialPrecomputation::default(),
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
            config
        );

        if config.precompute0_only {
            return Self {
                tokenizer,
                parser,
                precomputed0,
                precomputed1: BTreeMap::new(),
                precomputed2: BTreeMap::new(),
                precomputed3: BTreeMap::new(),
                llm_vocab,
                token_name_map,
                possible_matches: computed_possible_matches,
                trie0_god,
                trie1_god: Trie1GodWrapper::new(),
                trie2_god: Trie2GodWrapper::new(),
                trie3_god: Trie3GodWrapper::new(),
                trie3_traversal_data: None,
                post_commit_allow_check_mode: TerminalAllowanceCheckMode::default(),
                state_map_by_llm,
                terminal_map_by_llm,
                precompute0_vocab,
                precompute_vocab1: precompute_vocab,
                precompute2_vocab,
                precompute3_vocab,
                special_precomputation: SpecialPrecomputation::default(),
            };
        }

        // Check for overlapping original terminals in dummy_terminal_map
        let mut seen_originals = BTreeSet::new();
        for original_terminals in config.dummy_terminal_map.values() { // This is BTreeSet<Terminal> now
            for term in original_terminals {
                if !seen_originals.insert(term.clone()) {
                    panic!("Original terminal '{}' is mapped by multiple dummy terminals.", term);
                }
            }
        }

        let mut original_to_dummy_map: BTreeMap<TerminalID, TerminalID> = BTreeMap::new();
        for (dummy_name, original_terminals) in &config.dummy_terminal_map {
            let dummy_term = Terminal::regex_name(dummy_name);
            if let Some(&dummy_id) = parser.terminal_map.get_by_left(&dummy_term) {
                for original_terminal in original_terminals {
                    if let Some(&original_id) = parser.terminal_map.get_by_left(original_terminal) {
                        original_to_dummy_map.insert(original_id, dummy_id);
                    }
                }
            }
        }
        let (precomputed1, trie1_god) = Self::precompute1(
            &tokenizer,
            Some(&parser),
            Some(llm_vocab.clone()),
            &internal_llm_token_map_for_precompute,
            &token_name_map,
            &mut precompute_vocab,
            &terminal_follow_map,
            config,
            original_to_dummy_map,
        );

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
            &config.trie2,
        );

        // let mut stats2 = PrecomputeStats::default();
        // crate::constraint_extra::calculate_final_stats2(&precomputed2, &mut stats2, &trie2_god);
        // crate::constraint_extra::print_precompute_stats2(&stats2, &trie2_god);

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
            config,
            &mut precompute3_vocab,
        );

        let mut stats3 = PrecomputeStats::default();
        crate::constraint_extra::calculate_final_stats3(&precomputed3, &mut stats3, &trie3_god);
        crate::constraint_extra::print_precompute_stats3(&stats3, &trie3_god);

        let trie3_roots: Vec<_> = precomputed3.values().cloned().collect();
        let trie3_traversal_data = Trie::compute_traversal_data(&trie3_god, &trie3_roots);

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
            trie3_traversal_data,
            post_commit_allow_check_mode: TerminalAllowanceCheckMode::default(),
            state_map_by_llm,
            terminal_map_by_llm,
            precompute0_vocab,
            precompute_vocab1: precompute_vocab,
            precompute2_vocab,
            precompute3_vocab,
            special_precomputation: SpecialPrecomputation::default(),
        };

        gc.special_precomputation = gc.precompute_special();
        gc
    }
    pub fn new_with_config_and_precompute0_cache(
        tokenizer:        Regex,
        parser:           GLRParser,
        llm_token_map:    LLMTokenMap,
        token_name_map:   BiBTreeMap<Terminal, usize>,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
        precompute0_cache: Option<Precompute0Cache>,
    ) -> Self {
        let epsilon_terminal_group_ids: BTreeSet<_> = tokenizer.execute_from_state(&[], tokenizer.initial_state_id()).matches.iter().map(|token| token.id).collect();
        let epsilon_terminals: BTreeSet<&Terminal> = epsilon_terminal_group_ids.iter().map(|id| token_name_map.get_by_right(id).unwrap()).collect();
        assert!(epsilon_terminals.is_empty(), "Epsilon tokens (tokens that can match an empty string) are not supported by the grammar constraint. Got: {:?}", epsilon_terminals);
        let original_to_internal_map = Self::setup_llm_token_mappings(&llm_token_map, &tokenizer);

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

        // Build precomputed per-token (internal) maps.
        crate::debug!(2, "Building state_map_by_llm and terminal_map_by_llm");
        let state_map_by_llm = Self::build_state_map_by_llm(&tokenizer, &vocab_for_possible_matches.root);
        crate::debug!(2, "Built state_map_by_llm with {} entries", state_map_by_llm.len());
        let terminal_map_by_llm = Self::rearrange_possible_matches(&computed_possible_matches);

        let grammar_productions = &parser.productions;
        let grammar_term_map = &parser.terminal_map;

        crate::debug!(2, "Computing terminal follow sets");
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
            llm_token_map: llm_token_map.clone(),
            max_original_llm_token_id,
        });

        // Initialize per-stage vocabularies (start identical to global)
        let mut precompute0_vocab = StageVocab {
            original_to_internal: original_to_internal_map.clone(),
            internal_to_original: internal_to_original_map.clone(),
            internal_max_llm_token: internal_max_llm_token,
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
                trie3_traversal_data: None,
                post_commit_allow_check_mode: TerminalAllowanceCheckMode::default(),
                state_map_by_llm,
                terminal_map_by_llm,
                precompute0_vocab,
                precompute_vocab1: precompute_vocab,
                precompute2_vocab,
                precompute3_vocab,
                special_precomputation: SpecialPrecomputation::default(),
            };
        }

        // Check for overlapping original terminals in dummy_terminal_map
        let mut seen_originals = BTreeSet::new();
        for original_terminals in config.dummy_terminal_map.values() { // BTreeSet<Terminal>
            for term in original_terminals {
                if !seen_originals.insert(term.clone()) {
                    panic!("Original terminal '{}' is mapped by multiple dummy terminals.", term);
                }
            }
        }

        let mut original_to_dummy_map: BTreeMap<TerminalID, TerminalID> = BTreeMap::new();
        for (dummy_name, original_terminals) in &config.dummy_terminal_map {
            let dummy_term = Terminal::regex_name(dummy_name);
            if let Some(&dummy_id) = parser.terminal_map.get_by_left(&dummy_term) {
                for original_terminal in original_terminals {
                    if let Some(&original_id) = parser.terminal_map.get_by_left(original_terminal) {
                        original_to_dummy_map.insert(original_id, dummy_id);
                    }
                }
            }
        }
        // Maybe reuse precompute0 from cache
        let mut precomputed0_opt: Option<(Precomputed0, Trie0GodWrapper)> = None;
        if let Some(cache) = precompute0_cache {
            if cache.is_compatible(&tokenizer, &llm_token_map, max_original_llm_token_id, &original_to_internal_map) {
                crate::debug!(2, "Using cached precompute0");
                precompute0_vocab = cache.precompute0_vocab.clone();
                precompute_vocab = precompute0_vocab.clone();
                precompute2_vocab = precompute_vocab.clone();
                precompute3_vocab = precompute_vocab.clone();
                precomputed0_opt = Some((cache.precomputed0, cache.trie0_god));
            } else {
                crate::debug!(2, "Ignoring cached precompute0 (mismatch with tokenizer/vocab/mapping).");
            }
        }

        let (precomputed0, trie0_god) = if let Some(tuple) = precomputed0_opt {
            tuple
        } else {
            Self::precompute0(
                &tokenizer,
                Some(&parser),
                Some(llm_vocab.clone()),
                &internal_llm_token_map_for_precompute,
                &token_name_map,
                precompute0_vocab.internal_max_llm_token,
                &terminal_follow_map,
                parser.ignore_terminal_id,
                &mut computed_possible_matches,
                config
            )
        };

        if config.precompute0_only {
            return Self {
                tokenizer,
                parser,
                precomputed0,
                precomputed1: BTreeMap::new(),
                precomputed2: BTreeMap::new(),
                precomputed3: BTreeMap::new(),
                llm_vocab,
                token_name_map,
                possible_matches: computed_possible_matches,
                trie0_god,
                trie1_god: Trie1GodWrapper::new(),
                trie2_god: Trie2GodWrapper::new(),
                trie3_god: Trie3GodWrapper::new(),
                trie3_traversal_data: None,
                post_commit_allow_check_mode: TerminalAllowanceCheckMode::default(),
                state_map_by_llm,
                terminal_map_by_llm,
                precompute0_vocab,
                precompute_vocab1: precompute_vocab,
                precompute2_vocab,
                precompute3_vocab,
                special_precomputation: SpecialPrecomputation::default(),
            };
        }

        let (precomputed1, trie1_god) = Self::precompute1(
            &tokenizer,
            Some(&parser),
            Some(llm_vocab.clone()),
            &internal_llm_token_map_for_precompute,
            &token_name_map,
            &mut precompute_vocab,
            &terminal_follow_map,
            config,
            original_to_dummy_map,
        );

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
            &config.trie2,
        );

        let (precomputed3, trie3_god) = Self::precompute3(
            &precomputed1,
            &trie1_god,
            &tokenizer, Some(&parser), Some(llm_vocab.clone()), &internal_llm_token_map_for_precompute, &token_name_map, internal_max_llm_token, &terminal_follow_map, parser.ignore_terminal_id, &mut computed_possible_matches,
            config,
            &mut precompute3_vocab,
        );

        let mut stats3 = PrecomputeStats::default();
        crate::constraint_extra::calculate_final_stats3(&precomputed3, &mut stats3, &trie3_god);
        crate::constraint_extra::print_precompute_stats3(&stats3, &trie3_god);

        let trie3_roots: Vec<_> = precomputed3.values().cloned().collect();
        let trie3_traversal_data = Trie::compute_traversal_data(&trie3_god, &trie3_roots);

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
            trie3_traversal_data,
            post_commit_allow_check_mode: TerminalAllowanceCheckMode::default(),
            state_map_by_llm,
            terminal_map_by_llm,
            precompute0_vocab,
            precompute_vocab1: precompute_vocab,
            precompute2_vocab,
            precompute3_vocab,
            special_precomputation: SpecialPrecomputation::default(),
        };

        gc.special_precomputation = gc.precompute_special();
        gc
    }

    pub fn export_precompute0_cache(&self) -> Precompute0Cache {
        Precompute0Cache {
            tokenizer: self.tokenizer.clone(),
            llm_token_map: self.llm_vocab.llm_token_map.clone(),
            max_original_llm_token_id: self.llm_vocab.max_original_llm_token_id,
            precompute0_vocab: self.precompute0_vocab.clone(),
            precomputed0: self.precomputed0.clone(),
            trie0_god: self.trie0_god.clone(),
        }
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
        _possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
        config: &GrammarConstraintConfig,
    ) -> (BTreeMap<TokenizerStateID, PrecomputeNode0Index>, Trie0GodWrapper) {
        (BTreeMap::new(), Trie0GodWrapper::new())
    }

    fn has_llm_compatible_cycle0(
        arena: &Trie0GodWrapper,
        roots: &[PrecomputeNode0Index],
        internal_max_llm_token: usize,
    ) {
    }

    fn has_llm_compatible_cycle_temp(
        arena: &TempTrie1GodWrapper,
        roots: &[TempPrecomputeNode1Index],
        internal_max_llm_token: usize,
    ) {
        let mut visited: HashMap<TempPrecomputeNode1Index, RangeSetBlaze<usize>> = HashMap::new();
        let initial_tokens = RangeSetBlaze::from_iter(0..=internal_max_llm_token);

        for &root in roots {
            if let Some((cycle_path, llm_token_id)) = Self::detect_cycle_recursive_temp(
                root,
                None,
                initial_tokens.clone(),
                arena,
                &mut HashMap::new(),
                &mut visited,
                &mut Vec::new(),
            ) {
                let mut report = format!(
                    "LLM-compatible cycle detected in precompute1 trie for internal LLM token ID {}.\nCycle path:\n",
                    llm_token_id.0
                );
                for i in 0..cycle_path.len() {
                    let (node_idx, _) = cycle_path[i];
                    let next_i = (i + 1) % cycle_path.len();
                    let (next_node_idx, edge_to_next_opt) = &cycle_path[next_i];
                    let edge_str = edge_to_next_opt.as_ref().map_or_else(
                        || " (root edge)".to_string(),
                        |ek| format!("{:?}", ek),
                    );
                    report.push_str(&format!("  {} --[{}]--> {}\n", node_idx, edge_str, next_node_idx));
                }
                panic!("{}", report);
            }
        }
    }

    fn detect_cycle_recursive_temp(
        node_idx: TempPrecomputeNode1Index,
        edge_key_opt: Option<Option<GrammarTokenID>>,
        current_tokens: RangeSetBlaze<usize>,
        arena: &TempTrie1GodWrapper,
        recursion_stack: &mut HashMap<TempPrecomputeNode1Index, (RangeSetBlaze<usize>, usize)>,
        visited: &mut HashMap<TempPrecomputeNode1Index, RangeSetBlaze<usize>>,
        path: &mut Vec<(TempPrecomputeNode1Index, Option<Option<GrammarTokenID>>)>,
    ) -> Option<TempCycleReport1> {
        path.push((node_idx, edge_key_opt));

        if let Some((tokens_on_stack, path_start_idx)) = recursion_stack.get(&node_idx) {
            let intersection = &current_tokens & tokens_on_stack;
            if !intersection.is_empty() {
                let cycle_llm_token = intersection.iter().next().unwrap();
                let cycle_path = path[*path_start_idx..].to_vec();
                path.pop();
                return Some((cycle_path, LLMTokenID(cycle_llm_token)));
            }
        }

        let new_tokens_to_process = match visited.entry(node_idx) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let previously_visited_tokens = entry.get_mut();
                let new_unseen_tokens = &current_tokens - &*previously_visited_tokens;
                if new_unseen_tokens.is_empty() {
                    path.pop();
                    return None;
                }
                *previously_visited_tokens |= &current_tokens;
                new_unseen_tokens
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(current_tokens.clone());
                current_tokens.clone()
            }
        };

        recursion_stack.insert(node_idx, (current_tokens, path.len() - 1));

        let children_to_visit = if let Some(guard) = node_idx.read(arena) {
            guard.children().clone()
        } else {
            recursion_stack.remove(&node_idx);
            path.pop();
            return None;
        };

        for (edge_key, dest_map) in children_to_visit.iter() {
            for (child_idx, edge_tokens) in dest_map.iter() {
                let next_tokens = &new_tokens_to_process & edge_tokens;
                if !next_tokens.is_empty() {
                    if let Some(report) = Self::detect_cycle_recursive_temp(
                        *child_idx, Some(edge_key.clone()), next_tokens, arena, recursion_stack, visited, path,
                    ) {
                        return Some(report);
                    }
                }
            }
        }

        recursion_stack.remove(&node_idx);
        path.pop();
        None
    }

    fn has_llm_compatible_cycle(
        arena: &Trie1GodWrapper,
        roots: &[PrecomputeNode1Index],
        internal_max_llm_token: usize,
    ) {
        let mut visited: HashMap<PrecomputeNode1Index, LLMTokenBV> = HashMap::new();
        let initial_tokens = LLMTokenBV::ones(internal_max_llm_token + 1);

        for &root in roots {
            if let Some((cycle_path, llm_token_id)) = Self::detect_cycle_recursive(
                root,
                None,
                initial_tokens.clone(),
                arena,
                &mut HashMap::new(),
                &mut visited,
                &mut Vec::new(),
            ) {
                let mut report = format!(
                    "LLM-compatible cycle detected in precompute1 trie for internal LLM token ID {}.\nCycle path:\n",
                    llm_token_id.0
                );
                for i in 0..cycle_path.len() {
                    let (node_idx, _) = cycle_path[i];
                    let next_i = (i + 1) % cycle_path.len();
                    let (next_node_idx, edge_to_next_opt) = &cycle_path[next_i];
                    let edge_str = edge_to_next_opt.as_ref().map_or_else(
                        || " (root edge)".to_string(),
                        |ek| format!("{:?}", ek),
                    );
                    report.push_str(&format!("  {} --[{}]--> {}\n", node_idx, edge_str, next_node_idx));
                }
                panic!("{}", report);
            }
        }
    }

    fn detect_cycle_recursive(
        node_idx: PrecomputeNode1Index,
        edge_key_opt: Option<Option<GrammarTokenID>>,
        current_tokens: LLMTokenBV,
        arena: &Trie1GodWrapper,
        recursion_stack: &mut HashMap<PrecomputeNode1Index, (LLMTokenBV, usize)>,
        visited: &mut HashMap<PrecomputeNode1Index, LLMTokenBV>,
        path: &mut Vec<(PrecomputeNode1Index, Option<Option<GrammarTokenID>>)>,
    ) -> Option<CycleReport1> {
        path.push((node_idx, edge_key_opt));

        if let Some((tokens_on_stack, path_start_idx)) = recursion_stack.get(&node_idx) {
            let intersection = &current_tokens & tokens_on_stack;
            if !intersection.is_empty() {
                let cycle_llm_token = intersection.iter().next().unwrap();
                let cycle_path = path[*path_start_idx..].to_vec();
                path.pop();
                return Some((cycle_path, LLMTokenID(cycle_llm_token)));
            }
        }

        let new_tokens_to_process = match visited.entry(node_idx) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let previously_visited_tokens = entry.get_mut();
                let new_unseen_tokens = &current_tokens - &*previously_visited_tokens;
                if new_unseen_tokens.is_empty() {
                    path.pop();
                    return None;
                }
                *previously_visited_tokens |= &current_tokens;
                new_unseen_tokens
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(current_tokens.clone());
                current_tokens.clone()
            }
        };

        recursion_stack.insert(node_idx, (current_tokens, path.len() - 1));

        let children_to_visit = if let Some(guard) = node_idx.read(arena) {
            guard.children().clone()
        } else {
            recursion_stack.remove(&node_idx);
            path.pop();
            return None;
        };

        for (edge_key, dest_map) in children_to_visit.iter() {
            for (child_idx, edge_tokens) in dest_map.iter() {
                let next_tokens = &new_tokens_to_process & edge_tokens;
                if !next_tokens.is_empty() {
                    if let Some(report) = Self::detect_cycle_recursive(
                        *child_idx, Some(edge_key.clone()), next_tokens, arena, recursion_stack, visited, path,
                    ) {
                        return Some(report);
                    }
                }
            }
        }

        recursion_stack.remove(&node_idx);
        path.pop();
        None
    }

    pub fn precompute1(
        tokenizer: &Regex,
        parser: Option<&GLRParser>,
        llm_vocab: Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        token_name_map: &BiBTreeMap<Terminal, usize>,
        stage_vocab: &mut StageVocab,
        terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        config: &GrammarConstraintConfig,
        original_to_dummy_map: BTreeMap<TerminalID, TerminalID>,
    ) -> (BTreeMap<TokenizerStateID, PrecomputeNode1Index>, Trie1GodWrapper) {
        let mut dummy_terminal_penalties: BTreeMap<TerminalID, usize> = BTreeMap::new();
        if !config.dummy_terminal_penalties.is_empty() {
            if let Some(p) = parser {
                for (dummy_name, penalty) in &config.dummy_terminal_penalties {
                    let dummy_term = Terminal::regex_name(dummy_name);
                    if let Some(&dummy_id) = p.terminal_map.get_by_left(&dummy_term) {
                        dummy_terminal_penalties.insert(dummy_id, *penalty);
                    }
                }
            }
        } else {
            for dummy_tid in original_to_dummy_map.values() {
                *dummy_terminal_penalties.entry(*dummy_tid).or_default() += 1;
            }
        }

        let mut helper = Precomputer1::new(
            tokenizer,
            parser,
            llm_vocab,
            internal_llm_token_map,
            stage_vocab.internal_max_llm_token,
            MERGE_THRESHOLD,
            terminal_follow_map,
            parser.and_then(|p| p.ignore_terminal_id),
            token_name_map,
            original_to_dummy_map,
        );

        helper.run_dfs();
        let roots_before: Vec<_> = helper.roots.values().cloned().collect();
        Self::has_llm_compatible_cycle_temp(&helper.trie1_god, &roots_before, stage_vocab.internal_max_llm_token);

        let (mut precomputed1, trie1_god) = helper.finish();
        let roots_after: Vec<_> = precomputed1.values().cloned().collect();
        Self::has_llm_compatible_cycle(&trie1_god, &roots_after, stage_vocab.internal_max_llm_token);

        let mut stats = PrecomputeStats::default();
        crate::constraint_extra::calculate_final_stats1(&precomputed1, &mut stats, &trie1_god);
        crate::constraint_extra::print_precompute_stats1(&stats, token_name_map, &trie1_god);

        constraint_precompute1_utils::optimize_trie1_size(
            &mut precomputed1,
            &trie1_god,
            // Dummy values for trie0-dependent params
            &Trie0GodWrapper::new(),
            &HashMap::new(),
            parser.and_then(|p| p.ignore_terminal_id),
            stage_vocab.internal_max_llm_token,
            terminal_follow_map,
            &config.trie1,
            stage_vocab,
            token_name_map,
            &dummy_terminal_penalties,
        );

        (precomputed1, trie1_god)
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
        config: &Trie2Config,
    ) -> (Precomputed2, Trie2GodWrapper) {
        (BTreeMap::new(), Trie2GodWrapper::new())
    }

    /// Build a terminal -> (start_node, end_node) map in the given Trie3 arena.
    /// Each entry is a "template" subgraph for consuming a single grammar terminal:
    /// - start_node: entry point for this terminal
    /// - end_node: exit node for this terminal
    /// The subgraph between (start_node,end_node) encodes the stack checks derived from
    /// running the GLR state in hallucinated mode and flattening stacks via to_stacks().
    pub fn build_terminal_trie3_templates(
        parser: &GLRParser,
        trie3_god: &IntermediateTrie3GodWrapper,
        internal_max_llm_token: usize,
        config: &IntermediateTrie3Config,
    ) -> BTreeMap<TerminalID, (IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index)> {
        let mut out = BTreeMap::new();
        // Iterate terminals deterministically by ID
        let mut term_ids: Vec<TerminalID> = parser.terminal_map.iter().map(|(_l, r)| *r).collect();
        term_ids.sort_by_key(|t| t.0);

        for tid in term_ids {
            let (start, end) = Self::build_trie3_template_for_terminal(parser, trie3_god, tid);
            // Temporarily mark the end node as 'end' for optimization purposes
            end.write(trie3_god).unwrap().value.end = true;
            out.insert(tid, (start, end));
        }

        // Global, cross-template optimization pass (merge identical subgraphs, compress NoOp chains).
        let template_roots: Vec<_> = out.values().map(|(start, _end)| start.clone()).collect();
        let node_map = optimize_intermediate_trie3(
            &template_roots,
            trie3_god,
            |_, node| node.value.end,
            config,
        );

        // Update the start nodes in the template map
        let mut new_out = BTreeMap::new();
        for (tid, (start, end)) in out {
            let new_start = node_map.get(&start).unwrap_or(&start).clone();
            let new_end = node_map.get(&end).unwrap_or(&end).clone();
            new_out.insert(tid, (new_start, new_end));
        }

        // Revert the 'end' flag on the template end nodes.
        // The optimization process relies on the 'end' flag being set for the template end node,
        // but the final trie3 construction needs it to be false (as it's an internal node).
        for (_tid, (_start, end)) in new_out.iter() {
            end.write(trie3_god).unwrap().value.end = false;
        }

        new_out
    }


    /// Build a (start,end) template for a single terminal:
    /// 1) Seed hallucinated GLR state with an Acc that stores `start`.
    /// 2) Process the terminal with ContinueFromHallucinateState.
    /// 3) Flatten stacks to sequences and convert them into a Trie3 path using (-1)-pop edges
    ///    for state checks. Finally, converge all sequences into a single shared `end` node.
    fn build_trie3_template_for_terminal(
        parser: &GLRParser,
        trie3_god: &IntermediateTrie3GodWrapper,
        tid: TerminalID,
    ) -> (IntermediatePrecomputeNode3Index, IntermediatePrecomputeNode3Index) {
        // Create template start node
        println!("reduce_gss_stacks_to_trie3_paths_from_start: {:?}", tid);
        let start = IntermediatePrecomputeNode3Index::new(trie3_god.insert(IntermediatePrecomputeNode3::new(IntermediatePrecomputedNodeContents3::internal())));

        // Seed hallucinated GLR state with this start node in Acc
        let mut acc = Acc::new_fresh();
        acc.stored_trie_nodes_mut().insert(start.clone());
        let mut s = parser.init_parser_state_hallucinated_with_acc(acc).with_god(trie3_god.clone());
        let cfg = ProcessTokenAdvancedConfig {
            below_bottom_mode: BelowBottomReductionMode::ContinueFromHallucinateState,
            current_token: Some(tid),
            reset_cache: true,
        };
        s.process_token_advanced(tid, &cfg);

        // Flatten the active GSS into explicit stacks.
        let stacks = s.active_state.stack.inner.to_stacks();
        dbg!(&stacks);

        // This new function will build the paths for the stack *below* the shifted state.
        let final_nodes_map = Self::reduce_gss_stacks_to_trie3_paths_from_start(trie3_god, &stacks);

        // Create a single end node for all paths to converge to.
        let end = IntermediatePrecomputeNode3Index::new(trie3_god.insert(IntermediatePrecomputeNode3::new(IntermediatePrecomputedNodeContents3::internal())));

        // For each possible shifted state, add the final Push edge to the common end node.
        for (shifted_state_content, path_end_node) in final_nodes_map {
            let mut state_bv = StateIDBV::zeros();
            state_bv.insert(shifted_state_content.state_id.0);
            let inserter = EdgeInserter::new(
                trie3_god,
                path_end_node.as_arc().clone(),
                IntermediateTrie3EdgeKey::Push(state_bv), (), |_, _| {}, |_, _| {}, |_, _| {},
            );
            inserter.try_destination(end.clone()).expect("Failed to insert final Push edge in template");
        }

        (start, end)
    }

    /// Convert GSS stacks into Trie3 paths.
    /// This function processes stacks that result from a shift action. Each stack is a sequence of
    /// `ParseStateEdgeContent`. The first element (`items[0]`) is the new state after the shift.
    /// The rest of the elements (`items[1..]`) form the stack that was present before the shift.
    ///
    /// This function builds a trie representing the `items[1..]` part of all stacks, with `Push`
    /// operations for each state. It returns a map from the shifted state (`items[0]`) to the
    /// trie node that represents the end of the corresponding pre-shift stack path.
    fn reduce_gss_stacks_to_trie3_paths_from_start(
        trie3_god: &IntermediateTrie3GodWrapper,
        stacks: &[(Vec<ParseStateEdgeContent>, Acc)],
    ) -> BTreeMap<ParseStateEdgeContent, IntermediatePrecomputeNode3Index> {
        let mut final_nodes_map: BTreeMap<ParseStateEdgeContent, IntermediatePrecomputeNode3Index> = BTreeMap::new();

        for (items, acc) in stacks.iter() {
            let mut cur = IntermediatePrecomputeNode3Index::new(
                trie3_god.insert(IntermediatePrecomputeNode3::new(
                    IntermediatePrecomputedNodeContents3::internal()
                ))
            );
            for src in acc.stored_trie_nodes().iter() {
                let inserter = EdgeInserter::new(
                    trie3_god,
                    src.as_arc().clone(),
                    IntermediateTrie3EdgeKey::NoOp, (), |_, _| {}, |_, _| {}, |_, _| {},
                );
                inserter.try_destination(cur.clone()).expect("Failed to insert unconditional edge to template head");
            }

            let items = &items[1..]; // Skip first element
            if items.is_empty() {
                continue;
            }

            if items.len() > 0 {
                println!("ITEMS: {:?}", items);
            }

            let shifted_state_content = *items.last().unwrap();
            let pre_shift_stack = &items[..items.len() - 1];

            // Walk the pre-shift stack from top to bottom to build the trie path
            for state_content in pre_shift_stack {
                let mut state_bv = StateIDBV::zeros();
                state_bv.insert(state_content.state_id.0);
                let inserter = EdgeInserter::new(
                    trie3_god,
                    cur.as_arc().clone(),
                    IntermediateTrie3EdgeKey::Push(state_bv),
                    (),
                    |_, _| {},
                    |_, _| {},
                    |_, _| {},
                );
                let next = IntermediatePrecomputeNode3Index::new(
                    trie3_god.insert(IntermediatePrecomputeNode3::new(
                        IntermediatePrecomputedNodeContents3::internal()
                    ))
                );
                cur = inserter.try_destination(next)
                    .expect("Failed to insert Push edge in template chain");
            }

            final_nodes_map.insert(shifted_state_content, cur);
        }

        final_nodes_map
    }

    fn _process_and_rebuild_trie3_paths(
        intermediate_precomputed3: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
        intermediate_trie3_god: &IntermediateTrie3GodWrapper,
    ) {
        eliminate_pushes_and_pops(
            intermediate_precomputed3,
            intermediate_trie3_god,
        );
        // After elimination, assert that no pop operations are reachable from any push operations.
        // This is a key invariant that the elimination process should establish.
        crate::constraint_precompute3_challenge_elimination::assert_no_pops_reachable_from_pushes(
            intermediate_precomputed3,
            intermediate_trie3_god,
        );
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
        _possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
        config: &GrammarConstraintConfig,
        stage_vocab: &mut StageVocab,
    ) -> (Precomputed3, Trie3GodWrapper) {
        crate::debug!(2, "Precomputing Trie 3 (template-driven)...");
        let roots: Vec<PrecomputeNode1Index> = precomputed1.values().cloned().collect();
        // assert!(!Trie::has_cycle(trie1_god, roots));
        let mut intermediate_precomputed3 = BTreeMap::new();
        let intermediate_trie3_god = IntermediateTrie3GodWrapper::new();

        // Build per-terminal template subgraphs once in this arena.
        let terminal_templates = Self::build_terminal_trie3_templates(parser.unwrap(), &intermediate_trie3_god, internal_max_llm_token, &config.intermediate_trie3_templates);
        for (tid, (start, end)) in &terminal_templates {
            let terminal = token_name_map.get_by_right(&tid.0).unwrap();
            println!("\n--- Intermediate Trie3 Template for terminal {}: ---", terminal);
            println!("End node: {:?}", end);
            let mut options = crate::datastructures::trie::PrettyPrintOptions::default()
                .display_edge_keys_only()
                .display_nodes()
                .omit_depth()
                ;
            println!("{}", Trie::pretty_print_with_options(&intermediate_trie3_god, &[*start], &options));
        }

        if is_debug_level_enabled(2) {
            println!("\n--- Intermediate Trie3 Template Statistics ---");
            println!("{:<25} {:<5} {:>10} {:>10} {:>12}", "Terminal Name", "ID", "Nodes", "Edges", "Cyclic Nodes");
            println!("{:-<25} {:-<5} {:-<10} {:-<10} {:-<12}", "", "", "", "", "");

            let mut sorted_templates: Vec<_> = terminal_templates.iter().collect();
            sorted_templates.sort_by_key(|(tid, _)| *tid);

            for (tid, (start_node, _end_node)) in sorted_templates {
                let mut stats = crate::constraint_extra::PrecomputeStats::default();
                crate::constraint_extra::calculate_intermediate_stats3(
                    &[*start_node],
                    &mut stats,
                    &intermediate_trie3_god,
                );
                let terminal_name = parser.unwrap().terminal_map.get_by_right(tid).unwrap();

                let cyclic_nodes = IntermediatePrecomputeNode3::nodes_in_cycles(&intermediate_trie3_god, &[*start_node]);
                let cyclic_info = if cyclic_nodes.is_empty() {
                    "No".to_string()
                } else {
                    format!("Yes ({})", cyclic_nodes.len())
                };

                println!(
                    "{:<25} {:<5} {:>10} {:>10} {:>12}",
                    format!("'{}'", terminal_name),
                    tid.0,
                    stats.final_unique_nodes_count,
                    stats.final_edges_count,
                    cyclic_info
                );
            }
            println!("-----------------------------------------------------------------\n");

            // Aggregate stats across all templates: union nodes/edges and shared nodes count
            {
                use std::collections::{HashMap, HashSet};

                let mut union_nodes: HashSet<IntermediatePrecomputeNode3Index> = HashSet::new();
                let mut coverage: HashMap<IntermediatePrecomputeNode3Index, usize> = HashMap::new();

                for (_tid, (start_node, _end_node)) in &terminal_templates {
                    let nodes = Trie::all_nodes(&intermediate_trie3_god, &[*start_node]);
                    for n in nodes {
                        union_nodes.insert(n);
                        *coverage.entry(n).or_insert(0) += 1;
                    }
                }

                let mut union_edges = 0usize;
                for n in &union_nodes {
                    if let Some(g) = n.read(&intermediate_trie3_god) {
                        for (_ek, dm) in g.children() {
                            union_edges += dm.len();
                        }
                    }
                }
                let shared_nodes = coverage.values().filter(|&&c| c > 1).count();
                let shared_pct = if union_nodes.is_empty() { 0.0 } else { (shared_nodes as f64) * 100.0 / (union_nodes.len() as f64) };

                println!("Union Across All Templates:");
                println!("  Unique nodes: {}", union_nodes.len());
                println!("  Total edges: {}", union_edges);
                println!("  Nodes shared by >= 2 templates: {} ({:.1}%)", shared_nodes, shared_pct);
                println!("--------------------------------------------\n");
            }
        }

        if is_debug_level_enabled(3) {
            println!("\n--- Terminal Template Paths ---");
            let mut sorted_templates: Vec<_> = terminal_templates.iter().collect();
            sorted_templates.sort_by_key(|(tid, _)| *tid);

            for (tid, (start_node, end_node)) in sorted_templates {
                let terminal_name = parser.unwrap().terminal_map.get_by_right(tid).unwrap();
                println!("Template for terminal '{}' ({}):", terminal_name, tid.0);

                let template_paths = IntermediatePrecomputeNode3::get_all_paths(
                    &intermediate_trie3_god,
                    &[start_node.clone()],
                    |idx, _node| idx == *end_node
                );

                if template_paths.is_empty() {
                    println!("  (No paths found to end node)");
                }

                for (_root_value, path_edges) in &template_paths {
                    let edge_keys_str: Vec<_> = path_edges.iter()
                        .filter(|(ek, _, _)| !matches!(ek, IntermediateTrie3EdgeKey::NoOp))
                        .map(|(ek, _, _)| format!("{}", ek))
                        .collect();
                    if !edge_keys_str.is_empty() {
                        println!("  [{}]", edge_keys_str.join(", "));
                    }
                }
            }
            println!("--- End Terminal Template Paths ---\n");
        }

        // Group tokenizer states by shared Trie1 root
        let mut trie1_roots_to_tokenizer_states: BTreeMap<PrecomputeNode1Index, Vec<TokenizerStateID>> = BTreeMap::new();
        for (tokenizer_state_id, trie1_root) in precomputed1.iter() {
            trie1_roots_to_tokenizer_states.entry(trie1_root.clone()).or_default().push(*tokenizer_state_id);
        }

        // Create Trie3 roots and seed initial sets
        let mut initial_values_for_map: Vec<(PrecomputeNode1Index, (LLMTokenBV, BTreeSet<IntermediatePrecomputeNode3Index>))> = Vec::new();
        let all_tokens = LLMTokenBV::ones(internal_max_llm_token + 1);
        for (trie1_root, tokenizer_state_ids) in &trie1_roots_to_tokenizer_states {
            let trie3_root = IntermediatePrecomputeNode3Index::new(intermediate_trie3_god.insert(IntermediatePrecomputeNode3::new(IntermediatePrecomputedNodeContents3::internal())));
            for tokenizer_state_id in tokenizer_state_ids {
                intermediate_precomputed3.insert(*tokenizer_state_id, trie3_root.clone());
            }
            let mut seed = BTreeSet::new();
            seed.insert(trie3_root.clone());
            initial_values_for_map.push((trie1_root.clone(), (all_tokens.clone(), seed)));
        }

        // Shared end node for Trie1-end positions
        let trie3_end = IntermediatePrecomputeNode3Index::new(intermediate_trie3_god.insert(IntermediatePrecomputeNode3::new(IntermediatePrecomputedNodeContents3::leaf())));

        let trie1_roots_for_traversal: Vec<_> = initial_values_for_map.iter().map(|(idx, _)| *idx).collect();
        let traversal_data = Trie::compute_traversal_data(&trie1_god, &trie1_roots_for_traversal)
            .expect("Failed to compute traversal data for trie1 in precompute3");

        crate::debug!(2, "Entering precompute3 special_map_grouped");
        Trie::special_map_grouped(
            &trie1_god,
            &traversal_data,
            initial_values_for_map,
            // step: merge current set into a single node, then attach the terminal template or direct LLM edges
            |(current_tokens, current_nodes_set), edge_grammar_token_opt, destinations_map| {
                // Merge current set into a single node with unconditional edges
                let merged = IntermediatePrecomputeNode3Index::new(intermediate_trie3_god.insert(IntermediatePrecomputeNode3::new(IntermediatePrecomputedNodeContents3::internal())));
                for src in current_nodes_set.iter() {
                    let inserter = EdgeInserter::new(
                        &intermediate_trie3_god,
                        src.as_arc().clone(),
                        IntermediateTrie3EdgeKey::NoOp, (), |_, _| {}, |_, _| {}, |_, _| {},
                    );
                    inserter.try_destination(merged.clone()).expect("Failed to insert merge edge in Trie3 step");
                }

                let mut out = Vec::new();
                match edge_grammar_token_opt {
                    Some(tid) => {
                        // Copy the terminal's template and hook it after merged
                        let (templ_start, templ_end) = terminal_templates.get(&tid).expect("template for terminal missing");
                        let (copied_roots, id_map) = IntermediatePrecomputeNode3::deep_copy_subtrees_into(
                            &intermediate_trie3_god,
                            &intermediate_trie3_god,
                            &[*templ_start],
                        );
                        let copied_start = copied_roots.into_iter().next().unwrap();
                        let copied_end = id_map.get(templ_end).unwrap_or(&copied_start).clone();

                        // Connect merged -> copied_start unconditionally
                        {
                            let inserter = EdgeInserter::new(
                                &intermediate_trie3_god,
                                merged.as_arc().clone(),
                                IntermediateTrie3EdgeKey::NoOp, (), |_, _| {}, |_, _| {}, |_, _| {},
                            );
                            inserter.try_destination(copied_start.clone()).expect("Failed to hook merged node to copied template start");
                        }

                        // For each destination in Trie1, fork a node from copied_end with LLM tokens on the edge.
                        for (dst_node_wrapper, edge_bv) in destinations_map.iter() {
                            let next_tokens = &*current_tokens & edge_bv;
                            if next_tokens.is_empty() { continue; }

                            let next = IntermediatePrecomputeNode3Index::new(intermediate_trie3_god.insert(IntermediatePrecomputeNode3::new(IntermediatePrecomputedNodeContents3::internal())));
                            let inserter = EdgeInserter::new(
                                &intermediate_trie3_god,
                                copied_end.as_arc().clone(),
                                IntermediateTrie3EdgeKey::CheckLLM(edge_bv.clone()), (), |_, _| {}, |_, _| {}, |_, _| {},
                            );
                            let actual = inserter.try_destination(next.clone()).expect("Failed to add LLM edge after template end");
                            let mut s = BTreeSet::new();
                            s.insert(actual);
                            out.push((dst_node_wrapper.clone(), (next_tokens, s)));
                        }
                    }
                    None => {
                        // No grammar token on this edge: fan out directly from merged with LLM-token edges.
                        for (dst_node_wrapper, edge_bv) in destinations_map.iter() {
                            let next_tokens = &*current_tokens & edge_bv;
                            if next_tokens.is_empty() { continue; }

                            let next = IntermediatePrecomputeNode3Index::new(intermediate_trie3_god.insert(IntermediatePrecomputeNode3::new(IntermediatePrecomputedNodeContents3::internal())));
                            let inserter = EdgeInserter::new(
                                &intermediate_trie3_god,
                                merged.as_arc().clone(),
                                IntermediateTrie3EdgeKey::CheckLLM(edge_bv.clone()), (), |_, _| {}, |_, _| {}, |_, _| {},
                            );
                            let actual = inserter.try_destination(next.clone()).expect("Failed to add LLM edge on None-terminal branch");
                            let mut s = BTreeSet::new();
                            s.insert(actual);
                            out.push((dst_node_wrapper.clone(), (next_tokens, s)));
                        }
                    }
                }
                out
            },
            // merge sets
            |s1: &mut (LLMTokenBV, BTreeSet<IntermediatePrecomputeNode3Index>), s2: (LLMTokenBV, BTreeSet<IntermediatePrecomputeNode3Index>)| {
                s1.0 |= &s2.0;
                s1.1.extend(s2.1.into_iter());
            },
            // process: when we reach Trie1 end, attach nodes to a shared trie3_end node and continue
            |precomputed_node_data, (tokens, nodes_set)| {
                if tokens.is_empty() {
                    return false;
                }
                if precomputed_node_data.value.end {
                    for src in nodes_set.iter() {
                        let inserter = EdgeInserter::new(
                            &intermediate_trie3_god,
                            src.as_arc().clone(),
                            IntermediateTrie3EdgeKey::NoOp, (), |_, _| {}, |_, _| {}, |_, _| {},
                        );
                        inserter.try_destination(trie3_end.clone()).expect("Failed to insert end edge from nodes_set");
                    }
                }
                true
            },
        );

        // Ensure that the only node with end == true is trie3_end
        for node in Trie::all_nodes(&intermediate_trie3_god, &intermediate_precomputed3.values().cloned().collect::<Vec<_>>()) {
            let contents = &node.read(&intermediate_trie3_god).unwrap().value;
            if contents.end {
                assert_eq!(node, trie3_end);
            }
        }

        // --- New: Optimize intermediate trie before path processing ---
        crate::debug!(2, "Optimizing intermediate trie3...");
        let mut intermediate_roots: Vec<_> = intermediate_precomputed3.values().cloned().collect();
        let node_map = optimize_intermediate_trie3(
            &intermediate_roots,
            &intermediate_trie3_god,
            |_, node| node.value.end,
            &config.intermediate_trie3_main,
        );
        // Update the roots in the map after optimization
        intermediate_roots = intermediate_roots.into_iter().map(|x| node_map.get(&x).cloned().unwrap_or(x)).collect();
        for root in intermediate_precomputed3.values_mut() {
            if let Some(new_root) = node_map.get(root) {
                *root = new_root.clone();
            }
        }

        // println!("Intermediate trie3 before eliminating negative pops:");
        // let mut options = crate::datastructures::trie::PrettyPrintOptions::default()
        //     .display_edge_keys_only()
        //     .display_nodes()
        //     .omit_depth()
        //     ;
        // println!("{}", Trie::pretty_print_with_options(&intermediate_trie3_god, &intermediate_roots.iter().cloned().collect::<Vec<_>>(), &options));

        // --- New: Path extraction, elimination, and trie rebuilding ---
        crate::debug!(2, "Processing and rebuilding trie3 paths...");
        Self::_process_and_rebuild_trie3_paths(
            &mut intermediate_precomputed3,
            &intermediate_trie3_god,
        );
        intermediate_roots = intermediate_precomputed3.values().cloned().collect();

        // println!("Final intermediate trie3:");
        // let mut options = crate::datastructures::trie::PrettyPrintOptions::default()
        //     .display_edge_keys_only()
        //     .display_nodes()
        //     .omit_depth()
        //     ;
        // println!("{}", Trie::pretty_print_with_options(&intermediate_trie3_god, &intermediate_roots.iter().cloned().collect::<Vec<_>>(), &options));

        crate::debug!(2, "Optimizing intermediate trie3 again...");
        let mut intermediate_roots: Vec<_> = intermediate_precomputed3.values().cloned().collect();
        let node_map = optimize_intermediate_trie3(
            &intermediate_roots,
            &intermediate_trie3_god,
            |_, node| node.value.end,
            &config.intermediate_trie3_main,
        );
        // Update the roots in the map after optimization
        intermediate_roots = intermediate_roots.into_iter().map(|x| node_map.get(&x).cloned().unwrap_or(x)).collect();
        for root in intermediate_precomputed3.values_mut() {
            if let Some(new_root) = node_map.get(root) {
                *root = new_root.clone();
            }
        }

        // --- Convert intermediate trie to final Trie3 format ---
        crate::debug!(2, "Converting intermediate trie3 to final Trie3 format...");
        let (mut precomputed3, trie3_god) = Self::convert_intermediate_trie3_to_final(
            &intermediate_precomputed3,
            &intermediate_trie3_god,
            internal_max_llm_token,
        );

        // println!("Precompute3 trie before optimization:");
        // Self::_dump_precomputed3(
        //     &precomputed3,
        //     &stage_vocab.internal_to_original,
        //     &llm_vocab.as_ref().unwrap().llm_token_map,
        //     &trie3_god,
        // );

        crate::debug!(2, "Finished precomputing Trie 3.");
        let max_state_id = parser.unwrap().table.keys().map(|s| s.0).max().unwrap_or(0);
        optimize_trie3_size(&mut precomputed3, &trie3_god, &config.trie3, max_state_id, internal_max_llm_token, stage_vocab, parser.unwrap());
        (precomputed3, trie3_god)
    }

    fn convert_intermediate_trie3_to_final(
        intermediate_precomputed3: &BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
        intermediate_trie3_god: &IntermediateTrie3GodWrapper,
        internal_max_llm_token: usize,
    ) -> (Precomputed3, Trie3GodWrapper) {
        let trie3_god = Trie3GodWrapper::new();
        let mut precomputed3 = BTreeMap::new();
        let mut node_map: HashMap<IntermediatePrecomputeNode3Index, PrecomputeNode3Index> = HashMap::new();
        let mut q: VecDeque<IntermediatePrecomputeNode3Index> = VecDeque::new();

        let tokens_all = LLMTokenBV::ones(internal_max_llm_token + 1);
        let states_all = StateIDBV::max_ones();

        for (sid, old_root) in intermediate_precomputed3 {
            let new_root = *node_map.entry(*old_root).or_insert_with(|| {
                let new_node = PrecomputeNode3Index::new(trie3_god.insert(PrecomputeNode3::new(
                    PrecomputedNodeContents::root(internal_max_llm_token),
                )));
                q.push_back(*old_root);
                new_node
            });
            precomputed3.insert(*sid, new_root);
        }

        let mut visited = HashSet::new();
        while let Some(old_idx) = q.pop_front() {
            if !visited.insert(old_idx) { continue; }

            let new_idx = *node_map.get(&old_idx).unwrap();
            let old_guard = old_idx.read(intermediate_trie3_god).unwrap();

            for (edge_key, dest_map) in old_guard.children() {
                for (old_child_idx, _) in dest_map {
                    let new_child_idx = *node_map.entry(*old_child_idx).or_insert_with(|| {
                        let old_child_guard = old_child_idx.read(intermediate_trie3_god).unwrap();
                        let new_node_contents = if old_child_guard.value.end {
                            PrecomputedNodeContents::leaf()
                        } else {
                            PrecomputedNodeContents::internal()
                        };
                        let new_node = PrecomputeNode3Index::new(trie3_god.insert(PrecomputeNode3::new(new_node_contents)));
                        q.push_back(*old_child_idx);
                        new_node
                    });

                    let (final_key, final_value) = match edge_key {
                        IntermediateTrie3EdgeKey::Pop(n, states) => ((*n as isize, tokens_all.clone()), states.clone()),
                        IntermediateTrie3EdgeKey::Push(states) => ((0, tokens_all.clone()), states_all.clone()),
                        IntermediateTrie3EdgeKey::CheckLLM(tokens) => ((0, tokens.clone()), states_all.clone()),
                        IntermediateTrie3EdgeKey::NoOp => ((0, tokens_all.clone()), states_all.clone()),
                    };

                    trie3_god.insert_edge_simple(new_idx, new_child_idx, final_key, final_value);
                }
            }
        }

        (precomputed3, trie3_god)
    }

    pub fn precompute_special(
        &self,
    ) -> SpecialPrecomputation {
        crate::constraint_special_precompute::precompute_special(self)
    }

    pub fn dump_precomputed_special(&self) {
        crate::constraint_special_precompute::dump_precomputed_special(self);
    }

    pub fn all_internal_llm_tokens_bitset_precompute0(&self) -> LLMTokenBV {
        LLMTokenBV::ones(self.precompute0_vocab.internal_max_llm_token + 1)
    }

    pub fn all_internal_llm_tokens_bitset_precompute1(&self) -> LLMTokenBV {
        LLMTokenBV::ones(self.precompute_vocab1.internal_max_llm_token + 1)
    }

    pub fn all_internal_llm_tokens_bitset_precompute2(&self) -> LLMTokenBV {
        LLMTokenBV::ones(self.precompute2_vocab.internal_max_llm_token + 1)
    }

    pub fn all_internal_llm_tokens_bitset_precompute3(&self) -> LLMTokenBV {
        LLMTokenBV::ones(self.precompute3_vocab.internal_max_llm_token + 1)
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

    pub fn state_with_nodes(&self, nodes: Vec<(usize, Arc<GSSNode>)>) -> GrammarConstraintState<'_> {
        let mut state = BTreeMap::new();
        for (i, node) in nodes.into_iter() {
            state.insert(TokenizerStateID(i), self.parser.init_glr_parser(Some(self.llm_vocab.clone())));
        }
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

    // Stage-aware conversion (for Trie0)
    pub fn internal_bv_to_original_precompute0(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        self.internal_bv_to_original_with_map(internal_bv, &self.precompute0_vocab.internal_to_original, self.precompute0_vocab.internal_max_llm_token)
    }
    pub fn original_bv_to_internal_precompute0(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        self.original_bv_to_internal_with_map(original_bv, &self.precompute0_vocab.original_to_internal, self.precompute0_vocab.original_to_internal.len())
    }
    // Stage-aware conversion (for Trie1)
    pub fn internal_bv_to_original_precompute1(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        self.internal_bv_to_original_with_map(internal_bv, &self.precompute_vocab1.internal_to_original, self.precompute_vocab1.internal_max_llm_token)
    }
    pub fn original_bv_to_internal_precompute1(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        self.original_bv_to_internal_with_map(original_bv, &self.precompute_vocab1.original_to_internal, self.precompute_vocab1.original_to_internal.len())
    }
    // Stage-aware conversion (for Trie2)
    pub fn internal_bv_to_original_precompute2(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        self.internal_bv_to_original_with_map(internal_bv, &self.precompute2_vocab.internal_to_original, self.precompute2_vocab.internal_max_llm_token)
    }
    pub fn original_bv_to_internal_precompute2(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        self.original_bv_to_internal_with_map(original_bv, &self.precompute2_vocab.original_to_internal, self.precompute2_vocab.original_to_internal.len())
    }
    // Stage-aware conversion (for Trie3)
    pub fn internal_bv_to_original_precompute3(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
		self.internal_bv_to_original_with_map(internal_bv, &self.precompute3_vocab.internal_to_original, self.precompute3_vocab.internal_max_llm_token)
	}
    pub fn original_bv_to_internal_precompute3(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        self.original_bv_to_internal_with_map(original_bv, &self.precompute3_vocab.original_to_internal, self.precompute3_vocab.original_to_internal.len())
    }

    pub fn internal_to_original_precompute0(&self, original_id: LLMTokenID) -> Option<&HybridBitset> {
        self.precompute0_vocab.internal_to_original.get(&original_id.0)
    }
    pub fn original_to_internal_precompute0(&self, internal_id: LLMTokenID) -> Option<LLMTokenID> {
        self.precompute0_vocab.original_to_internal.get(&internal_id.0).map(|&orig_val| LLMTokenID(orig_val))
    }
    pub fn internal_to_original_precompute1(&self, original_id: LLMTokenID) -> Option<&HybridBitset> {
        self.precompute_vocab1.internal_to_original.get(&original_id.0)
    }
    pub fn original_to_internal_precompute1(&self, internal_id: LLMTokenID) -> Option<LLMTokenID> {
        self.precompute_vocab1.original_to_internal.get(&internal_id.0).map(|&orig_val| LLMTokenID(orig_val))
    }
    pub fn internal_to_original_precompute2(&self, original_id: LLMTokenID) -> Option<&HybridBitset> {
        self.precompute2_vocab.internal_to_original.get(&original_id.0)
    }
    pub fn original_to_internal_precompute2(&self, internal_id: LLMTokenID) -> Option<LLMTokenID> {
        self.precompute2_vocab.original_to_internal.get(&internal_id.0).map(|&orig_val| LLMTokenID(orig_val))
    }
    pub fn internal_to_original_precompute3(&self, original_id: LLMTokenID) -> Option<&HybridBitset> {
        self.precompute3_vocab.internal_to_original.get(&original_id.0)
    }
    pub fn original_to_internal_precompute3(&self, internal_id: LLMTokenID) -> Option<LLMTokenID> {
        self.precompute3_vocab.original_to_internal.get(&internal_id.0).map(|&orig_val| LLMTokenID(orig_val))
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

    fn original_bv_to_internal_with_map(
        &self,
        original_bv: &LLMTokenBV,
        original_to_internal: &BTreeMap<usize, usize>,
        _original_max_llm_token: usize,
    ) -> LLMTokenBV {
        let mut internal_bv = HybridBitset::zeros();
        if original_bv.is_all() {
            // Fast path for "all tokens"
            for &internal_id in original_to_internal.values() {
                internal_bv.insert(internal_id);
            }
        } else {
            for i in original_bv.iter() {
                if let Some(&internal_id) = original_to_internal.get(&i) {
                    internal_bv.insert(internal_id);
                }
            }
        }
        internal_bv
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
                let applicable_tokens_rangeset = child_vocab_node_ref.reachable_token_ids();
                result_map.entry(grammar_token_id).or_insert_with(LLMTokenBV::zeros)
                    .extend(applicable_tokens_rangeset.iter());
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
}

pub(crate) struct Precomputer0<'r> {
    pub(crate) tokenizer:        &'r Regex,
    pub(crate) parser:           Option<&'r GLRParser>,
    pub(crate) llm_vocab:        Option<Arc<LLMVocab>>,
    pub(crate) vocab:            VocabPrefixTree,
    pub(crate) roots:            BTreeMap<TokenizerStateID, PrecomputeNode0Index>,
    pub(crate) possible_matches: RefCell<BTreeMap<*const VocabPrefixTreeNode, BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>>>,
    pub(crate) all_llm_tokens:   HybridBitset,
    pub(crate) merge_threshold:  usize,
    pub(crate) pb:               ProgressBar,
    pub(crate) stats:            PrecomputeStats,
    pub(crate) terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    pub(crate) ignore_terminal_id: Option<TerminalID>,
    pub(crate) token_name_map:   &'r BiBTreeMap<Terminal, usize>,
    // One end node per final tokenizer state.
    pub(crate) end_nodes:        BTreeMap<TokenizerStateID, PrecomputeNode0Index>,
    pub(crate) trie0_god:        Trie0GodWrapper,
}

pub(crate) struct Precomputer1<'r> {
    pub(crate) tokenizer:        &'r Regex,
    pub(crate) parser:           Option<&'r GLRParser>,
    pub(crate) llm_vocab:        Option<Arc<LLMVocab>>,
    pub(crate) vocab:            VocabPrefixTree, // This will be moved out during dfs
    pub(crate) roots:            BTreeMap<TokenizerStateID, TempPrecomputeNode1Index>,
    pub(crate) possible_matches: RefCell<BTreeMap<*const VocabPrefixTreeNode, BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>>>,
    pub(crate) all_llm_tokens:   RangeSetBlaze<usize>,
    pub(crate) merge_threshold:  usize,
    pub(crate) pb:               ProgressBar,
    pub(crate) stats:            PrecomputeStats,
    pub(crate) terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    pub(crate) ignore_terminal_id: Option<TerminalID>,
    pub(crate) token_name_map:   &'r BiBTreeMap<Terminal, usize>,
    pub(crate) leaf_node:        TempPrecomputeNode1Index,
    pub(crate) dfs_stats:        DfsStats,
    pub(crate) trie1_god:        TempTrie1GodWrapper,
    pub(crate) original_to_dummy_map: BTreeMap<TerminalID, TerminalID>,
}

impl<'r> Precomputer0<'r> {}
impl<'r> Precomputer1<'r> {
    fn new(
        tokenizer:        &'r Regex,
        parser:           Option<&'r GLRParser>,
        llm_vocab:        Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        internal_max_llm_token: usize,
        merge_threshold:  usize,
        terminal_follow_map: &'r BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        ignore_terminal_id: Option<TerminalID>,
        token_name_map: &'r BiBTreeMap<Terminal, usize>,
        original_to_dummy_map: BTreeMap<TerminalID, TerminalID>,
    ) -> Self {
        let tokens: Vec<(usize, Vec<u8>)> = internal_llm_token_map
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone()))
            .collect();

        crate::debug!(2, "Building vocab prefix tree");
        let vocab = VocabPrefixTree::build(&tokens);
        crate::debug!(2, "Done building vocab prefix tree");

        let mut roots = BTreeMap::new();
        let trie1_god = TempTrie1GodWrapper::new();
        for sid in tokenizer.iter_states() {
            roots.insert(
                sid,
                TempPrecomputeNode1Index::new(trie1_god.insert(TempPrecomputeNode1::new(TempPrecomputedNodeContents::root(internal_max_llm_token)))),
            );
        }
        crate::debug!(2, "Created trie1 roots for {} tokenizer states", tokenizer.iter_states().count());

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

        let leaf_node = TempPrecomputeNode1Index::new(trie1_god.insert(TempPrecomputeNode1::new(TempPrecomputedNodeContents::leaf())));
        crate::debug!(2, "Created trie1 leaf node");

        Self {
            tokenizer,
            parser,
            llm_vocab,
            vocab,
            roots,
            possible_matches: RefCell::new(BTreeMap::new()), // This is for grammar tokens, not LLM tokens
            all_llm_tokens: RangeSetBlaze::from_iter(0..=internal_max_llm_token),
            merge_threshold,
            pb,
            stats: PrecomputeStats::default(),
            terminal_follow_map,
            ignore_terminal_id,
            token_name_map,
            leaf_node,
            dfs_stats: DfsStats::default(),
            trie1_god,
            original_to_dummy_map,
        }
    }

    fn get_leaf_node(&self) -> TempPrecomputeNode1Index {
        self.leaf_node.clone()
    }

    fn finish(self) -> (BTreeMap<TokenizerStateID, PrecomputeNode1Index>, Trie1GodWrapper) {
        let final_trie1_god = Trie1GodWrapper::new();
        let mut final_roots = BTreeMap::new();
        let mut node_map: HashMap<TempPrecomputeNode1Index, PrecomputeNode1Index> = HashMap::new();

        for (sid, temp_root) in &self.roots {
            let final_root = self.convert_trie1_recursive(
                *temp_root,
                &self.trie1_god,
                &final_trie1_god,
                &mut node_map,
            );
            final_roots.insert(*sid, final_root);
        }

        (final_roots, final_trie1_god)
    }

    fn convert_trie1_recursive(
        &self,
        temp_idx: TempPrecomputeNode1Index,
        temp_god: &TempTrie1GodWrapper,
        final_god: &Trie1GodWrapper,
        node_map: &mut HashMap<TempPrecomputeNode1Index, PrecomputeNode1Index>,
    ) -> PrecomputeNode1Index {
        if let Some(final_idx) = node_map.get(&temp_idx) {
            return *final_idx;
        }

        let temp_guard = temp_idx.read(temp_god).unwrap();
        let final_node_contents = PrecomputedNodeContents {
            end: temp_guard.value.end,
            live_tokens: HybridBitset { inner: crate::datastructures::cache::intern_l1(temp_guard.value.live_tokens.clone()) },
        };
        let new_node = PrecomputeNode1::new(final_node_contents);
        let final_idx = PrecomputeNode1Index::new(final_god.insert(new_node));
        node_map.insert(temp_idx, final_idx);

        let children_to_copy = temp_guard.children().clone();
        drop(temp_guard);

        if self.original_to_dummy_map.is_empty() {
            for (ek, dest_map) in children_to_copy {
                for (temp_child_idx, rs_blaze) in dest_map {
                    let final_child_idx = self.convert_trie1_recursive(temp_child_idx, temp_god, final_god, node_map);
                    let hybrid_bitset = HybridBitset { inner: crate::datastructures::cache::intern_l1(rs_blaze) };
                    final_god.insert_edge_simple(final_idx, final_child_idx, ek.clone(), hybrid_bitset);
                }
            }
        } else {
            let mut direct_edges = Vec::new();
            // Group injected edges by their dummy terminal ID.
            let mut injected_edges_by_dummy: BTreeMap<TerminalID, Vec<(Option<TerminalID>, OrderedHashMap<TempPrecomputeNode1Index, RangeSetBlaze<usize>>)>> = BTreeMap::new();

            for (ek, dest_map) in children_to_copy {
                if let Some(tid) = ek {
                    if let Some(dummy_tid) = self.original_to_dummy_map.get(&tid) {
                        injected_edges_by_dummy.entry(*dummy_tid).or_default().push((Some(tid), dest_map));
                        continue;
                    }
                }
                direct_edges.push((ek, dest_map));
            }

            for (ek, dest_map) in direct_edges {
                for (temp_child_idx, rs_blaze) in dest_map {
                    let final_child_idx = self.convert_trie1_recursive(temp_child_idx, temp_god, final_god, node_map);
                    let hybrid_bitset = HybridBitset { inner: crate::datastructures::cache::intern_l1(rs_blaze) };
                    final_god.insert_edge_simple(final_idx, final_child_idx, ek.clone(), hybrid_bitset);
                }
            }

            for (dummy_tid, edges) in injected_edges_by_dummy {
                let inter_node = PrecomputeNode1::new(PrecomputedNodeContents::internal());
                let inter_idx = PrecomputeNode1Index::new(final_god.insert(inter_node));
                let mut total_inter_bitset = HybridBitset::zeros();

                for (original_ek, dest_map) in edges {
                    for (temp_child_idx, rs_blaze) in dest_map {
                        let final_child_idx = self.convert_trie1_recursive(temp_child_idx, temp_god, final_god, node_map);
                        let hybrid_bitset = HybridBitset { inner: crate::datastructures::cache::intern_l1(rs_blaze) };
                        total_inter_bitset |= &hybrid_bitset;
                        final_god.insert_edge_simple(inter_idx, final_child_idx, original_ek, hybrid_bitset);
                    }
                }
                final_god.insert_edge_simple(final_idx, inter_idx, Some(dummy_tid), total_inter_bitset);
            }
        }

        final_idx
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
                *result_map.entry(grammar_token_id).or_insert_with(LLMTokenBV::zeros) |= HybridBitset::from(applicable_tokens);
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
            HashMap<TempPrecomputeNode1Index, RangeSetBlaze<usize>>,
        > = BTreeMap::new();

        for (sid, arc) in &self.roots {
            assoc
                .entry(*sid)
                .or_default()
                .insert(arc.clone(), self.all_llm_tokens.clone());
        }

        crate::debug!(2, "Starting precompute DFS");
        crate::debug!(6, "Roots for each tokenizer state:");
        for (sid, root) in &self.roots {
            crate::debug!(6, "  {}: {}", sid.0, root);
        }
        crate::profiler::reset();
        // Temporarily move vocab out of self to satisfy the borrow checker.
        let vocab = std::mem::replace(&mut self.vocab, VocabPrefixTree::new());
        self.dfs(&vocab.root, assoc);
        self.vocab = vocab; // Move it back.
        self.dfs_stats.print();
        crate::debug!(2, "Finished precompute DFS");
        self.pb.finish();
        crate::profiler::print_summary();
        crate::debug!(2, "Precomputation complete");
    }

    fn dfs(
        &mut self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<TokenizerStateID, HashMap<TempPrecomputeNode1Index, RangeSetBlaze<usize>>>,
    ) {
        self.pb.inc(1);
        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let mut work_queue: BTreeMap<
                usize,
                BTreeMap<TokenizerStateID, HashMap<TempPrecomputeNode1Index, RangeSetBlaze<usize>>>,
            > = BTreeMap::new();
            work_queue.insert(0, assoc_by_state.clone());

            let mut next_level_assoc: BTreeMap<_, HashMap<_, _>> = BTreeMap::new();

            // === OPTIMIZATION 1: Cache node data to avoid repeated lock acquisitions ===
            let mut node_cache: HashMap<TempPrecomputeNode1Index, (RangeSetBlaze<usize>, bool)> = HashMap::new();
            let get_node_data = |cache: &mut HashMap<_, _>, idx: TempPrecomputeNode1Index, god: &TempTrie1GodWrapper| {
                cache.entry(idx).or_insert_with(|| {
                    let guard = idx.read(god).unwrap();
                    (guard.value.live_tokens.clone(), guard.value.end)
                }).clone()
            };

            // === OPTIMIZATION 2: Batch all edge insertions and updates ===
            let mut pending_edges: Vec<(TempPrecomputeNode1Index, TempPrecomputeNode1Index, Option<GrammarTokenID>, RangeSetBlaze<usize>)> = Vec::new();
            let mut pending_live_token_updates: HashMap<TempPrecomputeNode1Index, RangeSetBlaze<usize>> = HashMap::new();

            // === OPTIMIZATION 3: Pre-compute child_vocab reachable tokens (used frequently) ===
            let child_reachable = child_vocab_node.reachable_token_ids();
            let child_token_id = child_vocab_node.token_id();

            // === OPTIMIZATION 4: Pre-compute possible_matches_at_end for all states we might need ===
            let mut possible_matches_cache: HashMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>> = HashMap::new();

            while let Some((pos, states_at_pos)) = work_queue.pop_first() {
                if pos == segment_bytes.len() {
                    for (tokenizer_state_id, nodes_with_tokens) in states_at_pos {
                        let entry = next_level_assoc.entry(tokenizer_state_id).or_default();
                        for (node, tokens) in nodes_with_tokens {
                            entry
                                .entry(node)
                                .or_insert_with(RangeSetBlaze::new)
                                .bitor_assign(&tokens);
                        }
                    }
                    continue;
                }

                for (tokenizer_state_id, precompute_nodes_with_tokens) in states_at_pos {
                    let exec_result = self.tokenizer.execute_from_state(&segment_bytes[pos..], tokenizer_state_id);

                    let possible_matches_at_end = if let Some(end_state_val) = exec_result.end_state {
                        let ts = TokenizerStateID(end_state_val);
                        possible_matches_cache.entry(ts).or_insert_with(|| {
                            self.possible_matches(child_vocab_node, ts)
                        })
                    } else {
                        &BTreeMap::new()
                    };

                    for match_info in &exec_result.matches {
                        let terminal_id = GrammarTokenID(match_info.id);
                        let next_pos = pos + match_info.width;

                        for (src_node_wrapper, src_contextual_tokens) in &precompute_nodes_with_tokens {
                            let src_node_idx = *src_node_wrapper;

                            // Use cache instead of repeated reads
                            let (src_live_tokens, _) = get_node_data(&mut node_cache, src_node_idx, &self.trie1_god);

                            // Handle exact end-of-segment match
                            if next_pos == segment_bytes.len() {
                                let mut edge_bv = RangeSetBlaze::new();
                                edge_bv.insert(child_token_id);
                                let final_edge_bv = &(&edge_bv & src_contextual_tokens) & &src_live_tokens;

                                if !final_edge_bv.is_empty() {
                                    let end_idx = self.get_leaf_node();
                                    pending_edges.push((src_node_idx, end_idx, Some(terminal_id), final_edge_bv.clone()));
                                    pending_live_token_updates.entry(end_idx)
                                        .or_insert_with(RangeSetBlaze::new)
                                        .bitor_assign(&final_edge_bv);
                                }
                            }

                            // Compute edge_bv once
                            let mut edge_bv = child_reachable.clone();
                            if next_pos == segment_bytes.len() {
                                edge_bv.remove(child_token_id);
                            }
                            if let Some(matches_for_terminal) = possible_matches_at_end.get(&terminal_id) {
                                edge_bv = &edge_bv - matches_for_terminal.inner.as_ref();
                            }

                            let edge_bv_for_inserter = &(&edge_bv & src_contextual_tokens) & &src_live_tokens;
                            if edge_bv_for_inserter.is_empty() { continue; }

                            let next_tokenizer_state = self.tokenizer.initial_state_id();
                            let dest_nodes_in_queue = work_queue.entry(next_pos)
                                .or_default()
                                .entry(next_tokenizer_state)
                                .or_default();

                            // Find or create destination node
                            let mut dest_node_opt = dest_nodes_in_queue.iter()
                                .filter_map(|(dest_node, dest_contextual_tokens)| {
                                    let (dest_live_tokens, is_end) = get_node_data(&mut node_cache, *dest_node, &self.trie1_god);
                                    if is_end { return None; }

                                    let risky_tokens = &edge_bv_for_inserter - dest_contextual_tokens;
                                    if risky_tokens.is_empty() || (&risky_tokens & &dest_live_tokens).is_empty() {
                                        Some(*dest_node)
                                    } else {
                                        None
                                    }
                                }).next();

                            if dest_node_opt.is_none() {
                                // Check existing children - read once
                                let children_of_src: Vec<TempPrecomputeNode1Index> = {
                                    let guard = src_node_idx.read(&self.trie1_god).unwrap();
                                    guard.children().values().flat_map(|m| m.keys().cloned()).collect()
                                };

                                dest_node_opt = children_of_src.iter()
                                    .filter(|child_arc| {
                                        let (child_live_tokens, is_end) = get_node_data(&mut node_cache, **child_arc, &self.trie1_god);
                                        !is_end && (&child_live_tokens & &edge_bv_for_inserter).is_empty()
                                    }).copied().next();
                            }

                            let result_node = dest_node_opt.unwrap_or_else(|| {
                                 let new_node = TempPrecomputeNode1::new(TempPrecomputedNodeContents::internal());
                                 let idx = TempPrecomputeNode1Index::new(self.trie1_god.insert(new_node));
                                 node_cache.insert(idx, (RangeSetBlaze::new(), false));
                                 idx
                            });

                            pending_edges.push((src_node_idx, result_node, Some(terminal_id), edge_bv_for_inserter.clone()));
                            pending_live_token_updates.entry(result_node)
                                .or_insert_with(RangeSetBlaze::new)
                                .bitor_assign(&edge_bv_for_inserter);

                            // Update cache
                            node_cache.entry(result_node)
                                .and_modify(|(live, _)| *live |= &edge_bv_for_inserter);

                            dest_nodes_in_queue.entry(result_node)
                                .or_insert_with(RangeSetBlaze::new)
                                .bitor_assign(&edge_bv_for_inserter);
                        }
                    }

                    // Handle continuation state
                    if let Some(end_state_val) = exec_result.end_state {
                        let final_tokenizer_state = TokenizerStateID(end_state_val);
                        let accessible_terminals = self.tokenizer.tokens_accessible_from_state(final_tokenizer_state);

                        for (src_node_wrapper, src_contextual_tokens) in &precompute_nodes_with_tokens {
                            let mut edge_bv = RangeSetBlaze::new();
                            edge_bv.insert(child_token_id);
                            let edge_bv_for_inserter = &edge_bv & src_contextual_tokens;
                            if edge_bv_for_inserter.is_empty() { continue; }

                            let src_node_idx = *src_node_wrapper;
                            let (src_live_tokens, _) = get_node_data(&mut node_cache, src_node_idx, &self.trie1_god);
                            let final_edge_bv = &edge_bv_for_inserter & &src_live_tokens;

                            if !final_edge_bv.is_empty() {
                                let end_idx = self.get_leaf_node();
                                for terminal_id in &accessible_terminals {
                                    pending_edges.push((src_node_idx, end_idx, Some(*terminal_id), final_edge_bv.clone()));
                                    pending_live_token_updates.entry(end_idx)
                                        .or_insert_with(RangeSetBlaze::new)
                                        .bitor_assign(&final_edge_bv);
                                }
                            }
                        }

                        let entry = next_level_assoc.entry(final_tokenizer_state).or_default();
                        for (node, tokens) in precompute_nodes_with_tokens {
                            entry.entry(node).or_default().bitor_assign(&tokens);
                        }
                    }
                }
            }

            // === OPTIMIZATION 5: Batch write all edges and updates ===
            for (src, dst, key, bv) in pending_edges {
                self.trie1_god.insert_edge_simple(src, dst, key, bv);
            }

            for (node_idx, live_tokens) in pending_live_token_updates {
                if let Some(mut guard) = node_idx.write(&self.trie1_god) {
                    guard.value.live_tokens |= &live_tokens;
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

#[derive(Default, Clone)]
struct TokenAcc {
    small: Vec<usize>,
    big: Vec<RangeSetBlaze<usize>>,
}

impl TokenAcc {
    fn add_token(&mut self, token: usize) {
        self.small.push(token);
    }

    fn add_bitset(&mut self, bitset: RangeSetBlaze<usize>) {
        if bitset.is_empty() {
            return;
        }
        if bitset.len() == 1 {
            self.small.push(bitset.iter().next().unwrap());
        } else {
            self.big.push(bitset);
        }
    }

    fn len(&self) -> usize {
        self.small.len() + self.big.len()
    }
}

pub type Trie0GodWrapper = GodWrapper<Option<(TerminalID, Option<TokenizerStateID>)>, HybridBitset, PrecomputedNodeContents0>;
pub type Trie0God = God<Option<(TerminalID, Option<TokenizerStateID>)>, HybridBitset, PrecomputedNodeContents>;
pub type Trie1GodWrapper = GodWrapper<Option<TerminalID>, HybridBitset, PrecomputedNodeContents>;
pub type Trie1God = God<Option<TerminalID>, HybridBitset, PrecomputedNodeContents>;
pub type Trie2GodWrapper = GodWrapper<(usize, Option<StateID>), HybridBitset, PrecomputedNodeContents>;
pub type Trie2God = God<(usize, Option<StateID>), HybridBitset, PrecomputedNodeContents>;
pub type Trie3GodWrapper = GodWrapper<(isize, LLMTokenBV), StateIDBV, PrecomputedNodeContents>;
pub type Trie3God = God<(isize, LLMTokenBV), StateIDBV, PrecomputedNodeContents>;

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
            let (gss_str, _) = print_gss_forest(&gss_roots, &self.parent.parser.terminal_map, &config);
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
        self.get_mask3()
        // self.get_mask4()
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

    pub fn num_unique_nodes(&self) -> usize {
        gather_gss_stats(
            &self.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
        ).unique_nodes()
    }

    pub fn get_mask3(&self) -> LLMTokenBV {
        let final_mask_internal = RefCell::new(HybridBitset::zeros());
        if self.state.is_empty() {
            return self.parent.internal_bv_to_original_precompute3(&final_mask_internal.into_inner());
        }
        let traversal_data = match &self.parent.trie3_traversal_data {
            Some(data) => data,
            None => {
                panic!("No traversal data for get_mask3, returning empty mask.");
            }
        };

        let mut initial_values_by_trie_node: BTreeMap<PrecomputeNode3Index, GLRParserState<'a>> = BTreeMap::new();
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
             return self.parent.internal_bv_to_original(&final_mask_internal.into_inner());
        }

        Trie::special_map_grouped(
            &self.parent.trie3_god,
            traversal_data,
            initial_values_for_map,
            // step_fn: (current_state, (pop, llm_token_bv), destinations_map)
            |glr_s, (pop, llm_token_bv_from_edge), dest_map| {
                let popped = glr_s.active_state.stack.popn(*pop as usize);
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
                        results.push((dest_idx.clone(), new_glr_s));
                    }
                }
                results
            },
            // merge_fn
            |glr_s1, glr_s2| glr_s1.merge_with(glr_s2),
            // process_fn: (precomputed_node_data, final_state_for_this_path)
            |precomputed_node_data, glr_s| {
                let mut glr_s_copy = glr_s.clone();
                let glr_active_tokens = glr_s_copy.active_state.stack.allowed_llm_tokens();
                let keep_going = glr_s_copy.is_ok();
                if precomputed_node_data.value.end {
                    if !glr_active_tokens.is_empty() {
                        *final_mask_internal.borrow_mut() |= &glr_active_tokens;
                    }
                }
                keep_going
            },
        );

        let final_mask_mapped = self.parent.internal_bv_to_original_precompute3(&final_mask_internal.into_inner());

        final_mask_mapped
    }

    pub fn get_mask4(&self) -> LLMTokenBV {
        crate::constraint_special_precompute::get_mask4(self)
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) { // original ID
        return self.commit_bytes(&self.parent.llm_vocab.llm_token_map.get_by_right(&llm_token_id).unwrap().clone());
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

pub(crate) type LLMTokenBV = HybridBitset;
pub(crate) type TerminalBV = HybridBitset;
/// A 2D bitset where L1 is tokenizer state and L2 is terminal ID.
pub type TerminalInfo = HybridL2Bitset;
