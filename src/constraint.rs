#![allow(clippy::too_many_arguments)]

use std::{
    borrow::Borrow,
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashMap, BinaryHeap},
    fmt::{self, Debug, Display, Formatter},
    iter::FromIterator,
    sync::Arc,
};
use std::cmp::Reverse;
 
use bimap::BiBTreeMap;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use ordered_hash_map::OrderedHashMap;
use range_set_blaze::RangeSetBlaze;

use crate::{
    constraint_extra::PrecomputeStats,
    constraint_precompute1_utils::{self, Trie1Config},
    datastructures::{
        hybrid_bitset::HybridBitset,
        hybrid_l2_bitset::HybridL2Bitset,
        leveled_gss::{LeveledGSS, Merge},
        trie::{EdgeInserter, Trie, Trie2Index},
        trie::{God, GodWrapper},
        vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode},
    },
    equivalence_analysis_finite_automata,
    finite_automata::Regex,
    glr::{
        analyze::compute_terminal_follow_sets,
        grammar::Terminal,
        parser::{GLRParser, GLRParserState},
    },
    interface::{CompiledGrammar, GrammarDefinition},
    json_serialization::{JSONConvertible, JSONNode},
    precompute4::full_dwa::{precompute4, Precomputed4},
    precompute4::weighted_automata::common::{StateID as WAStateID},
    precompute4::weighted_automata::{NWA, NWAStateID, Weight},
    profiler::{self, PROGRESS_BAR_ENABLED},
    r#macro::is_debug_level_enabled,
    tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID},
    types::{TerminalID as GrammarTokenID, TerminalID},
};
use profiler_macro::{time_it, timeit};
use std::collections::BTreeMap as StdMap;
use std::ops::BitOrAssign;
use rayon::prelude::*;
use im::HashSet;
use crate::datastructures::bitset::Bitset;
use crate::datastructures::gss_acc::Acc;
use crate::glr::parser::{ParseState, ParseStateEdgeContent};
use crate::glr::table::StateID;
use crate::precompute4::weighted_automata::bitset::SimpleBitset;

// ---------------------------------------------------------------------------
// Basic aliases
// ---------------------------------------------------------------------------

pub type LLMTokenBV = HybridBitset;
pub type TerminalBV = HybridBitset;
pub type StateIDBV = HybridBitset;
/// A 2D bitset where L1 is tokenizer state and L2 is terminal ID.
pub type TerminalInfo = HybridL2Bitset;

type GSSNode = LeveledGSS<ParseStateEdgeContent, Acc>;

// ---------------------------------------------------------------------------
// Terminal allowance mode
// ---------------------------------------------------------------------------

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
            other => Err(format!(
                "Expected JSON string for TerminalAllowanceCheckMode, got {:?}",
                other
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Vocab structures
// ---------------------------------------------------------------------------

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
    pub max_original_llm_token_id: usize,
    pub internal_to_original_sparse_matrix: Vec<Vec<(u16, u64)>>,
}

impl JSONConvertible for LLMVocab {
    fn to_json(&self) -> JSONNode {
        let mut m = StdMap::new();
        m.insert(
            "llm_token_map".to_string(),
            self.llm_token_map.to_json(),
        );
        m.insert(
            "max_original_llm_token_id".to_string(),
            self.max_original_llm_token_id.to_json(),
        );
        JSONNode::Object(m)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let llm_token_map = obj
                    .remove("llm_token_map")
                    .ok_or("LLMVocab: missing llm_token_map".to_string())
                    .and_then(|n| BiBTreeMap::<Vec<u8>, LLMTokenID>::from_json(n))?;
                let max_original_llm_token_id = obj
                    .remove("max_original_llm_token_id")
                    .ok_or("LLMVocab: missing max_original_llm_token_id".to_string())
                    .and_then(usize::from_json)?;
                Ok(LLMVocab {
                    llm_token_map,
                    max_original_llm_token_id,
                })
            }
            _ => Err("LLMVocab: expected object".to_string()),
        }
    }
}

impl JSONConvertible for StageVocab {
    fn to_json(&self) -> JSONNode {
        let mut m = StdMap::new();
        m.insert(
            "original_to_internal".to_string(),
            self.original_to_internal.to_json(),
        );
        let mut ito: Vec<(usize, Vec<usize>)> = Vec::new();
        for (k, bv) in &self.internal_to_original {
            ito.push((*k, bv.iter_up_to(self.max_original_llm_token_id).collect::<Vec<_>>()));
        }
        m.insert("internal_to_original".to_string(), ito.to_json());
        m.insert(
            "internal_max_llm_token".to_string(),
            self.internal_max_llm_token.to_json(),
        );
        m.insert(
            "max_original_llm_token_id".to_string(),
            self.max_original_llm_token_id.to_json(),
        );
        m.insert(
            "internal_to_original_sparse_matrix".to_string(),
            self.internal_to_original_sparse_matrix.to_json(),
        );
        JSONNode::Object(m)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let original_to_internal = obj
                    .remove("original_to_internal")
                    .ok_or("StageVocab: missing original_to_internal".to_string())
                    .and_then(BTreeMap::<usize, usize>::from_json)?;
                let internal_max_llm_token = obj
                    .remove("internal_max_llm_token")
                    .ok_or("StageVocab: missing internal_max_llm_token".to_string())
                    .and_then(usize::from_json)?;
                let max_original_llm_token_id = obj
                    .remove("max_original_llm_token_id")
                    .ok_or("StageVocab: missing max_original_llm_token_id".to_string())
                    .and_then(usize::from_json)?;
                let ito_vec: Vec<(usize, Vec<usize>)> = obj
                    .remove("internal_to_original")
                    .ok_or("StageVocab: missing internal_to_original".to_string())
                    .and_then(Vec::from_json)?;
                let internal_to_original: BTreeMap<usize, LLMTokenBV> = ito_vec
                    .into_iter()
                    .map(|(k, v)| (k, v.into_iter().collect()))
                    .collect();
                let internal_to_original_sparse_matrix =
                    match obj.remove("internal_to_original_sparse_matrix") {
                        Some(n) => Vec::<Vec<(u16, u64)>>::from_json(n)?,
                        None => {
                            // For backward compatibility, compute it if missing.
                            Self::build_internal_to_original_sparse_matrix(
                                &internal_to_original,
                                max_original_llm_token_id,
                                internal_max_llm_token,
                            )
                        }
                    };

                Ok(StageVocab {
                    original_to_internal,
                    internal_to_original,
                    internal_max_llm_token,
                    max_original_llm_token_id,
                    internal_to_original_sparse_matrix,
                })
            }
            _ => Err("StageVocab: expected object".to_string()),
        }
    }
}

impl StageVocab {
    pub(crate) fn build_internal_to_original_sparse_matrix(
        internal_to_original: &BTreeMap<usize, LLMTokenBV>,
        max_original_llm_token_id: usize,
        internal_max_llm_token: usize,
    ) -> Vec<Vec<(u16, u64)>> {
        type Word = u64;
        const WORD_BITS: usize = 64;

        let num_internal_tokens = internal_max_llm_token + 1;
        let mut sparse_matrix: Vec<Vec<(u16, Word)>> = vec![Vec::new(); num_internal_tokens];

        for (internal_id, original_bv) in internal_to_original.iter() {
            if *internal_id >= num_internal_tokens {
                continue;
            }

            let mut temp_row = BTreeMap::<u16, Word>::new();
            for original_id in original_bv.iter_up_to(max_original_llm_token_id) {
                if original_id > max_original_llm_token_id {
                    continue;
                }
                let word_idx = (original_id / WORD_BITS) as u16;
                let bit_idx = original_id % WORD_BITS;
                *temp_row.entry(word_idx).or_insert(0) |= 1 << bit_idx;
            }
            if !temp_row.is_empty() {
                sparse_matrix[*internal_id] = temp_row.into_iter().collect();
            }
        }
        sparse_matrix
    }

    /// Convert an internal BV (using `self.vocab`) back to original IDs.
    pub fn internal_bv_to_original(&self, internal_bv: &LLMTokenBV) -> Bitset {
        let mut internal_bv = internal_bv.clone();
        if internal_bv.is_all() {
            internal_bv = HybridBitset::ones(self.internal_max_llm_token + 1);
        }

        type Word = u64;
        const WORD_BITS: usize = 64;

        let max_original_id = self.max_original_llm_token_id;
        let original_vocab_size_words = (max_original_id / WORD_BITS) + 1;
        let num_internal_tokens = self.internal_max_llm_token + 1;

        let mut result_bitset_words = vec![0 as Word; original_vocab_size_words];
        for internal_id in internal_bv.iter_up_to(self.internal_max_llm_token) {
            if internal_id >= num_internal_tokens {
                continue;
            }
            // It's possible for an internal ID to exist in the bitvector but not have a
            // corresponding entry in the sparse matrix if it corresponds to no original tokens.
            if let Some(sparse_row) = self.internal_to_original_sparse_matrix.get(internal_id) {
                for &(word_idx, word) in sparse_row {
                    result_bitset_words[word_idx as usize] |= word;
                }
            }
        }

        Bitset::from_words_vec(result_bitset_words)
    }

    pub fn original_bv_to_internal(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        GrammarConstraint::original_bv_to_internal_with_map(
            original_bv,
            &self.original_to_internal,
            self.max_original_llm_token_id,
        )
    }
}

// ---------------------------------------------------------------------------
// Deduplicating map for large values
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupValueMap<K, V>
where
    K: Ord + Clone + Eq,
    V: Clone + Eq + std::hash::Hash,
{
    key_to_id: BTreeMap<K, usize>,
    id_to_value: BTreeMap<usize, V>,
    value_to_id: HashMap<V, usize>,
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
            value_to_id: HashMap::new(),
            next_id: 0,
        }
    }
}

impl<K, V> DedupValueMap<K, V>
where
    K: Ord + Clone + Eq,
    V: Clone + Eq + std::hash::Hash,
{
    pub fn new() -> Self { Self::default() }

    fn intern_value(&mut self, v: V) -> usize {
        if let Some(&id) = self.value_to_id.get(&v) { return id; }
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).expect("DedupValueMap ID overflow");
        self.id_to_value.insert(id, v.clone());
        self.value_to_id.insert(v, id);
        id
    }

    pub fn len(&self) -> usize { self.key_to_id.len() }
    pub fn is_empty(&self) -> bool { self.key_to_id.is_empty() }

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
        self.key_to_id
            .iter()
            .map(|(k, id)| (k, self.id_to_value.get(id).expect("dangling id")))
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
        let mut values_arr = Vec::new();
        for (id, v) in &self.id_to_value {
            values_arr.push(JSONNode::Array(vec![id.to_json(), v.to_json()]));
        }
        obj.insert("values".to_string(), JSONNode::Array(values_arr));
        let mut keys_arr = Vec::new();
        for (k, id) in &self.key_to_id {
            keys_arr.push(JSONNode::Array(vec![k.to_json(), id.to_json()]));
        }
        obj.insert("keys".to_string(), JSONNode::Array(keys_arr));
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        let next_id =
            usize::from_json(obj.remove("next_id").ok_or("DedupValueMap: missing 'next_id'")?)?;
        let values_arr = obj
            .remove("values")
            .ok_or("DedupValueMap: missing 'values'")?;
        let keys_arr = obj.remove("keys").ok_or("DedupValueMap: missing 'keys'")?;

        let mut id_to_value = BTreeMap::new();
        let mut value_to_id = HashMap::new();
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

        Ok(Self {
            key_to_id,
            id_to_value,
            value_to_id,
            next_id,
        })
    }
}

// ---------------------------------------------------------------------------
// Trie node contents
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedNodeContents {
    pub(crate) end: bool,
    pub(crate) live_tokens: LLMTokenBV,
}

impl PrecomputedNodeContents {
    pub(crate) fn root(internal_max_llm_token_id: usize) -> Self {
        Self {
            end: false,
            live_tokens: LLMTokenBV::ones(internal_max_llm_token_id + 1),
        }
    }

    pub(crate) fn internal() -> Self {
        timeit!("PrecomputedNodeContents::internal", {});
        Self {
            end: false,
            live_tokens: LLMTokenBV::zeros(),
        }
    }

    pub(crate) fn leaf() -> Self {
        Self {
            end: true,
            live_tokens: LLMTokenBV::zeros(),
        }
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
                let end = obj
                    .remove("clean_end")
                    .ok_or_else(|| "Missing field clean_end for PrecomputedNodeContents".to_string())
                    .and_then(bool::from_json)?;
                let live_tokens = obj
                    .remove("live_tokens")
                    .ok_or_else(|| "Missing field live_tokens for PrecomputedNodeContents".to_string())
                    .and_then(LLMTokenBV::from_json)?;
                Ok(PrecomputedNodeContents { end, live_tokens })
            }
            _ => Err("Expected JSONNode::Object for PrecomputedNodeContents".to_string()),
        }
    }
}

// Minimal node contents for the (now empty) Trie0 type, kept only because
// optimize_trie1_size still takes a Trie0GodWrapper in its signature.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedNodeContents0 {
    pub(crate) live_tokens: LLMTokenBV,
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
        Self {
            live_tokens: LLMTokenBV::zeros(),
            final_tokenizer_state: Some(final_sid),
        }
    }
}

impl JSONConvertible for PrecomputedNodeContents0 {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert(
            "clean_end".to_string(),
            self.final_tokenizer_state.is_some().to_json(),
        );
        obj.insert("live_tokens".to_string(), self.live_tokens.to_json());
        obj.insert(
            "final_tokenizer_state".to_string(),
            self.final_tokenizer_state.to_json(),
        );
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let live_tokens = obj
                    .remove("live_tokens")
                    .ok_or_else(|| "Missing field live_tokens for PrecomputedNodeContents0".to_string())
                    .and_then(LLMTokenBV::from_json)?;
                let final_tokenizer_state = obj
                    .remove("final_tokenizer_state")
                    .ok_or_else(|| {
                        "Missing field final_tokenizer_state for PrecomputedNodeContents0".to_string()
                    })
                    .and_then(Option::<TokenizerStateID>::from_json)?;
                Ok(PrecomputedNodeContents0 {
                    live_tokens,
                    final_tokenizer_state,
                })
            }
            _ => Err("Expected JSONNode::Object for PrecomputedNodeContents0".to_string()),
        }
    }
}

// Final precompute1 types
pub type PrecomputeNode1 =
    Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;
pub type PrecomputeNode1Index = Trie2Index;
pub type Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode1Index>;

// Trie0 is now only a type alias passed into optimize_trie1_size; we never build it.
pub type PrecomputeNode0 = Trie<
    Option<(GrammarTokenID, Option<TokenizerStateID>)>,
    LLMTokenBV,
    PrecomputedNodeContents0,
>;
pub type PrecomputeNode0Index = Trie2Index;

// God wrappers
pub type Trie0GodWrapper = GodWrapper<
    Option<(GrammarTokenID, Option<TokenizerStateID>)>,
    LLMTokenBV,
    PrecomputedNodeContents0,
>;
pub type Trie0God = God<
    Option<(GrammarTokenID, Option<TokenizerStateID>)>,
    LLMTokenBV,
    PrecomputedNodeContents0,
>;
pub type Trie1GodWrapper =
    GodWrapper<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;
pub type Trie1God = God<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GrammarConstraintConfig {
    pub trie1: Trie1Config,
    pub run_precompute4: bool,
    pub use_dummy_terminals: bool,
    pub dummy_terminal_map: BTreeMap<String, BTreeSet<Terminal>>,
    pub dummy_terminal_penalties: BTreeMap<String, usize>,
}

impl Default for GrammarConstraintConfig {
    fn default() -> Self {
        Self {
            trie1: Trie1Config::off(),
            run_precompute4: true,
            use_dummy_terminals: false,
            dummy_terminal_map: BTreeMap::new(),
            dummy_terminal_penalties: BTreeMap::new(),
        }
    }
}

impl GrammarConstraintConfig {
    pub fn off() -> Self { Self::default() }
}

// ---------------------------------------------------------------------------
// Main structure
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GrammarConstraint {
    pub tokenizer: Regex,
    pub parser: GLRParser,

    // Precomputations
    pub precomputed1: Precomputed,
    pub precomputed4: Precomputed4,

    pub llm_vocab: Arc<LLMVocab>,
    pub(crate) token_name_map: BiBTreeMap<Terminal, usize>,

    /// Tokenizer state -> grammar terminal -> internal LLM token bitset.
    pub possible_matches: BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,

    /// Internal-token -> start_tokenizer_state -> end_tokenizer_state.
    pub state_map_by_llm:
        DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TokenizerStateID>>,
    /// Internal-token -> start_tokenizer_state -> terminals.
    pub terminal_map_by_llm:
        DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TerminalBV>>,

    pub(crate) trie1_god: Trie1GodWrapper,

    pub run_precompute4: bool,
    pub post_commit_allow_check_mode: TerminalAllowanceCheckMode,

    pub vocab: StageVocab,

    /// Maps original terminal IDs to dummy terminal IDs (if any).
    pub(crate) original_to_dummy_map: BTreeMap<TerminalID, TerminalID>,
}

impl GrammarConstraint {
    pub fn assert_eq(&self, other: &Self) {
        assert_eq!(self.tokenizer, other.tokenizer);
        assert_eq!(self.parser, other.parser);

        assert_eq!(self.precomputed1.len(), other.precomputed1.len());
        for ((sid1, arc1), (sid2, arc2)) in
            self.precomputed1.iter().zip(other.precomputed1.iter())
        {
            assert_eq!(sid1, sid2);
            assert!(PrecomputeNode1::are_graphs_equal(
                &self.trie1_god,
                *arc1,
                &other.trie1_god,
                *arc2
            ));
        }

        assert_eq!(
            self.llm_vocab.llm_token_map,
            other.llm_vocab.llm_token_map
        );
        assert_eq!(self.token_name_map, other.token_name_map);
        assert_eq!(self.possible_matches, other.possible_matches);
        assert_eq!(
            self.post_commit_allow_check_mode,
            other.post_commit_allow_check_mode
        );
        assert_eq!(self.state_map_by_llm, other.state_map_by_llm);
        assert_eq!(self.terminal_map_by_llm, other.terminal_map_by_llm);
        assert_eq!(self.vocab, other.vocab);
        assert_eq!(self.original_to_dummy_map, other.original_to_dummy_map);

        // precomputed4 still has no PartialEq; skip.
    }
}

impl JSONConvertible for GrammarConstraint {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("tokenizer".to_string(), self.tokenizer.to_json());
        obj.insert("parser".to_string(), self.parser.to_json());
        obj.insert("precomputed1".to_string(), self.precomputed1.to_json());
        obj.insert("precomputed4".to_string(), self.precomputed4.to_json());
        obj.insert("token_name_map".to_string(), self.token_name_map.to_json());
        obj.insert(
            "possible_matches".to_string(),
            self.possible_matches.to_json(),
        );
        obj.insert("trie1_god".to_string(), self.trie1_god.to_json());
        obj.insert(
            "run_precompute4".to_string(),
            self.run_precompute4.to_json(),
        );
        obj.insert(
            "post_commit_allow_check_mode".to_string(),
            self.post_commit_allow_check_mode.to_json(),
        );
        obj.insert(
            "state_map_by_llm".to_string(),
            self.state_map_by_llm.to_json(),
        );
        obj.insert(
            "terminal_map_by_llm".to_string(),
            self.terminal_map_by_llm.to_json(),
        );
        obj.insert("vocab".to_string(), self.vocab.to_json());
        obj.insert("llm_vocab".to_string(), self.llm_vocab.to_json());
        obj.insert(
            "original_to_dummy_map".to_string(),
            self.original_to_dummy_map.to_json(),
        );
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let tokenizer = obj
                    .remove("tokenizer")
                    .ok_or_else(|| "Missing field tokenizer".to_string())
                    .and_then(Regex::from_json)?;
                let parser = obj
                    .remove("parser")
                    .ok_or_else(|| "Missing field parser".to_string())
                    .and_then(GLRParser::from_json)?;
                let precomputed1 = obj
                    .remove("precomputed1")
                    .ok_or_else(|| "Missing field precomputed1".to_string())
                    .and_then(Precomputed::from_json)?;
                let precomputed4 = obj
                    .remove("precomputed4")
                    .ok_or_else(|| "Missing field precomputed4".to_string())
                    .and_then(Precomputed4::from_json)?;

                let token_name_map = obj
                    .remove("token_name_map")
                    .ok_or_else(|| "Missing field token_name_map".to_string())
                    .and_then(|n| BiBTreeMap::<Terminal, usize>::from_json(n))?;

                // possible_matches: prefer new key, fall back to old *_precompute1 for compatibility
                let possible_matches = if let Some(n) = obj.remove("possible_matches") {
                    BTreeMap::<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>::from_json(n)?
                } else if let Some(n) = obj.remove("possible_matches_precompute1") {
                    BTreeMap::<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>::from_json(n)?
                } else {
                    BTreeMap::new()
                };

                let trie1_god = obj
                    .remove("trie1_god")
                    .ok_or_else(|| "Missing field trie1_god".to_string())
                    .and_then(Trie1GodWrapper::from_json)?;

                let run_precompute4 = obj
                    .remove("run_precompute4")
                    .map(bool::from_json)
                    .transpose()?
                    .unwrap_or(true);

                let post_commit_allow_check_mode =
                    match obj.remove("post_commit_allow_check_mode") {
                        Some(n) => TerminalAllowanceCheckMode::from_json(n)?,
                        None => TerminalAllowanceCheckMode::default(),
                    };

                let state_map_by_llm =
                    match obj.remove("state_map_by_llm") {
                        Some(n) => DedupValueMap::<
                            LLMTokenID,
                            BTreeMap<TokenizerStateID, TokenizerStateID>,
                        >::from_json(n)?,
                        None => DedupValueMap::new(),
                    };
                let terminal_map_by_llm =
                    match obj.remove("terminal_map_by_llm") {
                        Some(n) => DedupValueMap::<
                            LLMTokenID,
                            BTreeMap<TokenizerStateID, TerminalBV>,
                        >::from_json(n)?,
                        None => DedupValueMap::new(),
                    };

                // Handle llm_vocab deserialization with fallback
                let llm_vocab = if let Some(n) = obj.remove("llm_vocab") {
                    Arc::new(LLMVocab::from_json(n)?)
                } else {
                    // Fallback to old format
                    let max_original_llm_token_id = obj
                        .remove("max_original_llm_token_id")
                        .ok_or_else(|| "Missing field max_original_llm_token_id".to_string())
                        .and_then(usize::from_json)?;

                    let llm_token_map = obj
                        .remove("llm_token_map")
                        .ok_or_else(|| "Missing field llm_token_map".to_string())
                        .and_then(|n| BiBTreeMap::<Vec<u8>, LLMTokenID>::from_json(n))?;

                    Arc::new(LLMVocab {
                        llm_token_map,
                        max_original_llm_token_id,
                    })
                };

                // Stage vocab: new key "vocab", fall back to old names if present.
                let mut vocab_node = if let Some(n) = obj.remove("vocab") {
                    n
                } else if let Some(n) = obj.remove("precompute_vocab") {
                    n
                } else {
                    return Err(
                        "Missing stage vocab (vocab/precompute_vocab/precompute0_vocab)"
                            .to_string(),
                    );
                };

                // For backward compatibility, inject max_original_llm_token_id into vocab JSON if needed.
                if let JSONNode::Object(ref mut vocab_obj) = vocab_node {
                    if !vocab_obj.contains_key("max_original_llm_token_id") {
                        vocab_obj.insert(
                            "max_original_llm_token_id".to_string(),
                            llm_vocab.max_original_llm_token_id.to_json(),
                        );
                    }
                }
                let vocab = StageVocab::from_json(vocab_node)?;

                let original_to_dummy_map = match obj.remove("original_to_dummy_map") {
                    Some(n) => BTreeMap::<TerminalID, TerminalID>::from_json(n)?,
                    None => BTreeMap::new(),
                };

                let gc = GrammarConstraint {
                    tokenizer,
                    parser,
                    precomputed1,
                    precomputed4,                    llm_vocab,
                    token_name_map,
                    possible_matches,
                    state_map_by_llm,
                    terminal_map_by_llm,
                    trie1_god,
                    run_precompute4,
                    post_commit_allow_check_mode,
                    vocab,
                    original_to_dummy_map,
                };
                Ok(gc)
            }
            _ => Err("Expected JSONNode::Object for GrammarConstraint".to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// K-Way Merge Iterator for Strategy 11
// ---------------------------------------------------------------------------

struct KWayMergeIter<'a> {
    iters: Vec<std::slice::Iter<'a, u32>>,
    heap: BinaryHeap<(Reverse<u32>, usize)>, // (Reverse(value), iter_index)
}

impl<'a> KWayMergeIter<'a> {
    fn new(slices: Vec<&'a [u32]>) -> Self {
        let mut iters: Vec<_> = slices.into_iter().map(|s| s.iter()).collect();
        let mut heap = BinaryHeap::with_capacity(iters.len());

        for (i, iter) in iters.iter_mut().enumerate() {
            if let Some(&val) = iter.next() {
                heap.push((Reverse(val), i));
            }
        }
        Self { iters, heap }
    }
}

impl<'a> Iterator for KWayMergeIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((Reverse(val), i)) = self.heap.pop() {
            if let Some(&next_val) = self.iters[i].next() {
                self.heap.push((Reverse(next_val), i));
            }
            Some(val)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

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
        Self::build_with_config(
            compiled_grammar.tokenizer,
            compiled_grammar.glr_parser,
            llm_token_map,
            token_name_map,
            max_original_llm_token_id,
            config,
        )
    }

    pub fn new(
        tokenizer: Regex,
        parser: GLRParser,
        llm_token_map: LLMTokenMap,
        token_name_map: BiBTreeMap<Terminal, usize>,
        max_original_llm_token_id: usize,
    ) -> Self {
        Self::build_with_config(
            tokenizer,
            parser,
            llm_token_map,
            token_name_map,
            max_original_llm_token_id,
            &GrammarConstraintConfig::default(),
        )
    }

    pub fn new_with_config(
        tokenizer: Regex,
        parser: GLRParser,
        llm_token_map: LLMTokenMap,
        token_name_map: BiBTreeMap<Terminal, usize>,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
    ) -> Self {
        Self::build_with_config(
            tokenizer,
            parser,
            llm_token_map,
            token_name_map,
            max_original_llm_token_id,
            config,
        )
    }

    /// Convenience entry point from a `GrammarDefinition`.
    /// If `config.use_dummy_terminals` is `true`, the productions are rewritten
    /// using `config.dummy_terminal_map` and the resulting grammar is compiled.
    pub fn new_from_grammar_definition(
        grammar_definition: Arc<GrammarDefinition>,
        llm_token_map: LLMTokenMap,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
    ) -> Self {
        let initial_compiled_grammar =
            CompiledGrammar::from_definition(grammar_definition.clone());

        if !config.use_dummy_terminals {
            return Self::from_compiled_grammar_with_config(
                initial_compiled_grammar,
                llm_token_map,
                max_original_llm_token_id,
                config,
            );
        }

        // Rewriting productions with user-specified dummy terminals.
        let (final_productions, new_dummy_terminals) =
            crate::glr::analyze::rewrite_productions_with_dummies(
                &grammar_definition.productions,
                &config.dummy_terminal_map,
            );

        let final_compiled_grammar = if !new_dummy_terminals.is_empty() {
            let mut final_grammar_def = (*grammar_definition).clone();
            final_grammar_def.productions = final_productions;
            for dummy_terminal in new_dummy_terminals {
                if let Terminal::RegexName(name) = dummy_terminal {
                    final_grammar_def.add_external_terminal(&name);
                }
            }
            CompiledGrammar::from_definition(Arc::new(final_grammar_def))
        } else {
            initial_compiled_grammar
        };

        Self::from_compiled_grammar_with_config(
            final_compiled_grammar,
            llm_token_map,
            max_original_llm_token_id,
            config,
        )
    }

    /// Compute the mapping from original LLM token IDs to internal IDs,
    /// grouping tokens that are equivalent w.r.t. the tokenizer automaton.
    pub(crate) fn setup_llm_token_mappings(
        original_llm_token_map: &LLMTokenMap,
        tokenizer: &Regex,
    ) -> BTreeMap<usize, usize> {
        if original_llm_token_map.len() < 10 {
            return original_llm_token_map
                .iter()
                .map(|(_bytes, id)| (id.0, id.0))
                .collect();
        }

        let mut sorted_tokens: Vec<_> = original_llm_token_map.iter().collect();
        sorted_tokens.sort_by_key(|(bytes, _id)| *bytes);

        let mut llm_token_strings: Vec<Vec<u8>> = Vec::with_capacity(sorted_tokens.len());
        let mut original_ids: Vec<LLMTokenID> = Vec::with_capacity(sorted_tokens.len());

        for (bytes, id) in sorted_tokens {
            llm_token_strings.push(bytes.clone());
            original_ids.push(*id);
        }

        let initial_states: Vec<usize> = tokenizer.iter_states().map(|s| s.0).collect();

        let equivalence_classes =
            equivalence_analysis_finite_automata::find_equivalence_classes(
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
                crate::debug!(
                    2,
                    "  - Reduction factor: {:.2}x",
                    num_original_tokens as f64 / num_classes as f64
                );
            }

            let mut class_size_dist: BTreeMap<usize, usize> = BTreeMap::new();
            for string_indices in equivalence_classes.values() {
                *class_size_dist.entry(string_indices.len()).or_insert(0) += 1;
            }

            let mut dist_str =
                String::from("  - Class size distribution (top 10 largest):");
            let mut sorted_dist: Vec<_> = class_size_dist.into_iter().collect();
            sorted_dist.sort_by_key(|&(size, _count)| std::cmp::Reverse(size));
            for (size, count) in sorted_dist.iter().take(10) {
                dist_str.push_str(&format!("\n    - size {}: {} classes", size, count));
            }
            if sorted_dist.len() > 10 {
                dist_str.push_str("\n    - ...");
            }
            crate::debug!(2, "{}", dist_str);

            if num_original_tokens < 1000 {
                println!("All equivalence classes:");
                for (_signature, string_indices) in &equivalence_classes {
                    let members: Vec<String> = string_indices
                        .iter()
                        .map(|&idx| {
                            let bytes = &llm_token_strings[idx];
                            format!("{:?}", String::from_utf8_lossy(bytes))
                        })
                        .collect();
                    println!("- {}", members.join(", "));
                }
            }
        }

        let mut original_to_internal_map = BTreeMap::new();
        let mut internal_id_counter = 0;
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

    fn build_with_config(
        tokenizer: Regex,
        parser: GLRParser,
        llm_token_map: LLMTokenMap,
        token_name_map: BiBTreeMap<Terminal, usize>,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
    ) -> Self {
        // Epsilon tokens are not supported.
        let epsilon_terminal_group_ids: BTreeSet<_> = tokenizer
            .execute_from_state(&[], tokenizer.initial_state_id())
            .matches
            .iter()
            .map(|token| token.id)
            .collect();
        let epsilon_terminals: BTreeSet<&Terminal> = epsilon_terminal_group_ids
            .iter()
            .filter_map(|id| token_name_map.get_by_right(id))
            .collect();
        assert!(
            epsilon_terminals.is_empty(),
            "Epsilon tokens (tokens that can match an empty string) are not supported by the \
             grammar constraint. Got: {:?}",
            epsilon_terminals
        );

        // Global original<->internal mapping.
        let original_to_internal_map =
            Self::setup_llm_token_mappings(&llm_token_map, &tokenizer);
        let internal_max_llm_token = original_to_internal_map
            .values()
            .copied()
            .max()
            .unwrap_or(0);

        let mut internal_to_original_map: BTreeMap<usize, LLMTokenBV> = BTreeMap::new();
        for (orig, int_id) in &original_to_internal_map {
            internal_to_original_map
                .entry(*int_id)
                .or_default()
                .insert(*orig);
        }

        // Build internal LLM token map keyed by bytes.
        let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
        for (bytes, original_id) in llm_token_map.iter() {
            if let Some(internal_id_val) = original_to_internal_map.get(&original_id.0) {
                internal_llm_token_map.insert(bytes.clone(), LLMTokenID(*internal_id_val));
            }
        }

        // Vocab tree for internal tokens.
        crate::debug!(2, "Building vocab prefix tree for possible_matches computation");
        let internal_tokens_for_vocab: Vec<(usize, Vec<u8>)> =
            internal_llm_token_map.iter().map(|(b, id)| (id.0, b.clone())).collect();
        let vocab_tree = VocabPrefixTree::build(&internal_tokens_for_vocab);
        crate::debug!(2, "Done building vocab prefix tree");

        // possible_matches: tokenizer_state -> terminal -> internal-token bitset
        let mut computed_possible_matches =
            BTreeMap::<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>::new();
        let mut pm_cache: HashMap<
            (*const VocabPrefixTreeNode, TokenizerStateID),
            BTreeMap<GrammarTokenID, LLMTokenBV>,
        > = HashMap::new();
        crate::debug!(
            2,
            "Computing possible_matches for all {} tokenizer states",
            tokenizer.iter_states().count()
        );
        for sid in tokenizer.iter_states() {
            let matches_for_sid = Self::compute_possible_matches_for_vocab_node(
                &tokenizer,
                &vocab_tree.root,
                sid,
                &mut pm_cache,
            );
            computed_possible_matches.insert(sid, matches_for_sid);
        }
        crate::debug!(2, "Finished computing possible_matches");

        // Build per-token maps.
        crate::debug!(2, "Building state_map_by_llm and terminal_map_by_llm");
        let state_map_by_llm = Self::build_state_map_by_llm(&tokenizer, &vocab_tree.root);
        let terminal_map_by_llm = Self::rearrange_possible_matches(&computed_possible_matches);

        // Compute terminal follow sets, then map to IDs.
        crate::debug!(2, "Computing terminal follow sets");
        let terminal_follow_sets_named = compute_terminal_follow_sets(&parser.productions);
        let mut terminal_follow_map: BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>> =
            BTreeMap::new();
        for (terminal1, following_terminals) in terminal_follow_sets_named {
            let t1_id = *parser
                .terminal_map
                .get_by_left(&terminal1)
                .unwrap_or_else(|| panic!("Terminal {:?} from follow sets not found in map", terminal1));
            let mut following_ids = BTreeSet::new();
            for t2 in following_terminals {
                let t2_id = *parser.terminal_map.get_by_left(&t2).unwrap();
                following_ids.insert(t2_id);
            }
            if !following_ids.is_empty() {
                terminal_follow_map.insert(t1_id, following_ids);
            }
        }
        crate::debug!(
            2,
            "Computed terminal_follow_map_ids with {} entries.",
            terminal_follow_map.len()
        );

        let llm_vocab = Arc::new(LLMVocab {
            llm_token_map: llm_token_map.clone(),
            max_original_llm_token_id,
        });

        // Single stage vocab used everywhere.
        let mut vocab = StageVocab {
            original_to_internal: original_to_internal_map.clone(),
            internal_to_original: internal_to_original_map.clone(),
            internal_max_llm_token: internal_max_llm_token,
            // These will be finalized after trie optimization
            max_original_llm_token_id: 0,
            internal_to_original_sparse_matrix: vec![],
        };

        // Verify dummy-terminal map has no overlapping originals and build original->dummy map.
        let mut seen_originals = BTreeSet::new();
        for original_terminals in config.dummy_terminal_map.values() {
            for term in original_terminals {
                if !seen_originals.insert(term.clone()) {
                    panic!(
                        "Original terminal '{}' is mapped by multiple dummy terminals.",
                        term
                    );
                }
            }
        }

        let mut original_to_dummy_map: BTreeMap<TerminalID, TerminalID> = BTreeMap::new();
        for (dummy_name, original_terminals) in &config.dummy_terminal_map {
            let dummy_term = Terminal::regex_name(dummy_name);
            if let Some(&dummy_id) = parser.terminal_map.get_by_left(&dummy_term) {
                for original_terminal in original_terminals {
                    if let Some(&original_id) =
                        parser.terminal_map.get_by_left(original_terminal)
                    {
                        original_to_dummy_map.insert(original_id, dummy_id);
                    }
                }
            }
        }

        // Precompute1
        let precompute_vocab_before_p1 = vocab.clone();
        let (precomputed1, trie1_god) = Self::precompute1(
            &tokenizer,
            Some(&parser),
            Some(llm_vocab.clone()),
            &internal_llm_token_map,
            &token_name_map,
            &mut vocab,
            &terminal_follow_map,
            config,
            original_to_dummy_map.clone(),
        );

        // possible_matches for precompute1 vocab
        let mut possible_matches_precompute1 = computed_possible_matches.clone();
        if precompute_vocab_before_p1.original_to_internal != vocab.original_to_internal {
            crate::debug!(
                2,
                "Remapping LLM token IDs in possible_matches due to Trie1 optimization."
            );
            let mut old_to_new_map: BTreeMap<usize, usize> = BTreeMap::new();
            for (original_id, old_internal_id) in &precompute_vocab_before_p1.original_to_internal {
                if let Some(new_internal_id) = vocab.original_to_internal.get(original_id) {
                    old_to_new_map.insert(*old_internal_id, *new_internal_id);
                }
            }

            for terminal_map in possible_matches_precompute1.values_mut() {
                for llm_token_bv in terminal_map.values_mut() {
                    let mut new_bv = LLMTokenBV::zeros();
                    for old_id in llm_token_bv.iter_up_to(usize::MAX) { // TODO: This is a hack
                        if let Some(new_id) = old_to_new_map.get(&old_id) {
                            new_bv.insert(*new_id);
                        }
                    }
                    *llm_token_bv = new_bv;
                }
            }

            crate::debug!(2, "Done");
        }

        // Remap per-token maps to the final vocab as well, if needed.
        let (state_map_by_llm, terminal_map_by_llm) = if precompute_vocab_before_p1
            .original_to_internal
            != vocab.original_to_internal
        {
            let mut old_to_new_map: BTreeMap<usize, usize> = BTreeMap::new();
            for (original_id, old_internal_id) in
                &precompute_vocab_before_p1.original_to_internal
            {
                if let Some(new_internal_id) = vocab.original_to_internal.get(original_id) {
                    old_to_new_map.insert(*old_internal_id, *new_internal_id);
                }
            }

            let mut new_state_map_by_llm = DedupValueMap::new();
            for (old_llm_id, value) in state_map_by_llm.iter() {
                if let Some(new_id) = old_to_new_map.get(&old_llm_id.0) {
                    new_state_map_by_llm.insert(LLMTokenID(*new_id), value.clone());
                }
            }

            let mut new_terminal_map_by_llm = DedupValueMap::new();
            for (old_llm_id, value) in terminal_map_by_llm.iter() {
                if let Some(new_id) = old_to_new_map.get(&old_llm_id.0) {
                    new_terminal_map_by_llm.insert(LLMTokenID(*new_id), value.clone());
                }
            }

            (new_state_map_by_llm, new_terminal_map_by_llm)
        } else {
            (state_map_by_llm, terminal_map_by_llm)
        };

        // Precompute4 (DWA). Even if config.run_precompute4 is false, we build it;
        // there is no longer a trie3-based fallback.
        let max_internal_llm_token_id = vocab.internal_max_llm_token;
        let precomputed4 = precompute4(&parser, &precomputed1, &trie1_god, max_internal_llm_token_id);

        // Stats for precompute1
        let mut stats = PrecomputeStats::default();
        crate::constraint_extra::calculate_final_stats1(
            &precomputed1,
            &mut stats,
            &trie1_god,
        );
        crate::constraint_extra::print_precompute_stats1(
            &stats,
            &token_name_map,
            &trie1_god,
        );

        let internal_to_original_sparse_matrix = StageVocab::build_internal_to_original_sparse_matrix(
            &vocab.internal_to_original,
            max_original_llm_token_id,
            vocab.internal_max_llm_token,
        );
        vocab.max_original_llm_token_id = max_original_llm_token_id;
        vocab.internal_to_original_sparse_matrix = internal_to_original_sparse_matrix;

        let gc = GrammarConstraint {
            tokenizer,
            parser,
            precomputed1,
            precomputed4,
            llm_vocab,
            token_name_map,
            possible_matches: possible_matches_precompute1,
            state_map_by_llm,
            terminal_map_by_llm,
            trie1_god,
            run_precompute4: config.run_precompute4,
            post_commit_allow_check_mode: TerminalAllowanceCheckMode::default(),
            vocab,
            original_to_dummy_map,
        };
        gc
    }

    // -----------------------------------------------------------------------
    // Precompute1
    // -----------------------------------------------------------------------

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
                    "LLM-compatible cycle detected in precompute1 trie for internal LLM token ID \
                     {}.\nCycle path:\n",
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
                    report.push_str(&format!(
                        "  {} --[{}]--> {}\n",
                        node_idx, edge_str, next_node_idx
                    ));
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
    ) -> Option<(Vec<(PrecomputeNode1Index, Option<Option<GrammarTokenID>>)>, LLMTokenID)>
    {
        path.push((node_idx, edge_key_opt));

        if let Some((tokens_on_stack, path_start_idx)) = recursion_stack.get(&node_idx) {
            let intersection = &current_tokens & tokens_on_stack;
            if !intersection.is_empty() {
                let cycle_llm_token = intersection.iter_up_to(usize::MAX).next().unwrap();
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
                        *child_idx,
                        Some(edge_key.clone()),
                        next_tokens,
                        arena,
                        recursion_stack,
                        visited,
                        path,
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
        internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
        token_name_map: &BiBTreeMap<Terminal, usize>,
        stage_vocab: &mut StageVocab,
        terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
        config: &GrammarConstraintConfig,
        original_to_dummy_map: BTreeMap<TerminalID, TerminalID>,
    ) -> (BTreeMap<TokenizerStateID, PrecomputeNode1Index>, Trie1GodWrapper)
    {
        let mut dummy_terminal_penalties: BTreeMap<TerminalID, usize> =
            BTreeMap::new();
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

        // Reduce internal_llm_token_map to representatives to speed up precomputation
        let mut representative_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
        let mut seen_internal_ids = std::collections::HashSet::new();

        for (bytes, id) in internal_llm_token_map {
            if seen_internal_ids.insert(id.0) {
                representative_llm_token_map.insert(bytes.clone(), *id);
            }
        }

        let representative_states: Vec<TokenizerStateID> = tokenizer.iter_states().collect();

        let mut helper = Precomputer1::new(
            tokenizer,
            parser,
            llm_vocab,
            &representative_llm_token_map,
            stage_vocab.internal_max_llm_token,
            original_to_dummy_map,
            representative_states,
        );

        helper.run_dfs();

        let (mut precomputed1, trie1_god) = helper.finish();
        let roots_after: Vec<_> = precomputed1.values().cloned().collect();

        Self::has_llm_compatible_cycle(
            &trie1_god,
            &roots_after,
            stage_vocab.internal_max_llm_token,
        );

        let mut stats = PrecomputeStats::default();
        crate::constraint_extra::calculate_final_stats1(
            &precomputed1,
            &mut stats,
            &trie1_god,
        );
        crate::constraint_extra::print_precompute_stats1(
            &stats,
            token_name_map,
            &trie1_god,
        );

        // Trie1 optimization (size, vocab compression)
        constraint_precompute1_utils::optimize_trie1_size(
            &mut precomputed1,
            &trie1_god,
            // Dummy values for Trie0-dependent params (we no longer build Trie0).
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

    // -----------------------------------------------------------------------
    // Special precomputation
    // -----------------------------------------------------------------------

    pub fn dump_precomputed4(&self) {
        println!("\n--- Precomputed4 DWA ---");
        println!("{}", self.precomputed4);
    }

    // -----------------------------------------------------------------------
    // Vocab helpers
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    pub fn all_internal_llm_tokens_bitset(&self) -> LLMTokenBV {
        LLMTokenBV::ones(self.vocab.internal_max_llm_token + 1)
    }

    /// Convert an internal BV (using `self.vocab`) back to original IDs.
    pub fn internal_bv_to_original(&self, internal_bv: &LLMTokenBV) -> Bitset {
        self.vocab.internal_bv_to_original(internal_bv)
    }

    pub fn original_bv_to_internal(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        self.vocab.original_bv_to_internal(original_bv)
    }

    pub fn internal_to_original(&self, internal_id: LLMTokenID) -> Option<LLMTokenID> {
        self.vocab
            .internal_to_original
            .get(&internal_id.0)
            .and_then(|bv| bv.iter_up_to(self.vocab.internal_max_llm_token).next())
            .map(|v| LLMTokenID(v))
    }

    #[inline]
    pub fn original_id_to_internal(&self, original_id: LLMTokenID) -> Option<LLMTokenID> {
        self.vocab
            .original_to_internal
            .get(&original_id.0)
            .map(|v| LLMTokenID(*v))
    }

    fn original_bv_to_internal_with_map(
        original_bv: &LLMTokenBV,
        original_to_internal: &BTreeMap<usize, usize>,
        max_original_llm_token_id: usize,
    ) -> LLMTokenBV {
        let mut internal_bv = HybridBitset::zeros();
        if original_bv.is_all() {
            for &internal_id in original_to_internal.values() {
                internal_bv.insert(internal_id);
            }
        } else {
            for i in original_bv.iter_up_to(max_original_llm_token_id) {
                if let Some(&internal_id) = original_to_internal.get(&i) {
                    internal_bv.insert(internal_id);
                }
            }
        }
        internal_bv
    }

    // -----------------------------------------------------------------------
    // Possible-matches-related helpers
    // -----------------------------------------------------------------------

    /// Build per-token (internal id) mapping from initial tokenizer state to final tokenizer state
    /// after consuming the entire token.
    pub fn build_state_map_by_llm(
        tokenizer: &Regex,
        vocab_root: &VocabPrefixTreeNode,
    ) -> DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TokenizerStateID>> {
        let mut initial_map: BTreeMap<TokenizerStateID, TokenizerStateID> =
            BTreeMap::new();
        for sid in tokenizer.iter_states() {
            initial_map.insert(sid, sid);
        }
        let out = std::sync::Mutex::new(DedupValueMap::new());

        fn dfs(
            tokenizer: &Regex,
            node: &VocabPrefixTreeNode,
            current_map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
            out: &std::sync::Mutex<DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TokenizerStateID>>>,
        ) {
            // Optimization: Group by current state to avoid redundant regex executions.
            // Many start states map to the same current state.
            let mut target_to_sources: HashMap<TokenizerStateID, Vec<TokenizerStateID>> = HashMap::new();
            for (src, dst) in current_map {
                target_to_sources.entry(*dst).or_default().push(*src);
            }

            let children: Vec<_> = node.iter_children().collect();

            children.par_iter().for_each(|(segment_bytes, child)| {
                let mut next_map: BTreeMap<TokenizerStateID, TokenizerStateID> =
                    BTreeMap::new();

                for (cur, sources) in &target_to_sources {
                    let exec = tokenizer.execute_from_state(segment_bytes, *cur);
                    if let Some(end_state) = exec.end_state {
                        let end_sid = TokenizerStateID(end_state);
                        for src in sources {
                            next_map.insert(*src, end_sid);
                        }
                    }
                }

                if !next_map.is_empty() {
                    let tok_id = child.token_id();
                    {
                        let mut guard = out.lock().unwrap();
                        guard.insert(LLMTokenID(tok_id), next_map.clone());
                    }
                    dfs(tokenizer, child, &next_map, out);
                }
            });
        }

        dfs(tokenizer, vocab_root, &initial_map, &out);
        out.into_inner().unwrap()
    }

    /// Rearrange possible_matches: state -> terminal -> BV(tokens)
    /// into token -> state -> set(terminals).
    pub fn rearrange_possible_matches(
        pm: &BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    ) -> DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TerminalBV>> {
        let tmp = pm.iter().par_bridge()
            .map(|(sid, tmap)| {
                let mut local_map: BTreeMap<LLMTokenID, BTreeMap<TokenizerStateID, TerminalBV>> = BTreeMap::new();
                for (term, bv) in tmap {
                    if bv.is_all() { continue; }
                    for tok in bv.iter_up_to(usize::MAX) {
                        local_map.entry(LLMTokenID(tok))
                            .or_default()
                            .entry(*sid)
                            .or_default()
                            .insert(term.0);
                    }
                }
                local_map
            })
            .reduce(
                BTreeMap::new,
                |mut map_a, map_b| {
                    for (tok, state_map_b) in map_b {
                        let state_map_a = map_a.entry(tok).or_default();
                        state_map_a.extend(state_map_b);
                    }
                    map_a
                }
            );

        let mut out = DedupValueMap::new();
        for (tok, m) in tmp {
            out.insert(tok, m);
        }
        out
    }

    fn compute_possible_matches_for_vocab_node(
        tokenizer: &Regex,
        vocab_node: &VocabPrefixTreeNode,
        tokenizer_state_id: TokenizerStateID,
        cache: &mut HashMap<
            (*const VocabPrefixTreeNode, TokenizerStateID),
            BTreeMap<GrammarTokenID, LLMTokenBV>,
        >,
    ) -> BTreeMap<GrammarTokenID, LLMTokenBV> {
        let cache_key = (vocab_node as *const VocabPrefixTreeNode, tokenizer_state_id);
        if let Some(cached_result) = cache.get(&cache_key) {
            return cached_result.clone();
        }

        let mut result_map: BTreeMap<GrammarTokenID, LLMTokenBV> = BTreeMap::new();

        for (segment_bytes, child_vocab_arc) in vocab_node.iter_children() {
            let child_vocab_node_ref = child_vocab_arc;
            let exec_result =
                tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);

            for token_match in &exec_result.matches {
                let grammar_token_id = GrammarTokenID(token_match.id);
                let applicable_tokens_rangeset =
                    child_vocab_node_ref.reachable_token_ids();
                result_map
                    .entry(grammar_token_id)
                    .or_insert_with(LLMTokenBV::zeros)
                    .extend(applicable_tokens_rangeset.iter());
            }

            if let Some(final_state_val) = exec_result.end_state {
                let final_tokenizer_state_id = TokenizerStateID(final_state_val);

                let matches_possible_from_new_tokenizer_state: BTreeSet<_> =
                    tokenizer
                        .tokens_accessible_from_state(final_tokenizer_state_id)
                        .into_iter()
                        .collect();

                let matches_from_current_segment: BTreeSet<_> = exec_result
                    .matches
                    .iter()
                    .map(|m| GrammarTokenID(m.id))
                    .collect();

                let new_grammar_tokens_to_look_for =
                    &matches_possible_from_new_tokenizer_state
                        - &matches_from_current_segment;

                if !new_grammar_tokens_to_look_for.is_empty() {
                    let next_results = Self::compute_possible_matches_for_vocab_node(
                        tokenizer,
                        child_vocab_node_ref,
                        final_tokenizer_state_id,
                        cache,
                    );
                    for (token, bv) in next_results {
                        *result_map
                            .entry(token)
                            .or_insert_with(LLMTokenBV::zeros) |= bv;
                    }
                }
            }
        }
        cache.insert(cache_key, result_map.clone());
        result_map
    }

    // -----------------------------------------------------------------------
    // Top-level state construction
    // -----------------------------------------------------------------------

    pub fn init(&self) -> GrammarConstraintState<'_> {
        let mut state = BTreeMap::new();
        state.insert(
            self.tokenizer.initial_state_id(),
            self.parser.init_glr_parser(Some(self.llm_vocab.clone())),
        );
        GrammarConstraintState { parent: self, state }
    }

    pub fn state_with_nodes(
        &self,
        nodes: Vec<(usize, Arc<GSSNode>)>,
    ) -> GrammarConstraintState<'_> {
        todo!()
        // let mut state = BTreeMap::new();
        // for (i, node) in nodes.into_iter() {
        //     state.insert(
        //         TokenizerStateID(i),
        //         self.parser.init_glr_parser_from_stack(node),
        //     );
        // }
        // GrammarConstraintState { parent: self, state }
    }

    pub fn state_from_gss_map(&self, gss_map: &BTreeMap<TokenizerStateID, GSSNode>) -> GrammarConstraintState {
        let mut state = BTreeMap::new();
        for (i, node) in gss_map.iter() {
            state.insert(
                *i,
                self.parser.init_parse_state_with_gss(node.clone()),
            );
        }
        GrammarConstraintState { parent: self, state }
    }

    pub fn print_gss_nodes(
        &self,
        roots: &Vec<Arc<GSSNode>>,
        labels: Option<&[String]>,
    ) {
        // let config = GSSPrintConfig {
        //     labels,
        //     max_edges: 500,
        //     original_internal_bimap: None,
        //     llm_token_map: Some(&self.llm_vocab.llm_token_map),
        //     verbose: false,
        // };
        //
        // let (gss_str, _state_ids) =
        //     print_gss_forest(roots, &self.parser.terminal_map, &config);
        // println!("{}", gss_str);
    }
}

// ---------------------------------------------------------------------------
// Precomputer1
// ---------------------------------------------------------------------------

pub(crate) struct Precomputer1<'r> {
    pub(crate) tokenizer: &'r Regex,
    pub(crate) parser: Option<&'r GLRParser>,
    pub(crate) llm_vocab: Option<Arc<LLMVocab>>,
    pub(crate) vocab: VocabPrefixTree,
    pub(crate) roots: BTreeMap<TokenizerStateID, NWAStateID>,
    pub(crate) possible_matches: RefCell<
        BTreeMap<
            *const VocabPrefixTreeNode,
            BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>,
        >,
    >,
    pub(crate) all_llm_tokens: RangeSetBlaze<usize>,
    pub(crate) pb: ProgressBar,
    pub(crate) stats: PrecomputeStats,
    pub(crate) leaf_state: NWAStateID,
    pub(crate) nwa: NWA,
    pub(crate) live_tokens: HashMap<NWAStateID, RangeSetBlaze<usize>>,
    pub(crate) original_to_dummy_map: BTreeMap<TerminalID, TerminalID>,
}

impl<'r> Precomputer1<'r> {
    fn new(
        tokenizer: &'r Regex,
        parser: Option<&'r GLRParser>,
        llm_vocab: Option<Arc<LLMVocab>>,
        internal_llm_token_map: &BTreeMap<Vec<u8>, LLMTokenID>,
        internal_max_llm_token: usize,
        original_to_dummy_map: BTreeMap<TerminalID, TerminalID>,
        active_states: Vec<TokenizerStateID>,
    ) -> Self {
        let tokens: Vec<(usize, Vec<u8>)> = internal_llm_token_map
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone()))
            .collect();

        crate::debug!(2, "Building vocab prefix tree");
        let vocab = VocabPrefixTree::build(&tokens);
        crate::debug!(2, "Done building vocab prefix tree");

        let mut nwa = NWA::new();
        nwa.states.0.clear(); // Clear default start state
        let mut live_tokens = HashMap::new();

        let mut roots = BTreeMap::new();
        for sid in active_states {
            let root_state = nwa.add_state();
            live_tokens.insert(root_state, RangeSetBlaze::from_iter(0..=internal_max_llm_token));
            roots.insert(sid, root_state);
        }
        crate::debug!(
            2,
            "Created trie1 roots for {} representative tokenizer states",
            roots.len()
        );

        crate::debug!(2, "Counting vocab nodes for progress bar...");
        let total_nodes = count_vocab_nodes(&vocab.root);
        crate::debug!(2, "Counted {} vocab nodes", total_nodes);
        let pb = ProgressBar::new(total_nodes);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] \
                     [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta})",
                )
                .expect("progress-bar"),
        );
        if !PROGRESS_BAR_ENABLED {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }

        let leaf_state = nwa.add_state();
        nwa.states[leaf_state].final_weight = Some(Weight::all());
        live_tokens.insert(leaf_state, RangeSetBlaze::new());
        crate::debug!(2, "Created trie1 leaf state");

        Self {
            tokenizer,
            parser,
            llm_vocab,
            vocab,
            roots,
            possible_matches: RefCell::new(BTreeMap::new()),
            all_llm_tokens: RangeSetBlaze::from_iter(0..=internal_max_llm_token),
            pb,
            stats: PrecomputeStats::default(),
            leaf_state,
            nwa,
            live_tokens,
            original_to_dummy_map,
        }
    }

    fn get_leaf_node(&self) -> NWAStateID {
        self.leaf_state
    }

    fn finish(self) -> (BTreeMap<TokenizerStateID, PrecomputeNode1Index>, Trie1GodWrapper)
    {
        let final_trie1_god = Trie1GodWrapper::new();
        let mut final_roots = BTreeMap::new();
        let mut node_map: HashMap<
            NWAStateID,
            PrecomputeNode1Index,
        > = HashMap::new();

        for (sid, temp_root) in &self.roots {
            let final_root = self.convert_nwa_to_trie(
                *temp_root,
                &final_trie1_god,
                &mut node_map,
            );
            final_roots.insert(*sid, final_root);
        }

        (final_roots, final_trie1_god)
    }

    fn convert_nwa_to_trie(
        &self,
        state_id: NWAStateID,
        final_god: &Trie1GodWrapper,
        node_map: &mut HashMap<NWAStateID, PrecomputeNode1Index>,
    ) -> PrecomputeNode1Index {
        if let Some(final_idx) = node_map.get(&state_id) {
            return *final_idx;
        }

        let live = self.live_tokens.get(&state_id).cloned().unwrap_or_else(RangeSetBlaze::new);
        let is_end = self.nwa.states[state_id].final_weight.as_ref().map_or(false, |w| !w.is_empty());

        let final_node_contents = PrecomputedNodeContents {
            end: is_end,
            live_tokens: HybridBitset::from(live),
        };
        let new_node = PrecomputeNode1::new(final_node_contents);
        let final_idx = PrecomputeNode1Index::new(final_god.insert(new_node));
        node_map.insert(state_id, final_idx);

        // Group transitions by label
        let mut children_to_copy: BTreeMap<Option<GrammarTokenID>, Vec<(NWAStateID, RangeSetBlaze<usize>)>> = BTreeMap::new();
        for (label, targets) in &self.nwa.states[state_id].transitions {
            let grammar_token_id = GrammarTokenID(*label as usize);
            for (target, weight) in targets {
                // Convert SimpleBitset weight back to RangeSetBlaze
                let rsb = weight.rsb.clone();
                children_to_copy.entry(Some(grammar_token_id)).or_default().push((*target, rsb));
            }
        }

        if self.original_to_dummy_map.is_empty() {
            for (ek, dest_map) in children_to_copy {
                for (child_state_id, rs_blaze) in dest_map {
                    let final_child_idx = self.convert_nwa_to_trie(
                        child_state_id,
                        final_god,
                        node_map,
                    );
                    let hybrid_bitset = HybridBitset::from(rs_blaze);
                    final_god.insert_edge_simple(
                        final_idx,
                        final_child_idx,
                        ek.clone(),
                        hybrid_bitset,
                    );
                }
            }
        } else {
            let mut direct_edges = Vec::new();
            let mut injected_edges_by_dummy: BTreeMap<
                TerminalID,
                Vec<(
                    Option<TerminalID>,
                    OrderedHashMap<PrecomputeNode1Index, RangeSetBlaze<usize>>,
                )>,
            > = BTreeMap::new();

            for (ek, dest_map) in children_to_copy {
                if let Some(tid) = ek {
                    if let Some(dummy_tid) =
                        self.original_to_dummy_map.get(&tid)
                    {
                        injected_edges_by_dummy
                            .entry(*dummy_tid)
                            .or_default()
                            .push((Some(tid), dest_map.into_iter().map(|(s, w)| (Trie2Index::from(s), w)).collect()));
                        continue;
                    }
                }
                direct_edges.push((ek, dest_map));
            }

            for (ek, dest_map) in direct_edges {
                for (child_state_id, rs_blaze) in dest_map {
                    let final_child_idx = self.convert_nwa_to_trie(
                        child_state_id,
                        final_god,
                        node_map,
                    );
                    let hybrid_bitset = HybridBitset::from(rs_blaze);
                    final_god.insert_edge_simple(
                        final_idx,
                        final_child_idx,
                        ek.clone(),
                        hybrid_bitset,
                    );
                }
            }

            for (dummy_tid, edges) in injected_edges_by_dummy {
                let inter_node =
                    PrecomputeNode1::new(PrecomputedNodeContents::internal());
                let inter_idx =
                    PrecomputeNode1Index::new(final_god.insert(inter_node));
                let mut total_inter_bitset = HybridBitset::zeros();

                for (original_ek, dest_map) in edges {
                    for (child_state_id, rs_blaze) in dest_map {
                        let final_child_idx = self.convert_nwa_to_trie(
                            child_state_id.as_usize(),
                            final_god,
                            node_map,
                        );
                        let hybrid_bitset = HybridBitset::from(rs_blaze);
                        total_inter_bitset |= &hybrid_bitset;
                        final_god.insert_edge_simple(
                            inter_idx,
                            final_child_idx,
                            original_ek,
                            hybrid_bitset,
                        );
                    }
                }
                final_god.insert_edge_simple(
                    final_idx,
                    inter_idx,
                    Some(dummy_tid),
                    total_inter_bitset,
                );
            }
        }

        final_idx
    }

    fn possible_matches(
        &self,
        vocab_node: &VocabPrefixTreeNode,
        tokenizer_state_id: TokenizerStateID,
    ) -> BTreeMap<GrammarTokenID, LLMTokenBV> {
        let cache_key_ptr = vocab_node as *const VocabPrefixTreeNode;

        if let Some(cached_for_vocab_node) =
            self.possible_matches.borrow().get(&cache_key_ptr)
        {
            if let Some(cached_result) =
                cached_for_vocab_node.get(&tokenizer_state_id)
            {
                return cached_result.clone();
            }
        }

        let mut result_map: BTreeMap<GrammarTokenID, LLMTokenBV> = BTreeMap::new();

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let exec_result =
                self.tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);
            for token in &exec_result.matches {
                let grammar_token_id = GrammarTokenID(token.id);
                let applicable_tokens = child_vocab_node.reachable_token_ids();
                *result_map
                    .entry(grammar_token_id)
                    .or_insert_with(LLMTokenBV::zeros) |=
                    HybridBitset::from(applicable_tokens);
            }
            if let Some(final_state_val) = exec_result.end_state {
                let matches_possible_from_tokenizer_state: BTreeSet<_> = self
                    .tokenizer
                    .tokens_accessible_from_state(TokenizerStateID(final_state_val))
                    .into_iter()
                    .collect();
                let matches_here: BTreeSet<_> = exec_result
                    .matches
                    .iter()
                    .map(|m| GrammarTokenID(m.id))
                    .collect();
                let possible_new_matches =
                    &matches_possible_from_tokenizer_state - &matches_here;
                if !possible_new_matches.is_empty() {
                    let next_results = self.possible_matches(
                        child_vocab_node,
                        TokenizerStateID(final_state_val),
                    );
                    for (token, bv) in next_results {
                        *result_map
                            .entry(token)
                            .or_insert_with(LLMTokenBV::zeros) |= bv;
                    }
                }
            }
        }

        self.possible_matches
            .borrow_mut()
            .entry(cache_key_ptr)
            .or_default()
            .insert(tokenizer_state_id, result_map.clone());

        result_map
    }

    fn run_dfs(&mut self) {
        let mut assoc: BTreeMap<
            TokenizerStateID,
            HashMap<NWAStateID, RangeSetBlaze<usize>>,
        > = BTreeMap::new();

        for (sid, arc) in &self.roots {
            assoc
                .entry(*sid)
                .or_default()
                .insert(arc.clone(), self.all_llm_tokens.clone());
        }

        crate::debug!(2, "Starting precompute DFS for {} tokenizer states", self.roots.len());
        crate::debug!(6, "Roots for each tokenizer state:");
        for (sid, root) in &self.roots {
            crate::debug!(6, "  {}: {}", sid.0, root);
        }
        profiler::reset();
        let vocab = std::mem::replace(&mut self.vocab, VocabPrefixTree::new());
        self.dfs(&vocab.root, assoc);
        self.vocab = vocab;
        crate::debug!(2, "Finished precompute DFS");
        self.pb.finish();
        profiler::print_summary();
        crate::debug!(2, "Precomputation complete");
    }

    fn dfs(
        &mut self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<
            TokenizerStateID,
            HashMap<NWAStateID, RangeSetBlaze<usize>>,
        >,
    ) {
        self.pb.inc(1);

        // Structures for batching updates to avoid mutable borrow conflicts
        struct PendingEdge {
            src: NWAStateID,
            dst: NWAStateID,
            key: Option<GrammarTokenID>,
            bv: RangeSetBlaze<usize>,
        }
        let mut edges = Vec::new();
        let mut live_updates: HashMap<NWAStateID, RangeSetBlaze<usize>> = HashMap::new();

        // Local cache for node properties to avoid repeated lookups
        let mut node_data_cache: HashMap<NWAStateID, (RangeSetBlaze<usize>, bool)> = HashMap::new();

        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let child_token_id = child_vocab_node.token_id();
            let child_reachable = child_vocab_node.reachable_token_ids();

            let mut queue = BTreeMap::new();
            queue.insert(0, assoc_by_state.clone());
            let mut next_assoc = BTreeMap::new();

            // Cache possible_matches results for this child to avoid re-computation
            let mut pm_cache = HashMap::new();

            while let Some((pos, states)) = queue.pop_first() {
                // If we've consumed the whole segment, propagate states to the next vocab level
                if pos == segment_bytes.len() {
                    for (sid, nodes) in states {
                        let entry: &mut HashMap<NWAStateID, RangeSetBlaze<usize>> = next_assoc.entry(sid).or_default();
                        for (node, tokens) in nodes {
                            entry.entry(node).or_default().bitor_assign(&tokens);
                        }
                    }
                    continue;
                }

                // Execute tokenizer on the rest of the segment
                let rest = &segment_bytes[pos..];
                for (sid, nodes) in states {
                    let exec = self.tokenizer.execute_from_state(rest, sid);

                    // Pre-calculate possible matches if end_state is reached
                    // (Used for masking out tokens that are valid continuations)
                    let empty_pm = BTreeMap::new();
                    let current_pm = if let Some(end) = exec.end_state {
                        let ts = TokenizerStateID(end);
                        pm_cache.entry(ts).or_insert_with(|| self.possible_matches(child_vocab_node, ts))
                    } else {
                        &empty_pm
                    };

                    for (src, src_tokens) in nodes {
                        if !node_data_cache.contains_key(&src) {
                            let live = self.live_tokens.get(&src).cloned().unwrap_or_else(RangeSetBlaze::new);
                            let is_end = self.nwa.states[src].final_weight.as_ref().map_or(false, |w| !w.is_empty());
                            node_data_cache.insert(src, (live, is_end));
                        }
                        let (src_live, _) = node_data_cache.get(&src).unwrap();

                        // 1. Process Matches (Transitions)
                        for m in &exec.matches {
                            let tid = GrammarTokenID(m.id);
                            let next_pos = pos + m.width;

                            // Calculate edge mask (which LLM tokens allow this transition?)
                            let mut mask = child_reachable.clone();
                            if next_pos == segment_bytes.len() {
                                mask.remove(child_token_id);
                            }
                            if let Some(bad) = current_pm.get(&tid) {
                                mask = &mask - bad.inner.as_ref();
                            }

                            // Intersect with context and source live tokens
                            let final_mask = &(&mask & &src_tokens) & src_live;
                            if final_mask.is_empty() { continue; }

                            // A. Transition to Leaf (if match ends segment)
                            if next_pos == segment_bytes.len() {
                                let mut leaf_mask = RangeSetBlaze::from_iter([child_token_id]);
                                leaf_mask = &(&leaf_mask & &src_tokens) & src_live;
                                if !leaf_mask.is_empty() {
                                    let dst = self.get_leaf_node();
                                    edges.push(PendingEdge { src, dst, key: Some(tid), bv: leaf_mask.clone() });
                                    live_updates.entry(dst).or_default().bitor_assign(&leaf_mask);
                                }
                            }

                            // B. Transition to Next NWA Node
                            let next_sid = self.tokenizer.initial_state_id();
                            let dest_queue = queue.entry(next_pos).or_default().entry(next_sid).or_default();

                            // Attempt to merge into an existing destination node
                            let mut dst = None;

                            // Check candidates in queue
                            for (cand, cand_tokens) in dest_queue.iter() {
                                if !node_data_cache.contains_key(cand) {
                                    let live = self.live_tokens.get(cand).cloned().unwrap_or_else(RangeSetBlaze::new);
                                    let is_end = self.nwa.states[*cand].final_weight.as_ref().map_or(false, |w| !w.is_empty());
                                    node_data_cache.insert(*cand, (live, is_end));
                                }
                                let (cand_live, cand_end) = node_data_cache.get(cand).unwrap();
                                if *cand_end { continue; }
                                let risky = &final_mask - cand_tokens;
                                if risky.is_empty() || (&risky & cand_live).is_empty() {
                                    dst = Some(*cand);
                                    break;
                                }
                            }

                            // Check existing transitions from src
                            if dst.is_none() {
                                if let Some(targets) = self.nwa.states[src].transitions.get(&(tid.0 as i16)) {
                                    for (cand, _) in targets {
                                        if !node_data_cache.contains_key(cand) {
                                            let live = self.live_tokens.get(cand).cloned().unwrap_or_else(RangeSetBlaze::new);
                                            let is_end = self.nwa.states[*cand].final_weight.as_ref().map_or(false, |w| !w.is_empty());
                                            node_data_cache.insert(*cand, (live, is_end));
                                        }
                                        let (cand_live, cand_end) = node_data_cache.get(cand).unwrap();
                                        if !*cand_end && (cand_live & &final_mask).is_empty() {
                                            dst = Some(*cand);
                                            break;
                                        }
                                    }
                                }
                            }

                            // Create new node if no merge possible
                            let dst = dst.unwrap_or_else(|| {
                                let idx = self.nwa.add_state();
                                live_updates.insert(idx, RangeSetBlaze::new());
                                node_data_cache.insert(idx, (RangeSetBlaze::new(), false));
                                idx
                            });

                            edges.push(PendingEdge { src, dst, key: Some(tid), bv: final_mask.clone() });

                            // Update tracking for destination
                            live_updates.entry(dst).or_default().bitor_assign(&final_mask);
                            if let Some(d) = node_data_cache.get_mut(&dst) { d.0 |= &final_mask; }
                            dest_queue.entry(dst).or_default().bitor_assign(&final_mask);
                        }

                        // 2. Handle End State (Leaf transitions for next token)
                        if let Some(end) = exec.end_state {
                            let end_ts = TokenizerStateID(end);

                            // Check for valid leaf transitions
                            let mut leaf_bv = RangeSetBlaze::from_iter([child_token_id]);
                            leaf_bv = &(&leaf_bv & &src_tokens) & src_live;

                            if !leaf_bv.is_empty() {
                                for t in self.tokenizer.tokens_accessible_from_state(end_ts) {
                                    let dst = self.get_leaf_node();
                                    edges.push(PendingEdge { src, dst, key: Some(t), bv: leaf_bv.clone() });
                                    live_updates.entry(dst).or_default().bitor_assign(&leaf_bv);
                                }
                            }

                            // Propagate to next level via next_assoc
                            next_assoc.entry(end_ts).or_default().entry(src).or_default().bitor_assign(&src_tokens);
                        }
                    }
                }
            }

            // Apply batched updates
            for e in edges.drain(..) {
                let w = SimpleBitset::from_rsb(e.bv);
                self.nwa.add_transition(e.src, e.key.unwrap().0 as i16, e.dst, w).unwrap();
            }
            for (n, bv) in live_updates.drain() {
                self.live_tokens.entry(n).or_default().bitor_assign(&bv);
            }

            if !next_assoc.is_empty() {
                self.dfs(child_vocab_node, next_assoc);
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

// ---------------------------------------------------------------------------
// Merge implementation for leveled GSS
// ---------------------------------------------------------------------------

impl Merge for RangeSetBlaze<usize> {
    fn merge(&self, other: &Self) -> Self { self | other }
}

impl Merge for Arc<RangeSetBlaze<usize>> {
    fn merge(&self, other: &Self) -> Self {
        if Arc::ptr_eq(self, other) {
            return self.clone();
        }
        let mut merged = self.as_ref().clone();
        merged |= other.as_ref();
        if merged == **self {
            self.clone()
        } else if merged == **other {
            other.clone()
        } else {
            Arc::new(merged)
        }
    }
}

// ---------------------------------------------------------------------------
// GrammarConstraintState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GrammarConstraintState<'a> {
    pub parent: &'a GrammarConstraint,
    pub state: BTreeMap<TokenizerStateID, GLRParserState<'a>>,
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
        return Ok(());
        // writeln!(
        //     f,
        //     "GrammarConstraintState ({} active tokenizer states):",
        //     self.state.len()
        // )?;
        // if self.state.is_empty() { return Ok(()); }
        //
        // let mut gss_roots = Vec::new();
        // let mut tokenizer_state_info = Vec::new();
        //
        // for (tokenizer_state_id, glr_state) in &self.state {
        //     if !glr_state.stack.is_empty() {
        //         gss_roots.push(glr_state.stack.clone());
        //         tokenizer_state_info.push(format!(
        //             "  - Tokenizer State {:>3}: GSS Root ({} predecessors)",
        //             tokenizer_state_id.0,
        //             glr_state.stack.num_predecessors()
        //         ));
        //     } else {
        //         tokenizer_state_info.push(format!(
        //             "  - Tokenizer State {:>3}: (Empty GSS)",
        //             tokenizer_state_id.0
        //         ));
        //     }
        // }
        //
        // for info in tokenizer_state_info {
        //     writeln!(f, "{}", info)?;
        // }
        //
        // if !gss_roots.is_empty() {
        //     writeln!(f, "\nCombined GSS Forest (showing up to 50 nodes):")?;
        //     let config = GSSPrintConfig {
        //         labels: None,
        //         max_edges: 50,
        //         original_internal_bimap: None,
        //         llm_token_map: Some(&self.parent.llm_vocab.llm_token_map),
        //         verbose: false,
        //     };
        //     let (gss_str, _) =
        //         print_gss_forest(&gss_roots, &self.parent.parser.terminal_map, &config);
        //     write!(f, "{}", gss_str)?;
        // }

        Ok(())
    }
}

impl<'a> GrammarConstraintState<'a> {
    pub(crate) fn transform_gss_stacks<M, F>(&mut self, mut f: F)
    where
        M: Default,
        F: FnMut(&mut Arc<GSSNode>, &mut M),
    {
        let mut memo = M::default();
        for s in self.state.values_mut() {
            f(&mut Arc::new(s.stack.clone()), &mut memo);
        }
    }

    pub(crate) fn map_gss_stacks<M, F>(&mut self, mut f: F)
    where
        M: Default,
        F: FnMut(&mut Arc<GSSNode>, &mut M) -> Arc<GSSNode>,
    {
        let mut memo = M::default();
        for s in self.state.values_mut() {
            s.stack = f(&mut Arc::new(s.stack.clone()), &mut memo).as_ref().clone();
        }
    }

    pub fn compute_commit_maps(
        &self,
        llm_token_bytes: &[u8],
    ) -> (
        BTreeMap<TokenizerStateID, TokenizerStateID>,
        BTreeMap<TokenizerStateID, TerminalBV>,
    ) {
        let mut state_map: BTreeMap<TokenizerStateID, TokenizerStateID> =
            BTreeMap::new();
        let mut terminals_map: BTreeMap<TokenizerStateID, TerminalBV> =
            BTreeMap::new();
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
        // Trie3-based get_mask3 has been removed; we always use DWA now.
        self.get_mask4().into()
    }

    pub fn print_gss_stats(&self) {
        // println!("GrammarConstraintState Stats:");
        // println!("  - Active tokenizer states: {}", self.state.len());
        // if self.state.is_empty() {
        //     println!("  - GSS is empty.");
        //     return;
        // }
        // let stats = gather_gss_stats(
        //     &self
        //         .state
        //         .values()
        //         .map(|s| s.stack.as_ref())
        //         .collect::<Vec<_>>(),
        // );
        // println!("  - GSS Stats: {:#?}", stats);
        // todo!()
    }

    pub fn print_gss(&self) {
        // let roots: Vec<_> = self
        //     .state
        //     .values()
        //     .map(|s| s.stack.clone())
        //     .collect();
        // if roots.is_empty() {
        //     println!("GSS is empty.");
        //     return;
        // }
        // let labels: Vec<_> = self
        //     .state
        //     .keys()
        //     .map(|k| format!("Tokenizer State {}", k.0))
        //     .collect();
        // self.parent.print_gss_nodes(&roots, Some(&labels));
        // todo!()
    }

    pub fn explain_stack(&self) {
        // todo!()
        // for (state_id, state) in &self.state {
        //     println!("\n--- State {} ---", state_id.0);
        //     let mut seen = BTreeSet::new();
        //     let num_to_sample = 10;
        //     for i in 0..1000 {
        //         if let Some(sampled_path_edges) =
        //             sample_path(&[&state.stack], i)
        //         {
        //             let mut sampled_stack: Vec<usize> = sampled_path_edges
        //                 .iter()
        //                 .map(|edge| edge.state_id.0)
        //                 .collect();
        //             sampled_stack.reverse();
        //             if seen.contains(&sampled_stack) {
        //                 continue;
        //             }
        //             seen.insert(sampled_stack);
        //             if seen.len() >= num_to_sample {
        //                 break;
        //             }
        //         };
        //     }
        //     for sampled_stack in seen {
        //         println!("  Sampled stack: {:?}", sampled_stack);
        //     }
        //     if let Some(sampled_path_edges) =
        //         sample_path(&[&state.stack], 1)
        //     {
        //         let mut sampled_stack: Vec<_> = sampled_path_edges
        //             .iter()
        //             .map(|edge| edge.state_id)
        //             .collect();
        //         sampled_stack.reverse();
        //         let explanation =
        //             self.parent.parser.explain_stack(&sampled_stack);
        //         for line in explanation.lines() {
        //             println!("      {}", line);
        //         }
        //     };
        // }
    }

    pub fn num_unique_nodes(&self) -> usize {
        // gather_gss_stats(
        //     &self
        //         .state
        //         .values()
        //         .map(|s| s.stack.as_ref())
        //         .collect::<Vec<_>>(),
        // )
        // .unique_nodes()
        // todo!()
        0
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) {
        self.commit_bytes(
            &self
                .parent
                .llm_vocab
                .llm_token_map
                .get_by_right(&llm_token_id)
                .unwrap()
                .clone(),
        );
    }

    pub fn is_active(&self) -> bool { !self.state.is_empty() }

    pub fn is_valid(&self) -> bool {
        if self.state.is_empty() {
            return false;
        }
        if self.state.contains_key(&self.parent.tokenizer.initial_state_id()) {
            return true;
        }
        for (tid, glr_state) in self.state.iter() {
            for gtid in self.parent.tokenizer.tokens_accessible_from_state(TokenizerStateID(tid.0)) {
                let mut glr_state = glr_state.clone();
                glr_state.step(gtid);
                if glr_state.is_ok() {
                    return true;
                }
            }
        }
        false
    }

    pub fn state(&self) -> &BTreeMap<TokenizerStateID, GLRParserState<'a>> {
        &self.state
    }
}