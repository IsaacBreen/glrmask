#![allow(clippy::too_many_arguments)]

use std::{
    borrow::Borrow,
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::{self, Debug, Display, Formatter},
    iter::FromIterator,
    sync::Arc,
};

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
    precompute4::weighted_automata::common::StateID as WAStateID,
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
use crate::datastructures::gss_acc::Acc;
use crate::glr::parser::{ParseState, ParseStateEdgeContent};
use crate::glr::table::StateID;
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
            ito.push((*k, bv.iter().collect::<Vec<_>>()));
        }
        m.insert("internal_to_original".to_string(), ito.to_json());
        m.insert(
            "internal_max_llm_token".to_string(),
            self.internal_max_llm_token.to_json(),
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
                let ito_vec: Vec<(usize, Vec<usize>)> = obj
                    .remove("internal_to_original")
                    .ok_or("StageVocab: missing internal_to_original".to_string())
                    .and_then(Vec::from_json)?;
                let internal_to_original: BTreeMap<usize, LLMTokenBV> = ito_vec
                    .into_iter()
                    .map(|(k, v)| (k, v.into_iter().collect()))
                    .collect();
                Ok(StageVocab {
                    original_to_internal,
                    internal_to_original,
                    internal_max_llm_token,
                })
            }
            _ => Err("StageVocab: expected object".to_string()),
        }
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

// ---------------------------------------------------------------------------
// Temporary trie for building precompute1
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TempPrecomputedNodeContents {
    pub(crate) end: bool,
    pub(crate) live_tokens: RangeSetBlaze<usize>,
}

impl TempPrecomputedNodeContents {
    pub(crate) fn root(internal_max_llm_token_id: usize) -> Self {
        Self {
            end: false,
            live_tokens: RangeSetBlaze::from_iter(0..=internal_max_llm_token_id),
        }
    }

    pub(crate) fn internal() -> Self {
        Self {
            end: false,
            live_tokens: RangeSetBlaze::new(),
        }
    }

    pub(crate) fn leaf() -> Self {
        Self {
            end: true,
            live_tokens: RangeSetBlaze::new(),
        }
    }
}

type TempPrecomputeNode1 =
    Trie<Option<GrammarTokenID>, RangeSetBlaze<usize>, TempPrecomputedNodeContents>;
type TempPrecomputeNode1Index = Trie2Index;
type TempTrie1GodWrapper =
    GodWrapper<Option<GrammarTokenID>, RangeSetBlaze<usize>, TempPrecomputedNodeContents>;

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
            trie1: Trie1Config::default(),
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
        obj.insert(
            "llm_token_map".to_string(),
            self.llm_vocab.llm_token_map.to_json(),
        );
        obj.insert(
            "max_original_llm_token_id".to_string(),
            self.llm_vocab.max_original_llm_token_id.to_json(),
        );
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

                let llm_token_map = obj
                    .remove("llm_token_map")
                    .ok_or_else(|| "Missing field llm_token_map".to_string())
                    .and_then(|n| BiBTreeMap::<Vec<u8>, LLMTokenID>::from_json(n))?;
                let max_original_llm_token_id = obj
                    .remove("max_original_llm_token_id")
                    .ok_or_else(|| {
                        "Missing field max_original_llm_token_id".to_string()
                    })
                    .and_then(usize::from_json)?;
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

                // Stage vocab: new key "vocab", fall back to old names if present.
                let vocab = if let Some(n) = obj.remove("vocab") {
                    StageVocab::from_json(n)?
                } else if let Some(n) = obj.remove("precompute_vocab") {
                    StageVocab::from_json(n)?
                } else {
                    return Err("Missing stage vocab (vocab/precompute_vocab/precompute0_vocab)"
                        .to_string());
                };

                let original_to_dummy_map = match obj.remove("original_to_dummy_map") {
                    Some(n) => BTreeMap::<TerminalID, TerminalID>::from_json(n)?,
                    None => BTreeMap::new(),
                };

                let mut gc = GrammarConstraint {
                    tokenizer,
                    parser,
                    precomputed1,
                    precomputed4,
                    llm_vocab: Arc::new(LLMVocab {
                        llm_token_map,
                        max_original_llm_token_id,
                    }),
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
                    for old_id in llm_token_bv.iter() {
                        if let Some(new_id) = old_to_new_map.get(&old_id) {
                            new_bv.insert(*new_id);
                        }
                    }
                    *llm_token_bv = new_bv;
                }
            }
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
        let precomputed4 = precompute4(&parser, &precomputed1, &trie1_god);

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

        let mut gc = GrammarConstraint {
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

    fn has_llm_compatible_cycle_temp(
        arena: &TempTrie1GodWrapper,
        roots: &[TempPrecomputeNode1Index],
        internal_max_llm_token: usize,
    ) {
        let mut visited: HashMap<TempPrecomputeNode1Index, RangeSetBlaze<usize>> =
            HashMap::new();
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
                    "LLM-compatible cycle detected in precompute1 temp trie for internal LLM \
                     token ID {}.\nCycle path:\n",
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

    fn detect_cycle_recursive_temp(
        node_idx: TempPrecomputeNode1Index,
        edge_key_opt: Option<Option<GrammarTokenID>>,
        current_tokens: RangeSetBlaze<usize>,
        arena: &TempTrie1GodWrapper,
        recursion_stack: &mut HashMap<
            TempPrecomputeNode1Index,
            (RangeSetBlaze<usize>, usize),
        >,
        visited: &mut HashMap<TempPrecomputeNode1Index, RangeSetBlaze<usize>>,
        path: &mut Vec<(TempPrecomputeNode1Index, Option<Option<GrammarTokenID>>)>,
    ) -> Option<(Vec<(TempPrecomputeNode1Index, Option<Option<GrammarTokenID>>)>, LLMTokenID)>
    {
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

        let mut helper = Precomputer1::new(
            tokenizer,
            parser,
            llm_vocab,
            internal_llm_token_map,
            stage_vocab.internal_max_llm_token,
            original_to_dummy_map,
        );

        helper.run_dfs();
        let roots_before: Vec<_> = helper.roots.values().cloned().collect();
        Self::has_llm_compatible_cycle_temp(
            &helper.trie1_god,
            &roots_before,
            stage_vocab.internal_max_llm_token,
        );

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

    pub fn all_internal_llm_tokens_bitset(&self) -> LLMTokenBV {
        LLMTokenBV::ones(self.vocab.internal_max_llm_token + 1)
    }

    /// Convert an internal BV (using `self.vocab`) back to original IDs.
    pub fn internal_bv_to_original(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        self.internal_bv_to_original_with_map(
            internal_bv,
            &self.vocab.internal_to_original,
        )
    }


    pub fn original_bv_to_internal(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        self.original_bv_to_internal_with_map(
            original_bv,
            &self.vocab.original_to_internal,
        )
    }

    pub fn internal_to_original(&self, internal_id: LLMTokenID) -> Option<LLMTokenID> {
        self.vocab
            .internal_to_original
            .get(&internal_id.0)
            .and_then(|bv| bv.iter().next())
            .map(|v| LLMTokenID(v))
    }

    #[inline]
    pub fn original_id_to_internal(&self, original_id: LLMTokenID) -> Option<LLMTokenID> {
        self.vocab
            .original_to_internal
            .get(&original_id.0)
            .map(|v| LLMTokenID(*v))
    }

    fn internal_bv_to_original_with_map(
        &self,
        internal_bv: &LLMTokenBV,
        internal_to_original: &BTreeMap<usize, LLMTokenBV>,
    ) -> LLMTokenBV {
        if !internal_to_original.is_empty() {
            let i2o_num_entries = internal_to_original.len();
            let i2o_total_ranges: usize = internal_to_original
                .values()
                .map(|bv| bv.inner().ranges_len())
                .sum();
            let i2o_total_len: usize = internal_to_original.values().map(|bv| bv.len()).sum();
            let i2o_avg_ranges = i2o_total_ranges as f64 / i2o_num_entries as f64;
            let i2o_avg_len = i2o_total_len as f64 / i2o_num_entries as f64;

            println!("[perf] internal_bv_to_original_with_map stats:");
            println!("  - internal_to_original map:");
            println!("    - Entries: {}", i2o_num_entries);
            println!("    - Avg ranges per value: {:.2}", i2o_avg_ranges);
            println!("    - Avg len per value: {:.2}", i2o_avg_len);
            println!("  - input internal_bv:");
            println!("    - Total len: {}", internal_bv.len());
            println!("    - Num ranges: {}", internal_bv.inner().ranges_len());
        }

        let mut internal_bv = internal_bv.clone();
        if internal_bv.is_all() {
            internal_bv = HybridBitset::ones(self.vocab.internal_max_llm_token + 1);
        }

        // STRATEGY 1
        let instant = std::time::Instant::now();
        let mut original_bv_rsb = RangeSetBlaze::new();
        for i in internal_bv.iter() {
            if let Some(bv) = internal_to_original.get(&i) {
                original_bv_rsb |= bv.inner.as_ref();
            }
        }
        let elapsed1 = instant.elapsed();

        let output_len = {
            let count_u128 = original_bv_rsb.len();
            count_u128.try_into().unwrap_or(usize::MAX)
        };
        let output_ranges = original_bv_rsb.ranges_len();
        println!("  - output original_bv:");
        println!("    - Total len: {}", output_len);
        println!("    - Num ranges: {}", output_ranges);

        println!("[perf] STRATEGY 1 (BTree + RangeSetBlaze): {:?}", elapsed1);

        // STRATEGY 2
        let mut i2o2: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (i, bv) in internal_to_original {
            i2o2.insert(*i, bv.inner.iter().collect::<HashSet<_>>());
        }
        let instant = std::time::Instant::now();
        let mut bv2: HashSet<usize> = HashSet::new();
        for i in internal_bv.iter() {
            if let Some(bv) = i2o2.get(&i) {
                bv2.extend(bv.iter().copied());
            }
        }
        println!("[perf] STRATEGY 2 (BTree + im::HashSet):    {:?}", instant.elapsed());

        // STRATEGY 3: HashMap lookup
        let i2o_hashmap: std::collections::HashMap<_, _> =
            internal_to_original.iter().map(|(k, v)| (*k, v)).collect();
        let instant = std::time::Instant::now();
        let mut original_bv_3 = RangeSetBlaze::new();
        for i in internal_bv.iter() {
            if let Some(bv) = i2o_hashmap.get(&i) {
                original_bv_3 |= bv.inner.as_ref();
            }
        }
        println!("[perf] STRATEGY 3 (HashMap + RangeSetBlaze): {:?}", instant.elapsed());

        // STRATEGY 4: Vec lookup
        let mut i2o_vec: Vec<Option<LLMTokenBV>> =
            vec![None; self.vocab.internal_max_llm_token + 1];
        for (i, bv) in internal_to_original.iter() {
            if *i < i2o_vec.len() {
                i2o_vec[*i] = Some(bv.clone());
            }
        }
        let instant = std::time::Instant::now();
        let mut original_bv_4 = RangeSetBlaze::new();
        for i in internal_bv.iter() {
            if let Some(Some(bv)) = i2o_vec.get(i) {
                original_bv_4 |= bv.inner.as_ref();
            }
        }
        println!("[perf] STRATEGY 4 (Vec + RangeSetBlaze):     {:?}", instant.elapsed());

        // STRATEGY 5: Rayon
        let instant = std::time::Instant::now();
        let _original_bv_5 = internal_bv.inner.ranges().par_bridge().map(|range| {
            let mut partial_bv = RangeSetBlaze::new();
            for i in range {
                if let Some(bv) = internal_to_original.get(&i) {
                    partial_bv |= bv.inner.as_ref();
                }
            }
            partial_bv
        }).reduce(RangeSetBlaze::new, |a, b| a | b);
        println!("[perf] STRATEGY 5 (Rayon):                  {:?}", instant.elapsed());

        // STRATEGY 6: Pre-computed Bitset Matrix
        let max_original_id = self.llm_vocab.max_original_llm_token_id;
        let original_vocab_size_words = (max_original_id / 64) + 1;
        let num_internal_tokens = self.vocab.internal_max_llm_token + 1;

        let mut internal_to_original_bitset_matrix: Vec<u64> =
            vec![0; num_internal_tokens * original_vocab_size_words];

        for (internal_id, original_bv) in internal_to_original.iter() {
            if *internal_id >= num_internal_tokens {
                continue;
            }
            let row_start_idx = *internal_id * original_vocab_size_words;

            for original_id in original_bv.iter() {
                if original_id > max_original_id {
                    continue;
                }
                let word_idx = original_id / 64;
                let bit_idx = original_id % 64;
                internal_to_original_bitset_matrix[row_start_idx + word_idx] |= 1 << bit_idx;
            }
        }

        let instant = std::time::Instant::now();
        let mut result_bitset_words = vec![0u64; original_vocab_size_words];
        for internal_id in internal_bv.iter() {
            if internal_id >= num_internal_tokens {
                continue;
            }
            let row_start_idx = internal_id * original_vocab_size_words;
            let bitset_slice = &internal_to_original_bitset_matrix
                [row_start_idx..row_start_idx + original_vocab_size_words];

            for i in 0..original_vocab_size_words {
                result_bitset_words[i] |= bitset_slice[i];
            }
        }
        // To be fair, we must convert back to the required type.
        let _original_bv_6 = result_bitset_words
            .iter()
            .enumerate()
            .flat_map(|(word_idx, &word)| {
                (0..64).filter_map(move |bit_idx| {
                    if (word >> bit_idx) & 1 == 1 {
                        Some(word_idx * 64 + bit_idx)
                    } else {
                        None
                    }
                })
            })
            .collect::<RangeSetBlaze<usize>>();
        println!("[perf] STRATEGY 6 (Bitset Matrix):          {:?}", instant.elapsed());

        HybridBitset::from(original_bv_rsb)
    }

    fn original_bv_to_internal_with_map(
        &self,
        original_bv: &LLMTokenBV,
        original_to_internal: &BTreeMap<usize, usize>,
    ) -> LLMTokenBV {
        let mut internal_bv = HybridBitset::zeros();
        if original_bv.is_all() {
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
        let mut out = DedupValueMap::new();

        fn dfs(
            tokenizer: &Regex,
            node: &VocabPrefixTreeNode,
            current_map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
            out: &mut DedupValueMap<
                LLMTokenID,
                BTreeMap<TokenizerStateID, TokenizerStateID>,
            >,
        ) {
            for (segment_bytes, child) in node.iter_children() {
                let mut next_map: BTreeMap<TokenizerStateID, TokenizerStateID> =
                    BTreeMap::new();
                for (start, cur) in current_map {
                    let exec = tokenizer.execute_from_state(&segment_bytes, *cur);
                    if let Some(end_state) = exec.end_state {
                        next_map.insert(*start, TokenizerStateID(end_state));
                    }
                }

                let tok_id = child.token_id();
                out.insert(LLMTokenID(tok_id), next_map.clone());

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
        let mut tmp: BTreeMap<
            LLMTokenID,
            BTreeMap<TokenizerStateID, TerminalBV>,
        > = BTreeMap::new();
        for (sid, tmap) in pm {
            for (term, bv) in tmap {
                if bv.is_all() {
                    // can't efficiently enumerate; skip; commit falls back for missing entries.
                    continue;
                }
                for tok in bv.iter() {
                    let tok_id = LLMTokenID(tok);
                    let per_state = tmp.entry(tok_id).or_default();
                    per_state
                        .entry(*sid)
                        .or_default()
                        .insert(term.0);
                }
            }
        }
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
    pub(crate) roots: BTreeMap<TokenizerStateID, TempPrecomputeNode1Index>,
    pub(crate) possible_matches: RefCell<
        BTreeMap<
            *const VocabPrefixTreeNode,
            BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>,
        >,
    >,
    pub(crate) all_llm_tokens: RangeSetBlaze<usize>,
    pub(crate) pb: ProgressBar,
    pub(crate) stats: PrecomputeStats,
    pub(crate) leaf_node: TempPrecomputeNode1Index,
    pub(crate) trie1_god: TempTrie1GodWrapper,
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
                TempPrecomputeNode1Index::new(trie1_god.insert(
                    TempPrecomputeNode1::new(
                        TempPrecomputedNodeContents::root(internal_max_llm_token),
                    ),
                )),
            );
        }
        crate::debug!(
            2,
            "Created trie1 roots for {} tokenizer states",
            tokenizer.iter_states().count()
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

        let leaf_node = TempPrecomputeNode1Index::new(trie1_god.insert(
            TempPrecomputeNode1::new(TempPrecomputedNodeContents::leaf()),
        ));
        crate::debug!(2, "Created trie1 leaf node");

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
            leaf_node,
            trie1_god,
            original_to_dummy_map,
        }
    }

    fn get_leaf_node(&self) -> TempPrecomputeNode1Index {
        self.leaf_node.clone()
    }

    fn finish(self) -> (BTreeMap<TokenizerStateID, PrecomputeNode1Index>, Trie1GodWrapper)
    {
        let final_trie1_god = Trie1GodWrapper::new();
        let mut final_roots = BTreeMap::new();
        let mut node_map: HashMap<
            TempPrecomputeNode1Index,
            PrecomputeNode1Index,
        > = HashMap::new();

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
            live_tokens: HybridBitset::from(temp_guard.value.live_tokens.clone()),
        };
        let new_node = PrecomputeNode1::new(final_node_contents);
        let final_idx = PrecomputeNode1Index::new(final_god.insert(new_node));
        node_map.insert(temp_idx, final_idx);

        let children_to_copy = temp_guard.children().clone();
        drop(temp_guard);

        if self.original_to_dummy_map.is_empty() {
            for (ek, dest_map) in children_to_copy {
                for (temp_child_idx, rs_blaze) in dest_map {
                    let final_child_idx = self.convert_trie1_recursive(
                        temp_child_idx,
                        temp_god,
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
                    OrderedHashMap<TempPrecomputeNode1Index, RangeSetBlaze<usize>>,
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
                            .push((Some(tid), dest_map));
                        continue;
                    }
                }
                direct_edges.push((ek, dest_map));
            }

            for (ek, dest_map) in direct_edges {
                for (temp_child_idx, rs_blaze) in dest_map {
                    let final_child_idx = self.convert_trie1_recursive(
                        temp_child_idx,
                        temp_god,
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
                    for (temp_child_idx, rs_blaze) in dest_map {
                        let final_child_idx = self.convert_trie1_recursive(
                            temp_child_idx,
                            temp_god,
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
            HashMap<TempPrecomputeNode1Index, RangeSetBlaze<usize>>,
        >,
    ) {
        self.pb.inc(1);
        for (segment_bytes, child_vocab_node) in vocab_node.iter_children() {
            let mut work_queue: BTreeMap<
                usize,
                BTreeMap<
                    TokenizerStateID,
                    HashMap<TempPrecomputeNode1Index, RangeSetBlaze<usize>>,
                >,
            > = BTreeMap::new();
            work_queue.insert(0, assoc_by_state.clone());

            let mut next_level_assoc: BTreeMap<_, HashMap<_, _>> = BTreeMap::new();

            let mut node_cache: HashMap<
                TempPrecomputeNode1Index,
                (RangeSetBlaze<usize>, bool),
            > = HashMap::new();
            let get_node_data = |cache: &mut HashMap<_, _>,
                                 idx: TempPrecomputeNode1Index,
                                 god: &TempTrie1GodWrapper| {
                cache
                    .entry(idx)
                    .or_insert_with(|| {
                        let guard = idx.read(god).unwrap();
                        (guard.value.live_tokens.clone(), guard.value.end)
                    })
                    .clone()
            };

            let mut pending_edges: Vec<(
                TempPrecomputeNode1Index,
                TempPrecomputeNode1Index,
                Option<GrammarTokenID>,
                RangeSetBlaze<usize>,
            )> = Vec::new();
            let mut pending_live_token_updates: HashMap<
                TempPrecomputeNode1Index,
                RangeSetBlaze<usize>,
            > = HashMap::new();

            let child_reachable = child_vocab_node.reachable_token_ids();
            let child_token_id = child_vocab_node.token_id();

            let mut possible_matches_cache: HashMap<
                TokenizerStateID,
                BTreeMap<GrammarTokenID, LLMTokenBV>,
            > = HashMap::new();

            while let Some((pos, states_at_pos)) = work_queue.pop_first() {
                if pos == segment_bytes.len() {
                    for (tokenizer_state_id, nodes_with_tokens) in states_at_pos {
                        let entry =
                            next_level_assoc.entry(tokenizer_state_id).or_default();
                        for (node, tokens) in nodes_with_tokens {
                            entry
                                .entry(node)
                                .or_insert_with(RangeSetBlaze::new)
                                .bitor_assign(&tokens);
                        }
                    }
                    continue;
                }

                for (tokenizer_state_id, precompute_nodes_with_tokens) in
                    states_at_pos
                {
                    let exec_result = self
                        .tokenizer
                        .execute_from_state(&segment_bytes[pos..], tokenizer_state_id);

                    let possible_matches_at_end =
                        if let Some(end_state_val) = exec_result.end_state {
                            let ts = TokenizerStateID(end_state_val);
                            possible_matches_cache
                                .entry(ts)
                                .or_insert_with(|| {
                                    self.possible_matches(child_vocab_node, ts)
                                })
                        } else {
                            &BTreeMap::new()
                        };

                    for match_info in &exec_result.matches {
                        let terminal_id = GrammarTokenID(match_info.id);
                        let next_pos = pos + match_info.width;

                        for (src_node_wrapper, src_contextual_tokens) in
                            &precompute_nodes_with_tokens
                        {
                            let src_node_idx = *src_node_wrapper;

                            let (src_live_tokens, _) =
                                get_node_data(&mut node_cache, src_node_idx, &self.trie1_god);

                            if next_pos == segment_bytes.len() {
                                let mut edge_bv = RangeSetBlaze::new();
                                edge_bv.insert(child_token_id);
                                let final_edge_bv = &(&edge_bv & src_contextual_tokens)
                                    & &src_live_tokens;

                                if !final_edge_bv.is_empty() {
                                    let end_idx = self.get_leaf_node();
                                    pending_edges.push((
                                        src_node_idx,
                                        end_idx,
                                        Some(terminal_id),
                                        final_edge_bv.clone(),
                                    ));
                                    pending_live_token_updates
                                        .entry(end_idx)
                                        .or_insert_with(RangeSetBlaze::new)
                                        .bitor_assign(&final_edge_bv);
                                }
                            }

                            let mut edge_bv = child_reachable.clone();
                            if next_pos == segment_bytes.len() {
                                edge_bv.remove(child_token_id);
                            }
                            if let Some(matches_for_terminal) =
                                possible_matches_at_end.get(&terminal_id)
                            {
                                edge_bv =
                                    &edge_bv - matches_for_terminal.inner.as_ref();
                            }

                            let edge_bv_for_inserter =
                                &(&edge_bv & src_contextual_tokens) & &src_live_tokens;
                            if edge_bv_for_inserter.is_empty() {
                                continue;
                            }

                            let next_tokenizer_state =
                                self.tokenizer.initial_state_id();
                            let dest_nodes_in_queue = work_queue
                                .entry(next_pos)
                                .or_default()
                                .entry(next_tokenizer_state)
                                .or_default();

                            let mut dest_node_opt = dest_nodes_in_queue
                                .iter()
                                .filter_map(
                                    |(dest_node, dest_contextual_tokens)| {
                                        let (dest_live_tokens, is_end) =
                                            get_node_data(
                                                &mut node_cache,
                                                *dest_node,
                                                &self.trie1_god,
                                            );
                                        if is_end {
                                            return None;
                                        }

                                        let risky_tokens =
                                            &edge_bv_for_inserter - dest_contextual_tokens;
                                        if risky_tokens.is_empty()
                                            || (&risky_tokens
                                                & &dest_live_tokens)
                                                .is_empty()
                                        {
                                            Some(*dest_node)
                                        } else {
                                            None
                                        }
                                    },
                                )
                                .next();

                            if dest_node_opt.is_none() {
                                let children_of_src: Vec<
                                    TempPrecomputeNode1Index,
                                > = {
                                    let guard =
                                        src_node_idx.read(&self.trie1_god).unwrap();
                                    guard
                                        .children()
                                        .values()
                                        .flat_map(|m| m.keys().cloned())
                                        .collect()
                                };

                                dest_node_opt = children_of_src
                                    .iter()
                                    .filter(|child_arc| {
                                        let (child_live_tokens, is_end) =
                                            get_node_data(
                                                &mut node_cache,
                                                **child_arc,
                                                &self.trie1_god,
                                            );
                                        !is_end
                                            && (&child_live_tokens
                                                & &edge_bv_for_inserter)
                                                .is_empty()
                                    })
                                    .copied()
                                    .next();
                            }

                            let result_node = dest_node_opt.unwrap_or_else(|| {
                                let new_node = TempPrecomputeNode1::new(
                                    TempPrecomputedNodeContents::internal(),
                                );
                                let idx = TempPrecomputeNode1Index::new(
                                    self.trie1_god.insert(new_node),
                                );
                                node_cache.insert(
                                    idx,
                                    (RangeSetBlaze::new(), false),
                                );
                                idx
                            });

                            pending_edges.push((
                                src_node_idx,
                                result_node,
                                Some(terminal_id),
                                edge_bv_for_inserter.clone(),
                            ));
                            pending_live_token_updates
                                .entry(result_node)
                                .or_insert_with(RangeSetBlaze::new)
                                .bitor_assign(&edge_bv_for_inserter);

                            node_cache
                                .entry(result_node)
                                .and_modify(|(live, _)| {
                                    *live |= &edge_bv_for_inserter
                                });

                            dest_nodes_in_queue
                                .entry(result_node)
                                .or_insert_with(RangeSetBlaze::new)
                                .bitor_assign(&edge_bv_for_inserter);
                        }
                    }

                    if let Some(end_state_val) = exec_result.end_state {
                        let final_tokenizer_state =
                            TokenizerStateID(end_state_val);
                        let accessible_terminals = self
                            .tokenizer
                            .tokens_accessible_from_state(final_tokenizer_state);

                        for (src_node_wrapper, src_contextual_tokens) in
                            &precompute_nodes_with_tokens
                        {
                            let mut edge_bv = RangeSetBlaze::new();
                            edge_bv.insert(child_token_id);
                            let edge_bv_for_inserter =
                                &edge_bv & src_contextual_tokens;
                            if edge_bv_for_inserter.is_empty() {
                                continue;
                            }

                            let src_node_idx = *src_node_wrapper;
                            let (src_live_tokens, _) =
                                get_node_data(&mut node_cache, src_node_idx, &self.trie1_god);
                            let final_edge_bv =
                                &edge_bv_for_inserter & &src_live_tokens;

                            if !final_edge_bv.is_empty() {
                                let end_idx = self.get_leaf_node();
                                for terminal_id in &accessible_terminals {
                                    pending_edges.push((
                                        src_node_idx,
                                        end_idx,
                                        Some(*terminal_id),
                                        final_edge_bv.clone(),
                                    ));
                                    pending_live_token_updates
                                        .entry(end_idx)
                                        .or_insert_with(RangeSetBlaze::new)
                                        .bitor_assign(&final_edge_bv);
                                }
                            }
                        }

                        let entry =
                            next_level_assoc.entry(final_tokenizer_state).or_default();
                        for (node, tokens) in precompute_nodes_with_tokens {
                            entry
                                .entry(node)
                                .or_default()
                                .bitor_assign(&tokens);
                        }
                    }
                }
            }

            // Batch writes
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
        self.get_mask4()
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
