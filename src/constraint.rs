#![allow(clippy::too_many_arguments)]

use crate::datastructures::hybrid_bitset::RangeSet;
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    fmt::{self, Debug, Display, Formatter},
    sync::{Arc, OnceLock},
};
use std::collections::BTreeMap as StdMap;

use bimap::BiBTreeMap;
use json_convertible_derive::JSONConvertible;
use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use profiler_macro::{time_it, timeit};

use crate::{
    datastructures::{
        leveled_gss::{LeveledGSS, Merge},
        vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode},
    },
    equivalence_analysis::{
        compute_combined_equivalence,
        VocabEquivalenceResult,
    },
    dfa_u8::{Regex, Tokenizer},
    glr::{
        analyze::{
            compute_always_allowed_terminal_follows, compute_ever_allowed_terminal_follows,
            compute_always_follow_sets,
            compute_self_extending_terminals,
        },
        grammar::Terminal,
        parser::{GLRParser, GLRParserState},
    },
    interface::{CompiledGrammar, GrammarDefinition},
    json_serialization::{JSONConvertible, JSONNode},
    precompute4::parser_dwa::{build_parser_dwa, ParserDWA},
    r#macro::is_debug_level_enabled,
    dfa_u8::{LLMTokenID, LLMTokenMap, TokenizerStateID},
    types::{TerminalID as GrammarTokenID, TerminalID},
};
use crate::datastructures::bitset::Bitset;
use crate::datastructures::gss_acc::TerminalsDisallowed;
use crate::glr::parser::{ExpectElse, ParseStateEdgeContent};
use crate::glr::table::{get_row, Stage7ShiftsAndReducesLookaheadValue, StateID as TableStateID, NonTerminalID};
use crate::dwa_i32::{DWA, NWA};
use crate::dwa_i32::{RangeSet as WARangeSet, Weight};

pub use crate::constraint_vocab::*;
use crate::constraint_precompute::{
    run_precompute1_with_possible_matches,
};

type GSSNode = LeveledGSS<ParseStateEdgeContent, TerminalsDisallowed>;

// Thread-local storage for verification mode
// ---------------------------------------------------------------------------
// Terminal allowance mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, JSONConvertible)]
#[json_convertible(rename_all = "snake_case")]
pub enum TerminalAllowanceCheckMode {
    None,
    ImmediateSets,
    ImmediateProbe,
    #[default]
    StepProbe,
}


fn profile_terminal_automata_dump_enabled() -> bool {
    match std::env::var("PROFILE_TERMINAL_DWA") {
        Ok(value) => {
            let normalized = value.trim();
            normalized.is_empty()
                || !(normalized == "0"
                    || normalized.eq_ignore_ascii_case("false")
                    || normalized.eq_ignore_ascii_case("off"))
        }
        Err(_) => false,
    }
}

fn format_terminal_for_debug(term: &Terminal) -> String {
    match term {
        Terminal::RegexName(name) => name.clone(),
        Terminal::Literal(bytes) => {
            if bytes.iter().all(|b| *b >= 0x20 && *b < 0x7f) {
                format!("'{}'", String::from_utf8_lossy(bytes))
            } else {
                format!(
                    "Lit[{}]",
                    bytes
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect::<Vec<_>>()
                        .join("")
                )
            }
        }
    }
}

fn dump_terminal_id_map(parser: &GLRParser) {
    println!("\n--- Terminal ID Map ---");
    let mut entries: Vec<(usize, String)> = parser
        .terminal_map
        .iter()
        .map(|(term, tid)| (tid.0, format_terminal_for_debug(term)))
        .collect();
    entries.sort_by_key(|(tid, _)| *tid);
    for (tid, name) in entries {
        println!("  {} -> {}", tid, name);
    }
}

/// Compute the token partition that optimize_dwa_and_vocab would produce,
/// without actually modifying anything.
fn compute_dwa_partition(
    dwa: &DWA,
    possible_matches: &BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    max_tok: usize,
) -> Vec<Vec<usize>> {
    // Collect unique weights
    let mut unique_weights = HashSet::with_capacity(dwa.states.0.len() * 3);
    let mut dwa_weights = HashSet::new();
    for state in &dwa.states.0 {
        if let Some(w) = &state.final_weight { unique_weights.insert(w.clone()); dwa_weights.insert(w.clone()); }
        for w in state.trans_weights.values() { unique_weights.insert(w.clone()); dwa_weights.insert(w.clone()); }
    }
    let mut possible_match_weights = HashSet::new();
    for inner_map in possible_matches.values() {
        for bv in inner_map.values() {
            let w = Weight::from(bv.clone());
            unique_weights.insert(w.clone());
            possible_match_weights.insert(w);
        }
    }
    
    crate::debug!(2, "DWA partition: {} unique weights from DWA, {} from possible_matches", 
                 dwa_weights.len(), possible_match_weights.len());
    
    // Debug: find weights that separate token 6 and 31
    let tok_a = 6usize;
    let tok_b = 31usize;
    let mut separating_dwa = 0;
    let mut separating_pm = 0;
    for w in &unique_weights {
        let has_a = w.contains(tok_a);
        let has_b = w.contains(tok_b);
        if has_a != has_b {
            let in_dwa = dwa_weights.contains(w);
            let in_pm = possible_match_weights.contains(w);
            if in_dwa { separating_dwa += 1; }
            if in_pm { separating_pm += 1; }
        }
    }
    if separating_dwa > 0 || separating_pm > 0 {
        crate::debug!(7, "Weights separating tokens {} and {}: {} from DWA, {} from possible_matches",
            tok_a, tok_b, separating_dwa, separating_pm);
    }

    // Partition tokens
    let mut token_to_class: Vec<usize> = vec![0; max_tok + 1];
    let mut class_to_tokens: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    class_to_tokens.insert(0, (0..=max_tok).collect());
    let mut num_classes = 1;

    let weights_vec: Vec<&Weight> = unique_weights.iter().filter(|w| !w.is_all_fast()).collect();

    for w in weights_vec.iter() {
        let mut tokens_in_w_by_class: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        for t in w.iter_up_to(max_tok) {
            if t <= max_tok {
                tokens_in_w_by_class.entry(token_to_class[t]).or_default().push(t);
            }
        }
        for (old_cid, present_tokens) in tokens_in_w_by_class {
            let old_group = class_to_tokens.get_mut(&old_cid).unwrap();
            if present_tokens.len() < old_group.len() {
                let new_cid = num_classes;
                num_classes += 1;
                let present_set: FxHashSet<usize> = present_tokens.iter().cloned().collect();
                old_group.retain(|t| !present_set.contains(t));
                for &t in &present_tokens { token_to_class[t] = new_cid; }
                class_to_tokens.insert(new_cid, present_tokens);
            }
        }
    }

    class_to_tokens.into_values().collect()
}

fn optimize_dwa_and_vocab(
    _dwa: &mut DWA,
    _vocab: &mut StageVocab,
    _possible_matches: &mut BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
) {
    // TODO: Re-enable once AbstractWeight transition is complete
    // Body commented out as part of the Weight -> AbstractWeight migration.
    // The function relies on Weight::is_all_fast() and .rsb field access
    // which need to be updated for the new abstraction.
    crate::debug!(4, "DWA Vocab Optimization: Disabled (AbstractWeight migration)");
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GrammarConstraintConfig {
}

impl Default for GrammarConstraintConfig {
    fn default() -> Self {
        Self {
        }
    }
}

impl GrammarConstraintConfig {
    pub fn off() -> Self { Self::default() }
}

const EOS_TOKEN_BYTES: &[u8] = b"<|endoftext|>";

fn find_eos_token_id(vocab_trie: &LLMVocabTrie) -> Option<usize> {
    vocab_trie
        .iter()
        .find_map(|(id, bytes)| if bytes == EOS_TOKEN_BYTES { Some(id) } else { None })
}

// ---------------------------------------------------------------------------
// Main structure
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GrammarConstraint {
    pub tokenizer: Tokenizer,
    pub parser: GLRParser,

    /// The Parser DWA - the core precomputed artifact for O(1) mask queries.
    /// 
    /// This deterministic weighted automaton encodes how grammar terminals
    /// interact with parse stacks, with weights being sparse bitvectors
    /// over LLM token equivalence classes.
    pub parser_dwa: ParserDWA,

    /// LLM vocabulary stored as a trie for efficient lookup and compact serialization.
    pub vocab_trie: Arc<LLMVocabTrie>,
    
    /// Legacy field - kept for backward compatibility during migration.
    /// Will be removed in a future version.
    #[deprecated(note = "Use vocab_trie instead")]
    pub commit_vocab: Arc<CommitVocab>,
    pub(crate) token_name_map: BiBTreeMap<Terminal, usize>,

    /// Tokenizer state -> grammar terminal -> internal LLM token bitset.
    pub possible_matches: BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,

    #[serde(skip, default)]
    pub terminal_to_greedy_group: Vec<Option<usize>>,

    /// Vocabulary mappings for the Parser DWA stage.
    pub parser_dwa_vocab: StageVocab,

    /// Optional EOS token id derived from the vocab bytes.
    pub eos_token_id: Option<usize>,
    
    /// Number of tokenizer states (M in weight-heavy encoding).
    /// When representative compression is enabled, this becomes the number of rep states.
    /// When > 0, indicates weight-heavy mode where weights are in N×M space.
    /// When 0, indicates symbol-heavy mode (tsid as initial transition labels).
    pub num_tsids: usize,

    /// Maps raw tokenizer state ID → internal tsid index (0..num_tsids-1).
    /// In weight-heavy mode, this compresses the tokenizer state space to only
    /// representative states and (optionally) applies a reordering permutation
    /// to minimize range counts in DWA weights.
    /// Empty in symbol-heavy mode.
    #[serde(default)]
    pub state_to_internal_tsid: Vec<usize>,

    /// Cached per-tsid masks for weight-heavy mode.
    /// Indexed by internal_tsid (0..num_tsids).
    /// Each mask has points: tsid + n*M for n in 0..=max_llm_token.
    /// Initialized lazily on first use.
    #[serde(skip)]
    pub(crate) cached_tsid_masks: OnceLock<Vec<crate::dwa_i32::common::Weight>>,
}

impl GrammarConstraint {
    /// Get the cached tsid mask for a given internal_tsid.
    /// The mask has points: tsid + n * num_tsids for n in 0..=max_llm_token.
    /// When terminals_disallowed is empty, the seed phase result is exactly this mask
    /// (since expand(full_vocab) covers all N×M space, intersect with tsid_mask = tsid_mask).
    pub(crate) fn get_tsid_mask(&self, internal_tsid: usize) -> &crate::dwa_i32::common::Weight {
        let masks = self.cached_tsid_masks.get_or_init(|| {
            let max_llm_token = self.parser_dwa_vocab.internal_max_llm_token;
            let num_tsids = self.num_tsids;
            (0..num_tsids)
                .map(|tsid| {
                    crate::dwa_i32::weight_expansion::create_tsid_mask(tsid, num_tsids, max_llm_token)
                })
                .collect()
        });
        &masks[internal_tsid]
    }

    /// Save GrammarConstraint to a binary cache file using bincode + LZ4 compression.
    pub fn save_to_cache(&self, path: &std::path::Path) -> Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| format!("bincode serialize error: {}", e))?;
        let compressed = lz4_flex::compress_prepend_size(&bytes);
        std::fs::write(path, &compressed).map_err(|e| format!("write error: {}", e))?;
        Ok(())
    }

    /// Load GrammarConstraint from a binary cache file.
    pub fn load_from_cache(path: &std::path::Path) -> Result<Self, String> {
        let compressed = std::fs::read(path).map_err(|e| format!("read error: {}", e))?;
        let bytes = lz4_flex::decompress_size_prepended(&compressed).map_err(|e| format!("lz4 decompress error: {}", e))?;
        bincode::deserialize(&bytes).map_err(|e| format!("bincode deserialize error: {}", e))
    }

    /// Backward compatibility accessor for precomputed4
    #[deprecated(since = "0.3.0", note = "Use parser_dwa instead")]
    pub fn precomputed4(&self) -> &ParserDWA {
        &self.parser_dwa
    }
    
    /// Backward compatibility accessor for precompute4_vocab
    #[deprecated(since = "0.3.0", note = "Use parser_dwa_vocab instead")]
    pub fn precompute4_vocab(&self) -> &StageVocab {
        &self.parser_dwa_vocab
    }

    pub fn assert_eq(&self, other: &Self) {
        assert_eq!(self.tokenizer, other.tokenizer);
        assert_eq!(self.parser, other.parser);
        // Note: parser_dwa is skipped as it may differ due to runtime computation
        assert_eq!(self.token_name_map, other.token_name_map);
        assert_eq!(self.possible_matches, other.possible_matches);
        assert_eq!(self.parser_dwa_vocab, other.parser_dwa_vocab);
        assert_eq!(self.vocab_trie, other.vocab_trie);
        assert_eq!(self.eos_token_id, other.eos_token_id);
        assert_eq!(self.num_tsids, other.num_tsids);
    }
    
    /// Get the weight dimensions for this constraint.
    /// 
    /// In weight-heavy mode: `num_tokens × num_tsids` (full 2D space)
    /// In symbol-heavy mode: `num_tokens × 1` (degenerate 2D space)
    pub fn weight_dimensions(&self) -> crate::dwa_i32::heavy_weight::WeightDimensions {
        let num_tokens = self.parser_dwa_vocab.internal_max_llm_token + 1;
        let num_tsids = if self.num_tsids > 0 { self.num_tsids } else { 1 };
        crate::dwa_i32::heavy_weight::WeightDimensions::new(num_tokens, num_tsids)
    }
}

// ---------------------------------------------------------------------------
// Intermediate JSON types for GrammarConstraint serialization
// ---------------------------------------------------------------------------

/// Pooled representation of possible_matches for efficient serialization.
/// Instead of storing the full bitset for each (state, terminal) pair,
/// we store an index into a shared pool of unique bitsets.
#[derive(Debug, Clone, JSONConvertible)]
struct PossibleMatchesJSON {
    /// Pool of unique bitsets
    matches_pool: Vec<LLMTokenBV>,
    /// state_id (as string) -> terminal_id (as string) -> pool index
    state_terminal_indices: BTreeMap<String, BTreeMap<String, usize>>,
}

impl PossibleMatchesJSON {
    fn from_possible_matches(pm: &BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>) -> Self {
        let mut bitset_pool: Vec<LLMTokenBV> = Vec::new();
        let mut bitset_map: BTreeMap<LLMTokenBV, usize> = BTreeMap::new();
        let mut state_terminal_indices: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();

        for (state_id, inner) in pm {
            let mut new_inner = BTreeMap::new();
            for (term_id, bv) in inner {
                let idx = if let Some(&i) = bitset_map.get(bv) {
                    i
                } else {
                    let i = bitset_pool.len();
                    bitset_pool.push(bv.clone());
                    bitset_map.insert(bv.clone(), i);
                    i
                };
                new_inner.insert(term_id.0.to_string(), idx);
            }
            state_terminal_indices.insert(state_id.0.to_string(), new_inner);
        }

        PossibleMatchesJSON {
            matches_pool: bitset_pool,
            state_terminal_indices,
        }
    }

    fn to_possible_matches(self) -> Result<BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>, String> {
        let mut possible_matches = BTreeMap::new();
        for (sid_str, inner_map) in self.state_terminal_indices {
            let sid: usize = sid_str.parse().map_err(|_| "Invalid state ID key")?;
            let mut inner = BTreeMap::new();
            for (tid_str, idx) in inner_map {
                let tid: usize = tid_str.parse().map_err(|_| "Invalid terminal ID key")?;
                let bv = self.matches_pool.get(idx).ok_or("Pool index out of bounds")?.clone();
                inner.insert(TerminalID(tid), bv);
            }
            possible_matches.insert(TokenizerStateID(sid), inner);
        }
        Ok(possible_matches)
    }
}

/// Intermediate JSON representation of GrammarConstraint.
/// Uses struct field names for clear serialization.
#[derive(Debug, Clone)]
struct GrammarConstraintJSON {
    tokenizer_dfa: crate::finite_automata::DFA,
    dwa: DWA,
    vocab: StageVocab,
    parser: GLRParser,
    token_name_map: BiBTreeMap<Terminal, usize>,
    possible_matches: PossibleMatchesJSON,
    /// New trie-based vocab format (preferred)
    vocab_trie: Option<LLMVocabTrie>,
    /// Legacy commit_vocab format (for backward compatibility)
    commit_vocab: Option<CommitVocab>,
    /// Full original LLM vocab (optional for backward compat)
    original_llm_vocab: Option<LLMVocab>,
    /// Fallback: just max_original_llm_token_id if full vocab not available
    max_orig_id: Option<usize>,
    /// Number of tsids for weight-heavy mode (0 = symbol-heavy)
    num_tsids: usize,
    /// Mapping from raw tokenizer state ID to internal tsid index
    state_to_internal_tsid: Vec<usize>,
}

impl JSONConvertible for GrammarConstraintJSON {
    fn to_json(&self) -> JSONNode {
        let mut obj = std::collections::BTreeMap::new();
        obj.insert("tokenizer_dfa".to_string(), self.tokenizer_dfa.to_json());
        obj.insert("dwa".to_string(), self.dwa.to_json());
        obj.insert("vocab".to_string(), self.vocab.to_json());
        obj.insert("parser".to_string(), self.parser.to_json());
        obj.insert("token_name_map".to_string(), self.token_name_map.to_json());
        obj.insert("possible_matches".to_string(), self.possible_matches.to_json());
        
        // Only serialize non-None optional fields
        if let Some(ref trie) = self.vocab_trie {
            obj.insert("vocab_trie".to_string(), trie.to_json());
        }
        if let Some(ref cv) = self.commit_vocab {
            obj.insert("commit_vocab".to_string(), cv.to_json());
        }
        if let Some(ref llm_vocab) = self.original_llm_vocab {
            obj.insert("original_llm_vocab".to_string(), llm_vocab.to_json());
        }
        if let Some(max_id) = self.max_orig_id {
            obj.insert("max_orig_id".to_string(), JSONNode::UInt(max_id as u128));
        }
        
        // num_tsids (0 = symbol-heavy, > 0 = weight-heavy)
        if self.num_tsids > 0 {
            obj.insert("num_tsids".to_string(), JSONNode::UInt(self.num_tsids as u128));
        }

        // state_to_internal_tsid
        if !self.state_to_internal_tsid.is_empty() {
            let arr: Vec<JSONNode> = self.state_to_internal_tsid.iter()
                .map(|&v| JSONNode::UInt(v as u128))
                .collect();
            obj.insert("state_to_internal_tsid".to_string(), JSONNode::Array(arr));
        }

        JSONNode::Object(obj)
    }
    
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(obj) => {
                Ok(Self {
                    tokenizer_dfa: JSONConvertible::from_json(
                        obj.get("tokenizer_dfa").ok_or("Missing tokenizer_dfa")?.clone()
                    )?,
                    dwa: JSONConvertible::from_json(
                        obj.get("dwa").ok_or("Missing dwa")?.clone()
                    )?,
                    vocab: JSONConvertible::from_json(
                        obj.get("vocab").ok_or("Missing vocab")?.clone()
                    )?,
                    parser: JSONConvertible::from_json(
                        obj.get("parser").ok_or("Missing parser")?.clone()
                    )?,
                    token_name_map: JSONConvertible::from_json(
                        obj.get("token_name_map").ok_or("Missing token_name_map")?.clone()
                    )?,
                    possible_matches: JSONConvertible::from_json(
                        obj.get("possible_matches").ok_or("Missing possible_matches")?.clone()
                    )?,
                    // Optional fields
                    vocab_trie: obj.get("vocab_trie")
                        .map(|n| LLMVocabTrie::from_json(n.clone()))
                        .transpose()?,
                    commit_vocab: obj.get("commit_vocab")
                        .filter(|n| !matches!(n, JSONNode::Null))
                        .map(|n| CommitVocab::from_json(n.clone()))
                        .transpose()?,
                    original_llm_vocab: obj.get("original_llm_vocab")
                        .filter(|n| !matches!(n, JSONNode::Null))
                        .map(|n| LLMVocab::from_json(n.clone()))
                        .transpose()?,
                    max_orig_id: obj.get("max_orig_id")
                        .and_then(|n| match n {
                            JSONNode::UInt(v) => Some(*v as usize),
                            JSONNode::Int(v) => Some(*v as usize),
                            _ => None,
                        }),
                    num_tsids: obj.get("num_tsids")
                        .and_then(|n| match n {
                            JSONNode::UInt(v) => Some(*v as usize),
                            JSONNode::Int(v) => Some(*v as usize),
                            _ => None,
                        })
                        .unwrap_or(0),
                    state_to_internal_tsid: obj.get("state_to_internal_tsid")
                        .and_then(|n| match n {
                            JSONNode::Array(arr) => {
                                let vals: Result<Vec<usize>, _> = arr.iter().map(|item| match item {
                                    JSONNode::UInt(v) => Ok(*v as usize),
                                    JSONNode::Int(v) => Ok(*v as usize),
                                    _ => Err("Expected integer in state_to_internal_tsid"),
                                }).collect();
                                vals.ok()
                            }
                            _ => None,
                        })
                        .unwrap_or_default(),
                })
            }
            _ => Err("Expected object for GrammarConstraintJSON".to_string()),
        }
    }
}

impl JSONConvertible for GrammarConstraint {
    fn to_json(&self) -> JSONNode {
        let mut dwa = self.parser_dwa.clone();

        let intermediate = GrammarConstraintJSON {
            tokenizer_dfa: self.tokenizer.dfa().clone(),
            dwa,
            vocab: self.parser_dwa_vocab.clone(),
            parser: self.parser.clone(),
            token_name_map: self.token_name_map.clone(),
            possible_matches: PossibleMatchesJSON::from_possible_matches(&self.possible_matches),
            // Serialize the new trie format
            vocab_trie: Some((*self.vocab_trie).clone()),
            // Don't serialize the legacy format anymore
            commit_vocab: None,
            original_llm_vocab: None,
            max_orig_id: Some(self.parser_dwa_vocab.max_original_llm_token_id),
            num_tsids: self.num_tsids,
            state_to_internal_tsid: self.state_to_internal_tsid.clone(),
        };
        intermediate.to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let intermediate = GrammarConstraintJSON::from_json(node)?;

        let tokenizer = Tokenizer::new(Regex { dfa: intermediate.tokenizer_dfa });
        let possible_matches = intermediate.possible_matches.to_possible_matches()?;

        // Load vocab_trie, with fallback to legacy formats
        let vocab_trie = if let Some(trie) = intermediate.vocab_trie {
            // New trie format
            Arc::new(trie)
        } else if let Some(ref cv) = intermediate.commit_vocab {
            // Convert from legacy commit_vocab format
            Arc::new(LLMVocabTrie::from_commit_vocab(cv))
        } else if let Some(ref legacy_vocab) = intermediate.original_llm_vocab {
            // Convert from very old full vocab format
            Arc::new(LLMVocabTrie::from_token_map(&legacy_vocab.llm_token_map))
        } else {
            // Empty vocab fallback
            let max_orig_id = intermediate.max_orig_id.unwrap_or(0);
            Arc::new(LLMVocabTrie::empty(max_orig_id))
        };
        
        // Build legacy commit_vocab for backward compatibility
        // This can be removed once all code is migrated to use vocab_trie
        // Note: Commit equivalence analysis has been removed - just use stored data or empty fallback
        #[allow(deprecated)]
        let commit_vocab = if let Some(cv) = intermediate.commit_vocab {
            Arc::new(cv)
        } else {
            let max_orig_id = intermediate.max_orig_id.unwrap_or(0);
            Arc::new(CommitVocab::new(
                Vec::new(),
                vec![CommitVocab::INVALID_REPRESENTATIVE; max_orig_id + 1],
            ))
        };

        // Set global dimensions when deserializing
        let num_tsids_for_dims = if intermediate.num_tsids > 0 { intermediate.num_tsids } else { 1 };
        crate::datastructures::set_global_dims_all_threads(
            intermediate.vocab.internal_max_llm_token,
            num_tsids_for_dims,
        );

        let eos_token_id = find_eos_token_id(&vocab_trie);

        #[allow(deprecated)]
        Ok(GrammarConstraint {
            tokenizer,
            parser: intermediate.parser,
            parser_dwa: intermediate.dwa,
            vocab_trie,
            commit_vocab,
            token_name_map: intermediate.token_name_map,
            possible_matches,
            terminal_to_greedy_group: Vec::new(),
            parser_dwa_vocab: intermediate.vocab,
            eos_token_id,
            num_tsids: intermediate.num_tsids,
            state_to_internal_tsid: intermediate.state_to_internal_tsid,
            cached_tsid_masks: OnceLock::new(),
        })
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
        Self::build_with_config_and_grammar(
            compiled_grammar.tokenizer,
            compiled_grammar.glr_parser,
            llm_token_map,
            token_name_map,
            max_original_llm_token_id,
            config,
            compiled_grammar.definition,
        )
    }

    pub fn new(
        tokenizer: Tokenizer,
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
        tokenizer: Tokenizer,
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
    pub fn new_from_grammar_definition(
        grammar_definition: Arc<GrammarDefinition>,
        llm_token_map: LLMTokenMap,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
    ) -> Self {
        let compiled_grammar = CompiledGrammar::from_definition(grammar_definition.clone());
        
        Self::from_compiled_grammar_with_config(
            compiled_grammar,
            llm_token_map,
            max_original_llm_token_id,
            config,
        )
    }

    /// Combined setup that computes both internal mappings and commit vocab in a single pass.
    /// Also computes state equivalence analysis once for all states.
    /// Returns: (original_to_internal_map, commit_vocab, internal_llm_token_map, mask_classes, state_to_rep, representative_states)
    fn setup_combined(
        llm_token_map: &LLMTokenMap,
        tokenizer: &Tokenizer,
        max_original_llm_token_id: usize,
        grammar_group_ids: &std::collections::BTreeSet<usize>,
        suffix_group_mask: Option<&[bool]>,
        ever_allowed_by_group: Option<&[Vec<bool>]>,
        group_to_class: Option<&[usize]>,
    ) -> (
        BTreeMap<usize, usize>,
        CommitVocab,
        BTreeMap<Vec<u8>, LLMTokenID>,
        VocabEquivalenceResult,
        BTreeMap<TokenizerStateID, TokenizerStateID>,
        Vec<usize>,
    ) {
        if llm_token_map.is_empty() {
            return (
                BTreeMap::new(),
                CommitVocab::new(Vec::new(), Vec::new()),
                BTreeMap::new(),
                Default::default(),
                BTreeMap::new(),
                Vec::new(),
            );
        }

        let profile_setup_combined = std::env::var("PROFILE_BUILD_TOKENIZER").is_ok()
            || std::env::var("PROFILE_SETUP_COMBINED").is_ok();
        let setup_combined_start = std::time::Instant::now();
        let mut timing_sort_tokens = std::time::Duration::ZERO;
        let mut timing_collect_strings = std::time::Duration::ZERO;
        let mut timing_all_states = std::time::Duration::ZERO;
        let mut timing_compute_equivalence = std::time::Duration::ZERO;
        let mut timing_build_state_to_rep = std::time::Duration::ZERO;
        let mut timing_filter_states = std::time::Duration::ZERO;
        let mut timing_build_original_to_internal = std::time::Duration::ZERO;
        let mut timing_commit_vocab_prep = std::time::Duration::ZERO;
        let mut timing_build_internal_llm_token_map = std::time::Duration::ZERO;
        let mut timing_commit_vocab = std::time::Duration::ZERO;

        // Sort tokens by bytes for consistent ordering
        let mut sorted_tokens: Vec<_> = llm_token_map.iter().collect();
        let sort_start = std::time::Instant::now();
        timeit!("setup_combined::sort_tokens", {
            sorted_tokens.sort_by_key(|(bytes, _id)| *bytes);
        });
        timing_sort_tokens = sort_start.elapsed();

        let mut llm_token_strings: Vec<Vec<u8>> = Vec::with_capacity(sorted_tokens.len());
        let mut original_ids: Vec<usize> = Vec::with_capacity(sorted_tokens.len());
        let mut highest_original_id = 0usize;

        let collect_start = std::time::Instant::now();
        timeit!("setup_combined::collect_token_strings", {
            for (bytes, id) in &sorted_tokens {
                highest_original_id = highest_original_id.max(id.0);
                llm_token_strings.push((*bytes).clone());
                original_ids.push(id.0);
            }
        });
        timing_collect_strings = collect_start.elapsed();

        // Get ALL states for equivalence analysis
        let all_states_start = std::time::Instant::now();
        let all_states: Vec<usize> = tokenizer.iter_states().map(|s| s.0).collect();
        timing_all_states = all_states_start.elapsed();
        
        // Use combined equivalence analysis
        // State reduction threshold of 0 means always apply state reduction
        let compute_equivalence_start = std::time::Instant::now();
        let combined_result = timeit!("setup_combined::compute_equivalence", {
            compute_combined_equivalence(
                tokenizer,
                &llm_token_strings,
                &all_states,
                suffix_group_mask,
                ever_allowed_by_group,
                group_to_class,
            )
        });
        timing_compute_equivalence = compute_equivalence_start.elapsed();
        
        // Derive state_to_rep and representative_states from state_classes
        let mut state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = BTreeMap::new();
        let mut representative_set: BTreeSet<usize> = BTreeSet::new();
        
        let build_state_to_rep_start = std::time::Instant::now();
        timeit!("setup_combined::build_state_to_rep", {
            for class in &combined_result.state_classes {
                // Pick the first (smallest) state as representative
                if let Some(&rep) = class.iter().next() {
                    representative_set.insert(rep);
                    for &state in class {
                        state_to_rep.insert(TokenizerStateID(state), TokenizerStateID(rep));
                    }
                }
            }
        });
        timing_build_state_to_rep = build_state_to_rep_start.elapsed();
        
        let representative_states: Vec<usize> = representative_set.into_iter().collect();

        // Filter states for vocab equivalence to only grammar-relevant ones
        // (This filtering was already done inside compute_combined_equivalence? Let's check)
        // Actually the combined analysis used all_states, so we need to filter here for logging
        let filter_states_start = std::time::Instant::now();
        let initial_states_for_vocab: Vec<usize> = if !grammar_group_ids.is_empty() {
            let filtered: Vec<usize> = representative_states
                .iter()
                .copied()
                .filter(|&sid| {
                    let st = &tokenizer.dfa().states[sid];
                    st.finalizers.iter().any(|gid| grammar_group_ids.contains(&gid))
                        || st.possible_future_group_ids.iter().any(|gid| grammar_group_ids.contains(&gid))
                })
                .collect();

            if !filtered.is_empty() && filtered.len() < representative_states.len() {
                if crate::r#macro::is_debug_level_enabled(3) {
                    crate::debug!(
                        3,
                        "Pruned states for vocab equivalence: {} -> {} (grammar groups {})",
                        representative_states.len(),
                        filtered.len(),
                        grammar_group_ids.len()
                    );
                }
            }
            filtered
        } else {
            representative_states.clone()
        };
        timing_filter_states = filter_states_start.elapsed();

        crate::debug!(
            3,
            "Vocab equivalence analysis: {} initial states, {} tokens",
            initial_states_for_vocab.len(),
            llm_token_strings.len()
        );

        let mask_classes = combined_result.vocab_classes;

        if crate::r#macro::is_debug_level_enabled(3) {
            let num_original_tokens = llm_token_strings.len();
            crate::debug!(
                3,
                "Combined Equivalence Analysis: {} tokens -> {} mask classes",
                num_original_tokens,
                mask_classes.len(),
            );
        }

        // Build original_to_internal map AND track best representative per class (combined)
        // Use Vec for O(1) access instead of BTreeMap O(log n)
        let build_original_to_internal_start = std::time::Instant::now();
        let mut original_to_internal_vec: Vec<usize> = vec![usize::MAX; highest_original_id + 1];
        let mut best_rep_by_internal: Vec<usize> = Vec::with_capacity(mask_classes.len());
        let mut internal_id_counter = 0;
        timeit!("setup_combined::build_original_to_internal", {
            for string_indices in &mask_classes {
                if string_indices.is_empty() {
                    continue;
                }
                let internal_id = internal_id_counter;
                internal_id_counter += 1;
                // Find shortest representative while iterating
                let best_idx = *string_indices
                    .iter()
                    .min_by_key(|&&idx| (llm_token_strings[idx].len(), &llm_token_strings[idx]))
                    .unwrap();
                best_rep_by_internal.push(best_idx);
                for &string_index in string_indices {
                    let original_llm_id = original_ids[string_index];
                    original_to_internal_vec[original_llm_id] = internal_id;
                }
            }
        });
        // Convert to BTreeMap for compatibility with rest of code
        let original_to_internal_map: BTreeMap<usize, usize> = original_to_internal_vec
            .into_iter()
            .enumerate()
            .filter(|&(_, v)| v != usize::MAX)
            .collect();
        timing_build_original_to_internal = build_original_to_internal_start.elapsed();

        // TEMP: disable commit vocab optimization
        let commit_vocab_prep_start = std::time::Instant::now();
        let representatives: Vec<Vec<u8>> = (0..llm_token_strings.len()).map(|i| llm_token_strings[i].clone()).collect();
        let original_to_representative = (0..llm_token_strings.len()).map(|i| i as u32).collect();
        timing_commit_vocab_prep = commit_vocab_prep_start.elapsed();

        // Build internal_llm_token_map using best representatives we already computed
        // This avoids iterating 50K tokens again!
        let build_internal_map_start = std::time::Instant::now();
        let internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = timeit!(
            "setup_combined::build_internal_llm_token_map",
            {
                best_rep_by_internal
                    .into_iter()
                    .enumerate()
                    .map(|(internal_id, string_idx)| {
                        (llm_token_strings[string_idx].clone(), LLMTokenID(internal_id))
                    })
                    .collect()
            }
        );
        timing_build_internal_llm_token_map = build_internal_map_start.elapsed();
        
        crate::debug!(
            4,
            "internal_llm_token_map has {} representative entries (was {} total) - built in combined pass",
            internal_llm_token_map.len(),
            llm_token_strings.len()
        );

        let commit_vocab_start = std::time::Instant::now();
        let commit_vocab = CommitVocab::new(representatives, original_to_representative);
        timing_commit_vocab = commit_vocab_start.elapsed();

        if profile_setup_combined {
            let total = setup_combined_start.elapsed();
            let sum = timing_sort_tokens
                + timing_collect_strings
                + timing_all_states
                + timing_compute_equivalence
                + timing_build_state_to_rep
                + timing_filter_states
                + timing_build_original_to_internal
                + timing_commit_vocab_prep
                + timing_build_internal_llm_token_map
                + timing_commit_vocab;
            let other = total.saturating_sub(sum);

            crate::timing!("TIMING: setup_combined::sort_tokens {:?}", timing_sort_tokens);
            crate::timing!("TIMING: setup_combined::collect_token_strings {:?}", timing_collect_strings);
            crate::timing!("TIMING: setup_combined::all_states {:?}", timing_all_states);
            crate::timing!("TIMING: setup_combined::compute_equivalence {:?}", timing_compute_equivalence);
            crate::timing!("TIMING: setup_combined::build_state_to_rep {:?}", timing_build_state_to_rep);
            crate::timing!("TIMING: setup_combined::filter_states {:?}", timing_filter_states);
            crate::timing!("TIMING: setup_combined::build_original_to_internal {:?}", timing_build_original_to_internal);
            crate::timing!("TIMING: setup_combined::commit_vocab_prep {:?}", timing_commit_vocab_prep);
            crate::timing!(
                "TIMING: setup_combined::build_internal_llm_token_map {:?}",
                timing_build_internal_llm_token_map
            );
            crate::timing!("TIMING: setup_combined::commit_vocab {:?}", timing_commit_vocab);
            crate::timing!("TIMING: setup_combined::other {:?}", other);
            crate::timing!("TIMING: setup_combined::total {:?}", total);
        }
        (original_to_internal_map, commit_vocab, internal_llm_token_map, mask_classes, state_to_rep, representative_states)
    }

    /// Reorder tokenizer DFA states so equivalence classes are contiguous, keeping start state 0.
    /// Returns updated (state_to_rep, representative_states).
    fn reorder_tokenizer_states(
        tokenizer: &mut Tokenizer,
        state_to_rep: &BTreeMap<TokenizerStateID, TokenizerStateID>,
        representative_states: &[usize],
    ) -> (BTreeMap<TokenizerStateID, TokenizerStateID>, Vec<usize>) {
        let num_states = tokenizer.max_state();
        if num_states == 0 {
            return (state_to_rep.clone(), representative_states.to_vec());
        }

        let start_state = tokenizer.initial_state_id().0;
        let start_rep = state_to_rep
            .get(&TokenizerStateID(start_state))
            .copied()
            .unwrap_or(TokenizerStateID(start_state))
            .0;

        let mut rep_to_states: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (state, rep) in state_to_rep {
            rep_to_states.entry(rep.0).or_default().push(state.0);
        }

        let mut rep_keys: Vec<usize> = rep_to_states.keys().copied().collect();
        rep_keys.sort_unstable();
        if let Some(pos) = rep_keys.iter().position(|&r| r == start_rep) {
            rep_keys.swap(0, pos);
        }

        let mut old_to_new = vec![usize::MAX; num_states];
        let mut next = 0usize;
        for rep in rep_keys {
            let mut states = rep_to_states.remove(&rep).unwrap_or_default();
            states.sort_unstable();
            if rep == start_rep {
                if let Some(pos) = states.iter().position(|&s| s == start_state) {
                    states.swap(0, pos);
                }
            }
            for s in states {
                if s < old_to_new.len() {
                    old_to_new[s] = next;
                    next += 1;
                }
            }
        }

        if next != num_states || old_to_new.iter().any(|&v| v == usize::MAX) {
            crate::debug!(2, "State reorder incomplete (assigned {} of {}); falling back to identity", next, num_states);
            old_to_new = (0..num_states).collect();
        }

        if old_to_new[start_state] != 0 {
            crate::debug!(2, "State reorder did not keep start state at 0 ({}); falling back to identity", old_to_new[start_state]);
            old_to_new = (0..num_states).collect();
        }

        tokenizer.reorder_states(&old_to_new);

        let mut new_state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = BTreeMap::new();
        for (state, rep) in state_to_rep {
            let new_state = TokenizerStateID(old_to_new[state.0]);
            let new_rep = TokenizerStateID(old_to_new[rep.0]);
            new_state_to_rep.insert(new_state, new_rep);
        }

        let mut new_representative_states: Vec<usize> = representative_states
            .iter()
            .map(|s| old_to_new[*s])
            .collect();
        new_representative_states.sort_unstable();
        new_representative_states.dedup();

        (new_state_to_rep, new_representative_states)
    }

    #[time_it("GrammarConstraint::build_with_config")]
    fn build_with_config(
        tokenizer: Tokenizer,
        parser: GLRParser,
        llm_token_map: LLMTokenMap,
        token_name_map: BiBTreeMap<Terminal, usize>,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
    ) -> Self {
        crate::profiler::reset();
        let constraint = Self::build_with_config_inner(
            tokenizer,
            parser,
            llm_token_map,
            token_name_map,
            max_original_llm_token_id,
            config,
            None,
        );
        if crate::r#macro::is_debug_level_enabled(4) {
            let build_total = crate::profiler::sum_subtree_total_time("GrammarConstraint::build_with_config_inner");
            crate::debug!(4, "Profiler build_with_config_inner total: {:.3}s", build_total.as_secs_f64());
            let run_precompute1_total = crate::profiler::sum_subtree_total_time("run_precompute1");
            crate::debug!(4, "Profiler run_precompute1 total: {:.3}s", run_precompute1_total.as_secs_f64());
        }
        if crate::profiler::profiling_enabled() {
            crate::profiler::print_summary();
            let total_own = crate::profiler::sum_flat_own_time();
            println!("Profiler flat own-time total: {:.3}s", total_own.as_secs_f64());
            let run_precompute1_total = crate::profiler::sum_subtree_total_time("run_precompute1");
            println!(
                "Profiler run_precompute1 total: {:.3}s",
                run_precompute1_total.as_secs_f64()
            );
        }
        constraint
    }

    #[time_it("GrammarConstraint::build_with_config_and_grammar")]
    fn build_with_config_and_grammar(
        tokenizer: Tokenizer,
        parser: GLRParser,
        llm_token_map: LLMTokenMap,
        token_name_map: BiBTreeMap<Terminal, usize>,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
        grammar_definition: Arc<GrammarDefinition>,
    ) -> Self {
        crate::profiler::reset();
        let constraint = Self::build_with_config_inner(
            tokenizer,
            parser,
            llm_token_map,
            token_name_map,
            max_original_llm_token_id,
            config,
            Some(grammar_definition),
        );
        if crate::r#macro::is_debug_level_enabled(4) {
            let build_total = crate::profiler::sum_subtree_total_time("GrammarConstraint::build_with_config_inner");
            crate::debug!(4, "Profiler build_with_config_inner total: {:.3}s", build_total.as_secs_f64());
            let run_precompute1_total = crate::profiler::sum_subtree_total_time("run_precompute1");
            crate::debug!(4, "Profiler run_precompute1 total: {:.3}s", run_precompute1_total.as_secs_f64());
        }
        if crate::profiler::profiling_enabled() {
            crate::profiler::print_summary();
            let total_own = crate::profiler::sum_flat_own_time();
            println!("Profiler flat own-time total: {:.3}s", total_own.as_secs_f64());
            let run_precompute1_total = crate::profiler::sum_subtree_total_time("run_precompute1");
            println!(
                "Profiler run_precompute1 total: {:.3}s",
                run_precompute1_total.as_secs_f64()
            );
        }
        constraint
    }

    #[time_it("GrammarConstraint::build_with_config_inner")]
    fn build_with_config_inner(
        tokenizer: Tokenizer,
        parser: GLRParser,
        llm_token_map: LLMTokenMap,
        token_name_map: BiBTreeMap<Terminal, usize>,
        max_original_llm_token_id: usize,
        config: &GrammarConstraintConfig,
        grammar_definition: Option<Arc<GrammarDefinition>>,
    ) -> Self {
        crate::debug!(5, "GrammarConstraint build_with_config_inner: start");
        let mut tokenizer = tokenizer;

        // Epsilon tokens are not supported.
        let epsilon_terminal_group_ids: BTreeSet<_> = tokenizer
            .execute_from_state(&[], tokenizer.initial_state_id())
            .matches
            .iter()
            .map(|token| token.id)
            .collect();
        assert!(
            epsilon_terminal_group_ids.is_empty(),
            "Epsilon tokens are not supported."
        );

        // Collect grammar-relevant group IDs from token_name_map
        let grammar_group_ids: std::collections::BTreeSet<usize> = token_name_map.right_values().copied().collect();
        let verify_equivalence = std::env::var("VERIFY_EQUIVALENCE").is_ok();

        // Compute suffix_group_mask for vocab equivalence analysis: which tokenizer groups
        // can appear in suffix positions (i.e., appear in at least one terminal's follow set).
        // Groups not in any follow set can only be "first" terminals and are irrelevant in suffixes.
        let nwa_follow_prune_disabled = std::env::var("NWA_FOLLOW_PRUNE")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false);

        let suffix_group_mask: Option<Vec<bool>> = {
            if nwa_follow_prune_disabled {
                None
            } else {
                let ever_allowed = compute_ever_allowed_terminal_follows(&parser.productions);
                // Collect all terminals that appear in any follow set (union of all ever_allowed values)
                let mut suffix_relevant_terminals: std::collections::BTreeSet<Terminal> = std::collections::BTreeSet::new();
                for (_term, follows) in &ever_allowed {
                    for follow in follows {
                        suffix_relevant_terminals.insert(follow.clone());
                    }
                }
                // Map to group IDs via token_name_map
                let max_group_id = token_name_map.right_values().copied().max().unwrap_or(0);
                let mut mask = vec![false; max_group_id + 1];
                for terminal in &suffix_relevant_terminals {
                    if let Some(&gid) = token_name_map.get_by_left(terminal) {
                        mask[gid] = true;
                    }
                }
                let num_relevant = mask.iter().filter(|&&b| b).count();
                let total = mask.len();
                crate::debug!(
                    4,
                    "Vocab equiv suffix_group_mask: {}/{} groups relevant in suffix positions",
                    num_relevant,
                    total,
                );
                Some(mask)
            }
        };

        // Compute per-group follow masks for projected suffix hashing.
        // ever_allowed_by_group[g][h] = true means group h can follow group g.
        // Controlled by VOCAB_FOLLOW_PRUNE env var (default: enabled when NWA prune is enabled)
        let vocab_follow_prune_disabled = std::env::var("VOCAB_FOLLOW_PRUNE")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false);
        let ever_allowed_by_group: Option<Vec<Vec<bool>>> = {
            if nwa_follow_prune_disabled || vocab_follow_prune_disabled {
                None
            } else {
                let ever_allowed = compute_ever_allowed_terminal_follows(&parser.productions);
                let max_group_id = token_name_map.right_values().copied().max().unwrap_or(0);
                let num_groups = max_group_id + 1;
                let mut masks: Vec<Vec<bool>> = vec![vec![true; num_groups]; num_groups];
                // For each terminal with follow data, set its mask
                for (term, follows) in &ever_allowed {
                    if let Some(&gid) = token_name_map.get_by_left(term) {
                        // Replace the all-true default with the actual follow set
                        let mask = &mut masks[gid];
                        mask.fill(false);
                        for follow_term in follows {
                            if let Some(&follow_gid) = token_name_map.get_by_left(follow_term) {
                                mask[follow_gid] = true;
                            }
                        }
                    }
                }
                let groups_with_follow: usize = ever_allowed.len();
                crate::debug!(
                    4,
                    "Vocab equiv ever_allowed_by_group: {}/{} groups have follow data",
                    groups_with_follow,
                    num_groups,
                );
                // Diagnostic: follow matrix density analysis
                if std::env::var("ANALYZE_FOLLOW_MATRIX").is_ok() {
                    let mut total_allowed = 0usize;
                    let mut total_cells = 0usize;
                    let mut all_true_rows = 0usize;
                    let mut unique_rows: std::collections::HashSet<Vec<bool>> = std::collections::HashSet::new();
                    for gid in 0..num_groups {
                        let row = &masks[gid];
                        let allowed_count = row.iter().filter(|&&b| b).count();
                        total_allowed += allowed_count;
                        total_cells += num_groups;
                        if allowed_count == num_groups {
                            all_true_rows += 1;
                        }
                        unique_rows.insert(row.clone());
                    }
                    eprintln!("FOLLOW MATRIX ANALYSIS:");
                    eprintln!("  Groups: {}", num_groups);
                    eprintln!("  Groups with follow data: {}", groups_with_follow);
                    eprintln!("  Groups without follow data (all-true rows): {}", all_true_rows);
                    eprintln!("  Density: {}/{} = {:.1}%", total_allowed, total_cells, 100.0 * total_allowed as f64 / total_cells as f64);
                    eprintln!("  Unique rows: {}", unique_rows.len());
                    // Show per-row stats for groups WITH follow data
                    let mut row_sizes: Vec<(usize, usize)> = Vec::new();
                    for gid in 0..num_groups {
                        let row = &masks[gid];
                        let allowed_count = row.iter().filter(|&&b| b).count();
                        if allowed_count < num_groups {
                            row_sizes.push((gid, allowed_count));
                        }
                    }
                    row_sizes.sort_by_key(|&(_, c)| c);
                    eprintln!("  Non-trivial rows (gid, allowed_count):");
                    for &(gid, count) in row_sizes.iter().take(20) {
                        eprintln!("    gid={}: {}/{} ({:.1}%)", gid, count, num_groups, 100.0 * count as f64 / num_groups as f64);
                    }
                    if row_sizes.len() > 20 {
                        eprintln!("    ... and {} more", row_sizes.len() - 20);
                    }
                }
                Some(masks)
            }
        };

        // Compute group equivalence classes from the follow matrix.
        // Two groups are grammar-equivalent if they have the same row AND column
        // in the ever_allowed matrix. Using class IDs instead of raw group IDs
        // in suffix hashing merges tokens that differ only in grammar-equivalent groups.
        let group_to_class: Option<Vec<usize>> = ever_allowed_by_group.as_ref().map(|masks| {
            use std::collections::HashMap;
            let num_groups = masks.len();
            // For each group, build signature = (row, column) in follow matrix
            let mut sig_to_class: HashMap<Vec<u8>, usize> = HashMap::new();
            let mut g2c: Vec<usize> = Vec::with_capacity(num_groups);
            let mut next_class = 0usize;
            for gid in 0..num_groups {
                // Encode signature as packed bits: row bits then column bits
                let row = &masks[gid];
                let total_bits = num_groups * 2;
                let num_bytes = (total_bits + 7) / 8;
                let mut sig = vec![0u8; num_bytes];
                for j in 0..num_groups {
                    if row[j] {
                        sig[j / 8] |= 1 << (j % 8);
                    }
                    // Column: does parent j allow group gid?
                    if masks[j][gid] {
                        let bit = num_groups + j;
                        sig[bit / 8] |= 1 << (bit % 8);
                    }
                }
                let class = *sig_to_class.entry(sig).or_insert_with(|| {
                    let c = next_class;
                    next_class += 1;
                    c
                });
                g2c.push(class);
            }
            crate::debug!(
                2,
                "Group equivalence: {} groups -> {} grammar classes (from follow matrix)",
                num_groups,
                next_class,
            );
            g2c
        });

        // Combined equivalence analysis - computes state equivalence, vocab equivalence, and internal mappings
        // State equivalence is computed ONCE and reused for both vocab analysis and building maps
        crate::debug!(5, "setup_combined: start");
        let setup_start = std::time::Instant::now();
        let (
            original_to_internal_map,
            commit_vocab_data,
            mut internal_llm_token_map,
            mask_classes,
            mut state_to_rep,
            mut representative_states,
        ) = timeit!("setup_combined", {
            Self::setup_combined(
                &llm_token_map,
                &tokenizer,
                max_original_llm_token_id,
                &grammar_group_ids,
                suffix_group_mask.as_deref(),
                ever_allowed_by_group.as_deref(),
                group_to_class.as_deref(),
            )
        });
        crate::timing!("TIMING: setup_combined {:?}", setup_start.elapsed());
        crate::debug!(5, "setup_combined: end");
        let commit_vocab = Arc::new(commit_vocab_data);

        if std::env::var("ENABLE_DFA_STATE_REORDER").map(|v| v == "1").unwrap_or(false) {
            let reorder_start = std::time::Instant::now();
            let (new_state_to_rep, new_representatives) = Self::reorder_tokenizer_states(
                &mut tokenizer,
                &state_to_rep,
                &representative_states,
            );
            state_to_rep = new_state_to_rep;
            representative_states = new_representatives;
            crate::debug!(4, "DFA state reorder complete in {:?}", reorder_start.elapsed());
        } else {
            crate::debug!(4, "DFA state reorder disabled (set ENABLE_DFA_STATE_REORDER=1 to enable)");
        }

        let internal_max_llm_token = original_to_internal_map
            .values()
            .copied()
            .max()
            .unwrap_or(0);

        crate::debug!(4, "Building internal_to_original_map");
        crate::debug!(5, "build_internal_to_original_map: start");
        let internal_map_start = std::time::Instant::now();
        let internal_to_original_map: BTreeMap<usize, LLMTokenBV> = timeit!(
            "build_internal_to_original_map",
            {
                // Optimized: Batch collect then create RangeSets from iterators (faster than individual inserts)
                let mut groups: Vec<Vec<usize>> = vec![Vec::new(); internal_max_llm_token + 1];
                for (orig, int_id) in &original_to_internal_map {
                    groups[*int_id].push(*orig);
                }
                groups
                    .into_iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_empty())
                    .map(|(int_id, origs)| (int_id, LLMTokenBV::from_iter(origs)))
                    .collect()
            }
        );
        crate::timing!(
            "TIMING: build_internal_to_original_map {:?}",
            internal_map_start.elapsed()
        );
        crate::debug!(5, "build_internal_to_original_map: end");

        // internal_llm_token_map was already computed in setup_combined - no need to iterate 50K tokens again!

        // Vocab tree for internal tokens.
        crate::debug!(4, "Building internal vocab prefix tree");
        crate::debug!(5, "build_internal_vocab_tree: start");
        let vocab_tree_start = std::time::Instant::now();
        let internal_tokens_for_vocab: Vec<(usize, Vec<u8>)> =
            internal_llm_token_map.iter().map(|(b, id)| (id.0, b.clone())).collect();
        
        let vocab_tree = timeit!("build_internal_vocab_tree", {
            VocabPrefixTree::build(&internal_tokens_for_vocab)
        });
        crate::timing!(
            "TIMING: build_internal_vocab_tree {:?}",
            vocab_tree_start.elapsed()
        );
        crate::debug!(5, "build_internal_vocab_tree: end");
        crate::debug!(4, "Done building internal vocab prefix tree");

        // State equivalence already computed in setup_combined - reuse it
        crate::debug!(4, "Using precomputed state equivalence: {} representative states", representative_states.len());

        crate::debug!(4, "Computing maps and possible_matches (fast parallel pass)");
        crate::debug!(5, "build_maps_and_possible_matches: start");
        let maps_start = std::time::Instant::now();
        let computed_possible_matches: BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>> =
            timeit!("build_maps_and_possible_matches", {
                // Build group_id -> terminal_index mapping
                // token_name_map maps Terminal -> group_id (from tokenizer regex)
                // parser.terminal_map maps Terminal -> TerminalID (index used in DWA)
                // We need to convert tokenizer group_ids to parser terminal IDs
                let group_id_to_terminal_idx: BTreeMap<usize, usize> = token_name_map
                    .iter()
                    .filter_map(|(terminal, group_id)| {
                        parser.terminal_map.get_by_left(terminal).map(|tid| (*group_id, tid.0))
                    })
                    .collect();
                
                // Only compute for representative states, then expand to non-representatives
                let rep_possible_matches = Self::build_maps_and_matches_for_reps(
                    &tokenizer,
                    &vocab_tree.root,
                    &group_id_to_terminal_idx,
                    &representative_states,
                );
                
                // Expand results to all states via state_to_rep mapping
                let mut computed_possible_matches: BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>> = BTreeMap::new();
                for (s, rep) in &state_to_rep {
                    if let Some(rep_map) = rep_possible_matches.get(rep) {
                        computed_possible_matches.insert(*s, rep_map.clone());
                    }
                }
                computed_possible_matches
            });
        crate::timing!(
            "TIMING: build_maps_and_possible_matches {:?}",
            maps_start.elapsed()
        );
        crate::debug!(5, "build_maps_and_possible_matches: end");

        let allowed_follows_by_label = Arc::new(Vec::new());

        let mut labels: HashSet<crate::dwa_i32::Label> = compute_self_extending_terminals(&parser)
            .into_iter()
            .map(|tid| tid.0 as crate::dwa_i32::Label)
            .collect();
        let always_allowed = compute_always_allowed_terminal_follows(&parser.productions);
        for (term, follows) in &always_allowed {
            crate::debug!(5, "TERMINAL_ALWAYS_ALLOWED_FOLLOW: {:?} -> {:?}", term, follows);
        }
        let mut always_allowed_by_label: Vec<Vec<crate::dwa_i32::Label>> =
            vec![Vec::new(); parser.terminal_map.len()];
        for (term, follows) in &always_allowed {
            if let Some(tid) = parser.terminal_map.get_by_left(term) {
                let entry = &mut always_allowed_by_label[tid.0];
                for follow in follows {
                    if let Some(fid) = parser.terminal_map.get_by_left(follow) {
                        entry.push(fid.0 as crate::dwa_i32::Label);
                    }
                }
            }
        }
        let always_allowed_by_label = Arc::new(always_allowed_by_label);

        // Compute ever-allowed (union of follow sets across all occurrences).
        // The complement gives "never follows" — safe for unconditional pruning.
        let ever_allowed = compute_ever_allowed_terminal_follows(&parser.productions);
        let mut ever_allowed_by_label: Vec<Vec<crate::dwa_i32::Label>> =
            vec![Vec::new(); parser.terminal_map.len()];
        for (term, follows) in &ever_allowed {
            if let Some(tid) = parser.terminal_map.get_by_left(term) {
                let entry = &mut ever_allowed_by_label[tid.0];
                for follow in follows {
                    if let Some(fid) = parser.terminal_map.get_by_left(follow) {
                        entry.push(fid.0 as crate::dwa_i32::Label);
                    }
                }
            }
        }
        let ever_allowed_by_label = Arc::new(ever_allowed_by_label);

        if std::env::var("DEBUG_TERMINAL_ALWAYS_ALLOWED_FOLLOW_MAP")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            let total_terminals = parser.terminal_map.len();
            let mut non_empty = 0usize;
            let mut self_follow = 0usize;
            for (term, follows) in &always_allowed {
                if !follows.is_empty() {
                    non_empty += 1;
                }
                if follows.contains(term) {
                    self_follow += 1;
                }
            }
            eprintln!(
                "TERMINAL_ALWAYS_ALLOWED_FOLLOW: total_terminals={}, non_empty_entries={}, self_follow_entries={}",
                total_terminals,
                non_empty,
                self_follow,
            );
        }
        let self_extending_labels_for_collapse: Option<Arc<HashSet<crate::dwa_i32::Label>>> =
            Some(Arc::new(labels));

        if std::env::var("DEBUG_TERMINAL_ALWAYS_FOLLOW_MAP")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            let always_follow_map = compute_always_follow_sets(&parser);
            let total_terminals = parser.terminal_map.len();
            let mut size_counts: BTreeMap<usize, usize> = BTreeMap::new();
            let mut singleton_pairs: Vec<(GrammarTokenID, GrammarTokenID)> = Vec::new();
            let mut multi_entries: Vec<(GrammarTokenID, usize)> = Vec::new();
            let mut multi_count = 0usize;

            for (tid, follows) in &always_follow_map {
                let size = follows.len();
                *size_counts.entry(size).or_insert(0) += 1;
                if size == 1 {
                    if let Some(&only) = follows.iter().next() {
                        singleton_pairs.push((*tid, only));
                    }
                } else if size > 1 {
                    multi_count += 1;
                    multi_entries.push((*tid, size));
                }
            }

            eprintln!(
                "TERMINAL_ALWAYS_FOLLOW: total_terminals={}, non_empty_entries={}, multi_follow_entries={}, size_counts={:?}",
                total_terminals,
                always_follow_map.len(),
                multi_count,
                size_counts,
            );

            let print_limit = 40usize;
            if !singleton_pairs.is_empty() {
                eprintln!("TERMINAL_ALWAYS_FOLLOW: singleton follows (showing up to {})", print_limit);
                for (tid, only) in singleton_pairs.iter().take(print_limit) {
                    let t1 = parser
                        .terminal_map
                        .get_by_right(tid)
                        .map(|t| format!("{:?}", t))
                        .unwrap_or_else(|| format!("TerminalID({})", tid.0));
                    let t2 = parser
                        .terminal_map
                        .get_by_right(only)
                        .map(|t| format!("{:?}", t))
                        .unwrap_or_else(|| format!("TerminalID({})", only.0));
                    eprintln!("TERMINAL_ALWAYS_FOLLOW: {} -> {}", t1, t2);
                }
            }

            if !multi_entries.is_empty() {
                eprintln!("TERMINAL_ALWAYS_FOLLOW: multi follows (showing up to {})", print_limit);
                for (tid, size) in multi_entries.iter().take(print_limit) {
                    let t1 = parser
                        .terminal_map
                        .get_by_right(tid)
                        .map(|t| format!("{:?}", t))
                        .unwrap_or_else(|| format!("TerminalID({})", tid.0));
                    eprintln!("TERMINAL_ALWAYS_FOLLOW: {} -> {} terminals", t1, size);
                }
            }
        }

        if std::env::var("DEBUG_TERMINAL_SELF_EXTENDING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            let self_extending = compute_self_extending_terminals(&parser);
            let total_terminals = parser.terminal_map.len();
            eprintln!(
                "TERMINAL_SELF_EXTENDING: count={}, total_terminals={}",
                self_extending.len(),
                total_terminals
            );
            let print_limit = 60usize;
            if !self_extending.is_empty() {
                eprintln!("TERMINAL_SELF_EXTENDING: terminals (showing up to {})", print_limit);
                for tid in self_extending.iter().take(print_limit) {
                    let name = parser
                        .terminal_map
                        .get_by_right(tid)
                        .map(|t| format!("{:?}", t))
                        .unwrap_or_else(|| format!("TerminalID({})", tid.0));
                    eprintln!("TERMINAL_SELF_EXTENDING: {}", name);
                }
            }
        }

        // commit_vocab is already computed in setup_combined above

        let mut vocab = StageVocab {
            original_to_internal: original_to_internal_map.clone(),
            internal_to_original: internal_to_original_map.clone(),
            internal_max_llm_token,
            max_original_llm_token_id,
            internal_to_original_sparse_matrix: vec![],
        };

        // Number of tokenizer states for weight-heavy encoding.
        // Always use representative count (internal tsid compression).
        let weight_heavy_enabled = crate::constraint_precompute::is_weight_heavy_enabled();
        let num_tsids = if weight_heavy_enabled {
            representative_states.len()
        } else {
            0
        };

        // Build state_to_internal_tsid: raw tokenizer state ID → internal tsid (0..num_tsids-1)
        // This MUST be built before precompute1 so the DWA weights use internal indices.
        let mut state_to_internal_tsid: Vec<usize> = if weight_heavy_enabled {
            let rep_to_internal: std::collections::HashMap<usize, usize> = representative_states
                .iter()
                .enumerate()
                .map(|(idx, &rep)| (rep, idx))
                .collect();
            let num_raw_states = tokenizer.dfa().states.len();
            let mut mapping = vec![0usize; num_raw_states];
            for (&state, &rep) in &state_to_rep {
                if let Some(&internal) = rep_to_internal.get(&rep.0) {
                    if state.0 < num_raw_states {
                        mapping[state.0] = internal;
                    }
                }
            }
            crate::debug!(3, "Built state_to_internal_tsid: {} raw states -> {} internal tsids",
                num_raw_states, representative_states.len());
            mapping
        } else {
            vec![]
        };

        // Set global dimensions for RangeSet operations (LLMTokenBV::max_ones(), etc.)
        // In symbol-heavy mode (num_tsids == 0), we use num_tsids=1 for dimension calculations
        crate::datastructures::set_global_dims_all_threads(
            internal_max_llm_token,
            if num_tsids > 0 { num_tsids } else { 1 },
        );

        let disable_suffix_prune = true;
        let enable_suffix_prune = false;
        let suffix_prune_enabled = false;
        let do_nwa_suffix_prune = false;
        let do_dwa_suffix_prune = false;
        let need_suffix_cache = false;
        let suffix_parser_cache = if need_suffix_cache {
            grammar_definition
                .as_ref()
                .map(|grammar_def| Arc::new(crate::interface::build_suffix_parser_cache(
                    grammar_def,
                    &parser.terminal_map,
                )))
        } else {
            None
        };

        let ignored_terminals = if let Some(cache) = suffix_parser_cache.as_ref() {
            cache.ignored_terminals.clone()
        } else {
            let mut ignored = vec![false; parser.terminal_map.len()];
            if let Some(ref grammar_def) = grammar_definition {
                for tid in &grammar_def.ignore_terminal_ids {
                    if tid.0 < ignored.len() {
                        ignored[tid.0] = true;
                    }
                }
            } else {
                for tid in &parser.ignore_terminal_ids {
                    if tid.0 < ignored.len() {
                        ignored[tid.0] = true;
                    }
                }
            }
            ignored
        };

        let ignored_terminals = Arc::new(ignored_terminals);
        let terminal_to_greedy_group = {
            let mut mapping = vec![None; parser.terminal_map.len()];
            if let Some(grammar_def) = grammar_definition.as_ref() {
                let terminal_name_to_tid: HashMap<String, usize> = parser
                    .terminal_map
                    .iter()
                    .map(|(term, tid)| {
                        let formatted = match term {
                            Terminal::RegexName(name) => name.clone(),
                            Terminal::Literal(bytes) => {
                                let mut escaped = String::new();
                                for ch in String::from_utf8_lossy(bytes).chars() {
                                    match ch {
                                        '\\' => escaped.push_str("\\\\"),
                                        '\'' => escaped.push_str("\\'"),
                                        _ => escaped.extend(ch.escape_default()),
                                    }
                                }
                                format!("'{}'", escaped)
                            }
                        };
                        (formatted, tid.0)
                    })
                    .collect();

                for (group_idx, group) in grammar_def.greedy_groups.iter().enumerate() {
                    for terminal_name in &group.terminals {
                        if let Some(&tid) = terminal_name_to_tid.get(terminal_name) {
                            mapping[tid] = Some(group_idx);
                        }
                    }
                }
            }
            mapping
        };

        let dump_terminal_automata = profile_terminal_automata_dump_enabled();
        if dump_terminal_automata {
            dump_terminal_id_map(&parser);
        }

        // Pre-DWA dimension reordering: predict good token and tsid orderings
        // from possible_matches data, BEFORE building the terminal DWA.
        // This permutes the internal tsid mapping and token IDs without touching the DFA.
        let pre_dwa_reorder_enabled = weight_heavy_enabled
            && std::env::var("ENABLE_PRE_DWA_REORDER")
                .map(|v| v == "1")
                .unwrap_or(false);
        let mut did_pre_dwa_reorder = false;
        if pre_dwa_reorder_enabled {
            let reorder_start = std::time::Instant::now();
            crate::debug!(3, "Pre-DWA reorder: predicting orderings from possible_matches...");

            // Predict orderings operating on internal tsid space (0..num_tsids-1)
            // and internal token space (0..max_internal_token)
            let (token_perm, tsid_perm) =
                crate::dwa_i32::reorder::predict_orderings_from_possible_matches(
                    &computed_possible_matches,
                    vocab.internal_max_llm_token,
                    num_tsids,
                    &state_to_internal_tsid,
                );

            // Apply tsid permutation to state_to_internal_tsid values (no DFA reorder!)
            // The tsid_perm operates on 0..num_tsids-1 (internal space).
            for internal in state_to_internal_tsid.iter_mut() {
                *internal = tsid_perm[*internal];
            }

            // Apply token permutation: remap internal token IDs
            let mut new_internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
            for (bytes, old_id) in &internal_llm_token_map {
                new_internal_llm_token_map.insert(bytes.clone(), LLMTokenID(token_perm[old_id.0]));
            }
            internal_llm_token_map = new_internal_llm_token_map;

            // Update original_to_internal
            for (_orig, int_id) in vocab.original_to_internal.iter_mut() {
                *int_id = token_perm[*int_id];
            }

            // Rebuild internal_to_original
            let max_int = vocab.internal_max_llm_token;
            let mut groups: Vec<Vec<usize>> = vec![Vec::new(); max_int + 1];
            for (orig, int_id) in &vocab.original_to_internal {
                groups[*int_id].push(*orig);
            }
            vocab.internal_to_original = groups
                .into_iter()
                .enumerate()
                .filter(|(_, v)| !v.is_empty())
                .map(|(int_id, origs)| (int_id, LLMTokenBV::from_iter(origs)))
                .collect();

            did_pre_dwa_reorder = true;
            crate::debug!(
                3,
                "Pre-DWA reorder complete in {:?} (tokens: {}, internal_tsids: {})",
                reorder_start.elapsed(),
                vocab.internal_max_llm_token + 1,
                num_tsids,
            );
        }

        crate::debug!(4, "Running precompute1 (weight_heavy={}, num_tsids={})...", weight_heavy_enabled, num_tsids);
        crate::debug!(5, "run_precompute1: start");
        let precompute_start = std::time::Instant::now();
        let (mut terminal_dwa, mut possible_matches_precompute1) = timeit!("run_precompute1", {
            run_precompute1_with_possible_matches(
                &tokenizer,
                &internal_llm_token_map,
                vocab.internal_max_llm_token,
                parser.terminal_map.len(),
                state_to_rep.clone(),
                num_tsids,
                state_to_internal_tsid.clone(),
                suffix_parser_cache.clone(),
                self_extending_labels_for_collapse.clone(),
                ignored_terminals.clone(),
                allowed_follows_by_label.clone(),
                always_allowed_by_label.clone(),
                ever_allowed_by_label.clone(),
                terminal_to_greedy_group.clone(),
            )
        });
        crate::timing!("TIMING: run_precompute1 {:?}", precompute_start.elapsed());
        crate::debug!(5, "run_precompute1: end");

        crate::debug!(4, "Done precompute1. Terminal DWA (before pruning): {}", terminal_dwa.stats());

        let post_precompute1_start = std::time::Instant::now();

        // Prune terminal DWA using suffix grammar if grammar definition is available
        // This removes transitions that can't possibly lead to valid parses
        crate::debug!(4, "Checking suffix prune conditions: grammar_def={}, enabled={}, env_enable={}, env_disable={}",
            grammar_definition.is_some(),
            suffix_prune_enabled,
            enable_suffix_prune,
            disable_suffix_prune
        );
        let mut did_prune = false;
        if suffix_prune_enabled {
            if let Some(cache) = suffix_parser_cache.as_ref() {
                crate::debug!(4, "Pruning terminal DWA with suffix grammar...");
                let prune_start = std::time::Instant::now();
                let terminals_count = parser.terminal_map.len();
                let (kept, pruned) = crate::interface::prune_dwa_with_suffix_cache(
                    &mut terminal_dwa,
                    cache,
                    terminals_count,
                );
                crate::debug!(5, "Suffix grammar pruning in {:?}", prune_start.elapsed());
                crate::timing!("TIMING: suffix_prune {:?}", prune_start.elapsed());
                crate::debug!(4, "Suffix grammar pruning complete. Kept={}, pruned={}", kept, pruned);
                did_prune = pruned > 0;
            } else if let Some(ref grammar_def) = grammar_definition {
                crate::debug!(4, "Pruning terminal DWA with suffix grammar...");
                let prune_start = std::time::Instant::now();
                let terminals_count = parser.terminal_map.len();
                let (kept, pruned) = crate::interface::prune_dwa_with_suffix_grammar(
                    &mut terminal_dwa,
                    grammar_def,
                    &parser.terminal_map,
                    terminals_count,
                );
                crate::debug!(5, "Suffix grammar pruning in {:?}", prune_start.elapsed());
                crate::timing!("TIMING: suffix_prune {:?}", prune_start.elapsed());
                crate::debug!(4, "Suffix grammar pruning complete. Kept={}, pruned={}", kept, pruned);
                did_prune = pruned > 0;
            }
        }

        // Re-minimize after pruning (some states may now be mergeable)
        if did_prune {
            crate::debug!(4, "Re-minimizing terminal DWA after suffix pruning...");
            let before_stats = terminal_dwa.stats();
            let rem_start = std::time::Instant::now();
            let _dwa_type_guard = crate::dwa_i32::minimization::graph_coloring::set_current_dwa_type(
                Some("terminal_post_prune"),
            );
            terminal_dwa.minimize();
            crate::debug!(5, "Terminal DWA re-minimize in {:?}", rem_start.elapsed());
            crate::timing!(
                "TIMING: terminal_dwa_reminimize {:?}",
                rem_start.elapsed()
            );
            crate::debug!(4, "Terminal DWA re-minimization: {} -> {}",
                before_stats, terminal_dwa.stats());
        }

        // Now print terminal DWA stats (after pruning and re-minimization)
        crate::debug!(3, "Terminal DWA (final): {}", 
            terminal_dwa.stats());

        // Analyze effective equivalence classes in the terminal DWA
        if std::env::var("ANALYZE_TERMINAL_DWA_EQUIV").is_ok() {
            use std::collections::{BTreeSet, HashMap, HashSet};
            use std::hash::{Hash, Hasher};
            let start_analysis = std::time::Instant::now();
            
            // For each weight in the terminal DWA, the weight is a RangeMapWeight: token → tsid_set.
            // 
            // TOKEN equivalence: Two tokens are equiv if for every weight, they map to the same tsid_set.
            //   Approach: For each weight, iterate range_values to get (token_range, tsid_set) pairs.
            //   For each token, its "column" across all weights is the tuple of tsid_sets.
            //   Number of distinct columns = number of token equivalence classes.
            //
            // TSID equivalence: Two TSIDs are equiv if for every weight×token, they're both in or both out.
            //   Approach: Invert each weight to tsid → token_set. For each TSID, its "row" across all
            //   weights is the tuple of token_sets. Number of distinct rows = TSID eq classes.
            
            // Step 1: Collect all weights as (weight_idx, RangeMapWeight ref) 
            let mut all_rangemaps: Vec<&crate::datastructures::RangeMapWeight> = Vec::new();
            let mut weight_count = 0usize;
            let mut non_rangemap_count = 0usize;
            
            for state in &terminal_dwa.states.0 {
                for (_, w) in &state.trans_weights {
                    weight_count += 1;
                    match w {
                        crate::datastructures::AbstractWeight::RangeMap(rm) => {
                            all_rangemaps.push(rm);
                        }
                        _ => { non_rangemap_count += 1; }
                    }
                }
                if let Some(fw) = &state.final_weight {
                    weight_count += 1;
                    match fw {
                        crate::datastructures::AbstractWeight::RangeMap(rm) => {
                            all_rangemaps.push(rm);
                        }
                        _ => { non_rangemap_count += 1; }
                    }
                }
            }
            
            crate::timing!(
                "ANALYZE_EQUIV: {} weights total, {} RangeMaps, {} non-RangeMap. DWA has {} states.",
                weight_count,
                all_rangemaps.len(),
                non_rangemap_count,
                terminal_dwa.states.0.len(),
            );
            
            // Step 2: Collect all token range boundaries across all weights.
            // This gives us a partition of the token space where each interval maps to
            // the same tsid_set in every weight.
            let mut token_boundaries: BTreeSet<usize> = BTreeSet::new();
            token_boundaries.insert(0);
            let max_token = vocab.internal_max_llm_token;
            
            for rm in &all_rangemaps {
                for (token_range, _tsid_set) in rm.map.range_values() {
                    token_boundaries.insert(*token_range.start());
                    token_boundaries.insert(token_range.end().wrapping_add(1));
                }
            }
            
            // Build token intervals
            let token_bps: Vec<usize> = token_boundaries.into_iter().filter(|&t| t <= max_token + 1).collect();
            let mut token_intervals: Vec<(usize, usize)> = Vec::new();
            for i in 0..token_bps.len() {
                let start = token_bps[i];
                let end = if i + 1 < token_bps.len() { token_bps[i+1].saturating_sub(1) } else { max_token };
                if start <= end {
                    token_intervals.push((start, end));
                }
            }
            
            crate::timing!(
                "ANALYZE_EQUIV: {} token boundaries -> {} token intervals",
                token_bps.len(),
                token_intervals.len(),
            );
            
            // Step 3: TOKEN EQUIVALENCE CLASSES
            // For each token interval, compute signature = for each weight, hash of the tsid_set it maps to.
            // Use the representative token (start of interval) since all tokens in the interval
            // have the same tsid_set mapping in every weight.
            let tok_eq_start = std::time::Instant::now();
            let mut token_sig_map: HashMap<Vec<u64>, usize> = HashMap::new(); // sig -> count of intervals
            
            for &(tok_start, _tok_end) in &token_intervals {
                let mut sig: Vec<u64> = Vec::with_capacity(all_rangemaps.len());
                for rm in &all_rangemaps {
                    if let Some(tsid_set) = rm.map.get(tok_start) {
                        // Hash the tsid_set by its ranges
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        for r in tsid_set.ranges() {
                            r.start().hash(&mut hasher);
                            r.end().hash(&mut hasher);
                        }
                        sig.push(hasher.finish());
                    } else {
                        sig.push(0); // empty
                    }
                }
                *token_sig_map.entry(sig).or_insert(0) += 1;
            }
            
            let num_token_eq_classes = token_sig_map.len();
            let mut tok_class_sizes: Vec<usize> = token_sig_map.values().copied().collect();
            tok_class_sizes.sort_unstable();
            tok_class_sizes.reverse();
            let top10_tok: Vec<usize> = tok_class_sizes.iter().take(10).copied().collect();
            
            crate::timing!(
                "ANALYZE_EQUIV: Token eq classes: {} (from {} intervals). Top10 sizes: {:?}. ({:?})",
                num_token_eq_classes,
                token_intervals.len(),
                top10_tok,
                tok_eq_start.elapsed(),
            );
            
            // Step 4: TSID EQUIVALENCE CLASSES (inverted)
            // For each TSID, its signature = for each weight × token_interval, is this TSID present?
            // Collect TSID range boundaries across all weights.
            let tsid_eq_start = std::time::Instant::now();
            let mut tsid_boundaries: BTreeSet<usize> = BTreeSet::new();
            tsid_boundaries.insert(0);
            
            for rm in &all_rangemaps {
                for (_token_range, tsid_set) in rm.map.range_values() {
                    for tsid_range in tsid_set.ranges() {
                        tsid_boundaries.insert(*tsid_range.start());
                        tsid_boundaries.insert(tsid_range.end().wrapping_add(1));
                    }
                }
            }
            
            let tsid_bps: Vec<usize> = tsid_boundaries.into_iter().filter(|&t| t <= num_tsids + 1).collect();
            let mut tsid_intervals: Vec<(usize, usize)> = Vec::new();
            for i in 0..tsid_bps.len() {
                let start = tsid_bps[i];
                let end = if i + 1 < tsid_bps.len() { tsid_bps[i+1].saturating_sub(1) } else { num_tsids };
                if start <= end {
                    tsid_intervals.push((start, end));
                }
            }
            
            // For each TSID interval, compute: for each (weight_idx, token_interval_idx), is rep_tsid present?
            // We hash this as a compact signature.
            let mut tsid_sig_map: HashMap<Vec<u8>, usize> = HashMap::new();
            
            for &(tsid_start, _tsid_end) in &tsid_intervals {
                let rep_tsid = tsid_start;
                // Bit-packed signature: 1 bit per (weight, token_interval) context
                let total_bits = all_rangemaps.len() * token_intervals.len();
                let mut sig = vec![0u8; (total_bits + 7) / 8];
                let mut bit_idx = 0usize;
                
                for rm in &all_rangemaps {
                    for &(tok_start, _tok_end) in &token_intervals {
                        let present = rm.map.get(tok_start)
                            .map_or(false, |tsid_set| tsid_set.contains(rep_tsid));
                        if present {
                            sig[bit_idx / 8] |= 1 << (bit_idx % 8);
                        }
                        bit_idx += 1;
                    }
                }
                
                *tsid_sig_map.entry(sig).or_insert(0) += 1;
            }
            
            let num_tsid_eq_classes = tsid_sig_map.len();
            let mut tsid_class_sizes: Vec<usize> = tsid_sig_map.values().copied().collect();
            tsid_class_sizes.sort_unstable();
            tsid_class_sizes.reverse();
            let top10_tsid: Vec<usize> = tsid_class_sizes.iter().take(10).copied().collect();
            
            crate::timing!(
                "ANALYZE_EQUIV: TSID eq classes: {} (from {} intervals). Top10 sizes: {:?}. ({:?})",
                num_tsid_eq_classes,
                tsid_intervals.len(),
                top10_tsid,
                tsid_eq_start.elapsed(),
            );
            
            // Summary comparison with setup_combined
            crate::timing!(
                "ANALYZE_EQUIV SUMMARY: setup_combined: {} token reps (from {}), {} state reps. Terminal DWA: {} token eq classes, {} TSID eq classes.",
                vocab.original_to_internal.len(),
                max_token + 1,
                num_tsids,
                num_token_eq_classes,
                num_tsid_eq_classes,
            );
        }

        if let Some((token_id, labels)) = crate::debug_path_weight::parse_debug_path_weight_env() {
            let ignore_final = crate::debug_path_weight::debug_path_weight_ignore_final();
            let weight = if ignore_final {
                crate::debug_path_weight::check_dwa_path_weight_no_final(&terminal_dwa, &labels)
            } else {
                crate::debug_path_weight::check_dwa_path_weight(&terminal_dwa, &labels)
            };
            let num_tsids = crate::datastructures::get_num_tsids();
            let contains = crate::debug_path_weight::weight_contains_token(&weight, token_id, num_tsids);
            eprintln!(
                "DEBUG_PATH_WEIGHT stage=terminal_dwa_final token={} labels={:?} contains={} weight_len={} ignore_final={}",
                token_id,
                labels,
                contains,
                weight.len(),
                ignore_final,
            );
        }

        // Compute domain_max for weight trimming and metrics
        let domain_max = if weight_heavy_enabled {
            let n = vocab.internal_max_llm_token;
            let m = num_tsids;
            n.saturating_mul(m).saturating_add(m.saturating_sub(1))
        } else {
            vocab.internal_max_llm_token
        };
        
        // Trim weights to domain_max to remove unnecessary range extensions to usize::MAX
        terminal_dwa.trim_weights_to_domain(domain_max);

        if std::env::var("DEBUG_TERMINAL_SELF_EXTENDING_CHAINS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        {
            let self_extending = compute_self_extending_terminals(&parser);
            let terminals_count = parser.terminal_map.len();
            let mut self_ext_transitions = 0usize;

            for state in &terminal_dwa.states.0 {
                for (&label, _) in &state.transitions {
                    let label_usize = label as usize;
                    if label_usize < terminals_count
                        && self_extending.contains(&GrammarTokenID(label_usize))
                    {
                        self_ext_transitions += 1;
                    }
                }
            }

            let mut terminals_with_repeatable_chain = 0usize;
            let mut max_chain_len_overall = 0usize;
            let mut repeatable_edges = 0usize;
            let mut repeatable_states = 0usize;
            let mut terminal_chain_lengths: Vec<(GrammarTokenID, usize)> = Vec::new();

            for tid in &self_extending {
                let label = tid.0 as i32;
                let mut next_t: Vec<Option<usize>> = vec![None; terminal_dwa.states.len()];
                for (sid, state) in terminal_dwa.states.0.iter().enumerate() {
                    if let Some(&dst) = state.transitions.get(&label) {
                        next_t[sid] = Some(dst);
                    }
                }

                let mut memo: Vec<Option<usize>> = vec![None; terminal_dwa.states.len()];
                let mut visiting = vec![false; terminal_dwa.states.len()];
                fn chain_len(
                    state: usize,
                    next_t: &[Option<usize>],
                    memo: &mut [Option<usize>],
                    visiting: &mut [bool],
                ) -> usize {
                    if let Some(len) = memo[state] {
                        return len;
                    }
                    if visiting[state] {
                        return 1;
                    }
                    visiting[state] = true;
                    let len = if let Some(next) = next_t[state] {
                        1 + chain_len(next, next_t, memo, visiting)
                    } else {
                        0
                    };
                    visiting[state] = false;
                    memo[state] = Some(len);
                    len
                }

                let mut max_len_t = 0usize;
                let mut has_repeat_chain = false;
                for sid in 0..terminal_dwa.states.len() {
                    if next_t[sid].is_none() {
                        continue;
                    }
                    let len = chain_len(sid, &next_t, &mut memo, &mut visiting);
                    max_len_t = max_len_t.max(len);
                    if len >= 2 {
                        has_repeat_chain = true;
                        repeatable_states += 1;
                    }
                    if let Some(next) = next_t[sid] {
                        if next_t[next].is_some() {
                            repeatable_edges += 1;
                        }
                    }
                }

                if has_repeat_chain {
                    terminals_with_repeatable_chain += 1;
                }
                max_chain_len_overall = max_chain_len_overall.max(max_len_t);
                terminal_chain_lengths.push((*tid, max_len_t));
            }

            terminal_chain_lengths.sort_by(|a, b| b.1.cmp(&a.1));
            let top_limit = 10usize;
            let top_chains: Vec<String> = terminal_chain_lengths
                .iter()
                .take(top_limit)
                .map(|(tid, len)| {
                    let name = parser
                        .terminal_map
                        .get_by_right(tid)
                        .map(|t| format!("{:?}", t))
                        .unwrap_or_else(|| format!("TerminalID({})", tid.0));
                    format!("{}: {}", name, len)
                })
                .collect();

            eprintln!(
                "TERMINAL_SELF_EXT_CHAINS: terminals_self_extending={}, transitions_self_extending={}, terminals_with_repeatable_chain={}, repeatable_edges={}, repeatable_states={}, max_chain_len={}, top_chain_lengths={:?}",
                self_extending.len(),
                self_ext_transitions,
                terminals_with_repeatable_chain,
                repeatable_edges,
                repeatable_states,
                max_chain_len_overall,
                top_chains,
            );
        }

        // Weight complexity instrumentation (unique weights are interned).
        if crate::r#macro::is_debug_level_enabled(5) {
            crate::debug!(5, "Terminal DWA weight complexity (unique): total_ranges_unique={} (total_ranges_all={})", 
                terminal_dwa.num_ranges_interned(),
                terminal_dwa.num_ranges(),
            );
        }
        
        if crate::r#macro::is_debug_level_enabled(4) {
            if let Some(num_paths) = terminal_dwa.count_paths() {
                crate::debug!(4, "Terminal DWA: {} paths (count_paths)", num_paths);
            }
            if let Some(avg_path_len) = terminal_dwa.average_path_length() {
                crate::debug!(4, "Terminal DWA average path length (exact): {:.2}", avg_path_len);
            }
            if let Some((total_paths, total_length, avg)) = terminal_dwa.average_path_length_debug() {
                crate::debug!(4, "Terminal DWA: total_paths={}, total_length={}, avg={:.2} (debug)", 
                             total_paths, total_length, avg);
            }
            // Sample 100 paths and print their lengths
            {
                let mut rng = rand::thread_rng();
                let sample_paths = terminal_dwa.sample_paths(100, &mut rng);
                let lengths: Vec<_> = sample_paths.iter().map(|p| p.len()).collect();
                let min_len = lengths.iter().min().copied().unwrap_or(0);
                let max_len = lengths.iter().max().copied().unwrap_or(0);
                let sum_len: usize = lengths.iter().sum();
                let avg_len = sum_len as f64 / lengths.len() as f64;
                crate::debug!(4, "Terminal DWA sample (n=100): min={}, max={}, avg={:.2}", min_len, max_len, avg_len);
            }
            if let Some(est_avg) = terminal_dwa.estimate_average_path_length(10000) {
                crate::debug!(4, "Terminal DWA average path length (estimated, 10k samples): {:.2}", est_avg);
            }

            // Analyze terminal DWA structure (skip if empty)
            if !terminal_dwa.states.0.is_empty() {
                let start_state = &terminal_dwa.states[terminal_dwa.body.start_state];
                let start_transitions = start_state.transitions.len();
                let unique_trans_targets: std::collections::HashSet<_> = start_state.transitions.values().copied().collect();
                crate::debug!(4, "Terminal DWA start state: {} transitions to {} unique targets", 
                    start_transitions, unique_trans_targets.len());
            } else {
                crate::debug!(4, "Terminal DWA is empty (no valid tokens for this grammar/vocabulary combination)");
            }
        }

        // Sample and print terminal DWA paths when DWA_SAMPLE_PATHS is set.
        // Sampling controls:
        // - DWA_SAMPLE_PATHS: number of paths to print (default: 10)
        // - DWA_SAMPLE_LONG: bias toward longer paths
        // - DWA_SAMPLE_MIN_LEN: minimum path length to accept
        // - DWA_SAMPLE_MAX_ATTEMPTS / DWA_SAMPLE_MAX_STEPS / DWA_SAMPLE_END_PROB / DWA_SAMPLE_MAX_TOKENS
        // Focus mode (single-token debugging):
        // - DWA_SAMPLE_FOCUS_TOKEN_ID: internal token id (or comma-separated ids)
        // - DWA_SAMPLE_FOCUS_TOKEN_TEXT: token bytes as literal text (e.g. '","')
        // - DWA_SAMPLE_FOCUS_TOKEN_HEX: token bytes as hex (e.g. 222c22)
        // - DWA_SAMPLE_FOCUS_ONLY: keep only paths whose final weight contains focus token(s)
        if std::env::var("DWA_SAMPLE_PATHS").is_ok() {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let parse_env_bool = |key: &str| -> bool {
                std::env::var(key)
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes") || v.eq_ignore_ascii_case("on"))
                    .unwrap_or(false)
            };
            let num_sample_paths: usize = std::env::var("DWA_SAMPLE_PATHS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10);
            let sample_long = parse_env_bool("DWA_SAMPLE_LONG");
            let min_len: Option<usize> = std::env::var("DWA_SAMPLE_MIN_LEN")
                .ok()
                .and_then(|s| s.parse().ok());
            let sample_max_tokens: usize = std::env::var("DWA_SAMPLE_MAX_TOKENS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3);
            let target_samples = if sample_long {
                num_sample_paths.saturating_mul(20).max(num_sample_paths)
            } else {
                num_sample_paths
            };
            let max_attempts: usize = std::env::var("DWA_SAMPLE_MAX_ATTEMPTS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| target_samples.saturating_mul(50).max(target_samples));
            let max_steps: usize = std::env::var("DWA_SAMPLE_MAX_STEPS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(if sample_long { 2048 } else { 512 });
            let end_prob: f64 = std::env::var("DWA_SAMPLE_END_PROB")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .map(|p| p.clamp(0.0, 1.0))
                .unwrap_or(if sample_long { 0.1 } else { 0.3 });
            
            // In weight-heavy mode, token IDs in weights are expanded: id = internal_id * num_tsids + tsid_offset
            // We need to convert back to internal token ID to look up bytes
            let num_tsids_for_conversion = if weight_heavy_enabled {
                tokenizer.dfa().states.len()
            } else {
                0
            };
            
            // Build reverse map from TerminalID to human-readable terminal name
            let terminals_count = parser.terminal_map.len();
            let tid_to_name: std::collections::BTreeMap<usize, String> = parser.terminal_map
                .iter()
                .map(|(term, tid)| {
                    let name = match term {
                        crate::glr::grammar::Terminal::Literal(bytes) => {
                            // Convert bytes to string if ASCII printable
                            if bytes.iter().all(|b| *b >= 0x20 && *b < 0x7f) {
                                format!("'{}'", String::from_utf8_lossy(bytes))
                            } else {
                                format!("Lit[{}]", bytes.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(""))
                            }
                        }
                        crate::glr::grammar::Terminal::RegexName(name) => name.clone(),
                    };
                    (tid.0, name)
                })
                .collect();

            let tid_to_literal_bytes: std::collections::BTreeMap<usize, Vec<u8>> = parser.terminal_map
                .iter()
                .filter_map(|(term, tid)| match term {
                    crate::glr::grammar::Terminal::Literal(bytes) => Some((tid.0, bytes.clone())),
                    crate::glr::grammar::Terminal::RegexName(_) => None,
                })
                .collect();

            let reachable_terminals: std::collections::BTreeSet<usize> = {
                let mut reachable = std::collections::BTreeSet::new();
                if !terminal_dwa.states.0.is_empty() {
                    let mut visited = vec![false; terminal_dwa.states.len()];
                    let mut queue = std::collections::VecDeque::new();
                    visited[terminal_dwa.body.start_state] = true;
                    queue.push_back(terminal_dwa.body.start_state);
                    while let Some(state_id) = queue.pop_front() {
                        let state = &terminal_dwa.states[state_id];
                        for (&label, &next_state) in &state.transitions {
                            if let Some(w) = state.trans_weights.get(&label) {
                                if w.is_empty() {
                                    continue;
                                }
                            }
                            let label_usize = label as usize;
                            if label_usize < terminals_count {
                                reachable.insert(label_usize);
                            }
                            if !visited[next_state] {
                                visited[next_state] = true;
                                queue.push_back(next_state);
                            }
                        }
                    }
                }
                reachable
            };

            let print_terminals = parse_env_bool("DWA_PRINT_TERMINALS");
            if print_terminals {
                crate::debug!(5, "Terminal list (reachable {} of {}):", reachable_terminals.len(), terminals_count);
                for tid in reachable_terminals.iter() {
                    let name = tid_to_name
                        .get(tid)
                        .cloned()
                        .unwrap_or_else(|| format!("T{}", tid));
                    crate::debug!(5, "  T{}: {}", tid, name);
                }
            }
            
            // Build reverse map from internal token ID to bytes (sorted by length for shortest)
            let mut internal_id_to_bytes: std::collections::BTreeMap<usize, Vec<u8>> = std::collections::BTreeMap::new();
            for (bytes, id) in &internal_llm_token_map {
                // Only keep the shortest bytes for each internal ID
                internal_id_to_bytes.entry(id.0)
                    .and_modify(|existing| {
                        if bytes.len() < existing.len() {
                            *existing = bytes.clone();
                        }
                    })
                    .or_insert_with(|| bytes.clone());
            }

            let format_token = |internal_id: usize| -> String {
                let token_str = internal_id_to_bytes.get(&internal_id)
                    .map(|bytes| {
                        let escaped: String = bytes.iter()
                            .map(|&b| {
                                if b == b'\'' {
                                    "\\'".to_string()
                                } else if b >= 0x20 && b < 0x7f && b != b'\\' {
                                    (b as char).to_string()
                                } else {
                                    format!("\\x{:02x}", b)
                                }
                            })
                            .collect();
                        format!("'{}'", escaped)
                    })
                    .unwrap_or_else(|| format!("?tok{}", internal_id));
                format!("{}:{}", internal_id, token_str)
            };

            let parse_hex_bytes = |raw: &str| -> Option<Vec<u8>> {
                let cleaned: String = raw
                    .trim()
                    .chars()
                    .filter(|c| !c.is_ascii_whitespace() && *c != '_' && *c != ':')
                    .collect();
                if cleaned.is_empty() || cleaned.len() % 2 != 0 {
                    return None;
                }
                let mut out = Vec::with_capacity(cleaned.len() / 2);
                let bytes = cleaned.as_bytes();
                let mut idx = 0usize;
                while idx < bytes.len() {
                    let pair = std::str::from_utf8(&bytes[idx..idx + 2]).ok()?;
                    let byte = u8::from_str_radix(pair, 16).ok()?;
                    out.push(byte);
                    idx += 2;
                }
                Some(out)
            };

            let mut focus_token_ids: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
            let mut focus_requests: Vec<String> = Vec::new();

            if let Ok(raw_ids) = std::env::var("DWA_SAMPLE_FOCUS_TOKEN_ID") {
                for part in raw_ids.split(',') {
                    let trimmed = part.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if let Ok(id) = trimmed.parse::<usize>() {
                        focus_token_ids.insert(id);
                        focus_requests.push(format!("id={}", id));
                    }
                }
            }

            if let Ok(raw_text) = std::env::var("DWA_SAMPLE_FOCUS_TOKEN_TEXT") {
                if !raw_text.is_empty() {
                    let target_bytes = raw_text.as_bytes();
                    let mut matches = 0usize;
                    for (token_bytes, token_id) in &internal_llm_token_map {
                        if token_bytes.as_slice() == target_bytes {
                            focus_token_ids.insert(token_id.0);
                            matches += 1;
                        }
                    }
                    focus_requests.push(format!("text='{}' matches={}", raw_text, matches));
                }
            }

            if let Ok(raw_hex) = std::env::var("DWA_SAMPLE_FOCUS_TOKEN_HEX") {
                if !raw_hex.trim().is_empty() {
                    if let Some(target_bytes) = parse_hex_bytes(raw_hex.as_str()) {
                        let mut matches = 0usize;
                        for (token_bytes, token_id) in &internal_llm_token_map {
                            if token_bytes.as_slice() == target_bytes.as_slice() {
                                focus_token_ids.insert(token_id.0);
                                matches += 1;
                            }
                        }
                        focus_requests.push(format!("hex={} matches={}", raw_hex, matches));
                    } else {
                        crate::debug!(4, "DWA_SAMPLE_FOCUS_TOKEN_HEX parse error: '{}'", raw_hex);
                    }
                }
            }

            let focus_only = std::env::var("DWA_SAMPLE_FOCUS_ONLY")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes") || v.eq_ignore_ascii_case("on"))
                .unwrap_or(!focus_token_ids.is_empty());

            let weight_contains_internal_token = |weight: &crate::dwa_i32::Weight, internal_id: usize| -> bool {
                crate::debug_path_weight::weight_contains_token(weight, internal_id, num_tsids_for_conversion)
            };

            crate::debug!(
                5,
                "DWA_SAMPLE config: n={}, target_samples={}, long={}, min_len={:?}, max_attempts={}, max_steps={}, end_prob={:.3}, max_tokens={}, focus_only={}",
                num_sample_paths,
                target_samples,
                sample_long,
                min_len,
                max_attempts,
                max_steps,
                end_prob,
                sample_max_tokens,
                focus_only
            );
            if !focus_token_ids.is_empty() {
                let focused: Vec<String> = focus_token_ids
                    .iter()
                    .copied()
                    .map(format_token)
                    .collect();
                crate::debug!(5, "DWA_SAMPLE focus tokens: [{}]", focused.join(", "));
            } else if !focus_requests.is_empty() {
                crate::debug!(5, "DWA_SAMPLE focus requested but unresolved: {}", focus_requests.join("; "));
            }

            if let Ok(token_list) = std::env::var("DEBUG_TOKEN_BYTES") {
                let mut ids: Vec<usize> = Vec::new();
                for part in token_list.split(',') {
                    let trimmed = part.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if let Ok(id) = trimmed.parse::<usize>() {
                        ids.push(id);
                    }
                }
                if !ids.is_empty() {
                    ids.sort_unstable();
                    ids.dedup();
                    for id in ids {
                        if let Some(bytes) = internal_id_to_bytes.get(&id) {
                            crate::debug!(4, "DEBUG_TOKEN_BYTES id={} token={} len={}", id, format_token(id), bytes.len());
                        } else {
                            crate::debug!(4, "DEBUG_TOKEN_BYTES id={} token=?", id);
                        }
                    }
                }
            }
            
            let mut collected: Vec<(String, usize, crate::dwa_i32::Weight, Option<Vec<u8>>, Vec<crate::dwa_i32::Label>)> = Vec::new();
            let mut attempts = 0usize;

            while collected.len() < target_samples && attempts < max_attempts {
                let mut current_state = terminal_dwa.body.start_state;
                let mut path_strs: Vec<String> = Vec::new();
                let mut path_labels: Vec<crate::dwa_i32::Label> = Vec::new();
                let mut path_literal_bytes: Option<Vec<u8>> = Some(Vec::new());
                let mut path_weight = crate::dwa_i32::Weight::all();
                let mut steps = 0usize;
                let mut collected_this_attempt = false;

                loop {
                    // Check if we can end here with a non-empty weight
                    let end_weight = terminal_dwa.states[current_state]
                        .final_weight
                        .as_ref()
                        .map(|fw| {
                            let mut w = path_weight.clone();
                            w &= fw;
                            w
                        });

                    // Collect viable transitions (non-empty weight intersection)
                    let mut choices: Vec<(crate::dwa_i32::Label, crate::dwa_i32::StateID, crate::dwa_i32::Weight)> = Vec::new();
                    for (&label, &next_state) in &terminal_dwa.states[current_state].transitions {
                        if let Some(w) = terminal_dwa.states[current_state].trans_weights.get(&label) {
                            let mut next_w = path_weight.clone();
                            next_w &= w;
                            if !next_w.is_empty() {
                                choices.push((label, next_state, next_w));
                            }
                        }
                    }

                    if let Some(w) = end_weight {
                        if !w.is_empty() && (choices.is_empty() || rng.gen_bool(end_prob)) {
                            let focus_hit = !focus_token_ids.is_empty()
                                && focus_token_ids
                                    .iter()
                                    .copied()
                                    .any(|token_id| weight_contains_internal_token(&w, token_id));
                            let focus_ok = !focus_only || focus_hit;
                            if focus_ok && min_len.map_or(true, |min| path_strs.len() >= min) {
                                let path_str = path_strs.join(" → ");
                                collected.push((path_str, path_strs.len(), w, path_literal_bytes.clone(), path_labels.clone()));
                                collected_this_attempt = true;
                                break;
                            }
                        }
                    }

                    if choices.is_empty() || steps >= max_steps {
                        break;
                    }

                    let idx = rng.gen_range(0..choices.len());
                    let (label, next_state, next_weight) = choices.swap_remove(idx);

                    let label_usize = label as usize;
                    let name = if label_usize < terminals_count {
                        tid_to_name.get(&label_usize)
                            .cloned()
                            .unwrap_or_else(|| format!("T{}", label_usize))
                    } else {
                        format!("TSID{}", label_usize - terminals_count)
                    };
                    path_strs.push(name);
                    path_labels.push(label);

                    path_literal_bytes = match path_literal_bytes {
                        Some(mut bytes) => {
                            if label_usize < terminals_count {
                                if let Some(lit) = tid_to_literal_bytes.get(&label_usize) {
                                    bytes.extend_from_slice(lit);
                                    Some(bytes)
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        None => None,
                    };

                    path_weight = next_weight;
                    current_state = next_state;
                    steps += 1;
                }

                if !collected_this_attempt {
                    attempts += 1;
                    continue;
                }

                attempts += 1;
            }

            let mut deduped: std::collections::BTreeMap<String, (usize, crate::dwa_i32::Weight, Option<Vec<u8>>, Vec<crate::dwa_i32::Label>)> = std::collections::BTreeMap::new();
            for (path, len, w, lit, labels) in collected.into_iter() {
                deduped.entry(path).or_insert((len, w, lit, labels));
            }
            let mut collected: Vec<(String, usize, crate::dwa_i32::Weight, Option<Vec<u8>>, Vec<crate::dwa_i32::Label>)> = deduped
                .into_iter()
                .map(|(path, (len, w, lit, labels))| (path, len, w, lit, labels))
                .collect();

            let candidates_len = collected.len();
            let mut distinct_len_count = 0usize;
            if sample_long {
                collected.sort_by(|a, b| b.1.cmp(&a.1));
                let mut selected: Vec<(String, usize, crate::dwa_i32::Weight, Option<Vec<u8>>, Vec<crate::dwa_i32::Label>)> = Vec::new();
                let mut used_lengths = std::collections::BTreeSet::new();
                for (path, len, w, lit, labels) in collected.iter() {
                    if used_lengths.insert(*len) {
                        selected.push((path.clone(), *len, w.clone(), lit.clone(), labels.clone()));
                        if selected.len() >= num_sample_paths {
                            break;
                        }
                    }
                }
                if selected.len() < num_sample_paths {
                    for (path, len, w, lit, labels) in collected.iter() {
                        if selected.len() >= num_sample_paths {
                            break;
                        }
                        if selected.iter().any(|(p, _, _, _, _)| p == path) {
                            continue;
                        }
                        selected.push((path.clone(), *len, w.clone(), lit.clone(), labels.clone()));
                    }
                }
                distinct_len_count = used_lengths.len();
                collected = selected;
            } else if collected.len() > num_sample_paths {
                collected.sort_by(|a, b| b.1.cmp(&a.1));
                collected.truncate(num_sample_paths);
            }

            crate::debug!(5, "Terminal DWA sample paths (n={}, non-empty weights):", collected.len());
            let trace_samples = parse_env_bool("DWA_SAMPLE_TRACE");
            for (i, (path_str, path_len, w, path_literal_bytes, path_labels)) in collected.iter().enumerate() {
                let weight_len = w.len();
                let max_tokens = if sample_max_tokens == 0 {
                    0
                } else if weight_len <= 20 {
                    usize::MAX
                } else {
                    sample_max_tokens
                };
                let mut token_samples: Vec<usize> = Vec::new();
                let mut seen_tokens = std::collections::BTreeSet::new();
                for range in w.ranges() {
                    let mut pos = *range.start();
                    let end = *range.end();
                    while pos <= end && token_samples.len() < max_tokens {
                        let internal_id = if num_tsids_for_conversion > 0 {
                            pos / num_tsids_for_conversion
                        } else {
                            pos
                        };
                        if seen_tokens.insert(internal_id) {
                            token_samples.push(internal_id);
                        }
                        if pos == usize::MAX {
                            break;
                        }
                        pos = pos.saturating_add(1);
                    }
                    if token_samples.len() >= max_tokens {
                        break;
                    }
                }
                let formatted: Vec<String> = token_samples.iter().copied().map(format_token).collect();
                let tokens_str = if token_samples.is_empty() {
                    if weight_len <= 20 { "tokens_all=[]".to_string() } else { "tokens_sample=[]".to_string() }
                } else {
                    if weight_len <= 20 {
                        format!("tokens_all=[{}]", formatted.join(", "))
                    } else {
                        format!("tokens_sample=[{}]", formatted.join(", "))
                    }
                };
                let validation_note = match path_literal_bytes.as_ref() {
                    Some(path_bytes) if !token_samples.is_empty() => {
                        let path_len_bytes = path_bytes.len();
                        let mut invalid_short: Vec<String> = Vec::new();
                        let mut invalid_prefix: Vec<String> = Vec::new();
                        for (internal_id, token_str) in token_samples.iter().zip(formatted.iter()) {
                            if let Some(token_bytes) = internal_id_to_bytes.get(internal_id) {
                                if path_len_bytes > token_bytes.len() {
                                    invalid_short.push(token_str.clone());
                                } else if !token_bytes.starts_with(path_bytes) {
                                    invalid_prefix.push(token_str.clone());
                                }
                            }
                        }
                        if invalid_short.is_empty() && invalid_prefix.is_empty() {
                            String::new()
                        } else {
                            let mut parts = Vec::new();
                            if !invalid_short.is_empty() {
                                parts.push(format!("invalid_short=[{}]", invalid_short.join(", ")));
                            }
                            if !invalid_prefix.is_empty() {
                                parts.push(format!("invalid_prefix=[{}]", invalid_prefix.join(", ")));
                            }
                            format!(", literal_len={}, {}", path_len_bytes, parts.join(" "))
                        }
                    }
                    _ => String::new(),
                };
                let focus_hits: Vec<String> = focus_token_ids
                    .iter()
                    .copied()
                    .filter(|token_id| weight_contains_internal_token(w, *token_id))
                    .map(format_token)
                    .collect();
                let focus_note = if focus_token_ids.is_empty() {
                    String::new()
                } else {
                    format!(", focus_tokens=[{}]", focus_hits.join(", "))
                };
                let weight_info = format!("weight_len={}, {}{}", weight_len, tokens_str, validation_note);
                crate::debug!(5, "  Path {}: {} (len={}, {}{})", i, path_str, path_len, weight_info, focus_note);

                if trace_samples {
                    let mut current_state = terminal_dwa.body.start_state;
                    let mut acc = crate::dwa_i32::Weight::all();
                    for (step_idx, label) in path_labels.iter().enumerate() {
                        let Some((next_state, w_edge)) = terminal_dwa.states[current_state].get_transition(*label) else {
                            break;
                        };
                        acc &= w_edge;
                        let step_weight_len = acc.len();
                        let max_step_tokens = if sample_max_tokens == 0 {
                            0
                        } else if step_weight_len <= 20 {
                            usize::MAX
                        } else {
                            sample_max_tokens
                        };
                        let mut step_samples: Vec<usize> = Vec::new();
                        let mut step_seen = std::collections::BTreeSet::new();
                        for range in acc.ranges() {
                            let mut pos = *range.start();
                            let end = *range.end();
                            while pos <= end && step_samples.len() < max_step_tokens {
                                let internal_id = if num_tsids_for_conversion > 0 {
                                    pos / num_tsids_for_conversion
                                } else {
                                    pos
                                };
                                if step_seen.insert(internal_id) {
                                    step_samples.push(internal_id);
                                }
                                if pos == usize::MAX {
                                    break;
                                }
                                pos = pos.saturating_add(1);
                            }
                            if step_samples.len() >= max_step_tokens {
                                break;
                            }
                        }
                        let step_formatted: Vec<String> = step_samples.iter().copied().map(format_token).collect();
                        let step_tokens = if step_samples.is_empty() {
                            if step_weight_len <= 20 { "tokens_all=[]".to_string() } else { "tokens_sample=[]".to_string() }
                        } else if step_weight_len <= 20 {
                            format!("tokens_all=[{}]", step_formatted.join(", "))
                        } else {
                            format!("tokens_sample=[{}]", step_formatted.join(", "))
                        };
                        let label_usize = *label as usize;
                        let label_name = if label_usize < terminals_count {
                            tid_to_name.get(&label_usize)
                                .cloned()
                                .unwrap_or_else(|| format!("T{}", label_usize))
                        } else {
                            format!("TSID{}", label_usize.saturating_sub(terminals_count))
                        };
                        let step_focus_hits: Vec<String> = focus_token_ids
                            .iter()
                            .copied()
                            .filter(|token_id| weight_contains_internal_token(&acc, *token_id))
                            .map(format_token)
                            .collect();
                        let step_focus_note = if focus_token_ids.is_empty() {
                            String::new()
                        } else {
                            format!(", focus_tokens=[{}]", step_focus_hits.join(", "))
                        };
                        crate::debug!(5, "    Step {}: {} weight_len={}, {}{}", step_idx, label_name, step_weight_len, step_tokens, step_focus_note);
                        current_state = next_state;
                        if acc.is_empty() {
                            break;
                        }
                    }
                }
            }
            if collected.len() < num_sample_paths {
                crate::debug!(5, "  (collected {} non-empty paths after {} attempts)", collected.len(), attempts);
            }
            if sample_long {
                crate::debug!(5, "  (DWA_SAMPLE_LONG=1: selected {} paths across {} distinct lengths from {} candidates)", collected.len(), distinct_len_count, candidates_len);
                if candidates_len > 0 && distinct_len_count < 3 {
                    crate::debug!(5, "  (length diversity limited: {} distinct lengths in candidates)", distinct_len_count);
                }
            }

            let check_token_path_lengths = std::env::var("CHECK_TOKEN_PATH_LENGTHS")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if check_token_path_lengths {
                use std::collections::VecDeque;

                let n_states = terminal_dwa.states.len();
                let mut indegree = vec![0usize; n_states];
                for state in terminal_dwa.states.0.iter() {
                    for &next_state in state.transitions.values() {
                        indegree[next_state] = indegree[next_state].saturating_add(1);
                    }
                }

                let mut queue: VecDeque<usize> = VecDeque::new();
                for (sid, &deg) in indegree.iter().enumerate() {
                    if deg == 0 {
                        queue.push_back(sid);
                    }
                }

                let mut topo: Vec<usize> = Vec::with_capacity(n_states);
                while let Some(sid) = queue.pop_front() {
                    topo.push(sid);
                    for &next_state in terminal_dwa.states[sid].transitions.values() {
                        indegree[next_state] = indegree[next_state].saturating_sub(1);
                        if indegree[next_state] == 0 {
                            queue.push_back(next_state);
                        }
                    }
                }

                if topo.len() != n_states {
                    crate::debug!(4, "CHECK_TOKEN_PATH_LENGTHS: DWA may be cyclic; topo_len={}, states={}", topo.len(), n_states);
                }

                let start_state = terminal_dwa.body.start_state;
                crate::debug!(
                    4,
                    "CHECK_TOKEN_PATH_LENGTHS: context terminal_dwa states={} start_state={} terminals_count={} num_tsids_for_conversion={}",
                    n_states,
                    start_state,
                    terminals_count,
                    num_tsids_for_conversion
                );

                let weight_contains_token = |weight: &crate::dwa_i32::Weight, internal_id: usize| -> bool {
                    if num_tsids_for_conversion == 0 {
                        weight.contains(internal_id)
                    } else {
                        let start = internal_id.saturating_mul(num_tsids_for_conversion);
                        let end = start.saturating_add(num_tsids_for_conversion.saturating_sub(1));
                        for range in weight.ranges() {
                            let r_start = *range.start();
                            let r_end = *range.end();
                            if r_start > end {
                                break;
                            }
                            if r_end >= start {
                                return true;
                            }
                        }
                        false
                    }
                };

                let compute_exact_max_len = |internal_id: usize| -> Option<usize> {
                    if num_tsids_for_conversion == 0 {
                        return None;
                    }
                    let num_tsids = num_tsids_for_conversion;
                    let mut best_len: Vec<Vec<Option<usize>>> = vec![vec![None; num_tsids]; n_states];
                    for tsid in 0..num_tsids {
                        best_len[start_state][tsid] = Some(0);
                    }
                    for &sid in topo.iter() {
                        let state = &terminal_dwa.states[sid];
                        for (&label, &next_state) in &state.transitions {
                            let Some(weight) = state.trans_weights.get(&label) else {
                                continue;
                            };
                            let add = if (label as usize) < terminals_count { 1 } else { 0 };
                            for tsid in 0..num_tsids {
                                let Some(curr_len) = best_len[sid][tsid] else {
                                    continue;
                                };
                                let pos = internal_id.saturating_mul(num_tsids).saturating_add(tsid);
                                if !weight.contains(pos) {
                                    continue;
                                }
                                let cand = curr_len.saturating_add(add);
                                if best_len[next_state][tsid].map_or(true, |v| cand > v) {
                                    best_len[next_state][tsid] = Some(cand);
                                }
                            }
                        }
                    }
                    let mut max_len: Option<usize> = None;
                    for sid in 0..n_states {
                        for tsid in 0..num_tsids {
                            if let Some(len) = best_len[sid][tsid] {
                                if max_len.map_or(true, |v| len > v) {
                                    max_len = Some(len);
                                }
                            }
                        }
                    }
                    max_len
                };

                let mut violations: Vec<(usize, usize, usize, String)> = Vec::new();
                let mut short_reports: Vec<(usize, usize, usize, bool, Option<String>)> = Vec::new();
                for (&internal_id, bytes) in internal_id_to_bytes.iter() {
                    let token_len = bytes.len();
                    if token_len == 0 {
                        continue;
                    }

                    let mut best_len: Vec<Option<usize>> = vec![None; n_states];
                    let mut prev: Vec<Option<(usize, crate::dwa_i32::Label)>> = vec![None; n_states];
                    best_len[start_state] = Some(0);

                    for &sid in topo.iter() {
                        let Some(curr_len) = best_len[sid] else { continue; };
                        let state = &terminal_dwa.states[sid];
                        for (&label, &next_state) in &state.transitions {
                            if let Some(weight) = state.trans_weights.get(&label) {
                                if !weight_contains_token(weight, internal_id) {
                                    continue;
                                }
                                let label_usize = label as usize;
                                let add = if label_usize < terminals_count { 1 } else { 0 };
                                let cand = curr_len.saturating_add(add);
                                if best_len[next_state].map_or(true, |v| cand > v) {
                                    best_len[next_state] = Some(cand);
                                    prev[next_state] = Some((sid, label));
                                }
                            }
                        }
                    }

                    let mut max_len: Option<usize> = None;
                    let mut max_state: Option<usize> = None;
                    for (sid, state) in terminal_dwa.states.0.iter().enumerate() {
                        if let Some(final_weight) = &state.final_weight {
                            if !weight_contains_token(final_weight, internal_id) {
                                continue;
                            }
                            if let Some(len) = best_len[sid] {
                                if max_len.map_or(true, |v| len > v) {
                                    max_len = Some(len);
                                    max_state = Some(sid);
                                }
                            }
                        }
                    }

                    let mut max_len = match max_len {
                        Some(len) => len,
                        None => {
                            if token_len <= 5 {
                                short_reports.push((internal_id, token_len, 0, false, None));
                            }
                            continue;
                        }
                    };

                    let mut is_violation = max_len > token_len;
                    if is_violation {
                        if let Some(exact_max) = compute_exact_max_len(internal_id) {
                            max_len = exact_max;
                            is_violation = max_len > token_len;
                        }
                    }
                    let mut path_str: Option<String> = None;
                    let mut labels_full: Option<Vec<usize>> = None;
                    let mut full_path_str: Option<String> = None;

                    if is_violation {
                        let mut labels: Vec<usize> = Vec::new();
                        let mut labels_all: Vec<usize> = Vec::new();
                        let mut cur = max_state.unwrap_or(start_state);
                        while cur != start_state {
                            if let Some((prev_state, label)) = prev[cur] {
                                let label_usize = label as usize;
                                labels_all.push(label_usize);
                                if label_usize < terminals_count {
                                    labels.push(label_usize);
                                }
                                cur = prev_state;
                            } else {
                                break;
                            }
                        }
                        labels.reverse();
                        labels_all.reverse();
                        labels_full = Some(labels_all.clone());

                        let path = labels
                            .iter()
                            .map(|tid| {
                                if let Some(name) = tid_to_name.get(tid) {
                                    format!("{}({})", name, tid)
                                } else {
                                    format!("T{}({})", tid, tid)
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(" → ");
                        path_str = Some(path);

                        let full_path = labels_all
                            .iter()
                            .map(|lid| {
                                if *lid < terminals_count {
                                    if let Some(name) = tid_to_name.get(lid) {
                                        format!("{}({})", name, lid)
                                    } else {
                                        format!("T{}({})", lid, lid)
                                    }
                                } else {
                                    let tsid = lid.saturating_sub(terminals_count);
                                    format!("TSID{}({})", tsid, lid)
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(" → ");
                        full_path_str = Some(full_path);
                    }

                    if token_len <= 5 {
                        short_reports.push((internal_id, token_len, max_len, is_violation, path_str.clone()));
                    }

                    if is_violation {
                        if let Some(labels_all) = &labels_full {
                            crate::debug!(
                                4,
                                "  CHECK_TOKEN_PATH_LENGTHS: violation_context token_internal_id={} labels_raw={:?} labels_full={:?} full_path={}",
                                internal_id,
                                labels_all
                                    .iter()
                                    .filter(|lid| **lid < terminals_count)
                                    .copied()
                                    .collect::<Vec<usize>>(),
                                labels_all,
                                full_path_str.as_deref().unwrap_or("<none>")
                            );
                        }
                        if let Some(path) = path_str {
                            violations.push((internal_id, token_len, max_len, path));
                        }
                    }
                }

                if !violations.is_empty() {
                    violations.sort_by(|a, b| (b.2.saturating_sub(b.1)).cmp(&a.2.saturating_sub(a.1)));
                    crate::debug!(4, "CHECK_TOKEN_PATH_LENGTHS: {} violations (showing up to 20)", violations.len());
                    for (internal_id, token_len, max_len, path_str) in violations.iter().take(20) {
                        crate::debug!(4, "  token={} (len={}) max_path_len={} path={}", format_token(*internal_id), token_len, max_len, path_str);
                    }

                    if !short_reports.is_empty() {
                        short_reports.sort_by(|a, b| a.0.cmp(&b.0));
                        crate::debug!(4, "CHECK_TOKEN_PATH_LENGTHS: short token report (len<=5)");
                        for (internal_id, token_len, max_len, is_violation, path_str) in short_reports.iter() {
                            if *is_violation {
                                let off_by = max_len.saturating_sub(*token_len);
                                if let Some(path) = path_str {
                                    crate::debug!(4, "  token={} (len={}) max_path_len={} -- VIOLATION (off by {}) path={}", format_token(*internal_id), token_len, max_len, off_by, path);
                                } else {
                                    crate::debug!(4, "  token={} (len={}) max_path_len={} -- VIOLATION (off by {})", format_token(*internal_id), token_len, max_len, off_by);
                                }
                            } else {
                                crate::debug!(4, "  token={} (len={}) max_path_len={} -- OK", format_token(*internal_id), token_len, max_len);
                            }
                        }
                    }
                } else {
                    crate::debug!(4, "CHECK_TOKEN_PATH_LENGTHS: no violations");
                }
            }
        }
        // EPSILON EXPLOSION EXPERIMENT - Terminal DWA
        // This code tests whether replacing labeled tsid transitions with epsilon transitions
        // causes a transition explosion. Enable with: TEST_EPSILON_EXPLOSION=1
        //
        // Confirmed results (ApolloRouter schema):
        // - Original: 5,952 states, 45,284 transitions  
        // - Modified: 634 states, 315,507 transitions (6.97x explosion!)
        //
        // Conclusion: The proposed refactor to encode tsid in weights (using epsilons) is blocked
        // by this explosion. See TODO.md for details.
        if std::env::var("TEST_EPSILON_EXPLOSION").is_ok() {
            use crate::dwa_i32::nwa::NWA;
            use crate::dwa_i32::Weight;
            use std::collections::{HashSet, VecDeque};
            
            crate::debug!(1, "=== EPSILON EXPLOSION EXPERIMENT: Terminal DWA ===");
            
            // Get the valid LLM token range
            let max_llm_token = vocab.internal_max_llm_token;
            let valid_tokens = Weight::from_iter(0..=max_llm_token);
            crate::debug!(1, "Valid LLM token range: 0..={}, cardinality: {}", 
                max_llm_token, valid_tokens.len());
            
            // First, minimize the original terminal DWA with rustfst to verify it's minimal
            let mut orig_dwa_for_min = terminal_dwa.clone();
            orig_dwa_for_min.minimize();
            crate::debug!(1, "Original terminal DWA (after minimize_with_rustfst): {}",
                orig_dwa_for_min.stats());
            
            // Only dump verbose DWA structure and DOT graphs if DUMP_DWA_DOT is set
            // (These can produce 1000s of lines of output)
            if std::env::var("DUMP_DWA_DOT").is_ok() {
            // PRINT THE ACTUAL DWA STRUCTURE with human-readable names
            crate::debug!(1, "\n=== TERMINAL DWA STRUCTURE (minimized) ===");
            crate::debug!(1, "Start state: {}", orig_dwa_for_min.body.start_state);
            
            // Build a map from tokenizer state ID to a human-readable description
            let mut tsid_names: std::collections::HashMap<usize, String> = std::collections::HashMap::new();
            tsid_names.insert(0, "INITIAL".to_string());
            
            // BFS to find shortest example string for each state
            let mut queue = std::collections::VecDeque::new();
            queue.push_back((0, Vec::new())); // (state, bytes)
            let mut visited = std::collections::HashSet::new();
            visited.insert(0);
            
            // Limit BFS depth to avoid infinite loops in cyclic graphs
            let mut iterations = 0;
            while let Some((curr, bytes)) = queue.pop_front() {
                iterations += 1;
                if iterations > 10000 { break; }
                
                // Explore transitions
                if curr < tokenizer.dfa().states.len() {
                    // Collect transitions for this state
                    // The DFA structure in typical regex crates might be different, let's look at available fields
                    // Assuming we have access to transitions. 
                    // Since specific crate internals might be hidden, we rely on the fact that we can iterate 256 bytes
                    // Optimization: only check ASCII + some others? Or rely on .transitions field if public?
                    // Let's rely on `tokenizer.dfa().states[curr].transitions` which is a sparse map or array
                    
                    // We need to iterate edges. The Tokenizer struct seems to wrap a DFA.
                    // Let's look at how `tokenizer.dfa().states` is defined.
                    // Based on previous simple usage: `tokenizer.dfa().states[sid].transitions.get(byte)`
                    // We can just iterate the transitions map directly if it's a map.
                    
                    for (byte, &next) in &tokenizer.dfa().states[curr].transitions {
                        if !visited.contains(&next) {
                            visited.insert(next);
                            let mut new_bytes = bytes.clone();
                            new_bytes.push(byte);
                            
                            // Name this state
                            let s = String::from_utf8_lossy(&new_bytes);
                            tsid_names.insert(next, format!("after {}", s.escape_default()));
                            
                            if new_bytes.len() < 10 { // Don't make path too long
                                queue.push_back((next, new_bytes));
                            }
                        }
                    }
                }
            }
            
            for (sid, state) in orig_dwa_for_min.states.0.iter().enumerate() {
                let final_str = match &state.final_weight {
                    Some(w) => format!("FINAL(weight len={})", w.len()),
                    None => "non-final".to_string(),
                };
                crate::debug!(1, "State {}: {} transitions, {}", sid, state.transitions.len(), final_str);
                for (&label, &target) in &state.transitions {
                    // Decode label: if >= terminals_count, it's a tokenizer state ID
                    let label_str = if label >= parser.terminal_map.len() as i32 {
                        let tsid = (label - parser.terminal_map.len() as i32) as usize;
                        let name = tsid_names.get(&tsid)
                            .cloned()
                            .unwrap_or_else(|| format!("tsid:{}", tsid));
                        format!("TSID[{}]", name)
                    } else {
                        // Look up terminal name and show the actual bytes
                        let term = parser.terminal_map.get_by_right(&crate::types::TerminalID(label as usize));
                        match term {
                            Some(Terminal::Literal(bytes)) => {
                                // Show as escaped string
                                let s = String::from_utf8_lossy(bytes);
                                format!("\"{}\"", s.escape_default())
                            },
                            Some(t) => format!("{:?}", t),
                            None => format!("T{}", label),
                        }
                    };
                    // Get weight from target state
                    let target_weight_str = match &orig_dwa_for_min.states[target].final_weight {
                        Some(w) if w.len() <= 10 => format!("{:?}", w.to_rsb_allow_expansion().iter().collect::<Vec<_>>()),
                        Some(w) if w.len() == u64::MAX as usize => "ALL".to_string(),
                        Some(w) => format!("(len={})", w.len()),
                        None => "".to_string(),
                    };
                    let weight_part = if target_weight_str.is_empty() { "".to_string() } else { format!(" [w:{}]", target_weight_str) };
                    crate::debug!(1, "  --[{}]--> state {}{}", label_str, target, weight_part);
                }
            }
            crate::debug!(1, "=== END TERMINAL DWA STRUCTURE ===\n");

            // PRINT GRAPHVIZ DOT - combine labels for edges between same node pairs
            crate::debug!(1, "\n=== TERMINAL DWA DOT ===");
            crate::debug!(1, "digraph TerminalDWA {{");
            crate::debug!(1, "  rankdir=LR;");
            crate::debug!(1, "  node [shape=circle, style=filled, fillcolor=white];");
            crate::debug!(1, "  start [shape=point];");
            crate::debug!(1, "  start -> {};", orig_dwa_for_min.body.start_state);

            // Collect edges grouped by (source, target)
            let mut edge_labels: std::collections::HashMap<(usize, usize), (Vec<String>, bool)> = std::collections::HashMap::new();
            
            for (sid, state) in orig_dwa_for_min.states.0.iter().enumerate() {
                let shape = if state.final_weight.is_some() { "doublecircle" } else { "circle" };
                let node_color = if state.final_weight.is_some() { "lightblue" } else { "white" };
                crate::debug!(1, "  {} [shape={}, fillcolor={}, label=\"{}\"];", sid, shape, node_color, sid);

                for (&label, &target) in &state.transitions {
                    let (label_str, is_tsid) = if label >= parser.terminal_map.len() as i32 {
                        let tsid = (label - parser.terminal_map.len() as i32) as usize;
                        let name = tsid_names.get(&tsid)
                            .cloned()
                            .unwrap_or_else(|| format!("tsid:{}", tsid));
                        (format!("TSID[{}]", name), true)
                    } else {
                        let term = parser.terminal_map.get_by_right(&crate::types::TerminalID(label as usize));
                        let s = match term {
                            Some(Terminal::Literal(bytes)) => {
                                let s = String::from_utf8_lossy(bytes);
                                s.replace("\"", "'").replace("\\", "\\\\")
                            },
                            Some(t) => format!("{:?}", t).replace("\"", "'"),
                            None => format!("T{}", label),
                        };
                        (s, false)
                    };
                    
                    let entry = edge_labels.entry((sid, target)).or_insert_with(|| (vec![], false));
                    entry.0.push(label_str);
                    entry.1 |= is_tsid;
                }
            }
            
            // Output combined edges
            for ((src, tgt), (labels, has_tsid)) in &edge_labels {
                let color = if *has_tsid { "blue" } else { "black" };
                // Combine labels with newlines, truncate if too many
                let combined_label = if labels.len() <= 5 {
                    labels.join("\\n")
                } else {
                    format!("{}\\n...+{} more", labels[..3].join("\\n"), labels.len() - 3)
                };
                crate::debug!(1, "  {} -> {} [label=\"{}\", color={}, fontcolor={}];", 
                    src, tgt, combined_label, color, color);
            }
            crate::debug!(1, "}}");
            crate::debug!(1, "=== END TERMINAL DWA DOT ===\n");
            // end of first DOT dump, but we keep the DUMP_DWA_DOT block open for the second dump
            
            
            let terminal_nwa_orig = NWA::from_dwa(&terminal_dwa);
            let _orig_trans = terminal_dwa.states.num_transitions();
            let _orig_states = terminal_dwa.states.len();
            
            let start_id = terminal_dwa.body.start_state;
            let start_out_degree = terminal_dwa.states[start_id].transitions.len();
            crate::debug!(1, "Original start state has {} outgoing transitions", start_out_degree);
            
            let first_hop_states: std::collections::HashSet<_> = terminal_dwa.states[start_id]
                .transitions.values().cloned().collect();
            crate::debug!(1, "{} unique first-hop states from start", first_hop_states.len());
            
            // Count the out-degree of each first-hop state to understand structure
            let mut first_hop_out_trans = 0;
            let mut second_hop_states: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for &s in &first_hop_states {
                first_hop_out_trans += terminal_dwa.states[s].transitions.len();
                for &t in terminal_dwa.states[s].transitions.values() {
                    second_hop_states.insert(t);
                }
            }
            crate::debug!(1, "{} total outgoing transitions from first-hop states", first_hop_out_trans);
            crate::debug!(1, "{} unique second-hop states", second_hop_states.len());
            
            // KEY INSIGHT: How many second-hop states are shared between first-hop states?
            // This determines the explosion factor!
            let mut second_hop_reachability: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            for &s in &first_hop_states {
                for &t in terminal_dwa.states[s].transitions.values() {
                    *second_hop_reachability.entry(t).or_insert(0) += 1;
                }
            }
            let shared_count = second_hop_reachability.values().filter(|&&v| v > 1).count();
            let max_sharing = second_hop_reachability.values().max().copied().unwrap_or(0);
            crate::debug!(1, "{} second-hop states are reachable from >1 first-hop state (max sharing: {})",
                shared_count, max_sharing);
            
            let mut terminal_nwa_mod = terminal_nwa_orig.clone();
            let start_state = terminal_nwa_mod.body.start_states[0];
            
            let start_trans = std::mem::take(&mut terminal_nwa_mod.states[start_state].transitions);
            let num_eps = start_trans.values().map(|v| v.len()).sum::<usize>();
            for (_, targets) in start_trans {
                for (target, weight) in targets {
                    terminal_nwa_mod.add_epsilon(start_state, target, weight);
                }
            }
            
            crate::debug!(1, "Replaced with {} epsilon transitions", num_eps);
            
            // ANALYZE: What WEIGHTS are on the epsilon transitions?
            // This is crucial - different weights cause subset differentiation
            let eps_with_weights: Vec<_> = terminal_nwa_mod.states[start_state].epsilons.iter().collect();
            let unique_weights: std::collections::HashSet<_> = eps_with_weights.iter().map(|(_, w)| w.len()).collect();
            crate::debug!(1, "Epsilon transitions: {} total, {} unique weight cardinalities", 
                eps_with_weights.len(), unique_weights.len());
            
            // Sample some weights
            let mut weight_samples: Vec<_> = eps_with_weights.iter().take(5)
                .map(|(_, w)| w.len())
                .collect();
            weight_samples.sort();
            crate::debug!(1, "Sample weight cardinalities: {:?}", weight_samples);
            
            // Check if weights are all Weight::all()
            let all_weights_are_all = eps_with_weights.iter().all(|(_, w)| w.is_all_fast());
            crate::debug!(1, "All epsilon weights are Weight::all(): {}", all_weights_are_all);
            
            // ANALYZE: What does the NWA look like after epsilon replacement?
            // Count epsilon targets and their out-degrees
            let eps_targets: std::collections::HashSet<_> = terminal_nwa_mod.states[start_state].epsilons
                .iter().map(|(t, _)| *t).collect();
            crate::debug!(1, "Epsilon targets from start: {} states", eps_targets.len());
            
            // Count labeled transitions from epsilon targets
            let mut labeled_trans_from_eps_targets = 0;
            let mut label_histogram: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
            for &target in &eps_targets {
                for (&label, dsts) in terminal_nwa_mod.states[target].transitions.iter() {
                    labeled_trans_from_eps_targets += dsts.len();
                    *label_histogram.entry(label).or_insert(0) += dsts.len();
                }
            }
            crate::debug!(1, "Total labeled transitions from epsilon targets: {}", labeled_trans_from_eps_targets);
            crate::debug!(1, "Unique labels from epsilon targets: {}", label_histogram.len());
            
            // Find labels that have multiple sources (cause subset explosion)
            let multi_source_labels: Vec<_> = label_histogram.iter()
                .filter(|(_, &count)| count > 1)
                .map(|(&label, &count)| (label, count))
                .collect();
            crate::debug!(1, "Labels with multiple sources: {} (max count: {})",
                multi_source_labels.len(),
                multi_source_labels.iter().map(|(_, c)| *c).max().unwrap_or(0));
            
            // Sample a few high-sharing labels
            let mut sorted_labels: Vec<_> = multi_source_labels.clone();
            sorted_labels.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            for (label, count) in sorted_labels.iter().take(5) {
                crate::debug!(1, "  Label {} has {} source states", label, count);
            }
            
            // DEEP ANALYSIS: For the highest-sharing label, where do the transitions GO?
            if let Some(&(highest_label, source_count)) = sorted_labels.first() {
                let mut targets_for_label: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                let mut weights_for_label: std::collections::HashSet<usize> = std::collections::HashSet::new();
                
                for &eps_target in &eps_targets {
                    if let Some(dsts) = terminal_nwa_mod.states[eps_target].transitions.get(&highest_label) {
                        for (dst, weight) in dsts {
                            *targets_for_label.entry(*dst).or_insert(0) += 1;
                            weights_for_label.insert(weight.len());
                        }
                    }
                }
                
                crate::debug!(1, "For label {} with {} sources:", highest_label, sorted_labels[0].1);
                crate::debug!(1, "  {} unique target states", targets_for_label.len());
                crate::debug!(1, "  {} unique weight cardinalities", weights_for_label.len());
                
                // How many targets are shared?
                let shared_targets = targets_for_label.values().filter(|&&v| v > 1).count();
                let max_target_sharing = targets_for_label.values().max().copied().unwrap_or(0);
                crate::debug!(1, "  {} targets reachable from multiple sources (max: {})", 
                    shared_targets, max_target_sharing);
            }
            
            crate::debug!(1, "Starting determinize...");
            let mut mod_dwa = terminal_nwa_mod.determinize();
            crate::debug!(1, "After determinize: {}", mod_dwa.stats());
            
            // EXPORT DWAs to JSON for Python analysis
            if std::env::var("EXPORT_DWA_JSON").is_ok() {
                let export_dir = std::path::Path::new("temp");
                std::fs::create_dir_all(export_dir).ok();
                
                // Export original terminal DWA
                let orig_path = export_dir.join("terminal_dwa_original.json");
                if let Err(e) = orig_dwa_for_min.export_to_json_file(&orig_path) {
                    crate::debug!(1, "Warning: failed to export original DWA: {}", e);
                } else {
                    crate::debug!(1, "Exported original terminal DWA to {}", orig_path.display());
                }
                
                // Export modified terminal DWA (after epsilon, before minimize)
                let mod_path = export_dir.join("terminal_dwa_modified.json");
                if let Err(e) = mod_dwa.export_to_json_file(&mod_path) {
                    crate::debug!(1, "Warning: failed to export modified DWA: {}", e);
                } else {
                    crate::debug!(1, "Exported modified terminal DWA to {}", mod_path.display());
                }
            }
            
            // ANALYSIS: What does the start state look like after epsilon?
            let mod_start = mod_dwa.body.start_state;
            let mod_start_out = mod_dwa.states[mod_start].transitions.len();
            crate::debug!(1, "Modified start state {} has {} outgoing transitions", mod_start, mod_start_out);
            
            // Count unique target states from start
            let mod_start_targets: std::collections::HashSet<_> = mod_dwa.states[mod_start]
                .transitions.values().cloned().collect();
            crate::debug!(1, "Modified start reaches {} unique states", mod_start_targets.len());
            
            // Sample some states reachable from start
            let mut total_second_hop = 0;
            let mut sample_states = vec![];
            for (i, &target) in mod_start_targets.iter().take(5).enumerate() {
                let out_degree = mod_dwa.states[target].transitions.len();
                total_second_hop += out_degree;
                sample_states.push((target, out_degree));
                crate::debug!(1, "  Sample target {}: state {} has {} outgoing", i, target, out_degree);
            }
            
            if mod_start_targets.len() > 5 {
                // Compute average for all states
                let mut total = 0;
                for &target in &mod_start_targets {
                    total += mod_dwa.states[target].transitions.len();
                }
                let avg = total as f64 / mod_start_targets.len() as f64;
                crate::debug!(1, "  Average outgoing from start targets: {:.1}", avg);
            }
            
            // WEIGHT-BASED ANALYSIS: How many transitions have empty weights when intersected with valid tokens?
            let mut empty_weight_trans = 0;
            let mut nonempty_weight_trans = 0;
            for state in mod_dwa.states.0.iter() {
                for (_, &target) in state.transitions.iter() {
                    let is_empty = match mod_dwa.states[target].final_weight.as_ref() {
                        Some(w) => (&(w & &valid_tokens)).is_empty(),
                        None => false, // No final weight means not constrained
                    };
                    if is_empty {
                        empty_weight_trans += 1;
                    } else {
                        nonempty_weight_trans += 1;
                    }
                }
            }
            crate::debug!(1, "Weight analysis: {} transitions lead to empty weights, {} lead to non-empty",
                empty_weight_trans, nonempty_weight_trans);
            
            // BACKWARD REACHABILITY ANALYSIS with weights
            // Find which states can reach a final state with non-empty accumulated weight
            let mut can_reach_final_nonempty: HashSet<usize> = HashSet::new();
            let mut worklist = VecDeque::new();
            
            // Initialize: final states with non-empty weight
            for (sid, state) in mod_dwa.states.0.iter().enumerate() {
                if let Some(ref fw) = state.final_weight {
                    let intersected = fw & &valid_tokens;
                    if !intersected.is_empty() {
                        can_reach_final_nonempty.insert(sid);
                        worklist.push_back(sid);
                    }
                }
            }
            crate::debug!(1, "Initial final states with non-empty weight: {}", can_reach_final_nonempty.len());
            
            // Backward propagation
            // Build reverse edge map
            let mut reverse_edges: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
            for (sid, state) in mod_dwa.states.0.iter().enumerate() {
                for (_, &target) in state.transitions.iter() {
                    reverse_edges.entry(target).or_default().push(sid);
                }
            }
            
            while let Some(state) = worklist.pop_front() {
                if let Some(predecessors) = reverse_edges.get(&state) {
                    for &pred in predecessors {
                        if !can_reach_final_nonempty.contains(&pred) {
                            can_reach_final_nonempty.insert(pred);
                            worklist.push_back(pred);
                        }
                    }
                }
            }
            crate::debug!(1, "States that can reach final with non-empty weight: {} / {} total",
                can_reach_final_nonempty.len(), mod_dwa.states.len());
            
            // Count useful transitions (both source and target can reach final with non-empty weight)
            let mut useful_trans = 0;
            let mut useless_trans = 0;
            for (sid, state) in mod_dwa.states.0.iter().enumerate() {
                if can_reach_final_nonempty.contains(&sid) {
                    for (_, &target) in state.transitions.iter() {
                        if can_reach_final_nonempty.contains(&target) {
                            useful_trans += 1;
                        } else {
                            useless_trans += 1;
                        }
                    }
                } else {
                    useless_trans += state.transitions.len();
                }
            }
            crate::debug!(1, "Useful transitions: {}, Useless: {} (could be removed)",
                useful_trans, useless_trans);
            
            mod_dwa.minimize();
            let mod_states = mod_dwa.states.len();
            let mod_trans = mod_dwa.states.num_transitions();
            
            let mod_start_id = mod_dwa.body.start_state;
            let mod_start_out = mod_dwa.states[mod_start_id].transitions.len();
            crate::debug!(1, "Modified start has {} outgoing transitions", mod_start_out);
            
            crate::debug!(1, "After minimize_with_rustfst: {}", mod_dwa.stats());
            crate::debug!(1, "TERMINAL DWA: Original {} -> Modified {}",
                orig_dwa_for_min.stats(), mod_dwa.stats());
            crate::debug!(1, "TERMINAL DWA: State factor {:.2}x, Trans factor {:.2}x",
                mod_states as f64 / orig_dwa_for_min.states.len() as f64, 
                mod_trans as f64 / orig_dwa_for_min.states.num_transitions() as f64);
            
            if mod_trans > orig_dwa_for_min.states.num_transitions() {
                crate::debug!(1, "TERMINAL DWA RESULT: EXPLOSION! ({:.2}x expansion)", 
                    mod_trans as f64 / orig_dwa_for_min.states.num_transitions() as f64);
            } else {
                crate::debug!(1, "TERMINAL DWA RESULT: No explosion (reduction or same)");
            }
            
            // PRINT MODIFIED DWA DOT (after epsilon merge + minimize) - combine labels
            // DEBUG: Transition weight inspection - uncomment to verify state non-mergeability
            // States 105, 106, 107 have same final_weight but DIFFERENT trans_weights:
            // - State 105: trans_weight = 0..=11, 13..=∞
            // - State 106: trans_weight = 0..=5, 7..=∞  
            // - State 107: trans_weight = 0..=66, 68..=∞
            // This is why minimization correctly keeps them separate.
            // for sid in [34, 35, 36, 105, 106, 107] {
            //     if sid < mod_dwa.states.len() {
            //         let state = &mod_dwa.states.0[sid];
            //         crate::debug!(1, "State {}: final={:?}", sid, state.final_weight);
            //         for (&label, &target) in &state.transitions {
            //             let weight = state.trans_weights.get(&label);
            //             crate::debug!(1, "  trans {} -> {}: weight={:?}", label, target, weight);
            //         }
            //     }
            // }

            crate::debug!(1, "\n=== MODIFIED TERMINAL DWA DOT ===");
            crate::debug!(1, "digraph ModifiedTerminalDWA {{");
            crate::debug!(1, "  rankdir=LR;");
            crate::debug!(1, "  node [shape=circle, style=filled, fillcolor=white];");
            crate::debug!(1, "  start [shape=point];");
            crate::debug!(1, "  start -> {};", mod_dwa.body.start_state);

            // Collect edges grouped by (source, target)
            let mut edge_labels: std::collections::HashMap<(usize, usize), (Vec<String>, bool)> = std::collections::HashMap::new();

            for (sid, state) in mod_dwa.states.0.iter().enumerate() {
                let shape = if state.final_weight.is_some() { "doublecircle" } else { "circle" };
                let node_color = if state.final_weight.is_some() { "lightblue" } else { "white" };
                crate::debug!(1, "  {} [shape={}, fillcolor={}, label=\"{}\"];", sid, shape, node_color, sid);

                for (&label, &target) in &state.transitions {
                    let (label_str, is_tsid) = if label >= parser.terminal_map.len() as i32 {
                        let tsid = (label - parser.terminal_map.len() as i32) as usize;
                        let name = tsid_names.get(&tsid)
                            .cloned()
                            .unwrap_or_else(|| format!("tsid:{}", tsid));
                        (format!("TSID[{}]", name), true)
                    } else {
                        let term = parser.terminal_map.get_by_right(&crate::types::TerminalID(label as usize));
                        let s = match term {
                            Some(Terminal::Literal(bytes)) => {
                                let s = String::from_utf8_lossy(bytes);
                                s.replace("\"", "'").replace("\\", "\\\\")
                            },
                            Some(t) => format!("{:?}", t).replace("\"", "'"),
                            None => format!("T{}", label),
                        };
                        (s, false)
                    };
                    
                    let entry = edge_labels.entry((sid, target)).or_insert_with(|| (vec![], false));
                    entry.0.push(label_str);
                    entry.1 |= is_tsid;
                }
            }
            
            // Output combined edges
            for ((src, tgt), (labels, has_tsid)) in &edge_labels {
                let color = if *has_tsid { "blue" } else { "black" };
                // Combine labels with newlines, truncate if too many
                let combined_label = if labels.len() <= 5 {
                    labels.join("\\n")
                } else {
                    format!("{}\\n...+{} more", labels[..3].join("\\n"), labels.len() - 3)
                };
                crate::debug!(1, "  {} -> {} [label=\"{}\", color={}, fontcolor={}];", 
                    src, tgt, combined_label, color, color);
            }
            crate::debug!(1, "}}");
            crate::debug!(1, "=== END MODIFIED TERMINAL DWA DOT ===\n");
            } // end DUMP_DWA_DOT block - both DOT dumps complete
        }

        if verify_equivalence {
            crate::debug!(2, "VERIFY_EQUIVALENCE: Running optimize_dwa_and_vocab on terminal_dwa...");
            let vocab_before = vocab.internal_max_llm_token;

            let dwa_partition = compute_dwa_partition(&terminal_dwa, &possible_matches_precompute1, vocab.internal_max_llm_token);
            let actual_classes = dwa_partition.len();

            let expected_classes = mask_classes.len();
            crate::debug!(2, "VERIFY_EQUIVALENCE: DWA partition has {} classes (from {} tokens)", 
                         actual_classes, vocab_before + 1);
            crate::debug!(2, "VERIFY_EQUIVALENCE: Expected {} classes from Simple equivalence analysis", expected_classes);

            if expected_classes != actual_classes {
                crate::debug!(7, "VERIFY_EQUIVALENCE FAILED: Simple={} vs DWA={} classes", expected_classes, actual_classes);

                let sorted_tokens: Vec<_> = internal_llm_token_map.iter().collect();
                let llm_token_strings: Vec<Vec<u8>> = sorted_tokens.iter().map(|(b, _)| (*b).clone()).collect();
                let initial_states: Vec<usize> = tokenizer.iter_states().map(|s| s.0).collect();

                let mut dwa_token_to_class: Vec<usize> = vec![0; llm_token_strings.len()];
                for (class_id, tokens) in dwa_partition.iter().enumerate() {
                    for &tok_id in tokens {
                        if tok_id < dwa_token_to_class.len() {
                            dwa_token_to_class[tok_id] = class_id;
                        }
                    }
                }

                let mut simple_token_to_class: Vec<usize> = vec![0; llm_token_strings.len()];
                for (class_id, indices) in mask_classes.iter().enumerate() {
                    for &idx in indices {
                        if idx < simple_token_to_class.len() {
                            simple_token_to_class[idx] = class_id;
                        }
                    }
                }

                let mut examples_simple_coarser = Vec::new();
                let mut examples_dwa_coarser = Vec::new();

                for i in 0..llm_token_strings.len().min(10000) {
                    for j in (i + 1)..llm_token_strings.len().min(10000) {
                        let simple_same = simple_token_to_class[i] == simple_token_to_class[j];
                        let dwa_same = dwa_token_to_class[i] == dwa_token_to_class[j];

                        if simple_same && !dwa_same && examples_simple_coarser.len() < 5 {
                            examples_simple_coarser.push((i, j));
                        }
                        if !simple_same && dwa_same && examples_dwa_coarser.len() < 5 {
                            examples_dwa_coarser.push((i, j));
                        }
                    }
                }

                if !examples_simple_coarser.is_empty() {
                    crate::debug!(7, "Examples where Simple groups together but DWA separates:");
                    let (i, j) = examples_simple_coarser[0];
                    let s1 = String::from_utf8_lossy(&llm_token_strings[i]);
                    let s2 = String::from_utf8_lossy(&llm_token_strings[j]);
                    crate::debug!(7, "  {:?} (idx {}) and {:?} (idx {})", s1, i, s2, j);

                    let tok_i = &llm_token_strings[i];
                    let tok_j = &llm_token_strings[j];
                    let mut prefix_len = 0;
                    while prefix_len < tok_i.len() && prefix_len < tok_j.len() &&
                          tok_i[prefix_len] == tok_j[prefix_len] {
                        prefix_len += 1;
                    }
                    crate::debug!(7, "    Shared prefix length: {} ({:?})", prefix_len,
                        String::from_utf8_lossy(&tok_i[..prefix_len]));

                    for &init_state in initial_states.iter().take(5) {
                        let mut curr = init_state;
                        let mut dead = false;
                        for &byte in &tok_i[..prefix_len] {
                            if let Some(&next) = tokenizer.dfa().states[curr].transitions.get(byte) {
                                curr = next;
                            } else {
                                dead = true;
                                break;
                            }
                        }
                        if dead { continue; }

                        let state_after_prefix = curr;

                        let mut curr_i = state_after_prefix;
                        let mut dead_i = false;
                        for &byte in &tok_i[prefix_len..] {
                            if let Some(&next) = tokenizer.dfa().states[curr_i].transitions.get(byte) {
                                curr_i = next;
                            } else {
                                dead_i = true;
                                break;
                            }
                        }

                        let mut curr_j = state_after_prefix;
                        let mut dead_j = false;
                        for &byte in &tok_j[prefix_len..] {
                            if let Some(&next) = tokenizer.dfa().states[curr_j].transitions.get(byte) {
                                curr_j = next;
                            } else {
                                dead_j = true;
                                break;
                            }
                        }

                        let accessible_i = if dead_i { Vec::new() } else {
                            tokenizer.dfa().states[curr_i].possible_future_group_ids.iter().copied().collect::<Vec<_>>()
                        };
                        let accessible_j = if dead_j { Vec::new() } else {
                            tokenizer.dfa().states[curr_j].possible_future_group_ids.iter().copied().collect::<Vec<_>>()
                        };

                        let exec_i = tokenizer.execute_from_state(tok_i, TokenizerStateID(init_state));
                        let exec_j = tokenizer.execute_from_state(tok_j, TokenizerStateID(init_state));
                        let matches_i: Vec<(usize, usize)> = exec_i.matches.iter().map(|m| (m.id, m.width)).collect();
                        let matches_j: Vec<(usize, usize)> = exec_j.matches.iter().map(|m| (m.id, m.width)).collect();

                        crate::debug!(7, "    From init_state {}, after prefix {:?}:", init_state,
                            String::from_utf8_lossy(&tok_i[..prefix_len]));
                        crate::debug!(7, "      {:?} suffix {:?}: dead={}, final={}, accessible={:?}, MATCHES={:?}",
                            s1, String::from_utf8_lossy(&tok_i[prefix_len..]), dead_i, curr_i, accessible_i, matches_i);
                        crate::debug!(7, "      {:?} suffix {:?}: dead={}, final={}, accessible={:?}, MATCHES={:?}",
                            s2, String::from_utf8_lossy(&tok_j[prefix_len..]), dead_j, curr_j, accessible_j, matches_j);
                    }
                }
                if !examples_dwa_coarser.is_empty() {
                    crate::debug!(7, "Examples where DWA groups together but Simple separates:");
                    for (i, j) in &examples_dwa_coarser {
                        let s1 = String::from_utf8_lossy(&llm_token_strings[*i]);
                        let s2 = String::from_utf8_lossy(&llm_token_strings[*j]);
                        crate::debug!(7, "  {:?} (idx {}) and {:?} (idx {})", s1, i, s2, j);
                    }
                }

                panic!("VERIFY_EQUIVALENCE FAILED: Simple equivalence produced {} classes, but DWA partition produced {} classes. \
                       Difference: {} (Simple is {})",
                       expected_classes, actual_classes,
                       (expected_classes as isize - actual_classes as isize).abs(),
                       if expected_classes < actual_classes { "too coarse (under-discriminating)" } else { "too fine (over-discriminating)" });
            }
            crate::debug!(2, "✓ VERIFY_EQUIVALENCE PASSED: Simple equivalence matches DWA partition ({} classes)", expected_classes);
        }
        // Normal mode: Skip vocab optimization on terminal_dwa (optimization happens on parser_dwa below)

        if dump_terminal_automata {
            println!("\n--- Terminal DWA ---");
            println!("{}", terminal_dwa);
        }

        // Build Parser DWA
        let max_internal_llm_token_id = vocab.internal_max_llm_token;
        // Note: vocab.internal_max_llm_token might have changed due to optimization, which is fine.

        // Post-terminal-DWA reorder: reorder token and tsid dimensions in the terminal DWA
        // to minimize range fragmentation. This makes parser DWA construction faster and
        // gives the parser DWA better initial ordering.
        let terminal_reorder_enabled = weight_heavy_enabled
            && std::env::var("ENABLE_TERMINAL_DWA_REORDER")
                .map(|v| v != "0")
                .unwrap_or(true);
        if terminal_reorder_enabled && num_tsids > 0 {
            crate::debug!(3, "Reordering terminal DWA dimensions...");
            let reorder_start = std::time::Instant::now();
            let (token_perm, tsid_perm) = crate::dwa_i32::reorder::reorder_dwa_dimensions(
                &mut terminal_dwa,
                max_internal_llm_token_id,
                num_tsids,
            );
            crate::timing!("TIMING: terminal_dwa_reorder {:?}", reorder_start.elapsed());

            // Apply tsid_perm to state_to_internal_tsid values.
            // tsid_perm operates on internal space (0..num_tsids-1).
            // No state_to_rep or representative_states changes needed — those stay in raw state IDs.
            for internal in state_to_internal_tsid.iter_mut() {
                *internal = tsid_perm[*internal];
            }

            // Apply token_perm to internal_llm_token_map
            let mut new_internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
            for (bytes, old_id) in &internal_llm_token_map {
                new_internal_llm_token_map.insert(bytes.clone(), LLMTokenID(token_perm[old_id.0]));
            }
            internal_llm_token_map = new_internal_llm_token_map;

            // Update original_to_internal
            for (_orig, int_id) in vocab.original_to_internal.iter_mut() {
                *int_id = token_perm[*int_id];
            }

            // Rebuild internal_to_original
            let max_int = vocab.internal_max_llm_token;
            let mut groups: Vec<Vec<usize>> = vec![Vec::new(); max_int + 1];
            for (orig, int_id) in &vocab.original_to_internal {
                groups[*int_id].push(*orig);
            }
            vocab.internal_to_original = groups
                .into_iter()
                .enumerate()
                .filter(|(_, v)| !v.is_empty())
                .map(|(int_id, origs)| (int_id, LLMTokenBV::from_iter(origs)))
                .collect();

            crate::debug!(
                3,
                "Terminal DWA reorder complete in {:?}",
                reorder_start.elapsed(),
            );

            // Update possible_matches: permute LLMTokenBV values to new internal token IDs
            for state_matches in possible_matches_precompute1.values_mut() {
                for bv in state_matches.values_mut() {
                    let old_rsb = bv.inner.as_ref().clone();
                    let mut new_rsb = range_set_blaze::RangeSetBlaze::new();
                    for r in old_rsb.ranges() {
                        for old_id in *r.start()..=*r.end() {
                            if old_id < token_perm.len() {
                                new_rsb.insert(token_perm[old_id]);
                            }
                        }
                    }
                    *bv = crate::datastructures::hybrid_bitset::RangeSet::from(new_rsb);
                }
            }
        }

        // Convert the lexical DWA to NWA and build the Parser DWA.
        crate::timing!(
            "PHASE_TIMING: post_precompute1_to_parser_dwa_start = {:?}",
            post_precompute1_start.elapsed()
        );
        crate::debug!(3, "Building Parser DWA");
        crate::debug!(5, "terminal_nwa_from_dwa: start");
        let terminal_nwa_start = std::time::Instant::now();
        let terminal_nwa = timeit!("terminal_nwa_from_dwa", {
            NWA::from_dwa(&terminal_dwa)
        });
        crate::timing!(
            "PHASE_TIMING: terminal_nwa_from_dwa = {:?}",
            terminal_nwa_start.elapsed()
        );
        crate::timing!(
            "TIMING: terminal_nwa_from_dwa {:?}",
            terminal_nwa_start.elapsed()
        );
        crate::debug!(5, "terminal_nwa_from_dwa: end");
        crate::debug!(5, "build_parser_dwa: start");
        let parser_dwa_start = std::time::Instant::now();
        let mut parser_dwa = build_parser_dwa(&parser, &terminal_nwa);
        crate::timing!(
            "PHASE_TIMING: build_parser_dwa = {:?}",
            parser_dwa_start.elapsed()
        );
        crate::timing!("TIMING: build_parser_dwa {:?}", parser_dwa_start.elapsed());
        crate::debug!(5, "build_parser_dwa: end");

        if std::env::var("VERIFY_PARSER_DWA_MINIMIZE").is_ok() {
            let before_states = parser_dwa.states.len();
            let before_transitions = parser_dwa.states.num_transitions();
            let mut check = parser_dwa.clone();
            check.minimize();
            let after_states = check.states.len();
            let after_transitions = check.states.num_transitions();
            eprintln!(
                "VERIFY_PARSER_DWA_MINIMIZE: states {} -> {}, transitions {} -> {}",
                before_states,
                after_states,
                before_transitions,
                after_transitions
            );
            if before_states != after_states || before_transitions != after_transitions {
                eprintln!(
                    "VERIFY_PARSER_DWA_MINIMIZE: WARNING - minimize not idempotent"
                );
            }
        }

        // EPSILON EXPLOSION EXPERIMENT - Parser DWA from epsilon terminal NWA
        // Test: Build Parser DWA from the epsilon-modified terminal NWA
        // to see if Parser DWA build time is faster/slower
        if std::env::var("TEST_EPSILON_EXPLOSION").is_ok() {
            use crate::dwa_i32::nwa::NWA;
            use std::time::Instant;
            
            crate::debug!(1, "=== EPSILON EXPLOSION EXPERIMENT: Parser DWA from epsilon terminal NWA ===");
            crate::debug!(1, "Original Parser DWA build time: <see profiler summary>");
            crate::debug!(1, "Original terminal DWA: {}",
                terminal_dwa.stats());
            
            // First, minimize the original parser DWA to verify baseline
            let mut orig_parser_dwa_for_min = parser_dwa.clone();
            orig_parser_dwa_for_min.minimize();
            crate::debug!(1, "Original Parser DWA (after minimize_with_rustfst): {}",
                orig_parser_dwa_for_min.stats());
            
            // First, create the epsilon-modified terminal DWA (same as terminal DWA experiment)
            let mut terminal_nwa_mod = terminal_nwa.clone();
            let start_state = terminal_nwa_mod.body.start_states[0];
            
            let terminals_count = parser.terminal_map.len();
            
            // Replace labeled start transitions with epsilons, keeping track of the first one
            let start_trans = std::mem::take(&mut terminal_nwa_mod.states[start_state].transitions);
            let mut first_target_saved: Option<(crate::dwa_i32::common::StateID, crate::dwa_i32::Weight)> = None;
            let mut first_label_saved: Option<crate::dwa_i32::common::Label> = None;
            
            for (label, targets) in start_trans {
                for (target, weight) in targets {
                    if first_target_saved.is_none() {
                        // Save the first one to add back as a labeled transition
                        first_target_saved = Some((target, weight.clone()));
                        first_label_saved = Some(label);
                        // Add the labeled transition back (keep tsid 0)
                        terminal_nwa_mod.add_transition(start_state, label, target, weight).unwrap();
                    } else {
                        // All others become epsilon transitions
                        terminal_nwa_mod.add_epsilon(start_state, target, weight);
                    }
                }
            }
            
            let num_eps = terminal_nwa_mod.states[start_state].epsilons.len();
            crate::debug!(1, "Terminal NWA: Kept 1 labeled transition (label {}), replaced rest with {} epsilon transitions",
                first_label_saved.unwrap_or(0), num_eps);
            
            // Determinize and minimize the modified terminal NWA
            crate::debug!(1, "Determinizing and minimizing modified terminal NWA...");
            let det_start = Instant::now();
            let mut terminal_dwa_mod = terminal_nwa_mod.determinize();
            terminal_dwa_mod.minimize();
            crate::debug!(1, "Modified terminal DWA (after minimize): {} (took {:?})",
                terminal_dwa_mod.stats(), det_start.elapsed());
            
            // Now build Parser DWA from this modified terminal NWA
            let terminal_nwa_for_parser = NWA::from_dwa(&terminal_dwa_mod);
            crate::debug!(1, "Building Parser DWA from modified terminal NWA...");
            let parser_start = Instant::now();
            let mut parser_dwa_mod = build_parser_dwa(&parser, &terminal_nwa_for_parser);
            let build_time = parser_start.elapsed();
            crate::debug!(1, "Modified Parser DWA (before minimize): {} (took {:?})",
                parser_dwa_mod.stats(), build_time);
            
            parser_dwa_mod.minimize();
            crate::debug!(1, "Modified Parser DWA (after minimize): {}",
                parser_dwa_mod.stats());
            
            // Compare with original (minimized)
            let orig_term_min = {
                let mut t = terminal_dwa.clone();
                t.minimize();
                t
            };
            
            crate::debug!(1, "COMPARISON (all minimized):");
            crate::debug!(1, "  Original terminal DWA: {}",
                orig_term_min.stats());
            crate::debug!(1, "  Modified terminal DWA: {}",
                terminal_dwa_mod.stats());
            crate::debug!(1, "  Original Parser DWA: {}",
                orig_parser_dwa_for_min.stats());
            crate::debug!(1, "  Modified Parser DWA: {}",
                parser_dwa_mod.stats());
        }

        // Check if weight-heavy mode is enabled via environment variable
        let weight_heavy = crate::constraint_precompute::is_weight_heavy_enabled();
        // num_tsids uses the representative count (same as earlier in the build)
        let num_tsids = if weight_heavy {
            representative_states.len()
        } else {
            0
        };
        
        // Compute domain_max for weight trimming
        let domain_max = if weight_heavy {
            let n = vocab.internal_max_llm_token;
            let m = num_tsids;
            n.saturating_mul(m).saturating_add(m.saturating_sub(1))
        } else {
            vocab.internal_max_llm_token
        };
        
        // Trim weights to domain_max to remove unnecessary range extensions to usize::MAX
        crate::debug!(5, "parser_dwa.trim_weights_to_domain: start");
        let trim_start = std::time::Instant::now();
        timeit!("parser_dwa.trim_weights_to_domain", {
            parser_dwa.trim_weights_to_domain(domain_max);
        });
        crate::timing!(
            "TIMING: parser_dwa_trim_weights {:?}",
            trim_start.elapsed()
        );
        crate::debug!(5, "parser_dwa.trim_weights_to_domain: end");
        
        // Symbol-heavy mode: apply clip_weights and optimize_dwa_and_vocab
        // These optimizations assume N-space weights and reduce the token space
        if !weight_heavy {
            let optimize_start = std::time::Instant::now();
            parser_dwa.states.clip_weights(vocab.internal_max_llm_token);
            optimize_dwa_and_vocab(&mut parser_dwa, &mut vocab, &mut possible_matches_precompute1);
            crate::timing!(
                "TIMING: optimize_dwa_and_vocab {:?}",
                optimize_start.elapsed()
            );
        }

        // Weight complexity instrumentation for the parser DWA.
        if crate::r#macro::is_debug_level_enabled(5) {
            crate::debug!(5, "Parser DWA weight complexity (unique): total_ranges_unique={} (total_ranges_all={})", 
                parser_dwa.num_ranges_interned(),
                parser_dwa.num_ranges(),
            );
        }

        // state_to_internal_tsid was already built before precompute1 and updated
        // by terminal DWA reorder. No need to rebuild it here.

        // Post-DWA dimension reordering: permute token and tsid IDs to minimize
        // total range counts in the weight pool.
        // By default, skip if pre-DWA reorder was already applied.
        // Can be forced on with ENABLE_DWA_REORDER=1.
        let reorder_env = std::env::var("ENABLE_DWA_REORDER").ok();
        let reorder_enabled = weight_heavy && match &reorder_env {
            Some(v) if v == "1" => true,  // explicit on: always run
            Some(v) if v == "0" => false, // explicit off: never run
            _ => !did_pre_dwa_reorder,    // default: run only if pre-DWA didn't run
        };
        if reorder_enabled && num_tsids > 0 {
            crate::debug!(3, "Reordering parser DWA dimensions...");
            let reorder_start = std::time::Instant::now();
            let (token_perm, tsid_perm) = crate::dwa_i32::reorder::reorder_dwa_dimensions(
                &mut parser_dwa,
                vocab.internal_max_llm_token,
                num_tsids,
            );
            crate::timing!("TIMING: reorder_dwa_dimensions {:?}", reorder_start.elapsed());

            // Update state_to_internal_tsid: compose with tsid permutation
            // state → old_internal → new_internal
            for internal_tsid in state_to_internal_tsid.iter_mut() {
                *internal_tsid = tsid_perm[*internal_tsid];
            }

            // Update original_to_internal vocab mapping: compose with token permutation
            // original → old_internal → new_internal
            for internal_id in vocab.original_to_internal.values_mut() {
                *internal_id = token_perm[*internal_id];
            }
            // Update internal_to_original: rebuild from the updated original_to_internal
            vocab.internal_to_original.clear();
            for (&original, &internal) in &vocab.original_to_internal {
                vocab.internal_to_original
                    .entry(internal)
                    .or_insert_with(|| crate::datastructures::hybrid_bitset::RangeSet::default())
                    .insert(original);
            }

            // Update possible_matches: permute LLMTokenBV values to new internal token IDs
            for state_matches in possible_matches_precompute1.values_mut() {
                for bv in state_matches.values_mut() {
                    let old_rsb = bv.inner.as_ref().clone();
                    let mut new_rsb = range_set_blaze::RangeSetBlaze::new();
                    for r in old_rsb.ranges() {
                        for old_id in *r.start()..=*r.end() {
                            if old_id < token_perm.len() {
                                new_rsb.insert(token_perm[old_id]);
                            }
                        }
                    }
                    *bv = crate::datastructures::hybrid_bitset::RangeSet::from(new_rsb);
                }
            }

            crate::debug!(3, "DWA dimension reorder complete in {:?}", reorder_start.elapsed());
        }

        crate::debug!(5, "build_internal_to_original_sparse_matrix: start");
        let sparse_start = std::time::Instant::now();
        let internal_to_original_sparse_matrix = timeit!(
            "build_internal_to_original_sparse_matrix",
            {
                StageVocab::build_internal_to_original_sparse_matrix(
                    &vocab.internal_to_original,
                    max_original_llm_token_id,
                    vocab.internal_max_llm_token,
                )
            }
        );
        crate::timing!(
            "TIMING: build_internal_to_original_sparse_matrix {:?}",
            sparse_start.elapsed()
        );
        crate::debug!(5, "build_internal_to_original_sparse_matrix: end");
        vocab.max_original_llm_token_id = max_original_llm_token_id;
        vocab.internal_to_original_sparse_matrix = internal_to_original_sparse_matrix;

        // Build the new trie-based vocab from the LLM token map
        crate::debug!(5, "LLMVocabTrie::from_token_map: start");
        let vocab_trie_start = std::time::Instant::now();
        let vocab_trie = Arc::new(timeit!("LLMVocabTrie::from_token_map", {
            LLMVocabTrie::from_token_map(&llm_token_map)
        }));
        crate::timing!("TIMING: build_vocab_trie {:?}", vocab_trie_start.elapsed());
        crate::debug!(5, "LLMVocabTrie::from_token_map: end");
        let eos_token_id = find_eos_token_id(&vocab_trie);
        
        crate::debug!(5, "GrammarConstraint build_with_config_inner: end");
        #[allow(deprecated)]
        GrammarConstraint {
            tokenizer,
            parser,
            parser_dwa,
            possible_matches: possible_matches_precompute1,
            terminal_to_greedy_group,
            vocab_trie,
            commit_vocab,
            token_name_map,
            parser_dwa_vocab: vocab,
            eos_token_id,
            num_tsids,
            state_to_internal_tsid,
            cached_tsid_masks: OnceLock::new(),
        }
    }

    pub fn dump_vocab(&self) {
        println!("\n--- Parser DWA Vocab ---");
        println!("Internal to original mapping:");
        for (i, s) in self.parser_dwa_vocab.internal_to_original.iter() {
            println!("  {}: {:?}", i, s);
        }
    }
    
    /// Check if this constraint is in weight-heavy mode.
    pub fn is_weight_heavy(&self) -> bool {
        self.num_tsids > 0
    }

    pub(crate) fn mutually_greedy_group_terminals(&self, terminal_id: usize) -> Vec<usize> {
        let Some(Some(group_idx)) = self.terminal_to_greedy_group.get(terminal_id) else {
            return vec![terminal_id];
        };

        self.terminal_to_greedy_group
            .iter()
            .enumerate()
            .filter_map(|(tid, group)| {
                if *group == Some(*group_idx) {
                    Some(tid)
                } else {
                    None
                }
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Special precomputation
    // -----------------------------------------------------------------------

    /// Dump the Parser DWA for debugging.
    pub fn dump_parser_dwa(&self) {
        println!("\n--- Parser DWA ---");
        println!("{}", self.parser_dwa);
    }
    
    /// Deprecated alias for dump_parser_dwa
    #[deprecated(since = "0.3.0", note = "Use dump_parser_dwa instead")]
    pub fn dump_precomputed4(&self) {
        self.dump_parser_dwa();
    }

    // -----------------------------------------------------------------------
    // Vocab helpers
    // -----------------------------------------------------------------------

    pub fn all_internal_llm_tokens_bitset(&self) -> LLMTokenBV {
        LLMTokenBV::ones(self.parser_dwa_vocab.internal_max_llm_token + 1)
    }

    pub fn internal_bv_to_original(&self, internal_bv: &LLMTokenBV) -> Bitset {
        self.parser_dwa_vocab.internal_bv_to_original(internal_bv)
    }

    pub fn original_bv_to_internal(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        self.parser_dwa_vocab.original_bv_to_internal(original_bv)
    }

    pub fn internal_to_original(&self, internal_id: LLMTokenID) -> Option<LLMTokenID> {
        self.parser_dwa_vocab
            .internal_to_original
            .get(&internal_id.0)
            .and_then(|bv| bv.iter_up_to(self.parser_dwa_vocab.internal_max_llm_token).next())
            .map(|v| LLMTokenID(v))
    }

    #[inline]
    pub fn original_id_to_internal(&self, original_id: LLMTokenID) -> Option<LLMTokenID> {
        self.parser_dwa_vocab
            .original_to_internal
            .get(&original_id.0)
            .map(|v| LLMTokenID(*v))
    }

    // -----------------------------------------------------------------------
    // Possible-matches-related helpers
    // -----------------------------------------------------------------------

    /// Optimized version that only computes for representative states
    pub fn build_maps_and_matches_for_reps(
        tokenizer: &Tokenizer,
        vocab_root: &VocabPrefixTreeNode,
        group_id_to_terminal_idx: &BTreeMap<usize, usize>,
        representative_states: &[usize],
    ) -> BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>> {
        let start_time = std::time::Instant::now();
        
        // Build mapping from representative state to index in our output array
        let rep_to_idx: BTreeMap<usize, usize> = representative_states
            .iter()
            .enumerate()
            .map(|(idx, &state)| (state, idx))
            .collect();
        let num_reps = representative_states.len();
        
        // Flatten DFA for fast access (u32::MAX represents None)
        struct FastDFA {
            transitions: Vec<u32>,           // size = num_states * 256
            finalizers: Vec<Vec<TerminalID>>, // size = num_states
        }

        // Recursive DFS helper - uses sparse HashMap storage
        fn process_vocab_node_dfs_sparse(
            node: &VocabPrefixTreeNode,
            current_states: Vec<(u32, Vec<u32>)>,
            dfa: &FastDFA,
            out_matches: &mut HashMap<(u32, u32), LLMTokenBV>,
        ) {
            for (edge_bytes, child) in node.iter_children() {
                let reachable_bv: LLMTokenBV = child.reachable_token_ids().into();
                let mut next_grouped_map: HashMap<u32, Vec<u32>> = HashMap::new();

                for (start, sources) in &current_states {
                    let mut curr = *start;
                    let mut valid = true;
                    let mut triggered_terminals: Vec<TerminalID> = Vec::new();

                    for &b in edge_bytes.iter() {
                        let offset = (curr as usize) * 256 + (b as usize);
                        let next = dfa.transitions[offset];
                        if next == u32::MAX {
                            valid = false;
                            break;
                        }
                        curr = next;

                        let fins = &dfa.finalizers[curr as usize];
                        if !fins.is_empty() {
                            triggered_terminals.extend_from_slice(fins);
                        }
                    }

                    if valid {
                        if !triggered_terminals.is_empty() {
                            triggered_terminals.sort_unstable();
                            triggered_terminals.dedup();
                            for &src in sources {
                                for &tid in &triggered_terminals {
                                    let key = (src, tid.0 as u32);
                                    out_matches.entry(key)
                                        .and_modify(|existing| existing.union_with(&reachable_bv))
                                        .or_insert_with(|| reachable_bv.clone());
                                }
                            }
                        }
                        next_grouped_map.entry(curr).or_default().extend_from_slice(sources);
                    }
                }

                let next_grouped: Vec<(u32, Vec<u32>)> =
                    next_grouped_map.into_iter().collect();

                if !next_grouped.is_empty() {
                    process_vocab_node_dfs_sparse(
                        child,
                        next_grouped,
                        dfa,
                        out_matches,
                    );
                }
            }
        }

        let dfa = tokenizer.dfa();
        let num_states = dfa.states.len();
        let build_fast_dfa_start = std::time::Instant::now();
        let mut transitions = vec![u32::MAX; num_states * 256];
        let mut finalizers = Vec::with_capacity(num_states);

        let mut max_terminal_idx = 0;
        for (i, state) in dfa.states.iter().enumerate() {
            for (byte, &next) in &state.transitions {
                transitions[i * 256 + (byte as usize)] = next as u32;
            }
            let mut state_fins = Vec::new();
            for gid in &state.finalizers {
                // Convert group_id to terminal_index using the mapping
                if let Some(&terminal_idx) = group_id_to_terminal_idx.get(&gid) {
                    if terminal_idx > max_terminal_idx { max_terminal_idx = terminal_idx; }
                    state_fins.push(TerminalID(terminal_idx));
                }
                // If group_id is not in the mapping, it's not a grammar terminal, skip it
            }
            finalizers.push(state_fins);
        }

        let fast_dfa = FastDFA {
            transitions,
            finalizers,
        };
        let build_fast_dfa_time = build_fast_dfa_start.elapsed();

        let num_terminals = max_terminal_idx + 1;
        
        crate::debug!(4, "build_maps_and_matches_for_reps: num_states={}, num_reps={}, num_terminals={}",
            num_states, num_reps, num_terminals);

        // Group initial states by current state - only include representative states
        // sources are indices into representative_states (0..num_reps)
        let mut initial_states: Vec<(u32, Vec<u32>)> = Vec::with_capacity(num_reps);
        for (rep_idx, &rep_state) in representative_states.iter().enumerate() {
            initial_states.push((rep_state as u32, vec![rep_idx as u32]));
        }

        // Parallel processing of root children using sparse storage
        let root_children: Vec<_> = vocab_root.iter_children().collect();
        let dfs_start = std::time::Instant::now();
        let final_matches_map: HashMap<(u32, u32), LLMTokenBV> = root_children
            .par_iter()
            .map(|(edge_bytes, child_node)| {
                let mut local_matches: HashMap<(u32, u32), LLMTokenBV> = HashMap::new();

                // Manually process the first edge to bootstrap recursion
                let mut next_grouped_map: HashMap<u32, Vec<u32>> = HashMap::new();
                let reachable_bv: LLMTokenBV =
                    child_node.reachable_token_ids().into();

                for (start, sources) in &initial_states {
                    let mut curr = *start;
                    let mut valid = true;
                    let mut triggered_terminals = Vec::new();

                    for &b in edge_bytes.iter() {
                        let offset = (curr as usize) * 256 + (b as usize);
                        let next = fast_dfa.transitions[offset];
                        if next == u32::MAX {
                            valid = false;
                            break;
                        }
                        curr = next;
                        if !fast_dfa.finalizers[curr as usize].is_empty() {
                            triggered_terminals.extend_from_slice(&fast_dfa.finalizers[curr as usize]);
                        }
                    }
                    if valid {
                        if !triggered_terminals.is_empty() {
                            triggered_terminals.sort_unstable();
                            triggered_terminals.dedup();
                            for &src in sources {
                                for &tid in &triggered_terminals {
                                    let key = (src, tid.0 as u32);
                                    local_matches.entry(key)
                                        .and_modify(|existing| existing.union_with(&reachable_bv))
                                        .or_insert_with(|| reachable_bv.clone());
                                }
                            }
                        }
                        next_grouped_map.entry(curr).or_default().extend_from_slice(sources);
                    }
                }

                let next_grouped: Vec<(u32, Vec<u32>)> =
                    next_grouped_map.into_iter().collect();
                if !next_grouped.is_empty() {
                    process_vocab_node_dfs_sparse(
                        child_node,
                        next_grouped,
                        &fast_dfa,
                        &mut local_matches,
                    );
                }

                local_matches
            })
            .reduce(
                HashMap::new,
                |mut a, b| {
                    for (key, val_bv) in b {
                        a.entry(key)
                            .and_modify(|existing| existing.union_with(&val_bv))
                            .or_insert(val_bv);
                    }
                    a
                },
            );
        let dfs_time = dfs_start.elapsed();

        let assemble_start = std::time::Instant::now();
        let mut possible_matches: BTreeMap<
            TokenizerStateID,
            BTreeMap<TerminalID, LLMTokenBV>,
        > = BTreeMap::new();

        for ((rep_idx, tid), bv) in final_matches_map {
            let src = TokenizerStateID(representative_states[rep_idx as usize]);
            possible_matches
                .entry(src)
                .or_default()
                .insert(TerminalID(tid as usize), bv);
        }

        let assemble_time = assemble_start.elapsed();

        crate::debug!(4, "build_maps_and_matches_for_reps: completed in {:?}", start_time.elapsed());
        let total_time = start_time.elapsed();
        let accounted = build_fast_dfa_time + dfs_time + assemble_time;
        let unaccounted = total_time.saturating_sub(accounted);
        crate::debug!(5, "build_maps_and_matches_for_reps breakdown: fast_dfa={:?}, dfs={:?}, assemble={:?}, unaccounted={:?}",
            build_fast_dfa_time, dfs_time, assemble_time, unaccounted);
        possible_matches
    }

    pub fn rearrange_possible_matches(
        pm: &BTreeMap<TokenizerStateID, BTreeMap<TerminalID, LLMTokenBV>>,
    ) -> DedupValueMap<LLMTokenID, BTreeMap<TokenizerStateID, TerminalBV>> {
        let mut triples: Vec<(u32, u32, u32)> = pm
            .par_iter()
            .flat_map(|(sid, tmap)| {
                let mut local_triples = Vec::new();
                for (term, bv) in tmap {
                    if !bv.is_all() {
                        for tok in bv.iter_up_to(usize::MAX) {
                            local_triples
                                .push((tok as u32, sid.0 as u32, term.0 as u32));
                        }
                    }
                }
                local_triples
            })
            .collect();

        triples.par_sort_unstable_by_key(|t| t.0);

        let mut out = DedupValueMap::new();
        if triples.is_empty() {
            return out;
        }

        let mut current_token = triples[0].0;
        let mut current_map: BTreeMap<TokenizerStateID, TerminalBV> = BTreeMap::new();

        for (tok, sid, term) in triples {
            if tok != current_token {
                out.insert(LLMTokenID(current_token as usize), current_map);
                current_map = BTreeMap::new();
                current_token = tok;
            }
            current_map
                .entry(TokenizerStateID(sid as usize))
                .or_default()
                .insert(term as usize);
        }
        out.insert(LLMTokenID(current_token as usize), current_map);

        out
    }

    // -----------------------------------------------------------------------
    // Top-level state construction
    // -----------------------------------------------------------------------

    pub fn init(&self) -> GrammarConstraintState<'_> {
        let mut state = BTreeMap::new();
        state.insert(
            self.tokenizer.initial_state_id(),
            self.parser.init_glr_parser(None),
        );
        GrammarConstraintState { parent: self, state }
    }

    pub fn state_with_nodes(
        &self,
        _nodes: Vec<(usize, Arc<GSSNode>)>,
    ) -> GrammarConstraintState<'_> {
        todo!()
    }

    pub fn state_from_gss_map(
        &self,
        gss_map: &BTreeMap<TokenizerStateID, GSSNode>,
    ) -> GrammarConstraintState {
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
        _roots: &Vec<Arc<GSSNode>>,
        _labels: Option<&[String]>,
    ) {
        // Unimplemented
    }
}

// ---------------------------------------------------------------------------
// Merge implementation for leveled GSS
// ---------------------------------------------------------------------------

impl Merge for RangeSetBlaze<usize> {
    fn merge(&self, other: &Self) -> Self { self | other }
}

impl Merge for RangeSet {
    fn merge(&self, other: &Self) -> Self { self | other }
}

impl Merge for Weight {
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
        writeln!(
            f,
            "GrammarConstraintState ({} active tokenizer states):",
            self.state.len()
        )?;
        Ok(())
    }
}

impl<'a> GrammarConstraintState<'a> {
    fn mutually_greedy_group(&self, terminal_id: &usize) -> Vec<usize> {
        self.parent.mutually_greedy_group_terminals(*terminal_id)
    }

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
            s.stack = f(&mut Arc::new(s.stack.clone()), &mut memo)
                .as_ref()
                .clone();
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
                for terminal in self.mutually_greedy_group(&token.id) {
                    terminals.insert(terminal);
                }
            }
            terminals_map.insert(*tokenizer_state_id, terminals);
        }
        (state_map, terminals_map)
    }

    /// Returns the allowed token mask as a sparse range-based set.
    ///
    /// Note: For most use cases, prefer `get_mask()` which returns a dense `Bitset`
    /// that is more efficient for ML framework integration.
    #[deprecated(since = "0.2.0", note = "Use get_mask() which returns a dense Bitset. This method will be removed in a future version.")]
    pub fn get_mask_rangeset(&self) -> LLMTokenBV {
        self.get_mask().into()
    }

    pub fn print_gss_stats(&self) {
        // Unimplemented
    }

    pub fn print_gss(&self) {
        print!("{}", self.format_gss());
    }

    pub fn format_gss(&self) -> String {
        use std::fmt::Write;
        let mut memo = HashSet::new();
        let mut out = String::new();
        for (tsid, state) in self.state.iter() {
            writeln!(out, "Tokenizer State ID: {:?}", tsid).unwrap();
            writeln!(out, "{}", state.stack.to_graph_string_with_memo(&mut memo, false)).unwrap();
        }
        out
    }

    pub fn explain_stack(&self) {
        // Unimplemented
    }

    pub fn num_unique_nodes(&self) -> usize {
        0
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) -> Result<(), String> {
        let token_bytes = self
            .parent
            .vocab_trie
            .token_bytes(llm_token_id)
            .ok_or_else(|| format!("LLM token ID {} not found in vocabulary trie", llm_token_id.0))?;
        self.commit_bytes(token_bytes);
        Ok(())
    }

    pub fn is_active(&self) -> bool { !self.state.is_empty() }

    pub fn is_valid(&self) -> bool {
        if self.state.is_empty() {
            return false;
        }
        for (tid, glr_state) in self.state.iter() {
            for gtid in
                self.parent
                    .tokenizer
                    .tokens_accessible_from_state(TokenizerStateID(tid.0))
            {
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

    /// Check if the current state represents a complete valid parse.
    /// Returns true if the generated text so far is a complete valid string
    /// in the grammar, meaning EOS should be a valid token.
    pub fn is_complete(&self) -> bool {
        let initial_tsid = self.parent.tokenizer.initial_state_id();
        let parser_state = match self.state.get(&initial_tsid) {
            Some(ps) => ps,
            None => return false,
        };

        if parser_state.stack.is_empty() {
            return false;
        }

        let table = &self.parent.parser.table;

        // Collect stack top states into a worklist
        let mut heads: BTreeMap<TableStateID, Vec<Vec<ParseStateEdgeContent>>> = BTreeMap::new();
        for (path, _acc) in parser_state.stack.to_stacks() {
            if let Some(top) = path.last() {
                heads.entry(top.state_id).or_default().push(path);
            }
        }

        // Worklist: try all possible reduces exhaustively to find acceptance
        let max_iterations = 200;
        for _ in 0..max_iterations {
            let mut new_heads: BTreeMap<TableStateID, Vec<Vec<ParseStateEdgeContent>>> = BTreeMap::new();
            let mut found_acceptance = false;

            for (state_id, paths) in &heads {
                if let Some(row) = get_row(table, *state_id) {
                    let mut reduces: Vec<(NonTerminalID, usize)> = Vec::new();
                    if let Some(default) = &row.default_reduce {
                        Self::collect_reduces(default, &mut reduces);
                    }

                    if reduces.is_empty() {
                        continue;
                    }

                    reduces.sort();
                    reduces.dedup();

                    for (nt_id, len) in &reduces {
                        for path in paths {
                            if path.len() < *len + 1 {
                                continue; // Not enough stack elements to pop
                            }
                            // After popping `len` elements, the exposed state is at position path.len() - len - 1
                            let exposed_idx = path.len() - *len - 1;
                            let from_state = path[exposed_idx].state_id;

                            if let Some(row2) = get_row(table, from_state) {
                                if let Some(goto) = row2.gotos.get(nt_id) {
                                    if goto.accept {
                                        found_acceptance = true;
                                    }
                                    if let Some(next_id) = goto.state_id {
                                        let mut new_path = path[..=exposed_idx].to_vec();
                                        new_path.push(ParseStateEdgeContent { state_id: next_id });
                                        new_heads.entry(next_id).or_default().push(new_path);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if found_acceptance {
                return true;
            }
            if new_heads.is_empty() {
                return false;
            }
            heads = new_heads;
        }

        false
    }

    fn collect_reduces(action: &Stage7ShiftsAndReducesLookaheadValue, reduces: &mut Vec<(NonTerminalID, usize)>) {
        match action {
            Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, .. } => {
                reduces.push((*nonterminal_id, *len));
            }
            Stage7ShiftsAndReducesLookaheadValue::Split { reduces: split_reduces, .. } => {
                for (len, nts) in split_reduces {
                    for (nt_id, _) in nts {
                        reduces.push((*nt_id, *len));
                    }
                }
            }
            _ => {}
        }
    }
}
