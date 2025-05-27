// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::BitOr;
use std::sync::{Arc, Mutex};
use std::cell::RefCell;

use bimap::BiBTreeMap;
use bitvec::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

use crate::constraint_extra::{calculate_final_stats, print_precompute_stats, PrecomputeStats};
use crate::datastructures::charmap::TrieMap;
use crate::datastructures::gss::{gather_gss_stats, print_gss_forest, prune_and_transform_recursive, prune_and_transform_recursive_canonical, simplify_gss_forest, GSSNode, PathAccumulator}; // Import PathAccumulator
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::{EdgeInserter, Trie};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::datastructures::ArcPtrWrapper;
use crate::finite_automata::Regex;
use crate::glr::parser::{
    GLRParser, GLRParserState, ParseState, ParseStateEdgeContent, // Remove MergeAndIntersect from this import
};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::BTreeMap as StdMap;

pub type LLMTokenBV = HybridBitset;
pub type GrammarTokenBV = BitVec; // BitVec is not easily JSONConvertible without a specific strategy (e.g., as Vec<bool> or Vec<usize> of set bits)
                                  // For now, if GrammarTokenBV needs serialization, it will be complex.

// -----------------------------------------------------------------------------
// Small data-types used by the constraint
// -----------------------------------------------------------------------------
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LLMTokenInfo {
    pub active:       LLMTokenBV,
    pub intersection: LLMTokenBV,
    pub terminals:    Arc<GSSNode<TerminalID, ()>>,
}

impl Default for LLMTokenInfo {
    fn default() -> Self {
        // For the PathAccumulator trait, Default must provide the identity for UNION.
        // For LLMTokenInfo, this means active=empty, intersection=ALL_ONES.
        // This requires knowing the capacity. If HybridBitset::new() is empty, and
        // HybridBitset::ones(capacity) makes all ones, then this works IF capacity is known.
        // Assuming HybridBitset::new() is identity for active (OR-like) and
        // ALL_ONES is identity for intersection (AND-like).
        // This is problematic for generic Default. Let's assume Default provides
        // the correct identity for union based on the Bitset's default state
        // or that LLMTokenInfo is constructed with capacity elsewhere.
        // Let's explicitly make it empty for active and all_ones (requires capacity).
        // Since Default cannot take capacity, this is a hack.
        // The actual identity_for_union should be context dependent or HybridBitset needs a default capacity.
        // For now, let's use HybridBitset::new() for both, acknowledging the limitation.
        // The correct initialization in GrammarConstraint::init provides the ALL_ONES intersection.
        Self {
            active:       Default::default(), // Empty set
            intersection: HybridBitset::max_ones(),
            terminals:    Arc::new(GSSNode::new_default()),
        }
    }
}

impl PathAccumulator for LLMTokenInfo {
    fn union(&self, other: &Self) -> Self {
        let mut terminals = self.terminals.clone();
        Arc::make_mut(&mut terminals).merge(other.terminals.as_ref().clone());
        Self {
            active:       &self.active | &other.active,
            intersection: &self.intersection & &other.intersection, // Intersection field becomes stricter (AND)
            terminals,
        }
    }

    fn pop(&self, right: &Self) -> Self {
        // Keep the right terminals
        // assert that the total number of paths in the right terminals is equal to or greater than the total number of paths in the left terminals
        // todo: put this in something that can check debug level == 3
        if false {
            let left_terminals_paths = self.terminals.iter_paths().collect::<Vec<_>>();
            let right_terminals_paths = right.terminals.iter_paths().collect::<Vec<_>>();
            assert!(left_terminals_paths.len() <= right_terminals_paths.len());
        }
        Self {
            active:       &self.active & &right.active,
            intersection: &self.intersection & &right.intersection,
            terminals:    right.terminals.clone(),
        }
    }
}


impl std::fmt::Debug for LLMTokenInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const MAX_ITEMS: usize = 10;

        let fmt_bv = |bv: &LLMTokenBV| -> String {
            // let ids: Vec<_> = bv.iter().collect();
            // match ids.len() > MAX_ITEMS {
            //     true  => format!("[{:?}… ({} total)]", &ids[..MAX_ITEMS], ids.len()),
            //     false => format!("{:?}", ids),
            // }
            format!("{:?}", bv)
        };

        f.debug_struct("LLMTokenInfo")
            .field("active", &fmt_bv(&self.active))
            .field("intersection", &fmt_bv(&self.intersection))
            .field("terminals", &self.terminals)
            .finish()
    }
}

// Remove MergeAndIntersect impl for LLMTokenInfo

// -----------------------------------------------------------------------------
// Pre-computation node values
// -----------------------------------------------------------------------------
#[derive(Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedFinalizer {
    pub content: BTreeMap<TokenizerStateID, LLMTokenBV>,
}

// Manual impl for PrecomputedFinalizer - unchanged
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
                                   .and_then(|n| BTreeMap::<TokenizerStateID, LLMTokenBV>::from_json(n))?;
                Ok(PrecomputedFinalizer { content })
            }
            _ => Err("Expected JSONNode::Object for PrecomputedFinalizer".to_string()),
        }
    }
}


impl PrecomputedFinalizer {
    fn new(tokens: LLMTokenBV, tokenizer_state: TokenizerStateID) -> Self {
        Self {
            content: BTreeMap::from([(tokenizer_state, tokens)]),
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrecomputedNodeContents {
    finalizers: BTreeMap<GrammarTokenID, PrecomputedFinalizer>,
    pub clean_end: Option<LLMTokenBV>,
    pub active: LLMTokenBV, // Add this line
}

// Manual impl for PrecomputedNodeContents - unchanged
impl JSONConvertible for PrecomputedNodeContents {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("finalizers".to_string(), self.finalizers.to_json());
        obj.insert("clean_end".to_string(), self.clean_end.to_json());
        obj.insert("active".to_string(), self.active.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let finalizers = obj.remove("finalizers").ok_or_else(|| "Missing field finalizers for PrecomputedNodeContents".to_string())
                                    .and_then(|n| BTreeMap::<GrammarTokenID, PrecomputedFinalizer>::from_json(n))?;
                let clean_end = obj.remove("clean_end").ok_or_else(|| "Missing field clean_end for PrecomputedNodeContents".to_string())
                                   .and_then(Option::<LLMTokenBV>::from_json)?;
                let active = obj.remove("active").ok_or_else(|| "Missing field active for PrecomputedNodeContents".to_string())
                                  .and_then(LLMTokenBV::from_json)?;
                Ok(PrecomputedNodeContents { finalizers, clean_end, active })
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
        llm_token: LLMTokenID, // This is internal ID
        tokenizer_state: TokenizerStateID,
    ) {
        let mut bv = HybridBitset::new();
        bv.insert(llm_token.0); // llm_token.0 is already internal

        self.finalizers
            .entry(grammar_token)
            .and_modify(|f| {
                f.content
                    .entry(tokenizer_state)
                    .and_modify(|existing| *existing |= &bv)
                    .or_insert(bv.clone());
            })
            .or_insert_with(|| PrecomputedFinalizer::new(bv, tokenizer_state));
    }
}

// Pre-computation graph node / alias types
pub type PrecomputeNode =
    Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;

pub type Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>;
// JSONConvertible for Precomputed will depend on PrecomputeNode.

// -----------------------------------------------------------------------------
// GrammarConstraint – public facing object
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarConstraint {
    pub(crate) tokenizer:        Regex,
    pub(crate) parser:           GLRParser,
    pub(crate) precomputed:      Precomputed,
    pub(crate) llm_token_map:    BiBTreeMap<Vec<u8>, LLMTokenID>, // Stores original LLMTokenIDs (bytes -> original ID)
    pub(crate) token_name_map:   BiBTreeMap<String, usize>,
    pub(crate) max_original_llm_token_id: usize, // Max ID from the input llm_token_map

    // Mapping between original LLMTokenID.0 and internal LLMTokenID.0 (usize)
    // The number of unique internal tokens is derived from the size of this bimap.
    pub(crate) original_to_internal_id_bimap: BiBTreeMap<usize, usize>, // original_id.0 <-> internal_id.0
    pub(crate) internal_max_llm_token: usize, // Number of unique internal LLM tokens (capacity for bitsets)
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
                Ok(GrammarConstraint {
                    tokenizer,
                    parser,
                    precomputed,
                    llm_token_map,
                    token_name_map,
                    max_original_llm_token_id,
                    original_to_internal_id_bimap,
                    internal_max_llm_token,
                })
            }
            _ => Err("Expected JSONNode::Object for GrammarConstraint".to_string()),
        }
    }
}


impl GrammarConstraint {
    // Helper function to set up LLM token mappings - unchanged
    pub(crate) fn setup_llm_token_mappings(
        original_llm_token_map: &LLMTokenMap, // Input: Original BiBTreeMap<Vec<u8>, LLMTokenID>
    ) -> BiBTreeMap<usize, usize> // Returns original_id.0 <-> internal_id.0 bimap
    {
        // // TODO: delete this
        // // Temporarily just don't map
        // let mut original_to_internal_id_bimap = BiBTreeMap::new();
        // for (_, i) in original_llm_token_map.iter() {
        //     original_to_internal_id_bimap.insert(i.0, i.0);
        // }
        // return original_to_internal_id_bimap;

        // 1. Create sorted list of tokens to define internal mapping
        let mut sorted_tokens_with_original_ids: Vec<(Vec<u8>, LLMTokenID)> = original_llm_token_map
            .iter()
            .map(|(bytes, original_id)| (bytes.clone(), *original_id))
            .collect();
        sorted_tokens_with_original_ids.sort_by(|(bytes_a, _), (bytes_b, _)| bytes_a.cmp(bytes_b));

        // 2. Build the original_to_internal_id_bimap
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
        llm_token_map:    LLMTokenMap, // This is BiBTreeMap<Vec<u8>, LLMTokenID> with original IDs
        token_name_map:   BiBTreeMap<String, usize>,
        max_original_llm_token_id: usize, // Max ID of original LLMTokenIDs from input
    ) -> Self {
        let original_to_internal_id_bimap = Self::setup_llm_token_mappings(&llm_token_map);

        let internal_max_llm_token = original_to_internal_id_bimap.iter().map(|(_, id)| *id).max().unwrap_or(0);

        // Reconstruct the internal_llm_token_map for precomputation (bytes -> internal LLMTokenID)
        let mut internal_llm_token_map_for_precompute = BiBTreeMap::new();
        for (bytes, original_id) in llm_token_map.iter() {
            if let Some(internal_id_val) = original_to_internal_id_bimap.get_by_left(&original_id.0) {
                internal_llm_token_map_for_precompute.insert(bytes.clone(), LLMTokenID(*internal_id_val));
            }
        }

        let precomputed = Self::precompute(
            &tokenizer,
            &internal_llm_token_map_for_precompute, // Pass the map with internal IDs
            &token_name_map,
            internal_max_llm_token, // Pass the number of internal tokens
        );

        Self {
            tokenizer,
            parser,
            precomputed,
            llm_token_map, // Store the original llm_token_map (bytes -> original LLMTokenID)
            token_name_map,
            max_original_llm_token_id,
            original_to_internal_id_bimap,
            internal_max_llm_token,
        }
    }

    // -------------------------------------------------------------------------
    // PRE-COMPUTATION (heavy but now readable ☺)
    // -------------------------------------------------------------------------
    pub fn precompute(
        tokenizer:        &Regex,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>, // Renamed
        token_name_map:   &BiBTreeMap<String, usize>,
        internal_max_llm_token: usize,                       // Number of internal tokens
    ) -> Precomputed {
        // 1.  Kick off a helper object that contains all large mutable state.
        let mut helper = Precomputer::new(
            tokenizer,
            internal_llm_token_map,    // Use new parameter name
            internal_max_llm_token, // Use new parameter name
            100, // merge threshold
        );

        // 2.  Run the DFS over the vocabulary prefix tree.
        helper.run_dfs();

        // 3. Merge nodes.
        helper.merge_nodes();

        // 4.  Collect statistics & finish progress-bar.
        helper.finish(token_name_map)
    }

    // -------------------------------------------------------------------------
    // Runtime interface -------------------------------------------------------------------
    // -------------------------------------------------------------------------
    pub fn init(&self) -> GrammarConstraintState<'_> {
        // Create the initial LLMTokenInfo accumulator for the root of the GSS.
        // This represents the state where all original LLM tokens are initially possible.
        let base_set_for_info = HybridBitset::ones(self.internal_max_llm_token + 1);
        let initial_llm_token_acc = LLMTokenInfo { // Renamed from info
            active:       base_set_for_info.clone(),
            intersection: base_set_for_info,
            terminals:    Arc::new(GSSNode::new_default()),
        };
        let mut state = BTreeMap::new();
        state.insert(
            self.tokenizer.initial_state_id(),
            // Initialize GLRParserState with the initial accumulator
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

    #[allow(dead_code)] // Might be useful later
    fn original_bv_to_internal(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        let mut internal_bv = HybridBitset::new();
        for original_id_val in original_bv.iter() {
            let internal_id_val = self.original_to_internal_id_bimap.get_by_left(&(original_id_val as usize)).expect(format!("Original ID {} not found in original_to_internal_id_bimap", original_id_val).as_str());
            internal_bv.insert(*internal_id_val as usize);
        }
        internal_bv
    }

    fn internal_bv_to_original(&self, internal_bv: &LLMTokenBV) -> LLMTokenBV {
        let mut original_bv = HybridBitset::new();
        for internal_id_val in internal_bv.iter() {
            let original_id_val = self.original_to_internal_id_bimap.get_by_right(&(internal_id_val as usize)).expect(format!("Internal ID {} not found in original_to_internal_id_bimap", internal_id_val).as_str());
            original_bv.insert(*original_id_val as usize);
        }
        original_bv
    }
}

// -----------------------------------------------------------------------------
// Internal helper object that owns the gnarly DFS logic
// -----------------------------------------------------------------------------
struct Precomputer<'r> {
    tokenizer:        &'r Regex,
    vocab:            VocabPrefixTree,
    roots:            BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>>,
    possible_matches: RefCell<BTreeMap<*const VocabPrefixTreeNode, BTreeMap<TokenizerStateID, BTreeMap<GrammarTokenID, LLMTokenBV>>>>,
    all_llm_tokens:   HybridBitset,
    merge_threshold:  usize,
    pb:               ProgressBar,
    stats:            PrecomputeStats,
}

impl<'r> Precomputer<'r> {
    fn new(
        tokenizer:        &'r Regex,
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>, // Renamed
        internal_max_llm_token: usize,                       // Number of internal tokens
        merge_threshold:  usize,
    ) -> Self {
        // -- Build vocab prefix tree ------------------------------------------------------
        let tokens: Vec<(usize, Vec<u8>)> = internal_llm_token_map // Use internal map
            .iter()
            .map(|(bytes, id)| (id.0 as usize, bytes.clone())) // id.0 is already internal
            .collect();

        crate::debug!(2, "Building vocab prefix tree");
        let vocab = VocabPrefixTree::build(&tokens);
        crate::debug!(2, "Done building vocab prefix tree");

        // -- Root nodes (one per tokenizer state) -----------------------------------------
        let mut roots = BTreeMap::new();
        for sid in 0..tokenizer.max_state() {
            roots.insert(
                TokenizerStateID(sid),
                Arc::new(Mutex::new(PrecomputeNode::new(
                    PrecomputedNodeContents::default(),
                ))),
            );
        }

        // -- Progress bar -----------------------------------------------------------------
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
        }
    }

    fn possible_matches(&self, vocab_node: &VocabPrefixTreeNode, tokenizer_state_id: TokenizerStateID) -> BTreeMap<GrammarTokenID, LLMTokenBV> {
        let cache_key_ptr = vocab_node as *const VocabPrefixTreeNode;

        // Cache lookup (read-only borrow first)
        if let Some(cached_for_vocab_node) = self.possible_matches.borrow().get(&cache_key_ptr) {
            if let Some(cached_result) = cached_for_vocab_node.get(&tokenizer_state_id) {
                return cached_result.clone();
            }
        }

        // If not in cache, compute:
        let mut result_map: BTreeMap<GrammarTokenID, LLMTokenBV> = BTreeMap::new();

        // Recursively call on children and merge their results.
        for (segment_bytes, child_vocab_arc) in vocab_node.iter_children() {
            let exec_result = self.tokenizer.execute_from_state(&segment_bytes, tokenizer_state_id);
            for token in &exec_result.matches {
                let grammar_token_id = GrammarTokenID(token.id);
                let applicable_tokens = child_vocab_arc.reachable_token_ids();
                *result_map.entry(grammar_token_id).or_insert_with(LLMTokenBV::new) |= applicable_tokens;
            }
            if let Some(final_state_val) = exec_result.end_state {
                // No point in continuing if all possible future matches are for tokens that we've already matched.
                let matches_possible_from_tokenizer_state: BTreeSet<_> = self.tokenizer.tokens_accessible_from_state(TokenizerStateID(final_state_val)).into_iter().collect();
                let matches_here: BTreeSet<_> = exec_result.matches.iter().map(|m| GrammarTokenID(m.id)).collect();
                let possible_new_matches = &matches_possible_from_tokenizer_state - &matches_here;
                if !possible_new_matches.is_empty() {
                    // We have a new match, so we need to continue.
                    let next_results = self.possible_matches(child_vocab_arc, TokenizerStateID(final_state_val));
                    for (token, bv) in next_results {
                        *result_map.entry(token).or_insert_with(LLMTokenBV::new) |= bv;
                    }
                }
            }
        }

        // Cache it.
        self.possible_matches.borrow_mut().entry(cache_key_ptr).or_default().insert(tokenizer_state_id, result_map.clone());

        result_map
    }

    // -------------------------------------------------------------------------
    // Public driver
    // -------------------------------------------------------------------------
    fn run_dfs(&mut self) {
        // Entry associations: every tokenizer state starts at the vocab root.
        let mut assoc: BTreeMap<
            TokenizerStateID,
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
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
        // Merge equal nodes (first pass: collect unique representatives).
        crate::debug!(2, "Merging nodes: first collecting unique roots");
        let total = self.roots.len() as u64;
        // let pb1 = ProgressBar::new(total);
        // pb1.set_style(self.pb.style().clone());

        let mut unique = BTreeMap::new();
        for (tokenizer_state_id, root) in &self.roots {
            // pb1.inc(1);
            crate::debug!(4, "Processing root {:?}", tokenizer_state_id);
            let new_root = unique
                .entry(root.lock().unwrap().clone())
                .or_insert_with(|| root.clone());
            *new_root = root.clone();
        }
        // pb1.finish_with_message(format!("Collected {} unique nodes", unique.len()));

        // Second pass: rewrite each root to its unique representative.
        crate::debug!(2, "Merging nodes: second pass rewriting roots");
        // let pb2 = ProgressBar::new(total);
        // pb2.set_style(self.pb.style().clone());

        for (_tokenizer_state_id, root) in &mut self.roots {
            // pb2.inc(1);
            let new_root = unique
                .get(&root.lock().unwrap().clone())
                .unwrap()
                .clone();
            *root = new_root;
        }
        // pb2.finish_with_message("Rewritten all roots to unique representatives");
    }

    // -------------------------------------------------------------------------
    // Finalise: stats, cycle check, unwrap Arcs -> Precomputed
    // -------------------------------------------------------------------------
    fn finish(mut self, token_name_map: &BiBTreeMap<String, usize>) -> Precomputed {
        // Cycle check
        crate::debug!(2, "Checking for cycles in precomputed graph…");
        for (sid, root) in &self.roots {
            if PrecomputeNode::has_any_cycle(root.clone()) {
                panic!(
                    "Cycle detected in precomputed graph for tokenizer_state_id {:?}",
                    sid
                );
            }
        }

        // Stats
        calculate_final_stats(&self.roots, &mut self.stats);
        print_precompute_stats(&self.stats, token_name_map);

        // Turn Arc<Mutex<…>> roots into plain PrecomputeNode roots.
        let mut out = Precomputed::new();
        let mut clones = 0;

        for (sid, arc) in self.roots {
            match Arc::try_unwrap(arc) {
                Ok(mutex) => out.insert(
                    sid,
                    mutex.into_inner().expect("Mutex poisoned during unwrap"),
                ),
                Err(arc_val) => { // Renamed arc to arc_val
                    clones += 1;
                    out.insert(sid, arc_val.lock().unwrap().clone()) // Use arc_val
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

    // -------------------------------------------------------------------------
    // DEPTH-FIRST SEARCH -------------------------------------------------------
    // -------------------------------------------------------------------------
    fn dfs(
        &self,
        vocab_node: &VocabPrefixTreeNode,
        assoc_by_state: BTreeMap<
            TokenizerStateID,
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        >,
    ) {
        self.pb.inc(1);

        // Merge policy per tokenizer state
        let mut effective: BTreeMap<
            TokenizerStateID,
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        > = BTreeMap::new();

        for (sid, set) in assoc_by_state {
            let merged = self.merge_handles(&set);
            if !merged.is_empty() {
                effective.insert(sid, merged);
            }
        }

        // ---------------------------------------------------------------------
        // Explore each outgoing byte segment
        // ---------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // A single byte segment of the vocab prefix tree
    // -------------------------------------------------------------------------
    fn process_segment(
        &self,
        segment_bytes: &[u8],
        child_vocab_of_segment: &VocabPrefixTreeNode,
        sources_per_state: &BTreeMap<
            TokenizerStateID,
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        >,
    ) {
        // Maps used while consuming the segment byte-by-byte.
        let mut next_level: BTreeMap<
            TokenizerStateID,
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        > = BTreeMap::new();

        let mut queue: BTreeMap<
            usize,
            BTreeMap<TokenizerStateID, BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>>,
        > = BTreeMap::new();

        // Seed queue with offset 0.
        for (sid, set) in sources_per_state {
            queue
                .entry(0)
                .or_default()
                .entry(*sid)
                .or_default()
                .extend(set.clone());
        }

        while let Some((offset, map_at_offset)) = queue.pop_first() {
            for (state_before, src_set) in map_at_offset {
                if src_set.is_empty() {
                    continue;
                }

                let merged_src_set = self.merge_handles(&src_set); // Renamed src_set to merged_src_set
                if merged_src_set.is_empty() { // Use merged_src_set
                    continue;
                }

                let suffix      = &segment_bytes[offset..];
                let exec_result = self
                    .tokenizer
                    .execute_from_state(suffix, state_before);
                // let merged_src_nodes_ptr_strs = merged_src_set.iter().map(|node| format!("{:p}", Arc::as_ptr(node))).collect::<Vec<_>>().join(", ");
                // crate::debug!(4, "Executed tokenizer from state {:?} on suffix {:?}. Results: {:?}. Nodes are: {}", state_before.0, String::from_utf8_lossy(suffix), exec_result, merged_src_nodes_ptr_strs);

                let possible_future_matches: BTreeMap<GrammarTokenID, LLMTokenBV> = exec_result.end_state.map_or_else(BTreeMap::new, |end_state_id| {
                    self.possible_matches(&child_vocab_of_segment, TokenizerStateID(end_state_id))
                });

                // -------------------------------------------------------------
                // Matches inside suffix
                // -------------------------------------------------------------
                for m in &exec_result.matches {
                    let grammar_tok = GrammarTokenID(m.id);
                    let match_end_offset = offset + m.width;
                    let active_tokens = child_vocab_of_segment.reachable_token_ids().clone();
                    let tokens_with_future_match = possible_future_matches.get(&grammar_tok).cloned().unwrap_or_default();
                    // Apply greediness: If a future match is possible, don't match now.
                    let edge_tokens = active_tokens - tokens_with_future_match;

                    if !edge_tokens.is_empty() {
                        for src in &merged_src_set {
                            // let merged_src_nodes_ptr_strs = merged_src_set.iter().map(|node| format!("{:p}", Arc::as_ptr(node)).as_str()).collect::<Vec<_>>().join(", ");
                            // crate::debug!(4, "Adding edge from {:p} to {:?} with edge tokens {:?}. Nodes are: {}", Arc::as_ptr(src.as_arc()), grammar_tok, edge_tokens, merged_src_nodes_ptr_strs);
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

                // -------------------------------------------------------------
                // Final tokenizer state after reading entire suffix
                // -------------------------------------------------------------
                if let Some(final_state_val) = exec_result.end_state {
                    let final_sid = TokenizerStateID(final_state_val);
                    for src in &merged_src_set { // Use merged_src_set
                        // propagate association for next DFS level
                        next_level
                            .entry(final_sid)
                            .or_default()
                            .insert(src.clone());

                        // push possible finalizers
                        let mut guard = src.as_arc().lock().unwrap();
                        for gtid in self
                            .tokenizer
                            .tokens_accessible_from_state(final_sid)
                        {
                            crate::debug!(5, "Pushing finalizer info for token {:?} in state {:?}", gtid.0, final_sid.0);
                            guard.value.push_finalizer_info(
                                gtid,
                                LLMTokenID(child_vocab_of_segment.token_id()), // This is internal ID
                                final_sid,
                            );
                        }
                    }
                }
            }
        }

        // Recurse into the child vocab node.
        self.dfs(child_vocab_of_segment, next_level);
    }

    // Insert or merge an edge out of `source_arc`.
    #[allow(clippy::too_many_arguments)]
    fn insert_edge(
        &self,
        source_arc: Arc<Mutex<PrecomputeNode>>,
        grammar_tok: GrammarTokenID,
        edge_tokens: LLMTokenBV,
        final_llm_token_id_at_child_vocab: usize, // This is internal ID
        match_end_offset_in_segment: usize,
        segment_len: usize,
        queue: &mut BTreeMap<
            usize,
            BTreeMap<TokenizerStateID, BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>>,
        >,
        next_level: &mut BTreeMap<
            TokenizerStateID,
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        >,
    ) {
        let mut inserter = EdgeInserter::new(
            source_arc.clone(),
            Some(grammar_tok),
            edge_tokens.clone(), // Clone for inserter
            |existing: &mut HybridBitset, new_bv_ref: HybridBitset| *existing |= new_bv_ref,
        );

        // First try existing children
        inserter = inserter.try_children();

        // gather potential targets
        let mut pot: Vec<Arc<Mutex<PrecomputeNode>>> = Vec::new();

        let gather_set = |set: &BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
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

        // As last resort – children of current key
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

        // Add this block to update the target node's active set
        {
            let mut guard = target.lock().unwrap();
            guard.value.active |= &edge_tokens;
        }


        let handle = ArcPtrWrapper::new(target.clone());

        if match_end_offset_in_segment == segment_len {
            crate::debug!(5, "Marking clean end for child vocab node {:p} representing LLM token {:?}", handle.as_ref(), final_llm_token_id_at_child_vocab);
            next_level
                .entry(TokenizerStateID(0)) // Default tokenizer state for clean ends
                .or_default()
                .insert(handle);

            // mark clean_end
            let mut g = target.lock().unwrap();
            g.value
                .clean_end
                .get_or_insert_with(HybridBitset::new)
                .insert(final_llm_token_id_at_child_vocab); // final_llm_token_id_at_child_vocab is internal
        } else {
            queue
                .entry(match_end_offset_in_segment)
                .or_default()
                .entry(TokenizerStateID(0)) // Default tokenizer state for intermediate segment matches
                .or_default()
                .insert(handle);
        }
    }

    // -------------------------------------------------------------------------
    // Merge logic (union vs fresh node)
    // -------------------------------------------------------------------------
    fn merge_handles(
        &self,
        set: &BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
    ) -> BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>> {
        if set.len() <= self.merge_threshold {
            return set.clone();
        }

        let merged_node_arc = Arc::new(Mutex::new(PrecomputeNode::new( // Renamed merged to merged_node_arc
            PrecomputedNodeContents::default(),
        )));

        for child_wrapper in set { // Renamed child to child_wrapper
            let edge_tokens_for_merge = self.all_llm_tokens.clone();
            let mut inserter = EdgeInserter::new(
                child_wrapper.as_arc().clone(), // Source of the edge
                None::<GrammarTokenID>,   // Key for the edge (epsilon)
                edge_tokens_for_merge.clone(), // Data for the edge
                |existing_edge_data: &mut HybridBitset, new_edge_data: HybridBitset| *existing_edge_data |= new_edge_data,
            );

            // Try to reuse an existing child of `child_wrapper.as_arc()`
            inserter = inserter.try_children();

            // If no suitable existing child was found, make the new `merged_node_arc` the destination
            if inserter.clone_into_option().is_none() {
                inserter = inserter.try_destination(merged_node_arc.clone()); // Use merged_node_arc
            }

            // Now, `inserter.clone_into_option()` should contain Some(destination_node).
            // Update the `active` field of this destination_node.
            if let Some(destination_node_arc) = inserter.clone_into_option() {
                let mut guard = destination_node_arc.lock().unwrap();
                guard.value.active |= &edge_tokens_for_merge;
            }
        }

        let mut out = BTreeSet::new();
        out.insert(ArcPtrWrapper::new(merged_node_arc)); // Use merged_node_arc
        out
    }
}

// -----------------------------------------------------------------------------
// Tiny helper (private) – node counter for progress bar
// -----------------------------------------------------------------------------
fn count_vocab_nodes(node: &VocabPrefixTreeNode) -> u64 {
    1 + node
        .children()
        .values()
        .map(|c| count_vocab_nodes(c))
        .sum::<u64>()
}

// -----------------------------------------------------------------------------
// Runtime state object
// -----------------------------------------------------------------------------
#[derive(Debug, Clone)]
// GLRParserState is now generic over A: PathAccumulator (LLMTokenInfo)
pub struct GrammarConstraintState<'a> {
    parent: &'a GrammarConstraint,
    state:  BTreeMap<TokenizerStateID, GLRParserState<'a, LLMTokenInfo>>,
}
// GrammarConstraintState is a runtime state, likely not for direct serialization.

impl<'a> GrammarConstraintState<'a> {
    pub fn get_mask(&mut self) -> LLMTokenBV {
        let mut internal_mask = HybridBitset::new(); // This will be composed of internal IDs
        for (_tokenizer_state_id, glr_parser_state) in &self.state {
            // Access the acc field on the GSS node
            internal_mask |= &glr_parser_state.active_state.stack.acc().active; // .active is already internal
        }
        self.parent.internal_bv_to_original(&internal_mask) // Convert back to original IDs
    }

    pub fn step_with_all_llm_tokens(&mut self) {
        // This creates a bitset of all *internal* LLM tokens
        let all_internal_llm_tokens = HybridBitset::ones(self.parent.internal_max_llm_token + 1);
        self.step(&all_internal_llm_tokens);
    }

    pub fn step_with_llm_token(&mut self, llm_token_id: LLMTokenID) { // llm_token_id is original
        // Convert original LLMTokenID to internal LLMTokenID
        if let Some(internal_llm_id) = self.parent.original_id_to_internal(llm_token_id) {
            let mut internal_llm_tokens_bv = HybridBitset::new();
            internal_llm_tokens_bv.insert(internal_llm_id.0 as usize);
            self.step(&internal_llm_tokens_bv); // step expects internal IDs
        } else {
            // Token ID not in map, treat as if it matches nothing by stepping with an empty set.
            let empty_set = HybridBitset::new();
            self.step(&empty_set);
        }
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) { // llm_token_id is original
        let all_true_set = HybridBitset::ones(self.parent.internal_max_llm_token + 1);

        // Convert original LLMTokenID to internal LLMTokenID for the closure
        let internal_llm_id_val = self.parent.original_id_to_internal(llm_token_id).unwrap().0;

        let closure = |t: &LLMTokenInfo| -> Option<(LLMTokenInfo, bool)> {
            let mut t = t.clone();
            if t.active.contains(internal_llm_id_val) { // .active is internal, compare with internal ID
                if t.intersection == all_true_set {
                    Some((t, false))
                } else {
                    t.active = all_true_set.clone();
                    t.intersection = all_true_set.clone();
                    Some((t, true))
                }
            } else { // Original token ID not found in mapping, so it cannot be active
                None
            }
        };

        let mut memo = HashMap::new();
        self.state.retain(|_tokenizer_state_id, glr_state| {
            let parse_state = &mut glr_state.active_state;
            // crate::debug!(4, "Pruning parse state {:?}", parse_state.key());
            let maybe_new_node = prune_and_transform_recursive(&parse_state.stack, &closure, &mut memo);
            // crate::debug!(4, "Pruned parse state {:?}. Got new node {:?}.", parse_state.key(), maybe_new_node);
            if let Some(new_node) = maybe_new_node {
                parse_state.stack = new_node;
            }
            !glr_state.active_state.stack.predecessors_with_values().is_empty()
        });

        crate::debug!(4, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        crate::debug!(4, "Committed llm_token_id {:?} to grammar constraint state", llm_token_id);
        // Print each GSS
        for (tokenizer_state_id, glr_state) in self.state.iter() {
            glr_state.log_gss(format!("After committing llm_token_id {:?}, from tokenizer_state_id {:?}", llm_token_id, tokenizer_state_id).as_str(), GrammarTokenID(0));
        }
    }

    pub fn step_with_llm_token_sequence(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.step_with_llm_token(llm_token_id);
        }
    }

    fn prepare_initial_nodes_and_values_for_special_map(&mut self, llm_tokens: &LLMTokenBV) -> Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'a, LLMTokenInfo>)> {
        let mut initial_nodes_and_values: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'_, LLMTokenInfo>)> = Vec::new();
        let mut tokenizer_state_id_to_parse_states: BTreeMap<TokenizerStateID, GLRParserState<'_, LLMTokenInfo>> = BTreeMap::new();

        for (tokenizer_state_id, state_val) in &self.state { // Renamed state to state_val
            let mut cloned_state = state_val.clone(); // Use state_val
            Arc::make_mut(&mut cloned_state.active_state.stack).acc.active &= llm_tokens;
            tokenizer_state_id_to_parse_states.insert(*tokenizer_state_id, cloned_state);
        }

        crate::debug!(4, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        crate::debug!(4, "Printing initial nodes and values for tokenizer states");
        for tokenizer_state_id_val in tokenizer_state_id_to_parse_states.keys() { // Renamed tokenizer_state_id
            let glr_state_before = &self.state[&tokenizer_state_id_val]; // Use tokenizer_state_id_val
            let glr_state_after = &tokenizer_state_id_to_parse_states[&tokenizer_state_id_val]; // Use tokenizer_state_id_val
            glr_state_before.log_gss(format!("Existing initial nodes and values for tokenizer state {}", tokenizer_state_id_val.0).as_str(), GrammarTokenID(0)); // Use tokenizer_state_id_val
            glr_state_after.log_gss(format!("Prepared (stage 1) initial nodes and values for tokenizer state {}", tokenizer_state_id_val.0).as_str(), GrammarTokenID(0)); // Use tokenizer_state_id_val
        }
        crate::debug!(4, "----------------------------------------------------------------");

        for (tokenizer_state_id, state_val) in tokenizer_state_id_to_parse_states { // Renamed state to state_val
            let token_trie_node = self.parent.precomputed[&tokenizer_state_id].clone();
            let token_trie_arc_mutex = Arc::new(Mutex::new(token_trie_node));
            initial_nodes_and_values.push((token_trie_arc_mutex, state_val)); // Use state_val
        }


        initial_nodes_and_values
    }

    pub fn step(&mut self, llm_tokens: &LLMTokenBV) { // llm_tokens are internal IDs
        crate::debug!(2, "Stepping grammar constraint state with tokenizer states {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens);

        // Initialize the counter
        let step_counts = RefCell::new(BTreeMap::<GrammarTokenID, usize>::new());

        self.state = BTreeMap::new();

        Trie::special_map(
            initial_nodes_and_values,
            |glr_parse_state, grammar_token_id_opt, edge_llm_tokens, child_node| { // Renamed grammar_token_id to grammar_token_id_opt
                let node_ptr = std::ptr::addr_of!(*child_node);
                crate::debug!(3, "Processing grammar node {:p} token {:?}. Terminals: {:?}", node_ptr, grammar_token_id_opt.map(|gtid| gtid.0), gather_gss_stats(&[glr_parse_state.active_state.stack.acc.terminals.clone()])); // Use grammar_token_id_opt
                let mut cloned_glr_parse_state = glr_parse_state.clone();
                let current_active_tokens = cloned_glr_parse_state.active_state.stack.acc.active.clone();
                Arc::make_mut(&mut cloned_glr_parse_state.active_state.stack).acc.intersection &= &current_active_tokens;
                Arc::make_mut(&mut cloned_glr_parse_state.active_state.stack).acc.active &= edge_llm_tokens;
                if let Some(gtid) = grammar_token_id_opt { // Use grammar_token_id_opt
                    *step_counts.borrow_mut().entry(*gtid).or_insert(0) += 1;
                }
                grammar_token_id_opt.map(|gtid| {
                    Arc::make_mut(&mut cloned_glr_parse_state.active_state.stack).acc.terminals = Arc::new(cloned_glr_parse_state.active_state.stack.acc.terminals.as_ref().clone().push(gtid));
                    cloned_glr_parse_state.step(gtid);
                }); // Use grammar_token_id_opt
                if cloned_glr_parse_state.active_state.stack.is_empty() {
                    crate::debug!(3, "No active states after processing grammar token {:?}", grammar_token_id_opt.map(|gtid| gtid.0)); // Use grammar_token_id_opt
                    return None;
                } else {
                    crate::debug!(3, "Processed grammar token {:?}.", grammar_token_id_opt.map(|gtid| gtid.0)); // Use grammar_token_id_opt
                    Some(cloned_glr_parse_state)
                }
            },
            |managed_parse_state1, managed_parse_state2| {
                managed_parse_state1.merge_with(managed_parse_state2);
            },
            |node, current_glr_parse_state| {
                // Simplify the GSS forest
                let nodes: Vec<_> = vec![current_glr_parse_state.active_state.stack.clone()];
                let simplified_states = simplify_gss_forest(&nodes);
                current_glr_parse_state.active_state.stack = simplified_states[0].clone();

                let mut active_llm_tokens = HybridBitset::new();
                active_llm_tokens |= &current_glr_parse_state.active_state.stack.acc.active;
                let node_ptr = std::ptr::addr_of!(*node);
                crate::debug!(3, "Processing node {:p}, {} LLM tokens, {} finalizers", node_ptr, active_llm_tokens.len(), node.value.finalizers().len()); // Use .finalizers()
                if let Some(clean_end) = &node.value.clean_end { // clean_end has internal IDs
                    let mut final_glr_parse_state = current_glr_parse_state.clone();
                    let current_active_tokens = final_glr_parse_state.active_state.stack.acc.active.clone();
                    Arc::make_mut(&mut final_glr_parse_state.active_state.stack).acc.intersection &= &current_active_tokens;
                    Arc::make_mut(&mut final_glr_parse_state.active_state.stack).acc.active &= clean_end; // clean_end is internal

                    crate::debug!(3, "At clean end state");
                    if final_glr_parse_state.is_ok() {
                        crate::debug!(3, "GLR parse state at clean end is OK");
                        if let Some(existing) = self.state.get_mut(&TokenizerStateID(0)) {
                            existing.merge_with(final_glr_parse_state.clone());
                        } else {
                            self.state.insert(TokenizerStateID(0), final_glr_parse_state.clone());
                        }
                    }
                }

                for (possible_final_grammar_token, precomputed_finalizer) in node.value.finalizers().iter() { // Use .finalizers()
                    let mut possible_next_glr_parse_state = current_glr_parse_state.clone();
                    crate::debug!(3, "Stepping semi-final GLR parse state");
                    *step_counts.borrow_mut().entry(*possible_final_grammar_token).or_insert(0) += 1;

                    // TODO: put this in something that can check debug level == 3
                    if true {
                        let terminals = possible_next_glr_parse_state.active_state.stack.acc.terminals.clone();
                        crate::debug!(3, "Terminal history (candidates, NOT ALL CORRECT) before stepping:");
                        for path in terminals.iter_paths() {
                            let path_str = path.iter().map(|t| self.parent.token_name_map.get_by_right(&t.0).map(|s| s.clone()).unwrap_or("<Unknown Name>".to_string())).collect::<Vec<_>>().join(" → ");
                            crate::debug!(3, "  {}", path_str);
                        }
                    }
                    possible_next_glr_parse_state.step(*possible_final_grammar_token);
                    if possible_next_glr_parse_state.is_ok() {
                        crate::debug!(3, "Semi-final GLR parse state is OK");
                        for (tokenizer_state_id, llm_tokens_from_finalizer) in &precomputed_finalizer.content { // llm_tokens_from_finalizer are internal
                            let mut glr_parse_state_filtered = current_glr_parse_state.clone(); // Start from current_glr_parse_state for filtering
                            let current_active_tokens = glr_parse_state_filtered.active_state.stack.acc.active.clone();
                            Arc::make_mut(&mut glr_parse_state_filtered.active_state.stack).acc.intersection &= &current_active_tokens;
                            Arc::make_mut(&mut glr_parse_state_filtered.active_state.stack).acc.active &= llm_tokens_from_finalizer; // llm_tokens_from_finalizer are internal

                            crate::debug!(3, "Processing finalizer for token_state_id {:?}", tokenizer_state_id);
                            if glr_parse_state_filtered.is_ok() { // This is current_glr_parse_state filtered by finalizer's llm_tokens
                                crate::debug!(3, "Finalizer is compatible with current GLR state (pre-step by final_grammar_token)");
                                if let Some(existing) = self.state.get_mut(tokenizer_state_id) {
                                    existing.merge_with(glr_parse_state_filtered.clone());
                                } else {
                                    self.state.insert(*tokenizer_state_id, glr_parse_state_filtered.clone());
                                }
                            }
                        }
                    }
                }
                !current_glr_parse_state.active_state.stack.is_empty()
            },
        );

        // Simplify the GSS forest
        let mut roots = Vec::new();
        let mut tokenizer_state_id_to_root_pos = BTreeMap::new();
        for (i, (tokenizer_state_id, glr_state)) in self.state.iter().enumerate() { // Renamed tokenizer_state_id
            let root = glr_state.active_state.stack.clone();
            roots.push(root);
            tokenizer_state_id_to_root_pos.insert(*tokenizer_state_id, i);
        }
        crate::debug!(2, "Before simplifying GSS forest: {:?}", gather_gss_stats(&roots));
        let mut i = 0;
        let simplified_roots = simplify_gss_forest(&roots);
        crate::debug!(2, "After simplifying GSS forest: {:?}", gather_gss_stats(&simplified_roots));
        for (tokenizer_state_id, glr_state) in self.state.iter_mut() {
            let simplified_root = simplified_roots[tokenizer_state_id_to_root_pos[&tokenizer_state_id]].clone();
            glr_state.active_state.stack = simplified_root;
            i += 1;
        }

        // Print each GSS
        for (tokenizer_state_id, glr_state) in self.state.iter() {
            glr_state.log_gss(format!("After simplifying GSS for state {}", tokenizer_state_id.0).as_str(), GrammarTokenID(0));
        }

        // Print GLRParserState::step call counts
        let mut sorted_counts: Vec<(GrammarTokenID, usize)> = step_counts.into_inner().into_iter().collect();
        if !sorted_counts.is_empty() {
            println!("--- GLRParserState::step call counts (this GrammarConstraintState::step) ---");
            sorted_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))); // Sort by count desc, then by ID asc

            for (gtid, count) in sorted_counts {
                let token_name = self
                    .parent
                    .token_name_map
                    .get_by_right(&gtid.0)
                    .map(|s| s.as_str())
                    .unwrap_or("<Unknown Name>");
                println!("  Token {} (ID: {}): {} calls", token_name, gtid.0, count);
            }
            println!("--------------------------------------------------------------------------");
        } else {
            println!("--- GLRParserState::step was not called in this GrammarConstraintState::step ---");
        }
    }

    pub fn is_active(&self) -> bool {
        !self.state.is_empty()
    }
}

