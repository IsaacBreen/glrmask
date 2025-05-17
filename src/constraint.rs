// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::BitOr;
use std::sync::{Arc, Mutex};
use std::cell::RefCell; // Added this line

use bimap::BiBTreeMap;
use bitvec::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Serialize, Deserialize, Serializer, Deserializer};

use crate::constraint_extra::{calculate_final_stats, print_precompute_stats, PrecomputeStats};
use crate::datastructures::charmap::TrieMap;
use crate::datastructures::gss::{prune_and_transform_recursive, prune_and_transform_recursive_canonical, simplify_gss_forest};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::{EdgeInserter, Trie};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::datastructures::ArcPtrWrapper;
use crate::finite_automata::Regex;
use crate::glr::parser::{
    MergeAndIntersect, GLRParser, GLRParserState, ParseState, ParseStateNodeContent,
};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use crate::types::TerminalID as GrammarTokenID;
use crate::constraint_serializer;

pub type LLMTokenBV = HybridBitset;
pub type GrammarTokenBV = BitVec;

// -----------------------------------------------------------------------------
// Small data-types used by the constraint
// -----------------------------------------------------------------------------
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LLMTokenInfo {
    pub active:       LLMTokenBV,
    pub intersection: LLMTokenBV,
}

impl Default for LLMTokenInfo {
    fn default() -> Self {
        Self {
            active:       Default::default(),
            intersection: Default::default(),
        }
    }
}

impl std::fmt::Debug for LLMTokenInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const MAX_ITEMS: usize = 10;

        let fmt_bv = |bv: &LLMTokenBV| -> String {
            let ids: Vec<_> = bv.iter().collect();
            match ids.len() > MAX_ITEMS {
                true  => format!("[{:?}… ({} total)]", &ids[..MAX_ITEMS], ids.len()),
                false => format!("{:?}", ids),
            }
        };

        f.debug_struct("LLMTokenInfo")
            .field("active", &fmt_bv(&self.active))
            .field("intersection", &fmt_bv(&self.intersection))
            .finish()
    }
}

impl MergeAndIntersect for LLMTokenInfo {
    fn merge(&self, other: &Self) -> Self {
        Self {
            active:       &self.active | &other.active,
            intersection: &self.intersection & &other.intersection,
        }
    }
    fn intersect(&self, other: &Self) -> Self {
        Self {
            active:       &self.active & &other.active,
            intersection: &self.intersection & &other.intersection,
        }
    }
}

// -----------------------------------------------------------------------------
// Pre-computation node values
// -----------------------------------------------------------------------------
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct PrecomputedFinalizer {
    pub content: BTreeMap<TokenizerStateID, LLMTokenBV>,
}

impl PrecomputedFinalizer {
    fn new(tokens: LLMTokenBV, tokenizer_state: TokenizerStateID) -> Self {
        Self {
            content: BTreeMap::from([(tokenizer_state, tokens)]),
        }
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct PrecomputedNodeContents {
    finalizers: BTreeMap<GrammarTokenID, PrecomputedFinalizer>,
    pub clean_end: Option<LLMTokenBV>,
    pub active: LLMTokenBV, // Add this line
}

impl PrecomputedNodeContents {
    pub(crate) fn finalizers(&self) -> &BTreeMap<GrammarTokenID, PrecomputedFinalizer> {
        &self.finalizers
    }

    fn push_finalizer_info(
        &mut self,
        grammar_token: GrammarTokenID,
        llm_token: LLMTokenID,
        tokenizer_state: TokenizerStateID,
    ) {
        let mut bv = HybridBitset::new();
        bv.insert(llm_token.0);

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
// Need manual Serialize/Deserialize for Trie with Arc<Mutex<>> children
pub type PrecomputeNode =
    Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;

// This type alias is used in the GrammarConstraint struct, which is where the custom serializer is applied.
// The custom serializer handles the BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>> structure.
pub type Precomputed = BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>>;


// -----------------------------------------------------------------------------
// GrammarConstraint – public facing object
// -----------------------------------------------------------------------------
#[derive(Debug, Clone)] // Remove Serialize, Deserialize here, will be implemented manually
pub struct GrammarConstraint {
    pub(crate) tokenizer:        Regex,
    pub(crate) parser:           GLRParser,
    pub(crate) precomputed:      Precomputed, // BTreeMap<TokenizerStateID, Arc<Mutex<PrecomputeNode>>>
    pub(crate) llm_token_map:    BiBTreeMap<Vec<u8>, LLMTokenID>, // Stores original LLMTokenIDs (bytes -> original ID)
    pub(crate) token_name_map:   BiBTreeMap<String, usize>,
    pub(crate) max_original_llm_token_id: usize, // Max ID from the input llm_token_map

    // Mapping between original LLMTokenID.0 and internal LLMTokenID.0 (usize)
    // The number of unique internal tokens is derived from the size of this bimap.
    pub(crate) original_to_internal_id_bimap: BiBTreeMap<usize, usize>, // original_id.0 <-> internal_id.0
    pub(crate) internal_max_llm_token: usize, // Number of unique internal LLM tokens (capacity for bitsets)
}

// Define a helper struct for serializing/deserializing GrammarConstraint
#[derive(Serialize, Deserialize)]
struct GrammarConstraintSerdeHelper {
    tokenizer: Regex,
    parser: GLRParser,
    #[serde(with = "constraint_serializer")]
    precomputed: Precomputed, // This uses the custom module
    llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID>,
    token_name_map: BiBTreeMap<String, usize>,
    max_original_llm_token_id: usize,
    original_to_internal_id_bimap: BiBTreeMap<usize, usize>,
    internal_max_llm_token: usize,
}


impl Serialize for GrammarConstraint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let helper = GrammarConstraintSerdeHelper {
            tokenizer: self.tokenizer.clone(), // Assuming Regex is Clone
            parser: self.parser.clone(),       // Assuming GLRParser is Clone
            precomputed: self.precomputed.clone(), // Clone the BTreeMap of Arcs
            llm_token_map: self.llm_token_map.clone(),
            token_name_map: self.token_name_map.clone(),
            max_original_llm_token_id: self.max_original_llm_token_id,
            original_to_internal_id_bimap: self.original_to_internal_id_bimap.clone(),
            internal_max_llm_token: self.internal_max_llm_token,
        };
        helper.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for GrammarConstraint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper = GrammarConstraintSerdeHelper::deserialize(deserializer)?;
        Ok(GrammarConstraint {
            tokenizer: helper.tokenizer,
            parser: helper.parser,
            precomputed: helper.precomputed,
            llm_token_map: helper.llm_token_map,
            token_name_map: helper.token_name_map,
            max_original_llm_token_id: helper.max_original_llm_token_id,
            original_to_internal_id_bimap: helper.original_to_internal_id_bimap,
            internal_max_llm_token: helper.internal_max_llm_token,
        })
    }
}


impl GrammarConstraint {
    // Helper function to set up LLM token mappings
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
            999999999999, // merge threshold
        );

        // 2.  Run the DFS over the vocabulary prefix tree.
        helper.run_dfs();

        // 3.  Collect statistics & finish progress-bar.
        helper.finish(token_name_map)
    }

    // -------------------------------------------------------------------------
    // Runtime interface -------------------------------------------------------------------
    // -------------------------------------------------------------------------
    pub fn init(&self) -> GrammarConstraintState<'_> {
        let base_set_for_info = HybridBitset::ones(self.internal_max_llm_token + 1);
        let info = LLMTokenInfo {
            active:       base_set_for_info.clone(),
            intersection: base_set_for_info,
        };
        let mut state = BTreeMap::new();
        state.insert(
            self.tokenizer.initial_state_id(),
            self.parser.init_glr_parser_with_t(info),
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
            all_llm_tokens: HybridBitset::ones(internal_max_llm_token + 1),
            merge_threshold,
            pb,
            stats: PrecomputeStats::default(),
        }
    }

    fn definitely_matches(&self, vocab_node: &VocabPrefixTreeNode, tokenizer_state_id: TokenizerStateID) -> BTreeMap<GrammarTokenID, LLMTokenBV> {
        // Tells us which LLM tokens could match (starting from the vocab node) the specified grammar token.
        // TODO: Implement this. Ensure it's cached.
        todo!()
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
        self.dfs(&self.vocab.root, assoc);
        crate::debug!(2, "Finished precompute DFS");
        self.pb.finish_with_message("Precomputation complete");
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

        // The Precomputed type now holds `Arc<Mutex<PrecomputeNode>>`.
        // We no longer need to unwrap/clone the roots into plain PrecomputeNode.
        // The roots map is already the correct type.
        let out = self.roots;

        // We don't need the cloning warning check anymore as we are not unwrapping/cloning into plain nodes here.

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
                .extend(set.iter().cloned());
        }

        while let Some((offset, map_at_offset)) = queue.pop_first() {
            for (state_before, src_set) in map_at_offset {
                if src_set.is_empty() {
                    continue;
                }

                let src_set = self.merge_handles(&src_set);
                if src_set.is_empty() {
                    continue;
                }

                let suffix      = &segment_bytes[offset..];
                let exec_result = self
                    .tokenizer
                    .execute_from_state(suffix, state_before);
                crate::debug!(4, "Executed tokenizer from state {:?} on suffix {:?}. Results: {:?}", state_before.0, String::from_utf8_lossy(suffix), exec_result);

                // -------------------------------------------------------------
                // Matches inside suffix
                // -------------------------------------------------------------
                for m in &exec_result.matches {
                    let grammar_tok = GrammarTokenID(m.id);
                    let match_end_offset = offset + m.width;
                    let edge_tokens = child_vocab_of_segment.reachable_token_ids().clone();


                    for src in &src_set {
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

                // -------------------------------------------------------------
                // Final tokenizer state after reading entire suffix
                // -------------------------------------------------------------
                if let Some(final_state_val) = exec_result.end_state {
                    let final_sid = TokenizerStateID(final_state_val);
                    for src in &src_set {
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
                            crate::debug!(4, "Pushing finalizer info for token {:?} in state {:?}", gtid.0, final_sid.0);
                            guard.value.push_finalizer_info(
                                gtid,
                                LLMTokenID(child_vocab_of_segment.token_id()),
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
        final_llm_token_id_at_child_vocab: usize,
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
                          pot: &mut Vec<Arc<Mutex<PrecomputeNode>>>| {
            pot.extend(
                set.iter()
                    .map(|h| h.as_arc().clone()),
            );
        };

        if match_end_offset_in_segment < segment_len {
            if let Some(map_at_offset) = queue.get(&match_end_offset_in_segment) {
                for set_of_nodes_at_offset_and_any_state in map_at_offset.values() {
                    gather_set(set_of_nodes_at_offset_and_any_state, &mut pot);
                }
            }
        } else {
             for set_of_nodes_at_next_level_any_state in next_level.values() {
                gather_set(set_of_nodes_at_next_level_any_state, &mut pot);
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
            crate::debug!(4, "Marking clean end for child vocab node {:p} representing LLM token {:?}", handle.as_ref(), final_llm_token_id_at_child_vocab);
            next_level
                .entry(TokenizerStateID(0))
                .or_default()
                .insert(handle);

            // mark clean_end
            let mut g = target.lock().unwrap();
            g.value
                .clean_end
                .get_or_insert_with(HybridBitset::new)
                .insert(final_llm_token_id_at_child_vocab);
        } else {
            queue
                .entry(match_end_offset_in_segment)
                .or_default()
                .entry(TokenizerStateID(0))
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

        let merged = Arc::new(Mutex::new(PrecomputeNode::new(
            PrecomputedNodeContents::default(),
        )));

        for child in set {
            let edge_tokens_for_merge = self.all_llm_tokens.clone();
            let mut inserter = EdgeInserter::new(
                child.as_arc().clone(), // Source of the edge
                None::<GrammarTokenID>,   // Key for the edge (epsilon)
                edge_tokens_for_merge.clone(), // Data for the edge
                |existing_edge_data: &mut HybridBitset, new_edge_data| *existing_edge_data |= new_edge_data,
            );

            // Try to reuse an existing child of `child.as_arc()`
            inserter = inserter.try_children();

            // If no suitable existing child was found, make the new `merged` node the destination
            if inserter.clone_into_option().is_none() {
                inserter = inserter.try_destination(merged.clone());
            }

            // Now, `inserter.clone_into_option()` should contain Some(destination_node).
            // Update the `active` field of this destination_node.
            if let Some(destination_node_arc) = inserter.clone_into_option() {
                let mut guard = destination_node_arc.lock().unwrap();
                guard.value.active |= &edge_tokens_for_merge;
            }
        }

        let mut out = BTreeSet::new();
        out.insert(ArcPtrWrapper::new(merged));
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
#[derive(Debug, Clone)] // Not serialized directly
pub struct GrammarConstraintState<'a> {
    parent: &'a GrammarConstraint,
    state:  BTreeMap<TokenizerStateID, GLRParserState<'a, LLMTokenInfo>>,
}

impl<'a> GrammarConstraintState<'a> {
    pub fn get_mask(&mut self) -> LLMTokenBV {
        let mut internal_mask = HybridBitset::new(); // This will be composed of internal IDs
        for (_tokenizer_state_id, glr_parser_state) in &self.state {
            for active_state in glr_parser_state.active_states.values() {
                internal_mask |= &active_state.stack.peek().t.active; // .active is already internal
            }
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
        let all_true_token_info = LLMTokenInfo {
            active: all_true_set.clone(),
            intersection: all_true_set.clone(),
        };

        // Convert original LLMTokenID to internal LLMTokenID for the closure
        // Ensure the original ID is actually in the map before attempting unwrap
        let internal_llm_id_val = match self.parent.original_id_to_internal(llm_token_id) {
             Some(internal_id) => internal_id.0,
             None => {
                 // If the token ID isn't in the map, this commit can't be valid.
                 // The state should become invalid. Clear all active states.
                 self.state.clear();
                 return;
             }
        };

        let closure = |content: &ParseStateNodeContent<LLMTokenInfo>| -> Option<(ParseStateNodeContent<LLMTokenInfo>, bool)> {
            if content.t.active.contains(internal_llm_id_val) { // .active is internal, compare with internal ID
                if content.t.intersection == all_true_set {
                    Some((ParseStateNodeContent { state_id: content.state_id, t: all_true_token_info.clone() }, false))
                } else {
                    Some((ParseStateNodeContent { state_id: content.state_id, t: all_true_token_info.clone() }, true))
                }
            } else { // Original token ID not found in mapping, so it cannot be active
                None
            }
        };

        let mut memo = HashMap::new();
        self.state.retain(|_tokenizer_state_id, glr_state| {
            glr_state.active_states.retain(|_key, parse_state| {
                // crate::debug!(4, "Pruning parse state {:?}", parse_state.key());
                let maybe_new_node = prune_and_transform_recursive(&parse_state.stack, &closure, &mut memo);
                // crate::debug!(4, "Pruned parse state {:?}. Got new node {:?}.", parse_state.key(), maybe_new_node);
                if let Some(new_node) = maybe_new_node {
                    parse_state.stack = new_node;
                    true
                } else {
                    false
                }
            });
            !glr_state.active_states.is_empty()
        });
    }

    pub fn step_with_llm_token_sequence(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.step_with_llm_token(llm_token_id);
        }
    }

    fn prepare_initial_nodes_and_values_for_special_map(&mut self, llm_tokens: &LLMTokenBV) -> Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'_, LLMTokenInfo>)> {
        let mut initial_nodes_and_values: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'_, LLMTokenInfo>)> = Vec::new();
        let mut tokenizer_state_id_to_parse_states: BTreeMap<TokenizerStateID, GLRParserState<'_, LLMTokenInfo>> = BTreeMap::new();

        for (tokenizer_state_id, state) in &self.state {
            let mut cloned_state = state.clone();
            for parse_state in cloned_state.active_states.values_mut() {
                Arc::make_mut(&mut parse_state.stack).value.t.active &= llm_tokens;
            }
            tokenizer_state_id_to_parse_states.insert(*tokenizer_state_id, cloned_state);
        }

        crate::debug!(4, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        crate::debug!(4, "Printing initial nodes and values for tokenizer states");
        for tokenizer_state_id in tokenizer_state_id_to_parse_states.keys() {
            let glr_state_before = &self.state[&tokenizer_state_id];
            let glr_state_after = &tokenizer_state_id_to_parse_states[&tokenizer_state_id];
            glr_state_before.log_gss(format!("Existing initial nodes and values for tokenizer state {}", tokenizer_state_id.0).as_str(), GrammarTokenID(0));
            glr_state_after.log_gss(format!("Prepared (stage 1) initial nodes and values for tokenizer state {}", tokenizer_state_id.0).as_str(), GrammarTokenID(0));
        }
        crate::debug!(4, "----------------------------------------------------------------");

        for (tokenizer_state_id, state) in tokenizer_state_id_to_parse_states {
            // The roots map already holds Arc<Mutex<PrecomputeNode>>
            if let Some(token_trie_arc_mutex) = self.parent.precomputed.get(&tokenizer_state_id) {
                 initial_nodes_and_values.push((token_trie_arc_mutex.clone(), state));
            } else {
                 // If a tokenizer state from self.state has no corresponding precomputed root,
                 // something is wrong, or this state should perhaps be filtered out earlier.
                 // For now, we'll just skip it.
                 eprintln!("Warning: Tokenizer state {} in state.keys() has no corresponding precomputed root.", tokenizer_state_id.0);
            }

        }


        initial_nodes_and_values
    }

    pub fn step(&mut self, llm_tokens: &LLMTokenBV) {
        crate::debug!(2, "Stepping grammar constraint state with tokenizer states {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens);

        // Initialize the counter
        let step_counts = RefCell::new(BTreeMap::<GrammarTokenID, usize>::new());

        self.state = BTreeMap::new();

        Trie::special_map(
            initial_nodes_and_values,
            |glr_parse_state, grammar_token_id, edge_llm_tokens, child_node| {
                let node_ptr = std::ptr::addr_of!(*child_node);
                crate::debug!(3, "Processing grammar node {:p} token {:?} with {} active states", node_ptr, grammar_token_id.map(|gtid| gtid.0), glr_parse_state.active_states.len());
                let mut cloned_glr_parse_state = glr_parse_state.clone();
                cloned_glr_parse_state.active_states.retain(|_key, parse_state| {
                    let current_active_tokens = parse_state.stack.value.t.active.clone();
                    Arc::make_mut(&mut parse_state.stack).value.t.intersection &= &current_active_tokens;
                    Arc::make_mut(&mut parse_state.stack).value.t.active &= edge_llm_tokens;
                    // // TODO: delete this
                    // if parse_state.stack.value.t.active.is_empty() {
                    //     crate::debug!(4, "Pruning parse state {:?} because it has no active tokens", parse_state.key());
                    // }
                    !parse_state.stack.value.t.active.is_empty()
                });
                if let Some(gtid) = grammar_token_id {
                    *step_counts.borrow_mut().entry(*gtid).or_insert(0) += 1;
                }
                grammar_token_id.map(|gtid| cloned_glr_parse_state.step(gtid));
                if cloned_glr_parse_state.active_states.is_empty() {
                    crate::debug!(3, "No active states after processing grammar token {:?}", grammar_token_id.map(|gtid| gtid.0));
                    return None;
                } else {
                    crate::debug!(3, "Processed grammar token {:?}, {} active states.", grammar_token_id.map(|gtid| gtid.0), cloned_glr_parse_state.active_states.len());
                    Some(cloned_glr_parse_state)
                }
            },
            |managed_parse_state1, managed_parse_state2| {
                managed_parse_state1.merge_with(managed_parse_state2);
            },
            |node, current_glr_parse_state| {
                let mut active_llm_tokens = HybridBitset::new();
                for parse_state in current_glr_parse_state.active_states.values() {
                    active_llm_tokens |= &parse_state.stack.peek().t.active;
                }
                let node_ptr = std::ptr::addr_of!(*node);
                crate::debug!(3, "Processing node {:p} with {} active states, {} LLM tokens, {} finalizers", node_ptr, current_glr_parse_state.active_states.len(), active_llm_tokens.len(), node.value.finalizers().len()); // Use .finalizers()
                if let Some(clean_end) = &node.value.clean_end {
                    let mut final_glr_parse_state = current_glr_parse_state.clone();
                    final_glr_parse_state.active_states.retain(|_key, parse_state| {
                        let current_active_tokens = parse_state.stack.value.t.active.clone();
                        Arc::make_mut(&mut parse_state.stack).value.t.intersection &= &current_active_tokens;
                        Arc::make_mut(&mut parse_state.stack).value.t.active &= clean_end;
                        !parse_state.stack.value.t.active.is_empty()
                    });
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
                    possible_next_glr_parse_state.step(*possible_final_grammar_token);
                    if possible_next_glr_parse_state.is_ok() {
                        crate::debug!(3, "Semi-final GLR parse state is OK");
                        for (tokenizer_state_id, llm_tokens_from_finalizer) in &precomputed_finalizer.content {
                            let mut glr_parse_state_filtered = current_glr_parse_state.clone(); // Start from current_glr_parse_state for filtering
                            glr_parse_state_filtered.active_states.retain(|_key, parse_state| {
                                let current_active_tokens = parse_state.stack.value.t.active.clone();
                                Arc::make_mut(&mut parse_state.stack).value.t.intersection &= &current_active_tokens;
                                Arc::make_mut(&mut parse_state.stack).value.t.active &= llm_tokens_from_finalizer;
                                !parse_state.stack.value.t.active.is_empty()
                            });

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
                !current_glr_parse_state.active_states.is_empty()
            },
        );

        // Simplify the GSS forest
        let mut roots = Vec::new();
        for (tokenizer_state_id, glr_state) in self.state.iter() {
            for active_state in glr_state.active_states.values() {
                let root = active_state.stack.clone();
                roots.push(root);
            }
        }
        let mut i = 0;
        let simplified_roots = simplify_gss_forest(&roots);
        for (tokenizer_state_id, glr_state) in self.state.iter_mut() {
            // Collect keys to avoid iterator invalidation during retain
            let keys: Vec<_> = glr_state.active_states.keys().cloned().collect();
            for key in keys {
                if let Some(active_state) = glr_state.active_states.get_mut(&key) {
                    if i < simplified_roots.len() {
                         active_state.stack = simplified_roots[i].clone();
                         i += 1;
                    } else {
                         // This implies a mismatch between original roots count and simplified roots count.
                         // This should not happen if simplify_gss_forest preserves the number of roots.
                         eprintln!("Warning: Mismatch in root counts during GSS simplification update.");
                         // As a fallback, remove the active state if no simplified root is available
                         glr_state.active_states.remove(&key);
                    }
                }
            }
            // Ensure no extra simplified roots are left over
            if i < simplified_roots.len() {
                 eprintln!("Warning: {} simplified roots were not assigned to any active state.", simplified_roots.len() - i);
            }
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
                println!("  Token \"{}\" (ID: {}): {} calls", token_name, gtid.0, count);
            }
            println!("--------------------------------------------------------------------------");
        } else {
            println!("--- GLRParserState::step was not called in this GrammarConstraintState::step ---");
        }
    }
}
