// src/constraint.rs
#![allow(clippy::too_many_arguments)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::BitOr;
use std::sync::{Arc, Mutex};

use bimap::BiBTreeMap;
use bitvec::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

use crate::constraint_extra::{calculate_final_stats, print_precompute_stats, PrecomputeStats};
use crate::datastructures::charmap::TrieMap;
use crate::datastructures::gss::{prune_and_transform_roots, GSSNode}; // Use prune_and_transform_roots
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::{EdgeInserter, Trie};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::datastructures::ArcPtrWrapper;
use crate::finite_automata::Regex;
use crate::glr::parser::{
    MergeAndIntersect, GLRParser, GLRParserState, ParseState, // ParseStateNodeContent is removed
};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use crate::types::TerminalID as GrammarTokenID;
use crate::glr::table::StateID; // Import StateID

pub type LLMTokenBV = HybridBitset;
pub type GrammarTokenBV = BitVec;

// -----------------------------------------------------------------------------
// Small data-types used by the constraint
// -----------------------------------------------------------------------------
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
#[derive(Default, Debug, Clone)]
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

#[derive(Default, Debug, Clone)]
pub struct PrecomputedNodeContents {
    finalizers: BTreeMap<GrammarTokenID, PrecomputedFinalizer>,
    pub clean_end: Option<LLMTokenBV>,
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
// PrecomputeNode is Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>
pub type PrecomputeNode =
    Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>;
pub type Precomputed = BTreeMap<TokenizerStateID, PrecomputeNode>;

// -----------------------------------------------------------------------------
// GrammarConstraint – public facing object
// -----------------------------------------------------------------------------
#[derive(Debug, Clone)]
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

        let internal_max_llm_token = original_to_internal_id_bimap.iter().map(|(_, id)| *id).max().unwrap_or(0); // handle empty map

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
        internal_llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>, // Renamed and now contains internal IDs
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

        // 3.  Collect statistics & finish progress-bar.
        helper.finish(token_name_map)
    }

    // -------------------------------------------------------------------------
    // Runtime interface -------------------------------------------------------------------
    // -------------------------------------------------------------------------
    pub fn init(&self) -> GrammarConstraintState<'_> {
        let base_set_for_info = HybridBitset::ones(self.internal_max_llm_token);
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
            all_llm_tokens: HybridBitset::ones(internal_max_llm_token),
            merge_threshold,
            pb,
            stats: PrecomputeStats::default(),
        }
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
        let mut yellow = HashSet::new();
        self.dfs(&self.vocab.root, assoc, &mut yellow);
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

        // Turn Arc<Mutex<…>> roots into plain PrecomputeNode roots.
        let mut out = Precomputed::new();
        let mut clones = 0;

        for (sid, arc) in self.roots {
            match Arc::try_unwrap(arc) {
                Ok(mutex) => out.insert(
                    sid,
                    mutex.into_inner().expect("Mutex poisoned during unwrap"),
                ),
                Err(arc) => {
                    clones += 1;
                    out.insert(sid, arc.lock().unwrap().clone())
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
        yellow: &mut HashSet<*const VocabPrefixTreeNode>,
    ) {
        self.pb.inc(1);

        let vocab_node_ptr = vocab_node as *const VocabPrefixTreeNode;
        if yellow.contains(&vocab_node_ptr) {
            // This vocab_node is already in the current DFS processing path, skip.
            crate::debug!(4, "Skipping vocab node {:p} because it's already in the current DFS processing path", vocab_node_ptr);
            return;
        }
        crate::debug!(4, "Processing vocab node {:p}", vocab_node_ptr); //
        yellow.insert(vocab_node_ptr);

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

            self.process_segment(segment_bytes, child_vocab_ref, &effective, yellow);
        }

        yellow.remove(&vocab_node_ptr);
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
        yellow: &mut HashSet<*const VocabPrefixTreeNode>,
    ) {
        // Maps used while consuming the segment byte-by-byte.
        let mut next_level: BTreeMap<
            TokenizerStateID,
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        > = BTreeMap::new();

        let mut queue: BTreeMap<
            usize, // offset in segment_bytes
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
                    // Edge tokens are all tokens reachable *from* this vocab node *without consuming further bytes*
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
        self.dfs(child_vocab_of_segment, next_level, yellow);
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
            edge_tokens,
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
            let mut ins = EdgeInserter::new(
                child.as_arc().clone(),
                None::<GrammarTokenID>,
                self.all_llm_tokens.clone(),
                |ev: &mut HybridBitset, new| *ev |= new, // Pass new by reference
            )
            .try_children();

            if ins.clone_into_option().is_none() {
                let _ = ins.try_destination(merged.clone());
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
#[derive(Debug, Clone)]
pub struct GrammarConstraintState<'a> {
    parent: &'a GrammarConstraint,
    state:  BTreeMap<TokenizerStateID, GLRParserState<'a, LLMTokenInfo>>,
}

impl<'a> GrammarConstraintState<'a> {
    pub fn get_mask(&mut self) -> LLMTokenBV {
        let mut internal_mask = HybridBitset::new(); // This will be composed of internal IDs
        for (_tokenizer_state_id, glr_parser_state) in &self.state {
            for active_state in glr_parser_state.active_states.values() {
                // Access current_t on the ParseState
                internal_mask |= &active_state.current_t.active; // .active is already internal
            }
        }
        self.parent.internal_bv_to_original(&internal_mask) // Convert back to original IDs
    }

    pub fn step_with_all_llm_tokens(&mut self) {
        // This creates a bitset of all *internal* LLM tokens
        let all_internal_llm_tokens = HybridBitset::ones(self.parent.internal_max_llm_token);
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
        let all_true_set = HybridBitset::ones(self.parent.internal_max_llm_token);
        let all_true_token_info = LLMTokenInfo {
            active: all_true_set.clone(),
            intersection: all_true_set,
        };
        let all_true_intersection = all_true_token_info.intersection.clone();

        // Convert original LLMTokenID to internal LLMTokenID for the closure
        let maybe_internal_llm_id_val = self.parent.original_id_to_internal(llm_token_id)
                                          .map(|id| id.0 as usize);

        // Closure for prune_and_transform_roots
        // Takes (node_content, edge_value_to_this_node)
        // Returns Option<(new_node_content, new_edge_value_to_this_node, continue_recursion)>
        let closure = |state_id: &StateID, llm_info_to_node: &LLMTokenInfo| -> Option<(StateID, LLMTokenInfo, bool)> {
             if let Some(internal_llm_id_val) = maybe_internal_llm_id_val {
                 // Check if the required LLM token is in the active set *on the edge leading to this node*
                 if llm_info_to_node.active.contains(internal_llm_id_val) {
                     // The new value for the edge is `all_true_token_info`
                     let new_llm_info_for_edge = all_true_token_info.clone();

                     // Decide whether to continue recursion based on intersection
                     let continue_recursion = llm_info_to_node.intersection != all_true_intersection;

                     Some((*state_id, new_llm_info_for_edge, continue_recursion))
                 } else {
                     // The required token is not active on the edge to this node, prune this path
                     None
                 }
             } else {
                 // Original token ID not found in mapping, cannot be active. Prune this path.
                 None
             }
        };

        // Collect root GSS nodes from the active states to pass to prune_and_transform_roots
        // Need to pass roots as Vec<(Arc<GSSNode<N, E>>, E)> where E is the edge value to the root.
        // For initial roots, there's no preceding edge within the GSS, so we can pass a default E or handle it.
        // In the context of ParseState, each active ParseState represents a path ending at `parse_state.gss_node`,
        // and `parse_state.current_t` is the semantic value *at* that node. This is not the edge value *to* the node.
        // The `prune_and_transform_roots` as defined expects an edge value associated with each root.

        // Let's rethink `commit`. The pruning should happen based on the *accumulated* semantic value *at* each GSS node.
        // The `current_t` in `ParseState` already represents this accumulated value.
        // The prune logic needs to check `parse_state.current_t.active`.
        // If a ParseState is pruned, its associated GSS node and its predecessors via that path should be pruned unless
        // other ParseStates still use those nodes/paths.

        // The `prune_and_transform_recursive` function should operate on `Arc<GSSNode<N, E>>` and the accumulated value `T`.
        // Let's redefine the closure signature and the prune functions.
        // Closure: `Fn(&N, &E, &T_accumulated) -> Option<(N, E, T_accumulated, bool)>` ? No, this is getting complex.

        // Let's stick to the GSS definition and `prune_and_transform_roots` as defined in gss.rs
        // It expects `Vec<(Arc<GSSNode<N, E>>, E)>` as roots and `Fn(&N, &E) -> Option<(N, E, bool)>`.
        // This means the pruning/transformation is based on the node's content (N) and the incoming edge value (E).

        // The commit logic needs to filter/transform ParseStates based on their `current_t`.
        // This suggests `prune_and_transform_roots` might not be the right tool directly on the GSS forest of ParseStates.
        // Instead, we iterate through `active_states`, apply the commit logic based on `current_t`, and then rebuild/clean up the GSS.

        self.state.retain(|_tokenizer_state_id, glr_state| {
             let mut new_active_states: BTreeMap<StateID, ParseState<T>> = BTreeMap::new();
             for (_key, parse_state) in std::mem::take(&mut glr_state.active_states) {
                 // Apply commit logic based on parse_state.current_t
                 if let Some(internal_llm_id_val) = maybe_internal_llm_id_val {
                     if parse_state.current_t.active.contains(internal_llm_id_val) {
                         // This ParseState is kept, transform its current_t
                         let mut transformed_parse_state = parse_state;
                         transformed_parse_state.current_t = all_true_token_info.clone(); // Transform T

                         // Insert into the new map, merging with existing if key (StateID) is same
                         new_active_states.insert_with(transformed_parse_state.key(), transformed_parse_state, |existing, new_s| existing.merge(new_s));
                     } else {
                         // Prune this ParseState
                         crate::debug!(4, "Pruning ParseState due to commit token not being active in current_t");
                     }
                 } else {
                     // Original token ID not found, prune all states
                     crate::debug!(4, "Pruning ParseState due to original token ID not found in map");
                 }
             }
             glr_state.active_states = new_active_states;
             !glr_state.active_states.is_empty()
         });

         // After pruning ParseStates, the underlying GSS nodes might have become unreachable.
         // The `Arc` mechanism and the `Drop` implementation in `GSSNode` should handle cleanup.
         // However, shared nodes might still be referenced by other tokenizer states.
         // This seems correct.
    }

    pub fn step_with_llm_token_sequence(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.step_with_llm_token(llm_token_id);
        }
    }

    fn prepare_initial_nodes_and_values_for_special_map(&mut self, llm_tokens: &LLMTokenBV) -> Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'a, LLMTokenInfo>)> {
        let mut initial_nodes_and_values: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'_, LLMTokenInfo>)> = Vec::new();
        let mut tokenizer_state_id_to_parse_states: BTreeMap<TokenizerStateID, GLRParserState<'_, LLMTokenInfo>> = BTreeMap::new();

        for (tokenizer_state_id, state) in &self.state {
            let mut cloned_state = state.clone();
            // Filter/intersect the current_t.active on the ParseStates
            cloned_state.active_states.retain(|_key, parse_state| {
                Arc::make_mut(&mut parse_state.current_t).active &= llm_tokens; // Modify current_t's active set
                // Prune if active set becomes empty
                !parse_state.current_t.active.is_empty()
            });
             // Keep the GLRParserState only if it still has active ParseStates
            if !cloned_state.active_states.is_empty() {
                 tokenizer_state_id_to_parse_states.insert(*tokenizer_state_id, cloned_state);
            }
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
            let token_trie_node = self.parent.precomputed[&tokenizer_state_id].clone();
            let token_trie_arc_mutex = Arc::new(Mutex::new(token_trie_node));
            initial_nodes_and_values.push((token_trie_arc_mutex, state));
        }


        initial_nodes_and_values
    }

    pub fn step(&mut self, llm_tokens: &LLMTokenBV) {
        crate::debug!(2, "Stepping grammar constraint state with tokenizer states {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens);

        self.state = BTreeMap::new(); // Clear current state, will be rebuilt

        Trie::special_map(
            initial_nodes_and_values,
            |mut glr_parse_state, grammar_token_id, edge_llm_tokens, child_node| { // mut glr_parse_state to allow modification
                let node_ptr = std::ptr::addr_of!(*child_node);
                crate::debug!(3, "Processing grammar node {:p} token {:?} with {} active states", node_ptr, grammar_token_id.map(|gtid| gtid.0), glr_parse_state.active_states.len());

                // Filter/intersect the active LLM tokens on the ParseStates using the edge_llm_tokens from the Trie edge
                glr_parse_state.active_states.retain(|_key, parse_state| {
                    // current_t is the semantic value *at* the source node (before this edge)
                    // The active tokens on the edge should be intersected with the current_t.active
                    // and the intersection of the edge tokens and current_t.intersection should be propagated.
                    // Let's update current_t based on the edge_llm_tokens from the trie.

                    let current_active_tokens_at_source = parse_state.current_t.active.clone();

                    // The semantic value propagated *through* this trie edge
                    let mut propagated_t = parse_state.current_t.clone();
                    propagated_t.active &= edge_llm_tokens;
                    propagated_t.intersection &= edge_llm_tokens; // Intersect with edge tokens

                    // Update the ParseState's current_t to the propagated value *after* the edge traversal
                    // This seems incorrect. The `current_t` is the value *at the GSS node*.
                    // The semantic values `E` are on the GSS edges.
                    // The `special_map` gives us `edge_llm_tokens` which is the set of LLM tokens compatible with this trie edge (GrammarTokenID).
                    // We need to filter/transform the active ParseStates based on these `edge_llm_tokens`.

                    // Let's try filtering the ParseStates based on if their `current_t.active` has any overlap with `edge_llm_tokens`.
                    // Then, for the ones that remain, the `current_t.active` should be intersected with `edge_llm_tokens`.
                    // And `current_t.intersection` should also be intersected.

                    Arc::make_mut(&mut parse_state.current_t).active &= edge_llm_tokens;
                    Arc::make_mut(&mut parse_state.current_t).intersection &= edge_llm_tokens; // Also intersect intersection

                    // Prune the ParseState if its active set becomes empty after intersection
                    !parse_state.current_t.active.is_empty()
                });

                if let Some(gtid) = grammar_token_id {
                    // Step the GLR parser state with the grammar token
                    glr_parse_state.step(gtid);
                }

                if glr_parse_state.active_states.is_empty() {
                    crate::debug!(3, "No active states after processing grammar token {:?}", grammar_token_id.map(|gtid| gtid.0));
                    None
                } else {
                    crate::debug!(3, "Processed grammar token {:?}, {} active states.", grammar_token_id.map(|gtid| gtid.0), glr_parse_state.active_states.len());
                    Some(glr_parse_state)
                }
            },
            |managed_parse_state1, managed_parse_state2| {
                managed_parse_state1.merge_with(managed_parse_state2);
            },
            |node, mut current_glr_parse_state| { // mut current_glr_parse_state to allow modification
                let mut active_llm_tokens = HybridBitset::new();
                for parse_state in current_glr_parse_state.active_states.values() {
                    active_llm_tokens |= &parse_state.current_t.active; // Access current_t
                }
                let node_ptr = std::ptr::addr_of!(*node);
                crate::debug!(3, "Processing node {:p} with {} active states, {} LLM tokens, {} finalizers", node_ptr, current_glr_parse_state.active_states.len(), active_llm_tokens.len(), node.value.finalizers().len()); // Use .finalizers()

                // Handle clean end
                if let Some(clean_end_llm_tokens) = &node.value.clean_end {
                    let mut final_glr_parse_state = current_glr_parse_state.clone();
                    final_glr_parse_state.active_states.retain(|_key, parse_state| {
                        // Intersect with clean_end tokens
                        Arc::make_mut(&mut parse_state.current_t).active &= clean_end_llm_tokens;
                        Arc::make_mut(&mut parse_state.current_t).intersection &= clean_end_llm_tokens; // Also intersect intersection
                        !parse_state.current_t.active.is_empty()
                    });
                    crate::debug!(3, "At clean end state");
                    if final_glr_parse_state.is_ok() {
                        crate::debug!(3, "GLR parse state at clean end is OK");
                        // Merge the final_glr_parse_state into the overall constraint state for TokenizerStateID(0)
                        if let Some(existing) = self.state.get_mut(&TokenizerStateID(0)) {
                            existing.merge_with(final_glr_parse_state); // Pass the modified state
                        } else {
                            self.state.insert(TokenizerStateID(0), final_glr_parse_state);
                        }
                    }
                }

                // Handle finalizers
                for (possible_final_grammar_token, precomputed_finalizer) in node.value.finalizers().iter() { // Use .finalizers()
                    let mut glr_parse_state_after_finalizer_step = current_glr_parse_state.clone();
                    crate::debug!(3, "Stepping semi-final GLR parse state with grammar token {:?}", possible_final_grammar_token.0);
                    glr_parse_state_after_finalizer_step.step(*possible_final_grammar_token); // Step with the finalizer grammar token

                    if glr_parse_state_after_finalizer_step.is_ok() {
                        crate::debug!(3, "Semi-final GLR parse state after grammar step is OK");
                        for (tokenizer_state_id, llm_tokens_from_finalizer) in &precomputed_finalizer.content {
                            // For each finalizer entry, filter the states *after* the grammar step
                            let mut glr_parse_state_filtered = glr_parse_state_after_finalizer_step.clone(); // Start from state *after* grammar step
                             glr_parse_state_filtered.active_states.retain(|_key, parse_state| {
                                 // Intersect with finalizer's LLM tokens
                                 Arc::make_mut(&mut parse_state.current_t).active &= llm_tokens_from_finalizer;
                                 Arc::make_mut(&mut parse_state.current_t).intersection &= llm_tokens_from_finalizer; // Also intersect intersection
                                 !parse_state.current_t.active.is_empty()
                             });

                            crate::debug!(3, "Processing finalizer for token_state_id {:?}", tokenizer_state_id);
                            if glr_parse_state_filtered.is_ok() {
                                crate::debug!(3, "Finalizer is compatible after filtering");
                                if let Some(existing) = self.state.get_mut(tokenizer_state_id) {
                                    existing.merge_with(glr_parse_state_filtered); // Merge the filtered state
                                } else {
                                    self.state.insert(*tokenizer_state_id, glr_parse_state_filtered);
                                }
                            }
                        }
                    }
                }
                // The node is kept in the trie special_map if there are ANY active states associated with it at the end of processing this node.
                !current_glr_parse_state.active_states.is_empty() // This might be incorrect logic now. The trie node should be kept if it leads to *any* valid state in *any* tokenizer state after processing. The special_map returns the filtered/transformed GLRParserState for THIS branch.
                                                                   // The check `!current_glr_parse_state.active_states.is_empty()` here determines if this branch of the Trie traversal should continue. This seems correct.
            },
        );
    }
}
