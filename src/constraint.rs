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
use crate::datastructures::gss::prune_and_transform_recursive;
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
        // TODO: delete this
        // Temporarily just don't map
        let mut original_to_internal_id_bimap = BiBTreeMap::new();
        let mut internal_id_counter = 0;
        for _ in 0..original_llm_token_map.len() {
            original_to_internal_id_bimap.insert(internal_id_counter, internal_id_counter);
            internal_id_counter += 1;
        }
        return original_to_internal_id_bimap;

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

        let internal_max_llm_token = original_to_internal_id_bimap.len();

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
            if let Some(internal_id_val) = self.original_to_internal_id_bimap.get_by_left(&(original_id_val as usize)) {
                internal_bv.insert(*internal_id_val as usize);
            }
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
        crate::debug!(4, "Processing vocab node {:p}", vocab_node_ptr);
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
                internal_mask |= &active_state.stack.peek().t.active; // .active is already internal
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

        let closure = |content: &ParseStateNodeContent<LLMTokenInfo>| -> Option<(ParseStateNodeContent<LLMTokenInfo>, bool)> {
            if let Some(internal_llm_id_val) = maybe_internal_llm_id_val {
                if content.t.active.contains(internal_llm_id_val) { // .active is internal, compare with internal ID
                    if content.t.intersection == all_true_intersection {
                         Some((ParseStateNodeContent { state_id: content.state_id, t: all_true_token_info.clone() }, false))
                    } else {
                         Some((ParseStateNodeContent { state_id: content.state_id, t: all_true_token_info.clone() }, true))
                    }
                } else { // Original token ID not found in mapping, so it cannot be active
                    None
                }
            } else { // Original token ID not found in mapping, so it cannot be active
                None
            }
        };

        let mut memo = HashMap::new();
        self.state.retain(|_tokenizer_state_id, glr_state| {
            glr_state.active_states.retain(|_key, parse_state| {
                let maybe_new_node = prune_and_transform_recursive(&parse_state.stack, &closure, &mut memo);
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

    fn prepare_initial_nodes_and_values_for_special_map(&mut self, llm_tokens: &LLMTokenBV) -> Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'a, LLMTokenInfo>)> {
        let mut initial_nodes_and_values: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'_, LLMTokenInfo>)> = Vec::new();
        let mut tokenizer_state_id_to_parse_states: BTreeMap<TokenizerStateID, GLRParserState<'_, LLMTokenInfo>> = BTreeMap::new();

        for (tokenizer_state_id, state) in &self.state {
            let mut cloned_state = state.clone();
            for parse_state in cloned_state.active_states.values_mut() {
                Arc::make_mut(&mut parse_state.stack).value.t.active &= llm_tokens;
            }
            tokenizer_state_id_to_parse_states.insert(*tokenizer_state_id, cloned_state);
        }

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
                    !parse_state.stack.value.t.active.is_empty()
                });
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
                    active_llm_tokens |= &parse_state.stack.value.t.active;
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
    }
}


