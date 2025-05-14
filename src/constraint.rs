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
use crate::datastructures::gss::{prune_and_transform_recursive, GSSNode}; // Import GSSNode
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::{EdgeInserter, Trie};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::datastructures::ArcPtrWrapper;
use crate::finite_automata::Regex;
use crate::glr::parser::{
    MergeAndIntersect, GLRParser, GLRParserState, ParseState, // ParseStateNodeContent removed
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
// GSSNode generic parameter T is the edge value type, which is LLMTokenBV for precomputation edges.
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

        let internal_max_llm_token = original_to_internal_id_bimap.iter().map(|(_, id)| *id).max().expect("Internal max token ID"); // Max ID could be 0 if no tokens

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
        let base_set_for_info = HybridBitset::ones(self.internal_max_llm_token + 1); // Need size `internal_max_llm_token + 1` for bitset indices up to `internal_max_llm_token`
        let info = LLMTokenInfo {
            active:       base_set_for_info.clone(),
            intersection: base_set_for_info,
        };
        let mut state = BTreeMap::new();

        // Initialize GLRParserState with the correct initial state and LLMTokenInfo
        let initial_glr_state = self.parser.init_glr_parser_with_t(info);
        state.insert(
            self.tokenizer.initial_state_id(),
            initial_glr_state,
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
                Arc::new(Mutex::new(PrecomputeNode::new(StateID(0), PrecomputedNodeContents::default()))), // Precompute nodes don't have a meaningful StateID themselves, use 0 as placeholder? Or None? Let's use 0 for now.
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
            all_llm_tokens: HybridBitset::ones(internal_max_llm_token + 1), // Need size `internal_max_llm_token + 1`
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
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>, // GSSNode<LLMTokenBV>
        > = BTreeMap::new();

        for (sid, arc) in &self.roots {
            assoc
                .entry(*sid)
                .or_default()
                .insert(ArcPtrWrapper::new(arc.clone()));
        }

        crate::debug!(2, "Starting precompute DFS");
        let mut yellow = HashSet::new();
        // Start DFS from the vocab root node (no edge value leading to it)
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
                effective.insert(*sid, merged);
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
        // (byte_offset_in_segment, tokenizer_state_id) -> set of (ArcPtrWrapper<Mutex<PrecomputeNode>>)
        let mut queue: BTreeMap<
            usize,
            BTreeMap<TokenizerStateID, BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>>,
        > = BTreeMap::new();

        // next_level stores associations for the next DFS level (after processing the entire segment)
        let mut next_level: BTreeMap<
            TokenizerStateID,
            BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>,
        > = BTreeMap::new();


        // Seed queue with offset 0.
        for (sid, set) in sources_per_state {
            queue
                .entry(0) // Starting at offset 0
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

                // Merge nodes that arrive at the same (offset, state_before)
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
                // Matches inside suffix (Grammar Token Found)
                // -------------------------------------------------------------
                for m in &exec_result.matches {
                    let grammar_tok = GrammarTokenID(m.id);
                    let match_end_offset_in_segment = offset + m.width;
                    let edge_llm_tokens = child_vocab_of_segment.reachable_token_ids().clone();

                    for src_wrap in &src_set {
                        // src_wrap points to a PrecomputeNode (GSSNode<LLMTokenBV>)
                        let source_arc = src_wrap.as_arc().clone(); // Arc<Mutex<GSSNode<LLMTokenBV>>>

                        // The destination node should represent the point after consuming `grammar_tok`.
                        // This requires inserting an edge (grammar_tok, edge_llm_tokens) from source_arc.
                        let mut inserter = EdgeInserter::new(
                            source_arc.clone(),
                            Some(grammar_tok),
                            edge_llm_tokens.clone(),
                            |existing: &mut HybridBitset, new_bv_ref: HybridBitset| *existing |= new_bv_ref, // Edge values are LLMTokenBV
                        );

                        // Try to find an existing child node with the same grammar_tok edge
                        inserter = inserter.try_children();

                        // Gather potential targets from queue or next_level based on match_end_offset
                        let mut pot_target_handles: BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>> = BTreeSet::new();

                        if match_end_offset_in_segment < segment_bytes.len() {
                            if let Some(map_at_offset) = queue.get(&match_end_offset_in_segment) {
                                for nodes_at_offset in map_at_offset.values() {
                                     pot_target_handles.extend(nodes_at_offset.iter().cloned());
                                }
                            }
                        } else { // Match reaches or exceeds segment end, potential targets are from next_level
                             for nodes_at_next_level in next_level.values() {
                                 pot_target_handles.extend(nodes_at_next_level.iter().cloned());
                             }
                        }

                        // Try to reuse one of the potential target nodes as the destination for the new edge
                        for target_handle in pot_target_handles {
                            inserter = inserter.try_destination(target_handle.as_arc().clone());
                        }


                        // Get or create the target node for this edge
                        let target_arc_mutex = inserter
                            .else_create_destination_with_value(PrecomputedNodeContents::default())
                            .expect("Failed to get or create destination node"); // Should always succeed

                         // Propagate the association for the target node based on the offset
                        let target_handle = ArcPtrWrapper::new(target_arc_mutex.clone());

                        if match_end_offset_in_segment < segment_bytes.len() {
                            // If match ends within the segment, the target node continues processing the rest of the segment.
                            // It enters the queue at match_end_offset with tokenizer state 0 (since a grammar token matched).
                             queue
                                .entry(match_end_offset_in_segment)
                                .or_default()
                                .entry(TokenizerStateID(0)) // Tokenizer state resets after a match
                                .or_default()
                                .insert(target_handle);
                        } else {
                            // If match ends at or after segment end, the target node is part of the next DFS level.
                             next_level
                                .entry(TokenizerStateID(0)) // Tokenizer state resets after a match
                                .or_default()
                                .insert(target_handle);

                            // Mark clean_end for the target node if this segment is a complete LLM token
                            // The target node corresponds to being AT the end of consuming a grammar token
                            // which finishes at the end of the segment.
                            if match_end_offset_in_segment == segment_bytes.len() {
                                 crate::debug!(4, "Marking clean end for precompute node {:p} at offset {} for LLM token {:?}",
                                               Arc::as_ptr(&target_arc_mutex), match_end_offset_in_segment, child_vocab_of_segment.token_id());

                                 let mut guard = target_arc_mutex.lock().unwrap();
                                 guard.value
                                    .clean_end
                                    .get_or_insert_with(HybridBitset::new)
                                    .insert(child_vocab_of_segment.token_id());
                            }
                        }
                    }
                }

                // -------------------------------------------------------------
                // Final tokenizer state after reading entire suffix (no match)
                // -------------------------------------------------------------
                if let Some(final_state_val) = exec_result.end_state {
                    let final_sid = TokenizerStateID(final_state_val);
                    if offset + suffix.len() == segment_bytes.len() { // Reached end of segment without a match
                        for src_wrap in &src_set {
                            // Propagate association for the next DFS level.
                            // The source node itself is carried over to the next level,
                            // but associated with the new tokenizer state.
                             next_level
                                .entry(final_sid)
                                .or_default()
                                .insert(src_wrap.clone()); // src_wrap points to the GSSNode (PrecomputeNode)

                            // Add finalizers from the source node if any grammar tokens are accessible
                            // from the final tokenizer state.
                            let mut guard = src_wrap.as_arc().lock().unwrap(); // Lock the source node
                            for gtid in self
                                .tokenizer
                                .tokens_accessible_from_state(final_sid)
                            {
                                crate::debug!(4, "Pushing finalizer info for token {:?} in state {:?} on node {:p}", gtid.0, final_sid.0, Arc::as_ptr(&src_wrap.as_arc()));
                                guard.value.push_finalizer_info(
                                    gtid,
                                    LLMTokenID(child_vocab_of_segment.token_id()), // The LLM token corresponding to the current vocab node
                                    final_sid,
                                );
                            }
                        }
                    } else {
                        // Tokenizer ended mid-segment without a match.
                        // This implies an invalid token sequence, so these paths are dead ends for precomputation.
                        crate::debug!(4, "Tokenizer ended mid-segment without match: state {:?}, offset {}, suffix {:?}", final_sid.0, offset, String::from_utf8_lossy(suffix));
                    }
                }
            }
        }

        // Recurse into the child vocab node.
        if !next_level.is_empty() {
             self.dfs(child_vocab_of_segment, next_level, yellow);
        } else {
             crate::debug!(4, "No active paths to recurse for vocab node {:p}", vocab_node_of_segment as *const VocabPrefixTreeNode);
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

        // Create a single new node to merge into. PrecomputeNode's StateID is not relevant here.
        let merged_node_arc_mutex = Arc::new(Mutex::new(PrecomputeNode::new(StateID(0), PrecomputedNodeContents::default())));

        for node_handle in set {
            let source_arc_mutex = node_handle.as_arc().clone(); // Arc<Mutex<GSSNode<LLMTokenBV>>>

            // Insert an edge with None key (representing merging different paths)
            let mut inserter = EdgeInserter::new(
                source_arc_mutex.clone(),
                None::<GrammarTokenID>, // Merge edge has None key
                self.all_llm_tokens.clone(), // Edge value for merge is all possible LLM tokens initially
                |existing: &mut HybridBitset, new_bv_ref: HybridBitset| *existing |= new_bv_ref,
            )
            .try_children(); // Check if there's already a merge edge

            // Try to use the dedicated merged_node as the destination
            inserter = inserter.try_destination(merged_node_arc_mutex.clone());

            // This merge logic is slightly different from the GSS merge.
            // Here, we want all predecessors of the nodes in `set` to become predecessors of `merged_node`.
            // The edge value on the new edge to `merged_node` should be the union of edge values leading to nodes in `set`.

            // This simple EdgeInserter doesn't directly support the GSS-style predecessor merging.
            // A direct GSSNode merge operation is needed here.
            // Let's implement the GSS merge logic manually for this set of nodes.

            let mut merged_predecessors: BTreeSet<PredecessorLink<LLMTokenBV>> = BTreeSet::new();
            let mut merged_value = PrecomputedNodeContents::default(); // This is wrong, value should be on the edge

             // Need to get the *predecessors* of the nodes in `set`
             for node_handle in set {
                let source_arc_mutex = node_handle.as_arc().clone();
                let guard = source_arc_mutex.lock().unwrap(); // Lock the node

                // The value on the edge leading INTO `merged_node_arc_mutex` from a predecessor
                // should be the union of values on edges leading into the nodes in `set` from that same predecessor.
                // This structure doesn't easily give us that info directly.

                // Let's rethink the merge here. In precomputation, we are merging GSSNode<LLMTokenBV>.
                // The LLMTokenBV is on the edge.
                // When merging a set of nodes {N1, N2, ... Nk} all having the same StateID (which is 0 for PrecomputeNode)
                // but arrived via different paths (different predecessors and edge values),
                // the new merged node M should have predecessors that are the union of all predecessors of N1..Nk.
                // If a predecessor P leads to N1 via edge value E1 and to N2 via edge value E2,
                // then P leads to M via edge value E1.merge(E2).

                // Let's collect all unique predecessors and their associated edge values for the nodes in `set`.
                let mut predecessor_edge_values: BTreeMap<ArcPtrWrapper<GSSNode<LLMTokenBV>>, LLMTokenBV> = BTreeMap::new();

                for node_handle in set {
                    let source_arc_mutex = node_handle.as_arc().clone();
                    let guard = source_arc_mutex.lock().unwrap(); // Lock the node

                    for pred_link in &guard.predecessors {
                        predecessor_edge_values
                             .entry(pred_link.node.clone()) // Key is the predecessor node
                             .and_modify(|existing_edge_val| *existing_edge_val = existing_edge_val.merge(&pred_link.edge_value)) // Merge edge values
                             .or_insert(pred_link.edge_value.clone()); // Insert if new predecessor
                    }

                    // Merge the PrecomputedNodeContents (finalizers, clean_end)
                    // This content is AT the node, not on the edge.
                    // Merging means taking the union of finalizers and clean_end sets.
                    // This is where the original GSSNode value was useful.
                    // With value on edge, PrecomputeNodeContent is now the node's value.
                    merged_value.clean_end = match (&merged_value.clean_end, &guard.value.clean_end) {
                        (Some(a), Some(b)) => Some(a | b),
                        (Some(a), None) => Some(a.clone()),
                        (None, Some(b)) => Some(b.clone()),
                        (None, None) => None,
                    };
                    // Merge finalizers (BTreeMap<GrammarTokenID, PrecomputedFinalizer>)
                    for (gtid, finalizer) in &guard.value.finalizers {
                         merged_value.finalizers.entry(*gtid)
                            .and_modify(|existing_finalizer| {
                                 // Merge content (BTreeMap<TokenizerStateID, LLMTokenBV>)
                                 for (tsid, llm_bv) in &finalizer.content {
                                     existing_finalizer.content.entry(*tsid)
                                         .and_modify(|existing_llm_bv| *existing_llm_bv |= llm_bv)
                                         .or_insert(llm_bv.clone());
                                 }
                            })
                            .or_insert_with(|| finalizer.clone());
                    }

                }

                // Create new PredecessorLink for the merged node
                for (pred_node_handle, merged_edge_value) in predecessor_edge_values {
                     merged_predecessors.insert(PredecessorLink {
                         node: pred_node_handle,
                         edge_value: merged_edge_value,
                     });
                }

                // Create the new merged node
                let new_merged_gss_node = GSSNode::new_with_predecessors(
                    StateID(0), // Precompute nodes use StateID(0)
                    merged_predecessors.into_iter().collect()
                );

                // The value of this merged node should be the merged PrecomputedNodeContents
                let merged_node_arc_mutex = Arc::new(Mutex::new(new_merged_gss_node));
                merged_node_arc_mutex.lock().unwrap().value = merged_value; // Assign merged value here

            }


            let mut out = BTreeSet::new();
            out.insert(ArcPtrWrapper::new(merged_node_arc_mutex.clone()));
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
        for (_tokenizer_state_id, glr_parser_state) in &mut self.state { // Need mut to call log_gss if needed
            // Active states are now represented by predecessors of the head node
            for link in &glr_parser_state.head.predecessors {
                // The edge value (LLMTokenInfo) holds the active bitset
                internal_mask |= &link.edge_value.active;
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
            intersection: all_true_set,
        };
        let all_true_intersection = all_true_token_info.intersection.clone();

        // Convert original LLMTokenID to internal LLMTokenID for the closure
        let maybe_internal_llm_id_val = self.parent.original_id_to_internal(llm_token_id)
                                          .map(|id| id.0 as usize);

        // Closure now operates on the (StateID of the node, &LLMTokenInfo on the edge leading to it)
        // and returns Option<((StateID of new node, new LLMTokenInfo for edge), ContinueRecursion)>
        let closure = |(state_id, edge_val): &(StateID, &LLMTokenInfo)| -> Option<((StateID, LLMTokenInfo), bool)> {
            if let Some(internal_llm_id_val) = maybe_internal_llm_id_val {
                if edge_val.active.contains(internal_llm_id_val) { // Check active on edge value
                    // Keep the same state_id for the node, but potentially change the edge value
                    let new_edge_val = if edge_val.intersection == all_true_intersection {
                         all_true_token_info.clone() // Reset intersection
                    } else {
                         edge_val.clone() // Keep original edge value if not fully accepted yet
                    };
                    // Continue recursion unless the edge value implies a final state check might happen
                    // For now, let's continue recursion always during commit, pruning handles the rest.
                    Some(((state_id.clone(), new_edge_val), true))
                } else { // Original token ID not active on this edge, prune this path
                    None
                }
            } else { // Original token ID not in mapping, should not be active on any edge. Prune.
                None
            }
        };

        // Memoization map for prune_and_transform_recursive needs to key by (GSSNode pointer, edge value)
        // This is tricky due to edge value being part of the key for memoization.
        // Let's use a simpler memo key just based on node pointer for now, acknowledging this is not perfect
        // if the outcome of the closure depends *only* on the edge value for a shared node.
        // A better memo key would be HashMap<(*const GSSNode<T>, T), ...> which is complex.
        // For now, HashMap<*const GSSNode<LLMTokenInfo>, Option<(Arc<GSSNode<LLMTokenInfo>>, LLMTokenInfo)>>
        // The return value of prune_and_transform_recursive is Option<(Arc<GSSNode<T>>, T)>
        let mut memo: HashMap<*const GSSNode<LLMTokenInfo>, Option<(Arc<GSSNode<LLMTokenInfo>>, LLMTokenInfo)>> = HashMap::new();


        self.state.retain(|_tokenizer_state_id, glr_state| {
            let mut new_head_links = BTreeSet::new();
            let current_head_links: Vec<_> = glr_state.head.predecessors.iter().cloned().collect(); // Clone links to iterate

            for link in current_head_links {
                // Call prune_and_transform_recursive for each root of this GLR state's GSS forest (each predecessor of the head)
                if let Some((new_node_arc, new_edge_val)) = prune_and_transform_recursive(
                    &link.node.as_arc(), // This is Arc<GSSNode<LLMTokenInfo>>
                    &link.edge_value,   // This is &LLMTokenInfo
                    &closure,
                    &mut memo,
                ) {
                    new_head_links.insert(crate::datastructures::gss::PredecessorLink { // Use full path
                        node: crate::datastructures::ArcPtrWrapper::new(new_node_arc), // Use full path
                        edge_value: new_edge_val,
                    });
                }
            }

            if new_head_links.is_empty() {
                // If no links remain, this GLRParserState becomes inactive for this tokenizer state
                false // for retain
            } else {
                // Update the head node with the new set of links
                glr_state.head = Arc::new(GSSNode::new_with_predecessors(
                    crate::glr::parser::DUMMY_HEAD_STATE_ID, // Use the defined constant
                    new_head_links.into_iter().collect()
                ));
                true // for retain
            }
        });
    }

    pub fn step_with_llm_token_sequence(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.step_with_llm_token(llm_token_id);
        }
    }

    fn prepare_initial_nodes_and_values_for_special_map(&mut self, llm_tokens: &LLMTokenBV) -> Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'a, LLMTokenInfo>)> {
        let mut initial_nodes_and_values: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'_, LLMTokenInfo>)> = Vec::new();

        // Prepare GLRParserState copies for each tokenizer state, filtering ParseStates
        let mut tokenizer_state_id_to_parse_states: BTreeMap<TokenizerStateID, GLRParserState<'_, LLMTokenInfo>> = BTreeMap::new();

        for (tokenizer_state_id, state) in &self.state {
            let mut cloned_state = state.clone();
            let mut new_head_links = BTreeSet::new();
             // Iterate over predecessors of the head node
            let current_links: Vec<_> = cloned_state.head.predecessors.iter().cloned().collect();
            for link in current_links {
                // Filter based on the input llm_tokens bitset
                let mut filtered_edge_value = link.edge_value.clone();
                filtered_edge_value.active &= llm_tokens; // Filter active tokens on the edge

                if !filtered_edge_value.active.is_empty() {
                    new_head_links.insert(crate::datastructures::gss::PredecessorLink { // Use full path
                        node: link.node.clone(), // Keep the same GSSNode pointer
                        edge_value: filtered_edge_value, // Use the filtered edge value
                    });
                }
            }
            // Update the cloned state's head node
            cloned_state.head = Arc::new(GSSNode::new_with_predecessors(
                crate::glr::parser::DUMMY_HEAD_STATE_ID, // Use the defined constant
                new_head_links.into_iter().collect()
            ));


            if cloned_state.is_ok() { // Only include if some ParseStates remain
                tokenizer_state_id_to_parse_states.insert(*tokenizer_state_id, cloned_state);
            }
        }

        crate::debug!(4, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        crate::debug!(4, "Printing initial nodes and values for tokenizer states");
        for tokenizer_state_id in tokenizer_state_id_to_parse_states.keys() {
            let glr_state_after = &tokenizer_state_id_to_parse_states[&tokenizer_state_id];
            // log_gss takes TerminalID, use a dummy value
            glr_state_after.log_gss(format!("Prepared (stage 1) initial nodes and values for tokenizer state {}", tokenizer_state_id.0).as_str(), GrammarTokenID(0));
        }
        crate::debug!(4, "----------------------------------------------------------------");


        for (tokenizer_state_id, state) in tokenizer_state_id_to_parse_states {
            // Get the corresponding precomputed trie root for this tokenizer state
            if let Some(token_trie_node) = self.parent.precomputed.get(&tokenizer_state_id) {
                 let token_trie_arc_mutex = Arc::new(Mutex::new(token_trie_node.clone()));
                 initial_nodes_and_values.push((token_trie_arc_mutex, state));
            } else {
                crate::debug!(4, "Warning: No precomputed trie found for tokenizer state {:?}", tokenizer_state_id);
            }
        }

        initial_nodes_and_values
    }

    pub fn step(&mut self, llm_tokens: &LLMTokenBV) {
        crate::debug!(2, "Stepping grammar constraint state with tokenizer states {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens);

        // Clear the current state before populating it with results from special_map
        self.state = BTreeMap::new();

        Trie::special_map(
            initial_nodes_and_values,
            |mut glr_parse_state, grammar_token_id, edge_llm_tokens, child_node| {
                let node_ptr = std::ptr::addr_of!(*child_node);
                crate::debug!(3, "Processing grammar node {:p} token {:?} with {} active states", node_ptr, grammar_token_id.map(|gtid| gtid.0), glr_parse_state.head.predecessors.len()); // Use head.predecessors.len()

                // Filter the active states based on the edge_llm_tokens
                let mut new_head_links = BTreeSet::new();
                 // Iterate over predecessors of the head node
                let current_links: Vec<_> = glr_parse_state.head.predecessors.iter().cloned().collect();
                for link in current_links {
                    let mut filtered_edge_value = link.edge_value.clone();
                    filtered_edge_value.active &= &edge_llm_tokens; // Filter active tokens on the edge

                    if !filtered_edge_value.active.is_empty() {
                         // The intersection value is based on the edge value *leading to the node*
                         // where the grammar token action was taken. This is the link.edge_value.intersection.
                         // This part might need refinement depending on exact semantic interpretation.
                         // For now, let's update the intersection based on the *new* active set.
                         // Filtered edge value already has active tokens.
                         // Let's update intersection based on the original intersection and the new active set.
                         filtered_edge_value.intersection &= &link.edge_value.active; // Intersect original active with new active
                         filtered_edge_value.intersection &= &edge_llm_tokens; // Intersect with edge tokens


                        new_head_links.insert(crate::datastructures::gss::PredecessorLink { // Use full path
                            node: link.node.clone(), // Keep the same GSSNode pointer
                            edge_value: filtered_edge_value, // Use the filtered edge value
                        });
                    }
                }
                // Update the GLR state's head node based on filtered links
                glr_parse_state.head = Arc::new(GSSNode::new_with_predecessors(
                    crate::glr::parser::DUMMY_HEAD_STATE_ID, // Use the defined constant
                    new_head_links.into_iter().collect()
                ));


                if glr_parse_state.is_ok() {
                     // If there are still active states after filtering, perform the GLR step
                     grammar_token_id.map(|gtid| glr_parse_state.step(gtid));

                     if glr_parse_state.is_ok() {
                         crate::debug!(3, "Processed grammar token {:?}, {} active states.", grammar_token_id.map(|gtid| gtid.0), glr_parse_state.head.predecessors.len()); // Use head.predecessors.len()
                         Some(glr_parse_state)
                     } else {
                         crate::debug!(3, "No active states after processing grammar token {:?}", grammar_token_id.map(|gtid| gtid.0));
                         None // No active states after GLR step, prune this path
                     }
                } else {
                    crate::debug!(3, "No active states after filtering by edge LLM tokens {:?}", edge_llm_tokens);
                    None // No active states after filtering, prune this path
                }
            },
            |managed_parse_state1, managed_parse_state2| {
                // This is where GLRParserState::merge_with is called
                managed_parse_state1.merge_with(managed_parse_state2);
            },
            |node, current_glr_parse_state| {
                // This is the finalize callback. `node` is a PrecomputeNode (GSSNode<LLMTokenBV>),
                // `current_glr_parse_state` is the merged GLR state arriving at this node.

                // Aggregate active LLM tokens from the edges leading to the current active GLR states
                let mut active_llm_tokens = HybridBitset::new();
                for link in &current_glr_parse_state.head.predecessors {
                     active_llm_tokens |= &link.edge_value.active;
                }

                let node_ptr = std::ptr::addr_of!(*node);
                crate::debug!(3, "Processing precompute node {:p} in finalize step with {} active GLR states, {} LLM tokens, {} finalizers",
                              node_ptr,
                              current_glr_parse_state.head.predecessors.len(), // Use head.predecessors.len()
                              active_llm_tokens.len(),
                              node.value.finalizers().len()); // Use .finalizers()

                let mut keep_glr_state = false; // Flag to determine if current_glr_parse_state should be propagated

                // 1. Handle Clean End
                if let Some(clean_end_tokens) = &node.value.clean_end {
                    // Filter the GLR state paths that are compatible with clean_end_tokens
                    let mut final_glr_parse_state = current_glr_parse_state.clone();
                    let mut new_head_links = BTreeSet::new();
                    let current_links: Vec<_> = final_glr_parse_state.head.predecessors.iter().cloned().collect(); // Clone links

                    for link in current_links {
                        let mut filtered_edge_value = link.edge_value.clone();
                        filtered_edge_value.active &= clean_end_tokens; // Filter active tokens on the edge

                         if !filtered_edge_value.active.is_empty() {
                             // Update intersection based on original active and clean_end
                             filtered_edge_value.intersection &= &link.edge_value.active;
                             filtered_edge_value.intersection &= clean_end_tokens;

                             new_head_links.insert(crate::datastructures::gss::PredecessorLink { // Use full path
                                node: link.node.clone(), // Keep the same GSSNode pointer
                                edge_value: filtered_edge_value, // Use the filtered edge value
                             });
                        }
                    }
                    final_glr_parse_state.head = Arc::new(GSSNode::new_with_predecessors(
                         crate::glr::parser::DUMMY_HEAD_STATE_ID, // Use the defined constant
                         new_head_links.into_iter().collect()
                    ));


                    if final_glr_parse_state.is_ok() {
                        crate::debug!(3, "At clean end state, GLR parse state is OK. Propagating to TokenizerStateID(0)");
                         // Merge the final_glr_parse_state into self.state entry for TokenizerStateID(0)
                        let tokenizer_state_id_zero = TokenizerStateID(0);
                        if let Some(existing) = self.state.get_mut(&tokenizer_state_id_zero) {
                            existing.merge_with(final_glr_parse_state);
                        } else {
                            self.state.insert(tokenizer_state_id_zero, final_glr_parse_state);
                        }
                         // The current_glr_parse_state might also be needed for finalizers, so don't return false yet.
                         keep_glr_state = true;
                    }
                }

                // 2. Handle Finalizers
                for (possible_final_grammar_token, precomputed_finalizer) in node.value.finalizers().iter() {
                    // For each finalizer, we need to apply the grammar token step and then filter by tokenizer state and LLM tokens
                    let mut glr_state_after_final_grammar_token_step = current_glr_parse_state.clone();
                    crate::debug!(3, "Stepping GLR parse state with finalizer grammar token {:?}", possible_final_grammar_token.0);
                    glr_state_after_final_grammar_token_step.step(*possible_final_grammar_token);

                    if glr_state_after_final_grammar_token_step.is_ok() {
                        crate::debug!(3, "GLR state after final grammar token step is OK.");
                        for (tokenizer_state_id, llm_tokens_from_finalizer) in &precomputed_finalizer.content {
                            // Filter the GLR state paths after the grammar step based on finalizer's LLM tokens
                            let mut glr_parse_state_filtered_by_finalizer = glr_state_after_final_grammar_token_step.clone(); // Start from the state *after* the grammar step

                             let mut new_head_links = BTreeSet::new();
                             // Iterate over predecessors of the head node
                            let current_links: Vec<_> = glr_parse_state_filtered_by_finalizer.head.predecessors.iter().cloned().collect(); // Clone links

                            for link in current_links {
                                let mut filtered_edge_value = link.edge_value.clone();
                                filtered_edge_value.active &= llm_tokens_from_finalizer; // Filter active tokens on the edge

                                 if !filtered_edge_value.active.is_empty() {
                                     // Update intersection based on original active and finalizer tokens
                                     filtered_edge_value.intersection &= &link.edge_value.active;
                                     filtered_edge_value.intersection &= llm_tokens_from_finalizer;

                                     new_head_links.insert(crate::datastructures::gss::PredecessorLink { // Use full path
                                        node: link.node.clone(), // Keep the same GSSNode pointer
                                        edge_value: filtered_edge_value, // Use the filtered edge value
                                     });
                                }
                            }
                             glr_parse_state_filtered_by_finalizer.head = Arc::new(GSSNode::new_with_predecessors(
                                crate::glr::parser::DUMMY_HEAD_STATE_ID, // Use the defined constant
                                new_head_links.into_iter().collect()
                            ));


                            if glr_parse_state_filtered_by_finalizer.is_ok() {
                                crate::debug!(3, "Finalizer is compatible with GLR state (after step). Propagating to TokenizerStateID {:?}", tokenizer_state_id);
                                // Merge the filtered state into self.state entry for this tokenizer_state_id
                                if let Some(existing) = self.state.get_mut(tokenizer_state_id) {
                                    existing.merge_with(glr_parse_state_filtered_by_finalizer);
                                } else {
                                    self.state.insert(*tokenizer_state_id, glr_parse_state_filtered_by_finalizer);
                                }
                                keep_glr_state = true;
                            }
                        }
                    }
                }

                // Keep the current_glr_parse_state propagating in the Trie::special_map
                // if it's still active and was used in either clean end or finalizer processing.
                // The special_map finalize function expects a boolean indicating if the *input* value
                // (current_glr_parse_state) should continue.
                // Given the changes, the current_glr_parse_state is now a merged state representing
                // paths arriving at the PrecomputeNode. Whether it continues depends on whether
                // any of these paths successfully transitioned either via clean_end or a finalizer
                // into a new state managed by self.state.
                // So, return true if any resulting state was merged into self.state.
                keep_glr_state
            },
        );
    }
}

