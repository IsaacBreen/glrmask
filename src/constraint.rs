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
use crate::datastructures::gss::{prune_and_transform_recursive, prune_and_transform_roots, BulkMerge, GSSTrait, GSSNode, ParseStateNodeContent}; // Import needed GSS items
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::trie::{EdgeInserter, Trie};
use crate::datastructures::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::datastructures::ArcPtrWrapper;
use crate::finite_automata::Regex;
use crate::glr::parser::{
    MergeAndIntersect, GLRParser, GLRParserState, // ParseState is removed
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
            active:       Default::default(), // Should maybe be all ones by default? Or empty? Depends on usage.
            intersection: Default::default(), // Should maybe be all ones by default?
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
     fn default() -> Self {
         // Default should probably represent a state where all tokens are initially possible.
         // This depends on where Default is used (e.g., initial state label).
         // If used for the initial GSS label, it should allow all tokens.
         // If used as a placeholder, empty might be better.
         // Let's keep it as empty for now, and rely on explicit initialization (like in GrammarConstraint::init).
         Self {
             active: HybridBitset::new(),
             intersection: HybridBitset::new(),
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

        let internal_max_llm_token = original_to_internal_id_bimap.iter().map(|(_, id)| *id).max().unwrap_or(0); // Handle empty map case

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
        // The initial LLMTokenInfo should have all tokens active and intersecting
        let initial_t_value = LLMTokenInfo {
            active: HybridBitset::ones(self.internal_max_llm_token),
            intersection: HybridBitset::ones(self.internal_max_llm_token),
        };

        // Initialize the parser state for the initial tokenizer state
        let initial_glr_state = self.parser.init_glr_parser_with_t(initial_t_value);

        let mut state = BTreeMap::new();
        // The initial state corresponds to the tokenizer being in its initial state
        state.insert(
            self.tokenizer.initial_state_id(),
            initial_glr_state,
        );

        GrammarConstraintState { parent: self, state }
    }

    #[inline]
    fn original_id_to_internal(&self, original_id: LLMTokenID) -> Option<LLMTokenID> {
        self.original_to_internal_id_bimap.get_by_left(&(original_id.0)).map(|internal_val| LLMTokenID(*internal_val))
    }

    #[inline]
    fn internal_id_to_original(&self, internal_id: LLMTokenID) -> Option<LLMTokenID> {
        self.original_to_internal_id_bimap.get_by_right(&(internal_id.0)).map(|original_val| LLMTokenID(*original_val))
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
        // Precompute nodes are not GSS nodes, they are nodes in the Trie representing the vocab.
        // They don't have `value` field anymore.
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
            usize, // Offset within the segment_bytes
            BTreeMap<TokenizerStateID, BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>>, // Map from tokenizer state to a set of PrecomputeNodes
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

                let src_set = self.merge_handles(&src_set); // Merge PrecomputeNodes at this offset and tokenizer state
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
                    // The LLM tokens reachable from child_vocab_of_segment at this point (end of segment)
                    // are those that are suffixes of the vocabulary prefix ending with child_vocab_of_segment.
                    // The edge_tokens should be the set of LLM tokens that *could* follow the prefix
                    // from the root of the vocab tree down to child_vocab_of_segment.
                    // The VocabPrefixTreeNode stores `reachable_token_ids` which should be this set.
                    let edge_tokens = child_vocab_of_segment.reachable_token_ids().clone();

                    for src_wrap in &src_set {
                         let src_arc = src_wrap.as_arc().clone();
                         self.insert_edge(
                            src_arc, // Source PrecomputeNode (Arc<Mutex<PrecomputeNode>>)
                            Some(grammar_tok), // Edge key (GrammarTokenID)
                            edge_tokens.clone(), // Edge value (LLMTokenBV) - LLM tokens reachable *through* this edge
                            child_vocab_of_segment.token_id(), // LLM token ID if child_vocab_of_segment is a terminal node
                            match_end_offset, // Offset in segment where match ended
                            segment_bytes.len(), // Total segment length
                            &mut queue, // Queue for processing matches within the segment
                            &mut next_level // Map for results that reach the end of the segment
                        );
                    }
                }

                // -------------------------------------------------------------
                // Final tokenizer state after reading entire suffix
                // -------------------------------------------------------------
                if let Some(final_state_val) = exec_result.end_state {
                    let final_sid = TokenizerStateID(final_state_val);

                    // If the tokenizer reaches a state after consuming the full segment,
                    // this means the segment is a prefix of some LLM token(s).
                    // The PrecomputeNode corresponding to the end of this segment should
                    // be associated with the `final_sid` for the next level of DFS.
                    // The edge leading *to* the node representing the end of the segment
                    // doesn't have a GrammarTokenID key, it's the "implicit" edge of the trie structure.

                    // Find or create the PrecomputeNode corresponding to the end of the segment.
                    // This is the `child_vocab_of_segment` itself.
                    // We need to associate this `child_vocab_of_segment`'s corresponding PrecomputeNode handle(s)
                    // with the `final_sid` in `next_level`.
                    // The `src_set` contains handles to PrecomputeNodes at the *start* of the segment.
                    // We need to find the nodes at the *end* of the segment, reached via the trie path defined by `segment_bytes`.

                    // This requires traversing the PrecomputeNode trie in parallel with the VocabPrefixTree.
                    // The `insert_edge` helper is designed for edges with GrammarTokenID keys.
                    // The connection between `src_set` (nodes before segment) and the nodes after segment (corresponding to `child_vocab_of_segment`)
                    // is not via a GrammarTokenID edge in the PrecomputeNode trie, but via the trie structure itself.

                    // Let's reconsider the structure of `Precomputer::roots` and the DFS association.
                    // `roots`: Map from TokenizerStateID to `Arc<Mutex<PrecomputeNode>>`.
                    // `assoc`: Map from TokenizerStateID to `BTreeSet<ArcPtrWrapper<Mutex<PrecomputeNode>>>`.
                    // This `assoc` maps a tokenizer state to a set of PrecomputeNodes that can be reached when the tokenizer is in that state *and* we have consumed the vocabulary prefix leading to that PrecomputeNode.

                    // When processing a segment `S` from `vocab_node` to `child_vocab_of_segment`, starting with tokenizer state `state_before` and PrecomputeNodes `src_set` (associated with `vocab_node` and `state_before`):
                    // Tokenizer consumes `S`, ends in `final_sid`.
                    // The nodes corresponding to the end of the segment (`child_vocab_of_segment`) should now be associated with `final_sid`.
                    // These nodes are reached from `src_set` by following the path corresponding to `S` *within the PrecomputeNode trie structure*.

                    // This means we need to traverse the PrecomputeNode trie using `segment_bytes` as keys, starting from `src_set`.
                    // The `Trie::get_or_create_node` method seems relevant, but it takes a `key` which is `Option<GrammarTokenID>`.
                    // The path in the PrecomputeNode trie from `vocab_node` to `child_vocab_of_segment` is not keyed by GrammarTokenID, but by the bytes of the vocabulary segment.

                    // This implies a mismatch in the trie key types. The PrecomputeNode trie is keyed by `Option<GrammarTokenID>`, but the DFS logic uses `VocabPrefixTree` edges which are byte segments.

                    // Let's look at the `PrecomputeNode` definition: `Trie<Option<GrammarTokenID>, LLMTokenBV, PrecomputedNodeContents>`. The keys are `Option<GrammarTokenID>`.
                    // The `VocabPrefixTree` is `Trie<char, Vec<u8>, usize>`. The edges are characters, node values are token IDs. This was changed from bytes -> token id in the original code. Let's assume the `VocabPrefixTree` is keyed by bytes for now based on `iter_children()`.

                    // If the PrecomputeNode trie is keyed by `Option<GrammarTokenID>`, how do byte segments from the tokenizer relate?
                    // A byte segment from the tokenizer matches a GrammarTokenID if it's a full token match.
                    // So, when `execute_from_state` finds a match `m` for `segment_bytes[offset..offset+m.width]`, this byte sequence corresponds to `GrammarTokenID(m.id)`.
                    // The edge in the PrecomputeNode trie should use this `GrammarTokenID`.

                    // The `insert_edge` helper already does this: it inserts an edge with key `Some(grammar_tok)`.
                    // This covers the "matches inside suffix" part.

                    // What about the "end state" part?
                    // If the tokenizer consumes the full `segment_bytes` and ends in `final_sid`,
                    // we are at the PrecomputeNode corresponding to `child_vocab_of_segment`.
                    // The nodes in `src_set` were the PrecomputeNodes corresponding to `vocab_node`.
                    // The association in `next_level` should map `final_sid` to the PrecomputeNodes for `child_vocab_of_segment`.
                    // How do we find these nodes? They are children of `src_set` nodes via the path corresponding to `segment_bytes`.

                    // The `Trie::special_map` logic might provide a clue. It seems to traverse the trie structure itself.
                    // `Precomputer::dfs` traverses the VocabPrefixTree (`vocab_node`, `child_vocab_of_segment`).
                    // It also manages an association (`assoc_by_state`) between TokenizerStates and *sets of PrecomputeNodes*.

                    // Let's assume the `PrecomputeNode` trie structure *mirrors* the `VocabPrefixTree` structure in terms of byte paths, even though its explicit keys are `Option<GrammarTokenID>`.
                    // This means if `vocab_node` has a child `child_vocab` via byte sequence `B`, the corresponding `PrecomputeNode` for `vocab_node` has a child PrecomputeNode via some path corresponding to `B`.
                    // This interpretation seems incorrect given the `Option<GrammarTokenID>` keys.

                    // Alternative interpretation:
                    // The `PrecomputeNode` trie nodes correspond *directly* to `VocabPrefixTree` nodes.
                    // The map `roots` maps TokenizerStateID to the root of the PrecomputeNode trie.
                    // The association `assoc_by_state` maps a TokenizerStateID to a *set of nodes in the PrecomputeNode trie*. A node in this set corresponds to a specific `VocabPrefixTree` node, reachable from the root of the PrecomputeNode trie by following the byte path from the root of the VocabPrefixTree to the corresponding VocabPrefixTree node.

                    // Let's trace the `process_segment` again with this interpretation.
                    // `sources_per_state`: map from tokenizer state `S_start` to `Set<P_start>` where `P_start` is a PrecomputeNode corresponding to `vocab_node`.
                    // Segment `segment_bytes` from `vocab_node` to `child_vocab_of_segment`.
                    // Tokenizer executes `segment_bytes` from `S_start`, ends in `S_end`.
                    // The nodes corresponding to `child_vocab_of_segment` are reached from `P_start` by following the path for `segment_bytes` in the PrecomputeNode trie.
                    // The `insert_edge` handles edges keyed by `Some(GrammarTokenID)`. These occur for *matches* within the segment.
                    // The PrecomputeNode trie also has edges keyed by `None` for the structural path corresponding to the vocabulary prefix itself.

                    // This indicates the `PrecomputeNode` trie should be keyed by `Option<GrammarTokenID>` OR `Bytes`.
                    // But it's defined as `Trie<Option<GrammarTokenID>, ...>`.

                    // Let's re-examine `Trie::special_map`. Its key type is `K`. In `constraint.rs`, `K` is `Option<GrammarTokenID>`.
                    // It is called with `initial_nodes_and_values: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'a, LLMTokenInfo>)>`.
                    // This vector contains pairs of (PrecomputeNode root, GLR state).
                    // The traversal `Trie::special_map` uses the keys `Option<GrammarTokenID>`.

                    // This implies the PrecomputeNode trie structure is:
                    // Root (corresponding to empty prefix)
                    //   --Some(GT1)--> Node1
                    //   --Some(GT2)--> Node2
                    //   --None--> Node3 (This edge type doesn't seem to be used or generated).

                    // Let's look at how `Precomputer` builds the `PrecomputeNode` trie.
                    // `insert_edge` is the only place where children are added (`EdgeInserter`).
                    // It adds children with key `Some(grammar_tok)`.
                    // It also seems to create a destination node `target` with value `PrecomputedNodeContents::default()`.
                    // And then associates this `target` with `TokenizerStateID(0)` in `next_level` if it's a "clean end".

                    // The `clean_end` logic in `insert_edge` seems to link the PrecomputeNode `target` (corresponding to the end of the segment)
                    // to `TokenizerStateID(0)` in `next_level` *regardless* of the actual tokenizer state reached (`exec_result.end_state`).
                    // This looks like a bug or a misunderstanding of the precomputation logic.

                    // Let's revisit the end state propagation.
                    // When `execute_from_state(segment_bytes, state_before)` results in `end_state = final_sid`,
                    // the PrecomputeNodes corresponding to `child_vocab_of_segment` should be associated with `final_sid`.
                    // These nodes are reached from `src_set` by following the trie path for `segment_bytes`.
                    // Since the PrecomputeNode trie has `Option<GrammarTokenID>` keys, this implies that the structure should be:
                    // A PrecomputeNode for a vocab prefix P has children for `Some(GT)` if `GT` is a token that can immediately follow P.
                    // It also needs a way to represent extending the prefix P with bytes that don't form a complete token yet. This would be an edge keyed by the byte or a sequence of bytes, or perhaps a `None` key followed by byte-keyed edges.

                    // Given the current `PrecomputeNode` definition `Trie<Option<GrammarTokenID>, ...>`, the intended structure must be:
                    // Node for prefix P has a child via `Some(GT)` if GT can extend P.
                    // It must also have an edge (likely `None` key?) leading to a subtrie that represents extensions of P that are prefixes of other tokens.

                    // The `insert_edge` function seems to model the `Some(GrammarTokenID)` edges corresponding to token matches.
                    // The `process_segment` function is trying to model the structural path through the vocabulary trie using `segment_bytes`.

                    // There seems to be a conceptual mismatch between the VocabPrefixTree traversal (byte segments) and the PrecomputeNode trie structure (`Option<GrammarTokenID>` keys).

                    // Let's assume, for the sake of applying the GSS changes, that the precomputation logic *as written* is intended, even if its structure is unusual.
                    // `insert_edge` adds edges keyed by `Some(grammar_tok)`.
                    // The end-of-segment logic (`exec_result.end_state`) associates the `child_vocab_of_segment` with `final_sid` by adding `src.clone()` to `next_level.entry(final_sid).or_default()`.
                    // This means the PrecomputeNodes at the start of the segment (`src_set`) are carried forward and associated with the new tokenizer state (`final_sid`) at the *end* of the segment.
                    // This seems to link the tokenizer state transition across a segment to the PrecomputeNodes at the *start* of the segment, not the end.

                    // Let's assume the association `next_level` means: if the tokenizer is in state `S` after consuming the segment, we are at the PrecomputeNodes associated with `next_level[S]`.

                    // Back to applying the GSS changes:

                    // `prepare_initial_nodes_and_values_for_special_map`
                    // Input: `llm_tokens: &LLMTokenBV` (internal IDs).
                    // Output: `Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'a, LLMTokenInfo>)>`.
                    // This seems correct. It gets the initial PrecomputeNode roots (one per tokenizer state) and the corresponding initial GLR states.
                    // The filtering `Arc::make_mut(&mut parse_state.stack).value.t.active &= llm_tokens;` was filtering the T value on the *node*.
                    // Now T is on the *edge label*. The GLRParserState has a `head: ParseStack<T>`.
                    // The active states are represented by the edges leading *into* the head.
                    // Filtering means keeping only those edges where the label's `t.active` intersects with `llm_tokens`.
                    // This requires traversing the edges of `glr_state.head`, creating a new node with the filtered edges, and setting the new head.

                    let mut initial_nodes_and_values: Vec<(Arc<Mutex<PrecomputeNode>>, GLRParserState<'_, LLMTokenInfo>)> = Vec::new();

                    for (tokenizer_state_id, state) in &self.state {
                        let mut cloned_glr_state = state.clone();

                        // Filter the edges leading into the head node based on llm_tokens
                        let mut filtered_edges: BTreeSet<_> = BTreeSet::new();
                        let mut head_kept = false; // Did we keep any edges leading to the head?

                        for edge in &cloned_glr_state.head.predecessors {
                            let mut new_label = edge.label.clone();
                            new_label.t.active &= llm_tokens; // Filter the active set

                            if !new_label.t.active.is_empty() {
                                // If the edge is still active after filtering, keep it with the new label
                                filtered_edges.insert(crate::datastructures::gss::GSSEdge { // Use explicit path for GSSEdge
                                     pred: edge.pred.clone(), // Keep the predecessor
                                     label: new_label, // Use the new filtered label
                                });
                                head_kept = true;
                            }
                        }

                        // Create a new head node with the filtered edges.
                        // If no edges were kept, the new head will have an empty predecessor set.
                        cloned_glr_state.head = Arc::new(GSSNode { predecessors: filtered_edges });

                        // Only include this GLR state if the head node is still valid (has incoming edges)
                        if cloned_glr_state.is_ok() {
                             let token_trie_node = self.parent.precomputed[&tokenizer_state_id].clone();
                             let token_trie_arc_mutex = Arc::new(Mutex::new(token_trie_node));
                             initial_nodes_and_values.push((token_trie_arc_mutex, cloned_glr_state));
                        }
                    }

                    crate::debug!(4, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
                    crate::debug!(4, "Printing initial nodes and values for tokenizer states (after filtering by LLM tokens)");
                    for (t_node_arc, glr_state) in &initial_nodes_and_values {
                         // Note: GLR state log_gss expects a vector of roots, but it's adapted for a single head.
                         glr_state.log_gss(format!("Prepared (stage 1) initial nodes and values for tokenizer state corresponding to trie node {:p}", Arc::as_ptr(t_node_arc)).as_str(), GrammarTokenID(0));
                    }
                    crate::debug!(4, "----------------------------------------------------------------");


                    initial_nodes_and_values // Return the filtered initial states
                }


        // `GrammarConstraintState::step`
        // Calls `prepare_initial_nodes_and_values_for_special_map` (done).
        // Calls `Trie::special_map`.
        // `special_map` arguments:
        // 1. `initial_nodes_and_values`: Handled.
        // 2. Map closure: `|glr_parse_state, grammar_token_id, edge_llm_tokens, child_node|`
        //    - `glr_parse_state`: `&mut GLRParserState<'a, LLMTokenInfo>`.
        //    - `grammar_token_id`: `Option<GrammarTokenID>` (the edge key in PrecomputeNode trie).
        //    - `edge_llm_tokens`: `&LLMTokenBV` (the edge value in PrecomputeNode trie).
        //    - `child_node`: `&Arc<Mutex<PrecomputeNode>>` (the destination node in PrecomputeNode trie).
        //    Inside this closure:
        //    - Filter `glr_parse_state.head` edges by `edge_llm_tokens`. This means iterating `glr_parse_state.head.predecessors`, updating `t.active &= edge_llm_tokens`, creating a new head with filtered edges.
        //    - If `grammar_token_id` is `Some(gtid)`, call `cloned_glr_parse_state.step(gtid)`. This calls the GLR parser's step function, which modifies the GLR state's head.
        //    - Return `Option<GLRParserState>`.

        // 3. Merge closure: `|managed_parse_state1, managed_parse_state2|`
        //    - `managed_parse_state1`: `&mut GLRParserState<'a, LLMTokenInfo>`
        //    - `managed_parse_state2`: `GLRParserState<'a, LLMTokenInfo>`
        //    - Call `managed_parse_state1.merge_with(managed_parse_state2)` (updated).

        // 4. Finalizer closure: `|node, current_glr_parse_state|`
        //    - `node`: `&Arc<Mutex<PrecomputeNode>>` (the PrecomputeNode being finalized).
        //    - `current_glr_parse_state`: `&mut GLRParserState<'a, LLMTokenInfo>` (the aggregated GLR state when reaching this PrecomputeNode).
        //    Inside this closure:
        //    - Check `node.value.clean_end`. If present, filter `current_glr_parse_state.head` edges by `clean_end`. If resulting state is OK, merge it into `self.state.entry(TokenizerStateID(0))`.
        //    - Iterate `node.value.finalizers()`. For each `(final_grammar_token, precomputed_finalizer)`:
        //        - Filter `current_glr_parse_state.head` edges by the finalizer's LLM tokens (from `precomputed_finalizer.content`). This requires iterating `precomputed_finalizer.content` (`BTreeMap<TokenizerStateID, LLMTokenBV>`).
        //        - For each `(tokenizer_state_id, llm_tokens_from_finalizer)` in `precomputed_finalizer.content`:
        //            - Filter `current_glr_parse_state.head` by `llm_tokens_from_finalizer`.
        //            - Call `filtered_glr_state.step(*final_grammar_token)`.
        //            - If the result is OK, merge it into `self.state.entry(*tokenizer_state_id)`.

        pub fn step(&mut self, llm_tokens: &LLMTokenBV)
        where T: Clone + Eq + Hash + Default + MergeAndIntersect // Bounds needed for GSS ops and MergeAndIntersect
        {
            crate::debug!(2, "Stepping grammar constraint state with tokenizer states {:?}", self.state.keys().map(|k| k.0).collect::<Vec<_>>());

            // Prepare initial states: map tokenizer states to (PrecomputeNode root, GLR state)
            let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens);

            // Clear the current state map, results will be merged back into it by the finalizer closure
            self.state.clear();

            Trie::special_map(
                initial_nodes_and_values,
                |glr_parse_state, grammar_token_id, edge_llm_tokens, child_node| -> Option<GLRParserState<'a, LLMTokenInfo>> {
                    // This closure is called for each edge traversal in the PrecomputeNode trie.
                    // `glr_parse_state` is the aggregated GLR state at the start of this edge.
                    // `grammar_token_id` is the key on the edge (Option<GrammarTokenID>).
                    // `edge_llm_tokens` is the value on the edge (LLMTokenBV).
                    // `child_node` is the destination PrecomputeNode.

                    let node_ptr = std::ptr::addr_of!(*child_node);
                    crate::debug!(3, "Processing trie edge with key {:?} to node {:p}. Aggregated GLR head has {} edges.",
                                   grammar_token_id.map(|gtid| gtid.0), node_ptr, glr_parse_state.head.predecessors.len());

                    let mut cloned_glr_parse_state = glr_parse_state.clone();

                    // Filter the edges leading into the GLR head by the edge_llm_tokens
                    let mut filtered_edges: BTreeSet<_> = BTreeSet::new();
                     for edge in &cloned_glr_parse_state.head.predecessors {
                          let mut new_label = edge.label.clone();
                          new_label.t.active &= edge_llm_tokens; // Filter active tokens

                          if !new_label.t.active.is_empty() {
                              filtered_edges.insert(crate::datastructures::gss::GSSEdge {
                                   pred: edge.pred.clone(),
                                   label: new_label,
                              });
                          }
                     }
                     cloned_glr_parse_state.head = Arc::new(GSSNode { predecessors: filtered_edges });


                    if cloned_glr_parse_state.is_ok() {
                        // If the GLR state is still valid after filtering by edge LLM tokens:
                        if let Some(gtid) = grammar_token_id {
                            // If the edge key is a GrammarTokenID, step the GLR parser with it.
                            cloned_glr_parse_state.step(gtid);
                        }
                        // If grammar_token_id is None, this edge represents a structural vocabulary prefix extension,
                        // not a grammar token match. We don't step the GLR parser with a GrammarTokenID.
                        // The GLR state simply propagates along this path.

                        if cloned_glr_parse_state.is_ok() {
                             crate::debug!(3, "Processed trie edge with key {:?}, GLR head now has {} edges.",
                                            grammar_token_id.map(|gtid| gtid.0), cloned_glr_parse_state.head.predecessors.len());
                             Some(cloned_glr_parse_state)
                        } else {
                            crate::debug!(3, "GLR state became empty after processing trie edge with key {:?}.", grammar_token_id.map(|gtid| gtid.0));
                             None // Prune this path if the GLR state becomes empty
                        }
                    } else {
                        crate::debug!(3, "GLR state was empty after filtering by edge LLM tokens {:?}.", edge_llm_tokens);
                        None // Prune this path if the GLR state is empty initially
                    }
                },
                |managed_parse_state1, managed_parse_state2| {
                    // Merge two GLRParserStates
                    managed_parse_state1.merge_with(managed_parse_state2);
                },
                |node, current_glr_parse_state| {
                    // This closure is called when reaching a PrecomputeNode (`node`).
                    // `current_glr_parse_state` is the merged GLR state at this node.

                    let node_ptr = std::ptr::addr_of!(*node);
                    crate::debug!(3, "Finalizing at precompute node {:p}. Current GLR head has {} edges.",
                                   node_ptr, current_glr_parse_state.head.predecessors.len());

                    if current_glr_parse_state.is_ok() { // Only process if the state is valid
                        // Handle clean end
                        if let Some(clean_end_llm_tokens) = &node.value.clean_end {
                            let mut final_glr_parse_state = current_glr_parse_state.clone();
                            // Filter GLR head edges by clean_end tokens
                            let mut filtered_edges: BTreeSet<_> = BTreeSet::new();
                            for edge in &final_glr_parse_state.head.predecessors {
                                let mut new_label = edge.label.clone();
                                new_label.t.active &= clean_end_llm_tokens;
                                if !new_label.t.active.is_empty() {
                                     filtered_edges.insert(crate::datastructures::gss::GSSEdge {
                                         pred: edge.pred.clone(),
                                         label: new_label,
                                     });
                                }
                            }
                            final_glr_parse_state.head = Arc::new(GSSNode { predecessors: filtered_edges });

                            if final_glr_parse_state.is_ok() {
                                crate::debug!(3, "Reached clean end. Merging GLR state into TokenizerStateID(0).");
                                // Merge the resulting state into the main state map for TokenizerStateID(0)
                                self.state.entry(TokenizerStateID(0))
                                    .and_modify(|existing| existing.merge_with(final_glr_parse_state.clone()))
                                    .or_insert(final_glr_parse_state);
                            }
                        }

                        // Handle finalizers (matches ending at this node)
                        for (possible_final_grammar_token, precomputed_finalizer) in node.value.finalizers().iter() {
                            crate::debug!(3, "Processing finalizer for grammar token {:?}.", possible_final_grammar_token.0);

                            // The finalizer has potential next tokenizer states and associated LLM tokens
                            for (tokenizer_state_id, llm_tokens_from_finalizer) in &precomputed_finalizer.content {
                                crate::debug!(3, "  Finalizer leads to TokenizerStateID {:?} with LLM tokens {:?}.",
                                               tokenizer_state_id.0, llm_tokens_from_finalizer);

                                let mut glr_parse_state_filtered = current_glr_parse_state.clone();
                                // Filter GLR head edges by the finalizer's LLM tokens
                                let mut filtered_edges: BTreeSet<_> = BTreeSet::new();
                                for edge in &glr_parse_state_filtered.head.predecessors {
                                    let mut new_label = edge.label.clone();
                                    new_label.t.active &= llm_tokens_from_finalizer;
                                     // The intersection is done during the GLR reduce step based on T value of node len steps back.
                                     // The finalizer LLM tokens filter which *paths* are valid for this finalizer.
                                     // Let's update the intersection here too, as this path represents a completed token sequence.
                                     // new_label.t.intersection &= llm_tokens_from_finalizer; // No, intersection is for prefixes

                                    if !new_label.t.active.is_empty() {
                                         filtered_edges.insert(crate::datastructures::gss::GSSEdge {
                                            pred: edge.pred.clone(),
                                            label: new_label,
                                         });
                                    }
                                }
                                glr_parse_state_filtered.head = Arc::new(GSSNode { predecessors: filtered_edges });


                                if glr_parse_state_filtered.is_ok() {
                                     // If the state is still valid after filtering by finalizer tokens,
                                     // step the GLR parser with the final grammar token.
                                     glr_parse_state_filtered.step(*possible_final_grammar_token);

                                     if glr_parse_state_filtered.is_ok() {
                                         crate::debug!(3, "  Finalizer step successful. Merging GLR state into TokenizerStateID {:?}.", tokenizer_state_id.0);
                                         // Merge the resulting state into the main state map for the finalizer's tokenizer state
                                         self.state.entry(*tokenizer_state_id)
                                             .and_modify(|existing| existing.merge_with(glr_parse_state_filtered.clone()))
                                             .or_insert(glr_parse_state_filtered);
                                     } else {
                                         crate::debug!(3, "  GLR state became empty after finalizer step.");
                                     }
                                } else {
                                    crate::debug!(3, "  GLR state was empty after filtering by finalizer LLM tokens.");
                                }
                            }
                        }
                    } else {
                        crate::debug!(3, "Aggregated GLR state at node {:p} is empty. Pruning.", node_ptr);
                    }

                    // Return true if the current GLR state at this trie node is OK, to keep traversing children from this node.
                    // This controls whether special_map continues down this path in the PrecomputeNode trie.
                    // It should continue as long as the aggregated GLR state is not empty.
                    current_glr_parse_state.is_ok()
                },
            );
        }
    }
