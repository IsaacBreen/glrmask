//! Immutable compiled artifact shape.
//!
//! `Constraint` is the public runtime object.  Mathematically, its serialized
//! fields are the compiled artifact: Parser DWA, GLR table, lexer/tokenizer,
//! CanMatch relation, token-space quotients, and token byte tables.  Its
//! `#[serde(skip)]` fields are derived runtime caches rebuilt by
//! `artifact::caches`.
//!
//! A future internal refactor may rename this concrete storage object to
//! `CompiledArtifact` and make public `Constraint` a lightweight `Arc` owner.
//! This chunk keeps the public type stable while isolating the construction
//! contract in `CompiledArtifactParts`.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::grammar::flat::TerminalID;
use crate::parser::glr::table::GLRTable;
use crate::runtime::token_space::final_mask_mapping::FinalMaskMapping;
use rustc_hash::FxHashMap;

use super::cache_types::{
    DenseWeightBufMaskCache,
    DenseWeightMaskCache,
    FastDwaTransitions,
    FastTokenizerTransitions,
    InternalTokenBufMasks,
    RuntimeCaches,
    SeedStateBufMasks,
    SeedStateDenseMasks,
    SeedTerminalDenseMasks,
    SparseWeightBufMaskCache,
};
use super::dense::{empty_dense_words, DenseWords};
use super::templates::TemplateDfasByTerminal;
use super::token_space::CanMatchByTerminal;

/// Compile-phase output needed to assemble a runtime constraint.
///
/// This separates the semantic artifact from the derived caches.  The compile
/// pipeline should construct this value and then call
/// [`Constraint::from_compiled_parts`], rather than knowing every cache field
/// that happens to live in `Constraint`.
pub(crate) struct CompiledArtifactParts {
    pub(crate) parser_dwa: DWA,
    pub(crate) table: GLRTable,
    pub(crate) terminal_display_names: Vec<String>,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) ignore_terminal: Option<TerminalID>,
    pub(crate) can_match: CanMatchByTerminal,
    pub(crate) state_to_internal_tsid: Vec<u32>,
    pub(crate) internal_tsid_to_states: Vec<Vec<u32>>,
    pub(crate) template_dfas_by_terminal: TemplateDfasByTerminal,
    pub(crate) original_token_to_internal: Vec<u32>,
    pub(crate) internal_token_to_tokens: Vec<Vec<u32>>,
    pub(crate) eos_token_id: Option<u32>,
    pub(crate) token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    pub(crate) internal_token_bytes: BTreeMap<u32, Vec<u8>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Constraint {
    pub(crate) parser_dwa: DWA,
    pub(crate) table: GLRTable,
    #[serde(default)]
    pub(crate) terminal_display_names: Vec<String>,
    pub(crate) tokenizer: Tokenizer,
    #[serde(default)]
    pub(crate) ignore_terminal: Option<TerminalID>,

    /// can_match keyed by grammar terminal id.
    ///
    /// Each Weight maps final shared internal tokenizer-state ids to token sets
    /// in the final shared constraint-internal vocab space. Parser-DWA weights
    /// and can_match weights are reconciled into this same space during
    /// compilation.
    pub(crate) can_match: CanMatchByTerminal,
    pub(crate) state_to_internal_tsid: Vec<u32>,
    pub(crate) internal_tsid_to_states: Vec<Vec<u32>>,
    #[serde(skip)]
    pub(crate) template_dfas_by_terminal: TemplateDfasByTerminal,
    /// Original token -> final shared constraint-internal token id.
    ///
    /// This is not necessarily equal to the parser-DWA compaction vocab map
    /// produced before can-match reconciliation. It may contain additional
    /// splits required by can_match.
    #[serde(default)]
    pub(crate) original_token_to_internal: Vec<u32>,
    /// Final shared constraint-internal token id -> original token ids.
    ///
    /// Parser-DWA weights and Constraint.can_match bitmaps both use these
    /// final internal token ids.
    #[serde(default)]
    pub(crate) internal_token_to_tokens: Vec<Vec<u32>>,
    pub(crate) eos_token_id: Option<u32>,
    #[serde(default)]
    pub(crate) json_u_prefix_token_id: Option<u32>,
    #[serde(skip)]
    pub(crate) json_escape_prefix_buf_mask: Box<[u32]>,
    pub(crate) token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    #[serde(default)]
    pub(crate) internal_token_bytes: BTreeMap<u32, Vec<u8>>,
    #[serde(skip)]
    pub(crate) token_bytes_dense: Vec<Option<Box<[u8]>>>,

    /// Precomputed bitmask fragments for each internal token.
    /// `internal_token_buf_masks[i]` contains (word_index, or_mask) pairs
    /// for all original tokens that map to internal token `i`.
    #[serde(skip)]
    pub(crate) internal_token_buf_masks: Vec<InternalTokenBufMasks>,
    /// Precomputed combined buf output for each group of 64 internal tokens.
    /// `word_group_buf_masks[w]` is the combined mask for internal tokens [w*64 .. (w+1)*64).
    /// Used as a fast path in `or_to_buf` when a dense word is all-ones (!0u64).
    #[serde(skip)]
    pub(crate) word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 128 internal tokens.
    #[serde(skip)]
    pub(crate) pair_word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 256 internal tokens.
    #[serde(skip)]
    pub(crate) quad_word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 512 internal tokens.
    #[serde(skip)]
    pub(crate) super_word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 1024 internal tokens.
    #[serde(skip)]
    pub(crate) mega_word_group_buf_masks: Vec<Box<[u32]>>,
    /// Precomputed dense output masks for groups of 2048 internal tokens.
    #[serde(skip)]
    pub(crate) giga_word_group_buf_masks: Vec<Box<[u32]>>,
    /// Sparse OR-union for each 64-token internal word group.
    #[serde(skip)]
    pub(crate) word_group_sparse_masks: Vec<InternalTokenBufMasks>,
    /// Dense prefix-unions of 64-token internal word groups.
    ///
    /// `word_group_prefix_buf_masks[i]` is the OR-union of word groups
    /// `[0, i)`. Internal-token groups are disjoint in original-token space,
    /// so `prefix[end] & !prefix[start]` is the exact dense mask for a full
    /// internal-word run `[start, end)`.
    #[serde(skip)]
    pub(crate) word_group_prefix_buf_masks: Vec<Box<[u32]>>,
    /// Prefix sums of `word_group_sparse_masks[i].len()`.
    #[serde(skip)]
    pub(crate) word_group_sparse_prefix_entries: Vec<usize>,
    #[serde(skip)]
    pub(crate) quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    #[serde(skip)]
    pub(crate) byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    pub(crate) word_group_sparse_total_entries: usize,
    #[serde(skip)]
    pub(crate) word_group_sparse_max_entries: usize,
    /// Precomputed buf output for the full internal token universe (OR of all word_group_buf_masks).
    #[serde(skip)]
    pub(crate) all_tokens_buf_mask: Box<[u32]>,
    #[serde(skip)]
    pub(crate) internal_token_dense_words: usize,
    #[serde(skip)]
    pub(crate) weight_token_dense_masks: DenseWeightMaskCache,
    #[serde(skip)]
    pub(crate) weight_token_buf_masks: DenseWeightBufMaskCache,
    #[serde(skip)]
    pub(crate) weight_token_sparse_buf_masks: SparseWeightBufMaskCache,
    /// Precomputed dense bitmask for the seed phase: for each (tokenizer_state, terminal_id),
    /// the dense bitmap of internal tokens that terminal covers in that state.
    #[serde(skip)]
    pub(crate) seed_terminal_dense: SeedTerminalDenseMasks,
    /// Precomputed dense seed baseline for each ORIGINAL tokenizer state.
    ///
    /// seed_state_dense[s] is the dense bitmap of final shared internal token ids
    /// whose original token bytes are lexically live from original tokenizer state s.
    #[serde(skip)]
    pub(crate) seed_state_dense: SeedStateDenseMasks,
    /// Exact hash lookup for `seed_state_dense` -> `seed_state_dense` index.
    #[serde(skip)]
    pub(crate) seed_state_by_dense_hash: FxHashMap<u64, Vec<usize>>,
    /// Optional pre-expanded output masks for expensive seed-state dense masks.
    #[serde(skip)]
    pub(crate) seed_state_buf_masks: SeedStateBufMasks,
    /// Dense bitmap of the full internal token universe.
    #[serde(skip, default = "empty_dense_words")]
    pub(crate) seed_universe_dense: DenseWords,
    /// Fast DWA transition lookup (FxHashMap instead of BTreeMap).
    /// Built from parser_dwa.states at load/build time.
    #[serde(skip)]
    pub(crate) dwa_fast_transitions: FastDwaTransitions,
    /// Dense tokenizer transition lookup for commit-time byte scans.
    #[serde(skip)]
    pub(crate) tokenizer_fast_transitions: FastTokenizerTransitions,
    /// Dense buf masks for "heavy" internal tokens (those with many buf entries).
    /// Indexed by internal token ID; None for light tokens.
    #[serde(skip)]
    pub(crate) heavy_token_dense_masks: Vec<Option<Box<[u32]>>>,
    /// Flattened contiguous array of all internal token buf mask entries.
    /// All tokens' (word_index, or_mask) pairs concatenated in token order.
    /// Improves cache locality vs separate Vec allocations per token.
    #[serde(skip)]
    pub(crate) internal_token_buf_flat: Box<[(u16, u32)]>,
    /// Offsets into `internal_token_buf_flat` for each internal token.
    /// `internal_token_buf_flat[offsets[i]..offsets[i+1]]` gives token i's entries.
    /// Length = n_internal + 1 (sentinel at end).
    #[serde(skip)]
    pub(crate) internal_token_buf_offsets: Box<[u32]>,
    /// Pre-computed total cost (sum of entry counts) for all internal tokens.
    /// Used to avoid O(n_internal) cost analysis in the convert phase.
    #[serde(skip)]
    pub(crate) total_internal_buf_cost: usize,
    /// Indices of heavy tokens for fast iteration. Length == n_heavy_tokens.
    #[serde(skip)]
    pub(crate) heavy_token_indices: Vec<usize>,
    /// Total cost of all heavy tokens combined (n_heavy × buf_len).
    #[serde(skip)]
    pub(crate) heavy_total_cost: usize,
    /// Average cost per light token: (total_cost - heavy_total) / n_light.
    /// Pre-multiplied by 256 for fixed-point arithmetic to avoid float.
    #[serde(skip)]
    pub(crate) light_avg_cost_x256: usize,
    /// Exact materialization cost per internal token, after heavy-token dense masks
    /// have been chosen.
    #[serde(skip)]
    pub(crate) internal_token_buf_op_costs: Vec<usize>,
    /// Exact materialization cost per 64-token internal word group.
    #[serde(skip)]
    pub(crate) word_group_buf_op_costs: Vec<usize>,
    /// Self-contained final internal-token -> original-token bitset materializer.
    #[serde(skip)]
    pub(crate) final_mask_mapping: FinalMaskMapping,
}


impl Constraint {
    /// Assemble a runtime constraint from semantic compile outputs and empty
    /// derived cache storage.
    pub(crate) fn from_compiled_parts(parts: CompiledArtifactParts) -> Self {
        let caches = RuntimeCaches::default();
        let json_u_prefix_token_id = parts
            .token_bytes
            .iter()
            .find_map(|(&token_id, bytes)| (bytes.as_slice() == b"\\u").then_some(token_id));

        Self {
            parser_dwa: parts.parser_dwa,
            table: parts.table,
            terminal_display_names: parts.terminal_display_names,
            tokenizer: parts.tokenizer,
            ignore_terminal: parts.ignore_terminal,
            can_match: parts.can_match,
            state_to_internal_tsid: parts.state_to_internal_tsid,
            internal_tsid_to_states: parts.internal_tsid_to_states,
            template_dfas_by_terminal: parts.template_dfas_by_terminal,
            original_token_to_internal: parts.original_token_to_internal,
            internal_token_to_tokens: parts.internal_token_to_tokens,
            eos_token_id: parts.eos_token_id,
            json_u_prefix_token_id,
            json_escape_prefix_buf_mask: caches.json_escape_prefix_buf_mask,
            token_bytes: parts.token_bytes,
            internal_token_bytes: parts.internal_token_bytes,
            token_bytes_dense: caches.token_bytes_dense,
            internal_token_buf_masks: caches.internal_token_buf_masks,
            word_group_buf_masks: caches.word_group_buf_masks,
            pair_word_group_buf_masks: caches.pair_word_group_buf_masks,
            quad_word_group_buf_masks: caches.quad_word_group_buf_masks,
            super_word_group_buf_masks: caches.super_word_group_buf_masks,
            mega_word_group_buf_masks: caches.mega_word_group_buf_masks,
            giga_word_group_buf_masks: caches.giga_word_group_buf_masks,
            word_group_sparse_masks: caches.word_group_sparse_masks,
            word_group_prefix_buf_masks: caches.word_group_prefix_buf_masks,
            word_group_sparse_prefix_entries: caches.word_group_sparse_prefix_entries,
            quad_group_sparse_masks: caches.quad_group_sparse_masks,
            byte_group_sparse_masks: caches.byte_group_sparse_masks,
            word_group_sparse_total_entries: caches.word_group_sparse_total_entries,
            word_group_sparse_max_entries: caches.word_group_sparse_max_entries,
            all_tokens_buf_mask: caches.all_tokens_buf_mask,
            internal_token_dense_words: caches.internal_token_dense_words,
            weight_token_dense_masks: caches.weight_token_dense_masks,
            weight_token_buf_masks: caches.weight_token_buf_masks,
            weight_token_sparse_buf_masks: caches.weight_token_sparse_buf_masks,
            seed_terminal_dense: caches.seed_terminal_dense,
            seed_state_dense: caches.seed_state_dense,
            seed_state_by_dense_hash: caches.seed_state_by_dense_hash,
            seed_state_buf_masks: caches.seed_state_buf_masks,
            seed_universe_dense: caches.seed_universe_dense,
            dwa_fast_transitions: caches.dwa_fast_transitions,
            tokenizer_fast_transitions: caches.tokenizer_fast_transitions,
            heavy_token_dense_masks: caches.heavy_token_dense_masks,
            internal_token_buf_flat: caches.internal_token_buf_flat,
            internal_token_buf_offsets: caches.internal_token_buf_offsets,
            total_internal_buf_cost: caches.total_internal_buf_cost,
            heavy_token_indices: caches.heavy_token_indices,
            heavy_total_cost: caches.heavy_total_cost,
            light_avg_cost_x256: caches.light_avg_cost_x256,
            internal_token_buf_op_costs: caches.internal_token_buf_op_costs,
            word_group_buf_op_costs: caches.word_group_buf_op_costs,
            final_mask_mapping: caches.final_mask_mapping,
        }
    }
}
