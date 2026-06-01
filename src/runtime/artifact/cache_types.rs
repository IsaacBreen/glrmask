//! Derived runtime cache storage for a compiled artifact.
//!
//! The serialized mathematical artifact is not enough for fast Mask/Commit.
//! After construction or load, we derive dense/sparse materialization tables,
//! fast transition arrays, seed masks, and profile caches.  The `Constraint`
//! struct still stores these fields directly for compatibility with the current
//! runtime code; `RuntimeCaches` is the named aggregate used when constructing a
//! fresh artifact and documents exactly which fields are derived rather than
//! semantic.

use rustc_hash::FxHashMap;

use crate::sets::weight::Weight;
use crate::grammar::flat::TerminalID;
use crate::runtime::token_space::final_mask_mapping::FinalMaskMapping;

use super::dense::{empty_dense_words, DenseWords};

/// Sparse output mask for one runtime-internal token.
pub(crate) type InternalTokenBufMasks = Vec<(u16, u32)>;
/// Cache from Weight pointer/id to dense internal-token mask.
pub(crate) type DenseWeightMaskCache = FxHashMap<usize, DenseWords>;
/// Cache from Weight pointer/id to dense output-vocabulary mask.
pub(crate) type DenseWeightBufMaskCache = FxHashMap<usize, Box<[u32]>>;
/// Cache from Weight pointer/id to sparse output-vocabulary mask.
pub(crate) type SparseWeightBufMaskCache = FxHashMap<usize, Box<[(u16, u32)]>>;
/// Dense seed masks indexed by `(original tokenizer state, grammar terminal)`.
pub(crate) type SeedTerminalDenseMasks = FxHashMap<(u32, TerminalID), DenseWords>;
/// Dense seed baseline masks indexed by original tokenizer state.
pub(crate) type SeedStateDenseMasks = Vec<DenseWords>;
/// Optional materialized output masks for expensive seed baselines.
pub(crate) type SeedStateBufMasks = Vec<Option<Box<[u32]>>>;
/// Parser-DWA transition table specialized for fast runtime lookup.
pub(crate) type FastDwaTransitions = Vec<FxHashMap<i32, (u32, Weight)>>;
/// Tokenizer DFA transition table specialized for byte-wise commit.
pub(crate) type FastTokenizerTransitions = Vec<Box<[u32; 256]>>;

/// All fields that can be rebuilt from the serialized compiled artifact.
#[derive(Debug, Clone)]
pub(crate) struct RuntimeCaches {
    pub(crate) json_escape_prefix_buf_mask: Box<[u32]>,
    pub(crate) token_bytes_dense: Vec<Option<Box<[u8]>>>,
    pub(crate) internal_token_buf_masks: Vec<InternalTokenBufMasks>,
    pub(crate) word_group_buf_masks: Vec<Box<[u32]>>,
    pub(crate) pair_word_group_buf_masks: Vec<Box<[u32]>>,
    pub(crate) quad_word_group_buf_masks: Vec<Box<[u32]>>,
    pub(crate) super_word_group_buf_masks: Vec<Box<[u32]>>,
    pub(crate) mega_word_group_buf_masks: Vec<Box<[u32]>>,
    pub(crate) giga_word_group_buf_masks: Vec<Box<[u32]>>,
    pub(crate) word_group_sparse_masks: Vec<InternalTokenBufMasks>,
    pub(crate) word_group_prefix_buf_masks: Vec<Box<[u32]>>,
    pub(crate) word_group_sparse_prefix_entries: Vec<usize>,
    pub(crate) quad_group_sparse_masks: Vec<InternalTokenBufMasks>,
    pub(crate) byte_group_sparse_masks: Vec<InternalTokenBufMasks>,
    pub(crate) word_group_sparse_total_entries: usize,
    pub(crate) word_group_sparse_max_entries: usize,
    pub(crate) all_tokens_buf_mask: Box<[u32]>,
    pub(crate) internal_token_dense_words: usize,
    pub(crate) weight_token_dense_masks: DenseWeightMaskCache,
    pub(crate) weight_token_buf_masks: DenseWeightBufMaskCache,
    pub(crate) weight_token_sparse_buf_masks: SparseWeightBufMaskCache,
    pub(crate) seed_terminal_dense: SeedTerminalDenseMasks,
    pub(crate) seed_state_dense: SeedStateDenseMasks,
    pub(crate) seed_state_by_dense_hash: FxHashMap<u64, Vec<usize>>,
    pub(crate) seed_state_buf_masks: SeedStateBufMasks,
    pub(crate) seed_universe_dense: DenseWords,
    pub(crate) dwa_fast_transitions: FastDwaTransitions,
    pub(crate) tokenizer_fast_transitions: FastTokenizerTransitions,
    pub(crate) heavy_token_dense_masks: Vec<Option<Box<[u32]>>>,
    pub(crate) internal_token_buf_flat: Box<[(u16, u32)]>,
    pub(crate) internal_token_buf_offsets: Box<[u32]>,
    pub(crate) total_internal_buf_cost: usize,
    pub(crate) heavy_token_indices: Vec<usize>,
    pub(crate) heavy_total_cost: usize,
    pub(crate) light_avg_cost_x256: usize,
    pub(crate) internal_token_buf_op_costs: Vec<usize>,
    pub(crate) word_group_buf_op_costs: Vec<usize>,
    pub(crate) final_mask_mapping: FinalMaskMapping,
}

impl Default for RuntimeCaches {
    fn default() -> Self {
        Self {
            json_escape_prefix_buf_mask: Box::new([]),
            token_bytes_dense: Vec::new(),
            internal_token_buf_masks: Vec::new(),
            word_group_buf_masks: Vec::new(),
            pair_word_group_buf_masks: Vec::new(),
            quad_word_group_buf_masks: Vec::new(),
            super_word_group_buf_masks: Vec::new(),
            mega_word_group_buf_masks: Vec::new(),
            giga_word_group_buf_masks: Vec::new(),
            word_group_sparse_masks: Vec::new(),
            word_group_prefix_buf_masks: Vec::new(),
            word_group_sparse_prefix_entries: Vec::new(),
            quad_group_sparse_masks: Vec::new(),
            byte_group_sparse_masks: Vec::new(),
            word_group_sparse_total_entries: 0,
            word_group_sparse_max_entries: 0,
            all_tokens_buf_mask: Box::new([]),
            internal_token_dense_words: 0,
            weight_token_dense_masks: FxHashMap::default(),
            weight_token_buf_masks: FxHashMap::default(),
            weight_token_sparse_buf_masks: FxHashMap::default(),
            seed_terminal_dense: FxHashMap::default(),
            seed_state_dense: Vec::new(),
            seed_state_by_dense_hash: FxHashMap::default(),
            seed_state_buf_masks: Vec::new(),
            seed_universe_dense: empty_dense_words(),
            dwa_fast_transitions: Vec::new(),
            tokenizer_fast_transitions: Vec::new(),
            heavy_token_dense_masks: Vec::new(),
            internal_token_buf_flat: Box::new([]),
            internal_token_buf_offsets: Box::new([]),
            total_internal_buf_cost: 0,
            heavy_token_indices: Vec::new(),
            heavy_total_cost: 0,
            light_avg_cost_x256: 0,
            internal_token_buf_op_costs: Vec::new(),
            word_group_buf_op_costs: Vec::new(),
            final_mask_mapping: FinalMaskMapping::default(),
        }
    }
}
