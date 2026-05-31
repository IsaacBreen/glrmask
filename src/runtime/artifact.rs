use std::collections::BTreeMap;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::table::GLRTable;
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;

use super::mask_mapping::FinalMaskMapping;

pub(crate) type PossibleMatchesByTerminal = BTreeMap<TerminalID, Weight>;
pub(crate) type DenseWords = Arc<[u64]>;

pub(crate) fn empty_dense_words() -> DenseWords {
    Arc::<[u64]>::from(Vec::<u64>::new().into_boxed_slice())
}

pub(crate) type InternalTokenBufMasks = Vec<(u16, u32)>;
pub(crate) type DenseWeightMaskCache = FxHashMap<usize, DenseWords>;
pub(crate) type DenseWeightBufMaskCache = FxHashMap<usize, Box<[u32]>>;
pub(crate) type SparseWeightBufMaskCache = FxHashMap<usize, Box<[(u16, u32)]>>;
pub(crate) type SeedTerminalDenseMasks = FxHashMap<(u32, TerminalID), DenseWords>;
pub(crate) type SeedStateDenseMasks = Vec<DenseWords>;
pub(crate) type SeedStateBufMasks = Vec<Option<Box<[u32]>>>;
pub(crate) type FastDwaTransitions = Vec<FxHashMap<i32, (u32, Weight)>>;
pub(crate) type FastTokenizerTransitions = Vec<Box<[u32; 256]>>;
pub(crate) type TemplateDfasByTerminal = Vec<Option<Arc<UnweightedDfa>>>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Constraint {
    pub(crate) parser_dwa: DWA,
    pub(crate) table: GLRTable,
    #[serde(default)]
    pub(crate) terminal_display_names: Vec<String>,
    pub(crate) tokenizer: Tokenizer,
    #[serde(default)]
    pub(crate) ignore_terminal: Option<TerminalID>,

    /// possible_matches keyed by grammar terminal id.
    ///
    /// Each Weight maps final shared internal tokenizer-state ids to token sets
    /// in the final shared constraint-internal vocab space. Parser-DWA weights
    /// and possible_matches weights are reconciled into this same space during
    /// compilation.
    pub(crate) possible_matches: PossibleMatchesByTerminal,
    pub(crate) state_to_internal_tsid: Vec<u32>,
    pub(crate) internal_tsid_to_states: Vec<Vec<u32>>,
    #[serde(skip)]
    pub(crate) template_dfas_by_terminal: TemplateDfasByTerminal,
    /// Original token -> final shared constraint-internal token id.
    ///
    /// This is not necessarily equal to the parser-DWA compaction vocab map
    /// produced before possible-match reconciliation. It may contain additional
    /// splits required by possible_matches.
    #[serde(default)]
    pub(crate) original_token_to_internal: Vec<u32>,
    /// Final shared constraint-internal token id -> original token ids.
    ///
    /// Parser-DWA weights and Constraint.possible_matches bitmaps both use these
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
