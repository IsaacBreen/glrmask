use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::{TerminalSelfLoopBytes, Tokenizer};
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::weighted::dwa::DWA;
use crate::ds::leveled_gss::LeveledGSS;
use crate::compiler::glr::table::GLRTable;
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::ds::u8set::U8Set;
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
pub(crate) type FastDwaTransitions = Vec<FxHashMap<i32, (u32, Weight)>>;
pub(crate) type FastTokenizerTransitions = Vec<Box<[u32; 256]>>;
pub(crate) type TemplateDfasByTerminal = Vec<Option<Arc<CommitTemplateDfas>>>;

/// A lazy vocabulary split for a terminal quotient-loop byte set.  Tokens in
/// `safe_mask` can stay inside the current lexer residual language for their
/// entire byte string; only `exception_trie` needs exact dynamic traversal.
#[derive(Debug)]
pub(crate) struct DynamicLoopPartition {
    pub(crate) exception_trie: Arc<VocabPrefixTree>,
    pub(crate) safe_mask: Box<[u32]>,
    pub(crate) safe_token_count: usize,
    pub(crate) exception_token_count: usize,
}

/// Exact direct-continuation partition for one lexer/parser residual state.
///
/// A token in `safe_mask` reaches a lexer state from which these unchanged
/// parser stacks can still eventually consume a terminal. It is therefore
/// admissible without inspecting terminal matches inside the token. The
/// exception trie contains every other token and is processed by the normal
/// exact walk.
#[derive(Debug)]
pub(crate) struct DynamicContinuationPartition {
    pub(crate) tokenizer_state: u32,
    /// Keeps the pointer used for cache identity live, so an allocator cannot
    /// recycle it for an unrelated parser GSS.
    pub(crate) stacks: LeveledGSS<u32, ()>,
    pub(crate) safe_mask: Arc<[u32]>,
    pub(crate) exception_trie: Arc<VocabPrefixTree>,
    pub(crate) safe_canonical_tokens: usize,
    pub(crate) exception_canonical_tokens: usize,
}

/// Runtime-only vocabulary data for direct dynamic mask generation.
#[derive(Debug, Clone)]
pub(crate) struct DynamicMaskVocab {
    pub(crate) trie: Arc<VocabPrefixTree>,
    /// Each trie leaf stores one canonical token id. This restores every vocab
    /// id that has the same byte string.
    pub(crate) token_ids: Arc<BTreeMap<u32, Box<[u32]>>>,
    /// Canonical byte strings retained solely for lazy loop partitions.
    pub(crate) canonical_token_bytes: Arc<BTreeMap<u32, Box<[u8]>>>,
    /// Flat form used by hot lazy continuation-partition construction without
    /// B-tree iteration overhead.
    pub(crate) canonical_tokens: Arc<[(u32, Box<[u8]>)]>,
    /// Alias lists parallel to `canonical_tokens`. These avoid a B-tree lookup
    /// per token while constructing a partition mask.
    pub(crate) canonical_aliases: Arc<[Box<[u32]>]>,
    pub(crate) output_mask_words: usize,
    /// Built only when direct dynamic masking is used.  Keeping this cache in
    /// the runtime-only artifact avoids serializing it and keeps tokenizer
    /// simplification/construction paths independent of dynamic-mask details.
    pub(crate) terminal_self_loop_bytes: Arc<OnceLock<TerminalSelfLoopBytes>>,
    pub(crate) loop_partitions: Arc<Mutex<FxHashMap<U8Set, Arc<DynamicLoopPartition>>>>,
    /// A small strong-reference cache. Entries are keyed by lexer state and
    /// parser-GSS identity; retaining the GSS makes the identity immune to
    /// allocator-address reuse after a state is dropped.
    pub(crate) continuation_partitions:
        Arc<Mutex<Vec<Arc<DynamicContinuationPartition>>>>,
}

impl DynamicMaskVocab {
    #[inline]
    pub(crate) fn terminal_self_loop_bytes(
        &self,
        tokenizer: &Tokenizer,
    ) -> &TerminalSelfLoopBytes {
        self.terminal_self_loop_bytes
            .get_or_init(|| tokenizer.terminal_self_loop_bytes_map())
    }

    pub(crate) fn loop_partition(&self, loop_bytes: U8Set) -> Arc<DynamicLoopPartition> {
        let mut partitions = self
            .loop_partitions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(partition) = partitions.get(&loop_bytes) {
            return partition.clone();
        }

        let mut safe_mask = vec![0u32; self.output_mask_words];
        let mut exception_entries = Vec::<(usize, &[u8])>::new();
        let mut safe_token_count = 0usize;
        for (index, &(canonical_token_id, ref bytes)) in self.canonical_tokens.iter().enumerate() {
            if bytes.iter().all(|&byte| loop_bytes.contains(byte)) {
                let token_ids = &self.canonical_aliases[index];
                safe_token_count += token_ids.len();
                for &token_id in token_ids.iter() {
                    let word = token_id as usize / 32;
                    debug_assert!(word < safe_mask.len());
                    safe_mask[word] |= 1u32 << (token_id % 32);
                }
            } else {
                exception_entries.push((canonical_token_id as usize, bytes.as_ref()));
            }
        }

        let exception_token_count = exception_entries.len();
        let partition = Arc::new(DynamicLoopPartition {
            exception_trie: Arc::new(VocabPrefixTree::build_presorted(&exception_entries)),
            safe_mask: safe_mask.into_boxed_slice(),
            safe_token_count,
            exception_token_count,
        });
        partitions.insert(loop_bytes, partition.clone());
        partition
    }
}

pub(crate) fn lookup_dynamic_continuation_partition(
    vocab: &DynamicMaskVocab,
    tokenizer_state: u32,
    stacks: &LeveledGSS<u32, ()>,
) -> Option<Arc<DynamicContinuationPartition>> {
    let partitions = vocab
        .continuation_partitions
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    partitions
        .iter()
        .find(|partition| {
            partition.tokenizer_state == tokenizer_state && partition.stacks.ptr_eq(stacks)
        })
        .cloned()
}

pub(crate) fn cache_dynamic_continuation_partition(
    vocab: &DynamicMaskVocab,
    partition: Arc<DynamicContinuationPartition>,
) -> Arc<DynamicContinuationPartition> {
    const MAX_CACHED_CONTINUATION_PARTITIONS: usize = 64;
    let mut partitions = vocab
        .continuation_partitions
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(existing) = partitions.iter().find(|existing| {
        existing.tokenizer_state == partition.tokenizer_state
            && existing.stacks.ptr_eq(&partition.stacks)
    }) {
        return existing.clone();
    }
    if partitions.len() == MAX_CACHED_CONTINUATION_PARTITIONS {
        partitions.remove(0);
    }
    partitions.push(partition.clone());
    partition
}

impl Default for DynamicMaskVocab {
    fn default() -> Self {
        Self {
            trie: Arc::new(VocabPrefixTree::new()),
            token_ids: Arc::new(BTreeMap::new()),
            canonical_token_bytes: Arc::new(BTreeMap::new()),
            canonical_tokens: Arc::from(Vec::<(u32, Box<[u8]>)>::new().into_boxed_slice()),
            canonical_aliases: Arc::from(Vec::<Box<[u32]>>::new().into_boxed_slice()),
            output_mask_words: 0,
            terminal_self_loop_bytes: Arc::new(OnceLock::new()),
            loop_partitions: Arc::new(Mutex::new(FxHashMap::default())),
            continuation_partitions: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CommitTemplateDfas {
    pub(crate) pop: UnweightedDfa,
    pub(crate) read: UnweightedDfa,
    pub(crate) push: UnweightedDfa,
    pub(crate) pop_to_read: Vec<Option<u32>>,
    pub(crate) pop_to_push: Vec<Option<u32>>,
    pub(crate) read_to_push: Vec<Option<u32>>,
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

    /// Runtime-only vocabulary data for direct dynamic masking.
    #[serde(skip, default)]
    pub(crate) dynamic_mask_vocab: DynamicMaskVocab,

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
