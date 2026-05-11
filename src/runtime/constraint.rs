use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use rayon::prelude::*;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::ParserGSS;
use crate::compiler::glr::table::GLRTable;
use crate::grammar::flat::TerminalID;
use crate::ds::weight::Weight;

use super::state::ConstraintState;

/// Runtime possible-matches table.
///
/// Outer key:
///   grammar terminal id.
///
/// Inner key (in each Weight):
///   final shared internal tokenizer-state id.
///
/// Value (in each Weight):
///   final shared constraint-internal token ids.
///
/// This is symmetric with terminal DWA weights: terminal DWA transitions carry
/// a Weight over source states; possible_matches carries a Weight over source
/// states for each terminal.
///
/// The bitmap is NOT in original token space and is NOT in the raw parser-DWA
/// vocab-compaction space. During compilation, raw possible_matches are computed
/// in original token space, then remapped into a constraint vocab that refines
/// the parser-DWA vocab by possible-match signature and seed-state lexical-live
/// signature. Parser-DWA weights are remapped into this same token space.
pub(crate) type PossibleMatchesByTerminal = BTreeMap<TerminalID, Weight>;
type DenseWords = Arc<[u64]>;

fn empty_dense_words() -> DenseWords {
    Arc::<[u64]>::from(Vec::<u64>::new().into_boxed_slice())
}
type InternalTokenBufMasks = Vec<(u16, u32)>;
type DenseWeightMaskCache = FxHashMap<usize, DenseWords>;
type SeedTerminalDenseMasks = FxHashMap<(u32, TerminalID), DenseWords>;
type SeedStateDenseMasks = Vec<DenseWords>;
type FastDwaTransitions = Vec<FxHashMap<i32, (u32, Weight)>>;
type FastTokenizerTransitions = Vec<Box<[u32; 256]>>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Constraint {
    pub(crate) parser_dwa: DWA,
    pub(crate) table: GLRTable,
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
}

/// Dense buf OR: `buf[i] |= mask[i]` for all i in min(buf.len(), mask.len()).
/// Processes u64 chunks for reduced loop overhead and better throughput.
#[inline(always)]
fn or_dense_buf(buf: &mut [u32], mask: &[u32]) {
    let n = buf.len().min(mask.len());
    // Process pairs of u32 as u64 for 2× fewer iterations.
    let n_pairs = n / 2;
    if n_pairs > 0 {
        let buf64 = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u64, n_pairs) };
        let mask64 = unsafe { std::slice::from_raw_parts(mask.as_ptr() as *const u64, n_pairs) };
        for (b, &m) in buf64.iter_mut().zip(mask64.iter()) {
            *b |= m;
        }
    }
    // Handle trailing u32.
    if n % 2 == 1 {
        buf[n - 1] |= mask[n - 1];
    }
}

/// Dense buf AND-NOT: `buf[i] &= !mask[i]` for all i in min(buf.len(), mask.len()).
/// Processes u64 chunks for reduced loop overhead and better throughput.
#[inline(always)]
fn andnot_dense_buf(buf: &mut [u32], mask: &[u32]) {
    let n = buf.len().min(mask.len());
    let n_pairs = n / 2;
    if n_pairs > 0 {
        let buf64 = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u64, n_pairs) };
        let mask64 = unsafe { std::slice::from_raw_parts(mask.as_ptr() as *const u64, n_pairs) };
        for (b, &m) in buf64.iter_mut().zip(mask64.iter()) {
            *b &= !m;
        }
    }
    if n % 2 == 1 {
        buf[n - 1] &= !mask[n - 1];
    }
}

#[inline(always)]
fn copy_dense_buf(buf: &mut [u32], mask: &[u32]) {
    let n = buf.len().min(mask.len());
    buf[..n].copy_from_slice(&mask[..n]);
}

#[inline(always)]
fn or_sparse_buf_entries(buf: &mut [u32], entries: &[(u16, u32)]) {
    for &(word_idx, mask) in entries {
        unsafe {
            let slot = buf.get_unchecked_mut(word_idx as usize);
            *slot |= mask;
        }
    }
}

#[inline(always)]
fn andnot_sparse_buf_entries(buf: &mut [u32], entries: &[(u16, u32)]) {
    for &(word_idx, mask) in entries {
        unsafe {
            let slot = buf.get_unchecked_mut(word_idx as usize);
            *slot &= !mask;
        }
    }
}

#[derive(Default, Clone, Copy)]
pub(crate) struct DenseToBufProfileStats {
    pub dense_words_visited: u64,
    pub normal_full_word_hits: u64,
    pub complement_full_word_hits: u64,
    pub complement_full_byte_groups: u64,
    pub complement_full_nibble_groups: u64,
    pub complement_remaining_bits: u64,
    pub normal_token_iterations: u64,
    pub complement_token_iterations: u64,
    pub normal_sparse_entries: u64,
    pub complement_sparse_entries: u64,
    pub complement_heavy_dense_clears: u64,
    pub complement_max_sparse_span: u64,
    pub group_or_sparse_entries: u64,
    pub group_andnot_sparse_entries: u64,
    pub complement_path_used: bool,
}

#[derive(Default, Clone, Copy)]
pub(crate) struct DeltaReplayProfileStats {
    pub added_word_group_hits: u64,
    pub added_word_group_entries: u64,
    pub removed_word_group_hits: u64,
    pub removed_word_group_entries: u64,
    pub added_byte_group_hits: u64,
    pub added_byte_group_entries: u64,
    pub removed_byte_group_hits: u64,
    pub removed_byte_group_entries: u64,
    pub added_token_iterations: u64,
    pub added_token_entries: u64,
    pub removed_token_iterations: u64,
    pub removed_token_entries: u64,
}

#[inline(always)]
fn count_complement_subgroups(missing: u64, valid_mask: u64) -> (u32, u32, u32) {
    let mut byte_groups = 0u32;
    let mut nibble_groups = 0u32;
    let mut remaining_bits = 0u32;

    for byte_idx in 0..8 {
        let shift = byte_idx * 8;
        let byte_valid = ((valid_mask >> shift) & 0xff) as u8;
        if byte_valid == 0 {
            continue;
        }

        let byte_missing = ((missing >> shift) & 0xff) as u8;
        if byte_valid == 0xff && byte_missing == 0xff {
            byte_groups += 1;
            continue;
        }

        for nibble_idx in 0..2 {
            let nibble_shift = nibble_idx * 4;
            let nibble_valid = (byte_valid >> nibble_shift) & 0x0f;
            if nibble_valid == 0 {
                continue;
            }

            let nibble_missing = (byte_missing >> nibble_shift) & 0x0f;
            if nibble_valid == 0x0f && nibble_missing == 0x0f {
                nibble_groups += 1;
            } else {
                remaining_bits += nibble_missing.count_ones();
            }
        }
    }

    (byte_groups, nibble_groups, remaining_bits)
}

impl Constraint {
    pub(crate) fn internal_token_materialization_cost(&self, internal_token: usize) -> u64 {
        if internal_token < self.heavy_token_dense_masks.len()
            && self.heavy_token_dense_masks[internal_token].is_some()
        {
            return self.mask_len() as u64;
        }
        if internal_token + 1 >= self.internal_token_buf_offsets.len() {
            return 0;
        }
        (self.internal_token_buf_offsets[internal_token + 1]
            - self.internal_token_buf_offsets[internal_token]) as u64
    }

    pub(crate) fn estimate_internal_dense_to_buf_cost(&self, dense: &[u64]) -> u64 {
        let all_mask = &self.all_tokens_buf_mask;
        let sparse_word_groups = &self.word_group_sparse_masks;
        let offsets = &self.internal_token_buf_offsets;
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };
        if n_internal == 0 || dense.is_empty() {
            return 0;
        }

        let n_set: usize = dense.iter().map(|w| w.count_ones() as usize).sum();
        let buf_len = self.mask_len();
        if n_set >= n_internal && !all_mask.is_empty() {
            return buf_len as u64;
        }
        if n_set == 0 {
            return 0;
        }

        let heavy = &self.heavy_token_dense_masks;
        let n_missing = n_internal - n_set;
        let n_heavy_set = if !self.heavy_token_indices.is_empty() {
            let mut count = 0usize;
            for &idx in &self.heavy_token_indices {
                if idx < n_internal {
                    let wi = idx / 64;
                    let bit = idx % 64;
                    if wi < dense.len() && (dense[wi] >> bit) & 1 != 0 {
                        count += 1;
                    }
                }
            }
            count
        } else {
            0
        };
        let n_light_set = n_set.saturating_sub(n_heavy_set);
        let n_heavy_missing = self.heavy_token_indices.len().saturating_sub(n_heavy_set);
        let est_set_cost = n_heavy_set * buf_len + (n_light_set * self.light_avg_cost_x256) / 256;
        let est_missing_cost = n_heavy_missing * buf_len
            + (n_missing.saturating_sub(n_heavy_missing) * self.light_avg_cost_x256) / 256;
        let copy_overhead = buf_len;

        if !all_mask.is_empty() && est_missing_cost + copy_overhead < est_set_cost {
            let mut cost = buf_len as u64;
            for (wi, &w) in dense.iter().enumerate() {
                if wi * 64 >= n_internal {
                    break;
                }
                let remaining = n_internal - wi * 64;
                let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                let missing = !w & valid_mask;
                if missing == 0 {
                    continue;
                }
                if missing == valid_mask {
                    if let Some(group_mask) = sparse_word_groups.get(wi) {
                        cost += group_mask.len() as u64;
                        continue;
                    }
                }
                let mut bits = missing;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let internal_token = wi * 64 + bit;
                    cost += self.internal_token_materialization_cost(internal_token);
                    bits &= bits - 1;
                }
            }
            cost
        } else {
            let mut cost = 0u64;
            for (wi, &w) in dense.iter().enumerate() {
                if wi * 64 >= n_internal {
                    break;
                }
                let remaining = n_internal - wi * 64;
                let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                let valid_bits = w & valid_mask;
                if valid_bits == 0 {
                    continue;
                }
                if valid_bits == valid_mask {
                    if let Some(group_mask) = sparse_word_groups.get(wi) {
                        cost += group_mask.len() as u64;
                        continue;
                    }
                }
                let mut bits = valid_bits;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let internal_token = wi * 64 + bit;
                    cost += self.internal_token_materialization_cost(internal_token);
                    bits &= bits - 1;
                }
            }
            cost
        }
    }

    pub(crate) fn apply_internal_dense_delta_to_buf(
        &self,
        previous_dense: &[u64],
        current_dense: &[u64],
        buf: &mut [u32],
    ) -> DeltaReplayProfileStats {
        let mut stats = DeltaReplayProfileStats::default();
        let offsets = &self.internal_token_buf_offsets;
        let flat = &self.internal_token_buf_flat;
        let heavy = &self.heavy_token_dense_masks;
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };

        if n_internal == 0 {
            return stats;
        }

        let word_len = previous_dense.len().max(current_dense.len());
        for wi in 0..word_len {
            if wi * 64 >= n_internal {
                break;
            }
            let remaining = n_internal - wi * 64;
            let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
            let previous = previous_dense.get(wi).copied().unwrap_or(0) & valid_mask;
            let current = current_dense.get(wi).copied().unwrap_or(0) & valid_mask;

            let mut added = current & !previous;
            if added == valid_mask {
                if let Some(group_mask) = self.word_group_sparse_masks.get(wi) {
                    stats.added_word_group_hits += 1;
                    stats.added_word_group_entries += group_mask.len() as u64;
                    or_sparse_buf_entries(buf, group_mask);
                    continue;
                }
            }
            for byte_idx in 0..8 {
                let shift = byte_idx * 8;
                let byte_valid = (valid_mask >> shift) & 0xff;
                let byte_bits = (added >> shift) & 0xff;
                if byte_valid == 0xff && byte_bits == 0xff {
                    if let Some(group_mask) = self.byte_group_sparse_masks.get(wi * 8 + byte_idx) {
                        stats.added_byte_group_hits += 1;
                        stats.added_byte_group_entries += group_mask.len() as u64;
                        or_sparse_buf_entries(buf, group_mask);
                        added &= !(0xffu64 << shift);
                    }
                }
            }
            while added != 0 {
                stats.added_token_iterations += 1;
                let bit = added.trailing_zeros() as usize;
                let internal_token = wi * 64 + bit;
                if internal_token < heavy.len() {
                    if let Some(ref dense_mask) = heavy[internal_token] {
                        stats.added_token_entries += dense_mask.len() as u64;
                        or_dense_buf(buf, dense_mask);
                        added &= added - 1;
                        continue;
                    }
                }
                let start = offsets[internal_token] as usize;
                let end = offsets[internal_token + 1] as usize;
                stats.added_token_entries += (end - start) as u64;
                or_sparse_buf_entries(buf, &flat[start..end]);
                added &= added - 1;
            }

        }

        for wi in 0..word_len {
            if wi * 64 >= n_internal {
                break;
            }
            let remaining = n_internal - wi * 64;
            let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
            let previous = previous_dense.get(wi).copied().unwrap_or(0) & valid_mask;
            let current = current_dense.get(wi).copied().unwrap_or(0) & valid_mask;

            let mut removed = previous & !current;
            if removed == valid_mask {
                if let Some(group_mask) = self.word_group_sparse_masks.get(wi) {
                    stats.removed_word_group_hits += 1;
                    stats.removed_word_group_entries += group_mask.len() as u64;
                    andnot_sparse_buf_entries(buf, group_mask);
                    continue;
                }
            }
            for byte_idx in 0..8 {
                let shift = byte_idx * 8;
                let byte_valid = (valid_mask >> shift) & 0xff;
                let byte_bits = (removed >> shift) & 0xff;
                if byte_valid == 0xff && byte_bits == 0xff {
                    if let Some(group_mask) = self.byte_group_sparse_masks.get(wi * 8 + byte_idx) {
                        stats.removed_byte_group_hits += 1;
                        stats.removed_byte_group_entries += group_mask.len() as u64;
                        andnot_sparse_buf_entries(buf, group_mask);
                        removed &= !(0xffu64 << shift);
                    }
                }
            }
            while removed != 0 {
                stats.removed_token_iterations += 1;
                let bit = removed.trailing_zeros() as usize;
                let internal_token = wi * 64 + bit;
                if internal_token < heavy.len() {
                    if let Some(ref dense_mask) = heavy[internal_token] {
                        stats.removed_token_entries += dense_mask.len() as u64;
                        andnot_dense_buf(buf, dense_mask);
                        removed &= removed - 1;
                        continue;
                    }
                }
                let start = offsets[internal_token] as usize;
                let end = offsets[internal_token + 1] as usize;
                stats.removed_token_entries += (end - start) as u64;
                andnot_sparse_buf_entries(buf, &flat[start..end]);
                removed &= removed - 1;
            }
        }

        stats
    }

    pub(crate) fn rebuild_runtime_caches(&mut self) {
        let profile = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let total_started_at = profile.then(std::time::Instant::now);
        let primary_started_at = profile.then(std::time::Instant::now);
        let (
            internal_token_buf_masks,
            tokenizer_fast_transitions,
            (dense_mask_words, dense_masks),
            fast_transitions,
        ) = if rayon::current_num_threads() == 1 {
            (
                self.compute_buf_masks(),
                self.compute_tokenizer_fast_transitions(),
                self.compute_dense_token_masks(),
                self.compute_fast_transitions(),
            )
        } else {
            let ((internal_token_buf_masks, tokenizer_fast_transitions), ((dense_mask_words, dense_masks), fast_transitions)) = rayon::join(
                || rayon::join(
                    || self.compute_buf_masks(),
                    || self.compute_tokenizer_fast_transitions(),
                ),
                || rayon::join(
                    || self.compute_dense_token_masks(),
                    || self.compute_fast_transitions(),
                ),
            );
            (
                internal_token_buf_masks,
                tokenizer_fast_transitions,
                (dense_mask_words, dense_masks),
                fast_transitions,
            )
        };
        let primary_ms = primary_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        self.internal_token_buf_masks = internal_token_buf_masks;
        self.word_group_buf_masks = Vec::new();
        let block_started_at = profile.then(std::time::Instant::now);
        let ((word_group_sparse_masks, word_group_sparse_total_entries, word_group_sparse_max_entries), (byte_group_sparse_masks, _, _)) =
            if rayon::current_num_threads() == 1 {
                (
                    self.compute_token_block_sparse_masks(64),
                    self.compute_token_block_sparse_masks(8),
                )
            } else {
                rayon::join(
                    || self.compute_token_block_sparse_masks(64),
                    || self.compute_token_block_sparse_masks(8),
                )
            };
        self.word_group_sparse_masks = word_group_sparse_masks;
        self.byte_group_sparse_masks = byte_group_sparse_masks;
        self.word_group_sparse_total_entries = word_group_sparse_total_entries;
        self.word_group_sparse_max_entries = word_group_sparse_max_entries;
        let block_ms = block_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let derived_started_at = profile.then(std::time::Instant::now);
        self.pair_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(2);
        self.quad_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(4);
        self.super_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(8);
        self.mega_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(16);
        self.giga_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(32);
        self.all_tokens_buf_mask = self.compute_all_tokens_buf_mask();
        self.heavy_token_dense_masks = self.compute_heavy_token_dense_masks();
        let (flat, offsets) = Self::compute_flat_buf_masks(&self.internal_token_buf_masks);
        self.internal_token_buf_flat = flat;
        self.internal_token_buf_offsets = offsets;
        self.total_internal_buf_cost = Self::compute_total_internal_buf_cost(
            &self.internal_token_buf_offsets,
            &self.heavy_token_dense_masks,
            self.mask_len(),
        );

        // Precompute heavy token stats for fast path decision in convert.
        let buf_len = self.mask_len();
        let n_internal = if self.internal_token_buf_offsets.len() > 1 {
            self.internal_token_buf_offsets.len() - 1
        } else {
            0
        };
        self.heavy_token_indices = self.heavy_token_dense_masks.iter().enumerate().filter_map(|(i, m)| if m.is_some() { Some(i) } else { None }).collect();
        self.heavy_total_cost = self.heavy_token_indices.len() * buf_len;
        let n_light = n_internal.saturating_sub(self.heavy_token_indices.len());
        let light_total = self.total_internal_buf_cost.saturating_sub(self.heavy_total_cost);
        self.light_avg_cost_x256 = if n_light > 0 { (light_total * 256) / n_light } else { 0 };

        self.token_bytes_dense = Vec::new();
        self.internal_token_dense_words = dense_mask_words;
        self.weight_token_dense_masks = dense_masks;
        self.dwa_fast_transitions = fast_transitions;
        self.tokenizer_fast_transitions = tokenizer_fast_transitions;
        let derived_ms = derived_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let seed_started_at = profile.then(std::time::Instant::now);
        self.build_seed_dense_masks();
        let seed_ms = seed_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][runtime_finalize] primary_ms={:.3} block_masks_ms={:.3} derived_masks_ms={:.3} seed_dense_ms={:.3} total_ms={:.3}",
                primary_ms,
                block_ms,
                derived_ms,
                seed_ms,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    fn compute_tokenizer_fast_transitions(&self) -> FastTokenizerTransitions {
        let build = |state| {
                let dfa_state = &self.tokenizer.dfa.states()[state as usize];
                let mut flat = Box::new([u32::MAX; 256]);
                for (byte, &target) in dfa_state.transitions.iter() {
                    flat[byte as usize] = target;
                }
                flat
            };
        if rayon::current_num_threads() == 1 {
            (0..self.tokenizer.num_states()).map(build).collect()
        } else {
            (0..self.tokenizer.num_states()).into_par_iter().map(build).collect()
        }
    }

    fn compute_buf_masks(&self) -> Vec<InternalTokenBufMasks> {
        if self.internal_token_to_tokens.is_empty() {
            return Vec::new();
        }

        if !self.original_token_to_internal.is_empty() {
            let mut masks = vec![Vec::<(u16, u32)>::new(); self.internal_token_to_tokens.len()];
            for (original, &internal) in self.original_token_to_internal.iter().enumerate() {
                if internal == u32::MAX {
                    continue;
                }
                let internal = internal as usize;
                let Some(mask) = masks.get_mut(internal) else {
                    continue;
                };
                let word = (original as u32 / 32) as u16;
                let bit = original as u32 % 32;
                if let Some((last_word, last_mask)) = mask.last_mut() {
                    if *last_word == word {
                        *last_mask |= 1u32 << bit;
                        continue;
                    }
                }
                mask.push((word, 1u32 << bit));
            }
            return masks;
        }

        if rayon::current_num_threads() == 1 {
            self.internal_token_to_tokens
                .iter()
                .map(|originals| Self::build_internal_token_buf_mask(originals))
                .collect()
        } else {
            self.internal_token_to_tokens
                .par_iter()
                .map(|originals| Self::build_internal_token_buf_mask(originals))
                .collect()
        }
    }

    fn compute_token_block_sparse_masks(&self, block_size: usize) -> (Vec<InternalTokenBufMasks>, usize, usize) {
        if self.internal_token_buf_masks.is_empty() {
            return (Vec::new(), 0, 0);
        }
        let n_groups = self.internal_token_buf_masks.len().div_ceil(block_size);
        let mask_words = self.mask_len();
        let build_group = |group_id: usize| {
                let group_start = group_id * block_size;
                let group_end = (group_start + block_size).min(self.internal_token_buf_masks.len());
                let mut dense = vec![0u32; mask_words];
                let mut touched = Vec::<u16>::new();
                for token_masks in &self.internal_token_buf_masks[group_start..group_end] {
                    for &(word_idx, mask) in token_masks {
                        let slot = &mut dense[word_idx as usize];
                        if *slot == 0 {
                            touched.push(word_idx);
                        }
                        *slot |= mask;
                    }
                }
                touched.sort_unstable();
                touched
                    .into_iter()
                    .map(|word_idx| (word_idx, dense[word_idx as usize]))
                    .collect()
            };
        let groups: Vec<InternalTokenBufMasks> = if rayon::current_num_threads() == 1 {
            (0..n_groups).map(build_group).collect()
        } else {
            (0..n_groups).into_par_iter().map(build_group).collect()
        };
        let total_entries = groups.iter().map(Vec::len).sum();
        let max_entries = groups.iter().map(Vec::len).max().unwrap_or(0);
        (groups, total_entries, max_entries)
    }

    fn compute_sliding_word_group_dense_masks(&self, word_group_len: usize) -> Vec<Box<[u32]>> {
        if self.internal_token_buf_masks.is_empty() || word_group_len == 0 {
            return Vec::new();
        }
        let n_word_groups = self.internal_token_buf_masks.len().div_ceil(64);
        let mask_words = self.mask_len();
        let build_group = |word_group_start: usize| {
            let group_start = word_group_start * 64;
            let group_end = (group_start + word_group_len * 64).min(self.internal_token_buf_masks.len());
            let mut dense = vec![0u32; mask_words];
            for token_masks in &self.internal_token_buf_masks[group_start..group_end] {
                for &(word_idx, mask) in token_masks {
                    dense[word_idx as usize] |= mask;
                }
            }
            dense.into_boxed_slice()
        };
        if rayon::current_num_threads() == 1 {
            (0..n_word_groups).map(build_group).collect()
        } else {
            (0..n_word_groups).into_par_iter().map(build_group).collect()
        }
    }

    fn compute_all_tokens_buf_mask(&self) -> Box<[u32]> {
        let buf_words = self.mask_len();
        let mut combined = vec![0u32; buf_words];
        for group in &self.word_group_sparse_masks {
            for &(word_idx, mask) in group {
                combined[word_idx as usize] |= mask;
            }
        }
        combined.into_boxed_slice()
    }

    /// Build dense buf masks for internal tokens with many sparse entries.
    /// A token with >THRESHOLD entries benefits from a sequential 16KB scan
    /// instead of thousands of indexed read-modify-writes.
    fn compute_heavy_token_dense_masks(&self) -> Vec<Option<Box<[u32]>>> {
        let buf_words = self.mask_len();
        if buf_words == 0 {
            return Vec::new();
        }
        // Threshold: use dense when sparse entries are large enough that a
        // sequential scan beats many indexed read-modify-writes.
        // Dense OR costs ~buf_words ops; sparse OR costs ~n_entries ops.
        // With buf in L1 cache (≤16KB), sparse random writes are fast,
        // so we only go dense when entries exceed half the buffer size.
        let threshold = buf_words / 4;
        let build = |sparse: &InternalTokenBufMasks| {
                if sparse.len() > threshold {
                    let mut dense = vec![0u32; buf_words];
                    for &(word_idx, mask) in sparse {
                        dense[word_idx as usize] |= mask;
                    }
                    Some(dense.into_boxed_slice())
                } else {
                    None
                }
            };
        if rayon::current_num_threads() == 1 {
            self.internal_token_buf_masks.iter().map(build).collect()
        } else {
            self.internal_token_buf_masks.par_iter().map(build).collect()
        }
    }

    /// Flatten all per-token sparse entries into a single contiguous array
    /// with an offset table. Improves cache locality during convert phase.
    fn compute_flat_buf_masks(masks: &[InternalTokenBufMasks]) -> (Box<[(u16, u32)]>, Box<[u32]>) {
        let total: usize = masks.iter().map(|m| m.len()).sum();
        let mut flat = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(masks.len() + 1);
        for m in masks {
            offsets.push(flat.len() as u32);
            flat.extend_from_slice(m);
        }
        offsets.push(flat.len() as u32);
        (flat.into_boxed_slice(), offsets.into_boxed_slice())
    }

    /// Pre-compute total cost for all internal tokens (sum of entry counts,
    /// with heavy tokens counted at buf_len).
    fn compute_total_internal_buf_cost(
        offsets: &[u32],
        heavy: &[Option<Box<[u32]>>],
        buf_len: usize,
    ) -> usize {
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };
        let mut total: usize = 0;
        for idx in 0..n_internal {
            if idx < heavy.len() && heavy[idx].is_some() {
                total += buf_len;
            } else {
                total += (offsets[idx + 1] - offsets[idx]) as usize;
            }
        }
        total
    }

    fn compute_dense_token_bytes(&self) -> Vec<Option<Box<[u8]>>> {
        let Some(max_token_id) = self.max_original_token_id() else {
            return Vec::new();
        };

        let mut dense = vec![None; max_token_id as usize + 1];
        for (&token_id, bytes) in self.token_bytes.iter() {
            dense[token_id as usize] = Some(bytes.clone().into_boxed_slice());
        }
        dense
    }

    fn compute_fast_transitions(&self) -> FastDwaTransitions {
        let build = |state: &crate::automata::weighted_u32::dwa::DWAState| {
                state
                    .transitions
                    .iter()
                    .map(|(&label, (target, weight))| (label, (*target, weight.clone())))
                    .collect()
            };
        if rayon::current_num_threads() == 1 {
            self.parser_dwa.states().iter().map(build).collect()
        } else {
            self.parser_dwa.states().par_iter().map(build).collect()
        }
    }

    fn compute_dense_token_masks(&self) -> (usize, DenseWeightMaskCache) {
        let internal_token_dense_words = self.internal_token_to_tokens.len().div_ceil(64);
        if internal_token_dense_words == 0 {
            return (0, DenseWeightMaskCache::default());
        }

        let mut unique_sets: FxHashMap<usize, &RangeSetBlaze<u32>> = FxHashMap::default();
        for state in self.parser_dwa.states() {
            if let Some(final_weight) = &state.final_weight {
                Self::collect_weight_token_sets(final_weight, &mut unique_sets);
            }
            for (_, weight) in state.transitions.values() {
                Self::collect_weight_token_sets(weight, &mut unique_sets);
            }
        }

        let build = |(key, token_set): (usize, &RangeSetBlaze<u32>)| {
                (
                    key,
                    Self::dense_words_from_internal_set_with_words(token_set, internal_token_dense_words),
                )
            };
        let dense_masks: DenseWeightMaskCache = if rayon::current_num_threads() == 1 {
            unique_sets.into_iter().map(build).collect()
        } else {
            unique_sets.into_par_iter().map(build).collect()
        };

        (internal_token_dense_words, dense_masks)
    }

    /// Build precomputed bitmask fragments for each internal token.
    pub(crate) fn build_buf_masks(&mut self) {
        self.internal_token_buf_masks = self.compute_buf_masks();
        self.word_group_buf_masks = Vec::new();
        let (word_group_sparse_masks, word_group_sparse_total_entries, word_group_sparse_max_entries) =
            self.compute_token_block_sparse_masks(64);
        let (byte_group_sparse_masks, _, _) = self.compute_token_block_sparse_masks(8);
        self.word_group_sparse_masks = word_group_sparse_masks;
        self.byte_group_sparse_masks = byte_group_sparse_masks;
        self.word_group_sparse_total_entries = word_group_sparse_total_entries;
        self.word_group_sparse_max_entries = word_group_sparse_max_entries;
        self.pair_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(2);
        self.quad_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(4);
        self.super_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(8);
        self.mega_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(16);
        self.giga_word_group_buf_masks = self.compute_sliding_word_group_dense_masks(32);
        self.all_tokens_buf_mask = self.compute_all_tokens_buf_mask();
    }

    pub(crate) fn build_dense_token_bytes(&mut self) {
        self.token_bytes_dense = self.compute_dense_token_bytes();
    }

    /// Build fast transition lookup tables from the DWA's BTreeMap transitions.
    pub(crate) fn build_fast_transitions(&mut self) {
        self.dwa_fast_transitions = self.compute_fast_transitions();
    }

    pub(crate) fn build_dense_token_masks(&mut self) {
        let (internal_token_dense_words, dense_masks) = self.compute_dense_token_masks();
        self.internal_token_dense_words = internal_token_dense_words;
        self.weight_token_dense_masks = dense_masks;
    }

    /// Precompute dense bitmaps for the seed phase: one bitmap per (state, terminal)
    /// pair, plus the universe bitmap. This lets seed_weight_dense use bitwise ANDNOT
    /// instead of RangeSetBlaze subtraction.
    pub(crate) fn build_seed_dense_masks(&mut self) {
        let dw = self.internal_token_dense_words;
        if dw == 0 {
            self.seed_terminal_dense.clear();
            self.seed_state_dense.clear();
            self.seed_universe_dense = empty_dense_words();
            return;
        }

        if self.tokenizer_fast_transitions.len() != self.tokenizer.num_states() as usize {
            self.tokenizer_fast_transitions = self.compute_tokenizer_fast_transitions();
        }

        let universe = self.internal_token_universe();
        self.seed_universe_dense = self.dense_words_from_internal_set(&universe);

        self.seed_terminal_dense = self.build_seed_terminal_dense_masks();
        self.seed_state_dense = self.build_seed_state_dense_masks();
    }

    fn collect_weight_token_sets<'a>(
        weight: &'a Weight,
        unique_sets: &mut FxHashMap<usize, &'a RangeSetBlaze<u32>>,
    ) {
        for token_set in weight.unique_token_sets() {
            let key = token_set as *const RangeSetBlaze<u32> as usize;
            unique_sets.entry(key).or_insert(token_set);
        }
    }

    fn dense_words_from_internal_set_with_words(
        internal_tokens: &RangeSetBlaze<u32>,
        dense_word_count: usize,
    ) -> DenseWords {
        let mut words = vec![0u64; dense_word_count];
        for internal_token in internal_tokens.iter() {
            let index = internal_token as usize / 64;
            let bit = internal_token as usize % 64;
            if let Some(word) = words.get_mut(index) {
                *word |= 1u64 << bit;
            }
        }
        Arc::from(words.into_boxed_slice())
    }

    fn dense_words_from_internal_set(&self, internal_tokens: &RangeSetBlaze<u32>) -> DenseWords {
        Self::dense_words_from_internal_set_with_words(internal_tokens, self.internal_token_dense_words)
    }

    pub fn start(&self) -> ConstraintState<'_> {
        let state = self.initial_state_map();
        ConstraintState {
            constraint: self,
            state,
            buffers: Default::default(),
            generation: 0,
            mask_cache: Mutex::new(None),
            mask_scratch: Mutex::new(Default::default()),
        }
    }

    pub fn mask_len(&self) -> usize {
        self.token_bytes
            .keys()
            .max()
            .map(|token_id| (*token_id as usize / 32) + 1)
            .unwrap_or(0)
    }

    pub fn num_parser_states(&self) -> u32 {
        self.table.num_states
    }

    pub(crate) fn parser_dwa(&self) -> &DWA {
        &self.parser_dwa
    }

    pub(crate) fn possible_matches_for_state(
        &self,
        tokenizer_state: u32,
    ) -> BTreeMap<TerminalID, RangeSetBlaze<u32>> {
        // possible_matches weights are finalized into the same shared TSID and
        // token spaces as parser-DWA weights. Original-state keyed views used
        // by TerminalsDisallowed are rebuilt separately during seed-mask
        // precomputation.
        let internal_tsid = self.internal_tsid_for_state(tokenizer_state);
        self.possible_matches
            .iter()
            .filter_map(|(&terminal, weight)| {
                let tokens = weight.tokens_for_tsid(internal_tsid);
                if tokens.is_empty() {
                    None
                } else {
                    Some((terminal, self.expand_internal_token_set(&tokens)))
                }
            })
            .collect()
    }

    pub(crate) fn internal_tsid_for_state(&self, tokenizer_state: u32) -> u32 {
        self.state_to_internal_tsid
            .get(tokenizer_state as usize)
            .copied()
            .unwrap_or(tokenizer_state)
    }

    pub(crate) fn internal_token_for_original(&self, token_id: u32) -> u32 {
        self.original_token_to_internal
            .get(token_id as usize)
            .copied()
            .filter(|internal_id| *internal_id != u32::MAX)
            .unwrap_or(token_id)
    }

    fn final_internal_token_for_original(&self, token_id: u32) -> Option<u32> {
        let internal = *self.original_token_to_internal.get(token_id as usize)?;

        if internal == u32::MAX {
            return None;
        }

        if !self.internal_token_to_tokens.is_empty()
            && internal as usize >= self.internal_token_to_tokens.len()
        {
            return None;
        }

        Some(internal)
    }

    pub(crate) fn internal_token_universe(&self) -> RangeSetBlaze<u32> {
        if self.internal_token_to_tokens.is_empty() {
            let Some(max_token_id) = self.max_original_token_id() else {
                return RangeSetBlaze::new();
            };
            return RangeSetBlaze::from_iter([0..=max_token_id]);
        }

        RangeSetBlaze::from_iter([0..=self.internal_token_to_tokens.len().saturating_sub(1) as u32])
    }

    pub(crate) fn expand_internal_token_set(
        &self,
        internal_tokens: &RangeSetBlaze<u32>,
    ) -> RangeSetBlaze<u32> {
        if self.internal_token_to_tokens.is_empty() {
            return internal_tokens.clone();
        }

        let all_ids = self.collect_original_token_ids(internal_tokens);
        Self::range_set_from_sorted_ids(&all_ids)
    }

    pub(crate) fn possible_matches_for_state_internal(
        &self,
        tokenizer_state: u32,
    ) -> Option<BTreeMap<TerminalID, RangeSetBlaze<u32>>> {
        // Return possible_matches in the final shared constraint-internal vocab
        // space. These ids match parser-DWA weight token ids after reconciliation.
        let internal_tsid = self.internal_tsid_for_state(tokenizer_state);
        let mut result = BTreeMap::new();
        for (&terminal, weight) in &self.possible_matches {
            let tokens = weight.tokens_for_tsid(internal_tsid);
            if !tokens.is_empty() {
                result.insert(terminal, tokens);
            }
        }
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    fn build_internal_token_buf_mask(originals: &[u32]) -> InternalTokenBufMasks {
        let mut result = Vec::<(u16, u32)>::new();
        let mut current_word = None::<u16>;
        let mut current_mask = 0u32;
        for &original in originals {
            let word = (original / 32) as u16;
            let bit = original % 32;
            match current_word {
                None => {
                    current_word = Some(word);
                    current_mask = 1u32 << bit;
                }
                Some(current) if current == word => {
                    current_mask |= 1u32 << bit;
                }
                Some(current) if current < word => {
                    result.push((current, current_mask));
                    current_word = Some(word);
                    current_mask = 1u32 << bit;
                }
                Some(_) => {
                    return Self::build_internal_token_buf_mask_unsorted(originals);
                }
            }
        }
        if let Some(word) = current_word {
            result.push((word, current_mask));
        }
        result
    }

    fn build_internal_token_buf_mask_unsorted(originals: &[u32]) -> InternalTokenBufMasks {
        let mut word_map = BTreeMap::<u16, u32>::new();
        for &original in originals {
            let word = (original / 32) as u16;
            let bit = original % 32;
            *word_map.entry(word).or_default() |= 1u32 << bit;
        }
        word_map.into_iter().collect()
    }

    fn max_original_token_id(&self) -> Option<u32> {
        self.token_bytes.keys().next_back().copied()
    }

    fn build_seed_terminal_dense_masks(&self) -> SeedTerminalDenseMasks {
        self.possible_matches
            .iter()
            .flat_map(|(&terminal_id, weight)| {
                weight
                    .compact_entries()
                    .unwrap_or_default()
                    .into_iter()
                    .flat_map(move |(start, end, token_set)| {
                        let dense = self.dense_words_from_internal_set(token_set.as_ref());
                        let mut entries = Vec::new();
                        for internal_tsid in start..=end {
                            if let Some(states) = self.internal_tsid_to_states.get(internal_tsid as usize) {
                                for &tokenizer_state in states {
                                    entries.push(((tokenizer_state, terminal_id), dense.clone()));
                                }
                            } else {
                                entries.push(((internal_tsid, terminal_id), dense.clone()));
                            }
                        }
                        entries.into_iter()
                    })
            })
            .collect()
    }

    fn precompute_node_reachable_dense(
        node: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
        dense_words: usize,
        cache: &mut FxHashMap<usize, Vec<u64>>,
    ) {
        let ptr = node as *const _ as usize;
        if cache.contains_key(&ptr) {
            return;
        }
        let mut dense = vec![0u64; dense_words];
        if node.has_token() {
            let id = node.token_id();
            let word = id / 64;
            let bit = id % 64;
            if word < dense_words {
                dense[word] |= 1u64 << bit;
            }
        }
        for (_, child) in node.iter_children() {
            Self::precompute_node_reachable_dense(child, dense_words, cache);
            let child_ptr = child as *const _ as usize;
            let child_dense = cache.get(&child_ptr).unwrap();
            for i in 0..dense_words {
                dense[i] |= child_dense[i];
            }
        }
        cache.insert(ptr, dense);
    }

    fn walk_seed_trie(
        node: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
        state: u32,
        mask: &mut [u64],
        flat_transitions: &[Box<[u32; 256]>],
        terminal_states: &[bool],
        node_reachable_dense: &FxHashMap<usize, Vec<u64>>,
    ) {
        if node.has_token() {
            let id = node.token_id();
            let word = id / 64;
            let bit = id % 64;
            if word < mask.len() {
                mask[word] |= 1u64 << bit;
            }
        }
        for (segment, child) in node.iter_children() {
            let mut current_state = state;
            let mut terminal_hit = false;
            let mut blocked = false;
            for &byte in segment {
                let next = flat_transitions[current_state as usize][byte as usize];
                if next == u32::MAX {
                    blocked = true;
                    break;
                }
                if terminal_states[next as usize] {
                    terminal_hit = true;
                    break;
                }
                current_state = next;
            }
            if blocked {
                continue;
            }
            if terminal_hit {
                let child_ptr = child as *const _ as usize;
                if let Some(child_dense) = node_reachable_dense.get(&child_ptr) {
                    for i in 0..mask.len() {
                        mask[i] |= child_dense[i];
                    }
                }
            } else {
                Self::walk_seed_trie(child, current_state, mask, flat_transitions, terminal_states, node_reachable_dense);
            }
        }
    }

    fn build_seed_state_dense_masks(&self) -> SeedStateDenseMasks {
        let state_count = self.tokenizer.num_states() as usize;
        let dense_words = self.internal_token_dense_words;
        if state_count == 0 || dense_words == 0 {
            return (0..state_count)
                .map(|_| Arc::from(vec![0u64; dense_words].into_boxed_slice()))
                .collect();
        }

        let terminal_state = |state| self.tokenizer.matched_terminals_iter(state).next().is_some();
        let terminal_states: Vec<bool> = if rayon::current_num_threads() == 1 {
            (0..self.tokenizer.num_states()).map(terminal_state).collect()
        } else {
            (0..self.tokenizer.num_states())
                .into_par_iter()
                .map(terminal_state)
                .collect()
        };
        let owned_flat_transitions;
        let flat_transitions: &[Box<[u32; 256]>] = if self.tokenizer_fast_transitions.len() == state_count {
            &self.tokenizer_fast_transitions
        } else {
            owned_flat_transitions = self.compute_tokenizer_fast_transitions();
            &owned_flat_transitions
        };

        // Build trie over internal token byte strings so common prefixes
        // are shared instead of re-simulated for every token.
        let trie = crate::ds::vocab_prefix_tree::VocabPrefixTree::build_owned(
            self.internal_token_bytes
                .iter()
                .map(|(&token_id, bytes)| (token_id as usize, bytes.clone()))
                .collect(),
        );

        // Precompute a dense reachable-token bitmap for every trie node.
        let mut node_reachable_dense: FxHashMap<usize, Vec<u64>> = FxHashMap::default();
        Self::precompute_node_reachable_dense(&trie.root, dense_words, &mut node_reachable_dense);

        let root_ptr = &trie.root as *const _ as usize;
        let all_tokens_dense: DenseWords = Arc::from(
            node_reachable_dense
                .get(&root_ptr)
                .expect("root reachable-token bitmap must be precomputed")
                .clone()
                .into_boxed_slice(),
        );

        let compute_for_state = |start_state: u32| -> DenseWords {
            let start_idx = start_state as usize;
            if terminal_states.get(start_idx).copied().unwrap_or(false) {
                return Arc::clone(&all_tokens_dense);
            }

            let mut mask = vec![0u64; dense_words];
            Self::walk_seed_trie(
                &trie.root,
                start_state,
                &mut mask,
                flat_transitions,
                &terminal_states,
                &node_reachable_dense,
            );
            Arc::from(mask.into_boxed_slice())
        };

        let mapping_covers_states = self.state_to_internal_tsid.len() >= state_count;
        if mapping_covers_states && !self.internal_tsid_to_states.is_empty() {
            let max_mapped_internal = self
                .state_to_internal_tsid
                .iter()
                .take(state_count)
                .filter(|&&internal| internal != u32::MAX)
                .map(|&internal| internal as usize)
                .max()
                .unwrap_or(0);
            let class_count = self.internal_tsid_to_states.len().max(max_mapped_internal + 1);
            let mut representatives: Vec<Option<u32>> = vec![None; class_count];

            for (internal_tsid, states) in self.internal_tsid_to_states.iter().enumerate() {
                if internal_tsid < representatives.len() {
                    representatives[internal_tsid] = states.first().copied();
                }
            }
            for (state, &internal_tsid) in self.state_to_internal_tsid.iter().take(state_count).enumerate() {
                if internal_tsid == u32::MAX {
                    continue;
                }
                let internal_idx = internal_tsid as usize;
                if let Some(slot) = representatives.get_mut(internal_idx) {
                    slot.get_or_insert(state as u32);
                }
            }

            let build_class_mask =
                |representative: &Option<u32>| (*representative).map(|state| compute_for_state(state));
            let class_masks: Vec<Option<DenseWords>> = if rayon::current_num_threads() == 1 {
                representatives.iter().map(build_class_mask).collect()
            } else {
                representatives.par_iter().map(build_class_mask).collect()
            };

            let build_state_mask = |state: usize| {
                    let internal_tsid = self.state_to_internal_tsid[state];
                    if internal_tsid != u32::MAX {
                        if let Some(Some(mask)) = class_masks.get(internal_tsid as usize) {
                            return Arc::clone(mask);
                        }
                    }
                    compute_for_state(state as u32)
                };
            return if rayon::current_num_threads() == 1 {
                (0..state_count).map(build_state_mask).collect()
            } else {
                (0..state_count).into_par_iter().map(build_state_mask).collect()
            };
        }

        // Fallback for older/deserialized constraints without state-class maps.
        if rayon::current_num_threads() == 1 {
            (0..state_count as u32).map(compute_for_state).collect()
        } else {
            (0..state_count as u32)
                .into_par_iter()
                .map(compute_for_state)
                .collect()
        }
    }

    fn or_internal_token_masks_to_buf(&self, internal_token: usize, buf: &mut [u32]) {
        let masks = &self.internal_token_buf_masks[internal_token];
        for &(word_idx, mask) in masks {
            buf[word_idx as usize] |= mask;
        }
    }

    fn sparse_word_group_entries_in(&self, start: usize, len: usize) -> usize {
        self.word_group_sparse_masks[start..start + len]
            .iter()
            .map(Vec::len)
            .sum()
    }

    fn or_full_internal_word_run_to_buf(
        &self,
        mut wi: usize,
        end: usize,
        buf: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        while wi < end {
            let remaining = end - wi;
            let block = if remaining >= 32
                && self
                    .giga_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| dense.len() < self.sparse_word_group_entries_in(wi, 32))
            {
                Some((32, &self.giga_word_group_buf_masks[wi]))
            } else if remaining >= 16
                && self
                    .mega_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| dense.len() < self.sparse_word_group_entries_in(wi, 16))
            {
                Some((16, &self.mega_word_group_buf_masks[wi]))
            } else if remaining >= 8
                && self
                    .super_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| dense.len() < self.sparse_word_group_entries_in(wi, 8))
            {
                Some((8, &self.super_word_group_buf_masks[wi]))
            } else if remaining >= 4
                && self
                    .quad_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| dense.len() < self.sparse_word_group_entries_in(wi, 4))
            {
                Some((4, &self.quad_word_group_buf_masks[wi]))
            } else if remaining >= 2
                && self
                    .pair_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| dense.len() < self.sparse_word_group_entries_in(wi, 2))
            {
                Some((2, &self.pair_word_group_buf_masks[wi]))
            } else {
                None
            };

            if let Some((block_len, dense_mask)) = block {
                stats.normal_full_word_hits += block_len as u64;
                stats.group_or_sparse_entries += dense_mask.len() as u64;
                or_dense_buf(buf, dense_mask);
                wi += block_len;
                continue;
            }

            if let Some(group_mask) = self.word_group_sparse_masks.get(wi) {
                stats.normal_full_word_hits += 1;
                stats.group_or_sparse_entries += group_mask.len() as u64;
                or_sparse_buf_entries(buf, group_mask);
            }
            wi += 1;
        }
    }

    fn andnot_full_internal_word_run_from_buf(
        &self,
        mut wi: usize,
        end: usize,
        buf: &mut [u32],
        stats: &mut DenseToBufProfileStats,
    ) {
        while wi < end {
            let remaining = end - wi;
            let block = if remaining >= 32
                && self
                    .giga_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| dense.len() < self.sparse_word_group_entries_in(wi, 32))
            {
                Some((32, &self.giga_word_group_buf_masks[wi]))
            } else if remaining >= 16
                && self
                    .mega_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| dense.len() < self.sparse_word_group_entries_in(wi, 16))
            {
                Some((16, &self.mega_word_group_buf_masks[wi]))
            } else if remaining >= 8
                && self
                    .super_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| dense.len() < self.sparse_word_group_entries_in(wi, 8))
            {
                Some((8, &self.super_word_group_buf_masks[wi]))
            } else if remaining >= 4
                && self
                    .quad_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| dense.len() < self.sparse_word_group_entries_in(wi, 4))
            {
                Some((4, &self.quad_word_group_buf_masks[wi]))
            } else if remaining >= 2
                && self
                    .pair_word_group_buf_masks
                    .get(wi)
                    .is_some_and(|dense| dense.len() < self.sparse_word_group_entries_in(wi, 2))
            {
                Some((2, &self.pair_word_group_buf_masks[wi]))
            } else {
                None
            };

            if let Some((block_len, dense_mask)) = block {
                stats.complement_full_word_hits += block_len as u64;
                stats.group_andnot_sparse_entries += dense_mask.len() as u64;
                andnot_dense_buf(buf, dense_mask);
                wi += block_len;
                continue;
            }

            if let Some(group_mask) = self.word_group_sparse_masks.get(wi) {
                stats.complement_full_word_hits += 1;
                stats.group_andnot_sparse_entries += group_mask.len() as u64;
                andnot_sparse_buf_entries(buf, group_mask);
            }
            wi += 1;
        }
    }

    /// Convert a merged internal token dense bitmap to the output buffer.
    /// Uses a contiguous flat entry array for cache-friendly sequential access,
    /// with word_group fast paths for fully-set 64-bit words and heavy token
    /// dense masks for tokens with many buf entries.
    pub(crate) fn or_internal_dense_to_buf(
        &self,
        dense: &[u64],
        buf: &mut [u32],
        buf_zeroed: bool,
    ) -> DenseToBufProfileStats {
        let mut stats = DenseToBufProfileStats::default();
        let all_mask = &self.all_tokens_buf_mask;
        let sparse_word_groups = &self.word_group_sparse_masks;
        let offsets = &self.internal_token_buf_offsets;
        let flat = &self.internal_token_buf_flat;
        let n_internal = if offsets.len() > 1 { offsets.len() - 1 } else { 0 };

        if n_internal == 0 || dense.is_empty() {
            return stats;
        }

        // Count set bits to choose path.
        let n_set: usize = dense.iter().map(|w| w.count_ones() as usize).sum();

        // Super-fast path: all internal tokens set → OR all_tokens_buf_mask.
        if n_set >= n_internal && !all_mask.is_empty() {
            if buf_zeroed {
                copy_dense_buf(buf, all_mask);
            } else {
                or_dense_buf(buf, all_mask);
            }
            return stats;
        }

        if n_set == 0 {
            return stats;
        }

        // Count total sparse entries for set vs missing tokens to pick the cheaper path.
        // Heavy tokens use dense masks — count them at dense cost (buf.len() per token).
        // Use precomputed averages + O(n_heavy) heavy check for fast path decision.
        let heavy = &self.heavy_token_dense_masks;
        let buf_len = buf.len();
        let n_missing = n_internal - n_set;

        let n_heavy_set = if !self.heavy_token_indices.is_empty() {
            let mut count = 0usize;
            for &idx in &self.heavy_token_indices {
                if idx < n_internal {
                    let wi = idx / 64;
                    let bit = idx % 64;
                    if wi < dense.len() && (dense[wi] >> bit) & 1 != 0 {
                        count += 1;
                    }
                }
            }
            count
        } else {
            0
        };
        let n_light_set = n_set.saturating_sub(n_heavy_set);
        let n_heavy_missing = self.heavy_token_indices.len().saturating_sub(n_heavy_set);

        let est_set_cost = n_heavy_set * buf_len
            + (n_light_set * self.light_avg_cost_x256) / 256;
        let est_missing_cost = n_heavy_missing * buf_len
            + (n_missing.saturating_sub(n_heavy_missing) * self.light_avg_cost_x256) / 256;
        let copy_overhead = buf_len;

        if !all_mask.is_empty() && est_missing_cost + copy_overhead < est_set_cost {
            stats.complement_path_used = true;
            // Complement-sparse path: start from all_tokens, subtract missing tokens.
            if buf_zeroed {
                copy_dense_buf(buf, all_mask);
            } else {
                or_dense_buf(buf, all_mask);
            }
            let mut wi = 0usize;
            while wi < dense.len() {
                if wi * 64 >= n_internal {
                    break;
                }
                stats.dense_words_visited += 1;
                let w = dense[wi];
                let remaining = n_internal - wi * 64;
                let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                let missing = !w & valid_mask;
                if missing == 0 {
                    wi += 1;
                    continue;
                }
                if missing == valid_mask {
                    let run_start = wi;
                    wi += 1;
                    while wi < dense.len() && wi * 64 < n_internal {
                        let remaining = n_internal - wi * 64;
                        if remaining < 64 || dense[wi] != 0 {
                            break;
                        }
                        stats.dense_words_visited += 1;
                        wi += 1;
                    }
                    self.andnot_full_internal_word_run_from_buf(run_start, wi, buf, &mut stats);
                    continue;
                }
                let mut bits = missing;
                let (full_bytes, full_nibbles, remaining_bits) =
                    count_complement_subgroups(bits, valid_mask);
                stats.complement_full_byte_groups += full_bytes as u64;
                stats.complement_full_nibble_groups += full_nibbles as u64;
                stats.complement_remaining_bits += remaining_bits as u64;
                while bits != 0 {
                    stats.complement_token_iterations += 1;
                    let bit = bits.trailing_zeros() as usize;
                    let internal_token = wi * 64 + bit;
                    if internal_token < heavy.len() {
                        if let Some(ref dense_mask) = heavy[internal_token] {
                            stats.complement_heavy_dense_clears += 1;
                            andnot_dense_buf(buf, dense_mask);
                            bits &= bits - 1;
                            continue;
                        }
                    }
                    let start = offsets[internal_token] as usize;
                    let end = offsets[internal_token + 1] as usize;
                    let span = end.saturating_sub(start) as u64;
                    stats.complement_sparse_entries += span;
                    stats.complement_max_sparse_span = stats.complement_max_sparse_span.max(span);
                    andnot_sparse_buf_entries(buf, &flat[start..end]);
                    bits &= bits - 1;
                }
                wi += 1;
            }
        } else {
            // Normal path: process sparse light tokens and dense heavy tokens.
            let mut wi = 0usize;
            while wi < dense.len() {
                if wi * 64 >= n_internal {
                    break;
                }
                stats.dense_words_visited += 1;
                let w = dense[wi];
                let remaining = n_internal - wi * 64;
                let valid_mask = if remaining >= 64 { !0u64 } else { (1u64 << remaining) - 1 };
                let valid_bits = w & valid_mask;
                if valid_bits == 0 {
                    wi += 1;
                    continue;
                }
                if valid_bits == valid_mask {
                    let run_start = wi;
                    wi += 1;
                    while wi < dense.len() && wi * 64 < n_internal {
                        let remaining = n_internal - wi * 64;
                        if remaining < 64 || dense[wi] != !0u64 {
                            break;
                        }
                        stats.dense_words_visited += 1;
                        wi += 1;
                    }
                    self.or_full_internal_word_run_to_buf(run_start, wi, buf, &mut stats);
                    continue;
                }
                let mut bits = valid_bits;
                while bits != 0 {
                    stats.normal_token_iterations += 1;
                    let bit = bits.trailing_zeros() as usize;
                    let internal_token = wi * 64 + bit;
                    if internal_token < n_internal {
                        if internal_token < heavy.len() {
                            if let Some(ref dense_mask) = heavy[internal_token] {
                                or_dense_buf(buf, dense_mask);
                                bits &= bits - 1;
                                continue;
                            }
                        }
                        let start = offsets[internal_token] as usize;
                        let end = offsets[internal_token + 1] as usize;
                        stats.normal_sparse_entries += end.saturating_sub(start) as u64;
                        or_sparse_buf_entries(buf, &flat[start..end]);
                    }
                    bits &= bits - 1;
                }
                wi += 1;
            }
        }

        stats
    }

    fn or_original_token_to_buf(&self, token_id: u32, buf: &mut [u32]) {
        let word = token_id as usize / 32;
        let bit = token_id as usize % 32;
        if let Some(slot) = buf.get_mut(word) {
            *slot |= 1u32 << bit;
        }
    }

    fn initial_state_map(&self) -> BTreeMap<u32, ParserGSS> {
        let initial_tok_state = self.tokenizer.initial_state();
        let parser_gss = ParserGSS::from_stacks(&[(vec![0u32], TerminalsDisallowed::new())]);
        BTreeMap::from([(initial_tok_state, parser_gss)])
    }

    fn collect_original_token_ids(&self, internal_tokens: &RangeSetBlaze<u32>) -> Vec<u32> {
        let total_estimate: usize = internal_tokens
            .iter()
            .filter_map(|token| self.internal_token_to_tokens.get(token as usize))
            .map(Vec::len)
            .sum();
        let mut all_ids = Vec::with_capacity(total_estimate);
        for internal_token in internal_tokens.iter() {
            if let Some(originals) = self.internal_token_to_tokens.get(internal_token as usize) {
                all_ids.extend_from_slice(originals);
            }
        }
        all_ids.sort_unstable();
        all_ids.dedup();
        all_ids
    }

    fn range_set_from_sorted_ids(ids: &[u32]) -> RangeSetBlaze<u32> {
        let Some((&first, rest)) = ids.split_first() else {
            return RangeSetBlaze::new();
        };

        let mut ranges = Vec::new();
        let mut start = first;
        let mut end = first;
        for &id in rest {
            if id == end + 1 {
                end = id;
            } else {
                ranges.push(start..=end);
                start = id;
                end = id;
            }
        }
        ranges.push(start..=end);
        RangeSetBlaze::from_iter(ranges)
    }

}
