//! L1 terminal DWA: direct 2-state construction for terminals with max path
//! length ≤ 1.

pub(crate) mod max_length;

use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Ranges key with pre-computed hash for O(1) HashMap lookups.
/// The hash is computed in the parallel traversal phase so the sequential
/// interning loop avoids re-hashing large range vectors.
#[derive(Clone)]
struct PreHashedRanges {
    hash: u64,
    ranges: Vec<(u32, u32)>,
}

/// Hash contribution of a single (start, end) range.
#[inline(always)]
fn range_hash_val(s: u32, e: u32) -> u64 {
    let v = (s as u64) | ((e as u64) << 32);
    v.wrapping_mul(0x517cc1b727220a95)
}

impl PreHashedRanges {
    fn new(ranges: Vec<(u32, u32)>) -> Self {
        let mut h: u64 = 0;
        for &(s, e) in &ranges {
            h = h.wrapping_add(range_hash_val(s, e));
        }
        let hash = (ranges.len() as u64).wrapping_add(h);
        Self { hash, ranges }
    }
}

impl PartialEq for PreHashedRanges {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.ranges == other.ranges
    }
}

impl Eq for PreHashedRanges {}

impl Hash for PreHashedRanges {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

/// Lazy range representation: stores references to walk_cache range slices
/// instead of copying. Hash is computed over all referenced ranges using the
/// same commutative scheme as PreHashedRanges, so it matches the hash of the
/// fully-merged range set exactly when no inter-ref adjacency merges occur.
/// For interning, equality is checked via ref identity (ptr + len) — safe
/// because each walk_cache entry's Vec has a unique address and different
/// entry sets always produce different token ID sets.
#[derive(Clone)]
struct LazyRanges<'a> {
    refs: Vec<&'a [(u32, u32)]>,
    hash: u64,
    total_len: usize,
}

impl<'a> LazyRanges<'a> {
    fn new(refs: Vec<&'a [(u32, u32)]>) -> Self {
        // Compute hash over MERGED ranges by streaming through refs.
        // This produces the same hash as hashing the fully materialized
        // merged output, enabling correct interning across different
        // contributing entry sets that merge to the same result.
        let mut h: u64 = 0;
        let mut total_len: usize = 0;
        let mut merged_count: usize = 0;
        let mut current: Option<(u32, u32)> = None;

        for &slice in &refs {
            total_len += slice.len();
            for &(s, e) in slice {
                if let Some((cs, ref mut ce)) = current {
                    if s <= ce.saturating_add(1) {
                        *ce = (*ce).max(e);
                    } else {
                        h = h.wrapping_add(range_hash_val(cs, *ce));
                        merged_count += 1;
                        current = Some((s, e));
                    }
                } else {
                    current = Some((s, e));
                }
            }
        }
        if let Some((s, e)) = current {
            h = h.wrapping_add(range_hash_val(s, e));
            merged_count += 1;
        }
        let hash = (merged_count as u64).wrapping_add(h);
        Self { refs, hash, total_len }
    }

    /// Materialize into merged ranges.
    fn materialize(&self) -> Vec<(u32, u32)> {
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(self.total_len);
        for &slice in &self.refs {
            if let Some((&first, rest)) = slice.split_first() {
                if let Some(last) = merged.last_mut() {
                    if first.0 <= last.1.saturating_add(1) {
                        last.1 = last.1.max(first.1);
                    } else {
                        merged.push(first);
                    }
                } else {
                    merged.push(first);
                }
                merged.extend_from_slice(rest);
            }
        }
        merged
    }
}

impl<'a> PartialEq for LazyRanges<'a> {
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash { return false; }
        // First try fast path: identical ref lists always produce identical output.
        if self.refs.len() == other.refs.len()
            && self.refs.iter().zip(other.refs.iter()).all(|(&a, &b)| {
                std::ptr::eq(a.as_ptr(), b.as_ptr()) && a.len() == b.len()
            })
        {
            return true;
        }
        // Slow path: streaming merged-range comparison.
        self.materialize() == other.materialize()
    }
}

impl<'a> Eq for LazyRanges<'a> {}

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::grammar::flat::TerminalID;
use crate::compiler::stages::compact::{compact_from_env, CompactMode};
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::id_map_and_terminal_dwa::merge::{
    LocalIdMapTerminalDwa, identity_original_to_local_state,
};
use crate::ds::weight::{Weight, shared_rangeset};
use crate::Vocab;

use super::l2p::equivalence_analysis::compat::TokenizerView;
use super::types::{TerminalColoring, TerminalDwaPhaseProfile, compile_profile_enabled, debug_profile_enabled};

/// Maximum L1 equivalence class count before falling back to L2+.
///
/// When the tokenizer DFA has more than this many distinct equivalence classes
/// for the active L1 terminals, the L1 trie traversal becomes more expensive
/// than L2P's NWA-based approach.
pub(crate) const MAX_L1_TSIDS: usize = 50;

/// Quickly count L1 equivalence classes for the given active terminals.
///
/// Used by the partition builder to decide whether L1 should be attempted
/// *before* launching the parallel L1/L2P build, avoiding a wasteful
/// L2P double-build when L1 would be skipped.
pub(crate) fn count_l1_equivalence_classes(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    active_terminals: &[bool],
) -> usize {
    let states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();
    let tokenizer_view = TokenizerView::new_filtered(tokenizer, active_terminals);
    let token_bytes: Vec<&[u8]> = vocab.entries.values().map(|b| b.as_slice()).collect();
    let equiv_mapping = max_length::find_state_equivalence_classes_byte_restricted(
        &tokenizer_view,
        &token_bytes,
        &states,
    );
    let mut seen = rustc_hash::FxHashSet::default();
    for &rep in &equiv_mapping {
        seen.insert(rep);
    }
    seen.len()
}

/// Build an L1 id_map and terminal DWA for the given vocab and terminal set.
///
/// Uses max-length state equivalence and an identity vocab map, then traverses
/// the vocab tree to accumulate `terminal -> Weight` before building the final
/// 2-state DWA directly.
///
/// Returns `None` if the vocab is empty or no terminal matches exist.
/// The caller should pre-check `count_l1_equivalence_classes()` and merge
/// L1 terminals into L2+ when the count exceeds `MAX_L1_TSIDS`.
pub(crate) fn build_l1_id_map_and_terminal_dwa(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    _terminal_coloring: &TerminalColoring,
    _use_terminal_coloring: bool,
    _ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
    flat_trans: &Arc<[u32]>,
) -> Option<LocalIdMapTerminalDwa> {
    if vocab.is_empty() {
        return None;
    }

    let total_started_at = Instant::now();
    let id_map_started_at = Instant::now();
    let (mut id_map, sorted_entries, state_to_rep, id_map_profile) = build_l1_id_map(tokenizer, vocab, active_terminals, flat_trans);
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    let num_terminals = grammar.num_terminals as u32;
    let dwa_started_at = Instant::now();
    let (mut dwa, terminal_profile) = build_l1_terminal_dwa(
        tokenizer,
        sorted_entries,
        &mut id_map,
        &state_to_rep,
        num_terminals,
        active_terminals,
        flat_trans.as_ref(),
    )?;
    let dwa_stats_before_compact = dwa.stats();
    let terminal_build_ms = dwa_started_at.elapsed().as_secs_f64() * 1000.0;

    let profiling = compile_profile_enabled() || debug_profile_enabled();
    let tsids_before_compact = id_map.num_tsids();
    let tokens_before_compact = id_map.num_internal_tokens();

    let compact_started_at = Instant::now();
    let compact_report = compact_from_env(
        &mut dwa,
        &mut id_map,
        "GLRMASK_COMPACT_L1",
        CompactMode::Fast,
        profiling,
    );
    let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
    let dwa_stats_after_compact = dwa.stats();
    let tsids_after_compact = id_map.num_tsids();
    let tokens_after_compact = id_map.num_internal_tokens();
    let compact_tsid_shrink_pct = if tsids_before_compact > 0 {
        (tsids_before_compact as f64 - tsids_after_compact as f64) * 100.0 / tsids_before_compact as f64
    } else {
        0.0
    };
    let compact_vocab_shrink_pct = if tokens_before_compact > 0 {
        (tokens_before_compact as f64 - tokens_after_compact as f64) * 100.0 / tokens_before_compact as f64
    } else {
        0.0
    };

    if profiling {
        let stats_str = if let Some(stats) = compact_report.profile_stats {
            format!(
                " compact_tsids_before={} compact_tsids_after={} compact_tokens_before={} compact_tokens_after={} compact_tsid_shrink_pct={:.2} compact_vocab_shrink_pct={:.2} compact_weight_ranges_before={} compact_weight_ranges_after={} compact_token_ranges_before={} compact_token_ranges_after={}",
                stats.tsids_before, stats.tsids_after,
                stats.tokens_before, stats.tokens_after,
                compact_tsid_shrink_pct, compact_vocab_shrink_pct,
                stats.weight_ranges_before, stats.weight_ranges_after,
                stats.token_ranges_before, stats.token_ranges_after,
            )
        } else {
            format!(
                " compact_tsids_before={} compact_tsids_after={} compact_tokens_before={} compact_tokens_after={} compact_tsid_shrink_pct={:.2} compact_vocab_shrink_pct={:.2}",
                tsids_before_compact, tsids_after_compact,
                tokens_before_compact, tokens_after_compact,
                compact_tsid_shrink_pct, compact_vocab_shrink_pct,
            )
        };
        eprintln!(
            "[glrmask/profile][l1] partition={} vocab_tokens={} tsids={} rep_states={} state_equiv_ms={:.3} token_identity_map_ms={:.3} id_map_ms={:.3} internal_vocab_ms={:.3} vocab_tree_build_ms={:.3} state_seed_ms={:.3} token_set_intern_ms={:.3} tsid_profile_merge_ms={:.3} tsid_profile_merge_before={} tsid_profile_merge_after={} vocab_tree_traversal_ms={:.3} direct_terminal_dwa_ms={:.3} dwa_states={} dwa_transitions={} dwa_transition_pairs={} dwa_interned_ranges_before_compact={} dwa_interned_ranges_after_compact={} terminal_build_ms={:.3} compact_ms={:.3} determinize=none minimize=none prune=none total_ms={:.3}{}",
            partition_label,
            vocab.entries.len(),
            id_map.num_tsids(),
            id_map.tokenizer_states.representative_original_ids.len(),
            id_map_profile.state_equiv_ms,
            id_map_profile.token_identity_map_ms,
            id_map_ms,
            terminal_profile.internal_vocab_ms,
            terminal_profile.vocab_tree_build_ms,
            terminal_profile.state_seed_ms,
            terminal_profile.token_set_intern_ms,
            terminal_profile.tsid_profile_merge_ms,
            terminal_profile.tsid_profile_merge_before,
            terminal_profile.tsid_profile_merge_after,
            terminal_profile.vocab_tree_traversal_ms,
            terminal_profile.direct_terminal_dwa_ms,
            dwa_stats_before_compact.states,
            dwa_stats_before_compact.transitions,
            dwa_stats_before_compact.transition_pairs,
            dwa_stats_before_compact.interned_ranges,
            dwa_stats_after_compact.interned_ranges,
            terminal_build_ms,
            compact_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
            stats_str,
        );
    }

    // Fast iteration: exit after L1 for a specific partition
    if let Ok(exit_label) = std::env::var("GLRMASK_EXIT_AFTER_L1") {
        if exit_label == partition_label {
            eprintln!("[glrmask/debug] EXIT_AFTER_L1={} triggered.", partition_label);
            std::process::exit(0);
        }
    }

    Some(LocalIdMapTerminalDwa {
        id_map,
        dwa,
        original_to_local_state: identity_original_to_local_state(tokenizer.num_states() as usize),
        dropped_original_state_tsid_fallback: None,
        profile: TerminalDwaPhaseProfile {
            id_map_ms,
            terminal_dwa_ms: terminal_build_ms,
            compact_ms,
        },
    })
}

fn build_l1_id_map<'a>(tokenizer: &Tokenizer, vocab: &'a Vocab, active_terminals: &[bool], flat_trans: &Arc<[u32]>) -> (InternalIdMap, Vec<(u32, &'a [u8])>, Vec<u32>, L1IdMapProfile) {
    let states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();

    // Max-length bounded state equivalence: merge DFA states that behave
    // identically when only tokens up to the max vocab token length are
    // considered. Filtering by active_terminals lets us also merge states
    // that differ only by inactive terminal finalizers/futures.
    let state_equiv_started_at = Instant::now();
    let tokenizer_view = TokenizerView::new_filtered_from_flat_trans(flat_trans, tokenizer, active_terminals);
    let view_ms = state_equiv_started_at.elapsed().as_secs_f64() * 1000.0;
    let token_bytes: Vec<&[u8]> = vocab
        .entries
        .values()
        .map(|b| b.as_slice())
        .collect();
    let equiv_mapping = max_length::find_state_equivalence_classes_byte_restricted(
        &tokenizer_view,
        &token_bytes,
        &states,
    );
    let equiv_algo_ms = state_equiv_started_at.elapsed().as_secs_f64() * 1000.0 - view_ms;
    // Build representative → internal_id mapping
    let mut rep_to_internal: FxHashMap<usize, u32> = FxHashMap::default();
    let mut state_original_to_internal = vec![u32::MAX; states.len()];
    let mut state_representatives = Vec::new();
    for (i, &rep) in equiv_mapping.iter().enumerate() {
        let state_id = states[i];
        let internal_id = *rep_to_internal.entry(rep).or_insert_with(|| {
            let id = state_representatives.len() as u32;
            state_representatives.push(rep as u32);
            id
        });
        state_original_to_internal[state_id] = internal_id;
    }

    // Token-based pre-refinement for large DFAs: walk a small sample of tokens
    // through the kstep representatives to further reduce state count before
    // the expensive terminal DWA traversal.
    const L1_TOKEN_REFINE_MIN_REPS: usize = 200;
    const L1_TOKEN_REFINE_DFA_THRESHOLD: usize = 16_000;
    const L1_TOKEN_REFINE_BATCH_SIZE: usize = 200;
    let num_dfa_states = tokenizer.num_states() as usize;
    if state_representatives.len() >= L1_TOKEN_REFINE_MIN_REPS
        && num_dfa_states >= L1_TOKEN_REFINE_DFA_THRESHOLD
    {
        let kstep_reps: Vec<usize> = state_representatives.iter().map(|&s| s as usize).collect();
        let pre_refine_count = kstep_reps.len();
        let token_refine_mapping = super::l2p::equivalence_analysis::state::fast
            ::find_state_equivalence_classes_ex_with_rep_confirmation(
                &tokenizer_view,
                &token_bytes,
                &kstep_reps,
                &[], // skip_groups
                Some(1), // max_batches: single sample batch
                Some(L1_TOKEN_REFINE_BATCH_SIZE),
                Some(true), // early_stop
            );
        // Compose kstep → token_refine mappings
        let mut refined_rep_to_internal: FxHashMap<usize, u32> = FxHashMap::default();
        let mut refined_representatives = Vec::new();
        let mut kstep_internal_to_refined = vec![0u32; kstep_reps.len()];
        for (kstep_idx, &refined_rep) in token_refine_mapping.iter().enumerate() {
            let internal = *refined_rep_to_internal.entry(refined_rep).or_insert_with(|| {
                let id = refined_representatives.len() as u32;
                refined_representatives.push(refined_rep as u32);
                id
            });
            kstep_internal_to_refined[kstep_idx] = internal;
        }
        // Update state_original_to_internal with refined mapping
        for sto in state_original_to_internal.iter_mut() {
            if *sto != u32::MAX {
                *sto = kstep_internal_to_refined[*sto as usize];
            }
        }
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][l1_token_refine] dfa_states={} kstep_reps={} refined_reps={} sample_tokens={} batch_size={}",
                num_dfa_states,
                pre_refine_count,
                refined_representatives.len(),
                token_bytes.len(),
                L1_TOKEN_REFINE_BATCH_SIZE,
            );
        }
        state_representatives = refined_representatives;
    }

    // Build state_to_rep: original_state → representative_state (for trie traversal)
    let mut state_to_rep = vec![0u32; states.len()];
    for (state_id, &internal) in state_original_to_internal.iter().enumerate() {
        if internal != u32::MAX {
            state_to_rep[state_id] = state_representatives[internal as usize];
        }
    }
    let state_equiv_ms = state_equiv_started_at.elapsed().as_secs_f64() * 1000.0;
    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][l1_id_map] state_equiv breakdown: view={:.1}ms equiv_algo={:.1}ms mapping={:.1}ms total={:.1}ms tokens={} dfa_states={} reps={}",
            view_ms,
            equiv_algo_ms,
            state_equiv_ms - view_ms - equiv_algo_ms,
            state_equiv_ms,
            vocab.entries.len(),
            states.len(),
            state_representatives.len(),
        );
    }

    // Sort token IDs first by first byte, then by length, then lexicographically.
    // Keeping first-byte buckets contiguous preserves cheap whole-bucket unions,
    // while length-major order can reduce fragmentation for length-sensitive
    // token sets before the later compact pass.
    let token_identity_started_at = Instant::now();
    let mut token_id_bytes: Vec<(u32, &[u8])> = vocab
        .entries
        .iter()
        .map(|(&id, bytes)| (id, bytes.as_slice()))
        .collect();
    token_id_bytes.sort_unstable_by(|(_, left_bytes), (_, right_bytes)| {
        left_bytes
            .first()
            .cmp(&right_bytes.first())
            .then(left_bytes.len().cmp(&right_bytes.len()))
            .then(left_bytes.cmp(right_bytes))
    });
    let mut token_original_to_internal = vec![u32::MAX; vocab.max_token_id() as usize + 1];
    let token_ids_sorted: Vec<u32> = token_id_bytes
        .iter()
        .enumerate()
        .map(|(internal_id, &(original_id, _))| {
            token_original_to_internal[original_id as usize] = internal_id as u32;
            original_id
        })
        .collect();
    let token_identity_map_ms = token_identity_started_at.elapsed().as_secs_f64() * 1000.0;

    (
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
                state_original_to_internal,
                state_representatives.len() as u32,
                state_representatives,
            ),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_with_representatives(
                token_original_to_internal,
                token_ids_sorted.len() as u32,
                token_ids_sorted,
            ),
        },
        token_id_bytes,
        state_to_rep,
        L1IdMapProfile {
            state_equiv_ms,
            token_identity_map_ms,
        },
    )
}

fn build_l1_terminal_dwa(
    tokenizer: &Tokenizer,
    sorted_entries: Vec<(u32, &[u8])>,
    id_map: &mut InternalIdMap,
    state_to_rep: &[u32],
    num_terminals: u32,
    active_terminals: &[bool],
    flat_trans: &[u32],
) -> Option<(DWA, L1TerminalBuildProfile)> {
    let total_started_at = std::time::Instant::now();
    let internal_vocab_ms = 0.0;

    if sorted_entries.is_empty() {
        return None;
    }

    let vocab_tree_build_ms = 0.0;

    let state_seed_started_at = Instant::now();
    let mut states_to_initial_tsids = FxHashMap::<u32, Vec<u32>>::default();
    for (internal_tsid, representative_state) in id_map.tokenizer_states.iter_representative_ids().enumerate() {
        states_to_initial_tsids
            .entry(representative_state)
            .or_default()
            .push(internal_tsid as u32);
    }
    let state_seed_ms = state_seed_started_at.elapsed().as_secs_f64() * 1000.0;
    let representative_states = &id_map.tokenizer_states.representative_original_ids;
    let state_original_to_internal = &id_map.tokenizer_states.original_to_internal;
    let dead = u32::MAX;
    let dead_internal = u32::MAX;
    // Extra sentinel row at the end: all entries map to dead_internal.
    // This allows branchless dead-state lookups in the batched walk by
    // substituting sentinel_internal for dead_internal before indexing.
    let sentinel_internal = representative_states.len() as u32;
    // Transposed layout: rep_flat_trans[byte * stride + state] instead of
    // [state * 256 + byte]. When walking all targets for the same byte,
    // accesses become sequential instead of stride-256, and each byte
    // column (stride * 4 bytes) fits in L1 cache.
    let stride = representative_states.len() + 1;
    let mut rep_flat_trans = vec![dead_internal; stride * 256];
    for (internal_state, &rep_state) in representative_states.iter().enumerate() {
        let base = rep_state as usize * 256;
        for byte in 0..256usize {
            let next = flat_trans[base + byte];
            rep_flat_trans[byte * stride + internal_state] = if next == dead {
                dead_internal
            } else {
                state_original_to_internal[next as usize]
            };
        }
    }

    // Batch simulation: for each unique start state, simulate all tokens through
    // the DFA and accumulate end_state_rep → (tsid → token_ids).
    // Parallelized across start states using rayon.
    let num_dfa_states = tokenizer.num_states() as usize;

    let traversal_started_at = Instant::now();

    // Parallel traversal: each start_state processed independently.
    // Each (end_rep, tsid) pair is unique across start groups since TSIDs
    // partition deterministically into start groups. We exploit this by using
    // Arc from the start and skipping merging entirely.
    let start_states_list: Vec<(&u32, &Vec<u32>)> = states_to_initial_tsids.iter().collect();
    let mut empty_token_indices = Vec::<usize>::new();
    let mut token_indices_by_first_byte = vec![Vec::<usize>::new(); 256];
    for (internal_token_id, &(_original_id, token_bytes)) in sorted_entries.iter().enumerate() {
        if let Some(&first_byte) = token_bytes.first() {
            token_indices_by_first_byte[first_byte as usize].push(internal_token_id);
        } else {
            empty_token_indices.push(internal_token_id);
        }
    }

    let mut suffixes_by_first_byte = vec![Vec::<&[u8]>::new(); 256];
    let mut suffix_lcps_by_first_byte = vec![Vec::<usize>::new(); 256];
    for first_byte in 0..256 {
        let token_ids = &token_indices_by_first_byte[first_byte];
        if token_ids.is_empty() {
            continue;
        }

        let suffixes = &mut suffixes_by_first_byte[first_byte];
        let lcps = &mut suffix_lcps_by_first_byte[first_byte];
        suffixes.reserve(token_ids.len());
        lcps.reserve(token_ids.len());

        let mut previous_suffix: &[u8] = &[];
        for &internal_token_id in token_ids {
            let suffix_bytes = &sorted_entries[internal_token_id].1[1..];
            lcps.push(common_prefix_len(previous_suffix, suffix_bytes));
            suffixes.push(suffix_bytes);
            previous_suffix = suffix_bytes;
        }
    }

    // Compute suffix_subtree_bytes per first_byte: the set of all bytes
    // appearing in suffixes (bytes[1..]) of tokens starting with that byte.
    // Used for self-loop optimization in the walk cache.
    let mut suffix_subtree_bytes: Vec<[u64; 4]> = vec![[0u64; 4]; 256];
    for &(_original_id, token_bytes) in &sorted_entries {
        if let Some(&first) = token_bytes.first() {
            for &byte in &token_bytes[1..] {
                suffix_subtree_bytes[first as usize][byte as usize >> 6] |= 1u64 << (byte & 63);
            }
        }
    }

    // Precompute suffix_first_bytes per bucket: the set of bytes appearing
    // at position [1] (first suffix byte) of tokens in each first_byte bucket.
    // Also track whether any single-byte tokens exist per bucket.
    // Used for dead-walk elimination and fingerprint dedup.
    let mut suffix_first_bytes_by_bucket: Vec<[u64; 4]> = vec![[0u64; 4]; 256];
    let mut has_empty_suffix_by_bucket = vec![false; 256];
    for &(_original_id, token_bytes) in &sorted_entries {
        if let Some(&first) = token_bytes.first() {
            if token_bytes.len() <= 1 {
                has_empty_suffix_by_bucket[first as usize] = true;
            } else {
                let b = token_bytes[1];
                suffix_first_bytes_by_bucket[first as usize][b as usize >> 6] |= 1u64 << (b & 63);
            }
        }
    }

    let skipped_walks = std::sync::atomic::AtomicUsize::new(0);
    let skipped_tokens = std::sync::atomic::AtomicUsize::new(0);
    let fingerprint_dedup_walks = std::sync::atomic::AtomicUsize::new(0);
    let fingerprint_dedup_eliminated = std::sync::atomic::AtomicUsize::new(0);
    let phase1_transition_steps = std::sync::atomic::AtomicU64::new(0);
    let phase1_successful_tokens = std::sync::atomic::AtomicU64::new(0);
    let phase1_same_end_rep_hits = std::sync::atomic::AtomicU64::new(0);
    let p2_collect_ns = std::sync::atomic::AtomicU64::new(0);
    let p2_build_ns = std::sync::atomic::AtomicU64::new(0);
    let p2_total_ranges = std::sync::atomic::AtomicU64::new(0);
    // Walk cache: compute once per unique (first_byte, target) and cache
    // the raw merged ranges. Self-loop optimization: if the target state
    // has self-loops on all suffix bytes, all tokens end at the target
    // state and the walk can be skipped entirely.
        // Phase 1: Identify unique (first_byte, target_rep) pairs.
        // Equivalent post-byte target states produce identical suffix walks,
        // so canonicalize on the representative to avoid duplicate work.
        let mut unique_targets: FxHashMap<(u8, u32), ()> = FxHashMap::default();
        for (&start_state, _) in &states_to_initial_tsids {
            let start_internal = state_original_to_internal[start_state as usize];
            for (byte, token_ids) in token_indices_by_first_byte.iter().enumerate() {
                if token_ids.is_empty() { continue; }
                let target_internal = rep_flat_trans[byte * stride + start_internal as usize];
                if target_internal != dead_internal {
                    unique_targets
                        .entry((byte as u8, target_internal))
                        .or_default();
                }
            }
        }
        let unique_walk_keys: Vec<(u8, u32)> = unique_targets.into_keys().collect();

        if debug_profile_enabled() {
            // Count walks per first_byte and tokens per first_byte for work distribution.
            let mut walks_per_byte = [0u32; 256];
            for &(byte, _) in &unique_walk_keys {
                walks_per_byte[byte as usize] += 1;
            }
            let mut total_token_iterations = 0u64;
            let mut max_walks_for_byte = 0u32;
            let mut max_tokens_per_byte = 0usize;
            let mut active_bytes = 0u32;
            for byte in 0..256 {
                let w = walks_per_byte[byte];
                let t = token_indices_by_first_byte[byte].len();
                if t > 0 {
                    active_bytes += 1;
                    total_token_iterations += w as u64 * t as u64;
                    max_walks_for_byte = max_walks_for_byte.max(w);
                    max_tokens_per_byte = max_tokens_per_byte.max(t);
                }
            }
            eprintln!(
                "[glrmask/debug][l1_walk_distrib] unique_walks={} active_bytes={} max_walks_per_byte={} max_tokens_per_byte={} total_token_iterations={}",
                unique_walk_keys.len(), active_bytes, max_walks_for_byte, max_tokens_per_byte, total_token_iterations,
            );
        }

        // Precompute self-loop mask per target state.
        let mut self_loop_masks: FxHashMap<u32, [u64; 4]> = FxHashMap::default();
        for &(_, target) in &unique_walk_keys {
            self_loop_masks.entry(target).or_insert_with(|| {
                let mut mask = [0u64; 4];
                for byte in 0..=255u8 {
                    if rep_flat_trans[byte as usize * stride + target as usize] == target {
                        mask[byte as usize >> 6] |= 1u64 << (byte & 63);
                    }
                }
                mask
            });
        }

        // Parallel walk batched by first_byte: all targets for the same byte
        // are walked simultaneously in one pass over the token list.
        // This breaks the serial dependency chain across targets, enabling
        // memory-level parallelism (independent L2 accesses can overlap).
        let walk_cache: FxHashMap<(u8, u32), Vec<(u32, Vec<(u32, u32)>)>> = {
            // Group unique walk keys by first_byte.
            let mut walks_by_byte: FxHashMap<u8, Vec<u32>> = FxHashMap::default();
            for &(byte, target) in &unique_walk_keys {
                walks_by_byte.entry(byte).or_default().push(target);
            }
            let byte_groups: Vec<(u8, Vec<u32>)> = walks_by_byte.into_iter().collect();

            let all_batches: Vec<Vec<((u8, u32), Vec<(u32, Vec<(u32, u32)>)>)>> = byte_groups
                .par_iter()
                .map(|(first_byte, all_targets)| {
                    let byte = *first_byte;
                    let bucket_idx = byte as usize;
                    let token_ids = &token_indices_by_first_byte[bucket_idx];
                    let suffixes = &suffixes_by_first_byte[bucket_idx];
                    let suffix_lcps = &suffix_lcps_by_first_byte[bucket_idx];
                    let subtree = &suffix_subtree_bytes[byte as usize];

                    // Separate self-loop targets from targets that need walking.
                    let mut selfloop_targets: Vec<u32> = Vec::new();
                    let mut walk_targets: Vec<u32> = Vec::new();
                    for &target in all_targets {
                        let mask = &self_loop_masks[&target];
                        let can_skip = (subtree[0] & !mask[0]) == 0
                            && (subtree[1] & !mask[1]) == 0
                            && (subtree[2] & !mask[2]) == 0
                            && (subtree[3] & !mask[3]) == 0;
                        if can_skip {
                            selfloop_targets.push(target);
                        } else {
                            walk_targets.push(target);
                        }
                    }

                    let mut results: Vec<((u8, u32), Vec<(u32, Vec<(u32, u32)>)>)> = Vec::new();

                    // Handle self-loop targets.
                    if !selfloop_targets.is_empty() {
                        skipped_walks.fetch_add(selfloop_targets.len(), std::sync::atomic::Ordering::Relaxed);
                        skipped_tokens.fetch_add(selfloop_targets.len() * token_ids.len(), std::sync::atomic::Ordering::Relaxed);
                        let first = *token_ids.first().unwrap() as u32;
                        let last = *token_ids.last().unwrap() as u32;
                        for &target in &selfloop_targets {
                            let end_rep = representative_states[target as usize];
                            results.push(((byte, target), vec![(end_rep, vec![(first, last)])]));
                        }
                    }

                    if walk_targets.is_empty() {
                        return results;
                    }

                    // Fingerprint dedup: group walk targets by their
                    // first-suffix-byte transition pattern. Two targets that
                    // transition to the same next-state for every first-suffix-byte
                    // produce identical walk results (all subsequent walk steps
                    // proceed from the same state). Targets that are dead on ALL
                    // first-suffix-bytes produce empty results and are eliminated.
                    // This is only valid when no single-byte tokens exist in this
                    // bucket (empty suffix → final state = target state, which
                    // differs between targets).
                    let sfb = &suffix_first_bytes_by_bucket[bucket_idx];
                    let has_empty = has_empty_suffix_by_bucket[bucket_idx];
                    // rep_idx → list of non-representative targets in the same
                    // fingerprint group (excludes the representative itself)
                    let mut dedup_others: Option<Vec<Vec<u32>>> = None;
                    if !has_empty {
                        // Collect unique suffix first bytes for fingerprint keys
                        let mut sfb_list: Vec<u8> = Vec::new();
                        for w in 0..4u8 {
                            let mut bits = sfb[w as usize];
                            while bits != 0 {
                                let offset = bits.trailing_zeros() as u8;
                                sfb_list.push(w * 64 + offset);
                                bits &= bits - 1;
                            }
                        }

                        if !sfb_list.is_empty() {
                            // Compute fingerprint for each target and group
                            let mut fp_groups: FxHashMap<Vec<u32>, Vec<u32>> = FxHashMap::default();
                            for &target in &walk_targets {
                                let fp: Vec<u32> = sfb_list.iter()
                                    .map(|&b| rep_flat_trans[b as usize * stride + target as usize])
                                    .collect();
                                fp_groups.entry(fp).or_default().push(target);
                            }

                            // Separate dead groups (all entries are dead_internal)
                            // from live groups
                            let mut deduped_targets: Vec<u32> = Vec::new();
                            let mut others: Vec<Vec<u32>> = Vec::new();
                            let mut dead_eliminated = 0usize;
                            let mut dup_eliminated = 0usize;

                            for (fp, group) in &fp_groups {
                                let all_dead = fp.iter().all(|&s| s == dead_internal);
                                if all_dead {
                                    // All targets in this group produce empty walk results
                                    dead_eliminated += group.len();
                                    continue;
                                }
                                let rep = group[0];
                                deduped_targets.push(rep);
                                let group_others: Vec<u32> = group[1..].to_vec();
                                dup_eliminated += group_others.len();
                                others.push(group_others);
                            }

                            let total_eliminated = dead_eliminated + dup_eliminated;
                            if total_eliminated > 0 {
                                fingerprint_dedup_walks.fetch_add(walk_targets.len(), std::sync::atomic::Ordering::Relaxed);
                                fingerprint_dedup_eliminated.fetch_add(total_eliminated, std::sync::atomic::Ordering::Relaxed);
                                dedup_others = Some(others);
                                walk_targets = deduped_targets;
                            }
                        }
                    }

                    if walk_targets.is_empty() {
                        return results;
                    }

                    let num_walk = walk_targets.len();
                    let mut local_transition_steps = 0u64;
                    let mut local_successful_tokens = 0u64;

                    // suffix_states: flat [pos * num_walk + target_idx]
                    // Position 0 = initial target states (before any suffix bytes).
                    let mut suffix_states: Vec<u32> = walk_targets.clone();

                    // Per-target run-flush state.
                    let mut run_end_reps: Vec<u32> = vec![u32::MAX; num_walk];
                    let mut run_starts: Vec<u32> = vec![0; num_walk];
                    let mut run_ends: Vec<u32> = vec![0; num_walk];
                    let mut end_rep_maps: Vec<FxHashMap<u32, Vec<(u32, u32)>>> =
                        (0..num_walk).map(|_| FxHashMap::default()).collect();

                    for (bucket_pos, &internal_token_id) in token_ids.iter().enumerate() {
                        let suffix_bytes = suffixes[bucket_pos];
                        let lcp_len = suffix_lcps[bucket_pos];

                        // Truncate all targets to lcp_len + 1 positions.
                        suffix_states.truncate((lcp_len + 1) * num_walk);

                        // Walk remaining suffix bytes with all targets in parallel.
                        for byte_pos in lcp_len..suffix_bytes.len() {
                            let b = suffix_bytes[byte_pos];
                            let base = byte_pos * num_walk;
                            let col_base = b as usize * stride;
                            for t in 0..num_walk {
                                let prev_state = suffix_states[base + t];
                                // Sentinel substitution: dead_internal → sentinel row
                                // (returns dead_internal), avoiding bounds issues.
                                let safe = if prev_state == dead_internal { sentinel_internal } else { prev_state };
                                let next_state = rep_flat_trans[col_base + safe as usize];
                                suffix_states.push(next_state);
                                local_transition_steps += (prev_state != dead_internal) as u64;
                            }
                        }

                        // Record final states for each target.
                        let end_base = suffix_bytes.len() * num_walk;
                        let token_id = internal_token_id as u32;
                        for t in 0..num_walk {
                            let final_state = suffix_states[end_base + t];
                            if final_state != dead_internal {
                                let end_rep = representative_states[final_state as usize];
                                local_successful_tokens += 1;
                                if run_end_reps[t] == end_rep
                                    && run_ends[t].wrapping_add(1) == token_id
                                {
                                    run_ends[t] = token_id;
                                } else {
                                    // Flush previous run for this target.
                                    if run_end_reps[t] != u32::MAX {
                                        end_rep_maps[t]
                                            .entry(run_end_reps[t])
                                            .or_default()
                                            .push((run_starts[t], run_ends[t]));
                                    }
                                    run_end_reps[t] = end_rep;
                                    run_starts[t] = token_id;
                                    run_ends[t] = token_id;
                                }
                            } else {
                                // Dead: flush current run for this target.
                                if run_end_reps[t] != u32::MAX {
                                    end_rep_maps[t]
                                        .entry(run_end_reps[t])
                                        .or_default()
                                        .push((run_starts[t], run_ends[t]));
                                    run_end_reps[t] = u32::MAX;
                                }
                            }
                        }
                    }

                    // Flush remaining runs.
                    for t in 0..num_walk {
                        if run_end_reps[t] != u32::MAX {
                            end_rep_maps[t]
                                .entry(run_end_reps[t])
                                .or_default()
                                .push((run_starts[t], run_ends[t]));
                        }
                    }

                    phase1_transition_steps.fetch_add(local_transition_steps, std::sync::atomic::Ordering::Relaxed);
                    phase1_successful_tokens.fetch_add(local_successful_tokens, std::sync::atomic::Ordering::Relaxed);

                    // Package per-target results.
                    for (t, map) in end_rep_maps.into_iter().enumerate() {
                        let target = walk_targets[t];
                        let entries: Vec<(u32, Vec<(u32, u32)>)> = map
                            .into_iter()
                            .map(|(end_rep, ranges)| {
                                debug_assert!(
                                    ranges.windows(2).all(|w| w[0].1 < w[1].0),
                                    "Phase 1 ranges should be sorted and non-overlapping"
                                );
                                (end_rep, ranges)
                            })
                            .collect();

                        // Expand results to all targets in the same fingerprint
                        // group if fingerprint dedup was applied.
                        if let Some(ref others) = dedup_others {
                            for &other_target in &others[t] {
                                results.push(((byte, other_target), entries.clone()));
                            }
                        }

                        results.push(((byte, target), entries));
                    }

                    results
                })
                .collect();

            let mut cache: FxHashMap<(u8, u32), Vec<(u32, Vec<(u32, u32)>)>> = FxHashMap::default();
            for batch in all_batches {
                for (key, value) in batch {
                    cache.insert(key, value);
                }
            }
            cache
        };

        let phase1_ms = traversal_started_at.elapsed().as_secs_f64() * 1000.0;
        let phase1_wall_ms = phase1_ms;

        // Precompute unique end_reps and dense index for Phase 2.
        // This allows replacing HashMap with a flat array.
        let mut all_end_reps: Vec<u32> = walk_cache.values()
            .flat_map(|results| results.iter().map(|(end_rep, _)| *end_rep))
            .collect();
        // Also include all state_to_rep values for states in start_states_list
        // (needed for empty token handling).
        for (&start_state, _) in &states_to_initial_tsids {
            all_end_reps.push(state_to_rep[start_state as usize]);
        }
        all_end_reps.sort_unstable();
        all_end_reps.dedup();
        let n_end_reps = all_end_reps.len();
        // Dense mapping: end_rep → index in [0..n_end_reps)
        let mut end_rep_to_idx = vec![usize::MAX; num_dfa_states];
        for (i, &rep) in all_end_reps.iter().enumerate() {
            end_rep_to_idx[rep as usize] = i;
        }

        // Build indexed walk_cache: (byte, target) → Vec of (end_rep_idx, &ranges, entry_hash, entry_range_count).
        // entry_hash is precomputed from the ranges so Phase 2 can combine hashes
        // in O(entries) instead of O(ranges).
        let indexed_walk_cache: FxHashMap<(u8, u32), Vec<(usize, &[(u32, u32)], u64, usize)>> = walk_cache
            .iter()
            .map(|(&key, results)| {
                let indexed: Vec<(usize, &[(u32, u32)], u64, usize)> = results
                    .iter()
                    .map(|(end_rep, ranges)| {
                        let mut h: u64 = 0;
                        for &(s, e) in ranges.as_slice() {
                            h = h.wrapping_add(range_hash_val(s, e));
                        }
                        let entry_hash = (ranges.len() as u64).wrapping_add(h);
                        (end_rep_to_idx[*end_rep as usize], ranges.as_slice(), entry_hash, ranges.len())
                    })
                    .collect();
                (key, indexed)
            })
            .collect();

        if debug_profile_enabled() {
            eprintln!(
                "[glrmask/debug][l1_phase2_setup] unique_end_reps={} walk_cache_entries={}",
                n_end_reps, walk_cache.len()
            );
        }

        // Phase 2: For each start_state, collect walk_cache references per
        // end_rep and build LazyRanges. Instead of copying 25M+ ranges into
        // per-start-state Vecs, store references to walk_cache range slices.
        // Materialization deferred to interning (only ~463 unique sets).

        // Pre-build empty token ranges (shared across all start_states).
        let empty_token_ranges: Vec<(u32, u32)> = {
            let mut ranges = Vec::new();
            for &internal_token_id in &empty_token_indices {
                append_token_id_range(&mut ranges, internal_token_id as u32);
            }
            ranges
        };
        // Precompute hash for empty token ranges.
        let empty_token_hash: u64 = {
            let mut h: u64 = 0;
            for &(s, e) in &empty_token_ranges {
                h = h.wrapping_add(range_hash_val(s, e));
            }
            (empty_token_ranges.len() as u64).wrapping_add(h)
        };

        let per_thread_results: Vec<Vec<(u32, u32, LazyRanges<'_>)>> = start_states_list
            .par_iter()
            .map(|&(&start_state, ref initial_tsids)| {
                let collect_start = Instant::now();
                let start_internal = state_original_to_internal[start_state as usize];

                // Track only touched end_reps for this start state instead of
                // allocating n_end_reps buckets every time.
                let mut touched_positions: FxHashMap<usize, usize> = FxHashMap::default();
                let mut touched_end_reps: Vec<(usize, Vec<&[(u32, u32)]>, u64, usize)> = Vec::new();

                // Empty tokens: end_rep = start_rep
                if !empty_token_ranges.is_empty() {
                    let start_rep = state_to_rep[start_state as usize];
                    let start_rep_idx = end_rep_to_idx[start_rep as usize];
                    let position = if let Some(&position) = touched_positions.get(&start_rep_idx) {
                        position
                    } else {
                        let position = touched_end_reps.len();
                        touched_positions.insert(start_rep_idx, position);
                        touched_end_reps.push((start_rep_idx, Vec::new(), 0, 0));
                        position
                    };
                    let (_, refs, hash_accum, len_accum) = &mut touched_end_reps[position];
                    refs.push(empty_token_ranges.as_slice());
                    *hash_accum = hash_accum.wrapping_add(empty_token_hash);
                    *len_accum += empty_token_ranges.len();
                }

                for (byte, token_ids) in token_indices_by_first_byte.iter().enumerate() {
                    if token_ids.is_empty() { continue; }
                    let target_internal = rep_flat_trans[byte * stride + start_internal as usize];
                    if target_internal == dead_internal { continue; }
                    if let Some(results) = indexed_walk_cache.get(&(byte as u8, target_internal)) {
                        for &(end_rep_idx, ranges, entry_hash, entry_mc) in results {
                            let position = if let Some(&position) = touched_positions.get(&end_rep_idx) {
                                position
                            } else {
                                let position = touched_end_reps.len();
                                touched_positions.insert(end_rep_idx, position);
                                touched_end_reps.push((end_rep_idx, Vec::new(), 0, 0));
                                position
                            };
                            let (_, refs, hash_accum, len_accum) = &mut touched_end_reps[position];
                            refs.push(ranges);
                            *hash_accum = hash_accum.wrapping_add(entry_hash);
                            *len_accum += entry_mc;
                        }
                    }
                }
                p2_collect_ns.fetch_add(collect_start.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);

                // Finalize hashes and build LazyRanges entries.
                let build_start = Instant::now();
                let mut result: Vec<(u32, u32, LazyRanges)> = Vec::new();
                for (idx, refs, hash, total_len) in touched_end_reps.into_iter() {
                    p2_total_ranges.fetch_add(total_len as u64, std::sync::atomic::Ordering::Relaxed);
                    let end_rep = all_end_reps[idx];
                    let lazy = LazyRanges { refs, hash, total_len };
                    if initial_tsids.len() > 1 {
                        for &tsid in &initial_tsids[..initial_tsids.len() - 1] {
                            result.push((end_rep, tsid, lazy.clone()));
                        }
                    }
                    result.push((end_rep, *initial_tsids.last().unwrap(), lazy));
                }
                p2_build_ns.fetch_add(build_start.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                result
            })
            .collect();

    let phase2_wall_ms = traversal_started_at.elapsed().as_secs_f64() * 1000.0 - phase1_wall_ms;

    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][l1_phase2_detail] collect_thread_ms={:.1} build_thread_ms={:.1} total_ranges={} wall_ms={:.1}",
            p2_collect_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1_000_000.0,
            p2_build_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1_000_000.0,
            p2_total_ranges.load(std::sync::atomic::Ordering::Relaxed),
            phase2_wall_ms,
        );
    }

    // Sort-based intern: sort entries by hash, find hash-group boundaries,
    // then verify and build Arcs in parallel. LazyRanges are compared by
    // ref identity (fast pointer comparison) and materialized only for
    // unique groups.
    let token_set_intern_started_at = Instant::now();

    // Flatten all thread results into a single Vec.
    let mut all_entries: Vec<(u32, u32, LazyRanges<'_>)> =
        per_thread_results.into_iter().flatten().collect();

    // Sort by hash (fast u64 comparison). Equal hashes → same group candidate.
    all_entries.sort_unstable_by_key(|entry| entry.2.hash);

    // Find hash-group boundaries (sequential, fast — u64 comparison only).
    let mut hash_group_starts: Vec<usize> = vec![0];
    for k in 1..all_entries.len() {
        if all_entries[k].2.hash != all_entries[k - 1].2.hash {
            hash_group_starts.push(k);
        }
    }
    hash_group_starts.push(all_entries.len());

    // Process hash groups in parallel. Within each group, sub-group by
    // range equality. Cache the representative's materialization to avoid
    // re-materializing it for every comparison.
    let group_results: Vec<Vec<(usize, u32, Arc<RangeSetBlaze<u32>>)>> = hash_group_starts
        .par_windows(2)
        .map(|w| {
            let start = w[0];
            let end = w[1];
            let mut out = Vec::new();
            let mut sub_start = start;
            while sub_start < end {
                // Materialize the sub-group representative once.
                let rep_materialized = all_entries[sub_start].2.materialize();
                let mut sub_end = sub_start + 1;
                while sub_end < end {
                    // Fast path: ref pointer identity.
                    let candidate = &all_entries[sub_end].2;
                    let representative = &all_entries[sub_start].2;
                    let fast_match = candidate.refs.len() == representative.refs.len()
                        && candidate.refs.iter().zip(representative.refs.iter()).all(
                            |(&a, &b)| {
                                std::ptr::eq(a.as_ptr(), b.as_ptr())
                                    && a.len() == b.len()
                            },
                        );
                    if fast_match {
                        sub_end += 1;
                        continue;
                    }
                    // Slow path: materialize candidate, compare with cached rep.
                    if candidate.materialize() == rep_materialized {
                        sub_end += 1;
                    } else {
                        break;
                    }
                }
                let arc: Arc<RangeSetBlaze<u32>> = Arc::new(
                    rep_materialized
                        .iter()
                        .map(|&(s, e)| s..=e)
                        .collect(),
                );
                for k in sub_start..sub_end {
                    out.push((
                        all_entries[k].0 as usize,
                        all_entries[k].1,
                        arc.clone(),
                    ));
                }
                sub_start = sub_end;
            }
            out
        })
        .collect();

    // Merge parallel results into deferred_arced (sequential).
    let mut deferred_arced: Vec<Vec<(u32, Arc<RangeSetBlaze<u32>>)>> =
        vec![Vec::new(); num_dfa_states];
    // Count unique range sets: each distinct Arc pointer = one unique set.
    let unique_range_set_count;
    {
        let mut seen_arcs: rustc_hash::FxHashSet<usize> = rustc_hash::FxHashSet::default();
        for result in &group_results {
            for &(_, _, ref arc) in result {
                seen_arcs.insert(Arc::as_ptr(arc) as usize);
            }
        }
        unique_range_set_count = seen_arcs.len();
    }
    for result in group_results {
        for (end_rep, tsid, arc) in result {
            deferred_arced[end_rep].push((tsid, arc));
        }
    }

    let token_set_intern_ms = token_set_intern_started_at.elapsed().as_secs_f64() * 1000.0;
    let traversal_ms = traversal_started_at.elapsed().as_secs_f64() * 1000.0;
    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][l1_intern_detail] entries={} unique_sets={} hash_groups={} intern_ms={:.1}",
            all_entries.len(), unique_range_set_count, hash_group_starts.len() - 1, token_set_intern_ms,
        );
        let successful_tokens = phase1_successful_tokens.load(std::sync::atomic::Ordering::Relaxed);
        let same_end_rep_hits = phase1_same_end_rep_hits.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "[glrmask/debug][l1_phase1_ops] transition_steps={} successful_tokens={} same_end_rep_hits={} same_end_rep_rate={:.3}",
            phase1_transition_steps.load(std::sync::atomic::Ordering::Relaxed),
            successful_tokens,
            same_end_rep_hits,
            if successful_tokens == 0 {
                0.0
            } else {
                same_end_rep_hits as f64 / successful_tokens as f64
            },
        );
        eprintln!(
            "[glrmask/debug][l1_traversal] start_states={} phase1_walk_ms={:.1} phase2_assembly_ms={:.1} intern_ms={:.1} unique_range_sets={} skipped_walks={} skipped_tokens={} fp_dedup_walks={} fp_dedup_eliminated={} total_vt_ms={:.1}",
            start_states_list.len(), phase1_wall_ms, phase2_wall_ms, token_set_intern_ms,
            unique_range_set_count,
            skipped_walks.load(std::sync::atomic::Ordering::Relaxed),
            skipped_tokens.load(std::sync::atomic::Ordering::Relaxed),
            fingerprint_dedup_walks.load(std::sync::atomic::Ordering::Relaxed),
            fingerprint_dedup_eliminated.load(std::sync::atomic::Ordering::Relaxed),
            traversal_ms,
        );
    }

    let tsid_profile_merge_started_at = Instant::now();
    let tsid_profile_merge_before = id_map.num_tsids() as usize;
    let tsid_profile_merge_report = merge_deferred_equivalent_tsids(id_map, &mut deferred_arced);
    let tsid_profile_merge_after = tsid_profile_merge_report.tsids_after;
    let tsid_profile_merge_ms = tsid_profile_merge_started_at.elapsed().as_secs_f64() * 1000.0;
    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][l1_tsid_profile_merge] before={} after={} unique_arc_token_sets={} unique_range_token_sets={} profile_build_ms={:.3} group_ms={:.3} remap_ms={:.3} total_ms={:.3}",
            tsid_profile_merge_before,
            tsid_profile_merge_after,
            tsid_profile_merge_report.unique_arc_token_sets,
            tsid_profile_merge_report.unique_range_token_sets,
            tsid_profile_merge_report.profile_build_ms,
            tsid_profile_merge_report.group_ms,
            tsid_profile_merge_report.remap_ms,
            tsid_profile_merge_ms,
        );
    }

    let distribute_started_at = Instant::now();
    let arc_wrap_ms = 0.0; // Arc wrapping is now done inside the traversal

    // Build terminal → sorted deduped set of active DFA states (mapped to representatives)
    let inverse_started_at = Instant::now();
    let mut terminal_to_active_states: Vec<Vec<u32>> = vec![Vec::new(); num_terminals as usize];
    for state in 0..num_dfa_states {
        let state_u32 = state as u32;
        let rep = state_to_rep[state];
        for tid in tokenizer.dfa.finalizers(state_u32).iter() {
            if active_terminals.get(tid).copied().unwrap_or(false) {
                terminal_to_active_states[tid].push(rep);
            }
        }
        for tid in tokenizer.tokens_accessible_from_state(state_u32).iter() {
            if active_terminals.get(tid).copied().unwrap_or(false) {
                terminal_to_active_states[tid].push(rep);
            }
        }
    }
    for states in &mut terminal_to_active_states {
        states.sort_unstable();
        states.dedup();
    }
    let inverse_map_ms = inverse_started_at.elapsed().as_secs_f64() * 1000.0;

    // Pre-compute per-TSID full token set unions and contributing end_reps.
    // For each terminal, TSIDs whose contributing end_reps are all active reuse
    // the precomputed Arc; only TSIDs with some inactive end_reps are recomputed.
    let merge_started_at = Instant::now();

    let num_tsids = id_map.num_tsids() as usize;

    // Build per-TSID: full ranges union + contributing end_rep count.
    let mut tsid_full_ranges: Vec<Vec<(u32, u32)>> = (0..num_tsids).map(|_| Vec::new()).collect();
    let mut tsid_total_rep_counts = vec![0usize; num_tsids];
    for entries in &deferred_arced {
        for &(tsid, ref arc) in entries {
            tsid_total_rep_counts[tsid as usize] += 1;
            for r in arc.ranges() {
                tsid_full_ranges[tsid as usize].push((*r.start(), *r.end()));
            }
        }
    }
    let tsid_full_arc_cache: Vec<std::sync::OnceLock<Option<Arc<RangeSetBlaze<u32>>>>> =
        (0..num_tsids).map(|_| std::sync::OnceLock::new()).collect();

    // Group terminals by active_states to deduplicate identical computation
    let mut active_tids: Vec<usize> = (0..terminal_to_active_states.len())
        .filter(|&i| !terminal_to_active_states[i].is_empty())
        .collect();
    active_tids.sort_unstable_by(|&a, &b|
        terminal_to_active_states[a].cmp(&terminal_to_active_states[b]));
    let mut unique_groups: Vec<Vec<usize>> = Vec::new();
    for &tid in &active_tids {
        if let Some(last) = unique_groups.last_mut() {
            if terminal_to_active_states[last[0]] == terminal_to_active_states[tid] {
                last.push(tid); continue;
            }
        }
        unique_groups.push(vec![tid]);
    }

    let num_groups = unique_groups.len();
    let (end_rep_group_masks, words_per_mask) = build_end_rep_group_masks(
        &unique_groups,
        &terminal_to_active_states,
        deferred_arced.len(),
    );
    let mut tsid_group_contributions: Vec<Vec<(usize, Arc<RangeSetBlaze<u32>>)>> =
        (0..num_tsids).map(|_| Vec::new()).collect();
    for (end_rep, entries) in deferred_arced.iter().enumerate() {
        let mask_offset = end_rep * words_per_mask;
        let mask_slice = &end_rep_group_masks[mask_offset..mask_offset + words_per_mask];
        if mask_slice.iter().all(|&w| w == 0) {
            continue;
        }
        for &(tsid, ref arc) in entries {
            tsid_group_contributions[tsid as usize].push((end_rep, Arc::clone(arc)));
        }
    }

    let per_tsid_group_entries: Vec<Vec<(usize, u32, Arc<RangeSetBlaze<u32>>)>> =
        tsid_group_contributions
            .par_iter()
            .enumerate()
            .map(|(tsid, contributions)| {
                if contributions.is_empty() {
                    return Vec::new();
                }

                let mut group_counts = vec![0usize; num_groups];
                let mut group_ranges: Vec<Vec<(u32, u32)>> =
                    (0..num_groups).map(|_| Vec::new()).collect();
                let mut touched_groups: Vec<usize> = Vec::new();

                for &(end_rep, ref arc) in contributions {
                    let mask_offset = end_rep * words_per_mask;
                    let mask_slice = &end_rep_group_masks[mask_offset..mask_offset + words_per_mask];
                    for (word_idx, &word) in mask_slice.iter().enumerate() {
                        let mut remaining = word;
                        while remaining != 0 {
                            let bit_idx = remaining.trailing_zeros() as usize;
                            remaining &= remaining - 1;
                            let group_idx = word_idx * 64 + bit_idx;
                            if group_counts[group_idx] == 0 {
                                touched_groups.push(group_idx);
                            }
                            group_counts[group_idx] += 1;
                            for r in arc.ranges() {
                                group_ranges[group_idx].push((*r.start(), *r.end()));
                            }
                        }
                    }
                }

                touched_groups.sort_unstable();

                let mut out: Vec<(usize, u32, Arc<RangeSetBlaze<u32>>)> = Vec::new();
                out.reserve(touched_groups.len());
                for group_idx in touched_groups {
                    let shared = if group_counts[group_idx] == tsid_total_rep_counts[tsid] {
                        tsid_full_arc_cache[tsid]
                            .get_or_init(|| {
                                shared_rangeset_from_unsorted_pairs(
                                    tsid_full_ranges[tsid].as_slice(),
                                )
                            })
                            .clone()
                    } else {
                        shared_rangeset_from_unsorted_pairs(group_ranges[group_idx].as_slice())
                    };
                    if let Some(tokens) = shared {
                        out.push((group_idx, tsid as u32, tokens));
                    }
                }
                out
            })
            .collect();

    let mut group_weight_entries: Vec<Vec<(u32, Arc<RangeSetBlaze<u32>>)>> =
        (0..unique_groups.len()).map(|_| Vec::new()).collect();
    for entries in per_tsid_group_entries {
        for (group_idx, tsid, tokens) in entries {
            group_weight_entries[group_idx].push((tsid, tokens));
        }
    }
    for entries in &mut group_weight_entries {
        entries.sort_unstable_by_key(|&(tsid, _)| tsid);
    }

    let group_results: Vec<Option<(Vec<usize>, Weight)>> = unique_groups
        .iter()
        .enumerate()
        .map(|(group_idx, tids)| {
            let weight_entries = &group_weight_entries[group_idx];
            if weight_entries.is_empty() {
                return None;
            }
            let weight = Weight::from_per_tsid_shared(
                weight_entries.iter().map(|(t, a)| (*t, Arc::clone(a))));
            if weight.is_empty() { return None; }
            Some((tids.clone(), weight))
        })
        .collect();

    // Sequential DWA construction from grouped results
    let mut dwa = DWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let end_state = dwa.add_state();
    dwa.set_final_weight(end_state, Weight::all());
    let mut num_transitions = 0usize;

    for result in group_results.into_iter().flatten() {
        let (tids, weight) = result;
        for &tid in &tids {
            dwa.add_transition(dwa.start_state(), tid as i32, end_state, weight.clone());
            num_transitions += 1;
        }
    }

    if num_transitions == 0 {
        return None;
    }

    let dwa_stats = dwa.stats();

    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;
    let direct_terminal_dwa_ms = merge_ms;
    let distribute_ms = distribute_started_at.elapsed().as_secs_f64() * 1000.0;
    let vocab_tree_traversal_ms = traversal_ms;

    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][terminal_dwa] partition_build_l1_batch vocab={} tsids={} states={} transitions={} transition_pairs={} interned_ranges={} traversal_ms={:.1} arc_wrap_ms={:.1} inverse_map_ms={:.1} merge_ms={:.1} distribute_ms={:.1} total_ms={:.1}",
            sorted_entries.len(),
            id_map.num_tsids(),
            dwa_stats.states,
            dwa_stats.transitions,
            dwa_stats.transition_pairs,
            dwa_stats.interned_ranges,
            traversal_ms,
            arc_wrap_ms,
            inverse_map_ms,
            merge_ms,
            distribute_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Some((
        dwa,
        L1TerminalBuildProfile {
            internal_vocab_ms,
            vocab_tree_build_ms,
            state_seed_ms,
            token_set_intern_ms,
            tsid_profile_merge_ms,
            tsid_profile_merge_before,
            tsid_profile_merge_after,
            vocab_tree_traversal_ms,
            direct_terminal_dwa_ms,
        },
    ))
}

pub(crate) fn build_flat_transition_table(tokenizer: &Tokenizer) -> Vec<u32> {
    let dead = u32::MAX;
    let mut flat_trans = vec![dead; tokenizer.num_states() as usize * 256];
    for (state_idx, dfa_state) in tokenizer.dfa.states().iter().enumerate() {
        let base = state_idx * 256;
        for (byte, &target) in dfa_state.transitions.iter() {
            flat_trans[base + byte as usize] = target;
        }
    }
    flat_trans
}

fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    let limit = left.len().min(right.len());
    let mut index = 0usize;
    while index < limit && left[index] == right[index] {
        index += 1;
    }
    index
}

fn append_token_id_range(token_ranges: &mut Vec<(u32, u32)>, token_id: u32) {
    append_token_id_span(token_ranges, token_id, token_id);
}

fn append_token_id_span(token_ranges: &mut Vec<(u32, u32)>, start: u32, end: u32) {
    if let Some((_, last_end)) = token_ranges.last_mut() {
        if start <= last_end.saturating_add(1) {
            *last_end = (*last_end).max(end);
            return;
        }
    }
    token_ranges.push((start, end));
}

fn flush_end_rep_run(
    end_rep_token_ranges: &mut FxHashMap<u32, Vec<(u32, u32)>>,
    current_run_end_rep: &mut Option<u32>,
    current_run_start: &mut u32,
    current_run_end: &mut u32,
) {
    if let Some(end_rep) = current_run_end_rep.take() {
        append_token_id_span(
            end_rep_token_ranges.entry(end_rep).or_default(),
            *current_run_start,
            *current_run_end,
        );
    }
}

fn collect_l1_root_ranges_by_first_byte_lcp(
    start_state: u32,
    sorted_entries: &[(u32, &[u8])],
    empty_token_indices: &[usize],
    token_indices_by_first_byte: &[Vec<usize>],
    flat_trans: &[u32],
    state_to_rep: &[u32],
    end_rep_token_ranges: &mut FxHashMap<u32, Vec<(u32, u32)>>,
) {
    let dead = u32::MAX;
    let start_rep = state_to_rep[start_state as usize];
    for &internal_token_id in empty_token_indices {
        append_token_id_range(
            end_rep_token_ranges.entry(start_rep).or_default(),
            internal_token_id as u32,
        );
    }

    for (first_byte, token_ids) in token_indices_by_first_byte.iter().enumerate() {
        if token_ids.is_empty() {
            continue;
        }

        let first_target = flat_trans[start_state as usize * 256 + first_byte];
        if first_target == dead {
            continue;
        }

        let mut previous_suffix: &[u8] = &[];
        let mut suffix_states = vec![first_target];
        for &internal_token_id in token_ids {
            let token_bytes = sorted_entries[internal_token_id].1;
            let suffix_bytes = &token_bytes[1..];
            let lcp_len = common_prefix_len(previous_suffix, suffix_bytes);
            suffix_states.truncate(lcp_len + 1);

            let mut state = *suffix_states.last().unwrap_or(&first_target);
            if state == dead {
                suffix_states.resize(suffix_bytes.len() + 1, dead);
            } else {
                for &byte in &suffix_bytes[lcp_len..] {
                    state = flat_trans[state as usize * 256 + byte as usize];
                    suffix_states.push(state);
                    if state == dead {
                        suffix_states.resize(suffix_bytes.len() + 1, dead);
                        break;
                    }
                }
            }

            let final_state = suffix_states[suffix_bytes.len()];
            if final_state != dead {
                let end_rep = state_to_rep[final_state as usize];
                append_token_id_range(
                    end_rep_token_ranges.entry(end_rep).or_default(),
                    internal_token_id as u32,
                );
            }

            previous_suffix = suffix_bytes;
        }
    }
}

fn merge_ranges_in_place(ranges: &mut Vec<(u32, u32)>) {
    if ranges.is_empty() {
        return;
    }

    ranges.sort_unstable();
    let mut write_index = 0usize;
    for read_index in 1..ranges.len() {
        if ranges[read_index].0 <= ranges[write_index].1.saturating_add(1) {
            ranges[write_index].1 = ranges[write_index].1.max(ranges[read_index].1);
        } else {
            write_index += 1;
            ranges[write_index] = ranges[read_index];
        }
    }
    ranges.truncate(write_index + 1);
}

/// Like merge_ranges_in_place but assumes the input is already sorted.
/// Used in Phase 2 assembly where byte-order iteration guarantees sorted IDs.
fn merge_sorted_ranges_in_place(ranges: &mut Vec<(u32, u32)>) {
    if ranges.len() <= 1 {
        return;
    }
    let mut write_index = 0usize;
    for read_index in 1..ranges.len() {
        if ranges[read_index].0 <= ranges[write_index].1.saturating_add(1) {
            ranges[write_index].1 = ranges[write_index].1.max(ranges[read_index].1);
        } else {
            write_index += 1;
            ranges[write_index] = ranges[read_index];
        }
    }
    ranges.truncate(write_index + 1);
}

fn shared_rangeset_from_unsorted_pairs(
    ranges: &[(u32, u32)],
) -> Option<Arc<RangeSetBlaze<u32>>> {
    if ranges.is_empty() {
        return None;
    }

    let mut merged = ranges.to_vec();
    merge_ranges_in_place(&mut merged);
    Some(shared_rangeset(
        merged.iter().map(|&(start, end)| start..=end).collect(),
    ))
}

fn build_end_rep_group_masks(
    unique_groups: &[Vec<usize>],
    terminal_to_active_states: &[Vec<u32>],
    num_end_reps: usize,
) -> (Vec<u64>, usize) {
    let num_groups = unique_groups.len();
    let words_per_mask = num_groups.div_ceil(64);
    let mut end_rep_group_masks = vec![0u64; num_end_reps * words_per_mask];

    for (group_idx, tids) in unique_groups.iter().enumerate() {
        let word = group_idx / 64;
        let bit = 1u64 << (group_idx % 64);
        for &state in &terminal_to_active_states[tids[0]] {
            end_rep_group_masks[state as usize * words_per_mask + word] |= bit;
        }
    }

    (end_rep_group_masks, words_per_mask)
}

fn merge_deferred_equivalent_tsids(
    id_map: &mut InternalIdMap,
    deferred_arced: &mut [Vec<(u32, Arc<RangeSetBlaze<u32>>)>],
) -> L1TsidProfileMergeReport {
    let num_tsids = id_map.num_tsids() as usize;
    if num_tsids <= 1 {
        return L1TsidProfileMergeReport {
            tsids_after: num_tsids,
            unique_arc_token_sets: 0,
            unique_range_token_sets: 0,
            profile_build_ms: 0.0,
            group_ms: 0.0,
            remap_ms: 0.0,
        };
    }

    let profile_build_started_at = Instant::now();
    let mut profiles = vec![Vec::<(u32, u32)>::new(); num_tsids];
    let mut token_ctx_by_arc = FxHashMap::<usize, u32>::default();
    let mut next_token_ctx = 0u32;
    for (end_rep, entries) in deferred_arced.iter().enumerate() {
        for &(tsid, ref token_set) in entries {
            let arc_ptr = Arc::as_ptr(token_set) as usize;
            let token_ctx = *token_ctx_by_arc.entry(arc_ptr).or_insert_with(|| {
                let ctx = next_token_ctx;
                next_token_ctx += 1;
                ctx
            });
            profiles[tsid as usize].push((end_rep as u32, token_ctx));
        }
    }
    let profile_build_ms = profile_build_started_at.elapsed().as_secs_f64() * 1000.0;

    let group_started_at = Instant::now();
    let mut sorted_tsids: Vec<usize> = (0..num_tsids).collect();
    sorted_tsids.sort_by(|&left, &right| profiles[left].cmp(&profiles[right]));

    let mut tsid_perm = vec![0u32; num_tsids];
    let mut new_count = 1usize;
    tsid_perm[sorted_tsids[0]] = 0;
    for pair in sorted_tsids.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        if profiles[previous] != profiles[current] {
            new_count += 1;
        }
        tsid_perm[current] = (new_count - 1) as u32;
    }
    let group_ms = group_started_at.elapsed().as_secs_f64() * 1000.0;

    if new_count == num_tsids {
        return L1TsidProfileMergeReport {
            tsids_after: num_tsids,
            unique_arc_token_sets: token_ctx_by_arc.len(),
            unique_range_token_sets: token_ctx_by_arc.len(),
            profile_build_ms,
            group_ms,
            remap_ms: 0.0,
        };
    }

    let remap_started_at = Instant::now();
    apply_tsid_perm_to_id_map(&mut id_map.tokenizer_states, &tsid_perm, new_count);
    remap_deferred_arced_tsids(deferred_arced, &tsid_perm);
    let remap_ms = remap_started_at.elapsed().as_secs_f64() * 1000.0;

    L1TsidProfileMergeReport {
        tsids_after: new_count,
        unique_arc_token_sets: token_ctx_by_arc.len(),
        unique_range_token_sets: token_ctx_by_arc.len(),
        profile_build_ms,
        group_ms,
        remap_ms,
    }
}

fn remap_deferred_arced_tsids(
    deferred_arced: &mut [Vec<(u32, Arc<RangeSetBlaze<u32>>)>],
    tsid_perm: &[u32],
) {
    for entries in deferred_arced {
        if entries.is_empty() {
            continue;
        }

        let mut remapped: Vec<(u32, Arc<RangeSetBlaze<u32>>)> = std::mem::take(entries)
            .into_iter()
            .map(|(tsid, token_set)| (tsid_perm[tsid as usize], token_set))
            .collect();
        remapped.sort_unstable_by_key(|(tsid, _)| *tsid);

        let mut merged_entries = Vec::with_capacity(remapped.len());
        let mut idx = 0usize;
        while idx < remapped.len() {
            let tsid = remapped[idx].0;
            let token_set = Arc::clone(&remapped[idx].1);
            idx += 1;
            while idx < remapped.len() && remapped[idx].0 == tsid {
                idx += 1;
            }
            merged_entries.push((tsid, token_set));
        }

        *entries = merged_entries;
    }
}

fn apply_tsid_perm_to_id_map(id_map: &mut ManyToOneIdMap, perm: &[u32], new_count: usize) {
    let old_internal_to_originals = std::mem::take(&mut id_map.internal_to_originals);
    let old_representatives = std::mem::take(&mut id_map.representative_original_ids);

    for internal in &mut id_map.original_to_internal {
        if *internal != u32::MAX {
            *internal = perm[*internal as usize];
        }
    }

    let mut new_internal_to_originals = vec![Vec::new(); new_count];
    let mut new_representatives = vec![u32::MAX; new_count];
    for (old_internal, originals) in old_internal_to_originals.into_iter().enumerate() {
        let new_internal = perm[old_internal] as usize;
        new_internal_to_originals[new_internal].extend(originals);
        if new_representatives[new_internal] == u32::MAX {
            new_representatives[new_internal] = old_representatives[old_internal];
        }
    }

    id_map.internal_to_originals = new_internal_to_originals;
    id_map.representative_original_ids = new_representatives;
}

struct L1IdMapProfile {
    state_equiv_ms: f64,
    token_identity_map_ms: f64,
}

struct L1TsidProfileMergeReport {
    tsids_after: usize,
    unique_arc_token_sets: usize,
    unique_range_token_sets: usize,
    profile_build_ms: f64,
    group_ms: f64,
    remap_ms: f64,
}

struct L1TerminalBuildProfile {
    internal_vocab_ms: f64,
    vocab_tree_build_ms: f64,
    state_seed_ms: f64,
    token_set_intern_ms: f64,
    tsid_profile_merge_ms: f64,
    tsid_profile_merge_before: usize,
    tsid_profile_merge_after: usize,
    vocab_tree_traversal_ms: f64,
    direct_terminal_dwa_ms: f64,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::automata::lexer::ast::{byte, star};
    use crate::compiler::compile::build_tokenizer_from_exprs;

    fn naive_end_rep_sets(
        tokenizer: &Tokenizer,
        sorted_entries: &[(u32, &[u8])],
        start_state: u32,
        state_to_rep: &[u32],
    ) -> BTreeMap<u32, RangeSetBlaze<u32>> {
        let mut out = BTreeMap::new();
        for (internal_token_id, &(_original_id, token_bytes)) in sorted_entries.iter().enumerate() {
            let mut state = start_state;
            let mut blocked = false;
            for &byte in token_bytes {
                let Some(next_state) = tokenizer.step(state, byte) else {
                    blocked = true;
                    break;
                };
                state = next_state;
            }
            if !blocked {
                out.entry(state_to_rep[state as usize])
                    .or_insert_with(RangeSetBlaze::new)
                    .insert(internal_token_id as u32);
            }
        }
        out
    }

    #[test]
    fn test_l1_lexicographic_root_traversal_matches_naive_simulation() {
        let tokenizer = build_tokenizer_from_exprs(&[star(byte(b'a'))]);
        let mut token_entries = vec![
            (10u32, b"a".to_vec()),
            (11u32, b"a".to_vec()),
            (12u32, b"aa".to_vec()),
            (13u32, b"aaa".to_vec()),
            (14u32, b"b".to_vec()),
        ];
        token_entries.sort_unstable_by(|left, right| left.1.cmp(&right.1));
        let sorted_entries: Vec<(u32, &[u8])> = token_entries
            .iter()
            .map(|(token_id, bytes)| (*token_id, bytes.as_slice()))
            .collect();

        let flat_trans = build_flat_transition_table(&tokenizer);
        let state_to_rep: Vec<u32> = (0..tokenizer.num_states()).collect();
        let mut end_rep_ranges = FxHashMap::<u32, Vec<(u32, u32)>>::default();
        let mut empty_token_indices = Vec::<usize>::new();
        let mut token_indices_by_first_byte = vec![Vec::<usize>::new(); 256];
        for (internal_token_id, &(_original_id, token_bytes)) in sorted_entries.iter().enumerate() {
            if let Some(&first_byte) = token_bytes.first() {
                token_indices_by_first_byte[first_byte as usize].push(internal_token_id);
            } else {
                empty_token_indices.push(internal_token_id);
            }
        }

        collect_l1_root_ranges_by_first_byte_lcp(
            tokenizer.initial_state(),
            &sorted_entries,
            &empty_token_indices,
            &token_indices_by_first_byte,
            &flat_trans,
            &state_to_rep,
            &mut end_rep_ranges,
        );

        let actual: BTreeMap<u32, RangeSetBlaze<u32>> = end_rep_ranges
            .into_iter()
            .map(|(end_rep, ranges)| {
                (
                    end_rep,
                    ranges.into_iter().map(|(start, end)| start..=end).collect(),
                )
            })
            .collect();
        let expected = naive_end_rep_sets(
            &tokenizer,
            &sorted_entries,
            tokenizer.initial_state(),
            &state_to_rep,
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_end_rep_group_masks_handle_more_than_32_groups() {
        let terminal_to_active_states: Vec<Vec<u32>> = (0..65).map(|i| vec![i as u32]).collect();
        let unique_groups: Vec<Vec<usize>> = (0..65).map(|i| vec![i]).collect();
        let (end_rep_group_masks, words_per_mask) =
            build_end_rep_group_masks(&unique_groups, &terminal_to_active_states, 65);

        assert_eq!(words_per_mask, 2);
        for group_idx in 0..65 {
            let base = group_idx * words_per_mask;
            let word = group_idx / 64;
            let bit = 1u64 << (group_idx % 64);
            for word_idx in 0..words_per_mask {
                let expected = if word_idx == word { bit } else { 0 };
                assert_eq!(end_rep_group_masks[base + word_idx], expected);
            }
        }
    }
}
