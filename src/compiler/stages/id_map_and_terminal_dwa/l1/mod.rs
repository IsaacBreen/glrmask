//! L1 terminal DWA: direct 2-state construction for terminals with max path
//! length ≤ 1.

use crate::automata::lexer::Lexer;
pub(crate) mod max_length;

use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Exact first-byte target profiles computed for L1 state equivalence.
///
/// The L1 terminal-DWA builder needs the same whole-token walks *and* the same
/// active-terminal signature at each tokenizer state. Reusing both avoids a
/// second full tokenizer-state scan after exact equivalence has already proved
/// the signatures. Profile signature IDs reserve zero for the empty signature;
/// the direct DWA builder numbers non-empty signatures from zero, so the cached
/// direct-builder view stores that shifted numbering explicitly.
type L1WalkProfile = Arc<[(u32, Arc<[(u32, u32)]>)]>;

fn freeze_l1_walk_profile(runs: &[(u32, u32, u32)]) -> L1WalkProfile {
    let mut positions = FxHashMap::<u32, usize>::default();
    let mut grouped = Vec::<(u32, Vec<(u32, u32)>)>::new();
    for &(signature_id, start, end) in runs {
        if signature_id == 0 {
            continue;
        }
        let direct_signature_id = signature_id - 1;
        let position = if let Some(&position) = positions.get(&direct_signature_id) {
            position
        } else {
            let position = grouped.len();
            positions.insert(direct_signature_id, position);
            grouped.push((direct_signature_id, Vec::new()));
            position
        };
        grouped[position].1.push((start, end));
    }
    Arc::from(
        grouped
            .into_iter()
            .map(|(signature_id, ranges)| (signature_id, Arc::from(ranges)))
            .collect::<Vec<_>>(),
    )
}

fn freeze_l1_walk_profile_from_direct(profile: Vec<(u32, Vec<(u32, u32)>)>) -> L1WalkProfile {
    Arc::from(
        profile
            .into_iter()
            .map(|(signature_id, ranges)| (signature_id, Arc::from(ranges)))
            .collect::<Vec<_>>(),
    )
}

#[derive(Debug)]
struct L1ExactProfileReuse {
    target_to_profile_id: FxHashMap<(u8, u32), u32>,
    walk_profiles_by_id: Vec<L1WalkProfile>,
    direct_terminal_signatures: Arc<[Vec<u32>]>,
    direct_state_to_terminal_signature: Arc<[u32]>,
}

impl L1ExactProfileReuse {
    fn materialize_walk_cache(&self) -> FxHashMap<(u8, u32), L1WalkProfile> {
        let profiling = compile_profile_enabled();
        let total_started_at = profiling.then(Instant::now);
        let mut cache = FxHashMap::default();
        for (&target, &profile_id) in &self.target_to_profile_id {
            if profile_id != 0 {
                cache.insert(target, Arc::clone(&self.walk_profiles_by_id[profile_id as usize]));
            }
        }
        if let Some(total_started_at) = total_started_at {
            eprintln!(
                "[glrmask/profile][l1_exact_profile_materialize] profiles={} targets={} profile_build_ms=0.000 target_clone_ms={:.3} total_ms={:.3}",
                self.walk_profiles_by_id.len(),
                self.target_to_profile_id.len(),
                total_started_at.elapsed().as_secs_f64() * 1000.0,
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        cache
    }
}

/// Ranges key with pre-computed hash for O(1) HashMap lookups.
/// The hash is computed in the parallel traversal phase so the sequential
/// interning loop avoids re-hashing large range vectors.
#[derive(Clone)]
struct PreHashedRanges {
    hash: u64,
    ranges: Vec<(u32, u32)>,
}

#[derive(Debug)]
struct L1IdentityVocabOrder {
    token_ids_sorted: Arc<[u32]>,
    token_entries_sorted: Arc<[(u32, Arc<[u8]>)]>,
    original_to_internal: Arc<[u32]>,
    token_buckets: L1SortedTokenBuckets,
}

impl crate::vocab::VocabDerivedArtifact for L1IdentityVocabOrder {}

fn l1_identity_vocab_order(vocab: &Vocab) -> Arc<L1IdentityVocabOrder> {
    if let Some(cached) = vocab.vocab_derived_cache_get::<L1IdentityVocabOrder>() {
        return cached;
    }

    let mut token_entries_sorted: Vec<(u32, Arc<[u8]>)> = vocab
        .entries
        .iter()
        .map(|(&id, bytes)| (id, Arc::<[u8]>::from(bytes.clone().into_boxed_slice())))
        .collect();

    token_entries_sorted.sort_unstable_by(|(_, left_bytes), (_, right_bytes)| {
        left_bytes
            .first()
            .cmp(&right_bytes.first())
            .then(left_bytes.cmp(right_bytes))
    });

    let mut token_original_to_internal = vec![u32::MAX; vocab.max_token_id() as usize + 1];
    let token_ids_sorted: Vec<u32> = token_entries_sorted
        .iter()
        .enumerate()
        .map(|(internal_id, (original_id, _))| {
            token_original_to_internal[*original_id as usize] = internal_id as u32;
            *original_id
        })
        .collect();

    let token_buckets = build_l1_sorted_token_buckets(&token_entries_sorted);
    let order = Arc::new(L1IdentityVocabOrder {
        token_ids_sorted: token_ids_sorted.into(),
        token_entries_sorted: token_entries_sorted.into(),
        original_to_internal: token_original_to_internal.into(),
        token_buckets,
    });
    vocab.vocab_derived_cache_set(Arc::clone(&order));
    order
}

pub(crate) fn prepare_l1_identity_vocab_order(vocab: &Vocab) {
    let _ = l1_identity_vocab_order(vocab);
}

fn skip_max_length_for_partition(partition_label: &str) -> bool {
    if partition_label == "p5" {
        return true;
    }
    static SKIPPED_PARTITIONS: OnceLock<Vec<String>> = OnceLock::new();
    SKIPPED_PARTITIONS
        .get_or_init(|| {
            std::env::var("GLRMASK_SKIP_MAX_LENGTH_PARTITIONS")
                .ok()
                .map(|value| {
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|label| !label.is_empty())
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default()
        })
        .iter()
        .any(|label| label == partition_label)
}

fn skip_l1_max_length_for_partition(partition_label: &str) -> bool {
    if matches!(partition_label, "p4" | "p6") {
        return true;
    }
    static SKIPPED_L1_PARTITIONS: OnceLock<Vec<String>> = OnceLock::new();
    SKIPPED_L1_PARTITIONS
        .get_or_init(|| {
            std::env::var("GLRMASK_SKIP_L1_MAX_LENGTH_PARTITIONS")
                .ok()
                .map(|value| {
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|label| !label.is_empty())
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default()
        })
        .iter()
        .any(|label| label == partition_label)
}

fn l1_max_length_min_states() -> usize {
    static MIN_STATES: OnceLock<usize> = OnceLock::new();
    *MIN_STATES.get_or_init(|| {
        std::env::var("GLRMASK_L1_MAX_LENGTH_MIN_STATES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(128)
    })
}

/// Above this many unprojected states, the bounded prepass can cost more than
/// the exact token-signature pass it is meant to shrink. Skipping it remains
/// exact because L1 then classifies every candidate state directly.
fn l1_max_length_large_state_skip_threshold() -> usize {
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("GLRMASK_L1_MAX_LENGTH_LARGE_STATE_SKIP_THRESHOLD")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(16_384)
    })
}

#[inline]
fn should_skip_max_length_for_partition(
    partition_label: &str,
    initial_state_count: usize,
    projected_by_global: bool,
) -> bool {
    skip_max_length_for_partition(partition_label)
        || skip_l1_max_length_for_partition(partition_label)
        || initial_state_count < l1_max_length_min_states()
        || initial_state_count >= l1_max_length_large_state_skip_threshold()
        || (projected_by_global && initial_state_count <= 8192)
}

fn fast_projected_l1_id_map_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_L1_FAST_PROJECTED_ID_MAP")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true)
    })
}

fn fast_projected_l1_id_map_max_tsids() -> usize {
    static MAX_TSID: OnceLock<usize> = OnceLock::new();
    *MAX_TSID.get_or_init(|| {
        std::env::var("GLRMASK_L1_FAST_PROJECTED_ID_MAP_MAX_TSIDS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(16_384)
    })
}

#[inline]
fn should_use_fast_projected_l1_id_map(
    initial_state_map: Option<&ManyToOneIdMap>,
    num_dfa_states: usize,
) -> bool {
    if !fast_projected_l1_id_map_enabled() {
        return false;
    }
    let Some(map) = initial_state_map else {
        return false;
    };
    let projected_states = map.num_internal_ids() as usize;
    map.original_to_internal.len() == num_dfa_states
        && projected_states < num_dfa_states
        && projected_states <= fast_projected_l1_id_map_max_tsids()
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
        Self {
            refs,
            hash,
            total_len,
        }
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
        if self.hash != other.hash {
            return false;
        }
        // First try fast path: identical ref lists always produce identical output.
        if self.refs.len() == other.refs.len()
            && self
                .refs
                .iter()
                .zip(other.refs.iter())
                .all(|(&a, &b)| std::ptr::eq(a.as_ptr(), b.as_ptr()) && a.len() == b.len())
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
use crate::compiler::stages::mapped_artifact::MappedArtifact;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::id_map_and_terminal_dwa::types::LocalIdMapTerminalDwa;
use crate::ds::weight::{shared_rangeset, Weight};
use crate::grammar::flat::TerminalID;
use crate::Vocab;

use super::l2p::equivalence_analysis::compat::{compute_byte_classes, TokenizerView};
use super::types::{compile_profile_enabled, TerminalColoring, TerminalDwaPhaseProfile};

fn l1_exact_profile_reuse_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_L1_EXACT_PROFILE_REUSE")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true)
    })
}

fn l1_sequential_group_assembly_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_L1_SEQUENTIAL_GROUP_ASSEMBLY")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true)
    })
}

fn compact_l1_terminal_dwa_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_COMPACT_L1_TERMINAL_DWA")
            .map(|value| {
                let trimmed = value.trim();
                !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
            })
            .unwrap_or(true)
    })
}

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
    let mut relevant_bytes = [false; 256];
    for bytes in &token_bytes {
        for &byte in *bytes {
            relevant_bytes[byte as usize] = true;
        }
    }
    let byte_to_class = compute_byte_classes(tokenizer_view.dfa());
    let equiv_mapping = super::l2p::equivalence_analysis::state::max_length::find_state_equivalence_classes_byte_restricted(
        &tokenizer_view,
        &token_bytes,
        &states,
        Some(&byte_to_class),
        Some(active_terminals),
        Some(&relevant_bytes),
    );
    let mut seen = rustc_hash::FxHashSet::default();
    for &rep in &equiv_mapping {
        seen.insert(rep);
    }
    let mut max_length_representatives: Vec<usize> = seen.into_iter().collect();
    max_length_representatives.sort_unstable();

    let order = l1_identity_vocab_order(vocab);
    let flat_trans = build_flat_transition_table(tokenizer);
    let (exact_mapping, _) = find_l1_exact_state_equivalence_by_token_signatures(
        tokenizer,
        order.as_ref(),
        &max_length_representatives,
        active_terminals,
        flat_trans.as_slice(),
        None,
    );
    exact_mapping
        .into_iter()
        .collect::<rustc_hash::FxHashSet<_>>()
        .len()
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
    transitions_by_byte: Option<&[u32]>,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> Option<LocalIdMapTerminalDwa> {
    if vocab.is_empty() {
        return None;
    }

    let total_started_at = Instant::now();
    let id_map_started_at = Instant::now();
    let (mut id_map, vocab_order, _state_to_rep, id_map_profile, exact_profile_reuse) = build_l1_id_map(
        partition_label,
        tokenizer,
        vocab,
        active_terminals,
        flat_trans,
        transitions_by_byte,
        initial_state_map,
    );
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    let num_terminals = grammar.num_terminals as u32;
    let dwa_started_at = Instant::now();
    let (dwa, terminal_profile) = build_l1_terminal_dwa(
        tokenizer,
        vocab_order.as_ref(),
        &mut id_map,
        num_terminals,
        active_terminals,
        flat_trans.as_ref(),
        exact_profile_reuse
            .as_ref()
            .filter(|_| l1_exact_profile_reuse_enabled()),
    )?;
    let dwa_stats_before_compact = dwa.stats();
    let terminal_build_ms = dwa_started_at.elapsed().as_secs_f64() * 1000.0;

    let profiling = compile_profile_enabled();
    let tsids_before_compact = id_map.num_tsids();
    let tokens_before_compact = id_map.num_internal_tokens();

    let mut mapped_dwa = MappedArtifact::new(dwa, id_map);
    let (compact_report, compact_ms) = if compact_l1_terminal_dwa_enabled() {
        let compact_started_at = Instant::now();
        let compact_report = if profiling {
            mapped_dwa.compact_dimensions_fast_l1_with_stats()
        } else {
            mapped_dwa.compact_dimensions_fast_l1()
        };
        let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
        (Some(compact_report), compact_ms)
    } else {
        (None, 0.0)
    };
    let dwa_stats_after_compact = mapped_dwa.artifact().stats();
    let tsids_after_compact = mapped_dwa.id_map().num_tsids();
    let tokens_after_compact = mapped_dwa.id_map().num_internal_tokens();
    let (dwa, id_map) = mapped_dwa.into_parts();
    let compact_tsid_shrink_pct = if tsids_before_compact > 0 {
        (tsids_before_compact as f64 - tsids_after_compact as f64) * 100.0
            / tsids_before_compact as f64
    } else {
        0.0
    };
    let compact_vocab_shrink_pct = if tokens_before_compact > 0 {
        (tokens_before_compact as f64 - tokens_after_compact as f64) * 100.0
            / tokens_before_compact as f64
    } else {
        0.0
    };

    if profiling {
        let stats_str = if let Some(stats) = compact_report.as_ref().and_then(|report| report.profile_stats) {
            format!(
                " compact_tsids_before={} compact_tsids_after={} compact_tokens_before={} compact_tokens_after={} compact_tsid_shrink_pct={:.2} compact_vocab_shrink_pct={:.2} compact_token_ranges_before={} compact_token_ranges_after={}",
                stats.tsids_before, stats.tsids_after,
                stats.tokens_before, stats.tokens_after,
                compact_tsid_shrink_pct, compact_vocab_shrink_pct,
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
            "[glrmask/profile][l1] partition={} vocab_tokens={} tsids={} rep_states={} initial_states_considered={} max_length_skipped={} max_token_len={} token_len_gt_4={} token_len_gt_8={} token_len_gt_16={} token_len_gt_32={} token_len_gt_64={} state_equiv_ms={:.3} max_length_state_equiv_ms={:.3} exact_state_equiv_ms={:.3} max_length_reps={} exact_reps={} token_identity_map_ms={:.3} id_map_ms={:.3} internal_vocab_ms={:.3} vocab_tree_build_ms={:.3} state_seed_ms={:.3} token_set_intern_ms={:.3} tsid_profile_merge_ms={:.3} tsid_profile_merge_before={} tsid_profile_merge_after={} vocab_tree_traversal_ms={:.3} direct_terminal_dwa_ms={:.3} dwa_states={} dwa_transitions={} dwa_transition_pairs={} dwa_interned_ranges_before_compact={} dwa_interned_ranges_after_compact={} terminal_build_ms={:.3} compact_ms={:.3} determinize=none minimize=none prune=none total_ms={:.3}{}",
            partition_label,
            vocab.entries.len(),
            id_map.num_tsids(),
            id_map.tokenizer_states.representative_original_ids.len(),
            id_map_profile.initial_states_considered,
            id_map_profile.max_length_skipped,
            id_map_profile.max_token_len,
            id_map_profile.token_len_gt_4,
            id_map_profile.token_len_gt_8,
            id_map_profile.token_len_gt_16,
            id_map_profile.token_len_gt_32,
            id_map_profile.token_len_gt_64,
            id_map_profile.state_equiv_ms,
            id_map_profile.max_length_state_equiv_ms,
            id_map_profile.exact_state_equiv_ms,
            id_map_profile.max_length_reps,
            id_map_profile.exact_reps,
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

    Some(LocalIdMapTerminalDwa {
        id_map,
        dwa,
        profile: TerminalDwaPhaseProfile {
            id_map_ms,
            terminal_dwa_ms: terminal_build_ms,
            compact_ms,
            ..TerminalDwaPhaseProfile::default()
        },
    })
}

fn build_l1_id_map<'a>(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &'a Vocab,
    active_terminals: &[bool],
    flat_trans: &Arc<[u32]>,
    transitions_by_byte: Option<&[u32]>,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> (
    InternalIdMap,
    Arc<L1IdentityVocabOrder>,
    Vec<u32>,
    L1IdMapProfile,
    Option<L1ExactProfileReuse>,
) {
    let num_dfa_states = tokenizer.num_states() as usize;
    let states: Vec<usize> = match initial_state_map {
        Some(map) => map
            .representative_original_ids
            .iter()
            .map(|&s| s as usize)
            .collect(),
        None => (0..num_dfa_states).collect(),
    };
    let projected_by_global = initial_state_map.is_some() && states.len() < num_dfa_states;

    if should_use_fast_projected_l1_id_map(initial_state_map, num_dfa_states) {
        let token_bytes: Vec<&[u8]> = vocab
            .entries
            .values()
            .map(|bytes| bytes.as_slice())
            .collect();
        let token_len_stats = token_length_stats(&token_bytes);
        let max_token_len = token_bytes
            .iter()
            .map(|bytes| bytes.len())
            .max()
            .unwrap_or(0);
        let (vocab_tokens, vocab_order, token_identity_map_ms) =
            build_l1_identity_vocab_map(vocab);
        let tokenizer_states = initial_state_map
            .expect("checked by should_use_fast_projected_l1_id_map")
            .clone();
        let state_to_rep = state_to_representative_vector(&tokenizer_states, num_dfa_states);
        let exact_reps = tokenizer_states.num_internal_ids() as usize;

        return (
            InternalIdMap {
                tokenizer_states,
                vocab_tokens,
            },
            vocab_order,
            state_to_rep,
            L1IdMapProfile {
                initial_states_considered: states.len(),
                max_length_skipped: true,
                max_token_len,
                token_len_gt_4: token_len_stats.gt_4,
                token_len_gt_8: token_len_stats.gt_8,
                token_len_gt_16: token_len_stats.gt_16,
                token_len_gt_32: token_len_stats.gt_32,
                token_len_gt_64: token_len_stats.gt_64,
                state_equiv_ms: 0.0,
                max_length_state_equiv_ms: 0.0,
                exact_state_equiv_ms: 0.0,
                max_length_reps: exact_reps,
                exact_reps,
                token_identity_map_ms,
            },
            None,
        );
    }

    // Max-length bounded state equivalence: merge DFA states that behave
    // identically when only tokens up to the max vocab token length are
    // considered. Filtering by active_terminals lets us also merge states
    // that differ only by inactive terminal finalizers/futures.
    let order = l1_identity_vocab_order(vocab);
    let token_id_bytes = order.token_entries_sorted.as_ref();
    let token_len_stats = token_length_stats_from_entries(token_id_bytes);
    let max_token_len = token_id_bytes
        .iter()
        .map(|(_, bytes)| bytes.len())
        .max()
        .unwrap_or(0);
    let max_length_skipped =
        should_skip_max_length_for_partition(partition_label, states.len(), projected_by_global);
    let state_equiv_started_at = Instant::now();
    let mut view_ms = 0.0;
    let equiv_mapping = if max_length_skipped {
        states.clone()
    } else {
        let tokenizer_view =
            TokenizerView::new_filtered_from_flat_trans(flat_trans, tokenizer, active_terminals);
        view_ms = state_equiv_started_at.elapsed().as_secs_f64() * 1000.0;
        let token_bytes: Vec<&[u8]> = token_id_bytes
            .iter()
            .map(|(_, bytes)| bytes.as_ref())
            .collect();
        let mut relevant_bytes = [false; 256];
        for bytes in &token_bytes {
            for &byte in *bytes {
                relevant_bytes[byte as usize] = true;
            }
        }
        let byte_to_class = compute_byte_classes(tokenizer_view.dfa());
        super::l2p::equivalence_analysis::state::max_length::find_state_equivalence_classes_byte_restricted(
            &tokenizer_view,
            &token_bytes,
            &states,
            Some(&byte_to_class),
            Some(active_terminals),
            Some(&relevant_bytes),
        )
    };

    // Token IDs are first-byte bucketed and lexicographic within each bucket.
    // This preserves cheap whole-bucket unions and maximizes LCP reuse in the
    // exact suffix-profile walks used by both equivalence and terminal-DWA build.
    let token_sort_ms = 0.0;

    let max_length_ms = if max_length_skipped {
        0.0
    } else {
        state_equiv_started_at.elapsed().as_secs_f64() * 1000.0 - view_ms - token_sort_ms
    };
    let exact_started_at = Instant::now();
    let mut max_length_representatives = equiv_mapping.clone();
    max_length_representatives.sort_unstable();
    max_length_representatives.dedup();
    let (exact_mapping, exact_profile_reuse) = find_l1_exact_state_equivalence_by_token_signatures(
        tokenizer,
        order.as_ref(),
        &max_length_representatives,
        active_terminals,
        flat_trans.as_ref(),
        transitions_by_byte,
    );
    let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
    let mut max_rep_to_exact_rep = FxHashMap::<usize, usize>::default();
    for (&max_rep, &exact_rep) in max_length_representatives.iter().zip(exact_mapping.iter()) {
        max_rep_to_exact_rep.insert(max_rep, exact_rep);
    }

    // Build representative → internal_id mapping, composing through initial_state_map when present
    let mut rep_to_internal: FxHashMap<usize, u32> = FxHashMap::default();
    let mut state_original_to_internal = vec![u32::MAX; num_dfa_states];
    let mut state_representatives = Vec::new();
    for (i, &rep) in equiv_mapping.iter().enumerate() {
        let state_id = states[i];
        let exact_rep = max_rep_to_exact_rep[&rep];
        let internal_id = *rep_to_internal.entry(exact_rep).or_insert_with(|| {
            let id = state_representatives.len() as u32;
            state_representatives.push(exact_rep as u32);
            id
        });
        state_original_to_internal[state_id] = internal_id;
    }

    // When initial_state_map is present, compose: all DFA states map through
    // initial_state_map → max-length equivalence
    if let Some(init_map) = initial_state_map {
        for (orig_state, &init_internal) in init_map.original_to_internal.iter().enumerate() {
            if init_internal == u32::MAX
                || (init_internal as usize) >= init_map.representative_original_ids.len()
            {
                continue;
            }
            let init_rep = init_map.representative_original_ids[init_internal as usize] as usize;
            let final_internal = state_original_to_internal[init_rep];
            if final_internal != u32::MAX {
                state_original_to_internal[orig_state] = final_internal;
            }
        }
    }

    // Keep this exact: the L1 terminal DWA indexes weights by TSID, so every
    // original state in a TSID must have the same whole-token terminal
    // signature for every token. Sampling tokens here is not a proof and has
    // caused order-sensitive mask/commit mismatches.

    // Build state_to_rep: original_state → representative_state (for trie traversal)
    let mut state_to_rep = vec![0u32; num_dfa_states];
    for (state_id, &internal) in state_original_to_internal.iter().enumerate() {
        if internal != u32::MAX {
            state_to_rep[state_id] = state_representatives[internal as usize];
        }
    }
    let state_equiv_ms = state_equiv_started_at.elapsed().as_secs_f64() * 1000.0;

    let token_map_started_at = Instant::now();
    let token_original_to_internal = order.original_to_internal.to_vec();
    let token_ids_sorted = order.token_ids_sorted.to_vec();
    let token_identity_map_ms =
        token_sort_ms + token_map_started_at.elapsed().as_secs_f64() * 1000.0;
    let exact_reps = state_representatives.len();

    (
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
                state_original_to_internal,
                exact_reps as u32,
                state_representatives,
            ),
            vocab_tokens: ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                token_original_to_internal,
                token_ids_sorted,
            ),
        },
        order,
        state_to_rep,
        L1IdMapProfile {
            initial_states_considered: states.len(),
            max_length_skipped,
            max_token_len,
            token_len_gt_4: token_len_stats.gt_4,
            token_len_gt_8: token_len_stats.gt_8,
            token_len_gt_16: token_len_stats.gt_16,
            token_len_gt_32: token_len_stats.gt_32,
            token_len_gt_64: token_len_stats.gt_64,
            state_equiv_ms,
            max_length_state_equiv_ms: max_length_ms,
            exact_state_equiv_ms,
            max_length_reps: max_length_representatives.len(),
            exact_reps,
            token_identity_map_ms,
        },
        exact_profile_reuse,
    )
}

fn build_l1_identity_vocab_map(vocab: &Vocab) -> (ManyToOneIdMap, Arc<L1IdentityVocabOrder>, f64) {
    let token_identity_started_at = Instant::now();
    let order = l1_identity_vocab_order(vocab);
    let token_original_to_internal = order.original_to_internal.to_vec();
    let token_ids_sorted = order.token_ids_sorted.to_vec();

    let token_identity_map_ms = token_identity_started_at.elapsed().as_secs_f64() * 1000.0;
    (
        ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
            token_original_to_internal,
            token_ids_sorted,
        ),
        order,
        token_identity_map_ms,
    )
}

fn state_to_representative_vector(state_map: &ManyToOneIdMap, num_dfa_states: usize) -> Vec<u32> {
    let mut state_to_rep = vec![0u32; num_dfa_states];
    for (state_id, &internal) in state_map.original_to_internal.iter().enumerate() {
        if internal != u32::MAX {
            if let Some(&rep) = state_map.representative_original_ids.get(internal as usize) {
                state_to_rep[state_id] = rep;
            }
        }
    }
    state_to_rep
}

struct TokenLengthStats {
    gt_4: usize,
    gt_8: usize,
    gt_16: usize,
    gt_32: usize,
    gt_64: usize,
}

fn token_length_stats(tokens: &[&[u8]]) -> TokenLengthStats {
    let mut stats = TokenLengthStats {
        gt_4: 0,
        gt_8: 0,
        gt_16: 0,
        gt_32: 0,
        gt_64: 0,
    };
    for token in tokens {
        let len = token.len();
        if len > 4 {
            stats.gt_4 += 1;
        }
        if len > 8 {
            stats.gt_8 += 1;
        }
        if len > 16 {
            stats.gt_16 += 1;
        }
        if len > 32 {
            stats.gt_32 += 1;
        }
        if len > 64 {
            stats.gt_64 += 1;
        }
    }
    stats
}

fn token_length_stats_from_entries(tokens: &[(u32, Arc<[u8]>)]) -> TokenLengthStats {
    let mut stats = TokenLengthStats {
        gt_4: 0,
        gt_8: 0,
        gt_16: 0,
        gt_32: 0,
        gt_64: 0,
    };
    for (_, token) in tokens {
        let len = token.len();
        if len > 4 {
            stats.gt_4 += 1;
        }
        if len > 8 {
            stats.gt_8 += 1;
        }
        if len > 16 {
            stats.gt_16 += 1;
        }
        if len > 32 {
            stats.gt_32 += 1;
        }
        if len > 64 {
            stats.gt_64 += 1;
        }
    }
    stats
}

#[inline]
fn l1_transition(
    flat_trans: &[u32],
    transitions_by_byte: Option<&[u32]>,
    num_tokenizer_states: usize,
    state: u32,
    byte: usize,
) -> u32 {
    if let Some(transitions_by_byte) = transitions_by_byte {
        transitions_by_byte[byte * num_tokenizer_states + state as usize]
    } else {
        flat_trans[state as usize * 256 + byte]
    }
}

fn find_l1_exact_state_equivalence_by_token_signatures(
    tokenizer: &Tokenizer,
    vocab_order: &L1IdentityVocabOrder,
    states: &[usize],
    active_terminals: &[bool],
    flat_trans: &[u32],
    transitions_by_byte: Option<&[u32]>,
) -> (Vec<usize>, Option<L1ExactProfileReuse>) {
    if states.len() <= 1 {
        return (states.to_vec(), None);
    }
    let profile_enabled = compile_profile_enabled();
    let total_started_at = profile_enabled.then(Instant::now);

    // Exact L1 equivalence has a useful factorization.  For a non-empty token
    // b·suffix, the contribution of a start state s depends on s only through
    // delta(s, b); the rest of the token walk is shared by every state with the
    // same first-byte target.  The old implementation re-walked every token for
    // every candidate state.  Here we precompute each distinct
    // (first_byte, first_target) suffix profile once, intern those profiles, and
    // then classify a start state by the small vector of profile IDs reached by
    // its first-byte transitions.
    let sorted_entries = vocab_order.token_entries_sorted.as_ref();
    let terminal_signature_started_at = profile_enabled.then(Instant::now);
    let (state_to_terminal_signature, terminal_signatures) =
        build_l1_state_to_terminal_signatures(tokenizer, active_terminals);
    let terminal_signature_ms = terminal_signature_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    let token_buckets = &vocab_order.token_buckets;
    let dead = u32::MAX;
    let num_tokenizer_states = tokenizer.num_states() as usize;

    let nonempty_first_bytes: Vec<usize> = token_buckets
        .token_indices_by_first_byte
        .iter()
        .enumerate()
        .filter_map(|(byte, token_ids)| (!token_ids.is_empty()).then_some(byte))
        .collect();

    let unique_targets_started_at = profile_enabled.then(Instant::now);
    let mut unique_targets = FxHashSet::<(u8, u32)>::default();
    for &state in states {
        for &byte in &nonempty_first_bytes {
            let target = l1_transition(
                flat_trans,
                transitions_by_byte,
                num_tokenizer_states,
                state as u32,
                byte,
            );
            if target != dead {
                unique_targets.insert((byte as u8, target));
            }
        }
    }

    let mut unique_targets: Vec<(u8, u32)> = unique_targets.into_iter().collect();
    unique_targets.sort_unstable();
    let unique_targets_len = unique_targets.len();
    let unique_targets_ms = unique_targets_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });

    let target_profiles_started_at = profile_enabled.then(Instant::now);
    let mut targets_by_first_byte = vec![Vec::<u32>::new(); 256];
    for (byte, target) in unique_targets {
        targets_by_first_byte[byte as usize].push(target);
    }
    let byte_target_groups: Vec<(u8, Vec<u32>)> = targets_by_first_byte
        .into_iter()
        .enumerate()
        .filter_map(|(byte, targets)| (!targets.is_empty()).then_some((byte as u8, targets)))
        .collect();
    let build_byte_profiles = |(byte, targets): &(u8, Vec<u32>)| {
        let byte_idx = *byte as usize;
        l1_bucket_suffix_signature_profiles_packed(
            *byte,
            targets,
            sorted_entries,
            &token_buckets.token_indices_by_first_byte[byte_idx],
            &token_buckets.suffix_lcps_by_first_byte[byte_idx],
            &token_buckets.suffix_subtree_bytes[byte_idx],
            &token_buckets.suffix_first_bytes_by_bucket[byte_idx],
            token_buckets.has_empty_suffix_by_bucket[byte_idx],
            &state_to_terminal_signature,
            flat_trans,
            transitions_by_byte,
            num_tokenizer_states,
        )
    };
    let target_profile_batches: Vec<Vec<((u8, u32), Arc<[(u32, u32, u32)]>)>> =
        if rayon::current_num_threads() == 1 {
            byte_target_groups.iter().map(build_byte_profiles).collect()
        } else {
            byte_target_groups.par_iter().map(build_byte_profiles).collect()
        };
    let target_profiles: Vec<((u8, u32), Arc<[(u32, u32, u32)]>)> =
        target_profile_batches.into_iter().flatten().collect();
    let target_profiles_ms = target_profiles_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });

    let profile_intern_started_at = profile_enabled.then(Instant::now);
    let empty_profile: Arc<[(u32, u32, u32)]> = Arc::from([]);
    let mut profile_to_id = FxHashMap::<Arc<[(u32, u32, u32)]>, u32>::default();
    profile_to_id.insert(Arc::clone(&empty_profile), 0);
    let mut profiles_by_id = vec![empty_profile];
    let mut next_profile_id = 1u32;
    let mut target_to_profile_id = FxHashMap::<(u8, u32), u32>::default();
    for (target_key, profile) in target_profiles {
        let profile_id = if let Some(&profile_id) = profile_to_id.get(&profile) {
            profile_id
        } else {
            let profile_id = next_profile_id;
            next_profile_id += 1;
            profile_to_id.insert(Arc::clone(&profile), profile_id);
            profiles_by_id.push(profile);
            profile_id
        };
        target_to_profile_id.insert(target_key, profile_id);
    }
    let profile_ids_len = next_profile_id as usize;
    let walk_profiles_by_id: Vec<L1WalkProfile> = profiles_by_id
        .iter()
        .map(|profile| freeze_l1_walk_profile(profile.as_ref()))
        .collect();
    let profile_intern_ms = profile_intern_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });

    // State keys are a fixed-width vector of per-first-byte profile IDs.  The
    // old representation allocated one Vec per tokenizer state and hashed the
    // whole Vec into a map.  Materialize the profile lookup densely once, then
    // use a collision-checked scalar fingerprint for grouping.  Equality is
    // still the full exact profile vector.
    let state_keys_started_at = profile_enabled.then(Instant::now);
    let mut byte_slots = [u16::MAX; 256];
    for (slot, &byte) in nonempty_first_bytes.iter().enumerate() {
        byte_slots[byte] = slot as u16;
    }
    let mut profile_by_first_byte =
        vec![0u32; nonempty_first_bytes.len() * num_tokenizer_states];
    for (&(byte, target), &profile_id) in &target_to_profile_id {
        let slot = byte_slots[byte as usize];
        debug_assert_ne!(slot, u16::MAX);
        profile_by_first_byte[slot as usize * num_tokenizer_states + target as usize] = profile_id;
    }
    let has_empty_tokens = !token_buckets.empty_token_indices.is_empty();
    let hash_state_key = |&state: &usize| {
        let mut hash = 0x9e37_79b9_7f4a_7c15u64;
        if has_empty_tokens {
            hash ^= state_to_terminal_signature[state] as u64;
            hash = hash.rotate_left(17).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        }
        for (slot, &byte) in nonempty_first_bytes.iter().enumerate() {
            let target = l1_transition(
                flat_trans,
                transitions_by_byte,
                num_tokenizer_states,
                state as u32,
                byte,
            );
            let profile_id = if target == dead {
                0
            } else {
                profile_by_first_byte[slot * num_tokenizer_states + target as usize]
            };
            hash ^= (profile_id as u64)
                .wrapping_add(0x9e37_79b9_7f4a_7c15)
                .wrapping_add((slot as u64).wrapping_mul(0x517c_c1b7_2722_0a95));
            hash = hash.rotate_left(17).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        }
        hash
    };
    let state_key_hashes: Vec<u64> = if rayon::current_num_threads() == 1 {
        states.iter().map(hash_state_key).collect()
    } else {
        states.par_iter().map(hash_state_key).collect()
    };
    let state_keys_ms = state_keys_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });

    let group_started_at = profile_enabled.then(Instant::now);
    let same_state_key = |left: usize, right: usize| {
        if has_empty_tokens
            && state_to_terminal_signature[left] != state_to_terminal_signature[right]
        {
            return false;
        }
        for (slot, &byte) in nonempty_first_bytes.iter().enumerate() {
            let left_target = l1_transition(
                flat_trans,
                transitions_by_byte,
                num_tokenizer_states,
                left as u32,
                byte,
            );
            let right_target = l1_transition(
                flat_trans,
                transitions_by_byte,
                num_tokenizer_states,
                right as u32,
                byte,
            );
            let left_profile = if left_target == dead {
                0
            } else {
                profile_by_first_byte[slot * num_tokenizer_states + left_target as usize]
            };
            let right_profile = if right_target == dead {
                0
            } else {
                profile_by_first_byte[slot * num_tokenizer_states + right_target as usize]
            };
            if left_profile != right_profile {
                return false;
            }
        }
        true
    };
    let mut primary_representative_by_hash = FxHashMap::<u64, usize>::default();
    let mut collision_representatives_by_hash = FxHashMap::<u64, Vec<usize>>::default();
    let mut mapping = Vec::<usize>::with_capacity(states.len());
    let mut groups_len = 0usize;
    for (&state, &hash) in states.iter().zip(&state_key_hashes) {
        let representative = if let Some(&primary) = primary_representative_by_hash.get(&hash) {
            if same_state_key(primary, state) {
                primary
            } else {
                let representatives = collision_representatives_by_hash
                    .entry(hash)
                    .or_insert_with(|| vec![primary]);
                if let Some(&existing) = representatives
                    .iter()
                    .find(|&&existing| same_state_key(existing, state))
                {
                    existing
                } else {
                    representatives.push(state);
                    groups_len += 1;
                    state
                }
            }
        } else {
            primary_representative_by_hash.insert(hash, state);
            groups_len += 1;
            state
        };
        mapping.push(representative);
    }
    let group_ms = group_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });

    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][l1_exact_equiv_detail] states={} first_bytes={} unique_targets={} profile_ids={} groups={} terminal_signature_ms={:.3} unique_targets_ms={:.3} target_profiles_ms={:.3} profile_intern_ms={:.3} state_keys_ms={:.3} group_ms={:.3} total_ms={:.3}",
            states.len(),
            nonempty_first_bytes.len(),
            unique_targets_len,
            profile_ids_len,
            groups_len,
            terminal_signature_ms,
            unique_targets_ms,
            target_profiles_ms,
            profile_intern_ms,
            state_keys_ms,
            group_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    (
        mapping,
        Some(L1ExactProfileReuse {
            target_to_profile_id,
            walk_profiles_by_id,
            // Exact-equivalence signature ids reserve zero for empty. The
            // direct DWA builder uses `u32::MAX` for empty and zero-based ids
            // for non-empty signatures, matching `materialize_walk_cache`.
            direct_terminal_signatures: terminal_signatures[1..].to_vec().into(),
            direct_state_to_terminal_signature: state_to_terminal_signature
                .into_iter()
                .map(|signature_id| {
                    if signature_id == 0 {
                        u32::MAX
                    } else {
                        signature_id - 1
                    }
                })
                .collect::<Vec<_>>()
                .into(),
        }),
    )
}

fn l1_target_self_loop_covers_suffix_subtree(
    target: u32,
    suffix_subtree: &[u64; 4],
    flat_trans: &[u32],
) -> bool {
    for word in 0..4usize {
        let mut bits = suffix_subtree[word];
        while bits != 0 {
            let offset = bits.trailing_zeros() as usize;
            let byte = word * 64 + offset;
            if flat_trans[target as usize * 256 + byte] != target {
                return false;
            }
            bits &= bits - 1;
        }
    }
    true
}

fn l1_bucket_suffix_signature_profiles_batched(
    first_byte: u8,
    targets: &[u32],
    sorted_entries: &[(u32, Arc<[u8]>)],
    token_ids: &[usize],
    suffix_lcps: &[usize],
    suffix_subtree: &[u64; 4],
    suffix_first_bytes: &[u64; 4],
    has_empty_suffix: bool,
    state_to_terminal_signature: &[u32],
    flat_trans: &[u32],
) -> Vec<((u8, u32), Vec<(u32, u32, u32)>)> {
    let dead = u32::MAX;
    let mut results = Vec::<((u8, u32), Vec<(u32, u32, u32)>)>::with_capacity(targets.len());
    if token_ids.is_empty() {
        for &target in targets {
            results.push(((first_byte, target), Vec::new()));
        }
        return results;
    }

    let mut walk_targets = Vec::<u32>::new();
    for &target in targets {
        if l1_target_self_loop_covers_suffix_subtree(target, suffix_subtree, flat_trans) {
            let mut profile = Vec::<(u32, u32, u32)>::new();
            let sig_id = state_to_terminal_signature[target as usize];
            if sig_id != 0 {
                if let (Some(&first), Some(&last)) = (token_ids.first(), token_ids.last()) {
                    profile.push((sig_id, first as u32, last as u32));
                }
            }
            results.push(((first_byte, target), profile));
        } else {
            walk_targets.push(target);
        }
    }

    if walk_targets.is_empty() {
        return results;
    }

    let mut dedup_others: Option<Vec<Vec<u32>>> = None;
    if !has_empty_suffix {
        let mut first_suffix_bytes = Vec::<u8>::new();
        for word in 0..4u8 {
            let mut bits = suffix_first_bytes[word as usize];
            while bits != 0 {
                let offset = bits.trailing_zeros() as u8;
                first_suffix_bytes.push(word * 64 + offset);
                bits &= bits - 1;
            }
        }

        if !first_suffix_bytes.is_empty() {
            let mut fp_groups = FxHashMap::<Vec<u32>, Vec<u32>>::default();
            for &target in &walk_targets {
                let fp: Vec<u32> = first_suffix_bytes
                    .iter()
                    .map(|&byte| flat_trans[target as usize * 256 + byte as usize])
                    .collect();
                fp_groups.entry(fp).or_default().push(target);
            }

            let mut deduped_targets = Vec::<u32>::new();
            let mut others = Vec::<Vec<u32>>::new();
            for (fp, group) in fp_groups {
                if fp.iter().all(|&state| state == dead) {
                    for target in group {
                        results.push(((first_byte, target), Vec::new()));
                    }
                    continue;
                }
                deduped_targets.push(group[0]);
                others.push(group[1..].to_vec());
            }
            if deduped_targets.len() < walk_targets.len() {
                walk_targets = deduped_targets;
                dedup_others = Some(others);
            }
        }
    }

    if walk_targets.is_empty() {
        return results;
    }

    let num_walk = walk_targets.len();
    let mut profiles = vec![Vec::<(u32, u32, u32)>::new(); num_walk];
    let mut suffix_states = walk_targets.clone();

    for (bucket_pos, &internal_token_id) in token_ids.iter().enumerate() {
        let suffix_bytes = &sorted_entries[internal_token_id].1[1..];
        let lcp_len = suffix_lcps[bucket_pos].min(suffix_states.len() / num_walk - 1);
        suffix_states.truncate((lcp_len + 1) * num_walk);

        for byte_pos in lcp_len..suffix_bytes.len() {
            let byte = suffix_bytes[byte_pos];
            let base = byte_pos * num_walk;
            for target_idx in 0..num_walk {
                let previous_state = suffix_states[base + target_idx];
                let next_state = if previous_state == dead {
                    dead
                } else {
                    flat_trans[previous_state as usize * 256 + byte as usize]
                };
                suffix_states.push(next_state);
            }
        }

        let end_base = suffix_bytes.len() * num_walk;
        let token_id = internal_token_id as u32;
        for target_idx in 0..num_walk {
            let final_state = suffix_states[end_base + target_idx];
            if final_state == dead {
                continue;
            }
            let sig_id = state_to_terminal_signature[final_state as usize];
            if sig_id != 0 {
                append_l1_signature_profile_run(&mut profiles[target_idx], sig_id, token_id);
            }
        }
    }

    for (target_idx, (target, profile)) in walk_targets.into_iter().zip(profiles).enumerate() {
        if let Some(ref others) = dedup_others {
            for &other_target in &others[target_idx] {
                results.push(((first_byte, other_target), profile.clone()));
            }
        }
        results.push(((first_byte, target), profile));
    }
    results
}


const L1_NONE: u32 = u32::MAX;

/// Compact suffix trie for one first-byte vocabulary bucket. Token IDs are
/// byte-sorted, so each node's subtree is one contiguous token interval.
#[derive(Clone, Copy)]
struct L1PackedSuffixTrieNode {
    incoming_byte: u8,
    terminal_token: u32,
    subtree_start: u32,
    subtree_end: u32,
    first_child: u32,
    last_child: u32,
    next_sibling: u32,
    first_edge: u32,
    edge_len: u32,
}

impl L1PackedSuffixTrieNode {
    fn root() -> Self {
        Self {
            incoming_byte: 0,
            terminal_token: L1_NONE,
            subtree_start: L1_NONE,
            subtree_end: L1_NONE,
            first_child: L1_NONE,
            last_child: L1_NONE,
            next_sibling: L1_NONE,
            first_edge: 0,
            edge_len: 0,
        }
    }

    fn child(byte: u8) -> Self {
        Self {
            incoming_byte: byte,
            ..Self::root()
        }
    }
}

#[derive(Clone, Copy)]
struct L1PackedSuffixTrieEdge {
    byte: u8,
    child: u32,
}

struct L1PackedSuffixTrie {
    nodes: Vec<L1PackedSuffixTrieNode>,
    edges: Vec<L1PackedSuffixTrieEdge>,
}

impl L1PackedSuffixTrie {
    fn build(
        sorted_entries: &[(u32, Arc<[u8]>)],
        token_ids: &[usize],
        suffix_lcps: &[usize],
    ) -> Self {
        let mut nodes = vec![L1PackedSuffixTrieNode::root()];
        let mut path = vec![0u32];

        for (bucket_pos, &internal_token_id) in token_ids.iter().enumerate() {
            let suffix = &sorted_entries[internal_token_id].1[1..];
            let lcp = suffix_lcps[bucket_pos]
                .min(suffix.len())
                .min(path.len().saturating_sub(1));
            path.truncate(lcp + 1);
            for &byte in &suffix[lcp..] {
                let parent = *path.last().expect("suffix trie path") as usize;
                let child = nodes.len() as u32;
                nodes.push(L1PackedSuffixTrieNode::child(byte));
                if nodes[parent].first_child == L1_NONE {
                    nodes[parent].first_child = child;
                } else {
                    let last = nodes[parent].last_child as usize;
                    nodes[last].next_sibling = child;
                }
                nodes[parent].last_child = child;
                path.push(child);
            }

            let token_id = internal_token_id as u32;
            let terminal = *path.last().expect("suffix trie terminal path") as usize;
            debug_assert_eq!(nodes[terminal].terminal_token, L1_NONE);
            nodes[terminal].terminal_token = token_id;
        }

        // Token IDs follow the byte-sorted vocabulary order. Every suffix-trie
        // subtree therefore spans one contiguous leaf-token interval. Computing
        // that interval once bottom-up avoids rewriting it along each token path.
        for node_index in (0..nodes.len()).rev() {
            let mut subtree_start = nodes[node_index].terminal_token;
            let mut subtree_end = nodes[node_index].terminal_token;
            let mut child = nodes[node_index].first_child;
            while child != L1_NONE {
                let child_node = nodes[child as usize];
                if subtree_start == L1_NONE || child_node.subtree_start < subtree_start {
                    subtree_start = child_node.subtree_start;
                }
                if subtree_end == L1_NONE || child_node.subtree_end > subtree_end {
                    subtree_end = child_node.subtree_end;
                }
                child = child_node.next_sibling;
            }
            debug_assert_ne!(subtree_start, L1_NONE);
            debug_assert_ne!(subtree_end, L1_NONE);
            nodes[node_index].subtree_start = subtree_start;
            nodes[node_index].subtree_end = subtree_end;
        }

        let mut edges = Vec::with_capacity(nodes.len().saturating_sub(1));
        for node_index in 0..nodes.len() {
            let first_edge = edges.len() as u32;
            let mut child = nodes[node_index].first_child;
            while child != L1_NONE {
                let child_node = nodes[child as usize];
                edges.push(L1PackedSuffixTrieEdge {
                    byte: child_node.incoming_byte,
                    child,
                });
                child = child_node.next_sibling;
            }
            nodes[node_index].first_edge = first_edge;
            nodes[node_index].edge_len = edges.len() as u32 - first_edge;
        }
        Self { nodes, edges }
    }
}

#[derive(Clone, Copy, Default)]
struct L1PackedProductNodeData {
    states_start: u32,
    states_len: u32,
    behaviors_start: u32,
    records_start: u32,
}

#[derive(Clone, Copy, Default)]
struct L1PackedProductEdgeData {
    map_start: u32,
}

#[derive(Clone, Copy)]
struct L1PackedProductBehaviorRecord {
    terminal_signature: u32,
    child_behaviors_start: u32,
    uniform_signature: u32,
    hash_next: u32,
}

#[inline]
fn l1_packed_hash_behavior(terminal_signature: u32, child_behaviors: &[u32]) -> u64 {
    let mut hash = terminal_signature as u64 ^ 0x9e37_79b9_7f4a_7c15;
    for &child in child_behaviors {
        hash = hash.rotate_left(13) ^ (child as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
        hash = hash.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    hash
}

fn l1_packed_uniform_signature(
    trie: &L1PackedSuffixTrie,
    node_index: usize,
    behavior_id: u32,
    data: &[L1PackedProductNodeData],
    records: &[L1PackedProductBehaviorRecord],
) -> u32 {
    if behavior_id == 0 {
        return 0;
    }
    let node = trie.nodes[node_index];
    if node.edge_len == 0 {
        return behavior_id;
    }
    if node.terminal_token == L1_NONE && node.edge_len == 1 {
        let child = trie.edges[node.first_edge as usize].child as usize;
        return l1_packed_uniform_signature(trie, child, behavior_id, data, records);
    }
    records[data[node_index].records_start as usize + behavior_id as usize - 1].uniform_signature
}

fn l1_packed_append_behavior(
    trie: &L1PackedSuffixTrie,
    node_index: usize,
    behavior_id: u32,
    data: &[L1PackedProductNodeData],
    records: &[L1PackedProductBehaviorRecord],
    record_child_behaviors: &[u32],
    profile: &mut Vec<(u32, u32, u32)>,
) {
    if behavior_id == 0 {
        return;
    }
    let node = trie.nodes[node_index];
    if node.edge_len == 0 {
        append_l1_signature_profile_run(profile, behavior_id, node.subtree_start);
        return;
    }
    if node.terminal_token == L1_NONE && node.edge_len == 1 {
        let child = trie.edges[node.first_edge as usize].child as usize;
        l1_packed_append_behavior(
            trie,
            child,
            behavior_id,
            data,
            records,
            record_child_behaviors,
            profile,
        );
        return;
    }

    let record = records[data[node_index].records_start as usize + behavior_id as usize - 1];
    if record.uniform_signature != 0 {
        append_l1_signature_profile_run(profile, record.uniform_signature, node.subtree_start);
        if node.subtree_end != node.subtree_start {
            profile.last_mut().expect("uniform packed profile run").2 = node.subtree_end;
        }
        return;
    }
    if record.terminal_signature != 0 {
        append_l1_signature_profile_run(profile, record.terminal_signature, node.terminal_token);
    }
    let children_start = record.child_behaviors_start as usize;
    for edge_offset in 0..node.edge_len as usize {
        let edge = trie.edges[node.first_edge as usize + edge_offset];
        l1_packed_append_behavior(
            trie,
            edge.child as usize,
            record_child_behaviors[children_start + edge_offset],
            data,
            records,
            record_child_behaviors,
            profile,
        );
    }
}

fn l1_bucket_suffix_signature_profiles_packed(
    first_byte: u8,
    targets: &[u32],
    sorted_entries: &[(u32, Arc<[u8]>)],
    token_ids: &[usize],
    suffix_lcps: &[usize],
    suffix_subtree: &[u64; 4],
    suffix_first_bytes: &[u64; 4],
    has_empty_suffix: bool,
    state_to_terminal_signature: &[u32],
    flat_trans: &[u32],
    transitions_by_byte: Option<&[u32]>,
    num_lexer_states: usize,
) -> Vec<((u8, u32), Arc<[(u32, u32, u32)]>)> {
    let profiling = compile_profile_enabled();
    let total_started_at = profiling.then(Instant::now);
    let dead = u32::MAX;
    let mut results = Vec::<((u8, u32), Arc<[(u32, u32, u32)]>)>::with_capacity(targets.len());
    if token_ids.is_empty() {
        for &target in targets {
            results.push(((first_byte, target), Arc::from([])));
        }
        return results;
    }

    let mut walk_targets = Vec::<u32>::new();
    for &target in targets {
        if l1_target_self_loop_covers_suffix_subtree(target, suffix_subtree, flat_trans) {
            let mut profile = Vec::<(u32, u32, u32)>::new();
            let sig_id = state_to_terminal_signature[target as usize];
            if sig_id != 0 {
                profile.push((sig_id, token_ids[0] as u32, *token_ids.last().unwrap() as u32));
            }
            results.push(((first_byte, target), Arc::from(profile)));
        } else {
            walk_targets.push(target);
        }
    }
    if walk_targets.is_empty() {
        return results;
    }

    let mut dedup_others: Option<Vec<Vec<u32>>> = None;
    if !has_empty_suffix {
        let mut first_suffix_bytes = Vec::<u8>::new();
        for word in 0..4u8 {
            let mut bits = suffix_first_bytes[word as usize];
            while bits != 0 {
                let offset = bits.trailing_zeros() as u8;
                first_suffix_bytes.push(word * 64 + offset);
                bits &= bits - 1;
            }
        }
        if !first_suffix_bytes.is_empty() {
            let mut fp_groups = FxHashMap::<Vec<u32>, Vec<u32>>::default();
            for &target in &walk_targets {
                let fp: Vec<u32> = first_suffix_bytes
                    .iter()
                    .map(|&byte| {
                        l1_transition(
                            flat_trans,
                            transitions_by_byte,
                            num_lexer_states,
                            target,
                            byte as usize,
                        )
                    })
                    .collect();
                fp_groups.entry(fp).or_default().push(target);
            }
            let mut deduped_targets = Vec::<u32>::new();
            let mut others = Vec::<Vec<u32>>::new();
            for (fp, group) in fp_groups {
                if fp.iter().all(|&state| state == dead) {
                    for target in group {
                        results.push(((first_byte, target), Arc::from([])));
                    }
                    continue;
                }
                deduped_targets.push(group[0]);
                others.push(group[1..].to_vec());
            }
            if deduped_targets.len() < walk_targets.len() {
                walk_targets = deduped_targets;
                dedup_others = Some(others);
            }
        }
    }
    if walk_targets.is_empty() {
        return results;
    }

    let trie_started_at = profiling.then(Instant::now);
    let trie = L1PackedSuffixTrie::build(sorted_entries, token_ids, suffix_lcps);
    let trie_ms = trie_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let propagate_started_at = profiling.then(Instant::now);
    let mut data = vec![L1PackedProductNodeData::default(); trie.nodes.len()];
    let mut edge_data = vec![L1PackedProductEdgeData::default(); trie.edges.len()];
    let mut states = Vec::<u32>::new();
    let mut transition_maps = Vec::<u32>::new();
    states.extend_from_slice(&walk_targets);
    data[0].states_start = 0;
    data[0].states_len = walk_targets.len() as u32;

    let mut seen_stamp = vec![0u32; num_lexer_states];
    let mut seen_index = vec![0u32; num_lexer_states];
    let mut stamp = 0u32;
    for node_index in 0..trie.nodes.len() {
        let node = trie.nodes[node_index];
        let parent_start = data[node_index].states_start as usize;
        let parent_len = data[node_index].states_len as usize;
        if parent_len == 0 {
            continue;
        }
        for edge_offset in 0..node.edge_len as usize {
            let edge_index = node.first_edge as usize + edge_offset;
            let edge = trie.edges[edge_index];
            let child = edge.child as usize;
            edge_data[edge_index].map_start = transition_maps.len() as u32;
            let child_start = states.len() as u32;
            if parent_len == 1 {
                let next = l1_transition(
                    flat_trans,
                    transitions_by_byte,
                    num_lexer_states,
                    states[parent_start],
                    edge.byte as usize,
                );
                if next == dead {
                    transition_maps.push(L1_NONE);
                } else {
                    states.push(next);
                    transition_maps.push(0);
                }
            } else if parent_len == 2 {
                let first = l1_transition(
                    flat_trans,
                    transitions_by_byte,
                    num_lexer_states,
                    states[parent_start],
                    edge.byte as usize,
                );
                let second = l1_transition(
                    flat_trans,
                    transitions_by_byte,
                    num_lexer_states,
                    states[parent_start + 1],
                    edge.byte as usize,
                );
                let first_index = if first == dead {
                    L1_NONE
                } else {
                    states.push(first);
                    0
                };
                let second_index = if second == dead {
                    L1_NONE
                } else if second == first && first != dead {
                    0
                } else {
                    let index = states.len() as u32 - child_start;
                    states.push(second);
                    index
                };
                transition_maps.push(first_index);
                transition_maps.push(second_index);
            } else {
                stamp = stamp.wrapping_add(1);
                if stamp == 0 {
                    seen_stamp.fill(0);
                    stamp = 1;
                }
                for state_offset in 0..parent_len {
                    let state = states[parent_start + state_offset];
                    let next = l1_transition(
                        flat_trans,
                        transitions_by_byte,
                        num_lexer_states,
                        state,
                        edge.byte as usize,
                    );
                    if next == dead {
                        transition_maps.push(L1_NONE);
                    } else if seen_stamp[next as usize] == stamp {
                        transition_maps.push(seen_index[next as usize]);
                    } else {
                        let child_index = states.len() as u32 - child_start;
                        seen_stamp[next as usize] = stamp;
                        seen_index[next as usize] = child_index;
                        states.push(next);
                        transition_maps.push(child_index);
                    }
                }
            }
            data[child].states_start = child_start;
            data[child].states_len = states.len() as u32 - child_start;
        }
    }

    let propagate_ms = propagate_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let behavior_started_at = profiling.then(Instant::now);
    let mut behavior_ids = Vec::<u32>::with_capacity(states.len());
    let mut records = Vec::<L1PackedProductBehaviorRecord>::new();
    let mut record_child_behaviors = Vec::<u32>::new();
    // Behavior interning is local to each trie node. Reusing the table avoids
    // an allocation and rehash-table growth for every branching node while
    // preserving the node-local behavior IDs and collision chains.
    let mut behavior_hash_heads = FxHashMap::<u64, u32>::default();
    let mut unary_behavior_ids = FxHashMap::<(u32, u32), u32>::default();
    let mut binary_behavior_ids = FxHashMap::<(u32, u32, u32), u32>::default();
    let mut scratch_children = Vec::<u32>::new();
    for node_index in (0..trie.nodes.len()).rev() {
        let node = trie.nodes[node_index];
        let state_start = data[node_index].states_start as usize;
        let state_len = data[node_index].states_len as usize;
        data[node_index].behaviors_start = behavior_ids.len() as u32;
        data[node_index].records_start = records.len() as u32;
        if state_len == 0 {
            continue;
        }
        if node.edge_len == 0 {
            for state_offset in 0..state_len {
                behavior_ids.push(state_to_terminal_signature[states[state_start + state_offset] as usize]);
            }
            continue;
        }
        if node.terminal_token == L1_NONE && node.edge_len == 1 {
            let edge_index = node.first_edge as usize;
            let child = trie.edges[edge_index].child as usize;
            let child_behavior_start = data[child].behaviors_start as usize;
            let map_start = edge_data[edge_index].map_start as usize;
            for state_offset in 0..state_len {
                let child_state_index = transition_maps[map_start + state_offset];
                behavior_ids.push(if child_state_index == L1_NONE {
                    0
                } else {
                    behavior_ids[child_behavior_start + child_state_index as usize]
                });
            }
            continue;
        }
        if node.edge_len == 1 {
            let edge_index = node.first_edge as usize;
            let child = trie.edges[edge_index].child as usize;
            let child_behavior_start = data[child].behaviors_start as usize;
            let map_start = edge_data[edge_index].map_start as usize;
            unary_behavior_ids.clear();
            for state_offset in 0..state_len {
                let state = states[state_start + state_offset];
                let terminal_signature = state_to_terminal_signature[state as usize];
                let child_state_index = transition_maps[map_start + state_offset];
                let child_behavior = if child_state_index == L1_NONE {
                    0
                } else {
                    behavior_ids[child_behavior_start + child_state_index as usize]
                };
                if terminal_signature == 0 && child_behavior == 0 {
                    behavior_ids.push(0);
                    continue;
                }
                let key = (terminal_signature, child_behavior);
                let behavior_id = if let Some(&id) = unary_behavior_ids.get(&key) {
                    id
                } else {
                    let child_behaviors_start = record_child_behaviors.len() as u32;
                    record_child_behaviors.push(child_behavior);
                    let child_uniform = l1_packed_uniform_signature(
                        &trie,
                        child,
                        child_behavior,
                        &data,
                        &records,
                    );
                    let uniform_signature = if terminal_signature != 0
                        && child_uniform == terminal_signature
                    {
                        terminal_signature
                    } else {
                        0
                    };
                    let id = records.len() as u32 - data[node_index].records_start + 1;
                    records.push(L1PackedProductBehaviorRecord {
                        terminal_signature,
                        child_behaviors_start,
                        uniform_signature,
                        hash_next: L1_NONE,
                    });
                    unary_behavior_ids.insert(key, id);
                    id
                };
                behavior_ids.push(behavior_id);
            }
            continue;
        }
        if node.edge_len == 2 {
            let first_edge_index = node.first_edge as usize;
            let second_edge_index = first_edge_index + 1;
            let first_child = trie.edges[first_edge_index].child as usize;
            let second_child = trie.edges[second_edge_index].child as usize;
            let first_behavior_start = data[first_child].behaviors_start as usize;
            let second_behavior_start = data[second_child].behaviors_start as usize;
            let first_map_start = edge_data[first_edge_index].map_start as usize;
            let second_map_start = edge_data[second_edge_index].map_start as usize;
            binary_behavior_ids.clear();
            for state_offset in 0..state_len {
                let state = states[state_start + state_offset];
                let terminal_signature = if node.terminal_token == L1_NONE {
                    0
                } else {
                    state_to_terminal_signature[state as usize]
                };
                let first_state_index = transition_maps[first_map_start + state_offset];
                let first_behavior = if first_state_index == L1_NONE {
                    0
                } else {
                    behavior_ids[first_behavior_start + first_state_index as usize]
                };
                let second_state_index = transition_maps[second_map_start + state_offset];
                let second_behavior = if second_state_index == L1_NONE {
                    0
                } else {
                    behavior_ids[second_behavior_start + second_state_index as usize]
                };
                if terminal_signature == 0 && first_behavior == 0 && second_behavior == 0 {
                    behavior_ids.push(0);
                    continue;
                }
                let key = (terminal_signature, first_behavior, second_behavior);
                let behavior_id = if let Some(&id) = binary_behavior_ids.get(&key) {
                    id
                } else {
                    let child_behaviors_start = record_child_behaviors.len() as u32;
                    record_child_behaviors.push(first_behavior);
                    record_child_behaviors.push(second_behavior);
                    let first_uniform = l1_packed_uniform_signature(
                        &trie,
                        first_child,
                        first_behavior,
                        &data,
                        &records,
                    );
                    let second_uniform = l1_packed_uniform_signature(
                        &trie,
                        second_child,
                        second_behavior,
                        &data,
                        &records,
                    );
                    let uniform_signature = if node.terminal_token == L1_NONE {
                        if first_uniform != 0 && first_uniform == second_uniform {
                            first_uniform
                        } else {
                            0
                        }
                    } else if terminal_signature != 0
                        && first_uniform == terminal_signature
                        && second_uniform == terminal_signature
                    {
                        terminal_signature
                    } else {
                        0
                    };
                    let id = records.len() as u32 - data[node_index].records_start + 1;
                    records.push(L1PackedProductBehaviorRecord {
                        terminal_signature,
                        child_behaviors_start,
                        uniform_signature,
                        hash_next: L1_NONE,
                    });
                    binary_behavior_ids.insert(key, id);
                    id
                };
                behavior_ids.push(behavior_id);
            }
            continue;
        }

        let child_count = node.edge_len as usize;
        behavior_hash_heads.clear();
        scratch_children.clear();
        if scratch_children.capacity() < child_count {
            scratch_children.reserve(child_count - scratch_children.capacity());
        }
        for state_offset in 0..state_len {
            let state = states[state_start + state_offset];
            let terminal_signature = if node.terminal_token == L1_NONE {
                0
            } else {
                state_to_terminal_signature[state as usize]
            };
            scratch_children.clear();
            for edge_offset in 0..child_count {
                let edge_index = node.first_edge as usize + edge_offset;
                let child = trie.edges[edge_index].child as usize;
                let child_state_index = transition_maps[edge_data[edge_index].map_start as usize + state_offset];
                scratch_children.push(if child_state_index == L1_NONE {
                    0
                } else {
                    behavior_ids[data[child].behaviors_start as usize + child_state_index as usize]
                });
            }
            if terminal_signature == 0 && scratch_children.iter().all(|&id| id == 0) {
                behavior_ids.push(0);
                continue;
            }
            let hash = l1_packed_hash_behavior(terminal_signature, &scratch_children);
            let mut found = None;
            let mut candidate = behavior_hash_heads.get(&hash).copied().unwrap_or(L1_NONE);
            while candidate != L1_NONE {
                let record = records[data[node_index].records_start as usize + candidate as usize - 1];
                if record.terminal_signature == terminal_signature
                    && record_child_behaviors[record.child_behaviors_start as usize..record.child_behaviors_start as usize + child_count]
                        == scratch_children[..]
                {
                    found = Some(candidate);
                    break;
                }
                candidate = record.hash_next;
            }
            let behavior_id = if let Some(id) = found {
                id
            } else {
                let child_behaviors_start = record_child_behaviors.len() as u32;
                record_child_behaviors.extend_from_slice(&scratch_children);
                let mut uniform_signature = if node.terminal_token == L1_NONE {
                    0
                } else {
                    terminal_signature
                };
                if uniform_signature != 0 {
                    for edge_offset in 0..child_count {
                        let edge_index = node.first_edge as usize + edge_offset;
                        let child = trie.edges[edge_index].child as usize;
                        let child_uniform = l1_packed_uniform_signature(
                            &trie,
                            child,
                            scratch_children[edge_offset],
                            &data,
                            &records,
                        );
                        if child_uniform != uniform_signature {
                            uniform_signature = 0;
                            break;
                        }
                    }
                } else if node.terminal_token == L1_NONE {
                    let mut candidate_uniform = 0u32;
                    for edge_offset in 0..child_count {
                        let edge_index = node.first_edge as usize + edge_offset;
                        let child = trie.edges[edge_index].child as usize;
                        let child_uniform = l1_packed_uniform_signature(
                            &trie,
                            child,
                            scratch_children[edge_offset],
                            &data,
                            &records,
                        );
                        if child_uniform == 0
                            || (candidate_uniform != 0 && candidate_uniform != child_uniform)
                        {
                            candidate_uniform = 0;
                            break;
                        }
                        candidate_uniform = child_uniform;
                    }
                    uniform_signature = candidate_uniform;
                }
                let id = records.len() as u32 - data[node_index].records_start + 1;
                records.push(L1PackedProductBehaviorRecord {
                    terminal_signature,
                    child_behaviors_start,
                    uniform_signature,
                    hash_next: behavior_hash_heads.get(&hash).copied().unwrap_or(L1_NONE),
                });
                behavior_hash_heads.insert(hash, id);
                id
            };
            behavior_ids.push(behavior_id);
        }
    }

    let behavior_ms = behavior_started_at.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
    let materialize_started_at = profiling.then(Instant::now);
    let root_behavior_start = data[0].behaviors_start as usize;
    let mut profiles_by_behavior = FxHashMap::<u32, Arc<[(u32, u32, u32)]>>::default();
    for (target_index, &target) in walk_targets.iter().enumerate() {
        let behavior_id = behavior_ids[root_behavior_start + target_index];
        let profile = if let Some(profile) = profiles_by_behavior.get(&behavior_id) {
            Arc::clone(profile)
        } else {
            let mut profile = Vec::new();
            l1_packed_append_behavior(
                &trie,
                0,
                behavior_id,
                &data,
                &records,
                &record_child_behaviors,
                &mut profile,
            );
            let profile: Arc<[(u32, u32, u32)]> = Arc::from(profile);
            profiles_by_behavior.insert(behavior_id, Arc::clone(&profile));
            profile
        };
        if let Some(ref others) = dedup_others {
            for &other_target in &others[target_index] {
                results.push(((first_byte, other_target), Arc::clone(&profile)));
            }
        }
        results.push(((first_byte, target), profile));
    }
    let materialize_ms = materialize_started_at.map_or(0.0, |started| {
        started.elapsed().as_secs_f64() * 1000.0
    });
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][l1_packed_product] first_byte={} tokens={} targets={} trie_nodes={} states={} trie_ms={:.3} propagate_ms={:.3} behavior_ms={:.3} materialize_ms={:.3} total_ms={:.3}",
            first_byte,
            token_ids.len(),
            walk_targets.len(),
            trie.nodes.len(),
            states.len(),
            trie_ms,
            propagate_ms,
            behavior_ms,
            materialize_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    results
}


#[derive(Debug)]
struct L1SortedTokenBuckets {
    empty_token_indices: Vec<usize>,
    token_indices_by_first_byte: Vec<Vec<usize>>,
    suffix_lcps_by_first_byte: Vec<Vec<usize>>,
    suffix_subtree_bytes: Vec<[u64; 4]>,
    suffix_first_bytes_by_bucket: Vec<[u64; 4]>,
    has_empty_suffix_by_bucket: Vec<bool>,
}

fn build_l1_sorted_token_buckets(sorted_entries: &[(u32, Arc<[u8]>)]) -> L1SortedTokenBuckets {
    let mut empty_token_indices = Vec::<usize>::new();
    let mut token_indices_by_first_byte = vec![Vec::<usize>::new(); 256];
    for (internal_token_id, (_original_id, token_bytes)) in sorted_entries.iter().enumerate() {
        if let Some(&first_byte) = token_bytes.first() {
            token_indices_by_first_byte[first_byte as usize].push(internal_token_id);
        } else {
            empty_token_indices.push(internal_token_id);
        }
    }

    let mut suffix_lcps_by_first_byte = vec![Vec::<usize>::new(); 256];
    let mut suffix_subtree_bytes: Vec<[u64; 4]> = vec![[0u64; 4]; 256];
    let mut suffix_first_bytes_by_bucket: Vec<[u64; 4]> = vec![[0u64; 4]; 256];
    let mut has_empty_suffix_by_bucket = vec![false; 256];
    for first_byte in 0..256 {
        let token_ids = &token_indices_by_first_byte[first_byte];
        if token_ids.is_empty() {
            continue;
        }

        let lcps = &mut suffix_lcps_by_first_byte[first_byte];
        lcps.reserve(token_ids.len());

        let mut previous_suffix: &[u8] = &[];
        for &internal_token_id in token_ids {
            let token_bytes = &sorted_entries[internal_token_id].1;
            let suffix_bytes = &token_bytes[1..];
            lcps.push(common_prefix_len(previous_suffix, suffix_bytes));
            if suffix_bytes.is_empty() {
                has_empty_suffix_by_bucket[first_byte] = true;
            } else {
                let b = suffix_bytes[0];
                suffix_first_bytes_by_bucket[first_byte][b as usize >> 6] |= 1u64 << (b & 63);
                for &byte in suffix_bytes {
                    suffix_subtree_bytes[first_byte][byte as usize >> 6] |= 1u64 << (byte & 63);
                }
            }
            previous_suffix = suffix_bytes;
        }
    }

    L1SortedTokenBuckets {
        empty_token_indices,
        token_indices_by_first_byte,
        suffix_lcps_by_first_byte,
        suffix_subtree_bytes,
        suffix_first_bytes_by_bucket,
        has_empty_suffix_by_bucket,
    }
}

fn collect_active_terminal_signature(
    tokenizer: &Tokenizer,
    state: u32,
    active_terminals: &[bool],
) -> Vec<u32> {
    let mut signature = Vec::<u32>::new();
    for tid in tokenizer.matched_terminals_iter(state) {
        if active_terminals.get(tid as usize).copied().unwrap_or(false) {
            signature.push(tid);
        }
    }
    for tid in tokenizer.tokens_accessible_from_state(state).iter() {
        if active_terminals.get(tid).copied().unwrap_or(false) {
            signature.push(tid as u32);
        }
    }
    signature.sort_unstable();
    signature.dedup();
    signature
}

/// Intern the exact active-terminal signature of every tokenizer state.
///
/// Signature zero is the empty signature. Keeping the reverse table lets the
/// L1 terminal-DWA builder reuse the same proof object rather than redoing this
/// full state scan with a separately numbered signature map.
fn build_l1_state_to_terminal_signatures(
    tokenizer: &Tokenizer,
    active_terminals: &[bool],
) -> (Vec<u32>, Vec<Vec<u32>>) {
    let mut signature_to_id = FxHashMap::<Vec<u32>, u32>::default();
    signature_to_id.insert(Vec::new(), 0);
    let mut terminal_signatures = vec![Vec::new()];
    let mut state_to_terminal_signature = vec![0u32; tokenizer.num_states() as usize];

    for state in 0..tokenizer.num_states() as usize {
        let signature =
            collect_active_terminal_signature(tokenizer, state as u32, active_terminals);
        let next_signature_id = terminal_signatures.len() as u32;
        let sig_id = *signature_to_id.entry(signature.clone()).or_insert_with(|| {
            terminal_signatures.push(signature);
            next_signature_id
        });
        state_to_terminal_signature[state] = sig_id;
    }

    if compile_profile_enabled() {
        let mut nonempty_states = 0usize;
        let mut singleton_states = 0usize;
        let mut membership_sum = 0usize;
        let mut max_membership = 0usize;
        for &signature_id in &state_to_terminal_signature {
            let len = terminal_signatures[signature_id as usize].len();
            if len != 0 {
                nonempty_states += 1;
                membership_sum += len;
                max_membership = max_membership.max(len);
                if len == 1 {
                    singleton_states += 1;
                }
            }
        }
        let mut signature_histogram = [0usize; 6];
        for signature in &terminal_signatures {
            let bucket = match signature.len() {
                0 => 0,
                1 => 1,
                2..=4 => 2,
                5..=16 => 3,
                17..=64 => 4,
                _ => 5,
            };
            signature_histogram[bucket] += 1;
        }
        eprintln!(
            "[glrmask/profile][l1_terminal_signatures] states={} signature_ids={} nonempty_states={} singleton_states={} membership_sum={} max_membership={} signature_histogram={:?}",
            state_to_terminal_signature.len(),
            terminal_signatures.len(),
            nonempty_states,
            singleton_states,
            membership_sum,
            max_membership,
            signature_histogram,
        );
    }

    (state_to_terminal_signature, terminal_signatures)
}

fn l1_token_signature_profile_for_state(
    start_state: u32,
    sorted_entries: &[(u32, Arc<[u8]>)],
    buckets: &L1SortedTokenBuckets,
    state_to_terminal_signature: &[u32],
    flat_trans: &[u32],
) -> Vec<(u32, u32, u32)> {
    let dead = u32::MAX;
    let mut profile = Vec::<(u32, u32, u32)>::new();
    let start_sig = state_to_terminal_signature[start_state as usize];
    for &internal_token_id in &buckets.empty_token_indices {
        append_l1_signature_profile_run(&mut profile, start_sig, internal_token_id as u32);
    }

    for (first_byte, token_ids) in buckets.token_indices_by_first_byte.iter().enumerate() {
        if token_ids.is_empty() {
            continue;
        }

        let first_target = flat_trans[start_state as usize * 256 + first_byte];
        if first_target == dead {
            for &internal_token_id in token_ids {
                append_l1_signature_profile_run(&mut profile, 0, internal_token_id as u32);
            }
            continue;
        }

        let suffix_lcps = &buckets.suffix_lcps_by_first_byte[first_byte];
        let mut suffix_states = vec![first_target];
        for (bucket_pos, &internal_token_id) in token_ids.iter().enumerate() {
            let suffix_bytes = &sorted_entries[internal_token_id].1[1..];
            let lcp_len = suffix_lcps[bucket_pos].min(suffix_states.len().saturating_sub(1));
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
            let sig_id = if final_state == dead {
                0
            } else {
                state_to_terminal_signature[final_state as usize]
            };
            append_l1_signature_profile_run(&mut profile, sig_id, internal_token_id as u32);
        }
    }

    profile
}

fn append_l1_signature_profile_run(profile: &mut Vec<(u32, u32, u32)>, sig_id: u32, token_id: u32) {
    if let Some((last_sig, _start, end)) = profile.last_mut() {
        if *last_sig == sig_id && end.wrapping_add(1) == token_id {
            *end = token_id;
            return;
        }
    }
    profile.push((sig_id, token_id, token_id));
}

fn build_l1_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab_order: &L1IdentityVocabOrder,
    id_map: &mut InternalIdMap,
    num_terminals: u32,
    active_terminals: &[bool],
    flat_trans: &[u32],
    exact_profile_reuse: Option<&L1ExactProfileReuse>,
) -> Option<(DWA, L1TerminalBuildProfile)> {
    let total_started_at = std::time::Instant::now();
    let internal_vocab_ms = 0.0;
    let sorted_entries = vocab_order.token_entries_sorted.as_ref();
    let token_buckets = &vocab_order.token_buckets;

    if sorted_entries.is_empty() {
        return None;
    }

    let vocab_tree_build_ms = 0.0;

    let state_seed_started_at = Instant::now();
    let mut states_to_initial_tsids = FxHashMap::<u32, Vec<u32>>::default();
    for (internal_tsid, representative_state) in id_map
        .tokenizer_states
        .iter_representative_ids()
        .enumerate()
    {
        states_to_initial_tsids
            .entry(representative_state)
            .or_default()
            .push(internal_tsid as u32);
    }
    let state_seed_ms = state_seed_started_at.elapsed().as_secs_f64() * 1000.0;
    let dead = u32::MAX;
    // Exact equivalence has already built this state-signature index on the
    // normal L1 path. Reuse it verbatim. The fallback keeps the direct builder
    // self-contained for callers that intentionally bypass exact-profile reuse.
    let fallback_signatures = exact_profile_reuse.is_none().then(|| {
        let (state_to_exact_signature, signatures_with_empty) =
            build_l1_state_to_terminal_signatures(tokenizer, active_terminals);
        let terminal_signatures: Vec<Vec<u32>> =
            signatures_with_empty.into_iter().skip(1).collect();
        let state_to_terminal_signature: Vec<u32> = state_to_exact_signature
            .into_iter()
            .map(|signature_id| {
                if signature_id == 0 {
                    u32::MAX
                } else {
                    signature_id - 1
                }
            })
            .collect();
        (terminal_signatures, state_to_terminal_signature)
    });
    let (terminal_signatures, state_to_terminal_signature): (&[Vec<u32>], &[u32]) =
        if let Some(reuse) = exact_profile_reuse {
            (
                reuse.direct_terminal_signatures.as_ref(),
                reuse.direct_state_to_terminal_signature.as_ref(),
            )
        } else {
            let (terminal_signatures, state_to_terminal_signature) = fallback_signatures
                .as_ref()
                .expect("missing fallback L1 terminal signatures");
            (terminal_signatures.as_slice(), state_to_terminal_signature.as_slice())
        };

    // Batch simulation: for each unique start state, simulate all tokens through
    // the DFA and accumulate terminal_signature(final concrete state) → (tsid → token_ids).
    // Parallelized across start states using rayon.

    let traversal_started_at = Instant::now();

    // Parallel traversal: each start_state processed independently.
    // Each (terminal_signature, tsid) pair is unique across start groups since TSIDs
    // partition deterministically into start groups. We exploit this by using
    // Arc from the start and skipping merging entirely.
    let start_states_list: Vec<(&u32, &Vec<u32>)> = states_to_initial_tsids.iter().collect();
    let empty_token_indices = token_buckets.empty_token_indices.as_slice();
    let token_indices_by_first_byte = &token_buckets.token_indices_by_first_byte;
    let suffix_lcps_by_first_byte = &token_buckets.suffix_lcps_by_first_byte;
    let suffix_subtree_bytes = &token_buckets.suffix_subtree_bytes;
    let suffix_first_bytes_by_bucket = &token_buckets.suffix_first_bytes_by_bucket;
    let has_empty_suffix_by_bucket = &token_buckets.has_empty_suffix_by_bucket;
    // Walk cache: compute once per unique (first_byte, target) and cache
    // the raw merged ranges. Self-loop optimization: if the target state
    // has self-loops on all suffix bytes, all tokens end at the target
    // state and the walk can be skipped entirely.
    let walk_cache: FxHashMap<(u8, u32), L1WalkProfile> =
        if let Some(exact_profile_reuse) = exact_profile_reuse {
            exact_profile_reuse.materialize_walk_cache()
        } else {
    // Phase 1: Identify unique concrete (first_byte, target_state) pairs.
    // State equivalence is valid for whole-token walks from a start state,
    // but it is not necessarily closed over suffixes after the first byte.
    // Walk token suffixes from the concrete post-first-byte DFA state and
    // only map the final state back to a representative.
    let mut unique_targets: FxHashMap<(u8, u32), ()> = FxHashMap::default();
    for (&start_state, _) in &states_to_initial_tsids {
        for (byte, token_ids) in token_indices_by_first_byte.iter().enumerate() {
            if token_ids.is_empty() {
                continue;
            }
            let target_state = flat_trans[start_state as usize * 256 + byte];
            if target_state != dead {
                unique_targets
                    .entry((byte as u8, target_state))
                    .or_default();
            }
        }
    }
    let unique_walk_keys: Vec<(u8, u32)> = unique_targets.into_keys().collect();

    // Precompute self-loop mask per target state.
    let mut self_loop_masks: FxHashMap<u32, [u64; 4]> = FxHashMap::default();
    for &(_, target) in &unique_walk_keys {
        self_loop_masks.entry(target).or_insert_with(|| {
            let mut mask = [0u64; 4];
            for byte in 0..=255u8 {
                if flat_trans[target as usize * 256 + byte as usize] == target {
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
    {
        // Group unique walk keys by first_byte.
        let mut walks_by_byte: FxHashMap<u8, Vec<u32>> = FxHashMap::default();
        for &(byte, target) in &unique_walk_keys {
            walks_by_byte.entry(byte).or_default().push(target);
        }
        let byte_groups: Vec<(u8, Vec<u32>)> = walks_by_byte.into_iter().collect();

        let build_byte_batch = |(first_byte, all_targets): &(u8, Vec<u32>)| {
                let byte = *first_byte;
                let bucket_idx = byte as usize;
                let token_ids = &token_indices_by_first_byte[bucket_idx];
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
                    let first = *token_ids.first().unwrap() as u32;
                    let last = *token_ids.last().unwrap() as u32;
                    for &target in &selfloop_targets {
                        let sig_id = state_to_terminal_signature[target as usize];
                        if sig_id != u32::MAX {
                            results.push(((byte, target), vec![(sig_id, vec![(first, last)])]));
                        }
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
                            let fp: Vec<u32> = sfb_list
                                .iter()
                                .map(|&b| flat_trans[target as usize * 256 + b as usize])
                                .collect();
                            fp_groups.entry(fp).or_default().push(target);
                        }

                        // Separate dead groups (all entries are dead)
                        // from live groups
                        let mut deduped_targets: Vec<u32> = Vec::new();
                        let mut others: Vec<Vec<u32>> = Vec::new();
                        let mut dead_eliminated = 0usize;
                        let mut dup_eliminated = 0usize;

                        for (fp, group) in &fp_groups {
                            let all_dead = fp.iter().all(|&s| s == dead);
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
                            dedup_others = Some(others);
                            walk_targets = deduped_targets;
                        }
                    }
                }

                if walk_targets.is_empty() {
                    return results;
                }

                let num_walk = walk_targets.len();

                // suffix_states: flat [pos * num_walk + target_idx]
                // Position 0 = initial target states (before any suffix bytes).
                let mut suffix_states: Vec<u32> = walk_targets.clone();

                // Per-target run-flush state.
                let mut run_signature_ids: Vec<u32> = vec![u32::MAX; num_walk];
                let mut run_starts: Vec<u32> = vec![0; num_walk];
                let mut run_ends: Vec<u32> = vec![0; num_walk];
                let mut signature_maps: Vec<FxHashMap<u32, Vec<(u32, u32)>>> =
                    (0..num_walk).map(|_| FxHashMap::default()).collect();

                for (bucket_pos, &internal_token_id) in token_ids.iter().enumerate() {
                    let suffix_bytes = &sorted_entries[internal_token_id].1[1..];
                    let lcp_len = suffix_lcps[bucket_pos];

                    // Truncate all targets to lcp_len + 1 positions.
                    suffix_states.truncate((lcp_len + 1) * num_walk);

                    // Walk remaining suffix bytes with all targets in parallel.
                    for byte_pos in lcp_len..suffix_bytes.len() {
                        let b = suffix_bytes[byte_pos];
                        let base = byte_pos * num_walk;
                        for t in 0..num_walk {
                            let prev_state = suffix_states[base + t];
                            let next_state = if prev_state == dead {
                                dead
                            } else {
                                flat_trans[prev_state as usize * 256 + b as usize]
                            };
                            suffix_states.push(next_state);
                        }
                    }

                    // Record final states for each target.
                    let end_base = suffix_bytes.len() * num_walk;
                    let token_id = internal_token_id as u32;
                    for t in 0..num_walk {
                        let final_state = suffix_states[end_base + t];
                        if final_state != dead {
                            let sig_id = state_to_terminal_signature[final_state as usize];
                            if sig_id == u32::MAX {
                                if run_signature_ids[t] != u32::MAX {
                                    signature_maps[t]
                                        .entry(run_signature_ids[t])
                                        .or_default()
                                        .push((run_starts[t], run_ends[t]));
                                    run_signature_ids[t] = u32::MAX;
                                }
                                continue;
                            }
                            if run_signature_ids[t] == sig_id
                                && run_ends[t].wrapping_add(1) == token_id
                            {
                                run_ends[t] = token_id;
                            } else {
                                // Flush previous run for this target.
                                if run_signature_ids[t] != u32::MAX {
                                    signature_maps[t]
                                        .entry(run_signature_ids[t])
                                        .or_default()
                                        .push((run_starts[t], run_ends[t]));
                                }
                                run_signature_ids[t] = sig_id;
                                run_starts[t] = token_id;
                                run_ends[t] = token_id;
                            }
                        } else {
                            // Dead: flush current run for this target.
                            if run_signature_ids[t] != u32::MAX {
                                signature_maps[t]
                                    .entry(run_signature_ids[t])
                                    .or_default()
                                    .push((run_starts[t], run_ends[t]));
                                run_signature_ids[t] = u32::MAX;
                            }
                        }
                    }
                }

                // Flush remaining runs.
                for t in 0..num_walk {
                    if run_signature_ids[t] != u32::MAX {
                        signature_maps[t]
                            .entry(run_signature_ids[t])
                            .or_default()
                            .push((run_starts[t], run_ends[t]));
                    }
                }

                // Package per-target results.
                for (t, map) in signature_maps.into_iter().enumerate() {
                    let target = walk_targets[t];
                    let entries: Vec<(u32, Vec<(u32, u32)>)> = map
                        .into_iter()
                        .map(|(sig_id, ranges)| {
                            debug_assert!(
                                ranges.windows(2).all(|w| w[0].1 < w[1].0),
                                "Phase 1 ranges should be sorted and non-overlapping"
                            );
                            (sig_id, ranges)
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
            };
        let all_batches: Vec<Vec<((u8, u32), Vec<(u32, Vec<(u32, u32)>)>)>> =
            if rayon::current_num_threads() == 1 {
                byte_groups.iter().map(build_byte_batch).collect()
            } else {
                byte_groups.par_iter().map(build_byte_batch).collect()
            };

        let mut cache: FxHashMap<(u8, u32), Vec<(u32, Vec<(u32, u32)>)>> = FxHashMap::default();
        for batch in all_batches {
            for (key, value) in batch {
                cache.insert(key, value);
            }
        }
        cache
            .into_iter()
            .map(|(target, profile)| (target, freeze_l1_walk_profile_from_direct(profile)))
            .collect()
    }
        };

    // Build indexed walk_cache: (byte, target) → Vec of (signature_id, &ranges, entry_hash, entry_range_count).
    // entry_hash is precomputed from the ranges so Phase 2 can combine hashes
    // in O(entries) instead of O(ranges).
    let indexed_cache_started_at = compile_profile_enabled().then(Instant::now);
    let indexed_walk_cache: FxHashMap<(u8, u32), Vec<(usize, &[(u32, u32)], u64, usize)>> =
        walk_cache
            .iter()
            .map(|(&key, results)| {
                let indexed: Vec<(usize, &[(u32, u32)], u64, usize)> = results
                    .iter()
                    .map(|(sig_id, ranges)| {
                        let ranges = ranges.as_ref();
                        let mut h: u64 = 0;
                        for &(s, e) in ranges {
                            h = h.wrapping_add(range_hash_val(s, e));
                        }
                        let entry_hash = (ranges.len() as u64).wrapping_add(h);
                        (
                            *sig_id as usize,
                            ranges,
                            entry_hash,
                            ranges.len(),
                        )
                    })
                    .collect();
                (key, indexed)
            })
            .collect();
    if let Some(indexed_cache_started_at) = indexed_cache_started_at {
        eprintln!(
            "[glrmask/profile][l1_indexed_walk_cache] targets={} total_ms={:.3}",
            indexed_walk_cache.len(),
            indexed_cache_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    // Phase 2: For each start_state, collect walk_cache references per

    // Pre-build empty token ranges (shared across all start_states).
    let empty_token_ranges: Vec<(u32, u32)> = {
        let mut ranges = Vec::new();
        for &internal_token_id in empty_token_indices {
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

    let build_start_state_results = |&(&start_state, ref initial_tsids): &(&u32, &Vec<u32>)| {
            let collect_start = Instant::now();

            // Track only touched terminal signatures for this start state instead of
            // allocating all signature buckets every time.
            let mut touched_positions: FxHashMap<usize, usize> = FxHashMap::default();
            let mut touched_signatures: Vec<(usize, Vec<&[(u32, u32)]>, u64, usize)> = Vec::new();

            // Empty tokens: terminal signature at the start state.
            if !empty_token_ranges.is_empty() {
                let sig_id = state_to_terminal_signature[start_state as usize];
                if sig_id != u32::MAX {
                    let sig_idx = sig_id as usize;
                    let position = if let Some(&position) = touched_positions.get(&sig_idx) {
                        position
                    } else {
                        let position = touched_signatures.len();
                        touched_positions.insert(sig_idx, position);
                        touched_signatures.push((sig_idx, Vec::new(), 0, 0));
                        position
                    };
                    let (_, refs, hash_accum, len_accum) = &mut touched_signatures[position];
                    refs.push(empty_token_ranges.as_slice());
                    *hash_accum = hash_accum.wrapping_add(empty_token_hash);
                    *len_accum += empty_token_ranges.len();
                }
            }

            for (byte, token_ids) in token_indices_by_first_byte.iter().enumerate() {
                if token_ids.is_empty() {
                    continue;
                }
                let target_state = flat_trans[start_state as usize * 256 + byte];
                if target_state == dead {
                    continue;
                }
                if let Some(results) = indexed_walk_cache.get(&(byte as u8, target_state)) {
                    for &(sig_idx, ranges, entry_hash, entry_mc) in results {
                        let position = if let Some(&position) = touched_positions.get(&sig_idx) {
                            position
                        } else {
                            let position = touched_signatures.len();
                            touched_positions.insert(sig_idx, position);
                            touched_signatures.push((sig_idx, Vec::new(), 0, 0));
                            position
                        };
                        let (_, refs, hash_accum, len_accum) = &mut touched_signatures[position];
                        refs.push(ranges);
                        *hash_accum = hash_accum.wrapping_add(entry_hash);
                        *len_accum += entry_mc;
                    }
                }
            }

            // Finalize hashes and build LazyRanges entries.
            let mut result: Vec<(u32, u32, LazyRanges)> = Vec::new();
            for (sig_idx, refs, hash, total_len) in touched_signatures.into_iter() {
                let lazy = LazyRanges {
                    refs,
                    hash,
                    total_len,
                };
                if initial_tsids.len() > 1 {
                    for &tsid in &initial_tsids[..initial_tsids.len() - 1] {
                        result.push((sig_idx as u32, tsid, lazy.clone()));
                    }
                }
                result.push((sig_idx as u32, *initial_tsids.last().unwrap(), lazy));
            }
            result
        };
    let per_thread_results: Vec<Vec<(u32, u32, LazyRanges<'_>)>> =
        if rayon::current_num_threads() == 1 {
            start_states_list.iter().map(build_start_state_results).collect()
        } else {
            start_states_list
                .par_iter()
                .map(build_start_state_results)
                .collect()
        };

    let start_state_collect_ms = traversal_started_at.elapsed().as_secs_f64() * 1000.0;

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
    let process_hash_group = |w: &[usize]| {
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
                    let fast_match =
                        candidate.refs.len() == representative.refs.len()
                            && candidate.refs.iter().zip(representative.refs.iter()).all(
                                |(&a, &b)| {
                                    std::ptr::eq(a.as_ptr(), b.as_ptr()) && a.len() == b.len()
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
                let arc: Arc<RangeSetBlaze<u32>> =
                    Arc::new(rep_materialized.iter().map(|&(s, e)| s..=e).collect());
                for k in sub_start..sub_end {
                    out.push((all_entries[k].0 as usize, all_entries[k].1, arc.clone()));
                }
                sub_start = sub_end;
            }
            out
        };
    let group_results: Vec<Vec<(usize, u32, Arc<RangeSetBlaze<u32>>)>> =
        if rayon::current_num_threads() == 1 {
            hash_group_starts.windows(2).map(process_hash_group).collect()
        } else {
            hash_group_starts
                .par_windows(2)
                .map(process_hash_group)
                .collect()
        };

    // Merge parallel results into deferred_arced (sequential).
    let mut deferred_arced: Vec<Vec<(u32, Arc<RangeSetBlaze<u32>>)>> =
        vec![Vec::new(); terminal_signatures.len()];
    for result in group_results {
        for (sig_id, tsid, arc) in result {
            deferred_arced[sig_id as usize].push((tsid, arc));
        }
    }

    let token_set_intern_ms = token_set_intern_started_at.elapsed().as_secs_f64() * 1000.0;
    let traversal_ms = traversal_started_at.elapsed().as_secs_f64() * 1000.0;

    let tsid_profile_merge_started_at = Instant::now();
    let tsid_profile_merge_before = id_map.num_tsids() as usize;
    let tsid_profile_merge_report = merge_deferred_equivalent_tsids(id_map, &mut deferred_arced);
    let tsid_profile_merge_after = tsid_profile_merge_report.tsids_after;
    let tsid_profile_merge_ms = tsid_profile_merge_started_at.elapsed().as_secs_f64() * 1000.0;

    let distribute_started_at = Instant::now();

    // Build terminal -> terminal-signature ids. Each signature is the exact
    // set of active terminals produced by the full-token end state.
    let mut terminal_to_signatures: Vec<Vec<u32>> = vec![Vec::new(); num_terminals as usize];
    for (sig_id, terminals) in terminal_signatures.iter().enumerate() {
        for &terminal in terminals {
            terminal_to_signatures[terminal as usize].push(sig_id as u32);
        }
    }

    // Pre-compute per-TSID full token set unions and contributing signatures.
    // For each terminal, TSIDs whose contributing signatures are all active reuse
    // the precomputed Arc; only TSIDs with some inactive signatures are recomputed.
    let merge_started_at = Instant::now();

    let num_tsids = id_map.num_tsids() as usize;

    // Build per-TSID: full ranges union + contributing signature count.
    let full_ranges_started_at = Instant::now();
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

    let full_ranges_ms = full_ranges_started_at.elapsed().as_secs_f64() * 1000.0;

    // Group terminals by active_states to deduplicate identical computation
    let terminal_group_started_at = Instant::now();
    let mut active_tids: Vec<usize> = (0..terminal_to_signatures.len())
        .filter(|&i| !terminal_to_signatures[i].is_empty())
        .collect();
    active_tids
        .sort_unstable_by(|&a, &b| terminal_to_signatures[a].cmp(&terminal_to_signatures[b]));
    let mut unique_groups: Vec<Vec<usize>> = Vec::new();
    for &tid in &active_tids {
        if let Some(last) = unique_groups.last_mut() {
            if terminal_to_signatures[last[0]] == terminal_to_signatures[tid] {
                last.push(tid);
                continue;
            }
        }
        unique_groups.push(vec![tid]);
    }

    let terminal_group_ms = terminal_group_started_at.elapsed().as_secs_f64() * 1000.0;
    let num_groups = unique_groups.len();
    let contribution_seed_started_at = Instant::now();
    let signature_groups = build_end_rep_groups(
        &unique_groups,
        &terminal_to_signatures,
        deferred_arced.len(),
    );
    let mut tsid_group_contributions: Vec<Vec<(usize, Arc<RangeSetBlaze<u32>>)>> =
        (0..num_tsids).map(|_| Vec::new()).collect();
    for (sig_id, entries) in deferred_arced.iter().enumerate() {
        if signature_groups[sig_id].is_empty() {
            continue;
        }
        for &(tsid, ref arc) in entries {
            tsid_group_contributions[tsid as usize].push((sig_id, Arc::clone(arc)));
        }
    }

    let contribution_seed_ms = contribution_seed_started_at.elapsed().as_secs_f64() * 1000.0;
    let per_tsid_group_entries_started_at = Instant::now();
    let single_thread_direct_group_assembly =
        rayon::current_num_threads() == 1 && l1_sequential_group_assembly_enabled();
    let group_weight_entries: Vec<Vec<(u32, Arc<RangeSetBlaze<u32>>)>> =
        if single_thread_direct_group_assembly {
            // In the single-threaded case, avoid allocating `num_groups` fresh
            // counters and range buffers for every TSID. Reuse sparse scratch
            // and emit directly into each group, which is already ordered by
            // ascending TSID because this loop is sequential.
            let mut group_weight_entries: Vec<Vec<(u32, Arc<RangeSetBlaze<u32>>)>> =
                (0..num_groups).map(|_| Vec::new()).collect();
            let mut group_counts = vec![0usize; num_groups];
            let mut group_ranges: Vec<Vec<(u32, u32)>> =
                (0..num_groups).map(|_| Vec::new()).collect();
            let mut touched_groups = Vec::<usize>::new();

            for (tsid, contributions) in tsid_group_contributions.iter().enumerate() {
                if contributions.is_empty() {
                    continue;
                }

                for &group_idx in &touched_groups {
                    group_counts[group_idx] = 0;
                    group_ranges[group_idx].clear();
                }
                touched_groups.clear();

                for &(sig_id, ref arc) in contributions {
                    for &group_idx in &signature_groups[sig_id] {
                        if group_counts[group_idx] == 0 {
                            touched_groups.push(group_idx);
                        }
                        group_counts[group_idx] += 1;
                        group_ranges[group_idx].extend(
                            arc.ranges().map(|range| (*range.start(), *range.end())),
                        );
                    }
                }

                for &group_idx in &touched_groups {
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
                        group_weight_entries[group_idx].push((tsid as u32, tokens));
                    }
                }
            }

            group_weight_entries
        } else {
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

                        for &(sig_id, ref arc) in contributions {
                            for &group_idx in &signature_groups[sig_id] {
                                if group_counts[group_idx] == 0 {
                                    touched_groups.push(group_idx);
                                }
                                group_counts[group_idx] += 1;
                                for range in arc.ranges() {
                                    group_ranges[group_idx]
                                        .push((*range.start(), *range.end()));
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
                                shared_rangeset_from_unsorted_pairs(
                                    group_ranges[group_idx].as_slice(),
                                )
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
            group_weight_entries
        };
    let per_tsid_group_entries_ms =
        per_tsid_group_entries_started_at.elapsed().as_secs_f64() * 1000.0;
    let group_weight_entries_ms = 0.0;

    let group_results: Vec<Option<(Vec<usize>, Weight)>> = unique_groups
        .iter()
        .enumerate()
        .map(|(group_idx, tids)| {
            let weight_entries = &group_weight_entries[group_idx];
            if weight_entries.is_empty() {
                return None;
            }
            let weight = Weight::from_per_tsid_shared(
                weight_entries.iter().map(|(t, a)| (*t, Arc::clone(a))),
            );
            if weight.is_empty() {
                return None;
            }
            Some((tids.clone(), weight))
        })
        .collect();

    // Sequential DWA construction from grouped results
    let dwa_build_started_at = Instant::now();
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
    let dwa_build_ms = dwa_build_started_at.elapsed().as_secs_f64() * 1000.0;

    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l1_terminal_assembly] start_collect_ms={:.3} token_intern_ms={:.3} full_ranges_ms={:.3} terminal_group_ms={:.3} contribution_seed_ms={:.3} per_tsid_group_entries_ms={:.3} group_weight_entries_ms={:.3} dwa_build_ms={:.3} total_direct_ms={:.3}",
            start_state_collect_ms,
            token_set_intern_ms,
            full_ranges_ms,
            terminal_group_ms,
            contribution_seed_ms,
            per_tsid_group_entries_ms,
            group_weight_entries_ms,
            dwa_build_ms,
            merge_ms,
        );
    }
    let direct_terminal_dwa_ms = merge_ms;
    let distribute_ms = distribute_started_at.elapsed().as_secs_f64() * 1000.0;
    let vocab_tree_traversal_ms = traversal_ms;

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
    for state_idx in 0..tokenizer.num_states() as usize {
        let base = state_idx * 256;
        for (byte, target) in tokenizer.transitions_from(state_idx as u32) {
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
    sorted_entries: &[(u32, Arc<[u8]>)],
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
            let token_bytes = sorted_entries[internal_token_id].1.as_ref();
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

fn shared_rangeset_from_unsorted_pairs(ranges: &[(u32, u32)]) -> Option<Arc<RangeSetBlaze<u32>>> {
    if ranges.is_empty() {
        return None;
    }

    let mut merged = ranges.to_vec();
    merge_ranges_in_place(&mut merged);
    Some(shared_rangeset(
        merged.iter().map(|&(start, end)| start..=end).collect(),
    ))
}

fn build_end_rep_groups(
    unique_groups: &[Vec<usize>],
    terminal_to_active_states: &[Vec<u32>],
    num_end_reps: usize,
) -> Vec<Vec<usize>> {
    let mut groups_by_end_rep = vec![Vec::new(); num_end_reps];
    for (group_idx, tids) in unique_groups.iter().enumerate() {
        for &state in &terminal_to_active_states[tids[0]] {
            groups_by_end_rep[state as usize].push(group_idx);
        }
    }
    groups_by_end_rep
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
    initial_states_considered: usize,
    max_length_skipped: bool,
    max_token_len: usize,
    token_len_gt_4: usize,
    token_len_gt_8: usize,
    token_len_gt_16: usize,
    token_len_gt_32: usize,
    token_len_gt_64: usize,
    state_equiv_ms: f64,
    max_length_state_equiv_ms: f64,
    exact_state_equiv_ms: f64,
    max_length_reps: usize,
    exact_reps: usize,
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
mod packed_suffix_product_tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex;

    #[test]
    fn packed_suffix_profiles_match_batched_profiles() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Choice(vec![
                Expr::U8Seq(b"ac".to_vec()),
                Expr::U8Seq(b"ba".to_vec()),
            ]),
            Expr::U8Seq(b"cab".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let sorted_entries: Vec<(u32, Arc<[u8]>)> = vec![
            (0, Arc::from(&b""[..])),
            (1, Arc::from(&b"a"[..])),
            (2, Arc::from(&b"ab"[..])),
            (3, Arc::from(&b"abc"[..])),
            (4, Arc::from(&b"abd"[..])),
            (5, Arc::from(&b"ac"[..])),
            (6, Arc::from(&b"b"[..])),
            (7, Arc::from(&b"ba"[..])),
            (8, Arc::from(&b"bb"[..])),
            (9, Arc::from(&b"c"[..])),
            (10, Arc::from(&b"cab"[..])),
        ];
        let buckets = build_l1_sorted_token_buckets(&sorted_entries);
        let active_terminals = vec![true, false, true, true];
        let (state_to_terminal_signature, _) =
            build_l1_state_to_terminal_signatures(&tokenizer, &active_terminals);
        let flat_trans = build_flat_transition_table(&tokenizer);
        let targets: Vec<u32> = (0..tokenizer.num_states()).collect();

        for first_byte in 0..256usize {
            let token_ids = &buckets.token_indices_by_first_byte[first_byte];
            if token_ids.is_empty() {
                continue;
            }
            let mut expected = l1_bucket_suffix_signature_profiles_batched(
                first_byte as u8,
                &targets,
                &sorted_entries,
                token_ids,
                &buckets.suffix_lcps_by_first_byte[first_byte],
                &buckets.suffix_subtree_bytes[first_byte],
                &buckets.suffix_first_bytes_by_bucket[first_byte],
                  buckets.has_empty_suffix_by_bucket[first_byte],
                  &state_to_terminal_signature,
                  &flat_trans,
            );
            let mut actual = l1_bucket_suffix_signature_profiles_packed(
                first_byte as u8,
                &targets,
                &sorted_entries,
                token_ids,
                &buckets.suffix_lcps_by_first_byte[first_byte],
                &buckets.suffix_subtree_bytes[first_byte],
                &buckets.suffix_first_bytes_by_bucket[first_byte],
                buckets.has_empty_suffix_by_bucket[first_byte],
                  &state_to_terminal_signature,
                  &flat_trans,
                  None,
                  tokenizer.num_states() as usize,
            );
            expected.sort_unstable_by_key(|(key, _)| *key);
            actual.sort_unstable_by_key(|(key, _)| *key);
            let actual: Vec<((u8, u32), Vec<(u32, u32, u32)>)> = actual
                .into_iter()
                .map(|(key, profile)| (key, profile.as_ref().to_vec()))
                .collect();
            assert_eq!(actual, expected, "first byte {first_byte}");
        }
    }

    #[test]
    fn suffix_trie_bottom_up_ranges_cover_prefix_and_siblings() {
        let entries: Vec<(u32, Arc<[u8]>)> = vec![
            (0, Arc::from(b"a".as_slice())),
            (1, Arc::from(b"ab".as_slice())),
            (2, Arc::from(b"ac".as_slice())),
            (3, Arc::from(b"b".as_slice())),
        ];
        let trie = L1PackedSuffixTrie::build(&entries, &[0, 1, 2], &[0, 0, 0]);
        assert_eq!((trie.nodes[0].subtree_start, trie.nodes[0].subtree_end), (0, 2));
        let first_child = trie.nodes[0].first_child as usize;
        let second_child = trie.nodes[first_child].next_sibling as usize;
        assert_eq!((trie.nodes[first_child].subtree_start, trie.nodes[first_child].subtree_end), (1, 1));
        assert_eq!((trie.nodes[second_child].subtree_start, trie.nodes[second_child].subtree_end), (2, 2));
    }

    #[test]
    fn sparse_end_rep_groups_match_terminal_membership() {
        let groups = vec![vec![0usize, 2usize], vec![1usize], vec![3usize]];
        let terminal_to_end_reps = vec![vec![0u32, 2], vec![1], vec![0, 2], vec![3]];
        assert_eq!(
            build_end_rep_groups(&groups, &terminal_to_end_reps, 4),
            vec![vec![0], vec![1], vec![0], vec![2]],
        );
    }

    #[test]
    fn frozen_walk_profiles_preserve_signature_ranges() {
        let empty: Arc<[(u32, u32, u32)]> = Arc::from([]);
        let profile_one: Arc<[(u32, u32, u32)]> = Arc::from([
            (1, 2, 3),
            (2, 4, 4),
            (1, 7, 8),
            (0, 9, 10),
        ]);
        let reuse = L1ExactProfileReuse {
            target_to_profile_id: [((b'a', 17), 1u32)].into_iter().collect(),
            walk_profiles_by_id: vec![
                freeze_l1_walk_profile(&empty),
                freeze_l1_walk_profile(&profile_one),
            ],
            direct_terminal_signatures: Arc::from([]),
            direct_state_to_terminal_signature: Arc::from([]),
        };
        let cache = reuse.materialize_walk_cache();
        let profile = cache.get(&(b'a', 17)).expect("profile present");
        let grouped: Vec<(u32, Vec<(u32, u32)>)> = profile
            .iter()
            .map(|(signature, ranges)| (*signature, ranges.as_ref().to_vec()))
            .collect();
        assert_eq!(grouped, vec![(0, vec![(2, 3), (7, 8)]), (1, vec![(4, 4)])]);
    }

    #[test]
    fn exact_state_hash_partition_matches_direct_token_profiles() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Choice(vec![Expr::U8Seq(b"ac".to_vec()), Expr::U8Seq(b"ba".to_vec())]),
            Expr::U8Seq(b"cab".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let vocab = Vocab::new(
            vec![
                (0, b"".to_vec()),
                (1, b"a".to_vec()),
                (2, b"ab".to_vec()),
                (3, b"abc".to_vec()),
                (4, b"abd".to_vec()),
                (5, b"ac".to_vec()),
                (6, b"b".to_vec()),
                (7, b"ba".to_vec()),
                (8, b"bb".to_vec()),
                (9, b"c".to_vec()),
                (10, b"cab".to_vec()),
            ],
            None,
        );
        let active_terminals = vec![true, false, true, true];
        let order = l1_identity_vocab_order(&vocab);
        let flat_trans = build_flat_transition_table(&tokenizer);
        let states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();
          let (mapping, _) = find_l1_exact_state_equivalence_by_token_signatures(
            &tokenizer,
            &order,
            &states,
              &active_terminals,
              &flat_trans,
                None,
            );
        let num_states = tokenizer.num_states() as usize;
        let mut transitions_by_byte = vec![u32::MAX; num_states * 256];
        for state in 0..num_states {
            for byte in 0..256usize {
                transitions_by_byte[byte * num_states + state] = flat_trans[state * 256 + byte];
            }
        }
        let (transposed_mapping, _) = find_l1_exact_state_equivalence_by_token_signatures(
            &tokenizer,
            &order,
            &states,
            &active_terminals,
            &flat_trans,
            Some(&transitions_by_byte),
        );
        assert_eq!(transposed_mapping, mapping);
        let (state_to_signature, _) =
            build_l1_state_to_terminal_signatures(&tokenizer, &active_terminals);
        let profiles: Vec<Vec<(u32, u32, u32)>> = states
            .iter()
            .map(|&state| {
                l1_token_signature_profile_for_state(
                    state as u32,
                    order.token_entries_sorted.as_ref(),
                    &order.token_buckets,
                    &state_to_signature,
                    &flat_trans,
                )
            })
            .collect();

        for left in 0..states.len() {
            for right in 0..states.len() {
                assert_eq!(
                    mapping[left] == mapping[right],
                    profiles[left] == profiles[right],
                    "state pair ({left}, {right})"
                );
            }
        }
    }

}
