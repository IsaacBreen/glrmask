use crate::automata::lexer::Lexer;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::OnceLock;
use std::time::Instant;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use super::state_equivalence::global_token_position::GlobalTokenPositionStatePartition;
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::ignore_transparent_disallowed_follows;
use super::state_equivalence::{
    build_state_map_from_subset_representatives, resolve_l2p_pipeline_config,
    run_state_equivalence_pipeline, StateEquivalenceScope,
};
use crate::ds::bitset::BitSet;
use super::compat::TokenizerView;
use super::disallowed_follows::normalize_disallowed_follows;
use super::shared::{
    TokenDedup,
    expand_vocab_classes,
    hash_byte_class_seq,
    tokenizer_group_count,
};
use super::state::fast as state_equivalence_analysis;
use super::vocab::fast as vocab_equivalence_analysis;

fn deduplicate_tokens_by_byte_class<'a, S: AsRef<[u8]>>(
    tokens: &'a [S],
    byte_to_class: &[u8; 256],
) -> TokenDedup<'a> {
    let mut hash_to_repr = HashMap::with_capacity(tokens.len() / 2);
    let mut representative_token_bytes = Vec::new();
    let mut original_to_repr = Vec::with_capacity(tokens.len());

    for token in tokens {
        let bytes = token.as_ref();
        let repr_idx = *hash_to_repr
            .entry(hash_byte_class_seq(bytes, byte_to_class))
            .or_insert_with(|| {
                let idx = representative_token_bytes.len();
                representative_token_bytes.push(bytes);
                idx
            });
        original_to_repr.push(repr_idx);
    }

    TokenDedup {
        representative_token_bytes,
        original_to_repr,
    }
}

fn build_state_map(
    state_classes: &BTreeSet<BTreeSet<usize>>,
    num_dfa_states: usize,
) -> ManyToOneIdMap {
    let mut original_to_internal = vec![u32::MAX; num_dfa_states];
    let mut internal_to_originals = Vec::new();
    let mut representative_original_ids = Vec::new();

    for class in state_classes {
        let internal_id = internal_to_originals.len() as u32;
        let originals: Vec<u32> = class.iter().map(|&state| state as u32).collect();
        for &state in &originals {
            original_to_internal[state as usize] = internal_id;
        }
        representative_original_ids
            .push(*originals.first().expect("state class must be non-empty"));
        internal_to_originals.push(originals);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn build_state_map_composed(
    state_classes: &BTreeSet<BTreeSet<usize>>,
    num_dfa_states: usize,
    initial_state_map: &ManyToOneIdMap,
) -> ManyToOneIdMap {
    let mut rep_to_new_internal = vec![u32::MAX; num_dfa_states];
    let mut new_internal_to_originals: Vec<Vec<u32>> = Vec::new();
    for class in state_classes {
        let internal_id = new_internal_to_originals.len() as u32;
        let originals: Vec<u32> = class.iter().map(|&state| state as u32).collect();
        for &state in &originals {
            rep_to_new_internal[state as usize] = internal_id;
        }
        new_internal_to_originals.push(originals);
    }

    let mut composed_original_to_internal = vec![u32::MAX; num_dfa_states];
    let mut composed_internal_to_originals = vec![Vec::new(); new_internal_to_originals.len()];
    let mut composed_reps = vec![u32::MAX; new_internal_to_originals.len()];

    for (orig_state, &init_internal) in initial_state_map.original_to_internal.iter().enumerate() {
        if init_internal == u32::MAX
            || (init_internal as usize) >= initial_state_map.representative_original_ids.len()
        {
            continue;
        }
        let init_rep = initial_state_map.representative_original_ids[init_internal as usize] as usize;
        let new_internal = rep_to_new_internal[init_rep];
        if new_internal != u32::MAX {
            composed_original_to_internal[orig_state] = new_internal;
            let bucket = &mut composed_internal_to_originals[new_internal as usize];
            if bucket.is_empty() {
                composed_reps[new_internal as usize] = init_rep as u32;
            }
            bucket.push(orig_state as u32);
        }
    }

    ManyToOneIdMap {
        original_to_internal: composed_original_to_internal,
        internal_to_originals: composed_internal_to_originals,
        representative_original_ids: composed_reps,
    }
}

fn build_vocab_map(
    vocab_classes: &BTreeSet<Vec<usize>>,
    token_ids: &[u32],
    max_token_id: u32,
) -> ManyToOneIdMap {
    let mut ordered_vocab_classes: Vec<(u32, Vec<u32>)> = vocab_classes
        .iter()
        .map(|class| {
            let mut min_tid = u32::MAX;
            let mut originals: Vec<u32> = Vec::with_capacity(class.len());
            for &idx in class {
                let tid = token_ids[idx];
                originals.push(tid);
                if tid < min_tid {
                    min_tid = tid;
                }
            }
            (min_tid, originals)
        })
        .collect();
    ordered_vocab_classes.sort_unstable_by_key(|(rep_tid, _)| *rep_tid);

    let mut original_to_internal = vec![u32::MAX; (max_token_id + 1) as usize];
    let mut internal_to_originals = Vec::with_capacity(ordered_vocab_classes.len());
    let mut representative_original_ids = Vec::with_capacity(ordered_vocab_classes.len());

    for (internal_id, (rep_tid, originals)) in ordered_vocab_classes.into_iter().enumerate() {
        for &token_id in &originals {
            original_to_internal[token_id as usize] = internal_id as u32;
        }
        representative_original_ids.push(rep_tid);
        internal_to_originals.push(originals);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

/// An identity map is an exact conservative token quotient. It is useful for
/// small structural partitions where vocab equivalence has no materialization
/// benefit, while its analysis cost dominates the enclosing partition wall.
fn build_identity_vocab_map(token_ids: &[u32], max_token_id: u32) -> ManyToOneIdMap {
    let mut token_ids = token_ids.to_vec();
    token_ids.sort_unstable();
    token_ids.dedup();

    let mut original_to_internal = vec![u32::MAX; max_token_id as usize + 1];
    let mut internal_to_originals = Vec::with_capacity(token_ids.len());
    let mut representatives = Vec::with_capacity(token_ids.len());
    for (internal, token_id) in token_ids.into_iter().enumerate() {
        original_to_internal[token_id as usize] = internal as u32;
        internal_to_originals.push(vec![token_id]);
        representatives.push(token_id);
    }
    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids: representatives,
    }
}

/// Conservative vocabulary quotient for epsilon tokenizers: only byte-for-byte
/// aliases are merged. They necessarily have identical lexer behaviour, and
/// merging them is required because the vocabulary prefix tree has one leaf
/// per byte string. The final mask mapping expands the class back to every
/// original token id.
fn build_exact_byte_vocab_map(
    token_ids: &[u32],
    token_bytes: &[&[u8]],
    max_token_id: u32,
) -> ManyToOneIdMap {
    debug_assert_eq!(token_ids.len(), token_bytes.len());
    let mut classes = BTreeMap::<Vec<u8>, Vec<u32>>::new();
    for (&token_id, &bytes) in token_ids.iter().zip(token_bytes) {
        classes.entry(bytes.to_vec()).or_default().push(token_id);
    }

    let mut ordered = classes.into_values().collect::<Vec<_>>();
    for class in &mut ordered {
        class.sort_unstable();
    }
    ordered.sort_unstable_by_key(|class| class[0]);

    let mut original_to_internal = vec![u32::MAX; max_token_id as usize + 1];
    let mut representatives = Vec::with_capacity(ordered.len());
    for (internal, class) in ordered.iter().enumerate() {
        representatives.push(class[0]);
        for &original in class {
            original_to_internal[original as usize] = internal as u32;
        }
    }
    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals: ordered,
        representative_original_ids: representatives,
    }
}

struct PreparedEquivalenceInputs<'a> {
    max_token_id: u32,
    token_ids: Vec<u32>,
    token_bytes: Vec<&'a [u8]>,
    initial_states: Vec<usize>,
}

fn prepare_equivalence_inputs<'a>(
    tokenizer: &Tokenizer,
    vocab: &'a Vocab,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> PreparedEquivalenceInputs<'a> {
    let max_token_id = vocab.max_token_id();
    let mut token_bytes: Vec<&[u8]> = Vec::with_capacity(vocab.len());
    let mut token_ids: Vec<u32> = Vec::with_capacity(vocab.len());
    for (&tid, bytes) in vocab.entries.iter() {
        token_ids.push(tid);
        token_bytes.push(bytes.as_slice());
    }

    let initial_states = match initial_state_map {
        Some(map) => map.representative_original_ids.iter().map(|&s| s as usize).collect(),
        None => (0..tokenizer.num_states() as usize).collect(),
    };

    PreparedEquivalenceInputs {
        max_token_id,
        token_ids,
        token_bytes,
        initial_states,
    }
}


struct CombinedEquivalenceResult {
    vocab_classes: BTreeSet<Vec<usize>>,
    state_classes: BTreeSet<BTreeSet<usize>>,
}

pub(crate) struct CombinedEquivalenceProfile {
    pub(crate) initial_states_considered: usize,
    pub(crate) max_length_skipped: bool,
    pub(crate) max_token_len: usize,
    pub(crate) token_len_gt_4: usize,
    pub(crate) token_len_gt_8: usize,
    pub(crate) token_len_gt_16: usize,
    pub(crate) token_len_gt_32: usize,
    pub(crate) token_len_gt_64: usize,
    pub(crate) raw_analysis_base_init_ms: f64,
    pub(crate) analysis_view_build_ms: f64,
    /// Sub-timing within `analysis_view_build_ms`; do not add separately.
    pub(crate) active_mask_filter_ms: f64,
    pub(crate) effective_follows_normalize_ms: f64,
    pub(crate) prepare_inputs_ms: f64,
    pub(crate) byte_class_setup_ms: f64,
    pub(crate) vocab_analysis_dfa_build_ms: f64,
    pub(crate) token_dedup_ms: f64,
    pub(crate) restricted_observation_state_equiv_ms: f64,
    pub(crate) max_length_state_equiv_ms: f64,
    pub(crate) vocab_equiv_ms: f64,
    pub(crate) exact_state_equiv_ms: f64,
    pub(crate) id_map_finalize_ms: f64,
    pub(crate) restricted_observation_reps: usize,
    pub(crate) max_length_reps: usize,
    pub(crate) exact_reps: usize,
    pub(crate) exact_rep_confirmation_used: bool,
}

struct TokenLengthStats {
    gt_4: usize,
    gt_8: usize,
    gt_16: usize,
    gt_32: usize,
    gt_64: usize,
}

fn skip_max_length_for_partition(partition_label: &str) -> bool {
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

#[inline]
fn direct_refinement_work_is_no_larger(
    direct_token_bytes: usize,
    max_token_len: usize,
    active_byte_count: usize,
) -> bool {
    direct_token_bytes <= max_token_len.saturating_mul(active_byte_count)
}

#[inline]
fn should_skip_max_length_for_partition(
    partition_label: &str,
    initial_state_count: usize,
    projected_by_global: bool,
    direct_token_bytes: usize,
    max_token_len: usize,
    active_byte_count: usize,
) -> bool {
    if skip_max_length_for_partition(partition_label)
        || (projected_by_global && initial_state_count <= 8192)
    {
        return true;
    }

    // Both routes are exact. The k-bounded prepass performs up to one
    // state-transition refinement for every (length, active-byte) pair;
    // direct finite-vocabulary refinement must at least read every token byte.
    // Prefer the direct route when its input bound is no larger, avoiding a
    // prepass that cannot amortize its own full-DFA scans.
    direct_refinement_work_is_no_larger(
        direct_token_bytes,
        max_token_len,
        active_byte_count,
    )
}

const EXACT_REP_CONFIRMATION_MIN_STATES: usize = 2_000;
const EXACT_REP_CONFIRMATION_MIN_TOKENS: usize = 200;
const RAW_QUOTIENT_TINY_VOCAB_MAX_TOKENS: usize = 16;
const RAW_QUOTIENT_TINY_VOCAB_MAX_BYTES: usize = 64;
const RAW_QUOTIENT_STRUCTURAL_BOUNDARY_MAX_TOKENS: usize = 48;
const RAW_QUOTIENT_STRUCTURAL_BOUNDARY_MAX_BYTES: usize = 256;

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

/// Compose the raw restricted-observation quotient with its final exact
/// representative map without materializing ordered sets of raw states.
///
/// Scanning raw states in ascending order assigns dense IDs in the same order
/// as the generic `BTreeSet` path: each class is ordered by its first raw
/// member, and that member is also the retained raw representative.
fn compose_raw_quotient_state_map(
    pre_state_map: &ManyToOneIdMap,
    final_representative_for_preclass: &[usize],
) -> ManyToOneIdMap {
    assert_eq!(
        pre_state_map.internal_to_originals.len(),
        final_representative_for_preclass.len(),
        "raw quotient and exact representative map disagree",
    );
    let preclass_count = pre_state_map.internal_to_originals.len();
    let mut final_key_to_internal = vec![u32::MAX; preclass_count];
    let mut original_to_internal = vec![u32::MAX; pre_state_map.original_to_internal.len()];
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representative_original_ids = Vec::<u32>::new();

    for (raw_state, &preclass) in pre_state_map.original_to_internal.iter().enumerate() {
        if preclass == u32::MAX {
            continue;
        }
        let preclass = preclass as usize;
        assert!(preclass < preclass_count, "invalid raw quotient class");
        let final_key = final_representative_for_preclass[preclass];
        assert!(final_key < preclass_count, "invalid exact representative");
        let internal = if final_key_to_internal[final_key] == u32::MAX {
            let next = internal_to_originals.len() as u32;
            final_key_to_internal[final_key] = next;
            internal_to_originals.push(Vec::new());
            representative_original_ids.push(raw_state as u32);
            next
        } else {
            final_key_to_internal[final_key]
        };
        original_to_internal[raw_state] = internal;
        internal_to_originals[internal as usize].push(raw_state as u32);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn build_internal_id_map_from_combined_result(
    tokenizer: &Tokenizer,
    initial_state_map: Option<&ManyToOneIdMap>,
    prepared: &PreparedEquivalenceInputs<'_>,
    result: &CombinedEquivalenceResult,
) -> InternalIdMap {
    let num_dfa_states = tokenizer.num_states() as usize;
    let state_map = match initial_state_map {
        Some(init_map) => build_state_map_composed(&result.state_classes, num_dfa_states, init_map),
        None => build_state_map(&result.state_classes, num_dfa_states),
    };
    let vocab_map = build_vocab_map(&result.vocab_classes, &prepared.token_ids, prepared.max_token_id);

    InternalIdMap {
        tokenizer_states: state_map,
        vocab_tokens: vocab_map,
    }
}

fn try_analyze_equivalences_with_token_boundary_partition(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    effective_disallowed: &BTreeMap<u32, BitSet>,
    active_groups: Option<&[bool]>,
    _shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    _shared_analysis_dfa_cache: Option<&super::vocab::fast::SharedVocabAnalysisDfaCache>,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    token_boundary_partition: &GlobalTokenPositionStatePartition,
    effective_follows_prepare_ms: f64,
    pre_normalized_disallowed_follows: Option<&[BitSet]>,
) -> Option<(InternalIdMap, CombinedEquivalenceProfile)> {
    let active_groups = active_groups?;
    let seed = token_boundary_partition.as_many_to_one();
    let num_states = tokenizer.num_states() as usize;
    if seed.original_to_internal.len() != num_states
        || seed.representative_original_ids.is_empty()
        || vocab.entries.values().any(Vec::is_empty)
    {
        return None;
    }

    let prepare_inputs_started_at = Instant::now();
    let prepared = prepare_equivalence_inputs(tokenizer, vocab, None);
    if prepared.token_bytes.is_empty() {
        return None;
    }
    let token_len_stats = token_length_stats(&prepared.token_bytes);
    let max_token_len = prepared
        .token_bytes
        .iter()
        .map(|token| token.len())
        .max()
        .unwrap_or(0);
    let prepare_inputs_ms = prepare_inputs_started_at.elapsed().as_secs_f64() * 1000.0;

    let compatible_flat_trans = flat_trans.filter(|transitions| {
        transitions.len() == tokenizer.num_states() as usize * 256
    });
    let analysis_view_started_at = Instant::now();
    let tokenizer_view = match compatible_flat_trans {
        Some(transitions) => TokenizerView::new_filtered_from_flat_trans(
            transitions,
            tokenizer,
            active_groups,
        ),
        None => TokenizerView::new_filtered(tokenizer, active_groups),
    };
    let analysis_view_build_ms = analysis_view_started_at.elapsed().as_secs_f64() * 1000.0;

    // The C-seeded state/vocab equivalence walks only ever follow the
    // partition's token bytes, so byte classes only need to be exact for the
    // bytes that actually appear in the vocabulary. Restricting the byte-class
    // scan to those relevant bytes makes base construction proportional to the
    // partition's tiny alphabet instead of the full 18k-state x 256-byte table.
    // The base is partition-local (its byte classes depend on this partition's
    // relevant bytes), so it uses local caches rather than the cross-partition
    // shared ones to avoid contaminating other partitions.
    let mut relevant_bytes = [false; 256];
    for token in &prepared.token_bytes {
        for &byte in token.iter() {
            relevant_bytes[byte as usize] = true;
        }
    }
    let local_vocab_dfa_cache = vocab_equivalence_analysis::SharedVocabDfaCache::new();
    let local_analysis_dfa_cache =
        vocab_equivalence_analysis::SharedVocabAnalysisDfaCache::default();
    let compatible_shared_base = Some(local_vocab_dfa_cache.get_or_init(|| {
        vocab_equivalence_analysis::SharedVocabDfaBase::build_from_dfa_relevant(
            tokenizer_view.dfa(),
            &relevant_bytes,
        )
    }));

    let follows_normalize_started_at = Instant::now();
    let owned_normalized_disallowed_follows;
    let normalized_disallowed_follows = if let Some(rows) = pre_normalized_disallowed_follows {
        rows
    } else {
        owned_normalized_disallowed_follows =
            normalize_disallowed_follows(tokenizer_group_count(&tokenizer_view), effective_disallowed);
        &owned_normalized_disallowed_follows
    };
    let effective_follows_normalize_ms = effective_follows_prepare_ms
        + follows_normalize_started_at.elapsed().as_secs_f64() * 1000.0;

    let byte_class_started_at = Instant::now();
    let byte_to_class = compatible_shared_base
        .map(vocab_equivalence_analysis::SharedVocabDfaBase::byte_to_class)
        .unwrap_or_else(|| super::compat::compute_byte_classes(tokenizer_view.dfa()));
    let byte_class_setup_ms = byte_class_started_at.elapsed().as_secs_f64() * 1000.0;
    let token_dedup_started_at = Instant::now();
    let dedup = deduplicate_tokens_by_byte_class(&prepared.token_bytes, &byte_to_class);
    let token_dedup_ms = token_dedup_started_at.elapsed().as_secs_f64() * 1000.0;

    let seed_states = seed
        .representative_original_ids
        .iter()
        .map(|&state| state as usize)
        .collect::<Vec<_>>();
    if seed_states.iter().any(|&state| state >= num_states) {
        return None;
    }
    let exact_rep_confirmation_used = seed_states.len() >= EXACT_REP_CONFIRMATION_MIN_STATES
        && dedup.representative_token_bytes.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
    let exact_started_at = Instant::now();
    let state_representatives = if exact_rep_confirmation_used {
        state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
            &tokenizer_view,
            &dedup.representative_token_bytes,
            &seed_states,
            &normalized_disallowed_follows,
            None,
            None,
            Some(true),
            compatible_shared_base,
        )
    } else {
        state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
            &tokenizer_view,
            &dedup.representative_token_bytes,
            &seed_states,
            &normalized_disallowed_follows,
            compatible_shared_base,
        )
    };
    let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;

    let tokenizer_states = build_state_map_from_subset_representatives(
        &seed_states,
        &state_representatives,
        num_states,
        Some(seed),
    );
    let mut final_state_representatives = tokenizer_states
        .representative_original_ids
        .iter()
        .map(|&state| state as usize)
        .collect::<Vec<_>>();
    final_state_representatives.sort_unstable();
    final_state_representatives.dedup();

    let vocab_equiv_started_at = Instant::now();
    let (vocab_classes, vocab_analysis_dfa_build_ms) = if partition_label == "p8" {
        // P8's L2P vocabulary is deliberately tiny and its current exact
        // analysis produces an identity token partition. Identity is exact in
        // every schema, so avoid constructing a large analysis DFA merely to
        // rediscover that no token aliases are useful on this partition.
        (
            (0..prepared.token_ids.len())
                .map(|token_index| vec![token_index])
                .collect(),
            0.0,
        )
    } else {
        let (dedup_vocab_classes, build_ms) =
            vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
                &tokenizer_view,
                &dedup.representative_token_bytes,
                &final_state_representatives,
                effective_disallowed,
                Some(&byte_to_class),
                None,
                Some(&local_vocab_dfa_cache),
                Some(&local_analysis_dfa_cache),
            );
        (
            expand_vocab_classes(
                dedup_vocab_classes,
                &dedup.original_to_repr,
                dedup.representative_token_bytes.len(),
            ),
            build_ms,
        )
    };
    let vocab_equiv_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;

    let finalize_started_at = Instant::now();
    let internal_id_map = InternalIdMap {
        tokenizer_states,
        vocab_tokens: build_vocab_map(&vocab_classes, &prepared.token_ids, prepared.max_token_id),
    };
    let id_map_finalize_ms = finalize_started_at.elapsed().as_secs_f64() * 1000.0;
    let exact_reps = internal_id_map.tokenizer_states.num_internal_ids() as usize;

    Some((
        internal_id_map,
        CombinedEquivalenceProfile {
            initial_states_considered: seed_states.len(),
            max_length_skipped: true,
            max_token_len,
            token_len_gt_4: token_len_stats.gt_4,
            token_len_gt_8: token_len_stats.gt_8,
            token_len_gt_16: token_len_stats.gt_16,
            token_len_gt_32: token_len_stats.gt_32,
            token_len_gt_64: token_len_stats.gt_64,
            raw_analysis_base_init_ms: 0.0,
            analysis_view_build_ms,
            active_mask_filter_ms: 0.0,
            effective_follows_normalize_ms,
            prepare_inputs_ms,
            byte_class_setup_ms,
            vocab_analysis_dfa_build_ms,
            token_dedup_ms,
            restricted_observation_state_equiv_ms: 0.0,
            max_length_state_equiv_ms: 0.0,
            vocab_equiv_ms,
            exact_state_equiv_ms,
            id_map_finalize_ms,
            restricted_observation_reps: seed_states.len(),
            max_length_reps: seed_states.len(),
            exact_reps,
            exact_rep_confirmation_used,
        },
    ))
}

fn try_analyze_equivalences_with_raw_quotient(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    effective_disallowed: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    active_groups: Option<&[bool]>,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    initial_state_map: Option<&ManyToOneIdMap>,
    effective_follows_prepare_ms: f64,
    precomputed_raw_observations: Option<(&[u32], &[u32])>,
) -> Option<(InternalIdMap, CombinedEquivalenceProfile)> {
    let active_groups = active_groups?;
    let flat_trans = flat_trans?;
    if flat_trans.len() != tokenizer.num_states() as usize * 256 {
        return None;
    }

    let prepare_inputs_started_at = Instant::now();
    let prepared = prepare_equivalence_inputs(tokenizer, vocab, initial_state_map);
    let token_len_stats = token_length_stats(&prepared.token_bytes);
    let prepare_inputs_ms = prepare_inputs_started_at.elapsed().as_secs_f64() * 1000.0;

    // The direct raw refinement sees every byte that can occur in this local
    // vocabulary. It is deliberately at least as fine as the later deduped
    // quotient analysis.
    let mut raw_relevant_bytes = [false; 256];
    let mut direct_token_bytes = 0usize;
    let mut max_token_len = 0usize;
    for token in &prepared.token_bytes {
        direct_token_bytes += token.len();
        max_token_len = max_token_len.max(token.len());
        for &byte in *token {
            raw_relevant_bytes[byte as usize] = true;
        }
    }
    let active_byte_count = raw_relevant_bytes.iter().filter(|&&active| active).count();
    let projected_by_global = prepared.initial_states.len() < tokenizer.num_states() as usize;
    let skip_raw_quotient = should_skip_max_length_for_partition(
        partition_label,
        prepared.initial_states.len(),
        projected_by_global,
        direct_token_bytes,
        max_token_len,
        active_byte_count,
    );
    // The generic route still performs restricted observation, but for a tiny
    // local vocabulary it then keeps the full raw analysis DFA alive for the
    // exact state and token phases.  Materializing the exact restricted raw
    // quotient is cheaper in that regime: it avoids the 18k-state shared-base
    // setup and lets both remaining phases operate on the quotient.  Keep the
    // bound deliberately narrow; larger short-token partitions (notably p7)
    // can retain too much raw topology for this to win.
    let tiny_raw_quotient = prepared.token_bytes.len() <= RAW_QUOTIENT_TINY_VOCAB_MAX_TOKENS
        && direct_token_bytes <= RAW_QUOTIENT_TINY_VOCAB_MAX_BYTES;
    let structural_boundary_raw_quotient = matches!(partition_label, "p7" | "p8")
        && prepared.token_bytes.len() <= RAW_QUOTIENT_STRUCTURAL_BOUNDARY_MAX_TOKENS
        && direct_token_bytes <= RAW_QUOTIENT_STRUCTURAL_BOUNDARY_MAX_BYTES;
    if skip_raw_quotient && !tiny_raw_quotient && !structural_boundary_raw_quotient {
        return None;
    }

    // P8's local vocabulary has one common leading quote byte. For this one
    // partition, the full behavior of a source state on every local token is
    // determined by its quote successor. Evaluate one source representative
    // per distinct successor, then lift the exact result back to all raw
    // sources. This is a conservative factorization: equal quote successors
    // have identical complete token paths by definition.
    if partition_label == "p8" && super::super::p8_first_byte_factorization_allowed() {
        let common_first = prepared
            .token_bytes
            .first()
            .and_then(|token| token.first())
            .copied()?;
        if prepared
            .token_bytes
            .iter()
            .any(|token| token.first().copied() != Some(common_first))
        {
            return None;
        }

        // Every production P8 token has a nonempty identifier prefix after
        // the quote. Model terminal matches at the quote successor as
        // zero-position events, then analyze only the remaining suffix.
        // A one-byte quote token invalidates that factorization: its acceptance
        // is observed at the quote successor itself, not after a nonempty
        // suffix. Let the generic exact route handle such atypical partitions.
        let use_seeded_suffix_factorization =
            prepared.token_bytes.iter().all(|token| token.len() > 1);
        if !use_seeded_suffix_factorization {
            return None;
        }
        let mut target_to_source = BTreeMap::<u32, usize>::new();
        let mut source_representative = vec![usize::MAX; tokenizer.num_states() as usize];
        let mut target_by_source = vec![u32::MAX; tokenizer.num_states() as usize];
        for source in 0..tokenizer.num_states() as usize {
            let target = flat_trans[source * 256 + common_first as usize];
            if target == u32::MAX {
                continue;
            }
            let representative = *target_to_source.entry(target).or_insert(source);
            source_representative[source] = representative;
            target_by_source[source] = target;
        }
        let representative_sources: Vec<usize> = target_to_source.values().copied().collect();
        let quote_targets: Vec<usize> = target_to_source.keys().map(|&target| target as usize).collect();
        let suffix_tokens: Vec<&[u8]> = prepared
            .token_bytes
            .iter()
            .map(|token| &token[1..])
            .collect();
        debug_assert!(suffix_tokens.iter().all(|suffix| !suffix.is_empty()));
        let view_started_at = Instant::now();
        let analysis_view = TokenizerView::new_filtered_from_flat_trans(
            flat_trans,
            tokenizer,
            active_groups,
        );
        let analysis_view_build_ms = view_started_at.elapsed().as_secs_f64() * 1000.0;
        let exact_started_at = Instant::now();
        let representative_state_reps = if use_seeded_suffix_factorization {
            state_equivalence_analysis::find_state_equivalence_classes_with_sparse_disallowed_and_raw_transitions_with_initial_finalizers(
                &analysis_view,
                &suffix_tokens,
                &quote_targets,
                effective_disallowed,
            )
        } else {
            state_equivalence_analysis::find_state_equivalence_classes_with_sparse_disallowed_and_raw_transitions(
                &analysis_view,
                &prepared.token_bytes,
                &representative_sources,
                effective_disallowed,
            )
        };
        let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
        let mut source_rep_to_behavior_rep = HashMap::<usize, usize>::new();
        if use_seeded_suffix_factorization {
            for (&target, &behavior_rep) in quote_targets.iter().zip(&representative_state_reps) {
                source_rep_to_behavior_rep.insert(target, behavior_rep);
            }
        } else {
            for (&source, &behavior_rep) in representative_sources
                .iter()
                .zip(&representative_state_reps)
            {
                source_rep_to_behavior_rep.insert(source, behavior_rep);
            }
        }

        let id_map_finalize_started_at = Instant::now();
        let mut behavior_rep_to_internal = HashMap::<Option<usize>, u32>::new();
        let mut original_to_internal = vec![u32::MAX; tokenizer.num_states() as usize];
        let mut internal_to_originals = Vec::<Vec<u32>>::new();
        let mut representative_original_ids = Vec::<u32>::new();
        for source in 0..tokenizer.num_states() as usize {
            let behavior_rep = if use_seeded_suffix_factorization {
                (target_by_source[source] != u32::MAX)
                    .then(|| source_rep_to_behavior_rep[&(target_by_source[source] as usize)])
            } else {
                (source_representative[source] != usize::MAX).then(|| {
                    source_rep_to_behavior_rep[&source_representative[source]]
                })
            };
            let next = internal_to_originals.len() as u32;
            let internal = *behavior_rep_to_internal.entry(behavior_rep).or_insert_with(|| {
                internal_to_originals.push(Vec::new());
                representative_original_ids.push(source as u32);
                next
            });
            original_to_internal[source] = internal;
            internal_to_originals[internal as usize].push(source as u32);
        }
        let tokenizer_states = ManyToOneIdMap {
            original_to_internal,
            internal_to_originals,
            representative_original_ids,
        };
        let exact_reps = tokenizer_states.num_internal_ids() as usize;
        let internal_id_map = InternalIdMap {
            tokenizer_states,
            vocab_tokens: build_identity_vocab_map(&prepared.token_ids, prepared.max_token_id),
        };
        let id_map_finalize_ms = id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;
        if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
            eprintln!(
                "[glrmask/profile][p8_first_byte_factorization] first_byte={} quote_targets={} seeded_suffix_factorized={} exact_reps={} view_ms={:.3} state_equiv_ms={:.3}",
                common_first,
                representative_sources.len(),
                use_seeded_suffix_factorization,
                exact_reps,
                analysis_view_build_ms,
                exact_state_equiv_ms,
            );
        }
        return Some((
            internal_id_map,
            CombinedEquivalenceProfile {
                initial_states_considered: prepared.initial_states.len(),
                max_length_skipped: false,
                max_token_len,
                token_len_gt_4: token_len_stats.gt_4,
                token_len_gt_8: token_len_stats.gt_8,
                token_len_gt_16: token_len_stats.gt_16,
                token_len_gt_32: token_len_stats.gt_32,
                token_len_gt_64: token_len_stats.gt_64,
                raw_analysis_base_init_ms: 0.0,
                analysis_view_build_ms,
                active_mask_filter_ms: 0.0,
                effective_follows_normalize_ms: effective_follows_prepare_ms,
                prepare_inputs_ms,
                byte_class_setup_ms: 0.0,
                vocab_analysis_dfa_build_ms: 0.0,
                token_dedup_ms: 0.0,
                restricted_observation_state_equiv_ms: 0.0,
                max_length_state_equiv_ms: 0.0,
                vocab_equiv_ms: 0.0,
                exact_state_equiv_ms,
                id_map_finalize_ms,
                restricted_observation_reps: representative_sources.len(),
                max_length_reps: representative_sources.len(),
                exact_reps,
                exact_rep_confirmation_used: false,
            },
        ));
    }

    let restricted_started_at = Instant::now();
    let raw_restricted = super::state_equivalence::restricted_observation::compute_state_map_raw(
        tokenizer,
        flat_trans,
        active_groups,
        &raw_relevant_bytes,
        precomputed_raw_observations,
    )?;
    let restricted_observation_state_equiv_ms =
        restricted_started_at.elapsed().as_secs_f64() * 1000.0;
    if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
        eprintln!(
            "[glrmask/profile][raw_restricted_observation] partition={} labels_ms={:.3} refine_ms={:.3} certificate_ms={:.3} rounds={} reps={}",
            partition_label,
            raw_restricted.label_ms,
            raw_restricted.refine_ms,
            raw_restricted.certificate_ms,
            raw_restricted.rounds,
            raw_restricted.state_map.num_internal_ids(),
        );
    }
    let raw_observation_ids = raw_restricted.raw_observation_ids;
    let observation_representatives = raw_restricted.observation_representatives;
    let pre_state_map = raw_restricted.state_map;

    let use_compact_quotient_view = partition_label == "p7"
        && !super::super::l2p_terminal_interchangeability_strict_reference_enabled();
    let view_build_started_at = Instant::now();
    let (analysis_view, prebuilt_local_base, active_mask_filter_ms) = if use_compact_quotient_view {
        if let Some((view, base, build_ms)) =
            TokenizerView::new_filtered_quotient_from_flat_trans_with_observation_cache_and_relevant_base(
                flat_trans,
                tokenizer,
                active_groups,
                &pre_state_map,
                &raw_observation_ids,
                &observation_representatives,
                &raw_relevant_bytes,
            )
        {
            (view, Some(base), build_ms)
        } else {
            let (view, build_ms) =
                TokenizerView::new_filtered_quotient_from_flat_trans_with_observation_cache(
                    flat_trans,
                    tokenizer,
                    active_groups,
                    &pre_state_map,
                    &raw_observation_ids,
                    &observation_representatives,
                );
            (view, None, build_ms)
        }
    } else {
        let (view, build_ms) =
            TokenizerView::new_filtered_quotient_from_flat_trans_with_observation_cache(
                flat_trans,
                tokenizer,
                active_groups,
                &pre_state_map,
                &raw_observation_ids,
                &observation_representatives,
            );
        (view, None, build_ms)
    };
    let analysis_view_build_ms = view_build_started_at.elapsed().as_secs_f64() * 1000.0;

    // This exact byte partition is now over at most a few hundred quotient
    // rows. It replaces the previous 18k-state shared-base cold setup.
    let byte_class_setup_started_at = Instant::now();
    let local_shared_base = prebuilt_local_base.or_else(|| {
        super::vocab::fast::SharedVocabDfaBase::build_from_dfa_for_relevant_bytes(
            analysis_view.dfa(),
            &raw_relevant_bytes,
        )
    });
    let byte_to_class = local_shared_base
        .as_ref()
        .map(|base| base.byte_to_class())
        .unwrap_or_else(|| super::compat::compute_byte_classes(analysis_view.dfa()));
    let byte_class_setup_ms = byte_class_setup_started_at.elapsed().as_secs_f64() * 1000.0;

    let token_dedup_started_at = Instant::now();
    let dedup = deduplicate_tokens_by_byte_class(&prepared.token_bytes, &byte_to_class);
    let token_dedup_ms = token_dedup_started_at.elapsed().as_secs_f64() * 1000.0;

    // In strict-reference mode the primary build retains the old cloned dense
    // representation. The recursive reference runs with the guard suppressed
    // and therefore exercises the production borrowed/sparse form, so the
    // existing terminal-DWA equality check crosses the representations.
    let strict_reference_mode =
        super::super::l2p_terminal_interchangeability_strict_reference_enabled();
    let use_sparse_follow_rows = partition_label == "p8"
        && !strict_reference_mode;
    let use_borrowed_follow_rows = partition_label == "p7" && !strict_reference_mode;
    let follows_normalize_started_at = Instant::now();
    let group_count = tokenizer_group_count(&analysis_view);
    let normalized_disallowed_follows = (!use_sparse_follow_rows && !use_borrowed_follow_rows).then(|| {
        normalize_disallowed_follows(group_count, effective_disallowed)
    });
    let borrowed_disallowed_follows = use_borrowed_follow_rows.then(|| {
        (0..group_count)
            .map(|terminal| {
                effective_disallowed
                    .get(&(terminal as u32))
                    .filter(|bits| !bits.is_zero())
            })
            .collect::<Vec<_>>()
    });
    let effective_follows_normalize_ms = if use_sparse_follow_rows {
        effective_follows_prepare_ms
    } else {
        effective_follows_prepare_ms
            + follows_normalize_started_at.elapsed().as_secs_f64() * 1000.0
    };
    let pre_reduced_states: Vec<usize> = (0..pre_state_map.internal_to_originals.len()).collect();
    let exact_rep_confirmation_used = pre_reduced_states.len() >= EXACT_REP_CONFIRMATION_MIN_STATES
        && dedup.representative_token_bytes.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
    let exact_started_at = Instant::now();
    let reduced_state_reps_for_pre_reduced = if use_sparse_follow_rows {
        if exact_rep_confirmation_used {
            state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_sparse_disallowed_and_shared_base(
                &analysis_view,
                &dedup.representative_token_bytes,
                &pre_reduced_states,
                effective_disallowed,
                None,
                None,
                Some(true),
                local_shared_base.as_ref(),
            )
        } else {
            state_equivalence_analysis::find_state_equivalence_classes_with_sparse_disallowed_and_shared_base(
                &analysis_view,
                &dedup.representative_token_bytes,
                &pre_reduced_states,
                effective_disallowed,
                local_shared_base.as_ref(),
            )
        }
    } else if use_borrowed_follow_rows && exact_rep_confirmation_used {
        state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_borrowed_disallowed_and_shared_base(
            &analysis_view,
            &dedup.representative_token_bytes,
            &pre_reduced_states,
            borrowed_disallowed_follows
                .as_deref()
                .expect("borrowed follow rows must be present"),
            None,
            None,
            Some(true),
            local_shared_base.as_ref(),
        )
    } else if use_borrowed_follow_rows {
        state_equivalence_analysis::find_state_equivalence_classes_with_borrowed_disallowed_and_shared_base(
            &analysis_view,
            &dedup.representative_token_bytes,
            &pre_reduced_states,
            borrowed_disallowed_follows
                .as_deref()
                .expect("borrowed follow rows must be present"),
            local_shared_base.as_ref(),
        )
    } else if exact_rep_confirmation_used {
        state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
            &analysis_view,
            &dedup.representative_token_bytes,
            &pre_reduced_states,
            normalized_disallowed_follows
                .as_deref()
                .expect("dense follow rows must be present"),
            None,
            None,
            Some(true),
            local_shared_base.as_ref(),
        )
    } else {
        state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
            &analysis_view,
            &dedup.representative_token_bytes,
            &pre_reduced_states,
            normalized_disallowed_follows
                .as_deref()
                .expect("dense follow rows must be present"),
            local_shared_base.as_ref(),
        )
    };
    let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut final_state_representatives = reduced_state_reps_for_pre_reduced.clone();
    final_state_representatives.sort_unstable();
    final_state_representatives.dedup();

    // P8 is the structural-boundary route. Its local source vocabulary is
    // deliberately tiny, while the generic token quotient can cost more than
    // the entire NWA/DWA construction and frequently leaves every local token
    // distinct. Retaining the identity token map is exact by construction: it
    // only declines a possible compression, and leaves the independently exact
    // scanner-state quotient unchanged.
    if matches!(partition_label, "p7" | "p8") {
        let id_map_finalize_started_at = Instant::now();
        let tokenizer_states = compose_raw_quotient_state_map(
            &pre_state_map,
            &reduced_state_reps_for_pre_reduced,
        );
        let exact_reps = tokenizer_states.num_internal_ids() as usize;
        let internal_id_map = InternalIdMap {
            tokenizer_states,
            vocab_tokens: build_identity_vocab_map(&prepared.token_ids, prepared.max_token_id),
        };
        let id_map_finalize_ms = id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;
        return Some((
            internal_id_map,
            CombinedEquivalenceProfile {
                initial_states_considered: prepared.initial_states.len(),
                max_length_skipped: false,
                max_token_len,
                token_len_gt_4: token_len_stats.gt_4,
                token_len_gt_8: token_len_stats.gt_8,
                token_len_gt_16: token_len_stats.gt_16,
                token_len_gt_32: token_len_stats.gt_32,
                token_len_gt_64: token_len_stats.gt_64,
                raw_analysis_base_init_ms: 0.0,
                analysis_view_build_ms,
                active_mask_filter_ms,
                effective_follows_normalize_ms,
                prepare_inputs_ms,
                byte_class_setup_ms,
                vocab_analysis_dfa_build_ms: 0.0,
                token_dedup_ms,
                restricted_observation_state_equiv_ms,
                max_length_state_equiv_ms: 0.0,
                vocab_equiv_ms: 0.0,
                exact_state_equiv_ms,
                id_map_finalize_ms,
                restricted_observation_reps: pre_state_map.num_internal_ids() as usize,
                max_length_reps: pre_state_map.num_internal_ids() as usize,
                exact_reps,
                exact_rep_confirmation_used,
            },
        ));
    }

    let vocab_equiv_started_at = Instant::now();
    let (dedup_vocab_classes, vocab_analysis_dfa_build_ms) =
        vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
            &analysis_view,
            &dedup.representative_token_bytes,
            &final_state_representatives,
            effective_disallowed,
            Some(&byte_to_class),
            None,
            None,
            None,
        );
    let vocab_equiv_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;

    let id_map_finalize_started_at = Instant::now();
    let vocab_classes = expand_vocab_classes(
        dedup_vocab_classes,
        &dedup.original_to_repr,
        dedup.representative_token_bytes.len(),
    );
    let tokenizer_states = compose_raw_quotient_state_map(
        &pre_state_map,
        &reduced_state_reps_for_pre_reduced,
    );
    let exact_reps = tokenizer_states.num_internal_ids() as usize;
    let internal_id_map = InternalIdMap {
        tokenizer_states,
        vocab_tokens: build_vocab_map(&vocab_classes, &prepared.token_ids, prepared.max_token_id),
    };
    let id_map_finalize_ms = id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;

    Some((
        internal_id_map,
        CombinedEquivalenceProfile {
            initial_states_considered: prepared.initial_states.len(),
            max_length_skipped: false,
            max_token_len,
            token_len_gt_4: token_len_stats.gt_4,
            token_len_gt_8: token_len_stats.gt_8,
            token_len_gt_16: token_len_stats.gt_16,
            token_len_gt_32: token_len_stats.gt_32,
            token_len_gt_64: token_len_stats.gt_64,
            raw_analysis_base_init_ms: 0.0,
            analysis_view_build_ms,
            active_mask_filter_ms,
            effective_follows_normalize_ms,
            prepare_inputs_ms,
            byte_class_setup_ms,
            vocab_analysis_dfa_build_ms,
            token_dedup_ms,
            restricted_observation_state_equiv_ms,
            max_length_state_equiv_ms: 0.0,
            vocab_equiv_ms,
            exact_state_equiv_ms,
            id_map_finalize_ms,
            restricted_observation_reps: pre_state_map.num_internal_ids() as usize,
            max_length_reps: pre_state_map.num_internal_ids() as usize,
            exact_reps,
            exact_rep_confirmation_used,
        },
    ))
}

pub(crate) fn analyze_equivalences_with_group_filter(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    disallowed_follows_are_ignore_transparent: bool,
    pre_normalized_disallowed_follows: Option<&[BitSet]>,
    active_groups: Option<&[bool]>,
    shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&super::vocab::fast::SharedVocabAnalysisDfaCache>,
    shared_base_setup_ms: f64,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    initial_state_map: Option<&ManyToOneIdMap>,
    token_boundary_partition: Option<&GlobalTokenPositionStatePartition>,
    precomputed_raw_observations: Option<(&[u32], &[u32])>,
) -> (InternalIdMap, CombinedEquivalenceProfile) {
    analyze_equivalences_impl(
        partition_label,
        tokenizer,
        vocab,
        disallowed_follows,
        ignore_terminal,
        disallowed_follows_are_ignore_transparent,
        pre_normalized_disallowed_follows,
        active_groups,
        shared_vocab_dfa_cache,
        shared_analysis_dfa_cache,
        shared_base_setup_ms,
        flat_trans,
        initial_state_map,
        token_boundary_partition,
        precomputed_raw_observations,
    )
}

/// Combined equivalence analysis over a flattened tokenizer DFA.
///
/// Uses state equivalence (k-step hashing plus token-based refinement) followed
/// by vocab equivalence (parallel batched with byte-class compression).
fn analyze_equivalences_impl(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
    disallowed_follows_are_ignore_transparent: bool,
    pre_normalized_disallowed_follows: Option<&[BitSet]>,
    active_groups: Option<&[bool]>,
    shared_vocab_dfa_cache: Option<&super::vocab::fast::SharedVocabDfaCache>,
    shared_analysis_dfa_cache: Option<&super::vocab::fast::SharedVocabAnalysisDfaCache>,
    shared_base_setup_ms: f64,
    flat_trans: Option<&std::sync::Arc<[u32]>>,
    initial_state_map: Option<&ManyToOneIdMap>,
    token_boundary_partition: Option<&GlobalTokenPositionStatePartition>,
    precomputed_raw_observations: Option<(&[u32], &[u32])>,
) -> (InternalIdMap, CombinedEquivalenceProfile) {
    // Every equivalence implementation below assumes that a lexer
    // configuration is one deterministic physical state. With epsilon edges,
    // token behaviour is a set-of-states computation and merging either raw
    // states or vocabulary entries using those DFA-only signatures is
    // unsound. Keep exact identity coordinates until a set-aware quotient is
    // implemented. This only declines compression.
    if tokenizer.has_epsilon_transitions() {
        let prepare_started_at = Instant::now();
        let prepared = prepare_equivalence_inputs(tokenizer, vocab, initial_state_map);
        let token_len_stats = token_length_stats(&prepared.token_bytes);
        let max_token_len = prepared
            .token_bytes
            .iter()
            .map(|token| token.len())
            .max()
            .unwrap_or(0);
        let state_ids = (0..tokenizer.num_states()).collect::<Vec<_>>();
        let tokenizer_states =
            ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                state_ids.clone(),
                state_ids,
            );
        let id_map = InternalIdMap {
            tokenizer_states,
            vocab_tokens: build_exact_byte_vocab_map(
                &prepared.token_ids,
                &prepared.token_bytes,
                prepared.max_token_id,
            ),
        };
        return (
            id_map,
            CombinedEquivalenceProfile {
                initial_states_considered: prepared.initial_states.len(),
                max_length_skipped: true,
                max_token_len,
                token_len_gt_4: token_len_stats.gt_4,
                token_len_gt_8: token_len_stats.gt_8,
                token_len_gt_16: token_len_stats.gt_16,
                token_len_gt_32: token_len_stats.gt_32,
                token_len_gt_64: token_len_stats.gt_64,
                raw_analysis_base_init_ms: 0.0,
                analysis_view_build_ms: 0.0,
                active_mask_filter_ms: 0.0,
                effective_follows_normalize_ms: 0.0,
                prepare_inputs_ms: prepare_started_at.elapsed().as_secs_f64() * 1000.0,
                byte_class_setup_ms: 0.0,
                vocab_analysis_dfa_build_ms: 0.0,
                token_dedup_ms: 0.0,
                restricted_observation_state_equiv_ms: 0.0,
                max_length_state_equiv_ms: 0.0,
                vocab_equiv_ms: 0.0,
                exact_state_equiv_ms: 0.0,
                id_map_finalize_ms: 0.0,
                restricted_observation_reps: tokenizer.num_states() as usize,
                max_length_reps: tokenizer.num_states() as usize,
                exact_reps: tokenizer.num_states() as usize,
                exact_rep_confirmation_used: false,
            },
        );
    }

    let follows_prepare_started_at = Instant::now();
    let token_path_disallowed_follows = (!disallowed_follows_are_ignore_transparent)
        .then(|| ignore_transparent_disallowed_follows(disallowed_follows, ignore_terminal));
    let effective_follows_prepare_ms = follows_prepare_started_at.elapsed().as_secs_f64() * 1000.0;
    let effective_disallowed = token_path_disallowed_follows
        .as_ref()
        .unwrap_or(disallowed_follows);
    if let Some(token_boundary_partition) = token_boundary_partition {
        if let Some(result) = try_analyze_equivalences_with_token_boundary_partition(
            partition_label,
            tokenizer,
            vocab,
            effective_disallowed,
            active_groups,
            shared_vocab_dfa_cache,
            shared_analysis_dfa_cache,
            flat_trans,
            token_boundary_partition,
            effective_follows_prepare_ms,
            pre_normalized_disallowed_follows,
        ) {
            return result;
        }
    }
    if let Some(result) = try_analyze_equivalences_with_raw_quotient(
        partition_label,
        tokenizer,
        vocab,
        effective_disallowed,
        ignore_terminal,
        active_groups,
        flat_trans,
        initial_state_map,
        effective_follows_prepare_ms,
        precomputed_raw_observations,
    ) {
        return result;
    }
    // The raw tokenizer is the only lexer coordinate in L2P. Retain the
    // compatibility check defensively, since this cache is shared by callers.
    let compatible_flat_trans = flat_trans.filter(|ft| {
        ft.len() == tokenizer.num_states() as usize * 256
    });
    let analysis_view_build_started_at = Instant::now();
    let tokenizer_view = match (active_groups, compatible_flat_trans) {
        (Some(active_groups), Some(ft)) => TokenizerView::new_filtered_from_flat_trans(ft, tokenizer, active_groups),
        (Some(active_groups), None) => TokenizerView::new_filtered(tokenizer, active_groups),
        (None, Some(ft)) => TokenizerView::new_from_flat_trans(ft, tokenizer),
        _ => TokenizerView::new(tokenizer),
    };
    let analysis_view_build_ms = analysis_view_build_started_at.elapsed().as_secs_f64() * 1000.0;

    let prepare_inputs_started_at = Instant::now();
    let prepared = prepare_equivalence_inputs(tokenizer, vocab, initial_state_map);
    let token_len_stats = token_length_stats(&prepared.token_bytes);
    let prepare_inputs_ms = prepare_inputs_started_at.elapsed().as_secs_f64() * 1000.0;

    let raw_analysis_base_started_at = Instant::now();
    if let Some(cache) = shared_vocab_dfa_cache {
        cache.get_or_init(|| vocab_equivalence_analysis::SharedVocabDfaBase::build_from_dfa(tokenizer_view.dfa()));
    }
    let raw_analysis_base_init_ms = shared_base_setup_ms
        + raw_analysis_base_started_at.elapsed().as_secs_f64() * 1000.0;

    let byte_class_setup_started_at = Instant::now();
    let compatible_cache = shared_vocab_dfa_cache
        .and_then(|cache| cache.get())
        .filter(|base| base.is_compatible_with_dfa(tokenizer_view.dfa()));
    let byte_to_class = compatible_cache
        .map(|base| base.byte_to_class())
        .unwrap_or_else(|| super::compat::compute_byte_classes(tokenizer_view.dfa()));
    let byte_class_setup_ms = byte_class_setup_started_at.elapsed().as_secs_f64() * 1000.0;

    let token_dedup_started_at = Instant::now();
    let dedup = deduplicate_tokens_by_byte_class(&prepared.token_bytes, &byte_to_class);
    let token_dedup_ms = token_dedup_started_at.elapsed().as_secs_f64() * 1000.0;
    let max_token_len = dedup
        .representative_token_bytes
        .iter()
        .map(|token| token.len())
        .max()
        .unwrap_or(0);
    let mut relevant_bytes = [false; 256];
    for token in &dedup.representative_token_bytes {
        for &byte in *token {
            relevant_bytes[byte as usize] = true;
        }
    }
    let direct_token_bytes: usize = dedup
        .representative_token_bytes
        .iter()
        .map(|token| token.len())
        .sum();
    let active_byte_count = relevant_bytes.iter().filter(|&&active| active).count();
    let projected_by_global = prepared.initial_states.len() < tokenizer.num_states() as usize;
    let max_length_skipped = should_skip_max_length_for_partition(
        partition_label,
        prepared.initial_states.len(),
        projected_by_global,
        direct_token_bytes,
        max_token_len,
        active_byte_count,
    );
    let pipeline_config = resolve_l2p_pipeline_config(!max_length_skipped);
    let (pre_state_map, pipeline_profile) = run_state_equivalence_pipeline(
        tokenizer,
        vocab,
        initial_state_map,
        active_groups,
        StateEquivalenceScope::L2p,
        &pipeline_config,
        Some(&tokenizer_view),
        Some(&byte_to_class),
    );
    // Restricted observation is a fixed point over the vocabulary byte
    // alphabet. When its map is also output-labelled congruent, token paths can
    // be evaluated exactly on the quotient instead of reinitializing 18k raw
    // lexer states for the exact and vocabulary phases.
    let analysis_quotient = tokenizer_view
        .is_relevant_byte_congruent(&pre_state_map, &relevant_bytes)
        .then(|| tokenizer_view.quotient_by_state_map(&pre_state_map));
    let uses_analysis_quotient = analysis_quotient.is_some();
    let analysis_view = analysis_quotient.as_ref().unwrap_or(&tokenizer_view);
    let pre_reduced_states: Vec<usize> = if uses_analysis_quotient {
        (0..pre_state_map.internal_to_originals.len()).collect()
    } else {
        pre_state_map
            .representative_original_ids
            .iter()
            .map(|&state| state as usize)
            .collect()
    };
    let follows_normalize_started_at = Instant::now();
    let normalized_disallowed_follows =
        normalize_disallowed_follows(tokenizer_group_count(analysis_view), effective_disallowed);
    let effective_follows_normalize_ms = effective_follows_prepare_ms
        + follows_normalize_started_at.elapsed().as_secs_f64() * 1000.0;

    // First finalize state equivalence against every deduplicated token.
    // A final state representative is indistinguishable from each state it
    // represents for every token, so token equivalence may then be computed
    // over only those representatives. This reverses two exact quotients and
    // avoids classifying tokens over the much larger pre-refinement state set.
    let exact_rep_confirmation_used = pre_reduced_states.len() >= EXACT_REP_CONFIRMATION_MIN_STATES
        && dedup.representative_token_bytes.len() >= EXACT_REP_CONFIRMATION_MIN_TOKENS;
    let exact_started_at = Instant::now();
    let reduced_state_reps_for_pre_reduced = if exact_rep_confirmation_used {
        state_equivalence_analysis::find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed_and_shared_base(
            analysis_view,
            &dedup.representative_token_bytes,
            &pre_reduced_states,
            &normalized_disallowed_follows,
            None,
            None,
            Some(true),
            (!uses_analysis_quotient).then_some(compatible_cache).flatten(),
        )
    } else {
        state_equivalence_analysis::find_state_equivalence_classes_with_disallowed_and_shared_base(
            analysis_view,
            &dedup.representative_token_bytes,
            &pre_reduced_states,
            &normalized_disallowed_follows,
            (!uses_analysis_quotient).then_some(compatible_cache).flatten(),
        )
    };
    let exact_state_equiv_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;

    let representative_states = if uses_analysis_quotient {
        prepared
            .initial_states
            .iter()
            .map(|&state| {
                let pre_internal = pre_state_map.original_to_internal[state] as usize;
                let final_internal = reduced_state_reps_for_pre_reduced[pre_internal];
                pre_state_map.representative_original_ids[final_internal] as usize
            })
            .collect::<Vec<_>>()
    } else {
        let rep_to_final: BTreeMap<usize, usize> = pre_reduced_states
            .iter()
            .copied()
            .zip(reduced_state_reps_for_pre_reduced.iter().copied())
            .collect();
        prepared
            .initial_states
            .iter()
            .map(|&state| {
                let pre_internal = pre_state_map.original_to_internal[state];
                let pre_rep = pre_state_map.representative_original_ids[pre_internal as usize] as usize;
                rep_to_final[&pre_rep]
            })
            .collect::<Vec<_>>()
    };
    let mut final_state_representatives = reduced_state_reps_for_pre_reduced;
    final_state_representatives.sort_unstable();
    final_state_representatives.dedup();

    let vocab_equiv_started_at = Instant::now();
    let (dedup_vocab_classes, vocab_analysis_dfa_build_ms) =
        vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter_profiled(
            analysis_view,
            &dedup.representative_token_bytes,
            &final_state_representatives,
            effective_disallowed,
            Some(&byte_to_class),
            if uses_analysis_quotient { None } else { active_groups },
            if uses_analysis_quotient { None } else { shared_vocab_dfa_cache },
            if uses_analysis_quotient { None } else { shared_analysis_dfa_cache },
        );
    let vocab_equiv_ms = vocab_equiv_started_at.elapsed().as_secs_f64() * 1000.0;

    let id_map_finalize_started_at = Instant::now();
    let vocab_classes = expand_vocab_classes(
        dedup_vocab_classes,
        &dedup.original_to_repr,
        dedup.representative_token_bytes.len(),
    );
    let state_classes =
        state_equivalence_analysis::mapping_to_equivalence_classes(&prepared.initial_states, &representative_states);
    let exact_reps = state_classes.len();
    let result = CombinedEquivalenceResult {
        vocab_classes,
        state_classes,
    };
    let internal_id_map =
        build_internal_id_map_from_combined_result(tokenizer, initial_state_map, &prepared, &result);
    let id_map_finalize_ms = id_map_finalize_started_at.elapsed().as_secs_f64() * 1000.0;

    (
        internal_id_map,
        CombinedEquivalenceProfile {
            initial_states_considered: prepared.initial_states.len(),
            max_length_skipped: pipeline_profile.max_length_skipped,
            max_token_len,
            token_len_gt_4: token_len_stats.gt_4,
            token_len_gt_8: token_len_stats.gt_8,
            token_len_gt_16: token_len_stats.gt_16,
            token_len_gt_32: token_len_stats.gt_32,
            token_len_gt_64: token_len_stats.gt_64,
            raw_analysis_base_init_ms,
            analysis_view_build_ms,
            active_mask_filter_ms: 0.0,
            effective_follows_normalize_ms,
            prepare_inputs_ms,
            byte_class_setup_ms,
            vocab_analysis_dfa_build_ms,
            token_dedup_ms,
            restricted_observation_state_equiv_ms: pipeline_profile.restricted_observation_state_equiv_ms,
            max_length_state_equiv_ms: pipeline_profile.max_length_state_equiv_ms,
            vocab_equiv_ms,
            exact_state_equiv_ms,
            id_map_finalize_ms,
            restricted_observation_reps: pipeline_profile.restricted_observation_reps,
            max_length_reps: pipeline_profile.max_length_reps,
            exact_reps,
            exact_rep_confirmation_used,
        },
    )
}

#[cfg(test)]
mod prepass_selection_tests {
    use super::*;
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::compat::{
        FlatDfa, FlatDfaState,
    };
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::shared::representative_tokens_for_vocab_classes;
    use std::sync::Arc;

    fn partition_from_representatives<T: Ord + Copy>(
        values: &[T],
        representatives: &[T],
    ) -> BTreeSet<BTreeSet<T>> {
        let mut by_representative = BTreeMap::<T, BTreeSet<T>>::new();
        for (&value, &representative) in values.iter().zip(representatives) {
            by_representative.entry(representative).or_default().insert(value);
        }
        by_representative.into_values().collect()
    }

    fn synthetic_view() -> TokenizerView {
        let state_count = 5usize;
        let mut transitions = vec![u32::MAX; state_count * 256];
        let set = |transitions: &mut [u32], state: usize, byte: u8, target: u32| {
            transitions[state * 256 + byte as usize] = target;
        };
        set(&mut transitions, 0, b'a', 1);
        set(&mut transitions, 0, b'b', 2);
        set(&mut transitions, 1, b'a', 1);
        set(&mut transitions, 1, b'b', 3);
        set(&mut transitions, 2, b'a', 3);
        set(&mut transitions, 2, b'b', 2);
        set(&mut transitions, 3, b'a', 3);
        set(&mut transitions, 3, b'b', 3);
        // State 4 is behaviorally identical to state 1.
        set(&mut transitions, 4, b'a', 4);
        set(&mut transitions, 4, b'b', 3);

        TokenizerView {
            flat_dfa: FlatDfa {
                start_state: 0,
                transitions: Arc::from(transitions),
                states: vec![
                    FlatDfaState {
                        finalizers: vec![],
                        possible_future_group_ids: vec![0, 1],
                    },
                    FlatDfaState {
                        finalizers: vec![0],
                        possible_future_group_ids: vec![0, 1],
                    },
                    FlatDfaState {
                        finalizers: vec![1],
                        possible_future_group_ids: vec![0, 1],
                    },
                    FlatDfaState {
                        finalizers: vec![0, 1],
                        possible_future_group_ids: vec![0, 1],
                    },
                    FlatDfaState {
                        finalizers: vec![0],
                        possible_future_group_ids: vec![0, 1],
                    },
                ],
            },
        }
    }

    #[test]
    fn state_then_vocab_equivalence_matches_vocab_then_state() {
        let view = synthetic_view();
        let tokens: Vec<&[u8]> = vec![
            b"a", b"b", b"aa", b"ab", b"ba", b"bb", b"x", b"y",
        ];
        let states: Vec<usize> = (0..view.dfa().states.len()).collect();
        let byte_to_class = super::super::compat::compute_byte_classes(view.dfa());
        let disallowed = BTreeMap::<u32, BitSet>::new();
        let normalized = normalize_disallowed_follows(2, &disallowed);

        let old_vocab = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &states,
            &disallowed,
            Some(&byte_to_class),
            None,
            None,
            None,
        );
        let old_token_reps = representative_tokens_for_vocab_classes(&old_vocab, &tokens);
        let old_state_reps =
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed(
                &view,
                &old_token_reps,
                &states,
                &normalized,
            );

        let reversed_state_reps =
            state_equivalence_analysis::find_state_equivalence_classes_with_disallowed(
                &view,
                &tokens,
                &states,
                &normalized,
            );
        let final_state_reps: Vec<usize> = {
            let mut reps = reversed_state_reps.clone();
            reps.sort_unstable();
            reps.dedup();
            reps
        };
        let reversed_vocab = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
            &view,
            &tokens,
            &final_state_reps,
            &disallowed,
            Some(&byte_to_class),
            None,
            None,
            None,
        );

        assert_eq!(
            partition_from_representatives(&states, &old_state_reps),
            partition_from_representatives(&states, &reversed_state_reps),
        );
        assert_eq!(old_vocab, reversed_vocab);
    }

    #[test]
    fn selects_direct_refinement_when_byte_bounded_prepass_cannot_amortize() {
        // Small vocabulary with many relevant bytes: direct token walks are
        // cheaper than a full k-bounded byte refinement per lexer state.
        assert!(direct_refinement_work_is_no_larger(100, 10, 41));
        // Larger vocabulary over a smaller byte alphabet amortizes the exact
        // prepass and should retain it.
        assert!(!direct_refinement_work_is_no_larger(900, 14, 19));
    }
}
