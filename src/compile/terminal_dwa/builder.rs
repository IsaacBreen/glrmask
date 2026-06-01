//! Top-level Terminal-DWA builder.
//!
//! This is the orchestration layer only.  It is allowed to time phases, allocate
//! shared caches, spawn partition builds, and merge results.  It should not own
//! the details of vocabulary partitioning, state equivalence, direct-partition
//! construction, pair-partition construction, or id-map reconciliation.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::parser::glr::analysis::AnalyzedGrammar;
use crate::compile::id_space::{InternalIdMap, ManyToOneIdMap, MappedArtifact};
use crate::compile::terminal_dwa::classify;
use crate::compile::terminal_dwa::direct_partition;
use crate::compile::terminal_dwa::global_state_map;
use crate::compile::terminal_dwa::merge;
use crate::compile::terminal_dwa::pair_partition;
use crate::compile::terminal_dwa::partition;
use crate::compile::terminal_dwa::types::{
    compile_profile_enabled,
    LocalIdMapTerminalDwa,
    TerminalColoring,
    TerminalDwaPhaseProfile,
};
use crate::compile::terminal_dwa::vocab_partition;
use crate::sets::bitset::BitSet;
use crate::grammar::flat::TerminalID;
use crate::Vocab;

/// Build the global `(InternalIdMap, DWA)` for the full vocabulary.
///
/// IMPORTANT: the JSON-schema importer must not feed this stage large numbers
/// of short, grammar-visible alnum-ish terminals. A terminal is
/// "grammar-visible" when it is either a named terminal or an inline
/// literal/pattern that appears directly in a nonterminal rule body.
///
/// Those terminals are pathological for terminal-DWA construction when all of
/// the following hold:
///
/// 1. they match one of a broad character class but only a small bounded
///    number of characters (classic bad cases: `[a-z]`, `[a-z]{1,3}`,
///    bare URI/hostname/JSON-string body fragments);
/// 2. they do not carry stabilizing punctuation with them, especially on the
///    trailing edge; and
/// 3. they are grammar-visible rather than internal-only.
///
/// This creates explosive same-prefix ambiguity in the terminal DWA: e.g. a
/// visible `[a-z]` and `[a-z][a-z]` force the DWA to keep many competing
/// terminal continuations alive. The importer therefore deliberately fuses
/// punctuation into visible terminals when possible, and keeps short generic
/// bodies internal-only whenever possible. Do not "simplify" that structure
/// away without re-checking schemas like `Github_hard---o1051` and
/// `uuid_maxlength5000`.
///
/// `JSON_NUMBER` is the known exception: it technically fits the heuristic, but
/// in practice it has not shown the same DWA blow-up.
///
/// The algorithm is:
///
/// 1. Choose a vocabulary partitioning strategy.
/// 2. Build each partition's local `(InternalIdMap, DWA)` in parallel via
///    [`partition::build_partition_terminal_dwa`].
/// 3. Merge the local results via [`merge::merge_id_maps_and_terminal_dwas`].
/// 4. Return one mapped Terminal-DWA artifact whose local ids are reconciled
///    against the original tokenizer states and caller token ids.
pub(crate) fn build_terminal_dwa_with_precomputed_global_max_length(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    flat_trans: Arc<[u32]>,
    global_max_length_state_map: &ManyToOneIdMap,
    external_classify_cache: Option<&classify::SharedClassifyCache>,
) -> (MappedArtifact<DWA>, TerminalDwaPhaseProfile) {
    let total_started_at = Instant::now();
    let mut profile = TerminalDwaPhaseProfile::default();

    // Shared cache for terminal classification byte sets. The DFA scanning
    // (reachable_bytes, first_bytes, last_bytes) is identical across partitions;
    // only the vocab-dependent classification differs. Reuse external cache if
    // provided (already populated by pipeline pre-classification), otherwise
    // create a fresh one for partition sharing.
    let owned_classify_cache = classify::SharedClassifyCache::new();
    let shared_classify_cache: &classify::SharedClassifyCache =
        external_classify_cache.unwrap_or(&owned_classify_cache);

    let partition_vocab_started_at = Instant::now();
    let sub_vocabs = vocab_partition::choose_terminal_dwa_sub_vocabs(
        tokenizer,
        vocab,
        grammar,
        disallowed_follows,
        shared_classify_cache,
    );
    let partition_vocab_ms = partition_vocab_started_at.elapsed().as_secs_f64() * 1000.0;
    profile.id_map_ms += partition_vocab_ms;

    // Lazily-initialized shared compact transition table cache.  The first
    // partition to reach vocab_build_dfa builds the cache from its simplified
    // tokenizer's FlatDfa.  Subsequent partitions reuse it when compatible.
    let shared_vocab_dfa_cache =
        pair_partition::equivalence_analysis::vocab::fast::SharedVocabDfaCache::new();
    let shared_simplify_cache = pair_partition::SharedSimplifyCache::default();
    let shared_disallowed_follow_dfa_cache =
        pair_partition::postprocess::SharedDisallowedFollowDfaCache::new();

    let partition_results: Vec<(Option<(LocalIdMapTerminalDwa, f64)>, usize)> = sub_vocabs
        .par_iter()
        .enumerate()
        .map(|(idx, sub_vocab)| {
            let started_at = Instant::now();
            let label = format!("p{}", idx);
            let result = partition::build_partition_terminal_dwa(
                &label,
                tokenizer,
                sub_vocab,
                terminal_coloring,
                use_terminal_coloring,
                ignore_terminal,
                grammar,
                disallowed_follows,
                &flat_trans,
                Some(global_max_length_state_map),
                Some(&shared_vocab_dfa_cache),
                Some(&shared_simplify_cache),
                Some(&shared_disallowed_follow_dfa_cache),
                Some(shared_classify_cache),
            )
            .map(|pair| (pair, started_at.elapsed().as_secs_f64() * 1000.0));
            (result, idx)
        })
        .collect();

    let partition_ms: Vec<f64> = {
        let mut ms = vec![0.0; sub_vocabs.len()];
        for (result, idx) in &partition_results {
            ms[*idx] = result.as_ref().map(|(_, m)| *m).unwrap_or(0.0);
        }
        ms
    };
    let dominant_partition_profile = partition_results
        .iter()
        .filter_map(|(result, _)| result.as_ref().map(|(pair, ms)| (pair.profile, *ms)))
        .max_by(|(_, left_ms), (_, right_ms)| left_ms.total_cmp(right_ms))
        .map(|(phase_profile, _)| phase_profile)
        .unwrap_or_default();

    let mut pairs: Vec<LocalIdMapTerminalDwa> = Vec::new();
    for (result, _idx) in partition_results {
        if let Some((pair, _)) = result {
            pairs.push(pair);
        }
    }

    if pairs.is_empty() {
        let num_states = tokenizer.num_states() as usize;
        let empty_map = InternalIdMap {
            tokenizer_states: ManyToOneIdMap {
                original_to_internal: vec![0u32; num_states],
                internal_to_originals: vec![(0..num_states as u32).collect()],
                representative_original_ids: vec![0],
            },
            vocab_tokens: ManyToOneIdMap {
                original_to_internal: Vec::new(),
                internal_to_originals: Vec::new(),
                representative_original_ids: Vec::new(),
            },
        };
        return (MappedArtifact::new(DWA::new(1, 0), empty_map), profile);
    }

    let num_tokenizer_states = tokenizer.num_states() as usize;
    let max_token_id = vocab.max_token_id();

    let did_global_merge = pairs.len() > 1;
    let merge_started_at = Instant::now();
    let (merged, global_merge_profile) = if !did_global_merge {
        // Single partition — already compacted by partition merge. Skip a
        // redundant global compact pass.
        (
            pairs.into_iter().next().unwrap(),
            TerminalDwaPhaseProfile::default(),
        )
    } else {
        let merged = merge::merge_id_maps_and_terminal_dwas(pairs, num_tokenizer_states, max_token_id);
        let global_merge_profile = merged.profile;
        (merged, global_merge_profile)
    };
    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;
    profile.add_assign(dominant_partition_profile);
    profile.add_assign(global_merge_profile);
    profile.global_merge_ms = if did_global_merge { merge_ms } else { 0.0 };
    let split_terminal_dwa_total_ms = total_started_at.elapsed().as_secs_f64() * 1000.0;
    profile.split_terminal_dwa_total_ms = split_terminal_dwa_total_ms;

    if compile_profile_enabled() {
        let partition_detail: String = sub_vocabs
            .iter()
            .enumerate()
            .map(|(i, sv)| {
                format!(
                    "p{}_tokens={} p{}_ms={:.3}",
                    i,
                    sv.entries.len(),
                    i,
                    partition_ms[i]
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "[glrmask/profile][split_terminal_dwa] partition_vocab_ms={:.3} {} global_merge_ms={:.3} split_terminal_dwa_total_ms={:.3} accounted_id_map_ms={:.3} accounted_terminal_dwa_ms={:.3} accounted_compact_ms={:.3} accounted_total_ms={:.3} total_ms={:.3}",
            partition_vocab_ms,
            partition_detail,
            merge_ms,
            split_terminal_dwa_total_ms,
            profile.id_map_ms,
            profile.terminal_dwa_ms,
            profile.compact_ms,
            profile.total_ms(),
            split_terminal_dwa_total_ms,
        );
    }

    (MappedArtifact::new(merged.dwa, merged.id_map), profile)
}

pub(crate) fn build_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    external_classify_cache: Option<&classify::SharedClassifyCache>,
) -> (MappedArtifact<DWA>, TerminalDwaPhaseProfile, ManyToOneIdMap) {
    let mut profile = TerminalDwaPhaseProfile::default();

    let flat_trans_started_at = Instant::now();
    let flat_trans: Arc<[u32]> = Arc::from(direct_partition::build_flat_transition_table(tokenizer));
    let flat_trans_ms = flat_trans_started_at.elapsed().as_secs_f64() * 1000.0;

    let global_max_length_started_at = Instant::now();
    let global_max_length_state_map =
        global_state_map::build_global_max_length_state_map(tokenizer, vocab, &flat_trans);
    let global_max_length_ms = global_max_length_started_at.elapsed().as_secs_f64() * 1000.0;

    let (mapped_dwa, mut inner_profile) = build_terminal_dwa_with_precomputed_global_max_length(
        tokenizer,
        vocab,
        terminal_coloring,
        use_terminal_coloring,
        ignore_terminal,
        grammar,
        disallowed_follows,
        flat_trans,
        &global_max_length_state_map,
        external_classify_cache,
    );
    inner_profile.terminal_dwa_ms += flat_trans_ms;
    inner_profile.id_map_ms += global_max_length_ms;
    profile.add_assign(inner_profile);

    (mapped_dwa, profile, global_max_length_state_map)
}
