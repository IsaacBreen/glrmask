//! Top-level id_map + terminal DWA builder.
//!
//! The canonical path splits the vocabulary into 3 character-type partitions,
//! builds a per-partition `(InternalIdMap, DWA)`, and merges the results into
//! the final global `(InternalIdMap, DWA)`.

pub(crate) mod classify;
pub(crate) mod grammar_helpers;
pub(crate) mod l1;
pub(crate) mod l2p;
pub(crate) mod merge;
pub(crate) mod partition;
pub(crate) mod types;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::bitset::BitSet;
use crate::grammar::flat::TerminalID;
use crate::Vocab;

use classify::classify_vocab_char_type;
use l2p::equivalence_analysis::compat::{TokenizerView, compute_byte_classes};
use l2p::equivalence_analysis::state::max_length::find_state_equivalence_classes_byte_restricted;
use types::{
    compile_profile_enabled, TerminalColoring,
    TerminalDwaPhaseProfile,
};

pub(crate) fn build_global_max_length_state_map(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    flat_trans: &Arc<[u32]>,
) -> ManyToOneIdMap {
    let started_at = Instant::now();
    let tokenizer_view = TokenizerView::new_from_flat_trans(
        flat_trans,
        tokenizer,
    );
    let token_bytes: Vec<&[u8]> = vocab.entries.values().map(|bytes| bytes.as_slice()).collect();
    let mut relevant_bytes = [false; 256];
    for bytes in &token_bytes {
        for &byte in *bytes {
            relevant_bytes[byte as usize] = true;
        }
    }
    let byte_to_class = compute_byte_classes(tokenizer_view.dfa());
    let states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();
    let mapping = find_state_equivalence_classes_byte_restricted(
        &tokenizer_view,
        &token_bytes,
        &states,
        Some(&byte_to_class),
        None,
        Some(&relevant_bytes),
    );

    let mut rep_to_internal = FxHashMap::<usize, u32>::default();
    let mut original_to_internal = vec![u32::MAX; states.len()];
    let mut representative_original_ids = Vec::new();
    for (state, &rep) in mapping.iter().enumerate() {
        let internal = *rep_to_internal.entry(rep).or_insert_with(|| {
            let id = representative_original_ids.len() as u32;
            representative_original_ids.push(rep as u32);
            id
        });
        original_to_internal[state] = internal;
    }

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][global_max_length] states={} reps={} tokens={} ms={:.3}",
            states.len(),
            representative_original_ids.len(),
            token_bytes.len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    ManyToOneIdMap::from_original_to_internal_with_representatives(
        original_to_internal,
        representative_original_ids.len() as u32,
        representative_original_ids,
    )
}

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
/// 1. Splits vocab into 3 partitions by leading-byte character type.
/// 2. Builds each partition's `(InternalIdMap, DWA)` in parallel via
///    [`partition::build_partition_id_map_and_terminal_dwa`].
/// 3. Merges the 3 results via [`merge::merge_id_maps_and_terminal_dwas`].
pub(crate) fn build_id_map_and_terminal_dwa(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    external_classify_cache: Option<&classify::SharedClassifyCache>,
) -> (InternalIdMap, DWA, TerminalDwaPhaseProfile, ManyToOneIdMap) {
    let total_started_at = Instant::now();
    let mut profile = TerminalDwaPhaseProfile::default();


    // Shared cache for terminal classification byte sets. The DFA scanning
    // (reachable_bytes, first_bytes, last_bytes) is identical across partitions;
    // only the vocab-dependent classification differs. Reuse external cache if
    // provided (already populated by compile.rs pre-classification), otherwise
    // create a fresh one for partition sharing.
    let owned_classify_cache = classify::SharedClassifyCache::new();
    let shared_classify_cache: &classify::SharedClassifyCache =
        external_classify_cache.unwrap_or(&owned_classify_cache);

    let partition_vocab_started_at = Instant::now();
    let mut partition_entries: Vec<Vec<(u32, Vec<u8>)>> = (0..7).map(|_| Vec::new()).collect();
    for (&token_id, bytes) in &vocab.entries {
        let idx = classify_vocab_char_type(bytes) as usize;
        partition_entries[idx].push((token_id, bytes.clone()));
    }
    let sub_vocabs: Vec<Vocab> = partition_entries.into_iter().map(|entries| Vocab::new(entries, None)).collect();
    let partition_vocab_ms = partition_vocab_started_at.elapsed().as_secs_f64() * 1000.0;
    profile.id_map_ms += partition_vocab_ms;

    // Build flat DFA transition table once (shared across all partitions).
    let flat_trans_started_at = Instant::now();
    let flat_trans: Arc<[u32]> = Arc::from(l1::build_flat_transition_table(tokenizer));
    profile.terminal_dwa_ms += flat_trans_started_at.elapsed().as_secs_f64() * 1000.0;

    let global_max_length_started_at = Instant::now();
    let global_max_length_state_map = build_global_max_length_state_map(tokenizer, vocab, &flat_trans);
    profile.id_map_ms += global_max_length_started_at.elapsed().as_secs_f64() * 1000.0;

    // Lazily-initialized shared compact transition table cache.
    // The first partition to reach vocab_build_dfa will build the cache from
    // its simplified tokenizer's FlatDfa (same transitions as original when
    // minimize is skipped). Subsequent partitions reuse it, skipping the
    // ~120ms transpose + byte-class computation each (~480ms CPU saved).
    // The cache validates state counts, so it's safely ignored when simplify
    // changes the DFA (reducing state count via minimization).
    let shared_vocab_dfa_cache = l2p::equivalence_analysis::vocab::fast::SharedVocabDfaCache::new();
    let shared_simplify_cache = l2p::SharedSimplifyCache::default();

    use rayon::prelude::*;
    let partition_results: Vec<(Option<(merge::LocalIdMapTerminalDwa, f64)>, usize)> = sub_vocabs
        .par_iter()
        .enumerate()
        .map(|(idx, sub_vocab)| {
            let started_at = Instant::now();
            let label = format!("p{}", idx);
            let result = partition::build_partition_id_map_and_terminal_dwa(
                &label,
                tokenizer,
                sub_vocab,
                terminal_coloring,
                use_terminal_coloring,
                ignore_terminal,
                grammar,
                disallowed_follows,
                &flat_trans,
                Some(&global_max_length_state_map),
                Some(&shared_vocab_dfa_cache),
                Some(&shared_simplify_cache),
                Some(&shared_classify_cache),
            ).map(|pair| (pair, started_at.elapsed().as_secs_f64() * 1000.0));
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

    // Collect non-None results.
    let mut pairs: Vec<merge::LocalIdMapTerminalDwa> = Vec::new();
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
        return (empty_map, DWA::new(1, 0), profile, global_max_length_state_map);
    }

    let num_tokenizer_states = tokenizer.num_states() as usize;
    let max_token_id = vocab.max_token_id();

    let merge_started_at = Instant::now();
    let (merged, global_merge_profile) = if pairs.len() == 1 {
        // Single partition — already compacted by partition merge. Skip redundant global compact.
        (pairs.into_iter().next().unwrap(), TerminalDwaPhaseProfile::default())
    } else {
        let merged = merge::merge_id_maps_and_terminal_dwas(
            pairs,
            num_tokenizer_states,
            max_token_id,
        );
        let global_merge_profile = merged.profile;
        (merged, global_merge_profile)
    };
    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;
    profile.add_assign(dominant_partition_profile);
    profile.add_assign(global_merge_profile);

    if compile_profile_enabled() {
        let partition_detail: String = sub_vocabs.iter().enumerate()
            .map(|(i, sv)| format!("p{}_tokens={} p{}_ms={:.3}", i, sv.entries.len(), i, partition_ms[i]))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "[glrmask/profile][split_terminal_dwa] partition_vocab_ms={:.3} {} global_merge_ms={:.3} accounted_id_map_ms={:.3} accounted_terminal_dwa_ms={:.3} accounted_compact_ms={:.3} accounted_total_ms={:.3} total_ms={:.3}",
            partition_vocab_ms,
            partition_detail,
            merge_ms,
            profile.id_map_ms,
            profile.terminal_dwa_ms,
            profile.compact_ms,
            profile.total_ms(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    (merged.id_map, merged.dwa, profile, global_max_length_state_map)
}
