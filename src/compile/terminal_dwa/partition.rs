//! Per-partition terminal DWA builder.
//!
//! Given a partition vocab and shared parameters, classify terminals into direct-partition
//! and pair-partition, build those two pieces independently, then merge them into a
//! single `(InternalIdMap, DWA)` for the partition.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::parser::glr::analysis::AnalyzedGrammar;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::compile::terminal_dwa::classify::classify_terminal_path_lengths;
use crate::compile::terminal_dwa::types::{
    LocalIdMapTerminalDwa, TerminalColoring, TerminalPathLength, compile_profile_enabled,
};
use crate::compile::terminal_dwa::merge::merge_local_id_maps_and_terminal_dwas;
use crate::ds::bitset::BitSet;
use crate::grammar::flat::TerminalID;
use crate::Vocab;

/// Build an id_map and terminal DWA for a single vocab partition.
///
/// 1. Classify terminal path lengths into direct partition / pair-partition masks.
/// 2. Build direct partition and pair-partition `(InternalIdMap, DWA)` pairs in parallel.
/// 3. Merge the two results.
/// 4. Return a single `(InternalIdMap, DWA)`.
///
/// Returns `None` if the vocab is empty.
pub(crate) fn build_partition_terminal_dwa(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    flat_trans: &Arc<[u32]>,
    initial_state_map: Option<&ManyToOneIdMap>,
    _shared_vocab_dfa_cache: Option<&super::pair_partition::equivalence_analysis::vocab::fast::SharedVocabDfaCache>,
    shared_simplify_cache: Option<&super::pair_partition::SharedSimplifyCache>,
    shared_disallowed_follow_dfa_cache: Option<&super::pair_partition::postprocess::SharedDisallowedFollowDfaCache>,
    shared_classify_cache: Option<&super::classify::SharedClassifyCache>,
) -> Option<LocalIdMapTerminalDwa> {
    if vocab.is_empty() {
        return None;
    }

    let total_started_at = Instant::now();
    let num_terminals = grammar.num_terminals as u32;
    // Classify terminals into direct partition (single-byte paths) vs pair-partition by default.
    // Set GLRMASK_FORCE_ALL_PAIR_PARTITION=1 to skip direct partition and route everything through pair-partition.
    let force_all_pair_partition = std::env::var("GLRMASK_FORCE_ALL_PAIR_PARTITION").map_or(false, |v| v == "1");

    let classify_started_at = Instant::now();
    let terminal_path_lengths = if force_all_pair_partition {
        vec![TerminalPathLength::MultiStep; num_terminals as usize]
    } else {
        classify_terminal_path_lengths(tokenizer, vocab, disallowed_follows, num_terminals, shared_classify_cache)
    };
    let classify_ms = classify_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut direct_partition_mask = vec![false; num_terminals as usize];
    let mut pair_partition_mask = vec![false; num_terminals as usize];
    let mut has_direct_partition = false;
    let mut has_pair_partition = false;
    let mut num_zero = 0usize;
    let mut num_one = 0usize;
    let mut num_multi_step = 0usize;
    for (i, len) in terminal_path_lengths.iter().enumerate() {
        match len {
            TerminalPathLength::One => {
                direct_partition_mask[i] = true;
                has_direct_partition = true;
                num_one += 1;
            }
            TerminalPathLength::MultiStep => {
                pair_partition_mask[i] = true;
                has_pair_partition = true;
                num_multi_step += 1;
            }
            TerminalPathLength::Zero => {
                num_zero += 1;
            }
        }
    }

    // Build direct partition and pair-partition terminal DWAs in parallel.
    let (direct_partition_result, pair_partition_result) = rayon::join(
        || {
            if has_direct_partition {
                let started_at = Instant::now();
                let result = super::direct_partition::build_direct_partition_terminal_dwa(
                    partition_label,
                    tokenizer,
                    vocab,
                    terminal_coloring,
                    use_terminal_coloring,
                    ignore_terminal,
                    grammar,
                    &direct_partition_mask,
                    flat_trans,
                    initial_state_map,
                );
                (result, started_at.elapsed().as_secs_f64() * 1000.0)
            } else {
                (None, 0.0)
            }
        },
        || {
            if has_pair_partition {
                let started_at = Instant::now();
                let result = super::pair_partition::build_pair_partition_terminal_dwa(
                    partition_label,
                    tokenizer,
                    vocab,
                    terminal_coloring,
                    use_terminal_coloring,
                    ignore_terminal,
                    grammar,
                    &pair_partition_mask,
                    disallowed_follows,
                    _shared_vocab_dfa_cache,
                    shared_simplify_cache,
                    shared_disallowed_follow_dfa_cache,
                    // pair-partition currently uses the original tokenizer unchanged (`simplify_ms=0`), and
                    // equivalence analysis verifies flat-table compatibility before using it.
                    Some(flat_trans),
                    initial_state_map,
                );
                (result, started_at.elapsed().as_secs_f64() * 1000.0)
            } else {
                (None, 0.0)
            }
        },
    );

    let (direct_partition_pair, direct_partition_ms) = direct_partition_result;
    let (pair_partition_pair, pair_partition_ms) = pair_partition_result;
    let dominant_branch_profile = match (direct_partition_pair.as_ref(), pair_partition_pair.as_ref()) {
        (Some(direct_partition), Some(pair_partition)) => {
            if direct_partition_ms >= pair_partition_ms { direct_partition.profile } else { pair_partition.profile }
        }
        (Some(direct_partition), None) => direct_partition.profile,
        (None, Some(pair_partition)) => pair_partition.profile,
        (None, None) => return None,
    };

    // Collect non-None results and merge.
    let mut pairs = Vec::new();
    if let Some(direct_partition) = direct_partition_pair {
        pairs.push(direct_partition);
    }
    if let Some(pair_partition) = pair_partition_pair {
        pairs.push(pair_partition);
    }

    let num_tokenizer_states = tokenizer.num_states() as usize;
    let max_token_id = vocab.max_token_id();
    let merge_started_at = Instant::now();
    let mut merged = merge_local_id_maps_and_terminal_dwas(
        pairs,
        num_tokenizer_states,
        max_token_id,
    );
    merged.profile.add_assign(dominant_branch_profile);
    merged.profile.id_map_ms += classify_ms;
    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][partition] label={} vocab_tokens={} length0={} length1={} length_pair_partition={} classify_ms={:.3} direct_partition_ms={:.3} pair_partition_ms={:.3} merge_ms={:.3} accounted_id_map_ms={:.3} accounted_terminal_dwa_ms={:.3} accounted_compact_ms={:.3} accounted_total_ms={:.3} total_ms={:.3}",
            partition_label,
            vocab.entries.len(),
            num_zero,
            num_one,
            num_multi_step,
            classify_ms,
            direct_partition_ms,
            pair_partition_ms,
            merge_ms,
            merged.profile.id_map_ms,
            merged.profile.terminal_dwa_ms,
            merged.profile.compact_ms,
            merged.profile.total_ms(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Some(merged)
}
