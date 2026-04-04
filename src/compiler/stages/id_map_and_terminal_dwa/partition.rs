//! Per-partition terminal DWA builder.
//!
//! Given a partition vocab and shared parameters, classify terminals into L1
//! and L2+, build those two pieces independently, then merge them into a
//! single `(InternalIdMap, DWA)` for the partition.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::classify_terminal_path_lengths;
use crate::compiler::stages::id_map_and_terminal_dwa::merge::{LocalIdMapTerminalDwa, merge_local_id_maps_and_terminal_dwas};
use crate::compiler::stages::id_map_and_terminal_dwa::types::{
    TerminalColoring, TerminalPathLength, compile_profile_enabled, debug_profile_enabled,
};
use crate::ds::bitset::BitSet;
use crate::Vocab;

/// Build an id_map and terminal DWA for a single vocab partition.
///
/// 1. Classify terminal path lengths into L1 / L2+ masks.
/// 2. Build L1 and L2+ `(InternalIdMap, DWA)` pairs in parallel.
/// 3. Merge the two results.
/// 4. Return a single `(InternalIdMap, DWA)`.
///
/// Returns `None` if the vocab is empty.
pub(crate) fn build_partition_id_map_and_terminal_dwa(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    flat_trans: &Arc<[u32]>,
    _shared_vocab_dfa_cache: Option<&super::l2p::equivalence_analysis::vocab::fast::SharedVocabDfaCache>,
    shared_classify_cache: Option<&super::classify::SharedClassifyCache>,
) -> Option<LocalIdMapTerminalDwa> {
    if vocab.is_empty() {
        return None;
    }

    let total_started_at = Instant::now();
    let num_terminals = grammar.num_terminals as u32;
    // Classify terminals into L1 (single-byte paths) vs L2+ by default.
    // Set GLRMASK_FORCE_ALL_L2P=1 to skip L1 and route everything through L2P.
    let force_all_l2p = std::env::var("GLRMASK_FORCE_ALL_L2P").map_or(false, |v| v == "1");

    let classify_started_at = Instant::now();
    let terminal_path_lengths = if force_all_l2p {
        vec![TerminalPathLength::TwoPlus; num_terminals as usize]
    } else {
        classify_terminal_path_lengths(tokenizer, vocab, disallowed_follows, num_terminals, shared_classify_cache)
    };
    let classify_ms = classify_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut l1_mask = vec![false; num_terminals as usize];
    let mut l2p_mask = vec![false; num_terminals as usize];
    let mut has_l1 = false;
    let mut has_l2p = false;
    let mut num_zero = 0usize;
    let mut num_one = 0usize;
    let mut num_two_plus = 0usize;
    for (i, len) in terminal_path_lengths.iter().enumerate() {
        match len {
            TerminalPathLength::One => {
                l1_mask[i] = true;
                has_l1 = true;
                num_one += 1;
            }
            TerminalPathLength::TwoPlus => {
                l2p_mask[i] = true;
                has_l2p = true;
                num_two_plus += 1;
            }
            TerminalPathLength::Zero => {
                num_zero += 1;
            }
        }
    }

    if debug_profile_enabled() {
        let l1_ids: Vec<u32> = terminal_path_lengths.iter().enumerate()
            .filter(|(_, l)| **l == TerminalPathLength::One)
            .map(|(i, _)| i as u32)
            .collect();
        let l2p_ids: Vec<u32> = terminal_path_lengths.iter().enumerate()
            .filter(|(_, l)| **l == TerminalPathLength::TwoPlus)
            .map(|(i, _)| i as u32)
            .collect();
        let zero_ids: Vec<u32> = terminal_path_lengths.iter().enumerate()
            .filter(|(_, l)| **l == TerminalPathLength::Zero)
            .map(|(i, _)| i as u32)
            .collect();
        eprintln!(
            "[glrmask/debug][partition_classify] label={} l1_terminal_ids={:?} l2p_terminal_ids={:?} zero_terminal_ids={:?}",
            partition_label, l1_ids, l2p_ids, zero_ids,
        );
    }

    // Build L1 and L2+ terminal DWAs in parallel.
    let (l1_result, l2p_result) = rayon::join(
        || {
            if has_l1 {
                let started_at = Instant::now();
                let result = super::l1::build_l1_id_map_and_terminal_dwa(
                    partition_label,
                    tokenizer,
                    vocab,
                    terminal_coloring,
                    use_terminal_coloring,
                    ignore_terminal,
                    grammar,
                    &l1_mask,
                    flat_trans,
                );
                (result, started_at.elapsed().as_secs_f64() * 1000.0)
            } else {
                (None, 0.0)
            }
        },
        || {
            if has_l2p {
                let started_at = Instant::now();
                let result = super::l2p::build_l2p_id_map_and_terminal_dwa(
                    partition_label,
                    tokenizer,
                    vocab,
                    terminal_coloring,
                    use_terminal_coloring,
                    ignore_terminal,
                    grammar,
                    &l2p_mask,
                    disallowed_follows,
                    _shared_vocab_dfa_cache,
                    None,
                );
                (result, started_at.elapsed().as_secs_f64() * 1000.0)
            } else {
                (None, 0.0)
            }
        },
    );

    let (l1_pair, l1_ms) = l1_result;
    let (l2p_pair, l2p_ms) = l2p_result;
    let dominant_branch_profile = match (l1_pair.as_ref(), l2p_pair.as_ref()) {
        (Some(l1), Some(l2p)) => {
            if l1_ms >= l2p_ms { l1.profile } else { l2p.profile }
        }
        (Some(l1), None) => l1.profile,
        (None, Some(l2p)) => l2p.profile,
        (None, None) => return None,
    };

    // Collect non-None results and merge.
    let mut pairs = Vec::new();
    if let Some(l1) = l1_pair {
        pairs.push(l1);
    }
    if let Some(l2p) = l2p_pair {
        pairs.push(l2p);
    }

    let num_tokenizer_states = tokenizer.num_states() as usize;
    let max_token_id = vocab.max_token_id();
    let merge_started_at = Instant::now();
    let mut merged = merge_local_id_maps_and_terminal_dwas(
        &format!("partition:{partition_label}"),
        pairs,
        num_tokenizer_states,
        max_token_id,
    );
    merged.profile.add_assign(dominant_branch_profile);
    merged.profile.id_map_ms += classify_ms;
    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;

    if compile_profile_enabled() || debug_profile_enabled() {
        eprintln!(
            "[glrmask/profile][partition] label={} vocab_tokens={} length0={} length1={} length2plus={} classify_ms={:.3} l1_ms={:.3} l2p_ms={:.3} merge_ms={:.3} accounted_id_map_ms={:.3} accounted_terminal_dwa_ms={:.3} accounted_compact_ms={:.3} accounted_total_ms={:.3} total_ms={:.3}",
            partition_label,
            vocab.entries.len(),
            num_zero,
            num_one,
            num_two_plus,
            classify_ms,
            l1_ms,
            l2p_ms,
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
