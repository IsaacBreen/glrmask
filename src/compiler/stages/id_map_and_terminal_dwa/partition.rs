//! Per-partition terminal DWA builder.
//!
//! Given a partition vocab and shared parameters, classify terminals into L1
//! and L2+, build those two pieces independently, then merge them into a
//! single `(InternalIdMap, DWA)` for the partition.

use crate::automata::lexer::Lexer;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::{
    classify_terminal_path_lengths, split_vocab_for_active_l2p_terminals,
};
use crate::compiler::stages::id_map_and_terminal_dwa::grammar_helpers::ignore_transparent_disallowed_follows;
use crate::compiler::stages::id_map_and_terminal_dwa::types::{
    LocalIdMapTerminalDwa, TerminalColoring, TerminalDwaPhaseProfile, TerminalPathLength,
    compile_profile_enabled,
};
use crate::compiler::stages::id_map_and_terminal_dwa::merge::merge_local_id_maps_and_terminal_dwas;
use crate::ds::bitset::BitSet;
use crate::grammar::flat::TerminalID;
use crate::Vocab;

fn split_l2p_vocab_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_SPLIT_L2P_VOCAB")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || trimmed == "1" || trimmed.eq_ignore_ascii_case("true")
            })
            .unwrap_or(true)
    })
}

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
    initial_state_map: Option<&ManyToOneIdMap>,
    _shared_vocab_dfa_cache: Option<&super::l2p::equivalence_analysis::vocab::fast::SharedVocabDfaCache>,
    shared_simplify_cache: Option<&super::l2p::SharedSimplifyCache>,
    shared_disallowed_follow_dfa_cache: Option<&super::l2p::postprocess::SharedDisallowedFollowDfaCache>,
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

    let token_path_disallowed_follows =
        ignore_transparent_disallowed_follows(disallowed_follows, ignore_terminal);

    let classify_started_at = Instant::now();
    let terminal_path_lengths = if force_all_l2p {
        vec![TerminalPathLength::TwoPlus; num_terminals as usize]
    } else {
        classify_terminal_path_lengths(
            tokenizer,
            vocab,
            &token_path_disallowed_follows,
            num_terminals,
            shared_classify_cache,
        )
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

    let use_l2p_vocab_split = has_l2p && split_l2p_vocab_enabled();
    let l2p_vocab_split = use_l2p_vocab_split.then(|| {
        split_vocab_for_active_l2p_terminals(
            tokenizer,
            vocab,
            &token_path_disallowed_follows,
            num_terminals,
            &l2p_mask,
            shared_classify_cache,
        )
    });

    // Build L1 and L2+ terminal DWAs in parallel. L2+ terminals get an
    // additional token split: only tokens that can actually cross an active
    // L2+ terminal boundary go through the expensive L2P NWA builder; the
    // remaining active-terminal-relevant tokens are routed through the cheap
    // L1-style builder over the same L2P terminal set.
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
                    initial_state_map,
                );
                (result, started_at.elapsed().as_secs_f64() * 1000.0)
            } else {
                (None, 0.0)
            }
        },
        || {
            if has_l2p {
                let started_at = Instant::now();
                let Some(split) = l2p_vocab_split.as_ref() else {
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
                        shared_simplify_cache,
                        shared_disallowed_follow_dfa_cache,
                        // L2P currently uses the original tokenizer unchanged (`simplify_ms=0`), and
                        // equivalence analysis verifies flat-table compatibility before using it.
                        Some(flat_trans),
                        initial_state_map,
                    );
                    return (
                        result.into_iter().map(|result| (result, started_at.elapsed().as_secs_f64() * 1000.0)).collect(),
                        started_at.elapsed().as_secs_f64() * 1000.0,
                    );
                };
                let ((boundary_result, boundary_ms), (single_result, single_ms)) = rayon::join(
                    || {
                        if split.boundary_vocab.is_empty() {
                            (None, 0.0)
                        } else {
                            let started_at = Instant::now();
                            let result = super::l2p::build_l2p_id_map_and_terminal_dwa(
                                partition_label,
                                tokenizer,
                                &split.boundary_vocab,
                                terminal_coloring,
                                use_terminal_coloring,
                                ignore_terminal,
                                grammar,
                                &l2p_mask,
                                disallowed_follows,
                                _shared_vocab_dfa_cache,
                                shared_simplify_cache,
                                shared_disallowed_follow_dfa_cache,
                                // L2P currently uses the original tokenizer unchanged (`simplify_ms=0`), and
                                // equivalence analysis verifies flat-table compatibility before using it.
                                Some(flat_trans),
                                initial_state_map,
                            );
                            (result, started_at.elapsed().as_secs_f64() * 1000.0)
                        }
                    },
                    || {
                        if split.single_vocab.is_empty() {
                            (None, 0.0)
                        } else {
                            let started_at = Instant::now();
                            let result = super::l1::build_l1_id_map_and_terminal_dwa(
                                partition_label,
                                tokenizer,
                                &split.single_vocab,
                                terminal_coloring,
                                use_terminal_coloring,
                                ignore_terminal,
                                grammar,
                                &l2p_mask,
                                flat_trans,
                                initial_state_map,
                            );
                            (result, started_at.elapsed().as_secs_f64() * 1000.0)
                        }
                    },
                );

                let mut results = Vec::new();
                if let Some(result) = boundary_result {
                    results.push((result, boundary_ms));
                }
                if let Some(result) = single_result {
                    results.push((result, single_ms));
                }
                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][l2p_vocab_split] partition={} total_tokens={} boundary_tokens={} single_tokens={} irrelevant_tokens={} boundary_ms={:.3} single_ms={:.3}",
                        partition_label,
                        vocab.entries.len(),
                        split.boundary_tokens,
                        split.single_tokens,
                        split.irrelevant_tokens,
                        boundary_ms,
                        single_ms,
                    );
                }
                (results, started_at.elapsed().as_secs_f64() * 1000.0)
            } else {
                (Vec::new(), 0.0)
            }
        },
    );

    let (l1_pair, l1_ms) = l1_result;
    let (l2p_pairs, l2p_ms) = l2p_result;
    let mut dominant_branch: Option<(f64, TerminalDwaPhaseProfile)> = None;
    if let Some(l1) = l1_pair.as_ref() {
        dominant_branch = Some((l1_ms, l1.profile));
    }
    for (pair, elapsed_ms) in &l2p_pairs {
        if dominant_branch.map_or(true, |(current_ms, _)| *elapsed_ms > current_ms) {
            dominant_branch = Some((*elapsed_ms, pair.profile));
        }
    }
    let Some((_, dominant_branch_profile)) = dominant_branch else {
        return None;
    };

    // Collect non-None results and merge.
    let mut pairs = Vec::new();
    if let Some(l1) = l1_pair {
        pairs.push(l1);
    }
    for (l2p, _) in l2p_pairs {
        pairs.push(l2p);
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
