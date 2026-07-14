//! Per-partition terminal DWA builder.
//!
//! Given a partition vocab and shared parameters, classify terminals into L1
//! and L2+, build those two pieces independently, then merge them into a
//! single `(InternalIdMap, DWA)` for the partition.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::Lexer;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::compiler::stages::id_map_and_terminal_dwa::classify::{
    classify_terminal_path_lengths, split_vocab_for_active_l2p_terminals,
};
use crate::compiler::stages::id_map_and_terminal_dwa::types::{
    PartitionTerminalDwas, TerminalColoring, TerminalDwaPhaseProfile, TerminalPathLength,
    compile_profile_enabled, compile_profile_join,
};
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
/// 3. Preserve the L1, L2P, and split-off-L2P-vocab L1 pieces separately so
///    callers can merge like families across all vocabulary partitions.
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
    always_allowed_follows: &[Vec<TerminalID>],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    token_path_disallowed_follows: &Arc<BTreeMap<u32, BitSet>>,
    normalized_token_path_disallowed_follows: &Arc<[BitSet]>,
    flat_trans: &Arc<[u32]>,
    initial_state_map: Option<&ManyToOneIdMap>,
    shared_vocab_dfa_cache: Option<&super::l2p::equivalence_analysis::vocab::fast::SharedVocabDfaCache>,
    shared_original_vocab_dfa_cache: Option<&super::l2p::equivalence_analysis::vocab::fast::SharedVocabDfaCache>,
    shared_original_vocab_analysis_dfa_cache: Option<&super::l2p::equivalence_analysis::vocab::fast::SharedVocabAnalysisDfaCache>,
    shared_transition_cache: Option<&std::sync::OnceLock<super::l2p::equivalence_analysis::compat::FlatTransitionCache>>,
    shared_ti_output_cache: Option<&super::l2p::SharedTiTokenizerOutputCache>,
    shared_classify_cache: Option<&super::classify::SharedClassifyCache>,
) -> Option<PartitionTerminalDwas> {
    if vocab.is_empty() {
        return None;
    }

    let total_started_at = Instant::now();
    let pre_classify_setup_started_at = Instant::now();
    let num_terminals = grammar.num_terminals as u32;
    // Classify terminals into L1 (single-byte paths) vs L2+ by default.
    // Set GLRMASK_FORCE_ALL_L2P=1 to skip L1 and route everything through L2P.
    let force_all_l2p =
        std::env::var("GLRMASK_FORCE_ALL_L2P").map_or(false, |v| v == "1");

    let pre_classify_setup_ms =
        pre_classify_setup_started_at.elapsed().as_secs_f64() * 1000.0;

    let classify_started_at = Instant::now();
    let terminal_path_lengths = if force_all_l2p {
        vec![TerminalPathLength::TwoPlus; num_terminals as usize]
    } else {
        classify_terminal_path_lengths(
            partition_label,
            tokenizer,
            vocab,
            token_path_disallowed_follows.as_ref(),
            num_terminals,
            shared_classify_cache,
        )
    };
    let classify_ms = classify_started_at.elapsed().as_secs_f64() * 1000.0;

    let routing_started_at = Instant::now();
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
            flat_trans,
            vocab,
            token_path_disallowed_follows,
            num_terminals,
            &l2p_mask,
            shared_classify_cache,
        )
    });
    let has_split_l1 = l2p_vocab_split
        .as_ref()
        .is_some_and(|split| split.single_tokens != 0);
    // Classification already initializes this shared byte-major DFA table.
    // L1 exact equivalence walks many states at a fixed token byte, for which
    // the transposed layout avoids a 256-word stride through the row-major table.
    let l1_transitions_by_byte = (has_l1 || has_split_l1).then(|| {
        shared_classify_cache
            .and_then(|cache| cache.get())
            .map(|bytesets| bytesets.transitions_by_byte())
    }).flatten();
    let shared_l1_generic_nfa_topology = if has_l1
        && has_split_l1
        && tokenizer.has_epsilon_transitions()
        && !tokenizer.has_deterministic_dispatch()
        && super::l1::l1_generic_nfa_token_bounded_view_enabled(
            tokenizer.num_states() as usize,
            vocab.entries.len(),
        )
    {
        let raw_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
        let tokens = vocab
            .entries
            .values()
            .map(|bytes| bytes.as_ref())
            .collect::<Vec<_>>();
        Some(
            super::l2p::equivalence_analysis::state_equivalence::nfa::build_token_bounded_analysis_topology(
                tokenizer,
                &raw_states,
                &tokens,
            ),
        )
    } else {
        None
    };

    let routing_ms = routing_started_at.elapsed().as_secs_f64() * 1000.0;

    // Build L1 and L2+ terminal DWAs in parallel. L2+ terminals get an
    // additional token split: only tokens that can actually cross an active
    // L2+ terminal boundary go through the expensive L2P NWA builder; the
    // remaining active-terminal-relevant tokens are routed through the cheap
    // L1-style builder over the same L2P terminal set.
    let branch_build_started_at = Instant::now();
    let (l1_result, l2p_result) = compile_profile_join(
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
                    l1_transitions_by_byte,
                    initial_state_map,
                    shared_l1_generic_nfa_topology.as_ref(),
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
                        always_allowed_follows,
                        &l2p_mask,
                        disallowed_follows,
                        Some(token_path_disallowed_follows.as_ref()),
                        Some(normalized_token_path_disallowed_follows.as_ref()),
                        shared_vocab_dfa_cache,
                        shared_original_vocab_dfa_cache,
                        shared_original_vocab_analysis_dfa_cache,
                        shared_transition_cache,
                        shared_ti_output_cache,
                        // All L2P work keeps raw lexer-state coordinates; equivalence
                        // analysis verifies flat-table compatibility before using it.
                        Some(flat_trans),
                        initial_state_map,
                    );
                    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                    return ((result, 0.0), (None, 0.0), elapsed_ms);
                };
                let ((boundary_result, boundary_ms), (single_result, single_ms)) = compile_profile_join(
                    || {
                        if split.boundary_tokens == 0 {
                            (None, 0.0)
                        } else {
                            let started_at = Instant::now();
                            let boundary_vocab = split.boundary_vocab(vocab);
                            if std::env::var_os("GLRMASK_DUMP_L2P_BOUNDARY_VOCAB").is_some()
                                && matches!(partition_label, "p7" | "p8")
                            {
                                eprintln!(
                                    "[glrmask/dump][l2p_boundary_vocab] partition={} count={}",
                                    partition_label,
                                    boundary_vocab.entries.len(),
                                );
                                for (&token_id, bytes) in boundary_vocab.entries.iter() {
                                    eprintln!(
                                        "[glrmask/dump][l2p_boundary_vocab] partition={} token_id={} bytes={:?}",
                                        partition_label,
                                        token_id,
                                        bytes,
                                    );
                                }
                            }
                            let result = super::l2p::build_l2p_id_map_and_terminal_dwa(
                                partition_label,
                                tokenizer,
                                &boundary_vocab,
                                terminal_coloring,
                                use_terminal_coloring,
                                ignore_terminal,
                                grammar,
                                always_allowed_follows,
                                &l2p_mask,
                                disallowed_follows,
                                Some(token_path_disallowed_follows.as_ref()),
                                Some(normalized_token_path_disallowed_follows.as_ref()),
                                shared_vocab_dfa_cache,
                                shared_original_vocab_dfa_cache,
                                shared_original_vocab_analysis_dfa_cache,
                                shared_transition_cache,
                                shared_ti_output_cache,
                                        // All L2P work keeps raw lexer-state coordinates; equivalence
                                // analysis verifies flat-table compatibility before using it.
                                Some(flat_trans),
                                initial_state_map,
                            );
                            (result, started_at.elapsed().as_secs_f64() * 1000.0)
                        }
                    },
                    || {
                        if split.single_tokens == 0 {
                            (None, 0.0)
                        } else {
                            let started_at = Instant::now();
                            let single_vocab = split.single_vocab(vocab);
                            let result = super::l1::build_l1_id_map_and_terminal_dwa(
                                partition_label,
                                tokenizer,
                                &single_vocab,
                                terminal_coloring,
                                use_terminal_coloring,
                                ignore_terminal,
                                grammar,
                                &l2p_mask,
                                flat_trans,
                                l1_transitions_by_byte,
                                initial_state_map,
                                shared_l1_generic_nfa_topology.as_ref(),
                            );
                            (result, started_at.elapsed().as_secs_f64() * 1000.0)
                        }
                    },
                );

                if compile_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][l2p_vocab_split] partition={} total_tokens={} adjacent_tokens={} boundary_tokens={} single_tokens={} irrelevant_tokens={} boundary_ms={:.3} single_ms={:.3}",
                        partition_label,
                        vocab.entries.len(),
                        split.adjacent_tokens,
                        split.boundary_tokens,
                        split.single_tokens,
                        split.irrelevant_tokens,
                        boundary_ms,
                        single_ms,
                    );
                }
                (
                    (boundary_result, boundary_ms),
                    (single_result, single_ms),
                    started_at.elapsed().as_secs_f64() * 1000.0,
                )
            } else {
                ((None, 0.0), (None, 0.0), 0.0)
            }
        },
    );
    let branch_build_wall_ms = branch_build_started_at.elapsed().as_secs_f64() * 1000.0;

    let post_branch_started_at = Instant::now();
    let (l1_pair, l1_ms) = l1_result;
    let ((l2p_pair, l2p_boundary_ms), (l2p_single_l1_pair, l2p_single_ms), l2p_ms) =
        l2p_result;
    let mut dominant_branch: Option<(f64, TerminalDwaPhaseProfile)> = None;
    if let Some(l1) = l1_pair.as_ref() {
        dominant_branch = Some((l1_ms, l1.profile));
    }
    if let Some(l2p) = l2p_pair.as_ref() {
        if dominant_branch.map_or(true, |(current_ms, _)| l2p_boundary_ms > current_ms) {
            dominant_branch = Some((l2p_boundary_ms, l2p.profile));
        }
    }
    if let Some(split_l1) = l2p_single_l1_pair.as_ref() {
        if dominant_branch.map_or(true, |(current_ms, _)| l2p_single_ms > current_ms) {
            dominant_branch = Some((l2p_single_ms, split_l1.profile));
        }
    }
    let Some((_, dominant_branch_profile)) = dominant_branch else {
        return None;
    };
    let post_branch_ms = post_branch_started_at.elapsed().as_secs_f64() * 1000.0;

    let profile_bookkeeping_started_at = Instant::now();
    let mut partition_profile = dominant_branch_profile;
    partition_profile.id_map_ms += classify_ms;
    let profile_bookkeeping_ms =
        profile_bookkeeping_started_at.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_started_at.elapsed().as_secs_f64() * 1000.0;
    let accounted_wall_ms = pre_classify_setup_ms
        + classify_ms
        + routing_ms
        + branch_build_wall_ms
        + post_branch_ms
        + profile_bookkeeping_ms;
    let timing_residual_ms = (total_ms - accounted_wall_ms).max(0.0);

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][partition] label={} vocab_tokens={} length0={} length1={} length2plus={} pre_classify_setup_ms={:.3} classify_ms={:.3} routing_ms={:.3} branch_build_wall_ms={:.3} l1_branch_wall_ms={:.3} l2p_branch_wall_ms={:.3} l2p_boundary_wall_ms={:.3} l2p_single_l1_wall_ms={:.3} post_branch_ms={:.3} profile_bookkeeping_ms={:.3} critical_path_id_map_ms={:.3} critical_path_terminal_dwa_ms={:.3} critical_path_compact_ms={:.3} critical_path_profile_ms={:.3} accounted_wall_ms={:.3} timing_residual_ms={:.3} total_ms={:.3}",
            partition_label,
            vocab.entries.len(),
            num_zero,
            num_one,
            num_two_plus,
            pre_classify_setup_ms,
            classify_ms,
            routing_ms,
            branch_build_wall_ms,
            l1_ms,
            l2p_ms,
            l2p_boundary_ms,
            l2p_single_ms,
            post_branch_ms,
            profile_bookkeeping_ms,
            partition_profile.id_map_ms,
            partition_profile.terminal_dwa_ms,
            partition_profile.compact_ms,
            partition_profile.total_ms(),
            accounted_wall_ms,
            timing_residual_ms,
            total_ms,
        );
    }

    let result = PartitionTerminalDwas {
        l1: l1_pair,
        l2p: l2p_pair,
        l2p_single_l1: l2p_single_l1_pair,
        profile: partition_profile,
    };
    debug_assert!(!result.is_empty());
    Some(result)
}
