//! Terminal-DWA and scan-relation phase group.
//!
//! The same lexer DFA and vocabulary bytes induce two different relations:
//!
//! 1. the Terminal DWA relation, which sees complete terminal strings, and
//! 2. the scan relation, which sees byte fragments that may end in the middle of
//!    a terminal match.
//!
//! They are close enough to share caches and byte-transition precomputation, but
//! different enough that they must not share an equivalence proof.

use std::sync::Arc;
use std::time::Instant;

use crate::Vocab;
use crate::compile::pipeline::context::{
    GrammarAnalysisOutput,
    TerminalAndScanOutput,
    TerminalScanSupport,
};
use crate::compile::profiling::{CompilePhaseProfile, elapsed_ms};
use crate::compile::scan_relation;
use crate::compile::terminal_dwa::classify::{
    SharedClassifyCache,
    classify_terminal_path_lengths,
};
use crate::grammar::flat::GrammarDef;

/// Precompute vocabulary/lexer data shared by Terminal-DWA and scan-relation work.
pub(crate) fn precompute_terminal_scan_support(
    analysis: &GrammarAnalysisOutput,
    vocab: &Vocab,
    profile: &mut CompilePhaseProfile,
) -> TerminalScanSupport {
    let shared_classify_cache = SharedClassifyCache::new();

    let classify_started_at = Instant::now();
    let _terminal_path_lengths = classify_terminal_path_lengths(
        &analysis.tokenizer,
        vocab,
        &analysis.disallowed_follows,
        analysis.analyzed_grammar.num_terminals,
        Some(&shared_classify_cache),
    );
    profile.classify_ms = elapsed_ms(classify_started_at);

    let flat_trans_started_at = Instant::now();
    let flat_transitions: Arc<[u32]> = Arc::from(
        crate::compile::terminal_dwa::direct_partition::build_flat_transition_table(
            &analysis.tokenizer,
        ),
    );
    let flat_trans_ms = elapsed_ms(flat_trans_started_at);

    let global_max_length_started_at = Instant::now();
    let global_max_length_state_map = crate::compile::terminal_dwa::build_global_max_length_state_map(
        &analysis.tokenizer,
        vocab,
        &flat_transitions,
    );
    let global_max_length_ms = elapsed_ms(global_max_length_started_at);

    profile.terminal_dwa_ms += flat_trans_ms;
    profile.id_map_ms += global_max_length_ms;

    TerminalScanSupport {
        shared_classify_cache,
        flat_transitions,
        global_max_length_state_map,
    }
}

/// Build the Terminal DWA and scan relation in parallel.
pub(crate) fn build_terminal_dwa_and_scan_relation(
    prepared_grammar: &GrammarDef,
    analysis: &GrammarAnalysisOutput,
    support: &TerminalScanSupport,
    vocab: &Vocab,
    profile: &mut CompilePhaseProfile,
) -> TerminalAndScanOutput {
    let ((terminal_dwa, terminal_phase_profile), scan_relation_result) = rayon::join(
        || {
            crate::compile::terminal_dwa::build_terminal_dwa_with_precomputed_global_max_length(
                &analysis.tokenizer,
                vocab,
                &analysis.terminal_coloring,
                true,
                prepared_grammar.ignore_terminal,
                &analysis.analyzed_grammar,
                &analysis.disallowed_follows,
                Arc::clone(&support.flat_transitions),
                &support.global_max_length_state_map,
                Some(&support.shared_classify_cache),
            )
        },
        || {
            scan_relation::compute_scan_relation_for_vocab(
                &analysis.tokenizer,
                vocab,
                scan_relation::ScanRelationConfig,
            )
        },
    );

    profile.id_map_ms += terminal_phase_profile.id_map_ms;
    profile.terminal_dwa_ms += terminal_phase_profile.terminal_dwa_ms;
    profile.compact_ms += terminal_phase_profile.compact_ms;
    profile.split_terminal_dwa_total_ms = terminal_phase_profile.split_terminal_dwa_total_ms;
    profile.global_merge_ms = terminal_phase_profile.global_merge_ms;

    let scan_relation_profile = scan_relation_result.profile;
    profile.scan_relation_collect_ms = scan_relation_profile.scan_relation_collect_ms;
    profile.can_match_materialize_ms = scan_relation_profile.scan_relation_vocab_ms;

    TerminalAndScanOutput {
        terminal_dwa,
        scan_relation: scan_relation_result,
    }
}
