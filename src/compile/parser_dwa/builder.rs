//! High-level Parser-DWA build entrypoints.
//!
//! This file deliberately contains only phase ordering.  It should read like
//! the proof outline for the construction:
//!
//! 1. Compose the Terminal DWA with parser stack-effect templates to obtain a
//!    parser NWA.
//! 2. Resolve temporary negative labels produced by template construction.
//! 3. Determinize over parser-state labels while retaining support sets.
//! 4. Derive legal fallback/default domains from those supports.
//! 5. Normalize defaults and final weights.
//! 6. Re-determinize fallback semantics.
//! 7. Optionally minimize the resulting weighted DWA.

use std::time::Instant;

use crate::Vocab;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::parser::glr::analysis::AnalyzedGrammar;
use crate::parser::glr::table::GLRTable;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::compiler::stages::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::stages::templates::Templates;
use crate::compile::terminal_dwa::types::compile_profile_enabled;

use super::compose_nwa::build_parser_nwa_from_terminal_dwa;
use super::determinize::{
    build_possible_outgoing_ids_by_state, determinize_parser_dwa_with_fallbacks,
    determinize_with_supports,
};
use super::options::ParserDwaOptions;
use super::optimize::{optimize_parser_dwa_defaults, subtract_final_weights_from_outgoing_dwa};
use super::profiling::{elapsed_ms, ParserDwaProfile};

/// Named inputs to Parser-DWA construction.
///
/// The `vocab` and `id_map` arguments are retained because they are part of
/// the surrounding compile-stage contract.  The current mathematical
/// construction does not need them directly: the Terminal DWA already carries
/// the lexer-state/token-pair weights, and the templates already carry parser
/// stack-effect recognizers.
pub(crate) struct ParserDwaBuildInputs<'a> {
    pub(crate) table: &'a GLRTable,
    pub(crate) grammar: &'a AnalyzedGrammar,
    pub(crate) terminal_dwa: &'a DWA,
    pub(crate) templates: Templates,
    pub(crate) vocab: &'a Vocab,
    pub(crate) id_map: &'a InternalIdMap,
}

/// Named output from Parser-DWA construction.
pub(crate) struct ParserDwaBuildOutput {
    pub(crate) dwa: DWA,
    pub(crate) profile: ParserDwaProfile,
}

/// Build the Parser DWA from a Terminal DWA and precomputed terminal templates.
pub(crate) fn build_parser_dwa_from_terminal_dwa_with_templates(
    inputs: ParserDwaBuildInputs<'_>,
) -> ParserDwaBuildOutput {
    let ParserDwaBuildInputs {
        table,
        grammar,
        terminal_dwa,
        templates,
        vocab: _vocab,
        id_map: _id_map,
    } = inputs;

    let total_started_at = Instant::now();
    let profiling_enabled = compile_profile_enabled();
    let (terminal_dwa_transition_count, terminal_dwa_interned_ranges) = if profiling_enabled {
        let stats = terminal_dwa.stats();
        (stats.transitions, stats.interned_ranges)
    } else {
        (0, 0)
    };

    let Some((mut parser_nwa, parser_nwa_profile)) =
        build_parser_nwa_from_terminal_dwa(terminal_dwa, grammar, templates)
    else {
        let profile = ParserDwaProfile::empty(
            terminal_dwa.states().len(),
            terminal_dwa_transition_count,
            terminal_dwa_interned_ranges,
            false,
            elapsed_ms(total_started_at),
        );
        if profiling_enabled {
            profile.emit_detail();
        }
        return ParserDwaBuildOutput {
            dwa: DWA::new(0, 0),
            profile,
        };
    };

    let resolve_negative_started_at = Instant::now();
    resolve_negative_codes_in_nwa(&mut parser_nwa);
    let resolve_negative_ms = elapsed_ms(resolve_negative_started_at);

    let support_determinize_started_at = Instant::now();
    let determinized = determinize_with_supports(&parser_nwa, Some(table.num_states));
    let support_determinize_ms = elapsed_ms(support_determinize_started_at);
    let mut parser_dwa_pre_minimize = determinized.dwa;

    let possible_outgoing_started_at = Instant::now();
    let possible_by_state = build_possible_outgoing_ids_by_state(
        &parser_nwa,
        &determinized.supports,
        table.num_states,
    );
    let possible_outgoing_ms = elapsed_ms(possible_outgoing_started_at);

    let default_opt_started_at = Instant::now();
    optimize_parser_dwa_defaults(
        &mut parser_dwa_pre_minimize,
        &possible_by_state,
        table.num_states,
    );
    let default_opt_ms = elapsed_ms(default_opt_started_at);

    let subtract_final_started_at = Instant::now();
    subtract_final_weights_from_outgoing_dwa(&mut parser_dwa_pre_minimize);
    let subtract_final_ms = elapsed_ms(subtract_final_started_at);

    let fallback_determinize_started_at = Instant::now();
    parser_dwa_pre_minimize = determinize_parser_dwa_with_fallbacks(
        &parser_dwa_pre_minimize,
        &possible_by_state,
        table.num_states,
    );
    let fallback_determinize_ms = elapsed_ms(fallback_determinize_started_at);

    let pre_minimize_state_count = parser_dwa_pre_minimize.states().len();
    let pre_minimize_transition_count = parser_dwa_pre_minimize.num_transitions();
    let options = ParserDwaOptions::from_environment(
        pre_minimize_state_count,
        pre_minimize_transition_count,
    );

    let (minimized, minimize_ms, post_minimize_state_count, post_minimize_transition_count) =
        if options.skip_minimization {
            (
                parser_dwa_pre_minimize,
                0.0,
                pre_minimize_state_count,
                pre_minimize_transition_count,
            )
        } else {
            let minimize_started_at = Instant::now();
            let minimized = minimize(&parser_dwa_pre_minimize);
            let minimize_ms = elapsed_ms(minimize_started_at);
            let post_minimize_state_count = minimized.states().len();
            let post_minimize_transition_count = minimized.num_transitions();
            (
                minimized,
                minimize_ms,
                post_minimize_state_count,
                post_minimize_transition_count,
            )
        };

    let profile = ParserDwaProfile {
        terminal_dwa_states: terminal_dwa.states().len(),
        terminal_dwa_transitions: terminal_dwa_transition_count,
        terminal_dwa_interned_ranges,
        parser_nwa_built: true,
        parser_nwa_states: parser_nwa.states().len(),
        parser_nwa_start_states: parser_nwa.start_states().len(),
        pre_minimize_states: pre_minimize_state_count,
        pre_minimize_transitions: pre_minimize_transition_count,
        post_minimize_states: post_minimize_state_count,
        post_minimize_transitions: post_minimize_transition_count,
        minimize_skipped: options.skip_minimization,
        state_prep_ms: parser_nwa_profile.state_prep_ms,
        compose_state_ms: parser_nwa_profile.compose_state_ms,
        parser_nwa_build_ms: parser_nwa_profile.parser_nwa_build_ms,
        resolve_negative_ms,
        support_determinize_ms,
        possible_outgoing_ms,
        default_opt_ms,
        subtract_final_ms,
        fallback_determinize_ms,
        minimize_ms,
        total_ms: elapsed_ms(total_started_at),
    };

    if profiling_enabled {
        profile.emit_detail();
    }

    ParserDwaBuildOutput {
        dwa: minimized,
        profile,
    }
}

/// Compatibility wrapper used by the existing compile pipeline.
pub(crate) fn build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    terminal_dwa: &DWA,
    templates: Templates,
    vocab: &Vocab,
    id_map: &InternalIdMap,
) -> DWA {
    build_parser_dwa_from_terminal_dwa_with_templates(ParserDwaBuildInputs {
        table,
        grammar,
        terminal_dwa,
        templates,
        vocab,
        id_map,
    })
    .dwa
}
