//! Compile-time profiling records and profile emission.
//!
//! This module is deliberately presentation-oriented: the compile graph records
//! timings in [`CompilePhaseProfile`], and this module decides how those records
//! are rendered.  Pipeline phases should update fields on the profile; they
//! should not print ad-hoc strings themselves.  Keeping the side effect here
//! makes the pipeline a mathematical object first: a composition of functions
//! from grammar/vocabulary inputs to a runtime artifact plus a report.

use crate::compile::options::env_flag_enabled;
use crate::compile::template_dfa::characterize::TerminalCharacterizationProfile;
use crate::compile::template_dfa::compile_dfa::TemplateCompileProfile;
use std::time::Instant;

/// Return whether compile-summary profiling is enabled by the environment.
pub(crate) fn compile_profile_summary_enabled() -> bool {
    env_flag_enabled("GLRMASK_PROFILE_COMPILE_SUMMARY")
}

/// Return whether any compile-time profiling that affects the compile graph is enabled.
pub(crate) fn compile_profile_enabled() -> bool {
    env_flag_enabled("GLRMASK_PROFILE_COMPILE") || compile_profile_summary_enabled()
}

/// Convert an [`Instant`] into milliseconds.
pub(crate) fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

/// Timing and size counters collected by one compile run.
///
/// The fields intentionally mirror the phase graph, not the historical file
/// layout.  Some fields are still more fine-grained than the paper objects
/// because they correspond to expensive implementation choices that we need to
/// preserve while refactoring.
#[derive(Debug, Default, Clone)]
pub(crate) struct CompilePhaseProfile {
    pub(crate) prepare_ms: f64,
    pub(crate) tokenizer_build_ms: f64,
    pub(crate) analyze_grammar_ms: f64,
    pub(crate) glr_table_ms: f64,
    pub(crate) terminal_coloring_ms: f64,
    pub(crate) disallowed_follows_ms: f64,
    pub(crate) analysis_wall_ms: f64,
    pub(crate) classify_ms: f64,
    pub(crate) id_map_ms: f64,
    pub(crate) terminal_dwa_ms: f64,
    pub(crate) templates_ms: f64,
    pub(crate) compact_ms: f64,
    pub(crate) split_terminal_dwa_total_ms: f64,
    pub(crate) global_merge_ms: f64,
    pub(crate) scan_relation_collect_ms: f64,
    pub(crate) can_match_materialize_ms: f64,
    pub(crate) shared_id_reconcile_ms: f64,
    pub(crate) can_match_pipeline_ms: f64,
    pub(crate) terminal_dwa_interned_ranges_before_can_match_reconcile: usize,
    pub(crate) can_match_interned_ranges_before_can_match_reconcile: usize,
    pub(crate) terminal_can_match_joint_interned_ranges_before_reconcile: usize,
    pub(crate) terminal_can_match_joint_interned_ranges: usize,
    pub(crate) internal_token_bytes_ms: f64,
    pub(crate) parser_dwa_ms: f64,
    pub(crate) parser_dwa_interned_ranges: usize,
    pub(crate) can_match_interned_ranges: usize,
    pub(crate) parser_can_match_joint_interned_ranges: usize,
    pub(crate) finalize_ms: f64,
    pub(crate) compile_ms: f64,
    pub(crate) total_ms: f64,
}

/// Destination for profile lines.
///
/// The default implementation writes to stderr, but tests and future bindings can
/// pass a different sink without changing the compile graph.
pub(crate) trait CompileProfileSink {
    fn emit_line(&mut self, line: &str);
}

/// Profile sink used by the current environment-variable based diagnostics.
pub(crate) struct StderrCompileProfileSink;

impl CompileProfileSink for StderrCompileProfileSink {
    fn emit_line(&mut self, line: &str) {
        eprintln!("{line}");
    }
}

/// Render the one-line compile profile summary used by existing benchmarks.
pub(crate) fn compile_profile_summary_line(
    source_kind: Option<&str>,
    import_ms: Option<f64>,
    profile: &CompilePhaseProfile,
) -> String {
    let source = source_kind.unwrap_or("grammar");
    let import_fragment = import_ms
        .map(|ms| format!(" import_ms={ms:.3}"))
        .unwrap_or_default();

    format!(
        "[glrmask/profile][compile] source={}{} prepare_ms={:.3} tokenizer_build_ms={:.3} analyze_grammar_ms={:.3} glr_table_ms={:.3} terminal_coloring_ms={:.3} disallowed_follows_ms={:.3} analysis_wall_ms={:.3} classify_ms={:.3} id_map_ms={:.3} terminal_dwa_ms={:.3} split_terminal_dwa_total_ms={:.3} global_merge_ms={:.3} templates_ms={:.3} compact_ms={:.3} scan_relation_collect_ms={:.3} can_match_materialize_ms={:.3} shared_id_reconcile_ms={:.3} can_match_pipeline_ms={:.3} terminal_dwa_interned_ranges_before_can_match_reconcile={} can_match_interned_ranges_before_can_match_reconcile={} terminal_can_match_joint_interned_ranges_before_reconcile={} terminal_can_match_joint_interned_ranges={} internal_token_bytes_ms={:.3} parser_dwa_ms={:.3} parser_dwa_interned_ranges={} can_match_interned_ranges={} parser_can_match_joint_interned_ranges={} finalize_ms={:.3} compile_ms={:.3} total_ms={:.3}",
        source,
        import_fragment,
        profile.prepare_ms,
        profile.tokenizer_build_ms,
        profile.analyze_grammar_ms,
        profile.glr_table_ms,
        profile.terminal_coloring_ms,
        profile.disallowed_follows_ms,
        profile.analysis_wall_ms,
        profile.classify_ms,
        profile.id_map_ms,
        profile.terminal_dwa_ms,
        profile.split_terminal_dwa_total_ms,
        profile.global_merge_ms,
        profile.templates_ms,
        profile.compact_ms,
        profile.scan_relation_collect_ms,
        profile.can_match_materialize_ms,
        profile.shared_id_reconcile_ms,
        profile.can_match_pipeline_ms,
        profile.terminal_dwa_interned_ranges_before_can_match_reconcile,
        profile.can_match_interned_ranges_before_can_match_reconcile,
        profile.terminal_can_match_joint_interned_ranges_before_reconcile,
        profile.terminal_can_match_joint_interned_ranges,
        profile.internal_token_bytes_ms,
        profile.parser_dwa_ms,
        profile.parser_dwa_interned_ranges,
        profile.can_match_interned_ranges,
        profile.parser_can_match_joint_interned_ranges,
        profile.finalize_ms,
        profile.compile_ms,
        profile.total_ms,
    )
}

/// Emit the compile-summary line to an explicit sink.
pub(crate) fn emit_compile_profile_summary_to_sink(
    source_kind: Option<&str>,
    import_ms: Option<f64>,
    profile: &CompilePhaseProfile,
    sink: &mut dyn CompileProfileSink,
) {
    if compile_profile_summary_enabled() {
        let line = compile_profile_summary_line(source_kind, import_ms, profile);
        sink.emit_line(&line);
    }
}

/// Backward-compatible entry point used by import frontends.
pub(crate) fn emit_compile_profile_summary(
    source_kind: Option<&str>,
    import_ms: Option<f64>,
    profile: &CompilePhaseProfile,
) {
    let mut sink = StderrCompileProfileSink;
    emit_compile_profile_summary_to_sink(source_kind, import_ms, profile, &mut sink);
}

/// Render and emit the detailed template-build profile.
///
/// This centralizes the existing template profile side-effect.  The line format
/// is intentionally unchanged so old benchmark parsing scripts do not break.
pub(crate) fn emit_template_profile_summary(
    characterization_profile: &TerminalCharacterizationProfile,
    template_profile: &TemplateCompileProfile,
) {
    if !compile_profile_enabled() {
        return;
    }

    let mut sink = StderrCompileProfileSink;
    sink.emit_line(&format!(
        "[glrmask/profile][templates] terminals={} action_signature_classes={} action_quotient_hits={} max_action_signature_multiplicity={} characterization_signature_ms={:.3} characterization_ms={:.3} characterization_fanout_ms={:.3} characterization_validation_ms={:.3} characterization_total_ms={:.3} characterization_quotient_disabled={} unique_characterizations={} compiled_characterizations={} template_quotient_hits={} max_characterization_multiplicity={} build_nfa_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} template_fanout_ms={:.3} template_validation_ms={:.3} template_total_ms={:.3} template_wall_ms={:.3} template_minimize_skipped={} avg_nfa_states={:.2} avg_nfa_transitions={:.2} avg_premin_dfa_states={:.2} avg_premin_dfa_transitions={:.2} avg_dfa_states={:.2} avg_dfa_transitions={:.2} max_dfa_states={} max_dfa_transitions={}",
        characterization_profile.terminals,
        characterization_profile.unique_action_signatures,
        characterization_profile.quotient_hits,
        characterization_profile.max_action_signature_multiplicity,
        characterization_profile.signature_ms,
        characterization_profile.characterize_ms,
        characterization_profile.fanout_ms,
        characterization_profile.validation_ms,
        characterization_profile.total_ms,
        characterization_profile.quotient_disabled,
        template_profile.unique_characterizations,
        template_profile.compiled_characterizations,
        template_profile.quotient_hits,
        template_profile.max_characterization_multiplicity,
        template_profile.build_nfa_ms,
        template_profile.determinize_ms,
        template_profile.minimize_ms,
        template_profile.fanout_ms,
        template_profile.validation_ms,
        template_profile.total_ms,
        template_profile.wall_ms,
        template_profile.minimize_skipped,
        template_profile.avg_nfa_states(),
        template_profile.avg_nfa_transitions(),
        template_profile.avg_premin_dfa_states(),
        template_profile.avg_premin_dfa_transitions(),
        template_profile.avg_dfa_states(),
        template_profile.avg_dfa_transitions(),
        template_profile.max_dfa_states,
        template_profile.max_dfa_transitions,
    ));
}
