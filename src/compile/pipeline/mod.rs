//! Compile pipeline orchestration.
//!
//! This module is intentionally small.  It names the compile graph and delegates
//! each mathematical construction to a phase module.  The old implementation hid
//! the graph in one long local-variable soup; this version makes the pipeline a
//! sequence of typed transformations:
//!
//! ```text
//! GrammarDef × Vocab
//!   -> normalized GrammarDef
//!   -> tokenizer + grammar/table facts
//!   -> Terminal DWA + scan relation + templates
//!   -> Parser DWA + shared internal ID space
//!   -> runtime Constraint
//! ```
//!
//! No phase in this module asks environment questions directly, and no phase in
//! this module prints profile lines directly.  Those side effects are centralized
//! in `compile::options`, `compile::thread_pool`, and `compile::profiling`.

mod analysis;
mod context;
mod counts;
mod finalize;
mod phases;
mod reconcile;
mod templates;
mod terminal_scan;

use std::time::Instant;

use context::TemplateOutput;

use crate::Vocab;
use crate::compile::profiling::{
    CompilePhaseProfile,
    compile_profile_summary_enabled,
    elapsed_ms,
    emit_compile_profile_summary,
};
use crate::compile::thread_pool::run_with_compile_thread_pool;
use crate::compiler::grammar::transforms::prepare_grammar_transforms_only;
use crate::grammar::flat::GrammarDef;
use crate::runtime::Constraint;

/// Compile an owned grammar into a runtime constraint.
pub(crate) fn compile_owned(grammar: GrammarDef, vocab: &Vocab) -> Constraint {
    if compile_profile_summary_enabled() {
        let (constraint, profile) = compile_owned_profiled(grammar, vocab);
        emit_compile_profile_summary(None, None, &profile);
        return constraint;
    }

    let prepared_grammar = prepare_grammar_transforms_only(grammar);
    compile_prepared(prepared_grammar, vocab)
}

/// Compile an owned grammar and return a structured compile-phase profile.
pub(crate) fn compile_owned_profiled(
    grammar: GrammarDef,
    vocab: &Vocab,
) -> (Constraint, CompilePhaseProfile) {
    let total_started_at = Instant::now();
    let prepare_started_at = Instant::now();
    let prepared_grammar = prepare_grammar_transforms_only(grammar);
    let prepare_ms = elapsed_ms(prepare_started_at);

    let (constraint, mut profile) = compile_prepared_with_profile(prepared_grammar, vocab);
    profile.prepare_ms = prepare_ms;
    profile.total_ms = elapsed_ms(total_started_at);
    (constraint, profile)
}

/// Compile a grammar that has already been normalized by the import layer.
pub(crate) fn compile_prepared(prepared_grammar: GrammarDef, vocab: &Vocab) -> Constraint {
    compile_prepared_with_profile(prepared_grammar, vocab).0
}

/// Compile a prepared grammar and return the profile produced by the phase graph.
pub(crate) fn compile_prepared_with_profile(
    prepared_grammar: GrammarDef,
    vocab: &Vocab,
) -> (Constraint, CompilePhaseProfile) {
    run_with_compile_thread_pool(|| {
        let compile_started_at = Instant::now();
        let mut profile = CompilePhaseProfile::default();

        let analysis = analysis::build_grammar_analysis(&prepared_grammar, vocab, &mut profile);
        let terminal_scan_support =
            terminal_scan::precompute_terminal_scan_support(&analysis, vocab, &mut profile);

        // Keep the phase boundaries explicit.  A later performance pass may recover
        // safe parallel execution between `build_terminal_dwa_and_scan_relation` and
        // `build_templates` by returning phase-local profile deltas rather than
        // mutating the shared profile inside both branches.
        let terminal_and_scan = terminal_scan::build_terminal_dwa_and_scan_relation(
            &prepared_grammar,
            &analysis,
            &terminal_scan_support,
            vocab,
            &mut profile,
        );
        let template_output = templates::build_templates(&analysis, &mut profile);
        let TemplateOutput {
            templates,
            template_dfas_by_terminal,
        } = template_output;

        let reconciled = reconcile::reconcile_and_build_parser_dwa(
            &analysis,
            terminal_and_scan,
            templates,
            vocab,
            &mut profile,
        );

        let constraint = finalize::finalize_runtime_constraint(
            prepared_grammar,
            analysis,
            reconciled,
            template_dfas_by_terminal,
            vocab,
            &mut profile,
        );
        profile.compile_ms = elapsed_ms(compile_started_at);

        (constraint, profile)
    })
}
