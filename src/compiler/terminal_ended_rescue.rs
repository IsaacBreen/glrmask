//! Phase-2 template rescue: rewrite the original source grammar into
//! terminal-ended normal form, then compile it with the ordinary template
//! pipeline.
//!
//! The legacy template attempt always runs first and is unchanged. This module
//! is reached only after that attempt returns a typed strategy decline. Any
//! failure here is surfaced to the public boundary, which then runs the existing
//! exact TokenTrie fallback.

#[cfg(test)]
use std::cell::Cell;

use crate::compiler::build_budget::BuildBudgetScope;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::terminal_ended::{
    certify_terminal_ended, prepare_terminal_ended, CompileBudget, CompileDecline, CompileLimits,
};
use crate::compiler::grammar::transforms::prepare_terminal_ended_grammar;
use crate::compiler::pipeline::try_compile_terminal_ended;
use crate::grammar::flat::{GrammarDef, Terminal};
use crate::{Constraint, Error, Result, Vocab};

#[cfg(test)]
thread_local! {
    /// Test-only observation of whether the rescue route was entered on this
    /// thread. The decision is made on the caller thread, outside the compile
    /// pool, so this is not perturbed by concurrent test compiles.
    static RESCUE_ENTRIES: Cell<u32> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_test_rescue_entries() {
    RESCUE_ENTRIES.with(|entries| entries.set(0));
}

#[cfg(test)]
pub(crate) fn test_rescue_entries() -> u32 {
    RESCUE_ENTRIES.with(Cell::get)
}

fn map_decline(decline: CompileDecline) -> Error {
    match decline {
        CompileDecline::ResourceLimit {
            stage,
            kind,
            used,
            limit,
        } => Error::ResourceLimit(format!(
            "terminal_ended {stage:?} {kind:?} exceeded limit: used={used} limit={limit}"
        )),
        CompileDecline::StrategyUnsupported { stage, reason } => Error::StrategyUnsupported(
            format!("terminal_ended {stage:?}: {reason}"),
        ),
    }
}

/// Features the phase-2 rewrite does not preserve. Rejecting them here keeps
/// the route's exactness contract local rather than relying on a later stage.
pub(crate) fn check_rescue_scope(grammar: &GrammarDef) -> Result<()> {
    if grammar
        .terminals
        .iter()
        .any(|terminal| matches!(terminal, Terminal::MarkerProbe { .. }))
    {
        return Err(Error::StrategyUnsupported(
            "terminal-ended rescue does not preserve marker terminals".into(),
        ));
    }
    if grammar.direct_regular_automaton.is_some() {
        return Err(Error::StrategyUnsupported(
            "terminal-ended rescue does not preserve a frontend direct-regular automaton".into(),
        ));
    }
    Ok(())
}

/// Attempt the terminal-ended rescue for one original source grammar.
///
/// A fresh template-build budget scope is entered for the rescue; the legacy
/// attempt has already completed and dropped its own scope.
pub(crate) fn rescue(source: &GrammarDef, vocab: &Vocab) -> Result<Constraint> {
    #[cfg(test)]
    RESCUE_ENTRIES.with(|entries| entries.set(entries.get() + 1));

    check_rescue_scope(source)?;

    let source_start_nullable = source.start_is_nullable();
    let budget_scope = BuildBudgetScope::enter();
    budget_scope.install(|| crate::error::catch_template_attempt(|| {
        let limits = CompileLimits::default();
        let mut budget = CompileBudget::new();
        let prepared =
            prepare_terminal_ended(source, &limits, &mut budget).map_err(map_decline)?;
        // `prepare_terminal_ended` certifies internally; keep the explicit
        // obligation at this boundary so the route cannot drift.
        certify_terminal_ended(&prepared).map_err(map_decline)?;
        if prepared.rules.is_empty() {
            // The rewrite can prove an empty language (or an epsilon-only
            // language). Both are better handled by the exact TokenTrie
            // fallback than by a template with no ordinary productions.
            return Err(Error::StrategyUnsupported(
                "terminal-ended rescue has no ordinary productions".into(),
            ));
        }

        let grammar = prepare_terminal_ended_grammar(source.clone(), prepared);
        let analyzed = AnalyzedGrammar::from_grammar_def(&grammar);
        analyzed
            .check_dynamic_table_build_normal_form()
            .map_err(|message| {
                Error::StrategyUnsupported(format!(
                    "terminal-ended grammar precondition violations: {message}"
                ))
            })?;

        try_compile_terminal_ended(grammar, vocab, source_start_nullable)
    }))
}
