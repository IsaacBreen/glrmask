//! Phase-2 template rescue: rewrite the original source grammar into
//! terminal-ended normal form, then compile it with the ordinary template
//! pipeline.
//!
//! The legacy template attempt always runs first and is unchanged. This module
//! is reached only when `GLRMASK_EXPERIMENT_TERMINAL_ENDED_RESCUE=1` opts in
//! and that attempt returns a typed strategy decline. Any failure here is
//! surfaced to the public boundary, which returns the original typed decline
//! rather than recompiling with TokenTrie. Without the switch, compilation
//! returns the normal route's decline unchanged.

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

#[cfg(test)]
thread_local! {
    /// Test-only overrides of the rescue experiment switches. Process
    /// environment variables are unsuitable here because lib tests run in
    /// parallel on shared threads.
    static RESCUE_ENABLED_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static TE_FIRST_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Test-only override; `None` restores the environment-variable behaviour.
#[cfg(test)]
pub(crate) fn set_test_terminal_ended_rescue(value: Option<bool>) {
    RESCUE_ENABLED_OVERRIDE.with(|slot| slot.set(value));
}

/// Test-only override; `None` restores the environment-variable behaviour.
#[cfg(test)]
pub(crate) fn set_test_terminal_ended_first(value: Option<bool>) {
    TE_FIRST_OVERRIDE.with(|slot| slot.set(value));
}

#[cfg(test)]
fn terminal_ended_rescue_override() -> Option<bool> {
    RESCUE_ENABLED_OVERRIDE.with(Cell::get)
}

#[cfg(not(test))]
fn terminal_ended_rescue_override() -> Option<bool> {
    None
}

#[cfg(test)]
fn terminal_ended_first_override() -> Option<bool> {
    TE_FIRST_OVERRIDE.with(Cell::get)
}

#[cfg(not(test))]
fn terminal_ended_first_override() -> Option<bool> {
    None
}

/// Experiment controls read once per top-level compile.
///
/// The terminal-ended rescue is opt-in: a normal template decline is returned
/// unchanged unless `GLRMASK_EXPERIMENT_TERMINAL_ENDED_RESCUE=1` (or the
/// test-only override) enables it.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RescueExperiment {
    pub(crate) terminal_ended_rescue: bool,
    pub(crate) terminal_ended_first: bool,
    pub(crate) report_route: bool,
}

impl RescueExperiment {
    pub(crate) fn read() -> Self {
        let terminal_ended_rescue = terminal_ended_rescue_override().unwrap_or_else(|| {
            std::env::var_os("GLRMASK_EXPERIMENT_TERMINAL_ENDED_RESCUE")
                .is_some_and(|value| value.to_str() == Some("1"))
        });
        let terminal_ended_first = terminal_ended_rescue
            && terminal_ended_first_override().unwrap_or_else(|| {
                std::env::var_os("GLRMASK_EXPERIMENT_TERMINAL_ENDED_FIRST")
                    .is_some_and(|value| value.to_str() == Some("1"))
            });
        let report_route = std::env::var_os("GLRMASK_REPORT_ROUTE").is_some();
        Self {
            terminal_ended_rescue,
            terminal_ended_first,
            report_route,
        }
    }
}

/// Which path produced the result reported by `GLRMASK_REPORT_ROUTE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    TerminalEnded,
    Normal,
    TokenTrieFallback,
}

impl Route {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::TerminalEnded => "terminal_ended",
            Self::Normal => "normal",
            Self::TokenTrieFallback => "token_trie_fallback",
        }
    }
}

/// Outcome of the TE-first attempt (or of the legacy post-decline rescue when
/// TE-first is off).
#[derive(Clone, Debug, Default)]
pub(crate) enum RescueOutcome {
    #[default]
    NotTried,
    Ok,
    Declined(String),
}

impl RescueOutcome {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::NotTried => "not_tried",
            Self::Ok => "ok",
            Self::Declined(_) => "declined",
        }
    }

    pub(crate) fn reason(&self) -> String {
        match self {
            Self::Declined(reason) => short_reason(reason),
            _ => "-".to_owned(),
        }
    }
}

fn short_reason(reason: &str) -> String {
    let reason = reason.replace(['\n', '\r', ' ', '\t'], "_");
    let reason = reason.trim_matches('_');
    if reason.is_empty() {
        "-".to_owned()
    } else {
        reason.chars().take(160).collect()
    }
}

/// Emit exactly one route line for a top-level compile when requested.
pub(crate) fn report_route(
    enabled: bool,
    route: Route,
    strategy: impl std::fmt::Debug,
    experiment: RescueExperiment,
    outcome: &RescueOutcome,
) {
    if !enabled {
        return;
    }
    eprintln!(
        "[glrmask/route] route={} strategy={:?} te_first={} te_outcome={} te_reason={}",
        route.as_str(),
        strategy,
        u8::from(experiment.terminal_ended_first),
        outcome.as_str(),
        outcome.reason(),
    );
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
            // language). Both are better handled by returning a typed decline
            // than by a template with no ordinary productions.
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
