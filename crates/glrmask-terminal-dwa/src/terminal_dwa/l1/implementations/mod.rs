//! Swappable L1 builders and exact cross-checking machinery.
//!
//! Controls:
//! - `GLRMASK_L1_IMPLEMENTATION=projected|quotient|auto|scalar|trie|bulk|dense|frontier`
//! - `GLRMASK_L1_CHECK_AGAINST=none|single|production|scalar|trie|bulk|dense|frontier|other`
//! - `GLRMASK_L1_EXPERIMENT_PARTITIONS=p2,p5` scopes both controls.
//! - `GLRMASK_PROFILE_L1_IMPLEMENTATIONS=1` prints per-implementation timings.
//!
//! Defaults are projected, no checker, all partitions.

mod auto;
mod bulk;
mod common;
mod dense;
mod frontier;
mod projected;
pub(super) use projected::prepare_finite_vocab_projection;
mod quotient;
pub mod scalar;
mod support;
mod trie;
mod verify;

use std::sync::Arc;
use std::time::Instant;

use super::L1IdentityVocabOrder;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::grammar::flat::TerminalID;
use crate::terminal_dwa::l2p::equivalence_analysis::state_equivalence::nfa::{
    TokenBoundedAnalysisTopology, TokenBoundedAnalysisTrie,
};
use crate::terminal_dwa::types::{LocalIdMapTerminalDwa, TerminalColoring};
use crate::{Vocab, compiler::glr::analysis::AnalyzedGrammar};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Implementation {
    Projected,
    Quotient,
    Auto,
    Scalar,
    Trie,
    Bulk,
    Dense,
    Frontier,
}

impl Implementation {
    fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "projected" | "single" | "single-group" | "prefix-dfa" => Self::Projected,
            "quotient" | "production" | "prod" | "existing" | "established" => Self::Quotient,
            "auto" | "hybrid" => Self::Auto,
            "scalar" | "reference" | "ref" => Self::Scalar,
            "trie" | "optimized" | "opt" => Self::Trie,
            "bulk" | "dag" => Self::Bulk,
            "dense" | "chunked" => Self::Dense,
            "frontier" | "weighted" => Self::Frontier,
            other => panic!("unknown L1 implementation {other:?}; expected projected, quotient, auto, scalar, trie, bulk, dense, or frontier"),
        }
    }

    fn other(self) -> Self {
        match self {
            Self::Quotient => Self::Projected,
            Self::Projected | Self::Auto | Self::Scalar | Self::Trie | Self::Bulk | Self::Dense | Self::Frontier => Self::Quotient,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    pub use_implementation: Implementation,
    pub check_against: Option<Implementation>,
}

impl Default for Plan {
    fn default() -> Self {
        Self { use_implementation: Implementation::Projected, check_against: None }
    }
}

impl Plan {
    pub fn from_env(partition: &str) -> Self {
        if let Ok(scope) = std::env::var("GLRMASK_L1_EXPERIMENT_PARTITIONS") {
            let selected = scope.split(',').any(|candidate| candidate.trim() == partition);
            if !selected {
                return Self::default();
            }
        }
        let use_implementation = std::env::var("GLRMASK_L1_IMPLEMENTATION")
            .ok().map_or(Implementation::Projected, |value| Implementation::parse(&value));
        let check_against = std::env::var("GLRMASK_L1_CHECK_AGAINST").ok().and_then(|value| {
            match value.trim().to_ascii_lowercase().as_str() {
                "" | "0" | "false" | "none" => None,
                "1" | "true" | "other" => Some(use_implementation.other()),
                value => Some(Implementation::parse(value)),
            }
        });
        assert_ne!(check_against, Some(use_implementation), "L1 cannot check an implementation against itself");
        Self { use_implementation, check_against }
    }
}

#[derive(Clone, Copy)]
pub struct BuildInput<'a> {
    pub partition_label: &'a str,
    pub tokenizer: &'a Tokenizer,
    pub vocab: &'a Vocab,
    pub terminal_coloring: &'a TerminalColoring,
    pub use_terminal_coloring: bool,
    pub ignore_terminal: Option<TerminalID>,
    pub grammar: &'a AnalyzedGrammar,
    pub active_terminals: &'a [bool],
    pub flat_trans: &'a Arc<[u32]>,
    pub transitions_by_byte: Option<&'a [u32]>,
    pub initial_state_map: Option<&'a ManyToOneIdMap>,
    pub shared_generic_nfa_topology: Option<&'a TokenBoundedAnalysisTopology>,
    pub shared_generic_nfa_trie: Option<&'a TokenBoundedAnalysisTrie>,
    pub subset_parent_order: Option<&'a L1IdentityVocabOrder>,
}

fn run(implementation: Implementation, input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    match implementation {
        Implementation::Projected => projected::build(input),
        Implementation::Quotient => quotient::build(input),
        Implementation::Auto => auto::build(input),
        Implementation::Scalar => scalar::build(input),
        Implementation::Trie => trie::build(input),
        Implementation::Bulk => bulk::build(input),
        Implementation::Dense => dense::build(input),
        Implementation::Frontier => frontier::build(input),
    }
}

pub fn build_with_plan(input: BuildInput<'_>, plan: Plan) -> Option<LocalIdMapTerminalDwa> {
    let resolved = plan.use_implementation;
    let selected_started = Instant::now();
    let selected = run(resolved, input);
    let selected_ms = selected_started.elapsed().as_secs_f64() * 1000.0;

    let mut check_ms = 0.0;
    let resolved_checker = plan.check_against;
    if let Some(checker) = resolved_checker.filter(|&checker| checker != resolved) {
        let check_started = Instant::now();
        let expected = run(checker, input);
        verify::assert_equivalent(input, resolved, selected.as_ref(), checker, expected.as_ref());
        check_ms = check_started.elapsed().as_secs_f64() * 1000.0;
    }

    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_implementation] partition={} selected={:?} resolved={:?} selected_ms={:.3} checker={:?} resolved_checker={:?} check_ms={:.3}",
            input.partition_label,
            plan.use_implementation,
            resolved,
            selected_ms,
            plan.check_against,
            resolved_checker,
            check_ms,
        );
    }
    selected
}

pub fn build_from_env(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    build_with_plan(input, Plan::from_env(input.partition_label))
}
