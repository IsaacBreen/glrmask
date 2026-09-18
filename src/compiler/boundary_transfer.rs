//! Prepared local signed transfers for the static linker (advisor v3 design).
//!
//! The boundary compiler reuses ordinary component terminal transfers in
//! scoped provider coordinates, plus small Entry (slot CALL) and Finish
//! (EOF RETURN) transfer exports built with the existing local
//! characterization machinery. Controls belong in the expanded signed parser
//! program around template fragments (Ready ports), never as fake
//! terminal-DWA transitions, and negative-label cancellation runs only after
//! the complete boundary transfer is assembled.
//!
//! Provenance: specialist-advisor
//! `glrmask-prepared-static-linker-full-architecture-reconsideration-v3-2026-09-18`
//! §§4-5, 8-9, 13-15. This module implements milestones A (scoped ordinary
//! local signed transfers) and B (scoped Entry/Finish preserving the full
//! stack relation). Milestones C/D (Ready-port signed-NWA composition +
//! existing cancellation) build on these exports; see the loud declines in
//! `assemble_boundary_transfer_query` until the composer lands.
//!
//! Invariants enforced here (loud `Err`, never silent truncation):
//! - Entry preserves the FULL stack relation: raw characterization escapes
//!   gain the scoped child-start push; reduces/nt-rereduces are unchanged.
//!   No negative resolution runs on Entry alone (that would discard the saved
//!   continuation and child-start pushes the continuation needs).
//! - Entry is exact only for provider-supported slot shapes. The provider
//!   drops genuine shift/reduce slot splits, so such slots are a loud
//!   `unsupported` error, not a best-effort CALL.
//! - Finish uses `characterize_finish_transfer` (Accept -> Return endpoint
//!   policy with nullable extras wherever the provider emits them).
//! - All incoming links to one shared child must agree on
//!   child-start/return-pop/nullability (the provider dispatches Finish by
//!   first-incoming-link, which must not define semantics accidentally).

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use glrmask_parser_dwa::__private::resolve_negatives::resolve_negative_codes_in_nwa;

use crate::automata::weighted_u32::dwa::DWA;
use crate::automata::weighted_u32::nwa::{NWA, NwaBody};
use crate::compiler::constraint_compose::{
    CompiledSubgrammarInput, PublishedStaticBoundaryShard, WalkBoundaryShardWork,
    WalkShardPublishProfile, build_segmented_parser_links, publish_signed_boundary_shard_work,
};
use crate::compiler::glr::analysis::EOF;
use crate::compiler::glr::labels::is_negative_label;
use crate::compiler::glr::parser::ScopedSubgrammarLink;
use crate::compiler::glr::table::{Action, GLRTable};
use crate::compiler::stages::parser_dwa::normalize_weighted_parser_stack_nwa_for_parser_state_count;
use crate::compiler::stages::templates::Templates;
use crate::compiler::stages::templates::characterize::{
    FinishEndpointPolicy, FinishTransfer, InitialEscape, InitialReduce, NtEscape, NtRereduce,
    StackMatcher, TerminalCharacterization, characterize_finish_transfer,
    characterize_selected_terminals_for_terminal_count,
};
use crate::ds::weight::Weight;
use crate::grammar::flat::TerminalID;
use crate::runtime::Constraint;

/// Disjoint-union injection of one component's local parser states into the
/// scoped provider coordinate. Must agree with the runtime provider offsets
/// (`DisjointComponentActionProvider`); the link validates the full layout.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StateInjection {
    pub offset: u32,
}

impl StateInjection {
    pub(crate) fn scope_state(&self, local: u32) -> Result<u32, String> {
        self.offset.checked_add(local).ok_or_else(|| {
            format!("scoped parser-state coordinate overflow: offset {} + local {local}", self.offset)
        })
    }
}

fn scope_matcher(matcher: &StackMatcher, injection: &StateInjection) -> Result<StackMatcher, String> {
    match matcher {
        StackMatcher::Any => Ok(StackMatcher::Any),
        StackMatcher::State(local) => Ok(StackMatcher::State(injection.scope_state(*local)?)),
        StackMatcher::States(locals) => {
            let mut scoped = Vec::with_capacity(locals.len());
            for &local in locals {
                scoped.push(injection.scope_state(local)?);
            }
            Ok(StackMatcher::States(scoped))
        }
    }
}

fn scope_pop(
    pop: &[StackMatcher],
    injection: &StateInjection,
) -> Result<Vec<StackMatcher>, String> {
    pop.iter().map(|matcher| scope_matcher(matcher, injection)).collect()
}

fn scope_pushes(pushes: &[u32], injection: &StateInjection) -> Result<Vec<u32>, String> {
    pushes.iter().map(|&local| injection.scope_state(local)).collect()
}

/// Scope a local characterization into the provider coordinate, preserving
/// the full stack relation (matchers and pushes, reductions included).
/// Wildcards (`Any`) are preserved as wildcards: the literal provider `popn`
/// does not inspect the component of every popped state, and scoping must not
/// narrow that behavior.
pub(crate) fn scope_characterization(
    local: &TerminalCharacterization,
    injection: &StateInjection,
) -> Result<TerminalCharacterization, String> {
    let mut escapes = Vec::with_capacity(local.escapes.len());
    for escape in &local.escapes {
        escapes.push(InitialEscape {
            pop: scope_pop(&escape.pop, injection)?,
            pushes: scope_pushes(&escape.pushes, injection)?,
        });
    }
    let mut reduces = Vec::with_capacity(local.reduces.len());
    for reduce in &local.reduces {
        reduces.push(InitialReduce {
            pop: scope_pop(&reduce.pop, injection)?,
            nonterminal: reduce.nonterminal,
        });
    }
    let mut nt_escapes = Vec::with_capacity(local.nt_escapes.len());
    for escape in &local.nt_escapes {
        nt_escapes.push(NtEscape {
            source_nonterminal: escape.source_nonterminal,
            pop: scope_pop(&escape.pop, injection)?,
            pushes: scope_pushes(&escape.pushes, injection)?,
        });
    }
    let mut nt_rereduces = Vec::with_capacity(local.nt_rereduces.len());
    for rereduce in &local.nt_rereduces {
        nt_rereduces.push(NtRereduce {
            source_nonterminal: rereduce.source_nonterminal,
            pop: scope_pop(&rereduce.pop, injection)?,
            target_nonterminal: rereduce.target_nonterminal,
        });
    }
    Ok(TerminalCharacterization {
        escapes,
        reduces,
        nt_escapes,
        nt_rereduces,
        all_nts: local.all_nts.clone(),
    })
}

/// Per-transfer instrumentation recorded for every local summary (advisor
/// Prototype 1 deliverable).
#[derive(Debug, Clone, Default)]
pub(crate) struct TransferDiagnostics {
    /// `None` when the nt re-reduction graph is acyclic; `Some(len)` with the
    /// cycle length otherwise (boundedness not certified).
    pub cycle_status: Option<usize>,
    /// Maximum input-read length (pop matcher count) over all escapes/reduces.
    pub max_input_read: usize,
    /// Maximum push length over all escapes.
    pub max_push_len: usize,
    pub num_escapes: usize,
    pub num_reduces: usize,
    pub num_nt_escapes: usize,
    pub num_nt_rereduces: usize,
    /// Sorted endpoint-shape labels, e.g. `escape`, `reduce`, `nt_escape`,
    /// `nt_rereduce`, `return_pop1`, `return_pop2`.
    pub endpoint_kinds: Vec<String>,
}

fn is_return_escape(escape: &InitialEscape) -> Option<u32> {
    // Return pops are pop-only escapes: every pop beyond the first matcher is
    // `Any` (the executor's `R_Any^{r-1}` tail) and there are no pushes... but
    // a Return with a longer live segment keeps the remainder as pushes, so
    // only the pushes-empty + Any-tail shape is classified here; the rest is
    // reported as a plain escape. This is instrumentation only.
    if !escape.pushes.is_empty() || escape.pop.is_empty() {
        return None;
    }
    let extra = &escape.pop[1..];
    if extra.iter().all(|matcher| *matcher == StackMatcher::Any) {
        Some(escape.pop.len() as u32)
    } else {
        None
    }
}

pub(crate) fn diagnose_transfer(characterization: &TerminalCharacterization) -> TransferDiagnostics {
    let mut diagnostics = TransferDiagnostics {
        cycle_status: characterization.find_cycle().map(|cycle| cycle.len()),
        ..TransferDiagnostics::default()
    };
    let mut kinds = BTreeSet::new();
    for escape in &characterization.escapes {
        diagnostics.max_input_read = diagnostics.max_input_read.max(escape.pop.len());
        diagnostics.max_push_len = diagnostics.max_push_len.max(escape.pushes.len());
        match is_return_escape(escape) {
            Some(pop) => kinds.insert(format!("return_pop{pop}")),
            None => kinds.insert("escape".to_string()),
        };
    }
    for reduce in &characterization.reduces {
        diagnostics.max_input_read = diagnostics.max_input_read.max(reduce.pop.len());
        kinds.insert("reduce".to_string());
    }
    for escape in &characterization.nt_escapes {
        diagnostics.max_input_read = diagnostics.max_input_read.max(escape.pop.len());
        diagnostics.max_push_len = diagnostics.max_push_len.max(escape.pushes.len());
        kinds.insert("nt_escape".to_string());
    }
    for rereduce in &characterization.nt_rereduces {
        diagnostics.max_input_read = diagnostics.max_input_read.max(rereduce.pop.len());
        kinds.insert("nt_rereduce".to_string());
    }
    diagnostics.num_escapes = characterization.escapes.len();
    diagnostics.num_reduces = characterization.reduces.len();
    diagnostics.num_nt_escapes = characterization.nt_escapes.len();
    diagnostics.num_nt_rereduces = characterization.nt_rereduces.len();
    diagnostics.endpoint_kinds = kinds.into_iter().collect();
    diagnostics
}

/// Scoped transfer export: immutable reusable data plus its link-time
/// instantiation context. Exit destinations are contextual (per continuation
/// port), so instantiated fragments must never be shared across unrelated
/// lexical destinations without a key containing the continuation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransferKind {
    OrdinaryShift,
    Call,
    Return,
    LocalControlStep,
}

#[derive(Debug, Clone)]
pub(crate) struct ScopedTransfer {
    pub characterization: TerminalCharacterization,
    pub component: u32,
    pub kind: TransferKind,
    pub diagnostics: TransferDiagnostics,
}

fn action_shape_label(action: &Action) -> &'static str {
    match action {
        Action::Shift(..) => "shift",
        Action::StackShifts(..) => "stack-shifts",
        Action::GuardedStackShifts(..) => "guarded-stack-shifts",
        Action::Reduce(..) => "reduce",
        Action::Split { .. } => "split",
        Action::Accept => "accept",
        Action::ReplaceShifts(..) => "replace-shifts",
        Action::Skip => "skip",
    }
}

/// Validate that a parent slot terminal supports exact static Entry.
///
/// The provider replaces the ordinary slot shift by CALL only for pure shifts
/// (no reductions) and passes pure reductions through as local Entry
/// reductions. Every other slot shape — shift/reduce splits (dropped by the
/// provider), accepting splits, optimized shift forms, Skip/Accept — is a
/// loud unsupported error: silently reusing the ordinary characterization
/// would preserve alternatives DynamicDirect drops (or vice versa).
pub(crate) fn validate_slot_entry_shape(
    parent_table: &GLRTable,
    slot_terminal: TerminalID,
) -> Result<(), String> {
    if parent_table
        .forwarded_shifts
        .iter()
        .any(|&(_, terminal)| terminal == slot_terminal)
    {
        return Err(format!(
            "static Entry unsupported: forwarded shift involving slot terminal {slot_terminal} needs provider-conformance audit",
        ));
    }
    for state in 0..parent_table.num_states {
        let Some(action) = parent_table.action(state, slot_terminal) else {
            continue;
        };
        match action {
            Action::Shift(..) => {}
            Action::Split {
                shift: Some(_),
                reduces,
                accept: false,
            } if reduces.is_empty() => {}
            Action::Reduce(..) => {}
            Action::Split {
                shift: None,
                reduces,
                accept: false,
            } if !reduces.is_empty() => {}
            Action::Split {
                shift: Some(_),
                reduces,
                accept,
            } if !reduces.is_empty() || *accept => {
                return Err(format!(
                    "static Entry unsupported: parent state {state} slot {slot_terminal} has a shift/reduce/accept split (shift={} reduces={} accept={accept}) that the provider drops; refusing to guess call-site semantics",
                    action.shift_target().is_some(),
                    reduces.len(),
                ));
            }
            other => {
                return Err(format!(
                    "static Entry unsupported: parent state {state} slot {slot_terminal} has action {} which the provider Entry arm does not implement",
                    action_shape_label(other),
                ));
            }
        }
    }
    Ok(())
}

/// Instantiate Entry by attaching the scoped child start to every successful
/// escape push list of the (already scoped) local slot characterization.
///
/// Non-replacing CALL is `R(I) P(I) P(Q) P(S)`; replacing CALL is
/// `R(I) P(Q) P(S)`. Reductions and nt re-reductions are unchanged: Entry
/// reductions retain the selected Entry symbol atomically (the executor never
/// interleaves Entry mid-terminal-reduction-chain). No negative resolution
/// runs here.
pub(crate) fn instantiate_entry(
    scoped_slot: TerminalCharacterization,
    scoped_child_start: u32,
    parent_component: u32,
) -> ScopedTransfer {
    let mut characterization = scoped_slot;
    for escape in &mut characterization.escapes {
        escape.pushes.push(scoped_child_start);
    }
    for escape in &mut characterization.nt_escapes {
        escape.pushes.push(scoped_child_start);
    }
    let diagnostics = diagnose_transfer(&characterization);
    ScopedTransfer {
        characterization,
        component: parent_component,
        kind: TransferKind::Call,
        diagnostics,
    }
}

/// All incoming links to one shared child must agree on child-start,
/// return-pop, and nullability: Finish dispatches by first incoming link.
pub(crate) fn validate_shared_child_links(links: &[ScopedSubgrammarLink]) -> Result<(), String> {
    for (index, link) in links.iter().enumerate() {
        for other in &links[..index] {
            if other.child_component != link.child_component {
                continue;
            }
            if other.child_start != link.child_start
                || other.return_pop != link.return_pop
                || other.child_start_nullable != link.child_start_nullable
            {
                return Err(format!(
                    "static link unsupported: child component {} has disagreeing incoming links \
                     (child_start {} vs {}, return_pop {} vs {}, nullable {} vs {}); \
                     allocate separate component instances sharing the prepared artifact",
                    link.child_component,
                    other.child_start,
                    link.child_start,
                    other.return_pop,
                    link.return_pop,
                    other.child_start_nullable,
                    link.child_start_nullable,
                ));
            }
        }
    }
    Ok(())
}

/// Canonical child-start contract: the provider validates only that
/// `child_start` is in range, while return-pop derivation assumes standalone
/// start state 0 and its canonical root goto. Prepared links enforce the
/// canonical contract here; noncanonical entries need a separate completion
/// certificate.
pub(crate) fn validate_canonical_child_start(
    child_table: &GLRTable,
    link: &ScopedSubgrammarLink,
) -> Result<(), String> {
    if link.child_start != 0 {
        return Err(format!(
            "static link unsupported: child component {} has noncanonical child_start {} (canonical contract is 0); provide a separate completion certificate",
            link.child_component, link.child_start,
        ));
    }
    if link.return_pop != 1 && link.return_pop != 2 {
        return Err(format!(
            "static link unsupported: child component {} has non-canonical return_pop {} (expected 1|2)",
            link.child_component, link.return_pop,
        ));
    }
    if child_table.control_terminals.len() != 0 {
        return Err(format!(
            "static link unsupported: child component {} carries retained linker controls; classify them as controls before static reuse",
            link.child_component,
        ));
    }
    Ok(())
}

/// Build the scoped Finish transfer for one child link: local EOF
/// characterization with the Accept -> Return endpoint policy, in the scoped
/// coordinate. The caller declines the bounded flat composition when
/// `has_local_eof_effects` is set (ordinary EOF stack effects remaining
/// inside the child need outer control-choice points).
pub(crate) fn instantiate_finish(
    child_table: &GLRTable,
    link: &ScopedSubgrammarLink,
    injection: &StateInjection,
) -> Result<(ScopedTransfer, bool), String> {
    validate_canonical_child_start(child_table, link)?;
    let policy = FinishEndpointPolicy {
        return_pop: link.return_pop,
        nullable_child_start: link.child_start_nullable.then(|| link.child_start),
    };
    let FinishTransfer {
        characterization,
        has_local_eof_effects,
    } = characterize_finish_transfer(child_table, &policy)?;
    let characterization = scope_characterization(&characterization, injection)?;
    let diagnostics = diagnose_transfer(&characterization);
    Ok((
        ScopedTransfer {
            characterization,
            component: link.child_component,
            kind: TransferKind::Return,
            diagnostics,
        },
        has_local_eof_effects,
    ))
}

/// Boundary-transfer query assembly (milestone C scaffold).
///
/// The control-aware signed parser NWA composes scoped ordinary transfers at
/// lexical terminal edges with Entry/Finish/retained-control fragments at
/// Ready ports, then runs the existing negative-label cancellation once over
/// the complete query. That composer is not implemented yet; this scaffold
/// exists so call sites fail loudly (link-time decline) instead of routing
/// through the refuted splice-template or global-elimination paths.
pub(crate) fn assemble_boundary_transfer_query() -> Result<(), String> {
    Err("prepared static link: control-aware signed-NWA composer not implemented yet \
         (scoped Entry/Finish exports are ready; splice templates and global \
         exact_control_elimination must not be used as substitutes)"
        .to_string())
}

// ---------------------------------------------------------------------------
// Milestones C/D: control-aware signed parser-NWA compiler.
// ---------------------------------------------------------------------------
//
// For the supported flat prototype (flat component-instance DAG, effectively
// nonnullable bound children, no retained local controls, canonical EOF
// completion), the boundary parser is a signed action-word program:
//
// - one Ready(k) port per crossing-terminal-automaton vertex k;
// - each real terminal-DWA edge k -t-> k' substitutes t's scoped signed local
//   transfer fragment from Ready(k) to Ready(k'), stamping the lexical weight;
// - zero-width Entry/Finish transfer fragments loop at every Ready port
//   (the C* over-approximation; infeasible interleavings denote the empty
//   relation and cancel out exactly);
// - existing resolve_negative_codes_in_nwa runs ONCE over the assembled query;
//   output pushes are preserved until then (never resolve Entry alone).
//
// Controls are never fake terminal-DWA labels and never connect inside another
// selected symbol's reduction continuation: fragments are cloned per use and
// linked only at Ready ports and fragment exits. Downstream, the exact same
// resolver + table-free normalizer as the established path is reused, so no
// new DEFAULT/wildcard semantics is introduced. The one table-construction tag
// the old path consulted (`ExperimentalCoreMerged` driving grouped
// cancellation) is replaced by an explicit `false`: grouped cancellation is a
// performance variant (additionally env-gated), never a semantic requirement.

/// Link-time context for the signed-transfer boundary compiler: intact local
/// tables, provider-layout injections, and validated control contracts. No
/// composed or provider-level table exists anywhere in this object.
pub(crate) struct SignedLinkContext<'a> {
    pub parent_table: &'a GLRTable,
    pub child_tables: Vec<&'a GLRTable>,
    pub links: Vec<ScopedSubgrammarLink>,
    pub terminal_offsets: Vec<u32>,
    pub num_terminals: u32,
    pub state_offsets: Vec<u32>,
    pub total_scoped_states: u32,
    pub global_ignores: bool,
    pub ignore_terminals: Vec<Option<TerminalID>>,
    /// Composed-id unbound slot terminals with empty-language semantics: the
    /// live placeholder shifts in the component tables are NOT characterized.
    pub unbound_slots: BTreeSet<TerminalID>,
}

impl<'a> SignedLinkContext<'a> {
    fn component_table(&self, component: u32) -> Result<&'a GLRTable, String> {
        if component == 0 {
            return Ok(self.parent_table);
        }
        self.child_tables.get(component as usize - 1).copied().ok_or_else(|| {
            format!("signed link references unknown component {component}")
        })
    }

    pub(crate) fn injection(&self, component: u32) -> Result<StateInjection, String> {
        let offset = self.state_offsets.get(component as usize).copied().ok_or_else(|| {
            format!("signed link has no state offset for component {component}")
        })?;
        Ok(StateInjection { offset })
    }

    /// Composed terminal id -> (owning component, component-local terminal id).
    pub(crate) fn terminal_owner(&self, terminal: TerminalID) -> Result<(u32, TerminalID), String> {
        let owner = self
            .terminal_offsets
            .iter()
            .rposition(|&offset| offset <= terminal)
            .ok_or_else(|| format!("signed link terminal {terminal} has no owning component"))?;
        let local = terminal.checked_sub(self.terminal_offsets[owner]).ok_or_else(|| {
            format!("signed link terminal {terminal} underflows its component offset")
        })?;
        Ok((owner as u32, local))
    }
}

/// Build and validate the signed-link context for one flat composition.
///
/// Loud declines (never silent): nested or control-bearing components,
/// disagreeing incoming links to a shared child, noncanonical
/// child-start/return-pop, provider-unsupported slot shapes, forwarded shifts
/// involving slots. The packed splice (rules + terminal layout) is still built
/// by the caller for grammar/follows analysis and layout pins, but no parser
/// behavior is ever derived from it.
pub(crate) fn build_signed_link_context<'a>(
    parent: &'a Constraint,
    children: &'a [CompiledSubgrammarInput<'a>],
    terminal_offsets: &[u32],
    num_terminals: u32,
    global_ignores: bool,
    unbound_slots: BTreeSet<TerminalID>,
) -> Result<SignedLinkContext<'a>, String> {
    let mut tables = Vec::with_capacity(children.len() + 1);
    tables.push(&parent.table);
    for child in children {
        tables.push(&child.constraint.table);
    }
    for (index, table) in tables.iter().enumerate() {
        if !table.control_terminals.is_empty() {
            return Err(format!(
                "signed static link unsupported: component {index} carries retained linker controls; classify them as controls before static reuse",
            ));
        }
    }
    let links = build_segmented_parser_links(children)?;
    validate_shared_child_links(&links)?;
    let mut state_offsets = Vec::with_capacity(tables.len());
    let mut total = 0u32;
    for table in &tables {
        state_offsets.push(total);
        total = total.checked_add(table.num_states).ok_or_else(|| {
            "signed link scoped parser-state coordinate overflow".to_string()
        })?;
    }
    let mut ignore_terminals = Vec::with_capacity(tables.len());
    ignore_terminals.push(parent.ignore_terminal);
    for child in children {
        ignore_terminals.push(child.constraint.ignore_terminal);
    }
    let child_tables = tables[1..].to_vec();
    let context = SignedLinkContext {
        parent_table: tables[0],
        child_tables,
        links,
        terminal_offsets: terminal_offsets.to_vec(),
        num_terminals,
        state_offsets,
        total_scoped_states: total,
        global_ignores,
        ignore_terminals,
        unbound_slots,
    };
    // Fail fast on provider-unsupported slot shapes and noncanonical children.
    for link in &context.links {
        validate_slot_entry_shape(
            context.component_table(link.parent_component)?,
            link.slot_terminal,
        )?;
        let child_table = context.component_table(link.child_component)?;
        validate_canonical_child_start(child_table, link)?;
    }
    Ok(context)
}

/// The provider's standalone-ignore semantics as a reusable transfer:
/// Identity on every local state (R_I P_I), scoped by the caller.
fn identity_transfer(num_states: u32) -> TerminalCharacterization {
    TerminalCharacterization {
        escapes: (0..num_states)
            .map(|state| InitialEscape {
                pop: vec![StackMatcher::State(state)],
                pushes: vec![state],
            })
            .collect(),
        reduces: Vec::new(),
        nt_escapes: Vec::new(),
        nt_rereduces: Vec::new(),
        all_nts: BTreeSet::new(),
    }
}

/// The exact empty relation (for unbound slot terminals): no escapes, no
/// reductions, no continuations. Equivalent to emptying the terminal in a
/// private table copy, without building any composed table.
fn empty_transfer() -> TerminalCharacterization {
    TerminalCharacterization {
        escapes: Vec::new(),
        reduces: Vec::new(),
        nt_escapes: Vec::new(),
        nt_rereduces: Vec::new(),
        all_nts: BTreeSet::new(),
    }
}

/// Fragment library for one shard: scoped ordinary transfers for the shard's
/// emitted terminals plus per-link Entry/Finish exports, compiled once
/// through the standard template compiler. Entry/Finish fragments are keyed by
/// synthetic terminal ids above the composed domain (never emitted by the
/// lexical DWA, never confused with real terminals).
pub(crate) struct FragmentLibrary {
    pub templates: Templates,
    /// Synthetic fragment key per link index for Entry / Finish exports.
    pub entry_keys: Vec<TerminalID>,
    pub finish_keys: Vec<TerminalID>,
    pub ordinary_terms: usize,
    pub templates_ms: f64,
}

fn entry_fragment_key(num_terminals: u32, link_index: usize) -> Result<TerminalID, String> {
    (num_terminals as usize)
        .checked_add(2 * link_index)
        .and_then(|key| u32::try_from(key).ok())
        .ok_or_else(|| "signed link entry fragment key overflow".to_string())
}

fn finish_fragment_key(num_terminals: u32, link_index: usize) -> Result<TerminalID, String> {
    (num_terminals as usize)
        .checked_add(2 * link_index + 1)
        .and_then(|key| u32::try_from(key).ok())
        .ok_or_else(|| "signed link finish fragment key overflow".to_string())
}

/// Build the scoped fragment library for one shard's emitted terminal set.
///
/// Every emitted composed terminal resolves to exactly one scoped transfer:
/// the owner's local characterization (empty relation when the terminal has
/// no parser action anywhere — verified by scan, never assumed), the Identity
/// transfer for a component-standalone ignore terminal (the provider applies
/// Identity before consulting table rows), or the empty relation for unbound
/// slots. A resolvable-but-missing characterization is a loud error.
pub(crate) fn build_fragment_library(
    context: &SignedLinkContext,
    emitted: &[bool],
    start_component: u32,
) -> Result<FragmentLibrary, String> {
    let templates_started = Instant::now();
    let mut combined: BTreeMap<TerminalID, TerminalCharacterization> = BTreeMap::new();
    let mut ordinary_terms = 0usize;
    for (terminal, demanded) in emitted.iter().enumerate() {
        if !demanded {
            continue;
        }
        let terminal = terminal as TerminalID;
        if (terminal as usize) >= context.num_terminals as usize {
            return Err(format!(
                "signed link shard {start_component} emits terminal {terminal} outside the composed domain {}",
                context.num_terminals,
            ));
        }
        if context.unbound_slots.contains(&terminal) {
            combined.insert(terminal, empty_transfer());
            ordinary_terms += 1;
            continue;
        }
        let (owner, local) = context.terminal_owner(terminal)?;
        let table = context.component_table(owner)?;
        if !context.global_ignores
            && context.ignore_terminals.get(owner as usize).copied().flatten() == Some(local)
        {
            let injection = context.injection(owner)?;
            combined.insert(
                terminal,
                scope_characterization(&identity_transfer(table.num_states), &injection)?,
            );
            ordinary_terms += 1;
            continue;
        }
        if local >= table.num_terminals {
            return Err(format!(
                "signed link terminal {terminal} resolves to local {local} outside component {owner} domain {}",
                table.num_terminals,
            ));
        }
        let mut local_selected = vec![false; table.num_terminals as usize];
        local_selected[local as usize] = true;
        let characterized = characterize_selected_terminals_for_terminal_count(
            table,
            table.num_terminals,
            &local_selected,
        );
        let local_characterization = characterized.get(&local);
        let scoped = match local_characterization {
            Some(characterization) => {
                let injection = context.injection(owner)?;
                scope_characterization(characterization, &injection)?
            }
            None => {
                // No characterization entry: exact only when the terminal has
                // no parser action anywhere in the local table.
                let has_action = (0..table.num_states)
                    .any(|state| table.action(state, local).is_some());
                if has_action {
                    return Err(format!(
                        "signed link terminal {terminal} (component {owner} local {local}) has parser actions but no characterization",
                    ));
                }
                empty_transfer()
            }
        };
        combined.insert(terminal, scoped);
        ordinary_terms += 1;
    }
    let mut entry_keys = Vec::with_capacity(context.links.len());
    let mut finish_keys = Vec::with_capacity(context.links.len());
    for (link_index, link) in context.links.iter().enumerate() {
        // Slot Entry: local slot characterization, scoped, plus child start.
        let parent_table = context.component_table(link.parent_component)?;
        let mut slot_selected = vec![false; parent_table.num_terminals as usize];
        slot_selected[link.slot_terminal as usize] = true;
        let slot_characterized = characterize_selected_terminals_for_terminal_count(
            parent_table,
            parent_table.num_terminals,
            &slot_selected,
        );
        let local_slot = slot_characterized.get(&link.slot_terminal).ok_or_else(|| {
            format!(
                "signed link slot terminal {} has no characterization in parent component {}",
                link.slot_terminal, link.parent_component,
            )
        })?;
        let parent_injection = context.injection(link.parent_component)?;
        let scoped_slot = scope_characterization(local_slot, &parent_injection)?;
        let child_injection = context.injection(link.child_component)?;
        let scoped_child_start = child_injection.scope_state(link.child_start)?;
        let entry = instantiate_entry(scoped_slot, scoped_child_start, link.parent_component);
        let entry_key = entry_fragment_key(context.num_terminals, link_index)?;
        combined.insert(entry_key, entry.characterization);
        entry_keys.push(entry_key);
        // Child Finish under this link's endpoint policy.
        let child_table = context.component_table(link.child_component)?;
        let (finish, has_local_eof_effects) =
            instantiate_finish(child_table, link, &child_injection)?;
        if has_local_eof_effects {
            return Err(format!(
                "signed static link unsupported: child component {} performs ordinary local EOF stack work; composing through it needs outer control-choice points",
                link.child_component,
            ));
        }
        let finish_key = finish_fragment_key(context.num_terminals, link_index)?;
        combined.insert(finish_key, finish.characterization);
        finish_keys.push(finish_key);
    }
    let templates = Templates::from_characterizations(&combined);
    let templates_ms = templates_started.elapsed().as_secs_f64() * 1000.0;
    Ok(FragmentLibrary {
        templates,
        entry_keys,
        finish_keys,
        ordinary_terms,
        templates_ms,
    })
}

/// Clone a template fragment into the arena, stamp every edge/epsilon with
/// `weight`, and redirect accepting finals to `continuation`. Mirrors the
/// established bundle-substitution primitive exactly (same weight stamping,
/// same final redirection); the only difference is the caller chooses
/// Ready-port endpoints instead of a table-driven bundle walk.
fn append_weighted_fragment(
    arena: &mut NWA,
    template: &NWA,
    weight: &Weight,
    continuation: u32,
) -> Result<NwaBody, String> {
    let offset = u32::try_from(arena.states().len())
        .map_err(|_| "signed parser NWA arena overflow".to_string())?;
    let body = arena.append_with_body(template);
    let appended_len = template.states().len();
    for state_id in offset as usize..offset as usize + appended_len {
        let state = arena
            .states_mut()
            .get_mut(state_id)
            .ok_or_else(|| "signed parser NWA fragment range overflow".to_string())?;
        for targets in state.transitions.values_mut() {
            for (_, edge_weight) in targets {
                *edge_weight = weight.clone();
            }
        }
        for (_, epsilon_weight) in &mut state.epsilons {
            *epsilon_weight = weight.clone();
        }
    }
    for state_id in offset as usize..offset as usize + appended_len {
        let takes_final = arena
            .states_mut()
            .get_mut(state_id)
            .ok_or_else(|| "signed parser NWA fragment range overflow".to_string())?
            .final_weight
            .take()
            .is_some();
        if takes_final {
            arena.add_epsilon(state_id as u32, continuation, weight.clone());
        }
    }
    Ok(body)
}

/// Output of one signed-shard compilation.
pub(crate) struct SignedShardOutput {
    pub parser_dwa: DWA,
    pub templates_ms: f64,
    pub compose_ms: f64,
    pub resolve_ms: f64,
    pub normalize_ms: f64,
    pub signed_states: usize,
    pub signed_transitions: usize,
    pub terms: usize,
}

/// Compile one shard: Ready-port assembly, single exact negative resolution,
/// table-free positive normalization.
///
/// Every real terminal-DWA edge substitutes its scoped transfer fragment;
/// Entry/Finish fragments loop zero-width at every Ready port (the bounded
/// flat-class control program). Negative labels are resolved exactly once over
/// the whole query; a surviving negative label afterwards is a loud error.
/// Normalization takes only the scoped parser-state count — no table, hence
/// no table-dependent optimization can consult a mismatched object.
pub(crate) fn compile_signed_shard_parser(
    context: &SignedLinkContext,
    library: &FragmentLibrary,
    shard_dwa: &DWA,
    start_component: u32,
) -> Result<SignedShardOutput, String> {
    let compose_started = Instant::now();
    let mut arena = NWA::new(0, 0);
    let mut ready = vec![u32::MAX; shard_dwa.states().len()];
    for (index, state) in shard_dwa.states().iter().enumerate() {
        let port = arena.add_state();
        ready[index] = port;
        if let Some(weight) = state.final_weight.as_ref() {
            if !weight.is_empty() {
                arena.set_final_weight(port, weight.clone());
            }
        }
    }
    let start_index = shard_dwa.start_state() as usize;
    let start_port = ready.get(start_index).copied().ok_or_else(|| {
        format!("signed link shard {start_component} has no lexical start state")
    })?;
    if start_port == u32::MAX {
        return Err(format!(
            "signed link shard {start_component} never allocated its start port"
        ));
    }
    arena.set_start_states(vec![start_port]);
    // Ordinary terminal edges: substitute the scoped transfer fragment.
    for (index, state) in shard_dwa.states().iter().enumerate() {
        for (label, target, weight) in state.transitions.entries() {
            if label < 0 {
                return Err(format!(
                    "signed link shard {start_component} carries negative terminal-DWA label {label}; controls must not be lexical labels",
                ));
            }
            if weight.is_empty() {
                continue;
            }
            let terminal = label as TerminalID;
            let fragment = library.templates.by_terminal_nwa.get(&terminal).ok_or_else(|| {
                format!(
                    "signed link shard {start_component} emits terminal {terminal} with no scoped transfer"
                )
            })?;
            let target_port = ready.get(target as usize).copied().ok_or_else(|| {
                format!("signed link shard {start_component} edge targets unknown state {target}")
            })?;
            let body = append_weighted_fragment(&mut arena, fragment, weight, target_port)?;
            for start in body.start_states {
                arena.add_epsilon(ready[index], start, Weight::all());
            }
        }
    }
    // Zero-width control loops at every Ready port. Entry/Finish fragments
    // carry the full scoped stack relation, so infeasible interleavings
    // (e.g. Entry on a child-topped stack) denote the empty relation and
    // cancel out exactly; feasible ones realize K before each terminal.
    let all_weight = Weight::all();
    for (index, _) in shard_dwa.states().iter().enumerate() {
        for (link_index, _) in context.links.iter().enumerate() {
            let entry_fragment = library
                .templates
                .by_terminal_nwa
                .get(&library.entry_keys[link_index])
                .ok_or_else(|| {
                    format!("signed link entry fragment {link_index} missing from library")
                })?;
            let entry_body =
                append_weighted_fragment(&mut arena, entry_fragment, &all_weight, ready[index])?;
            for start in entry_body.start_states {
                arena.add_epsilon(ready[index], start, Weight::all());
            }
            let finish_fragment = library
                .templates
                .by_terminal_nwa
                .get(&library.finish_keys[link_index])
                .ok_or_else(|| {
                    format!("signed link finish fragment {link_index} missing from library")
                })?;
            let finish_body =
                append_weighted_fragment(&mut arena, finish_fragment, &all_weight, ready[index])?;
            for start in finish_body.start_states {
                arena.add_epsilon(ready[index], start, Weight::all());
            }
        }
    }
    let signed_states = arena.states().len();
    let signed_transitions = arena.num_transitions();
    let compose_ms = compose_started.elapsed().as_secs_f64() * 1000.0;
    // Single exact negative resolution over the whole assembled query.
    // `false` is deliberate: grouped cancellation is a performance variant
    // (additionally env-gated), and no table-construction tag exists here to
    // consult — correctness-first exact cancellation always.
    let resolve_started = Instant::now();
    let resolved_reverse_topo = resolve_negative_codes_in_nwa(&mut arena, false);
    let resolve_ms = resolve_started.elapsed().as_secs_f64() * 1000.0;
    for state in arena.states() {
        for (&label, _) in state.transitions.iter() {
            if is_negative_label(label) {
                return Err(format!(
                    "signed link shard {start_component} retains negative stack label {label} after exact resolution",
                ));
            }
        }
    }
    let normalize_started = Instant::now();
    let parser_dwa = normalize_weighted_parser_stack_nwa_for_parser_state_count(
        context.total_scoped_states,
        &arena,
    );
    let normalize_ms = normalize_started.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[glrmask/profile][signed_shard_compose] start_component={start_component} terms={} signed_states={signed_states} signed_transitions={signed_transitions} resolved_states={} resolved_transitions={} reverse_topo={} compose_ms={compose_ms:.3} resolve_ms={resolve_ms:.3} normalize_ms={normalize_ms:.3} parser_states={} parser_trans={}",
        library.ordinary_terms,
        arena.states().len(),
        arena.num_transitions(),
        resolved_reverse_topo.map(|layers| layers.len()).unwrap_or(usize::MAX),
        parser_dwa.num_states(),
        parser_dwa.num_transitions(),
    );
    Ok(SignedShardOutput {
        parser_dwa,
        templates_ms: library.templates_ms,
        compose_ms,
        resolve_ms,
        normalize_ms,
        signed_states,
        signed_transitions,
        terms: library.ordinary_terms,
    })
}

/// Publish one signed-compiled shard through the shared `StaticParser`
/// publication (shard-local TSID map in the scoped tokenizer coordinate).
pub(crate) fn publish_signed_shard(
    work: WalkBoundaryShardWork,
    output: SignedShardOutput,
    tokenizer_offsets: &[u32],
    component_state_counts: &[u32],
) -> Result<(PublishedStaticBoundaryShard, WalkShardPublishProfile), String> {
    publish_signed_boundary_shard_work(
        work,
        output.parser_dwa,
        WalkShardPublishProfile {
            templates_ms: output.templates_ms,
            materialize_ms: output.compose_ms + output.resolve_ms,
            normalize_ms: output.normalize_ms,
            terms: output.terms,
            parser_states: 0,
            parser_trans: 0,
        },
        tokenizer_offsets,
        component_state_counts,
    )
}

/// Strict-static trap flag: when `GLRMASK_STRICT_STATIC_TRAP_DYNAMIC=1`,
/// evaluating any `DynamicDirect` boundary shard panics loudly instead of
/// silently contributing exact dynamic masks on a claimed static path.
pub(crate) fn strict_static_dynamic_trap_enabled() -> bool {
    std::env::var_os("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC").is_some()
}

pub(crate) fn eof_terminal() -> TerminalID {
    EOF
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher_top_is(pop: &[StackMatcher], state: u32) -> bool {
        matches!(pop.first(), Some(StackMatcher::State(top)) if *top == state)
    }

    /// Literal concrete-stack interpreter for escapes (bottom-to-top stacks).
    /// Pop matchers apply top-down (`pop[0]` is the stack top); `Any` matches
    /// anything; pushes append bottom-to-top. Returns `None` when the escape
    /// does not apply.
    fn apply_escape(pop: &[StackMatcher], pushes: &[u32], stack: &[u32]) -> Option<Vec<u32>> {
        if stack.len() < pop.len() {
            return None;
        }
        for (offset, matcher) in pop.iter().enumerate() {
            let actual = stack[stack.len() - 1 - offset];
            match matcher {
                StackMatcher::Any => {}
                StackMatcher::State(want) => {
                    if actual != *want {
                        return None;
                    }
                }
                StackMatcher::States(wants) => {
                    if !wants.contains(&actual) {
                        return None;
                    }
                }
            }
        }
        let mut next = stack[..stack.len() - pop.len()].to_vec();
        next.extend_from_slice(pushes);
        Some(next)
    }

    fn apply_characterization(
        characterization: &TerminalCharacterization,
        stack: &[u32],
    ) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        for escape in &characterization.escapes {
            if let Some(next) = apply_escape(&escape.pop, &escape.pushes, stack) {
                out.push(next);
            }
        }
        out
    }

    /// Advisor §6 algebra, scoped single-link coordinates: parent target
    /// states `I_L/I_R`, continuations `Qx/Qy`, child start `S`, child
    /// interior `J`. `T_a; F; T_x` must admit the matching caller and reject
    /// the sibling caller; dropping the child-start push must break the chain.
    #[test]
    #[ignore]
    fn entry_finish_cancellation_distinguishes_call_sites() {
        const I_L: u32 = 10;
        const QX: u32 = 11;
        const QY: u32 = 12;
        const S: u32 = 100;
        const J: u32 = 101;
        const X: u32 = 13;
        // Slot escape R(I) P(I) P(Q) as characterized, then Entry appends S.
        let slot = TerminalCharacterization {
            escapes: vec![InitialEscape {
                pop: vec![StackMatcher::State(I_L)],
                pushes: vec![I_L, QX],
            }],
            reduces: vec![],
            nt_escapes: vec![],
            nt_rereduces: vec![],
            all_nts: BTreeSet::new(),
        };
        let entry = instantiate_entry(slot, S, 0);
        assert!(matcher_top_is(&entry.characterization.escapes[0].pop, I_L));
        assert_eq!(entry.characterization.escapes[0].pushes, vec![I_L, QX, S]);
        // Child terminal a: R(S) P(S) P(J).
        let child_ta = TerminalCharacterization {
            escapes: vec![InitialEscape {
                pop: vec![StackMatcher::State(S)],
                pushes: vec![S, J],
            }],
            reduces: vec![],
            nt_escapes: vec![],
            nt_rereduces: vec![],
            all_nts: BTreeSet::new(),
        };
        // Finish: R(J) R(Any) with return_pop=2.
        let finish = TerminalCharacterization {
            escapes: vec![InitialEscape {
                pop: vec![StackMatcher::State(J), StackMatcher::Any],
                pushes: vec![],
            }],
            reduces: vec![],
            nt_escapes: vec![],
            nt_rereduces: vec![],
            all_nts: BTreeSet::new(),
        };
        // Parent continuation terminal x: R(Qx) P(Qx) P(X).
        let parent_tx = TerminalCharacterization {
            escapes: vec![InitialEscape {
                pop: vec![StackMatcher::State(QX)],
                pushes: vec![QX, X],
            }],
            reduces: vec![],
            nt_escapes: vec![],
            nt_rereduces: vec![],
            all_nts: BTreeSet::new(),
        };
        // Matching caller: [I_L] -> E -> T_a -> F -> T_x.
        let stack = vec![I_L];
        let stack = apply_characterization(&entry.characterization, &stack)
            .into_iter()
            .next()
            .expect("entry applies at I_L");
        assert_eq!(stack, vec![I_L, QX, S]);
        let stack = apply_characterization(&child_ta, &stack)
            .into_iter()
            .next()
            .expect("child terminal applies on child start");
        assert_eq!(stack, vec![I_L, QX, S, J]);
        let stack = apply_characterization(&finish, &stack)
            .into_iter()
            .next()
            .expect("finish pops the child frame");
        assert_eq!(stack, vec![I_L, QX]);
        let stack = apply_characterization(&parent_tx, &stack)
            .into_iter()
            .next()
            .expect("matching continuation applies after return");
        assert_eq!(stack, vec![I_L, QX, X]);
        // Sibling caller carrying Qy: an entry from the sibling call site would
        // expose Qy, which T_x rejects.
        let sibling_stack = vec![I_L, QY];
        assert!(
            apply_characterization(&parent_tx, &sibling_stack).is_empty(),
            "sibling continuation must not match Qx transfer",
        );
        // Ablation: entry without the child-start push breaks the chain at T_a.
        let broken_entry = TerminalCharacterization {
            escapes: vec![InitialEscape {
                pop: vec![StackMatcher::State(I_L)],
                pushes: vec![I_L, QX],
            }],
            reduces: vec![],
            nt_escapes: vec![],
            nt_rereduces: vec![],
            all_nts: BTreeSet::new(),
        };
        let stack = apply_characterization(&broken_entry, &[I_L])
            .into_iter()
            .next()
            .unwrap();
        assert!(
            apply_characterization(&child_ta, &stack).is_empty(),
            "removing the child-start push must break the called-frame chain",
        );
    }

    #[test]
    #[ignore]
    fn shared_child_link_disagreement_declines_loudly() {
        use crate::compiler::glr::parser::ScopedSubgrammarLink;
        let links = vec![
            ScopedSubgrammarLink {
                parent_component: 0,
                slot_terminal: 3,
                child_component: 1,
                child_start: 0,
                return_pop: 1,
                child_start_nullable: false,
            },
            ScopedSubgrammarLink {
                parent_component: 0,
                slot_terminal: 4,
                child_component: 1,
                child_start: 0,
                return_pop: 2,
                child_start_nullable: false,
            },
        ];
        assert!(validate_shared_child_links(&links).is_err());
        let agreeing = vec![links[0]];
        assert!(validate_shared_child_links(&agreeing).is_ok());
    }

    #[test]
    #[ignore]
    fn composer_scaffold_declines_loudly_without_hidden_fallback() {
        assert!(assemble_boundary_transfer_query().is_err());
    }
}
